"""LLM-generated class descriptions and prompt rendering (Sec. 3.3-3.4).

A description file is a JSON object ``{class name: [description, ...]}`` produced by
``tools/generate_descriptions.py``. Several files are concatenated in the order given by
the configuration; the pool of a class is the concatenation of its lists.

For a class ``c`` with descriptions ``d_1..d_K`` the prompts are rendered in this order:

1. the M label prompts ``P^L_m(c)``;
2. the K x R description prompts ``P^D_r(d_k)``, template-major (all K of template 1,
   then all K of template 2, ...), lower-cased description, trailing period;
3. the bare class-name candidates (``"{c}."``).

The first M prompts form the label block used for the fused label embedding L_i; the rest
form the candidate block D_{i,k,r}.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence

from declip.config import REPO_ROOT, TemplatesConfig

Pool = dict[str, list[str]]


def load_descriptions(files: Sequence[str], root: str = REPO_ROOT) -> Pool:
    pool: dict[str, list[str]] = defaultdict(list)
    for rel in files:
        path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(f"description file not found: {path}")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        for classname, descriptions in data.items():
            pool[classname].extend(descriptions)
    return dict(pool)


def descriptions_per_class(pool: Pool) -> int:
    """K: the number of descriptions per class (the maximum when classes differ)."""
    return max((len(v) for v in pool.values()), default=0)


def validate_descriptions(pool: Pool, classnames: Iterable[str], k: int) -> None:
    classnames = list(classnames)
    missing = [c for c in classnames if c not in pool]
    if missing:
        raise ValueError(f"{len(missing)} class names have no descriptions, e.g. {missing[:5]}")
    empty = [c for c in classnames if not pool[c]]
    if empty:
        raise ValueError(f"classes without descriptions: {empty[:5]}")


def num_label_prompts(templates: TemplatesConfig) -> int:
    return len(templates.label)


def num_prompts(templates: TemplatesConfig, k: int) -> int:
    return (
        len(templates.label) + len(templates.description) * k + len(templates.class_name_candidate)
    )


def candidate_mask(classnames: Sequence[str], pool: Pool, templates: TemplatesConfig, k: int):
    """(C, N) boolean mask of the real candidates when classes have fewer than ``k``
    descriptions: :func:`render_prompts` cycles a short list to length ``k`` and the repeats
    are masked out here, so every original description counts once (as in DCLIP)."""
    import torch

    n = len(templates.description) * k + len(templates.class_name_candidate)
    mask = torch.ones(len(classnames), n, dtype=torch.bool)
    for i, classname in enumerate(classnames):
        count = min(len(pool[classname]), k)
        for r in range(len(templates.description)):
            mask[i, r * k + count : (r + 1) * k] = False
    return mask


def render_prompts(classname: str, pool: Pool, templates: TemplatesConfig, k: int) -> list[str]:
    prompts = [t.format(c=classname) for t in templates.label]
    descriptions = pool[classname][:k]
    if len(descriptions) < k:  # cycle short lists to a common length; see candidate_mask
        descriptions = [descriptions[i % len(descriptions)] for i in range(k)]
    for template in templates.description:
        prompts.extend(template.format(c=classname, d=d.lower()) + "." for d in descriptions)
    prompts.extend(t.format(c=classname) for t in templates.class_name_candidate)
    return prompts
