"""Configuration loading and prompt rendering."""

import os

import pytest

from declip.config import (
    DESCRIPTION_TEMPLATES,
    LABEL_TEMPLATES,
    REPO_ROOT,
    apply_override,
    load_config,
)
from declip.datasets import dataset_specs, union_classnames
from declip.descriptions import (
    descriptions_per_class,
    load_descriptions,
    num_prompts,
    render_prompts,
    validate_descriptions,
)

CONFIG = os.path.join(REPO_ROOT, "configs", "default.yaml")


def test_default_yaml_matches_paper_constants():
    cfg = load_config(CONFIG)
    assert cfg.encode.templates.label == LABEL_TEMPLATES and len(cfg.encode.templates.label) == 8
    assert (
        cfg.encode.templates.description == DESCRIPTION_TEMPLATES
        and len(cfg.encode.templates.description) == 7
    )
    assert cfg.aggregate.selection.tau == 0.95
    assert cfg.aggregate.alpha == 0.5
    assert len(cfg.encode.descriptions) == 5


def test_prompt_layout():
    cfg = load_config(CONFIG)
    pool = load_descriptions(cfg.encode.descriptions)
    k = descriptions_per_class(pool)
    assert k == 100
    specs = dataset_specs("weighted_f1")
    validate_descriptions(pool, union_classnames(specs), k)
    prompts = render_prompts("melanoma", pool, cfg.encode.templates, k)
    assert len(prompts) == num_prompts(cfg.encode.templates, k) == 8 + 7 * 100 + 1
    assert prompts[0] == "This is a skin image of melanoma"
    assert prompts[8].startswith("A skin lesion showing ") and prompts[8].endswith(".")
    assert prompts[-1] == "melanoma."


def test_base_and_override_chain():
    cfg = load_config(os.path.join(REPO_ROOT, "configs", "baselines", "dclip.yaml"))
    assert cfg.aggregate.selection.type == "none" and cfg.aggregate.renormalize is False
    assert cfg.encode.templates.label[0] == "This is a skin image of {c}"  # inherited from default
    cfg = load_config(os.path.join(REPO_ROOT, "configs", "ablations", "alpha.yaml"))
    assert cfg.aggregate.selection.type == "grouping"
    assert cfg.grid == {"aggregate.alpha": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    apply_override(cfg, "aggregate.alpha=0.6")
    assert cfg.aggregate.alpha == 0.6
    with pytest.raises(ValueError):
        apply_override(cfg, "aggregate.nonexistent=1")


def test_grid_rejects_encode_fields(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"base: {CONFIG}\ngrid:\n  encode.descriptions: [[a.json], [b.json]]\n")
    with pytest.raises(ValueError, match="aggregate"):
        load_config(str(bad))
