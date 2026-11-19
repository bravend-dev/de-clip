"""Experiment configuration.

A configuration is split along the boundary that matters for reproducibility:

* ``encode`` -- which description files and prompt templates are embedded (Sec. 3.3-3.4).
  Changing anything here requires re-encoding text.
* ``aggregate`` -- how the per-class prompt embeddings are refined into the final class text
  embedding (Sec. 3.5-3.6). Changing this only re-runs arithmetic on cached embeddings.
* ``evaluation`` -- which benchmark protocol and datasets are used (Sec. 4.1-4.2).

Configurations are YAML files. A file may name a ``base`` file whose values it overrides,
and a ``grid`` mapping ``aggregate.<field>: [values]`` that :mod:`declip.run` sweeps.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Label templates P^L_m, m = 1..M (Sec. 3.3). The order is part of the cache key.
LABEL_TEMPLATES: list[str] = [
    "This is a skin image of {c}",
    "This is a skin image of {c}.",
    "A skin image of {c}.",
    "An image of {c}, a skin condition.",
    "{c}, a skin disorder, is shown in this image.",
    "The skin lesion depicted is {c}.",
    "The skin cancer in this image is {c}.",
    "This image depicts {c}, a type of skin cancer.",
]

# Description templates P^D_r, r = 1..R (Sec. 3.4).
DESCRIPTION_TEMPLATES: list[str] = [
    "A skin lesion showing {d}",
    "This image displays {d}",
    "Close-up view of a lesion with {d}",
    "Visible {d} on the skin surface",
    "The presence of {d} is noted",
    "Skin condition with {d}",
    "{d}",
]

# The bare class name is kept as one extra candidate in the description block.
CLASS_NAME_CANDIDATE_TEMPLATES: list[str] = ["{c}."]

SELECTION_TYPES = ("grouping", "top_h", "none")
WEIGHTING_TYPES = ("shifted_linear", "uniform")
PROTOCOLS = ("weighted_f1", "accuracy")


@dataclass
class TemplatesConfig:
    label: list[str] = field(default_factory=lambda: list(LABEL_TEMPLATES))
    description: list[str] = field(default_factory=lambda: list(DESCRIPTION_TEMPLATES))
    class_name_candidate: list[str] = field(
        default_factory=lambda: list(CLASS_NAME_CANDIDATE_TEMPLATES)
    )


@dataclass
class EncodeConfig:
    """Description files (relative to the repository root) are concatenated in order."""

    descriptions: list[str] = field(default_factory=list)
    templates: TemplatesConfig = field(default_factory=TemplatesConfig)


@dataclass
class SelectionConfig:
    """Redundancy grouping, Sec. 3.5.2. ``top_h`` keeps the H best-scoring candidates."""

    type: str = "grouping"
    tau: float = 0.95
    top_h: int | None = None


@dataclass
class AggregateConfig:
    """Candidates are scored by anchor similarity, Eq. (7), then selected, weighted and blended."""

    selection: SelectionConfig = field(default_factory=SelectionConfig)
    weighting: str = "shifted_linear"
    alpha: float = 0.5
    # False keeps the plain mean of the unit candidate embeddings (no re-normalisation), i.e. the
    # score-mean rule of DCLIP / WaffleCLIP; only meaningful for alpha = 0 (description-only).
    renormalize: bool = True


@dataclass
class EvaluationConfig:
    """``datasets`` empty means every dataset of the protocol (see :mod:`declip.datasets`)."""

    protocol: str = "weighted_f1"
    datasets: list[str] = field(default_factory=list)
    meta_dir: str = "data/meta"  # evaluation CSVs: <meta_dir>/<protocol>/<dataset>.csv


@dataclass
class ExperimentConfig:
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    aggregate: AggregateConfig = field(default_factory=AggregateConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    grid: dict[str, list[Any]] = field(default_factory=dict)


def to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def _apply_mapping(obj: Any, mapping: dict, path: str = "") -> None:
    known = {f.name for f in fields(obj)}
    for key, value in mapping.items():
        if key not in known:
            raise ValueError(f"unknown config field '{path}{key}' (valid: {sorted(known)})")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"'{path}{key}' expects a mapping, got {type(value).__name__}")
            _apply_mapping(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)


def _coerce(text: str) -> Any:
    """'0.4' -> 0.4, 'true' -> True, 'null' -> None, '[1,2]' -> [1, 2]; other strings unchanged."""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def set_field(cfg: Any, path: str, value: Any) -> None:
    """Set a dotted field such as ``aggregate.selection.tau``."""
    parts = path.split(".")
    target = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"unknown config field '{path}'")
        target = getattr(target, part)
    if not hasattr(target, parts[-1]):
        raise ValueError(f"unknown config field '{path}'")
    setattr(target, parts[-1], value)


def apply_override(cfg: ExperimentConfig, assignment: str) -> None:
    """Apply a ``key=value`` override given on the command line."""
    if "=" not in assignment:
        raise ValueError(f"override must look like key=value, got '{assignment}'")
    path, raw = assignment.split("=", 1)
    set_field(cfg, path.strip(), _coerce(raw))


def _read_yaml_with_base(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    base = data.pop("base", None)
    if base is None:
        return data
    base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
    merged = _read_yaml_with_base(base_path)
    return _deep_merge(merged, data)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | None = None, overrides: list[str] | None = None) -> ExperimentConfig:
    """Defaults -> YAML file (with ``base`` chain) -> ``--set`` overrides."""
    cfg = ExperimentConfig()
    if path:
        data = _read_yaml_with_base(path)
        grid = data.pop("grid", None) or {}
        _apply_mapping(cfg, data)
        cfg.grid = {k: (v if isinstance(v, list) else [v]) for k, v in grid.items()}
    for assignment in overrides or []:
        apply_override(cfg, assignment)
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExperimentConfig) -> None:
    agg = cfg.aggregate
    if agg.selection.type not in SELECTION_TYPES:
        raise ValueError(f"aggregate.selection.type must be one of {SELECTION_TYPES}")
    if agg.weighting not in WEIGHTING_TYPES:
        raise ValueError(f"aggregate.weighting must be one of {WEIGHTING_TYPES}")
    if agg.selection.type == "top_h" and not agg.selection.top_h:
        raise ValueError("aggregate.selection.type=top_h requires aggregate.selection.top_h")
    if not 0.0 <= agg.alpha <= 1.0:
        raise ValueError(f"aggregate.alpha must be in [0, 1], got {agg.alpha}")
    if not agg.renormalize and agg.alpha != 0.0:
        raise ValueError("aggregate.renormalize=false (score-mean baseline) needs alpha=0")
    if cfg.evaluation.protocol not in PROTOCOLS:
        raise ValueError(f"evaluation.protocol must be one of {PROTOCOLS}")
    if not cfg.encode.templates.label:
        raise ValueError("encode.templates.label must not be empty")
    for key in cfg.grid:
        if not key.startswith("aggregate."):
            raise ValueError(
                f"grid can only sweep aggregate.* fields, got '{key}'. Fields under encode.* "
                "change the prompts and need a separate run."
            )


def dump_config(cfg: ExperimentConfig, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_dict(cfg), handle, indent=2, sort_keys=True)
