"""Command-line entry point: evaluate one configuration (or a grid) on one or all backbones.

    python -m declip.run --config configs/default.yaml --backbone dermlip
    python -m declip.run --config configs/ablations/alpha.yaml --backbone all
    python -m declip.run --batch jobs.txt     # one run per line, models and caches loaded once

Images and prompts are encoded once per backbone; every grid point only re-runs the
aggregation of Sec. 3.5-3.6 on the cached embeddings. Results are written as a CSV with one
row per grid point (per-dataset primary metric and their mean) plus a long-format CSV with
all metrics and the resolved configuration as JSON.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import logging
import os
import shlex
import sys
import time
from typing import Any

import torch

from declip.aggregate import build_class_embeddings
from declip.backbones import BACKBONES, load_backbone
from declip.cache import ImageEmbeddingCache, TextEmbeddingStore
from declip.config import (
    REPO_ROOT,
    ExperimentConfig,
    dump_config,
    load_config,
    set_field,
    validate_config,
)
from declip.datasets import dataset_specs, union_classnames
from declip.descriptions import (
    candidate_mask,
    descriptions_per_class,
    load_descriptions,
    num_label_prompts,
    validate_descriptions,
)
from declip.encode import encode_class_prompts, encode_images
from declip.metrics import evaluate

log = logging.getLogger("declip")

# Batch mode (--batch) keeps the last backbone, its text store and the image embeddings in
# memory across runs; a single run frees the model after encoding as before.
_SESSION: dict[str, Any] = {"reuse": False, "backbone": None, "store": None, "images": {}}


def _backbone(name: str, args: argparse.Namespace):
    held = _SESSION["backbone"]
    if held is not None and held[0] == (name, args.device, args.precision):
        return held[1], _SESSION["store"]
    _SESSION["backbone"] = _SESSION["store"] = None
    _SESSION["images"].clear()
    torch.cuda.empty_cache()
    backbone = load_backbone(name, device=args.device, precision=args.precision)
    store = TextEmbeddingStore(args.cache_dir, backbone.cache_key(), enabled=not args.no_cache)
    if _SESSION["reuse"]:
        _SESSION["backbone"], _SESSION["store"] = (
            (name, args.device, args.precision),
            backbone,
        ), store
    return backbone, store


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", help="YAML experiment configuration")
    parser.add_argument(
        "--batch",
        help="file with one run per line (the arguments of this command, optionally prefixed "
        "by 'python -m declip.run'); prints 'ok <out>' or 'FAIL <out>' per line",
    )
    parser.add_argument(
        "--backbone", default="dermlip", help=f"one of {sorted(BACKBONES)} or 'all'"
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="extend",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="override configuration fields, e.g. aggregate.alpha=0.5 (repeatable)",
    )
    parser.add_argument(
        "--grid",
        nargs="*",
        default=[],
        metavar="KEY=V1,V2",
        help="sweep aggregate.* fields; combined with the 'grid' key of the YAML",
    )
    parser.add_argument(
        "--out", default=None, help="output CSV (default: results/<config>_<backbone>.csv)"
    )
    parser.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, "cache"))
    parser.add_argument(
        "--no-cache", action="store_true", help="do not read or write embedding caches"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--precision",
        default="amp",
        choices=["amp", "fp32"],
        help="autocast fp16 for image encoding (amp) or plain fp32",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _parse_grid(items: list[str]) -> dict[str, list[Any]]:
    from declip.config import _coerce

    grid: dict[str, list[Any]] = {}
    for item in items:
        key, _, values = item.partition("=")
        grid[key.strip()] = [_coerce(v.strip()) for v in values.split(",")]
    return grid


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def run_backbone(
    cfg: ExperimentConfig, backbone_name: str, args: argparse.Namespace
) -> list[dict[str, Any]]:
    backbone, text_store = _backbone(backbone_name, args)
    specs = dataset_specs(cfg.evaluation.protocol, cfg.evaluation.datasets, cfg.evaluation.meta_dir)
    pool = load_descriptions(cfg.encode.descriptions)
    k = descriptions_per_class(pool)
    validate_descriptions(pool, union_classnames(specs), k)
    m = num_label_prompts(cfg.encode.templates)
    log.info(
        "%s: %d datasets, K=%d descriptions/class, M=%d label prompts",
        backbone_name,
        len(specs),
        k,
        m,
    )

    image_cache = ImageEmbeddingCache(args.cache_dir, enabled=not args.no_cache)

    prepared = []
    for spec in specs:
        image_key = (spec.name, spec.csv_path)
        if image_key not in _SESSION["images"]:
            images = encode_images(backbone, spec, image_cache, args.batch_size, args.workers)
            if not _SESSION["reuse"]:
                features, labels = images
            else:
                _SESSION["images"][image_key] = images
        if _SESSION["reuse"]:
            features, labels = _SESSION["images"][image_key]
        prompts = encode_class_prompts(
            backbone, list(spec.classnames), pool, cfg.encode.templates, k, text_store
        )
        mask = candidate_mask(list(spec.classnames), pool, cfg.encode.templates, k)
        prepared.append((spec, features, labels, prompts.to(backbone.device), mask, {}))
    text_store.flush()
    if not _SESSION["reuse"]:
        del backbone.model
        torch.cuda.empty_cache()

    rows = []
    for point in expand_grid(cfg.grid):
        point_cfg = copy.deepcopy(cfg)
        for key, value in point.items():
            set_field(point_cfg, key, value)
        validate_config(point_cfg)
        row: dict[str, Any] = {"backbone": backbone_name, **point}
        metrics_long = []
        primaries = []
        for spec, features, labels, prompts, mask, memo in prepared:
            class_embeddings = build_class_embeddings(prompts, point_cfg.aggregate, m, mask, memo)
            metrics = evaluate(features, labels, class_embeddings, spec.metric_set)
            row[spec.name] = round(100 * metrics[spec.primary_metric], 4)
            primaries.append(metrics[spec.primary_metric])
            metrics_long.extend(
                {"dataset": spec.name, "metric": name, "value": round(100 * v, 4)}
                for name, v in metrics.items()
            )
        row["mean"] = round(100 * sum(primaries) / len(primaries), 4)
        row["_metrics"] = metrics_long
        rows.append(row)
        log.info("%s %s -> mean %.2f", backbone_name, point or "(config)", row["mean"])
    return rows


def write_results(rows: list[dict[str, Any]], out_path: str, cfg: ExperimentConfig) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    columns = [c for c in rows[0] if c != "_metrics"]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in columns})
    long_path = os.path.splitext(out_path)[0] + ".metrics.csv"
    grid_keys = [
        c
        for c in columns
        if c not in ("backbone", "mean") and c not in {r for r in columns if r[0].isupper()}
    ]
    with open(long_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["backbone", *grid_keys, "dataset", "metric", "value"]
        )
        writer.writeheader()
        for row in rows:
            for item in row["_metrics"]:
                writer.writerow(
                    {"backbone": row["backbone"], **{k: row[k] for k in grid_keys}, **item}
                )
    dump_config(cfg, os.path.splitext(out_path)[0] + ".config.json")
    log.info("wrote %s, %s", out_path, long_path)


def run_batch(path: str) -> int:
    """Run every line of ``path`` in this process; returns the number of failed runs."""
    _SESSION["reuse"] = True
    failed = 0
    with open(path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    for line in lines:
        argv = shlex.split(line)
        if argv[:3] == ["python", "-m", "declip.run"]:
            argv = argv[3:]
        args = parse_args(argv)
        start = time.time()
        try:
            run_one(args)
            print(f"ok {args.out} {time.time() - start:.1f}s", flush=True)
        except Exception:  # keep going; the failed line is reported and can be re-run
            log.exception("failed: %s", line)
            print(f"FAIL {line}", flush=True)
            failed += 1
            _SESSION["backbone"] = _SESSION["store"] = None
            _SESSION["images"].clear()
            torch.cuda.empty_cache()
    return failed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.batch:
        return 1 if run_batch(args.batch) else 0
    if not args.config:
        raise SystemExit("--config or --batch is required")
    return run_one(args)


def run_one(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    cfg.grid.update(_parse_grid(args.grid))
    validate_config(cfg)

    backbones = sorted(BACKBONES) if args.backbone == "all" else [args.backbone]
    stem = os.path.splitext(os.path.basename(args.config))[0]
    out_path = args.out or os.path.join(REPO_ROOT, "results", f"{stem}_{args.backbone}.csv")
    rows: list[dict[str, Any]] = []
    for name in backbones:
        rows.extend(run_backbone(cfg, name, args))
    write_results(rows, out_path, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
