"""Variability of the DCLIP baseline over subsets of its descriptors.

DCLIP has one generated pool of 6-21 descriptors per class. As an analogue of the five
independent K = 100 runs used for CuPL and DE-CLIP, this script evaluates DCLIP on 5 random
subsets holding 71% of each class's descriptors (seeds 0-4), both protocols, all backbones, and
writes one row per (subset, backbone, protocol, dataset).

    python tools/dclip_subsets.py --out results/variance_dclip.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from declip.aggregate import build_class_embeddings  # noqa: E402
from declip.backbones import BACKBONES, load_backbone  # noqa: E402
from declip.cache import ImageEmbeddingCache, TextEmbeddingStore  # noqa: E402
from declip.config import load_config, validate_config  # noqa: E402
from declip.datasets import dataset_specs  # noqa: E402
from declip.descriptions import candidate_mask, num_label_prompts  # noqa: E402
from declip.encode import encode_class_prompts, encode_images  # noqa: E402
from declip.metrics import evaluate  # noqa: E402

log = logging.getLogger("dclip_subsets")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="all")
    parser.add_argument("--subsets", type=int, default=5)
    parser.add_argument("--keep", type=float, default=5 / 7, help="fraction of descriptors kept")
    parser.add_argument("--out", default=os.path.join(ROOT, "results", "variance_dclip.csv"))
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, "cache"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(os.path.join(ROOT, "configs", "baselines", "dclip.yaml"))
    cfg.grid = {}
    validate_config(cfg)
    m = num_label_prompts(cfg.encode.templates)
    with open(os.path.join(ROOT, cfg.encode.descriptions[0]), encoding="utf-8") as handle:
        full = json.load(handle)
    rows = []
    for name in sorted(BACKBONES) if args.backbone == "all" else [args.backbone]:
        backbone = load_backbone(name, device=args.device)
        image_cache = ImageEmbeddingCache(args.cache_dir)
        store = TextEmbeddingStore(args.cache_dir, backbone.cache_key())
        for protocol in ["weighted_f1", "accuracy"]:
            specs = dataset_specs(protocol)
            images = [encode_images(backbone, s, image_cache) for s in specs]
            for seed in range(args.subsets):
                rng = random.Random(seed)
                pool = {}
                for c, d in full.items():
                    n_keep = max(1, round(args.keep * len(d)))
                    pool[c] = [d[i] for i in sorted(rng.sample(range(len(d)), n_keep))]
                k = max(len(v) for v in pool.values())
                values = []
                for spec, (feats, labels) in zip(specs, images):
                    prompts = encode_class_prompts(
                        backbone,
                        list(spec.classnames),
                        pool,
                        cfg.encode.templates,
                        k,
                        store,
                    ).to(backbone.device)
                    mask = candidate_mask(list(spec.classnames), pool, cfg.encode.templates, k)
                    emb = build_class_embeddings(prompts, cfg.aggregate, m, mask)
                    value = 100 * evaluate(feats, labels, emb, spec.metric_set)[spec.primary_metric]
                    values.append(value)
                    rows.append(
                        dict(
                            backbone=name,
                            protocol=protocol,
                            seed=seed,
                            dataset=spec.name,
                            value=round(value, 4),
                        )
                    )
                rows.append(
                    dict(
                        backbone=name,
                        protocol=protocol,
                        seed=seed,
                        dataset="mean",
                        value=round(sum(values) / len(values), 4),
                    )
                )
                log.info("%s %s subset %d done", name, protocol, seed)
            store.flush()
        del backbone.model
        torch.cuda.empty_cache()
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %s (%d rows)", args.out, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
