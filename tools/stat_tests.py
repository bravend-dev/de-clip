#!/usr/bin/env python
"""Statistical tests for the key comparisons of the benchmark tables.

DE-CLIP vs. Default, DE-CLIP vs. CuPL and CuPL vs. Default, as reported in the "Statistical
Significance" table (Section 4.7).

Input: the five generation runs of the two benchmarks,
    results/benchmark_weighted_f1{,_run2,...,_run5}.csv   (nine datasets, Table 4 of the paper)
    results/benchmark_accuracy{,_run2,...,_run5}.csv      (four datasets, Table 3 of the paper)

For every comparison and backbone the script reports the difference of the average score over
datasets (mean over the five runs, i.e. the Delta columns of Tables 3-4) and the two-sided
p-value of a paired t-test over the five runs (Default is deterministic, so against it this is a
one-sample test of the five differences). The last row of each comparison is a two-sided
Wilcoxon signed-rank test over all dataset-backbone pairs (45 for weighted-F1, 20 for accuracy),
each pair being the five-run mean of the two methods, following Demsar (JMLR 2006); the number
of pairs favouring the first method is given in parentheses.

    python tools/stat_tests.py                 # markdown table on stdout (Table 11)
    python tools/stat_tests.py --csv results/stat_tests.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
RUN_SUFFIXES = ["", "_run2", "_run3", "_run4", "_run5"]
META = ("backbone", "aggregate.alpha", "aggregate.selection.type", "aggregate.weighting")
METHODS = {  # (alpha, selection, weighting) of the grid rows of configs/methods.yaml
    "Default": ("1.0", "none", "uniform"),
    "CuPL": ("0.0", "none", "uniform"),
    "DE-CLIP": ("0.5", "grouping", "shifted_linear"),
}
BACKBONES = [
    ("dermlip", "DermLIP"),
    ("make", "MAKE"),
    ("biomedclip", "BiomedCLIP"),
    ("pmc-clip", "PMC-CLIP"),
    ("openai-clip", "OpenAI-CLIP"),
]
PROTOCOLS = [("benchmark_weighted_f1", "wF1"), ("benchmark_accuracy", "Acc")]


def load(stem: str) -> tuple[dict[tuple[str, str], np.ndarray], list[str]]:
    """-> {(method, backbone): array[run, dataset]}, dataset names (without the 'mean' column)."""
    out: dict[tuple[str, str], list[list[float]]] = {}
    datasets: list[str] = []
    for suffix in RUN_SUFFIXES:
        with open(os.path.join(RESULTS, f"{stem}{suffix}.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        datasets = [c for c in rows[0] if c not in META and c != "mean"]
        for row in rows:
            key = (
                row["aggregate.alpha"],
                row["aggregate.selection.type"],
                row["aggregate.weighting"],
            )
            for method, spec in METHODS.items():
                if key == spec:
                    out.setdefault((method, row["backbone"]), []).append(
                        [float(row[d]) for d in datasets]
                    )
    arrays = {k: np.array(v) for k, v in out.items()}
    for k, v in arrays.items():
        if v.shape[0] != len(RUN_SUFFIXES):
            raise SystemExit(
                f"{stem}: expected {len(RUN_SUFFIXES)} runs for {k}, found {v.shape[0]}"
            )
    return arrays, datasets


def fmt_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


COMPARISONS = [("DE-CLIP", "Default"), ("DE-CLIP", "CuPL"), ("CuPL", "Default")]


def compute() -> dict:
    """{(method, base): {"rows": [(backbone, {proto: (delta, p)})], "all": {proto: (p, k, n)}}}"""
    out = {c: {"rows": [(name, {}) for _, name in BACKBONES], "all": {}} for c in COMPARISONS}
    for stem, proto in PROTOCOLS:
        data, _ = load(stem)
        for method, base in COMPARISONS:
            pairs = []
            for i, (slug, _) in enumerate(BACKBONES):
                a, b = data[(method, slug)], data[(base, slug)]
                a_avg, b_avg = a.mean(axis=1), b.mean(axis=1)  # average over datasets, per run
                delta = float(a_avg.mean() - b_avg.mean())
                p = float(stats.ttest_rel(a_avg, b_avg).pvalue)
                out[(method, base)]["rows"][i][1][proto] = (delta, p)
                pairs.extend(a.mean(axis=0) - b.mean(axis=0))  # five-run mean per dataset
            pairs = np.array(pairs)
            out[(method, base)]["all"][proto] = (
                float(stats.wilcoxon(pairs).pvalue),
                int((pairs > 0).sum()),
                len(pairs),
            )
    return out


PROTO_NAMES = [name for _, name in PROTOCOLS]


def render_md(res: dict) -> str:
    lines = ["| Comparison | Backbone | " + " | ".join(f"{m}: Δ (p)" for m in PROTO_NAMES) + " |"]
    lines.append("|---" * (len(PROTO_NAMES) + 2) + "|")
    for (method, base), r in res.items():
        label = f"{method} vs. {base}"
        for backbone, vals in r["rows"]:
            cells = " | ".join(f"{vals[m][0]:+.2f} ({fmt_p(vals[m][1])})" for m in PROTO_NAMES)
            lines.append(f"| {label} | {backbone} | {cells} |")
        cells = " | ".join(
            f"{fmt_p(p)} ({k}/{n})" for m in PROTO_NAMES for p, k, n in [r["all"][m]]
        )
        lines.append(f"| {label} | All pairs (Wilcoxon) | {cells} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--csv", help="also write the raw numbers to this CSV")
    args = ap.parse_args()
    res = compute()
    print(render_md(res))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "method",
                    "baseline",
                    "row",
                    "metric",
                    "delta_avg",
                    "p_value",
                    "test",
                    "n",
                    "pairs_favouring_method",
                ]
            )
            for (method, base), r in res.items():
                for backbone, vals in r["rows"]:
                    for m in PROTO_NAMES:
                        d, p = vals[m]
                        w.writerow(
                            [
                                method,
                                base,
                                backbone,
                                m,
                                f"{d:.4f}",
                                f"{p:.3e}",
                                "paired t-test over runs",
                                5,
                                "",
                            ]
                        )
                for m in PROTO_NAMES:
                    p, k, n = r["all"][m]
                    w.writerow(
                        [
                            method,
                            base,
                            "all pairs",
                            m,
                            "",
                            f"{p:.3e}",
                            "Wilcoxon signed-rank over dataset-backbone pairs",
                            n,
                            k,
                        ]
                    )


if __name__ == "__main__":
    main()
