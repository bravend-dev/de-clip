"""Render the paper tables (Tables 3-10) as Markdown from the CSV files in ``results/``.

    python tools/make_tables.py            # all tables that have results
    python tools/make_tables.py --table 5

The CSV files are those written by the scripts under ``scripts/`` (``scripts/run_all.sh``).
Table 11 (statistical tests) is printed by ``tools/stat_tests.py``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

BACKBONE_ORDER = ["dermlip", "make", "biomedclip", "pmc-clip", "openai-clip"]
BACKBONE_NAMES = {
    "dermlip": "DermLIP",
    "make": "MAKE",
    "biomedclip": "BiomedCLIP",
    "pmc-clip": "PMC-CLIP",
    "openai-clip": "OpenAI-CLIP",
}
# (alpha, selection, weighting) of the grid rows written by configs/methods.yaml and
# configs/ablations/component.yaml.
DEFAULT = ("1.0", "none", "uniform")
CUPL = ("0.0", "none", "uniform")
DECLIP = ("0.5", "grouping", "shifted_linear")
D1 = ("0.5", "none", "uniform")
COMPONENTS = {  # Table 5 rows (B0 and the Top-H rows are added separately)
    D1: ("D1", "+ Descriptions, unfiltered (alpha=0.5)"),
    ("0.5", "none", "shifted_linear"): ("D2", "D1 + Score weighting"),
    ("0.5", "grouping", "uniform"): ("D3", "D1 + Deduplication only (tau=0.95)"),
    DECLIP: ("D4", "D1 + Weighting + Deduplication"),
}
# Table 6: five K = 100 runs per generator (results/ablation_generator_<slug>_run<r>.csv).
GENERATORS = [
    ("gpt-5.2", "GPT-5.2"),
    ("gpt-oss-20b", "GPT-OSS-20B"),
    ("gemini-2.5-flash-lite", "Gemini-2.5-Flash-Lite"),
    ("gpt-oss-120b", "GPT-OSS-120B"),
    ("ling-3.0-flash", "Ling-3.0-Flash"),
    ("qwen3-30b-a3b", "Qwen3-30B-A3B"),
    ("mistral-small-3.2-24b", "Mistral-Small-3.2-24B"),
    ("deepseek-v3.2", "DeepSeek-V3.2"),
    ("llama-3.3-70b", "Llama-3.3-70B"),
]
RUNS = range(1, 6)
RUN_SUFFIXES = ["", "_run2", "_run3", "_run4", "_run5"]  # the five K = 100 runs (run 1 = default)
# Table 10: five single calls per setting (results/ablation_generation_<stem>_<n>.csv).
GENERATION_ROWS = [
    ("LLM (3 words, K=20)", "GPT-4.1-mini", "gpt-4.1-mini_call_k20_w3"),
    ("LLM (3 words, K=20)", "GPT-5-mini", "gpt-5-mini_call_k20_w3"),
    ("LLM (3 words, K=20)", "Gemini-3-Flash-Preview", "gemini-3-flash-preview_call_k20_w3"),
    ("LLM (3 words, K=20)", "GPT-5.2 (default)", "gpt-5.2_call_k20_w3"),
    ("Desc. length (GPT-5-mini, K=20)", "1 word", "gpt-5-mini_call_k20_w1"),
    ("Desc. length (GPT-5-mini, K=20)", "2 words", "gpt-5-mini_call_k20_w2"),
    ("Desc. length (GPT-5-mini, K=20)", "3 words", "gpt-5-mini_call_k20_w3"),
    ("Desc. length (GPT-5-mini, K=20)", "4 words", "gpt-5-mini_call_k20_w4"),
    ("Desc. length (GPT-5-mini, K=20)", "5 words", "gpt-5-mini_call_k20_w5"),
    ("Desc. length (GPT-5-mini, K=20)", "6 words", "gpt-5-mini_call_k20_w6"),
    ("Desc. per API call (=K; GPT-5-mini, 3 words)", "5", "gpt-5-mini_call_k5_w3"),
    ("Desc. per API call (=K; GPT-5-mini, 3 words)", "10", "gpt-5-mini_call_k10_w3"),
    ("Desc. per API call (=K; GPT-5-mini, 3 words)", "20", "gpt-5-mini_call_k20_w3"),
    ("Desc. per API call (=K; GPT-5-mini, 3 words)", "40", "gpt-5-mini_call_k40_w3"),
]


def read(name: str) -> list[dict[str, str]]:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_runs(stem: str) -> list[list[dict[str, str]]]:
    return [rows for rows in (read(f"{stem}{x}.csv") for x in RUN_SUFFIXES) if rows]


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def combo(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        row.get("aggregate.alpha", "0.5"),
        row.get("aggregate.selection.type", "grouping"),
        row.get("aggregate.weighting", "shifted_linear"),
    )


def dataset_columns(rows: list[dict[str, str]]) -> list[str]:
    return [c for c in rows[0] if c not in ("backbone", "mean") and not c.startswith("aggregate.")]


def markdown(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def rows_of(runs: list[list[dict[str, str]]], backbone: str, key=None) -> list[dict[str, str]]:
    """The row of ``backbone`` (and grid point ``key``) in every run / seed that has it."""
    out = []
    for rows in runs:
        row = next(
            (r for r in rows if r["backbone"] == backbone and (key is None or combo(r) == key)),
            None,
        )
        if row is not None:
            out.append(row)
    return out


def run_means(runs: list[list[dict[str, str]]], match) -> list[float]:
    """The 'mean' column of the row selected by ``match`` in every run."""
    return [float(next(r for r in rows if match(r))["mean"]) for rows in runs]


def table_benchmark(stem: str, suffix: str, title: str) -> str | None:
    """Tables 3-4: class-name baselines, WaffleCLIP, DCLIP, CuPL and DE-CLIP per backbone.

    CuPL and DE-CLIP are the mean over the five runs and WaffleCLIP the mean over five seeds,
    with Delta +- sd over runs / seeds. DCLIP uses its full descriptor pool; its sd is taken over
    random descriptor subsets (results/variance_dclip.csv, written by tools/dclip_subsets.py).
    """
    runs = read_runs(stem)
    if not runs:
        return None
    datasets = dataset_columns(runs[0])
    protocol = "accuracy" if suffix else "weighted_f1"
    subsets = read("variance_dclip.csv")
    body = []
    for backbone in BACKBONE_ORDER:
        default = rows_of(runs[:1], backbone, DEFAULT)  # deterministic: identical in every run
        if not default:
            continue
        ref = float(default[0]["mean"])
        dclip_sd = [
            float(r["value"])
            for r in subsets
            if r["backbone"] == backbone and r["protocol"] == protocol and r["dataset"] == "mean"
        ]
        methods = [
            ("Default (8 templates)", default, None),
            (
                "Class name, 1 template",
                rows_of([read(f"baseline_template_single{suffix}.csv")], backbone),
                None,
            ),
            (
                "Class name, 80 templates",
                rows_of([read(f"baseline_template_imagenet80{suffix}.csv")], backbone),
                None,
            ),
            (
                "WaffleCLIP",
                rows_of([read(f"baseline_waffle_seed{s}{suffix}.csv") for s in range(5)], backbone),
                None,
            ),
            ("DCLIP", rows_of([read(f"baseline_dclip{suffix}.csv")], backbone), dclip_sd),
            ("CuPL", rows_of(runs, backbone, CUPL), None),
            ("DE-CLIP (ours)", rows_of(runs, backbone, DECLIP), None),
        ]
        for label, rows, spread in methods:
            if not rows:
                continue
            values = {c: statistics.mean(float(r[c]) for r in rows) for c in datasets + ["mean"]}
            means = [float(r["mean"]) for r in rows]
            sd = statistics.stdev(spread) if spread and len(spread) > 1 else None
            if sd is None and len(means) > 1:
                sd = statistics.stdev(means)
            if label.startswith("Default"):
                delta = "--"
            else:
                delta = f"{values['mean'] - ref:+.2f}" + ("" if sd is None else f" ± {sd:.2f}")
            body.append(
                [
                    BACKBONE_NAMES[backbone],
                    label,
                    *[fmt(values[d]) for d in datasets],
                    fmt(values["mean"]),
                    delta,
                ]
            )
    return f"### {title}\n\n" + markdown(["Backbone", "Method", *datasets, "Avg.", "Delta"], body)


def table5() -> str | None:
    comp, top = read_runs("ablation_components"), read_runs("ablation_top_h")
    if not comp:
        return None
    b0 = statistics.mean(run_means(comp, lambda r: combo(r)[0] == "1.0"))
    body = [["B0", "Default class-name anchor (alpha=1.0)", fmt(b0), "--"]]
    for h in [r["aggregate.selection.top_h"] for r in (top[0] if top else [])]:
        v = statistics.mean(run_means(top, lambda r, h=h: r["aggregate.selection.top_h"] == h))
        body.append([f"Top-{h}", f"+ Top-H descriptions (H={h})", fmt(v), f"{v - b0:+.2f}"])
    for key, (cid, desc) in COMPONENTS.items():
        v = statistics.mean(run_means(comp, lambda r, key=key: combo(r) == key))
        body.append([cid, desc, fmt(v), f"{v - b0:+.2f}"])
    return (
        f"### Table 5: component ablation (DermLIP, weighted-F1, mean over {len(comp)} runs)\n\n"
        + markdown(["ID", "Configuration", "Avg. F1 (%)", "Delta vs. B0"], body)
    )


def table6() -> str | None:
    base = read("ablation_components.csv")
    if not base:
        return None
    b0 = next(float(r["mean"]) for r in base if combo(r)[0] == "1.0")
    body = []
    for slug, label in GENERATORS:
        runs = [read(f"ablation_generator_{slug}_run{r}.csv") for r in RUNS]
        runs = [rows for rows in runs if rows]
        if not runs:
            continue
        d1 = statistics.mean(run_means(runs, lambda r: combo(r) == D1))
        d4 = statistics.mean(run_means(runs, lambda r: combo(r) == DECLIP))
        body.append([label, fmt(d1), f"{d1 - b0:+.2f}", fmt(d4), f"{d4 - b0:+.2f}"])
    return (
        "### Table 6: description generator (DermLIP, weighted-F1, mean over 5 runs of K=100)\n\n"
        + markdown(["Generator", "D1", "Delta vs. B0", "D4", "Delta vs. B0"], body)
    )


def table_sweep(stem: str, column: str, title: str, label: str) -> str | None:
    """Tables 7-8: one column per backbone, average weighted-F1, mean over the five runs."""
    runs = read_runs(stem)
    if not runs:
        return None
    present = [b for b in BACKBONE_ORDER if any(r["backbone"] == b for r in runs[0])]
    values = [r[column] for r in runs[0] if r["backbone"] == present[0]]
    body = []
    for v in values:
        cells = [
            fmt(
                statistics.mean(
                    run_means(runs, lambda r, v=v, b=b: r["backbone"] == b and r[column] == v)
                )
            )
            for b in present
        ]
        body.append([v, *cells])
    return f"### {title} (mean over {len(runs)} runs)\n\n" + markdown(
        [label, *[BACKBONE_NAMES[b] for b in present]], body
    )


def table9() -> str | None:
    groups: dict[int, list[float]] = {}
    for p in glob.glob(os.path.join(RESULTS, "ablation_pool_size_K*.csv")):
        m = re.match(r"ablation_pool_size_K(\d+)(?:_p\d+)?\.csv$", os.path.basename(p))
        if m:
            groups.setdefault(int(m.group(1)), []).append(
                float(read(os.path.basename(p))[0]["mean"])
            )
    if not groups:
        return None
    body = [[str(k), fmt(statistics.mean(groups[k])), str(len(groups[k]))] for k in sorted(groups)]
    return (
        "### Table 9: pool size K (DermLIP, full DE-CLIP; mean over disjoint pools of "
        "K/20 GPT-5.2 calls)\n\n" + markdown(["K", "Avg. F1 (%)", "#Pools"], body)
    )


def table10() -> str | None:
    body = []
    for factor, setting, stem in GENERATION_ROWS:
        runs = [read(f"ablation_generation_{stem}_{n}.csv") for n in RUNS]
        values = [float(rows[0]["mean"]) for rows in runs if rows]
        if values:
            body.append([factor, setting, fmt(statistics.mean(values))])
    if not body:
        return None
    return (
        "### Table 10: sensitivity to description generation (DermLIP, mean over 5 calls)\n\n"
        + markdown(["Factor", "Setting", "Avg. F1 (%)"], body)
    )


TABLES = {
    "3": lambda: table_benchmark(
        "benchmark_accuracy", "_accuracy", "Table 3: accuracy benchmark (Derm1M protocol)"
    ),
    "4": lambda: table_benchmark(
        "benchmark_weighted_f1", "", "Table 4: weighted-F1 benchmark (nine datasets)"
    ),
    "5": table5,
    "6": table6,
    "7": lambda: table_sweep(
        "ablation_alpha_all_backbones",
        "aggregate.alpha",
        "Table 7: blending weight alpha (full DE-CLIP, average weighted-F1)",
        "alpha",
    ),
    "8": lambda: table_sweep(
        "ablation_tau_all_backbones",
        "aggregate.selection.tau",
        "Table 8: grouping threshold tau (full DE-CLIP, average weighted-F1)",
        "tau",
    ),
    "9": table9,
    "10": table10,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", choices=sorted(TABLES, key=int), help="render one table only")
    args = parser.parse_args(argv)
    for number in [args.table] if args.table else sorted(TABLES, key=int):
        text = TABLES[number]()
        if text:
            print(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
