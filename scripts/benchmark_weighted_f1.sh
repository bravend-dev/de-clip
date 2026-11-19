#!/usr/bin/env bash
# Weighted-F1 benchmark (Table 4): Default / CuPL / DE-CLIP on five backbones, nine datasets,
# each of the five runs (tables report the mean over the runs).
source "$(dirname "$0")/_common.sh"
for r in $RUNS; do
  $RUN --config configs/methods.yaml --backbone "${1:-all}" --set "$(run_set $r)" \
       --out "results/benchmark_weighted_f1$(run_suffix $r).csv"
done
