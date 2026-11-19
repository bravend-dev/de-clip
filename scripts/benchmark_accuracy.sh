#!/usr/bin/env bash
# Accuracy benchmark, Derm1M protocol (Table 3): Default / CuPL / DE-CLIP on five backbones,
# each of the five runs (tables report the mean over the runs).
source "$(dirname "$0")/_common.sh"
for r in $RUNS; do
  $RUN --config configs/methods.yaml --backbone "${1:-all}" --set "$(run_set $r)" \
       evaluation.protocol=accuracy --out "results/benchmark_accuracy$(run_suffix $r).csv"
done
