#!/usr/bin/env bash
# Component ablation (Table 5): B0, D1-D4 and the Top-H rows, on each of the five runs.
source "$(dirname "$0")/_common.sh"
for r in $RUNS; do
  $RUN --config configs/ablations/component.yaml --backbone "$BACKBONE" --set "$(run_set $r)" \
       --out "results/ablation_components$(run_suffix $r).csv"
  $RUN --config configs/ablations/component_top_h.yaml --backbone "$BACKBONE" --set "$(run_set $r)" \
       --out "results/ablation_top_h$(run_suffix $r).csv"
done
