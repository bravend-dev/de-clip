#!/usr/bin/env bash
# Grouping threshold tau (Table 8), full DE-CLIP, five backbones, each of the five runs.
source "$(dirname "$0")/_common.sh"
for r in $RUNS; do
  $RUN --config configs/ablations/tau.yaml --backbone all --set "$(run_set $r)" \
       --out "results/ablation_tau_all_backbones$(run_suffix $r).csv"
done
