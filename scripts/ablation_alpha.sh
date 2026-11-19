#!/usr/bin/env bash
# Blending weight alpha (Table 7), full DE-CLIP, five backbones, each of the five runs.
source "$(dirname "$0")/_common.sh"
for r in $RUNS; do
  $RUN --config configs/ablations/alpha.yaml --backbone all --set "$(run_set $r)" \
       --out "results/ablation_alpha_all_backbones$(run_suffix $r).csv"
done
