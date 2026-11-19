#!/usr/bin/env bash
# Baselines of Tables 3-4 on five backbones: DCLIP, WaffleCLIP (5 seeds) and the single-template
# and 80-ImageNet-template variants of the class-name baseline.
# PROTOCOL=accuracy bash scripts/benchmark_baselines.sh   for the Derm1M accuracy protocol.
source "$(dirname "$0")/_common.sh"
PROTOCOL="${PROTOCOL:-weighted_f1}"
SUFFIX=""; [[ $PROTOCOL == accuracy ]] && SUFFIX="_accuracy"
$RUN --config configs/baselines/dclip.yaml --backbone "${1:-all}" \
     --set evaluation.protocol=$PROTOCOL --out "results/baseline_dclip$SUFFIX.csv"
for seed in 0 1 2 3 4; do
  $RUN --config configs/baselines/waffle.yaml --backbone "${1:-all}" \
       --set evaluation.protocol=$PROTOCOL \
             "encode.descriptions=[\"descriptions/baselines/waffle_count15_seed$seed.json\"]" \
       --out "results/baseline_waffle_seed$seed$SUFFIX.csv"
done
for t in single imagenet80; do
  $RUN --config configs/baselines/template_$t.yaml --backbone "${1:-all}" \
       --set evaluation.protocol=$PROTOCOL --out "results/baseline_template_$t$SUFFIX.csv"
done
