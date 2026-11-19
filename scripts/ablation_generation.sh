#!/usr/bin/env bash
# Sensitivity to description generation (Table 10): generator, description length and
# descriptions per call, one LLM call per class, full DE-CLIP. Five independent calls per setting
# (descriptions/<generator>/call_k<K>_w<max words>_<n>.json, n = 1..5); tables report mean +- sd.
source "$(dirname "$0")/_common.sh"
SUFFIX=""; [[ $BACKBONE != dermlip ]] && SUFFIX="_$BACKBONE"
for f in gpt-5.2/call_k20_w3 gpt-4.1-mini/call_k20_w3 gpt-5-mini/call_k20_w3 \
         gemini-3-flash-preview/call_k20_w3 \
         gpt-5-mini/call_k20_w1 gpt-5-mini/call_k20_w2 gpt-5-mini/call_k20_w4 \
         gpt-5-mini/call_k20_w5 gpt-5-mini/call_k20_w6 \
         gpt-5-mini/call_k5_w3 gpt-5-mini/call_k10_w3 gpt-5-mini/call_k40_w3; do
  for n in 1 2 3 4 5; do
    $RUN --config configs/ablations/generation.yaml --backbone "$BACKBONE" \
         --set "encode.descriptions=[\"descriptions/${f}_$n.json\"]" \
         --out "results/ablation_generation_${f//\//_}_$n${SUFFIX}.csv"
  done
done
