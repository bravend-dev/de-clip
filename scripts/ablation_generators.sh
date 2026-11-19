#!/usr/bin/env bash
# Description generator ablation (Table 6): five independent K=100 pools per LLM, run r = calls
# 5r-4 .. 5r of 20 descriptions each (descriptions/<generator>/call_k20_w3_<n>.json).
# The GPT-5.2 run 1 pool is the default pool (configs/default.yaml).
source "$(dirname "$0")/_common.sh"
SUFFIX=""; [[ $BACKBONE != dermlip ]] && SUFFIX="_$BACKBONE"
for g in gpt-5.2 gpt-oss-20b gemini-2.5-flash-lite gpt-oss-120b ling-3.0-flash qwen3-30b-a3b \
         mistral-small-3.2-24b deepseek-v3.2 llama-3.3-70b; do
  for r in 1 2 3 4 5; do
    files=""
    for n in $(seq $((5 * r - 4)) $((5 * r))); do
      files="$files${files:+, }\"descriptions/$g/call_k20_w3_$n.json\""
    done
    $RUN --config configs/ablations/generators.yaml --backbone "$BACKBONE" \
         --set "encode.descriptions=[$files]" \
         --out "results/ablation_generator_${g}_run$r${SUFFIX}.csv"
  done
done
