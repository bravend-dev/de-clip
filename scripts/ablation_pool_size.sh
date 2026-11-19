#!/usr/bin/env bash
# Pool size K = 20 n (Table 9), n = 1 .. 10: five disjoint pools per K, pool j = GPT-5.2 calls
# (j-1)n+1 .. jn (K = 100 gives the five main runs). Pool 1 -> ablation_pool_size_K<K>.csv,
# pool j >= 2 -> ablation_pool_size_K<K>_p<j>.csv; tables report mean +- sd over the five pools.
source "$(dirname "$0")/_common.sh"
for n in $(seq 1 10); do
  for j in 1 2 3 4 5; do
    s=""; [[ $j != 1 ]] && s="_p$j"
    $RUN --config configs/ablations/pool_size.yaml --backbone "$BACKBONE" \
         --set "$(calls_set $(((j - 1) * n + 1)) $((j * n)))" \
         --out "results/ablation_pool_size_K$((n * 20))$s.csv"
  done
done
