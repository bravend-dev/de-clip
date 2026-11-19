#!/usr/bin/env bash
# Run every experiment of the paper in order. Each script can also be run on its own.
set -euo pipefail
cd "$(dirname "$0")/.."
for s in benchmark_accuracy benchmark_weighted_f1 ablation_components ablation_generators \
         ablation_alpha ablation_tau ablation_pool_size ablation_generation benchmark_baselines; do
  echo "== $s"; bash "scripts/$s.sh"
done
echo "== benchmark_baselines (accuracy protocol)"; PROTOCOL=accuracy bash scripts/benchmark_baselines.sh
echo "== dclip_subsets"; python tools/dclip_subsets.py --out results/variance_dclip.csv
