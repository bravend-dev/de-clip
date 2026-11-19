#!/usr/bin/env bash
# Same runs as run_all.sh, but fast: every run becomes one line of a `declip.run --batch` job file
# (models, image embeddings and text cache are loaded once per process instead of once per run),
# and the job file is split over parallel processes:
#   * one process for the five-backbone (--backbone all) runs,
#   * WORKERS - 1 processes for the single-backbone runs,
#   * the pool-size runs (largest pools) afterwards in POOL_WORKERS process(es), to bound RAM.
# Defaults fit one 24 GB GPU, 32 GB RAM and 16 cores; lower WORKERS if a process gets killed.
#   bash scripts/run_fast.sh                  # everything
#   RUNS=1 WORKERS=3 bash scripts/run_fast.sh # run 1 only, fewer processes
# RUNS and BACKBONE are passed to the experiment scripts as for run_all.sh; the baselines run under
# both protocols. tools/dclip_subsets.py is not included (run it separately).
# DRY_RUN=1 only writes and splits the job file.
# Job files, per-process logs and the list of failed runs go to logs/run_fast/<timestamp>/.
set -euo pipefail
cd "$(dirname "$0")/.."
WORKERS="${WORKERS:-4}"
POOL_WORKERS="${POOL_WORKERS:-1}"
CONDA_ENV="${CONDA_ENV:-declip}"
SCRIPTS="${SCRIPTS:-benchmark_accuracy benchmark_weighted_f1 ablation_components ablation_generators
  ablation_alpha ablation_tau ablation_pool_size ablation_generation benchmark_baselines}"

# Use the conda env unless it is already active (another env may import another declip copy).
if [[ ${CONDA_DEFAULT_ENV:-} != "$CONDA_ENV" ]] && command -v conda > /dev/null; then
  prefix="$(conda info --base)/envs/$CONDA_ENV"
  [[ -x $prefix/bin/python ]] && export PATH="$prefix/bin:$PATH"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
       MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
python -c "import declip, os, sys; assert os.path.samefile(os.path.dirname(declip.__file__), 'declip'), \
  f'python {sys.executable} imports declip from {declip.__file__}, not from this repository'"

DIR="logs/run_fast/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIR"
export JOBS_FILE="$DIR/jobs.txt"
: > "$JOBS_FILE"
for s in $SCRIPTS; do bash "scripts/$s.sh"; done
[[ " $SCRIPTS " == *" benchmark_baselines "* ]] && PROTOCOL=accuracy bash scripts/benchmark_baselines.sh
echo "$(wc -l < "$JOBS_FILE") runs -> $JOBS_FILE"

# Split: pool-size runs, --backbone all runs, and the rest in contiguous chunks (sorted by backbone
# so consecutive lines of a process reuse the loaded model).
python - "$JOBS_FILE" "$DIR" "$WORKERS" "$POOL_WORKERS" <<'PY'
import shlex, sys
jobs, out, workers, pool_workers = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
lines = [l.rstrip("\n") for l in open(jobs) if l.strip()]
def backbone(l):
    a = shlex.split(l)
    return a[a.index("--backbone") + 1] if "--backbone" in a else "dermlip"
pool = [l for l in lines if "configs/ablations/pool_size.yaml" in l]
every = [l for l in lines if l not in pool and backbone(l) == "all"]
rest = sorted((l for l in lines if l not in pool and backbone(l) != "all"), key=backbone)
def split(xs, n, tag):
    n = max(1, n); size = -(-len(xs) // n) if xs else 0
    for i in range(n):
        chunk = xs[i * size:(i + 1) * size]
        if chunk:
            open(f"{out}/{tag}_{i}.txt", "w").write("\n".join(chunk) + "\n")
if every and workers > 1:
    split(every, 1, "main_all")
    split(rest, workers - 1, "main")
else:
    split(every + rest, workers, "main")
split(pool, pool_workers, "pool")
PY

if [[ -n ${DRY_RUN:-} ]]; then wc -l "$DIR"/{main,pool}*.txt; exit 0; fi

run_phase() {  # run_phase TAG: one --batch process per shard file of that phase, in parallel
  local f pids=()
  for f in "$DIR/$1"_*.txt; do
    [[ -e $f ]] || continue
    python -m declip.run --batch "$f" > "${f%.txt}.out" 2> "${f%.txt}.err" &
    pids+=($!)
    echo "  $(basename "$f"): $(wc -l < "$f") runs, pid $!"
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
}
start=$SECONDS
echo "== main phase"; run_phase main
echo "== pool-size phase"; run_phase pool
cat "$DIR"/*.out > "$DIR/done.log"
# Runs not reported "ok" (failed, or never reached because a process was killed) -> failed.txt.
python - "$JOBS_FILE" "$DIR/done.log" "$DIR/failed.txt" <<'PY'
import shlex, sys
done = {l.split()[1] for l in open(sys.argv[2]) if l.startswith("ok ")}
left = [l for l in open(sys.argv[1]) if l.strip()
        and (a := shlex.split(l))[a.index("--out") + 1] not in done]
open(sys.argv[3], "w").writelines(left)
PY
total=$(wc -l < "$JOBS_FILE")
fail=$(wc -l < "$DIR/failed.txt")
echo "ok $((total - fail)) / $total in $(((SECONDS - start) / 60)) min; logs in $DIR"
if ((fail > 0)); then
  echo "$fail runs failed or did not finish: $DIR/failed.txt (errors in $DIR/*.err)."
  echo "Re-run them with: python -m declip.run --batch $DIR/failed.txt"
  exit 1
fi
