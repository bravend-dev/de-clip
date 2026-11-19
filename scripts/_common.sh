#!/usr/bin/env bash
# Shared prologue for the experiment scripts: run from the repository root, fail fast.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN="python -m declip.run"
if [[ -n ${JOBS_FILE:-} ]]; then  # set by scripts/run_fast.sh: append each run to a --batch job file
  _emit_job() { { printf 'python -m declip.run'; printf ' %q' "$@"; printf '\n'; } >> "$JOBS_FILE"; }
  RUN=_emit_job
fi
BACKBONE="${BACKBONE:-dermlip}"   # override with BACKBONE=make bash scripts/<experiment>.sh

# Five independent K = 100 runs: run r uses GPT-5.2 calls 5r-4 .. 5r (run 1 = configs/default.yaml).
# Ablations are evaluated on every run and reported as mean +- sd; outputs of run r >= 2 get "_run<r>".
RUNS="${RUNS:-1 2 3 4 5}"
calls_set() {  # calls_set FIRST LAST -> --set value with GPT-5.2 calls FIRST..LAST
  local f=""
  for n in $(seq "$1" "$2"); do f="$f${f:+, }\"descriptions/gpt-5.2/call_k20_w3_$n.json\""; done
  echo "encode.descriptions=[$f]"
}
run_set() { calls_set $((5 * $1 - 4)) $((5 * $1)); }
run_suffix() { [[ $1 == 1 ]] && echo "" || echo "_run$1"; }
