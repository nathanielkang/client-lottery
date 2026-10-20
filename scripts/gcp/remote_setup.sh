#!/bin/bash
# Paper protocol on a fresh VM checkout of this directory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements.txt
mkdir -p results
PROTOCOL="${PROTOCOL:-paper}"
CMD=(./venv/bin/python scripts/run_experiment.py --protocol "$PROTOCOL" --execution-host gcp-lv1-cpu --outdir results)
if [[ -n "${T:-}" ]]; then
  CMD+=(--T "$T")
fi
if [[ -n "${SEEDS:-}" ]]; then
  # shellcheck disable=SC2206
  SEED_ARR=($SEEDS)
  CMD+=(--seeds "${SEED_ARR[@]}")
fi
printf -v CMD_STR '%q ' "${CMD[@]}"
tmux kill-session -t lv1 2>/dev/null || true
tmux new-session -d -s lv1 "cd '$ROOT' && ${CMD_STR} > results/paper.log 2>&1; echo PAPER_PROTOCOL_DONE >> results/paper.log"
echo "TMUX_LV1_STARTED protocol=$PROTOCOL T=${T:-default}"
