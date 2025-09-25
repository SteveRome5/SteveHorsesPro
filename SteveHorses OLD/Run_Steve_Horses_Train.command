#!/usr/bin/env bash
set -euo pipefail
BASE="$HOME/Desktop/SteveHorsesPro"; PY="${PYTHON:-/usr/bin/python3}"
source "$BASE/env.sh"
mkdir -p "$BASE/logs" "$BASE/history"
echo "[TRAIN] $(date '+%F %T') starting..."
"$PY" "$BASE/steve_horses_train.py" --days-back 120 --min-rows 160 2>&1 | tee -a "$BASE/logs/train_run.log"
echo "[TRAIN] done."
