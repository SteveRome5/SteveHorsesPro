#!/bin/zsh
set -euo pipefail

BASE="$HOME/Desktop/SteveHorsesPro"
cd "$BASE"

export RACINGAPI_USER='WQaKSMwgmG8GnbkHgvRRCT0V'
export RACINGAPI_PASS='McYBoQViXSPvlNcvxQi1Z1py'
export PYTHONWARNINGS="ignore:NotOpenSSLWarning"

LOG="$BASE/logs/train_cron.log"
mkdir -p "$BASE/logs"

echo "[train-cron] $(date) start" | tee -a "$LOG"

if [ -f "$BASE/steve_horses_train.py" ]; then
  /usr/bin/python3 "$BASE/steve_horses_train.py" --harvest-days 2 2>&1 | tee -a "$LOG"
  /usr/bin/python3 "$BASE/steve_horses_train.py" --train-only --days-back 120 2>&1 | tee -a "$LOG"
fi

echo "[train-cron] $(date) done" | tee -a "$LOG"
