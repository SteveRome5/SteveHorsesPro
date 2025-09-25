# =========================================
# FILE: ~/Desktop/Run_Steve_Horses_Train.command
# PURPOSE: Run TRAIN fast and write predictions CSV for PRO
# =========================================
#!/usr/bin/env bash
set -Eeuo pipefail

BASE="${BASE:-$HOME/Desktop/SteveHorsesPro}"
OUT="$BASE/outputs"; LOG="$BASE/logs"; PY="${PYTHON:-/usr/bin/python3}"
mkdir -p "$OUT" "$LOG"

export FAST="${FAST:-1}"
export TRAIN_DATE="${TRAIN_DATE:-$(date +%F)}"
export MAJOR_TRACKS_ONLY="${MAJOR_TRACKS_ONLY:-Aqueduct Racetrack,Belmont Park,Saratoga Race Course,Churchill Downs,Keeneland,Gulfstream Park,Santa Anita Park,Del Mar,Oaklawn Park,Fair Grounds,Parx Racing,Woodbine,Monmouth Park,Tampa Bay Downs}"
export ALLOW_MINOR_TRACKS="${ALLOW_MINOR_TRACKS:-0}"

echo "[ENV] TRAIN_DATE=$TRAIN_DATE FAST=$FAST" | tee -a "$LOG/train_${TRAIN_DATE}.log"
"$PY" "$BASE/steve_horses_train.py" || true

csv="$OUT/predictions_${TRAIN_DATE}.csv"
[ -f "$csv" ] && echo "[OK] TRAIN wrote $csv" | tee -a "$LOG/train_${TRAIN_DATE}.log"
if command -v open >/dev/null 2>&1; then open "$csv" || true; fi