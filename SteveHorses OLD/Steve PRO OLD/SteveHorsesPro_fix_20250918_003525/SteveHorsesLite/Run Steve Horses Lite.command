#!/bin/zsh
set -euo pipefail

# Your Racing API credentials (leave as-is if your shell already has them saved)
export RACING_API_USER=''"$RACING_API_USER"''
export RACING_API_PASS=''"$RACING_API_PASS"''

# Conservative defaults; tweak if you want
export MIN_PRICE_PAD=${MIN_PRICE_PAD:-0.15}   # +15% over fair
export REFRESH_SNAPSHOT=${REFRESH_SNAPSHOT:-0} # set 1 to re-pull same day

BASE="$HOME/Desktop/SteveHorsesLite"
PY="$BASE/steve_horses_lite.py"
LOGS="$BASE/logs"; OUT="$BASE/outputs"; mkdir -p "$LOGS" "$OUT"
LOG="$LOGS/run_$(date +%Y%m%d_%H%M%S).log"

/usr/bin/env python3 "$PY" >>"$LOG" 2>&1 || true
LATEST=$(ls -t "$OUT"/*_horses_lite.html 2>/dev/null | head -n 1)
[[ -n "$LATEST" ]] && open "$LATEST" || open "$LOG"
