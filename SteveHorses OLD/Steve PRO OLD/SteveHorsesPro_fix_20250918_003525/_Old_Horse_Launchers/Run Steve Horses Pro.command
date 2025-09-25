#!/bin/zsh
set -euo pipefail
# Pass through saved creds (or the ones baked by the installer)
export RACING_API_USER="${RACING_API_USER:-WQaKSMwgmG8GnbkHgvRRCT0V}"
export RACING_API_PASS="${RACING_API_PASS:-McYBoQViXSPvlNcvxQi1Z1py}"

# Conservative default (pad fair price by 15%)
export MIN_PRICE_PAD="${MIN_PRICE_PAD:-0.15}"

BASE="$HOME/Desktop/SteveHorsesPro"
PY="$BASE/steve_horses_pro.py"
OUT="$BASE/outputs"; LOGS="$BASE/logs"
mkdir -p "$OUT" "$LOGS"
LOG="$LOGS/run_$(date +%Y%m%d_%H%M%S).log"

/usr/bin/env python3 "$PY" >>"$LOG" 2>&1 || true
LATEST="$(ls -t "$OUT"/*.html 2>/dev/null | head -n 1 || true)"
[[ -n "$LATEST" ]] && open "$LATEST" || open "$LOG"
