#!/bin/sh
BASE="$HOME/Desktop/SteveHorses"
OUT="$BASE/outputs"
LOG="$BASE/logs/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUT" "$BASE/logs"
/usr/bin/env python3 "$BASE/steve_horses_board.py" >>"$LOG" 2>&1
LATEST=$(ls -t "$OUT"/*.html 2>/dev/null | head -n 1)
if [ -n "$LATEST" ]; then
  open "$LATEST"
else
  open "$LOG"
fi
