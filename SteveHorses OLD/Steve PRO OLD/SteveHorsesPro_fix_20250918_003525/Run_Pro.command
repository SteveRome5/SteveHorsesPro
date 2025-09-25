#!/bin/zsh
set -euo pipefail

cd "$HOME/Desktop/SteveHorsesPro" || exit 1

# Load env
source "$HOME/.zprofile" >/dev/null 2>&1 || true
source "$HOME/.racing_api.env" >/dev/null 2>&1 || true

export PYTHONUNBUFFERED=1
echo "[RUN] $(date '+%F %T')  Steve Horses Pro"

python3 -u steve_horses_pro.py

# Open today’s report
OUT="outputs/$(date +%F)_horses_targets+full.html"
if [ -f "$OUT" ]; then
  open -a "Safari" "$OUT" 2>/dev/null || open "$OUT"
else
  echo "[WARN] No HTML found at $OUT"
fi

# Keep window open so you can see logs/errors
read -n 1 -s -r "?Press any key to close…"
