#!/bin/zsh
cd "$HOME/Desktop/SteveHorsesPro" || exit 1
# Optional: uncomment and fill in if you need credentials/env
# export RACINGAPI_USER="your_user"
# export RACINGAPI_PASS="your_pass"
# export PRO_MODE=1
/usr/bin/python3 steve_horses_pro.py
# Open today's if present, else the newest available
FILE="outputs/$(date +%F)_horses_targets+full.html"
if [ -f "$FILE" ]; then
  open "$FILE"
else
  LATEST="$(ls -t outputs/*_horses_targets+full*.html 2>/dev/null | head -n1)"
  [ -n "$LATEST" ] && open "$LATEST" || osascript -e 'display notification "No report generated" with title "PRO"'
fi
