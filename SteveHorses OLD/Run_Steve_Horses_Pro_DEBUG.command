#!/bin/zsh
set -euo pipefail

BASE="$HOME/Desktop/SteveHorsesPro"
source "$BASE/env.sh" 2>/dev/null || true
export PRO_MODE=1 PRO_DEBUG=1

/usr/bin/python3 -m py_compile "$BASE/steve_horses_pro.py"
/usr/bin/python3 "$BASE/steve_horses_pro.py"

# Print a short diagnostic tail with key markers (same lines you saw)
tail -n 120 "$BASE/logs/pro_run.log" | egrep -i "PLUMB|horse-db|GET fail" || true

OUT="$BASE/outputs/$(date +%F)_horses_targets+full.html"
[ -f "$OUT" ] && open "$OUT" || open "$BASE/outputs"
