#!/bin/zsh
set -euo pipefail
/usr/bin/env python3 "$HOME/Desktop/SteveHorsesPro/overlay_live_odds.py" >>"$HOME/Desktop/SteveHorsesPro/logs/live_overlay_$(date +%Y%m%d_%H%M%S).log" 2>&1 || true
