#!/bin/zsh
cd "$HOME/Desktop/SteveHorsesPro" || exit 1

# Load env once per run
if [[ -f ./env.sh ]]; then
  source ./env.sh
fi

# Write JT sidecar (real JT from API); continue even if it fails
/usr/bin/python3 jt_sidecar.py || echo "JT sidecar failed — continuing"

# Build the report
/usr/bin/python3 steve_horses_pro.py

# Open today’s HTML
open "./outputs/$(date +%F)_horses_targets+full.html"
