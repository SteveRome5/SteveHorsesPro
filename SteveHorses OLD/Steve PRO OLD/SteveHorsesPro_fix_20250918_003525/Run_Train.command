#!/bin/zsh
set -euo pipefail

cd "$HOME/Desktop/SteveHorsesPro" || exit 1

source "$HOME/.zprofile" >/dev/null 2>&1 || true
source "$HOME/.racing_api.env" >/dev/null 2>&1 || true

export PYTHONUNBUFFERED=1
echo "[RUN] $(date '+%F %T')  Steve Horses Train"

python3 -u steve_horses_train.py

read -n 1 -s -r "?Done. Press any key to close…"
