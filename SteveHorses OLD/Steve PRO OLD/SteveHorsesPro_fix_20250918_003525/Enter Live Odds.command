#!/bin/zsh
set -euo pipefail
PY="$HOME/Desktop/SteveHorsesPro/steve_horses_pro.py"
# Generate a runners skeleton CSV for today and open it for editing.
# Columns: track,race,number,horse,odds
/usr/bin/env python3 - "$PY" <<'PYGEN'
import sys, csv, os, json
from pathlib import Path
from datetime import date
import importlib.util

# import app functions without polluting
spec = importlib.util.spec_from_file_location("app", sys.argv[1])
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

day = date.today().isoformat()
cards, chosen, _ = app.fetch_cards_for_today(day)
odds_dir = app.ODDS
odds_path = odds_dir / f"{day}.csv"

with odds_path.open("w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["track","race","number","horse","odds"])
    for c in sorted(cards, key=lambda x:(x["track"], x["race_no"])):
        for rr in c["runners"]:
            wr.writerow([c["track"], c["race_no"], rr["no"], rr["name"], ""])
print(odds_path)
PYGEN

open "$HOME/Desktop/SteveHorsesPro/odds/$(date +%F).csv"
