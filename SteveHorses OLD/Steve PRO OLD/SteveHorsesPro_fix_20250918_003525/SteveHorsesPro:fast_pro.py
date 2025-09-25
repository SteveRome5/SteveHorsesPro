# fast_pro.py — builds today's report fast using your existing module
from datetime import date
from pathlib import Path
import webbrowser

import steve_horses_pro as shp  # your PF-35 Mach++ v3.8-handicap++learn (WHY) file

iso = date.today().isoformat()

# Build today’s cards + scratches using your existing helpers
cards, scr_summary, auto_summary, scr_details = shp.build_cards_and_scratches(iso)

# Render report with your existing HTML builder
html = shp.build_report(cards, iso, scr_summary, auto_summary, scr_details=scr_details)

# Write to outputs/latest.html (and also timestamped file if you want)
out_dir = shp.OUT_DIR
out_dir.mkdir(parents=True, exist_ok=True)
latest = out_dir / "latest.html"
latest.write_text(html, encoding="utf-8")

print(f"[ok] Wrote {latest}")
try:
    if latest.exists():
        webbrowser.open("file://" + str(latest))
except Exception:
    pass