from __future__ import annotations
from pathlib import Path
from datetime import date
import webbrowser, html

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "outputs"
LOG_DIR = BASE / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRACKS = ["Saratoga", "Del Mar", "Santa Anita", "Gulfstream Park", "Keeneland"]

# Valid dict. No stray braces, no set literals.
KNOWN_TRACK_COORDS = {
    "Saratoga": (43.083, -73.785),
    "Del Mar": (32.973, -117.259),
    "Santa Anita": (34.136, -118.039),
    "Gulfstream Park": (25.981, -80.143),
    "Keeneland": (38.040, -84.605),
}

CSS = """
<style>
  body{background:#0f2a2a;color:#e6f0f0;font-family:-apple-system,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:1200px;margin:24px auto;padding:0 12px}
  h1{font-size:28px;margin:0 0 8px}
  .sub{color:#9cd3cf;margin:0 0 16px}
  table{width:100%;border-collapse:collapse}
  th,td{padding:10px 12px}
  thead th{font-size:12px;letter-spacing:.04em;color:#9cd3cf;text-align:left;border-bottom:1px solid #1f3c3c}
  tbody tr:nth-child(odd){background:#123535}
  .right{text-align:right}
  .play{color:#32d296;font-weight:600}
  .pass{color:#9aa9a9}
</style>
"""

def build() -> Path:
    today = date.today()
    parts = []
    parts.append("<!doctype html><meta charset='utf-8'>")
    parts.append(CSS)
    parts.append("<div class='wrap'>")
    parts.append(f"<h1>Steve's Horses Board — {today:%Y-%m-%d}</h1>")
    parts.append("<p class='sub'>Data feed not connected yet. Placeholder proves the launcher + script are healthy.</p>")
    parts.append("<table><thead><tr>")
    for h in ["Time","Track","Race","BetLine","Steve's Line","EV%","Edge%","Play"]:
        parts.append(f"<th>{html.escape(h)}</th>")
    parts.append("</tr></thead><tbody>")
    for trk in TRACKS:
        parts.extend([
            "<tr>",
            "<td>—</td>",
            f"<td>{html.escape(trk)}</td>",
            "<td>—</td>",
            "<td class='right'>—</td>",
            "<td class='right'>—</td>",
            "<td class='right'>—</td>",
            "<td class='right'>—</td>",
            "<td class='pass'>NO DATA</td>",
            "</tr>"
        ])
    parts.append("</tbody></table>")
    parts.append("<p class='sub' style='margin-top:12px'>Tracks configured: " + ", ".join(TRACKS) + ".</p>")
    parts.append("</div>")
    out = OUT_DIR / f"{today:%Y-%m-%d}_horses_board.html"
    out.write_text("".join(parts), encoding="utf-8")
    return out

if __name__ == "__main__":
    out = build()
    try:
        webbrowser.open(f"file://{out}")
    except Exception:
        pass
    print(out)
