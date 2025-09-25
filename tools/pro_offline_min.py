#!/usr/bin/env python3
# offline board from local signals (no API, no extras)
import json, re, sys, datetime as dt
from pathlib import Path

BASE = Path.home() / "Desktop" / "SteveHorsesPro"
SIGS = BASE / "data" / "signals"
OUT  = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def load_rows(p):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = []
    if isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    rows = []
    for r in (data if isinstance(data, list) else []):
        race  = str(r.get("race") or r.get("r") or "")
        horse = (r.get("horse") or r.get("name") or "").strip()
        prob  = r.get("p") or r.get("prob") or r.get("win") or None
        flags = r.get("flags") or []
        rows.append({"race": race, "horse": horse, "p": prob, "flags": flags})
    return rows

# collect signals grouped by date
by_date = {}
if SIGS.exists():
    for p in SIGS.glob("*.json"):
        m = re.search(r"\d{4}-\d{2}-\d{2}", p.name)
        if m:
            by_date.setdefault(m.group(), []).append(p)

if not by_date:
    print("NO SIGNAL FILES FOUND in data/signals", file=sys.stderr)
    sys.exit(2)

day = sorted(by_date.keys())[-1]  # latest available date
files = sorted(by_date[day], key=lambda x: x.name.lower())

def html_escape(s): 
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def render_meet(track, races):
    out = []
    for rno in sorted(races):
        rows = races[rno]
        # fill uniform p when missing
        if not any(x["p"] for x in rows):
            n = max(1, len(rows))
            for x in rows: x["p"] = 1.0 / n
        # normalize per race just in case
        s = sum(x["p"] or 0.0 for x in rows) or 1.0
        for x in rows: x["p"] = (x["p"] or 0.0) / s

        out.append(f"<h3>{html_escape(track)} — Race {html_escape(rno)}</h3>")
        out.append("<table><thead><tr>"
                   "<th>#</th><th>Horse</th>"
                   "<th>Win% (Final)</th><th>Market%</th><th>Edge</th>"
                   "<th>Fair</th><th>Min Price</th><th>Notes</th><th>Source</th>"
                   "</tr></thead><tbody>")
        for i, x in enumerate(rows, 1):
            p = x['p']
            fair = 1.0/p if p>0 else 0.0
            minp = fair * 0.8
            out.append(
                "<tr>"
                f"<td>{i}</td>"
                f"<td>{html_escape(x['horse'])}</td>"
                f"<td>{p*100:0.2f}%</td>"
                f"<td>—</td>"
                f"<td>—</td>"
                f"<td>{int(fair):d}/1 • ${fair:0.2f} • {1/p:0.2f}</td>"
                f"<td>{int(minp):d}/1 • ${minp:0.2f}</td>"
                f"<td>—</td>"
                f"<td>PRO+DB(offline)</td>"
                "</tr>")
        out.append("</tbody></table>")
    return "\n".join(out)

# group rows by track → race
meets = {}
for p in files:
    track = p.stem.split("|",1)[0]
    races = {}
    for row in load_rows(p):
        races.setdefault(row["race"] or "?", []).append(row)
    if races:
        meets.setdefault(track, {}).update(races)

css = ("<style>body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;"
       "margin:24px}h1{margin:0 0 8px 0}table{border-collapse:collapse;width:100%;"
       "margin:12px 0}th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;"
       "font-size:14px}th{background:#f3f3f3}</style>")

html = [f"<!doctype html><meta charset='utf-8'>",
        f"<title>PF-35 (offline) — {day}</title>", css,
        f"<h1>PF-35 Mach++ (offline signals only) <span "
        f"style='font-size:12px;color:#666'>({day})</span></h1>",
        "<p style='color:#666'>Built from local signals only (no API). "
        "Market%, Edge will be blank; Fair/Min are derived from Win%.</p>"]

for track, races in sorted(meets.items()):
    html.append(f"<h2>{html_escape(track)}</h2>")
    html.append(render_meet(track, races))

out = OUT / f"{day}_horses_targets+full.html"
out.write_text("\n".join(html), encoding="utf-8")
print(str(out))
