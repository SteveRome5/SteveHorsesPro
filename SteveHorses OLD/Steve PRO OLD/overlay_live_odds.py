#!/usr/bin/env python3
from __future__ import annotations
import re, csv, html, webbrowser
from pathlib import Path
from datetime import date

BASE = Path.home()/ "Desktop" / "SteveHorsesPro"
OUT  = BASE / "outputs"
LIVE = BASE / "live"

def parse_odds_to_dec(s: str):
    if not s: return None
    t = str(s).strip().lower().replace(' ', '')
    if t in ("evs","even","evens"): return 2.0
    if t.endswith('d'):
        try: return float(t[:-1])
        except: pass
    m = re.fullmatch(r'(\d+)\s*[/\-:]\s*(\d+)', t)  # 7/2 or 7-2
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        return 1.0 + (num/den) if den>0 else None
    if re.fullmatch(r'[+-]?\d+', t):  # American
        a = int(t)
        return (1 + a/100.0) if a>0 else (1 + 100.0/abs(a))
    try:
        v = float(t)
        return v if v>1 else None
    except: return None

def dec_to_frac(dec: float) -> str:
    if not dec or dec<=1: return "—"
    v = dec - 1.0
    best=(9e9,"—")
    for den in (1,2,3,4,5,6,7,8,9,10,12,14,16,20):
        num = round(v*den); err = abs(v - num/den)
        if err<best[0]: best=(err,f"{int(num)}-{int(den)}")
    return best[1]

def load_targets_from_html(p: Path):
    """Read the Top Win Targets table from the model output."""
    txt = p.read_text(encoding="utf-8", errors="ignore")
    # isolate the first table after 'Top Win Targets'
    m = re.search(r"Top Win Targets.*?<table.*?>(?P<table>.*?)</table>", txt, re.S|re.I)
    if not m: return []
    table = m.group("table")
    rows=[]
    # <tr><td>Track</td><td>Race</td><td class='play'>Horse</td><td ...>12.3%</td><td ...>frac / $2 / 9.20d</td></tr>
    rre = re.compile(
        r"<tr>\s*<td>(?P<track>.*?)</td>\s*"
        r"<td>(?P<race>\d+)</td>\s*"
        r"<td[^>]*>(?P<horse>.*?)</td>\s*"
        r"<td[^>]*>(?P<pct>[\d.]+)%</td>\s*"
        r"<td[^>]*>(?P<price>[^<]+)</td>\s*</tr>", re.S|re.I)
    for m in rre.finditer(table):
        track = re.sub("<.*?>","",m.group("track")).strip()
        race  = int(m.group("race"))
        horse = html.unescape(re.sub("<.*?>","",m.group("horse")).strip())
        pct   = float(m.group("pct"))/100.0
        price_block = m.group("price")
        md = None
        dm = re.search(r'([0-9.]+)\s*d', price_block)
        if dm:
            md = float(dm.group(1))  # min decimal price to bet
        rows.append({"track":track,"race":race,"horse":horse,"p":pct,"min_dec":md})
    return rows

def main():
    today = date.today().isoformat()
    # pick newest html
    htmls = sorted(OUT.glob("*.html"), key=lambda p:p.stat().st_mtime, reverse=True)
    if not htmls:
        print("No model output in outputs/. Run your usual launcher first.")
        return
    src = htmls[0]
    targets = load_targets_from_html(src)
    if not targets:
        print("Couldn't find 'Top Win Targets' in", src.name)
        return

    # Prepare or read today's odds CSV
    odds_csv = LIVE / f"{today}_odds.csv"
    if not odds_csv.exists():
        odds_csv.write_text("track,race,horse,live_odds\n","utf-8")
        # seed with targets for convenience
        with odds_csv.open("a", newline="", encoding="utf-8") as f:
            w=csv.writer(f)
            for r in targets:
                w.writerow([r["track"], r["race"], r["horse"], ""])  # fill later
        # open for the user once, then exit
        print("Created:", odds_csv)
        print("Fill live_odds as 7-2, 15-1, +250 or decimal like 6.80, then re-run the overlay.")
        try:
            import subprocess; subprocess.run(["open", str(odds_csv)])
        except: pass
        return

    # read odds
    book={}
    with odds_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k=(row["track"].strip(), int(row["race"]), row["horse"].strip())
            book[k] = parse_odds_to_dec(row.get("live_odds",""))

    # compute overlay
    KELLY_CAP = float((Path.home()/".kelly_cap").read_text().strip()) if (Path.home()/".kelly_cap").exists() else 0.10

    enriched=[]
    for r in targets:
        k=(r["track"], r["race"], r["horse"])
        dec = book.get(k)
        p   = r["p"]
        fair = (1.0/p) if p>0 else None
        min_dec = r["min_dec"]
        edge = None
        ev   = None
        kelly=None
        play="PASS"
        if dec and min_dec:
            edge = (dec - min_dec)/min_dec
            ev   = p*dec - 1.0
            b = dec - 1.0
            kelly_raw = (p*b - (1-p))/b if b>0 else -1
            kelly = max(0.0, min(KELLY_CAP, kelly_raw))
            if dec >= min_dec: play="PLAY"
        enriched.append({**r, "live_dec":dec, "fair":fair, "edge":edge, "ev":ev, "kelly":kelly, "play":play})

    # write overlay html
    dst = OUT / f"{today}_horses_pro_live.html"
    parts=[]
    parts.append(f"<!doctype html><meta charset='utf-8'><title>Steve’s Horses Pro + Live — {today}</title>")
    parts.append("""
    <style>
    :root{--bg:#0f2027;--fg:#e6f1f5;--muted:#87a0ab;--row:#122a33;--play:#34d399;--warn:#f59e0b}
    body{background:var(--bg);color:var(--fg);font-family:-apple-system,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:22px}
    h1{margin:0 0 8px}.sub{color:var(--muted);margin:0 0 14px}
    table{width:100%;border-collapse:collapse;font-size:14px}
    th,td{padding:10px 8px} th{color:#a3c0cb;border-bottom:1px solid #23424d}
    tbody tr:nth-child(odd){background:var(--row)}
    .right{text-align:right}.mono{font-variant-numeric:tabular-nums}
    .play{color:var(--play);font-weight:700}
    .badge{display:inline-block;padding:2px 6px;border:1px solid #2a4c58;border-radius:6px;color:#9fb9c4;font-size:12px;margin-left:8px}
    </style>
    """)
    parts.append(f"<h1>Steve’s Horses Pro + Live — {today}</h1>")
    parts.append(f"<div class='sub'>Overlay uses your latest model page <span class='badge'>{html.escape(src.name)}</span> and odds from <span class='badge'>{html.escape(odds_csv.name)}</span>. PLAY shows only when live price ≥ min price.</div>")
    parts.append("<table><thead><tr><th>Track</th><th>Race</th><th>Horse</th>"
                 "<th class='right'>Model Win%</th><th class='right'>Min dec</th><th class='right'>Live dec</th>"
                 "<th class='right'>Edge%</th><th class='right'>EV%</th><th class='right'>Kelly%</th><th class='right'>Play</th></tr></thead><tbody>")
    # sort: PLAY first by edge desc, then others
    enriched.sort(key=lambda r:(r["play"]!="PLAY", -(r["edge"] or -9)))
    for r in enriched:
        p=r["p"]*100
        md=r["min_dec"]; ld=r["live_dec"]; ed=r["edge"]; ev=r["ev"]; ky=r["kelly"]
        parts.append("<tr>"
                     f"<td>{html.escape(r['track'])}</td>"
                     f"<td class='right mono'>{r['race']}</td>"
                     f"<td class='play'>{html.escape(r['horse'])}</td>"
                     f"<td class='right mono'>{p:0.1f}%</td>"
                     f"<td class='right mono'>{md:0.2f}d</td>"
                     f"<td class='right mono'>{'' if ld is None else f'{ld:0.2f}d'}</td>"
                     f"<td class='right mono'>{'' if ed is None else f'{ed*100:0.1f}%'}</td>"
                     f"<td class='right mono'>{'' if ev is None else f'{ev*100:0.1f}%'}</td>"
                     f"<td class='right mono'>{'' if ky is None else f'{ky*100:0.1f}%'}</td>"
                     f"<td class='right mono'>{r['play']}</td>"
                     "</tr>")
    parts.append("</tbody></table>")

    dst.write_text("".join(parts), encoding="utf-8")
    try: webbrowser.open(f"file://{dst}")
    except: pass
    print(dst)

if __name__=="__main__":
    main()
