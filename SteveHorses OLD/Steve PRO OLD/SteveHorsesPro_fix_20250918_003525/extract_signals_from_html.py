#!/usr/bin/env python3
# Extracts bets from today's report HTML and writes a signals CSV.
# No dependency on PRO; safe to run any time.

import sys, re, csv, os
from pathlib import Path
from datetime import date

BASE = Path.home() / "Desktop" / "SteveHorsesPro"
OUT_DIR = BASE / "outputs"
SIG_DIR = BASE / "signals"
SIG_DIR.mkdir(parents=True, exist_ok=True)

def find_today_html():
    iso = date.today().isoformat()
    p = OUT_DIR / f"{iso}_horses_targets+full.html"
    return p if p.exists() else None

def parse_money(s):
    if not s: return 0.0
    m = re.search(r"\$([\d,]+)", s)
    return float(m.group(1).replace(",","")) if m else 0.0

def parse_dec_odds(cell):
    # expects strings like "11/2 • $13.00 - 6.50" or "8/1 • $18.00 - 9.00" or "—"
    m = re.search(r"(\d+(?:\.\d+)?)\s*$", cell.replace("•"," ").strip())
    try:
        v = float(m.group(1))
        return v if v > 1 else None
    except:
        return None

def extract(html):
    # grab all rows where Bet column shows a $ amount (>0)
    rows = []
    # crude table row capture; we only care about PR, track, race, num, horse, min price, market, bet, flags
    # 1) split per race headers
    race_blocks = re.split(r"<h3>(.*?)\s+—\s+Race\s+(\d+)</h3>", html, flags=re.I)
    # race_blocks = [before, track, race, table..., track, race, table..., ...]
    for i in range(1, len(race_blocks), 3):
        track = re.sub(r"<.*?>","", race_blocks[i]).strip()
        race  = race_blocks[i+1].strip()
        table = race_blocks[i+2]
        # each <tr> ... </tr>
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.S|re.I):
            tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S|re.I)
            if len(tds) < 10:
                continue
            num_html, horse_html = tds[0], tds[1]
            market_cell, bet_cell, flags_cell = tds[7], tds[9], tds[8]
            bet_amt = parse_money(bet_cell)
            if bet_amt <= 0:
                continue
            num  = re.sub(r"<.*?>","", num_html).strip()
            horse= re.sub(r"<.*?>","", horse_html).strip()
            market_dec = parse_dec_odds(re.sub(r"<.*?>","", market_cell))
            flags= re.sub(r"<.*?>","", flags_cell).strip()
            rows.append({
                "date": date.today().isoformat(),
                "track": track,
                "race": race,
                "num": num,
                "horse": horse,
                "market_dec": f"{market_dec:.2f}" if market_dec else "",
                "stake": f"{int(bet_amt)}",
                "board_flags": flags,           # e.g. PRIME/ACTION + P/HIGH/CAP/DUTCH
                "status": "PENDING",            # will be updated by reconciler
                "result": "",
                "return": "",
                "profit": "",
            })
    return rows

def main():
    html_path = None
    if len(sys.argv) >= 2:
        html_path = Path(sys.argv[1])
    else:
        html_path = find_today_html()
    if not html_path or not html_path.exists():
        print("[extract] html not found")
        sys.exit(0)

    html = html_path.read_text(encoding="utf-8", errors="ignore")
    rows = extract(html)
    if not rows:
        print("[extract] no bets found")
        sys.exit(0)

    sig_csv = SIG_DIR / f"{date.today().isoformat()}_signals.csv"
    write_header = not sig_csv.exists()
    with sig_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header: w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[extract] wrote {sig_csv} rows={len(rows)}")

if __name__ == "__main__":
    main()