#!/usr/bin/env python3
# PF-35 locked — voting report (one pick per race), program numbers, robust race parsing
from __future__ import annotations
import os, ssl, json, html, base64, re, hashlib
from pathlib import Path
from datetime import date
from urllib.request import Request, urlopen
from urllib.parse import urlencode

VERSION = "PF-35 locked"

# Tracks you care about
TRACKS = ["Saratoga","Del Mar","Santa Anita Park","Gulfstream Park","Keeneland","Parx Racing","Finger Lakes"]
TRACK_IDS = {"SAR","DMR","SA","GP","KEE","PRX","FL"}

# Tunables (kept simple)
SHOW_NUMBERS = os.getenv("SHOW_NUMBERS","1") == "1"
TIE_MAX      = int(os.getenv("TIE_MAX","1"))       # 1 = exactly one pick per race
MIN_PAD      = float(os.getenv("MIN_PRICE_PAD","0.15"))  # 15% above fair
MIN_WIN      = float(os.getenv("MIN_WIN_PCT","0.10"))    # filter low-prob picks

# Folders
HOME = Path.home()
BASE = HOME/"Desktop"/"SteveHorsesPro"
OUT  = BASE/"outputs"
LOGS = BASE/"logs"
for p in (BASE,OUT,LOGS): p.mkdir(parents=True, exist_ok=True)

# API creds
RUSER = os.getenv("RACING_API_USER","").strip()
RPASS = os.getenv("RACING_API_PASS","").strip()
USE_API = bool(RUSER and RPASS)

CTX = ssl.create_default_context()

def _get(path, params=None):
    url = "https://api.theracingapi.com" + path + ("?" + urlencode(params) if params else "")
    req = Request(url, headers={"User-Agent":"Mozilla/5.0"})
    if USE_API:
        tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
        req.add_header("Authorization", "Basic " + tok)
    with urlopen(req, timeout=30, context=CTX) as r:
        return json.loads(r.read().decode("utf-8"))

def g(d,*ks, default=None):
    for k in ks:
        if isinstance(d,dict) and k in d and d[k] not in (None,""):
            return d[k]
    return default

def parse_rno(v, idx):
    if v is None: return idx
    s = str(v)
    m = re.search(r"\d+", s)
    if m:
        try: return int(m.group(0))
        except: pass
    try: return int(v)
    except: return idx

def parse_frac_or_dec(s):
    if s is None: return (None,None)
    t = str(s).strip().lower()
    if t in ("evs","even","evens"): return (2.0, 0.5)
    m = re.fullmatch(r"(\d+)\s*[/\-:]\s*(\d+)", t)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den>0: 
            dec = 1.0 + num/den
            return (dec, 1.0/dec)
    try:
        dec = float(t)
        if dec>1.0: return (dec, 1.0/dec)
    except: pass
    return (None,None)

def stable_seed(*parts) -> int:
    s = "|".join(map(str,parts))
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:8], 'big')

def p35_probs(runners, race_key):
    n = max(1, len(runners))
    out = []
    for r in runners:
        ml = g(r,"ml","morning_line","morningLine","ml_decimal","morningLineDecimal","morning_line_decimal")
        if isinstance(ml,(int,float)): dec = float(ml) if float(ml)>1 else None
        else: dec,_ = parse_frac_or_dec(ml)
        p = (1.0/dec) if dec and dec>1 else (1.0/n)
        post = g(r,"program_number","program","number","saddle","saddle_number","pp","post_position","horse_number")
        try: bias = 0.005 if 2 <= int(post) <= 7 else -0.005 if post not in (None,"") else 0.0
        except: bias = 0.0
        pri = ((stable_seed(race_key, g(r,"horse_name","name","runner_name","horse","runner","horseName","id")) % 1000)/1000 - 0.5)*0.01
        out.append(max(1e-6, p + bias + pri))
    s = sum(out)
    return [x/s for x in out]

def extract_races(meet_json):
    races = []
    cand = g(meet_json, "races","data","entries","cards", default=[])
    if isinstance(cand, dict): cand = cand.get("races") or cand.get("data") or cand.get("entries") or []
    if not isinstance(cand, list): return races
    for idx,rx in enumerate(cand,1):
        rno = parse_rno(g(rx,"race_number","raceNo","number","race","race_id"), idx)
        runners = g(rx,"runners","horses","entries","starters","fields", default=[])
        if isinstance(runners, dict):
            runners = runners.get("runners") or runners.get("entries") or list(runners.values())
        if not isinstance(runners, list): runners = []
        clean=[]
        for rr in runners:
            clean.append(rr if isinstance(rr,dict) else {"name":str(rr)})
        races.append({"race_number": rno, "runners": clean})
    races.sort(key=lambda x: x["race_number"])
    return races

def choose_meets_today():
    today = date.today().isoformat()
    js = _get("/v1/north-america/meets", {"start_date":today,"end_date":today}) if USE_API else {"meets":[]}
    chosen=[]
    for m in js.get("meets", []):
        trk = g(m,"track_name","track","meeting","course","name", default="")
        tid = g(m,"track_id","trackId","id", default="")
        if any(t.lower() in trk.lower() for t in TRACKS) or (tid in TRACK_IDS):
            chosen.append((trk or tid, g(m,"meet_id","id","meetId")))
    return chosen

def load_entries(meet_id):
    return _get(f"/v1/north-america/meets/{meet_id}/entries") if USE_API else {}

def dec_to_frac(dec):
    if not dec or dec<=1: return "—"
    v=dec-1.0; best="—"; err=9e9
    for den in (1,2,3,4,5,6,7,8,9,10,12,14,16,20,32):
        num=round(v*den); e=abs(v-num/den)
        if e<err: err=e; best=f"{int(num)}-{int(den)}"
    return best

def price_triplet(dec):
    return f"{dec_to_frac(dec)} / ${2*dec:0.2f} / {dec:0.2f}d"

def one_pick_rows(cards):
    rows=[]
    for track, races in cards:
        for r in races:
            if not r["runners"]: continue
            probs = p35_probs(r["runners"], (track, r["race_number"]))
            j = max(range(len(probs)), key=lambda i: probs[i])
            p = probs[j]
            if p < MIN_WIN:   # floor
                continue
            nm = g(r["runners"][j], "horse_name","name","runner_name","horse","runner","horseName", default="?")
            no = str(g(r["runners"][j], "program_number","program","number","saddle","saddle_number","pp","post_position","horse_number", default="")).strip()
            label = f"#{no} {nm}" if (SHOW_NUMBERS and no) else nm
            fair_dec = 1.0/max(p,1e-6)
            min_dec  = fair_dec*(1.0+MIN_PAD)
            rows.append((track, r["race_number"], label, p, price_triplet(min_dec)))
    rows.sort(key=lambda x: (x[0], x[1], -x[3]))
    return rows

def build_html(rows, status):
    today = date.today().isoformat()
    css = """
    <style>
      :root{--bg:#0f2027;--fg:#e6f1f5;--muted:#87a0ab;--row:#122a33;--play:#34d399}
      body{background:var(--bg);color:var(--fg);font-family:-apple-system,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:24px}
      h1{margin:0 0 8px;font-weight:800}.sub{color:var(--muted);margin:0 0 12px}
      .badge{display:inline-block;padding:2px 6px;border:1px solid #2a4c58;border-radius:6px;color:#9fb9c4;font-size:12px;margin-left:8px}
      table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 8px;text-align:left}
      th{color:#a3c0cb;border-bottom:1px solid #23424d;font-weight:600}tbody tr:nth-child(odd){background:var(--row)}
      .right{text-align:right}.mono{font-variant-numeric:tabular-nums}.play{color:var(--play);font-weight:700}
    </style>
    """
    head = f"<h1>Steve's Horses Pro — {today}</h1>"
    sub  = ("<div class='sub'>Tracks: Saratoga, Del Mar, Santa Anita Park, Gulfstream Park, Keeneland, Parx Racing, Finger Lakes. "
            "Data via The Racing API."
            f"<span class='badge'>{VERSION}</span>"
            f"<span class='badge'>Min price to bet = fractional / $2‑payout / decimal</span>"
            f"<span class='badge'>{html.escape(status)}</span></div>")
    body = ["<div class='track'>Top Win Targets</div>",
            "<table><thead><tr><th>Track</th><th>Race</th><th>Horse</th><th class='right'>Model Win%</th><th class='right'>Min price to bet (frac / $2 / dec)</th></tr></thead><tbody>"]
    for trk,rno,label,p,trip in rows:
        body.append(f"<tr><td>{html.escape(trk)}</td><td>{rno}</td><td class='play'>{html.escape(label)}</td>"
                    f"<td class='right mono'>{p*100:0.1f}%</td><td class='right mono'>{trip}</td></tr>")
    body.append("</tbody></table>")
    return "<!doctype html><meta charset='utf-8'>"+css+head+sub+"".join(body)

def main():
    if not USE_API:
        out = OUT/f"{date.today().isoformat()}_horses_targets.html"
        out.write_text(build_html([], "Missing API creds"), encoding="utf-8")
        print(out); return

    chosen = choose_meets_today()
    rows=[]; pulled=[]
    for trk, mid in chosen:
        try:
            mj = load_entries(mid) or {}
            races = extract_races(mj)
            if not races: continue
            pulled.append(trk)
            rows += one_pick_rows([(trk, races)])
        except Exception:
            continue
    status = ("NA/meets OK · chosen=" + ", ".join(pulled)) if pulled else "No targets today (missing ML or below floor)."
    out = OUT/f"{date.today().isoformat()}_horses_targets.html"
    out.write_text(build_html(rows, status), encoding="utf-8")
    print(out)

if __name__ == "__main__":
    main()
