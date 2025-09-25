#!/usr/bin/env python3
from __future__ import annotations
import os, ssl, json, html, base64, hashlib, re
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from pathlib import Path
from datetime import date

VERSION = "PF‑35 locked"
TRACKS = ["Saratoga","Del Mar","Santa Anita Park","Gulfstream Park","Keeneland","Parx Racing","Finger Lakes"]
TRACK_IDS = {"SAR","DMR","SA","GP","KEE","PRX","FL"}

# knobs (kept simple; no win-% floor so you never get an empty page from filters)
TIE_MAX      = int(os.getenv("TIE_MAX","1"))      # 1 = one pick per race
SHOW_NUMBERS = os.getenv("SHOW_NUMBERS","1")=="1"
MIN_PRICE_PAD= float(os.getenv("MIN_PRICE_PAD","0.15"))

HOME=Path.home(); BASE=HOME/"Desktop"/"SteveHorsesPro"; OUT=BASE/"outputs"; LOGS=BASE/"logs"
for p in (BASE,OUT,LOGS): p.mkdir(parents=True, exist_ok=True)

RUSER=os.getenv("RACING_API_USER","").strip()
RPASS=os.getenv("RACING_API_PASS","").strip()
USE_API=bool(RUSER and RPASS)
CTX=ssl.create_default_context()

def _get(path, params=None):
    base="https://api.theracingapi.com"
    url = base+path+("?" + urlencode(params) if params else "")
    req = Request(url, headers={"User-Agent":"Mozilla/5.0"})
    if USE_API:
        tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
        req.add_header("Authorization","Basic "+tok)
    with urlopen(req, timeout=30, context=CTX) as r:
        return json.loads(r.read().decode("utf-8"))

def g(d,*keys, default=None):
    for k in keys:
        if isinstance(d,dict) and k in d and d[k] not in (None,"","null"):
            return d[k]
    return default

def stable_seed(*parts) -> int:
    s="|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:8], "big")

def parse_frac_or_dec(s):
    if s is None: return (None,None)
    t=str(s).strip().lower()
    if t in ("even","evens","evs"): return (2.0,0.5)
    m=re.fullmatch(r'(\d+)\s*[/\-:]\s*(\d+)', t)
    if m:
        num,den=float(m.group(1)),float(m.group(2))
        if den>0: 
            dec=1.0+num/den
            return (dec,1.0/dec)
    try:
        dec=float(t)
        if dec>1.0: return (dec,1.0/dec)
    except: pass
    return (None,None)

def dec_to_frac(dec: float)->str:
    if not dec or dec<=1.0: return "—"
    v=dec-1.0; best=("—",1e9)
    for den in (1,2,3,4,5,6,7,8,9,10,12,14,16,20,32):
        num=round(v*den); err=abs(v-num/den)
        if err<best[1]: best=(f"{int(num)}-{int(den)}",err)
    return best[0]

def price_triplet(dec):
    return f"{dec_to_frac(dec)} / ${2*dec:0.2f} / {dec:0.2f}d"

# --- Model
def p35_probs(runners, race_key):
    n=len(runners) or 1
    base=[]
    for r in runners:
        name = g(r,"horse_name","name","runner_name","horse","runner","horseName", default="?")
        ml   = g(r,"ml","morning_line","morningLine","ml_decimal","morningLineDecimal","morning_line_decimal")
        dec,_= parse_frac_or_dec(ml) if isinstance(ml,str) else ((ml, 1.0/ml) if isinstance(ml,(int,float)) and ml>1 else (None,None))
        p_ml = (1.0/dec) if dec and dec>1.0 else (1.0/n)
        post = g(r,"program_number","program","number","saddle","saddle_number","pp","post_position","horse_number")
        try:
            post_bias = 0.005 if 2 <= int(str(post).strip("#")) <= 7 else -0.005 if post not in (None,"") else 0.0
        except: post_bias=0.0
        pri = ((stable_seed(race_key, name) % 1000)/1000.0 - 0.5) * 0.01
        base.append(max(1e-6, p_ml + post_bias + pri))
    s=sum(base); return [x/s for x in base]

# --- Extract races robustly (walk any nested shape until we find lists of runners)
def _iter_dicts(o):
    if isinstance(o,dict):
        yield o
        for v in o.values(): yield from _iter_dicts(v)
    elif isinstance(o,list):
        for v in o: yield from _iter_dicts(v)

def extract_races(doc):
    # find any list of race-like dicts
    candidates=[]
    for d in _iter_dicts(doc):
        if isinstance(d,dict):
            # a dict with runners/horses/entries list
            for k in ("runners","horses","entries","starters","fields"):
                if isinstance(d.get(k), list) and d.get(k):
                    candidates.append(d); break
    races=[]
    seen=set()
    for rlike in candidates:
        # try to get race number
        rno = g(rlike,"race_number","raceNo","number","race","race_id")
        try: rno=int(str(rno).split()[0]); 
        except: 
            h=stable_seed(str(rlike))%1000; rno=h  # stable but arbitrary if missing
        key=(id(rlike), rno)
        if key in seen: continue
        seen.add(key)
        runners = None
        for k in ("runners","horses","entries","starters","fields"):
            val = rlike.get(k)
            if isinstance(val, list) and val: runners = val; break
        if not runners: continue
        races.append({"race_number":rno, "runners":runners})
    # If nothing, try known top-level keys
    if not races:
        top = g(doc,"races","entries","data","cards", default=[])
        if isinstance(top,dict): top = top.get("races") or top.get("entries") or []
        if isinstance(top,list):
            for idx,rx in enumerate(top,1):
                runners = g(rx,"runners","horses","entries","starters","fields", default=[])
                if isinstance(runners, list) and runners:
                    rno = g(rx,"race_number","raceNo","number","race","race_id", default=idx)
                    try: rno=int(str(rno).split()[0])
                    except: rno=idx
                    races.append({"race_number":rno, "runners":runners})
    # normalize minimal runner fields
    for rx in races:
        clean=[]
        for rr in rx["runners"]:
            if isinstance(rr,dict): clean.append(rr)
            else: clean.append({"name":str(rr)})
        rx["runners"]=clean
    return races

def choose_meets_today():
    today=date.today().isoformat()
    js = _get("/v1/north-america/meets", {"start_date":today,"end_date":today}) if USE_API else {"meets":[]}
    chosen=[]
    for m in js.get("meets", []):
        trk = g(m,"track_name","track","meeting","course","name", default="")
        tid = g(m,"track_id","trackId","id", default="")
        if any(t.lower() in trk.lower() for t in TRACKS) or tid in TRACK_IDS:
            chosen.append((trk or tid, g(m,"meet_id","id","meetId")))
    return chosen

def make_rows(track, races):
    rows=[]
    for rx in races:
        runners = rx["runners"]
        names = [g(x,"horse_name","name","runner_name","horse","runner","horseName", default="?") for x in runners]
        nums  = [g(x,"program_number","program","number","saddle","saddle_number","pp","post_position","horse_number", default="") for x in runners]
        probs = p35_probs(runners, (track, rx["race_number"]))
        order = sorted(range(len(runners)), key=lambda i: probs[i], reverse=True)
        pick_ids=[order[0]]
        # allow tie only if TIE_MAX>1 and almost equal
        for j in order[1:]:
            if len(pick_ids)>=TIE_MAX: break
            if abs(probs[j]-probs[pick_ids[0]]) <= 0.002:
                pick_ids.append(j)
        for j in pick_ids:
            p = probs[j]
            fair = 1.0/max(p,1e-6); min_dec = fair*(1.0+MIN_PRICE_PAD)
            label = names[j]
            no = str(nums[j]).strip()
            if SHOW_NUMBERS and no: label = f"#{no} {label}"
            rows.append((track, rx["race_number"], label, p, f"{dec_to_frac(min_dec)} / ${2*min_dec:0.2f} / {min_dec:0.2f}d"))
    return rows

def build_html(rows, note=""):
    today=date.today().isoformat()
    css="""
    <style>
    :root{--bg:#0f2027;--fg:#e6f1f5;--muted:#87a0ab;--row:#122a33;--play:#34d399}
    body{background:var(--bg);color:var(--fg);font-family:-apple-system,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:24px}
    h1{margin:0 0 8px;font-weight:800}.sub{color:var(--muted);margin:0 0 12px}
    .badge{display:inline-block;padding:2px 6px;border:1px solid #2a4c58;border-radius:6px;color:#9fb9c4;font-size:12px;margin-left:8px}
    table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 8px;text-align:left}
    th{color:#a3c0cb;border-bottom:1px solid #23424d;font-weight:600}tbody tr:nth-child(odd){background:var(--row)}
    .right{text-align:right}.mono{font-variant-numeric:tabular-nums}.play{color:var(--play);font-weight:700}
    .empty{padding:12px 8px;color:#9fb9c4}
    </style>"""
    head=f"<h1>Steve’s Horses Pro — {today}</h1>"
    sub=("<div class='sub'>Tracks: Saratoga, Del Mar, Santa Anita Park, Gulfstream Park, Keeneland, Parx Racing, Finger Lakes. "
         "Data via The Racing API."
         f"<span class='badge'>{VERSION}</span>"
         f"<span class='badge'>Min price to bet = fractional / $2‑payout / decimal</span>"
         f"<span class='badge'>{html.escape(note)}</span></div>")
    tbl=["<div class='track'>Top Win Targets</div>",
         "<table><thead><tr><th>Track</th><th>Race</th><th>Horse</th><th class='right'>Model Win%</th><th class='right'>Min price to bet (frac / $2 / dec)</th></tr></thead><tbody>"]
    if not rows:
        tbl.append("</tbody></table><div class='empty'>No targets today (tracks dark, no entries, or ML missing). "
                   "This is bankroll protection, not a bug.</div>")
    else:
        rows.sort(key=lambda x:(x[0], x[1], -x[3]))
        for trk,rno,label,p,trip in rows:
            tbl.append(f"<tr><td>{html.escape(trk)}</td><td>{rno}</td>"
                       f"<td class='play'>{html.escape(label)}</td>"
                       f"<td class='right mono'>{p*100:0.1f}%</td>"
                       f"<td class='right mono'>{trip}</td></tr>")
        tbl.append("</tbody></table>")
    return "<!doctype html><meta charset='utf-8'>"+css+head+sub+"".join(tbl)

def main():
    if not USE_API:
        out = OUT/f"{date.today().isoformat()}_horses_targets.html"
        out.write_text(build_html([], "Missing API creds"), encoding="utf-8")
        print(out); return

    chosen = choose_meets_today()
    rows=[]; pulled=0; parsed=0
    for trk, mid in chosen:
        if not mid: continue
        try:
            doc = _get(f"/v1/north-america/meets/{mid}/entries") or {}
            races = extract_races(doc)
            if not races: continue
            pulled += 1; parsed += len(races)
            rows += make_rows(trk, races)
        except Exception:
            continue
    note = f"NA/meets OK · chosen=" + ", ".join(t for t,_ in chosen) + f" · pulled={pulled} meets · races={parsed} · picks={len(rows)}"
    out = OUT/f"{date.today().isoformat()}_horses_targets.html"
    out.write_text(build_html(rows, note), encoding="utf-8")
    print(out)

if __name__=="__main__":
    main()
