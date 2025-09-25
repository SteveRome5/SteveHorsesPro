#!/usr/bin/env python3
from __future__ import annotations
import os, ssl, json, html, base64, hashlib, re
from pathlib import Path
from datetime import date
from urllib.request import Request, urlopen
from urllib.parse import urlencode

VERSION = "PF-35 locked"

# ---------- knobs (env‑tunable) ----------
SHOW_NUMBERS = os.getenv("SHOW_NUMBERS","1") == "1"     # show "#3 Name"
TIE_MAX      = int(os.getenv("TIE_MAX","1"))            # ONE pick per race by default
TIE_BAND     = float(os.getenv("TIE_BAND","0.0000"))    # 0.0 => no “equal” second pick
MIN_PRICE_PAD= float(os.getenv("MIN_PRICE_PAD","0.20")) # 20% over fair
MIN_WIN_PCT  = float(os.getenv("MIN_WIN_PCT","0.18"))   # 18% floor
OUTPUT_MODE  = os.getenv("OUTPUT_MODE","votes")

TRACKS    = ["Saratoga","Del Mar","Santa Anita Park","Gulfstream Park","Keeneland","Parx Racing","Finger Lakes"]
TRACK_IDS = {"SAR","DMR","SA","GP","KEE","PRX","FL"}

HOME=Path.home(); BASE=HOME/"Desktop"/"SteveHorsesPro"; OUT=BASE/"outputs"; LOGS=BASE/"logs"
for p in (BASE,OUT,LOGS): p.mkdir(parents=True, exist_ok=True)

RUSER=os.getenv("RACING_API_USER","").strip()
RPASS=os.getenv("RACING_API_PASS","").strip()
USE_API = bool(RUSER and RPASS)
CTX = ssl.create_default_context()

def _get(path, params=None):
    base="https://api.theracingapi.com"
    url = base + path + ("?"+urlencode(params) if params else "")
    req = Request(url, headers={"User-Agent":"Mozilla/5.0"})
    if USE_API:
        token = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
        req.add_header("Authorization","Basic "+token)
    with urlopen(req, timeout=30, context=CTX) as r:
        return json.loads(r.read().decode("utf-8"))

def g(d,*keys, default=None):
    for k in keys:
        if isinstance(d,dict) and d.get(k) not in (None,""):
            return d[k]
    return default

def stable_seed(*parts)->int:
    import hashlib
    s = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:8],'big')

def parse_frac_or_dec(s):
    if s is None: return (None,None)
    t=str(s).strip().lower()
    if t in ("evs","even","evens"): return (2.0,0.5)
    m=re.fullmatch(r'(\d+)\s*[/\-:]\s*(\d+)',t)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
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
    v=dec-1.0
    best=("—",1e9)
    for den in (1,2,3,4,5,6,7,8,9,10,12,14,16,20,32):
        num=round(v*den)
        err=abs(v-num/den)
        if err<best[1]: best=(f"{int(num)}-{int(den)}",err)
    return best[0]

def p35_probs(runners, race_key):
    n=len(runners) or 1
    probs=[]
    for r in runners:
        name = g(r,"horse_name","name","runner","runner_name","horse","horseName", default="?")
        ml   = g(r,"ml","morning_line","morningLine","ml_decimal","morningLineDecimal","morning_line_decimal")
        if isinstance(ml,str):
            dec,_ = parse_frac_or_dec(ml)
        elif isinstance(ml,(int,float)) and ml>1.0:
            dec, _ = ml, 1.0/ml
        else:
            dec = None
        p_ml = (1.0/dec) if dec and dec>1.0 else (1.0/n)

        post = g(r,"post","pp","post_position","saddle","saddle_number","draw","program_number","number","horse_number")
        try:
            bias = 0.005 if 2<=int(post)<=7 else -0.005 if post not in (None,"") else 0.0
        except: 
            bias = 0.0

        pri = ((stable_seed(race_key, name) % 1000)/1000.0 - 0.5) * 0.01
        probs.append(max(1e-6, p_ml + bias + pri))
    s=sum(probs)
    return [x/s for x in probs]

def extract_races(meet_json):
    races=[]
    cand = g(meet_json,"races","data","entries","cards", default=[])
    if isinstance(cand,dict): cand = cand.get("races") or cand.get("data") or cand.get("entries") or []
    if isinstance(cand,list):
        for idx,rx in enumerate(cand,1):
            rno = g(rx,"race_number","raceNo","number","race","race_id", default=idx)
            runners = g(rx,"runners","horses","entries","starters","fields", default=[])
            if isinstance(runners,dict): runners = runners.get("runners") or runners.get("entries") or list(runners.values())
            clean=[]
            for rr in runners:
                clean.append(rr if isinstance(rr,dict) else {"name":str(rr)})
            try:
                rno_int = int(str(rno).split()[0])
            except:
                rno_int = idx
            races.append({"race_number": rno_int, "runners": clean})
    return races

def choose_meets_today():
    today=date.today().isoformat()
    js=_get("/v1/north-america/meets",{"start_date":today,"end_date":today}) if USE_API else {"meets":[]}
    chosen=[]
    for m in js.get("meets",[]):
        trk = g(m,"track_name","track","meeting","course","name", default="")
        tid = g(m,"track_id","trackId","id", default="")
        if any(t.lower() in trk.lower() for t in TRACKS) or (tid in TRACK_IDS):
            chosen.append((trk or tid, g(m,"meet_id","id","meetId")))
    return chosen, js

def load_meet_entries(meet_id):
    return _get(f"/v1/north-america/meets/{meet_id}/entries") if USE_API else {}

def fair_min_tuple(p):
    fair = 1.0/max(p,1e-6)
    minp = fair*(1.0+MIN_PRICE_PAD)
    return fair, minp

def price_triplet(dec):
    return f"{dec_to_frac(dec)} / ${2*dec:0.2f} / {dec:0.2f}d"

def make_votes_table(all_cards):
    rows=[]
    for (track, races) in all_cards:
        for r in races:
            runners = r["runners"]
            names = [g(x,"horse_name","name","runner_name","horse","runner","horseName", default="?") for x in runners]
            nums  = [g(x,"program_number","program","number","saddle","saddle_number","pp","post_position","horse_number", default="") for x in runners]
            probs = p35_probs(runners,(track,r["race_number"]))

            # rank by prob
            idx = sorted(range(len(runners)), key=lambda i: probs[i], reverse=True)
            best=[idx[0]]
            if TIE_MAX>1:
                for j in idx[1:]:
                    if len(best)>=TIE_MAX: break
                    if abs(probs[j]-probs[best[0]]) <= TIE_BAND:
                        best.append(j)

            for j in best:
                p=probs[j]
                if p < MIN_WIN_PCT:     # <- enforce your win% floor
                    continue
                fair,minp=fair_min_tuple(p)
                nm = names[j]
                no = str(nums[j]).strip()
                label = f"#{no} {nm}" if (SHOW_NUMBERS and no) else nm
                rows.append((track, r["race_number"], label, p, price_triplet(minp)))
    rows.sort(key=lambda x:(x[0],x[1],-x[3]))  # track, race, prob desc
    return rows

def build_html(rows,status_note):
    today=date.today().isoformat()
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
    head=f"<h1>Steve’s Horses Pro — {today}</h1>"
    sub=("<div class='sub'>Tracks: Saratoga, Del Mar, Santa Anita Park, Gulfstream Park, Keeneland, Parx Racing, Finger Lakes. "
         "Data via The Racing API."
         f"<span class='badge'>{VERSION}</span>"
         f"<span class='badge'>Min price to bet = fractional / $2‑payout / decimal</span>"
         f"<span class='badge'>{html.escape(status_note)}</span></div>")
    body=["<div class='track'>Top Win Targets</div>",
          "<table><thead><tr><th>Track</th><th>Race</th><th>Horse</th><th class='right'>Model Win%</th><th class='right'>Min price to bet (frac / $2 / dec)</th></tr></thead><tbody>"]
    if rows:
        for trk,rno,label,p,price in rows:
            body.append(f"<tr><td>{html.escape(trk)}</td><td>{rno}</td><td class='play'>{html.escape(label)}</td>"
                        f"<td class='right mono'>{p*100:0.1f}%</td><td class='right mono'>{price}</td></tr>")
    else:
        body.append("<tr><td colspan='5'>No targets today (dark or missing morning lines, or all picks fell below your win% floor).</td></tr>")
    body.append("</tbody></table>")
    return "<!doctype html><meta charset='utf-8'>"+css+head+sub+"".join(body)

def main():
    if not USE_API:
        status="Missing API creds. Launcher must export RACING_API_USER/RACING_API_PASS."
        (OUT/f"{date.today().isoformat()}_horses_targets.html").write_text(build_html([],status),encoding="utf-8")
        return

    chosen,_ = choose_meets_today()
    rows=[]; pulled=0
    for track,mid in chosen:
        try:
            mj=load_meet_entries(mid) or {}
            races=extract_races(mj)
            if not races: 
                continue
            pulled += 1
            rows += make_votes_table([(track,races)])
        except Exception:
            continue
    status = (f"NA/meets OK · chosen=" + ", ".join(t for t,_ in chosen)) if chosen else "No meets"
    out = OUT/f"{date.today().isoformat()}_horses_targets.html"
    out.write_text(build_html(rows,status),encoding="utf-8")
    print(out)

if __name__=="__main__":
    main()
