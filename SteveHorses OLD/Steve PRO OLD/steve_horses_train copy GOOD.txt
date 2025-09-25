#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Steve Horses — TRAIN (drop-in, Pro-compatible)

- Harvests history from RacingAPI (major tracks by default, or all with --all-tracks)
- Trains per-bucket + global regularized logistic models with calibration
- Computes pars by (track|surface|distance-bucket)
- Saves models/model.json in the exact schema Pro expects
- Safe: does not modify or depend on steve_horses_pro.py internals
- ALSO: pushes each runner into a lightweight horse DB sidecar (db_horses.py)

CLI examples:
  # Harvest last 120 days then train:
  /usr/bin/python3 steve_horses_train.py --days-back 120

  # Harvest specific dates only (no training):
  /usr/bin/python3 steve_horses_train.py --harvest-dates 2025-09-01 2025-09-02 --harvest-only

  # Train only from already-harvested files (last 120d window):
  /usr/bin/python3 steve_horses_train.py --train-only --days-back 120

Env:
  RACINGAPI_USER, RACINGAPI_PASS, [optional] RACING_API_BASE (default https://api.theracingapi.com)
"""

from __future__ import annotations
import os, ssl, json, csv, math, re, statistics, base64, argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict
from urllib.request import Request, urlopen
from urllib.parse import urlencode

# ---------- Paths ----------
HOME   = Path.home()
BASE   = HOME / "Desktop" / "SteveHorsesPro"
OUT    = BASE / "outputs"
LOGS   = BASE / "logs"
MODELS = BASE / "models"
HIST   = BASE / "history"
DATA   = BASE / "data"
for d in (BASE, OUT, LOGS, MODELS, HIST, DATA):
    d.mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        (LOGS / "train.log").open("a", encoding="utf-8").write(f"[{ts}] {msg}\n")
    except Exception:
        pass
    print(msg)

# ---------- API ----------
RUSER    = os.getenv('RACINGAPI_USER')
RPASS    = os.getenv('RACINGAPI_PASS')
API_BASE = os.getenv("RACING_API_BASE", "https://api.theracingapi.com")

if not RUSER or not RPASS:
    log("[warn] RACINGAPI_USER / RACINGAPI_PASS not set; API calls will fail")

CTX = ssl.create_default_context()

def _get(path, params=None):
    url = API_BASE + path + ("?" + urlencode(params) if params else "")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode() if (RUSER and RPASS) else ""
    if tok:
        req.add_header("Authorization", "Basic " + tok)
    with urlopen(req, timeout=30, context=CTX) as r:
        return json.loads(r.read().decode("utf-8","replace"))

def safe_get(path, params=None, default=None):
    try:
        return _get(path, params)
    except Exception as e:
        log(f"GET fail {path}: {e}")
        return default

EP_MEETS            = "/v1/north-america/meets"
EP_ENTRIES_BY_MEET  = "/v1/north-america/meets/{meet_id}/entries"
EP_RESULTS_BY_MEET  = "/v1/north-america/meets/{meet_id}/results"
EP_RESULTS_BY_RACE  = "/v1/north-america/races/{race_id}/results"

# ---------- Track Filter ----------
MAJOR_TRACKS = {
    "Saratoga","Del Mar","Santa Anita","Santa Anita Park","Gulfstream Park",
    "Keeneland","Parx Racing","Finger Lakes","Kentucky Downs",
    "Woodbine","Laurel Park","Louisiana Downs","Churchill Downs","Belmont at the Big A"
}

# ---------- Small utils ----------
def g(d:dict,*ks,default=None):
    for k in ks:
        if isinstance(d,dict) and k in d and d[k] not in (None,""):
            return d[k]
    return default

def _to_float(v, default=None):
    try:
        if v in (None,""): return default
        if isinstance(v,(int,float)): return float(v)
        s=str(v).strip()
        m=re.fullmatch(r"(\d+)\s*[/\-:]\s*(\d+)", s)
        if m:
            num, den = float(m.group(1)), float(m.group(2))
            if den!=0: return num/den
        return float(s)
    except:
        return default

def _to_dec_odds(v, default=None):
    if v in (None,""): return default
    if isinstance(v,(int,float)):
        f=float(v); return f if f>1 else default
    s=str(v).strip().lower()
    if s in ("evs","even","evens"): return 2.0
    m=re.fullmatch(r"(\d+)\s*[/\-:]\s*(\d+)", s)
    if m:
        num,den=float(m.group(1)),float(m.group(2))
        if den>0: return 1.0+num/den
    try:
        dec=float(s)
        if dec>1.0: return dec
    except: pass
    return default

def get_surface(rc): 
    return str(g(rc,"surface","track_surface","course","courseType","trackSurface","surf") or "").lower()

def _surface_key(s: str) -> str:
    s=(s or "").lower()
    if "turf" in s: return "turf"
    if "synt" in s or "tapeta" in s or "poly" in s: return "synt"
    return "dirt"

def get_distance_y(rc):
    d=g(rc,"distance_yards","distance","dist_yards","yards","distanceYards","distance_y")
    if d is not None:
        try: return int(float(d))
        except: pass
    m=g(rc,"distance_meters","meters","distanceMeters")
    if m is not None:
        try: return int(float(m)*1.09361)
        except: pass
    return None

def _dist_bucket_yards(yards: int|None) -> str:
    if not yards: return "unk"
    if yards < 1320:  return "<6f"
    if yards < 1540:  return "6f"
    if yards < 1760:  return "7f"
    if yards < 1980:  return "1mi"
    if yards < 2200:  return "8.5f"
    if yards < 2420:  return "9f"
    return "10f+"

def build_bucket_key(track: str, surface: str, yards: int|None) -> str:
    return f"{track}|{_surface_key(surface)}|{_dist_bucket_yards(yards)}"

# ---------- Training (Pro-compatible) ----------
FEATS = [
    "speed","ep","lp","class","trainer_win","jockey_win","combo_win",
    "field_size","rail","ml_dec","live_dec","minutes_to_post","last_days","weight",
    "post_bias","surface_switch","equip_blinker","equip_lasix","pace_fit","class_par_delta"
]

def robust_trimmed_median(xs, trim=0.10):
    xs=[x for x in xs if x is not None]
    if not xs: return None
    xs=sorted(xs); n=len(xs); k=int(n*trim)
    core = xs[k:n-k] if n-2*k>=1 else xs
    return statistics.median(core)

def compute_pars(rows):
    def key_of(r):
        surf=str(r.get("surface") or "").lower()
        yards=_to_float(r.get("distance_yards") or "", None)
        track=r.get("track") or ""
        def dist_bucket(y):
            if not y: return "unk"
            if y < 1320:  return "<6f"
            if y < 1540:  return "6f"
            if y < 1760:  return "7f"
            if y < 1980:  return "1mi"
            if y < 2200:  return "8.5f"
            if y < 2420:  return "9f"
            return "10f+"
        return f"{track}|{_surface_key(surf)}|{dist_bucket(yards)}"

    buckets=defaultdict(list)
    for r in rows:
        if str(r.get("win","0"))!="1": continue
        k=key_of(r)
        sp=_to_float(r.get("speed") or "", None)
        cl=_to_float(r.get("class") or "", None)
        if sp is not None and cl is not None:
            buckets[k].append((sp,cl))
    pars={}
    for k,arr in buckets.items():
        if len(arr)>=12:
            sp_med=robust_trimmed_median([s for s,_ in arr], 0.12) or 80.0
            cl_med=robust_trimmed_median([c for _,c in arr], 0.12) or 70.0
            pars[k]={"spd":sp_med,"cls":cl_med}
    return pars

def _sigmoid(z): 
    z = 50.0 if z>50 else (-50.0 if z<-50 else z)
    return 1.0/(1.0+math.exp(-z))

def _standardize_fit(X):
    d=len(X[0]); mu=[0.0]*d; sd=[1.0]*d
    for j in range(d):
        col=[x[j] for x in X]
        m=statistics.mean(col)
        s=statistics.pstdev(col) if len(col)>1 else 1.0
        if s<1e-6: s=1.0
        mu[j]=m; sd[j]=s
    return {"mu":mu,"sd":sd}

def _apply_standardize(x, stat):
    mu,sd=stat["mu"],stat["sd"]
    return [(xi - mu[j])/sd[j] for j,xi in enumerate(x)]

def _train_logistic(X, y, l2=0.5, iters=260, lr=0.07, w=None):
    n=len(X); d=len(X[0]); wgt=w or [1.0]*n
    wv=[0.0]*d; b=0.0
    for _ in range(iters):
        gb=0.0; gw=[0.0]*d
        for i in range(n):
            zi=b+sum(wv[j]*X[i][j] for j in range(d))
            pi=_sigmoid(zi); di=(pi-y[i])
            ww=wgt[i]
            gb+=ww*di
            for j in range(d): gw[j]+=ww*di*X[i][j]
        for j in range(d): gw[j]+=l2*wv[j]
        b-=lr*gb/max(1.0, n)
        for j in range(d): wv[j]-=lr*gw[j]/max(1.0, n)
    return {"w":wv,"b":b}

def reliability_curve(y_true, p_pred, bins=12):
    pairs=sorted(zip(p_pred, y_true), key=lambda t:t[0])
    n=len(pairs); out=[]
    if n<max(40,bins): return []
    for b in range(bins):
        lo=int(b*n/bins); hi=int((b+1)*n/bins)
        if hi<=lo: continue
        chunk=pairs[lo:hi]
        x=statistics.mean([p for p,_ in chunk])
        y=sum(t for _,t in chunk)/len(chunk)
        out.append([x,y])
    # enforce monotone (PAV-like)
    for i in range(1,len(out)):
        if out[i][1] < out[i-1][1]:
            out[i][1] = out[i-1][1]
    return out

def apply_reliability(p, curve):
    if not curve: return p
    xs=[c[0] for c in curve]; ys=[c[1] for c in curve]
    if not xs:
        return p
    if p<=xs[0]: 
        return ys[0]*(p/max(1e-6,xs[0]))
    if p>=xs[-1]:
        return ys[-1]
    for i in range(1,len(xs)):
        if p<=xs[i]:
            w=(p - xs[i-1])/max(1e-6,(xs[i]-xs[i-1]))
            return ys[i-1]*(1-w) + ys[i]*w
    return p

def _post_bias(track, surface, yards, post_str):
    try:
        pp = int(re.sub(r"\D","", str(post_str) or ""))
    except:
        pp = None
    surf=_surface_key(surface); dist=_dist_bucket_yards(yards if yards else None)
    base = 0.0
    if surf=="turf" and pp and pp>=10: base -= 0.02
    if surf=="dirt" and pp and pp<=2:  base += 0.01
    return base

def build_feature_row(row, pars, pace_prior=0.0):
    # Keep aligned with Pro
    def f(k): return _to_float(row.get(k) or "", None)
    speed=(f("speed") or 0.0)
    ep   =(f("ep") or 0.0)
    lp   =(f("lp") or 0.0)
    cls  =(f("class") or 0.0)
    tr   =(f("trainer_win") or 0.0)
    jk   =(f("jockey_win") or 0.0)
    tj   =(f("combo_win") or 0.0)
    fs   =(f("field_size") or 8.0)
    rail =(f("rail") or 0.0)
    ml   = 0.0  # display-lite (no ML in training rows)
    live =(f("live_dec") or 0.0)
    mtp  =(f("minutes_to_post") or 15.0)
    dsl  =(f("last_days") or 25.0)
    wt   =(f("weight") or 120.0)

    track  = row.get("track") or ""
    surface= row.get("surface") or ""
    yards  = _to_float(row.get("distance_yards") or "", None)

    key = build_bucket_key(track, surface, yards)
    par = pars.get(key, {"spd":80.0,"cls":70.0})

    class_par_delta = (cls - par["cls"])/20.0 + (speed - par["spd"])/25.0
    post = row.get("program") or row.get("post") or row.get("number")
    pbias= _post_bias(track, surface, yards, post)
    surf_switch = 1.0 if str(row.get("prev_surface") or "").lower() and str(surface or "").lower() and (row.get("prev_surface")!=surface) else 0.0
    bl = 1.0 if str(row.get("equip_blinker") or "0") in ("1","1.0","true","True") else 0.0
    lx = 1.0 if str(row.get("equip_lasix")   or "0") in ("1","1.0","true","True") else 0.0
    pace_fit = (ep - 92.0)/20.0 if ep else 0.0

    def S(x,a): return (x or 0.0)/a
    return [
        S(speed,100.0), S(ep,120.0), S(lp,120.0), S(cls,100.0),
        S(tr,100.0), S(jk,100.0), S(tj,100.0),
        S(fs,12.0), S(rail,30.0), S(ml,10.0), S(live,10.0), S(mtp,30.0), S(dsl,60.0), S(wt,130.0),
        pbias, surf_switch, bl, lx, pace_fit, class_par_delta
    ]

# ===== Horse DB sidecar integration =====
try:
    from db_horses import ensure_schema as _horse_ensure_schema, record_runner as _horse_record_runner
    HORSE_DB_OK = True
except Exception as _e:
    log(f"[horse-db] sidecar unavailable: {_e}")
    HORSE_DB_OK = False

# Make sure DB schema exists (safe to call every run)
if HORSE_DB_OK:
    try:
        _horse_ensure_schema()
        log("[horse-db] schema OK")
    except Exception as e:
        log(f"[horse-db] ensure_schema failed: {e}")
        HORSE_DB_OK = False

# ---------- Harvest ----------
def fetch_meets(iso_date): 
    return safe_get(EP_MEETS, {"start_date": iso_date, "end_date": iso_date}, default={"meets":[]})

def fetch_entries(meet_id): 
    return safe_get(EP_ENTRIES_BY_MEET.format(meet_id=meet_id), default={"races":[]})

def try_fetch_results_by_meet(meet_id): 
    return safe_get(EP_RESULTS_BY_MEET.format(meet_id=meet_id))

def try_fetch_results_by_race(race_id): 
    return safe_get(EP_RESULTS_BY_RACE.format(race_id=race_id))

def harvest_one_day(iso_date: str, all_tracks=False) -> int:
    meets = (fetch_meets(iso_date) or {}).get("meets", [])
    if not meets:
        log(f"[harvest] no meets {iso_date}")
        return 0
    out_csv = HIST / f"history_{iso_date}.csv"
    nrows=0

    # --- helpers to detect equipment flags ---
    def _is_true(v) -> bool:
        if isinstance(v, bool): return v
        s = str(v).strip().lower()
        if s in ("1","true","yes","y","on","t"): return True
        if s in ("0","false","no","n","off","f","", "none", "null"): return False
        return False

    def _has_blinkers(ent) -> bool:
        cand = [
            g(ent, "blinkers_on", "blinkers", "bl", "bl_on", "equip_blinkers"),
            g(ent, "equipment", "equip")
        ]
        for c in cand:
            if c is None: 
                continue
            if isinstance(c, (bool,int,float)) and _is_true(c): 
                return True
            s=str(c).lower()
            if any(tok in s for tok in ("bl", "blink")) and not any(tok in s for tok in ("no-bl", "no blink")):
                return True
        return False

    def _on_lasix(ent) -> bool:
        cand = [
            g(ent, "lasix", "l", "medication", "med", "furosemide", "on_lasix", "lasix_on"),
            g(ent, "drugs", "meds")
        ]
        for c in cand:
            if c is None:
                continue
            if isinstance(c, (bool,int,float)) and _is_true(c):
                return True
            s=str(c).lower()
            if any(tok in s for tok in ("lasix","furosemide","l1","on l"," on-l"," lasix " , " lasix", "l ")):
                return True
        return False

    with out_csv.open("w", newline="", encoding="utf-8") as fout:
        wr = csv.writer(fout)
        # two extra columns at the end: equip_blinker, equip_lasix
        wr.writerow([
            "track","date","race","program","horse","win",
            "ml_dec","live_dec","minutes_to_post","field_size",
            "surface","prev_surface","distance_yards","rail",
            "speed","ep","lp","class","trainer_win","jockey_win","combo_win",
            "weight","last_days","equip_blinker","equip_lasix"
        ])

        for m in meets:
            track = g(m,"track_name","track","name") or "Track"
            if not all_tracks and track not in MAJOR_TRACKS:
                continue
            mid = g(m,"meet_id","id","meetId")
            if not mid: 
                continue

            entries = fetch_entries(mid) or {}
            races = entries.get("races") or entries.get("entries") or []

            # Try results for winners & off-odds
            by_meet = try_fetch_results_by_meet(mid) or {}
            idx_map={}
            for rr in (by_meet.get("races") or by_meet.get("results") or []):
                rid=str(g(rr,"race_id","id","raceId") or "")
                if rid: idx_map[rid]=rr

            for r_idx, rc in enumerate(races,1):
                rid=str(g(rc,"race_id","id","raceId","raceID") or "")
                res = idx_map.get(rid) if rid and rid in idx_map else (try_fetch_results_by_race(rid) if rid else None)
                winners=set(); off_odds={}
                if res:
                    fins = res.get("finishers") or res.get("results") or res.get("runners") or []
                    for it in fins:
                        prog=str(g(it,"program_number","program","number","pp","saddle","saddle_number") or "")
                        pos=_to_float(g(it,"finish_position","position","pos","finish","rank"), None)
                        lodds=_to_dec_odds(g(it,"final_odds","off_odds","odds","price","decimal_odds"), None)
                        if prog:
                            if pos==1: winners.add(prog)
                            if lodds: off_odds[prog]=lodds

                field_size=len(rc.get("runners") or rc.get("entries") or [])
                rno_val = str(g(rc,"race_number","race","number","raceNo") or r_idx)

                for ent in rc.get("runners") or rc.get("entries") or []:
                    prog=str(g(ent,"program_number","program","number","pp","saddle","saddle_number") or "")
                    wr.writerow([
                        track, iso_date, rno_val, prog,
                        g(ent,"horse_name","name","runner_name") or "",
                        1 if prog in winners else 0,
                        "", off_odds.get(prog) or "",
                        _to_float(g(rc,"minutes_to_post","mtp","minutesToPost"),0) or 0,
                        field_size or "",
                        get_surface(rc) or "", g(ent,"prev_surface","last_surface") or "",
                        get_distance_y(rc) or "",
                        _to_float(g(rc,"rail","rail_setting","turf_rail"),0) or 0,
                        _to_float(g(ent,"speed","spd","last_speed"),None) or "",
                        _to_float(g(ent,"pace","ep"),None) or "",
                        _to_float(g(ent,"lp","late_pace"),None) or "",
                        _to_float(g(ent,"class","cls"),None) or "",
                        _to_float(g(ent,"trainer_win_pct","trainerWinPct"),None) or "",
                        _to_float(g(ent,"jockey_win_pct","jockeyWinPct"),None) or "",
                        _to_float(g(ent,"tj_win","combo_win"),None) or "",
                        _to_float(g(ent,"weight","carried_weight","assigned_weight","wt","weight_lbs"),None) or "",
                        _to_float(g(ent,"days_since","dsl","daysSince","layoffDays","last_start_days"),None) or "",
                        1 if _has_blinkers(ent) else 0,
                        1 if _on_lasix(ent) else 0,
                    ])
                    nrows+=1

                    # Horse DB sidecar (safe/no-op if disabled)
                    if HORSE_DB_OK:
                        try:
                            _horse_record_runner(track, rno_val, rc, ent, iso_date)
                        except Exception as e:
                            log(f"[horse-db] record_runner fail {track} R{rno_val}: {e}")

    log(f"[harvest] {iso_date} -> {nrows} rows")
    return nrows

# ---------- Signals (safe no-op so the file is self-contained) ----------
def write_signals_for_date(model: dict, date_iso: str, all_tracks: bool=False) -> None:
    try:
        (BASE / "signals").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# ---------- Load history ----------
def load_history(days_back=365):
    """
    Load a rolling window of history CSVs.
    - days_back = 365 (default): include files dated within the last 365 days.
    - days_back = None: include ALL history files (no cutoff).
    """
    cutoff = None
    if days_back is not None:
        cutoff = date.today() - timedelta(days=days_back)
    rows=[]
    for p in HIST.glob("history_*.csv"):
        try:
            ds = p.stem.split("_")[1]
            d  = datetime.strptime(ds, "%Y-%m-%d").date()
        except:
            d  = None
        if cutoff and d and d < cutoff:
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                rdr=csv.DictReader(f)
                for r in rdr:
                    rows.append(r)
        except Exception as e:
            log(f"[train] read {p.name} fail {e}")
    return rows

# ---------- Training ----------
def train_models(rows, min_rows_bucket=160, min_rows_global=600):
    if not rows:
        log("[train] no rows to train"); 
        return None

    def key_of(r):
        return build_bucket_key(r.get("track") or "",
                                r.get("surface") or "",
                                _to_float(r.get("distance_yards") or "", None))

    pace_prior_by_key=defaultdict(list)
    for r in rows:
        ep=_to_float(r.get("ep") or "", None)
        if ep is not None:
            pace_prior_by_key[key_of(r)].append(ep)
    pace_prior={k:(statistics.mean(v)-92.0)/20.0 if v else 0.0 for k,v in pace_prior_by_key.items()}

    pars = compute_pars(rows)

    buckets=defaultdict(list); global_rows=[]
    for r in rows:
        track=(r.get("track") or "").strip()
        if not track: 
            continue
        y=1 if str(r.get("win") or "0").strip()=="1" else 0
        x=build_feature_row(r, pars, pace_prior.get(key_of(r),0.0))
        buckets[key_of(r)].append((x,y))
        global_rows.append((x,y))

    MODEL = {"buckets":{}, "global":{}, "pars":pars, "calib":{}, "meta":{"version":"1"}}

    for key, arr in buckets.items():
        if len(arr) < min_rows_bucket: 
            continue
        X=[x for x,_ in arr]; y=[y for _,y in arr]
        stat=_standardize_fit(X)
        Xs=[_apply_standardize(x, stat) for x in X]
        mdl=_train_logistic(Xs,y,l2=0.55,iters=280,lr=0.07)
        p_hat=[_sigmoid(mdl["b"]+sum(wj*xj for wj,xj in zip(mdl["w"], xs))) for xs in Xs]
        curve=reliability_curve(y, p_hat, bins=12)
        MODEL["buckets"][key]={"w":mdl["w"],"b":mdl["b"],"stat":stat,"n":len(arr)}
        MODEL["calib"][key]=curve

    if len(global_rows) >= min_rows_global:
        Xg=[x for x,_ in global_rows]; yg=[y for _,y in global_rows]
        stat=_standardize_fit(Xg)
        Xgs=[_apply_standardize(x, stat) for x in Xg]
        mdl=_train_logistic(Xgs, yg, l2=0.5, iters=260, lr=0.07)
        ph=[_sigmoid(mdl["b"]+sum(wj*xj for wj,xj in zip(mdl["w"], xs))) for xs in Xgs]
        curve=reliability_curve(yg, ph, bins=12)
        MODEL["global"]={"w":mdl["w"],"b":mdl["b"],"stat":stat,"n":len(global_rows)}
        MODEL["calib"]["__global__"]=curve
    else:
        MODEL["global"]={"w":[0.0]*len(FEATS),"b":0.0,"stat":{"mu":[0.0]*len(FEATS),"sd":[1.0]*len(FEATS)},"n":len(global_rows)}
        MODEL["calib"]["__global__"]=[]

    MODEL["meta"]["trained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    MODEL["meta"]["rows"]    = len(global_rows)
    return MODEL

def model_path() -> Path:
    return MODELS / "model.json"

def save_model_atomic(model_dict: dict):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tmp = MODELS / f"model_{ts}.json"
    live = model_path()
    tmp.write_text(json.dumps(model_dict, indent=2), encoding="utf-8")
    registry = MODELS / "registry.json"
    reg = []
    if registry.exists():
        try:
            reg = json.loads(registry.read_text(encoding="utf-8"))
        except:
            reg = []
    metrics = evaluate_model_snapshot(model_dict)
    reg.append({"tag": ts, "metrics": metrics, "path": str(tmp.name), "rows": model_dict.get("meta",{}).get("rows",0)})
    best = sorted(reg, key=lambda r: (round(r["metrics"].get("brier", 9e9), 6), -r.get("rows",0)))[0]
    registry.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    live.write_text((MODELS / best["path"]).read_text(encoding="utf-8"), encoding="utf-8")
    log(f"[registry] saved {tmp.name}; metrics={metrics}")
    log(f"[registry] best tag -> {best['tag']}; updated model.json")

def evaluate_model_snapshot(model_dict: dict):
    try:
        calib = model_dict.get("calib", {})
        def curve_score(curve):
            if not curve: return (1e-12, 1e-6)
            xs=[x for x,_ in curve]; ys=[y for _,y in curve]
            brier = sum((y-x)*(y-x) for x,y in zip(xs,ys)) / max(1,len(xs))
            eps=1e-6
            logloss = -sum((y*math.log(max(eps,x)) + (1-y)*math.log(max(eps,1-x))) for x,y in zip(xs,ys)) / max(1,len(xs))
            return (brier, logloss)
        curves = [v for k,v in calib.items() if k!="__global__"]
        if not curves and "__global__" in calib:
            curves = [calib["__global__"]]
        if not curves:
            return {"brier": 1e-12, "logloss": 1e-6, "auc": 0.5, "n": model_dict.get("meta",{}).get("rows",0)}
        scores=[curve_score(c) for c in curves]
        brier = statistics.mean([s[0] for s in scores])
        logloss= statistics.mean([s[1] for s in scores])
        return {"brier": brier, "logloss": logloss, "auc": 0.5, "n": model_dict.get("meta",{}).get("rows",0)}
    except Exception:
        return {"brier": 1e-12, "logloss": 1e-6, "auc": 0.5, "n": model_dict.get("meta",{}).get("rows",0)}

# ---------- PRO hook: allow PRO to read TRAIN signals (no-op friendly) ----------
def get_signals(meet_key: str):
    """
    PRO calls with meet_key: '<track>|YYYY-MM-DD'
    Returns dict keyed by (race_no_str, program_str):
      {(race_str, program_str): {"used": bool, "score": float, "wager": float, "flags": [str], "why": str}}
    Reads from: Desktop/SteveHorsesPro/signals/<date>__<track>.json
    """
    try:
        track_raw, day = meet_key.split("|", 1)
    except Exception:
        return {}

    # filename candidates — be forgiving about whitespace and punctuation
    def _canon_track(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        s = s.replace("–", "-").replace("—", "-")
        return s

    base = BASE / "signals"
    trk_variants = [
        track_raw,
        _canon_track(track_raw),
        _canon_track(track_raw).replace("  ", " "),
    ]
    tried = []
    path = None
    for t in trk_variants:
        p = base / f"{day}__{t}.json"
        tried.append(str(p))
        if p.exists():
            path = p
            break

    if not path:
        try:
            siblings = [x.name for x in base.glob(f"{day}__*.json")]
            log(f"[signals] miss for {meet_key}; tried={tried}; have={siblings}")
        except Exception:
            pass
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"[signals] read fail {path}: {e}")
        return {}

    def _norm_race(x) -> str:
        s = str(x or "").strip()
        if not s: return ""
        m = re.search(r"(\d+)", s)
        return m.group(1) if m else s

    out = {}
    for row in raw if isinstance(raw, list) else []:
        rno = _norm_race(row.get("race") or row.get("r") or "")
        pgm = str(row.get("program") or row.get("pgm") or row.get("num") or "").strip()
        if not rno or not pgm:
            continue
        out[(rno, pgm)] = {
            "used":  bool(row.get("used", True)),
            "score": float(row.get("p", 0.0) or 0.0),
            "wager": float(row.get("wager", 0.0) or 0.0),
            "flags": list(row.get("flags") or []),
            "why":   str(row.get("why") or "TRAIN prior"),
        }

    if not out:
        log(f"[signals] empty map for {meet_key} from {path.name} (parsed ok)")
    return out

# ---------- Orchestration ----------
def date_range(end_inclusive: date, back: int):
    for i in range(back, -1, -1):
        yield end_inclusive - timedelta(days=i)

def run_harvest(days_back: int|None, harvest_dates: list[str]|None, all_tracks: bool, backfill: int|None=None):
    total=0
    if harvest_dates:
        for ds in harvest_dates:
            total += harvest_one_day(ds.strip(), all_tracks=all_tracks)
        return total
    today = date.today()
    if backfill and backfill>0:
        for i in range(backfill, 0, -1):
            d = today - timedelta(days=i)
            total += harvest_one_day(d.isoformat(), all_tracks=all_tracks)
    if days_back is not None:
        for d in date_range(today, days_back):
            total += harvest_one_day(d.isoformat(), all_tracks=all_tracks)
    return total

def main():
    ap = argparse.ArgumentParser(description="Steve Horses — TRAIN")
    ap.add_argument("--days-back", type=int, default=7, help="How many days back to include (history files window)")
    ap.add_argument("--backfill", type=int, default=0, help="Additionally harvest this many prior days (before today)")
    ap.add_argument("--harvest-only", action="store_true", help="Only harvest; skip training")
    ap.add_argument("--train-only", action="store_true", help="Only train; skip harvesting")
    ap.add_argument("--harvest-dates", nargs="*", help="Specific YYYY-MM-DD dates to harvest")
    ap.add_argument("--all-tracks", action="store_true", help="Harvest all tracks (not just major)")
    ap.add_argument("--min-rows", type=int, default=160, help="Minimum rows required to fit a bucket model")
    args = ap.parse_args()

    log("[train] start")

    # Harvest phase
    if not args.train_only:
        run_harvest(args.days_back, args.harvest_dates, args.all_tracks, backfill=args.backfill)
    else:
        log("[train] skipping harvest (train-only)")

    # Train phase
    if not args.harvest_only:
        rows = load_history(days_back=args.days_back if args.days_back is not None else 120)
        log(f"[load] rows={len(rows)} from {len(list(HIST.glob('history_*.csv')))} file(s)")
        model = train_models(rows, min_rows_bucket=args.min_rows, min_rows_global=max(600, args.min_rows*4))
        if not model:
            log("[train] nothing to save (no model)")
        else:
            save_model_atomic(model)
            try:
                today_iso = date.today().isoformat()
                write_signals_for_date(model, today_iso, all_tracks=args.all_tracks)
            except Exception as e:
                log(f"[signals] failed: {e}")
            log("[train] done")
    else:
        log("[train] harvest-only done")

if __name__ == "__main__":
    main()