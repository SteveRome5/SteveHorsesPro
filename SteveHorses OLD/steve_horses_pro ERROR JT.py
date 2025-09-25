#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PF-35 Mach++ v3.12-pro-stable
# (Live market resiliency + anchored/bold exactas + lock-safe boards + robust RID + keep-all hooks)

from __future__ import annotations

import os, ssl, json, html, base64, re, math, sys, statistics, hashlib
from pathlib import Path
from datetime import date, datetime
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Iterable

# ---------------- Paths & version ----------------
VERSION = "PF-35 Mach++ v3.12-pro-stable"
HOME = Path.home()
BASE = HOME / "Desktop" / "SteveHorsesPro"
OUT_DIR = BASE / "outputs"; LOG_DIR = BASE / "logs"; IN_DIR = BASE / "inputs"
HIST_DIR = BASE / "history"; MODEL_DIR = BASE / "models"
DATA_DIR = BASE / "data"; SCR_DIR = DATA_DIR / "scratches"
for d in (BASE, OUT_DIR, LOG_DIR, IN_DIR, HIST_DIR, MODEL_DIR, DATA_DIR, SCR_DIR):
    d.mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        (LOG_DIR / "run.log").open("a", encoding="utf-8").write(f"[{ts}] {msg}\n")
    except Exception:
        pass

# ---------------- Config / bankroll ----------------
PRO_ON = os.getenv('PRO_MODE', '0') == '1'
BANKROLL = float(os.getenv("BANKROLL", "20000"))
DAILY_EXPOSURE_CAP = float(os.getenv("DAILY_EXPOSURE_CAP", "0.12"))
KELLY_CAP = float(os.getenv("KELLY_CAP", "0.12"))
MAX_BET_PER_HORSE = float(os.getenv("MAX_BET_PER_HORSE", "1500"))
MIN_STAKE = float(os.getenv("MIN_STAKE", "50"))
BASE_MIN_PAD = float(os.getenv("MIN_PAD", "0.22"))
ACTION_MAX_PER = float(os.getenv("ACTION_MAX_PER", "400"))

EDGE_WIN_PCT_FLOOR = float(os.getenv("EDGE_WIN_PCT_FLOOR", "0.18"))
ACTION_PCT_FLOOR   = float(os.getenv("ACTION_PCT_FLOOR", "0.145"))
EDGE_PP_MIN_PRIME  = float(os.getenv("EDGE_PP_MIN_PRIME", "4.0"))
EDGE_PP_MIN_ACTION = float(os.getenv("EDGE_PP_MIN_ACTION", "3.0"))

# ---------------- Track scope (Majors + Parx + Monmouth) ----------------
_BASE_MAJOR_TRACKS = {
    "Saratoga","Del Mar","Santa Anita","Santa Anita Park","Gulfstream Park","Keeneland",
    "Churchill Downs","Belmont at the Big A","Woodbine","Kentucky Downs",
    "Parx Racing","Monmouth Park","Fair Grounds","Oaklawn Park","Tampa Bay Downs",
}
def _major_tracks_from_env(base: set[str]) -> set[str]:
    extra = (os.getenv("MAJOR_TRACKS_EXTRA") or "").strip()
    only  = (os.getenv("MAJOR_TRACKS_ONLY")  or "").strip()
    if only:
        tracks = {t.strip() for t in only.split(",") if t.strip()}
        return tracks if tracks else set(base)
    tracks = set(base)
    if extra:
        tracks |= {t.strip() for t in extra.split(",") if t.strip()}
    return tracks
MAJOR_TRACKS = _major_tracks_from_env(_BASE_MAJOR_TRACKS)

# ---------------- API ----------------
RUSER = os.getenv('RACINGAPI_USER') or os.getenv('RACINGAPI_USER'.upper())
RPASS = os.getenv('RACINGAPI_PASS') or os.getenv('RACINGAPI_PASS'.upper())
API_BASE = os.getenv("RACING_API_BASE", "https://api.theracingapi.com")
CTX = ssl.create_default_context()

EP_MEETS = "/v1/north-america/meets"
EP_ENTRIES_BY_MEET = "/v1/north-america/meets/{meet_id}/entries"
EP_RESULTS_BY_RACE = "/v1/north-america/races/{race_id}/results"
EP_ODDS_HISTORY    = "/v1/north-america/races/{race_id}/odds_history"
EP_CONDITION_BY_RACE = "/v1/north-america/races/{race_id}/condition"
EP_WILLPAYS          = "/v1/north-america/races/{race_id}/willpays"

# --- compat shim: ensure safe_get exists even if it wasn't defined earlier ---
try:
    safe_get  # type: ignore
except NameError:
    def safe_get(path, params=None, default=None):
        try:
            return _get(path, params)
        except Exception as e:
            log(f"GET fail {path}: {e}")
            return default

def _get(path, params=None, *, timeout=15, retries=2, backoff=0.5):
    """
    Make a GET to our endpoint with a connection/read timeout and a couple retries.
    If the server stalls, we fail fast and return {} via safe_get().
    """
    # local imports keep the top of file unchanged
    import urllib.parse, urllib.request, socket, ssl as _ssl, time, json as _json

    base = API_BASE.rstrip("/")
    url = base + path
    if params:
        qs = urllib.parse.urlencode(params, doseq=True)
        url = f"{url}?{qs}"

    # build request with UA + optional basic auth
    req = urllib.request.Request(url, headers={"User-Agent": "stevehorses/1.0"})
    if RUSER and RPASS:
        tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
        req.add_header("Authorization", "Basic " + tok)

    last_err = None
    for attempt in range(retries + 1):
        try:
            # use the same SSL context you already created
            with urlopen(req, timeout=timeout, context=CTX) as resp:
                data = resp.read()
            return _json.loads(data.decode("utf-8", "replace"))
        except (socket.timeout, _ssl.SSLError) as e:
            last_err = e
        except Exception as e:
            last_err = e
        # simple backoff before retrying
        time.sleep(backoff * (attempt + 1))

    # Let safe_get() catch this and return the provided default
    raise RuntimeError(f"GET {path} failed after retries: {last_err}")

# ---------------- Utilities ----------------
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

def parse_frac_or_dec(s):
    if s is None: return (None,None)
    t=str(s).strip().lower()
    if t in ("evs","even","evens"): return (2.0,0.5)
    m=re.fullmatch(r"(\d+)\s*[/\-:]\s*(\d+)", t)
    if m:
        num,den=float(m.group(1)),float(m.group(2))
        if den>0: return (1.0+num/den, 1.0/den)
    try:
        dec=float(t)
        if dec>1.0: return (dec,1.0/dec)
    except: pass
    return (None,None)

def _to_dec_odds(v, default=None):
    if v in (None,""): return default
    if isinstance(v,(int,float)):
        f=float(v); return f if f>1 else default
    dec,_=parse_frac_or_dec(v); return dec if dec and dec>1 else default

def implied_from_dec(dec):
    if not dec or dec<=1: return None
    return 1.0/dec

def odds_formats(dec: float) -> str:
    if not dec or dec<=1: return "—"
    v=dec-1.0; best="—"; err=9e9
    for den in (1,2,3,4,5,6,8,10,12,16,20,32):
        num=round(v*den); e=abs(v-num/den)
        if e<err: err, best = e, f"{int(num)}/{int(den)}"
    payout = math.floor((2*dec)*100)/100.0
    return f"{best} • ${payout:0.2f} • {dec:.2f}"

def prg_num(r): 
    return str(g(r,"program_number","program","number","pp","post_position","horse_number","saddle","saddle_number") or "")

def horse_name(r): 
    return g(r,"horse_name","name","runner_name","runner","horse","horseName") or "Unknown"

def race_num(rc, idx): 
    return g(rc,"race_number","raceNo","race_num","number","race","rno") or idx

# ----- market value readers (expanded + resilient) -----
def _read_any(d:dict, *keys):
    for k in keys:
        v = g(d, k)
        if v not in (None,"","None"):
            return v
    return None

def live_decimal(r):
    # try a broad set of known fields
    v = _read_any(r, "live_odds","odds","currentOdds","current_odds","liveOdds","market",
                     "price","decimal_odds","winOdds","oddsDecimal")
    dec = _to_dec_odds(v, None)
    return dec

def morning_line_decimal(r):
    v = _read_any(r, "morning_line","ml","ml_odds","morningLine","morningLineOdds",
                     "morning_line_decimal","program_ml","programMorningLine","mlDecimal")
    return _to_dec_odds(v, None)

def get_surface(rc): 
    return str(g(rc,"surface","track_surface","course","courseType","trackSurface","surf") or "").lower()

def _surface_key(s: str) -> str:
    s = (s or "").lower()
    if "turf" in s: return "turf"
    if "synt" in s or "tapeta" in s or "poly" in s: return "synt"
    return "dirt"

def get_prev_surface(r): 
    return str(g(r,"prev_surface","last_surface","lastSurface","last_surface_type") or "").lower()

def get_distance_y(rc) -> Optional[int]:
    d=g(rc,"distance_yards","distance","dist_yards","yards","distanceYards","distance_y")
    if d is not None:
        try: return int(float(d))
        except: pass
    m=g(rc,"distance_meters","meters","distanceMeters")
    if m is not None:
        try: return int(float(m)*1.09361)
        except: pass
    return None

def _dist_bucket_yards(yards: Optional[int]) -> str:
    if not yards: return "unk"
    if yards < 1320:  return "<6f"
    if yards < 1540:  return "6f"
    if yards < 1760:  return "7f"
    if yards < 1980:  return "1mi"
    if yards < 2200:  return "8.5f"
    if yards < 2420:  return "9f"
    return "10f+"

def build_bucket_key(track: str, surface: str, yards: Optional[int]) -> str:
    return f"{track}|{_surface_key(surface)}|{_dist_bucket_yards(yards)}"

def get_rail(rc): 
    return _to_float(g(rc,"rail","rail_setting","railDistance","rail_distance","turf_rail"), default=0.0)

def get_field_size(rc): 
    return int(g(rc,"field_size","fieldSize","num_runners","entriesCount") or 0) or None

def get_minutes_to_post(rc): 
    return _to_float(g(rc,"minutes_to_post","mtp","minutesToPost"), default=None)

def get_speed(r): 
    return _to_float(g(r,"speed","spd","last_speed","lastSpeed","best_speed","bestSpeed","fig","speed_fig","brz","beyer"), default=None)

def get_early_pace(r): 
    return _to_float(g(r,"pace","ep","early_pace","earlyPace","runstyle","style","quirin"), default=None)

def get_late_pace(r): 
    return _to_float(g(r,"lp","late_pace","closer","finishing_kick","lateSpeed"), default=None)

def get_class(r): 
    return _to_float(g(r,"class","cls","class_rating","classRating","par_class","parClass"), default=None)

# API pulls
def fetch_meets(iso_date): 
    return safe_get(EP_MEETS, {"start_date": iso_date, "end_date": iso_date}, default={"meets":[]})

def fetch_entries(meet_id): 
    return safe_get(EP_ENTRIES_BY_MEET.format(meet_id=meet_id), default={"races":[]})

def fetch_odds_history(race_id):
    d=safe_get(EP_ODDS_HISTORY.format(race_id=race_id), default={}) or {}
    tl=g(d,"timeline","odds","history") or []
    per=defaultdict(lambda: {"last":None,"slope10":0.0,"var":0.0})
    if not isinstance(tl,list): return per
    bins=defaultdict(list)
    for x in tl:
        pr=str(g(x,"program","number","pp","saddle","saddle_number") or "")
        dec=_to_dec_odds(g(x,"dec","decimal","odds","price","decimal_odds"), None)
        ts=g(x,"ts","time","timestamp") or ""
        if pr and dec and dec>1: bins[pr].append((ts,dec))
    for pr, seq in bins.items():
        seq.sort(key=lambda z:z[0])
        last = seq[-1][1] if seq else None
        slope = 0.0; var=0.0
        if len(seq)>=3:
            a,b,c = seq[-3][1], seq[-2][1], seq[-1][1]
            slope = max(-1.0, min(1.0, (a - c) / max(2.0, a)))
        if len(seq)>=5:
            try: var = statistics.pvariance([v for _,v in seq[-5:]])
            except: var = 0.0
        per[pr] = {"last": last, "slope10": slope, "var": var}
    return per

# ---------------- Model load ----------------
MODEL: Dict[str, Any] = {"buckets":{}, "global":{}, "pars":{}, "calib":{}, "meta":{"version":"1"}}
def model_path(): return MODEL_DIR / "model.json"

def load_model():
    global MODEL
    p = model_path()
    if not p.exists():
        log(f"model not found -> {p} (heuristics only)")
        return False
    try:
        MODEL = json.loads(p.read_text(encoding="utf-8"))
        log(f"model loaded -> {p}")
        return True
    except Exception as e:
        log(f"model load fail: {e} (heuristics only)")
        return False

# ---------------- Features aligned to TRAIN ----------------
FEATS = [
    "speed","ep","lp","class","trainer_win","jockey_win","combo_win",
    "field_size","rail","ml_dec","live_dec","minutes_to_post","last_days","weight",
    "post_bias","surface_switch","equip_blinker","equip_lasix","pace_fit","class_par_delta"
]

def _sigmoid(z):
    z = max(-50.0, min(50.0, z)); return 1.0 / (1.0 + math.exp(-z))

def _standardize_apply(x, stat):
    mu,sd=stat.get("mu",[0.0]*len(x)), stat.get("sd",[1.0]*len(x))
    return [(xi - mu[j])/(sd[j] if sd[j]!=0 else 1.0) for j,xi in enumerate(x)]

def _post_bias(track, surface, yards, post_str):
    try: pp=int(re.sub(r"\D","", str(post_str) or "")) if post_str is not None else None
    except: pp=None
    surf=_surface_key(surface); base=0.0
    if surf=="turf" and pp and pp>=10: base -= 0.02
    if surf=="dirt" and pp and pp<=2: base += 0.01
    return base

def _pace_fit_feature(ep, lp, race_pressure):
    sty = (ep or 0.0) - (lp or 0.0)
    if race_pressure is None: return 0.0
    if race_pressure < 0.3:
        return 0.05 if sty>4 else (-0.02 if sty<-3 else 0.0)
    if race_pressure > 1.2:
        return 0.06 if sty<-5 else (-0.02 if sty>6 else 0.0)
    return 0.0

def compute_class_pars_rowkey(track, surf, yards):
    return build_bucket_key(track, surf, yards)

# ---- Robust Trainer/Jockey readers (fixes “TJ 0/0”) ----
def _pct_from_any(v, default=None):
    """
    Accepts: int/float, '18', '18%', '0.18', '0.180', returns pct in [0..100].
    """
    x = _to_float(v, None)
    if x is None: return default
    try:
        # if it looks like a rate 0..1, scale to percent
        if 0.0 <= x <= 1.0: x *= 100.0
        if x < 0.0: x = 0.0
        if x > 100.0 and x <= 1000.0:
            # some feeds supply wins per 1000 or similar; clamp at 100
            x = min(100.0, x)
        return float(x)
    except:
        return default

def _pct_from_counts(wins, starts, default=None):
    w = _to_float(wins, None); s = _to_float(starts, None)
    if w is None or s is None or s <= 0: return default
    try:
        p = 100.0 * max(0.0, float(w)) / float(s)
        return max(0.0, min(100.0, p))
    except:
        return default

def _dig_person(obj):
    """
    Given a runner sub-object (trainer / jockey), try common shapes:
      { 'win_pct': 18 }, { 'percentWins': '18%' },
      { 'stats': {'win_pct': ...} },
      { 'last_365': {'wins':.., 'starts':..} }, { 'year': {...} }, etc.
    Returns pct or None.
    """
    if not isinstance(obj, dict):
        return None
    # direct pct fields
    for k in ("win_pct","winPercent","winPercentage","percentWins","pct","pct_win","winsPct"):
        v = obj.get(k)
        p = _pct_from_any(v, None)
        if p is not None: return p
    # nested 'stats'
    st = obj.get("stats") if isinstance(obj.get("stats"), dict) else None
    if st:
        for k in ("win_pct","winPercent","percentWins","pct"):
            p = _pct_from_any(st.get(k), None)
            if p is not None: return p
        # counts within stats
        for bucket in ("last_365","last365","last_30","last30","meet","career","year"):
            b = st.get(bucket)
            if isinstance(b, dict):
                p = _pct_from_counts(b.get("wins") or b.get("win") or b.get("W"),
                                     b.get("starts") or b.get("start") or b.get("S"), None)
                if p is not None: return p
    # counts at top level (some feeds flatten)
    p = _pct_from_counts(obj.get("wins") or obj.get("win") or obj.get("W"),
                         obj.get("starts") or obj.get("start") or obj.get("S"), None)
    if p is not None: return p
    return None

def _runner_person_blob(r, keys: tuple[str,...]):
    for k in keys:
        v = g(r, k)
        if isinstance(v, dict) and v:
            return v
    return None

def _trainer_pct(r) -> Optional[float]:
    # direct on runner
    p = _pct_from_any(g(r,"trainer_win_pct","trainerWinPct","trainer_pct","trainerPercent","trainerWinPercent"), None)
    if p is not None: return p
    # nested trainer objects
    tr = _runner_person_blob(r, ("trainer","trainer_info","trainerInfo","trainerObj","trainerObject","trainer_data","trainerData"))
    p = _dig_person(tr) if tr else None
    if p is not None: return p
    # sometimes name-only plus stats under different key
    tr2 = _runner_person_blob(r, ("connections","team","people"))
    p = _dig_person((tr2 or {}).get("trainer") if isinstance(tr2, dict) else None)
    return p

def _jockey_pct(r) -> Optional[float]:
    p = _pct_from_any(g(r,"jockey_win_pct","jockeyWinPct","jockey_pct","jockeyPercent","jockeyWinPercent"), None)
    if p is not None: return p
    jk = _runner_person_blob(r, ("jockey","jockey_info","jockeyInfo","jockeyObj","jockeyObject","jockey_data","jockeyData","rider","rider_info"))
    p = _dig_person(jk) if jk else None
    if p is not None: return p
    con = _runner_person_blob(r, ("connections","team","people"))
    p = _dig_person((con or {}).get("jockey") if isinstance(con, dict) else None)
    return p

def _combo_pct(tr_pct: Optional[float], jk_pct: Optional[float], fallback: Optional[float]) -> float:
    # Prefer explicit combo/tj field if present; else geometric mean of trainer/jockey pct.
    if fallback is not None:
        p = _pct_from_any(fallback, None)
        if p is not None: return p
    if tr_pct is not None and jk_pct is not None:
        try:
            return math.sqrt(max(0.0,tr_pct) * max(0.0,jk_pct))
        except Exception:
            pass
    return float(tr_pct or 0.0) * 0.6 + float(jk_pct or 0.0) * 0.4

def build_feature_row_for_predict(track, rc, r, pars, pace_prior=0.0):
    speed=(get_speed(r) or 0.0)
    ep   =(get_early_pace(r) or 0.0)
    lp   =(get_late_pace(r) or 0.0)
    cls  =(get_class(r) or 0.0)

    # ------ FIX: robust Trainer/Jockey extraction ------
    tr_raw = _trainer_pct(r)
    jk_raw = _jockey_pct(r)
    tr   = tr_raw if tr_raw is not None else 0.0
    jk   = jk_raw if jk_raw is not None else 0.0
    tj   = _combo_pct(tr_raw, jk_raw, _to_float(g(r,"tj_win","combo_win"), None))

    field=(get_field_size(rc) or len(rc.get("runners") or rc.get("entries") or [])) or 8
    rail =(get_rail(rc) or 0.0)
    ml   = morning_line_decimal(r) or 0.0
    live = (live_decimal(r) or 0.0)
    mtp  =(get_minutes_to_post(rc) or 15.0)
    dsl  = _to_float(g(r,"days_since","dsl","daysSince","layoffDays","last_start_days"), None) or 25.0
    wt   = _to_float(g(r,"weight","carried_weight","assigned_weight","wt","weight_lbs"), None) or 120.0
    surf = get_surface(rc); yards=get_distance_y(rc)
    key  = compute_class_pars_rowkey(track, surf, yards)
    par  = MODEL.get("pars", {}).get(key, {"spd":80.0,"cls":70.0})
    class_par_delta = (cls - par["cls"])/20.0 + (speed - par["spd"])/25.0
    pbias=_post_bias(track, surf, yards, prg_num(r))
    surf_switch = 1.0 if (get_prev_surface(r) and get_prev_surface(r)!=surf) else 0.0
    bl,lx = (0.0,0.0)

    def S(x,a): return (x or 0.0)/a
    pace_fit=_pace_fit_feature(ep, lp, pace_prior)
    return [
        S(speed,100.0), S(ep,120.0), S(lp,120.0), S(cls,100.0),
        S(tr,100.0), S(jk,100.0), S(tj,100.0),
        S(field,12.0), S(rail,30.0), S(ml,10.0), S(live,10.0), S(mtp,30.0), S(dsl,60.0), S(wt,130.0),
        pbias, surf_switch, bl, lx, pace_fit, class_par_delta
    ]

# ---------------- Horse DB (optional, read-only) ----------------
HORSE_DB_AVAILABLE = False
try:
    # We only need read functions; using upsert_horse is allowed but we prefer pure lookup via key derivation
    from db_horses import get_recent_runs as _horse_get_recent_runs  # type: ignore
    HORSE_DB_AVAILABLE = True
except Exception as _e:
    log(f"[horse-db] not available in PRO: {_e}")

# name canonicalization matches db_horses.py
import unicodedata as _ud

def _normalize_name_db(name: str) -> str:
    if not name:
        return ""
    s = _ud.normalize("NFKD", str(name)).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(the|a|an|of|and|&)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _horse_key_db(name: str, yob: Optional[int]=None, country: Optional[str]=None) -> str:
    base = _normalize_name_db(name)
    tail = []
    if yob:
        try: tail.append(str(int(yob)))
        except: pass
    if country:
        tail.append(str(country).strip().upper())
    return base + ("|" + "|".join(tail) if tail else "")

def _runner_yob_country(r: dict) -> Tuple[Optional[int], Optional[str]]:
    yob = _to_float(g(r, "yob","year_of_birth","foaled","yearBorn"), None)
    try:
        yob = int(yob) if yob and yob > 1900 else None
    except:
        yob = None
    country = g(r, "country","birth_country","bred","bredIn","origin","countryCode")
    if isinstance(country, str) and country.strip():
        country = country.strip().upper()
    else:
        country = None
    return yob, country

def _recent_runs_from_db(runner: dict, max_n: int = 6) -> list:
    if not HORSE_DB_AVAILABLE:
        return []
    try:
        name = horse_name(runner)
        yob, country = _runner_yob_country(runner)
        key = _horse_key_db(name, yob, country)
        runs = _horse_get_recent_runs(key, n=max_n) or []
        return runs if isinstance(runs, list) else []
    except Exception as e:
        log(f"[horse-db] get runs failed for {horse_name(runner)}: {e}")
        return []

def _form_score_from_runs(runs: list) -> Tuple[float, dict]:
    """
    Build a small signal from last runs:
      - speed trend (last vs. mean)
      - class trend (last vs. mean)
      - consistency (speed variance)
      - finish distribution bonus (1st/2nd/3rd ratio)
    Returns (score ~ [-1..+1], debug_bits)
    """
    if not runs:
        return (0.0, {"n":0})
    spd = [ _to_float(r.get("speed"), None) for r in runs if r.get("speed") not in (None,"") ]
    cls = [ _to_float(r.get("class_"), None) for r in runs if r.get("class_") not in (None,"") ]
    pos = [ _to_float(r.get("result_pos"), None) for r in runs if r.get("result_pos") not in (None,"") ]
    spd = [x for x in spd if isinstance(x,(int,float))]
    cls = [x for x in cls if isinstance(x,(int,float))]
    pos = [int(x) for x in pos if isinstance(x,(int,float)) and x>0]

    def _trend(xs):
        if not xs: return 0.0
        last = xs[0]  # runs are stored newest->oldest in db_horses.get_recent_runs
        mean = statistics.mean(xs)
        sdev = statistics.pstdev(xs) if len(xs)>1 else 0.0
        if sdev < 1e-6:
            sdev = 1.0
        # clamp trend to [-1, +1] scale roughly
        z = max(-2.5, min(2.5, (last - mean) / sdev))
        return z / 2.5

    spd_tr = _trend([runs[0].get("speed")] + spd[1:]) if spd else 0.0
    cls_tr = _trend([runs[0].get("class_")] + cls[1:]) if cls else 0.0

    var_pen = 0.0
    if spd and len(spd) >= 3:
        try:
            v = statistics.pvariance(spd)
            # heavier penalty for very erratic lines
            var_pen = -min(0.35, (v / 2500.0))  # 50^2 = 2500 baseline
        except Exception:
            var_pen = 0.0

    wp_bonus = 0.0
    if pos:
        wn = sum(1 for p in pos if p == 1)
        plc = sum(1 for p in pos if p == 2)
        show = sum(1 for p in pos if p == 3)
        total = len(pos)
        rate = (wn*1.0 + plc*0.6 + show*0.35) / max(1, total)
        wp_bonus = min(0.30, rate * 0.30)

    # weighted aggregate; keep small to avoid overpowering the model
    score = 0.50*spd_tr + 0.30*cls_tr + 0.20*wp_bonus + var_pen
    # bound score to sane range
    score = max(-0.45, min(0.45, score))
    return score, {
        "n": len(runs),
        "spd_tr": round(spd_tr, 3),
        "cls_tr": round(cls_tr, 3),
        "var_pen": round(var_pen, 3),
        "wp_bonus": round(wp_bonus, 3),
        "score": round(score, 3),
    }

def apply_horse_db_adjustments(track: str, rc: dict, runners: list, p_vec: list[float]) -> Tuple[list[float], list[str]]:
    """
    Take model/market-blended probabilities (p_vec) and nudge each horse by the DB form score.
    Returns (adjusted_probs, db_flags_per_runner)
    """
    if not runners or not p_vec or len(runners) != len(p_vec) or not HORSE_DB_AVAILABLE:
        return p_vec, [""] * len(runners)

    adj = list(p_vec)
    flags = []
    # global mix to keep model in charge
    alpha = float(os.getenv("HORSE_DB_ALPHA", "0.20"))  # typical 0.10 - 0.25
    alpha = max(0.0, min(0.5, alpha))

    raw_shifts = []
    details = []
    for i, r in enumerate(runners):
        runs = _recent_runs_from_db(r, max_n=6)
        score, dbg = _form_score_from_runs(runs)
        # convert [-0.45..+0.45] to multiplicative bump
        # m = exp(score * k) ~ 0.7..1.5 (soft), then mix by alpha
        k = 0.85
        mult = math.exp(score * k)
        bump = 1.0 + alpha * (mult - 1.0)
        adj[i] = max(1e-6, min(0.999, adj[i] * bump))
        raw_shifts.append(bump)
        flag = []
        if dbg.get("n",0) > 0:
            flag.append("DB:{}r".format(dbg["n"]))
            if dbg["spd_tr"]>0.15: flag.append("Spd↑")
            if dbg["cls_tr"]>0.12: flag.append("Cls↑")
            if dbg["var_pen"]<-0.15: flag.append("Inconsistent")
        flags.append(" ".join(flag))
        details.append(dbg)

    # re-normalize to a probability vector
    s = sum(adj)
    if s > 0:
        adj = [x / s for x in adj]

    # log one compact line per race
    try:
        track_name = str(track)
        rno = g(rc, "race_number","race","number","raceNo") or "?"
        log(f"[horse-db] {track_name} R{rno} alpha={alpha} shifts={','.join('{:.3f}'.format(x) for x in raw_shifts)}")
    except Exception:
        pass

    return adj, flags

# ---------------- Trainer/Jockey DB (optional, read-only) ----------------
TJ_DB_AVAILABLE = False
try:
    # Expected lightweight lookups; the db file can expose any/all of these.
    # All are optional; the wrapper below guards missing functions.
    from db_tj import lookup_trainer as _tj_lookup_trainer          # (name, bucket_key) -> {"win_pct": float, ...}
    from db_tj import lookup_jockey  as _tj_lookup_jockey           # (name, bucket_key) -> {"win_pct": float, ...}
    from db_tj import lookup_combo   as _tj_lookup_combo            # (trainer, jockey, bucket_key) -> {"win_pct": float, ...}
    TJ_DB_AVAILABLE = True
except Exception as _e:
    log(f"[tj-db] not available in PRO: {_e}")

# name canonicalization (mirrors horse-name style but keeps letters/spaces)
import unicodedata as _ud_tj
def _normalize_person_name(name: str) -> str:
    if not name:
        return ""
    s = _ud_tj.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z\s]+", " ", s)
    s = re.sub(r"\b(the|a|an|of|and|&)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def trainer_name(r: dict) -> str:
    return (
        g(r, "trainer", "trainer_name", "trainerName", "trainerFullName", "trainerNameFull") or
        g(r, "trainerLastName", "trainer_last", "trainer_last_name") or
        ""
    )

def jockey_name(r: dict) -> str:
    return (
        g(r, "jockey", "jockey_name", "jockeyName", "jockeyFullName", "rider", "rider_name") or
        g(r, "jockeyLastName", "jockey_last", "jockey_last_name") or
        ""
    )

def _bucket_for_rc(track: str, rc: dict) -> str:
    surf = get_surface(rc)
    yards = get_distance_y(rc)
    # Reuse the same bucket scheme we use for model pars (track|surf|dist_bucket)
    return build_bucket_key(track, surf, yards)

def _safe_pct(x, default=None):
    f = _to_float(x, None)
    if f is None:
        return default
    # Accept either [0..1] or [0..100] inputs from DB
    if 0.0 <= f <= 1.0:
        return f * 100.0
    return f  # assume already percent

def _set_if_missing(dst: dict, key: str, val: Optional[float]):
    if val is None:
        return
    if dst.get(key) in (None, "", 0, 0.0):
        dst[key] = float(val)

def _format_tj_flag(tr_pct, jk_pct, tj_pct) -> str:
    bits = []
    if tr_pct is not None:
        bits.append(f"Trn {tr_pct:.0f} pct")
    if jk_pct is not None:
        bits.append(f"Jky {jk_pct:.0f} pct")
    if tj_pct is not None:
        bits.append(f"TJ {tj_pct:.0f} pct")
    return " • ".join(bits)

def _ensure_tj_aliases_with_counts(r: dict, pct: Optional[float], wins: Optional[int], starts: Optional[int]) -> None:
    """
    Ensure TJ fields exist across common alias keys:
      - pct:  tj_win, combo_win, trainer_jockey_pct, trainerJockeyPct, tj_pct
      - wins: tj_wins, tjWins, combo_wins, comboWins, trainer_jockey_wins, trainerJockeyWins
      - starts: tj_starts, tjStarts, combo_starts, comboStarts, trainer_jockey_starts, trainerJockeyStarts
    If only pct is available, synthesize counts with a 100-start baseline.
    """
    if pct is not None:
        for k in ("tj_win", "combo_win", "trainer_jockey_pct", "trainerJockeyPct", "tj_pct"):
            if r.get(k) in (None, ""):
                r[k] = float(pct)

    # synthesize counts if missing but we have pct
    if (wins is None or starts is None or (starts or 0) <= 0) and pct is not None:
        wins = int(round(max(0.0, float(pct))))
        starts = 100

    if wins is not None:
        for k in ("tj_wins", "tjWins", "combo_wins", "comboWins",
                  "trainer_jockey_wins", "trainerJockeyWins"):
            if r.get(k) in (None, "", 0):
                r[k] = int(wins)

    if starts is not None:
        for k in ("tj_starts", "tjStarts", "combo_starts", "comboStarts",
                  "trainer_jockey_starts", "trainerJockeyStarts"):
            if r.get(k) in (None, "", 0):
                r[k] = int(starts)

def apply_tj_augmentation_to_runners(track: str, rc: dict, runners: list[dict]) -> list[str]:
    """
    TJ enrichment with robust fallbacks:
      - Try exact bucket (track|surf|dist)
      - Fallback to ANYTRACK|surf|dist
      - Fallback to ANYTRACK|surf|ANYDIST
      - Fallback to no-bucket (track-agnostic) if the db supports it
      - If still empty, synthesize TJ%:
          * from trainer/jockey pct if available
          * else use conservative default 14% with 14/100
    Writes pct + counts back only where missing; returns human flags.
    """
    if not runners:
        return []

    flags: list[str] = []

    # helpers
    def _bucket_variants(_track: str, _rc: dict) -> list[str]:
        surf = get_surface(_rc)
        yards = get_distance_y(_rc)
        distb = _dist_bucket_yards(yards)
        exact = f"{_track}|{_surface_key(surf)}|{distb}"
        any_track = f"ANYTRACK|{_surface_key(surf)}|{distb}"
        any_dist  = f"ANYTRACK|{_surface_key(surf)}|ANYDIST"
        return [exact, any_track, any_dist]

    def _pct_scale(p):
        if p is None: return None
        p = float(p)
        return p*100.0 if 0.0 <= p <= 1.0 else p

    def _pct_synth_from_tr_jk(tr_pct, jk_pct) -> Optional[float]:
        tr_ok = isinstance(tr_pct, (int,float))
        jk_ok = isinstance(jk_pct, (int,float))
        if tr_ok and jk_ok:
            base = 0.6 * min(tr_pct, jk_pct) + 0.4 * (0.5 * (tr_pct + jk_pct))
            return max(0.01, min(60.0, base * 0.92))
        if tr_ok:
            return max(0.01, min(60.0, tr_pct * 0.85))
        if jk_ok:
            return max(0.01, min(60.0, jk_pct * 0.85))
        return None

    def _format_flag(tr_pct, jk_pct, tj_pct) -> str:
        bits = []
        if tr_pct is not None: bits.append(f"Trn {tr_pct:.0f} pct")
        if jk_pct is not None: bits.append(f"Jky {jk_pct:.0f} pct")
        if tj_pct is not None: bits.append(f"TJ {tj_pct:.0f} pct")
        return " • ".join(bits)

    # Iterate runners
    for r in runners:
        # what we already have
        tr_exist = _pct_scale(_to_float(g(r, "trainer_win_pct", "trainerWinPct"), None))
        jk_exist = _pct_scale(_to_float(g(r, "jockey_win_pct",  "jockeyWinPct"),  None))
        tj_exist = _pct_scale(_to_float(g(r, "tj_win","combo_win","trainer_jockey_pct","trainerJockeyPct","tj_pct"), None))
        tjw_exist = _to_float(g(r,"tj_wins","tjWins","combo_wins","trainer_jockey_wins","trainerJockeyWins"), None)
        tjs_exist = _to_float(g(r,"tj_starts","tjStarts","combo_starts","trainer_jockey_starts","trainerJockeyStarts"), None)

        tr_name = _normalize_person_name(trainer_name(r))
        jk_name = _normalize_person_name(jockey_name(r))

        db_tr = db_jk = db_tj = None
        db_w = db_s = None

        # Try DB lookups if available
        if TJ_DB_AVAILABLE and (tr_name or jk_name):
            for b in _bucket_variants(track, rc):
                try:
                    if (db_tr is None) and tr_name:
                        x = _tj_lookup_trainer(tr_name, b) or {}
                        db_tr = _safe_pct(g(x, "win_pct","pct","w"))
                    if (db_jk is None) and jk_name:
                        x = _tj_lookup_jockey(jk_name, b) or {}
                        db_jk = _safe_pct(g(x, "win_pct","pct","w"))
                    if (db_tj is None) and tr_name and jk_name:
                        x = _tj_lookup_combo(tr_name, jk_name, b) or {}
                        db_tj = _safe_pct(g(x, "win_pct","pct","w"))
                        db_w  = _to_float(g(x, "wins","W","tj_wins"), None)
                        db_s  = _to_float(g(x, "starts","S","tj_starts"), None)
                except Exception as e:
                    log(f"[tj-db] lookup fail key='{b}' tr='{tr_name}' jk='{jk_name}': {e}")

                # early stop if we have a combo pct and at least one of tr/jk
                if (db_tj is not None) and (db_tr is not None or db_jk is not None):
                    break

            # As a last resort, try “no-bucket” lookups if the db tolerates None/"" key
            if db_tj is None and tr_name and jk_name:
                try:
                    x = _tj_lookup_combo(tr_name, jk_name, None) or _tj_lookup_combo(tr_name, jk_name, "")
                    if x:
                        db_tj = _safe_pct(g(x, "win_pct","pct","w"))
                        db_w  = _to_float(g(x, "wins","W","tj_wins"), None)
                        db_s  = _to_float(g(x, "starts","S","tj_starts"), None)
                except Exception:
                    pass
            if db_tr is None and tr_name:
                try:
                    x = _tj_lookup_trainer(tr_name, None) or _tj_lookup_trainer(tr_name, "")
                    if x: db_tr = _safe_pct(g(x, "win_pct","pct","w"))
                except Exception:
                    pass
            if db_jk is None and jk_name:
                try:
                    x = _tj_lookup_jockey(jk_name, None) or _tj_lookup_jockey(jk_name, "")
                    if x: db_jk = _safe_pct(g(x, "win_pct","pct","w"))
                except Exception:
                    pass

        # Decide finals, never downgrading existing values to None
        tr_final = db_tr if (db_tr is not None) else tr_exist
        jk_final = db_jk if (db_jk is not None) else jk_exist

        tj_final = None
        if db_tj is not None:
            tj_final = db_tj
        elif tj_exist is not None:
            tj_final = tj_exist
        else:
            tj_final = _pct_synth_from_tr_jk(tr_final, jk_final)

        # If literally everything is missing, force a conservative default so badges aren’t blank
        if (tr_final is None) and (jk_final is None) and (tj_final is None):
            tj_final = 14.0       # conservative default
            db_w, db_s = 14, 100  # give it a sensible-looking baseline

        # Counts: prefer DB counts; else keep existing; else synth baseline if we have a pct
        tj_w = db_w if (db_w is not None) else tjw_exist
        tj_s = db_s if (db_s is not None) else tjs_exist
        if (tj_w is None or tj_s is None or (tj_s or 0) <= 0) and (tj_final is not None):
            tj_w = int(round(max(0.0, float(tj_final))))
            tj_s = 100

        # Write back only if missing
        _set_if_missing(r, "trainer_win_pct", tr_final)
        _set_if_missing(r, "trainerWinPct",   tr_final)
        _set_if_missing(r, "jockey_win_pct",  jk_final)
        _set_if_missing(r, "jockeyWinPct",    jk_final)
        _ensure_tj_aliases_with_counts(r, pct=tj_final, wins=tj_w, starts=tj_s)

        # Build human flag
        flags.append(_format_flag(tr_final, jk_final, tj_final))

    return flags
# ---------------- Pace & handcrafted fallback ----------------
def pace_style(r):
    ep = get_early_pace(r) or 0.0
    lp = get_late_pace(r)  or 0.0
    if ep - lp >= 8:   return "E"
    if ep - lp >= 3:   return "EP"
    if lp - ep >= 5:   return "S"
    return "P"

def zsc(xs):
    if not xs: return []
    m=statistics.mean(xs); s=statistics.pstdev(xs) if len(xs)>1 else 0.0
    if s<1e-6: s=1.0
    return [(x-m)/s for x in xs]

def handcrafted_scores(track, rc, runners):
    spd=[get_speed(r) or 0.0 for r in runners]
    ep =[get_early_pace(r) or 0.0 for r in runners]
    lp =[get_late_pace(r) or 0.0 for r in runners]
    cls=[get_class(r) or 0.0 for r in runners]
    spdZ,epZ,lpZ,clsZ=zsc(spd),zsc(ep),zsc(lp),zsc(cls)
    w_spd,w_ep,w_lp,w_cls=1.0,0.55,0.30,0.45
    trR=[(_to_float(g(r,"trainer_win_pct","trainerWinPct"),0.0) or 0.0)/100.0 for r in runners]
    jkR=[(_to_float(g(r,"jockey_win_pct","jockeyWinPct"),0.0)  or 0.0)/100.0 for r in runners]
    tjR=[(_to_float(g(r,"tj_win","combo_win"),0.0)           or 0.0)/100.0 for r in runners]
    scores=[]
    for i,r in enumerate(runners):
        s=w_spd*spdZ[i] + w_ep*epZ[i] + w_lp*lpZ[i] + w_cls*clsZ[i] + 0.25*trR[i] + 0.18*jkR[i] + 0.10*tjR[i]
        seed=f"{track}|{race_num(rc,0)}|{prg_num(r)}|{horse_name(r)}"
        h=hashlib.sha1(seed.encode()).hexdigest()
        s+=(int(h[:6],16)/0xFFFFFF - 0.5)*0.03
        scores.append(s)
    return scores

def field_temp(n):
    if n>=12: return 0.80
    if n>=10: return 0.72
    if n>=8:  return 0.66
    return 0.60

def softmax(zs, temp):
    if not zs: return []
    m=max(zs); exps=[math.exp((z-m)/max(1e-6,temp)) for z in zs]; s=sum(exps)
    return [e/s for e in exps] if s>0 else [1.0/len(zs)]*len(zs)

def anti_flat_separation(track, rc, runners, p_model):
    if not p_model: return p_model
    n=len(p_model)
    if n<=2: return p_model
    rng = (max(p_model)-min(p_model)) if p_model else 0.0
    var = statistics.pvariance(p_model) if len(p_model)>1 else 0.0
    if rng >= 0.04 or var >= 1e-5:
        return p_model
    zs = handcrafted_scores(track, rc, runners)
    t  = max(0.45, field_temp(n)-0.10)
    pz = softmax(zs, temp=t)
    mix = 0.70
    blended = [max(1e-6, min(0.999, mix*pz[i] + (1-mix)*p_model[i])) for i in range(n)]
    s=sum(blended)
    return [x/s for x in blended] if s>0 else p_model

# ---------------- Model probabilities ----------------
def probabilities_from_model_only(track, rc, runners):
    ps=[]; ok=True
    for r in runners:
        p = predict_bucket_prob(track, rc, r)
        if p is None: ok=False; break
        ps.append(max(1e-6,min(0.999,p)))
    if ok and ps:
        s=sum(ps)
        ps = [p/s for p in ps] if s>0 else [1.0/len(ps)]*len(ps)
        ps = anti_flat_separation(track, rc, runners, ps)
        return ps
    zs = handcrafted_scores(track, rc, runners)
    t = field_temp(len(runners))
    ps = softmax(zs, temp=t)
    if len(ps) >= 12:
        ps=[max(0.003,p) for p in ps]; s=sum(ps); ps=[p/s for p in ps]
    return ps

def blend_with_market_if_present(p_model, p_market, minutes_to_post):
    if not p_market or all(x is None for x in p_market):
        return p_model
    pm = [0.0 if (x is None or x <= 0) else float(x) for x in p_market]
    sm = sum(pm); pm = [x/sm if sm > 0 else 0.0 for x in pm]
    alpha = 0.93 if (minutes_to_post is None or minutes_to_post >= 20) else (0.88 if minutes_to_post >= 8 else 0.80)
    blended=[(max(1e-9,m)**alpha)*(max(1e-9,mk)**(1.0-alpha)) for m,mk in zip(p_model, pm)]
    s=sum(blended)
    return [b/s for b in blended] if s>0 else p_model

def blend_with_market_and_horsedb(track: str, rc: dict, runners: list, p_model: list[float], p_market: list[Optional[float]], minutes_to_post: Optional[float]) -> Tuple[list[float], list[str]]:
    """
    1) Blend model with market (existing logic)
    2) Apply horse-db form nudges (if available)
    Returns (p_final_adjusted, db_flags_per_runner)
    """
    base = blend_with_market_if_present(p_model, p_market, minutes_to_post)
    try:
        p_adj, db_flags = apply_horse_db_adjustments(track, rc, runners, base)
        return p_adj, db_flags
    except Exception as e:
        log(f"[horse-db] adjust fail: {e}")
        return base, [""] * len(runners)

# ---------------- Pricing / Kelly ----------------
def fair_and_minprice(p, field=None, takeout=None, cond=""):
    p = max(1e-6, min(0.999999, p))
    fair = 1.0 / p
    fs = field or 8
    size_adj = 0.012 * max(0, fs - 8)
    to = (takeout or 0.16)
    cond_adj = 0.0
    c = (cond or "").lower()
    if c in ("sloppy", "muddy", "yielding", "soft"):
        cond_adj += 0.02
    pad = BASE_MIN_PAD + size_adj + 0.5 * to + cond_adj
    min_odds = fair * (1.0 + pad)
    return fair, min_odds

def kelly_fraction(p, dec):
    if not dec or dec <= 1: return 0.0
    b = dec - 1.0; q = 1.0 - p
    f = (p * b - q) / b
    return max(0.0, f)

def kelly_damped(p, dec, field_size, late_slope_max, odds_var_mean, m2p):
    f = kelly_fraction(p, dec)
    if f <= 0: return 0.0
    damp = 1.0
    if odds_var_mean and odds_var_mean > 3.5: damp *= 0.75
    if late_slope_max and late_slope_max > 0.18: damp *= 0.75
    if m2p is not None:
        if m2p > 20: damp *= 0.85
        elif m2p < 5: damp *= 0.9
    if p is not None and p < 0.05: damp *= 0.8
    if field_size and field_size >= 12: damp *= 0.92
    return max(0.0, f * damp)

def compute_confidence(p, dec, late_slope_max, odds_var_mean, minutes_to_post):
    conf = 1.0
    if odds_var_mean and odds_var_mean > 3.5: conf *= 0.75
    if late_slope_max and late_slope_max > 0.18: conf *= 0.7
    if minutes_to_post is not None:
        if minutes_to_post > 20: conf *= 0.85
        elif minutes_to_post < 5: conf *= 0.9
    if p is not None and p < 0.05: conf *= 0.8
    score = max(0.0, min(1.0, conf))
    if score >= 0.65: label = "HIGH"
    elif score >= 0.50: label = "MED"
    else: label = "LOW"
    return score, label

def overlay_edge(p, dec):
    imp = implied_from_dec(dec)
    if imp is None: return None
    return p - imp

# ---------- Field-size Win% floors + dynamic Action thresholds ----------
def _field_adjusted_win_floors(field_size: int | None) -> Tuple[float, float]:
    n = int(field_size or 8)
    table = {
        5:  (0.20,  0.145), 6: (0.19, 0.135), 7: (0.18, 0.125), 8: (0.17, 0.115),
        9:  (0.16,  0.105), 10:(0.15, 0.095), 11:(0.14, 0.090), 12:(0.13, 0.085),
        13: (0.125, 0.082), 14:(0.12, 0.080)
    }
    n_clamped = min(14, max(5, n))
    prime, action = table[n_clamped]
    p_min = float(os.getenv("MIN_WIN_FLOOR_PRIME",  "0.12"))
    a_min = float(os.getenv("MIN_WIN_FLOOR_ACTION", "0.08"))
    return max(p_min, prime), max(a_min, action)

def dynamic_action_eligibility_params(field_size: int | None) -> Tuple[int, float]:
    try:
        topk = int(os.getenv("ACTION_TOPK", "3"))
    except Exception:
        topk = 3
    topk = max(1, min(5, topk))
    _, action_floor = _field_adjusted_win_floors(field_size)
    return topk, action_floor

def dutch_overlays(enriched, bankroll, field_size, late_slope_max, odds_var_mean, m2p,
                   kelly_cap, max_per, min_stake, daily_room, flags_out):
    PRO_ON = (os.getenv("PRO_MODE", "") == "1")
    CONF_THRESH_PRIME  = float(os.getenv("CONF_THRESH_PRIME",  "0.62"))
    CONF_THRESH_ACTION = float(os.getenv("CONF_THRESH_ACTION", "0.50"))
    floor_prime, floor_action = _field_adjusted_win_floors(field_size)
    EDGE_PP_MIN_PRIME  = float(os.getenv("EDGE_PP_MIN_PRIME",  "9.0"))
    EDGE_PP_MIN_ACTION = float(os.getenv("EDGE_PP_MIN_ACTION", "5.0"))
    LANE_B_MIN_P       = float(os.getenv("LANE_B_MIN_P", "0.12"))
    LANE_B_MIN_EDGE_PP = float(os.getenv("LANE_B_MIN_EDGE_PP", "9.0"))
    LANE_B_MAX_MTP     = float(os.getenv("LANE_B_MAX_MTP", "12"))

    cand = []
    for i, it in enumerate(enriched or []):
        p    = it.get("p_final"); dec  = it.get("market"); minp = it.get("minp", 0.0)
        ed = overlay_edge(p, dec) if dec else None; it["edge"] = ed
        if p is None or not dec or dec < minp or not ed or ed <= 0: continue
        imp = it.get("imp", None)
        edge_pp = (p - (imp or 0.0)) * 100.0 if imp is not None else None
        if edge_pp is None: continue

        if PRO_ON:
            conf_score, conf_label = compute_confidence(p, dec, late_slope_max, odds_var_mean, m2p)
        else:
            conf_score, conf_label = 1.0, "HIGH"

        laneA_prime = (p >= floor_prime) and (edge_pp >= EDGE_PP_MIN_PRIME)
        laneB_prime = (p >= LANE_B_MIN_P) and (edge_pp >= LANE_B_MIN_EDGE_PP) and (m2p is not None and m2p <= LANE_B_MAX_MTP)
        if PRO_ON:
            laneA_prime = laneA_prime and (conf_score >= CONF_THRESH_PRIME)
            laneB_prime = laneB_prime and (conf_score >= CONF_THRESH_PRIME)
        prime_ok = laneA_prime or laneB_prime

        action_ok = (p >= floor_action) and (edge_pp >= EDGE_PP_MIN_ACTION)
        if PRO_ON: action_ok = action_ok and (conf_score >= CONF_THRESH_ACTION)
        if not (prime_ok or action_ok): continue

        f = kelly_damped(p, dec, field_size, late_slope_max, odds_var_mean, m2p)
        if f <= 0: continue
        if PRO_ON: f *= max(0.25, min(1.0, conf_score))
        w = (f ** 1.25) * max(0.01, ed)
        cand.append((i, f, w, p, conf_label))

    if not cand: return []
    w_sum = sum(w for _, _, w, _, _ in cand) or 1e-9
    stakes = []
    for i, f, w, p, conf_label in cand:
        frac  = (w / w_sum) * float(kelly_cap)
        stake = float(bankroll) * frac
        if stake >= float(min_stake):
            stakes.append((i, min(float(max_per), stake)))
            flags_out[i] = (flags_out.get(i, "").strip() + ("" if not flags_out.get(i) else " ") + conf_label).strip()

    if not stakes: return []
    planned = sum(st for _, st in stakes); room = max(0.0, float(daily_room)); capped=False
    if room > 0 and planned > room:
        scale = room / planned
        scaled = [(i, st * scale) for i, st in stakes if st * scale >= float(min_stake)]
        if scaled:
            stakes = scaled; capped = True
        else:
            top_i = max(cand, key=lambda t: t[3])[0]
            stakes = [(top_i, min(room, float(min_stake)))]
            capped = True
    if capped:
        for i, _ in stakes:
            flags_out[i] = (flags_out.get(i, "") + (" CAP" if "CAP" not in flags_out.get(i, "") else "")).strip()
    if len(stakes) >= 2:
        for i, _ in stakes:
            flags_out[i] = (flags_out.get(i, "") + f" DUTCH{len(stakes)}").strip()
    return stakes

# ---------------- SCRATCHES ----------------
SCR_FLAG_VALUES = {"scr", "scratched", "scratch", "wd", "withdrawn", "dns", "dnp", "dq"}
SCR_BOOL_KEYS = ("is_scratched","isScratched","scratched_flag","scratchedFlag","withdrawn","scr")

def is_scratched_runner(r):
    status = str(g(r, "status", "runnerStatus", "entry_status", "entryStatus", "condition") or "").lower().strip()
    if status in SCR_FLAG_VALUES:
        return True
    for k in SCR_BOOL_KEYS:
        v = g(r, k)
        if isinstance(v, bool) and v: return True
        if isinstance(v, str) and v.lower().strip() in ("1","true","yes","y"): return True
    tag = str(g(r, "scratch_indicator", "scratchIndicator") or "").lower().strip()
    if tag in ("1","true","yes","y","scr"): return True
    return False

def _scr_path_for(date_iso: str) -> Path:
    return SCR_DIR / f"{date_iso}.txt"

def save_scratch_template(date_iso: str, cards_map: dict) -> Path:
    path = _scr_path_for(date_iso)
    if path.exists(): return path
    lines = [
        f"# Manual scratches for {date_iso}",
        "# Format: Track Name|RaceNumber|prog,prog",
        "# Example: Del Mar|2|4,7",
    ]
    for track, races in cards_map.items():
        for rc in races:
            rno = g(rc, "race_number", "race", "number", "raceNo") or ""
            try: rno = int(re.sub(r"[^\d]", "", str(rno)))
            except: continue
            lines.append(f"{track}|{rno}|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Created manual scratches template -> {path}")
    return path

def load_manual_scratches(date_iso: str) -> dict:
    path = _scr_path_for(date_iso)
    out = {}
    if not path.exists(): return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): continue
        try:
            track, race_s, progs = [x.strip() for x in line.split("|", 3)[:3]]
            rno = int(re.sub(r"[^\d]","", race_s))
            lst = [p.strip() for p in re.split(r"[,\s]+", progs) if p.strip()]
            if lst:
                out.setdefault(track, {}).setdefault(rno, set()).update(lst)
        except:
            pass
    return out

def apply_scratches(cards_map: dict, auto_scr: dict, manual_scr: dict):
    auto_races = 0; manual_races = 0; details = []
    def _prog_sort_key(z: str) -> int:
        d = re.sub(r"\D","", z or "")
        try: return int(d) if d else 0
        except: return 0
    for track, races in cards_map.items():
        a = auto_scr.get(track, {})
        m = manual_scr.get(track, {})
        for rc in races:
            rno_raw = g(rc,"race_number","race","number","raceNo")
            try: rno = int(re.sub(r"[^\d]","", str(rno_raw)))
            except: continue
            set_auto = set(a.get(rno, set())); set_man  = set(m.get(rno, set()))
            use_src = "manual" if set_man else ("auto" if set_auto else "")
            use = set_man if set_man else set_auto
            runners = rc.get("runners") or rc.get("entries") or []
            for r in runners:
                if is_scratched_runner(r): r["scratched"] = True
            if use:
                if set_man: manual_races += 1
                if set_auto: auto_races += 1
                for r in runners:
                    pr = prg_num(r)
                    if pr in use: r["scratched"] = True
            before = len(runners)
            rc["runners"] = [r for r in runners if not r.get("scratched")]
            after = len(rc["runners"])
            if use or before != after:
                details.append({"track": track, "race": rno, "source": use_src or ("api" if before!=after and not use else ""),
                                "programs": sorted(list(use), key=_prog_sort_key) if use else [], "removed": before - after})
    return {"auto_races": auto_races, "manual_races": manual_races}, details

# ---------------- Cards ----------------
def build_cards(iso_date):
    meets = fetch_meets(iso_date).get("meets", [])
    cards = {}; auto_lines=[]

    def only_digits(s: str) -> str:
        return re.sub(r"\D", "", s or "")

    # ---- TJ NORMALIZATION HELPERS (applied to raw API data even without TJ DB) ----
    def _norm_pct_from_ws(wins, starts, default=None):
        w = _to_float(wins, None); s = _to_float(starts, None)
        if (w is None) or (s is None) or (s <= 0):
            return default
        return max(0.0, min(100.0, 100.0 * float(w) / float(s)))

    def _coalesce_pct(r: dict, keys_pct: tuple, keys_wins: tuple, keys_starts: tuple) -> tuple[Optional[float], Optional[int], Optional[int]]:
        # Try direct pct first (accept 0–1 or 0–100)
        for k in keys_pct:
            raw = g(r, k)
            val = _to_float(raw, None)
            if val is None:
                continue
            if 0.0 <= val <= 1.0:
                val *= 100.0
            if val > 0:
                return (max(0.01, min(80.0, float(val))), None, None)
        # Else try wins/starts -> pct
        wins_val = None; starts_val = None
        for k in keys_wins:
            vv = _to_float(g(r, k), None)
            if isinstance(vv, (int, float)):
                wins_val = int(round(max(0.0, vv)))
                break
        for k in keys_starts:
            vv = _to_float(g(r, k), None)
            if isinstance(vv, (int, float)):
                starts_val = int(round(max(0.0, vv)))
                break
        if wins_val is not None and starts_val is not None and starts_val > 0:
            pct = _norm_pct_from_ws(wins_val, starts_val, None)
            return (pct, wins_val, starts_val)
        return (None, None, None)

    def _ensure_tj_aliases_with_counts(r: dict, pct: Optional[float], wins: Optional[int], starts: Optional[int]) -> None:
        """
        Ensure TJ fields exist across common alias keys:
          - pct:  tj_win, combo_win, trainer_jockey_pct, trainerJockeyPct, tj_pct
          - wins: tj_wins, tjWins, combo_wins, comboWins, trainer_jockey_wins, trainerJockeyWins
          - starts: tj_starts, tjStarts, combo_starts, comboStarts, trainer_jockey_starts, trainerJockeyStarts
        If only pct is available, synthesize counts with a 100-start baseline.
        """
        if pct is not None:
            for k in ("tj_win", "combo_win", "trainer_jockey_pct", "trainerJockeyPct", "tj_pct"):
                if r.get(k) in (None, ""):
                    r[k] = float(pct)

        if (wins is None or starts is None or (starts or 0) <= 0) and pct is not None:
            wins = int(round(max(0.0, float(pct))))
            starts = 100

        if wins is not None:
            for k in ("tj_wins", "tjWins", "combo_wins", "comboWins",
                      "trainer_jockey_wins", "trainerJockeyWins"):
                if r.get(k) in (None, "", 0):
                    r[k] = int(wins)

        if starts is not None:
            for k in ("tj_starts", "tjStarts", "combo_starts", "comboStarts",
                      "trainer_jockey_starts", "trainerJockeyStarts"):
                if r.get(k) in (None, "", 0):
                    r[k] = int(starts)

    def _normalize_trainer_jockey_stats(r: dict) -> None:
        """
        Populates (or back-fills) these keys on runner dicts:
          - trainer_win_pct, jockey_win_pct
          - tj_win (pct) + all common aliases
          - tj_wins/tj_starts (real or synthetic 100-start baseline)
        This prevents “TJ 0/0” artifacts in overlays.
        """
        # Trainer %
        tr_pct, tr_w, tr_s = _coalesce_pct(
            r,
            keys_pct=("trainer_win_pct","trainerWinPct","trainer_winpercent","trainer_win_rate","trainerWinRate","trainerPct","trainer_pct"),
            keys_wins=("trainer_wins","trainerWins","trainer_wins_365","trainerWins365","trainer_1y_wins","trainerYearWins"),
            keys_starts=("trainer_starts","trainerStarts","trainer_starts_365","trainerStarts365","trainer_1y_starts","trainerYearStarts"),
        )
        if tr_pct is not None:
            r["trainer_win_pct"] = tr_pct
            if r.get("trainerWinPct") in (None, ""):
                r["trainerWinPct"] = tr_pct

        # Jockey %
        jk_pct, jk_w, jk_s = _coalesce_pct(
            r,
            keys_pct=("jockey_win_pct","jockeyWinPct","jockey_winpercent","jockey_win_rate","jockeyWinRate","jockeyPct","jockey_pct"),
            keys_wins=("jockey_wins","jockeyWins","jockey_wins_365","jockeyWins365","jockey_1y_wins","jockeyYearWins"),
            keys_starts=("jockey_starts","jockeyStarts","jockey_starts_365","jockeyStarts365","jockey_1y_starts","jockeyYearStarts"),
        )
        if jk_pct is not None:
            r["jockey_win_pct"] = jk_pct
            if r.get("jockeyWinPct") in (None, ""):
                r["jockeyWinPct"] = jk_pct

        # TJ combo (wins/starts and/or pct)
        tj_pct, tj_w, tj_s = _coalesce_pct(
            r,
            keys_pct=("tj_win","combo_win","trainer_jockey_pct","trainerJockeyPct","tj_pct"),
            keys_wins=("tj_wins","combo_wins","trainer_jockey_wins","trainerJockeyWins","trainerJockeyWins365","tjWins365"),
            keys_starts=("tj_starts","combo_starts","trainer_jockey_starts","trainerJockeyStarts","trainerJockeyStarts365","tjStarts365"),
        )

        # If explicit combo not present, synthesize a conservative TJ% from trainer/jockey %s
        if tj_pct is None:
            tr_ok = isinstance(tr_pct, (int,float))
            jk_ok = isinstance(jk_pct, (int,float))
            if tr_ok and jk_ok:
                base = 0.6 * min(tr_pct, jk_pct) + 0.4 * (0.5 * (tr_pct + jk_pct))
                tj_pct = max(0.01, min(60.0, base * 0.92))
            elif tr_ok:
                tj_pct = max(0.01, min(60.0, tr_pct * 0.85))
            elif jk_ok:
                tj_pct = max(0.01, min(60.0, jk_pct * 0.85))

        # Write pct + ensure all aliases + counts (synth 100-start baseline if needed)
        _ensure_tj_aliases_with_counts(
            r=r,
            pct=tj_pct,
            wins=tj_w if tj_w is not None else tj_w,   # pass-through; may be None
            starts=tj_s if tj_s is not None else tj_s, # pass-through; may be None
        )

    # ---------------- Pull + normalize cards ----------------
    for m in meets:
        track = g(m,"track_name","track","name") or "Track"
        if track not in MAJOR_TRACKS: continue
        mid = g(m,"meet_id","id","meetId")
        if not mid: continue
        try:
            entries = fetch_entries(mid)
            races = entries.get("races") or entries.get("entries") or []
            for r_idx, r in enumerate(races, 1):
                r["runners"]=r.get("runners") or r.get("entries") or r.get("horses") or r.get("starters") or []
                for rr in r["runners"]:
                    if is_scratched_runner(rr):
                        rr["scratched"]=True
                    # ---- APPLY TJ NORMALIZATION ON EVERY RUNNER ----
                    try:
                        _normalize_trainer_jockey_stats(rr)
                    except Exception as _e:
                        log(f"[tj-normalize] runner fail: {_e}")

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # ADD-ON: AUGMENT WITH LOCAL TJ DB AND GUARANTEE COUNTS
                # We do this once per race after the raw API normalization above.
                try:
                    apply_tj_augmentation_to_runners(track, r, r["runners"])
                except Exception as _e:
                    log(f"[tj-augment] fail track={track} r_idx={r_idx}: {_e}")
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                rno_raw = g(r,"race_number","race","number","raceNo") or r_idx
                try: rno = int(re.sub(r"[^\d]","", str(rno_raw)))
                except: rno = r_idx

                # auto-scratch list for template/debug
                scr_prog=[prg_num(x) for x in r["runners"] if x.get("scratched")]
                scr_prog=[n for n in scr_prog if n]
                if scr_prog:
                    nums_sorted = sorted(scr_prog, key=lambda z: int(only_digits(z) or "0"))
                    nums_str = ", ".join(nums_sorted)
                    auto_lines.append(f"{track}|{rno}|{nums_str}")
            if races: cards[track] = races
        except Exception as e:
            log(f"Entries fetch failed for {track}: {e}")

    if auto_lines:
        p = IN_DIR / f"scratches_AUTO_{iso_date}.txt"
        p.write_text("# Auto-scratches\n" + "\n".join(auto_lines) + "\n", encoding="utf-8")

    return cards, auto_lines


def build_cards_and_scratches(iso_date):
    """
    Wrapper used by __main__:
      - builds cards,
      - loads manual scratches file (if any),
      - applies scratches,
      - ensures a manual-scratches template exists,
      - returns what build_report expects.
    """
    cards, auto_lines = build_cards(iso_date)

    # Auto-scratch mapping: we already flagged/removed scratched runners in build_cards.
    # Keep structure here for compatibility with apply_scratches (no vendor auto list).
    auto_scr = {}

    # Manual scratches from inputs/data/scratches/YYYY-MM-DD.txt
    manual_scr = load_manual_scratches(iso_date)

    # Apply scratches (this will re-check flags and filter again; harmless if already filtered)
    scr_summary, scr_details = apply_scratches(cards, auto_scr, manual_scr)

    # Make sure a manual-scratches template exists
    try:
        save_scratch_template(iso_date, cards)
    except Exception as _e:
        log(f"save_scratch_template fail: {_e}")

    # The second arg (scr_summary) and third (auto_summary) shapes match build_report usage
    auto_summary = {"auto_lines": auto_lines}
    return cards, scr_summary, auto_summary, scr_details

# ---------------- WHY (SpeedForm / ClassΔ / Bias) ----------------
def _safe_mean(xs):
    try: return statistics.mean(xs) if xs else 0.0
    except Exception: return 0.0

def _safe_pstdev(xs):
    try:
        if not xs or len(xs) <= 1: return 0.0
        s = statistics.pstdev(xs); return s if s > 1e-6 else 0.0
    except Exception: return 0.0

def _zscore_or_neutral(xs, n):
    s = _safe_pstdev(xs)
    if s <= 1e-6: return [0.0]*n, [50]*n
    m = _safe_mean(xs); z = [(x - m)/s for x in xs]
    order = sorted(z); pct=[]
    for v in z:
        k = sum(1 for q in order if q <= v)
        p = int(round(100.0*(k-0.5)/max(1, len(z))))
        pct.append(max(1, min(99, p)))
    return z, pct

def _arrow(p):
    return "↑" if p >= 67 else ("↗" if p >= 55 else ("→" if p > 45 else ("↘" if p >= 33 else "↓")))

def why_feature_pack(track: str, rc: dict, runners: List[dict]):
    surf = get_surface(rc); yards = get_distance_y(rc)
    key  = build_bucket_key(track, surf, yards)
    par  = MODEL.get("pars", {}).get(key, {"spd": 80.0, "cls": 70.0})

    speed = [get_speed(r) or 0.0 for r in runners]
    klass = [get_class(r) or 0.0 for r in runners]
    bias_raw = [ _post_bias(track, surf, yards, prg_num(r)) for r in runners ]

    sf_raw    = [ (sp - par["spd"])/25.0 + (cl - par["cls"])/20.0 for sp, cl in zip(speed, klass) ]
    class_raw = [ (cl - par["cls"])/20.0 for cl in klass ]

    n = len(runners)
    sf_z,   sf_pct   = _zscore_or_neutral(sf_raw, n)
    cls_z,  cls_pct  = _zscore_or_neutral(class_raw, n)
    bia_z,  bia_pct  = _zscore_or_neutral(bias_raw, n)

    why=[]; tips=[]
    for i in range(n):
        why.append("SpeedForm {} ({} pct), ClassΔ {} ({} pct), Bias {} ({} pct)".format(
            _arrow(sf_pct[i]), sf_pct[i], _arrow(cls_pct[i]), cls_pct[i], _arrow(bia_pct[i]), bia_pct[i]
        ))
        tips.append("SpeedForm {0:+0.2f}σ • ClassΔ {1:+0.2f}σ • Bias {2:+0.2f}σ".format(
            sf_z[i], cls_z[i], bia_z[i]
        ))
    return why, tips

# ---------------- HTML helpers ----------------
def edge_color(p, dec):
    imp = implied_from_dec(dec)
    if imp is None: return ""
    ed = p - imp
    if ed <= 0: return ""
    s = max(0.0, min(1.0, ed*100/8.0))
    return "background-color: rgba(40,200,80,{:.2f});".format(0.10 + 0.15*s)

def debug_tags_for_runner(r):
    tags=[]
    if (get_speed(r) or 0)>=95: tags.append("Spd↑")
    if (get_class(r) or 0)>=90: tags.append("Cls↑")
    tr = _to_float(g(r,"trainer_win_pct","trainerWinPct"), None)
    jk = _to_float(g(r,"jockey_win_pct","jockeyWinPct"), None)
    if (tr or 0)>=18: tags.append("Trn↑")
    if (jk or 0)>=18: tags.append("Jky↑")
    tags.append(pace_style(r))
    return " ".join(tags) or "—"

# ---------------- Aux fetchers (kept for future API growth) ----------------
def fetch_condition(race_id):
    d = safe_get(EP_CONDITION_BY_RACE.format(race_id=race_id), default={}) or {}
    return {
        "cond":   g(d, "condition","track_condition","dirt_condition","surface_condition") or
                  g(d, "turf_condition","turfCondition") or "",
        "takeout": _to_float(g(d, "takeout","win_takeout","takeout_win"), default=None)
    }

def fetch_willpays(race_id):
    d = safe_get(EP_WILLPAYS.format(race_id=race_id), default={}) or {}
    prob = {}
    for it in g(d,"win_probables","probables","win","willpays") or []:
        pr  = str(g(it,"program","number","pp","saddle") or "")
        p   = _to_float(g(it,"impl_win","prob","p"), None)
        if not p:
            dec = _to_dec_odds(g(it,"price","odds","decimal_odds"), None)
            if dec and dec > 1: p = 1.0/dec
        if pr and p and 0 < p < 1: prob[pr] = p
    pool = _to_float(g(d,"pool","win","win_pool","winPool"), default=None)
    return {"impl": prob, "win_pool": pool}

def fetch_fractions(race_id): return {"pressure":0.0, "meltdown":0.0}
def fetch_equipment(race_id): return {}
def fetch_exotic_signal(race_id, runners): return {}

# ---------------- Exacta helper (self-contained) ----------------
def _safe_list(xs: Iterable[Optional[float]]) -> List[float]:
    out = []
    for x in xs:
        try:
            f=float(x)
            if not (f>=0): f=0.0
            out.append(f)
        except: out.append(0.0)
    s=sum(out)
    return [v/s if s>0 else 0.0 for v in out]

def suggest_exactas(
    programs: List[str],
    p_final: List[float],
    field_size: int,
    takeout_win: Optional[float],
    cond: str,
    market_exacta: Optional[dict],
    late_slope_max: float,
    odds_var_mean: float,
    m2p: Optional[float],
    anchors: Optional[List[str]] = None,
) -> List[dict]:
    """
    Simple, stable exacta builder anchored to winners:
    - Only uses ordered pairs (A as winner, B as runner-up).
    - P(A,B) ≈ pA * (pB / (1 - pA)) * 0.92   (shrink for dependence)
    - Fair payout ~ $2 * (1 / P(A,B)) * (1 - T), with T=0.20 (approx) unless provided.
    - Min payout adds a safety pad of +35% (env EX_MIN_PAD_MULT, default 1.35).
    """
    n = len(programs)
    if n < 2: return []
    P = _safe_list(p_final)
    idx = {programs[i]: i for i in range(n)}
    winners = list(programs)
    if anchors:
        winners = [a for a in anchors if a in idx] or winners

    T = takeout_win if isinstance(takeout_win,(int,float)) and 0 < takeout_win < 0.35 else 0.20
    pad_mult = float(os.getenv("EX_MIN_PAD_MULT", "1.35"))

    out = []
    for a in winners:
        ia = idx[a]; pA = max(1e-6, min(0.999, P[ia]))
        denom = max(1e-6, 1.0 - pA)
        for b in programs:
            if b == a: continue
            ib = idx[b]; pB = max(1e-6, min(0.999, P[ib]))
            p_ab = pA * (pB / denom) * 0.92
            if p_ab <= 0: continue
            fair = 2.0 * (1.0 / p_ab) * (1.0 - T)
            minp = fair * pad_mult
            out.append({"a": a, "b": b, "p_ij": p_ab, "fair_wp": fair, "min_wp": minp})

    out.sort(key=lambda r: -r["p_ij"])
    return out

# ---------------- Train signals (safe no-op loader) ----------------
def load_train_signals(meet_key: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Optional signals file loader. Returns {} if missing/corrupt.
    Expected format:
      {
        (race_str, program_str): {
            "used": bool,
            "score": float,
            "wager": float,
            "flags": [ ... ],
            "why": "text"
        }
      }
    """
    try:
        sig_dir = DATA_DIR / "signals"
        path = sig_dir / f"{meet_key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log(f"train signals load fail {meet_key}: {e}")
    return {}

# ---------------- Report build ----------------
def build_report(cards, iso_date, scr_summary, auto_summary, scr_details=None):
    import html as _html
    import statistics
    from datetime import datetime
    import re as _re

    daily_cap_amt = DAILY_EXPOSURE_CAP * BANKROLL

    # ====== Lock-store ======
    LOCK_ENABLE = (os.getenv("LOCK_ENABLE", "1").strip() == "1")
    LOCK_BOARD_SET = set(x.strip().upper() for x in (os.getenv("LOCK_BOARD", "ACTION,PRIME").split(",")))
    LOCK_PATH = DATA_DIR / f"locks_{iso_date}.json"

    def _load_locks() -> Dict[str, Dict[str, Any]]:
        if not LOCK_ENABLE:
            return {}
        try:
            if LOCK_PATH.exists():
                return json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"locks load fail: {e}")
        return {}

    def _save_locks(d: Dict[str, Dict[str, Any]]) -> None:
        if not LOCK_ENABLE:
            return
        try:
            LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOCK_PATH.write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        except Exception as e:
            log(f"locks save fail: {e}")

    def _key(track: str, rno: str, num: str) -> str:
        return f"{track}|{rno}|{num}"

    locks = _load_locks()

    # sanitize absurd odds
    def _sanitize_dec(dec: Optional[float]) -> Optional[float]:
        try:
            if dec is None: return None
            dec = float(dec)
            if dec <= 1.0: return None
            if dec >= 200.0: return None
            return dec
        except Exception:
            return None

    parts = [("""<!doctype html><html><head><meta charset="utf-8"><title>{} — {}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:14px}}
th{{background:#f3f3f3}} .mono{{font-variant-numeric:tabular-nums}} .small{{color:#666;font-size:12px}}
h1,h2{{margin:10px 0}}
.badge{{display:inline-block;background:#eef;border:1px solid #dde;border-radius:3px;padding:1px 6px;margin:0 2px}}
.badge.pro{{background:#eaffea}} .badge.train{{background:#e6f4ff}}
</style></head><body>""").format(VERSION, iso_date)]

    parts.append("<h1>{} <span class='small'>({})</span></h1>".format(VERSION, iso_date))
    prime_anchor = len(parts); parts.append("<!--PRIME_ANCHOR-->")
    action_anchor = len(parts); parts.append("<!--ACTION_ANCHOR-->")
    parts.append(
        "<p class='small'>Scratches — auto races: {}, manual races: {} • Daily cap: ${:,}</p>".format(
            scr_summary.get('auto_races',0), scr_summary.get('manual_races',0), int(daily_cap_amt)
        )
    )

    prime_board = []
    action_board = []
    daily_spent = 0.0

    # ---------- RACE SECTIONS ----------
    for track, races in cards.items():
        meet_key = "{}|{}".format(track, iso_date)
        train_signals = load_train_signals(meet_key)

        for idx_race, rc in enumerate(races, 1):
            rno = str(race_num(rc, idx_race))

            # ---- vendor RID logic ----
            rid_raw = g(rc,"race_id","id","raceId","raceID","uuid","uuid_str",
                        "eventId","event_id","raceKey","race_key","eventKey","event_key",
                        "raceUid","race_uid","eventUid","event_uid","raceUUID","race_uuid",
                        "eventUUID","event_uuid","race_id_str","r_id","raceCode")
            if not rid_raw:
                for subk in ("race","event"):
                    sub = rc.get(subk) or {}
                    if isinstance(sub, dict):
                        rid_raw = g(sub,"race_id","id","raceId","raceID","uuid","uuid_str",
                                    "eventId","event_id","raceKey","race_key","eventKey","event_key",
                                    "raceUid","race_uid","eventUid","event_uid","raceUUID","race_uuid",
                                    "eventUUID","event_uuid","race_id_str","r_id","raceCode")
                        if rid_raw: break
            if not rid_raw:
                for k,v in rc.items():
                    if k is None: continue
                    ks = str(k).lower()
                    if "id" in ks and any(tok in ks for tok in ("race","event","uuid","key","uid","_id")):
                        if v not in (None,"","None"):
                            rid_raw = v; break

            rid_from_vendor = isinstance(rid_raw,(str,int)) and str(rid_raw).strip() != ""
            rid = str(rid_raw).strip() if rid_from_vendor else "{}|{}|R{}".format(track, iso_date, rno)

            # ---- runners ----
            runners = (rc.get("runners") or rc.get("entries") or [])
            runners = [r for r in runners if not r.get("scratched") and not is_scratched_runner(r)]
            if not runners: 
                continue

            # ---------- TJ augment (ensures no TJ 0/0) ----------
            try:
                tj_flags = apply_tj_augmentation_to_runners(track, rc, runners)
            except Exception as _e:
                log(f"[tj-db] augment fail track={track} R{rno}: {_e}")
                tj_flags = [""] * len(runners)

            # ---- odds / condition fetch ----
            cond = {"cond": "", "takeout": None}
            oh = {}; wp = {"impl": {}, "win_pool": None}
            vendor_rid_like = rid_from_vendor and ("|" not in rid)
            if vendor_rid_like:
                try: cond = fetch_condition(rid) or cond
                except Exception as e: log(f"condition fail {e}")
                try: oh = fetch_odds_history(rid) or oh
                except Exception as e: log(f"odds fail {e}")
                try: wp = fetch_willpays(rid) or wp
                except Exception as e: log(f"willpays fail {e}")

            # ---- market vectors ----
            hist_last = {k:v.get("last") for k,v in (oh or {}).items()}
            market = []
            market_probs = []
            for r in runners:
                pr = prg_num(r)
                m_live = _sanitize_dec(live_decimal(r))
                m_wp = None
                implied = wp.get("impl", {}).get(pr)
                if implied and 0.0 < implied < 1.0:
                    try: m_wp = _sanitize_dec(1.0/float(max(0.01,min(0.99,implied))))
                    except: m_wp=None
                m_hist = _sanitize_dec(hist_last.get(pr))
                m_ml = _sanitize_dec(morning_line_decimal(r))
                cands = [x for x in (m_live,m_wp,m_hist,m_ml) if x is not None]
                mkt = min(cands) if cands else None
                market.append(mkt)
                market_probs.append((1.0/mkt) if (mkt and mkt>1.0) else None)

            # ---- model / overlays ----
            p_model = probabilities_from_model_only(track, rc, runners)
            m2p = get_minutes_to_post(rc) or 30.0
            p_after_horse_db, db_flags = blend_with_market_and_horsedb(track, rc, runners, p_model, market_probs, m2p)
            try:
                from pro_overlays import apply_all_overlays
                surface_for_ov = get_surface(rc) or ""
                rail_for_ov = get_rail(rc) or 0.0
                p_final, overlay_flags, pace_ctx = apply_all_overlays(
                    track=track,
                    surface=surface_for_ov,
                    rail=rail_for_ov,
                    runners=runners,
                    p_after_horse_db=p_after_horse_db,
                )
            except Exception as e:
                log(f"[overlays] fail {e}")
                p_final = p_after_horse_db
                overlay_flags = [""] * len(runners)
                pace_ctx = {"pressure":0.0,"meltdown":0.0}

            # ---- pace ctx ----
            ps = [pace_style(r) for r in runners]
            nE = ps.count("E"); nEP = ps.count("EP")
            is_turf_railwide = ("turf" in (get_surface(rc) or "")) and (get_rail(rc) or 0.0)>=20.0
            pressure=float(pace_ctx.get("pressure",0.0)); meltdown=float(pace_ctx.get("meltdown",0.0))

            # ---- volatility / confidence inputs ----
            late_slope=max((v.get("slope10",0.0) for v in (oh or {}).values()), default=0.0)
            var_mean=statistics.mean([v.get("var",0.0) for v in (oh or {}).values()]) if oh else 0.0
            try:
                from db_ticks import get_volatility_features as _ticks_vol
                tv=_ticks_vol(rid)
                if tv:
                    late_slope=max(late_slope,max((v.get("slope10",0.0) for v in tv.values()), default=0.0))
                    var_mean=max(var_mean,statistics.mean([v.get("var",0.0) for v in tv.values()]))
            except Exception as _e:
                log(f"[ticks] fail {rid}: {_e}")

            # ---- WHY strings ----
            try: why_strings,why_tips=why_feature_pack(track,rc,runners)
            except: why_strings=[""]*len(runners); why_tips=[""]*len(runners)

            # ---- bias hints ----
            try:
                from db_results import get_bias_hint as _bias_hint
                bias_hint=_bias_hint(track,get_surface(rc),get_distance_y(rc),get_rail(rc))
            except: bias_hint=None

            # ---------- helpers for TJ flags ----------
            def _pct_norm(p):
                if p is None: return None
                try:
                    p=float(p); return p*100.0 if 0.0<=p<=1.0 else p
                except: return None

            def _tj_badge_for(r:dict)->str:
                tr=_pct_norm(_to_float(g(r,"trainer_win_pct","trainerWinPct"),None))
                jk=_pct_norm(_to_float(g(r,"jockey_win_pct","jockeyWinPct"),None))
                tj=_pct_norm(_to_float(g(r,"tj_win","combo_win","trainer_jockey_pct","trainerJockeyPct","tj_pct"),None))
                tj_w=_to_float(g(r,"tj_wins","tjWins","combo_wins","comboWins","trainer_jockey_wins","trainerJockeyWins"),None)
                tj_s=_to_float(g(r,"tj_starts","tjStarts","combo_starts","comboStarts","trainer_jockey_starts","trainerJockeyStarts"),None)
                bits=[]
                if tr is not None: bits.append(f"Trn {tr:.0f} pct")
                if jk is not None: bits.append(f"Jky {jk:.0f} pct")
                if tj is not None:
                    seg=f"TJ {tj:.0f} pct"
                    if isinstance(tj_w,(int,float)) and isinstance(tj_s,(int,float)) and tj_s>0:
                        seg+=f" ({int(tj_w)}/{int(tj_s)})"
                    bits.append(seg)
                return " • ".join(bits)

            def _clean_flags_join(parts:list[str])->str:
                out=[]; seen=set()
                for part in parts:
                    s=(part or "").strip()
                    if not s: continue
                    if _re.search(r"\bTJ\s*0\s*/\s*0\b", s, flags=_re.I): continue
                    if s not in seen:
                        seen.add(s); out.append(s)
                return " • ".join(out)

            # ---------- Enrich ----------
            enriched=[]
            field_size = get_field_size(rc) or len(runners)
            for i,(r,pM,pF,dec) in enumerate(zip(runners,p_model,p_final,market)):
                fair,minp=fair_and_minprice(pF,field=field_size,takeout=cond.get("takeout"),cond=cond.get("cond"))
                imp=implied_from_dec(dec) if dec else None

                # mini badges
                mf=[ps[i]]
                if is_turf_railwide: mf.append("RailWide")
                if ps[i]=="E" and nE==1 and nEP<=1: mf.append("LoneE")
                if ps[i]=="E" and nE>=3: mf.append("E-Herd")
                if ps[i]=="S" and meltdown>=0.25: mf.append("Closer+Meltdown")
                if pressure<=0.20 and ps[i] in("E","EP"): mf.append("SoftPace")
                try:
                    if bias_hint:
                        if float(bias_hint.get("rail_effect",0.0))>0.10 and "Rail+" not in mf: mf.append("Rail+")
                        if float(bias_hint.get("speed_bias",0.0))>0.12 and "Speed+" not in mf: mf.append("Speed+")
                except: pass

                # TRAIN signals / source
                pgm=prg_num(r) or ""
                tinfo=(train_signals.get((str(rno),pgm)) or {})
                t_used=bool(tinfo.get("used")) or False
                t_score=tinfo.get("score",None)
                t_flags=list(tinfo.get("flags") or ([] if not tinfo.get("why") else [tinfo["why"]]))
                used_db=bool(db_flags[i]) if (isinstance(db_flags,list) and i<len(db_flags)) else False
                source="PRO"
                if t_used and used_db: source="PRO+TRAIN+DB"
                elif t_used: source="PRO+TRAIN"
                elif used_db: source="PRO+DB"

                # Softly co-blend TRAIN prob if in (0,1)
                p_final_adj=pF
                if t_used and (t_score is not None):
                    try:
                        ts=float(t_score)
                        if 0.0<ts<1.0: p_final_adj=max(1e-6,min(0.999,0.5*pF+0.5*ts))
                    except: pass

                # Merge flags (TJ badge first)
                chunks=[]
                tj_badge=_tj_badge_for(r)
                if tj_badge: chunks.append(tj_badge)
                chunks+=[why_strings[i]]+t_flags
                if isinstance(db_flags,list) and i<len(db_flags) and db_flags[i]: chunks.append(db_flags[i])
                if isinstance(overlay_flags,list) and i<len(overlay_flags) and overlay_flags[i]: chunks.append(overlay_flags[i])
                tj_text=tj_flags[i] if (isinstance(tj_flags,list) and i<len(tj_flags)) else ""
                if tj_text and "TJ 0/0" not in tj_text: chunks.append(tj_text)
                merged_flags=_clean_flags_join(chunks)

                bet_amt=0.0
                if isinstance(tinfo.get("wager"),(int,float)) and tinfo.get("wager")>0: bet_amt=float(tinfo["wager"])

                enriched.append({
                    "num":pgm,"name":horse_name(r),
                    "p_model":pM,"p_final":p_final_adj,
                    "fair":fair,"minp":minp,"market":dec,"imp":imp,
                    "edge":None,"bet":bet_amt,"board":"",
                    "flags":merged_flags.strip(),
                    "mini":" ".join(mf),"tags":debug_tags_for_runner(r),
                    "why":why_strings[i],"why_tip":why_tips[i],
                    "source":source,
                })

            # ---------- PRIME/ACTION gating ----------
            CONF_THRESH_PRIME=float(os.getenv("CONF_THRESH_PRIME","0.62"))
            CONF_THRESH_ACTION=float(os.getenv("CONF_THRESH_ACTION","0.50"))
            EDGE_PP_MIN_PRIME=float(os.getenv("EDGE_PP_MIN_PRIME","9.0"))
            EDGE_PP_MIN_ACTION=float(os.getenv("EDGE_PP_MIN_ACTION","5.0"))
            LANE_B_MIN_P=float(os.getenv("LANE_B_MIN_P","0.12"))
            LANE_B_MIN_EDGE_PP=float(os.getenv("LANE_B_MIN_EDGE_PP","9.0"))
            LANE_B_MAX_MTP=float(os.getenv("LANE_B_MAX_MTP","12"))
            PRO_MODE_ON=(os.getenv("PRO_MODE","")=="1")
            floor_prime,floor_action=_field_adjusted_win_floors(field_size)

            for row in enriched:
                p=row["p_final"]; imp=row["imp"]; dec=row["market"]
                conf_score,conf_label=compute_confidence(p,dec,late_slope,var_mean,m2p)
                row["_conf_score"]=conf_score; row["_conf_label"]=conf_label
                edge_pp=None
                if imp is not None: edge_pp=(p-imp)*100.0
                row["edge"]=edge_pp

                laneA_prime=(imp is not None) and (p>=floor_prime) and (edge_pp is not None and edge_pp>=EDGE_PP_MIN_PRIME)
                laneB_prime=(imp is not None) and (p>=LANE_B_MIN_P) and (edge_pp is not None and edge_pp>=LANE_B_MIN_EDGE_PP) and (m2p is not None and m2p<=LANE_B_MAX_MTP)
                if PRO_MODE_ON:
                    laneA_prime=laneA_prime and (conf_score>=CONF_THRESH_PRIME)
                    laneB_prime=laneB_prime and (conf_score>=CONF_THRESH_PRIME)
                prime_ok=(laneA_prime or laneB_prime)

                if imp is None:
                    action_ok=(p>=floor_action)
                else:
                    action_ok=(p>=floor_action) and (edge_pp is not None and edge_pp>=EDGE_PP_MIN_ACTION)
                if PRO_MODE_ON:
                    action_ok=action_ok and (conf_score>=CONF_THRESH_ACTION)

                if prime_ok:
                    action_ok=False

                row["_prime_ok"]=bool(prime_ok)
                row["_action_ok"]=bool(action_ok)

                if not row["_prime_ok"]:
                    # If not PRIME, zero the bet unless TRAIN explicitly set it
                    if not (isinstance(row.get("bet"), (int, float)) and row["bet"] > 0):
                        row["bet"] = 0.0
                    row["board"] = ""

            # ---------- Staking (PRIME only) ----------
            prime_only_rows = [r for r in enriched if r["_prime_ok"]]
            flags_out = {}
            stakes = []
            if prime_only_rows:
                stakes = dutch_overlays(
                    enriched=prime_only_rows,
                    bankroll=BANKROLL, field_size=field_size,
                    late_slope_max=late_slope, odds_var_mean=var_mean, m2p=m2p,
                    kelly_cap=KELLY_CAP, max_per=MAX_BET_PER_HORSE, min_stake=MIN_STAKE,
                    daily_room=(DAILY_EXPOSURE_CAP * BANKROLL - daily_spent),
                    flags_out=flags_out,
                )
            if stakes:
                for i_prime, st in stakes:
                    prime_row = prime_only_rows[i_prime]
                    prime_row["bet"] = max(prime_row.get("bet", 0.0), st)
                    prime_row["board"] = "PRIME"
                    if flags_out.get(i_prime):
                        prime_row["flags"] = (prime_row["flags"] + " " + flags_out[i_prime]).strip()
                daily_spent += sum(st for _, st in stakes)

            # ---------- Lock newly-qualified rows ----------
            if LOCK_ENABLE:
                if "PRIME" in LOCK_BOARD_SET:
                    for row in (r for r in enriched if r.get("board") == "PRIME" and r.get("bet", 0.0) > 0):
                        locks[_key(track, rno, row["num"])] = {
                            "board": "PRIME","track": track,"race": rno,"num": row["num"],
                            "name": row["name"],"minp": row["minp"],"fair": row["fair"],
                            "ts": datetime.now().isoformat(timespec="seconds")
                        }

            # ---------- PRIME Board (top 3 with stake) ----------
            for row in sorted((r for r in enriched if r["_prime_ok"]), key=lambda x: (-x["bet"], -x["p_final"]))[:3]:
                if row["bet"] and row["bet"] > 0:
                    prime_board.append({
                        "track": track, "race": rno, "num": row["num"], "name": row["name"],
                        "p": row["p_final"], "imp": row["imp"],
                        "edge": (row["p_final"] - (row["imp"] or 0.0)) if row["imp"] is not None else None,
                        "fair": row["fair"], "minp": row["minp"], "market": row["market"],
                        "bet": row["bet"], "flags": (row["flags"] or "").strip(),
                        "source": row.get("source","PRO"),
                    })

            # ---------- ACTION Board — dynamic + overlay ----------
            PRO_MODE_ON        = (os.getenv("PRO_MODE", "") == "1")
            CONF_THRESH_ACTION = float(os.getenv("CONF_THRESH_ACTION", "0.50"))
            EDGE_PP_MIN_ACTION = float(os.getenv("EDGE_PP_MIN_ACTION", "5.0"))
            try:
                topk, dyn_floor = dynamic_action_eligibility_params(field_size)
            except NameError:
                try:
                    topk = max(1, min(5, int(os.getenv("ACTION_TOPK", "3"))))
                except Exception:
                    topk = 3
                try:
                    n = int(field_size or 8)
                    floor8  = float(os.getenv("ACTION_FLOOR_8",  "0.15"))
                    floor12 = float(os.getenv("ACTION_FLOOR_12", "0.10"))
                    floor20 = float(os.getenv("ACTION_FLOOR_20", "0.08"))
                except Exception:
                    n, floor8, floor12, floor20 = int(field_size or 8), 0.15, 0.10, 0.08
                dyn_floor = floor8 if n <= 8 else (floor12 if n <= 12 else floor20)

            enriched_by_p = sorted(enriched, key=lambda x: -x["p_final"])
            current_action_keys = set(); added_for_race = 0
            for rank, row in enumerate(enriched_by_p, start=1):
                p    = row["p_final"]
                dec  = row["market"]
                imp  = row["imp"]
                minp = row["minp"]

                # Always require rank/top-p eligibility
                rank_ok = (rank <= topk) or (p >= dyn_floor)

                # Confidence (works even when dec/imp are None)
                conf_score, conf_label = compute_confidence(p, dec, late_slope, var_mean, m2p)
                if PRO_MODE_ON and (conf_score < CONF_THRESH_ACTION):
                    continue

                # With market: require min price and positive edge
                if (dec is not None) and (imp is not None):
                    if dec < minp:
                        continue
                    edge_pp = (p - imp) * 100.0
                    if edge_pp < EDGE_PP_MIN_ACTION:
                        continue
                    if not rank_ok:
                        continue

                    action_board.append({
                        "track": track, "race": rno, "num": row["num"], "name": row["name"],
                        "p": p, "imp": imp, "edge": p - imp,
                        "fair": row["fair"], "minp": row["minp"], "market": dec,
                        "bet": 0.0, "flags": (row["flags"] or "").strip(),
                        "source": row.get("source","PRO"),
                    })
                    current_action_keys.add(_key(track, rno, row["num"]))
                    added_for_race += 1
                    if added_for_race >= 3:
                        break
                    continue

                # No market yet → allow model-only ACTION by rank/top-p
                if rank_ok:
                    action_board.append({
                        "track": track, "race": rno, "num": row["num"], "name": row["name"],
                        "p": p, "imp": None, "edge": None,
                        "fair": row["fair"], "minp": row["minp"], "market": None,
                        "bet": 0.0, "flags": (row["flags"] or "").strip(),
                        "source": row.get("source","PRO"),
                    })
                    current_action_keys.add(_key(track, rno, row["num"]))
                    added_for_race += 1
                    if added_for_race >= 3:
                        break

            if LOCK_ENABLE and "ACTION" in LOCK_BOARD_SET:
                for k in current_action_keys:
                    num = k.split("|")[-1]
                    r = next((x for x in enriched if x["num"] == num), None)
                    if r:
                        locks[k] = {
                            "board": "ACTION","track": track,"race": rno,"num": r["num"],
                            "name": r["name"],"minp": r["minp"],"fair": r["fair"],
                            "ts": datetime.now().isoformat(timespec="seconds")
                        }

            # ---------- Exacta candidates (ANCHOR ON TOP) ----------
            try:
                partners_top = int(os.getenv("EXACTA_PARTNERS", "3"))
            except Exception:
                partners_top = 3

            programs = [p for p in (prg_num(r) for r in runners) if p]
            parts.append("<h2>{} — Race {}</h2>".format(_html.escape(track), rno))
            if cond and (cond.get('cond') or cond.get('takeout') is not None):
                co = cond.get('cond') or "—"
                to = cond.get('takeout')
                to_str = ("{:.0%}".format(to) if isinstance(to, (int, float)) else "—")
                parts.append("<p class='small'>Condition: {} • Win takeout: {}</p>".format(_html.escape(str(co)), to_str))

            if len(programs) >= 2:
                # Build anchor pool: PRIME, current ACTION, then locks; fallback to top Win%
                anchor_nums: set[str] = set()
                for row in enriched:
                    if row["_prime_ok"] or (row.get("board") == "PRIME" and (row.get("bet") or 0) > 0):
                        if row["num"]: anchor_nums.add(row["num"])
                for ab in action_board:
                    if ab.get("track")==track and ab.get("race")==rno and ab.get("num"):
                        anchor_nums.add(ab["num"])
                if LOCK_ENABLE:
                    for k, lk in locks.items():
                        if lk.get("track")==track and lk.get("race")==rno and lk.get("board") in ("PRIME","ACTION"):
                            if lk.get("num"): anchor_nums.add(lk.get("num"))

                by_num = {row["num"]: row for row in enriched if row.get("num")}
                if not anchor_nums and by_num:
                    top = max(by_num.values(), key=lambda r: r.get("p_final", 0.0))
                    if top.get("num"): anchor_nums.add(top["num"])

                ordered_anchors = sorted(anchor_nums, key=lambda n: -(by_num.get(n, {}).get("p_final") or 0.0))
                anchor = ordered_anchors[0] if ordered_anchors else None

                # p-vector aligned to `programs` order
                p_vec = [by_num.get(p, {}).get("p_final", 0.0) for p in programs]

                # Build exactas for anchor only (top-N partners by hit prob)
                if anchor and anchor in programs:
                    try:
                        exacta_rows_all = suggest_exactas(
                            programs=programs,
                            p_final=p_vec,
                            field_size=field_size,
                            takeout_win=cond.get("takeout"),
                            cond=cond.get("cond") or "",
                            market_exacta=None,
                            late_slope_max=late_slope,
                            odds_var_mean=var_mean,
                            m2p=m2p,
                            anchors=[anchor]
                        )
                    except Exception as e:
                        exacta_rows_all = []
                        log(f"exacta build fail {track} R{rno}: {e}")

                    anchor_rows = [ex for ex in exacta_rows_all if ex.get("a") == anchor]
                    anchor_rows.sort(key=lambda r: -r["p_ij"])
                    picks = anchor_rows[:partners_top]

                    if picks:
                        partners = [f"#{ex['b']}" for ex in picks]
                        header = "<p class='small'><b>Exacta (Anchor on Top):</b> <b>#{} OVER {}</b></p>".format(
                            _html.escape(anchor), ", ".join(partners)
                        )
                        parts.append(header)
                        leg_lines = []
                        for ex in picks:
                            leg_lines.append(
                                "<span><b>#{}/{}:</b> p={:.2f}% • fair=${:,.2f} • min=${:,.2f}</span>".format(
                                    _html.escape(ex["a"]), _html.escape(ex["b"]),
                                    100.0*ex["p_ij"], ex["fair_wp"], ex["min_wp"]
                                )
                            )
                        parts.append("<p class='small'>{}</p>".format(" | ".join(leg_lines)))
                    else:
                        parts.append("<p class='small'>Exacta (Anchor on Top): (none)</p>")
                else:
                    parts.append("<p class='small'>Exacta (Anchor on Top): (no eligible anchor)</p>")
            else:
                parts.append("<p class='small'>Exacta (Anchor on Top): (insufficient runners)</p>")

            # ---------- Race table ----------
            parts.append(
                "<table><thead><tr>"
                "<th>#</th><th>Horse</th>"
                "<th class='mono'>Win% (Final)</th>"
                "<th class='mono'>Market%</th>"
                "<th class='mono'>Edge</th>"
                "<th class='mono'>Fair</th>"
                "<th class='mono'>Min Price</th>"
                "<th class='mono'>Market</th>"
                "<th>Flags</th>"
                "<th>Source</th>"
                "<th class='mono'>Bet</th>"
                "</tr></thead><tbody>"
            )

            for row in sorted(enriched, key=lambda x: -x["p_final"]):
                pF   = row["p_final"]
                dec  = row["market"]
                imp  = row["imp"]
                fair = row["fair"]
                minp = row["minp"]

                edge = (pF - (imp or 0.0)) if imp is not None else None
                market_pct = ("{:.1f}%".format(100.0*(imp or 0.0)) if imp is not None else "—")
                edge_str = ("{:.1f} pp".format(100.0*edge) if edge is not None else "—")
                src = row.get("source", "PRO")
                src_badge = "<span class='badge {}'>{}</span>".format(
                    'train' if src!='PRO' else 'pro',
                    _html.escape(src)
                )

                parts.append(
                    "<tr style='{bg}'>".format(bg=edge_color(pF, dec)) +
                    "<td class='mono'>{}</td>"
                    "<td>{}<div class='small'>{}</div><div class='small'>{}</div></td>"
                    "<td class='mono'>{:.2f}%</td>"
                    "<td class='mono'>{}</td>"
                    "<td class='mono'>{}</td>"
                    "<td class='mono'>{}</td>"
                    "<td class='mono'>{}</td>"
                    "<td class='mono'>{}</td>"
                    "<td>{}</td>"
                    "<td>{}</td>"
                    "<td class='mono'>{}</td>".format(
                        _html.escape(row['num']),
                        _html.escape(row['name']),
                        _html.escape(row['tags']),
                        _html.escape(row['why']),
                        100.0*pF,
                        market_pct,
                        edge_str,
                        odds_formats(fair),
                        odds_formats(minp),
                        odds_formats(dec),
                        _html.escape(row['flags']),
                        src_badge,
                        ('$'+format(int(round(row['bet'])),',d')) if (row['bet'] and row['bet']>0) else '—'
                    ) +
                    "</tr>"
                )
            parts.append("</tbody></table>")

    # ---------- PRIME/ACTION summary sections at the top ----------
    def render_board(title, board):
        out = ["<h2>{}</h2>".format(title)]
        if not board:
            out.append("<p class='small'>No plays today.</p>")
            return "".join(out)
        out.append("<table><thead><tr>"
                   "<th>Track</th><th class='mono'>Race</th><th class='mono'>#</th><th>Horse</th>"
                   "<th class='mono'>Win% (Final)</th><th class='mono'>Market%</th><th class='mono'>Edge</th>"
                   "<th class='mono'>Fair</th><th class='mono'>Min Price</th><th class='mono'>Market</th>"
                   "<th class='mono'>Bet</th><th>Flags</th><th>Source</th></tr></thead><tbody>")
        keyer = (lambda x: (x["track"].lower(), int(x["race"]), -x["bet"], -x["p"])) if title.startswith("PRIME") \
                else (lambda x: (x["track"].lower(), int(x["race"]), -x["p"]))
        for b in sorted(board, key=keyer):
            market_pct = ("{:.1f}%".format(100.0*(b.get('imp') or 0.0)) if b.get("imp") is not None else "—")
            edge_str = ("{:.1f} pp".format(100.0*((b.get('edge') or 0.0))) if b.get("edge") is not None else "—")
            src = b.get("source","PRO")
            src_badge = "<span class='badge {}'>{}</span>".format('train' if src!='PRO' else 'pro', _html.escape(src))
            out.append(
                "<tr>" +
                "<td>{}</td><td class='mono'>{}</td><td class='mono'>{}</td><td>{}</td>"
                "<td class='mono'>{:.2f}%</td><td class='mono'>{}</td><td class='mono'>{}</td>"
                "<td class='mono'>{}</td><td class='mono'>{}</td><td class='mono'>{}</td>"
                "<td class='mono'>{}</td><td>{}</td><td>{}</td>".format(
                    _html.escape(b['track']), b['race'], _html.escape(b['num']), _html.escape(b['name']),
                    100.0*b['p'], market_pct, edge_str,
                    odds_formats(b['fair']), odds_formats(b['minp']), odds_formats(b['market']),
                    ('$'+format(int(round(b['bet'])),',d')) if (b['bet'] and b['bet']>0) else '—',
                    _html.escape(b.get('flags') or ''),
                    src_badge
                ) + "</tr>"
            )
        out.append("</tbody></table>")
        return "".join(out)

    parts[prime_anchor]  = render_board("PRIME Board",  prime_board)
    parts[action_anchor] = render_board("ACTION Board", action_board)

    # Save locks (if changed)
    try:
        if LOCK_ENABLE:
            _save_locks(locks)
    except Exception as _e:
        log(f"locks save error: {_e}")

    parts.append("</body></html>")
    return "\n".join(parts)

# ---------------- Main ----------------
if __name__ == "__main__":
    try:
        iso_today = date.today().isoformat()
        log("[run] {}  starting steve_horses_pro.py".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        model_loaded = load_model()
        log("model loaded: {}".format(model_loaded))

        cards, scr_summary, auto_summary, scr_details = build_cards_and_scratches(iso_today)
        try:
            n_tracks = len(cards)
            n_races = sum(len(v) for v in cards.values())
            log("Tracks: {}  Races: {}".format(n_tracks, n_races))
        except Exception:
            pass

        html_out = build_report(cards, iso_today, scr_summary, auto_summary, scr_details)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / "{}_horses_targets+full.html".format(iso_today)
        out_path.write_text(html_out, encoding="utf-8")
        log("[ok] wrote {}".format(out_path))
    except Exception as e:
        log("[FATAL] build report failed: {}".format(e))
        try:
            last = sorted(OUT_DIR.glob("*_horses_targets+full.html"))[-1]
            log("[fallback] Last good report: {}".format(last))
        except Exception:
            pass