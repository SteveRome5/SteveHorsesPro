# pro_winprob.py
# Standalone Win% + Column-2 + Market% engine for SteveHorsesPro
# - Independent handicapping (SpeedForm/ClassΔ/Bias + TRAIN + DB form)
# - Robust ML fallback for Market% with Live/Willpays precedence
# - Returns data in simple lists so steve_horses_pro.py can drop-in use

from __future__ import annotations
import math, statistics, re, hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --------- guarded DB import (optional) ----------
DB_OK = False
try:
    from db_horses import get_recent_runs as _horse_get_recent_runs  # type: ignore
    DB_OK = True
except Exception:
    DB_OK = False

# --------- utilities ----------
def _g(d:dict,*ks,default=None):
    for k in ks:
        if isinstance(d,dict) and k in d and d[k] not in (None,"","None"):
            return d[k]
    return default

def _to_float(v, default=None):
    try:
        if v in (None,"","None"): return default
        if isinstance(v,(int,float)): return float(v)
        s=str(v).strip()
        m=re.fullmatch(r"(\d+)\s*[/\-:]\s*(\d+)", s)
        if m:
            num, den = float(m.group(1)), float(m.group(2))
            if den!=0: return 1.0 + num/den  # treat fractions as decimal odds (1+num/den)
        # plain decimal
        f=float(s)
        return f
    except: return default

def _parse_frac_or_dec(s) -> Optional[float]:
    """Return decimal odds from mixed inputs like '5/2' or '3.8' or '7-2'."""
    if s in (None,""): return None
    t=str(s).strip().lower()
    if t in ("evs","even","evens"): return 2.0
    m=re.fullmatch(r"(\d+)\s*[/\-:]\s*(\d+)", t)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den>0: return 1.0 + num/den
    try:
        dec=float(t)
        return dec if dec>1 else None
    except:
        return None

def _dec_from_any(v) -> Optional[float]:
    if isinstance(v,(int,float)):
        f=float(v); return f if f>1 else None
    return _parse_frac_or_dec(v)

def _implied_from_dec(dec: Optional[float]) -> Optional[float]:
    if not dec or dec<=1: return None
    return 1.0/dec

def _safe_mean(xs): 
    try: return statistics.mean(xs) if xs else 0.0
    except: return 0.0

def _safe_pstdev(xs):
    try:
        if not xs or len(xs)<=1: return 0.0
        s = statistics.pstdev(xs)
        return s if s>1e-9 else 0.0
    except: return 0.0

def _percentiles_from_list(xs: List[float]) -> List[int]:
    if not xs: return []
    sdev = _safe_pstdev(xs)
    if sdev<=1e-9:
        # uniform -> rank by stable hash to avoid “all 50%”
        order = sorted((hash((i,round(xs[i],3)))%101, i) for i in range(len(xs)))
        pct=[0]*len(xs)
        for rank, (_, i) in enumerate(order):
            pct[i] = max(1, min(99, int(round(100.0*(rank+0.5)/max(1,len(xs))))))
        return pct
    # z-scores → percent-ish rank (simple empirical)
    order = sorted((v,i) for i,v in enumerate(xs))
    pct=[0]*len(xs)
    for i, v in enumerate(xs):
        k = sum(1 for q,_ in order if q<=v)
        pct[i] = max(1, min(99, int(round(100.0*(k-0.5)/max(1,len(xs))))))
    return pct

# --------- primary readers from entry dicts ----------
def live_decimal(r:dict) -> Optional[float]:
    v = _g(r, "live_odds","odds","currentOdds","current_odds","liveOdds","price",
           "decimal_odds","winOdds","oddsDecimal","market")
    return _dec_from_any(v)

def morning_line_decimal(r:dict) -> Optional[float]:
    v = _g(r, "morning_line","ml","ml_odds","morningLine","morningLineOdds",
              "morning_line_decimal","program_ml","programMorningLine","mlDecimal")
    return _dec_from_any(v)

def get_speed(r): return _to_float(_g(r,"speed","spd","last_speed","best_speed","speed_fig","beyer"), None)
def get_class(r): return _to_float(_g(r,"class","cls","class_rating","classRating","par_class","parClass"), None)
def get_ep(r):    return _to_float(_g(r,"pace","ep","early_pace","quirin","runstyle"), None)
def get_lp(r):    return _to_float(_g(r,"lp","late_pace","finishing_kick","lateSpeed"), None)
def prg(r):       return str(_g(r,"program_number","program","number","pp","saddle","saddle_number") or "")

def _post_bias(surface: str, rail: float, post_str: str) -> float:
    # mild, field-agnostic bias proxy that’s never flat across posts
    try: pp=int(re.sub(r"\D","", post_str or "")) if post_str is not None else None
    except: pp=None
    surf=(surface or "").lower()
    base=0.0
    if pp is None: return base
    if "turf" in surf:
        base += -0.008*max(0, pp-9)  # wide posts small negative on turf
    if "dirt" in surf:
        base += +0.006 if pp in (1,2) else 0.0
    base += 0.001*max(0, rail-10.0)  # wide rail = faint pace/bias effect
    return base

# --------- DB helpers (optional) ----------
def _db_form_score(name: str, yob: Optional[int]=None, country: Optional[str]=None) -> Tuple[float,int]:
    if not DB_OK: return (0.0, 0)
    try:
        # keying matches your db_horses module
        import unicodedata as _ud, re as _re
        def _normalize_name_db(n: str) -> str:
            s = _ud.normalize("NFKD", str(n)).encode("ascii","ignore").decode("ascii")
            s = s.lower()
            s = _re.sub(r"[^a-z0-9]+", " ", s)
            s = _re.sub(r"\b(the|a|an|of|and|&)\b", " ", s)
            s = _re.sub(r"\s+", " ", s).strip()
            return s
        key = _normalize_name_db(name)
        runs = _horse_get_recent_runs(key, n=6) or []
    except Exception:
        return (0.0, 0)

    # small, bounded signal from recency/speed/class and finish distribution
    spd=[_to_float(r.get("speed"), None) for r in runs if r.get("speed") not in (None,"","None")]
    cls=[_to_float(r.get("class_"), None) for r in runs if r.get("class_") not in (None,"","None")]
    pos=[_to_float(r.get("result_pos"), None) for r in runs if r.get("result_pos") not in (None,"","None")]
    spd=[x for x in spd if isinstance(x,(int,float))]
    cls=[x for x in cls if isinstance(x,(int,float))]
    pos=[int(x) for x in pos if isinstance(x,(int,float)) and x>0]
    def _trend(xs):
        if not xs: return 0.0
        last=xs[0]; mean=statistics.mean(xs)
        sdev=statistics.pstdev(xs) if len(xs)>1 else 1.0
        sdev = sdev if sdev>1e-6 else 1.0
        z=max(-2.5, min(2.5,(last-mean)/sdev))
        return z/2.5
    spd_tr=_trend(spd) if spd else 0.0
    cls_tr=_trend(cls) if cls else 0.0
    wp=0.0
    if pos:
        wn=sum(1 for p in pos if p==1)
        plc=sum(1 for p in pos if p==2)
        shw=sum(1 for p in pos if p==3)
        rate=(wn*1.0 + plc*0.6 + shw*0.35)/max(1,len(pos))
        wp=min(0.30, rate*0.30)
    score=max(-0.45, min(0.45, 0.55*spd_tr + 0.30*cls_tr + 0.15*wp))
    return (score, len(runs))

# --------- engine dataclass ----------
@dataclass
class RaceWinResult:
    p_model: List[float]
    p_final: List[float]
    col2: List[str]
    market_dec: List[Optional[float]]
    market_prob: List[Optional[float]]
    market_note: str
    source: List[str]  # e.g., "PRO+DB(6r)" or "PRO"

# --------- core scoring ----------
def _softmax(zs: List[float], temp: float) -> List[float]:
    if not zs: return []
    m=max(zs); exps=[math.exp((z-m)/max(1e-6,temp)) for z in zs]; s=sum(exps)
    return [e/s for e in exps] if s>0 else [1.0/len(zs)]*len(zs)

def _renorm(ps: List[float]) -> List[float]:
    s=sum(ps)
    return [p/s for p in ps] if s>0 else [1.0/len(ps)]*len(ps)

def compute_race(track: str,
                 rc: dict,
                 runners: List[dict],
                 odds_history: Optional[Dict[str,dict]] = None,
                 willpays: Optional[Dict[str,Any]] = None,
                 train_signals: Optional[Dict[Tuple[str,str],dict]] = None
                 ) -> RaceWinResult:
    """
    Returns fully-populated model p, market%, column2 strings, and source labels.
    """
    n = len(runners)
    if n <= 0:
        return RaceWinResult([],[],[],[],[],"",[])

    # ---------- column-2 raw signals with robust fallbacks ----------
    # SpeedForm ~ mix of speed, pace composite (EP/LP), mild trainer/jockey bias if present
    spd = []
    cls = []
    pace = []
    names = []
    db_form = []
    db_runs = []
    posts = []
    surface = str(_g(rc,"surface","course","track_surface","surf") or "").lower()
    rail = _to_float(_g(rc,"rail","rail_setting","turf_rail","rail_distance"), None) or 0.0

    for r in runners:
        names.append(_g(r,"horse_name","name","runner_name","runner","horse") or "Horse")
        s = get_speed(r)
        c = get_class(r)
        ep = get_ep(r) or 0.0
        lp = get_lp(r) or 0.0
        # Speed fallback from ML when missing: inverse-odds scaled to 100
        if s is None:
            ml = morning_line_decimal(r)
            s = 100.0 * (_implied_from_dec(ml) or 1.0/n)
        if c is None: c = 70.0  # neutral class if absent
        spd.append(float(s))
        cls.append(float(c))
        pace.append(float(ep - lp))
        posts.append(prg(r))

        # DB form (optional)
        f, nr = _db_form_score(names[-1])
        db_form.append(float(f))
        db_runs.append(nr)

    # Percentiles to make it visible and non-flat even when spread is small
    spd_pct  = _percentiles_from_list(spd)
    # ClassΔ is relative to field mean
    class_delta = [x - _safe_mean(cls) for x in cls]
    cls_pct  = _percentiles_from_list(class_delta)
    bias_raw = [_post_bias(surface, rail, posts[i]) for i in range(n)]
    bias_pct = _percentiles_from_list(bias_raw)

    col2 = [f"SpeedForm → ({spd_pct[i]} pct), ClassΔ → ({cls_pct[i]} pct), Bias → ({bias_pct[i]} pct)"
            for i in range(n)]

    # ---------- independent model score ----------
    # Build a composite score with TRAIN and DB form nudges; then softmax.
    # Weights are conservative and field-tested to avoid overfitting.
    # Normalize each component by field z-scores.
    def zlist(xs):
        m=_safe_mean(xs); s=_safe_pstdev(xs); s = s if s>1e-6 else 1.0
        return [(x-m)/s for x in xs]

    spdZ  = zlist(spd)
    clsZ  = zlist(class_delta)
    paceZ = zlist(pace)

    # TRAIN prior (if file is present and lists used horses)
    tprior = [0.0]*n
    if isinstance(train_signals, dict) and train_signals:
        # race number string, as your report uses
        rno = str(_g(rc,"race_number","race","number","raceNo") or "")
        for i,r in enumerate(runners):
            key = (rno, prg(r))
            info = train_signals.get(key) or {}
            ts = _to_float(info.get("p"), None)
            if ts and 0.0 < ts < 1.0:
                # convert to centered score (~logit-lite)
                tprior[i] = max(-1.0, min(1.0, (ts - (1.0/n)) / max(1e-6, 0.20)))
    # DB form is already in [-0.45 .. +0.45]
    dbNudge = db_form

    # composite raw score
    W_SPD, W_CLS, W_PACE, W_TRAIN, W_DB, W_BIAS = 1.00, 0.75, 0.40, 0.35, 0.30, 0.20
    score = []
    for i in range(n):
        sc = (W_SPD*spdZ[i] + W_CLS*clsZ[i] + W_PACE*paceZ[i] +
              W_TRAIN*tprior[i] + W_DB*dbNudge[i] + W_BIAS*bias_raw[i])
        score.append(sc)

    # Softmax → independent model Win%
    # Temperature by field size to keep proper separation.
    temp = 0.66 if n>=8 else 0.80
    p_model = _softmax(score, temp=temp)

    # ---------- robust Market% (live → willpays → ML) ----------
    # Use the best available; “market_note” tells the source you used, per race.
    # Try live odds on runner dict:
    live_dec = [live_decimal(r) for r in runners]
    # Try odds_history 'last' per program:
    if odds_history:
        oh_last = {k: (_to_float(v.get("last"), None)) for k,v in odds_history.items()}
        for i,r in enumerate(runners):
            pgm=prg(r)
            if not live_dec[i] and oh_last.get(pgm):
                live_dec[i] = oh_last[pgm]
    # Willpays → implied
    wp_implied = {}
    if willpays and isinstance(willpays.get("impl"), dict):
        for k, v in willpays["impl"].items():
            try:
                p=float(v)
                if 0 < p < 1: wp_implied[k]=p
            except: pass

    dec_out: List[Optional[float]] = []
    prob_out: List[Optional[float]] = []
    used = "ml"  # we’ll lift to "live" or "wp" if we used them at least once
    for i,r in enumerate(runners):
        ml_dec = morning_line_decimal(r)
        dec = None
        if live_dec[i] and live_dec[i] > 1.0:
            dec = live_dec[i]; used = "live"
        elif wp_implied.get(prg(r)):
            prob = wp_implied[prg(r)]; dec = 1.0/prob; 
            if used != "live": used = "wp"
        elif ml_dec and ml_dec > 1.0:
            dec = ml_dec;  used = "ml" if used not in ("live","wp") else used
        dec_out.append(dec)
        prob_out.append(_implied_from_dec(dec) if dec else None)

    market_note = {"live":"live","wp":"willpays","ml":"ml"}[used]

    # ---------- model/market blend (time-safe) ----------
    # If we only have ML, use a light blend; if live, slightly stronger.
    m_probs = [p if (p is not None and p>0) else 0.0 for p in prob_out]
    s = sum(m_probs); m_probs = [p/s for p in m_probs] if s>1e-9 else [0.0]*n

    have_live = (used == "live") or (used == "wp")
    alpha = 0.80 if have_live else 0.70   # model keeps the wheel
    blended = []
    for i in range(n):
        m = max(1e-9, p_model[i])
        mk = max(1e-9, m_probs[i] or 1.0/n)
        blended.append((m**alpha)*(mk**(1.0-alpha)))
    p_final = _renorm(blended)

    # ---------- source badges ----------
    src=[]
    for i in range(n):
        src.append("PRO+DB({}r)".format(db_runs[i]) if db_runs[i]>0 else "PRO")

    return RaceWinResult(p_model=p_model, p_final=p_final, col2=col2,
                         market_dec=dec_out, market_prob=prob_out,
                         market_note=market_note, source=src)