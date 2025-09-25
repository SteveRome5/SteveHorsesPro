# pro_overlays.py — PF-35 overlay stack (fast, pure-Python)
from __future__ import annotations
import math, statistics, re
from typing import List, Tuple, Dict, Any

def _g(d:dict,*ks,default=None):
    for k in ks:
        if isinstance(d,dict) and k in d and d[k] not in (None,"","None"):
            return d[k]
    return default

def _f(v, default=None):
    try:
        if v in (None,"","None"): return default
        return float(v)
    except: return default

def _speed(r): return _f(_g(r,"speed","spd","last_speed","lastSpeed"),0.0) or 0.0
def _ep(r):    return _f(_g(r,"pace","ep","early_pace","earlyPace"),0.0) or 0.0
def _lp(r):    return _f(_g(r,"lp","late_pace","latePace"),0.0) or 0.0
def _cls(r):   return _f(_g(r,"class","cls","class_rating"),0.0) or 0.0
def _tr(r):    return (_f(_g(r,"trainer_win_pct","trainerWinPct"),0.0) or 0.0) / 100.0
def _jk(r):    return (_f(_g(r,"jockey_win_pct","jockeyWinPct"),0.0)  or 0.0) / 100.0
def _tj(r):    return (_f(_g(r,"tj_win","combo_win"),0.0)            or 0.0) / 100.0
def _dsl(r):   return _f(_g(r,"days_since","dsl","daysSince","layoffDays","last_start_days"),30.0) or 30.0
def _prog(r):  return str(_g(r,"program_number","program","number","pp","saddle","saddle_number") or "")

def _z(arr: List[float]) -> Tuple[List[float], float, float]:
    if not arr: return [], 0.0, 1.0
    m = statistics.mean(arr)
    s = statistics.pstdev(arr) if len(arr)>1 else 1.0
    if s < 1e-6: s = 1.0
    return [ (x - m)/s for x in arr ], m, s

def _style(ep, lp):
    if ep - lp >= 8:  return "E"
    if ep - lp >= 3:  return "EP"
    if lp - ep >= 5:  return "S"
    return "P"

def _renorm(ps: List[float]) -> List[float]:
    ps = [max(1e-9, float(x)) for x in ps]
    s = sum(ps)
    return [x/s for x in ps] if s>0 else [1.0/len(ps)]*len(ps)

def _softmax(zs: List[float], temp: float) -> List[float]:
    if not zs: return []
    m = max(zs)
    exps = [math.exp((z - m)/max(1e-6,temp)) for z in zs]
    s = sum(exps)
    return [e/s for e in exps] if s>0 else [1.0/len(zs)]*len(zs)

def apply_all_overlays(track: str, surface: str, rail: float,
                       runners: List[Dict[str,Any]],
                       p_after_horse_db: List[float]) -> Tuple[List[float], List[str], Dict[str,float]]:
    n = len(runners)
    if n == 0:
        return [], [], {"pressure":0.0,"meltdown":0.0}
    base = _renorm(p_after_horse_db)

    spd = [_speed(r) for r in runners]
    ep  = [_ep(r)    for r in runners]
    lp  = [_lp(r)    for r in runners]
    cls = [_cls(r)   for r in runners]
    tr  = [_tr(r)    for r in runners]
    jk  = [_jk(r)    for r in runners]
    tj  = [_tj(r)    for r in runners]
    dsl = [_dsl(r)   for r in runners]

    spdZ,_,_ = _z(spd)
    clsZ,_,_ = _z(cls)

    # Pace pressure & meltdown proxies
    styles = [_style(ep[i],lp[i]) for i in range(n)]
    nE  = styles.count("E")
    nEP = styles.count("EP")
    frac_E = (nE + 0.5*nEP)/max(1,n)
    pressure = min(1.5, max(0.0, 0.20 + 0.65*frac_E))
    meltdown = min(1.0, max(0.0, (1.0 - frac_E) * 0.8 + (statistics.pvariance(ep) if len(ep)>1 else 0.0)/200.0))

    # Bias (very light): dirt-inside posts mild+, turf-wide rail mild-
    def _postnum(r):
        try:
            return int(re.sub(r"\D","", _prog(r)) or "0")
        except:
            return 0
    is_turf = ("turf" in (surface or "").lower())
    bias_raw = []
    for r in runners:
        bump = 0.0
        pn = _postnum(r)
        if not is_turf and pn in (1,2): bump += 0.10
        if is_turf and float(rail or 0.0) >= 20.0 and pn >= 10: bump -= 0.10
        bias_raw.append(bump)
    biasZ,_,_ = _z(bias_raw)

    # Trainer/Jockey synergy
    tj_syn = [0.0]*n
    for i in range(n):
        syn = 0.6*tr[i] + 0.4*jk[i] + 0.5*tj[i]
        tj_syn[i] = syn  # already 0..1 scale-ish

    # Form cycle: fresh 20–45 days = mild+, 0–8 = mild‑, 60+ = slight‑
    form = []
    for v in dsl:
        if 20 <= v <= 45: form.append(+0.12)
        elif v < 9:       form.append(-0.08)
        elif v >= 60:     form.append(-0.05)
        else:             form.append(0.0)

    # Multipliers (tight caps keep it stable and fast)
    flags = [[] for _ in range(n)]
    mul   = [1.0]*n
    for i in range(n):
        m1 = 1.0 + 0.045*spdZ[i]                    # SpeedForm
        if spdZ[i] >  0.60: flags[i].append("SF↑")
        elif spdZ[i] < -0.60: flags[i].append("SF↓")

        m2 = 1.0 + 0.035*clsZ[i]                    # ClassΔ
        if clsZ[i] >  0.60: flags[i].append("ClassΔ↑")
        elif clsZ[i] < -0.60: flags[i].append("ClassΔ↓")

        m3 = 1.0 + 0.030*biasZ[i]                   # Bias
        if biasZ[i] > 0.50: flags[i].append("Bias↑")
        elif biasZ[i] < -0.50: flags[i].append("Bias↓")

        # Pace pressure/meltdown interaction
        pace_tag = None
        if pressure >= 0.9 and styles[i] in ("E","EP"): 
            m4 = 1.06; pace_tag = "Pace+"
        elif meltdown >= 0.35 and styles[i] == "S":
            m4 = 1.07; pace_tag = "Closer+Meltdown"
        else:
            m4 = 1.00
        if pace_tag: flags[i].append(pace_tag)

        # Trainer/Jockey synergy
        m5 = 1.0 + min(0.06, max(-0.06, tj_syn[i]-0.12))

        # Form cycle
        m6 = 1.0 + form[i]

        mul[i] = max(0.85, min(1.15, m1*m2*m3*m4*m5*m6))

    p = [base[i]*mul[i] for i in range(n)]
    p = _renorm(p)

    # Still flat? force separation by proxy
    rng = (max(p)-min(p)) if p else 0.0
    var = statistics.pvariance(p) if len(p)>1 else 0.0
    if rng < 0.035 or var < 1e-5:
        pf_score = [0.6*spdZ[i] + 0.25*clsZ[i] + 0.15*(1 if styles[i] in ("E","EP") else 0) for i in range(n)]
        p = _softmax(pf_score, temp=0.60)

    out_flags = [" ".join(sorted(set(f))) if f else "" for f in flags]
    return p, out_flags, {"pressure": float(pressure), "meltdown": float(meltdown)}
