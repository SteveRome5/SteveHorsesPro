# pro_overlays.py
# Stable overlays for PF-35 Mach++ v3.12-pro-stable
# - Trainer/Jockey (TJ) overlay with flags
# - Cycle (recent activity) overlay
# - Pace Pressure context (pressure/meltdown diagnostics)
# - Bias nudge (minor rail/surface hint)
# - Shipper (very light)
# - Ensemble: mix model+market+nudges and return flags for HTML

from __future__ import annotations
import os, math, statistics, re
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

# ----------------- small utils -----------------
def _g(d: dict, *ks, default=None):
    for k in ks:
        if isinstance(d, dict) and k in d and d[k] not in (None, "", "None"):
            return d[k]
    return default

def _to_float(v, default=None):
    try:
        if v in (None, ""): return default
        if isinstance(v, (int, float)): return float(v)
        s = str(v).strip()
        m = re.fullmatch(r"(\d+)\s*[/\-:]\s*(\d+)", s)
        if m:  # fractional odds -> decimal-ish scalar; here we just fail through
            num, den = float(m.group(1)), float(m.group(2))
            if den != 0: return num / den
        return float(s)
    except Exception:
        return default

def _safe_mean(xs):
    try:
        return statistics.mean(xs) if xs else 0.0
    except Exception:
        return 0.0

def _norm_probs(ps: List[float]) -> List[float]:
    s = sum(ps)
    if s <= 0:
        return [1.0 / len(ps)] * len(ps) if ps else []
    return [max(1e-6, p) / s for p in ps]

def _cap01(x):
    return max(0.0, min(1.0, float(x)))

def _pace_style(r):
    ep = _to_float(_g(r, "pace", "ep", "early_pace", "earlyPace", "runstyle", "style", "quirin"), 0.0) or 0.0
    lp = _to_float(_g(r, "lp", "late_pace", "closer", "finishing_kick", "lateSpeed"), 0.0) or 0.0
    if ep - lp >= 8:   return "E"
    if ep - lp >= 3:   return "EP"
    if lp - ep >= 5:   return "S"
    return "P"

# ----------------- OVERLAYS -----------------

def overlay_trainer_jockey(runners: List[dict], p_after_horse_db: List[float]) -> Tuple[List[float], List[str]]:
    """
    TJ overlay:
      - reads trainer_win_pct, jockey_win_pct, combo_win (if present)
      - builds a 'hotness' score; adds flags like "TJ 22/18 ↑" or "TJ Hot"
      - multiplicative bump mixed by env TJ_ALPHA (default 0.15)
    """
    if not runners or not p_after_horse_db or len(runners) != len(p_after_horse_db):
        return p_after_horse_db, [""] * len(runners)

    enable = (os.getenv("TJ_ENABLE", "1") != "0")
    if not enable:
        return p_after_horse_db, [""] * len(runners)

    alpha = float(os.getenv("TJ_ALPHA", "0.15"))  # 0..0.35 safe
    alpha = max(0.0, min(0.5, alpha))

    # gather pct
    T = [(_to_float(_g(r, "trainer_win_pct", "trainerWinPct"), 0.0) or 0.0) for r in runners]
    J = [(_to_float(_g(r, "jockey_win_pct", "jockeyWinPct"), 0.0) or 0.0) for r in runners]
    C = [(_to_float(_g(r, "tj_win", "combo_win"), 0.0) or 0.0) for r in runners]

    # normalize to 0..1-ish (percentage inputs)
    tR = [t / 100.0 for t in T]
    jR = [j / 100.0 for j in J]
    cR = [c / 100.0 for c in C]

    # score: combo emphasized; fallback to T/J average
    score = []
    for i in range(len(runners)):
        base = cR[i] if cR[i] > 0 else (0.6 * tR[i] + 0.4 * jR[i])
        # center around ~0.15 baseline (typical tbred win%)
        s = max(-0.15, min(0.35, base - 0.15))
        score.append(s)

    # turn score into multiplier around 1.0
    # hot pairs can reach ~ +10–15% prob bump before renorm, cold slight down
    mult = [max(0.85, min(1.15, 1.0 + alpha * (s / 0.20))) for s in score]

    out = [p_after_horse_db[i] * mult[i] for i in range(len(runners))]
    out = _norm_probs(out)

    # build flags
    flags = []
    for i in range(len(runners)):
        t, j, c = T[i], J[i], C[i]
        hot = (c >= 20) or (t >= 20 and j >= 18)
        cold = (c > 0 and c < 10) or (t < 8 and j < 8)
        if hot:
            tag = f"TJ {int(round(t))}/{int(round(j))} ↑"
        elif cold:
            tag = f"TJ {int(round(t))}/{int(round(j))} ↓"
        else:
            # only print when we have at least one real pct
            if (t > 0 or j > 0 or c > 0):
                tag = f"TJ {int(round(t))}/{int(round(j))}"
            else:
                tag = ""
        flags.append(tag)
    return out, flags


def overlay_cycle(runners: List[dict], p_in: List[float]) -> Tuple[List[float], List[str]]:
    """
    Cycle overlay: gentle bump if days-since-last fits 15–45 window; penalize extreme layoffs.
    Uses env CYCLE_ALPHA (default 0.10). Safe & tiny.
    """
    if not runners or not p_in or len(runners) != len(p_in):
        return p_in, [""] * len(runners)
    enable = (os.getenv("CYCLE_ENABLE", "1") != "0")
    if not enable:
        return p_in, [""] * len(runners)

    alpha = float(os.getenv("CYCLE_ALPHA", "0.10"))
    alpha = max(0.0, min(0.4, alpha))

    def bump(dsl: Optional[float]) -> float:
        if dsl is None: return 1.0
        # sweet spot ~ 20–35
        if 15 <= dsl <= 45:   return 1.0 + alpha * 0.12
        if dsl < 10:          return 1.0 - alpha * 0.08
        if dsl > 90:          return 1.0 - alpha * 0.10
        return 1.0

    mult=[]; flags=[]
    for r in runners:
        dsl = _to_float(_g(r, "days_since","dsl","daysSince","layoffDays","last_start_days"), None)
        m = bump(dsl)
        mult.append(m)
        if dsl is None:
            flags.append("")
        elif 15 <= dsl <= 45:
            flags.append("Cycle✓")
        elif dsl < 10:
            flags.append("BackQuick")
        elif dsl > 90:
            flags.append("LongLayoff")
        else:
            flags.append("")

    out = _norm_probs([p_in[i] * mult[i] for i in range(len(p_in))])
    return out, flags


def overlay_pace_pressure(runners: List[dict], p_in: List[float]) -> Tuple[List[float], List[str], Dict[str, float]]:
    """
    Build race pace context:
      - pressure: abundance of E/EP types vs S
      - meltdown: chance closers benefit
    This overlay itself is neutral (no prob change) unless env PACE_ENABLE=1, but we always compute context.
    """
    if not runners:
        return p_in, [""] * 0, {"pressure": 0.0, "meltdown": 0.0}

    styles = [_pace_style(r) for r in runners]
    n = len(styles)
    nE = styles.count("E"); nEP = styles.count("EP"); nS = styles.count("S")
    pressure = (nE * 1.0 + 0.6 * nEP) / max(1, n)
    meltdown = max(0.0, min(1.0, (nS - nE) / max(1, n)))

    enable = (os.getenv("PACE_ENABLE", "1") != "0")
    if not enable or not p_in:
        # still return mini-flags for HTML
        flags = []
        for s in styles:
            if s == "E" and nE == 1 and nEP <= 1: flags.append("LoneE")
            elif s == "E" and nE >= 3:           flags.append("E-Herd")
            elif s == "S" and meltdown >= 0.25:  flags.append("Closer+Meltdown")
            else:                                 flags.append("")
        return p_in, flags, {"pressure": pressure, "meltdown": meltdown}

    # tiny tilt: help lone-E or strong closers in meltdown
    mult = []
    for s in styles:
        m = 1.0
        if s in ("E", "EP") and nE == 1 and nEP <= 1:
            m *= 1.02
        if s == "S" and meltdown >= 0.30:
            m *= 1.02
        mult.append(m)

    out = _norm_probs([p_in[i] * mult[i] for i in range(len(p_in))])

    flags = []
    for i, s in enumerate(styles):
        tag = []
        if s == "E" and nE == 1 and nEP <= 1: tag.append("LoneE")
        if s == "E" and nE >= 3:              tag.append("E-Herd")
        if s == "S" and meltdown >= 0.25:     tag.append("Closer+Meltdown")
        flags.append(" ".join(tag))
    return out, flags, {"pressure": pressure, "meltdown": meltdown}


def overlay_bias(surface: str, rail: float, runners: List[dict], p_in: List[float]) -> Tuple[List[float], List[str]]:
    """
    Micro bias: small nudge if wide rail on turf or inside bias on dirt. Extremely gentle.
    Env BIAS_ENABLE (default on), BIAS_ALPHA (default 0.06)
    """
    if not runners or not p_in:
        return p_in, [""] * len(runners)
    enable = (os.getenv("BIAS_ENABLE", "1") != "0")
    if not enable:
        return p_in, [""] * len(runners)

    alpha = float(os.getenv("BIAS_ALPHA", "0.06"))
    alpha = max(0.0, min(0.25, alpha))
    s = (surface or "").lower()
    mult = [1.0] * len(runners); flags=[""] * len(runners)

    # A simple heuristic: turf with big rail -> outside posts slightly penalized; dirt -> low posts tiny plus.
    def post_num(r):
        try:
            return int(re.sub(r"\D", "", str(_g(r, "program_number","program","number","pp","post_position","horse_number","saddle","saddle_number") or "")) or "0")
        except:
            return 0

    if "turf" in s and rail and rail >= 20.0:
        for i, r in enumerate(runners):
            p = post_num(r)
            if p >= 10:
                mult[i] *= (1.0 - 0.02 * alpha / 0.06)  # tiny down
                flags[i] = "RailWide"
    elif "dirt" in s:
        for i, r in enumerate(runners):
            p = post_num(r)
            if 1 <= p <= 2:
                mult[i] *= (1.0 + 0.015 * alpha / 0.06)
                flags[i] = "Inside+"

    out = _norm_probs([p_in[i] * mult[i] for i in range(len(p_in))])
    return out, flags


def overlay_shipper(runners: List[dict], p_in: List[float]) -> Tuple[List[float], List[str]]:
    """
    Very soft shipper tag if 'ship' flag/key exists (~cosmetic).
    Env SHIP_ENABLE (default on)
    """
    if not runners or not p_in:
        return p_in, [""] * len(runners)
    enable = (os.getenv("SHIP_ENABLE", "1") != "0")
    if not enable:
        return p_in, [""] * len(runners)

    flags=[]
    for r in runners:
        ship = str(_g(r, "ship", "shipper_flag", "is_shipper", "barn_ship") or "").strip().lower()
        if ship in ("1","true","y","yes"):
            flags.append("Shipper")
        else:
            flags.append("")
    # no price move; cosmetic
    return p_in, flags


# --------- apply_all_overlays (public) ---------
def apply_all_overlays(
    track: str,
    surface: str,
    rail: float,
    runners: List[dict],
    p_after_horse_db: List[float],
) -> Tuple[List[float], List[str], Dict[str, float]]:
    """
    Returns:
      p_final         : List[float] (normalized)
      overlay_flags   : List[str]   (merged per runner: TJ/Cycle/Pace/Bias/Ship)
      pace_ctx        : {"pressure": float, "meltdown": float}
    """
    N = len(runners)
    if N == 0:
        return [], [], {"pressure": 0.0, "meltdown": 0.0}

    # TJ
    p1, tj_flags = overlay_trainer_jockey(runners, p_after_horse_db)
    # Cycle
    p2, cyc_flags = overlay_cycle(runners, p1)
    # Pace pressure (also returns context)
    p3, pace_flags, pace_ctx = overlay_pace_pressure(runners, p2)
    # Bias
    p4, bias_flags = overlay_bias(surface, float(rail or 0.0), runners, p3)
    # Shipper (cosmetic)
    p5, ship_flags = overlay_shipper(runners, p4)

    # merge flags (dedupe while keeping order)
    merged = []
    for i in range(N):
        bits = [tj_flags[i], cyc_flags[i], pace_flags[i], bias_flags[i], ship_flags[i]]
        out = []
        seen = set()
        for b in bits:
            if b and b not in seen:
                out.append(b); seen.add(b)
        merged.append(" ".join(out).strip())

    return _norm_probs(p5), merged, pace_ctx


# Back-compat shim (old code may call this name)
def overlay_trainer_jockey_pulse(runners, p):
    # identical behavior to overlay_trainer_jockey (kept for older imports)
    return overlay_trainer_jockey(runners, p)