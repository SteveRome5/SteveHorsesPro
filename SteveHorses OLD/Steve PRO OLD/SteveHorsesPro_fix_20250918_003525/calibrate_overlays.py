#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_overlays.py
Learn overlay weights from historical results and emit a shell snippet that sets env vars.

Usage (recommended):
  python3 calibrate_overlays.py --days 365

Fallback (if DB loader can't find data):
  python3 calibrate_overlays.py --from-csv inputs/training_examples.csv

The script will write: outputs/overlay_tune_YYYY-MM-DD.sh
Source it like:
  source outputs/overlay_tune_2025-09-17.sh
"""

from __future__ import annotations
import os, sys, csv, json, math, statistics, argparse
from pathlib import Path
from datetime import date, datetime
from typing import List, Dict, Any, Tuple, Optional

# --- Paths (match your Pro layout) ---
HOME = Path.home()
BASE = HOME / "Desktop" / "SteveHorsesPro"
OUT_DIR = BASE / "outputs"; DATA_DIR = BASE / "data"; LOG_DIR = BASE / "logs"
for d in (OUT_DIR, DATA_DIR, LOG_DIR): d.mkdir(parents=True, exist_ok=True)

def log(s: str) -> None:
    try:
        (LOG_DIR / "calibrate_overlays.log").open("a", encoding="utf-8").write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {s}\n"
        )
    except Exception:
        pass

# -----------------------------
# 1) Data loading
# -----------------------------
EXPECTED_COLS = [
    # minimally useful columns (if you supply CSV)
    "track","surface","rail","post",
    "trainer_win_pct","jockey_win_pct","tj_win",
    "days_since","start_since_layoff",
    "ep","lp",
    "last_track","this_track",
    "win"  # 1 if horse won, else 0
]

def _to_float(v, default=None):
    try:
        if v in (None,"","NA","NaN"): return default
        return float(v)
    except: return default

def _to_int(v, default=None):
    try:
        if v in (None,"","NA","NaN"): return default
        return int(v)
    except: return default

def load_from_csv(path: Path) -> List[Dict[str,Any]]:
    rows=[]
    with path.open("r", encoding="utf-8", newline="") as f:
        rd=csv.DictReader(f)
        missing=[c for c in EXPECTED_COLS if c not in rd.fieldnames]
        if missing:
            log(f"CSV missing columns: {missing}")
        for r in rd:
            rows.append(r)
    return rows

def load_from_db(days: int) -> List[Dict[str,Any]]:
    """
    Best-effort DB loader. Tries to use db_results if present.
    We keep this generic; if your db_results exposes a different API,
    we'll just tell you to use --from-csv as fallback.
    """
    try:
        import db_results  # type: ignore
    except Exception as e:
        log(f"db_results import failed: {e}")
        return []

    rows=[]
    # Heuristic: look for a function that can yield runner-level records
    got=False
    for fname in ("export_training_rows","get_training_rows","iter_training_rows","iter_runner_rows"):
        fn=getattr(db_results, fname, None)
        if fn:
            try:
                tmp=list(fn(days=days))  # expect list[dict] or iterable
                for r in tmp:
                    rows.append(r)
                got=True
                break
            except Exception as e:
                log(f"db_results.{fname} days={days} failed: {e}")
    if not got:
        log("No compatible function found in db_results. Use --from-csv instead.")
    return rows

# -----------------------------
# 2) Feature engineering
# -----------------------------
def pace_style(ep: Optional[float], lp: Optional[float]) -> str:
    ep = ep or 0.0; lp = lp or 0.0
    if ep - lp >= 8:   return "E"
    if ep - lp >= 3:   return "EP"
    if lp - ep >= 5:   return "S"
    return "P"

def make_features(r: Dict[str,Any]) -> Tuple[List[float], int]:
    """
    Build a compact feature vector for overlays.
    Returns (x, y) where y in {0,1}
    """
    # y label
    y = _to_int(r.get("win"), 0) or 0

    surface = str(r.get("surface") or "").lower()
    rail    = _to_float(r.get("rail"), 0.0) or 0.0
    post    = _to_int(r.get("post"), None)

    tw = (_to_float(r.get("trainer_win_pct"), None) or 0.0)/100.0
    jw = (_to_float(r.get("jockey_win_pct"),  None) or 0.0)/100.0
    tj = (_to_float(r.get("tj_win"),           None) or 0.0)/100.0
    tj_base = max(tw, jw, tj)  # trainer/jockey "hotness" proxy

    dsl = _to_float(r.get("days_since"), None) or 0.0
    n_since = _to_int(r.get("start_since_layoff"), None)

    # cycle flags
    first_off_long   = 1.0 if (dsl >= 150 and (not n_since or n_since <= 1)) else 0.0
    second_off       = 1.0 if (n_since == 2) else 0.0
    third_off        = 1.0 if (n_since == 3) else 0.0
    fresh_21_60      = 1.0 if (21 <= dsl <= 60) else 0.0

    ep = _to_float(r.get("ep"), None)
    lp = _to_float(r.get("lp"), None)
    style = pace_style(ep, lp)
    is_closer = 1.0 if style == "S" else 0.0
    is_speed  = 1.0 if style in ("E","EP") else 0.0

    # bias-ish
    is_turf = 1.0 if "turf" in surface else 0.0
    rail_wide_bias = 1.0 if (is_turf and rail >= 20.0 and (post or 0) >= 9) else 0.0
    inside_bias    = 1.0 if ((not is_turf) and (post in (1,2))) else 0.0

    last_track = str(r.get("last_track") or "").lower()
    this_track = str(r.get("this_track") or "").lower()
    small = {"finger lakes","charlestown","mountaineer","penn national","canterbury","evangeline","lone star"}
    big   = {"saratoga","del mar","belmont","aqueduct","churchill downs","keeneland","gulfstream"}
    ship_up   = 1.0 if (last_track in small and this_track in big) else 0.0
    ship_down = 1.0 if (last_track in big and this_track in small) else 0.0

    x = [
        tj_base,             # 0 - trainer/jockey base hotness
        first_off_long,      # 1
        second_off,          # 2
        third_off,           # 3
        fresh_21_60,         # 4
        is_closer,           # 5
        is_speed,            # 6
        rail_wide_bias,      # 7
        inside_bias,         # 8
        ship_up,             # 9
        ship_down            # 10
    ]
    return x, y

# -----------------------------
# 3) Tiny ridge logistic regression
# -----------------------------
def sigmoid(z: float) -> float: return 1.0/(1.0 + math.exp(-max(-50.0, min(50.0, z))))

def fit_logreg_ridge(X: List[List[float]], y: List[int], reg: float=1.0, iters: int=500, lr: float=0.05) -> List[float]:
    if not X: return []
    n, d = len(X), len(X[0])
    w = [0.0]*d
    for _ in range(iters):
        grad = [0.0]*d
        for i in range(n):
            z = sum(w[j]*X[i][j] for j in range(d))
            p = sigmoid(z)
            err = p - (1.0 if y[i]==1 else 0.0)
            for j in range(d):
                grad[j] += err * X[i][j]
        # ridge penalty
        for j in range(d):
            grad[j] = grad[j]/n + reg*w[j]
            w[j] -= lr * grad[j]
    return w

# -----------------------------
# 4) Mapping coefficients -> overlay weights
# -----------------------------
def squash_positive(v: float, scale: float, hi: float) -> float:
    """
    Map any float to [0, hi] with a soft squashing that preserves ordering.
    hi is the max weight contribution.
    """
    # use tanh-like squashing
    return max(0.0, min(hi, hi * (1.0/(1.0 + math.exp(-v*scale)))))

def map_weights(w: List[float]) -> Dict[str,float]:
    """
    Convert coefficients into overlay weights:
      - W_TJ from w[0]
      - W_CYCLE from (w[1..4])
      - W_PACE from (w[5], w[6])
      - W_BIAS from (w[7], w[8])
      - W_SHIP from (w[9], w[10])
    The absolute “strength” (ALPHA) you already set (e.g., TJ_ALPHA) stays as-is unless you want to also tune it.
    """
    if not w: return {}
    # gentle caps so nothing dominates
    tj_w   = squash_positive(w[0], 1.50, 0.35)
    cyc_w  = squash_positive(sum(w[1:5]), 0.80, 0.30)
    pace_w = squash_positive((w[5] + w[6]), 0.80, 0.35)
    bias_w = squash_positive((w[7] + w[8]), 0.80, 0.25)
    ship_w = squash_positive((w[9] + w[10]),0.80, 0.20)
    return {
        "W_TJ":   round(tj_w, 3),
        "W_CYCLE":round(cyc_w, 3),
        "W_PACE": round(pace_w, 3),
        "W_BIAS": round(bias_w, 3),
        "W_SHIP": round(ship_w, 3),
    }

# -----------------------------
# 5) Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365, help="train on last N days from DB (if available)")
    ap.add_argument("--from-csv", type=str, default="", help="optional CSV path if DB loader not available")
    ap.add_argument("--reg", type=float, default=1.0, help="ridge regularization (higher = safer/smaller weights)")
    args = ap.parse_args()

    rows: List[Dict[str,Any]] = []
    if args.from_csv:
        p = Path(args.from_csv)
        if not p.exists():
            print(f"[ERR] CSV not found: {p}")
            print(f"Expected columns (at least a subset): {', '.join(EXPECTED_COLS)}")
            sys.exit(2)
        rows = load_from_csv(p)
        log(f"Loaded {len(rows)} rows from CSV {p}")
    else:
        rows = load_from_db(args.days)
        if not rows:
            print("[info] Could not load training rows from db_results.")
            print("       Provide a CSV instead, e.g.:")
            print("       inputs/training_examples.csv with columns:", ", ".join(EXPECTED_COLS))
            print("       Then run: python3 calibrate_overlays.py --from-csv inputs/training_examples.csv")
            sys.exit(0)

    X=[]; Y=[]
    for r in rows:
        try:
            x,y = make_features(r)
            X.append(x); Y.append(1 if y else 0)
        except Exception as e:
            # skip malformed
            continue

    if len(X) < 200:
        print(f"[info] Only {len(X)} usable training rows found — that’s light.")
        print("       We can still emit a cautious set of weights, but more history will improve calibration.")
    if not X:
        print("[ERR] No usable rows. Aborting.")
        sys.exit(2)

    w = fit_logreg_ridge(X, Y, reg=args.reg, iters=500, lr=0.05)
    weights = map_weights(w)

    # Compose output shell snippet
    iso = date.today().isoformat()
    outf = OUT_DIR / f"overlay_tune_{iso}.sh"
    lines = [
        "# Auto-generated overlay weights (calibrated)",
        f"# trained_on_rows={len(X)}  generated={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "# Usage:  source " + str(outf),
        "",
    ]
    for k,v in weights.items():
        lines.append(f"export {k}={v}")
    # keep your existing alphas unless you explicitly want to pin them here:
    # lines += ["export TJ_ALPHA=0.12", "export CYCLE_ALPHA=0.10", "export PACE_ALPHA=0.12", "export BIAS_ALPHA=0.08", "export SHIP_ALPHA=0.06"]

    outf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] Wrote {outf}")
    print("     To apply now, run:")
    print(f"       source {outf}")
    print("     Then re-run: python3 steve_horses_pro.py")

if __name__ == "__main__":
    main()