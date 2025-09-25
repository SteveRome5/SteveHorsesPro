#!/usr/bin/env python3
# Fast runner for Steve Horses Pro — keep all tweaks here, not in the big file.

import os, json, base64, time, datetime as dt, traceback
from urllib.parse import urlencode

# ========= 0) Credentials & warnings =========
os.environ.setdefault("RACINGAPI_USER", "WQaKSMwgmG8GnbkHgvRRCT0V")
os.environ.setdefault("RACINGAPI_PASS", "McYBoQViXSPvlNcvxQi1Z1py")
# Silence the LibreSSL/urllib3 noise completely
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# ========= 1) Your knobs live here (change safely) =========
PATCHES = {
    # bankroll & risk
    "BANKROLL": 20000.0,
    "DAILY_EXPOSURE_CAP": 0.12,
    "KELLY_CAP": 0.15,
    "MAX_BET_PER_HORSE": 1500.0,
    "MIN_STAKE": 50.0,
    "ACTION_MAX_PER": 400.0,

    # selection logic
    "EDGE_WIN_PCT_FLOOR": 0.20,   # PRIME min win%
    "ACTION_PCT_FLOOR":   0.13,   # ACTION min win%
    "EDGE_PP_MIN_PRIME":  3.0,    # PRIME edge (pp) vs market
    "EDGE_PP_MIN_ACTION": 5.0,    # ACTION edge (pp) vs market

    # pricing pad
    "BASE_MIN_PAD": 0.22,

    # odds source
    "USE_LIVE": True,             # True = use live if available, else WP/ML fallback

    # market blend schedule (alpha = weight on model; 1-alpha on market)
    "BLEND_ALPHA_SLOW": 0.93,     # >= 20 MTP
    "BLEND_ALPHA_MID":  0.88,     # 8–19 MTP
    "BLEND_ALPHA_LATE": 0.80,     # <= 7 MTP
}

# ========= 2) Import the engine =========
import steve_horses_pro as shp

# ========= 3) Apply patches (no editing the big file) =========
def _apply_patches():
    # Map simple constants
    for k, v in PATCHES.items():
        if hasattr(shp, k):
            setattr(shp, k, v)

    # Toggle live odds via env the engine already reads
    os.environ["LIVE_ODDS"] = "1" if PATCHES.get("USE_LIVE", True) else "0"

    # Override the market blend with your alpha schedule (optional but handy)
    def blend_with_market_if_present(p_model, p_market, minutes_to_post):
        if not p_market or all(x is None for x in p_market):
            return p_model
        pm = [0.0 if (x is None or x <= 0) else float(x) for x in p_market]
        sm = sum(pm)
        pm = [x/sm if sm > 0 else 0.0 for x in pm]

        if minutes_to_post is None:
            alpha = PATCHES["BLEND_ALPHA_MID"]
        elif minutes_to_post >= 20:
            alpha = PATCHES["BLEND_ALPHA_SLOW"]
        elif minutes_to_post >= 8:
            alpha = PATCHES["BLEND_ALPHA_MID"]
        else:
            alpha = PATCHES["BLEND_ALPHA_LATE"]

        blended = [(max(1e-9,m)**alpha)*(max(1e-9,mk)**(1.0-alpha)) for m,mk in zip(p_model, pm)]
        s = sum(blended)
        return [b/s for b in blended] if s>0 else p_model

    shp.blend_with_market_if_present = blend_with_market_if_present

# ========= 4) Helper: make a dated backup of the engine =========
def _backup_engine():
    try:
        base = os.path.dirname(shp.__file__)
        src  = os.path.join(base, "steve_horses_pro.py")
        date = dt.date.today().isoformat()
        dst  = os.path.join(base, "history", f"steve_horses_pro_{date}.py.bak")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(src) and not os.path.exists(dst):
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
    except Exception:
        pass

# ========= 5) Run end-to-end =========
def main():
    t0 = time.time()
    _backup_engine()
    _apply_patches()

    today = dt.date.today().isoformat()
    print(f"[info] Running cards for {today} …")

    # Build cards + scratches
    t1 = time.time()
    cards, scr_summary, auto_summary, scr_details = shp.build_cards_and_scratches(today)
    print(f"[info] Cards built in {time.time()-t1:0.1f}s; tracks={len(cards)}")

    # Train (optional: comment out if you want to skip)
    trained = shp.train_models_from_history(days_back=120, min_rows=160)
    print(f"[info] Trained buckets: {trained}")

    # Build report
    t2 = time.time()
    html = shp.build_report(cards, today, scr_summary, auto_summary, scr_details)
    out_path = os.path.join(str(shp.OUT_DIR), f"{today}_horses_targets+full.html")
    os.makedirs(str(shp.OUT_DIR), exist_ok=True)

    # Stamp settings at the top for traceability
    patch_note = "<!-- PATCHES " + json.dumps(PATCHES, sort_keys=True) + " -->\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(patch_note + html)

    print(f"[ok] Wrote {out_path} in {time.time()-t2:0.1f}s")
    print(f"[done] Total runtime {time.time()-t0:0.1f}s")

    # Open it
    try:
        os.system(f'open "{out_path}"')
    except Exception:
        pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[fatal]", e)
        traceback.print_exc()