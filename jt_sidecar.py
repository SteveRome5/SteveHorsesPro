#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# JT sidecar — builds data/signals/<track>|<YYYY-MM-DD>.json
# Uses REAL trainer/jockey fields from the same API your main script uses.

from __future__ import annotations
import os, json, base64, ssl, re
from datetime import date
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from pathlib import Path

# --- paths ---
HOME = Path.home()
ROOT = HOME / "Desktop" / "SteveHorsesPro"
DATA = ROOT / "data" / "signals"
DATA.mkdir(parents=True, exist_ok=True)

# --- API creds (same env vars as main) ---
RUSER = os.getenv('RACINGAPI_USER') or os.getenv('RACINGAPI_USER'.upper())
RPASS = os.getenv('RACINGAPI_PASS') or os.getenv('RACINGAPI_PASS'.upper())
API_BASE = os.getenv("RACING_API_BASE", "https://api.theracingapi.com")
CTX = ssl.create_default_context()

EP_MEETS           = "/v1/north-america/meets"
EP_ENTRIES_BY_MEET = "/v1/north-america/meets/{meet_id}/entries"

def _get(path, params=None):
    url = API_BASE + path + ("?" + urlencode(params) if params else "")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if RUSER and RPASS:
        tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
        req.add_header("Authorization", "Basic " + tok)
    with urlopen(req, timeout=30, context=CTX) as r:
        return json.loads(r.read().decode("utf-8","replace"))

def g(d: dict, *ks, default=None):
    for k in ks:
        if isinstance(d, dict) and k in d and d[k] not in (None, "", "None"):
            return d[k]
    return default

def _to_float(v, default=None):
    try:
        if v in (None, "", "None"): return default
        if isinstance(v, (int, float)): return float(v)
        return float(str(v).strip())
    except Exception:
        return default

def _meet_id(m, iso_date):
    return (
        g(m, "meet_id","meetId","id","uuid","key","meetUUID","meetUid") or
        f"{g(m,'track_id','trackId','track','abbr') or (g(m,'track_name') or '').strip().lower()}_{iso_date}"
    )

def _build_scores_for_race(runners):
    """
    Convert trainer/jockey/combo win rates into a **bounded** probability-like score,
    with flags. This is the *actual* JT signal from the vendor data, not a placeholder.
    """
    out = {}
    for r in runners or []:
        pgm = str(g(r,"program_number","program","number","pp","saddle","saddle_number") or "")
        if not pgm: continue
        tr = _to_float(g(r,"trainer_win_pct","trainerWinPct"), 0.0) / 100.0
        jk = _to_float(g(r,"jockey_win_pct","jockeyWinPct"),   0.0) / 100.0
        tj = _to_float(g(r,"tj_win","combo_win"),              0.0) / 100.0

        # Weighted blend; bounded so it never dominates the model.
        raw = 0.55*tr + 0.35*jk + 0.10*tj
        score = min(0.35, max(0.02, raw))   # keep it in (2%..35%)

        flags = []
        if tr >= 0.20: flags.append("TrainerHot")
        if jk >= 0.20: flags.append("JockeyHot")
        if tj >= 0.22: flags.append("ComboHot")

        out[pgm] = {
            "used": True,
            "score": round(score, 4),
            "flags": flags,
            "why": "JT blend (real %)"
        }
    return out

def _write_meet_sidecar(track, iso_date, races):
    rows = []
    for rc in races:
        rno = str(g(rc, "race_number","number","race","raceNo") or "")
        # normalize runners list like the main script does
        runners = (
            rc.get("runners") or rc.get("entries") or rc.get("horses") or rc.get("starters") or []
        )
        per_prog = _build_scores_for_race(runners)
        for pgm, row in per_prog.items():
            rows.append({"race": rno, "program": pgm, **row})

    path = DATA / f"{track}|{iso_date}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print("JT sidecar wrote", path)

def main():
    iso = date.today().isoformat()
    meets = _get(EP_MEETS, {"start_date": iso, "end_date": iso}).get("meets", []) or []

    for m in meets:
        track = g(m, "track_name","track","name") or "Track"
        mid = _meet_id(m, iso)
        try:
            entries = _get(EP_ENTRIES_BY_MEET.format(meet_id=mid)) or {}
            races = entries.get("races") or entries.get("entries") or []
            if races:
                _write_meet_sidecar(track, iso, races)
        except Exception as e:
            print("JT sidecar skip", track, ":", e)

if __name__ == "__main__":
    main()
