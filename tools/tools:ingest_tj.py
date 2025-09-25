#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trainer/Jockey daily combo backfill (robust extractor)
- Reads meets/races/results from the same API PRO uses
- Upserts per-day trainer/jockey combos into data/tj.sqlite:tj_daily
- Strong name extraction that tolerates many JSON shapes
"""

from __future__ import annotations
import os, ssl, json, base64, re, time, sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from collections import defaultdict

# ---------- Paths ----------
HOME = Path.home()
BASE = HOME / "Desktop" / "SteveHorsesPro"
DATA = BASE / "data"
TOOLS = BASE / "tools"
DATA.mkdir(parents=True, exist_ok=True)
(TOOLS).mkdir(parents=True, exist_ok=True)

# ---------- API (same as PRO) ----------
RUSER = os.getenv('RACINGAPI_USER') or os.getenv('RACINGAPI_USER'.upper())
RPASS = os.getenv('RACINGAPI_PASS') or os.getenv('RACINGAPI_PASS'.upper())
API_BASE = os.getenv("RACING_API_BASE", "https://api.theracingapi.com")
CTX = ssl.create_default_context()

EP_MEETS = "/v1/north-america/meets"
EP_ENTRIES_BY_MEET = "/v1/north-america/meets/{meet_id}/entries"
EP_RESULTS_BY_RACE  = "/v1/north-america/races/{race_id}/results"

def _get(path, params=None):
    url = API_BASE + path + ("?" + urlencode(params) if params else "")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if RUSER and RPASS:
        tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
        req.add_header("Authorization", "Basic " + tok)
    with urlopen(req, timeout=30, context=CTX) as r:
        return json.loads(r.read().decode("utf-8","replace"))

def safe_get(path, params=None, default=None):
    try:
        return _get(path, params)
    except Exception as e:
        print(f"[warn] GET fail {path}: {e}")
        return default

def g(d, *ks, default=None):
    for k in ks:
        if isinstance(d, dict) and k in d and d[k] not in (None, "", "None"):
            return d[k]
    return default

# ---------- Buckets (match PRO) ----------
def _surface_key(s: str) -> str:
    s = (s or "").lower()
    if "turf" in s: return "turf"
    if "synt" in s or "tapeta" in s or "poly" in s: return "synt"
    return "dirt"

def _dist_bucket_yards(yards: int | None) -> str:
    if not yards: return "unk"
    y = int(yards)
    if y < 1320:  return "<6f"
    if y < 1540:  return "6f"
    if y < 1760:  return "7f"
    if y < 1980:  return "1mi"
    if y < 2200:  return "8.5f"
    if y < 2420:  return "9f"
    return "10f+"

def build_bucket_key(track: str, surface: str | None, yards: int | None) -> str:
    return f"{track}|{_surface_key(surface or '')}|{_dist_bucket_yards(yards)}"

def _to_int(x, default=None):
    try: return int(float(x))
    except: return default

def _to_yards(rc) -> int | None:
    d = g(rc, "distance_yards","distance","dist_yards","yards","distanceYards","distance_y")
    if d is not None:
        return _to_int(d, None)
    m = g(rc,"distance_meters","meters","distanceMeters")
    if m is not None:
        try: return int(float(m)*1.09361)
        except: return None
    return None

# ---------- Name normalization (mirror PRO) ----------
import unicodedata as _ud
def _normalize_person_name(name: str) -> str:
    if not name: return ""
    s = _ud.normalize("NFKD", str(name)).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z\s]+", " ", s)
    s = re.sub(r"\b(the|a|an|of|and|&)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Pull a human name out of many possible shapes
def _pick_name(obj) -> str:
    if isinstance(obj, str):  # already a name
        return obj
    if not isinstance(obj, dict):
        return ""
    # common direct keys
    for k in ("full","full_name","name","display","displayName","trainerName","jockeyName",
              "riderName","jockey_full","trainer_full","first_last","firstLast"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # assemble from parts if present
    first = obj.get("first") or obj.get("first_name") or obj.get("given") or ""
    last  = obj.get("last")  or obj.get("last_name")  or obj.get("surname") or ""
    if isinstance(first,str) or isinstance(last,str):
        nm = f"{first} {last}".strip()
        if nm:
            return nm
    # nested wrappers
    for k in ("person","trainer","jockey","rider","nameObj","profile","info"):
        sub = obj.get(k)
        if isinstance(sub, (dict, str)):
            got = _pick_name(sub)
            if got: return got
    return ""

def _trainer_name_from_finisher(fin) -> str:
    # try multiple places
    for k in ("trainer","trainer_info","trainerInfo","connections","team","people"):
        v = fin.get(k)
        if isinstance(v, dict):
            # sometimes connections: {"trainer": {...}}
            if "trainer" in v:
                return _pick_name(v["trainer"])
            got = _pick_name(v)
            if got: return got
    # flat
    for k in ("trainer_name","trainerName","trainerFullName"):
        v = fin.get(k)
        if isinstance(v,str) and v.strip(): return v
    return ""

def _jockey_name_from_finisher(fin) -> str:
    for k in ("jockey","jockey_info","jockeyInfo","rider","rider_info","connections","team","people"):
        v = fin.get(k)
        if isinstance(v, dict):
            if "jockey" in v:  # connections: {"jockey": {...}}
                return _pick_name(v["jockey"])
            if "rider" in v:
                return _pick_name(v["rider"])
            got = _pick_name(v)
            if got: return got
    for k in ("jockey_name","jockeyName","jockeyFullName","rider_name"):
        v = fin.get(k)
        if isinstance(v,str) and v.strip(): return v
    return ""

def _finish_code(fin) -> str:
    return str(
        g(fin,"finish","finish_pos","finishPos","result","result_pos","position","placing","place") or ""
    ).strip().lower()

# ---------- DB ----------
DB = DATA / "tj.sqlite"
con = sqlite3.connect(str(DB))
con.executescript("""
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS tj_daily (
  dt TEXT NOT NULL,
  bucket TEXT NOT NULL,
  trainer_norm TEXT NOT NULL,
  jockey_norm TEXT NOT NULL,
  starts INTEGER NOT NULL,
  wins INTEGER NOT NULL,
  PRIMARY KEY (dt,bucket,trainer_norm,jockey_norm)
);
CREATE INDEX IF NOT EXISTS idx_tj_bucket ON tj_daily(bucket,trainer_norm,jockey_norm,dt);
""")
con.commit()

# ---------- API helpers ----------
def fetch_meets(iso_date): 
    d = safe_get(EP_MEETS, {"start_date": iso_date, "end_date": iso_date}, default={"meets":[]})
    return d.get("meets", [])

def fetch_entries(meet_id):
    d = safe_get(EP_ENTRIES_BY_MEET.format(meet_id=meet_id), default={"races":[]})
    return d.get("races") or d.get("entries") or []

def fetch_results(race_id):
    d = safe_get(EP_RESULTS_BY_RACE.format(race_id=race_id), default={}) or {}
    fins = g(d,"finishers","results","result","placings","order") or []
    # some vendors stash finishers under d["race"]["results"]
    if not fins and isinstance(d.get("race"), dict):
        fins = g(d["race"], "finishers","results","result") or []
    return fins

# ---------- Harvest ----------
def harvest_one(day: date) -> int:
    iso = day.isoformat()
    meets = fetch_meets(iso)
    combos = defaultdict(lambda: {"starts":0,"wins":0})
    races_count = 0

    for m in meets:
        track = g(m,"track_name","track","name") or "Track"
        mid = g(m,"meet_id","id","meetId")
        if not mid: continue
        races = fetch_entries(mid) or []
        for rc in races:
            # vendor race id for results:
            rid = g(rc,"race_id","id","raceId","raceID","eventId","event_id","raceKey","race_key",
                    "raceUid","race_uid","raceUUID","uuid","uuid_str")
            if not rid:
                continue
            surface = g(rc,"surface","track_surface","course","courseType","trackSurface","surf") or ""
            yards = _to_yards(rc)
            bucket = build_bucket_key(track, surface, yards)

            fins = fetch_results(rid) or []
            if not isinstance(fins, list):
                continue
            races_count += 1

            for fin in fins:
                tr_raw = _trainer_name_from_finisher(fin)
                jk_raw = _jockey_name_from_finisher(fin)
                tr = _normalize_person_name(tr_raw)
                jk = _normalize_person_name(jk_raw)
                if not (tr and jk): 
                    continue
                code = _finish_code(fin)
                won = (code in ("1","1st","win"))
                key = (iso, bucket, tr, jk)
                combos[key]["starts"] += 1
                combos[key]["wins"]   += (1 if won else 0)

            time.sleep(0.02)  # be gentle

    # upsert
    with con:
        for (dt,b,tr,jk),v in combos.items():
            con.execute(
                """INSERT INTO tj_daily(dt,bucket,trainer_norm,jockey_norm,starts,wins)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(dt,bucket,trainer_norm,jockey_norm)
                   DO UPDATE SET
                     starts = starts + excluded.starts,
                     wins   = wins   + excluded.wins""",
                (dt,b,tr,jk,v["starts"],v["wins"])
            )

    print(f"[{iso}] races={races_count} combos_upserted={len(combos)}")
    return len(combos)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--days", type=int, help="Backfill N days ending today (inclusive)")
    g.add_argument("--date", type=str, help="Single YYYY-MM-DD")
    args = ap.parse_args()

    if args.date:
        d0 = datetime.strptime(args.date, "%Y-%m-%d").date()
        harvest_one(d0)
    else:
        n = int(args.days or 120)
        today = date.today()
        total = 0
        for k in range(n-1, -1, -1):
            total += harvest_one(today - timedelta(days=k))
        tot_rows = con.execute("SELECT COUNT(*) FROM tj_daily").fetchone()[0]
        print(f"DONE. tj_daily rows: {tot_rows}  total upserts: {total}")

if __name__ == "__main__":
    main()