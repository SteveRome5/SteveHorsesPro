#!/usr/bin/env python3
# ingest_tj.py
# Robust, single-file ingestion for TJ data -> sqlite
# Compatible with Python 3.8+

from __future__ import print_function
import os
import sys
import time
import json
import base64
import sqlite3
import argparse
import traceback
from datetime import datetime, timedelta, date
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import ssl
import socket

# ---------------- Config ----------------
API_BASE = os.getenv("RACING_API_BASE", "https://api.theracingapi.com")
RUSER = os.getenv("RACINGAPI_USER") or os.getenv("RACINGAPI_USER".upper())
RPASS = os.getenv("RACINGAPI_PASS") or os.getenv("RACINGAPI_PASS".upper())

HOME = os.path.expanduser("~")
BASE = os.path.join(HOME, "Desktop", "SteveHorsesPro")
DATA_DIR = os.path.join(BASE, "data")
LOG_DIR = os.path.join(BASE, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "tj.sqlite")
CTX = ssl.create_default_context()

LOG_FILE = os.path.join(LOG_DIR, "ingest_tj.log")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line)

# ---------------- HTTP helpers ----------------
def _get(path, params=None, timeout=15, retries=2, backoff=0.6):
    """
    Return JSON from API path. Raises RuntimeError on persistent failures.
    """
    base = API_BASE.rstrip("/")
    url = base + path
    if params:
        qs = urlencode(params, doseq=True)
        url = f"{url}?{qs}"
    req = Request(url, headers={"User-Agent": "stevehorses/ingest_tj/1.0"})
    if RUSER and RPASS:
        tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
        req.add_header("Authorization", "Basic " + tok)
    last_exc = None
    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=timeout, context=CTX) as r:
                raw = r.read()
                try:
                    return json.loads(raw.decode("utf-8", "replace"))
                except Exception:
                    # sometimes API returns empty or non-json; return {}
                    return {}
        except Exception as e:
            last_exc = e
            # If 404 or similar, break early but we'll let caller handle None by safe_get
            # print minimal debug to logs
            log(f"HTTP error for {url!s}: {e}")
            # some errors are permanent (404) — if so, break retry loop
            if isinstance(e, Exception) and hasattr(e, 'code') and getattr(e, 'code') == 404:
                break
        time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"GET failed for {path} after {retries} retries: {last_exc}")

def safe_get(path, params=None, default=None):
    try:
        return _get(path, params)
    except Exception as e:
        log(f"GET fail {path} {params} -> {e}")
        return default

# ---------------- SQLite helpers ----------------
def ensure_db(conn):
    """
    Creates minimal schema if missing:
     - tj_daily: per-runner TJ stats
     - combos: pairings / combos (a simple store)
     - trainers/jockeys: optional lightweight rollup tables
    """
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS tj_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        track TEXT,
        race INTEGER,
        program TEXT,
        horse TEXT,
        trainer TEXT,
        jockey TEXT,
        tj_wins INTEGER,
        tj_starts INTEGER,
        tj_pct REAL,
        UNIQUE(date, track, race, program)
    );

    CREATE INDEX IF NOT EXISTS idx_tj_daily_date ON tj_daily(date);

    CREATE TABLE IF NOT EXISTS combos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        track TEXT,
        race INTEGER,
        program TEXT,
        combo TEXT,
        value REAL,
        UNIQUE(date, track, race, program, combo)
    );

    CREATE INDEX IF NOT EXISTS idx_combos_date ON combos(date);
    """)
    conn.commit()

def upsert_tj(conn, row):
    """
    row: dict with keys date, track, race, program, horse, trainer, jockey, tj_wins, tj_starts, tj_pct
    """
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO tj_daily (date, track, race, program, horse, trainer, jockey, tj_wins, tj_starts, tj_pct)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(date, track, race, program) DO UPDATE SET
      horse=excluded.horse,
      trainer=excluded.trainer,
      jockey=excluded.jockey,
      tj_wins=excluded.tj_wins,
      tj_starts=excluded.tj_starts,
      tj_pct=excluded.tj_pct
    """, (row.get("date"), row.get("track"), row.get("race"), row.get("program"),
          row.get("horse"), row.get("trainer"), row.get("jockey"),
          row.get("tj_wins"), row.get("tj_starts"), row.get("tj_pct")))
    conn.commit()

def upsert_combo(conn, row):
    """
    simple store of combos for later consumption
    row: date, track, race, program, combo (str like "1-2-3"), value (float)
    """
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO combos (date, track, race, program, combo, value)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(date, track, race, program, combo) DO UPDATE SET
      value=excluded.value
    """, (row.get("date"), row.get("track"), row.get("race"), row.get("program"), row.get("combo"), row.get("value")))
    conn.commit()

# ---------------- Parsing helpers ----------------
def _to_int(v, default=None):
    try:
        if v is None or v == "": return default
        return int(float(v))
    except Exception:
        return default

def _to_float(v, default=None):
    try:
        if v is None or v == "": return default
        return float(v)
    except Exception:
        return default

def _pct_from_any(v):
    """
    Accepts '18', '18%', 0.18, 0.180 -> returns pct as float in 0..100
    """
    if v is None: return None
    try:
        if isinstance(v, (int, float)):
            if 0.0 <= v <= 1.0:
                return v * 100.0
            return float(v)
        s = str(v).strip()
        if s.endswith("%"):
            s = s[:-1].strip()
        f = float(s)
        if 0.0 <= f <= 1.0:
            return f * 100.0
        return f
    except Exception:
        return None

def detect_race_id_from_raceobj(race_obj):
    """
    resilient extraction of vendor race id; many API variants exist.
    """
    for k in ("race_id", "id", "raceId", "uuid", "eventId", "raceKey", "race_key", "race_uid", "raceUid"):
        v = race_obj.get(k)
        if v:
            return str(v)
    # nested
    for subk in ("race", "event"):
        sub = race_obj.get(subk)
        if isinstance(sub, dict):
            for k in ("race_id","id","raceId","uuid"):
                v = sub.get(k)
                if v:
                    return str(v)
    return None

# ---------------- API wrappers ----------------
EP_MEETS = "/v1/north-america/meets"
EP_ENTRIES_BY_MEET = "/v1/north-america/meets/{meet_id}/entries"
EP_RESULTS_BY_RACE = "/v1/north-america/races/{race_id}/results"
EP_RACES_SEARCH = "/v1/north-america/races/search"  # sometimes helpful

def fetch_meets(iso_date):
    return safe_get(EP_MEETS, {"start_date": iso_date, "end_date": iso_date}, default={"meets": []})

def fetch_entries(meet_id):
    path = EP_ENTRIES_BY_MEET.format(meet_id=meet_id)
    return safe_get(path, default={})

def fetch_results(race_id):
    path = EP_RESULTS_BY_RACE.format(race_id=race_id)
    return safe_get(path, default={})

# ---------------- Core ingestion logic ----------------
def process_date(conn, iso_date, limit_tracks=None):
    """
    Build cards, then extract TJ info per runner and combos (if any).
    Returns summary dict.
    """
    meets_resp = fetch_meets(iso_date) or {}
    meets = meets_resp.get("meets") or []
    total_races = 0
    total_upserts = 0
    total_combos = 0

    for m in meets:
        track = m.get("track_name") or m.get("track") or m.get("name") or "Unknown"
        if limit_tracks and track not in limit_tracks:
            continue
        meet_id = m.get("meet_id") or m.get("id") or m.get("meetId")
        if not meet_id:
            log(f"skip meet missing meet_id for track={track}")
            continue
        entries = fetch_entries(meet_id) or {}
        races = entries.get("races") or entries.get("entries") or []
        if not races:
            # maybe API returns races under another key
            log(f"no races for meet {meet_id} / {track}")
            continue
        for rc in races:
            total_races += 1
            # canonical fields
            rno = rc.get("race_number") or rc.get("raceNo") or rc.get("race") or None
            try:
                rno_int = int(str(rno)) if rno is not None else None
            except Exception:
                rno_int = None

            runners = rc.get("runners") or rc.get("entries") or rc.get("horses") or []
            if not runners:
                # sometimes we need to fetch results or entries differently
                # attempt to discover vendor race_id and fetch entries via search or results endpoint
                vendor_race_id = detect_race_id_from_raceobj(rc)
                if vendor_race_id:
                    # try results fetch — many APIs return entries inside results as well
                    results = fetch_results(vendor_race_id) or {}
                    # results may contain "runners" or "results"
                    candidates = results.get("runners") or results.get("entries") or results.get("runners") or []
                    if candidates:
                        runners = candidates
                if not runners:
                    log(f"skip: cannot discover runners track={track} rn={rno_int} meet_id={meet_id}")
                    continue

            # For each runner try to extract tj info
            for r in runners:
                program = r.get("program") or r.get("pp") or r.get("number") or r.get("saddle") or r.get("horse_number") or ""
                horse = r.get("horse_name") or r.get("horse") or r.get("name") or ""
                trainer = r.get("trainer") or r.get("trainer_name") or r.get("trainerName") or ""
                jockey = r.get("jockey") or r.get("jockey_name") or r.get("jockeyName") or r.get("rider") or ""
                # TJ fields: look for various aliases
                tj_pct = None
                tj_wins = None
                tj_starts = None

                # try pct direct
                for k in ("tj_win", "combo_win", "trainer_jockey_pct", "trainerJockeyPct", "tj_pct"):
                    if k in r:
                        tj_pct = _pct_from_any(r.get(k))
                        break

                # try wins/starts keys
                for k in ("tj_wins","tjWins","combo_wins","trainer_jockey_wins","trainerJockeyWins","W"):
                    if k in r:
                        tj_wins = _to_int(r.get(k), None)
                        break

                for k in ("tj_starts","tjStarts","combo_starts","trainer_jockey_starts","trainerJockeyStarts","S"):
                    if k in r:
                        tj_starts = _to_int(r.get(k), None)
                        break

                # If pct missing but wins & starts present -> compute pct
                if tj_pct is None and tj_wins is not None and tj_starts:
                    try:
                        tj_pct = float(tj_wins) * 100.0 / float(tj_starts)
                    except Exception:
                        tj_pct = None

                # If pct exists but wins/starts missing, synthesize counts baseline 100 starts
                if tj_pct is not None:
                    if tj_starts is None:
                        tj_starts = 100
                    if tj_wins is None:
                        tj_wins = int(round(tj_pct))

                # guard bounds
                if tj_pct is not None:
                    if tj_pct < 0: tj_pct = 0.0
                    if tj_pct > 100: tj_pct = min(tj_pct, 100.0)

                row = {
                    "date": iso_date,
                    "track": track,
                    "race": rno_int,
                    "program": str(program) if program is not None else "",
                    "horse": horse,
                    "trainer": trainer,
                    "jockey": jockey,
                    "tj_wins": tj_wins,
                    "tj_starts": tj_starts,
                    "tj_pct": tj_pct
                }
                upsert_tj(conn, row)
                total_upserts += 1

            # optional: parse combos (if present in rc) and store them
            # combos schema varies; we do a lightweight extraction if fields exist
            combos_list = rc.get("combos") or rc.get("exactas") or rc.get("exotic_combos") or []
            if combos_list and isinstance(combos_list, list):
                for combo in combos_list:
                    # try to find program and combo string/value
                    prog = combo.get("program") or combo.get("a") or ""
                    combo_key = combo.get("combo") or combo.get("b") or combo.get("pair") or combo.get("legs") or None
                    value = _to_float(combo.get("value") or combo.get("payout") or combo.get("price"), None)
                    if combo_key:
                        upsert_combo(conn, {
                            "date": iso_date,
                            "track": track,
                            "race": rno_int,
                            "program": str(prog),
                            "combo": str(combo_key),
                            "value": value
                        })
                        total_combos += 1

    return {"date": iso_date, "races": total_races, "upserts": total_upserts, "combos": total_combos}

# ---------------- CLI and orchestration ----------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="ingest_tj.py - ingest trainer/jockey combo data into local sqlite")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--date", "-d", help="Date YYYY-MM-DD to ingest", type=str)
    g.add_argument("--days", "-D", help="Backfill N days ending yesterday", type=int)
    p.add_argument("--limit-track", help="Comma-separated list of tracks to restrict", type=str, default=None)
    return p.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    ensure_db(conn)
    if args.limit_track:
        limit_tracks = set([t.strip() for t in args.limit_track.split(",") if t.strip()])
    else:
        limit_tracks = None

    summaries = []
    if args.days:
        # backfill last N days up to yesterday
        for i in range(args.days):
            day = (date.today() - timedelta(days=(i+1))).isoformat()
            log(f"date {day}")
            try:
                s = process_date(conn, day, limit_tracks=limit_tracks)
                log(f"[{day}] races={s['races']} upserts={s['upserts']} combos_upserted={s['combos']}")
                summaries.append(s)
            except Exception as e:
                log(f"ERROR processing {day}: {e}")
                log(traceback.format_exc())
    else:
        day = args.date or date.today().isoformat()
        log(f"date {day}")
        try:
            s = process_date(conn, day, limit_tracks=limit_tracks)
            log(f"[{day}] races={s['races']} upserts={s['upserts']} combos_upserted={s['combos']}")
            summaries.append(s)
        except Exception as e:
            log(f"ERROR processing {day}: {e}")
            log(traceback.format_exc())

    # Totals
    total_r = sum(s.get("races",0) for s in summaries)
    total_u = sum(s.get("upserts",0) for s in summaries)
    total_c = sum(s.get("combos",0) for s in summaries)
    log(f"TOTAL races={total_r} upserts={total_u} combos_upserted={total_c}")
    conn.close()

if __name__ == "__main__":
    main(sys.argv[1:])