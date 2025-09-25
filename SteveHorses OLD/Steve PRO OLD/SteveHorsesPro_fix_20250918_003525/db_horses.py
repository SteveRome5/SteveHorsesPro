#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lightweight horse-history sidecar DB for SteveHorsesPro
- Pure stdlib (sqlite3)
- Safe to import from TRAIN; PRO can ignore it entirely
- Exposes:
    ensure_schema()
    upsert_horse(name, yob=None, country=None, sex=None) -> horse_key
    insert_run(horse_key, race_date, track, race_no, program, surface, dist_y,
               ml_dec, live_dec, speed, class_, ep, lp, equipment, result_pos=None)
    get_recent_runs(horse_key, n=6) -> list[dict]
    record_runner(track, rno, rc, runner, race_date=None)  # convenience wrapper
"""

from __future__ import annotations

import sqlite3
import re
import os
import unicodedata
from pathlib import Path
from datetime import date
from typing import Optional, Dict, Any, List

# ---------- Paths ----------
HOME = Path.home()
BASE = HOME / "Desktop" / "SteveHorsesPro"
DATA_DIR = BASE / "data"
DB_PATH = DATA_DIR / "horses.db"

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
            return num/den if den!=0 else default
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
        if den>0: return 1.0 + num/den
    try:
        dec=float(s)
        return dec if dec>1.0 else default
    except: 
        return default

def prg_num(r): 
    return str(g(r,"program_number","program","number","pp","post_position","horse_number","saddle","saddle_number") or "")

def horse_name(r): 
    return g(r,"horse_name","name","runner_name","runner","horse","horseName") or "Unknown"

def get_surface(rc): 
    return str(g(rc,"surface","track_surface","course","courseType","trackSurface","surf") or "").lower()

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

def morning_line_decimal(r):
    v = g(r, "morning_line","ml","ml_odds","morningLine","morningLineOdds",
            "morning_line_decimal","program_ml","programMorningLine","mlDecimal")
    return _to_dec_odds(v, None)

def live_decimal(r):
    v = g(r,"live_odds","odds","currentOdds","current_odds","liveOdds","market",
            "price","decimal_odds","winOdds","oddsDecimal")
    return _to_dec_odds(v, None)

def get_speed(r): return _to_float(g(r,"speed","spd","last_speed","best_speed","fig","speed_fig","brz","beyer"), None)
def get_early_pace(r): return _to_float(g(r,"pace","ep","early_pace","earlyPace","quirin"), None)
def get_late_pace(r):  return _to_float(g(r,"lp","late_pace","closer","lateSpeed"), None)
def get_class(r):      return _to_float(g(r,"class","cls","class_rating","classRating","par_class","parClass"), None)

def _normalize_name(name: str) -> str:
    if not name: return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(the|a|an|of|and|&)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _horse_key(name: str, yob: Optional[int]=None, country: Optional[str]=None) -> str:
    base = _normalize_name(name)
    tail = []
    if yob: tail.append(str(int(yob)))
    if country: tail.append(country.strip().upper())
    return base + ("|" + "|".join(tail) if tail else "")

# ---------- Schema ----------
_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
CREATE TABLE IF NOT EXISTS horses (
    horse_key   TEXT PRIMARY KEY,
    name_raw    TEXT NOT NULL,
    name_key    TEXT NOT NULL,
    yob         INTEGER,
    country     TEXT,
    sex         TEXT,
    first_seen  TEXT,
    last_seen   TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    horse_key   TEXT NOT NULL,
    race_date   TEXT NOT NULL,
    track       TEXT NOT NULL,
    race_no     TEXT NOT NULL,
    program     TEXT,
    surface     TEXT,
    dist_y      INTEGER,
    ml_dec      REAL,
    live_dec    REAL,
    speed       REAL,
    class_      REAL,
    ep          REAL,
    lp          REAL,
    equipment   TEXT,
    result_pos  INTEGER,
    UNIQUE(horse_key, race_date, track, race_no, program)
);
CREATE INDEX IF NOT EXISTS idx_runs_horse_date ON runs(horse_key, race_date DESC);
"""

# ---------- Connection helpers ----------
def _conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cn = sqlite3.connect(DB_PATH)
    cn.execute("PRAGMA foreign_keys = ON;")
    return cn

def ensure_schema() -> None:
    cn = _conn()
    try:
        cn.executescript(_SCHEMA)
        cn.commit()
    finally:
        cn.close()

# ---------- Upserts & Inserts ----------
def upsert_horse(name: str, yob: Optional[int]=None, country: Optional[str]=None, sex: Optional[str]=None) -> str:
    if not name: name = "Unknown"
    key = _horse_key(name, yob, country)
    name_key = _normalize_name(name)
    today = date.today().isoformat()
    cn=_conn()
    try:
        cn.execute("""
            INSERT INTO horses(horse_key, name_raw, name_key, yob, country, sex, first_seen, last_seen)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(horse_key) DO UPDATE SET
                name_raw=excluded.name_raw,
                name_key=excluded.name_key,
                yob=COALESCE(horses.yob, excluded.yob),
                country=COALESCE(horses.country, excluded.country),
                sex=COALESCE(horses.sex, excluded.sex),
                last_seen=excluded.last_seen
        """, (key, name, name_key, yob, country, sex, today, today))
        cn.commit()
    finally:
        cn.close()
    return key

def insert_run(
    horse_key: str, race_date: str, track: str, race_no: str, program: Optional[str],
    surface: Optional[str], dist_y: Optional[int], ml_dec: Optional[float],
    live_dec: Optional[float], speed: Optional[float], class_: Optional[float],
    ep: Optional[float], lp: Optional[float], equipment: Optional[str],
    result_pos: Optional[int] = None
) -> None:
    cn=_conn()
    try:
        cn.execute("""
            INSERT OR IGNORE INTO runs(
                horse_key, race_date, track, race_no, program, surface, dist_y,
                ml_dec, live_dec, speed, class_, ep, lp, equipment, result_pos
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (horse_key, race_date, track, str(race_no), program, surface, dist_y,
              ml_dec, live_dec, speed, class_, ep, lp, equipment, result_pos))
        cn.commit()
    finally:
        cn.close()

# ---------- Queries ----------
def get_recent_runs(horse_key: str, n: int = 6) -> List[Dict[str, Any]]:
    cn=_conn()
    try:
        cur = cn.execute("""
            SELECT race_date, track, race_no, program, surface, dist_y,
                   ml_dec, live_dec, speed, class_, ep, lp, equipment, result_pos
            FROM runs
            WHERE horse_key = ?
            ORDER BY race_date DESC
            LIMIT ?
        """, (horse_key, int(n)))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        cn.close()

# ---------- Convenience: shove a vendor runner + race into DB ----------
def record_runner(track: str, rno: str | int, rc: dict, runner: dict, race_date: Optional[str]=None) -> None:
    # Fail-closed: never raise
    try:
        name = horse_name(runner)
        yob  = _to_float(g(runner, "yob","year_of_birth","foaled","yearBorn"), None)
        yob  = int(yob) if yob and yob>1900 else None
        country = g(runner, "country","birth_country","bred","bredIn","origin","countryCode")
        sex = g(runner, "sex","gender","sx")
        program = prg_num(runner) or None

        surface = get_surface(rc) or None
        dist_y  = get_distance_y(rc)
        ml_dec  = morning_line_decimal(runner)
        live_dec= live_decimal(runner)
        speed   = get_speed(runner)
        class_  = get_class(runner)
        ep      = get_early_pace(runner)
        lp      = get_late_pace(runner)
        equip   = g(runner, "equipment","equip","blinkers","lasix","medication")

        pos = g(runner, "finish_position","finish","pos","placing","official_finish")
        try: result_pos = int(str(pos)) if pos not in (None,"") else None
        except: result_pos = None

        if not race_date:
            rd = g(rc, "race_date","date","iso_date","start_time","postTime","post_time")
            race_date = str(rd)[:10] if rd else date.today().isoformat()

        horse_key = upsert_horse(name=name, yob=yob, country=country, sex=sex)
        insert_run(
            horse_key=horse_key, race_date=race_date, track=str(track), race_no=str(rno),
            program=program, surface=surface, dist_y=dist_y,
            ml_dec=ml_dec, live_dec=live_dec, speed=speed, class_=class_,
            ep=ep, lp=lp, equipment=str(equip) if equip is not None else None,
            result_pos=result_pos
        )
    except Exception:
        # swallow; TRAIN should never crash due to sidecar
        pass