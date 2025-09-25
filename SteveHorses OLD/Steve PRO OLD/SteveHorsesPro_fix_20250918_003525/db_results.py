# db_results.py
# Results store (SQLite WAL) + simple bias index by (track|surf|bucket|rail_bin).
from __future__ import annotations

import os, sqlite3, statistics
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

# ---------- Paths ----------
DATA_DIR = Path(os.getenv("STEVE_DATA_DIR", str((Path.home() / "Desktop" / "SteveHorsesPro" / "data"))))
DB_DIR   = DATA_DIR / "results"
DB_PATH  = DB_DIR / "results.sqlite3"
DB_DIR.mkdir(parents=True, exist_ok=True)

def _connect() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")
    return c

def init() -> None:
    with _connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS races (
            rid TEXT PRIMARY KEY,
            race_date TEXT,
            track TEXT,
            race_num TEXT,
            surface TEXT,
            distance_yards INTEGER,
            rail REAL,
            cond TEXT,
            takeout_win REAL
        );
        CREATE TABLE IF NOT EXISTS results (
            rid TEXT NOT NULL,
            program TEXT NOT NULL,
            position INTEGER,             -- 1=win, 2=place, 3=show...
            odds_dec REAL,
            win_paid REAL,
            place_paid REAL,
            show_paid REAL,
            beaten_lengths REAL,
            speed REAL,
            class_ REAL,
            PRIMARY KEY (rid, program),
            FOREIGN KEY (rid) REFERENCES races(rid)
        );
        CREATE INDEX IF NOT EXISTS idx_results_rid_pos ON results(rid, position);

        CREATE TABLE IF NOT EXISTS exacta (
            rid TEXT PRIMARY KEY,
            a_program TEXT,
            b_program TEXT,
            payout REAL
        );

        -- Aggregated bias hints (very light, can be rebuilt any time)
        CREATE TABLE IF NOT EXISTS bias_index (
            key TEXT PRIMARY KEY,                 -- track|surf|bucket|railbin
            last_update TEXT,
            count INTEGER,
            inside_bias REAL,                     -- [-1..+1], + favors inside
            speed_bias REAL,                      -- [-1..+1], + favors speed/forward
            rail_effect REAL                      -- [-0.5..+0.5], + rail helps
        );
        """)
        c.commit()

# ---------- Helpers ----------
def _bucket_yards(y: Optional[int]) -> str:
    if not y: return "unk"
    if y < 1320:  return "<6f"
    if y < 1540:  return "6f"
    if y < 1760:  return "7f"
    if y < 1980:  return "1mi"
    if y < 2200:  return "8.5f"
    if y < 2420:  return "9f"
    return "10f+"

def _rail_bin(rail: Optional[float]) -> str:
    try:
        r = float(rail or 0.0)
    except:
        r = 0.0
    if r < 10:  return "0-9"
    if r < 20:  return "10-19"
    if r < 30:  return "20-29"
    return "30+"

def _key(track: str, surface: str, yards: Optional[int], rail: Optional[float]) -> str:
    return f"{(track or '').strip()}|{(surface or '').strip()}|{_bucket_yards(yards)}|{_rail_bin(rail)}"

# ---------- Writers ----------
def upsert_race_meta(
    rid: str,
    *,
    race_date: Optional[str],
    track: Optional[str],
    race_num: Optional[str],
    surface: Optional[str],
    distance_yards: Optional[int],
    rail: Optional[float],
    cond: Optional[str],
    takeout_win: Optional[float]
) -> None:
    init()
    with _connect() as c:
        c.execute("""
            INSERT INTO races (rid, race_date, track, race_num, surface, distance_yards, rail, cond, takeout_win)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(rid) DO UPDATE SET
                race_date=excluded.race_date, track=excluded.track, race_num=excluded.race_num,
                surface=excluded.surface, distance_yards=excluded.distance_yards, rail=excluded.rail,
                cond=excluded.cond, takeout_win=excluded.takeout_win
        """, (rid, race_date, track, race_num, surface, distance_yards, rail, cond, takeout_win))
        c.commit()

def upsert_results(rid: str, rows: List[Dict[str, Any]]) -> None:
    """
    rows: [{program, position, odds_dec, win_paid, place_paid, show_paid, beaten_lengths, speed, class_}, ...]
    """
    if not rows: return
    init()
    with _connect() as c:
        for r in rows:
            c.execute("""
                INSERT INTO results (rid, program, position, odds_dec, win_paid, place_paid, show_paid, beaten_lengths, speed, class_)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(rid,program) DO UPDATE SET
                    position=excluded.position, odds_dec=excluded.odds_dec,
                    win_paid=excluded.win_paid, place_paid=excluded.place_paid, show_paid=excluded.show_paid,
                    beaten_lengths=excluded.beaten_lengths, speed=excluded.speed, class_=excluded.class_
            """, (
                rid, str(r.get("program","")), r.get("position"),
                r.get("odds_dec"), r.get("win_paid"), r.get("place_paid"), r.get("show_paid"),
                r.get("beaten_lengths"), r.get("speed"), r.get("class_")
            ))
        c.commit()

def upsert_exacta(rid: str, a_program: str, b_program: str, payout: Optional[float]) -> None:
    init()
    with _connect() as c:
        c.execute("""
            INSERT INTO exacta (rid, a_program, b_program, payout)
            VALUES (?,?,?,?)
            ON CONFLICT(rid) DO UPDATE SET a_program=excluded.a_program, b_program=excluded.b_program, payout=excluded.payout
        """, (rid, str(a_program), str(b_program), payout))
        c.commit()

# ---------- Bias maintenance (simple & explainable) ----------
def update_bias_from_race(rid: str) -> None:
    """
    Extremely lightweight bias update:
      - inside_bias: +1 if winner program <= 3, -1 if winner program >= 10, else 0 (scaled by field size)
      - speed_bias:  if we have 'speed' figs: +1 if winner speed >= 90th pct of field, -1 if <= 10th pct (else 0)
      - rail_effect: +0.05 if rail bin >= 20 (turf rails wide), else 0
    Aggregated as moving average with counts.
    """
    init()
    with _connect() as c:
        meta = c.execute("SELECT track, surface, distance_yards, rail FROM races WHERE rid=?", (rid,)).fetchone()
        if not meta:
            return
        track, surface, yards, rail = meta
        key = _key(track or "", surface or "", yards, rail)

        rows = c.execute("SELECT program, position, speed FROM results WHERE rid=?", (rid,)).fetchall()
        if not rows:
            return
        # winner:
        winner_prog = None
        speeds: List[float] = []
        for prog, pos, sp in rows:
            try:
                if pos == 1:
                    winner_prog = int(re.sub(r"\\D","", str(prog) or "0") or "0")
            except:
                try:
                    winner_prog = int(str(prog))
                except:
                    winner_prog = None
            try:
                if sp is not None:
                    speeds.append(float(sp))
            except: pass

        inside = 0.0
        if winner_prog is not None:
            if winner_prog <= 3: inside = +1.0
            elif winner_prog >= 10: inside = -1.0
        speed_b = 0.0
        if speeds:
            try:
                speeds_sorted = sorted(speeds)
                p10 = speeds_sorted[max(0, int(0.10*(len(speeds_sorted)-1)))]
                p90 = speeds_sorted[int(0.90*(len(speeds_sorted)-1))]
                # winner speed value:
                w_sp = c.execute("SELECT speed FROM results WHERE rid=? AND position=1", (rid,)).fetchone()
                w_sp = float(w_sp[0]) if (w_sp and w_sp[0] is not None) else None
                if w_sp is not None:
                    if w_sp >= p90: speed_b = +1.0
                    elif w_sp <= p10: speed_b = -1.0
            except:
                pass

        rail_eff = 0.05 if (rail is not None and float(rail) >= 20.0 and "turf" in (surface or "").lower()) else 0.0

        prev = c.execute("SELECT count, inside_bias, speed_bias, rail_effect FROM bias_index WHERE key=?", (key,)).fetchone()
        if prev:
            cnt, ib, sb, reff = prev
            cnt2 = int(cnt or 0) + 1
            # simple running average (clamped)
            ib2 = max(-1.0, min(1.0, ((ib or 0.0)*cnt + inside) / cnt2))
            sb2 = max(-1.0, min(1.0, ((sb or 0.0)*cnt + speed_b) / cnt2))
            rf2 = max(-0.5, min(0.5, ((reff or 0.0)*cnt + rail_eff) / cnt2))
            c.execute("UPDATE bias_index SET last_update=?, count=?, inside_bias=?, speed_bias=?, rail_effect=? WHERE key=?",
                      (datetime.utcnow().isoformat(timespec="seconds"), cnt2, ib2, sb2, rf2, key))
        else:
            c.execute("INSERT INTO bias_index(key,last_update,count,inside_bias,speed_bias,rail_effect) VALUES (?,?,?,?,?,?)",
                      (key, datetime.utcnow().isoformat(timespec="seconds"), 1, inside, speed_b, rail_eff))
        c.commit()

def rebuild_bias() -> int:
    """Recompute bias table from all stored races. Returns #keys updated."""
    init()
    with _connect() as c:
        c.execute("DELETE FROM bias_index")
        c.commit()
    keys = set()
    with _connect() as c:
        cur = c.execute("SELECT rid FROM races")
        for (rid,) in cur.fetchall():
            update_bias_from_race(str(rid))
        cur2 = c.execute("SELECT COUNT(*) FROM bias_index")
        return int(cur2.fetchone()[0])

# ---------- Readers ----------
def get_result(rid: str) -> Dict[str, Any]:
    """Return metadata + ordered results for a race."""
    init()
    out: Dict[str, Any] = {"meta": {}, "results": []}
    with _connect() as c:
        m = c.execute("SELECT race_date,track,race_num,surface,distance_yards,rail,cond,takeout_win FROM races WHERE rid=?", (rid,)).fetchone()
        if m:
            out["meta"] = {
                "race_date": m[0], "track": m[1], "race_num": m[2],
                "surface": m[3], "distance_yards": m[4], "rail": m[5],
                "cond": m[6], "takeout_win": m[7]
            }
        for row in c.execute("SELECT program,position,odds_dec,win_paid,place_paid,show_paid,beaten_lengths,speed,class_ "
                             "FROM results WHERE rid=? ORDER BY position ASC NULLS LAST, program ASC", (rid,)).fetchall():
            out["results"].append({
                "program": row[0], "position": row[1], "odds_dec": row[2],
                "win_paid": row[3], "place_paid": row[4], "show_paid": row[5],
                "beaten_lengths": row[6], "speed": row[7], "class_": row[8]
            })
    return out

def get_bias_hint(track: str, surface: str, distance_yards: Optional[int], rail: Optional[float]) -> Dict[str, float]:
    """
    Returns a tiny bias hint (all in [-1..+1] except rail_effect [-0.5..+0.5]).
    Missing key => zeros.
    """
    init()
    k = _key(track or "", surface or "", distance_yards, rail)
    with _connect() as c:
        row = c.execute("SELECT inside_bias, speed_bias, rail_effect, count FROM bias_index WHERE key=?", (k,)).fetchone()
        if not row:
            return {"inside_bias": 0.0, "speed_bias": 0.0, "rail_effect": 0.0, "n": 0.0}
        ib, sb, rf, cnt = row
        return {
            "inside_bias": float(ib or 0.0),
            "speed_bias":  float(sb or 0.0),
            "rail_effect": float(rf or 0.0),
            "n":           float(cnt or 0.0)
        }