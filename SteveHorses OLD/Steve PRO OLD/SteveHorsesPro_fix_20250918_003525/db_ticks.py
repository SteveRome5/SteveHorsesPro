# db_ticks.py
# Lightweight market ticks store (SQLite WAL) with writer + reader utilities.
from __future__ import annotations

import os, sqlite3, math, statistics, re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterable, List, Dict, Any, Tuple

# ---------- Paths ----------
DATA_DIR = Path(os.getenv("STEVE_DATA_DIR", str((Path.home() / "Desktop" / "SteveHorsesPro" / "data"))))
DB_DIR   = DATA_DIR / "ticks"
DB_PATH  = DB_DIR / "ticks.sqlite3"
DB_DIR.mkdir(parents=True, exist_ok=True)

def _utc_now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# ---------- Connect / init ----------
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn

def init() -> None:
    with _connect() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS ticks (
            rid         TEXT NOT NULL,             -- vendor race id (or synthetic)
            race_date   TEXT,                      -- YYYY-MM-DD (optional but recommended)
            track       TEXT,                      -- human track name
            race_num    TEXT,                      -- '1','2',.. (string to match PRO)
            program     TEXT NOT NULL,             -- saddle / program number
            ts_utc      TEXT NOT NULL,             -- ISO 8601 UTC
            source      TEXT,                      -- 'live','wp','ml','hist'
            dec         REAL,                      -- decimal odds ( >1.0 )
            mtp         REAL,                      -- minutes to post (approx if known)
            win_pool    REAL,                      -- optional pool
            PRIMARY KEY (rid, program, ts_utc, source)
        );
        CREATE INDEX IF NOT EXISTS idx_ticks_rid_prog_ts ON ticks(rid, program, ts_utc);
        CREATE INDEX IF NOT EXISTS idx_ticks_rid_ts      ON ticks(rid, ts_utc);
        CREATE TABLE IF NOT EXISTS last_snapshot (
            rid       TEXT PRIMARY KEY,
            last_ts_utc TEXT,
            n_rows    INTEGER
        );
        """)
        c.commit()

# ---------- Sanitize ----------
def _to_dec(v: Any) -> Optional[float]:
    if v in (None, "", "None"): return None
    try:
        if isinstance(v,(int,float)):
            dv=float(v)
        else:
            s=str(v).strip().lower()
            m=re.fullmatch(r"(\d+)\s*[/\-:]\s*(\d+)", s)
            if m:
                num=float(m.group(1)); den=float(m.group(2)); 
                if den>0: dv=1.0+num/den
                else: return None
            elif s in ("evs","even","evens"):
                dv=2.0
            else:
                dv=float(s)
        if not (dv>1.0): return None
        if dv>200.0: return None
        return dv
    except:
        return None

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v in (None,"","None"): return None
        return float(v)
    except:
        return None

# ---------- Writers ----------
def add_tick(
    rid: str,
    program: str,
    dec: Any,
    *,
    ts_utc: Optional[str]=None,
    source: str="live",
    mtp: Optional[Any]=None,
    track: Optional[str]=None,
    race_num: Optional[str]=None,
    race_date: Optional[str]=None,
    win_pool: Optional[Any]=None
) -> bool:
    """Insert a single tick. Returns True on write."""
    init()
    d = _to_dec(dec)
    if d is None: 
        return False
    ts = ts_utc or _utc_now_iso()
    with _connect() as c:
        c.execute(
            "INSERT OR IGNORE INTO ticks (rid,race_date,track,race_num,program,ts_utc,source,dec,mtp,win_pool)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (str(rid).strip(), str(race_date or ""), str(track or ""), str(race_num or ""),
             str(program).strip(), ts, str(source or ""), d, _safe_float(mtp), _safe_float(win_pool))
        )
        cur = c.execute("SELECT COUNT(*) FROM ticks WHERE rid=?", (rid,))
        n = int(cur.fetchone()[0])
        c.execute("INSERT INTO last_snapshot(rid,last_ts_utc,n_rows) VALUES (?,?,?)"
                  " ON CONFLICT(rid) DO UPDATE SET last_ts_utc=excluded.last_ts_utc, n_rows=excluded.n_rows",
                  (rid, ts, n))
        c.commit()
    return True

def add_many_ticks(
    rid: str,
    rows: Iterable[Dict[str, Any]],
    *,
    track: Optional[str]=None,
    race_num: Optional[str]=None,
    race_date: Optional[str]=None
) -> int:
    """
    rows items may contain: program, dec, ts_utc, source, mtp, win_pool
    """
    init()
    count=0
    with _connect() as c:
        for r in rows:
            d = _to_dec(r.get("dec"))
            if d is None: 
                continue
            ts = r.get("ts_utc") or _utc_now_iso()
            c.execute(
                "INSERT OR IGNORE INTO ticks (rid,race_date,track,race_num,program,ts_utc,source,dec,mtp,win_pool)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (str(rid).strip(), str(race_date or ""), str(track or ""), str(race_num or ""),
                 str(r.get("program","")).strip(), ts, str(r.get("source","") or ""), d,
                 _safe_float(r.get("mtp")), _safe_float(r.get("win_pool")))
            )
            count+=1
        cur = c.execute("SELECT MAX(ts_utc), COUNT(*) FROM ticks WHERE rid=?", (rid,))
        last_ts, n = cur.fetchone()
        c.execute("INSERT INTO last_snapshot(rid,last_ts_utc,n_rows) VALUES (?,?,?)"
                  " ON CONFLICT(rid) DO UPDATE SET last_ts_utc=excluded.last_ts_utc, n_rows=excluded.n_rows",
                  (rid, last_ts, n or 0))
        c.commit()
    return count

def trim_keep_days(keep_days: int = 120) -> int:
    """Delete very old rows; returns deleted count."""
    if keep_days <= 0: 
        return 0
    init()
    cutoff = (datetime.utcnow() - timedelta(days=keep_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _connect() as c:
        cur = c.execute("SELECT COUNT(*) FROM ticks WHERE ts_utc < ?", (cutoff,))
        before = int(cur.fetchone()[0])
        c.execute("DELETE FROM ticks WHERE ts_utc < ?", (cutoff,))
        c.commit()
        return before

# ---------- Readers (PRO-safe) ----------
def get_latest_snapshot(rid: str) -> Dict[str, float]:
    """Return {program: last_decimal_odds} for a race id."""
    init()
    out: Dict[str, float] = {}
    with _connect() as c:
        # last per (rid, program) by ts
        cur = c.execute("""
            SELECT t1.program, t1.dec FROM ticks t1
            JOIN (
                SELECT program, MAX(ts_utc) AS ts
                FROM ticks WHERE rid=? GROUP BY program
            ) t2 ON t1.program=t2.program AND t1.ts_utc=t2.ts
            WHERE t1.rid=?
        """, (rid, rid))
        for prog, dec in cur.fetchall():
            try:
                out[str(prog)] = float(dec)
            except:
                pass
    return out

def get_series(rid: str, program: str, since_minutes: Optional[int] = None) -> List[Tuple[str, float]]:
    """Return [(ts_iso, dec), ...] time series for a single horse."""
    init()
    query = "SELECT ts_utc, dec FROM ticks WHERE rid=? AND program=?"
    args: List[Any] = [rid, program]
    if since_minutes and since_minutes > 0:
        cutoff = (datetime.utcnow() - timedelta(minutes=since_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")
        query += " AND ts_utc >= ?"; args.append(cutoff)
    query += " ORDER BY ts_utc ASC"
    out: List[Tuple[str, float]] = []
    with _connect() as c:
        for ts, dec in c.execute(query, tuple(args)).fetchall():
            try:
                out.append((str(ts), float(dec)))
            except:
                pass
    return out

def _slope_and_var(seq: List[float]) -> Tuple[float, float]:
    """
    Matches your PRO semantics:
      slope10 ≈ clamp((a - c) / max(2.0, a), [-1, +1]) on the last 3 points
      var     = pvariance on last 5 points
    """
    last = seq[-1] if seq else None
    slope = 0.0; var = 0.0
    if len(seq) >= 3:
        a, b, c = seq[-3], seq[-2], seq[-1]
        try:
            slope = max(-1.0, min(1.0, (a - c) / max(2.0, a)))
        except:
            slope = 0.0
    if len(seq) >= 5:
        try:
            var = statistics.pvariance(seq[-5:])
        except:
            var = 0.0
    return slope, var

def get_volatility_features(rid: str) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Returns { program: {'last': dec or None, 'slope10': float, 'var': float}, ... }
    """
    init()
    out: Dict[str, Dict[str, Optional[float]]] = {}
    with _connect() as c:
        # fetch all in order per program
        cur = c.execute("SELECT program, ts_utc, dec FROM ticks WHERE rid=? ORDER BY program, ts_utc ASC", (rid,))
        seq_prog: str = ""
        vals: List[float] = []
        last_val: Optional[float] = None
        def _flush():
            if not seq_prog: return
            slope, var = _slope_and_var(vals)
            out[seq_prog] = {"last": last_val, "slope10": slope, "var": var}
        for prog, ts, dec in cur.fetchall():
            prog = str(prog)
            if prog != seq_prog and seq_prog:
                _flush()
                vals = []
            seq_prog = prog
            try:
                dv = float(dec)
            except:
                continue
            vals.append(dv); last_val = dv
        if seq_prog:
            _flush()
    return out