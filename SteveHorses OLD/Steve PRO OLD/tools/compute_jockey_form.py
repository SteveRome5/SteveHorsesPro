#!/usr/bin/env python3
"""
Compute jockey/trainer form rollups from data/tj.sqlite (tj_daily).

Creates/refreshes:
  - tj_rollup_combo(window_days, bucket, trainer_norm, jockey_norm, starts, wins, win_pct)
  - tj_rollup_jockey(window_days, bucket, jockey_norm, starts, wins, win_pct)

Usage:
  python3 tools/compute_jockey_form.py                # default 365 days
  python3 tools/compute_jockey_form.py --days 120     # custom window
"""
import argparse
import sqlite3
from pathlib import Path
from datetime import date, timedelta

APP_DIR = Path(__file__).resolve().parents[1]
DB_PATH = APP_DIR / "data" / "tj.sqlite"

def iso_days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()

def ensure_schema(con: sqlite3.Connection):
    con.executescript("""
    PRAGMA journal_mode=WAL;

    CREATE TABLE IF NOT EXISTS tj_daily (
      dt TEXT NOT NULL,
      bucket TEXT NOT NULL,
      trainer_norm TEXT NOT NULL,
      jockey_norm TEXT NOT NULL,
      starts INTEGER NOT NULL,
      wins INTEGER NOT NULL,
      PRIMARY KEY (dt, bucket, trainer_norm, jockey_norm)
    );

    CREATE TABLE IF NOT EXISTS tj_rollup_combo (
      window_days INTEGER NOT NULL,
      bucket TEXT NOT NULL,
      trainer_norm TEXT NOT NULL,
      jockey_norm TEXT NOT NULL,
      starts INTEGER NOT NULL,
      wins INTEGER NOT NULL,
      win_pct REAL NOT NULL,
      PRIMARY KEY (window_days, bucket, trainer_norm, jockey_norm)
    );

    CREATE TABLE IF NOT EXISTS tj_rollup_jockey (
      window_days INTEGER NOT NULL,
      bucket TEXT NOT NULL,
      jockey_norm TEXT NOT NULL,
      starts INTEGER NOT NULL,
      wins INTEGER NOT NULL,
      win_pct REAL NOT NULL,
      PRIMARY KEY (window_days, bucket, jockey_norm)
    );

    CREATE INDEX IF NOT EXISTS idx_tj_daily_bucket ON tj_daily(bucket, dt);
    CREATE INDEX IF NOT EXISTS idx_tj_daily_people ON tj_daily(trainer_norm, jockey_norm, dt);
    """)

def build_rollups(con: sqlite3.Connection, days: int):
    cutoff = iso_days_ago(days)
    # Freshen snapshots for this window
    with con:
        con.execute("DELETE FROM tj_rollup_combo WHERE window_days = ?", (days,))
        con.execute("DELETE FROM tj_rollup_jockey WHERE window_days = ?", (days,))

        # Combo (trainer + jockey) per bucket
        con.execute(f"""
        INSERT INTO tj_rollup_combo(window_days, bucket, trainer_norm, jockey_norm, starts, wins, win_pct)
        SELECT
            ? as window_days,
            bucket,
            trainer_norm,
            jockey_norm,
            SUM(starts) as starts,
            SUM(wins)   as wins,
            CASE WHEN SUM(starts) > 0 THEN ROUND(100.0 * SUM(wins) * 1.0 / SUM(starts), 1) ELSE 0.0 END as win_pct
        FROM tj_daily
        WHERE dt >= ?
        GROUP BY bucket, trainer_norm, jockey_norm
        """, (days, cutoff))

        # Jockey-only per bucket (aggregating over all trainers)
        con.execute(f"""
        INSERT INTO tj_rollup_jockey(window_days, bucket, jockey_norm, starts, wins, win_pct)
        SELECT
            ? as window_days,
            bucket,
            jockey_norm,
            SUM(starts) as starts,
            SUM(wins)   as wins,
            CASE WHEN SUM(starts) > 0 THEN ROUND(100.0 * SUM(wins) * 1.0 / SUM(starts), 1) ELSE 0.0 END as win_pct
        FROM tj_daily
        WHERE dt >= ?
        GROUP BY bucket, jockey_norm
        """, (days, cutoff))

    c1 = con.execute("SELECT COUNT(*) FROM tj_rollup_combo  WHERE window_days = ?", (days,)).fetchone()[0]
    c2 = con.execute("SELECT COUNT(*) FROM tj_rollup_jockey WHERE window_days = ?", (days,)).fetchone()[0]
    # A few sample lines to make the log useful
    sample = con.execute("""
        SELECT bucket, trainer_norm, jockey_norm, starts, wins, win_pct
        FROM tj_rollup_combo
        WHERE window_days = ?
        ORDER BY starts DESC
        LIMIT 5
    """, (days,)).fetchall()

    print(f"[jockey-form] db={DB_PATH}")
    print(f"[jockey-form] window_days={days}  cutoff>={cutoff}")
    print(f"[jockey-form] rows combo={c1}  jockey={c2}")
    for b,tr,jk,st,wn,p in sample:
        print(f"  • {b} | {tr} / {jk}  {wn}/{st} = {p}%")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365, help="rolling window size (days)")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"[jockey-form] ERROR: {DB_PATH} not found. Run tools/ingest_tj.py first.")
        return 2

    con = sqlite3.connect(str(DB_PATH))
    try:
        ensure_schema(con)
        build_rollups(con, args.days)
    finally:
        con.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())