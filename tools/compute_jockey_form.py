#!/usr/bin/env python3
# tools/compute_jockey_form.py
# Rebuilds tj_rollup_combo & tj_rollup_jockey from tj_daily when wins exist in the window,
# otherwise falls back to the historical "combos" table.
#
# Robust to older schemas that used trainer_norm/jockey_norm in rollup tables.

from __future__ import annotations
import argparse
import sqlite3
from datetime import date, timedelta
from pathlib import Path
import sys

def log(msg: str) -> None:
    print(f"[jockey-form] {msg}", flush=True)

def db_path_from_repo() -> Path:
    here = Path(__file__).resolve()
    base = here.parent.parent  # repo root
    return base / "data" / "tj.sqlite"

def get_table_cols(cur: sqlite3.Cursor, table: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table});")
    return {row[1] for row in cur.fetchall()}

def ensure_rollup_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Desired canonical schemas
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS tj_rollup_combo(
      trainer   TEXT NOT NULL,
      jockey    TEXT NOT NULL,
      bucket    TEXT NOT NULL,
      starts    INTEGER NOT NULL,
      wins      INTEGER NOT NULL,
      win_pct   REAL NOT NULL,
      updated_at TEXT,
      PRIMARY KEY(trainer,jockey,bucket)
    );
    CREATE TABLE IF NOT EXISTS tj_rollup_jockey(
      jockey    TEXT NOT NULL,
      bucket    TEXT NOT NULL,
      starts    INTEGER NOT NULL,
      wins      INTEGER NOT NULL,
      win_pct   REAL NOT NULL,
      updated_at TEXT,
      PRIMARY KEY(jockey,bucket)
    );
    """)
    conn.commit()

    # If legacy columns exist (trainer_norm/jockey_norm), migrate them to canonical names
    def migrate_combo_if_needed():
        cols = get_table_cols(cur, "tj_rollup_combo")
        if "trainer" in cols and "jockey" in cols:
            return
        # If legacy shape, rebuild table
        if "trainer_norm" in cols and "jockey_norm" in cols:
            log("migrating legacy tj_rollup_combo schema → (trainer,jockey)")
            cur.executescript("""
            BEGIN;
            CREATE TABLE tj_rollup_combo_new(
              trainer   TEXT NOT NULL,
              jockey    TEXT NOT NULL,
              bucket    TEXT NOT NULL,
              starts    INTEGER NOT NULL,
              wins      INTEGER NOT NULL,
              win_pct   REAL NOT NULL,
              updated_at TEXT,
              PRIMARY KEY(trainer,jockey,bucket)
            );
            INSERT OR REPLACE INTO tj_rollup_combo_new
              (trainer,jockey,bucket,starts,wins,win_pct,updated_at)
            SELECT
              trainer_norm AS trainer,
              jockey_norm  AS jockey,
              bucket,
              starts,
              wins,
              CASE
                WHEN COALESCE(win_pct, -1) >= 0 THEN win_pct
                WHEN starts>0 THEN 1.0*wins/starts
                ELSE 0.0
              END AS win_pct,
              updated_at
            FROM tj_rollup_combo;
            DROP TABLE tj_rollup_combo;
            ALTER TABLE tj_rollup_combo_new RENAME TO tj_rollup_combo;
            COMMIT;
            """)
            conn.commit()

    def migrate_jockey_if_needed():
        cols = get_table_cols(cur, "tj_rollup_jockey")
        if "jockey" in cols:
            return
        if "jockey_norm" in cols:
            log("migrating legacy tj_rollup_jockey schema → (jockey)")
            cur.executescript("""
            BEGIN;
            CREATE TABLE tj_rollup_jockey_new(
              jockey    TEXT NOT NULL,
              bucket    TEXT NOT NULL,
              starts    INTEGER NOT NULL,
              wins      INTEGER NOT NULL,
              win_pct   REAL NOT NULL,
              updated_at TEXT,
              PRIMARY KEY(jockey,bucket)
            );
            INSERT OR REPLACE INTO tj_rollup_jockey_new
              (jockey,bucket,starts,wins,win_pct,updated_at)
            SELECT
              jockey_norm AS jockey,
              bucket,
              starts,
              wins,
              CASE
                WHEN COALESCE(win_pct, -1) >= 0 THEN win_pct
                WHEN starts>0 THEN 1.0*wins/starts
                ELSE 0.0
              END AS win_pct,
              updated_at
            FROM tj_rollup_jockey;
            DROP TABLE tj_rollup_jockey;
            ALTER TABLE tj_rollup_jockey_new RENAME TO tj_rollup_jockey;
            COMMIT;
            """)
            conn.commit()

    migrate_combo_if_needed()
    migrate_jockey_if_needed()

def choose_source(conn: sqlite3.Connection, cutoff_iso: str) -> str:
    cur = conn.cursor()
    # If tj_daily has wins in the window, use it; otherwise fall back to combos
    cur.execute("SELECT COALESCE(SUM(wins),0) FROM tj_daily WHERE dt >= ?", (cutoff_iso,))
    total_wins = cur.fetchone()[0] or 0
    if total_wins > 0:
        return "tj_daily"
    return "combos"

def rebuild_from_tj_daily(conn: sqlite3.Connection, cutoff_iso: str) -> tuple[int,int]:
    cur = conn.cursor()
    cur.executescript("BEGIN; DELETE FROM tj_rollup_combo; DELETE FROM tj_rollup_jockey; COMMIT;")
    conn.commit()

    cur.execute("""
        INSERT OR REPLACE INTO tj_rollup_combo
        (trainer,jockey,bucket,starts,wins,win_pct,updated_at)
        SELECT
          trainer_norm AS trainer,
          jockey_norm  AS jockey,
          bucket,
          SUM(starts) AS starts,
          SUM(wins)   AS wins,
          CASE WHEN SUM(starts)>0 THEN 1.0*SUM(wins)/SUM(starts) ELSE 0.0 END AS win_pct,
          DATE('now')
        FROM tj_daily
        WHERE dt >= ?
        GROUP BY 1,2,3
    """, (cutoff_iso,))
    cur.execute("""
        INSERT OR REPLACE INTO tj_rollup_jockey
        (jockey,bucket,starts,wins,win_pct,updated_at)
        SELECT
          jockey_norm AS jockey,
          bucket,
          SUM(starts) AS starts,
          SUM(wins)   AS wins,
          CASE WHEN SUM(starts)>0 THEN 1.0*SUM(wins)/SUM(starts) ELSE 0.0 END AS win_pct,
          DATE('now')
        FROM tj_daily
        WHERE dt >= ?
        GROUP BY 1,2
    """, (cutoff_iso,))
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM tj_rollup_combo")
    n_combo = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM tj_rollup_jockey")
    n_jky = cur.fetchone()[0]
    return n_combo, n_jky

def rebuild_from_combos(conn: sqlite3.Connection) -> tuple[int,int]:
    cur = conn.cursor()
    cur.executescript("BEGIN; DELETE FROM tj_rollup_combo; DELETE FROM tj_rollup_jockey; COMMIT;")
    conn.commit()

    cur.execute("""
        INSERT OR REPLACE INTO tj_rollup_combo
        (trainer,jockey,bucket,starts,wins,win_pct,updated_at)
        SELECT
          trainer,
          jockey,
          bucket,
          SUM(starts) AS starts,
          SUM(wins)   AS wins,
          CASE WHEN SUM(starts)>0 THEN 1.0*SUM(wins)/SUM(starts) ELSE 0.0 END AS win_pct,
          DATE('now')
        FROM combos
        GROUP BY 1,2,3
    """)
    cur.execute("""
        INSERT OR REPLACE INTO tj_rollup_jockey
        (jockey,bucket,starts,wins,win_pct,updated_at)
        SELECT
          jockey,
          bucket,
          SUM(starts) AS starts,
          SUM(wins)   AS wins,
          CASE WHEN SUM(starts)>0 THEN 1.0*SUM(wins)/SUM(starts) ELSE 0.0 END AS win_pct,
          DATE('now')
        FROM combos
        GROUP BY 1,2
    """)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM tj_rollup_combo")
    n_combo = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM tj_rollup_jockey")
    n_jky = cur.fetchone()[0]
    return n_combo, n_jky

def print_sample(cur: sqlite3.Cursor, table: str, who: str) -> None:
    # Show a few rows with wins>0 so we can see it's not all zeros
    if table == "combo":
        cur.execute("""
            SELECT trainer, jockey, bucket, wins, starts,
                   ROUND(100.0*win_pct,1)||'%' AS win_pct
            FROM tj_rollup_combo
            WHERE wins>0 AND starts>0
            ORDER BY wins DESC, starts DESC
            LIMIT 5
        """)
    else:
        cur.execute("""
            SELECT jockey, bucket, wins, starts,
                   ROUND(100.0*win_pct,1)||'%' AS win_pct
            FROM tj_rollup_jockey
            WHERE wins>0 AND starts>0
            ORDER BY wins DESC, starts DESC
            LIMIT 5
        """)
    rows = cur.fetchall()
    for r in rows:
        print("  •", " | ".join(str(x) for x in r), flush=True)

def main():
    parser = argparse.ArgumentParser(description="Compute jockey/trainer form rollups.")
    parser.add_argument("--days", type=int, default=365, help="window size in days (default 365)")
    args = parser.parse_args()

    db_path = db_path_from_repo()
    log(f"db={db_path}")
    cutoff = (date.today() - timedelta(days=args.days)).isoformat()
    log(f"window_days={args.days}  cutoff>={cutoff}")

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass

    ensure_rollup_tables(conn)

    src = choose_source(conn, cutoff)
    if src == "tj_daily":
        n_combo, n_jky = rebuild_from_tj_daily(conn, cutoff)
    else:
        log("tj_daily has zero wins; falling back to combos")
        n_combo, n_jky = rebuild_from_combos(conn)

    log(f"rows combo={n_combo}  jockey={n_jky}")

    cur = conn.cursor()
    print_sample(cur, "combo", "combo")
    print_sample(cur, "jockey", "jockey")

if __name__ == "__main__":
    try:
        main()
    except sqlite3.Error as e:
        log(f"sqlite error: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"error: {e}")
        sys.exit(1)