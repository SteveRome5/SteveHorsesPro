#!/usr/bin/env python3
# tools/jockey_form.py
# Build jockey "form" aggregates from tj_daily (trainer+jockey daily combos).
# Writes results to sqlite: data/tj.sqlite -> table: jockey_form
# Optional CSV export for a quick sanity check.

import argparse, pathlib, sqlite3, sys, csv
from datetime import date, datetime, timedelta

APP = pathlib.Path(__file__).resolve().parents[1]
DB  = APP / "data" / "tj.sqlite"

DDL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS jockey_form (
  asof         TEXT NOT NULL,             -- YYYY-MM-DD (when we computed)
  window_days  INTEGER NOT NULL,          -- lookback window
  scope        TEXT NOT NULL,             -- 'overall' or 'bucket'
  bucket       TEXT NOT NULL,             -- '' for overall; else e.g. 'Churchill Downs|dirt|unk'
  jockey_norm  TEXT NOT NULL,
  starts       INTEGER NOT NULL,
  wins         INTEGER NOT NULL,
  win_pct      REAL NOT NULL,
  PRIMARY KEY (asof, window_days, scope, bucket, jockey_norm)
);
CREATE INDEX IF NOT EXISTS idx_jf_jockey ON jockey_form(jockey_norm, asof);
"""

def open_db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB))
    con.executescript(DDL)
    return con

def as_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def rows_to_csv(rows, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asof","window_days","scope","bucket","jockey_norm","starts","wins","win_pct"])
        w.writerows(rows)

def compute_jockey_form(con, asof: date, window_days: int, write_csv: bool=False):
    d0 = asof - timedelta(days=window_days-1)
    aiso = asof.isoformat()

    # OVERALL per jockey across all buckets
    q_overall = """
      SELECT ? AS asof, ? AS window_days, 'overall' AS scope, '' AS bucket,
             jockey_norm,
             SUM(starts) AS starts, SUM(wins) AS wins,
             CASE WHEN SUM(starts)>0 THEN ROUND(100.0*SUM(wins)/SUM(starts),2) ELSE 0 END AS win_pct
      FROM tj_daily
      WHERE dt BETWEEN ? AND ?
      GROUP BY jockey_norm
      HAVING starts > 0
    """

    # PER BUCKET per jockey
    q_bucket = """
      SELECT ? AS asof, ? AS window_days, 'bucket' AS scope, bucket,
             jockey_norm,
             SUM(starts) AS starts, SUM(wins) AS wins,
             CASE WHEN SUM(starts)>0 THEN ROUND(100.0*SUM(wins)/SUM(starts),2) ELSE 0 END AS win_pct
      FROM tj_daily
      WHERE dt BETWEEN ? AND ?
      GROUP BY bucket, jockey_norm
      HAVING starts > 0
    """

    cur = con.cursor()

    # fetch & upsert helper
    def fetch_and_upsert(sql, label):
        rows = list(cur.execute(sql, (aiso, window_days, d0.isoformat(), aiso)))
        if not rows:
            print(f"[{aiso}] {label}: 0 rows")
            return 0
        with con:
            con.executemany("""
              INSERT INTO jockey_form(asof,window_days,scope,bucket,jockey_norm,starts,wins,win_pct)
              VALUES(?,?,?,?,?,?,?,?)
              ON CONFLICT(asof,window_days,scope,bucket,jockey_norm)
              DO UPDATE SET
                starts = excluded.starts,
                wins   = excluded.wins,
                win_pct= excluded.win_pct
            """, rows)
        print(f"[{aiso}] {label}: {len(rows)} rows upserted")
        return len(rows), rows

    n_overall, rows_overall = fetch_and_upsert(q_overall, "overall")
    n_bucket,  rows_bucket  = fetch_and_upsert(q_bucket,  "bucket")

    if write_csv:
        out = APP / "data" / f"jockey_form_{aiso}_{window_days}d.csv"
        all_rows = []
        if n_overall: all_rows += rows_overall
        if n_bucket:  all_rows += rows_bucket
        if all_rows:
            rows_to_csv(all_rows, out)
            print(f"[{aiso}] wrote CSV: {out}")

def main():
    ap = argparse.ArgumentParser(description="Compute jockey form aggregates from tj_daily.")
    ap.add_argument("--asof", type=str, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--days-back", type=int, default=60, help="Lookback window in days (default: 60)")
    ap.add_argument("--csv", action="store_true", help="Also write a CSV snapshot for sanity checks")
    args = ap.parse_args()

    asof = as_date(args.asof) if args.asof else date.today()
    window = int(args.days_back or 60)

    con = open_db()
    compute_jockey_form(con, asof, window, write_csv=args.csv)
    # small summary
    tot = con.execute("SELECT COUNT(*) FROM jockey_form WHERE asof=? AND window_days=?", (asof.isoformat(), window)).fetchone()[0]
    print(f"DONE. jockey_form rows for asof={asof} window={window}d: {tot}")
    con.close()

if __name__ == "__main__":
    sys.exit(main())