#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_training_csv.py  (scavenger edition)
Crawls any SQLite DB under data/ (or RESULTS_DB env) and exports runner-level rows
to outputs/training_examples.csv for calibrate_overlays.py.

It:
- Scans ALL tables, picks any with a plausible finish/win column
- Uses any columns it finds for features; missing fields are left blank
- Applies --days filter only if a date column exists; otherwise exports all
"""

from __future__ import annotations
import os, re, csv, sqlite3, argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

HOME = Path.home()
BASE = HOME / "Desktop" / "SteveHorsesPro"
DATA_DIR = BASE / "data"
OUT_DIR  = BASE / "outputs"
LOG_DIR  = BASE / "logs"
for d in (DATA_DIR, OUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "training_examples.csv"

EXPECTED = [
    "track","surface","rail","post",
    "trainer_win_pct","jockey_win_pct","tj_win",
    "days_since","start_since_layoff",
    "ep","lp",
    "last_track","this_track",
    "win",
]

FINISH_KEYS = {"finish_position","finish","pos","placing","final_pos","result_pos"}
WIN_KEYS    = {"win","is_winner","winner_flag"}
POST_KEYS   = {"post_position","program","program_number","pp","saddle","saddle_number","gate"}
TRACK_KEYS  = {"track","track_name","name","this_track"}
SURF_KEYS   = {"surface","track_surface","course"}
RAIL_KEYS   = {"rail","turf_rail","rail_setting","railDistance","rail_distance"}
DATE_KEYS   = {"race_date","date","start_date","event_date"}
DSL_KEYS    = {"days_since","dsl","layoff_days","last_start_days"}
NSL_KEYS    = {"start_since_layoff","starts_since_layoff","since_layoff"}
TRN_KEYS    = {"trainer_win_pct","trainerWinPct","trainer_win","trainer_pct"}
JKY_KEYS    = {"jockey_win_pct","jockeyWinPct","jockey_win","jockey_pct"}
TJ_KEYS     = {"tj_win","combo_win","trainer_jockey_win"}
EP_KEYS     = {"ep","early_pace","earlyPace","quirin"}
LP_KEYS     = {"lp","late_pace","latePace"}
LAST_TRK    = {"last_track","prev_track","prior_track"}

def log(msg: str) -> None:
    try:
        (LOG_DIR / "export_training_csv.log").open("a", encoding="utf-8").write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        )
    except Exception:
        pass

def find_sqlite() -> Optional[Path]:
    for k in ("RESULTS_DB","RESULTS_DB_PATH","DB_RESULTS_PATH"):
        v = os.getenv(k)
        if v and Path(v).exists():
            return Path(v)
    # common names
    for p in (
        DATA_DIR / "results.sqlite",
        DATA_DIR / "db" / "results.sqlite",
        DATA_DIR / "results.db",
        DATA_DIR / "racing_results.sqlite",
    ):
        if p.exists(): return p
    for p in DATA_DIR.rglob("*.sqlite"):
        return p
    for p in DATA_DIR.rglob("*.db"):
        return p
    return None

def table_columns(conn, tbl: str) -> List[str]:
    try:
        cur = conn.execute(f"PRAGMA table_info({tbl})")
        return [r[1] for r in cur.fetchall()]
    except:
        return []

def pick(colset: set, keys: set) -> Optional[str]:
    for k in keys:
        if k in colset: return k
    return None

def to_int(v):
    try:
        if v in (None,"","NA"): return None
        return int(v)
    except:
        # try extracting digits
        try:
            m = re.search(r"\d+", str(v))
            return int(m.group()) if m else None
        except:
            return None

def to_float(v):
    try:
        if v in (None,"","NA"): return None
        return float(v)
    except:
        return None

def first_present(row, *cols):
    for c in cols:
        if c and c in row.keys():
            v = row[c]
            if v not in (None,""):
                return v
    return None

def export_scavenger(db_path: Path, days: int) -> int:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]

    total_out = 0
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=EXPECTED)
        wr.writeheader()

        for tbl in tables:
            cols = table_columns(conn, tbl)
            colset = set(cols)
            if not colset: continue

            # Must have some outcome signal
            finish_col = pick(colset, FINISH_KEYS)
            win_col    = pick(colset, WIN_KEYS)
            if not (finish_col or win_col):
                continue

            # Optional date filter
            date_col = pick(colset, DATE_KEYS)
            where = ""; params = {}
            if date_col:
                where = f" WHERE date({date_col}) >= date('now', :offset)"
                params["offset"] = f"-{days} day"

            # Build SELECT * (cheap) and filter in Python
            sql = f"SELECT * FROM {tbl}{where}"
            try:
                cur = conn.execute(sql, params)
                rows = cur.fetchall()
            except Exception as e:
                log(f"scan fail {tbl}: {e}")
                continue

            # likely runner-level features present?
            has_any_feat = any(k in colset for k in (TRN_KEYS|JKY_KEYS|TJ_KEYS|EP_KEYS|LP_KEYS|DSL_KEYS|NSL_KEYS|POST_KEYS|TRACK_KEYS|SURF_KEYS))
            if not has_any_feat and len(rows) > 0:
                # still allow, we may at least have post/track
                pass

            for r in rows:
                # derive win
                win_val = None
                if win_col and (win_col in r.keys()):
                    v = r[win_col]
                    try:
                        win_val = int(v)
                        if win_val not in (0,1): win_val = 1 if int(v)==1 else 0
                    except:
                        s = str(v).strip().lower()
                        if s in ("y","yes","true","t","1"): win_val = 1
                        elif s in ("n","no","false","f","0"): win_val = 0
                if win_val is None and finish_col and finish_col in r.keys():
                    try:
                        win_val = 1 if int(r[finish_col]) == 1 else 0
                    except:
                        win_val = 0

                track  = first_present(r, *(TRACK_KEYS & colset))
                surface= first_present(r, *(SURF_KEYS  & colset))
                rail   = first_present(r, *(RAIL_KEYS  & colset))
                post   = first_present(r, *(POST_KEYS  & colset))
                trn    = first_present(r, *(TRN_KEYS   & colset))
                jky    = first_present(r, *(JKY_KEYS   & colset))
                tj     = first_present(r, *(TJ_KEYS    & colset))
                dsl    = first_present(r, *(DSL_KEYS   & colset))
                nsl    = first_present(r, *(NSL_KEYS   & colset))
                ep     = first_present(r, *(EP_KEYS    & colset))
                lp     = first_present(r, *(LP_KEYS    & colset))
                last_t = first_present(r, *(LAST_TRK   & colset))

                row_out = {
                    "track": track if track is not None else "",
                    "surface": surface if surface is not None else "",
                    "rail": rail if rail is not None else "",
                    "post": to_int(post),
                    "trainer_win_pct": to_float(trn),
                    "jockey_win_pct": to_float(jky),
                    "tj_win": to_float(tj),
                    "days_since": to_float(dsl),
                    "start_since_layoff": to_float(nsl),
                    "ep": to_float(ep),
                    "lp": to_float(lp),
                    "last_track": last_t if last_t is not None else "",
                    "this_track": track if track is not None else "",
                    "win": int(win_val or 0),
                }

                # if we have at least post + win OR (any feature + win), keep
                keep = (row_out["win"] in (0,1)) and (
                    row_out["post"] is not None or
                    any(row_out[k] not in (None,"") for k in ("trainer_win_pct","jockey_win_pct","tj_win","ep","lp","days_since"))
                )
                if keep:
                    wr.writerow(row_out)
                    total_out += 1

    if total_out == 0:
        print("[ERR] Export produced 0 rows (scavenger). Schema may hide outcome columns.")
    else:
        print(f"[ok] Wrote {CSV_PATH} (rows={total_out})")
    return total_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()

    db = None
    # try db_results hook first, if available and returns a CSV
    try:
        import db_results  # type: ignore
        for fname in ("export_training_rows","dump_training_rows","to_training_csv"):
            fn = getattr(db_results, fname, None)
            if fn:
                path = fn(days=args.days)
                if path and Path(path).exists():
                    src = Path(path)
                    CSV_PATH.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                    print(f"[ok] Wrote {CSV_PATH} (via db_results.{fname})")
                    return
    except Exception as e:
        log(f"db_results hook not used: {e}")

    db = os.getenv("RESULTS_DB")
    if db and Path(db).exists():
        n = export_scavenger(Path(db), args.days); 
        if n>0: return
    found = find_sqlite()
    if found:
        export_scavenger(found, args.days)
    else:
        print("[ERR] Could not find a results DB under data/. Set RESULTS_DB=/path/to/results.sqlite and re-run.")

if __name__ == "__main__":
    main()