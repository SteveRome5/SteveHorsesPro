# File: tools/verify_signals.py
"""
Verify that TRAIN writes signals and PRO reads/uses the same payloads.

Usage:
  python tools/verify_signals.py \
      --train /mnt/data/steve_horses_train.py \
      --pro   /mnt/data/steve_horses_pro.py \
      --limit 10 \
      --out   ./signals_check.html

What it checks:
- Train SIGDIR exists, Pro DATA_DIR/signals exists.
- File set overlap and mismatches.
- For each selected <Track>|<YYYY-MM-DD>.json:
  * Train JSON is loadable and non-empty.
  * Pro _load_signals(track,date) returns same-length payload.
  * Per row consistency on (race, program, used). Score compared with tolerance.
  * Pro _sig_used(...) equals row['used'].

Exit codes:
  0 = all green
  1 = soft issues (warnings only)
  2 = hard failures (mismatches)
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
from datetime import datetime
import html

# ---- minimal helpers ---------------------------------------------------
def import_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)                 # type: ignore
    return mod

def list_signal_files(dir_path: Path):
    return sorted([p for p in dir_path.glob("*.json") if "|" in p.stem])

def parse_meet_filename(p: Path):
    # <Track>|<YYYY-MM-DD>.json
    stem = p.stem
    if "|" not in stem:
        return None, None
    track, date_iso = stem.split("|", 1)
    return track, date_iso

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}

def to_int(s) -> int:
    try:
        import re
        return int(re.sub(r"[^\d]","", str(s) or "0") or "0")
    except Exception:
        return 0

# ---- core verification --------------------------------------------------
def verify(train_mod, pro_mod, limit: int = 10):
    # Resolve dirs
    train_sigdir: Path = getattr(train_mod, "SIGDIR", None)
    if not isinstance(train_sigdir, Path):
        raise RuntimeError("Train module lacks SIGDIR Path")
    pro_datadir: Path = getattr(pro_mod, "DATA_DIR", None)
    if not isinstance(pro_datadir, Path):
        raise RuntimeError("Pro module lacks DATA_DIR Path")
    pro_sigdir = pro_datadir / "signals"

    # Scan files
    t_files = list_signal_files(train_sigdir)
    p_files = list_signal_files(pro_sigdir)
    t_set = {f.name for f in t_files}
    p_set = {f.name for f in p_files}
    both = sorted(t_set & p_set)
    only_train = sorted(t_set - p_set)
    only_pro = sorted(p_set - t_set)

    # Choose subset, bias to newest by mtime
    files_by_mtime = sorted(
        [train_sigdir / name for name in both],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:max(1, limit)]

    results = {
        "meta": {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_sigdir": str(train_sigdir),
            "pro_sigdir": str(pro_sigdir),
            "counts": {
                "train_files": len(t_files),
                "pro_files": len(p_files),
                "both": len(both),
                "only_train": len(only_train),
                "only_pro": len(only_pro),
                "checked": len(files_by_mtime),
            },
        },
        "diff": {
            "only_train": only_train,
            "only_pro": only_pro,
        },
        "checks": [],
        "summary": {"warnings": 0, "failures": 0}
    }

    # Core comparisons
    fail = 0
    warn = 0
    for path in files_by_mtime:
        track, date_iso = parse_meet_filename(path)
        check = {
            "file": path.name,
            "track": track,
            "date": date_iso,
            "train_rows": None,
            "pro_rows": None,
            "issues": [],
        }

        # Load Train JSON
        train_rows = load_json(path)
        if isinstance(train_rows, dict) and "__error__" in train_rows:
            check["issues"].append(f"FAIL: train JSON unreadable: {train_rows['__error__']}")
            fail += 1
            results["checks"].append(check)
            continue
        if not isinstance(train_rows, list) or not train_rows:
            check["issues"].append("FAIL: train JSON is empty or not a list")
            fail += 1
            results["checks"].append(check)
            continue

        check["train_rows"] = len(train_rows)

        # Ask Pro to read signals for same meet
        try:
            pro_rows = pro_mod._load_signals(track, date_iso)  # noqa: protected access ok for diagnostic
        except Exception as e:
            check["issues"].append(f"FAIL: Pro _load_signals blew up: {type(e).__name__}: {e}")
            fail += 1
            results["checks"].append(check)
            continue

        if not isinstance(pro_rows, list):
            check["issues"].append("FAIL: Pro _load_signals did not return list")
            fail += 1
            results["checks"].append(check)
            continue

        check["pro_rows"] = len(pro_rows)

        # Compare counts
        if len(train_rows) != len(pro_rows):
            check["issues"].append(f"FAIL: row-count mismatch train={len(train_rows)} pro={len(pro_rows)}")
            fail += 1

        # Build lookup by (race, program)
        def key(r): return (to_int(r.get("race")), str(r.get("program") or "").strip())
        t_map = {key(r): r for r in train_rows if isinstance(r, dict)}
        p_map = {key(r): r for r in pro_rows if isinstance(r, dict)}

        # Hard compare existence and crucial fields
        for k, tr in t_map.items():
            pr = p_map.get(k)
            if pr is None:
                check["issues"].append(f"FAIL: Pro missing row {k}")
                fail += 1
                continue

            # used flag
            t_used = bool(tr.get("used"))
            p_used = bool(pr.get("used"))
            if t_used != p_used:
                check["issues"].append(f"FAIL: used mismatch for {k}: train={t_used} pro={p_used}")
                fail += 1

            # score tolerance
            t_score = tr.get("score", None)
            p_score = pr.get("score", None)
            if isinstance(t_score, (int, float)) and isinstance(p_score, (int, float)):
                if abs(float(t_score) - float(p_score)) > 1e-6:
                    check["issues"].append(f"WARN: score drift for {k}: {t_score} vs {p_score}")
                    warn += 1

            # verify Pro’s _sig_used agrees
            try:
                race_i = k[0]
                program = k[1]
                pro_used_call = pro_mod._sig_used(track, race_i, program, date_iso)
                if bool(pro_used_call) != t_used:
                    check["issues"].append(f"FAIL: _sig_used disagrees for {k}: _sig_used={pro_used_call} train.used={t_used}")
                    fail += 1
            except Exception as e:
                check["issues"].append(f"WARN: _sig_used call error for {k}: {type(e).__name__}: {e}")
                warn += 1

        # Soft sanity on required fields
        req = {"race", "program", "used", "score"}
        missing_any = [k for r in train_rows if isinstance(r, dict) for k in (req - set(r.keys()))]
        if missing_any:
            check["issues"].append(f"WARN: some rows missing required fields {sorted(set(missing_any))}")
            warn += 1

        results["checks"].append(check)

    results["summary"]["warnings"] = warn
    results["summary"]["failures"] = fail
    return results

def write_outputs(results: dict, out_html: Path, out_json: Path):
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    parts = []
    s = results["summary"]
    meta = results["meta"]
    parts.append("<!doctype html><meta charset='utf-8'>")
    parts.append("<style>body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px} .bad{color:#b00020} .warn{color:#9a6b00} .ok{color:#197300} pre{background:#f6f6f6;padding:12px;border-radius:6px;overflow:auto}</style>")
    parts.append(f"<h1>Train→Pro Signals Check</h1>")
    parts.append(f"<p><b>Train SIGDIR:</b> {html.escape(meta['train_sigdir'])}<br>"
                 f"<b>Pro SIGDIR:</b> {html.escape(meta['pro_sigdir'])}</p>")
    parts.append(f"<p>Files: train={meta['counts']['train_files']} pro={meta['counts']['pro_files']} both={meta['counts']['both']} checked={meta['counts']['checked']}</p>")

    if results["diff"]["only_train"]:
        parts.append("<p class='warn'><b>Only in Train:</b> " + ", ".join(map(html.escape, results["diff"]["only_train"])) + "</p>")
    if results["diff"]["only_pro"]:
        parts.append("<p class='warn'><b>Only in Pro:</b> " + ", ".join(map(html.escape, results["diff"]["only_pro"])) + "</p>")

    status = "ok" if s["failures"] == 0 else "bad"
    parts.append(f"<p class='{status}'><b>Summary:</b> failures={s['failures']}, warnings={s['warnings']}</p>")

    for chk in results["checks"]:
        ok = not any(msg.startswith("FAIL") for msg in chk["issues"])
        color = "ok" if ok else "bad"
        parts.append(f"<h2 class='{color}'>{html.escape(chk['file'])}</h2>")
        parts.append(f"<p>train_rows={chk.get('train_rows')} pro_rows={chk.get('pro_rows')}</p>")
        if chk["issues"]:
            parts.append("<ul>")
            for issue in chk["issues"]:
                cls = "warn" if issue.startswith("WARN") else "bad"
                parts.append(f"<li class='{cls}'>{html.escape(issue)}</li>")
            parts.append("</ul>")
        else:
            parts.append("<p class='ok'>OK</p>")

    out_html.write_text("\n".join(parts), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to steve_horses_train.py")
    ap.add_argument("--pro", required=True, help="Path to steve_horses_pro.py")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default="./signals_check.html")
    ap.add_argument("--json", default="./signals_check.json")
    args = ap.parse_args()

    train_mod = import_module_from_path("train_mod", args.train)
    pro_mod = import_module_from_path("pro_mod", args.pro)

    results = verify(train_mod, pro_mod, limit=args.limit)
    out_html = Path(args.out)
    out_json = Path(args.json)
    write_outputs(results, out_html, out_json)

    fails = results["summary"]["failures"]
    warns = results["summary"]["warnings"]
    print(f"[signals-check] wrote {out_html} and {out_json} | failures={fails} warnings={warns}")
    if fails:
        raise SystemExit(2)
    elif warns:
        raise SystemExit(1)
    else:
        raise SystemExit(0)

if __name__ == "__main__":
    main()