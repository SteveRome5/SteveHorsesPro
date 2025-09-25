
import re, json, sqlite3, html
from pathlib import Path

def html_escape(s): return html.escape(str(s) if s is not None else "")

def load_train_map(base: Path, track: str, ymd: str):
    path = base / "data" / "signals" / f"{track}|{ymd}.json"
    out = {}
    if not path.exists():
        return out
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return out
    rows = []
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        rows = obj.get("signals") or obj.get("rows") or obj
        if isinstance(rows, dict):
            # keyed like "race|pgm" or similar
            tmp = []
            for k,v in rows.items():
                if isinstance(k,str) and "|" in k:
                    sp = k.split("|")
                    if len(sp) >= 2:
                        tmp.append({"race": sp[-2], "program": sp[-1], **(v if isinstance(v,dict) else {})})
            rows = tmp
    for r in rows or []:
        race = str(r.get("race") or r.get("Race") or "").strip()
        pgm  = str(r.get("program") or r.get("Program") or r.get("pgm") or "").strip()
        if not race or not pgm: continue
        flags = r.get("flags")
        if isinstance(flags,str):
            flags = [t for t in re.split(r"[|,\\s]+", flags) if t]
        if not isinstance(flags,list):
            flags = []
        out[(race, pgm)] = {"flags": flags, "raw": r}
    return out

def open_runs_db(base: Path):
    db = base / "data" / "horses.db"
    if not db.exists():
        return None, None
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(runs);")
    cols = [c[1].lower() for c in cur.fetchall()]
    # pick a horse-name-like column
    candidates = [c for c in cols if "horse" in c or c == "name" or c == "horse_name"]
    horse_col = candidates[0] if candidates else None
    return conn, horse_col

def build_counts(conn, horse_col):
    if not conn or not horse_col:
        return {}
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT LOWER(TRIM({horse_col})), COUNT(*) FROM runs GROUP BY LOWER(TRIM({horse_col}))")
        return { (r[0] or "").strip(): int(r[1] or 0) for r in cur.fetchall() }
    except Exception:
        return {}

def extract_cells(row_html):
    # returns list of <td...>...</td> chunks
    return re.findall(r"<td[^>]*>.*?</td>", row_html, flags=re.S)

def td_text(td):
    # strip tags inside cell to get text
    inner = re.sub(r"<[^>]+>", "", td, flags=re.S)
    return html.unescape(inner).strip()

def replace_cell(cells, idx, new_html):
    cells[idx] = new_html
    return cells

def rebuild_row(cells):
    return "".join(cells)

def annotate_report(base: Path, in_path: Path, out_path: Path):
    html_text = in_path.read_text(encoding="utf-8")
    # discover date from filename
    mdate = re.search(r"(\\d{4}-\\d{2}-\\d{2})_horses_targets\\+full\\.html$", str(in_path))
    ymd = mdate.group(1) if mdate else None

    conn, horse_col = open_runs_db(base)
    counts_map = build_counts(conn, horse_col)

    # We’ll patch each table row
    def patch_row(match):
        row = match.group(0)
        cells = extract_cells(row)
        if len(cells) < 13:
            return row  # unexpected shape

        track = td_text(cells[0])
        race  = td_text(cells[1])
        pgm   = td_text(cells[2])
        horse = td_text(cells[3])

        # train map per track/day
        train_map = load_train_map(base, track, ymd) if ymd else {}

        # Build flags
        flags = []
        # DB count by horse name, case-insensitive
        key = horse.lower().strip()
        dbn = counts_map.get(key)
        if dbn and dbn > 0:
            flags.append(f"DB{dbn}")

        # TRAIN flags if present for (race, program)
        used_train = False
        if train_map:
            rec = train_map.get((race, pgm))
            if rec:
                used_train = True
                tflags = rec.get("flags") or []
                flags.extend([str(t) for t in tflags if t])

        # Replace Flags cell (#12 = index 11)
        flags_badges = " ".join([f"<span class='badge'>{html_escape(t)}</span>" for t in flags]) if flags else "—"
        cells[11] = f"<td>{flags_badges}</td>"

        # Update Source cell (#13 = index 12)
        src_html = cells[12]
        if used_train and ("+TRAIN" not in src_html):
            src_html = re.sub(r"(>PRO(?:\\+LOGIC)?)<", r"\\1+TRAIN<", src_html)
        cells[12] = src_html

        return rebuild_row(cells)

    new_html = re.sub(r"<tr[^>]*>.*?</tr>", patch_row, html_text, flags=re.S)
    out_path.write_text(new_html, encoding="utf-8")
    if conn: conn.close()

if __name__ == "__main__":
    import sys
    base = Path.cwd()
    if len(sys.argv) != 2:
        print("usage: python tools/postbuild_annotate.py outputs/YYYY-MM-DD_horses_targets+full.html")
        sys.exit(2)
    in_path = Path(sys.argv[1]).resolve()
    out_path = in_path.with_name(in_path.stem + "+ANN.html")
    annotate_report(base, in_path, out_path)
    print("[ok] wrote", out_path)
