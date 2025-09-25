
import re, json, sqlite3, html, math
from pathlib import Path
from statistics import fmean

# ---------- Config knobs (override via env vars later if you want) ----------
PRIME_WIN_MIN = 18.0
PRIME_EDGE_PP =  1.5
ACTION_WIN_MIN = 12.0
ACTION_EDGE_PP =  0.5
EXACTA_MAX = 3

def esc(x): return html.escape("" if x is None else str(x))

def parse_pct(s):
    try: return float(str(s).strip().rstrip("%"))
    except: return None

def colfind(cols, *keys):
    cols = [c.lower() for c in cols]
    for k in keys:
        for c in cols:
            if k in c: return c
    return None

def open_runs_db(base: Path):
    db = base / "data" / "horses.db"
    if not db.exists(): return None, None, None, None
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(runs)")
    cols = [c[1] for c in cur.fetchall()]
    horse_col = colfind(cols,"horse","name","horse_name")
    speed_col = colfind(cols,"speed")
    class_col = colfind(cols,"class")
    date_col  = colfind(cols,"date")  # best effort
    return conn, horse_col, speed_col, class_col if class_col!=speed_col else None, date_col

def db_counts_map(conn, horse_col):
    if not conn or not horse_col: return {}
    cur = conn.cursor()
    cur.execute(f"SELECT LOWER(TRIM({horse_col})), COUNT(*) FROM runs GROUP BY LOWER(TRIM({horse_col}))")
    return { (r[0] or "").strip(): int(r[1] or 0) for r in cur.fetchall() }

def recent_metrics(conn, horse_col, speed_col, class_col, date_col, horse_name, k=6):
    if not conn or not horse_col: return None, None
    key = (horse_name or "").lower().strip()
    cur = conn.cursor()
    order = f" ORDER BY {date_col} DESC" if date_col else ""
    cols = [horse_col]
    sel = []
    if speed_col: sel.append(speed_col)
    if class_col: sel.append(class_col)
    if not sel: return None, None
    q = f"SELECT {', '.join(sel)} FROM runs WHERE LOWER(TRIM({horse_col}))=?{order} LIMIT {k}"
    try:
        cur.execute(q,(key,))
        speeds, classes = [], []
        for row in cur.fetchall():
            if speed_col:
                v = row[0]
                try: v=float(v)
                except: v=None
                if v is not None: speeds.append(v)
            if class_col:
                idx = 1 if speed_col else 0
                v = row[idx]
                try: v=float(v)
                except: v=None
                if v is not None: classes.append(v)
        sf = fmean(speeds) if speeds else None
        if classes:
            half = max(1,len(classes)//2)
            cd = fmean(classes[:half]) - fmean(classes[half:]) if len(classes)>=2 else 0.0
        else:
            cd = None
        return sf, cd
    except Exception:
        return None, None

def load_train_candidates(base: Path, track: str, ymd: str):
    p = base / "data" / "signals" / f"{track}|{ymd}.json"
    if not p.exists(): return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    items = []
    if isinstance(obj,list):
        items = obj
    elif isinstance(obj,dict):
        if "signals" in obj and isinstance(obj["signals"], list):
            items = obj["signals"]
        else:
            # permissive: flatten dict of dicts
            for v in obj.values():
                if isinstance(v,list): items.extend(v)
    # normalize
    out = {}
    for r in items:
        race = str(r.get("race") or r.get("Race") or "").strip()
        pgm  = str(r.get("program") or r.get("Program") or r.get("pgm") or "").strip()
        name = str(r.get("horse") or r.get("Horse") or r.get("name") or "").strip().lower()
        flags = r.get("flags")
        if isinstance(flags,str):
            flags = [t for t in re.split(r"[|,\\s]+", flags) if t]
        if not isinstance(flags,list):
            flags = []
        if race and pgm:
            out[("rp",race,pgm)] = {"flags": flags, "name": name}
        if name:
            out[("nm",name)] = {"flags": flags, "name": name}
    return out

def exacta_pairs(programs, probs, anchors=None, k=3):
    # naive independence; anchor on top pick
    if not programs or not probs: return []
    if anchors: base = [programs.index(a) for a in anchors if a in programs][:1]
    else: base = [max(range(len(probs)), key=lambda i: probs[i])]
    i = base[0]
    pairs = []
    for j in range(len(probs)):
        if j==i: continue
        pij = probs[i]*probs[j]*(1.0 - probs[i])  # crude order factor
        pairs.append((i,j,pij))
    pairs.sort(key=lambda t: t[2], reverse=True)
    out=[]
    for i,j,pij in pairs[:k]:
        out.append({"a": programs[i], "b": programs[j], "p": pij})
    return out

def annotate(base: Path, in_path: Path, out_path: Path):
    html_text = in_path.read_text(encoding="utf-8")
    mdate = re.search(r"(\\d{4}-\\d{2}-\\d{2})_horses_targets\\+full(?:\\+ANN)?\\.html$", str(in_path))
    ymd = mdate.group(1) if mdate else None

    conn, horse_col, speed_col, class_col, date_col = open_runs_db(base)
    counts = db_counts_map(conn, horse_col)

    # collect rows to build boards later
    rows_meta = []  # [(key, win, edge, bet_text, row_html, start_idx, end_idx, fields...), ...]

    def patch_tbody(m):
        tbody = m.group(2)
        # replace only row <tr> inside this tbody
        def patch_row(mm):
            row = mm.group(0)
            cells = re.findall(r"<td[^>]*>.*?</td>", row, flags=re.S)
            if len(cells) < 13:
                return row

            get_txt = lambda td: html.unescape(re.sub(r"<[^>]+>","",td,flags=re.S)).strip()
            track = get_txt(cells[0])
            race  = get_txt(cells[1])
            pgm   = get_txt(cells[2])
            horse = get_txt(cells[3])
            winf  = parse_pct(get_txt(cells[4])) or 0.0
            markt = parse_pct(get_txt(cells[5]))
            edge  = get_txt(cells[6]).replace("pp","").strip()
            try: edgef = float(edge)
            except: edgef = None
            bet   = get_txt(cells[10])  # "Bet" column text

            # TRAIN presence (race+pgm, fallback by name)
            used_train = False
            train_flags = []
            if ymd:
                tmap = load_train_candidates(base, track, ymd)
                rec = tmap.get(("rp",race,pgm))
                if not rec and horse:
                    rec = tmap.get(("nm",horse.lower()))
                if rec:
                    used_train = True
                    train_flags = rec.get("flags") or []

            # Flags: DB count + SF + ΔC + TRAIN flags
            fl = []
            dbruns = counts.get((horse or "").lower().strip())
            if dbruns:
                fl.append(f"DB{dbruns}")
            if horse and (speed_col or class_col):
                sf, cd = recent_metrics(conn, horse_col, speed_col, class_col, date_col, horse, k=6)
                if sf is not None:
                    fl.append(f"SF:{sf:.1f}")
                if cd is not None:
                    sign = "▲" if cd>0 else ("▼" if cd<0 else "•")
                    fl.append(f"ΔC:{cd:+.1f}{sign}")
            for t in train_flags[:3]:
                fl.append(str(t))

            flags_html = " ".join([f"<span class='badge'>{esc(t)}</span>" for t in fl]) if fl else "—"
            cells[11] = f"<td>{flags_html}</td>"

            # Source: add +TRAIN only when we actually saw it
            src_html = cells[12]
            if used_train and "+TRAIN" not in src_html:
                src_html = re.sub(r"(>PRO(?:\\+LOGIC)?)<", r"\\1+TRAIN<", src_html)
            cells[12] = src_html

            # Row class for boards (derive if Bet blank)
            is_prime = False
            is_action = False
            if bet and bet != "—":
                # if your core tags Prime/Action, pick that up here (future)
                pass
            else:
                if edgef is not None and winf >= PRIME_WIN_MIN and edgef >= PRIME_EDGE_PP:
                    is_prime = True
                elif edgef is not None and winf >= ACTION_WIN_MIN and edgef >= ACTION_EDGE_PP:
                    is_action = True

            cls = "row-prime" if is_prime else ("row-action" if is_action else "")
            if cls:
                row = re.sub(r"<tr([^>]*)>", fr"<tr class='{cls}'\1>", row, 1)

            # Store for boards
            rows_meta.append({
                "track": track, "race": race, "pgm": pgm, "horse": horse,
                "win": winf, "edge": edgef, "is_prime": is_prime, "is_action": is_action
            })

            # Exacta helper (top-anchored, max 3)
            try:
                # crude: re-scan tbody section for same (track,race) to pull probs/programs
                # in practice we’ll compute per race later; this is a no-op here
                pass
            except Exception:
                pass

            # rebuild row
            row = re.sub(r"<td[^>]*>.*?</td>", lambda i: cells.pop(0), row, count=len(cells), flags=re.S)
            return row

        new_tbody = re.sub(r"<tr[^>]*>.*?</tr>", patch_row, tbody, flags=re.S)
        return m.group(1) + new_tbody + m.group(3)

    # patch only TBODY blocks
    new_html = re.sub(r"(<tbody>)(.*?)(</tbody>)", patch_tbody, html_text, flags=re.S)

    # Build boards content from rows_meta
    def board_list(kind):
        items = [r for r in rows_meta if r["is_"+kind]]
        if not items: return "<p class='small'>No plays today.</p>"
        # group -> show "Track R# — # Pgm Horse (Win%, Edge)"
        lines=[]
        for r in items:
            lines.append(f"<div class='small'><span class='badge pro'>{esc(r['track'])} R{esc(r['race'])}</span> "
                         f"#{esc(r['pgm'])} {esc(r['horse'])} • {r['win']:.2f}% • {r['edge']:+.1f} pp</div>")
        return "\\n".join(lines)

    # replace PRIME and ACTION sections’ contents only, never the headings
    def replace_board(h2_title, content_html):
        pat = re.compile(rf"(<h2>{re.escape(h2_title)}</h2>)(.*?)(?=<h2>|\\Z)", re.S)
        def repl(m):
            head = m.group(1)
            body = m.group(2)
            # replace first paragraph/lines with our content, keep scratch/cap paragraph intact
            # Find the first block after h2 up to the next block-level tag
            return head + "\\n" + content_html + "\\n"
        return pat.sub(repl, new_html)

    new_html = replace_board("PRIME Board", board_list("prime"))
    new_html = replace_board("ACTION Board", board_list("action"))

    out_path.write_text(new_html, encoding="utf-8")
    if conn: conn.close()

if __name__ == "__main__":
    import sys
    base = Path.cwd()
    if len(sys.argv) != 2:
        print("usage: python tools/postbuild_annotate_v2.py outputs/YYYY-MM-DD_horses_targets+full.html")
        sys.exit(2)
    in_path = Path(sys.argv[1]).resolve()
    out_path = in_path.with_name(in_path.stem + "+ANN.html")
    annotate(base, in_path, out_path)
    print("[ok] wrote", out_path)
