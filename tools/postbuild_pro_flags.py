
import re, sys, sqlite3, html
from pathlib import Path

def badge(t):      return "<span class='badge'>{}</span>".format(html.escape(t))
def pbadge(t):     return "<span class='badge pro'>{}</span>".format(html.escape(t))

def load_db_counts(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT horse_key, COUNT(*) FROM runs GROUP BY horse_key")
        m = {str(k): int(n) for (k, n) in cur.fetchall()}
        conn.close()
        return m
    except Exception:
        return {}

def add_class(tr_html, cls):
    if 'class=' in tr_html:
        return re.sub(r'class=["\']([^"\']*)["\']',
                      lambda m: 'class="{} {}"'.format(m.group(1), cls) if cls not in m.group(1).split() else m.group(0),
                      tr_html, count=1)
    else:
        return tr_html.replace("<tr", '<tr class="{}"'.format(cls), 1)

def replace_flags(in_html, db_counts):
    def repl_row(m):
        tr = m.group(0)

        # Extract horse_key if annotate added it (data-hk="..."); otherwise None
        hk_m = re.search(r'data-hk="([^"]+)"', tr)
        hk = hk_m.group(1) if hk_m else None

        # Pull SpeedForm / ClassΔ from the tooltip title we already emit in the HTML
        title_m = re.search(r"<span class=['\"]sub['\"][^>]*title=['\"]([^'\"]+)['\"]", tr)
        sf = cd = None
        if title_m:
            title = title_m.group(1)
            sfm = re.search(r"SpeedForm\s+([+\-−]?\d+(?:\.\d+)?)σ", title)
            cdm = re.search(r"ClassΔ\s+([+\-−]?\d+(?:\.\d+)?)σ", title)
            if sfm: sf = sfm.group(1).replace('−','-')
            if cdm: cd = cdm.group(1).replace('−','-')

        flags = []
        if sf is not None: flags.append("SF:{}".format(sf))
        if cd is not None: flags.append("ΔC:{}".format(cd))
        if hk and hk in db_counts: flags.append("DB{}".format(db_counts[hk]))

        # Locate the Flags cell (second-to-last <td>)
        tds = list(re.finditer(r'(<td[^>]*>)(.*?)(</td>)', tr, flags=re.S))
        if len(tds) < 2:
            return tr
        flags_td = tds[-2]
        old = re.sub(r"\s+"," ", flags_td.group(2)).strip()

        # Keep ACTION/PRIME labels as separate green badges
        add = []
        if re.search(r'\bPRIME\b', old):  add.append(pbadge("PRIME"))
        if re.search(r'\bACTION\b', old): add.append(pbadge("ACTION"))

        new_flags_html = (" ".join(add) + (" " if add and flags else "") + " ".join(badge(x) for x in flags)) if flags or add else "—"

        # Replace Flags cell content while preserving the td tag
        start, end = flags_td.span(2)  # only inner content
        tr2 = tr[:start] + new_flags_html + tr[end:]

        # Row coloring: prime/action
        bet_cell = tds[-1].group(2)
        if re.search(r'\$[0-9]', bet_cell):      tr2 = add_class(tr2, "row-action")
        if re.search(r'\bPRIME\b', old):         tr2 = add_class(tr2, "row-prime")

        return tr2

    return re.sub(r"<tr[^>]*>.*?</tr>", repl_row, in_html, flags=re.S)

def main():
    if len(sys.argv) != 2:
        print("usage: python tools/postbuild_pro_flags.py outputs/YYYY-MM-DD_horses_targets+full+ANN.html")
        sys.exit(2)
    ann_path = Path(sys.argv[1]).resolve()
    out_path = ann_path.with_name(ann_path.stem.replace("+ANN","") + "+PRO.html")

    html_text = ann_path.read_text(encoding="utf-8")
    db_counts = load_db_counts(Path("data/horses.db"))

    out_html = replace_flags(html_text, db_counts)
    out_path.write_text(out_html, encoding="utf-8")
    print("[ok] wrote", out_path)

if __name__ == "__main__":
    main()
