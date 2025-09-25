#!/usr/bin/env python3
from __future__ import annotations
import os, ssl, json, re, html, base64
from pathlib import Path
from datetime import date
from urllib.request import Request, urlopen
from urllib.parse import urlencode

VERSION = "v1.4"
TRACKS = ["Saratoga","Del Mar","Santa Anita Park","Gulfstream Park","Keeneland","Parx Racing","Finger Lakes"]
TRACK_IDS = {"SAR","DMR","SA","GP","KEE","PRX","FL"}
MIN_PRICE_PAD = float(os.getenv("MIN_PRICE_PAD","0.15"))  # 15% above fair
TOP_N = int(os.getenv("TOP_N","80"))

RACING_USER = os.getenv("RACING_API_USER","").strip()
RACING_PASS = os.getenv("RACING_API_PASS","").strip()
USE_API = bool(RACING_USER and RACING_PASS)

HOME = Path.home()
BASE = HOME/"Desktop"/"SteveHorsesPro"
OUT  = BASE/"outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- HTTP ----------
CTX = ssl.create_default_context()
def _get(path: str, params: dict|None=None):
    base = "https://api.theracingapi.com"
    url  = base + path + ("?" + urlencode(params) if params else "")
    req  = Request(url, headers={"User-Agent":"Mozilla/5.0"})
    if USE_API:
        tok = base64.b64encode(f"{RACING_USER}:{RACING_PASS}".encode()).decode()
        req.add_header("Authorization","Basic "+tok)
    with urlopen(req, timeout=30, context=CTX) as r:
        return json.loads(r.read().decode("utf-8"))

# ---------- odds ----------
def parse_decimal(s):
    if s is None: return (None,None)
    t = str(s).strip().lower()
    if t in ("even","evens","evs"): return (2.0,0.5)
    m = re.fullmatch(r'(\d+)\s*[/\-:]\s*(\d+)', t)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den>0:
            dec = 1.0 + num/den
            return (dec, 1.0/dec)
    m = re.fullmatch(r'[+-]?\d+', t)  # +250, -120
    if m:
        a = int(m.group(0))
        dec = 1 + (a/100.0 if a>0 else 100.0/abs(a))
        return (dec, 1.0/dec)
    try:
        dec = float(t)
        if dec>1.0: return (dec, 1.0/dec)
    except: pass
    return (None,None)

def frac(dec):
    if not dec or dec<=1: return "—"
    v = dec-1
    best = (9e9,"—")
    for den in (1,2,3,4,5,6,7,8,10,12,14,16,20,32):
        num = round(v*den); err = abs(v - num/den)
        if err<best[0]: best = (err, f"{int(num)}-{int(den)}")
    return best[1]

# ---------- helpers ----------
def pick(d, *keys, default=None):
    for k in keys:
        if isinstance(d,dict) and k in d and d[k] not in (None,""): return d[k]
    return default

def race_no(r, idx_fallback=None):
    for k in ("race_number","raceNumber","raceNo","race_no","number","Race","RaceNo",
              "sequence","seq","order","race_index","raceSequence","RaceNum"):
        v = r.get(k)
        if isinstance(v,int): return v
        if isinstance(v,str) and v.isdigit(): return int(v)
    nm = (r.get("name") or r.get("title") or r.get("RaceName") or "")
    m = re.search(r'\b(?:race|r)\s*(\d{1,2})\b', str(nm), re.I)
    if m: return int(m.group(1))
    for k in ("race_id","raceId","id","slug"):
        v = r.get(k)
        if isinstance(v,str):
            m = re.search(r'[Rr]\s*0*([1-9]\d?)\b', v) or re.search(r'[^0-9]([1-9]\d?)$', v)
            if m: return int(m.group(1))
    return idx_fallback if idx_fallback is not None else "?"

def program_no(e):
    # Prefer explicit program/saddle codes
    for k in ("program_number","programNumber","program","programNo",
              "saddle_number","saddleNumber","horse_number","horseNumber"):
        v = e.get(k)
        if v not in (None,""):
            return str(v).strip()
    # Then post/draw/gate/pp
    for k in ("post_position","postPosition","pp","draw","gate","stall","box"):
        v = e.get(k)
        if v not in (None,""):
            return str(v).strip()
    # Sometimes nested
    h = e.get("horse")
    if isinstance(h, dict):
        for k in ("program_number","programNumber","number"):
            v = h.get(k)
            if v not in (None,""):
                return str(v).strip()
    # Do NOT trust generic e["number"] (often 1 for all)
    return "?"

def horse_name(e):
    if isinstance(e.get("horse"),dict):
        for k in ("name","horse_name"):
            if e["horse"].get(k): return str(e["horse"][k])
    for k in ("horse_name","name","runner_name","entry_name","displayName"):
        if e.get(k): return str(e[k])
    return "?"

def morning_line(e):
    for k in ("morning_line","morningLine","ml","ml_odds","mline","line","program_odds","odds","win_odds"):
        v = e.get(k)
        if v not in (None,""): return v
    return None

def probs(entries):
    n = max(1,len(entries))
    uni = 1.0/n
    p_ml, have = [], 0
    for e in entries:
        dec, imp = parse_decimal(morning_line(e))
        if imp: have += 1; p_ml.append(imp)
        else:   p_ml.append(None)
    w = 0.70 * (have/n)
    s_known = sum(x for x in p_ml if x is not None) or 0.0
    out = []
    for imp in p_ml:
        if imp is not None and s_known>0:
            out.append(max(1e-6, w*(imp/s_known) + (1-w)*uni))
        else:
            out.append(max(1e-6, (1-w)*uni))
    s = sum(out)
    return [x/s for x in out], p_ml  # also expose ml-implied probs

# ---------- API pulls ----------
def todays_meets():
    d = date.today().isoformat()
    js = _get("/v1/north-america/meets", {"start_date":d,"end_date":d}) or {}
    all_meets = js.get("meets") or js.get("data") or []
    chosen = []
    for m in all_meets:
        tname = pick(m,"track_name","track","meeting","course","name", default="")
        tid   = pick(m,"track_id","trackId","id", default="")
        if any(t.lower() in str(tname).lower() for t in TRACKS) or tid in TRACK_IDS:
            chosen.append((tname or tid, pick(m,"meet_id","id","meetId","slug",default=""), tid))
    return chosen, len(all_meets)

def meet_entries(meet_id):
    js = _get(f"/v1/north-america/meets/{meet_id}/entries") or {}
    for k in ("races","entries","data","meetRaces"):
        if isinstance(js.get(k), list): return js[k]
    if isinstance(js.get("meet"),dict):
        for k in ("races","entries"):
            if isinstance(js["meet"].get(k), list): return js["meet"][k]
    return []

# ---------- HTML ----------
CSS = """
<style>
:root{--bg:#0f2027;--fg:#e6f1f5;--muted:#87a0ab;--row:#122a33;--play:#34d399}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:24px}
h1{margin:0 0 8px;font-weight:800}.sub{color:var(--muted);margin:0 0 16px}
.badge{display:inline-block;padding:2px 6px;border:1px solid #2a4c58;border-radius:6px;color:#9fb9c4;font-size:12px;margin-left:8px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 8px;text-align:left}
th{color:#a3c0cb;border-bottom:1px solid #23424d;font-weight:600}tbody tr:nth-child(odd){background:var(--row)}
.right{text-align:right}.mono{font-variant-numeric:tabular-nums}.play{color:var(--play);font-weight:700}
</style>
"""

def build_html(rows, badges):
    today = date.today().isoformat()
    parts = ["<!doctype html><meta charset='utf-8'>", CSS]
    parts.append(f"<h1>Steve’s Horses Pro — {today}</h1>")
    parts.append("<div class='sub'>Tracks: " + ", ".join(TRACKS) + ". Data via The Racing API. "
                 f"<span class='badge'>Steve’s Horses Pro {VERSION}</span>"
                 "<span class='badge'>Min price to bet = fractional / $2‑payout / decimal</span>"
                 + "".join(f"<span class='badge'>{html.escape(b)}</span>" for b in badges)
                 + "</div>")
    parts.append("<div class='track'>Top Win Targets</div>")
    parts.append("<table><thead><tr>"
                 "<th>Track</th><th>Race</th><th>No.</th><th>Horse</th>"
                 "<th class='right'>Model Win%</th><th class='right'>Min price to bet (frac / $2 / dec)</th>"
                 "</tr></thead><tbody>")
    for r in rows[:TOP_N]:
        dec = r["min_dec"]; frac_s = frac(dec); pay2 = f\"${dec*2:0.2f}\"
        parts.append(\"<tr>\"
            f\"<td>{html.escape(r['track'])}</td>\"
            f\"<td>{html.escape(str(r['race']))}</td>\"
            f\"<td>{html.escape(str(r['no']))}</td>\"
            f\"<td class='play'>{html.escape(r['horse'])}</td>\"
            f\"<td class='right mono'>{r['p']*100:0.1f}%</td>\"
            f\"<td class='right mono'>{frac_s} / {pay2} / {dec:0.2f}d</td>\"
            \"</tr>\")
    parts.append("</tbody></table>")
    out = OUT / f"{today}_horses_pro.html"
    out.write_text("".join(parts), encoding="utf-8")
    print(out)
    return out

# ---------- Main ----------
def main():
    chosen, total = todays_meets()
    if not chosen:
        return build_html([], ["No meets for selected tracks or credentials missing."])
    rows = []
    counts = {}
    for tname, mid, tid in chosen:
        races = meet_entries(mid)
        counts[tname or tid or "?"] = len(races)
        for idx, r in enumerate(races, 1):
            field = (r.get("entries") or r.get("runners") or r.get("horses") or r.get("field") or [])
            if not isinstance(field, list) or not field: continue
            ps, ml_imps = probs(field)
            rn = race_no(r, idx_fallback=idx)

            # Select one or two top targets per race:
            maxp = max(ps)
            tie_indices = [i for i,p in enumerate(ps) if abs(p - maxp) < 1e-9]
            if len(tie_indices) >= 2:
                # take at most two: prefer higher ML implied prob, then name
                def tie_key(i):
                    ml = ml_imps[i] or 0.0
                    nm = horse_name(field[i]).lower()
                    return (-ml, nm)
                best_two = sorted(tie_indices, key=tie_key)[:2]
                picks = best_two
            else:
                picks = [max(range(len(ps)), key=lambda i: ps[i])]

            for i in picks:
                e = field[i]
                horse = horse_name(e)
                p = ps[i]
                fair = 1.0/max(1e-6,p)
                min_dec = fair*(1.0+MIN_PRICE_PAD)
                rows.append({
                    "track": tname or tid or "?",
                    "race": rn,
                    "no": program_no(e),
                    "horse": horse,
                    "p": p,
                    "min_dec": min_dec
                })

    rows.sort(key=lambda x: (str(x["track"]),
                             int(x["race"]) if str(x["race"]).isdigit() else 999,
                             -x["p"], x["horse"]))
    chip = "NA/meets HTTP 200 · chosen=" + ", ".join(t for t,_,_ in chosen)
    chip_counts = " · " + " · ".join(f"{k}: races={v}" for k,v in sorted(counts.items()))
    return build_html(rows, [chip + chip_counts])

if __name__ == "__main__":
    main()
