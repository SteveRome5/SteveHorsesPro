from __future__ import annotations
import os, ssl, json, math, re, html, base64, hashlib, webbrowser
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from datetime import date, datetime

# ---------------- Config ----------------
TRACKS = [
    "Saratoga", "Del Mar", "Santa Anita Park",
    "Gulfstream Park", "Keeneland", "Parx Racing", "Finger Lakes"
]
TRACK_IDS = {"SAR","DMR","SA","GP","KEE","PRX","FL"}

TOP_N         = int(os.getenv("TOP_N","30"))                 # rows in Top Win Targets
OUTPUT_MODE   = os.getenv("OUTPUT_MODE","targets").strip()   # 'targets' or 'full'
MIN_PRICE_PAD = float(os.getenv("MIN_PRICE_PAD","0.15"))     # 15% above fair
KELLY_CAP     = float(os.getenv("KELLY_CAP","0.25"))

RACING_USER = os.getenv("RACING_API_USER","").strip()
RACING_PASS = os.getenv("RACING_API_PASS","").strip()
USE_API     = bool(RACING_USER and RACING_PASS)

BASE_URL = "https://api.theracingapi.com"

# -------------- HTTP --------------------
CTX = ssl.create_default_context()
def _http(url: str):
    req = Request(url, headers={"User-Agent":"Mozilla/5.0"})
    if USE_API:
        tok = base64.b64encode(f"{RACING_USER}:{RACING_PASS}".encode()).decode()
        req.add_header("Authorization","Basic "+tok)
    with urlopen(req, timeout=30, context=CTX) as r:
        return json.loads(r.read().decode("utf-8"))

def na_meets(day: str):
    url = f"{BASE_URL}/v1/north-america/meets?" + urlencode({"start_date":day,"end_date":day})
    js = _http(url)
    return js.get("meets", []) if isinstance(js, dict) else (js or [])

def na_entries(meet_id: str):
    url = f"{BASE_URL}/v1/north-america/meets/{meet_id}/entries"
    return _http(url) or {}

# -------------- Odds utils --------------
def parse_odds_to_decimal(text):
    if text is None: return (None,None)
    t = str(text).strip().lower()
    if not t: return (None,None)
    if t in ("evs","even","evens"): return (2.0, 0.5)
    m = re.fullmatch(r'(\d+)\s*[/\-:]\s*(\d+)', t)  # 5/2, 3-1
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den>0:
            dec = 1.0 + num/den
            return (dec, 1.0/dec)
    m = re.fullmatch(r'[+-]?\d+', t)               # American
    if m:
        a = int(m.group(0))
        dec = (1 + a/100.0) if a>0 else (1 + 100.0/abs(a))
        return (dec, 1.0/dec)
    try:
        dec = float(t)
        if dec>1.0: return (dec, 1.0/dec)
    except: pass
    return (None,None)

def dec_to_frac(dec):
    if not dec or dec<=1.0: return "—"
    v = dec - 1.0
    # if near an integer, show N-1 (so 8.05 -> 7-1)
    n = round(v)
    if abs(v - n) < 0.25:
        return f"{int(n)}-1"
    best = (1e9, "—")
    for den in (2,3,4,5,6,7,8,9,10,12,14,16):
        num = round(v*den)
        err = abs(v - num/den)
        if err < best[0]:
            best = (err, f"{int(num)}-{int(den)}")
    return best[1]

def two_dollar_payout(dec):
    return f"${(2.0*dec):.2f}"

def kelly_fraction(p, dec_odds):
    if not dec_odds or dec_odds<=1.0 or p<=0.0 or p>=1.0: return 0.0
    b = dec_odds - 1.0
    q = 1.0 - p
    k = (b*p - q)/b
    return max(0.0, min(KELLY_CAP, k))

# -------------- Modeling ----------------
def _deterministic_bump(horse_name: str) -> float:
    # Stable tiny nudge so ties don't flicker across runs
    h = hashlib.md5(horse_name.encode("utf-8")).hexdigest()
    v = int(h[:8], 16)/0xffffffff  # 0..1
    return (v - 0.5) * 0.04        # ±2%

def _pos_bias(post: int, field_size: int) -> float:
    if not post or post<1 or field_size<2: return 0.0
    rel = 1.0 - (post-1)/(field_size-1)          # 1.0 at rail, 0.0 outside
    return (rel - 0.5) * 0.06                    # ±3%

def model_probs(runners, field_size):
    # Build unnormalized weights anchored on ML when present
    weights = []
    for r in runners:
        name = r.get("name") or "?"
        dec_ml, p_ml = parse_odds_to_decimal(r.get("ml") or r.get("morning_line") or r.get("odds"))
        base = p_ml if p_ml else (1.0/field_size)
        w = base * (1.0 + _deterministic_bump(name) + _pos_bias(r.get("post",0), field_size))
        weights.append(max(w, 1e-6))
    s = sum(weights) or 1.0
    return [w/s for w in weights]

# -------------- Data normalize ----------
def get_cards_today():
    day = date.today().isoformat()
    meets = na_meets(day) if USE_API else []
    chosen = []
    for m in meets:
        track = m.get("track_name") or m.get("course") or m.get("track") or ""
        tid   = (m.get("track_id") or "").upper()
        if any(t.lower() in track.lower() for t in TRACKS) or tid in TRACK_IDS:
            chosen.append((track or tid, m.get("meet_id")))
    cards = []
    for track, mid in chosen:
        js = na_entries(mid) or {}
        races = js.get("races") or js.get("entries") or []
        for r in races:
            rno = r.get("race_number") or r.get("race") or r.get("number") or "?"
            post = r.get("post_time") or r.get("off") or r.get("race_time") or ""
            raw = r.get("runners") or r.get("entries") or r.get("horses") or []
            runners = []
            for rr in raw:
                runners.append({
                    "name": rr.get("horse") or rr.get("name") or rr.get("horse_name"),
                    "ml": rr.get("morning_line") or rr.get("ml") or rr.get("odds"),
                    "post": rr.get("post_position") or rr.get("draw") or rr.get("stall") or rr.get("post"),
                })
            cards.append({
                "track": track, "race": rno, "post": post, "runners": runners
            })
    return cards

# -------------- HTML --------------------
CSS = """
<style>
  :root{--bg:#0f2027;--fg:#e6f1f5;--muted:#87a0ab;--row:#122a33;--play:#34d399;--pass:#9ba6b2}
  body{background:var(--bg);color:var(--fg);font-family:-apple-system,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:24px}
  h1{margin:0 0 8px;font-weight:800}
  .sub{color:var(--muted);margin:0 0 16px}
  .badge{display:inline-block;padding:2px 6px;border:1px solid #2a4c58;border-radius:6px;color:#9fb9c4;font-size:12px;margin-left:8px}
  .track{margin-top:18px;font-weight:700;color:#bfe4f2}
  .race{margin:12px 0 6px;color:#a3c0cb}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th,td{padding:10px 8px;text-align:left}
  th{color:#a3c0cb;border-bottom:1px solid #23424d;font-weight:600}
  tbody tr:nth-child(odd){background:var(--row)}
  .right{text-align:right}
  .mono{font-variant-numeric:tabular-nums}
  .play{color:var(--play);font-weight:700}
  .pass{color:var(--pass)}
</style>
"""

def build_html(cards):
    today = date.today().isoformat()
    parts = []
    parts.append("<!doctype html><meta charset='utf-8'><title>Steve’s Horses Pro — "+today+"</title>")
    parts.append(CSS)
    parts.append("<h1>Steve’s Horses Pro — "+today+"</h1>")
    parts.append("<div class='sub'>Tracks: "+", ".join(TRACKS)+". Data via The Racing API."
                 " <span class='badge'>Min price to bet = fractional / $2 payout / decimal</span></div>")

    # ------------ Aggregate for targets ------------
    targets = []
    for c in cards:
        runners = c["runners"]
        if not runners: continue
        probs = model_probs(runners, len(runners))
        for r, p in zip(runners, probs):
            fair = 1.0/p
            minp = fair*(1.0+MIN_PRICE_PAD)
            targets.append({
                "track": c["track"], "race": c["race"], "horse": r.get("name") or "?",
                "p": p, "min_dec": minp
            })
    targets.sort(key=lambda x: (-x["p"], x["track"], str(x["race"]), x["horse"]))
    parts.append("<div class='track'>Top Win Targets</div>")
    parts.append("<table><thead><tr><th>Track</th><th>Race</th><th>Horse</th>"
                 "<th class='right'>Model Win%</th><th class='right'>Min price to bet (frac / $2 / dec)</th>"
                 "</tr></thead><tbody>")
    for row in targets[:TOP_N]:
        frac = dec_to_frac(row["min_dec"])
        pay  = two_dollar_payout(row["min_dec"])
        parts.append("<tr>"
                     f"<td>{html.escape(str(row['track']))}</td>"
                     f"<td>{html.escape(str(row['race']))}</td>"
                     f"<td class='play'>{html.escape(str(row['horse']))}</td>"
                     f"<td class='right mono'>{row['p']*100:.1f}%</td>"
                     f"<td class='right mono'>{frac} / {pay} / {row['min_dec']:.2f}d</td>"
                     "</tr>")
    parts.append("</tbody></table>")

    if OUTPUT_MODE == "targets":
        out = (Path.home()/ "Desktop" / "SteveHorsesPro" / "outputs" / f"{today}_horses_pro.html")
        out.write_text("".join(parts), encoding="utf-8")
        return out

    # ------------ Per-race tables ------------
    # Group by track then race number
    by_track = {}
    for c in cards:
        by_track.setdefault(c["track"], []).append(c)
    for trk, races in by_track.items():
        parts.append(f"<div class='track'>{html.escape(trk)}</div>")
        # sort numerically when possible
        def _key(x):
            try: return int(x.get("race") or 0)
            except: return 999
        for c in sorted(races, key=_key):
            parts.append(f"<div class='race'>Race {html.escape(str(c.get('race')))} · Post {html.escape(str(c.get('post') or ''))}</div>")
            runners = c["runners"]
            probs = model_probs(runners, len(runners)) if runners else []
            parts.append("<table><thead><tr>"
                         "<th>Horse</th><th class='right'>ML/odds</th><th class='right'>Model Win%</th>"
                         "<th class='right'>Fair (dec)</th><th class='right'>Min (dec)</th>"
                         "<th class='right'>Kelly%</th><th>Play</th></tr></thead><tbody>")
            for r, p in zip(runners, probs):
                dec_ml, p_ml = parse_odds_to_decimal(r.get("ml"))
                fair = 1.0/p
                minp = fair*(1.0+MIN_PRICE_PAD)
                kelly = kelly_fraction(p, dec_ml) if dec_ml else 0.0
                play  = ""
                if dec_ml and dec_ml >= minp:
                    play = "VALUE: ML ≥ min"
                parts.append("<tr>"
                             f"<td>{html.escape(r.get('name') or '?')}</td>"
                             f"<td class='right mono'>{html.escape(str(r.get('ml') or '—'))}</td>"
                             f"<td class='right mono'>{p*100:.1f}%</td>"
                             f"<td class='right mono'>{fair:.2f}d</td>"
                             f"<td class='right mono'>{minp:.2f}d</td>"
                             f"<td class='right mono'>{(kelly*100):.1f}%</td>"
                             f"<td class='{'play' if play else 'pass'}'>{play or 'PASS'}</td>"
                             "</tr>")
            parts.append("</tbody></table>")

    out = (Path.home()/ "Desktop" / "SteveHorsesPro" / "outputs" / f"{today}_horses_pro.html")
    out.write_text("".join(parts), encoding="utf-8")
    return out

if __name__ == "__main__":
    cards = get_cards_today()
    out = build_html(cards)
    try:
        webbrowser.open(f"file://{out}")
    except Exception:
        pass
    print(out)
