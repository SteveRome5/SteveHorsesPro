#!/usr/bin/env python3
# coding: utf-8
import os, json, re, math, statistics, datetime as dt
from pathlib import Path

BASE = Path(os.environ.get("BASE", str(Path.home()/ "Desktop" / "SteveHorsesPro")))
DATA = BASE / "data"
SIG1 = DATA / "signals"       # new layout: Track|YYYY-MM-DD.json
SIG2 = BASE / "signals"       # legacy layout: YYYY-MM-DD__Track.json
OUT  = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MAJORS = [s.strip() for s in os.environ.get("MAJOR_TRACKS_ONLY","").split(",") if s.strip()]

def _z(a):
    try:
        m,s = statistics.fmean(a), statistics.pstdev(a)
        return [(x-m)/(s if s>1e-9 else 1.0) for x in a]
    except Exception:
        return [0.0 for _ in a]

def _softmax(z, temp=0.66):
    if not z: return []
    m = max(z); x = [(v-m)/max(1e-6,temp) for v in z]
    ex = [math.exp(v) for v in x]
    s = sum(ex) or 1.0
    return [v/s for v in ex]

def fair_and_min_from_prob(p):
    """Return 'fair' and conservative 'min' money strings from a prob."""
    if p <= 0: return ("—","—")
    fair = f"${(1.0/p):.2f}"
    # 80% of fair as a conservative min (then rounded to 2 decimals)
    minp = max(0.01, 0.8/p)
    min_ = f"${(1.0/minp):.2f}"
    return fair, min_

def parse_hints(why):
    """Extract simple numeric hints from a 'why' string."""
    # examples seen: "SpeedForm ↑ (94 pct), ClassΔ ↑ (94 pct), Bias ↑ (94 pct)"
    sf = cls = bias = 0.0
    if not why: return (sf,cls,bias)
    m = re.findall(r"(SpeedForm|Class.?Δ|ClassΔ|ClassA|Bias)[^0-9\-+]*([\-+]?\d+(\.\d+)?)", why, flags=re.I)
    for k,v,_ in m:
        v = float(v)
        key = k.lower()
        if "speed" in key: sf  = v
        elif "class" in key: cls = v
        elif "bias" in key: bias = v
    return (sf,cls,bias)

def independent_p_from_hints(rows):
    """Compute non-flat p when TRAIN p is absent, from 'why' hints."""
    # build a score per runner from parsed hints, z-score, then softmax
    triples = [parse_hints(r.get("why","")) for r in rows]
    if not any(any(t) for t in triples):
        # no hints at all -> tiny geometric spread
        z = [0.05*i for i in range(len(rows))]
        return _softmax(z, temp=0.60)
    sf  = _z([t[0] for t in triples])
    cls = _z([t[1] for t in triples])
    bs  = _z([t[2] for t in triples])
    score = [0.6*sf[i] + 0.25*cls[i] + 0.15*bs[i] for i in range(len(rows))]
    return _softmax(score, temp=0.60)

def load_signals_for_day(day_iso):
    files = []
    # new
    for p in (SIG1).glob(f"*|{day_iso}.json"):
        files.append(p)
    # legacy
    for p in (SIG2).glob(f"{day_iso}__*.json"):
        if p not in files: files.append(p)
    out = []
    for p in files:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            track = p.name.split("|")[0] if "|" in p.name else p.stem.split("__",1)[1]
            out.append((track, raw))
        except Exception:
            pass
    return out

def latest_day_available():
    days = set()
    for p in (SIG1).glob("*|*.json"):
        try: days.add(p.name.split("|")[-1].split(".json")[0])
        except: pass
    for p in (SIG2).glob("*__*.json"):
        try: days.add(p.stem.split("__")[0])
        except: pass
    return sorted(days)[-1] if days else None

def build_meets(day_iso):
    meets = []
    for track, raw in load_signals_for_day(day_iso):
        if MAJORS and not any(track.startswith(m) for m in MAJORS):
            continue
        rows = [r for r in raw if isinstance(r, dict)]
        # group by race -> list of rows
        races = {}
        for r in rows:
            rno = str(r.get("race") or r.get("r") or "").strip()
            if not rno: continue
            races.setdefault(rno, []).append(r)
        if not races: continue
        # attach computed p and market placeholder
        meet = {"track": track, "day": day_iso, "races": []}
        for rno in sorted(races, key=lambda x: int(re.findall(r"\d+",x)[0])):
            rr = races[rno]
            # TRAIN p if present
            has_train = any(isinstance(r.get("p"), (int,float)) and r.get("p")>0 for r in rr)
            if has_train:
                ps = [max(1e-6, float(r.get("p") or 0.0)) for r in rr]
                s  = sum(ps) or 1.0
                p  = [v/s for v in ps]
            else:
                p = independent_p_from_hints(rr)
            # produce display rows
            disp = []
            for i, r in enumerate(rr):
                horse  = r.get("horse") or r.get("name") or r.get("h") or "—"
                pgm    = r.get("program") or r.get("pgm") or r.get("num") or "—"
                why    = r.get("why") or ""
                fair, minp = fair_and_min_from_prob(p[i])
                disp.append({
                    "pgm": pgm, "horse": horse, "p": p[i], "why": why,
                    "market": None, "fair": fair, "min": minp,
                    "source": "PRO+TRAIN" if has_train else "PRO",
                })
            # sort within race by probability
            disp.sort(key=lambda x: -x["p"])
            meet["races"].append({"rno": rno, "rows": disp})
        if meet["races"]:
            meets.append(meet)
    return meets

def html(meets, day_iso):
    CSS = """<style>
    body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px}
    table{border-collapse:collapse;width:100%;margin:16px 0}
    th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:14px}
    th{background:#f3f3f3}.mono{font-variant-numeric:tabular-nums}
    .badge{display:inline-block;padding:1px 6px;border-radius:6px;background:#eef;border:1px solid #ccd;font-size:12px}
    </style>"""
    out = ["<!doctype html><meta charset='utf-8'><title>PF-35 Mach++ v4 — PRO</title>", CSS]
    out.append(f"<h1>PF-35 Mach++ v4 <span style='font-size:12px;color:#666'>({day_iso})</span></h1>")
    out.append("<p style='color:#666;font-size:12px'>Built from local TRAIN signals. API/ML temporarily disabled.</p>")
    for m in meets:
        out.append(f"<h2>{m['track']}</h2>")
        for race in m["races"]:
            out.append(f"<h3>{m['track']} — Race {race['rno']}</h3>")
            out.append("<table><thead><tr><th>#</th><th>Horse</th><th>Win% (Final)</th><th>Market%</th><th>Edge</th><th>Fair</th><th>Min Price</th><th>Source</th></tr></thead><tbody>")
            for idx,row in enumerate(race["rows"],1):
                edge = "—"  # will be p - market once ML is wired
                mkt  = "—"
                out.append(
                    f"<tr><td class='mono'>{idx}</td>"
                    f"<td>{row['horse']}<br><span style='color:#666;font-size:12px'>{row['why']}</span></td>"
                    f"<td class='mono'>{row['p']*100:0.2f}%</td>"
                    f"<td class='mono'>{mkt}</td>"
                    f"<td class='mono'>{edge}</td>"
                    f"<td class='mono'>{row['fair']}</td>"
                    f"<td class='mono'>{row['min']}</td>"
                    f"<td><span class='badge'>{row['source']}</span></td></tr>"
                )
            out.append("</tbody></table>")
    return "\n".join(out)

def main():
    day = os.environ.get("DAY") or latest_day_available()
    if not day:
        print("No local signals found.")
        return
    meets = build_meets(day)
    out = OUT / f"{day}_horses_targets+full.html"
    out.write_text(html(meets, day), encoding="utf-8")
    print(str(out))

if __name__ == "__main__":
    main()