#!/usr/bin/env python3
# no_history_today.py — write a CSV of today's runners with no harvested history
import os, sys, csv, json, base64, urllib.request, urllib.parse, datetime
from pathlib import Path

BASE   = Path.home() / "Desktop" / "SteveHorsesPro"
DATA   = BASE / "data" / "harvest"
OUTDIR = BASE / "signals"
OUTDIR.mkdir(parents=True, exist_ok=True)

API_BASE = os.getenv("RACING_API_BASE", "https://api.theracingapi.com")
USER = os.getenv("RACINGAPI_USER")
PASS = os.getenv("RACINGAPI_PASS")

def get(path):
    if not USER or not PASS:
        raise RuntimeError("RACINGAPI_USER/PASS not set")
    url = API_BASE + path
    req = urllib.request.Request(url, headers={"User-Agent": "stevehorses/1.0"})
    tok = base64.b64encode(f"{USER}:{PASS}".encode()).decode()
    req.add_header("Authorization", "Basic " + tok)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))

def today_iso():
    return datetime.date.today().isoformat()

def harvest_history_names(days_back=120):
    names = set()
    if not DATA.exists():
        return names
    cutoff = datetime.date.today() - datetime.timedelta(days=days_back)
    for p in DATA.glob("*.csv"):
        try:
            # filename like 2025-09-10.csv
            d = datetime.date.fromisoformat(p.stem)
            if d < cutoff: 
                continue
        except Exception:
            pass
        try:
            with p.open(newline="", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    # be flexible about schema
                    for k in ("horse","runner","name","runner_name","Horse"):
                        if k in row and row[k]:
                            names.add(str(row[k]).strip())
                            break
        except Exception:
            continue
    return names

def fetch_todays_runners():
    runners = []  # list of (track, race_no, program, name)
    meets = get("/v1/north-america/meets")
    meet_list = meets.get("meets", [])
    for m in meet_list:
        track = m.get("track_name") or m.get("track") or m.get("name") or "Track"
        mid   = m.get("meet_id") or m.get("id") or m.get("meetId")
        if not mid:
            continue
        try:
            entries = get(f"/v1/north-america/meets/{mid}/entries")
        except Exception:
            continue
        races = entries.get("races") or entries.get("entries") or []
        for idx, r in enumerate(races, 1):
            rno = r.get("race_number") or r.get("race") or r.get("number") or idx
            rno = int("".join(c for c in str(rno) if c.isdigit()) or "0") or idx
            rs = r.get("runners") or r.get("entries") or r.get("horses") or []
            for rr in rs:
                prog = rr.get("program") or rr.get("program_number") or rr.get("number") or ""
                name = rr.get("name") or rr.get("horse") or rr.get("runner_name") or ""
                if name:
                    runners.append((track, rno, str(prog).strip(), str(name).strip()))
    return runners

def main():
    iso = today_iso()
    history = harvest_history_names(days_back=365)  # look back a full year for safety
    todays = fetch_todays_runners()

    missing = [(t, r, p, n) for (t, r, p, n) in todays if n not in history]
    out = OUTDIR / f"{iso}_no_history.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track","race","program","#name"])
        for t, r, p, n in sorted(missing, key=lambda x: (x[0].lower(), int(x[1]), x[3].lower())):
            w.writerow([t, r, p, n])

    print(f"[no_history] today runners: {len(todays)}  |  in history: {len(todays)-len(missing)}  |  missing: {len(missing)}")
    print(f"[no_history] written -> {out}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[no_history] ERROR: {e}", file=sys.stderr)
        sys.exit(1)