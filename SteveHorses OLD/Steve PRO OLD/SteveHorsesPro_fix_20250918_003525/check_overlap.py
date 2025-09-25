#!/usr/bin/env python3
import os, csv, glob, json, base64, ssl
from pathlib import Path
from datetime import date, timedelta, datetime
from urllib.request import Request, urlopen

BASE = Path.home() / "Desktop" / "SteveHorsesPro"
HIST_DIR = BASE / "history"

API_BASE = os.getenv("RACING_API_BASE", "https://api.theracingapi.com")
RUSER = os.environ["RACINGAPI_USER"]
RPASS = os.environ["RACINGAPI_PASS"]
CTX = ssl.create_default_context()

def _get(path, params=None):
    from urllib.parse import urlencode
    url = API_BASE + path + ("?" + urlencode(params) if params else "")
    req = Request(url, headers={"User-Agent":"Mozilla/5.0"})
    tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
    req.add_header("Authorization", "Basic " + tok)
    with urlopen(req, timeout=25, context=CTX) as r:
        return json.loads(r.read().decode("utf-8","replace"))

def today_entries_for_tracks(tracks_like):
    iso = date.today().isoformat()
    meets = _get("/v1/north-america/meets", {"start_date": iso, "end_date": iso}).get("meets", [])
    out = []
    for m in meets:
        tname = (m.get("track_name") or m.get("track") or m.get("name") or "").strip()
        tl = tname.lower()
        if any(k in tl for k in tracks_like):
            mid = m.get("meet_id") or m.get("id") or m.get("meetId")
            if not mid: continue
            entries = _get(f"/v1/north-america/meets/{mid}/entries").get("races", [])
            for rc in entries:
                runners = rc.get("runners") or rc.get("entries") or []
                for r in runners:
                    horse = (r.get("horse_name") or r.get("name") or r.get("runner_name") or "").strip()
                    if horse:
                        out.append((tname, horse))
    return out

def load_history_names(days_back=180):
    cutoff = date.today() - timedelta(days=days_back)
    names = set()
    files = sorted(glob.glob(str(HIST_DIR / "history_*.csv")))
    for f in files:
        try:
            ds = Path(f).stem.split("_",1)[1]
            d  = datetime.strptime(ds,"%Y-%m-%d").date()
        except Exception:
            d = None
        if d and d < cutoff: 
            continue
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                rdr = csv.DictReader(fh)
                for row in rdr:
                    nm = (row.get("horse") or "").strip()
                    if nm: names.add(nm)
        except Exception:
            continue
    return names

def main():
    # what we consider “Churchill/Belmont”
    keys = ["churchill", "belmont"]
    print("[check] fetching today’s entries for Churchill/Belmont…")
    entries = today_entries_for_tracks(keys)
    if not entries:
        print("No entries found today for Churchill/Belmont (or API returned none).")
        return
    hist = load_history_names(180)
    per_track = {}
    for track, horse in entries:
        per_track.setdefault(track, []).append(horse)
    print(f"[check] history horses loaded: {len(hist):,}")
    for track, horses in per_track.items():
        uniq = sorted(set(horses))
        have = [h for h in uniq if h in hist]
        miss = [h for h in uniq if h not in hist]
        print("\n—", track, "—")
        print(f"  today runners: {len(uniq)}  |  in history: {len(have)}  |  missing: {len(miss)}")
        if miss:
            print("  (no history found for):", ", ".join(miss[:12]) + (" …" if len(miss)>12 else ""))
    print("\nDone.")

if __name__ == "__main__":
    main()