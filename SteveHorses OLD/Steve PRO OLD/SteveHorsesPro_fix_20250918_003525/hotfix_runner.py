
#!/usr/bin/env python3
# Hotfix runner: robust entries/cards + safe race/meet-id handling (no dicts in URLs)
from datetime import date, datetime
import json, re, sys
from pathlib import Path

import steve_horses_pro as pro

# -------------- Full replacement: API helpers --------------
def _stripped(s):
    try: return str(s).strip()
    except: return ""

def _dig_id(obj):
    """
    Accepts: int/str -> returns clean str
             dict    -> tries common id keys recursively
    Returns "" if cannot find a simple id.
    """
    if obj is None: return ""
    if isinstance(obj, (int, float)):
        return _stripped(int(obj))
    if isinstance(obj, str):
        return _stripped(obj)
    if isinstance(obj, dict):
        # try common keys first
        for k in ("meet_id","id","meetId","meetID","uuid","uuid_str","meet_uid","uid","key","code"):
            v = obj.get(k)
            s = _dig_id(v)
            if s: return s
        # try any key that looks like an id
        for k, v in obj.items():
            ks = _stripped(k).lower()
            if "id" in ks or "uid" in ks or "uuid" in ks or "key" in ks:
                s = _dig_id(v)
                if s: return s
        return ""
    # unknown type
    return ""

# Replace fetch_entries with a robust version
def fetch_entries_robust(meet_id_obj):
    mid = _dig_id(meet_id_obj)
    if not mid:
        pro.log(f"[hotfix] Bad meet_id object -> {meet_id_obj!r}")
        return {"races":[]}
    path = pro.EP_ENTRIES_BY_MEET.format(meet_id=mid)
    try:
        return pro.safe_get(path, default={"races":[]}) or {"races":[]}
    except Exception as e:
        pro.log(f"[hotfix] entries fail mid={mid}: {e}")
        return {"races":[]}

# -------------- Full replacement: Cards + scratches --------------
def build_cards_hotfix(iso_date):
    """
    Same structure as your build_cards, but:
      * uses fetch_entries_robust
      * never creates bad URLs
      * keeps your majors filter intact
    """
    only_digits = lambda s: re.sub(r"\\D", "", s or "")
    meets = (pro.fetch_meets(iso_date) or {}).get("meets", []) or []
    cards = {}
    auto_lines = []

    for m in meets:
        track = pro.g(m,"track_name","track","name") or "Track"
        # respect your majors set
        if track not in pro.MAJOR_TRACKS:
            continue

        mid = pro.g(m,"meet_id","id","meetId") or m
        try:
            entries = fetch_entries_robust(mid)
            races = entries.get("races") or entries.get("entries") or []
            for r_idx, r in enumerate(races, 1):
                r["runners"] = r.get("runners") or r.get("entries") or r.get("horses") or r.get("starters") or []
                # mark API-scratched first
                for rr in r["runners"]:
                    if pro.is_scratched_runner(rr):
                        rr["scratched"] = True
                # normalize race number for template + logs
                rno_raw = pro.g(r,"race_number","race","number","raceNo") or r_idx
                try:
                    rno = int(re.sub(r"[^\\d]","", str(rno_raw)))
                except:
                    rno = r_idx

                scr_prog = [pro.prg_num(x) for x in r["runners"] if x.get("scratched")]
                scr_prog = [n for n in scr_prog if n]
                if scr_prog:
                    nums_sorted = sorted(scr_prog, key=lambda z: int(only_digits(z) or "0"))
                    auto_lines.append(f"{track}|{rno}|{', '.join(nums_sorted)}")

            if races:
                cards[track] = races
        except Exception as e:
            pro.log(f"[hotfix] Entries fetch failed for {track}: {e}")

    # write autoscrapes file (same as your code)
    if auto_lines:
        p = pro.IN_DIR / f"scratches_AUTO_{iso_date}.txt"
        try:
            p.write_text("# Auto-scratches\\n" + "\\n".join(auto_lines) + "\\n", encoding="utf-8")
        except Exception as e:
            pro.log(f"[hotfix] autoscr write fail: {e}")
    return cards, auto_lines

def build_cards_and_scratches_hotfix(iso_date):
    cards, auto_lines = build_cards_hotfix(iso_date)
    # keep your template + manual scratches flow
    pro.save_scratch_template(iso_date, cards)
    manual_scr = pro.load_manual_scratches(iso_date)

    # convert auto_lines -> map
    from collections import defaultdict
    auto_scr_map = defaultdict(lambda: defaultdict(set))
    for line in auto_lines:
        try:
            track, rno_s, progs = [x.strip() for x in line.split("|", 3)[:3]]
            rno = int(re.sub(r"[^\\d]","", rno_s))
            lst=[p.strip() for p in progs.split(",") if p.strip()]
            for pnum in lst:
                auto_scr_map[track][rno].add(pnum)
        except Exception:
            pass

    scr_summary, scr_details = pro.apply_scratches(cards, auto_scr_map, manual_scr)
    auto_summary={"auto_count": sum(len(x.split('|')[2].split(',')) for x in auto_lines) if auto_lines else 0}
    return cards, scr_summary, auto_summary, scr_details

def main():
    iso = date.today().isoformat()
    pro.log(f"[hotfix-run] starting at {datetime.now():%Y-%m-%d %H:%M:%S}")
    model_loaded = pro.load_model()
    pro.log(f"[hotfix-run] model loaded: {model_loaded}")

    # run with robust cards
    cards, scr_summary, auto_summary, scr_details = build_cards_and_scratches_hotfix(iso)
    try:
        n_tracks = len(cards)
        n_races = sum(len(v) for v in cards.values())
        pro.log(f"[hotfix-run] Tracks: {n_tracks}  Races: {n_races}")
        print("Tracks:", n_tracks, "Races:", n_races)
        print("Tracks list:", list(cards.keys()))
    except Exception:
        pass

    html_out = pro.build_report(cards, iso, scr_summary, auto_summary, scr_details)
    pro.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = pro.OUT_DIR / f"{iso}_horses_targets+full.html"
    out_path.write_text(html_out, encoding="utf-8")
    pro.log(f"[hotfix-run] wrote {out_path}")
    print("WROTE:", out_path)

if __name__ == "__main__":
    main()
