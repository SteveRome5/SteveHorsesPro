#!/usr/bin/env python3
import os, re, sys
BASE = os.path.expanduser("~/Desktop/SteveHorsesPro")
TARGET = os.path.join(BASE, "steve_horses_pro.py")
BACKUP = TARGET + ".bak_propatch"

def read(p):
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

def write(p, txt):
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)

def ensure_after(pattern, insert, txt, desc):
    """Insert `insert` right after the first line that matches `pattern` (regex).
       If already present, do nothing. Returns updated text."""
    if insert.strip() in txt:
        return txt
    m = re.search(pattern, txt)
    if not m:
        print(f"[patch] WARN: could not find anchor for {desc}; skipping that insert.")
        return txt
    idx = txt.find("\n", m.end())
    if idx == -1:
        idx = m.end()
    new = txt[:idx+1] + insert + ("\n" if not insert.endswith("\n") else "") + txt[idx+1:]
    print(f"[patch] inserted: {desc}")
    return new

def replace_function(header_re, new_body, txt, desc):
    """Replace a def ...(): function block that starts with header_re (regex).
       Assumes top-level function (starts with 'def')."""
    m = re.search(header_re, txt)
    if not m:
        print(f"[patch] ERROR: could not find {desc}.")
        sys.exit(1)
    start = m.start()
    # Find next top-level "def " after this function
    m2 = re.search(r"\ndef\s+\w+\(", txt[m.end():])
    end = len(txt) if not m2 else m.end() + m2.start()
    new_txt = txt[:start] + new_body + txt[end:]
    print(f"[patch] replaced: {desc}")
    return new_txt

def main():
    if not os.path.isfile(TARGET):
        print(f"[patch] ERROR: {TARGET} not found")
        sys.exit(1)

    src = read(TARGET)
    write(BACKUP, src)
    print(f"[patch] backup -> {BACKUP}")

    # 1) Add a tiny PRO switch right after imports (after 'from collections import defaultdict')
    pro_switch = (
        "PRO_ON = os.getenv('PRO_MODE', '0') == '1'\n"
        "# Confidence gates (only used when PRO_ON)\n"
        "CONF_THRESH_PRIME = 0.58\n"
        "CONF_THRESH_ACTION = 0.50\n"
        "RACE_SPEND_CAP_MULT = 1.00  # 1.00 = keep as-is; you can lower later\n"
    )
    src = ensure_after(r"from collections import defaultdict.*\n", pro_switch, src, "PRO switch & gates")

    # 2) Add compute_confidence helper right after kelly_damped
    if "def compute_confidence(" not in src:
        conf_fn = """\
def compute_confidence(p, dec, late_slope_max, odds_var_mean, minutes_to_post):
    \"\"\"Return (0..1) confidence from market stability & time context.
    No pool-size damp since user plays true track odds off-course.
    \"\"\"
    # Base: higher when market exists and minutes-to-post is not super early
    base = 0.55
    if dec and dec > 1:
        base += 0.10
    if minutes_to_post is not None:
        if minutes_to_post < 6:
            base += 0.08
        elif minutes_to_post < 12:
            base += 0.04
    # Stability dampers
    slope_pen = 0.12 * max(0.0, late_slope_max)         # big late drift penalized
    var_pen   = 0.04 * max(0.0, (odds_var_mean or 0.0)) # choppy odds penalized
    c = max(0.0, min(1.0, base - slope_pen - var_pen))
    # Very tiny boost for stronger p (so chalkier confident probs get a hair more)
    c = max(0.0, min(1.0, c + 0.10 * max(0.0, (p or 0.0) - 0.22)))
    return c
"""
        src = ensure_after(r"def kelly_damped\(", conf_fn, src, "compute_confidence()")

    # 3) Replace dutch_overlays with PRO-aware version
    new_dutch = """\
def dutch_overlays(enriched, bankroll, field_size, late_slope_max, odds_var_mean, m2p,
                   kelly_cap, max_per, min_stake, daily_room, flags_out):
    \"\"\"Win-bet allocator.
    - When PRO_ON is False: identical behavior to previous version.
    - When PRO_ON is True: applies confidence gates and scales Kelly by confidence.
    \"\"\"
    # Non-PRO path: preserve exact original logic
    if not PRO_ON:
        cand=[]
        for i,it in enumerate(enriched):
            p=it["p_final"]; dec=it["market"]; minp=it["minp"]
            ed = overlay_edge(p, dec) if dec else None
            it["edge"]=ed
            if not dec or dec<minp or not ed or ed<=0:
                continue
            if p < EDGE_WIN_PCT_FLOOR:
                continue
            imp = it.get("imp", None)
            edge_pp = (p - (imp or 0.0))*100.0 if imp is not None else None
            if edge_pp is None or edge_pp < EDGE_PP_MIN_PRIME:
                continue
            f = kelly_damped(p,dec,field_size,late_slope_max,odds_var_mean, m2p)
            if f<=0:
                continue
            w = (f ** 1.25) * max(0.01, ed)
            cand.append((i, f, w, p))
        if not cand:
            return []
        w_sum = sum(w for _,_,w,_ in cand)
        stakes=[]
        for i,f,w,p in cand:
            frac = (w / w_sum) * kelly_cap
            stake = bankroll * frac
            if stake>=min_stake:
                stakes.append((i, min(max_per, stake)))
        if not stakes:
            return []
        planned = sum(st for _,st in stakes)
        room = max(0.0, daily_room)
        capped = False
        if planned > room and room > 0:
            scale = room / planned
            scaled=[(i, st*scale) for i,st in stakes if st*scale >= min_stake]
            if scaled:
                stakes = scaled; capped = True
            else:
                top_i = max(cand, key=lambda t: t[3])[0]
                stakes = [(top_i, min(room, min_stake))]
                capped = True
        if capped:
            for i,_ in stakes:
                flags_out[i] = (flags_out.get(i,"") + (" CAP" if "CAP" not in flags_out.get(i,"") else "")).strip()
        if len(stakes) >= 2:
            for i,_ in stakes:
                flags_out[i] = (flags_out.get(i,"") + f" DUTCH{len(stakes)}").strip()
        return stakes

    # PRO path
    cand=[]
    for i,it in enumerate(enriched):
        p  = it["p_final"]
        dec= it["market"]
        minp=it["minp"]
        ed = overlay_edge(p, dec) if dec else None
        it["edge"]=ed

        if not dec or dec < minp or not ed or ed <= 0:
            continue

        # Confidence from market stability; minutes-to-post is m2p (race level)
        conf = compute_confidence(p, dec, late_slope_max, odds_var_mean, m2p)
        it["_conf"] = conf

        imp = it.get("imp", None)
        edge_pp = (p - (imp or 0.0))*100.0 if imp is not None else None

        # PRIME gate: higher prob & confidence
        prime_ok = (p >= EDGE_WIN_PCT_FLOOR) and (edge_pp is not None and edge_pp >= EDGE_PP_MIN_PRIME) and (conf >= CONF_THRESH_PRIME)
        # ACTION gate: looser prob/conf, still need positive edge_pp
        action_ok = (p >= ACTION_PCT_FLOOR) and (edge_pp is not None and edge_pp >= EDGE_PP_MIN_ACTION) and (conf >= CONF_THRESH_ACTION)

        if not (prime_ok or action_ok):
            continue

        f0 = kelly_damped(p, dec, field_size, late_slope_max, odds_var_mean, m2p)
        f  = f0 * conf  # scale Kelly by confidence
        if f <= 0:
            continue

        # Weight tilts prefer more confident + higher edge
        w = (f ** 1.25) * max(0.01, ed) * (0.80 + 0.40*conf)
        cand.append((i, f, w, p, prime_ok, action_ok))

    if not cand:
        return []

    # Build stakes with dutching
    w_sum = sum(w for _,_,w,_,_,_ in cand)
    stakes=[]
    for i,f,w,p,prime_ok,action_ok in cand:
        frac  = (w / w_sum) * kelly_cap
        stake = bankroll * frac
        if stake >= min_stake:
            stakes.append((i, min(max_per, stake)))

    if not stakes:
        return []

    # Race-level cap (multiplicative on the existing daily_room)
    planned = sum(st for _,st in stakes)
    room    = max(0.0, daily_room) * RACE_SPEND_CAP_MULT
    capped  = False
    if planned > room and room > 0:
        scale = room / planned
        scaled=[(i, st*scale) for i,st in stakes if st*scale >= min_stake]
        if scaled:
            stakes = scaled; capped = True
        else:
            # pick the most confident highest-p candidate
            top_i = max(cand, key=lambda t: (t[4], t[3], t[1]))[0]  # (prime_ok, p, f)
            stakes = [(top_i, min(room, min_stake))]
            capped = True

    # Flags
    if capped:
        for i,_ in stakes:
            flags_out[i] = (flags_out.get(i,"") + (" CAP" if "CAP" not in flags_out.get(i,"") else "")).strip()

    if len(stakes) >= 2:
        for i,_ in stakes:
            flags_out[i] = (flags_out.get(i,"") + f" DUTCH{len(stakes)}").strip()

    return stakes
"""
    src = replace_function(r"\ndef\s+dutch_overlays\(", new_dutch, src, "def dutch_overlays")

    write(TARGET, src)
    print("[patch] done. You can toggle the new behavior by setting PRO_MODE=1 in your launcher.")

if __name__ == "__main__":
    main()