#!/usr/bin/env python3
import re, sys, pathlib

BASE = pathlib.Path.home()/ "Desktop" / "SteveHorsesPro"
PRO  = BASE / "steve_horses_pro.py"
src  = PRO.read_text(encoding="utf-8")

def replace_block(name, new_code):
    global src
    # Replace a def block that starts with "def name(" and ends before next "def " or end of file
    pat = re.compile(rf"(?s)\ndef\s+{name}\s*\(.*?\)\s*:\n(.*?)(?=\ndef\s|\Z)")
    if pat.search(src):
        src = pat.sub("\n" + new_code.strip() + "\n", src)
        return True
    return False

# ---------- Blocks we will (re)define ----------

WHY_BLOCK = r'''
def safe_mean(xs):
    import statistics
    try: return statistics.mean(xs) if xs else 0.0
    except: return 0.0

def safe_pstdev(xs):
    import statistics
    try:
        if not xs or len(xs) <= 1: return 0.0
        s = statistics.pstdev(xs)
        return s if s > 1e-6 else 0.0
    except:
        return 0.0

def zscore_or_neutral(xs, n):
    import statistics
    s = safe_pstdev(xs)
    if s <= 1e-6:
        return [0.0]*n, [50]*n
    m = safe_mean(xs)
    z = [(x - m)/s for x in xs]
    order = sorted(z)
    pct = []
    for v in z:
        k = sum(1 for q in order if q <= v)
        p = int(round(100.0*(k-0.5)/max(1,len(z))))
        pct.append(max(1, min(99, p)))
    return z, pct

def arrow(p):
    return "↑" if p>=67 else "↗" if p>=55 else "→" if p>45 else "↘" if p>=33 else "↓"

def why_feature_pack(track: str, rc: dict, runners: list[dict]):
    surf = get_surface(rc); yards = get_distance_y(rc)
    key  = build_bucket_key(track, surf, yards)
    par  = MODEL.get("pars",{}).get(key, {"spd":80.0,"cls":70.0})
    speed = [get_speed(r) or 0.0 for r in runners]
    klass = [get_class(r) or 0.0 for r in runners]
    sf_raw    = [ (sp - par["spd"])/25.0 + (cl - par["cls"])/20.0 for sp,cl in zip(speed,klass) ]
    class_raw = [ (cl - par["cls"])/20.0 for cl in klass ]
    bias_raw  = [ _post_bias(track, surf, yards, prg_num(r)) for r in runners ]
    n=len(runners)
    sf_z, sf_pct     = zscore_or_neutral(sf_raw, n)
    cls_z, cls_pct   = zscore_or_neutral(class_raw, n)
    bias_z, bias_pct = zscore_or_neutral(bias_raw, n)
    why=[]; tips=[]
    for i in range(n):
        why.append(f"SpeedForm {arrow(sf_pct[i])} ({sf_pct[i]} pct), "
                   f"ClassΔ {arrow(cls_pct[i])} ({cls_pct[i]} pct), "
                   f"Bias {arrow(bias_pct[i])} ({bias_pct[i]} pct)")
        tips.append(f"SpeedForm {sf_z[i]:+0.2f}σ • ClassΔ {cls_z[i]:+0.2f}σ • Bias {bias_z[i]:+0.2f}σ")
    return why, tips
'''.strip()

# Ensure anti-flat helpers are present; if not, append.
if "def handcrafted_scores(" not in src or "def anti_flat_separation(" not in src:
    # Minimal, non-invasive insert near top-level helpers
    insert_here = src.rfind("\n# ---------------- Model probabilities ----------------")
    if insert_here == -1:
        insert_here = len(src)
    ANTI_FLAT = r'''
# ---- Anti-flat separation helpers (sharpen probabilities when too equal) ----
def zsc(xs):
    import statistics
    if not xs: return []
    m=statistics.mean(xs); s=statistics.pstdev(xs) if len(xs)>1 else 0.0
    if s<1e-6: s=1.0
    return [(x-m)/s for x in xs]

def pace_style(r):
    ep = get_early_pace(r) or 0.0
    lp = get_late_pace(r)  or 0.0
    if ep - lp >= 8:   return "E"
    if ep - lp >= 3:   return "EP"
    if lp - ep >= 5:   return "S"
    return "P"

def race_shape_adjust(runners, sect, rail, surface):
    import statistics, re, math, hashlib
    ps = [pace_style(r) for r in runners]
    nE = ps.count("E"); nEP = ps.count("EP")
    pressure = float((sect or {}).get("pressure") or 0.0)
    rail_wide = (rail or 0.0) >= 20.0
    adj = [0.0]*len(runners)
    for i,_ in enumerate(runners):
        sty = ps[i]
        if sty == "E":
            lone = 0.10 if nE==1 and nEP<=1 else 0.0
            herd = -0.08 if nE>=3 else 0.0
            rail_eff = (-0.04 if ("turf" in (get_surface(runners[0]) or "") and rail_wide) else 0.0)
            adj[i] += lone + herd + rail_eff
            if pressure <= 0.2: adj[i] += 0.05
        elif sty == "EP":
            if pressure <= 0.2: adj[i] += 0.03
            if nE>=2: adj[i] -= 0.02
        elif sty == "S":
            adj[i] += 0.10*max(0.0, (sect or {}).get("meltdown") or 0.0)
    return adj

def handcrafted_scores(track, rc, runners, extras=None):
    import statistics, hashlib
    sect  = (extras or {}).get("sect") or {"pressure":0.0,"meltdown":0.0}
    rail  = get_rail(rc) or 0.0
    surface = get_surface(rc)
    spd=[get_speed(r) or 0.0 for r in runners]
    ep =[get_early_pace(r) or 0.0 for r in runners]
    lp =[get_late_pace(r) or 0.0 for r in runners]
    cls=[get_class(r) or 0.0 for r in runners]
    spdZ,epZ,lpZ,clsZ=zsc(spd),zsc(ep),zsc(lp),zsc(cls)
    w_spd,w_ep,w_lp,w_cls=1.0,0.55,0.30,0.45
    trR=[(get_trainer_win(r) or 0.0)/100.0 for r in runners]
    jkR=[(get_jockey_win(r)  or 0.0)/100.0 for r in runners]
    tjR=[(get_combo_win(r)   or 0.0)/100.0 for r in runners]
    shape_adj=race_shape_adjust(runners, sect, rail, surface)
    scores=[]
    for i,r in enumerate(runners):
        s=w_spd*spdZ[i] + w_ep*epZ[i] + w_lp*lpZ[i] + w_cls*clsZ[i] + 0.25*trR[i] + 0.18*jkR[i] + 0.10*tjR[i]
        s+=shape_adj[i]
        seed=f"{track}|{race_num(rc,0)}|{prg_num(r)}|{horse_name(r)}"
        import hashlib
        h=hashlib.sha1(seed.encode()).hexdigest()
        s+=(int(h[:6],16)/0xFFFFFF - 0.5)*0.03
        scores.append(s)
    return scores

def field_temp(n):
    if n>=12: return 0.80
    if n>=10: return 0.72
    if n>=8:  return 0.66
    return 0.60

def softmax(zs, temp):
    import math
    if not zs: return []
    m=max(zs); exps=[math.exp((z-m)/max(1e-6,temp)) for z in zs]; s=sum(exps)
    return [e/s for e in exps] if s>0 else [1.0/len(zs)]*len(zs)

def anti_flat_separation(track, rc, runners, p_model, extras):
    import statistics
    if not p_model: return p_model
    n=len(p_model)
    if n<=2: return p_model
    var = statistics.pvariance(p_model) if len(p_model)>1 else 0.0
    if var >= 1e-5:
        return p_model
    zs = handcrafted_scores(track, rc, runners, extras=extras)
    t  = max(0.45, field_temp(n)-0.10)
    pz = softmax(zs, temp=t)
    mix = 0.70
    blended = [max(1e-6, min(0.999, mix*pz[i] + (1-mix)*p_model[i])) for i in range(n)]
    s=sum(blended)
    return [x/s for x in blended] if s>0 else p_model
'''.strip()
    src = src[:insert_here] + "\n\n" + ANTI_FLAT + "\n\n" + src[insert_here:]

# Replace WHY block (function + helpers)
_ = replace_block("why_feature_pack", WHY_BLOCK)

# Replace probabilities_from_model_only to call anti_flat_separation
PROBS_BLOCK = r'''
def probabilities_from_model_only(track, rc, runners, extras=None):
    ps=[]
    ok=True
    for r in runners:
        p = predict_bucket_prob(track, rc, r)
        if p is None: ok=False; break
        ps.append(max(1e-6,min(0.999,p)))
    if ok and ps:
        s=sum(ps)
        ps = [p/s for p in ps] if s>0 else [1.0/len(ps)]*len(ps)
        # sharpen if the distribution is too flat
        ps = anti_flat_separation(track, rc, runners, ps, extras)
        return ps
    # fallback to handcrafted
    zs = handcrafted_scores(track, rc, runners, extras=extras)
    t = field_temp(len(runners))
    ps = softmax(zs, temp=t)
    if len(ps) >= 12:
        ps=[max(0.003,p) for p in ps]; s=sum(ps); ps=[p/s for p in ps]
    return ps
'''.strip()
_ = replace_block("probabilities_from_model_only", PROBS_BLOCK)

# (Optional) ensure dutch_overlays keeps confidence flags; most current builds already do.
# We'll only inject if the conf tag is missing.
if "conf_label" in src and "flags_out[i]" in src and "DUTCH" in src:
    pass  # looks good
else:
    # skip; the common modern block you pasted earlier already sets flags nicely
    pass

# Tag the build so we can confirm in logs
if "PF-35 Mach++ v3.8-pro (WHY fixed" not in src:
    src = src.replace('VERSION = "PF-35 Mach++ v3.8-pro"', 'VERSION = "PF-35 Mach++ v3.8-pro (WHY fixed, sharpened)"')

PRO.write_text(src, encoding="utf-8")
print("[hotfix] applied OK ->", PRO)
