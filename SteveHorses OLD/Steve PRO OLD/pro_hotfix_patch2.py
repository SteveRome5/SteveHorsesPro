#!/usr/bin/env python3
from pathlib import Path
import re, sys, hashlib

PRO = Path("steve_horses_pro.py")
src = PRO.read_text(encoding="utf-8")

def replace_block(func_name, new_code):
    global src
    pat = re.compile(rf"def {re.escape(func_name)}\s*\(.*?\)\s*:(?:\n(?:    .*\n?)*)", re.DOTALL)
    if not pat.search(src):
        print(f"[hotfix] FAIL: cannot find {func_name}")
        sys.exit(2)
    src = pat.sub(new_code, src, count=1)

# --- 1) zscore_or_neutral: return N/A instead of fake 50 pct when no variance ---
ZSCORE_BLOCK = r'''
def zscore_or_neutral(xs, n):
    # If inputs are missing/constant, return neutral z and None percentiles (signals N/A).
    def _pstdev_safe(vs):
        import statistics
        if not vs or len(vs) <= 1:
            return 0.0
        try:
            s = statistics.pstdev(vs)
        except Exception:
            s = 0.0
        return s if s > 1e-9 else 0.0
    s = _pstdev_safe(xs)
    if s <= 1e-9:
        return [0.0]*n, [None]*n
    import statistics
    m = statistics.mean(xs) if xs else 0.0
    z = [(x - m)/s for x in xs]
    order = sorted(z)
    pct = []
    for v in z:
        k = sum(1 for q in order if q <= v)
        p = int(round(100.0*(k-0.5)/max(1,len(z))))
        pct.append(max(1, min(99, p)))
    return z, pct
'''.strip()

# --- 2) why_feature_pack: render N/A cleanly when pct is None ---
WHY_BLOCK = r'''
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

    def fmt(name, p):
        if p is None: return f"{name} N/A"
        if p >= 67: a="↑"
        elif p >= 55: a="↗"
        elif p > 45: a="→"
        elif p >= 33: a="↘"
        else: a="↓"
        return f"{name} {a} ({p} pct)"

    why=[]; tips=[]
    for i in range(n):
        why.append(", ".join([
            fmt("SpeedForm", sf_pct[i]),
            fmt("ClassΔ", cls_pct[i]),
            fmt("Bias", bias_pct[i]),
        ]))
        tips.append(f"SpeedForm {sf_z[i]:+0.2f}σ • ClassΔ {cls_z[i]:+0.2f}σ • Bias {bias_z[i]:+0.2f}σ")
    return why, tips
'''.strip()

# --- 3) handcrafted_scores: build separation when core figs are missing ---
HANDCRAFT_BLOCK = r'''
def handcrafted_scores(track, rc, runners, extras=None):
    # Use figs if present; otherwise build separation from trainer/jockey/combo and tiny post bias + market prior.
    sect  = (extras or {}).get("sect") or {"pressure":0.0,"meltdown":0.0}
    rail  = get_rail(rc) or 0.0
    surface = get_surface(rc)
    spd=[get_speed(r) or 0.0 for r in runners]
    ep =[get_early_pace(r) or 0.0 for r in runners]
    lp =[get_late_pace(r) or 0.0 for r in runners]
    cls=[get_class(r) or 0.0 for r in runners]

    has_core = any(spd) or any(ep) or any(lp) or any(cls)

    def zsc(xs):
        import statistics
        if not xs: return []
        m=statistics.mean(xs)
        s=statistics.pstdev(xs) if len(xs)>1 else 0.0
        if s<1e-6: s=1.0
        return [(x-m)/s for x in xs]

    if has_core:
        spdZ,epZ,lpZ,clsZ=zsc(spd),zsc(ep),zsc(lp),zsc(cls)
        w_spd,w_ep,w_lp,w_cls=1.0,0.55,0.30,0.45
    else:
        # Core figs missing: base on trainer/jockey/combo %, plus tiny post bias and market prior.
        spdZ=[0.0]*len(runners); epZ=[0.0]*len(runners); lpZ=[0.0]*len(runners); clsZ=[0.0]*len(runners)
        w_spd,w_ep,w_lp,w_cls=0.0,0.0,0.0,0.0

    trR=[(get_trainer_win(r) or 0.0)/100.0 for r in runners]
    jkR=[(get_jockey_win(r)  or 0.0)/100.0 for r in runners]
    tjR=[(get_combo_win(r)   or 0.0)/100.0 for r in runners]

    # market prior (if live odds exist, use implied as additive prior)
    market_prior=[]
    for r in runners:
        dec = live_decimal(r)
        if dec and dec>1: market_prior.append(1.0/dec)
        else: market_prior.append(0.0)

    # race shape adj (still helps if figs exist)
    ps = [pace_style(r) for r in runners]
    nE = ps.count("E"); nEP = ps.count("EP")
    pressure = float((extras or {}).get("sect",{}).get("pressure") or 0.0)
    rail_wide = (rail or 0.0) >= 20.0
    shape_adj=[0.0]*len(runners)
    for i,_ in enumerate(runners):
        sty = ps[i]
        if sty == "E":
            lone = 0.10 if nE==1 and nEP<=1 else 0.0
            herd = -0.08 if nE>=3 else 0.0
            rail_eff = (-0.04 if ("turf" in surface and rail_wide) else 0.0)
            shape_adj[i] += lone + herd + rail_eff
            if pressure <= 0.2: shape_adj[i] += 0.05
        elif sty == "EP":
            if pressure <= 0.2: shape_adj[i] += 0.03
            if nE>=2: shape_adj[i] -= 0.02
        elif sty == "S":
            md = float((extras or {}).get("sect",{}).get("meltdown") or 0.0)
            shape_adj[i] += 0.10*max(0.0, md)

    # tiny post bias
    post_bias_arr=[]
    yards=get_distance_y(rc); surf=get_surface(rc)
    for r in runners:
        post_bias_arr.append(_post_bias(track, surf, yards, prg_num(r)))

    # combine
    scores=[]
    for i,r in enumerate(runners):
        s = (1.00*spdZ[i] + 0.55*epZ[i] + 0.30*lpZ[i] + 0.45*clsZ[i])
        s += 0.32*trR[i] + 0.24*jkR[i] + 0.12*tjR[i]
        s += 0.50*market_prior[i]
        s += 0.50*post_bias_arr[i]
        s += shape_adj[i]
        # tiny deterministic tiebreak
        seed=f"{track}|{(g(rc,'race_number','race','number','raceNo') or '')}|{prg_num(r)}|{horse_name(r)}"
        h=int(hashlib.sha1(seed.encode()).hexdigest()[:6],16)/0xFFFFFF
        s += (h-0.5)*0.04
        scores.append(s)
    return scores
'''.strip()

# --- 4) anti_flat + probabilities_from_model_only: trigger blend on tight ranges ---
ANTIFLAT_BLOCK = r'''
def anti_flat_separation(track, rc, runners, p_model, extras):
    if not p_model: return p_model
    n=len(p_model)
    if n<=2: return p_model
    rng = (max(p_model)-min(p_model)) if p_model else 0.0
    if rng >= 0.04:
        return p_model
    zs = handcrafted_scores(track, rc, runners, extras=extras)
    # sharper temperature
    def field_temp(n):
        if n>=12: return 0.80
        if n>=10: return 0.72
        if n>=8:  return 0.66
        return 0.60
    def softmax(zs, temp):
        if not zs: return []
        m=max(zs); exps=[__import__("math").exp((z-m)/max(1e-6,temp)) for z in zs]; s=sum(exps)
        return [e/s for e in exps] if s>0 else [1.0/len(zs)]*len(zs)
    t  = max(0.45, field_temp(n)-0.10)
    pz = softmax(zs, temp=t)
    mix = 0.70
    blended = [max(1e-6, min(0.999, mix*pz[i] + (1-mix)*p_model[i])) for i in range(n)]
    s=sum(blended)
    return [x/s for x in blended] if s>0 else p_model

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
    else:
        zs = handcrafted_scores(track, rc, runners, extras=extras)
        def field_temp(n):
            if n>=12: return 0.80
            if n>=10: return 0.72
            if n>=8:  return 0.66
            return 0.60
        def softmax(zs, temp):
            if not zs: return []
            import math
            m=max(zs); exps=[math.exp((z-m)/max(1e-6,temp)) for z in zs]; s=sum(exps)
            return [e/s for e in exps] if s>0 else [1.0/len(zs)]*len(zs)
        t = field_temp(len(runners))
        ps = softmax(zs, temp=t)
        if len(ps) >= 12:
            ps=[max(0.003,p) for p in ps]; s=sum(ps); ps=[p/s for p in ps]

    # force anti-flat if range is too tight
    rng = (max(ps)-min(ps)) if ps else 0.0
    if rng < 0.04:
        ps = anti_flat_separation(track, rc, runners, ps, extras)
    return ps
'''.strip()

# Inject blocks
replace_block("zscore_or_neutral", ZSCORE_BLOCK)
replace_block("why_feature_pack", WHY_BLOCK)
replace_block("handcrafted_scores", HANDCRAFT_BLOCK)
replace_block("anti_flat_separation", ANTIFLAT_BLOCK.split("def probabilities_from_model_only")[0].strip())
replace_block("probabilities_from_model_only", "def probabilities_from_model_only" + ANTIFLAT_BLOCK.split("def probabilities_from_model_only")[1])

# Tag version
src = src.replace('VERSION = "PF-35 Mach++ v3.8-pro"', 'VERSION = "PF-35 Mach++ v3.8-pro (WHY fixed v3.8-pro+)"')

PRO.write_text(src, encoding="utf-8")
print("[hotfix] applied OK ->", PRO.resolve())
