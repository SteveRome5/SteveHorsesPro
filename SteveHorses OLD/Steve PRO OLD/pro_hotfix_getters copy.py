import re, sys, pathlib

PRO = pathlib.Path("steve_horses_pro.py")
src = PRO.read_text(encoding="utf-8")

def replace_func(name, body):
    # Replace def <name>(...) block up to next "def " at col 0
    pat = re.compile(rf"(?ms)^def\s+{name}\s*\(.*?\n(?:(?!^def\s).*\n)*")
    if not pat.search(src):
        return None
    return pat.sub(body.rstrip()+"\n", src)

def insert_antiflat(s):
    # Find probabilities_from_model_only and inject an anti-flat mix if range too small
    m = re.search(r"(?ms)^def\s+probabilities_from_model_only\s*\(.*?\):\s*(.*?)^\s*return\s+ps\s*$", s)
    if not m:
        return s  # different build name; skip safely
    block = m.group(0)
    if "anti_flat_separation" in block:
        return s  # already has it
    # Inject just before final "return ps"
    injected = re.sub(
        r"(?ms)(^\s*)return\s+ps\s*$",
        r"""\1rng = (max(ps)-min(ps)) if ps else 0.0
\1if rng < 0.04:
\1    # Break ties using handcrafted signals when the model/market is too flat
\1    extras = {'sect': {'pressure': 0.0, 'meltdown': 0.0}}
\1    runners = (rc.get('runners') or rc.get('entries') or [])
\1    zs = handcrafted_scores(track, rc, runners, extras=extras)
\1    from math import exp
\1    m = max(zs) if zs else 0.0
\1    exps = [exp(z-m)/max(1e-6, 0.55) for z in zs] if zs else []
\1    ssum = sum(exps)
\1    if ssum > 0:
\1        prior = [e/ssum for e in exps]
\1        ps = [0.7*prior[i] + 0.3*ps[i] for i in range(len(ps))]
\1        s2 = sum(ps)
\1        if s2 > 0: ps = [p/s2 for p in ps]
\1return ps""",
        block
    )
    return s.replace(block, injected)

# New, more tolerant getters (collect real values no matter the API variant)
GET_SPEED = r'''
def get_speed(r):
    from builtins import str as _s
    keys = ("speed","spd","last_speed","lastSpeed","best_speed","bestSpeed",
            "fig","speed_fig","beyer","brz","sf","bsf","prime_power","primePower")
    for k in keys:
        v = r.get(k)
        if v not in (None, ""):
            try: return float(str(v).strip())
            except: pass
    # nested spots occasionally used
    for k in ("figs","ratings","numbers"):
        d = r.get(k) or {}
        if isinstance(d, dict):
            for cand in ("speed","beyer","last","best"):
                v = d.get(cand)
                if v not in (None,""):
                    try: return float(_s(v).strip())
                    except: pass
    return None
'''

GET_EP = r'''
def get_early_pace(r):
    keys = ("pace","ep","early_pace","earlyPace","runstyle","style","quirin","ep_fig","early")
    for k in keys:
        v = r.get(k)
        if v not in (None, ""):
            try: return float(str(v).strip())
            except: pass
    return None
'''

GET_LP = r'''
def get_late_pace(r):
    keys = ("lp","late_pace","closer","finishing_kick","lateSpeed","lp_fig","late")
    for k in keys:
        v = r.get(k)
        if v not in (None, ""):
            try: return float(str(v).strip())
            except: pass
    return None
'''

GET_CLASS = r'''
def get_class(r):
    keys = ("class","cls","class_rating","classRating","par_class","parClass",
            "pclass","last_class","best_class")
    for k in keys:
        v = r.get(k)
        if v not in (None, ""):
            try: return float(str(v).strip())
            except: pass
    return None
'''

# Replace the four getters
new = replace_func("get_speed", GET_SPEED)
if new is None: 
    print("[hotfix] get_speed not found"); sys.exit(1)
src = new

for nm, blk in (("get_early_pace", GET_EP), ("get_late_pace", GET_LP), ("get_class", GET_CLASS)):
    new = replace_func(nm, blk)
    if new is None:
        print(f"[hotfix] {nm} not found")
        sys.exit(1)
    src = new

# Insert anti-flat guard (no-op if already present or different build)
src = insert_antiflat(src)

# Tag version so we can see it in logs/HTML
src = src.replace('VERSION = "', 'VERSION = "PF-35 Mach++ v3.8-pro (getters+antiflat) • ')

PRO.write_text(src, encoding="utf-8")
print("[hotfix] applied OK ->", PRO)
