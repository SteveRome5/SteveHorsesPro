#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pro Filtered Full — fast, filtered “full report” for one track (and optional race)
# - Uses existing model.json (no training/harvest)
# - Skips slow endpoints (exotics, equipment, odds_history)
# - Blends with live odds + willpays (fast enough) if available
# - Outputs same “full” layout, but scoped to your selection

from __future__ import annotations
import os, ssl, json, html, base64, re, math, sys, csv, statistics
from pathlib import Path
from datetime import date, datetime
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from collections import defaultdict

VERSION = "Pro Filtered Full v1.0"

# ---------- Paths ----------
HOME = Path.home()
BASE = HOME / "Desktop" / "SteveHorsesPro"
OUT_DIR = BASE / "outputs"; LOG_DIR = BASE / "logs"; MODEL_DIR = BASE / "models"
DATA_DIR = BASE / "data"
for d in (OUT_DIR, LOG_DIR, MODEL_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        (LOG_DIR / "run_filtered.log").open("a", encoding="utf-8").write(f"[{ts}] {msg}\n")
    except: pass

# ---------- Input (env prompts) ----------
ISO_DATE = os.getenv("PRO_DATE") or date.today().isoformat()
TRACK_FILTER = (os.getenv("PRO_TRACK") or "").strip()
RACE_FILTER = os.getenv("PRO_RACE")
try:
    RACE_FILTER = int(RACE_FILTER) if (RACE_FILTER and str(RACE_FILTER).strip().isdigit()) else None
except:
    RACE_FILTER = None

USE_LIVE = os.getenv("LIVE_ODDS", "1") == "1"

# ---------- API ----------
RUSER = os.getenv("RACINGAPI_USER", "WQaKSMwgmG8GnbkHgvRRCT0V")
RPASS = os.getenv("RACINGAPI_PASS", "McYBoQViXSPvlNcvxQi1Z1py")
API_BASE = os.getenv("RACING_API_BASE", "https://api.theracingapi.com")
CTX = ssl.create_default_context()

EP_MEETS = "/v1/north-america/meets"
EP_ENTRIES_BY_MEET = "/v1/north-america/meets/{meet_id}/entries"
EP_CONDITION_BY_RACE = "/v1/north-america/races/{race_id}/condition"
EP_WILLPAYS         = "/v1/north-america/races/{race_id}/willpays"

def _get(path, params=None):
    url = API_BASE + path + ("?" + urlencode(params) if params else "")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
    req.add_header("Authorization", "Basic " + tok)
    with urlopen(req, timeout=25, context=CTX) as r:
        raw = r.read().decode("utf-8","replace")
        return json.loads(raw)

def safe_get(path, params=None, default=None):
    try: return _get(path, params)
    except Exception as e:
        log(f"GET fail {path}: {e}")
        return default

# ---------- Small utils ----------
def g(d:dict,*ks,default=None):
    for k in ks:
        if isinstance(d,dict) and k in d and d[k] not in (None,""):
            return d[k]
    return default

def _to_float(v, default=None):
    try:
        if v in (None,""): return default
        if isinstance(v,(int,float)): return float(v)
        return float(str(v).strip())
    except: return default

def _to_dec_odds(v, default=None):
    if v in (None,""): return default
    try:
        if isinstance(v,(int,float)): f=float(v); return f if f>1 else default
        s=str(v).strip().lower()
        if s in ("evs","even","evens"): return 2.0
        m=re.fullmatch(r"(\d+)\s*[/\-:]\s*(\d+)", s)
        if m:
            num,den=float(m.group(1)),float(m.group(2))
            return 1.0 + (num/den) if den>0 else default
        dec=float(s); return dec if dec>1 else default
    except: return default

def implied_from_dec(dec): 
    return (1.0/dec) if (dec and dec>1) else None

def odds_formats(dec: float) -> str:
    if not dec or dec<=1: return "—"
    v=dec-1.0; best="—"; err=9e9
    for den in (1,2,3,4,5,6,8,10,12,16,20,32):
        num=round(v*den); e=abs(v-num/den)
        if e<err: err, best = e, f"{int(num)}/{int(den)}"
    payout = math.floor((2*dec)*100)/100.0
    return f"{best} • ${payout:0.2f} • {dec:.2f}"

def prg_num(r): 
    return str(g(r,"program_number","program","number","pp","post_position","horse_number","saddle","saddle_number") or "")

def horse_name(r): 
    return g(r,"horse_name","name","runner_name","runner","horse","horseName") or "Unknown"

def race_num(rc, idx): 
    return g(rc,"race_number","raceNo","race_num","number","race","rno") or idx

def get_surface(rc): 
    return str(g(rc,"surface","track_surface","course","courseType","trackSurface","surf") or "").lower()

def get_distance_y(rc):
    d=g(rc,"distance_yards","distance","yards","distanceYards","distance_y")
    if d is not None:
        try: return int(float(d))
        except: pass
    m=g(rc,"distance_meters","meters","distanceMeters")
    if m is not None:
        try: return int(float(m)*1.09361)
        except: pass
    return None

def get_rail(rc): 
    return _to_float(g(rc,"rail","rail_setting","railDistance","rail_distance","turf_rail"), default=0.0)

def get_field_size(rc): 
    return int(g(rc,"field_size","fieldSize","num_runners","entriesCount") or 0) or None

def get_minutes_to_post(rc): 
    return _to_float(g(rc,"minutes_to_post","mtp","minutesToPost"), default=None)

def get_speed(r):  return _to_float(g(r,"speed","spd","last_speed","best_speed","fig","speed_fig","beyer"), None)
def get_early_pace(r):  return _to_float(g(r,"pace","ep","early_pace","quirin"), None)
def get_late_pace(r):   return _to_float(g(r,"lp","late_pace","lateSpeed"), None)
def get_class(r):       return _to_float(g(r,"class","cls","class_rating"), None)
def get_trainer_win(r): return _to_float(g(r,"trainer_win_pct","trainerWinPct"), None)
def get_jockey_win(r):  return _to_float(g(r,"jockey_win_pct","jockeyWinPct"), None)
def get_combo_win(r):   return _to_float(g(r,"tj_win","combo_win"), None)

def live_decimal(r): 
    return _to_dec_odds(g(r,"live_odds","odds","currentOdds","liveOdds"))

# ---------- Model load (no training here) ----------
FEATS = [
    "speed","ep","lp","class","trainer_win","jockey_win","combo_win",
    "field_size","rail","ml_dec","live_dec","minutes_to_post","last_days","weight",
    "post_bias","surface_switch","equip_blinker","equip_lasix","pace_fit","class_par_delta"
]
MODEL = {"buckets":{}, "global":{}, "pars":{}, "calib":{}}

def load_model():
    p = MODEL_DIR / "model.json"
    if not p.exists(): 
        log("model.json missing — probabilities will fall back (slower/shallower)")
        return False
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
        MODEL.update(m)
        return True
    except Exception as e:
        log(f"model load fail: {e}")
        return False

def _sigmoid(z): 
    z=max(-50.0,min(50.0,z)); 
    return 1.0/(1.0+math.exp(-z))

def _surface_key(s: str) -> str:
    s=(s or "").lower()
    if "turf" in s: return "turf"
    if "synt" in s or "tapeta" in s or "poly" in s: return "synt"
    return "dirt"

def _dist_bucket_yards(yards: int|None) -> str:
    if not yards: return "unk"
    if yards < 1320:  return "<6f"
    if yards < 1540:  return "6f"
    if yards < 1760:  return "7f"
    if yards < 1980:  return "1mi"
    if yards < 2200:  return "8.5f"
    if yards < 2420:  return "9f"
    return "10f+"

def build_bucket_key(track: str, surface: str, yards: int|None) -> str:
    return f"{track}|{_surface_key(surface)}|{_dist_bucket_yards(yards)}"

def _apply_standardize(x, stat): 
    mu,sd=stat.get("mu",[0.0]*len(FEATS)), stat.get("sd",[1.0]*len(FEATS))
    return [(xi - mu[j])/ (sd[j] if sd[j] else 1.0) for j,xi in enumerate(x)]

def predict_bucket_prob(track: str, rc: dict, r: dict) -> float|None:
    surf = get_surface(rc); yards = get_distance_y(rc)
    key  = build_bucket_key(track, surf, yards)
    entry= MODEL.get("buckets",{}).get(key) or MODEL.get("global")
    if not entry or not entry.get("w"): return None

    # minimal features (ml_dec forced 0.0)
    row = {
        "track": track, "surface": surf, "distance_yards": yards,
        "speed": get_speed(r), "ep": get_early_pace(r), "lp": get_late_pace(r),
        "class": get_class(r), "trainer_win": get_trainer_win(r), "jockey_win": get_jockey_win(r),
        "combo_win": get_combo_win(r),
        "field_size": get_field_size(rc) or (len(rc.get("runners") or rc.get("entries") or [])),
        "rail": get_rail(rc),
        "ml_dec": 0.0, 
        "live_dec": (live_decimal(r) if USE_LIVE else 0.0) or 0.0,
        "minutes_to_post": get_minutes_to_post(rc) or 15.0,
        "last_days": _to_float(g(r,"days_since","dsl","last_start_days"), None),
        "weight": _to_float(g(r,"weight","carried_weight","assigned_weight","wt","weight_lbs"), None),
        "prev_surface": g(r,"prev_surface","last_surface","last_surface_type"),
        "program": prg_num(r)
    }

    # simple feature builder (par/light pace fit omitted for speed)
    def S(x,a): return ((x or 0.0)/a)
    x = [
        S(row["speed"],100.0), S(row["ep"],120.0), S(row["lp"],120.0), S(row["class"],100.0),
        S(row["trainer_win"],100.0), S(row["jockey_win"],100.0), S(row["combo_win"],100.0),
        S(row["field_size"],12.0), S(row["rail"],30.0), 0.0, S(row["live_dec"],10.0),
        S(row["minutes_to_post"],30.0), S(row["last_days"],60.0), S(row["weight"],130.0),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]

    xs = _apply_standardize(x, entry.get("stat", {"mu":[0.0]*len(FEATS),"sd":[1.0]*len(FEATS)}))
    z = entry["b"] + sum(wj*xj for wj,xj in zip(entry["w"], xs))
    p_raw = _sigmoid(z)
    curve = MODEL.get("calib",{}).get(key) or MODEL.get("calib",{}).get("__global__", [])
    if not curve: return max(1e-6, min(0.999, p_raw))
    xs_c=[c[0] for c in curve]; ys_c=[c[1] for c in curve]
    p=p_raw
    if p<=xs_c[0]: p = ys_c[0]*(p/max(1e-6,xs_c[0]))
    elif p>=xs_c[-1]:
        p = ys_c[-1] + (p - xs_c[-1])*(ys_c[-1]-ys_c[-2])/max(1e-6,(xs_c[-1]-xs_c[-2]))
    else:
        for i in range(1,len(xs_c)):
            if p<=xs_c[i]:
                w=(p - xs_c[i-1])/max(1e-6,(xs_c[i]-xs_c[i-1]))
                p = ys_c[i-1]*(1-w) + ys_c[i]*w
                break
    return max(1e-6, min(0.999, p))

# ---------- Pace-lite + softmax fallback ----------
def pace_style(r):
    ep = get_early_pace(r) or 0.0
    lp = get_late_pace(r)  or 0.0
    if ep - lp >= 8:   return "E"
    if ep - lp >= 3:   return "EP"
    if lp - ep >= 5:   return "S"
    return "P"

def field_temp(n):
    if n>=12: return 0.80
    if n>=10: return 0.72
    if n>=8:  return 0.66
    return 0.60

def softmax(zs, t):
    m=max(zs); exps=[math.exp((z-m)/max(1e-6,t)) for z in zs]; s=sum(exps)
    return [e/s for e in exps] if s>0 else [1.0/len(zs)]*len(zs)

def handcrafted_scores(track, rc, runners):
    spd=[get_speed(r) or 0.0 for r in runners]
    ep =[get_early_pace(r) or 0.0 for r in runners]
    lp =[get_late_pace(r) or 0.0 for r in runners]
    cls=[get_class(r) or 0.0 for r in runners]
    def z(xs):
        if not xs: return []
        m=statistics.mean(xs); s=statistics.pstdev(xs) if len(xs)>1 else 1.0
        if s<1e-6: s=1.0
        return [(x-m)/s for x in xs]
    spdZ,epZ,lpZ,clsZ=z(spd),z(ep),z(lp),z(cls)
    w_spd,w_ep,w_lp,w_cls=1.0,0.55,0.30,0.45
    scores=[w_spd*spdZ[i]+w_ep*epZ[i]+w_lp*lpZ[i]+w_cls*clsZ[i] for i in range(len(runners))]
    return scores

def probabilities(track, rc, runners):
    ps=[]; ok=True
    for r in runners:
        p = predict_bucket_prob(track, rc, r)
        if p is None: ok=False; break
        ps.append(max(1e-6,min(0.999,p)))
    if ok and ps:
        s=sum(ps); return [p/s for p in ps] if s>0 else [1.0/len(ps)]*len(ps)
    zs = handcrafted_scores(track, rc, runners)
    ps = softmax(zs, field_temp(len(runners)))
    if len(ps)>=12:
        ps=[max(0.003,p) for p in ps]; s=sum(ps); ps=[p/s for p in ps]
    return ps

# ---------- Market + condition (fast) ----------
def fetch_condition(race_id):
    d=safe_get(EP_CONDITION_BY_RACE.format(race_id=race_id), default={}) or {}
    return {"cond": g(d,"condition","track_condition","dirt_condition","surface_condition") or g(d,"turf_condition","turfCondition") or "",
            "takeout": _to_float(g(d,"takeout","win_takeout","takeout_win"), default=None)}

def fetch_willpays(race_id):
    d=safe_get(EP_WILLPAYS.format(race_id=race_id), default={}) or {}
    prob={}
    for it in g(d,"win_probables","probables","win","willpays") or []:
        pr=str(g(it,"program","number","pp","saddle") or "")
        p=_to_float(g(it,"impl_win","prob","p"), None)
        if pr and p: prob[pr]=max(0.01,min(0.99,p))
    pool=_to_float(g(d,"pool","win","win_pool","winPool"), default=None)
    return {"impl": prob, "win_pool": pool}

def blend_with_market_if_present(p_model, p_market, minutes_to_post):
    if not p_market or all(x is None for x in p_market): return p_model
    pm = [0.0 if (x is None or x <= 0) else float(x) for x in p_market]
    sm = sum(pm); pm = [x/sm if sm > 0 else 0.0 for x in pm]
    alpha = 0.93 if minutes_to_post >= 20 else (0.88 if minutes_to_post >= 8 else 0.80)
    blended=[(max(1e-9,m)**alpha)*(max(1e-9,mk)**(1.0-alpha)) for m,mk in zip(p_model, pm)]
    s=sum(blended)
    return [b/s for b in blended] if s>0 else p_model

# ---------- Build track cards (filtered) ----------
MAJOR_TRACKS = {
    "Saratoga","Del Mar","Santa Anita","Santa Anita Park","Gulfstream Park",
    "Keeneland","Parx Racing","Finger Lakes","Kentucky Downs",
    "Woodbine","Laurel Park","Louisiana Downs"
}

def fetch_meets(iso_date): 
    return safe_get(EP_MEETS, {"start_date": iso_date, "end_date": iso_date}, default={"meets":[]})

def fetch_entries(meet_id): 
    return safe_get(EP_ENTRIES_BY_MEET.format(meet_id=meet_id), default={"races":[]})

def is_scratched_runner(r):
    status = str(g(r,"status","runnerStatus","entry_status","entryStatus","condition") or "").lower().strip()
    if status in {"scr","scratched","scratch","wd","withdrawn","dns","dnp","dq"}: return True
    for k in ("is_scratched","isScratched","scratched_flag","scratchedFlag","withdrawn","scr"):
        v = g(r, k)
        if isinstance(v, bool) and v: return True
        if isinstance(v, str) and v.lower().strip() in ("1","true","yes","y"): return True
    tag = str(g(r,"scratch_indicator","scratchIndicator") or "").lower().strip()
    if tag in ("1","true","yes","y","scr"): return True
    return False

def build_filtered_cards(iso_date, track_name, race_num_opt=None):
    meets = fetch_meets(iso_date).get("meets", [])
    if not meets: return {}
    target = None
    for m in meets:
        t = g(m,"track_name","track","name") or ""
        if t.lower().strip() == track_name.lower().strip():
            target = m; break
    if not target:
        # try relaxed match on MAJOR_TRACKS aliasing
        for m in meets:
            t = g(m,"track_name","track","name") or ""
            if t in MAJOR_TRACKS and track_name.lower() in t.lower():
                target = m; break
    if not target: 
        return {}

    mid = g(target,"meet_id","id","meetId")
    if not mid: return {}
    entries = fetch_entries(mid) or {}
    races = entries.get("races") or entries.get("entries") or []
    if race_num_opt:
        # filter to given race number
        filt=[]
        for r in races:
            rno = g(r,"race_number","race","number","raceNo")
            try: rn = int(re.sub(r"[^\d]","", str(rno)))
            except: rn = None
            if rn == race_num_opt:
                filt.append(r)
        races = filt

    # normalize runners & scratches
    for r in races:
        r["runners"]=r.get("runners") or r.get("entries") or r.get("horses") or r.get("starters") or []
        for rr in r["runners"]:
            if is_scratched_runner(rr): rr["scratched"]=True
        r["runners"] = [rr for rr in r["runners"] if not rr.get("scratched")]
    return { (g(target,"track_name","track","name") or "Track"): [r for r in races if (r.get("runners") or [])] }

# ---------- HTML ----------
def edge_color(p, dec):
    imp = implied_from_dec(dec)
    if imp is None: return ""
    ed = p - imp
    if ed <= 0: return ""
    s = max(0.0, min(1.0, ed*100/8.0))
    return f"background-color: rgba(40,200,80,{0.10 + 0.15*s:.2f});"

def debug_tags_for_runner(r):
    tags=[]
    if (get_speed(r) or 0)>=95: tags.append("Spd↑")
    if (get_class(r) or 0)>=90: tags.append("Cls↑")
    if (get_trainer_win(r) or 0)>=18: tags.append("Trn↑")
    if (get_jockey_win(r) or 0)>=18: tags.append("Jky↑")
    if (get_combo_win(r) or 0)>=20: tags.append("TJ↑")
    tags.append(pace_style(r))
    return " ".join(tags) or "—"

def build_report(cards, iso_date):
    parts=[f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(VERSION)} — {html.escape(iso_date)}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:14px}}
th{{background:#f3f3f3}} .mono{{font-variant-numeric:tabular-nums}} .small{{color:#666;font-size:12px}} 
.bet{{background:#eef9f0}} .sub{{color:#555}}
</style></head><body>"""]
    parts.append(f"<h1>{html.escape(VERSION)} <span class='small'>({html.escape(iso_date)})</span></h1>")

    full_sections=[]
    prime_board=[]; action_board=[]
    daily_cap_amt = 0 # not allocating bankroll in filtered mode
    parts.append(f"<p class='small'>Filtered to {html.escape(TRACK_FILTER)}"
                 + (f", Race {RACE_FILTER}" if RACE_FILTER else "") + "</p>")

    for track, races in cards.items():
        for idx, rc in enumerate(races,1):
            rno=str(race_num(rc, idx))
            rid=str(g(rc,"race_id","id","raceId","raceID") or "")
            runners = (rc.get("runners") or [])
            if not runners: continue

            cond=fetch_condition(rid) if rid else {"cond":"", "takeout":None}
            wp=fetch_willpays(rid) if rid else {"impl":{}, "win_pool": None}

            # market (live odds or willpays)
            market=[]; market_probs=[]
            for r in runners:
                pr=prg_num(r)
                mkt=(live_decimal(r) if USE_LIVE else None)
                implied=wp.get("impl",{}).get(pr,None)
                if implied and implied>0:
                    dec_from_wp = 1.0/max(0.01,min(0.99, implied))
                    mkt = dec_from_wp if not mkt or mkt<=1 else min(mkt, dec_from_wp)
                market.append(mkt)
                market_probs.append((1.0/mkt) if (mkt and mkt>1) else None)

            p_model = probabilities(track, rc, runners)
            m2p = get_minutes_to_post(rc) or 30.0
            p_final = blend_with_market_if_present(p_model, market_probs, m2p)

            # enrich
            enriched=[]
            for r, pM, pF, dec in zip(runners, p_model, p_final, market):
                imp = implied_from_dec(dec) if dec else None
                fair = 1.0/max(1e-6,pF)
                # min price pad (simple)
                to = cond.get("takeout") or 0.16
                pad = 0.22 + 0.5*to + 0.012*max(0,(len(runners)-8))
                minp = fair*(1.0+pad)
                enriched.append({
                    "num": prg_num(r) or "", "name": horse_name(r),
                    "p_final": pF, "imp": imp, "fair": fair, "minp": minp,
                    "market": dec, "tags": debug_tags_for_runner(r),
                    "why": "—", "why_tip": ""
                })

            # Full race table
            rows=[f"<h3>{html.escape(track)} — Race {html.escape(str(rno))}</h3>"]
            rows.append("<table><thead><tr>"
                        "<th>#</th><th>Horse</th><th>Win% (Final)</th><th>Market%</th><th>Edge</th>"
                        "<th>Fair</th><th>Min Price</th><th>Market</th>"
                        "</tr></thead><tbody>")
            for it in enriched:
                style=edge_color(it["p_final"], it["market"])
                imp_pct = (it["imp"]*100.0 if it["imp"] is not None else None)
                edge_pp = ((it["p_final"] - (it["imp"] or 0))*100.0) if it["imp"] is not None else None
                rows.append(
                    f"<tr style='{style}'>"
                    f"<td class='mono'>{html.escape(it['num'])}</td>"
                    f"<td>{html.escape(it['name'])} <span class='small'>{html.escape(it['tags'])}</span></td>"
                    f"<td class='mono'><b>{it['p_final']*100:0.1f}%</b></td>"
                    f"<td class='mono'>{(imp_pct and f'{imp_pct:0.1f}%') or '—'}</td>"
                    f"<td class='mono'>{(edge_pp is not None and f'{edge_pp:+0.1f} pp') or '—'}</td>"
                    f"<td class='mono'>{odds_formats(it['fair'])}</td>"
                    f"<td class='mono'>{odds_formats(it['minp'])}</td>"
                    f"<td class='mono'>{odds_formats(it['market'])}</td>"
                    f"</tr>"
                )
            rows.append("</tbody></table>")
            full_sections.append("\n".join(rows))

    parts.append("<hr><h2>Full Races</h2>"); parts.extend(full_sections)
    parts.append(f"<p class='small'>Version {html.escape(VERSION)} — generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></body></html>")
    return "\n".join(parts)

def main():
    log(f"Filtered run: date={ISO_DATE}, track='{TRACK_FILTER}', race={RACE_FILTER}")
    if not TRACK_FILTER:
        print("ERROR: Set PRO_TRACK env (and optional PRO_RACE).")
        sys.exit(2)
    load_model()
    cards = build_filtered_cards(ISO_DATE, TRACK_FILTER, RACE_FILTER)
    out = OUT_DIR / f"{ISO_DATE}_{TRACK_FILTER.replace(' ','_')}{('_R'+str(RACE_FILTER)) if RACE_FILTER else ''}_filtered_full.html"
    if not cards:
        out.write_text(f"<h1>{html.escape(VERSION)} <span class='small'>({ISO_DATE})</span></h1><p>No races found for {html.escape(TRACK_FILTER)}.</p>", encoding="utf-8")
        print(out); 
        if sys.platform=="darwin": os.system(f"open '{out}'")
        return
    html_doc = build_report(cards, ISO_DATE)
    out.write_text(html_doc, encoding="utf-8")
    print(out)
    if sys.platform=="darwin":
        os.system(f"open '{out}'")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: sys.exit(130)