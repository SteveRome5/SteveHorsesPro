#!/usr/bin/env python3
# Ultra-fast single-race sheet (model-lite)

import os, sys, json, html, math, re, base64
from datetime import date
from urllib.request import Request, urlopen
from urllib.parse import urlencode

API_BASE = os.getenv("RACING_API_BASE", "https://api.theracingapi.com")
RUSER = os.getenv("RACINGAPI_USER", "")
RPASS = os.getenv("RACINGAPI_PASS", "")

def _get(path, params=None):
    url = API_BASE + path + ("?" + urlencode(params) if params else "")
    req = Request(url, headers={"User-Agent":"Mozilla/5.0"})
    if RUSER and RPASS:
        tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
        req.add_header("Authorization", "Basic " + tok)
    with urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8","replace"))

def g(d,*ks,default=None):
    for k in ks:
        if isinstance(d,dict) and k in d and d[k] not in (None,""):
            return d[k]
    return default

def prg_num(r):
    return str(g(r,"program_number","program","number","pp","saddle","saddle_number","post_position") or "")

def horse_name(r):
    return g(r,"horse_name","name","runner_name","runner","horse","horseName") or "Unknown"

def _to_float(v):
    try:
        if v in (None,""): return None
        if isinstance(v,(int,float)): return float(v)
        s=str(v).strip()
        m=re.fullmatch(r"(\d+)\s*[/\-:]\s*(\d+)", s)
        if m:
            num, den = float(m.group(1)), float(m.group(2))
            if den!=0: return num/den
        return float(s)
    except:
        return None

def get_speed(r): return _to_float(g(r,"speed","spd","last_speed","best_speed","fig","speed_fig","brz","beyer"))
def get_ep(r):    return _to_float(g(r,"pace","ep","early_pace","quirin"))
def get_lp(r):    return _to_float(g(r,"lp","late_pace","closer","lateSpeed"))
def get_cls(r):   return _to_float(g(r,"class","cls","class_rating","classRating","par_class","parClass"))

def zlist(xs):
    xs=[(x if x is not None else 0.0) for x in xs]
    n=len(xs)
    if n<=1: return [0.0]*n
    m=sum(xs)/n
    var=sum((x-m)**2 for x in xs)/n
    s=math.sqrt(var) if var>1e-12 else 1.0
    return [(x-m)/s for x in xs]

def softmax(zs, temp=0.66):
    if not zs: return []
    m=max(zs); exps=[math.exp((z-m)/max(1e-6,temp)) for z in zs]; s=sum(exps)
    return [e/s for e in exps]

MENU = [
    "Saratoga","Del Mar","Santa Anita","Gulfstream Park","Keeneland",
    "Parx Racing","Finger Lakes","Kentucky Downs","Woodbine",
    "Laurel Park","Louisiana Downs",
]

def choose_track_and_race():
    print("\nSelect a Track:")
    for i,t in enumerate(MENU,1):
        print(f" {i}) {t}")
    traw = input("\nEnter Track # (e.g. 1 for Saratoga): ").strip() or "1"
    try: ti=int(traw)
    except: ti=1
    ti=max(1,min(len(MENU),ti))
    track=MENU[ti-1]
    rraw = input("Enter Race # (default 1): ").strip() or "1"
    try: ri=int(rraw)
    except: ri=1
    return track, max(1,ri)

def find_meet_for_track(iso, track_name):
    meets=_get("/v1/north-america/meets", {"start_date": iso, "end_date": iso}).get("meets",[])
    for m in meets:
        if (g(m,"track_name","track","name") or "").strip().lower()==track_name.lower():
            return g(m,"meet_id","id","meetId")
    return None

def get_race_by_number(entries, rno):
    races = entries.get("races") or entries.get("entries") or []
    for r in races:
        n=g(r,"race_number","race","number","raceNo")
        try: n=int(re.sub(r"[^\d]","",str(n)))
        except: n=None
        if n==rno: return r
    return None

def make_sheet(track, rno, rc, out_path):
    runners = (rc.get("runners") or rc.get("entries") or [])
    runners = [r for r in runners if str(g(r,"status","runnerStatus","entry_status","condition") or "").lower() not in ("scr","scratched","scratch","wd","withdrawn")]
    names=[horse_name(r) for r in runners]
    nums=[prg_num(r) for r in runners]
    spd=[get_speed(r) or 0.0 for r in runners]
    ep =[get_ep(r)    or 0.0 for r in runners]
    lp =[get_lp(r)    or 0.0 for r in runners]
    cls=[get_cls(r)   or 0.0 for r in runners]

    spdZ, epZ, lpZ, clsZ = zlist(spd), zlist(ep), zlist(lp), zlist(cls)
    w_spd, w_ep, w_lp, w_cls = 1.0, 0.55, 0.30, 0.45
    zs=[w_spd*spdZ[i] + w_ep*epZ[i] + w_lp*lpZ[i] + w_cls*clsZ[i] for i in range(len(runners))]
    ps=softmax(zs, temp=0.66)

    rows=[]
    for i in range(len(runners)):
        rows.append(
            f"<tr><td class='mono'>{html.escape(nums[i])}</td>"
            f"<td>{html.escape(names[i])}</td>"
            f"<td class='mono'><b>{ps[i]*100:0.1f}%</b></td>"
            f"<td class='mono'>{spd[i]:.0f}</td>"
            f"<td class='mono'>{ep[i]:.0f}</td>"
            f"<td class='mono'>{lp[i]:.0f}</td>"
            f"<td class='mono'>{cls[i]:.0f}</td></tr>"
        )

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>FAST — {html.escape(track)} Race {rno}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px}}
table{{border-collapse:collapse;width:100%;margin:12px 0}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;font-size:14px}}
th{{background:#f3f3f3}} .mono{{font-variant-numeric:tabular-nums}}
</style></head><body>
<h1>FAST Sheet — {html.escape(track)} <span style="color:#666">Race {rno} · {date.today().isoformat()}</span></h1>
<p style="color:#666">Model-lite (speed/pace/class only). No exotics/market/conditions — built for speed.</p>
<table><thead><tr>
<th>#</th><th>Horse</th><th>Win% (Lite)</th><th>Speed</th><th>EP</th><th>LP</th><th>Class</th>
</tr></thead><tbody>
{''.join(rows)}
</tbody></table>
</body></html>"""
    with open(out_path,"w",encoding="utf-8") as f:
        f.write(page)
    print(f"[fast] wrote {out_path}")

def main():
    trk, rno = choose_track_and_race()
    iso = date.today().isoformat()
    meet_id = find_meet_for_track(iso, trk)
    if not meet_id:
        print(f"[fast] No meet found for {trk} on {iso}")
        return
    entries = _get(f"/v1/north-america/meets/{meet_id}/entries")
    rc = get_race_by_number(entries, rno)
    if not rc:
        print(f"[fast] Race {rno} not found at {trk} today")
        return
    out_dir = os.path.join(os.path.expanduser("~"), "Desktop", "SteveHorsesPro", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    safe_trk = re.sub(r"[^A-Za-z0-9]+","_", trk).strip("_")
    out_path = os.path.join(out_dir, f"{iso}_{safe_trk}_race{rno}_fast.html")
    make_sheet(trk, rno, rc, out_path)

if __name__ == "__main__":
    main()
