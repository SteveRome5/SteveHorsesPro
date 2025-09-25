fix_numbers () 
{ 
    local html="$1";
    /usr/bin/env python3 - "$html"  <<'PY'
import os, sys, json, html, base64, ssl, urllib.parse
from urllib.request import Request, urlopen
CTX = ssl.create_default_context()

html_path = sys.argv[1]
user = os.getenv("RACING_API_USER","").strip()
pwd  = os.getenv("RACING_API_PASS","").strip()
if not user or not pwd:
    sys.exit(0)  # nothing we can do

# fetch today's NA meets
from datetime import date
params = urllib.parse.urlencode({"start_date":date.today().isoformat(),"end_date":date.today().isoformat()})
u = f"https://api.theracingapi.com/v1/north-america/meets?{params}"
tok = base64.b64encode(f"{user}:{pwd}".encode()).decode()
req = Request(u, headers={"User-Agent":"Mozilla/5.0","Authorization":"Basic "+tok})
with urlopen(req, timeout=30, context=CTX) as r:
    meets = json.loads(r.read().decode()).get("meets",[])

# pull entries for every meet and build name->program mapping
name2prog = {}
for m in meets:
    mid = m.get("meet_id") or m.get("id")
    if not mid: continue
    u = f"https://api.theracingapi.com/v1/north-america/meets/{mid}/entries"
    req = Request(u, headers={"User-Agent":"Mozilla/5.0","Authorization":"Basic "+tok})
    try:
        with urlopen(req, timeout=30, context=CTX) as r:
            js = json.loads(r.read().decode())
    except Exception:
        continue
    for race in js.get("races", []):
        for e in race.get("entries", []):
            nm = (e.get("horse") or e.get("name") or "").strip()
            if not nm: continue
            prog = (e.get("program") or e.get("prog") or e.get("saddlecloth") 
                    or e.get("saddlecloth_number") or e.get("number") 
                    or e.get("post_position") or e.get("post") or e.get("pp"))
            if prog is None: continue
            prog = str(prog).strip()
            # Tidy e.g. "01" -> "1", keep things like "1A"
            if prog.isdigit(): prog = str(int(prog))
            name2prog.setdefault(nm, prog)

if not name2prog:
    sys.exit(0)

# rewrite HTML: prefix first occurrence of each exact horse name with "#X "
txt = open(html_path, "r", encoding="utf-8").read()
for nm, prog in sorted(name2prog.items(), key=lambda kv: -len(kv[0])):  # longer names first to avoid partials
    safe = html.escape(nm)
    txt = txt.replace(f">{safe}<", f">#{prog} {safe}<")

open(html_path, "w", encoding="utf-8").write(txt)
PY

}
