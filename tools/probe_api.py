#!/usr/bin/env python3
import os, requests, sys
BASE = os.environ.get("RACING_API_BASE","https://api.theracingapi.com").rstrip("/")
AUTH = (os.environ.get("RACINGAPI_USER",""), os.environ.get("RACINGAPI_PASS",""))
DAY  = sys.argv[1] if len(sys.argv)>1 else os.environ.get("DAY","")
def hit(p, params=None):
    u=f"{BASE}{p}"
    r=requests.get(u, params=params or {}, auth=AUTH, timeout=20)
    print(r.status_code, u, "\n", r.text[:400], "\n---")
hit("/v1/north-america/meets", {"start_date":DAY,"end_date":DAY})
# If 200, copy a meet_id printed by the first call and probe entries:
# hit(f"/v1/north-america/meets/<meet_id>/entries")