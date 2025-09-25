#!/usr/bin/env python3
# train_results_ingest.py — store official results and update bias
from __future__ import annotations
import os, ssl, json, base64, re
from datetime import date, datetime
from urllib.request import Request, urlopen
from urllib.parse import urlencode

API_BASE = os.getenv("RACING_API_BASE", "https://api.theracingapi.com")
RUSER = os.getenv('RACINGAPI_USER') or os.getenv('RACINGAPI_USER'.upper())
RPASS = os.getenv('RACINGAPI_PASS') or os.getenv('RACINGAPI_PASS'.upper())
CTX = ssl.create_default_context()

EP_MEETS = "/v1/north-america/meets"
EP_ENTRIES_BY_MEET = "/v1/north-america/meets/{meet_id}/entries"
EP_RESULTS_BY_RACE = "/v1/north-america/races/{race_id}/results"
EP_CONDITION_BY_RACE = "/v1/north-america/races/{race_id}/condition"

def _get(path, params=None):
    url = API_BASE + path + ("?" + urlencode(params) if params else "")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if RUSER and RPASS:
        tok = base64.b64encode(f"{RUSER}:{RPASS}".encode()).decode()
        req.add_header("Authorization", "Basic " + tok)
    with urlopen(req, timeout=20, context=CTX) as r:
        return json.loads(r.read().decode("utf-8","replace"))

def g(d:dict,*ks,default=None):
    for k in ks:
        if isinstance(d,dict) and k in d and d[k] not in (None,""):
            return d[k]
    return default

def prg_num(r): 
    return str(g(r,"program_number","program","number","pp","post_position","horse_number","saddle","saddle_number") or "")

def run_once():
    from db_results import upsert_race_meta, upsert_results, upsert_exacta, update_bias_from_race
    iso_today = date.today().isoformat()
    meets = (_get(EP_MEETS, {"start_date": iso_today, "end_date": iso_today}) or {}).get("meets", [])
    for m in meets:
        track = g(m,"track_name","track","name") or ""
        mid = g(m,"meet_id","id","meetId")
        if not mid: 
            continue
        entries = _get(EP_ENTRIES_BY_MEET.format(meet_id=mid)) or {}
        races = entries.get("races") or entries.get("entries") or []
        for r_idx, rc in enumerate(races, 1):
            rid = g(rc,"race_id","id","raceId","raceID") or ""
            rno = g(rc,"race_number","race","number","raceNo") or str(r_idx)
            if not rid:
                continue
            # results (if published)
            try:
                res = _get(EP_RESULTS_BY_RACE.format(race_id=rid)) or {}
            except:
                res = {}
            if not res:
                continue
            # condition / meta
            try:
                cond = _get(EP_CONDITION_BY_RACE.format(race_id=rid)) or {}
            except:
                cond = {}

            # meta
            surface = (g(cond,"surface","track_surface","course","courseType","trackSurface","surf") or
                       g(rc,"surface","track_surface","course","courseType","trackSurface","surf") or "")
            def _yards(rc) -> int | None:
                d=g(rc,"distance_yards","distance","dist_yards","yards","distanceYards","distance_y")
                if d:
                    try: return int(float(d))
                    except: pass
                m=g(rc,"distance_meters","meters","distanceMeters")
                if m:
                    try: return int(float(m)*1.09361)
                    except: pass
                return None
            upsert_race_meta(
                rid,
                race_date=iso_today,
                track=track,
                race_num=str(rno),
                surface=str(surface or ""),
                distance_yards=_yards(rc),
                rail=(g(cond,"rail","rail_setting","railDistance","rail_distance","turf_rail")),
                cond=(g(cond,"condition","track_condition","dirt_condition","surface_condition","turf_condition","turfCondition")),
                takeout_win=g(cond,"takeout","win_takeout","takeout_win"),
            )

            # ordered finishers
            rows=[]
            finishers = g(res,"finishers","results","order","positions") or []
            # If API returns horses inside res['runners'] with 'position', map that
            if not finishers:
                finishers = g(res,"runners") or []
            for it in finishers:
                pr = prg_num(it)
                pos = g(it,"position","finish","result","rank")
                if pos in (None,""): 
                    continue
                try:
                    pos = int(str(pos))
                except:
                    continue
                rows.append({
                    "program": pr,
                    "position": pos,
                    "odds_dec": g(it,"odds_dec","decimal_odds","odds","price"),
                    "win_paid": g(it,"win_paid","payout_win","win"),
                    "place_paid": g(it,"place_paid","payout_place","place"),
                    "show_paid": g(it,"show_paid","payout_show","show"),
                    "beaten_lengths": g(it,"beaten","beaten_lengths","margin"),
                    "speed": g(it,"speed","fig","beyer","brz","last_speed"),
                    "class_": g(it,"class","class_rating","classRating")
                })
            if rows:
                upsert_results(rid, rows)

            # exacta if present
            ex = g(res,"exacta") or {}
            if ex:
                a = g(ex,"a_program","a","first")
                b = g(ex,"b_program","b","second")
                upsert_exacta(rid, str(a or ""), str(b or ""), g(ex,"payout","payoff","price"))

            # bias update
            update_bias_from_race(rid)

if __name__ == "__main__":
    try:
        run_once()
        print("[results-ingest] ok", datetime.now().isoformat(timespec="seconds"))
    except Exception as e:
        print("[results-ingest] ERROR:", e)