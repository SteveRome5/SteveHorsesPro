#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
db_tj.py — Trainer/Jockey lookup shim for SteveHorsesPro.

Works in two modes:
1) If ~/Desktop/SteveHorsesPro/data/tj_stats.csv exists, it will load it and do exact lookups.
2) If no CSV is found, lookups return None, but callers can still synthesize TJ% from trainer/jockey %.
   This is intentional so Flags won't be "TJ 0/0" if upstream has tr/jk pcts.

CSV schema (header names are case-insensitive; extra columns are ignored):
  trainer,jockey,track,surface,dist_bucket,wins,starts,win_pct
- trainer, jockey: names (free text)
- track: track name matching your meets (e.g. "Del Mar")
- surface: "dirt"|"turf"|"synt" (or any text; we lowercase)
- dist_bucket: one of "<6f","6f","7f","1mi","8.5f","9f","10f+","unk"
- wins/starts: integers; win_pct optional (if omitted, computed as 100*wins/starts)

Bucket key MUST match Pro's scheme: f"{track}|{surface}|{dist_bucket}"
"""

from __future__ import annotations
import csv, os, re, unicodedata
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

# -------- Paths --------
HOME = Path.home()
BASE = HOME / "Desktop" / "SteveHorsesPro"
DATA_DIR = BASE / "data"
CSV_PATH = DATA_DIR / "tj_stats.csv"   # optional

# -------- Name + bucket normalization (must mirror Pro) --------
def _normalize_person_name(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z\s]+", " ", s)
    s = re.sub(r"\b(the|a|an|of|and|&)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _surface_key(s: str) -> str:
    s = (s or "").lower()
    if "turf" in s: return "turf"
    if "synt" in s or "tapeta" in s or "poly" in s: return "synt"
    return "dirt"

def _dist_bucket_yards(yards: Optional[int]) -> str:
    if not yards: return "unk"
    y = int(yards)
    if y < 1320:  return "<6f"
    if y < 1540:  return "6f"
    if y < 1760:  return "7f"
    if y < 1980:  return "1mi"
    if y < 2200:  return "8.5f"
    if y < 2420:  return "9f"
    return "10f+"

def _bucket_key(track: str, surface: str, dist_bucket: str) -> str:
    return f"{(track or '').strip()}|{_surface_key(surface)}|{(dist_bucket or 'unk').strip()}"

# -------- Safe parsing --------
def _to_float(v, default=None):
    try:
        if v in (None, ""): return default
        return float(v)
    except Exception:
        return default

def _to_int(v, default=None):
    try:
        if v in (None, ""): return default
        return int(float(v))
    except Exception:
        return default

def _safe_pct(win_pct: Optional[float], wins: Optional[int], starts: Optional[int]) -> Optional[float]:
    p = _to_float(win_pct, None)
    if p is None and wins is not None and starts and starts > 0:
        p = 100.0 * float(wins) / float(starts)
    if p is None: return None
    # accept either [0..1] or [0..100]
    if 0.0 <= p <= 1.0: p *= 100.0
    return max(0.0, min(100.0, p))

# -------- In-memory indices --------
# Single-entity (trainer-only, jockey-only) by (name_norm, bucket_key) and (name_norm, "*")
_TRAINER_IDX: Dict[Tuple[str,str], Dict[str,Any]] = {}
_JOCKEY_IDX:  Dict[Tuple[str,str], Dict[str,Any]] = {}
# Combo by (trainer_norm, jockey_norm, bucket_key) and global (trainer_norm, jockey_norm, "*")
_COMBO_IDX:   Dict[Tuple[str,str,str], Dict[str,Any]] = {}

def _load_csv_if_present() -> None:
    if not CSV_PATH.exists():
        return
    try:
        with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                tr_raw = (row.get("trainer") or row.get("Trainer") or "").strip()
                jk_raw = (row.get("jockey")  or row.get("Jockey")  or "").strip()
                track  = (row.get("track")   or row.get("Track")   or "").strip()
                surf   = (row.get("surface") or row.get("Surface") or "").strip()
                dbuck  = (row.get("dist_bucket") or row.get("bucket") or row.get("dist") or "").strip()
                wins   = _to_int(row.get("wins")   or row.get("Wins"), None)
                starts = _to_int(row.get("starts") or row.get("Starts"), None)
                pct    = _safe_pct(_to_float(row.get("win_pct") or row.get("WinPct") or row.get("pct"), None), wins, starts)

                if not (tr_raw or jk_raw):  # need at least one name
                    continue

                tr = _normalize_person_name(tr_raw) if tr_raw else ""
                jk = _normalize_person_name(jk_raw) if jk_raw else ""
                key = _bucket_key(track, surf, dbuck or "unk")

                rec = {"win_pct": pct, "wins": wins, "starts": starts}

                # trainer-only
                if tr and not jk:
                    _TRAINER_IDX[(tr, key)] = rec
                    _TRAINER_IDX[(tr, "*")]  = rec if (tr, "*") not in _TRAINER_IDX else _TRAINER_IDX[(tr, "*")]

                # jockey-only
                if jk and not tr:
                    _JOCKEY_IDX[(jk, key)] = rec
                    _JOCKEY_IDX[(jk, "*")]  = rec if (jk, "*") not in _JOCKEY_IDX else _JOCKEY_IDX[(jk, "*")]

                # combo
                if tr and jk:
                    _COMBO_IDX[(tr, jk, key)] = rec
                    _COMBO_IDX[(tr, jk, "*")] = rec if (tr, jk, "*") not in _COMBO_IDX else _COMBO_IDX[(tr, jk, "*")]
    except Exception:
        # Don’t bomb PRO if the CSV is weird; just run in synth mode
        pass

_load_csv_if_present()

# -------- Lookups expected by PRO --------
def _pick_best(rec_exact: Optional[dict], rec_global: Optional[dict]) -> Optional[dict]:
    # prefer exact bucket; else fall back to global
    return rec_exact or rec_global

def lookup_trainer(name_norm: str, bucket_key: str) -> Optional[Dict[str,Any]]:
    """
    name_norm: normalized person name (already lowercased/stripped by caller)
    bucket_key: 'Track|surface|dist_bucket' (same scheme as PRO)
    returns {"win_pct": float, "wins": int, "starts": int} or None
    """
    n = _normalize_person_name(name_norm)
    if not n:
        return None
    return _pick_best(_TRAINER_IDX.get((n, bucket_key)), _TRAINER_IDX.get((n, "*")))

def lookup_jockey(name_norm: str, bucket_key: str) -> Optional[Dict[str,Any]]:
    n = _normalize_person_name(name_norm)
    if not n:
        return None
    return _pick_best(_JOCKEY_IDX.get((n, bucket_key)), _JOCKEY_IDX.get((n, "*")))

def lookup_combo(trainer_norm: str, jockey_norm: str, bucket_key: str) -> Optional[Dict[str,Any]]:
    tr = _normalize_person_name(trainer_norm)
    jk = _normalize_person_name(jockey_norm)
    if not tr or not jk:
        return None
    # 1) direct combo if present
    rec = _pick_best(_COMBO_IDX.get((tr, jk, bucket_key)), _COMBO_IDX.get((tr, jk, "*")))
    if rec:
        return rec
    # 2) synth combo from individual trainer/jockey if both present
    tr_rec = lookup_trainer(tr, bucket_key)
    jk_rec = lookup_jockey(jk, bucket_key)
    if tr_rec or jk_rec:
        tr_pct = tr_rec["win_pct"] if tr_rec and tr_rec.get("win_pct") is not None else None
        jk_pct = jk_rec["win_pct"] if jk_rec and jk_rec.get("win_pct") is not None else None
        if tr_pct is not None and jk_pct is not None:
            base = 0.6 * min(tr_pct, jk_pct) + 0.4 * (0.5 * (tr_pct + jk_pct))
            pct  = max(0.01, min(60.0, base * 0.92))
        elif tr_pct is not None:
            pct  = max(0.01, min(60.0, tr_pct * 0.85))
        elif jk_pct is not None:
            pct  = max(0.01, min(60.0, jk_pct * 0.85))
        else:
            pct  = None
        if pct is not None:
            # prefer some counts if available, else synth baseline
            wins = (tr_rec or {}).get("wins") or (jk_rec or {}).get("wins")
            starts = (tr_rec or {}).get("starts") or (jk_rec or {}).get("starts")
            if not starts or starts <= 0:
                wins, starts = int(round(pct)), 100
            return {"win_pct": pct, "wins": int(wins), "starts": int(starts)}
    # 3) nothing
    return None