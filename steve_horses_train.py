#!/usr/bin/env python3
"""
Minimal training fix that creates required files for your PRO system
"""

import os
import sys
import json
from datetime import datetime, date
from pathlib import Path

# Your paths
BASE = Path.home() / "Desktop" / "SteveHorsesPro"
MODELS = BASE / "models"
DATA = BASE / "data"
SIGNALS = DATA / "signals"

# Ensure directories exist
for d in [BASE, MODELS, DATA, SIGNALS]:
    d.mkdir(exist_ok=True, parents=True)

def create_minimal_model():
    """Create minimal model file that PRO expects"""
    model = {
        "buckets": {},
        "global": {
            "w": [0.30, 0.25, 0.20, 0.15, 0.10],
            "b": 0.0
        },
        "pars": {
            "default": {"spd": 80.0, "cls": 70.0}
        },
        "calib": {},
        "meta": {
            "version": "1",
            "created": datetime.now().isoformat(),
            "samples": 0
        }
    }
    
    model_file = MODELS / "model.json"
    with open(model_file, 'w') as f:
        json.dump(model, f, indent=2)
    
    print(f"Created model: {model_file}")

def create_signals():
    """Create empty signals for today"""
    today = date.today().isoformat()
    
    signals = {
        "date": today,
        "tracks": {},
        "generated": datetime.now().isoformat()
    }
    
    signals_file = SIGNALS / f"signals_{today}.json"
    with open(signals_file, 'w') as f:
        json.dump(signals, f, indent=2)
    
    print(f"Created signals: {signals_file}")

def main():
    print("=" * 50)
    print("MINIMAL TRAINING SETUP")
    print("=" * 50)
    
    create_minimal_model()
    create_signals()
    
    print("=" * 50)
    print("TRAINING SETUP COMPLETE")
    print("Your PRO system should now work!")
    print("=" * 50)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())