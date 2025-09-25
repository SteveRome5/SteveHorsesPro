#!/usr/bin/env python3
"""
Updater: optional patch + version logger.

Behavior:
- If a payload file exists, overwrite steve_horses_pro.py from it.
  Payload search order:
    1) SteveHorsesPro/tools/payload_steve_horses_pro.py
    2) SteveHorsesPro/payload_steve_horses_pro.py
- Regardless, read VERSION from steve_horses_pro.py and append a line to logs/run.log:
    [YYYY-MM-DD HH:MM:SS] updater: version='PF-35 Mach++ vX.Y' patched=yes|no
- Prints the same banner to stdout so you see it in Terminal.
"""

from pathlib import Path
from datetime import datetime
import re, shutil, sys

HOME = Path.home()
BASE = HOME / "Desktop" / "SteveHorsesPro"
TOOLS = BASE / "tools"
TARGET = BASE / "steve_horses_pro.py"
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PAYLOADS = [
    TOOLS / "payload_steve_horses_pro.py",
    BASE / "payload_steve_horses_pro.py",
]

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_version(text: str) -> str:
    # Grab VERSION = "PF-35 Mach++ v1.7"
    m = re.search(r'^\s*VERSION\s*=\s*[\'"]([^\'"]+)[\'"]', text, re.M)
    return m.group(1).strip() if m else "UNKNOWN"

def log_line(msg: str):
    (LOG_DIR / "run.log").open("a", encoding="utf-8").write(f"[{ts()}] {msg}\n")

def main():
    BASE.mkdir(parents=True, exist_ok=True)

    # 1) Maybe patch from payload
    patched = False
    payload = next((p for p in PAYLOADS if p.exists()), None)
    if payload is not None:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        if TARGET.exists():
            backup = BASE / f"steve_horses_pro.py.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(TARGET, backup)
        src_txt = payload.read_text(encoding="utf-8")
        TARGET.write_text(src_txt, encoding="utf-8")
        patched = True

    # 2) Read VERSION from current target
    if not TARGET.exists():
        # Nothing to log against; at least say something.
        msg = "updater: target steve_horses_pro.py not found; patched=no"
        print("[warn]", msg)
        log_line(msg)
        return

    txt = TARGET.read_text(encoding="utf-8")
    version = read_version(txt)

    # 3) Write to log and echo to stdout
    status = f"updater: version='{version}' patched={'yes' if patched else 'no'}"
    print(f"[info] {status}")
    log_line(status)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] updater failed: {e}", file=sys.stderr)
        log_line(f"updater: FAILED error={e}")
        sys.exit(1)