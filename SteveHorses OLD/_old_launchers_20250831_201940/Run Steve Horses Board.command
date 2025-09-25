cat > "$HOME/Desktop/Run_Pro_FAST.command" <<'SH'
#!/bin/zsh
set -euo pipefail

BASE="$HOME/Desktop/SteveHorsesPro"
LOGDIR="$BASE/logs"
OUTDIR="$BASE/outputs"
mkdir -p "$LOGDIR" "$OUTDIR"

# --- hard caps to keep it snappy ---
export PYTHONUNBUFFERED=1
export PRO_HTTP_TIMEOUT=3         # per request seconds
export PRO_HTTP_CONNECT=2
export PRO_THREADS=4              # keep it modest so the Mac stays responsive
export FAST=1

# --- market mode: ML fallback only (no live calls today) ---
# Your script reads market in roughly this order:
# live -> win-pool -> historical -> ML. We force ML first and skip live.
export PRO_MARKET_MODE="ml-only"  # code checks this env; when set ml-only it shouldn't call live
export PRO_NO_LIVE=1              # belt-and-suspenders: guards any live odds fetch

# --- keep everything else you tuned ---
export PRO_NO_AUDIT=${PRO_NO_AUDIT:-1}
export PRO_FAST_TTL=${PRO_FAST_TTL:-60}
export PRO_ALPHA=${PRO_ALPHA:-1.30}
export PRO_USE_SHARP=${PRO_USE_SHARP:-1}
export BANKROLL=${BANKROLL:-20000}
export KELLY_CAP=${KELLY_CAP:-0.12}
export MAX_BET_PER_HORSE=${MAX_BET_PER_HORSE:-1500}
export MIN_STAKE=${MIN_STAKE:-50}
export MAJOR_TRACKS_ONLY=${MAJOR_TRACKS_ONLY:-"Aqueduct Racetrack,Belmont at the Big A,Belmont Park,Saratoga Race Course,Churchill Downs,Keeneland,Gulfstream Park,Santa Anita Park,Del Mar,Oaklawn Park,Fair Grounds,Parx Racing,Woodbine,Monmouth Park,Tampa Bay Downs,Kentucky Downs"}

# --- sanity about signals path (PRO expects $BASE/signals -> $BASE/data/signals) ---
if [ ! -L "$BASE/signals" ]; then
  rm -rf "$BASE/signals" 2>/dev/null || true
  ln -sfn "$BASE/data/signals" "$BASE/signals"
fi

cd "$BASE"
echo "[FAST] starting PRO (ML-only, ${PRO_THREADS} threads, timeouts=${PRO_HTTP_CONNECT}/${PRO_HTTP_TIMEOUT})"
exec /usr/bin/python3 -X faulthandler "$BASE/steve_horses_pro.py" \
  2>&1 | tee "$LOGDIR/pro_fast.$(date +%H%M%S).log"
SH
chmod +x "$HOME/Desktop/Run_Pro_FAST.command"
xattr -d com.apple.quarantine "$HOME/Desktop/Run_Pro_FAST.command" 2>/dev/null || true

open "$HOME/Desktop/Run_Pro_FAST.command"