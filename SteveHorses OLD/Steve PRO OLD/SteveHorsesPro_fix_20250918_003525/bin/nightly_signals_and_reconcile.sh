#!/bin/zsh
set -euo pipefail

BASE="$HOME/Desktop/SteveHorsesPro"
cd "$BASE"

export RACINGAPI_USER='WQaKSMwgmG8GnbkHgvRRCT0V'
export RACINGAPI_PASS='McYBoQViXSPvlNcvxQi1Z1py'
export PYTHONWARNINGS="ignore:NotOpenSSLWarning"

LOG="$BASE/logs/nightly_signals.log"
mkdir -p "$BASE/logs" "$BASE/signals" "$BASE/ledger"

echo "[nightly] $(date) start" | tee -a "$LOG"

TODAY="$(date +%F)"
CAND="$BASE/outputs/${TODAY}_horses_targets+full.html"
if [ -f "$CAND" ] && [ $(stat -f%z "$CAND" 2>/dev/null || echo 0) -ge 50000 ]; then
  REPORT="$CAND"
else
  REPORT="$(ls -t "$BASE"/outputs/*_horses_targets+full.html 2>/dev/null | head -n1 || true)"
fi

if [ -z "${REPORT:-}" ] || [ ! -f "$REPORT" ]; then
  echo "[nightly] no HTML report found — skipping" | tee -a "$LOG"
  exit 0
fi

BASENAME="$(basename "$REPORT" .html)"
OUT_CSV="$BASE/signals/${BASENAME/_horses_targets+full/}_signals.csv"

echo "[nightly] using report: $REPORT" | tee -a "$LOG"

if [ -f "$BASE/extract_signals_from_html.py" ]; then
  /usr/bin/python3 "$BASE/extract_signals_from_html.py" "$REPORT" "$OUT_CSV" 2>&1 | tee -a "$LOG"
  echo "[nightly] signals -> $OUT_CSV" | tee -a "$LOG"
fi

if [ -f "$BASE/ledger_reconcile.py" ] && [ -f "$OUT_CSV" ]; then
  /usr/bin/python3 "$BASE/ledger_reconcile.py" "$OUT_CSV" "$BASE/ledger/results_ledger.csv" 2>&1 | tee -a "$LOG"
  echo "[nightly] reconcile complete -> $BASE/ledger/results_ledger.csv" | tee -a "$LOG"
fi

if [ -f "$BASE/ledger_metrics.py" ] && [ -f "$BASE/ledger/results_ledger.csv" ]; then
  /usr/bin/python3 "$BASE/ledger_metrics.py" "$BASE/ledger/results_ledger.csv" > "$BASE/ledger/metrics.txt" 2>>"$LOG" || true
  echo "[nightly] metrics -> $BASE/ledger/metrics.txt" | tee -a "$LOG"
fi

echo "[nightly] $(date) done" | tee -a "$LOG"
