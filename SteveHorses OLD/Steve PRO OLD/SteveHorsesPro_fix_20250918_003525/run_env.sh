# --- Steve Horses env ---
export RACINGAPI_USER="YOUR_USERNAME_HERE"
export RACINGAPI_PASS="YOUR_PASSWORD_HERE"
# Optional base (leave as-is if my default works for you)
export RACING_API_BASE="https://api.theracingapi.com"

# Pro run-time knobs (tweak if you like)
export PRO_MODE="1"                     # turn on confidence gating & dampers
export BANKROLL="20000"
export DAILY_EXPOSURE_CAP="0.12"
export KELLY_CAP="0.15"
export MAX_BET_PER_HORSE="1500"
export MIN_STAKE="50"
export MIN_PAD="0.22"
export ACTION_MAX_PER="400"

export EDGE_WIN_PCT_FLOOR="0.20"
export ACTION_PCT_FLOOR="0.13"
export EDGE_PP_MIN_PRIME="3.0"
export EDGE_PP_MIN_ACTION="5.0"

# Confidence thresholds (used when PRO_MODE=1)
export CONF_THRESH_PRIME="0.58"
export CONF_THRESH_ACTION="0.50"

# Ensure we call the right Python
# If you use a specific python (e.g. /usr/local/bin/python3 or a pyenv), point to it here:
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"