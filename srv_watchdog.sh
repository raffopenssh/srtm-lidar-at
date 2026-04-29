#!/usr/bin/env bash
# srv watchdog: restarts srv.service if /api/v1/ping is unresponsive.
#
# Why: when all gunicorn workers wedge on outbound HTTPS calls, the
# socket listens but every request hangs.  systemd's own Restart= can't
# detect this (the process is alive).  A 5 s wall-clock probe + 3 strikes
# trips a forced restart.
#
# Runs every 60 s via srv-watchdog.timer.
set -u

STATE_FILE="${HOME}/.cache/srv_watchdog.state"
MAX_FAILURES=3
PING_TIMEOUT=5      # seconds for the whole probe
MIN_RESTART_INTERVAL=600    # seconds — don't loop-restart
LOG_TAG=srv-watchdog

mkdir -p "$(dirname "$STATE_FILE")"

failures=0
last_restart=0
if [[ -f $STATE_FILE ]]; then
    # shellcheck disable=SC1090
    . "$STATE_FILE"
fi

now=$(date +%s)
http_code=$(curl -s -o /dev/null -w '%{http_code}' \
    --connect-timeout 2 --max-time "$PING_TIMEOUT" \
    http://127.0.0.1:8000/api/v1/ping || echo 000)

if [[ $http_code == 200 ]]; then
    if (( failures > 0 )); then
        logger -t "$LOG_TAG" "recovered after $failures failures"
    fi
    failures=0
else
    failures=$((failures + 1))
    logger -t "$LOG_TAG" "ping failed (http=$http_code, strike $failures/$MAX_FAILURES)"
fi

if (( failures >= MAX_FAILURES )); then
    if (( now - last_restart < MIN_RESTART_INTERVAL )); then
        logger -t "$LOG_TAG" "would restart but last restart was $((now - last_restart))s ago — waiting"
    else
        logger -t "$LOG_TAG" "restarting srv.service (gunicorn wedged)"
        sudo /bin/systemctl restart srv.service || \
            logger -t "$LOG_TAG" "systemctl restart failed"
        last_restart=$now
        failures=0
    fi
fi

printf 'failures=%s\nlast_restart=%s\n' "$failures" "$last_restart" >"$STATE_FILE"
