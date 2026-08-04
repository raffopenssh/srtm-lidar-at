#!/bin/bash
# One-shot: clear the permanent-failure verdict on Soelden's two
# out-of-coverage blocks once the fleet is running code that can emit an
# out-of-coverage product for them (commit with _handle_out_of_coverage).
#
# Must clear on EVERY peer that holds the verdict *before* the primary --
# app._sync_peer_data now unions peers' failed_kgs into the primary's, so
# clearing the primary first would just re-import it on the next tick.
#
# Also clears retried_kgs (the processor only grants one fresh attempt per
# code ever; without this the blocks stay skipped even with an empty
# failed_kgs.json) and re-queues the parent so it drains promptly.
set -u
cd /home/exedev/srtm-lidar || exit 1
BLOCKS='80110-southeast-3 80110-southeast-6'
TARGET=$(git rev-parse --short=7 HEAD)
TOKEN=$(cat data/admin_token)

for i in $(seq 1 60); do
  BEHIND=$(curl -sm20 http://localhost:8000/api/v1/director/status \
    | python3 -c "
import json,sys
d=json.load(sys.stdin)
t='$TARGET'
print(sum(1 for p in (d.get('peers') or [])
          if (p.get('git_commit') or '')[:7] != t))" 2>/dev/null)
  [ "${BEHIND:-99}" = "0" ] && break
  echo "$(date -Is) waiting: $BEHIND peer(s) not yet on $TARGET"
  sleep 60
done

for pid in $(curl -sm20 http://localhost:8000/api/v1/director/status \
    | python3 -c "
import json,sys
print(' '.join(p['id'] for p in (json.load(sys.stdin).get('peers') or [])
                if p.get('id')))" 2>/dev/null); do
  [ "$pid" = "primary" ] && continue
  curl -sm30 -X POST -H "X-Admin-Token: $TOKEN" \
    "https://srtm-lidar-$pid.exe.xyz:8000/api/v1/admin/unfail_kgs" \
    -H 'Content-Type: application/json' \
    -d "{\"kgs\": [\"${BLOCKS// /\", \"}\"]}" >/dev/null 2>&1 \
    && echo "$(date -Is) cleared on $pid"
done

python3 - <<'PY'
import json, pathlib
blocks = {'80110-southeast-3', '80110-southeast-6'}
for name in ('failed_kgs.json', 'retried_kgs.json'):
    p = pathlib.Path('data/austria_processor') / name
    if not p.exists():
        continue
    cur = set(json.loads(p.read_text() or '[]'))
    if cur & blocks:
        p.write_text(json.dumps(sorted(cur - blocks), indent=2))
        print('cleared', name, sorted(cur & blocks))
PY

curl -sm30 -X POST -H "X-Admin-Token: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"kgs":["80110"],"position":0,"skip_processed":false}' \
  http://localhost:8000/api/v1/processing/queue
echo "$(date -Is) done"
