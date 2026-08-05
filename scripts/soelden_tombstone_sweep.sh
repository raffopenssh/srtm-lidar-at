#!/bin/bash
# Sweep the 117 Soelden product tombstones fleet-wide.
#
# 2026-08-05 UPDATE: the "loop until clean" strategy could never converge.
# /admin/drop_tombstones only cleared one gunicorn worker's in-memory dict
# plus disk; the sibling worker re-persisted the keys, and every peer
# re-broadcast them to every other peer via _sync_peer_data. The result was
# a reported-clean fleet that had 114 tombstones back on all 7 peers
# minutes later -- and peers' processors then dropped ~40 finished blocks
# from completed_codes and began re-running ~9 h of work per block.
#
# The real fix is the tombstone DROP JOURNAL in app.py
# (manifest_tombstone_drops.json): a drop is now recorded as a negative
# tombstone, honoured by both gunicorn workers, propagated peer-to-peer,
# and respected by austria_processor + peer_director. So one pass per node
# is sufficient and permanent. This script is now a one-shot fan-out.
#
# Background: re-queueing the split PARENT 80110 tombstoned all 117
# products of its 38 finished blocks fleet-wide (fixed at source in the
# "never invalidate COMPLETE sibling blocks" commit, but the already-
# stamped tombstones must be swept). A peer still on the old code has no
# endpoint to clear them, and its tombstones are NEWER than the restored
# manifest entries, so every 5-min peer-sync re-imports them to the
# primary. Hence: loop until clean, then verify.
set -u
cd /home/exedev/srtm-lidar || exit 1
TOKEN=$(cat data/admin_token)
BODY='{"prefix":"80110","keep":["80110_requeue"]}'
sweep() {  # $1 = base url
  curl -sm25 -X POST -H "X-Admin-Token: $TOKEN" \
    -H 'Content-Type: application/json' -d "$BODY" \
    "$1/api/v1/admin/drop_tombstones" 2>&1 | head -c 200
}
for i in $(seq 1 3); do
  DIRTY=0
  for pid in $(curl -sm20 http://localhost:8000/api/v1/director/status \
      | python3 -c "
import json,sys
print(' '.join(p['id'] for p in (json.load(sys.stdin).get('peers') or [])
                if p.get('id') and p['id']!='primary'))" 2>/dev/null); do
    N=$(curl -sm20 "https://srtm-lidar-$pid.exe.xyz:8000/api/v1/processing/peers" \
      | python3 -c "
import json,sys
t=(json.load(sys.stdin).get('tombstones') or {})
print(sum(1 for k in t if k.startswith('80110') and k!='80110_requeue'))" 2>/dev/null)
    [ "${N:-0}" -gt 0 ] 2>/dev/null || continue
    DIRTY=$((DIRTY+1))
    echo "$(date -Is) $pid dirty($N): $(sweep https://srtm-lidar-$pid.exe.xyz:8000)"
  done
  # Always re-clean the primary: a dirty peer may already have re-merged.
  sweep http://localhost:8000 >/dev/null
  if [ "$DIRTY" = "0" ]; then
    echo "$(date -Is) fleet clean after $i pass(es)"
    break
  fi
  sleep 60
done
echo "=== final state ==="
python3 -c "
import json
e=json.load(open('data/austria_processor/zenodo_manifest.json'))['entries']
k=[x for x in e if x.startswith('80110')]
t=json.load(open('data/austria_processor/manifest_tombstones.json'))
tk=[x for x in t if x.startswith('80110')]
print('manifest entries:', len(k), '| _json:', sum(1 for x in k if x.endswith('_json')))
print('tombstones:', tk)"
