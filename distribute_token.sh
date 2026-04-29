#!/bin/bash
# Distribute the cluster admin token to all peers in peers.json.
#
# Usage (from the primary):
#   bash distribute_token.sh
#
# For each peer:
#   1) Fetch its current token (loopback only -> not reachable from us;
#      we can't read it remotely). So we use install_token's bootstrap
#      mode: works iff the peer has no token yet (first-deploy or after
#      `rm data/admin_token; systemctl restart srv` has not yet
#      auto-generated one).
#   2) If install fails because the peer already has a different token,
#      print the manual recovery command.
#
# Pre-req: the peer is already running the new auth code. Run
# /api/v1/director/update_peers first to roll out the code, then this.
set -e
cd "$(dirname "$0")"
TOK=$(cat data/admin_token)
[ -z "$TOK" ] && { echo "no local token"; exit 1; }

python3 - <<PY
import json, requests, sys, pathlib
cfg = json.load(open('data/austria_processor/peers.json'))
tok = pathlib.Path('data/admin_token').read_text().strip()
peers = [p for p in cfg.get('peers', []) if p.get('url')]
print(f"Distributing token to {len(peers)} peers...")
for p in peers:
    pid, url = p['id'], p['url']
    try:
        r = requests.post(url.rstrip('/') + '/api/v1/admin/install_token',
                          json={'new_token': tok, 'current_token': tok},
                          headers={'X-Admin-Token': tok},
                          timeout=10)
        if r.ok:
            print(f"  {pid:6s} OK ({r.json().get('status')})")
        elif r.status_code == 404:
            print(f"  {pid:6s} 404 endpoint missing (run /director/update_peers first)")
        elif r.status_code == 401:
            print(f"  {pid:6s} 401 already has different token; manual fix needed:")
            print(f"           ssh into {pid} and run: echo {tok} > data/admin_token && sudo systemctl restart srv")
        else:
            print(f"  {pid:6s} HTTP {r.status_code} {r.text[:120]}")
    except Exception as e:
        print(f"  {pid:6s} ERR {e}")
PY
