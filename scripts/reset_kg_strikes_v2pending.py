#!/usr/bin/env python3
"""Fleet-wide reset of adaptive-split strikes (kg_strikes.json) for KGs that
are v1-complete but not yet v2-complete — i.e. everything still ahead of the
v2 upgrade / v2_regen / v2.2 re-run pipeline.

Why: since 37cea2a (2026-05) every "graceful" stop was a SIGKILL (runuser),
so each rollout counted a strike on whatever KG the peer was on (3b1b9bd).
Those strikes are not evidence about the KG; when such a KG is re-run for
v2.2 (v2_regen, manual re-queue) the splitter would shred it into tiny
blocks that bounce between cache-only peers. These are all small enough
that re-running them whole (or by plain tile-count split) is cheap.

Mechanics: ``PUT /api/v1/processing/kg_strikes {code: -1}`` drops the
code + its ``<code>-*`` block children (3b1b9bd+). Peers max-merge their
local copy back to the director on their next interrupt, so the reset
must land on EVERY peer — peers on pre-3b1b9bd code ignore negatives, so
re-run this once the whole fleet has rolled forward (``versions:`` line
in /process.txt). Idempotent.

Usage:
  python3 scripts/reset_kg_strikes_v2pending.py            # dry run
  python3 scripts/reset_kg_strikes_v2pending.py --apply
  python3 scripts/reset_kg_strikes_v2pending.py --apply --codes 63330 84006
  python3 scripts/reset_kg_strikes_v2pending.py --apply --include-v1pending
  python3 scripts/reset_kg_strikes_v2pending.py --apply --from-peers   # laggard peers, after primary already reset
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'data' / 'austria_processor'


def _committed(ent, key):
    e = ent.get(key)
    return (isinstance(e, dict) and 'error' not in str(e.get('status', ''))
            and int(e.get('size') or 0) > 0)


def fleet_strike_codes():
    """Union of strike codes across primary + every enabled peer. Needed
    because once the primary's own kg_strikes.json has been reset, the
    local file no longer names the codes that peers still carry (peers on
    old code at the time of the first run ignored the negatives)."""
    urls, requests = peer_urls()
    codes = set(json.load(open(DATA / 'kg_strikes.json')))
    for pid, url in urls.items():
        try:
            r = requests.get(url.rstrip('/') + '/api/v1/processing/kg_strikes', timeout=20)
            d = r.json()
            d = d.get('strikes', d) if isinstance(d, dict) else {}
            codes |= set(d)
        except Exception as e:
            print(f'{pid:8s} strikes GET ERR {str(e)[:60]}', file=sys.stderr)
    return codes


def select_codes(include_v1pending=False, from_peers=False):
    from kg_splitter import is_block_code, parent_kg_code
    from v21_products import v2_products_complete as v2c
    ent = json.load(open(DATA / 'zenodo_manifest.json'))['entries']
    strikes = fleet_strike_codes() if from_peers else json.load(open(DATA / 'kg_strikes.json'))
    parents = {parent_kg_code(c) if is_block_code(c) else c for c in strikes}
    out = {}
    for p in sorted(parents):
        v1 = _committed(ent, p + '_json')
        blocks = [k[:-5] for k in ent
                  if k.endswith('_json') and k.startswith(p + '-') and _committed(ent, k)]
        if v1 or blocks:
            codes = [p] if v1 else blocks
            if all(v2c(ent, c) for c in codes):
                continue                      # already v2-complete: irrelevant
            out[p] = 'v1done' if v1 else f'v1done({len(blocks)} blocks)'
        elif include_v1pending:
            out[p] = 'v1pending'
    return out


def peer_urls():
    import requests
    d = json.load(open(DATA / 'peers.json'))
    peers = d.get('peers', d) if isinstance(d, dict) else d
    urls = {'primary': 'http://127.0.0.1:8000'}
    for p in peers:
        if p.get('url') and p.get('enabled', True):
            urls[p['id']] = p['url']
    return urls, requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--codes', nargs='*')
    ap.add_argument('--include-v1pending', action='store_true')
    ap.add_argument('--from-peers', action='store_true',
                    help='source strike codes from the union of all peers (use after primary was already reset)')
    a = ap.parse_args()
    if a.codes:
        sel = {c: 'explicit' for c in a.codes}
    else:
        sel = select_codes(a.include_v1pending, a.from_peers)
    print(f'{len(sel)} parent KGs selected')
    for c, why in list(sel.items())[:15]:
        print(f'  {c} {why}')
    if len(sel) > 15:
        print('  …')
    if not a.apply:
        print('dry run — pass --apply')
        return
    tok = (ROOT / 'data' / 'admin_token').read_text().strip()
    body = {c: -1 for c in sel}
    urls, requests = peer_urls()
    for pid, url in urls.items():
        try:
            r = requests.put(url.rstrip('/') + '/api/v1/processing/kg_strikes',
                             json=body, timeout=30, headers={'X-Admin-Token': tok})
            j = r.json() if r.ok else {}
            print(f'{pid:8s} HTTP {r.status_code} reset={len(j.get("reset", []))} '
                  f'remaining={j.get("total")}')
        except Exception as e:
            print(f'{pid:8s} ERR {str(e)[:80]}')


if __name__ == '__main__':
    main()
