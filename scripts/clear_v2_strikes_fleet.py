#!/usr/bin/env python3
"""Fleet-wide clear of v2-upgrade strikes (each peer's ``v2_upgrade_failed.json``).

Selects entries by *fatal verify check* (``--checks``, forwarded to the peer's
``POST /api/v1/admin/v2_strikes/clear {"checks": [...]}``) and/or by a regex
over the recorded reason (``--reason-re``, resolved here per peer into an
explicit ``codes`` list), or explicit ``--codes``. Run after a verify-gate fix
so struck codes are re-dispatched. The director's fleet union
(``v2_strikes_fleet.json``) drops a peer's stale entries on its next full
status push (minutes); ``/process.txt`` → ``strikes_fleet=… struck_out=…``.

Usage:
  python3 scripts/clear_v2_strikes_fleet.py --checks parcel_segmentation_coverage_pct_ge_v1
  python3 scripts/clear_v2_strikes_fleet.py --reason-re 'download: HTTP 5\\d\\d' --apply
  python3 scripts/clear_v2_strikes_fleet.py --codes 62202 40224-northwest-1 --apply
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'austria_processor'


def peer_urls():
    d = json.load(open(DATA / 'peers.json'))
    peers = d.get('peers', d) if isinstance(d, dict) else d
    urls = {'primary': 'http://127.0.0.1:8000'}
    for p in peers:
        if p.get('url') and p.get('enabled', True):
            urls[p['id']] = p['url']
    return urls


def fatal_checks(reason: str) -> list[str]:
    # mirrors app.admin_v2_strikes_clear (check names = tokens followed by
    # ": " at section start / after "; " or "| "; details may contain ";")
    m = re.search(r' FAIL \d+/\d+ checks \| (.*?)(?: \| (?:warn|hint):|$)', reason)
    if not m:
        return []
    return re.findall(r'(?:^|\| |; )([a-z][a-z0-9_.]*): ', m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checks', nargs='*', default=[])
    ap.add_argument('--reason-re', default='')
    ap.add_argument('--codes', nargs='*', default=[])
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    if not (a.checks or a.reason_re or a.codes):
        ap.error('one of --checks / --reason-re / --codes required')
    import requests
    tok = (ROOT / 'data' / 'admin_token').read_text().strip()
    hdr = {'X-Admin-Token': tok}
    rre = re.compile(a.reason_re) if a.reason_re else None
    checks = set(a.checks)
    total = 0
    for pid, url in peer_urls().items():
        base = url.rstrip('/') + '/api/v1/admin/v2_strikes'
        try:
            d = requests.get(base, headers=hdr, timeout=20).json()
        except Exception as e:
            print(f'{pid:8s} GET ERR {str(e)[:70]}', file=sys.stderr)
            continue
        if not isinstance(d, dict) or not d:
            continue
        sel = set(a.codes) & set(d)
        for code, e in d.items():
            reason = str((e or {}).get('reason') or '')
            fc = fatal_checks(reason)
            if checks and fc and all(f in checks for f in fc):
                sel.add(code)
            if rre and rre.search(reason):
                sel.add(code)
        if not sel:
            continue
        total += len(sel)
        print(f'{pid:8s} {len(sel):3d} of {len(d):3d}: {" ".join(sorted(sel))[:150]}')
        if a.apply:
            try:
                r = requests.post(base + '/clear', json={'codes': sorted(sel)}, headers=hdr, timeout=20)
                j = r.json() if r.ok else {}
                print(f'         → HTTP {r.status_code} removed={len(j.get("removed", []))} remaining={j.get("remaining")}')
            except Exception as e:
                print(f'         → ERR {str(e)[:70]}', file=sys.stderr)
    print(f'{total} strike entries selected' + ('' if a.apply else ' — dry run, pass --apply'))


if __name__ == '__main__':
    main()
