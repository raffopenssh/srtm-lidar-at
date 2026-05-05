#!/usr/bin/env python3
"""Re-upload locally-fixed KG JSONs to their existing Zenodo draft depositions.

After `fix_frav_inflation.py` rewrote local KG JSONs to clamp inflated
per-parcel `area_summary` / `frav` values, the canonical Zenodo copies are
now stale. This script pushes the fixed JSONs back to Zenodo, replacing
the file in each KG's existing draft deposition.

Key constraint: only KG depositions that are still drafts can have their
files replaced. Published depositions need a new version (out of scope here
until we have a publish step).

Usage:
    python3 reupload_fixed_jsons.py --dry-run
    python3 reupload_fixed_jsons.py                     # all KGs in manifest
    python3 reupload_fixed_jsons.py --kg 91109          # one KG
    python3 reupload_fixed_jsons.py --only-stale        # skip KGs whose remote
                                                        #   was uploaded after the
                                                        #   local file was rewritten

Respects the Zenodo upload mutex when run on a peer (uses the same lock
broker as austria_processor.py).
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from zenodo_client import Client, Manifest, DEFAULT_TOKEN, landscape_metadata  # noqa: E402

try:
    from zenodo_lock import zenodo_upload_lock  # noqa: E402
except Exception:
    # Fall back to a no-op context manager if the helper isn't present.
    from contextlib import contextmanager
    @contextmanager
    def zenodo_upload_lock(*a, **kw):  # type: ignore
        yield

log = logging.getLogger('reupload')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

JSON_DIR = ROOT / 'data' / 'austria_processor' / 'json'
MANIFEST_PATH = ROOT / 'data' / 'austria_processor' / 'zenodo_manifest.json'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--kg', help='process only this KG code (else: all)')
    ap.add_argument('--dry-run', action='store_true',
                    help='show what would be uploaded')
    ap.add_argument('--only-stale', action='store_true',
                    help='only upload when local mtime > remote uploaded_at')
    ap.add_argument('--limit', type=int, default=0,
                    help='process at most N KGs (0=unlimited)')
    args = ap.parse_args()

    manifest = Manifest(MANIFEST_PATH)
    entries = manifest.entries()

    if args.kg:
        candidates = [args.kg]
    else:
        # Every JSON entry in the manifest has a corresponding *_json key.
        candidates = sorted({k[:-5] for k in entries if k.endswith('_json')})

    n_total = n_uploaded = n_skipped = n_failed = 0
    client = None if args.dry_run else Client(token=DEFAULT_TOKEN)

    for kg in candidates:
        if args.limit and n_uploaded >= args.limit:
            log.info('reached --limit %d; stopping', args.limit)
            break
        n_total += 1
        key = f'{kg}_json'
        entry = entries.get(key)
        if not entry:
            log.warning('%s: no manifest entry; skipping', kg)
            n_skipped += 1
            continue

        jp = JSON_DIR / f'{kg}.json'
        if not jp.exists():
            log.warning('%s: local JSON missing at %s', kg, jp)
            n_skipped += 1
            continue

        # Optional staleness gate: skip if Zenodo is already newer.
        if args.only_stale:
            remote_ts = entry.uploaded_at or 0
            local_ts = jp.stat().st_mtime
            if remote_ts >= local_ts - 1:
                log.info('%s: remote (uploaded_at=%d) ≥ local mtime (%d); skipping',
                         kg, int(remote_ts), int(local_ts))
                n_skipped += 1
                continue

        try:
            kg_name = json.loads(jp.read_text()).get('kg_name', '')
        except Exception:
            kg_name = ''

        def _meta(k, fn, ver, _kg=kg, _name=kg_name):
            return landscape_metadata(
                kg_code=_kg, kg_name=_name, version=ver, file_type='json',
            )

        size_kb = jp.stat().st_size / 1024
        log.info('%s: %s (%.0f KB) -> deposition %s',
                 kg, jp.name, size_kb, entry.depo_id or '?')
        if args.dry_run:
            n_uploaded += 1
            continue

        try:
            with zenodo_upload_lock(purpose='reupload_json', kg=kg):
                client.upload(
                    key=key,
                    local_path=str(jp),
                    version='v1',
                    meta_func=_meta,
                    manifest=manifest,
                )
            n_uploaded += 1
        except Exception as e:
            log.error('%s: upload failed: %s', kg, e)
            n_failed += 1

    log.info('done: total=%d uploaded=%d skipped=%d failed=%d',
             n_total, n_uploaded, n_skipped, n_failed)
    return 0 if n_failed == 0 else 2


if __name__ == '__main__':
    sys.exit(main())
