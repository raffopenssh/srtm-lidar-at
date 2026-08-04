#!/usr/bin/env python3
"""Find queue entries wedged by split-layout drift.

A parent KG can get stuck in retry_queue.json forever when its block
layout changed between runs (e.g. the adaptive strike-split raised the
block count from 8 to 15). The manifest then holds a mix of *old* and
*new* geometry blocks whose labels happen to cover every label of the
current layout:

  * process.html renders pips per **label** -> shows 15/15 done ("complete")
  * app._kg_coverage_complete() judges **geometry** -> finds a real hole,
    so GET /api/v1/processing/queue never prunes the code
  * austria_processor skips every block of the current layout because all
    those block *codes* are in completed_codes -> nothing schedulable

Result: an immortal queue entry with a genuine data hole.

Fix: delete the *old-geometry* block's `<code>_json` manifest entry so the
same label is re-processed with the current layout's bbox (which spans the
hole). This script reports which single deletion closes the coverage.

Usage: python3 scripts_wedged.py [--fix]
"""
import json
import sys
from pathlib import Path

import app  # noqa: E402
import kg_splitter as k  # noqa: E402
import search_index as si  # noqa: E402

MANIFEST = Path('data/austria_processor/zenodo_manifest.json')
QUEUE = Path('data/austria_processor/retry_queue.json')


def main(fix=False):
    mf = json.loads(MANIFEST.read_text())
    entries = mf['entries']
    completed = {c[:-5] for c in entries if c.endswith('_json')}
    conn = si.get_index()._conn()
    queue = json.loads(QUEUE.read_text())
    changed = False

    for code in queue:
        if '-' in code:
            continue
        row = conn.execute('SELECT * FROM kg WHERE kg_code=?', (code,)).fetchone()
        if not row or row['min_lon'] is None:
            continue
        bbox = {'min_lon': row['min_lon'], 'min_lat': row['min_lat'],
                'max_lon': row['max_lon'], 'max_lat': row['max_lat']}
        blocks = k.maybe_split_kg({'kg_code': code, 'kg_name': row['kg_name'],
                                   'bbox': bbox})
        if len(blocks) < 2:
            continue
        schedulable = [b['kg_code'] for b in blocks
                       if b['kg_code'] not in completed]
        done, _ = app._kg_coverage_complete(code, entries)
        if schedulable or done:
            continue

        print(f'WEDGED {code} ({row["kg_name"]}): '
              f'{len(blocks)} blocks in current layout, all labels present, '
              f'oracle says incomplete')
        parent = (bbox['min_lon'], bbox['min_lat'], bbox['max_lon'], bbox['max_lat'])
        bmap = {b['kg_code']: b['bbox'] for b in blocks}
        fam = [c for c in completed if c.startswith(code + '-')]
        for cand in sorted(fam):
            if cand not in bmap:
                continue
            rects = []
            for c in fam:
                if c == cand:
                    continue
                bb = app._read_bbox_cheap(c)
                if bb:
                    rects.append(bb)
            nb = bmap[cand]
            rects.append((nb['min_lon'], nb['min_lat'], nb['max_lon'], nb['max_lat']))
            if app._rects_cover_parent(parent, rects):
                print(f'  -> re-processing {cand} with current geometry '
                      f'closes the hole (delete {cand}_json)')
                if fix:
                    entries.pop(cand + '_json', None)
                    changed = True
                break
        else:
            print('  -> no single-block deletion closes it; inspect manually')

    if fix and changed:
        MANIFEST.write_text(json.dumps(mf, indent=1))
        print('manifest updated')


if __name__ == '__main__':
    main('--fix' in sys.argv)
