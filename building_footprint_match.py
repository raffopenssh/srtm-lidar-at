"""building_footprint_match — lazily join our index buildings to cadastre
footprint_ids (siedler LID-3).

Our ``building_footprints`` come from the same BEV cadastre footprints the
cadastre API serves, so centroids coincide to ~0.2 m. Matching is therefore
nearest-centroid within a size-scaled tolerance + area-ratio sanity check
(no polygons needed — the index only stores centroids).

Per KG, on first demand (query/buildings or prewarm), a background worker
fetches ``/spatial/footprints?bbox=<kg bbox>&geometry=0`` from the cadastre
API (~1.4 MB per KG, once), matches, and writes ``kg_buildings.footprint_id``
+ a ``kg_fp_match`` marker row. Single worker thread, bounded queue — the
primary is bandwidth-metered, so we never fan this out.
"""
from __future__ import annotations

import logging
import math
import queue
import threading
import time

log = logging.getLogger(__name__)

_q: "queue.Queue[str]" = queue.Queue(maxsize=200)
_queued: set = set()
_lock = threading.Lock()
_worker_started = False
REMATCH_AFTER_S = 30 * 86400
MAX_PER_HOUR = 60          # bandwidth guard (~80 MB/h worst case)
_recent: list = []


def _ensure_schema(c):
    try:
        c.execute('ALTER TABLE kg_buildings ADD COLUMN footprint_id TEXT')
    except Exception:
        pass
    c.execute('CREATE TABLE IF NOT EXISTS kg_fp_match (kg_code TEXT PRIMARY KEY, '
              'matched_at REAL, n_matched INTEGER, n_total INTEGER, n_footprints INTEGER)')


def status(idx, kgs):
    """{kg: 'ready'|'warming'|'unmatched'} and enqueue unmatched ones."""
    c = idx._conn()
    _ensure_schema(c)
    out = {}
    now = time.time()
    for kg in kgs:
        row = c.execute('SELECT matched_at FROM kg_fp_match WHERE kg_code=?', (kg,)).fetchone()
        if row and now - float(row[0] or 0) < REMATCH_AFTER_S:
            out[kg] = 'ready'
            continue
        out[kg] = 'warming' if enqueue(kg) else 'unmatched'
    return out


def enqueue(kg: str) -> bool:
    global _worker_started
    with _lock:
        if kg in _queued:
            return True
        # rate guard
        cutoff = time.time() - 3600
        while _recent and _recent[0] < cutoff:
            _recent.pop(0)
        if len(_recent) >= MAX_PER_HOUR:
            return False
        try:
            _q.put_nowait(kg)
        except queue.Full:
            return False
        _queued.add(kg)
        _recent.append(time.time())
        if not _worker_started:
            _worker_started = True
            threading.Thread(target=_worker, name='fp-match', daemon=True).start()
    return True


def _worker():
    import search_index as si
    while True:
        kg = _q.get()
        try:
            match_kg(si.get_index(), kg)
        except Exception as e:
            log.warning('fp-match %s: %s', kg, e)
        finally:
            with _lock:
                _queued.discard(kg)


def match_kg(idx, kg: str) -> dict:
    import cadastre_bridge as cb
    c = idx._conn()
    _ensure_schema(c)
    row = c.execute('SELECT min_lon, min_lat, max_lon, max_lat FROM kg WHERE kg_code=?',
                    (kg,)).fetchone()
    if not row:
        return {'error': 'unknown kg'}
    w, s, e, n = row
    fps = []
    try:
        d = cb.cadastre_proxy('/spatial/footprints', params={
            'west': w, 'south': s, 'east': e, 'north': n,
            'geometry': 0, 'limit': 20000})
        fps = [f for f in d.get('footprints') or [] if f.get('kg_code') == kg]
    except cb.CadastrePending:
        # re-queue later; don't stamp a marker
        threading.Timer(30, enqueue, args=(kg,)).start()
        return {'pending': True}
    blds = c.execute('SELECT rowid, centroid_lon, centroid_lat, footprint_area_sqm '
                     'FROM kg_buildings WHERE kg_code=?', (kg,)).fetchall()
    matched = []
    if fps and blds:
        import numpy as np
        from scipy.spatial import cKDTree
        lat0 = (s + n) / 2
        kx = 111320.0 * math.cos(math.radians(lat0)); ky = 110540.0
        P = np.array([[f['lon'] * kx, f['lat'] * ky] for f in fps])
        tree = cKDTree(P)
        for rid, lon, lat, a in blds:
            if lon is None or lat is None:
                continue
            dist, i = tree.query([lon * kx, lat * ky], k=1)
            f = fps[int(i)]
            fa = float(f.get('area_sqm') or 0)
            a = float(a or 0)
            tol = max(3.0, 0.6 * math.sqrt(max(a, fa, 1.0)))
            ratio = (a / fa) if fa > 0 else 1.0
            if dist <= tol and 0.3 <= ratio <= 3.0:
                matched.append((f['footprint_id'], rid))
    with idx._write_lock:
        c.execute('UPDATE kg_buildings SET footprint_id=NULL WHERE kg_code=?', (kg,))
        if matched:
            c.executemany('UPDATE kg_buildings SET footprint_id=? WHERE rowid=?', matched)
        c.execute('INSERT OR REPLACE INTO kg_fp_match VALUES (?,?,?,?,?)',
                  (kg, time.time(), len(matched), len(blds), len(fps)))
        c.commit()
    log.info('fp-match %s: %d/%d buildings ↔ %d cadastre footprints',
             kg, len(matched), len(blds), len(fps))
    return {'kg': kg, 'matched': len(matched), 'buildings': len(blds), 'footprints': len(fps)}
