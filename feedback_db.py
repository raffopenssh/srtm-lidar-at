"""Feedback + flag persistence layer.

Separate SQLite DB at data/feedback.sqlite so it can be backed up and
rebuilt independently of the search index. Auto-rebuilt of objects+flags
is idempotent on `quality_flags.scan_json`.

Responsibilities
----------------
*   `write_objects_and_flags(...)` — idempotent upsert per KG, called by
    `quality_flags.scan_json` after each new KG JSON is written.
*   `record_feedback(...)` — store a feedback row, attempt to resolve to
    an obj_ref if only a coord is given.
*   `resolve_point(lon, lat, hint=None)` — spatial join against
    `objects_rtree` returning best-match obj_ref + distance.
*   `match_text(text, kg_code=None)` — used by /api/v1/flags/match;
    tokenises a snippet ("102.2m tree") and ranks objects.
*   `flag_stats()`, `list_flags(...)`, `list_feedback(...)`.
*   `effective_overrides(obj_refs)` — per-segment overrides, used by
    select read paths to layer feedback over predictions.

The schema is built lazily on first use. Cheap migrations only.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

log = logging.getLogger(__name__)

DB_PATH = Path('data/feedback.sqlite')
_LOCK = threading.RLock()

# ---------------------------------------------------------------- schema

_SCHEMA = [
    '''CREATE TABLE IF NOT EXISTS objects (
        obj_ref TEXT PRIMARY KEY,
        kg_code TEXT NOT NULL,
        kind TEXT NOT NULL,
        obj_type TEXT,
        centroid_lon REAL,
        centroid_lat REAL,
        area_sqm REAL,
        height_max_m REAL,
        height_mean_m REAL,
        rf_confidence REAL,
        confidence REAL,
        attrs_json TEXT,
        rule_version TEXT,
        computed_at INTEGER NOT NULL
    )''',
    'CREATE INDEX IF NOT EXISTS objects_kg ON objects(kg_code)',
    'CREATE INDEX IF NOT EXISTS objects_type ON objects(obj_type)',
    'CREATE INDEX IF NOT EXISTS objects_height ON objects(height_max_m)',
    'CREATE INDEX IF NOT EXISTS objects_kind ON objects(kind)',
    '''CREATE VIRTUAL TABLE IF NOT EXISTS objects_rtree USING rtree(
        rowid, min_lon, max_lon, min_lat, max_lat
    )''',
    '''CREATE TABLE IF NOT EXISTS flags (
        id INTEGER PRIMARY KEY,
        obj_ref TEXT NOT NULL,
        kg_code TEXT NOT NULL,
        flag_code TEXT NOT NULL,
        severity TEXT NOT NULL,
        message TEXT,
        attrs_json TEXT,
        rule_version TEXT NOT NULL,
        centroid_lon REAL,
        centroid_lat REAL,
        computed_at INTEGER NOT NULL,
        UNIQUE(obj_ref, flag_code, rule_version)
    )''',
    'CREATE INDEX IF NOT EXISTS flags_kg ON flags(kg_code)',
    'CREATE INDEX IF NOT EXISTS flags_code ON flags(flag_code)',
    'CREATE INDEX IF NOT EXISTS flags_severity ON flags(severity)',
    'CREATE INDEX IF NOT EXISTS flags_obj ON flags(obj_ref)',
    '''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY,
        obj_ref TEXT,
        kg_code TEXT,
        point_lon REAL,
        point_lat REAL,
        resolved_obj_ref TEXT,
        resolved_kg_code TEXT,
        resolved_distance_m REAL,
        resolution_status TEXT,
        predicted_type TEXT,
        predicted_attrs_json TEXT,
        kind TEXT NOT NULL,
        corrected_type TEXT,
        corrected_attrs_json TEXT,
        user_id TEXT NOT NULL DEFAULT 'anon',
        user_role TEXT DEFAULT 'student',
        confidence TEXT,
        notes TEXT,
        source_app TEXT NOT NULL,
        context_text TEXT,
        created_at INTEGER NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        superseded_by INTEGER
    )''',
    'CREATE INDEX IF NOT EXISTS fb_obj ON feedback(resolved_obj_ref)',
    'CREATE INDEX IF NOT EXISTS fb_user ON feedback(user_id)',
    'CREATE INDEX IF NOT EXISTS fb_kg ON feedback(resolved_kg_code)',
    'CREATE INDEX IF NOT EXISTS fb_created ON feedback(created_at)',
]


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=NORMAL')
    c.execute('PRAGMA foreign_keys=ON')
    return c

_initialised = False

def ensure_schema(force: bool = False):
    global _initialised
    if _initialised and not force: return
    with _LOCK:
        c = _conn()
        for stmt in _SCHEMA:
            c.execute(stmt)
        c.commit(); c.close()
    _initialised = True


# ---------------------------------------------------------------- writes

def write_objects_and_flags(objects: list, flags: list, rule_version: str):
    """Replace all rows for the KGs covered by `objects`. Idempotent."""
    ensure_schema()
    if not objects and not flags:
        return
    kgs = sorted({o['kg_code'] for o in objects} | {f['kg_code'] for f in flags})
    now = int(time.time())
    with _LOCK:
        c = _conn()
        # Delete prior rows for these KGs (we own them entirely)
        for kg in kgs:
            # remove rtree rows by rowid first
            rids = [r[0] for r in c.execute('SELECT rowid FROM objects WHERE kg_code=?', (kg,))]
            for rid in rids:
                c.execute('DELETE FROM objects_rtree WHERE rowid=?', (rid,))
            c.execute('DELETE FROM objects WHERE kg_code=?', (kg,))
            c.execute('DELETE FROM flags WHERE kg_code=?', (kg,))
        # insert objects
        for o in objects:
            cur = c.execute('''INSERT INTO objects
                (obj_ref, kg_code, kind, obj_type, centroid_lon, centroid_lat,
                 area_sqm, height_max_m, height_mean_m, rf_confidence, confidence,
                 attrs_json, rule_version, computed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (o['obj_ref'], o['kg_code'], o['kind'], o.get('obj_type'),
                 o.get('centroid_lon'), o.get('centroid_lat'),
                 o.get('area_sqm'), o.get('height_max_m'), o.get('height_mean_m'),
                 o.get('rf_confidence'), o.get('confidence'),
                 json.dumps(o.get('attrs') or {}), rule_version, now))
            rowid = cur.lastrowid
            lon = o.get('centroid_lon'); lat = o.get('centroid_lat')
            if lon is not None and lat is not None:
                c.execute('INSERT INTO objects_rtree(rowid, min_lon, max_lon, min_lat, max_lat) VALUES (?,?,?,?,?)',
                          (rowid, lon, lon, lat, lat))
        # insert flags
        for f in flags:
            try:
                c.execute('''INSERT OR IGNORE INTO flags
                    (obj_ref, kg_code, flag_code, severity, message, attrs_json,
                     rule_version, centroid_lon, centroid_lat, computed_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)''',
                    (f['obj_ref'], f['kg_code'], f['flag_code'], f['severity'],
                     f.get('message'), json.dumps(f.get('attrs') or {}),
                     rule_version, f.get('centroid_lon'), f.get('centroid_lat'), now))
            except sqlite3.IntegrityError:
                pass
        c.commit(); c.close()


# ---------------------------------------------------------------- reads

def _row_to_dict(r):
    return dict(r) if r else None


def list_flags(kg_code=None, severity=None, flag_code=None, obj_type=None,
               bbox=None, min_value=None, kind=None,
               limit=200, offset=0, order='severity'):
    ensure_schema()
    where = []; args = []
    if kg_code:    where.append('f.kg_code=?'); args.append(kg_code)
    if severity:   where.append('f.severity=?'); args.append(severity)
    if flag_code:  where.append('f.flag_code=?'); args.append(flag_code)
    if obj_type:   where.append('o.obj_type=?'); args.append(obj_type)
    if kind:       where.append('o.kind=?'); args.append(kind)
    if bbox:
        w,s,e,n = bbox
        where.append('f.centroid_lon BETWEEN ? AND ? AND f.centroid_lat BETWEEN ? AND ?')
        args += [w, e, s, n]
    if min_value is not None:
        where.append("CAST(json_extract(f.attrs_json, '$.value') AS REAL) >= ?")
        args.append(min_value)
    sql = f'''SELECT f.*, o.obj_type, o.kind, o.height_max_m, o.area_sqm,
                     o.rf_confidence, o.confidence
              FROM flags f LEFT JOIN objects o ON f.obj_ref=o.obj_ref'''
    if where: sql += ' WHERE ' + ' AND '.join(where)
    if order == 'severity':
        sql += ''' ORDER BY CASE f.severity
                   WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                   WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END,
                   o.height_max_m DESC NULLS LAST'''
    elif order == 'value':
        sql += " ORDER BY CAST(json_extract(f.attrs_json, '$.value') AS REAL) DESC NULLS LAST"
    elif order == 'recent':
        sql += ' ORDER BY f.computed_at DESC'
    sql += ' LIMIT ? OFFSET ?'; args += [limit, offset]
    c = _conn()
    rows = [dict(r) for r in c.execute(sql, args)]
    c.close()
    for r in rows:
        if r.get('attrs_json'):
            try: r['attrs'] = json.loads(r.pop('attrs_json'))
            except Exception: r['attrs'] = {}
    return rows


def flag_stats():
    ensure_schema()
    c = _conn()
    counts = {}
    counts['total'] = c.execute('SELECT COUNT(*) FROM flags').fetchone()[0]
    counts['by_severity'] = {r['severity']: r['n']
        for r in c.execute('SELECT severity, COUNT(*) AS n FROM flags GROUP BY severity')}
    counts['by_code'] = {r['flag_code']: r['n']
        for r in c.execute('SELECT flag_code, COUNT(*) AS n FROM flags GROUP BY flag_code ORDER BY n DESC')}
    counts['by_type'] = {r['obj_type']: r['n']
        for r in c.execute('''SELECT o.obj_type, COUNT(*) AS n
                              FROM flags f JOIN objects o ON o.obj_ref=f.obj_ref
                              WHERE o.obj_type IS NOT NULL
                              GROUP BY o.obj_type ORDER BY n DESC''')}
    counts['top_kgs'] = [dict(r) for r in c.execute('''
        SELECT kg_code, COUNT(*) AS n,
               SUM(CASE WHEN severity IN ('high','critical') THEN 1 ELSE 0 END) AS n_serious
        FROM flags GROUP BY kg_code ORDER BY n_serious DESC, n DESC LIMIT 20''')]
    counts['n_objects'] = c.execute('SELECT COUNT(*) FROM objects').fetchone()[0]
    c.close()
    return counts


# ---------------------------------------------------------------- spatial resolve

EARTH_R = 6371000.0

def _haversine(lon1, lat1, lon2, lat2):
    if None in (lon1, lat1, lon2, lat2): return None
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2-lat1); dλ = math.radians(lon2-lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def resolve_point(lon: float, lat: float, hint: dict = None,
                  radius_m: float = 50.0, kg_code: str = None,
                  obj_type: str = None, kind: str = None) -> dict:
    """Find the most likely object at (lon, lat).

    Hint may provide: predicted_type, height_max_m, area_sqm, kg_code.
    Returns {'status', 'obj_ref'?, 'distance_m'?, 'candidates': [...]}.
    """
    ensure_schema()
    # rough deg per metre at this latitude
    dlat = radius_m / 111000.0
    dlon = radius_m / (111000.0 * max(0.1, math.cos(math.radians(lat))))
    where = ['min_lon BETWEEN ? AND ?', 'min_lat BETWEEN ? AND ?']
    args = [lon-dlon, lon+dlon, lat-dlat, lat+dlat]
    sql = f'''SELECT o.* FROM objects_rtree r JOIN objects o ON o.rowid=r.rowid
              WHERE {' AND '.join(where)}'''
    if kg_code:   sql += ' AND o.kg_code=?'; args.append(kg_code)
    if obj_type:  sql += ' AND o.obj_type=?'; args.append(obj_type)
    if kind:      sql += ' AND o.kind=?'; args.append(kind)
    c = _conn()
    rows = [dict(r) for r in c.execute(sql, args)]
    c.close()
    if not rows:
        return {'status': 'no_object', 'candidates': [], 'searched_radius_m': radius_m}
    h_target = (hint or {}).get('height_max_m')
    a_target = (hint or {}).get('area_sqm')
    t_target = (hint or {}).get('predicted_type')
    cands = []
    for r in rows:
        d = _haversine(lon, lat, r['centroid_lon'], r['centroid_lat']) or 1e9
        score = d  # lower is better
        # bonus for matching attributes from hint
        if t_target and r.get('obj_type') == t_target: score -= 25
        if h_target is not None and r.get('height_max_m') is not None:
            if abs(r['height_max_m'] - h_target) < 0.5: score -= 25
            elif abs(r['height_max_m'] - h_target) < 2: score -= 10
        if a_target is not None and r.get('area_sqm') is not None:
            if abs(r['area_sqm'] - a_target) < max(1, a_target*0.05): score -= 15
        cands.append({'obj_ref': r['obj_ref'], 'kg_code': r['kg_code'],
                      'kind': r['kind'], 'obj_type': r['obj_type'],
                      'distance_m': round(d, 2),
                      'height_max_m': r.get('height_max_m'),
                      'area_sqm': r.get('area_sqm'),
                      '_score': score})
    cands.sort(key=lambda x: x['_score'])
    best = cands[0]
    if len(cands) > 1 and abs(cands[1]['_score'] - best['_score']) < 5:
        status = 'ambiguous'
    else:
        status = 'resolved'
    for c_ in cands: c_.pop('_score', None)
    return {'status': status, 'obj_ref': best['obj_ref'],
            'kg_code': best['kg_code'], 'distance_m': best['distance_m'],
            'candidates': cands[:5]}


# ---------------------------------------------------------------- text matcher

_NUM_RE = re.compile(r'(\d+(?:[.,]\d+)?)\s*(m²|sqm|sq\.?m|m\^2|m3|m³|m|ha)?', re.I)
_TYPE_HINT_RE = None  # built on first call


def _build_type_hint_re():
    try:
        from object_segmentation import OBJECT_TYPES, GROUP_TYPES
        names = list(OBJECT_TYPES) + list(GROUP_TYPES)
    except Exception:
        names = ['tree', 'shrub', 'hedge', 'roof', 'building', 'mast',
                 'water', 'grass', 'crop', 'road', 'path', 'parking',
                 'orchard', 'vineyard', 'solar_panel', 'greenhouse',
                 'wall', 'fence', 'rock', 'bare_soil', 'fill', 'excavation']
    # longer first so 'wind_turbine' beats 'wind'
    names = sorted(set(n.lower() for n in names), key=len, reverse=True)
    return re.compile(r'\b(' + '|'.join(re.escape(n) for n in names) + r')\b', re.I)


def parse_snippet(text: str) -> dict:
    """Extract a hint dict from a free-text snippet like '102.2m tree'.
    Returns possibly-empty hint dict."""
    global _TYPE_HINT_RE
    if _TYPE_HINT_RE is None: _TYPE_HINT_RE = _build_type_hint_re()
    h = {}
    if not text: return h
    s = text.strip()
    # type
    tm = _TYPE_HINT_RE.search(s)
    if tm: h['predicted_type'] = tm.group(1).lower()
    # numbers w/ units; classify by unit
    for nm in _NUM_RE.finditer(s):
        try:
            val = float(nm.group(1).replace(',', '.'))
        except Exception:
            continue
        unit = (nm.group(2) or '').lower()
        if unit in ('m²','sqm','sq.m','m^2'):
            h.setdefault('area_sqm', val)
        elif unit == 'ha':
            h.setdefault('area_sqm', val*10000)
        elif unit in ('m3','m³'):
            h.setdefault('volume_m3', val)
        elif unit == 'm':
            # height candidate — first plain m wins
            h.setdefault('height_max_m', val)
        else:
            h.setdefault('_value', val)
    return h


def match_text(text: str, kg_code: str = None, lon: float = None, lat: float = None,
               radius_m: float = 200.0, limit: int = 8) -> dict:
    """Best-effort match a free-text snippet to known objects.

    Search order:
      1. If lon/lat given: spatial+attribute (most reliable).
      2. If kg_code given: filter to that KG, rank by attribute match.
      3. Global: rank by attribute match.
    """
    ensure_schema()
    hint = parse_snippet(text or '')
    if lon is not None and lat is not None:
        # spatial path — widen radius for typed text
        return resolve_point(lon, lat, hint=hint, radius_m=radius_m, kg_code=kg_code)
    where = []; args = []
    if kg_code: where.append('kg_code=?'); args.append(kg_code)
    if hint.get('predicted_type'):
        where.append('LOWER(obj_type)=?'); args.append(hint['predicted_type'])
    h = hint.get('height_max_m')
    if h is not None:
        where.append('height_max_m IS NOT NULL AND ABS(height_max_m - ?) < 1.0')
        args.append(h)
    sql = 'SELECT * FROM objects'
    if where: sql += ' WHERE ' + ' AND '.join(where)
    sql += ' ORDER BY '
    if h is not None:
        sql += 'ABS(height_max_m - ?), '
        args = [h] + args  # bind for ORDER BY
        # sqlite binds positionally — reorder: actually shift tail
        # We re-do args properly:
    # rebuild sanely
    where = []; args = []
    if kg_code: where.append('kg_code=?'); args.append(kg_code)
    if hint.get('predicted_type'):
        where.append('LOWER(obj_type)=?'); args.append(hint['predicted_type'])
    if h is not None:
        where.append('height_max_m IS NOT NULL AND ABS(height_max_m - ?) < 1.5')
        args.append(h)
    a = hint.get('area_sqm')
    if a is not None:
        where.append('area_sqm IS NOT NULL AND ABS(area_sqm - ?) < ?')
        args += [a, max(1.0, a*0.05)]
    sql = 'SELECT * FROM objects'
    if where: sql += ' WHERE ' + ' AND '.join(where)
    order_parts = []
    order_args = []
    if h is not None:
        order_parts.append('ABS(height_max_m - ?)')
        order_args.append(h)
    if a is not None:
        order_parts.append('ABS(area_sqm - ?)')
        order_args.append(a)
    order_parts.append('rf_confidence DESC NULLS LAST')
    sql += ' ORDER BY ' + ', '.join(order_parts) + ' LIMIT ?'
    args = args + order_args + [limit]
    c = _conn()
    rows = [dict(r) for r in c.execute(sql, args)]
    c.close()
    cands = []
    for r in rows:
        cands.append({
            'obj_ref': r['obj_ref'], 'kg_code': r['kg_code'],
            'kind': r['kind'], 'obj_type': r['obj_type'],
            'centroid_lon': r['centroid_lon'], 'centroid_lat': r['centroid_lat'],
            'height_max_m': r['height_max_m'], 'area_sqm': r['area_sqm'],
            'rf_confidence': r['rf_confidence'],
        })
    if not cands:
        return {'status': 'no_object', 'hint': hint, 'candidates': []}
    status = 'resolved' if len(cands) == 1 or (
        len(cands) > 1 and h is not None
        and cands[0].get('height_max_m') is not None
        and (cands[1].get('height_max_m') is None
             or abs(cands[0]['height_max_m']-h) + 0.3 < abs(cands[1]['height_max_m']-h))
    ) else 'ambiguous'
    return {'status': status, 'hint': hint,
            'obj_ref': cands[0]['obj_ref'], 'kg_code': cands[0]['kg_code'],
            'candidates': cands}


# ---------------------------------------------------------------- feedback

def record_feedback(payload: dict, user_id: str = 'anon', user_role: str = 'student',
                    source_app: str = 'web') -> dict:
    ensure_schema()
    obj_ref = payload.get('obj_ref')
    kg_code = payload.get('kg_code')
    point = payload.get('point') or {}
    lon = point.get('lon'); lat = point.get('lat')
    context_text = payload.get('context_text') or payload.get('selected_text')

    resolved_obj_ref = obj_ref
    resolved_kg_code = kg_code
    resolved_distance_m = None
    resolution_status = 'resolved' if obj_ref else 'pending'

    if not obj_ref:
        # try resolution: text first if provided, else coord
        hint = payload.get('predicted_attrs') or {}
        if context_text:
            r = match_text(context_text, kg_code=kg_code, lon=lon, lat=lat)
        elif lon is not None and lat is not None:
            r = resolve_point(lon, lat, hint=hint, kg_code=kg_code)
        else:
            r = {'status': 'no_object'}
        resolution_status = r.get('status', 'no_object')
        if r.get('obj_ref'):
            resolved_obj_ref = r['obj_ref']
            resolved_kg_code = r.get('kg_code')
            resolved_distance_m = r.get('distance_m')

    # If we have a resolved ref, fetch its kg_code to be safe
    if resolved_obj_ref and not resolved_kg_code:
        c = _conn()
        row = c.execute('SELECT kg_code FROM objects WHERE obj_ref=?', (resolved_obj_ref,)).fetchone()
        c.close()
        if row: resolved_kg_code = row['kg_code']

    now = int(time.time())
    with _LOCK:
        c = _conn()
        cur = c.execute('''INSERT INTO feedback
            (obj_ref, kg_code, point_lon, point_lat,
             resolved_obj_ref, resolved_kg_code, resolved_distance_m, resolution_status,
             predicted_type, predicted_attrs_json, kind, corrected_type, corrected_attrs_json,
             user_id, user_role, confidence, notes, source_app, context_text, created_at, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'active')''',
            (obj_ref, kg_code, lon, lat,
             resolved_obj_ref, resolved_kg_code, resolved_distance_m, resolution_status,
             payload.get('predicted_type'),
             json.dumps(payload.get('predicted_attrs') or {}),
             payload.get('kind') or 'report',
             payload.get('corrected_type'),
             json.dumps(payload.get('corrected_attrs') or {}),
             user_id, user_role, payload.get('confidence'),
             payload.get('notes'), source_app, context_text, now))
        fb_id = cur.lastrowid
        c.commit(); c.close()
    return {'id': fb_id, 'resolved_obj_ref': resolved_obj_ref,
            'resolved_kg_code': resolved_kg_code,
            'resolved_distance_m': resolved_distance_m,
            'resolution_status': resolution_status}


def list_feedback(kg_code=None, user=None, since=None, status='active',
                  obj_ref=None, limit=200, offset=0):
    ensure_schema()
    where = []; args = []
    if status: where.append('f.status=?'); args.append(status)
    if kg_code: where.append('f.resolved_kg_code=?'); args.append(kg_code)
    if user: where.append('f.user_id=?'); args.append(user)
    if since:
        try: where.append('f.created_at >= ?'); args.append(int(since))
        except Exception: pass
    if obj_ref: where.append('(f.resolved_obj_ref=? OR f.obj_ref=?)'); args += [obj_ref, obj_ref]
    sql = 'SELECT * FROM feedback f'
    if where: sql += ' WHERE ' + ' AND '.join(where)
    sql += ' ORDER BY f.created_at DESC LIMIT ? OFFSET ?'; args += [limit, offset]
    c = _conn()
    rows = [dict(r) for r in c.execute(sql, args)]
    c.close()
    for r in rows:
        for k in ('predicted_attrs_json', 'corrected_attrs_json'):
            if r.get(k):
                try: r[k.replace('_json','')] = json.loads(r.pop(k))
                except Exception: r.pop(k, None)
    return rows


# ---------------------------------------------------------------- effective overrides (transparent in queries)

def effective_overrides(obj_refs: Sequence[str]) -> dict:
    """Return {obj_ref: {effective_type, n_confirms, n_rejects, n_corrections,
    community_verified}} for the requested refs. Empty dict if none.

    Consensus rule: ≥2 students agree on a correction OR ≥1 trusted/admin.
    """
    if not obj_refs: return {}
    ensure_schema()
    qmarks = ','.join(['?'] * len(obj_refs))
    sql = f'''SELECT resolved_obj_ref AS ref, kind, corrected_type, user_role, confidence
              FROM feedback
              WHERE status='active' AND resolved_obj_ref IN ({qmarks})'''
    c = _conn()
    rows = c.execute(sql, list(obj_refs)).fetchall()
    c.close()
    by_ref = {}
    for r in rows:
        d = by_ref.setdefault(r['ref'], {
            'n_confirms': 0, 'n_rejects': 0, 'n_corrections': 0,
            'votes': {}, 'admin_correction': None,
        })
        if r['kind'] == 'confirm':
            d['n_confirms'] += 1
        elif r['kind'] == 'reject':
            d['n_rejects'] += 1
        elif r['kind'] in ('correct_type', 'correct'):
            d['n_corrections'] += 1
            ct = (r['corrected_type'] or '').strip()
            if not ct: continue
            w = 5 if r['user_role'] == 'admin' else (2 if r['user_role'] == 'trusted' else 1)
            d['votes'][ct] = d['votes'].get(ct, 0) + w
            if r['user_role'] in ('admin', 'trusted'):
                d['admin_correction'] = ct
    out = {}
    for ref, d in by_ref.items():
        # pick mode
        majority = None; max_w = 0
        for ct, w in d['votes'].items():
            if w > max_w: majority, max_w = ct, w
        verified = bool(d['admin_correction']) or max_w >= 2
        out[ref] = {
            'effective_type': d['admin_correction'] or (majority if verified else None),
            'community_verified': verified,
            'n_confirms': d['n_confirms'], 'n_rejects': d['n_rejects'],
            'n_corrections': d['n_corrections'],
        }
    return out


# ---------------------------------------------------------------- kg lookup helpers

def kg_with_flag_counts() -> list:
    ensure_schema()
    c = _conn()
    rows = [dict(r) for r in c.execute('''
        SELECT kg_code, COUNT(*) AS n_total,
               SUM(severity='critical') AS n_critical,
               SUM(severity='high')     AS n_high,
               SUM(severity='medium')   AS n_medium,
               SUM(severity='low')      AS n_low
        FROM flags GROUP BY kg_code
    ''')]
    c.close()
    return rows


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    ensure_schema()
    if len(sys.argv) > 1 and sys.argv[1] == 'stats':
        print(json.dumps(flag_stats(), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == 'reset':
        DB_PATH.unlink(missing_ok=True)
        ensure_schema(force=True)
        print('reset')
