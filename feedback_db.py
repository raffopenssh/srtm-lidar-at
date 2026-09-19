"""Feedback + flag persistence layer.

Separate SQLite DB at data/feedback.sqlite so it can be backed up and
rebuilt independently of the search index.

Storage model (schema v2, Sep 2026)
-----------------------------------
Only **flags** and **feedback** are stored. There is deliberately NO
`objects` table any more: the v1 schema mirrored every inspectable object
of every KG (parcels, parcel top objects, buildings, infra ... ~20k rows per
KG) *including* a full `attrs_json` copy of the source record. That made
feedback.sqlite a bloated second copy of `data/austria_processor/json`
(31 GB for 3.4k KGs, i.e. ~75 GB at full fleet -- larger than the JSONs
themselves and 14x the search index).

The KG JSON / v2 blob store *is* the object store. `resolve_point`,
`match_text`, `get_object` locate a KG via the search-index R-tree (or the
kg_code prefix of the obj_ref), decode that one document (small LRU) and
iterate `quality_flags.iter_objects` on it. A point lookup costs one JSON
decode (~50-200 ms cold, ~0 warm) instead of a 30 GB R-tree.

`flags` carries a denormalised copy of the few object columns the read
paths need (obj_type, kind, height, area, confidence, centroid) so listing
/ stats / bbox queries never need the object. Rule attrs collapse to a
single `value REAL` (that's all rules ever emit). `flag_events` is a
bounded audit ring (90 d) and no longer records rule-version-only bumps.

Everything in `flags` / `flag_events` is derived and can be regenerated
from the KG documents at any time:

    python3 feedback_db.py rebuild        # offline rebuild, atomic swap
    python3 quality_flags.py scan-all      # incremental re-scan into live DB

`feedback` + `feedback_events` are the ONLY primary data; `rebuild`
carries them over.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional, Sequence

log = logging.getLogger(__name__)

DB_PATH = Path('data/feedback.sqlite')
JSON_DIR = Path('data/austria_processor/json')
_LOCK = threading.RLock()

SCHEMA_VERSION = 2
FLAG_EVENTS_RETENTION_S = 90 * 86400
_PRUNE_EVERY_N_WRITES = 50

# ---------------------------------------------------------------- schema

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS flags (
        id INTEGER PRIMARY KEY,
        obj_ref TEXT NOT NULL,
        kg_code TEXT NOT NULL,
        flag_code TEXT NOT NULL,
        severity TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 1.0,
        message TEXT,
        value REAL,
        rule_version TEXT NOT NULL,
        obj_type TEXT,
        kind TEXT,
        height_max_m REAL,
        area_sqm REAL,
        rf_confidence REAL,
        confidence REAL,
        centroid_lon REAL,
        centroid_lat REAL,
        computed_at INTEGER NOT NULL,
        UNIQUE(obj_ref, flag_code)
    )""",
    'CREATE INDEX IF NOT EXISTS flags_kg ON flags(kg_code)',
    'CREATE INDEX IF NOT EXISTS flags_code ON flags(flag_code)',
    """CREATE TABLE IF NOT EXISTS flag_events (
        id INTEGER PRIMARY KEY,
        ts INTEGER NOT NULL,
        kind TEXT NOT NULL,           -- 'created'|'removed'|'changed'
        obj_ref TEXT NOT NULL,
        kg_code TEXT NOT NULL,
        flag_code TEXT NOT NULL,
        severity TEXT,
        weight REAL,
        rule_version TEXT
    )""",
    'CREATE INDEX IF NOT EXISTS fe_obj ON flag_events(obj_ref)',
    'CREATE INDEX IF NOT EXISTS fe_ts ON flag_events(ts)',
    """CREATE TABLE IF NOT EXISTS feedback_events (
        id INTEGER PRIMARY KEY,
        ts INTEGER NOT NULL,
        feedback_id INTEGER NOT NULL,
        kind TEXT NOT NULL,           -- 'submit'|'supersede'|'withdraw'|'resolve'
        obj_ref TEXT,
        kg_code TEXT,
        action TEXT,                  -- confirm|reject|correct_type|...
        corrected_type TEXT,
        user_id TEXT,
        user_role TEXT,
        weight REAL,
        notes TEXT
    )""",
    'CREATE INDEX IF NOT EXISTS fbe_obj ON feedback_events(obj_ref)',
    'CREATE INDEX IF NOT EXISTS fbe_kg  ON feedback_events(kg_code)',
    'CREATE INDEX IF NOT EXISTS fbe_ts  ON feedback_events(ts)',
    'CREATE INDEX IF NOT EXISTS fbe_fid ON feedback_events(feedback_id)',
    """CREATE TABLE IF NOT EXISTS feedback (
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
    )""",
    'CREATE INDEX IF NOT EXISTS fb_obj ON feedback(resolved_obj_ref)',
    'CREATE INDEX IF NOT EXISTS fb_user ON feedback(user_id)',
    'CREATE INDEX IF NOT EXISTS fb_kg ON feedback(resolved_kg_code)',
    'CREATE INDEX IF NOT EXISTS fb_created ON feedback(created_at)',
]


def _conn(path: Path = None) -> sqlite3.Connection:
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('PRAGMA synchronous=NORMAL')
    c.execute('PRAGMA foreign_keys=ON')
    return c


@contextlib.contextmanager
def _conn_ctx(path: Path = None):
    """Connection that is ALWAYS rolled back + closed, even on exception.

    Aug 2026 incident: `write_objects_and_flags` raised mid-write on a
    duplicate obj_ref and never reached its `c.commit(); c.close()`. The
    abandoned connection kept an open write transaction inside a long-lived
    gunicorn worker, so every later feedback write returned
    "database is locked" until srv was restarted. Any code path that WRITES
    must go through this.
    """
    c = _conn(path)
    try:
        yield c
        c.commit()
    except BaseException:
        try:
            c.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            c.close()
        except Exception:
            pass

_initialised = False
_legacy_schema = False


def is_legacy_schema(c: sqlite3.Connection) -> bool:
    """True when the DB still carries the v1 fat `objects` table."""
    return bool(c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='objects'").fetchone())


def _create_schema(c: sqlite3.Connection):
    # auto_vacuum must be set before any table exists to take effect on a
    # fresh file; harmless no-op on an existing one.
    c.execute('PRAGMA auto_vacuum=INCREMENTAL')
    for stmt in _SCHEMA:
        c.execute(stmt)
    c.execute(f'PRAGMA user_version={SCHEMA_VERSION}')


def ensure_schema(force: bool = False):
    global _initialised, _legacy_schema
    if _initialised and not force:
        if not _legacy_schema: return
        # Legacy mode was decided at boot; `feedback_db.py rebuild` swaps the
        # file underneath a running gunicorn, so re-probe (one sqlite_master
        # lookup) until we observe the v2 layout -- otherwise every query
        # keeps joining the now-dropped `objects` table until a restart.
        with _LOCK, _conn_ctx() as c:
            if is_legacy_schema(c): return
            _legacy_schema = False
            log.info('feedback_db: v2 schema detected at %s (rebuild landed) -- '
                     'leaving legacy compatibility mode', DB_PATH)
        return
    with _LOCK, _conn_ctx() as c:
        if is_legacy_schema(c):
            # v1 layout. Don't migrate in-place (rewriting a 30 GB file at
            # boot is not an option) -- keep serving from the legacy tables
            # via the compatibility shims below and nag until an operator
            # runs `python3 feedback_db.py rebuild`.
            _legacy_schema = True
            cols = {r[1] for r in c.execute('PRAGMA table_info(flags)')}
            if 'weight' not in cols:
                c.execute('ALTER TABLE flags ADD COLUMN weight REAL NOT NULL DEFAULT 1.0')
            for stmt in _SCHEMA:
                if 'TABLE IF NOT EXISTS feedback' in stmt or stmt.startswith('CREATE INDEX IF NOT EXISTS fb'):
                    c.execute(stmt)
            log.warning('feedback_db: legacy v1 schema (objects table) at %s -- '
                        'run `python3 feedback_db.py rebuild` to shrink it', DB_PATH)
        else:
            _legacy_schema = False
            _create_schema(c)
    _initialised = True


# ---------------------------------------------------------------- object access (from KG documents)

_DOC_CACHE_MAX = 8
_doc_cache: 'OrderedDict[str, tuple]' = OrderedDict()
_doc_lock = threading.Lock()


def _load_doc(code: str) -> Optional[dict]:
    """Decoded KG document for a plain or block code, or None."""
    try:
        import kg_v2_store
        d = kg_v2_store.get(code)
        if d is not None:
            return d
    except Exception:
        pass
    jp = JSON_DIR / f'{code}.json'
    if jp.exists():
        try:
            return json.loads(jp.read_text())
        except Exception as e:
            log.warning('feedback_db: cannot read %s: %s', jp, e)
    return None


def _doc_mtime(code: str) -> float:
    try:
        return (JSON_DIR / f'{code}.json').stat().st_mtime
    except OSError:
        return 0.0


def objects_for_code(code: str) -> list:
    """All inspectable objects (quality_flags.iter_objects) for one doc code.
    Small LRU keyed on code, invalidated by JSON mtime."""
    mt = _doc_mtime(code)
    with _doc_lock:
        hit = _doc_cache.get(code)
        if hit and hit[0] == mt:
            _doc_cache.move_to_end(code)
            return hit[1]
    doc = _load_doc(code)
    if doc is None:
        return []
    from quality_flags import iter_objects
    objs = list(iter_objects(doc, code))
    with _doc_lock:
        _doc_cache[code] = (mt, objs)
        while len(_doc_cache) > _DOC_CACHE_MAX:
            _doc_cache.popitem(last=False)
    return objs


def doc_codes_for_kg(kg_code: str) -> list:
    """Doc codes that hold objects for a parent KG: the plain code plus any
    split blocks ('63304-south'). Accepts a block code as-is."""
    kg_code = str(kg_code)
    out = []
    if (JSON_DIR / f'{kg_code}.json').exists():
        out.append(kg_code)
    if '-' not in kg_code:
        for jp in JSON_DIR.glob(f'{kg_code}-*.json'):
            out.append(jp.stem)
    try:
        import kg_v2_store
        if kg_v2_store.has(kg_code) and kg_code not in out:
            out.append(kg_code)
        if '-' not in kg_code:
            for c in kg_v2_store.codes_for_parent(kg_code):
                if c not in out:
                    out.append(c)
    except Exception:
        pass
    return out


def _kgs_near(lon: float, lat: float, radius_m: float) -> list:
    """Parent KG codes whose bbox intersects the search square."""
    dlat = radius_m / 111000.0
    dlon = radius_m / (111000.0 * max(0.1, math.cos(math.radians(lat))))
    try:
        import search_index
        idx = search_index.get_index()
        res = idx.query_bbox(lon - dlon, lat - dlat, lon + dlon, lat + dlat, limit=50)
        return [r['kg_code'] for r in res.get('results', []) if r.get('kg_code')]
    except Exception as e:
        log.debug('feedback_db._kgs_near: %s', e)
        return []


def get_object(obj_ref: str) -> Optional[dict]:
    """Object record (with full `attrs`) for an obj_ref, read from its KG doc."""
    if not obj_ref or ':' not in obj_ref:
        return None
    code = obj_ref.split(':', 1)[0]
    for o in objects_for_code(code):
        if o['obj_ref'] == obj_ref:
            return o
    return None


def _obj_public(o: dict) -> dict:
    return {k: o.get(k) for k in (
        'obj_ref', 'kg_code', 'kind', 'obj_type', 'centroid_lon', 'centroid_lat',
        'area_sqm', 'height_max_m', 'height_mean_m', 'rf_confidence', 'confidence')}


# ---------------------------------------------------------------- writes

_write_counter = 0


def _flag_value(f: dict):
    a = f.get('attrs') or {}
    v = a.get('value')
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def write_objects_and_flags(objects: list, flags: list, rule_version: str,
                            db_path: Path = None):
    """Replace all flag rows for the KGs covered by `objects`/`flags`.
    Idempotent. `objects` is used only to denormalise object columns onto
    the flag rows -- objects themselves are NOT stored (see module doc)."""
    global _write_counter
    if db_path is None:
        ensure_schema()
        if _legacy_schema:
            return _write_legacy(objects, flags, rule_version)
    if not objects and not flags:
        return
    kgs = sorted({o['kg_code'] for o in objects} | {f['kg_code'] for f in flags})
    now = int(time.time())
    by_ref = {o['obj_ref']: o for o in objects}
    with _LOCK, _conn_ctx(db_path) as c:
        prior = {}
        for kg in kgs:
            for r in c.execute('SELECT obj_ref, kg_code, flag_code, severity, weight '
                               'FROM flags WHERE kg_code=?', (kg,)):
                prior[(r['obj_ref'], r['flag_code'])] = dict(r)
            c.execute('DELETE FROM flags WHERE kg_code=?', (kg,))
        from quality_flags import SEV_WEIGHT  # local import to avoid cycle
        seen = set()
        rows = []
        events = []
        for f in flags:
            key = (f['obj_ref'], f['flag_code'])
            if key in seen:
                continue
            seen.add(key)
            w = f.get('weight') or SEV_WEIGHT.get(f.get('severity', 'low'), 1.0)
            o = by_ref.get(f['obj_ref']) or {}
            rows.append((
                f['obj_ref'], f['kg_code'], f['flag_code'], f['severity'], w,
                f.get('message'), _flag_value(f), rule_version,
                o.get('obj_type'), o.get('kind'), o.get('height_max_m'),
                o.get('area_sqm'), o.get('rf_confidence'), o.get('confidence'),
                f.get('centroid_lon', o.get('centroid_lon')),
                f.get('centroid_lat', o.get('centroid_lat')), now))
            old = prior.get(key)
            ev_kind = 'created' if not old else (
                'changed' if old.get('severity') != f.get('severity') else None)
            if ev_kind:
                events.append((now, ev_kind, f['obj_ref'], f['kg_code'], f['flag_code'],
                               f.get('severity'), w, rule_version))
        for key, old in prior.items():
            if key in seen: continue
            events.append((now, 'removed', key[0], old.get('kg_code') or '', key[1],
                           old.get('severity'), old.get('weight'), rule_version))
        c.executemany("""INSERT OR IGNORE INTO flags
            (obj_ref, kg_code, flag_code, severity, weight, message, value, rule_version,
             obj_type, kind, height_max_m, area_sqm, rf_confidence, confidence,
             centroid_lon, centroid_lat, computed_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        # A fresh DB (rebuild) has no prior state: every flag would be a
        # 'created' event, i.e. pure noise. Only record deltas.
        if prior:
            c.executemany("""INSERT INTO flag_events
                (ts, kind, obj_ref, kg_code, flag_code, severity, weight, rule_version)
                VALUES (?,?,?,?,?,?,?,?)""", events)
        _write_counter += 1
        if _write_counter % _PRUNE_EVERY_N_WRITES == 0:
            c.execute('DELETE FROM flag_events WHERE ts < ?', (now - FLAG_EVENTS_RETENTION_S,))
            c.execute('PRAGMA incremental_vacuum(4096)')
    with _doc_lock:
        for kg in kgs:
            _doc_cache.pop(kg, None)


def _write_legacy(objects: list, flags: list, rule_version: str):
    """v1 write path (objects table present). Kept so a not-yet-rebuilt DB
    keeps working; still drops attrs_json to stop the bleeding."""
    kgs = sorted({o['kg_code'] for o in objects} | {f['kg_code'] for f in flags})
    now = int(time.time())
    objects = list({o['obj_ref']: o for o in objects}.values())
    with _LOCK, _conn_ctx() as c:
        for kg in kgs:
            rids = [r[0] for r in c.execute('SELECT rowid FROM objects WHERE kg_code=?', (kg,))]
            for rid in rids:
                c.execute('DELETE FROM objects_rtree WHERE rowid=?', (rid,))
            c.execute('DELETE FROM objects WHERE kg_code=?', (kg,))
            c.execute('DELETE FROM flags WHERE kg_code=?', (kg,))
        for o in objects:
            cur = c.execute("""INSERT INTO objects
                (obj_ref, kg_code, kind, obj_type, centroid_lon, centroid_lat,
                 area_sqm, height_max_m, height_mean_m, rf_confidence, confidence,
                 attrs_json, rule_version, computed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,?,?)""",
                (o['obj_ref'], o['kg_code'], o['kind'], o.get('obj_type'),
                 o.get('centroid_lon'), o.get('centroid_lat'),
                 o.get('area_sqm'), o.get('height_max_m'), o.get('height_mean_m'),
                 o.get('rf_confidence'), o.get('confidence'), rule_version, now))
            lon = o.get('centroid_lon'); lat = o.get('centroid_lat')
            if lon is not None and lat is not None:
                c.execute('INSERT INTO objects_rtree(rowid, min_lon, max_lon, min_lat, max_lat) VALUES (?,?,?,?,?)',
                          (cur.lastrowid, lon, lon, lat, lat))
        from quality_flags import SEV_WEIGHT
        for f in flags:
            w = f.get('weight') or SEV_WEIGHT.get(f.get('severity', 'low'), 1.0)
            c.execute("""INSERT OR IGNORE INTO flags
                (obj_ref, kg_code, flag_code, severity, weight, message, attrs_json,
                 rule_version, centroid_lon, centroid_lat, computed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (f['obj_ref'], f['kg_code'], f['flag_code'], f['severity'], w,
                 f.get('message'), json.dumps(f.get('attrs') or {}),
                 rule_version, f.get('centroid_lon'), f.get('centroid_lat'), now))


# ---------------------------------------------------------------- reads

def _row_to_dict(r):
    return dict(r) if r else None


def list_flags(kg_code=None, severity=None, flag_code=None, obj_type=None,
               bbox=None, min_value=None, kind=None, obj_ref=None,
               limit=200, offset=0, order='severity'):
    ensure_schema()
    legacy = _legacy_schema
    # v2: object columns are denormalised onto the flag row; v1: join objects.
    o = 'o.' if legacy else 'f.'
    val = "CAST(json_extract(f.attrs_json, '$.value') AS REAL)" if legacy else 'f.value'
    where = []; args = []
    if obj_ref:    where.append('f.obj_ref=?'); args.append(obj_ref)
    if kg_code:    where.append('f.kg_code=?'); args.append(kg_code)
    if severity:   where.append('f.severity=?'); args.append(severity)
    if flag_code:  where.append('f.flag_code=?'); args.append(flag_code)
    if obj_type:   where.append(f'{o}obj_type=?'); args.append(obj_type)
    if kind:       where.append(f'{o}kind=?'); args.append(kind)
    if bbox:
        w,s,e,n = bbox
        where.append('f.centroid_lon BETWEEN ? AND ? AND f.centroid_lat BETWEEN ? AND ?')
        args += [w, e, s, n]
    if min_value is not None:
        where.append(f'{val} >= ?'); args.append(min_value)
    if legacy:
        sql = """SELECT f.*, o.obj_type, o.kind, o.height_max_m, o.area_sqm,
                        o.rf_confidence, o.confidence
                 FROM flags f LEFT JOIN objects o ON f.obj_ref=o.obj_ref"""
    else:
        sql = 'SELECT f.* FROM flags f'
    if where: sql += ' WHERE ' + ' AND '.join(where)
    if order == 'severity':
        sql += f""" ORDER BY CASE f.severity
                   WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                   WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END,
                   {o}height_max_m DESC NULLS LAST"""
    elif order == 'value':
        sql += f' ORDER BY {val} DESC NULLS LAST'
    elif order == 'recent':
        sql += ' ORDER BY f.computed_at DESC'
    sql += ' LIMIT ? OFFSET ?'; args += [limit, offset]
    c = _conn()
    rows = [dict(r) for r in c.execute(sql, args)]
    c.close()
    for r in rows:
        if 'attrs_json' in r:
            try: r['attrs'] = json.loads(r.pop('attrs_json') or '{}')
            except Exception: r['attrs'] = {}
        else:
            v = r.pop('value', None)
            r['attrs'] = {'value': v} if v is not None else {}
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
    if _legacy_schema:
        by_type_sql = """SELECT o.obj_type, COUNT(*) AS n
                         FROM flags f JOIN objects o ON o.obj_ref=f.obj_ref
                         WHERE o.obj_type IS NOT NULL
                         GROUP BY o.obj_type ORDER BY n DESC"""
    else:
        by_type_sql = """SELECT obj_type, COUNT(*) AS n FROM flags
                         WHERE obj_type IS NOT NULL GROUP BY obj_type ORDER BY n DESC"""
    counts['by_type'] = {r['obj_type']: r['n'] for r in c.execute(by_type_sql)}
    counts['top_kgs'] = [dict(r) for r in c.execute("""
        SELECT kg_code, COUNT(*) AS n,
               SUM(CASE WHEN severity IN ('high','critical') THEN 1 ELSE 0 END) AS n_serious
        FROM flags GROUP BY kg_code ORDER BY n_serious DESC, n DESC LIMIT 20""")]
    counts['n_flagged_objects'] = c.execute('SELECT COUNT(DISTINCT obj_ref) FROM flags').fetchone()[0]
    counts['n_kgs'] = c.execute('SELECT COUNT(DISTINCT kg_code) FROM flags').fetchone()[0]
    try:
        counts['db_bytes'] = os.path.getsize(DB_PATH)
    except OSError:
        pass
    counts['schema'] = 'v1-legacy' if _legacy_schema else f'v{SCHEMA_VERSION}'
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


def _objects_near(lon: float, lat: float, radius_m: float, kg_code: str = None) -> list:
    """Objects of every KG document that could contain (lon, lat). Parent KG
    codes are accepted even when objects were ingested under a split-block
    code like '63304-south'."""
    kgs = [str(kg_code)] if kg_code else _kgs_near(lon, lat, radius_m)
    codes = []
    for kg in kgs:
        for code in doc_codes_for_kg(kg):
            if code not in codes:
                codes.append(code)
    out = []
    for code in codes:
        out.extend(objects_for_code(code))
    return out


def resolve_point(lon: float, lat: float, hint: dict = None,
                  radius_m: float = 50.0, kg_code: str = None,
                  obj_type: str = None, kind: str = None) -> dict:
    """Find the most likely object at (lon, lat).

    Hint may provide: predicted_type, height_max_m, area_sqm, kg_code.
    Returns {'status', 'obj_ref'?, 'distance_m'?, 'candidates': [...]}.
    """
    # rough deg per metre at this latitude
    dlat = radius_m / 111000.0
    dlon = radius_m / (111000.0 * max(0.1, math.cos(math.radians(lat))))
    lo_lon, hi_lon, lo_lat, hi_lat = lon-dlon, lon+dlon, lat-dlat, lat+dlat
    rows = [o for o in _objects_near(lon, lat, radius_m, kg_code)
            if o.get('centroid_lon') is not None and o.get('centroid_lat') is not None
            and lo_lon <= o['centroid_lon'] <= hi_lon and lo_lat <= o['centroid_lat'] <= hi_lat
            and (not obj_type or o.get('obj_type') == obj_type)
            and (not kind or o.get('kind') == kind)]
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
    cands = _dedup_candidates(cands)
    best = cands[0]
    if len(cands) > 1 and abs((cands[1].get('_score') or 0) - (best.get('_score') or 0)) < 5:
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


_KIND_PRIORITY = {
    'building': 0, 'parcel': 1, 'top_tree': 2, 'top_obj': 3,
    'top_by_type': 4, 'new_building': 5, 'infra': 6,
}

def _dedup_candidates(cands: list, *, coord_decimals: int = 5,
                      h_round: float = 0.5, a_round: float = 5.0) -> list:
    """Collapse multiple obj_refs that point to the same physical object.

    The pipeline emits the same segment as `top_tree`, `top_obj`, and
    `top_by_type:<type>:rank` simultaneously — they share kg + centroid +
    height + area but have distinct refs. For flagging, the user wants
    *one* row to act on; we keep the most informative kind (lowest
    _KIND_PRIORITY) and stash the duplicate refs under `aliases`.
    """
    buckets = {}
    order = []
    for c in cands:
        lon = c.get('centroid_lon'); lat = c.get('centroid_lat')
        h = c.get('height_max_m'); a = c.get('area_sqm')
        key = (
            c.get('kg_code') or '',
            c.get('obj_type') or '',
            None if lon is None else round(lon, coord_decimals),
            None if lat is None else round(lat, coord_decimals),
            None if h is None else round(h / h_round) * h_round,
            None if a is None else round(a / a_round) * a_round,
        )
        if key not in buckets:
            buckets[key] = c
            c['aliases'] = []
            order.append(key)
        else:
            keep = buckets[key]
            new_p = _KIND_PRIORITY.get(c.get('kind'), 99)
            old_p = _KIND_PRIORITY.get(keep.get('kind'), 99)
            if new_p < old_p:
                c['aliases'] = keep.get('aliases', []) + [keep.get('obj_ref')]
                buckets[key] = c
            else:
                keep.setdefault('aliases', []).append(c.get('obj_ref'))
    return [buckets[k] for k in order]


def match_text(text: str, kg_code: str = None, lon: float = None, lat: float = None,
               radius_m: float = 200.0, limit: int = 8) -> dict:
    """Best-effort match a free-text snippet to known objects.

    Search order:
      1. If lon/lat given: spatial+attribute (most reliable).
      2. If kg_code given: filter to that KG, rank by attribute match.
      3. Global: rank by attribute match.
    """
    hint = parse_snippet(text or '')
    if lon is not None and lat is not None:
        # spatial path — widen radius for typed text
        return resolve_point(lon, lat, hint=hint, radius_m=radius_m, kg_code=kg_code)
    h = hint.get('height_max_m')
    a = hint.get('area_sqm')
    t = hint.get('predicted_type')

    def _ok(o):
        if t and (o.get('obj_type') or '').lower() != t: return False
        if h is not None and (o.get('height_max_m') is None or abs(o['height_max_m'] - h) >= 1.5): return False
        if a is not None and (o.get('area_sqm') is None or abs(o['area_sqm'] - a) >= max(1.0, a*0.05)): return False
        return True

    def _rank(o):
        k = []
        if h is not None: k.append(abs((o.get('height_max_m') or 0) - h))
        if a is not None: k.append(abs((o.get('area_sqm') or 0) - a))
        k.append(-(o.get('rf_confidence') or 0))
        return tuple(k)

    if kg_code:
        # KG-scoped: walk the KG document(s) directly.
        pool = []
        for code in doc_codes_for_kg(kg_code):
            pool.extend(objects_for_code(code))
        rows = sorted((o for o in pool if _ok(o)), key=_rank)[:limit]
    else:
        # Global text match without a location: there is no fleet-wide
        # object table any more, so rank over *flagged* objects (which carry
        # the denormalised attributes) -- those are the ones a user is
        # looking at in a report anyway.
        where = []; args = []
        if t: where.append('LOWER(obj_type)=?'); args.append(t)
        if h is not None:
            where.append('height_max_m IS NOT NULL AND ABS(height_max_m - ?) < 1.5'); args.append(h)
        if a is not None:
            where.append('area_sqm IS NOT NULL AND ABS(area_sqm - ?) < ?'); args += [a, max(1.0, a*0.05)]
        sql = ('SELECT obj_ref, kg_code, kind, obj_type, centroid_lon, centroid_lat, '
               'height_max_m, area_sqm, rf_confidence FROM flags')
        if where: sql += ' WHERE ' + ' AND '.join(where)
        order = []
        oargs = []
        if h is not None: order.append('ABS(height_max_m - ?)'); oargs.append(h)
        if a is not None: order.append('ABS(area_sqm - ?)'); oargs.append(a)
        order.append('rf_confidence DESC NULLS LAST')
        sql += ' GROUP BY obj_ref ORDER BY ' + ', '.join(order) + ' LIMIT ?'
        c = _conn()
        rows = [dict(r) for r in c.execute(sql, args + oargs + [limit])]
        c.close()
    cands = []
    for r in rows:
        cands.append({
            'obj_ref': r['obj_ref'], 'kg_code': r['kg_code'],
            'kind': r.get('kind'), 'obj_type': r.get('obj_type'),
            'centroid_lon': r.get('centroid_lon'), 'centroid_lat': r.get('centroid_lat'),
            'height_max_m': r.get('height_max_m'), 'area_sqm': r.get('area_sqm'),
            'rf_confidence': r.get('rf_confidence'),
        })
    cands = _dedup_candidates(cands)
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

_KNOWN_FEEDBACK_KEYS = {
    'obj_ref', 'kg_code', 'point', 'context_text', 'selected_text', 'kind',
    'predicted_type', 'predicted_attrs', 'corrected_type', 'corrected_attrs',
    'value', 'confidence', 'notes', 'note', 'user', 'token', 'source_app',
}


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
        if ':' in resolved_obj_ref:
            resolved_kg_code = resolved_obj_ref.split(':', 1)[0]

    # Accept both `notes` (canonical) and `note` (documented alias) — silently
    # dropping the payload cost a reporter their whole bug report (Jun 2026).
    fb_notes = payload.get('notes')
    if fb_notes in (None, ''):
        fb_notes = payload.get('note')

    # `value` is the documented key for a corrected type; `corrected_type` is
    # what the schema calls it. Accept both.
    fb_corrected = payload.get('corrected_type') or payload.get('value')

    now = int(time.time())
    role_w = {'admin': 5.0, 'trusted': 2.0, 'student': 1.0, 'anon': 0.5}.get(user_role, 1.0)
    fb_kind = payload.get('kind') or 'report'
    with _LOCK, _conn_ctx() as c:
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
             fb_kind,
             fb_corrected,
             json.dumps(payload.get('corrected_attrs') or {}),
             user_id, user_role, payload.get('confidence'),
             fb_notes, source_app, context_text, now))
        fb_id = cur.lastrowid
        c.execute('''INSERT INTO feedback_events
            (ts, feedback_id, kind, obj_ref, kg_code, action, corrected_type,
             user_id, user_role, weight, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
            (now, fb_id, 'submit', resolved_obj_ref or obj_ref,
             resolved_kg_code or kg_code, fb_kind,
             fb_corrected,
             user_id, user_role, role_w, fb_notes))
    out = {'id': fb_id, 'resolved_obj_ref': resolved_obj_ref,
           'resolved_kg_code': resolved_kg_code,
           'resolved_distance_m': resolved_distance_m,
           'resolution_status': resolution_status,
           'notes_stored': bool(fb_notes)}
    # Surface silently-dropped payloads instead of a bare ok:true.
    warnings = []
    unknown = [k for k in payload
               if k not in _KNOWN_FEEDBACK_KEYS]
    if unknown:
        warnings.append(f"ignored unknown field(s): {', '.join(sorted(unknown))}")
    if fb_kind == 'correct_type' and not fb_corrected:
        warnings.append("kind=correct_type without 'value'/'corrected_type' "
                        "— no correction recorded")
    if fb_kind in ('reject', 'report_missing') and not (fb_notes or context_text):
        warnings.append(f"kind={fb_kind} carries no note/context_text — "
                        "nothing but the vote was stored")
    if warnings:
        out['warnings'] = warnings
    return out


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

def predict_action_impact(obj_ref: str, kind: str = 'reject',
                          corrected_type: str = None,
                          user_role: str = 'student') -> dict:
    """Forecast what would happen if a user submitted `kind` on `obj_ref` now.

    Returns a dict the UI can display under each action option:
        weight_added, total_after, would_verify, current_consensus,
        flips_outcome, projected_effective_type, projected_status,
        rationale.
    """
    ensure_schema()
    role_w = {'admin': 5.0, 'trusted': 2.0, 'student': 1.0, 'anon': 0.5}.get(user_role, 1.0)
    cur = effective_overrides([obj_ref]).get(obj_ref) or {
        'effective_type': None, 'community_verified': False,
        'n_confirms': 0, 'n_rejects': 0, 'n_corrections': 0}
    c = _conn()
    obj = c.execute('SELECT obj_type FROM flags WHERE obj_ref=? LIMIT 1', (obj_ref,)).fetchone() \
        if not _legacy_schema else None
    flag_w = c.execute('SELECT COALESCE(SUM(weight), 0) AS w, COUNT(*) AS n '
                       'FROM flags WHERE obj_ref=?', (obj_ref,)).fetchone()
    c.close()
    predicted_type = obj['obj_type'] if obj else None
    if predicted_type is None:
        o = get_object(obj_ref)
        predicted_type = o.get('obj_type') if o else None
    out = {
        'current': dict(cur),
        'flag_weight': float(flag_w['w']) if flag_w else 0.0,
        'n_flags': int(flag_w['n']) if flag_w else 0,
        'role_weight': role_w,
        'kind': kind,
        'predicted_type': predicted_type,
    }
    if kind == 'confirm':
        out['rationale'] = (
            'Adds weight to the existing prediction; once two students or one '
            'trusted reviewer confirm, the prediction is locked as “community-verified”.'
        )
        out['n_confirms_after'] = cur['n_confirms'] + 1
        out['flips_outcome'] = False
    elif kind == 'reject':
        out['rationale'] = (
            'Records that the prediction is wrong but does not (yet) supply '
            'a replacement. Two rejections downgrade quality; the segment '
            'enters the resampling pool.'
        )
        out['n_rejects_after'] = cur['n_rejects'] + 1
        out['flips_outcome'] = (cur['n_rejects'] + 1) >= 2 and not cur['community_verified']
    elif kind in ('correct_type', 'correct'):
        out['rationale'] = (
            f"Suggests the correct type is '{corrected_type or '?'}'. "
            'Two students agreeing OR one trusted reviewer makes it the '
            'community-effective type — used in queries with '
            '`use_overrides=true` and added to the resampling pool.'
        )
        # current votes
        c = _conn()
        votes = {}
        for r in c.execute('''SELECT corrected_type, user_role FROM feedback
                              WHERE status='active' AND resolved_obj_ref=?
                              AND kind IN ('correct_type','correct')''', (obj_ref,)):
            ct = r['corrected_type']; ur = r['user_role'] or 'student'
            w = {'admin': 5, 'trusted': 2}.get(ur, 1)
            votes[ct] = votes.get(ct, 0) + w
        c.close()
        votes[corrected_type or '?'] = votes.get(corrected_type or '?', 0) + (5 if user_role=='admin' else 2 if user_role=='trusted' else 1)
        winner = max(votes.items(), key=lambda x: x[1]) if votes else (None, 0)
        out['projected_votes'] = votes
        out['projected_effective_type'] = winner[0] if winner[1] >= 2 or user_role in ('admin','trusted') else None
        out['flips_outcome'] = (
            out['projected_effective_type'] not in (None, predicted_type)
        )
    else:
        out['rationale'] = 'Recorded for review; no automated effect.'
        out['flips_outcome'] = False
    return out


def list_flag_events(kg_code=None, obj_ref=None, since=None,
                     kind=None, limit=200, offset=0):
    ensure_schema()
    where = []; args = []
    if kg_code: where.append('kg_code=?'); args.append(kg_code)
    if obj_ref: where.append('obj_ref=?'); args.append(obj_ref)
    if since:
        try: where.append('ts >= ?'); args.append(int(since))
        except Exception: pass
    if kind: where.append('kind=?'); args.append(kind)
    sql = 'SELECT * FROM flag_events'
    if where: sql += ' WHERE ' + ' AND '.join(where)
    sql += ' ORDER BY ts DESC LIMIT ? OFFSET ?'; args += [limit, offset]
    c = _conn(); rows = [dict(r) for r in c.execute(sql, args)]; c.close()
    return rows


def list_feedback_events(kg_code=None, obj_ref=None, since=None,
                         user_id=None, limit=200, offset=0):
    ensure_schema()
    where = []; args = []
    if kg_code: where.append('kg_code=?'); args.append(kg_code)
    if obj_ref: where.append('obj_ref=?'); args.append(obj_ref)
    if user_id: where.append('user_id=?'); args.append(user_id)
    if since:
        try: where.append('ts >= ?'); args.append(int(since))
        except Exception: pass
    sql = 'SELECT * FROM feedback_events'
    if where: sql += ' WHERE ' + ' AND '.join(where)
    sql += ' ORDER BY ts DESC LIMIT ? OFFSET ?'; args += [limit, offset]
    c = _conn(); rows = [dict(r) for r in c.execute(sql, args)]; c.close()
    return rows


def object_aggregates(obj_refs: Sequence[str]) -> dict:
    """For each obj_ref return aggregate flag weight + count + max severity.

    Useful in /flags and /flags/object responses so callers can sort or
    filter by 'agreement' (sum of weights = how many independent rules
    flagged this object, severity-weighted).
    """
    if not obj_refs: return {}
    ensure_schema()
    qmarks = ','.join(['?'] * len(obj_refs))
    c = _conn()
    rows = c.execute(
        f'''SELECT obj_ref,
                  COALESCE(SUM(weight),0) AS total_weight,
                  COUNT(*) AS n_flags,
                  GROUP_CONCAT(flag_code) AS codes,
                  GROUP_CONCAT(severity) AS sevs
            FROM flags WHERE obj_ref IN ({qmarks}) GROUP BY obj_ref''', list(obj_refs)).fetchall()
    c.close()
    out = {}
    for r in rows:
        sevs = (r['sevs'] or '').split(',')
        rank = max((SEV_ORDER.get(s, -1) for s in sevs), default=-1)
        max_sev = next((k for k,v in SEV_ORDER.items() if v == rank), None)
        out[r['obj_ref']] = {
            'total_weight': float(r['total_weight'] or 0),
            'n_flags': int(r['n_flags'] or 0),
            'codes': sorted(set((r['codes'] or '').split(','))) if r['codes'] else [],
            'max_severity': max_sev,
        }
    return out


SEV_ORDER = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}


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


# ---------------------------------------------------------------- rebuild

def _copy_feedback_tables(src_c: sqlite3.Connection, dst_c: sqlite3.Connection):
    """Carry the primary data (feedback + feedback_events) into a fresh DB."""
    for table in ('feedback', 'feedback_events'):
        try:
            rows = src_c.execute(f'SELECT * FROM {table}').fetchall()
        except sqlite3.OperationalError:
            continue
        if not rows:
            continue
        cols = rows[0].keys()
        dst_c.executemany(
            f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
            [tuple(r[k] for k in cols) for r in rows])
        log.info('rebuild: carried %d %s row(s)', len(rows), table)


def rebuild(json_dir: Path = JSON_DIR, progress_every: int = 200) -> dict:
    """Offline rebuild of the derived tables into a fresh slim DB, then an
    atomic swap. Safe while srv is running: writes made to the OLD file
    during the rebuild land in flags rows that this pass recomputes anyway
    (the JSON on disk is the source of truth). Feedback rows written during
    the window are re-copied right before the swap.

    Returns a stats dict. The old file is kept as `<DB>.v1.bak` until the
    operator deletes it.
    """
    import quality_flags
    tmp = DB_PATH.with_suffix('.sqlite.rebuild')
    for p in (tmp, Path(str(tmp) + '-wal'), Path(str(tmp) + '-shm')):
        p.unlink(missing_ok=True)
    t0 = time.time()
    with _conn_ctx(tmp) as c:
        _create_schema(c)
        c.execute('PRAGMA synchronous=OFF')
    files = sorted(json_dir.glob('*.json'))
    n_flags = 0
    for i, jp in enumerate(files, 1):
        try:
            data = json.loads(jp.read_text())
            res = quality_flags.scan_kg_data(data, jp.stem)
            write_objects_and_flags(res['objects'], res['flags'],
                                    quality_flags.RULE_VERSION, db_path=tmp)
            n_flags += len(res['flags'])
        except Exception as e:
            log.warning('rebuild: %s: %s', jp.name, e)
        if i % progress_every == 0:
            log.info('rebuild: %d/%d KGs, %d flags, %.0fs, %.1f MB',
                     i, len(files), n_flags, time.time() - t0,
                     os.path.getsize(tmp) / 1e6)
    # carry primary data + swap under the module lock so no writer is mid-txn
    with _LOCK:
        if DB_PATH.exists():
            with _conn_ctx(tmp) as dst, contextlib.closing(_conn()) as old:
                _copy_feedback_tables(old, dst)
            bak = DB_PATH.with_suffix('.sqlite.v1.bak')
            os.replace(DB_PATH, bak)
            for suf in ('-wal', '-shm'):
                p = Path(str(DB_PATH) + suf)
                if p.exists():
                    os.replace(p, Path(str(bak) + suf))
        else:
            bak = None
        with _conn_ctx(tmp) as c:
            c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        os.replace(tmp, DB_PATH)
        for suf in ('-wal', '-shm'):
            Path(str(tmp) + suf).unlink(missing_ok=True)
        ensure_schema(force=True)
    stats = {'kgs': len(files), 'flags': n_flags,
             'seconds': round(time.time() - t0, 1),
             'bytes': os.path.getsize(DB_PATH), 'backup': str(bak) if bak else None}
    log.info('rebuild: done %s', stats)
    return stats


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'stats'
    if cmd == 'stats':
        ensure_schema()
        print(json.dumps(flag_stats(), indent=2))
    elif cmd == 'rebuild':
        print(json.dumps(rebuild(), indent=2))
    elif cmd == 'reset':
        DB_PATH.unlink(missing_ok=True)
        ensure_schema(force=True)
        print('reset')
    else:
        print('usage: feedback_db.py [stats|rebuild|reset]')
