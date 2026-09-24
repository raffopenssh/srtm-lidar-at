"""llm_api — sibling-service contract for the Siedler Österreich game.

Implements the AHEAD list at https://siedler-oesterreich.exe.xyz:8000/llm/ahead
(service=srtm) and the shared per-KG spec from
https://cadastre-process-api.exe.xyz/api/v1/docs/llm.txt?section=integration.

Everything here answers from ``data/search_index.db`` (kg / kg_parcels /
kg_trees / kg_buildings) — never from a KG JSON, never from a GPKG — so
every call is a few ms and costs no Zenodo / BEV bandwidth on the metered
primary. "Warm" therefore means "ingested into the index"; the only cold
state is a KG whose product exists on Zenodo but hasn't been ingested yet
(``ready:false`` + ``retry_after_s``).

Endpoints (all GET unless noted, CORS ``*``, ETag via app after_request):
  /llm/kg/<code>[.json]      per-KG dossier (ALL-1 / ALL-1b)
  /llm/manifest.json         coverage + schema (ALL-2)
  /llm/kgs[?codes=a,b]       covered codes / batch dossiers (ALL-2 optional)
  /llm.txt                   plain-text contract (ALL-3)
  POST /api/v1/prewarm?kgs=  warm-up hint (ALL-4)
  /api/v1/trees/bbox         sampled tree apices (LID-2)
  /api/v1/query/buildings    buildings by bbox (LID-3)
The LID-1 slim fields are added to ``/api/v1/query/parcels`` in app.py via
``slim_parcel_rows`` below.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from flask import Blueprint, Response, jsonify, request

import search_index as si

log = logging.getLogger(__name__)

bp = Blueprint('llm_api', __name__)

SERVICE = 'srtm-lidar-at'
DATASET = 'BEV ALS LiDAR landscape segmentation (DTM/DSM/nDSM + ortho + Sentinel)'
BASE_URL = si.BASE_URL
LICENSE = 'CC-BY-4.0'
SOURCE = ('BEV ALS DTM/DSM 1 m + BEV DOP RGBI 0.2 m (CC BY 4.0), Copernicus '
          'Sentinel-1/2 + ESA WorldCover, Hansen GFC, Austrian cadastre')
KG_TOTAL = 8440

# Paths that browsers may call cross-origin (game runs in the browser).
CORS_PREFIXES = ('/llm', '/api/v1/query', '/api/v1/trees', '/api/v1/kg/',
                 '/api/v1/prewarm', '/tiles/', '/api/v1/parcel/',
                 '/api/v1/lookup', '/api/v1/index/status')


def cors_path(path: str) -> bool:
    return path == '/llm.txt' or any(path.startswith(p) for p in CORS_PREFIXES)


def add_cors(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, If-None-Match'
    resp.headers['Access-Control-Expose-Headers'] = 'ETag, X-Cache, X-Resolution-M'
    resp.headers['Access-Control-Max-Age'] = '86400'
    return resp


# ── helpers ─────────────────────────────────────────────────────────────

def _idx():
    return si.get_index()


def _code_ok(code: str) -> bool:
    return isinstance(code, str) and len(code) == 5 and code.isdigit()


def _no_data(code, extra=None, status=404):
    body = {'kg_code': code, 'error': 'no_data'}
    if extra:
        body.update(extra)
    return jsonify(body), status


_completed_cache = {'t': 0.0, 'set': frozenset()}
_completed_lock = threading.Lock()


def completed_codes() -> frozenset:
    """Parent KG codes with a committed ``_json`` on Zenodo (60 s cache).
    Cheap manifest-key scan — used to tell *pending ingest* from *never
    processed* for the ``ready:false`` contract."""
    now = time.time()
    if now - _completed_cache['t'] < 60:
        return _completed_cache['set']
    with _completed_lock:
        if now - _completed_cache['t'] < 60:
            return _completed_cache['set']
        out = set()
        try:
            mp = Path('data/austria_processor/zenodo_manifest.json')
            m = json.loads(mp.read_text())
            for k, v in (m.get('entries', m) or {}).items():
                if k.endswith('_json') and 'error' not in str((v or {}).get('status', '')):
                    out.add(k[:-5].split('-', 1)[0])
        except Exception as e:
            log.debug('completed_codes: %s', e)
        _completed_cache['set'] = frozenset(out)
        _completed_cache['t'] = time.time()
        return _completed_cache['set']


def _kg_state(code: str, idx=None):
    """'indexed' | 'pending' (product on Zenodo, not ingested yet) |
    'unprocessed' (known KG, no product) | 'unknown'."""
    idx = idx or _idx()
    c = idx._conn()
    row = c.execute('SELECT processed FROM kg WHERE kg_code=?', (code,)).fetchone()
    if row is None:
        return 'unknown'
    if row[0]:
        return 'indexed'
    return 'pending' if code in completed_codes() else 'unprocessed'


def pending_kgs_in_bbox(w, s, e, n, idx=None) -> list:
    """KGs touching bbox whose product exists but is not yet indexed."""
    idx = idx or _idx()
    allk = idx.kgs_in_bbox(w, s, e, n, processed_only=False)
    done = set(idx.kgs_in_bbox(w, s, e, n, processed_only=True))
    comp = completed_codes()
    return [k for k in allk if k not in done and k in comp]


def _num(v, nd=3):
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if f != f:
        return None
    return round(f, nd)


# ── LID-1 slim rows ─────────────────────────────────────────────────────

try:
    from parcel_compact import LETTER_TYPE as _LETTER_TYPE
except Exception:  # pragma: no cover
    _LETTER_TYPE = {}

_NATURAL = {'tree', 'shrub', 'grass', 'hedge', 'water', 'crop', 'orchard',
            'vineyard', 'garden', 'bare_soil', 'rock'}


def _fracs_from_frav(frav_json):
    if not frav_json:
        return {}, None
    try:
        fr = json.loads(frav_json)
    except Exception:
        return {}, None
    tot = float(sum(float(v) for v in fr.values()) or 0)
    if tot <= 0:
        return {}, None
    fracs = {}
    dom, dom_a = None, -1.0
    for k, v in fr.items():
        t = _LETTER_TYPE.get(k, k)
        f = float(v) / tot
        if f >= 0.02:
            fracs[t] = round(f, 3)
        if t in _NATURAL and float(v) > dom_a:
            dom, dom_a = t, float(v)
    return fracs, dom


def slim_parcel_rows(rows: list, idx=None, top_trees_min_h=25.0) -> list:
    """Attach the siedler slim fields to kg_parcels rows (in place).

    Adds: fracs, dom_terrain, tree_h{mean,max}, slope_deg, elev_m,
    aspect_deg, top_trees[{lon,lat,h,rf_conf}] (h ≥ 25 m)."""
    if not rows:
        return rows
    idx = idx or _idx()
    by_kg = {}
    for r in rows:
        by_kg.setdefault(r.get('kg_code'), []).append(r.get('parcel_id'))
    trees = {}
    for kg, pids in by_kg.items():
        try:
            trees[kg] = idx.trees_for_parcels(kg, pids, min_h=top_trees_min_h)
        except Exception as e:
            log.debug('trees_for_parcels %s: %s', kg, e)
            trees[kg] = {}
    for r in rows:
        fracs, dom = _fracs_from_frav(r.get('frav'))
        r['fracs'] = fracs
        r['dom_terrain'] = dom
        r['tree_h'] = {'mean': r.get('tree_h_mean'), 'max': r.get('tree_h_max')}
        r['slope_deg'] = r.get('slope_mean_deg')
        r['elev_m'] = r.get('elevation_m')
        r['aspect_deg'] = r.get('aspect_mean_deg')
        tl = trees.get(r.get('kg_code'), {}).get(r.get('parcel_id'), [])
        r['top_trees'] = [{'lon': t[0], 'lat': t[1], 'h': t[2], 'rf_conf': t[3]}
                          for t in tl]
        r.pop('frav', None)
    return rows


# ── /llm/kg/<code> ──────────────────────────────────────────────────────

METRICS_SCHEMA = {
    'total_area_sqm': 'number', 'parcel_count': 'integer',
    'elevation_min_m': 'number', 'elevation_max_m': 'number',
    'elevation_mean_m': 'number', 'slope_mean_deg': 'number',
    'steepness_max_deg': 'number', 'roughness_mean': 'number', 'tri_mean': 'number',
    'vegetated_fraction': 'number', 'forest_fraction': 'number',
    'ndvi_mean': 'number', 'shannon_diversity': 'number',
    'tree_count': 'integer', 'tree_canopy_sqm': 'number',
    'tree_mean_height_m': 'number', 'tree_stem_volume_m3': 'number',
    'building_count': 'integer', 'building_footprint_sqm': 'number',
    'building_mean_height_m': 'number', 'building_max_height_m': 'number',
    'building_stories_mean': 'number', 'building_pitched_pct': 'number',
    'new_building_count': 'integer', 'infrastructure_count': 'integer',
    'forest_loss_px_total': 'integer', 'forest_loss_px_5yr': 'integer',
    'net_volume_change_m3': 'number', 'temporal_stability': 'number',
    'quality_score': 'number', 'frac_<type>': 'number (landcover share, per object type)',
}

UNIT_GLOSSARY = {
    '*_sqm': 'square metres', '*_m': 'metres', '*_deg': 'degrees',
    '*_m3': 'cubic metres', '*_fraction|frac_*': 'share of KG area 0..1',
    '*_pct': 'percent 0..100',
    'forest_loss_px_*': 'Hansen GFC 30 m loss pixels intersecting the KG tiles '
                        '(relative index; tile overlap may double count)',
    'tree_count': 'segmented tree crowns (not single stems); n_apices in the '
                  'light GPKG holds single-tree apices',
    'quality_score': '0..1 composite classification quality',
}


def kg_dossier(code: str, idx=None, with_parcels=False, parcel_limit=2000):
    idx = idx or _idx()
    d = idx.query_kg(code)
    if not d or not d.get('processed'):
        return None
    g = d.get
    m = {
        'total_area_sqm': _num(g('total_area_sqm'), 1),
        'parcel_count': g('parcel_count'),
        'elevation_min_m': _num(g('elevation_min_m'), 1),
        'elevation_max_m': _num(g('elevation_max_m'), 1),
        'elevation_mean_m': _num(g('elevation_mean_m'), 1),
        'slope_mean_deg': _num(g('slope_mean_deg'), 2),
        'steepness_max_deg': _num(g('steepness_max_deg'), 1),
        'roughness_mean': _num(g('roughness_mean')),
        'tri_mean': _num(g('tri_mean')),
        'vegetated_fraction': _num(g('vegetated_fraction')),
        'ndvi_mean': _num(g('ndvi_mean')),
        'shannon_diversity': _num(g('shannon_diversity')),
        'tree_count': g('tree_count'),
        'tree_canopy_sqm': _num(g('tree_canopy_sqm'), 0),
        'tree_mean_height_m': _num(g('tree_mean_height_m'), 2),
        'tree_stem_volume_m3': _num(g('tree_stem_volume_m3'), 0),
        'building_count': g('building_count'),
        'building_footprint_sqm': _num(g('building_footprint_sqm'), 0),
        'building_mean_height_m': _num(g('building_mean_height_m'), 2),
        'building_max_height_m': _num(g('building_max_height_m'), 2),
        'building_stories_mean': _num(g('building_stories_mean'), 2),
        'building_pitched_pct': _num(g('building_pitched_pct'), 1),
        'new_building_count': g('new_building_count'),
        'infrastructure_count': g('infrastructure_count'),
        'net_volume_change_m3': _num(g('net_volume_change_m3'), 0),
        'temporal_stability': _num(g('temporal_stability')),
        'quality_score': _num(g('quality_score')),
    }
    forest = 0.0
    for lc in d.get('landcover') or []:
        t = lc.get('object_type')
        fr = _num(lc.get('fraction'))
        if t and fr is not None:
            m[f'frac_{t}'] = fr
            if t in ('tree', 'shrub', 'hedge'):
                forest += fr
    m['forest_fraction'] = round(forest, 3)
    hist = []
    tot = 0
    recent = 0
    for h in d.get('hansen_loss') or []:
        y = h.get('loss_year'); px = int(h.get('loss_pixels') or 0)
        if not y:
            continue
        tot += px
        if int(y) >= 2020:
            recent += px
        hist.append({'as_of': f'{int(y)}-12-31', 'forest_loss_px': px})
    m['forest_loss_px_total'] = tot
    m['forest_loss_px_5yr'] = recent
    hist.sort(key=lambda x: x['as_of'])
    gen = g('generated_at') or ''
    py = g('primary_year')
    out = {
        'service': SERVICE,
        'dataset': DATASET,
        'kg_code': code,
        'kg_name': g('kg_name'),
        'gemeinde_code': g('gemeinde_code'),
        'gemeinde_name': g('gemeinde_name'),
        'granularity': 'parcel',
        'as_of': f'{py}-12-31' if py else (gen[:10] or None),
        'updated_at': g('json_v2_uploaded_at') or g('json_uploaded_at') or gen or None,
        'source': SOURCE,
        'license': LICENSE,
        'attribution_url': f'{BASE_URL}/api/v1/attribution',
        'product_version': g('product_version'),
        'quality_grade': g('quality_grade'),
        'bbox': [g('min_lon'), g('min_lat'), g('max_lon'), g('max_lat')],
        'attributes': {
            'dominant_type': g('dominant_type'),
            'terrain_class': g('terrain_class'),
            'aspect_dominant': g('aspect_dominant'),
            'phenology_dominant': g('phenology_dominant'),
        },
        'unit_glossary': UNIT_GLOSSARY,
        'metrics': m,
        'history': hist,
        'parcels_url': f'{BASE_URL}/api/v1/query/parcels?kg={code}&limit=1000',
        'trees_url': f'{BASE_URL}/api/v1/trees/bbox?west={g("min_lon")}&south={g("min_lat")}'
                     f'&east={g("max_lon")}&north={g("max_lat")}&min_height=20',
        'links': d.get('_links') or {},
    }
    if with_parcels:
        r = idx.query_parcels_index(kg_code=code, limit=min(int(parcel_limit), 5000),
                                    sort='area_sqm', sort_dir='desc')
        rows = slim_parcel_rows(r.get('results') or [], idx)
        out['parcels'] = [{
            'parcel_id': p['parcel_id'],
            'metrics': {
                'area_sqm': p.get('area_sqm'), 'elev_m': p.get('elev_m'),
                'slope_deg': p.get('slope_deg'), 'aspect_deg': p.get('aspect_deg'),
                'vegetated_fraction': p.get('vegetated_fraction'),
                'forested_fraction': p.get('forested_fraction'),
                'ndsm_max_m': p.get('ndsm_max_m'),
                'building_count': p.get('building_count'),
                'tree_h_max': p['tree_h']['max'],
            },
            'fracs': p['fracs'], 'dom_terrain': p['dom_terrain'],
            'auto_class': p.get('auto_class'), 'auto_subclass': p.get('auto_subclass'),
            'top_trees': p['top_trees'],
        } for p in rows]
        out['parcels_total'] = r.get('total')
    return out


@bp.route('/llm/kg/<code>')
@bp.route('/llm/kg/<code>.json')
def llm_kg(code):
    code = (code or '').strip()
    if code.endswith('.json'):
        code = code[:-5]
    if not _code_ok(code):
        return _no_data(code, {'hint': 'kg_code is a 5-digit string'})
    try:
        idx = _idx()
        st = _kg_state(code, idx)
        if st == 'indexed':
            want_p = request.args.get('parcels', '').lower() in ('1', 'true', 'yes')
            body = kg_dossier(code, idx, with_parcels=want_p,
                              parcel_limit=request.args.get('limit', 2000))
            if body is None:
                return _no_data(code)
            resp = jsonify(body)
            resp.headers['Cache-Control'] = 'public, max-age=3600'
            return resp
        if st == 'pending':
            resp = jsonify({'kg_code': code, 'ready': False, 'status': 'pending_ingest',
                            'retry_after_s': 300,
                            'hint': 'product committed on Zenodo, index ingest pending'})
            resp.headers['Retry-After'] = '300'
            return resp, 202
        return _no_data(code, {'status': st})
    except Exception as e:
        log.exception('llm_kg %s', code)
        return _no_data(code, {'status': 'error', 'detail': str(e)}, status=404)


# ── /llm/manifest.json + /llm/kgs ───────────────────────────────────────

@bp.route('/llm/manifest.json')
def llm_manifest():
    idx = _idx()
    c = idx._conn()
    n = c.execute('SELECT COUNT(*) FROM kg WHERE processed=1').fetchone()[0]
    meta = {r[0]: r[1] for r in c.execute('SELECT key, value FROM index_meta')}
    body = {
        'service': SERVICE,
        'dataset': DATASET,
        'base_url': BASE_URL,
        'finest_granularity': 'parcel',
        'join_keys': ['kg_code', 'parcel_id', 'gemeinde_code'],
        'kg_endpoint': '/llm/kg/{kg_code}',
        'kgs_endpoint': '/llm/kgs?codes=a,b',
        'covered_kgs_url': f'{BASE_URL}/llm/kgs',
        'metrics_schema': METRICS_SCHEMA,
        'unit_glossary': UNIT_GLOSSARY,
        'kg_count': n,
        'kg_total': KG_TOTAL,
        'coverage_pct': round(100.0 * n / KG_TOTAL, 1),
        'updated_at': meta.get('last_incremental_at') or meta.get('built_at'),
        'license': LICENSE,
        'source': SOURCE,
        'viewport_endpoints': {
            'parcels': '/api/v1/query/parcels?bbox=w,s,e,n&limit=500',
            'trees': '/api/v1/trees/bbox?west&south&east&north&min_height=20&limit=2000',
            'buildings': '/api/v1/query/buildings?bbox=w,s,e,n&limit=500',
            'prewarm': 'POST /api/v1/prewarm?kgs=a,b',
        },
        'llm_txt': f'{BASE_URL}/llm.txt',
    }
    resp = jsonify(body)
    resp.headers['Cache-Control'] = 'public, max-age=3600'
    return resp


@bp.route('/llm/kgs')
def llm_kgs():
    idx = _idx()
    codes = request.args.get('codes', '').strip()
    if not codes:
        c = idx._conn()
        rows = [r[0] for r in c.execute(
            'SELECT kg_code FROM kg WHERE processed=1 ORDER BY kg_code')]
        resp = jsonify({'service': SERVICE, 'kg_count': len(rows), 'kgs': rows})
        resp.headers['Cache-Control'] = 'public, max-age=3600'
        return resp
    want = [x.strip() for x in codes.split(',') if x.strip()][:500]
    results = []
    for code in want:
        if not _code_ok(code):
            results.append({'kg_code': code, 'error': 'no_data', 'status': 'invalid'})
            continue
        st = _kg_state(code, idx)
        if st == 'indexed':
            d = kg_dossier(code, idx)
            if d is not None:
                d.pop('unit_glossary', None)
                results.append(d)
                continue
        results.append({'kg_code': code, 'error': 'no_data', 'status': st,
                        **({'ready': False, 'retry_after_s': 300} if st == 'pending' else {})})
    return jsonify({'service': SERVICE, 'count': len(results), 'results': results})


# ── /llm.txt ────────────────────────────────────────────────────────────

LLM_TXT = f"""# {SERVICE} — sibling-service contract (siedler / kohlschwarz)
base: {BASE_URL}
license: {LICENSE} — attribution required, see /api/v1/attribution (BEV, Copernicus, Hansen, cadastre)
full API reference: /api/v1/docs/llm.txt   dashboard: /process.html (text: /process.txt)

## Contract endpoints (GET, JSON, CORS *, ETag/304, gzip)
/llm/kg/{{kg_code}}            per-KG dossier: service, kg_code, gemeinde_code, granularity=parcel,
                              as_of, updated_at, metrics{{flat snake_case numbers}}, history[Hansen loss/yr],
                              attributes{{strings}}, parcels_url, trees_url.  ?parcels=1 embeds slim parcels.
                              404 {{kg_code,error:"no_data"}} when we hold nothing; 202 ready:false +
                              retry_after_s when the product exists but is not ingested yet.
/llm/manifest.json            service, kg_endpoint, kg_count (indexed), kg_total=8440, metrics_schema,
                              covered_kgs_url, updated_at.
/llm/kgs                      all covered kg codes.  /llm/kgs?codes=a,b (≤500) → {{results:[dossiers]}}
/llm.txt                      this file.

## Viewport endpoints (answered from the SQLite/R-tree index, never from a KG JSON or GPKG)
/api/v1/query/parcels?bbox=w,s,e,n&limit=500
    rows: parcel_id, kg_code, fracs{{type:frac≥0.02}}, dom_terrain, tree_h{{mean,max}}, auto_class,
    auto_subclass, slope_deg, elev_m, aspect_deg, top_trees[{{lon,lat,h,rf_conf}}] (h≥25) + all
    legacy columns. bbox filters on parcel centroid. ready:false + retry_after_s + pending_kgs when a
    touched KG is committed but not yet ingested. Also ?kg=CODE, 70+ min_/max_ filters (see docs).
/api/v1/trees/bbox?west&south&east&north&min_height=20&limit=2000
    {{trees:[{{lon,lat,h_m,crown_d_m,parcel_id,rf_conf,kg_code}}],count,truncated,ready,kgs,source}}
    source = the ≤5 tallest trees per cadastre parcel (sampled apices, index-backed). The complete
    apex set (~40k/KG) is in the light GPKG → POST /api/v3/trees (slower, product download).
/api/v1/query/buildings?bbox=w,s,e,n&limit=500
    rows: building_id (address id), footprint_id (cadastre, null when unmatched), ns, roof_type,
    max_height_m, mean_height_m, stories_est, footprint_area_sqm, lon, lat, kg_code.
POST /api/v1/prewarm?kgs=a,b   (≤50) → 200/202 {{already_warm[],queued[],pending[],not_processed[],unknown[]}}
    Index-backed endpoints are warm for every indexed KG; the hint only triggers slim-field
    backfill for legacy rows. Nothing to warm for not_processed KGs.

## Keys & rules
kg_code / gemeinde_code: 5-char strings (leading zeros kept). parcel_id = "KGCODE-GNR".
Coverage grows continuously (fleet processes all 8440 KGs); poll /llm/manifest.json kg_count.
Rate limit: be gentle — primary is bandwidth-metered. Cache by ETag; responses are gzip'd.
Additive-only evolution: fields are never removed or renamed.
"""


@bp.route('/llm.txt')
def llm_txt():
    resp = Response(LLM_TXT, mimetype='text/plain; charset=utf-8')
    resp.headers['Cache-Control'] = 'public, max-age=3600'
    return resp


# ── POST /api/v1/prewarm ────────────────────────────────────────────────

_prewarm_lock = threading.Lock()
_prewarm_inflight: set = set()


def _backfill_async(codes):
    def run():
        try:
            _idx().backfill_slim_fields(codes=codes, log_every=0)
        except Exception as e:
            log.warning('prewarm backfill: %s', e)
        finally:
            with _prewarm_lock:
                _prewarm_inflight.difference_update(codes)
    threading.Thread(target=run, name='prewarm-backfill', daemon=True).start()


@bp.route('/api/v1/prewarm', methods=['POST', 'GET'])
def prewarm():
    raw = request.args.get('kgs') or ''
    if not raw:
        body = request.get_json(silent=True) or {}
        v = body.get('kgs') or []
        raw = ','.join(v) if isinstance(v, list) else str(v)
    codes = []
    for x in raw.split(','):
        x = x.strip()
        if x and x not in codes:
            codes.append(x)
    if not codes:
        return jsonify({'error': 'kgs required (comma list, ≤50)'}), 400
    codes = codes[:50]
    idx = _idx()
    c = idx._conn()
    out = {'already_warm': [], 'queued': [], 'pending': [], 'not_processed': [], 'unknown': []}
    to_backfill = []
    for code in codes:
        if not _code_ok(code):
            out['unknown'].append(code)
            continue
        st = _kg_state(code, idx)
        if st == 'indexed':
            n_missing = c.execute(
                'SELECT COUNT(*) FROM kg_parcels WHERE kg_code=? AND frav IS NULL',
                (code,)).fetchone()[0]
            n_all = c.execute('SELECT COUNT(*) FROM kg_parcels WHERE kg_code=?',
                              (code,)).fetchone()[0]
            fp_st = 'ready'
            try:
                import building_footprint_match as _bfm
                fp_st = _bfm.status(idx, [code]).get(code, 'ready')
            except Exception:
                pass
            if (n_all and n_missing == n_all) or fp_st == 'warming':
                out['queued'].append(code)
                if n_all and n_missing == n_all:
                    to_backfill.append(code)
            else:
                out['already_warm'].append(code)
        elif st == 'pending':
            out['pending'].append(code)
        elif st == 'unprocessed':
            out['not_processed'].append(code)
        else:
            out['unknown'].append(code)
    if to_backfill:
        with _prewarm_lock:
            fresh = [k for k in to_backfill if k not in _prewarm_inflight]
            _prewarm_inflight.update(fresh)
        if fresh:
            _backfill_async(fresh)
    all_warm = not out['queued'] and not out['pending']
    out['ready'] = all_warm
    if not all_warm:
        out['retry_after_s'] = 30 if out['queued'] else 300
    return jsonify(out), (200 if all_warm else 202)


# ── GET /api/v1/trees/bbox ──────────────────────────────────────────────

def _bbox_args():
    a = request.args
    if a.get('bbox'):
        w, s, e, n = [float(x) for x in a['bbox'].split(',')]
    else:
        w = float(a['west']); s = float(a['south'])
        e = float(a['east']); n = float(a['north'])
    if w > e:
        w, e = e, w
    if s > n:
        s, n = n, s
    return w, s, e, n


@bp.route('/api/v1/trees/bbox')
def trees_bbox():
    try:
        w, s, e, n = _bbox_args()
    except Exception:
        return jsonify({'error': 'west,south,east,north (or bbox=w,s,e,n) required'}), 400
    if (e - w) * (n - s) > 0.25:
        return jsonify({'error': 'bbox too large (max 0.25 deg²)'}), 400
    min_h = float(request.args.get('min_height', 20) or 20)
    limit = int(request.args.get('limit', 2000) or 2000)
    idx = _idx()
    r = idx.query_trees_bbox(w, s, e, n, min_height=min_h, limit=limit)
    trees = [{'lon': t[2], 'lat': t[3], 'h_m': t[4], 'crown_d_m': t[5],
              'parcel_id': t[1], 'rf_conf': t[6], 'kg_code': t[0]} for t in r['rows']]
    pending = pending_kgs_in_bbox(w, s, e, n, idx)
    body = {
        'trees': trees, 'count': len(trees), 'truncated': r['truncated'],
        'ready': not pending, 'kgs': r['kgs'], 'min_height': min_h,
        'limit': min(limit, 2000),
        'source': 'index kg_trees: ≤5 tallest apices per cadastre parcel (sampled; '
                  'full apex set via POST /api/v3/trees)',
        'complete': False,
    }
    if pending:
        body['pending_kgs'] = pending
        body['retry_after_s'] = 300
    if not r['kgs'] and not pending:
        body['coverage'] = 'none'
    return jsonify(body)


# ── GET /api/v1/query/buildings ─────────────────────────────────────────

@bp.route('/api/v1/query/buildings')
def query_buildings():
    a = request.args
    idx = _idx()
    c = idx._conn()
    limit = max(1, min(int(a.get('limit', 500) or 500), 2000))
    offset = max(0, int(a.get('offset', 0) or 0))
    where, params = [], []
    kgs = None
    pending = []
    if a.get('bbox') or a.get('west'):
        try:
            w, s, e, n = _bbox_args()
        except Exception:
            return jsonify({'error': 'bad bbox'}), 400
        if (e - w) * (n - s) > 0.25:
            return jsonify({'error': 'bbox too large (max 0.25 deg²)'}), 400
        kgs = idx.kgs_in_bbox(w, s, e, n)
        pending = pending_kgs_in_bbox(w, s, e, n, idx)
        if not kgs:
            return jsonify({'buildings': [], 'count': 0, 'total': 0, 'ready': not pending,
                            'pending_kgs': pending or None, 'kgs': [],
                            'coverage': 'none' if not pending else 'pending'})
        where.append('kg_code IN (%s)' % ','.join('?' * len(kgs)))
        params.extend(kgs)
        where.append('centroid_lon >= ? AND centroid_lon <= ? AND centroid_lat >= ? AND centroid_lat <= ?')
        params.extend([w, e, s, n])
    if a.get('kg'):
        where.append('kg_code = ?'); params.append(a['kg'])
    for key, col, op in (('min_height', 'max_height_m', '>='), ('max_height', 'max_height_m', '<='),
                         ('min_stories', 'stories_est', '>='), ('max_stories', 'stories_est', '<='),
                         ('min_area', 'footprint_area_sqm', '>='), ('max_area', 'footprint_area_sqm', '<=')):
        if a.get(key):
            where.append(f'{col} {op} ?'); params.append(float(a[key]))
    if a.get('roof_type'):
        where.append('roof_type_hint = ?'); params.append(a['roof_type'])
    if not where:
        return jsonify({'error': 'bbox= (or kg=) required'}), 400
    wsql = ' AND '.join(where)
    import building_footprint_match as _bfm
    _bfm._ensure_schema(c)
    total = c.execute(f'SELECT COUNT(*) FROM kg_buildings WHERE {wsql}', params).fetchone()[0]
    rows = c.execute(
        f'SELECT kg_code, building_id, ns, roof_type_hint, max_height_m, mean_height_m, '
        f'stories_est, footprint_area_sqm, centroid_lon, centroid_lat, centroid_dtm_m, '
        f'footprint_id FROM kg_buildings WHERE {wsql} '
        f'ORDER BY max_height_m DESC LIMIT ? OFFSET ?',
        params + [limit, offset]).fetchall()
    blds = [{
        'kg_code': r[0], 'building_id': r[1] or None,
        'footprint_id': r[11],
        'ns': r[2], 'roof_type': r[3] or None, 'max_height_m': r[4], 'mean_height_m': r[5],
        'stories_est': r[6], 'footprint_area_sqm': r[7], 'lon': r[8], 'lat': r[9],
        'ground_m': r[10],
    } for r in rows]
    fp_status = {}
    try:
        fp_status = _bfm.status(idx, sorted({r[0] for r in rows} | set(kgs or [])))
    except Exception as e:
        log.debug('footprint match: %s', e)
    body = {'buildings': blds, 'count': len(blds), 'total': total, 'limit': limit,
            'offset': offset, 'ready': not pending, 'kgs': kgs,
            'footprint_match': fp_status,
            'footprint_ids_ready': all(v == 'ready' for v in fp_status.values())}
    if pending:
        body['pending_kgs'] = pending
        body['retry_after_s'] = 300
    return jsonify(body)
