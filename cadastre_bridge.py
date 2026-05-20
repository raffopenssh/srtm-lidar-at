"""cadastre_bridge.py — Cross-API bridge between srtm-lidar landscape analysis and Cadastre API.

Provides functions that combine Austrian cadastral data (parcels, buildings,
legal refs, protected areas) with landscape analysis data (NDVI, vegetation,
elevation, RF classification, Hansen forest loss) for nature conservation
assessment and parcel purchase opportunity analysis.
"""
from __future__ import annotations

import json
import logging
import time
import threading
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import requests

import search_index as si

log = logging.getLogger(__name__)

CADASTRE_API = 'https://cadastre-process-api.exe.xyz/api/v1'
CADASTRE_TIMEOUT = 30  # seconds (cadastre queries can be slow for large KGs)
JSON_DIR = Path('data/austria_processor/json')

# ═══════════════════════════════════════════════════════════════════════════
# Simple TTL cache for cadastre responses
# ═══════════════════════════════════════════════════════════════════════════

_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()
_CACHE_TTL = 30  # seconds
_CACHE_MAX = 500  # max entries


def _cache_get(key: str) -> Any | None:
    """Return cached value if fresh, else None."""
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: Any):
    """Store value in cache with current timestamp."""
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            # Evict oldest quarter
            sorted_keys = sorted(_cache, key=lambda k: _cache[k][0])
            for k in sorted_keys[:_CACHE_MAX // 4]:
                _cache.pop(k, None)
        _cache[key] = (time.time(), value)


# ═══════════════════════════════════════════════════════════════════════════
# Generic cadastre proxy
# ═══════════════════════════════════════════════════════════════════════════

def cadastre_proxy(endpoint: str, params: dict | None = None,
                   method: str = 'GET', json_body: Any = None,
                   timeout: int = CADASTRE_TIMEOUT) -> dict:
    """Generic proxy to cadastre API with error handling, timeout, and caching.

    Args:
        endpoint: API path (e.g. '/query', '/lookup'). Prepends CADASTRE_API.
        params: Query parameters dict.
        method: HTTP method ('GET' or 'POST').
        json_body: JSON body for POST requests.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response dict.

    Raises:
        CadastreError: On HTTP or connection errors.
    """
    url = f'{CADASTRE_API}{endpoint}'

    # Cache GET requests
    cache_key = None
    if method == 'GET':
        cache_key = f'{url}?{urlencode(params or {}, doseq=True)}'
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    try:
        if method == 'POST':
            resp = requests.post(url, params=params, json=json_body, timeout=timeout)
        else:
            resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if cache_key:
            _cache_set(cache_key, data)
        return data
    except requests.Timeout:
        raise CadastreError(f'Cadastre API timeout ({timeout}s) for {endpoint}')
    except requests.ConnectionError:
        raise CadastreError(f'Cadastre API unreachable: {endpoint}')
    except requests.HTTPError as e:
        raise CadastreError(f'Cadastre API error {e.response.status_code}: {endpoint}')
    except (ValueError, KeyError) as e:
        raise CadastreError(f'Cadastre API invalid response: {e}')


class CadastreError(Exception):
    """Raised when the cadastre API call fails."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Lookup proxy
# ═══════════════════════════════════════════════════════════════════════════

def lookup(q: str, type: str | None = None, limit: int = 20) -> list[dict]:
    """Proxy to cadastre /lookup — diacritics-insensitive search.

    Searches across Austria's federal register (EDM): Gemeinden, KGs,
    Ortschaften, PLZ. Handles umlauts gracefully ("Kofla" → "Köflach").

    Args:
        q: Search text (name, PLZ, code).
        type: Filter by entity type: 'plz', 'gemeinde', 'kg', 'ortschaft'.
        limit: Max results (1-200).

    Returns:
        List of match dicts with type, code, name, plz, gemeinde info.
    """
    params = {'q': q, 'limit': min(limit, 200)}
    if type:
        params['type'] = type
    return cadastre_proxy('/lookup', params=params)


# ═══════════════════════════════════════════════════════════════════════════
# Local landscape data loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_kg_json(kg_code: str) -> dict | None:
    """Load KG JSON from local file. Returns parsed dict or None."""
    jp = JSON_DIR / f'{kg_code}.json'
    if jp.exists():
        try:
            return json.loads(jp.read_text())
        except Exception as e:
            log.warning('Failed to load KG JSON %s: %s', kg_code, e)
    return None


def _find_parcel_in_kg(kg_data: dict, parcel_id: str) -> dict | None:
    """Find a parcel detail dict in KG JSON data."""
    for p in kg_data.get('parcels', {}).get('details', []):
        if p.get('parcel_id') == parcel_id:
            return p
    return None


def _extract_landscape_from_parcel(parcel_detail: dict) -> dict:
    """Extract landscape analysis fields from a parcel detail dict.

    Carries through the full per-parcel data from our KG JSON:
    - area_summary: {type: {area_sqm, fraction}} — segmentation result
    - height_distribution: {type: {min, max, mean}} — nDSM heights per type
    - classification: full RF/rules breakdown per type with confidence
    - vegetation, elevation, nDSM, temporal change
    """
    if not parcel_detail:
        return {}

    landscape = {}

    # Area summary (tree, grass, building, etc.)
    area_summary = parcel_detail.get('area_summary', {})
    if area_summary:
        landscape['area_summary'] = area_summary
        # Compute vegetation fraction from area_summary
        veg_types = ('tree', 'grass', 'shrub', 'crop', 'hedge',
                     'orchard', 'vineyard', 'garden')
        total_frac = sum(
            area_summary.get(t, {}).get('fraction', 0)
            for t in veg_types
        )
        landscape['vegetated_fraction'] = round(total_frac, 3)
        # Tree canopy
        tree = area_summary.get('tree', {})
        if tree:
            landscape['tree_canopy_sqm'] = tree.get('area_sqm', 0)
            landscape['tree_fraction'] = tree.get('fraction', 0)

    # Height distribution per type
    if parcel_detail.get('height_distribution'):
        landscape['height_distribution'] = parcel_detail['height_distribution']

    # Classification — full per-type breakdown
    # classification.by_type.{type}: segments, area_sqm, mean_confidence,
    #   rf_count, rules_count, rf_mean_confidence, diverged_count
    if parcel_detail.get('classification'):
        landscape['classification'] = parcel_detail['classification']

    # Vegetation flags
    if 'vegetated_fraction' in parcel_detail:
        landscape['vegetated_fraction'] = parcel_detail['vegetated_fraction']
    if 'is_vegetated' in parcel_detail:
        landscape['is_vegetated'] = parcel_detail['is_vegetated']

    # nDSM heights
    if parcel_detail.get('ndsm_max_m') is not None:
        landscape['ndsm_max_m'] = parcel_detail['ndsm_max_m']
    if parcel_detail.get('ndsm_mean_m') is not None:
        landscape['ndsm_mean_m'] = parcel_detail['ndsm_mean_m']

    # Elevation
    if parcel_detail.get('elevation_m') is not None:
        landscape['elevation_m'] = parcel_detail['elevation_m']

    # Forested fraction
    if 'forested_fraction' in parcel_detail:
        landscape['forested_fraction'] = parcel_detail['forested_fraction']

    # Dominant type
    if parcel_detail.get('dominant_type'):
        landscape['dominant_type'] = parcel_detail['dominant_type']

    # Temporal / Hansen
    for k in ('hansen_loss', 'hansen_loss_pixels', 'temporal_change', 'volume_change_m3'):
        if k in parcel_detail:
            landscape[k] = parcel_detail[k]

    return landscape


def _extract_landscape_from_kg_index(kg_code: str) -> dict | None:
    """Get KG-level landscape data from the search index."""
    try:
        idx = si.get_index()
        return idx.query_kg(kg_code)
    except Exception as e:
        log.warning('Search index query_kg %s: %s', kg_code, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Conservation score computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_conservation_score(legal_refs: list | None = None,
                               protected_area_relation: str | None = None,
                               vegetated_fraction: float = 0,
                               ndvi: float = 0,
                               tree_canopy_sqm: float = 0) -> int:
    """Compute a 0-100 conservation value score for a parcel.

    Scoring breakdown:
        Legal protection status: 0-30 pts
            Referenced in any law → +30
        Protected area proximity: 0-20 pts
            Within a protected area → +20
            Near a protected area → +10
        Vegetation fraction: 0-20 pts
            Linear scale: fraction × 20
        NDVI quality: 0-15 pts
            Linear scale: clamp(ndvi, 0, 0.8) / 0.8 × 15
        Tree canopy: 0-15 pts
            Log scale: min(log10(canopy+1)/4, 1) × 15

    Args:
        legal_refs: List of legal reference dicts (presence = protected).
        protected_area_relation: 'within', 'near', or None.
        vegetated_fraction: 0-1 fraction of parcel covered by vegetation.
        ndvi: Mean NDVI value (typically 0-0.8 for vegetation).
        tree_canopy_sqm: Tree canopy area in square metres.

    Returns:
        Integer score 0-100.
    """
    import math
    score = 0.0

    # Legal protection (0-30)
    if legal_refs:
        score += 30

    # Protected area (0-20)
    if protected_area_relation == 'within':
        score += 20
    elif protected_area_relation == 'near':
        score += 10

    # Vegetation fraction (0-20)
    score += min(max(vegetated_fraction, 0), 1.0) * 20

    # NDVI (0-15)
    if ndvi > 0:
        score += min(ndvi / 0.8, 1.0) * 15

    # Tree canopy (0-15) — log scale so small canopy still scores
    if tree_canopy_sqm > 0:
        score += min(math.log10(tree_canopy_sqm + 1) / 4.0, 1.0) * 15

    return min(round(score), 100)


# ═══════════════════════════════════════════════════════════════════════════
# Batch parcel landscape enrichment
# ═══════════════════════════════════════════════════════════════════════════

def batch_parcel_landscape(parcel_ids: list[str]) -> dict:
    """For a list of parcel IDs, return landscape analysis + cadastre data.

    For each parcel:
    1. Extract kg_code from parcel_id (digits before the '-')
    2. Load KG JSON (local file) for parcel-level landscape detail
    3. Fall back to search index for KG-level landscape summary
    4. Optionally fetch cadastre data (landuse, area, buildings) via batch query

    Args:
        parcel_ids: List of parcel IDs like ['63349-505/3', '75414-1314/1'].
                    Max 200 per request.

    Returns:
        Dict with 'results' list and '_warnings' if any errors occurred.
    """
    warnings = []
    results = []

    # Group parcels by KG code for efficient loading
    kg_parcels: dict[str, list[str]] = {}
    for pid in parcel_ids:
        if '-' not in pid:
            warnings.append(f'Invalid parcel_id format: {pid}')
            continue
        kg_code = pid.split('-')[0]
        kg_parcels.setdefault(kg_code, []).append(pid)

    # Load KG JSONs once per KG
    kg_json_cache: dict[str, dict | None] = {}
    kg_index_cache: dict[str, dict | None] = {}

    for kg_code in kg_parcels:
        kg_json_cache[kg_code] = _load_kg_json(kg_code)
        kg_index_cache[kg_code] = _extract_landscape_from_kg_index(kg_code)

    # Try to batch-fetch cadastre data for all parcels
    cadastre_data: dict[str, dict] = {}
    try:
        # Use the cadastre search/feature endpoint per-parcel, or batch spatial
        # Group by KG and query cadastre in bulk
        for kg_code, pids in kg_parcels.items():
            try:
                cad_resp = cadastre_proxy('/search/parcel', params={
                    'kg': kg_code,
                    'limit': min(len(pids) * 2, 1000),
                })
                for item in cad_resp.get('data', []):
                    pid = item.get('parcel_id')
                    if pid:
                        cadastre_data[pid] = item
            except CadastreError as e:
                warnings.append(f'Cadastre lookup failed for KG {kg_code}: {e}')
    except Exception as e:
        warnings.append(f'Cadastre batch fetch error: {e}')

    # Build results
    for pid in parcel_ids:
        if '-' not in pid:
            continue

        kg_code = pid.split('-')[0]
        entry = {'parcel_id': pid, 'kg_code': kg_code}

        # Cadastre data
        cad = cadastre_data.get(pid)
        if cad:
            entry['cadastre'] = {
                'area_sqm': cad.get('area_sqm'),
                'landuse_codes': cad.get('landuse_codes'),
                'landuse_summary': cad.get('landuse_summary'),
                'ez': cad.get('ez'),
                'status': cad.get('status'),
                'building_count': cad.get('building_count'),
                'lon': cad.get('lon'),
                'lat': cad.get('lat'),
                'legal_refs': cad.get('legal_refs'),
                'legal_contexts': cad.get('legal_contexts'),
            }
        else:
            entry['cadastre'] = None

        # Landscape data — prefer parcel-level from JSON
        kg_data = kg_json_cache.get(kg_code)
        parcel_detail = _find_parcel_in_kg(kg_data, pid) if kg_data else None

        if parcel_detail:
            entry['landscape'] = _extract_landscape_from_parcel(parcel_detail)
            entry['landscape']['_source'] = 'parcel_json'
        else:
            # Fall back to KG-level index data
            kg_idx = kg_index_cache.get(kg_code)
            if kg_idx:
                entry['landscape'] = {
                    '_source': 'kg_index',
                    'ndvi_mean': kg_idx.get('ndvi_mean'),
                    'vegetated_fraction': kg_idx.get('vegetated_fraction'),
                    'elevation_mean_m': kg_idx.get('elevation_mean_m'),
                    'slope_mean_deg': kg_idx.get('slope_mean_deg'),
                    'tree_canopy_sqm': kg_idx.get('tree_canopy_sqm'),
                    'tree_count': kg_idx.get('tree_count'),
                    'dominant_type': kg_idx.get('dominant_type'),
                    'quality_score': kg_idx.get('quality_score'),
                }
            else:
                entry['landscape'] = None

        # Conservation score
        legal_refs = (entry.get('cadastre') or {}).get('legal_refs')
        veg_frac = (entry.get('landscape') or {}).get('vegetated_fraction', 0) or 0
        ndvi_val = (entry.get('landscape') or {}).get('ndvi_mean', 0) or 0
        tree_sqm = (entry.get('landscape') or {}).get('tree_canopy_sqm', 0) or 0
        entry['conservation_score'] = compute_conservation_score(
            legal_refs=legal_refs,
            vegetated_fraction=veg_frac,
            ndvi=ndvi_val,
            tree_canopy_sqm=tree_sqm,
        )

        results.append(entry)

    out = {'results': results, 'total': len(results)}
    if warnings:
        out['_warnings'] = warnings
    return out


def batch_parcel_landscape_by_query(query_filters: dict, landscape_filters: dict | None = None,
                                    limit: int = 50, offset: int = 0) -> dict:
    """Find parcels via cadastre query, then enrich with landscape data.

    This is the "query mode" for batch: instead of specifying explicit parcel IDs,
    pass cadastre query filters (and optionally landscape filters) to find and
    enrich parcels in one call.

    Step 1: Query cadastre API with the cadastre-side filters.
    Step 2: For returned parcels, load landscape data from our index/JSON.
    Step 3: Apply landscape-side filters (NDVI, vegetation, tree canopy, etc.).
    Step 4: Compute conservation scores and return enriched results.

    Args:
        query_filters: Cadastre query params — any valid /api/v1/query params:
            kg, gemeinde, district, state, plz, landuse, min_area, max_area,
            has_buildings, status, ez, has_legal_refs, legal_context,
            min_lon, min_lat, max_lon, max_lat, sort, q.
        landscape_filters: Optional landscape-side post-filters:
            min_vegetated_fraction, max_vegetated_fraction,
            min_ndvi, max_ndvi,
            min_tree_canopy_sqm,
            min_elevation, max_elevation,
            min_conservation_score,
            dominant_type.
        limit: Max results after all filtering.
        offset: Offset for pagination.

    Returns:
        Dict with 'results', 'total', 'meta', and '_warnings'.
    """
    warnings = []

    # Build cadastre query params
    cad_params = dict(query_filters)
    # Request more from cadastre than we need, since landscape filters may reduce
    # the set. Fetch up to 5x the requested limit (capped at 1000).
    cad_fetch_limit = min((limit + offset) * 5, 1000)
    cad_params['limit'] = cad_fetch_limit
    cad_params['offset'] = 0  # We paginate after landscape filtering

    # Query cadastre
    try:
        cad_resp = cadastre_proxy('/query', params=cad_params)
    except CadastreError as e:
        return {'results': [], 'total': 0, '_warnings': [str(e)],
                'meta': {'cadastre_error': True}}

    cad_parcels = cad_resp.get('data', [])
    cad_stats = cad_resp.get('stats', {})
    cad_meta = cad_resp.get('meta', {})

    if not cad_parcels:
        return {'results': [], 'total': 0, 'meta': {
            'cadastre_total': cad_meta.get('total', 0),
            'cadastre_stats': cad_stats,
        }}

    # Group parcels by KG for efficient landscape loading
    kg_codes = set()
    for p in cad_parcels:
        pid = p.get('parcel_id', '')
        if '-' in pid:
            kg_codes.add(pid.split('-')[0])

    # Load KG landscape data
    kg_json_cache: dict[str, dict | None] = {}
    kg_index_cache: dict[str, dict | None] = {}
    for kg_code in kg_codes:
        kg_json_cache[kg_code] = _load_kg_json(kg_code)
        kg_index_cache[kg_code] = _extract_landscape_from_kg_index(kg_code)

    # Enrich and filter
    lf = landscape_filters or {}
    enriched = []

    for p in cad_parcels:
        pid = p.get('parcel_id', '')
        if '-' not in pid:
            continue
        kg_code = pid.split('-')[0]

        # Landscape enrichment
        kg_data = kg_json_cache.get(kg_code)
        parcel_detail = _find_parcel_in_kg(kg_data, pid) if kg_data else None

        if parcel_detail:
            landscape = _extract_landscape_from_parcel(parcel_detail)
            landscape['_source'] = 'parcel_json'
        else:
            kg_idx = kg_index_cache.get(kg_code)
            if kg_idx:
                landscape = {
                    '_source': 'kg_index',
                    'ndvi_mean': kg_idx.get('ndvi_mean'),
                    'vegetated_fraction': kg_idx.get('vegetated_fraction'),
                    'elevation_mean_m': kg_idx.get('elevation_mean_m'),
                    'slope_mean_deg': kg_idx.get('slope_mean_deg'),
                    'tree_canopy_sqm': kg_idx.get('tree_canopy_sqm'),
                    'tree_count': kg_idx.get('tree_count'),
                    'dominant_type': kg_idx.get('dominant_type'),
                    'quality_score': kg_idx.get('quality_score'),
                }
            else:
                landscape = {}

        # Apply landscape filters
        veg_frac = landscape.get('vegetated_fraction', 0) or 0
        ndvi_val = landscape.get('ndvi_mean', 0) or 0
        tree_sqm = landscape.get('tree_canopy_sqm', 0) or 0
        elev = landscape.get('elevation_mean_m') or landscape.get('elevation_mean')

        if lf.get('min_vegetated_fraction') is not None and veg_frac < lf['min_vegetated_fraction']:
            continue
        if lf.get('max_vegetated_fraction') is not None and veg_frac > lf['max_vegetated_fraction']:
            continue
        if lf.get('min_ndvi') is not None and ndvi_val < lf['min_ndvi']:
            continue
        if lf.get('max_ndvi') is not None and ndvi_val > lf['max_ndvi']:
            continue
        if lf.get('min_tree_canopy_sqm') is not None and tree_sqm < lf['min_tree_canopy_sqm']:
            continue
        if elev is not None:
            if lf.get('min_elevation') is not None and elev < lf['min_elevation']:
                continue
            if lf.get('max_elevation') is not None and elev > lf['max_elevation']:
                continue
        if lf.get('dominant_type') and landscape.get('dominant_type') != lf['dominant_type']:
            continue

        # Conservation score
        legal_refs = p.get('legal_refs')
        pa_relation = p.get('protected_area_relation')
        cons_score = compute_conservation_score(
            legal_refs=legal_refs,
            protected_area_relation=pa_relation,
            vegetated_fraction=veg_frac,
            ndvi=ndvi_val,
            tree_canopy_sqm=tree_sqm,
        )

        if lf.get('min_conservation_score') is not None and cons_score < lf['min_conservation_score']:
            continue

        enriched.append({
            'parcel_id': pid,
            'kg_code': kg_code,
            'cadastre': {
                'area_sqm': p.get('area_sqm'),
                'landuse_codes': p.get('landuse_codes'),
                'landuse_summary': p.get('landuse_summary'),
                'ez': p.get('ez'),
                'status': p.get('status'),
                'building_count': p.get('building_count'),
                'lon': p.get('lon'),
                'lat': p.get('lat'),
                'legal_refs': legal_refs,
                'legal_contexts': p.get('legal_contexts'),
            },
            'landscape': landscape,
            'conservation_score': cons_score,
        })

    total = len(enriched)

    # Sort by conservation score desc by default
    sort_key = lf.get('sort', 'conservation_score')
    sort_desc = lf.get('sort_dir', 'desc').lower() == 'desc'
    if sort_key == 'conservation_score':
        enriched.sort(key=lambda x: x.get('conservation_score', 0), reverse=sort_desc)
    elif sort_key == 'area':
        enriched.sort(key=lambda x: (x.get('cadastre') or {}).get('area_sqm', 0), reverse=sort_desc)
    elif sort_key == 'ndvi':
        enriched.sort(key=lambda x: (x.get('landscape') or {}).get('ndvi_mean', 0) or 0, reverse=sort_desc)
    elif sort_key == 'tree_canopy':
        enriched.sort(key=lambda x: (x.get('landscape') or {}).get('tree_canopy_sqm', 0) or 0, reverse=sort_desc)
    elif sort_key == 'vegetated_fraction':
        enriched.sort(key=lambda x: (x.get('landscape') or {}).get('vegetated_fraction', 0) or 0, reverse=sort_desc)

    # Paginate
    page = enriched[offset:offset + limit]

    result = {
        'results': page,
        'total': total,
        'offset': offset,
        'limit': limit,
        'meta': {
            'cadastre_total': cad_meta.get('total', 0),
            'cadastre_stats': cad_stats,
            'landscape_kgs_loaded': len(kg_codes),
            'landscape_kgs_with_json': sum(1 for v in kg_json_cache.values() if v),
            'landscape_kgs_with_index': sum(1 for v in kg_index_cache.values() if v),
            'query_filters': query_filters,
            'landscape_filters': landscape_filters,
        },
    }
    if warnings:
        result['_warnings'] = warnings
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Landscape-first parcel query (compound → parcels)
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_buildings_to_parcels(
        kg_codes: list[str],
        kg_json_map: dict[str, dict],
        roof_type: str | None = None,
        min_stories: int | None = None,
        max_stories: int | None = None,
) -> set[str] | None:
    """For each KG, collect building_footprints centroids that match the
    roof_type / stories filters, batch-lookup their parcel via
    POST /spatial/points (true point-in-polygon), and return the set of
    parcel_ids that contain at least one matching building.

    Returns None if no building filters are active (caller should not filter).
    Returns an empty set if filters are active but no buildings match.
    """
    if roof_type is None and min_stories is None and max_stories is None:
        return None

    # Collect candidate buildings from all KG JSONs
    candidates = []  # [{lon, lat, id=building_uid, stories, roof}]
    for kg_code in kg_codes:
        kg_data = kg_json_map.get(kg_code)
        if not kg_data:
            continue
        for b in kg_data.get('building_footprints', {}).get('details', []):
            s = b.get('stories_est')
            r = b.get('roof_type_hint', '')
            # Pre-filter: only buildings that match the criteria
            if min_stories is not None and (s is None or s < min_stories):
                continue
            if max_stories is not None and (s is None or s > max_stories):
                continue
            if roof_type is not None and r != roof_type:
                continue
            c = b.get('centroid', {})
            lon, lat = c.get('lon'), c.get('lat')
            if lon is None or lat is None:
                continue
            # uid = kg_code + index for dedup (building_id may be empty)
            uid = f"{kg_code}__{len(candidates)}"
            candidates.append({'lon': lon, 'lat': lat, 'id': uid})

    if not candidates:
        return set()

    # Batch POST /spatial/points in chunks of 1000
    matched_parcel_ids: set[str] = set()
    CHUNK = 1000
    for i in range(0, len(candidates), CHUNK):
        chunk = candidates[i:i + CHUNK]
        try:
            resp = cadastre_proxy(
                '/spatial/points',
                json_body={'points': chunk},
                timeout=60,
            )
            for result in resp.get('results', []):
                parcel = result.get('parcel')
                if parcel and parcel.get('parcel_id'):
                    matched_parcel_ids.add(parcel['parcel_id'])
        except CadastreError as e:
            log.warning('_resolve_buildings_to_parcels: cadastre error: %s', e)
            # On error return None so caller skips filtering rather than
            # incorrectly dropping all parcels
            return None

    return matched_parcel_ids


def landscape_parcel_query(compound_filters: dict,
                           parcel_filters: dict | None = None,
                           cadastre_enrich: bool = True,
                           limit: int = 100, offset: int = 0) -> dict:
    """Landscape-first parcel query: compound filter on KGs → expand to parcels.

    This is the core power query. It starts from OUR landscape index
    (RF classification confidence, aspect, roughness, elevation, NDVI, trees,
    etc.), finds matching KGs, then loads per-parcel landscape detail from
    our KG JSONs, applies parcel-level post-filters, and optionally enriches
    with cadastre data.

    Flow:
        1. compound_filters → search index → matching KG codes
        2. Load KG JSONs → extract per-parcel details (area_summary,
           classification, heights, vegetated_fraction, elevation)
        3. Apply parcel_filters (per-parcel landscape post-filters)
        4. Optionally enrich with cadastre data (landuse, legal refs, etc.)
        5. Score, sort, paginate, return

    Args:
        compound_filters: Any filters accepted by SearchIndex.query_compound():
            bbox, state, district, gemeinde, aspect, dominant_type, phenology,
            quality_grade, min_slope, min_roughness, min_elevation, max_elevation,
            min_tree_count, min_tree_canopy_sqm, min_ndvi, min_vegetated_fraction,
            max_buildings, min_new_buildings, min_confidence, min_rf_confidence,
            type_filters: [{type, min_confidence, min_area_sqm}, ...],
            landcover_filters: [{type, min_area_sqm, min_fraction}, ...],
            sort, sort_dir.
        parcel_filters: Per-parcel post-filters on our JSON data:
            min_vegetated_fraction: float (0-1)
            max_vegetated_fraction: float (0-1)
            min_elevation: float (metres)
            max_elevation: float (metres)
            types: list[str] — require these types in parcel area_summary
            min_type_fraction: float — min fraction for each required type
            min_ndsm_max: float — min nDSM max height (metres)
            max_ndsm_max: float — max nDSM max height
            min_parcel_area: float — min parcel area (sqm)
            max_parcel_area: float — max parcel area (sqm)
            is_vegetated: bool — only vegetated parcels
            min_slope: float — min slope_mean_deg
            max_slope: float — max slope_mean_deg
            min_tri: float — min TRI (terrain ruggedness index)
            max_tri: float — max TRI
            terrain_class: str — exact match (level, nearly_level, slightly_rugged, ...)
            aspect: str or list — dominant aspect (N, NE, E, SE, S, SW, W, NW)

            --- Per-type classification confidence (from classification.by_type) ---
            type_confidence: list[dict] — per-type confidence filters, e.g.:
              [{"type": "tree", "min_confidence": 0.7}]              combined
              [{"type": "tree", "min_rf_confidence": 0.8}]           RF only
              [{"type": "tree", "min_rules_count": 5}]              rules only
              [{"type": "tree", "min_confidence": 0.7, "min_area_sqm": 500}]
              [{"type": "tree", "max_diverged_pct": 10}]            low divergence
              Each dict: type (required), min_confidence, min_rf_confidence,
                         min_area_sqm, min_fraction, min_rf_count, min_rules_count,
                         max_diverged_pct
            min_confidence: float — min mean overall confidence (all types)
            min_rf_confidence: float — min mean RF confidence (all types)

            cadastre_landuse: str — require cadastre landuse code/abbr
            cadastre_has_buildings: bool — building presence filter
            cadastre_min_area: float — min cadastre area (sqm)
            cadastre_max_area: float — max cadastre area (sqm)

            --- Building attribute filters (resolved via /spatial/points centroid lookup) ---
            roof_type: str — "pitched" or "flat" — parcel must contain a building with this roof
            min_stories: int — parcel must contain a building with stories_est >= N
            max_stories: int — parcel must contain a building with stories_est <= N
            (These three work together: parcel must have ≥1 building matching ALL active criteria.)
            sort: str — conservation_score|vegetated_fraction|elevation|
                        ndsm_max|parcel_area
            sort_dir: str — asc|desc (default desc)
        cadastre_enrich: Whether to fetch cadastre data for matched parcels.
        limit: Max results (default 100, max 1000).
        offset: Pagination offset.

    Returns:
        {results: [{parcel_id, kg_code, kg_name, landscape: {...},
                    cadastre: {...}|null, conservation_score}],
         total, offset, limit,
         meta: {kgs_matched, kgs_with_json, parcels_scanned, ...}}
    """
    warnings = []
    idx = si.get_index()

    # Step 1: Compound query to find matching KGs
    # Fetch up to 500 KGs (we'll scan their JSONs for parcels)
    kg_limit = min(compound_filters.pop('kg_limit', 500), 500)
    kg_result = idx.query_compound(compound_filters, limit=kg_limit, offset=0)
    kg_codes = [r['kg_code'] for r in kg_result.get('results', [])]

    if not kg_codes:
        return {
            'results': [], 'total': 0, 'offset': offset, 'limit': limit,
            'meta': {'kgs_matched': 0, 'kgs_with_json': 0, 'parcels_scanned': 0,
                     'compound_total': kg_result.get('total', 0)}
        }

    # Step 2: Load KG JSONs, extract per-parcel details
    pf = parcel_filters or {}
    all_parcels = []
    kgs_with_json = 0
    parcels_scanned = 0

    # Cache loaded KG JSONs so step 3b (building resolver) can reuse them
    kg_json_map: dict[str, dict] = {}

    for kg_code in kg_codes:
        kg_data = _load_kg_json(kg_code)
        if not kg_data:
            continue
        kgs_with_json += 1
        kg_json_map[kg_code] = kg_data

        kg_name = kg_data.get('kg_name', '')
        kg_state = kg_data.get('state', '')

        for pd in kg_data.get('parcels', {}).get('details', []):
            parcels_scanned += 1
            pid = pd.get('parcel_id', '')
            if not pid:
                continue

            # Step 3: Apply parcel-level filters

            # Parcel area filter
            p_area = pd.get('area_sqm', 0)
            if pf.get('min_parcel_area') and p_area < pf['min_parcel_area']:
                continue
            if pf.get('max_parcel_area') and p_area > pf['max_parcel_area']:
                continue

            # Vegetation filter
            veg_frac = pd.get('vegetated_fraction', 0) or 0
            if pf.get('min_vegetated_fraction') is not None and veg_frac < pf['min_vegetated_fraction']:
                continue
            if pf.get('max_vegetated_fraction') is not None and veg_frac > pf['max_vegetated_fraction']:
                continue
            if pf.get('is_vegetated') is not None:
                is_veg = pd.get('is_vegetated', False)
                if pf['is_vegetated'] != is_veg:
                    continue

            # Forested fraction filter
            forest_frac = pd.get('forested_fraction', 0) or 0
            if pf.get('min_forested_fraction') is not None and forest_frac < pf['min_forested_fraction']:
                continue
            if pf.get('max_forested_fraction') is not None and forest_frac > pf['max_forested_fraction']:
                continue

            # Dominant type filter
            if pf.get('dominant_type'):
                if pd.get('dominant_type') != pf['dominant_type']:
                    continue

            # Hansen recent forest loss filter
            hansen = pd.get('hansen_loss', {})
            if pf.get('max_hansen_recent_5yr') is not None:
                recent = hansen.get('recent_5yr_pixels', 0)
                if recent > pf['max_hansen_recent_5yr']:
                    continue
            if pf.get('min_hansen_recent_5yr') is not None:
                recent = hansen.get('recent_5yr_pixels', 0)
                if recent < pf['min_hansen_recent_5yr']:
                    continue
            if pf.get('max_hansen_total') is not None:
                total_h = hansen.get('total_pixels', 0)
                if total_h > pf['max_hansen_total']:
                    continue
            if pf.get('min_hansen_total') is not None:
                total_h = hansen.get('total_pixels', 0)
                if total_h < pf['min_hansen_total']:
                    continue

            # Elevation filter
            elev = pd.get('elevation_m')
            if elev is not None:
                if pf.get('min_elevation') is not None and elev < pf['min_elevation']:
                    continue
                if pf.get('max_elevation') is not None and elev > pf['max_elevation']:
                    continue

            # Slope filter
            slope = pd.get('slope_mean_deg')
            if slope is not None:
                if pf.get('min_slope') is not None and slope < pf['min_slope']:
                    continue
                if pf.get('max_slope') is not None and slope > pf['max_slope']:
                    continue

            # TRI filter
            tri = pd.get('tri_mean')
            if tri is not None:
                if pf.get('min_tri') is not None and tri < pf['min_tri']:
                    continue
                if pf.get('max_tri') is not None and tri > pf['max_tri']:
                    continue

            # Terrain class filter
            if pf.get('terrain_class'):
                if pd.get('terrain_class') != pf['terrain_class']:
                    continue

            # Aspect filter
            if pf.get('aspect'):
                aspects = pf['aspect'] if isinstance(pf['aspect'], list) else [pf['aspect']]
                if pd.get('aspect_dominant') not in aspects:
                    continue

            # nDSM height filter
            ndsm_max = pd.get('ndsm_max_m')
            if ndsm_max is not None:
                if pf.get('min_ndsm_max') is not None and ndsm_max < pf['min_ndsm_max']:
                    continue
                if pf.get('max_ndsm_max') is not None and ndsm_max > pf['max_ndsm_max']:
                    continue

            # Type presence filter
            area_summary = pd.get('area_summary', {})
            required_types = pf.get('types')
            if required_types:
                min_frac = pf.get('min_type_fraction', 0)
                if not all(
                    area_summary.get(t, {}).get('fraction', 0) >= min_frac
                    for t in required_types
                ):
                    continue

            # Classification confidence filters
            cls = pd.get('classification', {})
            by_type = cls.get('by_type', {})

            # Overall confidence (combined RF + rules)
            if pf.get('min_confidence') is not None:
                if (cls.get('mean_confidence') or 0) < pf['min_confidence']:
                    continue

            # Overall RF confidence
            if pf.get('min_rf_confidence') is not None:
                if (cls.get('rf_mean_confidence') or 0) < pf['min_rf_confidence']:
                    continue

            # Per-type confidence filters
            # Each entry: {type, min_confidence, min_rf_confidence,
            #   min_area_sqm, min_fraction, min_rf_count, min_rules_count,
            #   max_diverged_pct}
            type_conf_filters = pf.get('type_confidence')
            if type_conf_filters:
                skip = False
                for tcf in type_conf_filters:
                    t = tcf.get('type', '')
                    bt = by_type.get(t, {})
                    # Type must exist in parcel
                    if not bt and not area_summary.get(t):
                        skip = True; break
                    # Combined confidence (RF or rules, whichever classified)
                    if tcf.get('min_confidence') is not None:
                        if (bt.get('mean_confidence') or 0) < tcf['min_confidence']:
                            skip = True; break
                    # RF-only confidence
                    if tcf.get('min_rf_confidence') is not None:
                        if (bt.get('rf_mean_confidence') or 0) < tcf['min_rf_confidence']:
                            skip = True; break
                    # Area (from classification or area_summary)
                    if tcf.get('min_area_sqm') is not None:
                        t_area = bt.get('area_sqm') or area_summary.get(t, {}).get('area_sqm', 0)
                        if t_area < tcf['min_area_sqm']:
                            skip = True; break
                    # Fraction (from area_summary)
                    if tcf.get('min_fraction') is not None:
                        if area_summary.get(t, {}).get('fraction', 0) < tcf['min_fraction']:
                            skip = True; break
                    # RF count (how many segments classified by RF)
                    if tcf.get('min_rf_count') is not None:
                        if (bt.get('rf_count') or 0) < tcf['min_rf_count']:
                            skip = True; break
                    # Rules count
                    if tcf.get('min_rules_count') is not None:
                        if (bt.get('rules_count') or 0) < tcf['min_rules_count']:
                            skip = True; break
                    # Divergence (RF predicted X but final was Y)
                    if tcf.get('max_diverged_pct') is not None:
                        segs = bt.get('segments', 0) or 1
                        div = bt.get('diverged_count', 0) or 0
                        if (100 * div / segs) > tcf['max_diverged_pct']:
                            skip = True; break
                if skip:
                    continue

            # Build landscape dict
            landscape = _extract_landscape_from_parcel(pd)
            landscape['_source'] = 'parcel_json'

            entry = {
                'parcel_id': pid,
                'kg_code': kg_code,
                'kg_name': kg_name,
                'state': kg_state,
                'landscape': landscape,
                'cadastre': None,  # filled in step 4
            }
            if pd.get('centroid'):
                entry['centroid'] = pd['centroid']
            if pd.get('elevation_m') is not None:
                entry['elevation_m'] = pd['elevation_m']

            all_parcels.append(entry)

    # Step 3b: Building attribute filter via /spatial/points centroid lookup
    # Resolve which parcel_ids contain a building matching roof_type / stories.
    # Uses building_footprint centroids from the KG JSON + cadastre point-in-polygon.
    _roof_type   = pf.get('roof_type')
    _min_stories = pf.get('min_stories')
    _max_stories = pf.get('max_stories')
    if _roof_type is not None or _min_stories is not None or _max_stories is not None:
        matched_by_building = _resolve_buildings_to_parcels(
            kg_codes=list(kg_json_map.keys()),
            kg_json_map=kg_json_map,
            roof_type=_roof_type,
            min_stories=_min_stories,
            max_stories=_max_stories,
        )
        if matched_by_building is not None:
            # None = lookup failed (error) → skip filter to avoid false negatives
            all_parcels = [e for e in all_parcels
                           if e['parcel_id'] in matched_by_building]
        else:
            warnings.append(
                'Building filter (roof_type/stories) skipped due to /spatial/points error'
            )

    total_matched = len(all_parcels)

    # Step 4: Optionally enrich with cadastre data
    if cadastre_enrich and all_parcels:
        # Group by KG, batch-query cadastre for each KG's parcels
        from collections import defaultdict
        kg_pids: dict[str, list[int]] = defaultdict(list)  # kg → indices into all_parcels
        for i, entry in enumerate(all_parcels):
            kg_pids[entry['kg_code']].append(i)

        for kg_code, indices in kg_pids.items():
            try:
                cad_resp = cadastre_proxy('/search/parcel', params={
                    'kg': kg_code,
                    'limit': 10000,
                })
                # Build lookup by parcel_id
                cad_map = {}
                for item in cad_resp.get('data', []):
                    cpid = item.get('parcel_id')
                    if cpid:
                        cad_map[cpid] = item

                for idx_i in indices:
                    pid = all_parcels[idx_i]['parcel_id']
                    cad = cad_map.get(pid)
                    if cad:
                        all_parcels[idx_i]['cadastre'] = {
                            'area_sqm': cad.get('area_sqm'),
                            'landuse_codes': cad.get('landuse_codes'),
                            'landuse_summary': cad.get('landuse_summary'),
                            'ez': cad.get('ez'),
                            'building_count': cad.get('building_count'),
                            'legal_refs': cad.get('legal_refs'),
                            'legal_contexts': cad.get('legal_contexts'),
                            'in_natura2000': cad.get('in_natura2000'),
                            'natura2000_sites': cad.get('natura2000_sites'),
                        }
            except CadastreError as e:
                warnings.append(f'Cadastre enrichment failed for KG {kg_code}: {e}')

        # Apply cadastre-side post-filters (only possible after enrichment)
        if any(k.startswith('cadastre_') for k in pf):
            filtered = []
            for entry in all_parcels:
                cad = entry.get('cadastre') or {}

                if pf.get('cadastre_has_buildings') is not None:
                    bc = cad.get('building_count', 0) or 0
                    if pf['cadastre_has_buildings'] and bc == 0:
                        continue
                    if not pf['cadastre_has_buildings'] and bc > 0:
                        continue

                if pf.get('cadastre_min_area') is not None:
                    ca = cad.get('area_sqm', 0) or 0
                    if ca < pf['cadastre_min_area']:
                        continue

                if pf.get('cadastre_max_area') is not None:
                    ca = cad.get('area_sqm', 0) or 0
                    if ca > pf['cadastre_max_area']:
                        continue

                if pf.get('cadastre_in_natura2000') is not None:
                    in_n2k = bool(cad.get('in_natura2000'))
                    if pf['cadastre_in_natura2000'] and not in_n2k:
                        continue
                    if not pf['cadastre_in_natura2000'] and in_n2k:
                        continue

                if pf.get('cadastre_natura2000_site'):
                    sites = cad.get('natura2000_sites') or []
                    target = pf['cadastre_natura2000_site']
                    if not any((s.get('sitecode') == target) for s in sites):
                        continue

                if pf.get('cadastre_natura2000_type'):
                    sites = cad.get('natura2000_sites') or []
                    target = str(pf['cadastre_natura2000_type']).upper()
                    # A=Birds/SPA, B=Habitats/SCI-SAC, C=both. 'C' satisfies A or B.
                    def _type_match(s):
                        st = (s.get('sitetype') or '').upper()
                        if target == 'C':
                            return st in ('A', 'B', 'C')
                        return st == target or st == 'C'
                    if not any(_type_match(s) for s in sites):
                        continue

                if pf.get('cadastre_natura2000_habitat'):
                    sites = cad.get('natura2000_sites') or []
                    target = str(pf['cadastre_natura2000_habitat']).lower()
                    if not any(target in [h.lower() for h in (s.get('habitats') or [])]
                               for s in sites):
                        continue

                if pf.get('cadastre_landuse'):
                    codes = cad.get('landuse_codes', '') or ''
                    summary = cad.get('landuse_summary', {}) or {}
                    target = pf['cadastre_landuse']
                    # Match against codes string or summary keys
                    if target not in codes and not any(target.lower() in k.lower() for k in summary):
                        continue

                filtered.append(entry)
            all_parcels = filtered
            total_matched = len(all_parcels)

    # Compute conservation scores
    for entry in all_parcels:
        legal_refs = (entry.get('cadastre') or {}).get('legal_refs')
        veg_frac = (entry.get('landscape') or {}).get('vegetated_fraction', 0) or 0
        ndvi_val = (entry.get('landscape') or {}).get('ndvi_mean', 0) or 0
        tree_sqm = (entry.get('landscape') or {}).get('tree_canopy_sqm', 0) or 0
        entry['conservation_score'] = compute_conservation_score(
            legal_refs=legal_refs,
            vegetated_fraction=veg_frac,
            ndvi=ndvi_val,
            tree_canopy_sqm=tree_sqm,
        )

    # Sort
    sort_key = pf.get('sort', 'conservation_score')
    sort_desc = pf.get('sort_dir', 'desc').lower() != 'asc'
    sort_funcs = {
        'conservation_score': lambda e: e.get('conservation_score', 0),
        'vegetated_fraction': lambda e: (e.get('landscape') or {}).get('vegetated_fraction', 0) or 0,
        'forested_fraction': lambda e: (e.get('landscape') or {}).get('forested_fraction', 0) or 0,
        'elevation': lambda e: e.get('elevation_m') or 0,
        'ndsm_max': lambda e: (e.get('landscape') or {}).get('ndsm_max_m', 0) or 0,
        'parcel_area': lambda e: e.get('landscape', {}).get('area_summary', {}).get('tree', {}).get('area_sqm', 0),
        'hansen_recent': lambda e: (e.get('landscape') or {}).get('hansen_loss', {}).get('recent_5yr_pixels', 0),
        'hansen_total': lambda e: (e.get('landscape') or {}).get('hansen_loss', {}).get('total_pixels', 0),
    }
    sort_fn = sort_funcs.get(sort_key, sort_funcs['conservation_score'])
    all_parcels.sort(key=sort_fn, reverse=sort_desc)

    # Paginate
    page = all_parcels[offset:offset + limit]

    result = {
        'results': page,
        'total': total_matched,
        'offset': offset,
        'limit': limit,
        'meta': {
            'compound_total_kgs': kg_result.get('total', 0),
            'kgs_matched': len(kg_codes),
            'kgs_with_json': kgs_with_json,
            'parcels_scanned': parcels_scanned,
            'parcels_matched': total_matched,
            'cadastre_enriched': cadastre_enrich,
        },
    }
    if warnings:
        result['_warnings'] = warnings
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Nature conservation screening
# ═══════════════════════════════════════════════════════════════════════════

def nature_conservation_screen(bbox: str | None = None,
                               state: str | None = None,
                               district: str | None = None,
                               gemeinde: str | None = None,
                               protected_area: str | None = None,
                               legal_context: str | None = None,
                               min_vegetated_fraction: float | None = None,
                               min_ndvi: float | None = None,
                               min_tree_canopy_sqm: float | None = None,
                               min_area_sqm: float | None = None,
                               max_area_sqm: float | None = None,
                               landuse: str | None = None,
                               has_buildings: bool | None = None,
                               sort: str = 'conservation_score',
                               limit: int = 50,
                               offset: int = 0) -> dict:
    """Nature conservation + parcel purchase opportunity finder.

    Cross-references four data sources:
    1. Cadastre: parcels matching area/landuse/building filters
    2. Protected areas (WDPA): proximity to or containment in protected areas
    3. Legal refs (RIS): parcels referenced in nature protection laws
    4. Landscape analysis: NDVI, vegetation fraction, tree canopy, classification

    Returns parcels ranked by conservation value score (0-100).

    Args:
        bbox: Bounding box 'w,s,e,n' in WGS84.
        state/district/gemeinde: Administrative filters.
        protected_area: Protected area name (uses cadastre spatial query).
        legal_context: Legal context filter (national_park, nature_protection, etc.).
        min_vegetated_fraction: Min vegetation fraction from landscape (0-1).
        min_ndvi: Min NDVI from landscape analysis.
        min_tree_canopy_sqm: Min tree canopy area in sq metres.
        min_area_sqm/max_area_sqm: Parcel area filter.
        landuse: Cadastre landuse code/abbreviation.
        has_buildings: Building presence filter.
        sort: Sort key (conservation_score, area, ndvi, tree_canopy).
        limit/offset: Pagination.

    Returns:
        Dict with ranked results, total count, and aggregation stats.
    """
    warnings = []

    # Step 1: If protected_area specified, use the spatial protected area query
    if protected_area:
        cad_params = {'area': protected_area, 'relation': 'within', 'limit': 500}
        if landuse:
            cad_params['landuse'] = landuse
        if min_area_sqm is not None:
            cad_params['min_area'] = min_area_sqm
        if max_area_sqm is not None:
            cad_params['max_area'] = max_area_sqm
        if has_buildings is not None:
            cad_params['has_buildings'] = str(has_buildings).lower()
        if legal_context:
            cad_params['legal_context'] = legal_context

        try:
            cad_resp = cadastre_proxy('/query/protected_area', params=cad_params)
        except CadastreError as e:
            warnings.append(f'Cadastre protected area query failed: {e}')
            cad_resp = {'data': [], 'meta': {}, 'stats': {}}
    else:
        # Step 1b: Standard query with admin/spatial/attribute filters
        cad_params = {'limit': 500}
        if bbox:
            parts = bbox.split(',')
            if len(parts) == 4:
                cad_params['min_lon'] = parts[0]
                cad_params['min_lat'] = parts[1]
                cad_params['max_lon'] = parts[2]
                cad_params['max_lat'] = parts[3]
        if state:
            cad_params['state'] = state
        if district:
            cad_params['district'] = district
        if gemeinde:
            cad_params['gemeinde'] = gemeinde
        if landuse:
            cad_params['landuse'] = landuse
        if min_area_sqm is not None:
            cad_params['min_area'] = min_area_sqm
        if max_area_sqm is not None:
            cad_params['max_area'] = max_area_sqm
        if has_buildings is not None:
            cad_params['has_buildings'] = str(has_buildings).lower()
        if legal_context:
            cad_params['legal_context'] = legal_context
            cad_params['has_legal_refs'] = 'true'

        try:
            cad_resp = cadastre_proxy('/query', params=cad_params)
        except CadastreError as e:
            warnings.append(f'Cadastre query failed: {e}')
            cad_resp = {'data': [], 'meta': {}, 'stats': {}}

    cad_parcels = cad_resp.get('data', [])
    cad_stats = cad_resp.get('stats', {})

    if not cad_parcels:
        result = {'results': [], 'total': 0, 'offset': offset, 'limit': limit,
                  'stats': cad_stats}
        if warnings:
            result['_warnings'] = warnings
        return result

    # Step 2: Load landscape data for matched KGs
    kg_codes = set()
    for p in cad_parcels:
        pid = p.get('parcel_id', '')
        if '-' in pid:
            kg_codes.add(pid.split('-')[0])

    kg_json_cache: dict[str, dict | None] = {}
    kg_index_cache: dict[str, dict | None] = {}
    for kg_code in kg_codes:
        kg_json_cache[kg_code] = _load_kg_json(kg_code)
        kg_index_cache[kg_code] = _extract_landscape_from_kg_index(kg_code)

    # Step 3: Enrich parcels with landscape + score
    enriched = []
    for p in cad_parcels:
        pid = p.get('parcel_id', '')
        if '-' not in pid:
            continue
        kg_code = pid.split('-')[0]

        # Landscape data
        kg_data = kg_json_cache.get(kg_code)
        parcel_detail = _find_parcel_in_kg(kg_data, pid) if kg_data else None

        if parcel_detail:
            landscape = _extract_landscape_from_parcel(parcel_detail)
            landscape['_source'] = 'parcel_json'
        else:
            kg_idx = kg_index_cache.get(kg_code)
            if kg_idx:
                landscape = {
                    '_source': 'kg_index',
                    'ndvi_mean': kg_idx.get('ndvi_mean'),
                    'vegetated_fraction': kg_idx.get('vegetated_fraction'),
                    'elevation_mean_m': kg_idx.get('elevation_mean_m'),
                    'tree_canopy_sqm': kg_idx.get('tree_canopy_sqm'),
                }
            else:
                landscape = {}

        veg_frac = landscape.get('vegetated_fraction', 0) or 0
        ndvi_val = landscape.get('ndvi_mean', 0) or 0
        tree_sqm = landscape.get('tree_canopy_sqm', 0) or 0

        # Apply landscape filters
        if min_vegetated_fraction is not None and veg_frac < min_vegetated_fraction:
            continue
        if min_ndvi is not None and ndvi_val < min_ndvi:
            continue
        if min_tree_canopy_sqm is not None and tree_sqm < min_tree_canopy_sqm:
            continue

        # Conservation score
        pa_relation = p.get('protected_area_relation')
        legal_refs = p.get('legal_refs')
        score = compute_conservation_score(
            legal_refs=legal_refs,
            protected_area_relation=pa_relation,
            vegetated_fraction=veg_frac,
            ndvi=ndvi_val,
            tree_canopy_sqm=tree_sqm,
        )

        enriched.append({
            'parcel_id': pid,
            'kg_code': kg_code,
            'conservation_score': score,
            'cadastre': {
                'area_sqm': p.get('area_sqm'),
                'landuse_codes': p.get('landuse_codes'),
                'landuse_summary': p.get('landuse_summary'),
                'ez': p.get('ez'),
                'building_count': p.get('building_count'),
                'lon': p.get('lon'),
                'lat': p.get('lat'),
                'legal_refs': legal_refs,
                'legal_contexts': p.get('legal_contexts'),
                'legal_status': p.get('legal_status'),
                'protected_area_relation': pa_relation,
            },
            'landscape': landscape,
        })

    # Sort
    total_enriched = len(enriched)
    sort_desc = True
    if sort == 'conservation_score':
        enriched.sort(key=lambda x: x['conservation_score'], reverse=True)
    elif sort == 'area':
        enriched.sort(key=lambda x: (x.get('cadastre') or {}).get('area_sqm', 0), reverse=True)
    elif sort == 'ndvi':
        enriched.sort(key=lambda x: (x.get('landscape') or {}).get('ndvi_mean', 0) or 0, reverse=True)
    elif sort == 'tree_canopy':
        enriched.sort(key=lambda x: (x.get('landscape') or {}).get('tree_canopy_sqm', 0) or 0, reverse=True)
    elif sort == 'vegetated_fraction':
        enriched.sort(key=lambda x: (x.get('landscape') or {}).get('vegetated_fraction', 0) or 0, reverse=True)

    page = enriched[offset:offset + limit]

    result = {
        'results': page,
        'total': total_enriched,
        'offset': offset,
        'limit': limit,
        'stats': {
            'cadastre_parcels_queried': len(cad_parcels),
            'landscape_enriched': total_enriched,
            'kgs_loaded': len(kg_codes),
            'kgs_with_local_json': sum(1 for v in kg_json_cache.values() if v),
            'avg_conservation_score': round(
                sum(e['conservation_score'] for e in enriched) / max(len(enriched), 1), 1
            ),
            'cadastre_stats': cad_stats,
        },
    }
    if warnings:
        result['_warnings'] = warnings
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Single parcel detail (combined APIs)
# ═══════════════════════════════════════════════════════════════════════════

def parcel_landscape_detail(parcel_id: str) -> dict:
    """Full detail for a single parcel combining both APIs.

    From cadastre: area, landuse, EZ, buildings, geometry centroid, legal refs.
    From landscape: elevation, slope, NDVI, vegetation fraction, object types,
                    RF classification, heights, temporal changes.

    Args:
        parcel_id: Parcel ID like '63349-505/3'.

    Returns:
        Dict with 'parcel_id', 'cadastre', 'landscape', 'conservation_score',
        and '_warnings' if partial data.
    """
    if '-' not in parcel_id:
        return {'error': f'Invalid parcel_id format: {parcel_id}'}

    kg_code = parcel_id.split('-')[0]
    gnr = parcel_id.split('-', 1)[1]
    warnings = []

    # Cadastre data
    cadastre = None
    try:
        cad = cadastre_proxy(f'/search/feature', params={'id': parcel_id})
        cad_items = cad.get('data', [])
        if cad_items:
            cadastre = cad_items[0]
    except CadastreError as e:
        warnings.append(f'Cadastre lookup failed: {e}')

    # Legal refs
    legal_refs = None
    try:
        legal = cadastre_proxy(f'/legal/parcel/{kg_code}/{gnr}')
        legal_refs = legal.get('refs', [])
    except CadastreError as e:
        warnings.append(f'Legal refs lookup failed: {e}')

    # Protected area check (use parcel centroid if available)
    protected_areas = []
    if cadastre and cadastre.get('lon') and cadastre.get('lat'):
        try:
            pa = cadastre_proxy('/search/protected_area', params={
                'contains_lon': cadastre['lon'],
                'contains_lat': cadastre['lat'],
            })
            protected_areas = pa.get('data', [])
        except CadastreError as e:
            warnings.append(f'Protected area check failed: {e}')

    # Landscape data
    landscape = None
    kg_data = _load_kg_json(kg_code)
    parcel_detail = _find_parcel_in_kg(kg_data, parcel_id) if kg_data else None

    if parcel_detail:
        landscape = _extract_landscape_from_parcel(parcel_detail)
        landscape['_source'] = 'parcel_json'
        # Include full parcel detail from JSON (area_summary, all fields)
        landscape['_raw_detail'] = parcel_detail
    else:
        kg_idx = _extract_landscape_from_kg_index(kg_code)
        if kg_idx:
            landscape = {
                '_source': 'kg_index',
                'ndvi_mean': kg_idx.get('ndvi_mean'),
                'vegetated_fraction': kg_idx.get('vegetated_fraction'),
                'elevation_mean_m': kg_idx.get('elevation_mean_m'),
                'slope_mean_deg': kg_idx.get('slope_mean_deg'),
                'tree_canopy_sqm': kg_idx.get('tree_canopy_sqm'),
                'tree_count': kg_idx.get('tree_count'),
                'dominant_type': kg_idx.get('dominant_type'),
                'quality_score': kg_idx.get('quality_score'),
                'landcover': kg_idx.get('landcover', []),
                'hansen_loss': kg_idx.get('hansen_loss', []),
            }

    # Conservation score
    veg_frac = (landscape or {}).get('vegetated_fraction', 0) or 0
    ndvi_val = (landscape or {}).get('ndvi_mean', 0) or 0
    tree_sqm = (landscape or {}).get('tree_canopy_sqm', 0) or 0
    pa_relation = 'within' if protected_areas else None

    score = compute_conservation_score(
        legal_refs=legal_refs,
        protected_area_relation=pa_relation,
        vegetated_fraction=veg_frac,
        ndvi=ndvi_val,
        tree_canopy_sqm=tree_sqm,
    )

    result = {
        'parcel_id': parcel_id,
        'kg_code': kg_code,
        'conservation_score': score,
        'cadastre': cadastre,
        'legal_refs': legal_refs,
        'protected_areas': protected_areas,
        'landscape': landscape,
    }
    if warnings:
        result['_warnings'] = warnings
    return result


# ═══════════════════════════════════════════════════════════════════════════
# KG combined profile
# ═══════════════════════════════════════════════════════════════════════════

def kg_combined_profile(kg_code: str) -> dict:
    """Combined KG profile from both APIs.

    From cadastre: parcel count, building count, landuse distribution, legal refs.
    From landscape: landcover, elevation, NDVI, trees, new buildings, quality.

    Args:
        kg_code: 5-digit KG code.

    Returns:
        Dict with 'kg_code', 'cadastre', 'landscape', 'legal', and '_warnings'.
    """
    warnings = []

    # Landscape data from our search index
    landscape = _extract_landscape_from_kg_index(kg_code)
    if not landscape:
        warnings.append(f'KG {kg_code} not found in landscape index')

    # Cadastre KG info
    cadastre_kg = None
    try:
        cad = cadastre_proxy('/search/kg', params={'code': kg_code})
        cad_items = cad.get('data', [])
        if cad_items:
            cadastre_kg = cad_items[0]
    except CadastreError as e:
        warnings.append(f'Cadastre KG lookup failed: {e}')

    # Cadastre landuse distribution
    landuse_dist = None
    try:
        lu = cadastre_proxy('/landuse/distribution', params={'kg': kg_code})
        landuse_dist = lu.get('data', [])
    except CadastreError as e:
        warnings.append(f'Cadastre landuse distribution failed: {e}')

    # Legal refs
    legal = None
    try:
        lr = cadastre_proxy(f'/legal/kg/{kg_code}')
        legal = {
            'total_refs': lr.get('total_refs', 0),
            'unique_laws': lr.get('unique_laws', 0),
            'legal_contexts': lr.get('legal_contexts', []),
            'refs': lr.get('refs', [])[:20],  # First 20 refs as sample
        }
    except CadastreError as e:
        warnings.append(f'Legal refs lookup failed: {e}')

    # Local JSON summary if available
    json_summary = None
    kg_data = _load_kg_json(kg_code)
    if kg_data:
        json_summary = {
            'parcel_count': len(kg_data.get('parcels', {}).get('details', [])),
            'building_summary': kg_data.get('buildings', {}).get('summary') if kg_data.get('buildings') else None,
            'summary': kg_data.get('summary'),
        }

    result = {
        'kg_code': kg_code,
        'cadastre': cadastre_kg,
        'cadastre_landuse': landuse_dist,
        'legal': legal,
        'landscape': landscape,
        'local_json_summary': json_summary,
    }
    if warnings:
        result['_warnings'] = warnings
    return result


# ═════════════════════════════════════════════════════════════════════════
# Processed-set intersection helpers (no per-KG enrichment, fast cross-API stats)
# ═════════════════════════════════════════════════════════════════════════

_PROCESSED_KGS_CACHE: dict[str, Any] = {'set': None, 'ts': 0.0}
_PROCESSED_KGS_TTL_S = 60.0

def _processed_kg_codes() -> set[str]:
    """Set of KG codes we have already processed (processed=1 in search_index).

    Cached 60s; cheap to refresh.
    """
    now = time.time()
    cached = _PROCESSED_KGS_CACHE.get('set')
    if cached is not None and (now - _PROCESSED_KGS_CACHE['ts']) < _PROCESSED_KGS_TTL_S:
        return cached
    codes: set[str] = set()
    try:
        rows = si.get_index()._conn().execute(
            'SELECT kg_code FROM kg WHERE processed=1'
        ).fetchall()
        codes = {r[0] for r in rows}
    except Exception as e:
        log.warning('processed_kg_codes lookup failed: %s', e)
    _PROCESSED_KGS_CACHE['set'] = codes
    _PROCESSED_KGS_CACHE['ts'] = now
    return codes


def habitat_processed_count(
    habitat: str | None = None,
    sitetype: str | None = None,
    sitecode: str | None = None,
    state: str | None = None,
    district: str | None = None,
    gemeinde: str | None = None,
    landuse: str | None = None,
    has_buildings: bool | None = None,
    breakdown: bool = False,
    limit: int = 0,
) -> dict:
    """Count Natura 2000 parcels (filtered by habitat / type / site / admin /
    landuse) that fall in a KG we have already processed.

    Single cadastre round-trip (indexed natura2000_parcel_index +
    natura2000_parcel_habitats join), then in-memory set intersection against
    our processed_kgs. No per-KG enrichment loop.

    Args:
      habitat: moor|floodplain|river|lake|wetland|forest|alpine|meadow|
               pasture|valley|hill|steppe|orchard|park|cave
      sitetype: A=Birds/SPA, B=Habitats/SCI-SAC, C=both (C matches A or B too)
      sitecode: restrict to a specific Natura 2000 sitecode (e.g. AT1205A00)
      state / district / gemeinde / landuse / has_buildings: cadastre filters
      breakdown: also return per-KG and per-state counts
      limit: if >0, also return up to `limit` parcel_ids (intersected)

    Returns:
      {
        filters: {...},
        cadastre_total: int           # matches in cadastre across all AT
        processed_total: int          # of those, in our processed set
        processed_fraction: 0..1
        unique_kgs_total: int
        unique_kgs_processed: int
        unprocessed_total: int        # cadastre_total - processed_total
        by_kg: {kg_code: count}       # only if breakdown=true
        by_state: {state: count}      # only if breakdown=true
        parcel_ids: [...]             # only if limit>0
        query_time_ms: float
      }
    """
    if not any([habitat, sitetype, sitecode]):
        raise ValueError('habitat_processed_count needs at least one of habitat/sitetype/sitecode')

    PAGE = 100000  # cadastre hard cap
    params: dict[str, Any] = {
        'limit': PAGE,
        'fields': 'parcel_id,kg_code,state_name',
        'with_stats': 'false',
    }
    if habitat:    params['natura2000_habitat'] = habitat
    if sitetype:   params['natura2000_type'] = sitetype
    if sitecode:   params['natura2000_site'] = sitecode
    elif not (habitat or sitetype):
        params['has_natura2000'] = 'true'
    if state:      params['state'] = state
    if district:   params['district'] = district
    if gemeinde:   params['gemeinde'] = gemeinde
    if landuse:    params['landuse'] = landuse
    if has_buildings is not None:
        params['has_buildings'] = 'true' if has_buildings else 'false'

    t0 = time.time()
    processed_set = _processed_kg_codes()

    by_kg_all: dict[str, int] = {}
    by_kg_proc: dict[str, int] = {}
    by_state_proc: dict[str, int] = {}
    processed_ids: list[str] = []
    processed_total = 0
    rows_seen = 0
    cadastre_total = 0

    # Auto-paginate through cadastre so big habitats (forest=~214k) aren't
    # truncated. ~10 pages max for the biggest tag; each page ~1.5s.
    offset = 0
    pages = 0
    MAX_PAGES = 25  # safety: 2.5M parcels worst case
    while True:
        params['offset'] = offset
        resp = cadastre_proxy('/query', params=params)
        rows = resp.get('data') or []
        if pages == 0:
            cadastre_total = int((resp.get('meta') or {}).get('total') or len(rows))
        rows_seen += len(rows)
        for r in rows:
            kg = r.get('kg_code')
            if not kg:
                continue
            by_kg_all[kg] = by_kg_all.get(kg, 0) + 1
            if kg in processed_set:
                processed_total += 1
                by_kg_proc[kg] = by_kg_proc.get(kg, 0) + 1
                st = r.get('state_name') or ''
                if st:
                    by_state_proc[st] = by_state_proc.get(st, 0) + 1
                if limit > 0 and len(processed_ids) < limit:
                    pid = r.get('parcel_id')
                    if pid:
                        processed_ids.append(pid)
        pages += 1
        if len(rows) < PAGE or rows_seen >= cadastre_total or pages >= MAX_PAGES:
            break
        offset += PAGE

    filters_out = {k: v for k, v in {
        'habitat': habitat, 'sitetype': sitetype, 'sitecode': sitecode,
        'state': state, 'district': district, 'gemeinde': gemeinde,
        'landuse': landuse, 'has_buildings': has_buildings,
    }.items() if v is not None}
    out: dict[str, Any] = {
        'filters': filters_out,
        'cadastre_total': cadastre_total,
        'processed_total': processed_total,
        'unprocessed_total': max(cadastre_total - processed_total, 0),
        'processed_fraction': round(processed_total / cadastre_total, 4) if cadastre_total else 0.0,
        'unique_kgs_total': len(by_kg_all),
        'unique_kgs_processed': len(by_kg_proc),
        'query_time_ms': round((time.time() - t0) * 1000, 1),
    }
    if rows_seen < cadastre_total:
        out['truncated'] = True
        out['rows_scanned'] = rows_seen
        out['pages_scanned'] = pages
    if breakdown:
        out['by_kg'] = dict(sorted(by_kg_proc.items(), key=lambda x: -x[1]))
        out['by_state'] = dict(sorted(by_state_proc.items(), key=lambda x: -x[1]))
    if limit > 0:
        out['parcel_ids'] = processed_ids
    return out
