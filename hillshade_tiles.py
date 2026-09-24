"""hillshade_tiles — XYZ hillshade / slope PNG tiles from the product-2.1
``terrain.grid25`` (25 m DTM) already held in ``data/kg_v2_store.db``
(siedler LID-4).

Rendered on the fly (numpy, ~20 ms/tile), no BEV / Zenodo access, no disk
cache (z12-17 over Austria would be tens of GB); ``Cache-Control`` is 1 y so
browsers + the exe.dev proxy hold them. Resolution is the 25 m grid (header
``X-Resolution-M: 25``) — at z17 it is smooth, but every pixel is real relief.
204 when no indexed KG with grid25 intersects the tile (or all nodata).

nDSM tiles are NOT available: grid25 carries DTM + slope + canopy fraction
only; a 1 m nDSM raster per KG lives in the full GPKG (hundreds of MB) and is
out of the primary's disk/bandwidth budget → ``/tiles/ndsm/…`` answers 501.
"""
from __future__ import annotations

import io
import logging
import math
import threading
from collections import OrderedDict

import numpy as np
from flask import Blueprint, Response, jsonify

log = logging.getLogger(__name__)
bp = Blueprint('hillshade_tiles', __name__)

TILE = 256
ZMIN, ZMAX = 10, 17
_grid_cache: "OrderedDict[str, list]" = OrderedDict()
_grid_lock = threading.Lock()
_GRID_CACHE_MAX = 600          # ~65 KB each → ≤40 MB
_tf = None


def _to3035():
    global _tf
    if _tf is None:
        from pyproj import Transformer
        _tf = Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)
    return _tf


def _grids(kg_code):
    with _grid_lock:
        g = _grid_cache.get(kg_code)
        if g is not None:
            _grid_cache.move_to_end(kg_code)
            return g
    import kg_v2_store as kvs
    import v21_products as v21
    out = []
    for code in kvs.codes_for_parent(kg_code) or []:
        try:
            sec = kvs.get_grid25(code)
            if sec and sec.get('terrain'):
                out.append(v21.decode_terrain_grid25(sec['terrain']))
        except Exception as e:  # noqa: BLE001
            log.debug('grid25 %s: %s', code, e)
    with _grid_lock:
        _grid_cache[kg_code] = out
        while len(_grid_cache) > _GRID_CACHE_MAX:
            _grid_cache.popitem(last=False)
    return out


def tile_bounds(z, x, y):
    n = 2 ** z
    lon0 = x / n * 360.0 - 180.0
    lon1 = (x + 1) / n * 360.0 - 180.0
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon0, lat0, lon1, lat1


def _sample(grid, E, N):
    a = grid['elev']; cell = grid['cell_m']
    rows, cols = a.shape
    fx = (E - grid['x0']) / cell - 0.5
    fy = (grid['y0'] - N) / cell - 0.5
    inside = (fx > -0.5) & (fx < cols - 0.5) & (fy > -0.5) & (fy < rows - 0.5)
    out = np.full(E.shape, np.nan, np.float32)
    if not inside.any():
        return out
    c0 = np.clip(np.floor(fx).astype(int), 0, cols - 1); c1 = np.clip(c0 + 1, 0, cols - 1)
    r0 = np.clip(np.floor(fy).astype(int), 0, rows - 1); r1 = np.clip(r0 + 1, 0, rows - 1)
    wx = np.clip(fx - c0, 0, 1); wy = np.clip(fy - r0, 0, 1)
    bil = (a[r0, c0] * (1 - wx) * (1 - wy) + a[r0, c1] * wx * (1 - wy)
           + a[r1, c0] * (1 - wx) * wy + a[r1, c1] * wx * wy)
    ci = np.clip(np.round(fx).astype(int), 0, cols - 1)
    ri = np.clip(np.round(fy).astype(int), 0, rows - 1)
    near = a[ri, ci]
    v = np.where(np.isfinite(bil), bil, near)
    out[inside] = v[inside]
    return out


def render(z, x, y, kind='hillshade'):
    """→ PNG bytes or None (no data)."""
    import search_index as si
    lon0, lat0, lon1, lat1 = tile_bounds(z, x, y)
    idx = si.get_index()
    kgs = idx.kgs_in_bbox(lon0, lat0, lon1, lat1)
    if not kgs:
        return None
    grids = [g for kg in kgs for g in _grids(kg)]
    if not grids:
        return None
    # pixel centres with a 1-px apron for gradients
    n = TILE + 2
    lons = lon0 + (np.arange(n) - 0.5) / TILE * (lon1 - lon0)
    # mercator-linear in y
    my0 = math.log(math.tan(math.pi / 4 + math.radians(lat1) / 2))
    my1 = math.log(math.tan(math.pi / 4 + math.radians(lat0) / 2))
    mys = my0 + (np.arange(n) - 0.5) / TILE * (my1 - my0)
    lats = np.degrees(2 * np.arctan(np.exp(mys)) - np.pi / 2)
    LON, LAT = np.meshgrid(lons, lats)
    E, N = _to3035().transform(LON, LAT)
    E = np.asarray(E, np.float64); N = np.asarray(N, np.float64)
    elev = np.full(E.shape, np.nan, np.float32)
    for g in grids:
        s = _sample(g, E, N)
        m = np.isnan(elev) & np.isfinite(s)
        elev[m] = s[m]
    core = elev[1:-1, 1:-1]
    if not np.isfinite(core).any():
        return None
    # metres per pixel (at tile centre)
    lat_c = math.radians((lat0 + lat1) / 2)
    mpp = 156543.03392 * math.cos(lat_c) / (2 ** z)
    filled = np.where(np.isfinite(elev), elev, np.nanmean(core))
    dzdy, dzdx = np.gradient(filled, mpp)
    if kind == 'slope':
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))[1:-1, 1:-1]
        val = np.clip(slope / 60.0, 0, 1)
        rgb = np.stack([255 - 200 * val, 255 - 120 * val, 255 - 60 * val], -1)
    else:
        az = math.radians(315.0); alt = math.radians(45.0)
        slope = np.arctan(np.hypot(dzdx, dzdy))
        aspect = np.arctan2(-dzdx, dzdy)
        hs = (math.sin(alt) * np.cos(slope)
              + math.cos(alt) * np.sin(slope) * np.cos(az - aspect))
        hs = np.clip(hs, 0, 1)[1:-1, 1:-1]
        g8 = (hs * 255)
        rgb = np.stack([g8, g8, g8], -1)
    alpha = np.where(np.isfinite(core), 255, 0).astype(np.uint8)
    img = np.dstack([rgb.astype(np.uint8), alpha])
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img, 'RGBA').save(buf, format='PNG', optimize=False, compress_level=6)
    return buf.getvalue()


def _tile_resp(z, x, y, kind):
    if not (ZMIN <= z <= ZMAX):
        return jsonify({'error': f'z must be {ZMIN}..{ZMAX}'}), 400
    if not (0 <= x < 2 ** z and 0 <= y < 2 ** z):
        return jsonify({'error': 'tile out of range'}), 400
    try:
        png = render(z, x, y, kind)
    except Exception as e:  # noqa: BLE001
        log.exception('tile %s/%d/%d/%d', kind, z, x, y)
        return jsonify({'error': str(e)}), 500
    if png is None:
        r = Response(status=204)
    else:
        r = Response(png, mimetype='image/png')
    r.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    r.headers['X-Resolution-M'] = '25'
    r.headers['Access-Control-Allow-Origin'] = '*'
    return r


@bp.route('/tiles/hillshade/<int:z>/<int:x>/<int:y>.png')
def hillshade_tile(z, x, y):
    return _tile_resp(z, x, y, 'hillshade')


@bp.route('/tiles/slope/<int:z>/<int:x>/<int:y>.png')
def slope_tile(z, x, y):
    return _tile_resp(z, x, y, 'slope')


@bp.route('/tiles/ndsm/<int:z>/<int:x>/<int:y>.png')
def ndsm_tile(z, x, y):
    return jsonify({'error': 'ndsm tiles not available',
                    'detail': 'only the 25 m DTM grid is held on the primary; 1 m nDSM '
                              'lives in the per-KG full GPKG on Zenodo (POST /api/v1/lidar/overlay '
                              'or /api/v1/kg/<code>/heightfield for on-demand rasters)'}), 501
