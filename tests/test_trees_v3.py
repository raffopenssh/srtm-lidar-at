"""Fixture-driven check of /api/v3/trees* against a synthetic 2.1 product.
Run: python3 tests/test_trees_v3.py [--live]   (live = also exercise BEV fallback)"""
import json, sys, os, tempfile, shutil
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shapely.geometry import box, mapping, Point
import tile_index as ti, v21_products as v21, kg_v2_store, trees_v3, search_index as si

PARENT = '60001'   # Aflenz Kurort
CODE = '60001'
kg = next(k for k in json.load(open('data/austria_processor/kg_list.json')) if k['kg_code'] == PARENT)
bb = kg['bbox']
bbox_wgs = (bb['min_lon'], bb['min_lat'], bb['max_lon'], bb['max_lat'])
bbox_3035 = ti.geometry_to_3035(box(*bbox_wgs)).envelope
minx, miny, maxx, maxy = bbox_3035.bounds

# --- synthetic tree_apices GPKG in the cache dir -----------------------------
tmp = tempfile.mkdtemp()
cache = si.GpkgCache(cache_dir=tmp)
si._gpkg_cache = cache
rng = np.random.default_rng(1)
N = 400
es = rng.uniform(minx + 500, maxx - 500, N); ns = rng.uniform(miny + 500, maxy - 500, N)
rows = []
for i in range(N):
    h = float(rng.uniform(3, 35))
    rows.append({"tree_id": f"t_{int(es[i]*10)}_{int(ns[i]*10)}", "e": float(es[i]), "n": float(ns[i]),
                 "h_m": h, "crown_r_m": 2.0, "crown_area_m2": 12.6, "detection_source": "ndsm",
                 "detection_conf": 0.8, "surface_class": "tree", "tree_likelihood": 0.9,
                 "stand_context": "tree" if i % 4 else "orchard", "segment_type": "tree",
                 "segment_type_conf": 0.7, "dh_per_year_m": 0.3, "als_year": 2022, "ndvi": 0.6,
                 "leaf_type_hint": "coniferous", "leaf_type_conf": 0.6, "dbh_est_cm": 20.0,
                 "volume_m3_est": 0.5})
gp = cache._path(CODE, trees_v3.GPKG_VARIANT)
v21.write_tree_apices(str(gp), rows)

# --- synthetic landcover grid25 (all 'tree'=code 1 in the middle band) --------
x0, y0 = v21.snap_down(minx), v21.snap_down(maxy) + 25
cols = int((maxx - x0) // 25) + 2; rws = int((y0 - miny) // 25) + 2
cls = np.zeros((rws, cols), np.uint8); cls[rws//4: 3*rws//4, :] = 1
# realistic east edge: the KG polygon ends at the WGS bbox's max_lon (mid-lat)
_ex = ti.geometry_to_3035(Point(bb['max_lon'], (bb['min_lat'] + bb['max_lat']) / 2)).x
cls[:, (x0 + (np.arange(cols) + 0.5) * 25) > _ex - 30] = 0
frac = np.full(cls.shape, 200, np.uint8)
lc = {"cell_m": 25, "cols": cols, "rows": rws, "origin": "nw", "crs": "EPSG:3035", "x0": x0, "y0": y0,
      "coding": "u8", "cls": v21._b64gz(cls.tobytes()), "cover_frac": v21._b64gz(frac.tobytes()),
      "legend": {"1": "tree"}, "n_classified": int((cls > 0).sum())}

# --- monkeypatch product resolution -------------------------------------------
trees_v3.manifest_entries = lambda: {f"{CODE}_light_gpkg_v2": {"version": "v2.1", "bucket_url": "x", "filename": "y", "size": 0}}
kg_v2_store.codes_for_parent = lambda p: [CODE] if p == PARENT else []
kg_v2_store.get_bbox = lambda c: bbox_wgs if c == CODE else None
kg_v2_store.get_grid25 = lambda c: {"terrain": None, "landcover": lc} if c == CODE else None

import app as A
client = A.app.test_client()

# AOI: 1 km box in the middle of the KG (fully product-covered → no live)
cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
aoi = ti.geometry_from_3035(box(cx - 500, cy - 500, cx + 500, cy + 500))
body = {"type": "Feature", "properties": {}, "geometry": mapping(aoi)}
r = client.post('/api/v3/trees?fallback_live=false', json=body)
d = r.get_json(); assert r.status_code == 200, d
print('v3/trees: n=', d['summary']['n_trees'], 'canopy_ha=', d['summary']['area_ha_canopy'],
      'stems/ha canopy=', d['summary']['stems_per_ha_canopy'], 'by_stand=', d['summary']['by_stand_context'],
      'cov=', d['meta']['coverage']['product_frac'], 'live_area=', d['meta']['coverage']['live_area_ha'])
n_expected = sum(1 for x, y in zip(es, ns) if cx-500 < x < cx+500 and cy-500 < y < cy+500)
assert d['summary']['n_trees'] == n_expected, (d['summary']['n_trees'], n_expected)
assert d['summary']['area_ha_canopy'] > 0
assert d['features'][0]['properties']['source'] == 'product'
assert d['meta']['source_mix']['product'] == [CODE]
assert d['meta']['coverage']['live_area_ha'] == 0

# filter
r = client.post('/api/v3/trees?fallback_live=false&stand_context=orchard&min_tree_height=10', json=body)
d2 = r.get_json(); assert all(f['properties']['stand_context'] == 'orchard' and f['properties']['height_m'] >= 10 for f in d2['features'])
print('filter ok', d2['summary']['n_trees'])

# by-polygons
fc = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"name": "A"}, "geometry": mapping(ti.geometry_from_3035(box(cx-500, cy-500, cx, cy+500)))},
    {"type": "Feature", "properties": {"name": "B"}, "geometry": mapping(ti.geometry_from_3035(box(cx, cy-500, cx+500, cy+500)))}]}
r = client.post('/api/v3/trees/by-polygons?fallback_live=false&include_trees=true', json=fc)
d3 = r.get_json(); assert r.status_code == 200, d3
print('by-polygons:', [(s['key'], s['summary']['n_trees'], s['summary']['area_ha_canopy']) for s in d3['stands']],
      d3['summary']['n_trees_assigned'])
assert d3['summary']['n_trees_assigned'] == d['summary']['n_trees']

# AOI outside every product → live needed; with fallback off we get zero trees + live_area>0
far = ti.geometry_from_3035(box(minx - 3000, miny - 3000, minx - 2000, miny - 2000))
r = client.post('/api/v3/trees?fallback_live=false', json={"type": "Feature", "properties": {}, "geometry": mapping(far)})
d4 = r.get_json(); assert r.status_code == 200, d4
print('uncovered:', d4['summary']['n_trees'], d4['meta']['coverage'], d4['meta']['source_mix'])
assert d4['meta']['coverage']['live_area_ha'] > 0 and not d4['meta']['coverage']['live_ran']

if '--live' in sys.argv:
    # small AOI straddling the product bbox edge → product part + live part
    ex = ti.geometry_to_3035(Point(bb['max_lon'], (bb['min_lat'] + bb['max_lat']) / 2)).x
    ey = ti.geometry_to_3035(Point(bb['max_lon'], (bb['min_lat'] + bb['max_lat']) / 2)).y
    edge = ti.geometry_from_3035(box(ex - 150, ey - 100, ex + 150, ey + 100))
    r = client.post('/api/v3/trees', json={"type": "Feature", "properties": {}, "geometry": mapping(edge)})
    d5 = r.get_json(); assert r.status_code == 200, d5
    print('live mix:', d5['summary']['by_source'], d5['meta']['coverage'], d5['summary']['area_ha_canopy'])
    assert d5['meta']['coverage']['live_ran'] and 0 < d5['meta']['coverage']['product_frac'] < 1
    # changes: product apices vs live nDSM (inside the product)
    small = ti.geometry_from_3035(box(cx - 150, cy - 150, cx + 150, cy + 150))
    r = client.post('/api/v3/changes/trees', json={"type": "Feature", "properties": {}, "geometry": mapping(small)})
    d6 = r.get_json(); assert r.status_code == 200, d6
    print('changes:', d6['summary']['by_status'], d6['epoch_dates'])

shutil.rmtree(tmp)
print('ALL OK')


def test_v3_params_keep_live_rejection_threshold():
    """2.4.1 regression: the v3 row-filter default (0.0) must not disable
    the live detector's v2.3 non-forest rejection threshold."""
    import app
    import tree_inventory as tv
    eff = app._tree_v3_params({})
    assert eff['min_tree_likelihood'] == tv.MIN_TREE_LIKELIHOOD
    assert eff['filter_min_tree_likelihood'] == 0.0
    eff = app._tree_v3_params({'min_tree_likelihood': '0.5'})
    assert eff['min_tree_likelihood'] == 0.5 and eff['filter_min_tree_likelihood'] == 0.5


def test_v3_filter_rows_uses_filter_threshold():
    import app
    eff = app._tree_v3_params({})
    rows = [{'h_m': 10, 'tree_likelihood': 0.02, 'surface_class': 'building'},
            {'h_m': 10, 'tree_likelihood': None}]
    # filter threshold is 0.0 → nothing dropped by likelihood here (the
    # detector is responsible for rejection); explicit param drops the roof
    assert len(app._tree_v3_filter_rows(rows, eff)) == 2
    eff = app._tree_v3_params({'min_tree_likelihood': '0.35'})
    assert len(app._tree_v3_filter_rows(rows, eff)) == 1
