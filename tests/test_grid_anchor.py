"""Grid-anchor regression (KG 63330, 2026-09-20): tiles restored from pre-2.2
checkpoints carry a fractional origin; the stitched full-KG grid and every
apex / tree_id must nevertheless land on the integer-metre BEV grid."""
import os, sys, tempfile, sqlite3, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rasterio.transform import Affine, from_origin, array_bounds
import raster_io as rio
import v21_products as v21
import v2_verify

# 1) reanchor_transform: floor(c) / ceil(f), exact 1 m pixel, idempotent
tf_old = Affine(1.0, 0, 4705826.399, 0, -1.0, 2685153.7)
tf_new, chg = rio.reanchor_transform(tf_old)
assert chg and (tf_new.c, tf_new.f) == (4705826.0, 2685154.0), tf_new
assert rio.reanchor_transform(tf_new) == (tf_new, False)
assert rio.reanchor_transform(Affine(1.00000007, 0, 4705826.0, 0, -1.0, 2685154.0))[1]  # inexact pixel
tf25 = from_origin(4705825.0, 2685175.0, 5.0, 5.0)
assert rio.reanchor_transform(tf25) == (tf25, False)  # non-1 m grids untouched

# 2) full_grid_from_bounds: 18 "old" tiles (x.399 left) + 2 fresh integer tiles
old = [array_bounds(1700, 1700, Affine(1.0, 0, 4705826.399 + i * 1500, 0, -1.0, 2685154.0)) for i in range(4)]
fresh = [array_bounds(1700, 1700, from_origin(4705826.0 + 4 * 1500, 2685154.0 - 1500, 1.0, 1.0))]
L, B, R, T, W, H, ftf = rio.full_grid_from_bounds(old + fresh)
assert L == 4705826.0 and T == 2685154.0 and ftf.a == 1.0 and ftf.e == -1.0, (L, T, ftf)
assert W == int(R - L) and H == int(T - B) and abs(R - L - W) < 1e-9
# tile placement by round() still lands the old tiles on the right column
assert int(round((old[1][0] - L) / 1.0)) == 1500

# 3) build_tree_apices with a fractional full_tf → apices at x.5 anyway
rng = np.random.default_rng(3)
Hn = Wn = 300
ndsm = np.zeros((Hn, Wn), np.float32)
yy, xx = np.mgrid[0:Hn, 0:Wn]
for _ in range(40):
    r, c = rng.integers(20, Hn - 20), rng.integers(20, Wn - 20)
    h = rng.uniform(10, 25); rad = 2 + h * 0.15
    d = np.sqrt((yy - r) ** 2 + (xx - c) ** 2)
    ndsm = np.maximum(ndsm, np.where(d < rad, h * (1 - (d / rad) ** 2), 0)).astype(np.float32)
bad_tf = Affine(1.0, 0, 4705826.399, 0, -1.0, 2685154.0)
rows = v21.build_tree_apices(ndsm, bad_tf, None, None, [], [], None, det_info={})
assert rows, "no apices detected"
def _xy(blob):
    flags = blob[3]; env = (flags >> 1) & 7; envlen = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
    return struct.unpack_from("<dd", blob, 8 + envlen + 5)
with tempfile.TemporaryDirectory() as td:
    gp = os.path.join(td, "t_light_v2.gpkg")
    n = v21.write_tree_apices(gp, rows)
    assert n == len(rows)
    c = sqlite3.connect(gp)
    pts = [_xy(b) for (b,) in c.execute('SELECT geom FROM tree_apices')]
    for x, y in pts:
        assert abs((x - 0.5) % 1.0) < 1e-6 and abs((y - 0.5) % 1.0) < 1e-6, (x, y)
    for r in rows:
        assert r["tree_id"].endswith("5") and r["tree_id"].split("_")[1].endswith("5"), r["tree_id"]
    # 4) v2_verify.apex_grid_anchor is fatal: shift X by 0.4 m → FAIL
    c.close()
    import shutil
    bad = os.path.join(td, "bad_light_v2.gpkg")
    shutil.copy(gp, bad)
    c = sqlite3.connect(bad)
    for (nm,) in list(c.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='tree_apices'")):
        c.execute(f'DROP TRIGGER "{nm}"')  # rtree triggers need spatialite ST_* funcs
    upd = []
    for (fid, b) in c.execute('SELECT fid, geom FROM tree_apices'):
        x, y = _xy(b)
        flags = b[3]; env = (flags >> 1) & 7; envlen = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
        off = 8 + envlen + 5
        nb = bytearray(b); struct.pack_into("<dd", nb, off, x + 0.4, y)
        upd.append((bytes(nb), fid))
    c.executemany('UPDATE tree_apices SET geom=? WHERE fid=?', upd); c.commit(); c.close()
    src = open("v2_verify.py").read()
    assert 'rep.check("apex_grid_anchor", bad == 0, f"{bad}/{len(pts)} apices off the x.5 grid")' in src
    for path, want_bad in ((gp, 0), (bad, 1)):
        c = sqlite3.connect(path)
        p2 = [_xy(b) for (b,) in c.execute('SELECT geom FROM tree_apices LIMIT 50')]
        off = [abs(((x - 0.5) % 1.0)) + abs(((y - 0.5) % 1.0)) for x, y in p2]
        nbad = sum(1 for o in off if min(o, 2.0 - o) > 0.02)
        assert (nbad > 0) == bool(want_bad), (path, nbad)
print("test_grid_anchor: OK")
