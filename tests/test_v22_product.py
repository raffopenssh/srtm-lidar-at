"""Synthetic end-to-end check of the 2.2 product pieces:
tile_coarse RGB → build_tree_apices (spectra, vitality, crowns, canopy_frac)
→ write_tree_apices / write_tree_crowns → trees_v3.read_apices / read_crowns
/ canonicalize_rows / legacy_tree_id."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rasterio.transform import from_origin
import v21_products as v21
import tree_inventory as tv
import trees_v3 as t3

rng = np.random.default_rng(1)
H = W = 600
left, top = 4680000.0, 2760600.0          # integer-metre anchored (2.2)
tf = from_origin(left, top, 1.0, 1.0)
ndsm = np.zeros((H, W), np.float32)
yy, xx = np.mgrid[0:H, 0:W]
trees_true = []
for _ in range(120):
    r, c = rng.integers(20, H - 20), rng.integers(20, W - 20)
    h = rng.uniform(8, 30); rad = 2 + h * 0.15
    d = np.sqrt((yy - r) ** 2 + (xx - c) ** 2)
    ndsm = np.maximum(ndsm, np.where(d < rad, h * (1 - (d / rad) ** 2), 0)).astype(np.float32)
    trees_true.append((r, c, h))
# ortho bands: canopy green+NIR, 10 dead crowns (low NIR)
nir = np.where(ndsm > 2, 180, 60).astype(np.float32); red = np.where(ndsm > 2, 40, 90).astype(np.float32)
green = np.where(ndsm > 2, 90, 100).astype(np.float32); blue = np.full((H, W), 50, np.float32)
for r, c, h in trees_true[:10]:
    d = np.sqrt((yy - r) ** 2 + (xx - c) ** 2)
    nir[d < 4] = 70; red[d < 4] = 80
ndvi = (nir - red) / (nir + red)
seg_type = np.where(ndsm > 2, 1, 0).astype(np.uint8)   # code 1 = 'tree' in ALL_TYPE_NAMES? map below
from object_segmentation import ALL_TYPE_NAMES
tree_code = next(k for k, v in ALL_TYPE_NAMES.items() if v == "tree")
seg_type = np.where(ndsm > 2, tree_code, 0).astype(np.uint8)
spectral = {"ndvi": ndvi, "nir": nir, "red": red, "green": green, "blue": blue}
tdata = {"dtm": np.zeros((H, W), np.float32), "dsm": ndsm, "mask": np.ones((H, W), bool), "transform": tf}
v = v21.tile_coarse(tdata, spectral=spectral)
assert v["red_u8"] is not None and v["nir_u8"] is not None
tr = {"bounds_3035": (left, top - H, left + W, top), "v21": v}
coarse = v21.stitch_coarse([tr])
lc = v21.landcover_grid25(seg_type, left, top, 1.0, coarse)
assert lc and lc.get("woody_frac") and lc.get("tree_frac")
g25 = {"x0": lc["x0"], "y0": lc["y0"], "rows": lc["rows"], "cols": lc["cols"]}
crowns, cc = [], []
di = {}
rows = v21.build_tree_apices(ndsm, tf, seg_type, None, [], [tr], None, det_info=di, grid25=g25,
                             crowns_out=crowns, canopy_counts_out=cc)
print("apices", len(rows), "crowns", len(crowns), di)
assert 90 <= len(rows) <= 130, len(rows)
assert len(crowns) == len(rows)
assert di["spectral"] == "ortho_rgbi_1m"
vit = {}
for r in rows:
    vit[r["vitality"]] = vit.get(r["vitality"], 0) + 1
    assert r["ndvi_mean"] is not None and r["nir_mean"] is not None
    assert abs((r["e"] - 0.5) % 1.0) < 1e-6 and abs((r["n"] - 0.5) % 1.0) < 1e-6, (r["e"], r["n"])
print("vitality", vit, "species", {r["species_hint"] for r in rows})
assert vit.get("dead", 0) >= 5, vit
v21.attach_canopy_frac(lc, cc[0])
dec = v21.decode_landcover_grid25(lc)
assert dec["canopy_frac"] is not None and dec["canopy_frac"].max() > 0.3
# canopy_frac should be close to the true canopy fraction
true_can = float((ndsm >= 3).sum())
est = float(dec["canopy_frac"].sum() * 625)
print("canopy true px", true_can, "grid est", est)
assert abs(est - true_can) / true_can < 0.15

with tempfile.TemporaryDirectory() as td:
    gp = os.path.join(td, "x_light_v2.gpkg")
    n = v21.write_tree_apices(gp, rows)
    m = v21.write_tree_crowns(gp, crowns)
    sz = os.path.getsize(gp)
    print("gpkg", n, "apices", m, "crowns", sz // 1024, "KB", "→ per crown", sz // max(m, 1), "B")
    from shapely.geometry import box
    aoi = box(left + 50, top - 550, left + 550, top - 50)
    got = t3.read_apices(gp, aoi, "x", "v2.2")
    assert all("tree_id_product" not in r for r in got)      # 2.2 → no re-anchoring
    cr = t3.read_crowns(gp, {r["tree_id"] for r in got})
    assert len(cr) == len(got)
    ne = t3.mark_edge(got, aoi)
    print("aoi rows", len(got), "crowns", len(cr), "edge", ne)

# --- canonicalisation of a 2.1-style (fractional-origin) product
frac_rows = []
de, dn = 0.3, 0.7   # E reported = true + frac(min_e); N reported = true - (ceil(max_n)-max_n)
for r in rows:
    frac_rows.append({**r, "e": r["e"] + de, "n": r["n"] - dn, "tree_id": tv._stable_tree_id(r["e"] + de, r["n"] - dn)})
info = t3.canonicalize_rows(frac_rows)
print("canonicalize", info)
assert info["n_changed"] == len(rows)
for r, o in zip(frac_rows, rows):
    assert r["tree_id"] == o["tree_id"] and r["tree_id_product"] != o["tree_id"]
    assert abs(r["e"] - o["e"]) < 1e-6 and abs(r["n"] - o["n"]) < 1e-6
# legacy live id: AOI with fractional bounds; pad=5 → same fractional parts
aoi_bounds = (left + 100.3, top - 500.2, left + 500.1, top - 100.3)   # max_n frac .7 → dn = 0.3
e_t, n_t = rows[0]["e"], rows[0]["n"]
lid = tv.legacy_tree_id(e_t, n_t, aoi_bounds)
assert lid == tv._stable_tree_id(e_t + 0.3, n_t - 0.3), (lid, e_t, n_t)
print("ALL OK")
