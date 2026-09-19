"""trees v3 — product-backed tree inventory (``/api/v3/trees*``).

The 2.1 product ships a ``tree_apices`` point layer in every ``_light_v2.gpkg``
(apex inventory on the stitched 1 m nDSM, ndsm_only, cadastre building mask;
see ``docs/v2.1-product-spec.md``) plus a 25 m ``landcover.grid25`` index in
the JSON (``kg_v2_store.get_grid25``).  v3 answers an AOI from those frozen
products instead of re-running the detector on live BEV rasters:

* AOI → parent KGs (``search_index`` R-tree) → product codes
  (``kg_v2_store.codes_for_parent``) whose manifest ``<code>_light_gpkg_v2``
  carries ``version == v21_products.MANIFEST_VERSION``.
* light GPKG through ``search_index.GpkgCache`` (variant ``light_v2``,
  Bearer-authenticated bucket URL — drafts 403 without it).
* apices inside the AOI (R-tree bbox filter + exact containment), deduped by
  ``tree_id`` (location-derived, so an apex owned by two overlapping product
  bboxes collapses to one row).
* canopy denominator from ``landcover.grid25`` cells whose class is a woody
  stand type (``STAND_CONTEXT_TYPES``) and whose centre lies in the AOI.
* parts of the AOI without a 2.1 product fall back to the live v2 detector
  (``app._tree_v2_inventory``); live trees within ``LIVE_DEDUPE_M`` of a
  product apex are dropped (same detector, same nDSM → same apex).

Disk rule: nothing here is persisted — GPKGs go through the 1 GB LRU cache,
grid25 comes from the small store column.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np
from shapely.geometry import GeometryCollection, Point, box
from shapely.ops import unary_union
from shapely.prepared import prep

log = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/austria_processor/zenodo_manifest.json")
GPKG_VARIANT = "light_v2"
#: live-fallback trees closer than this to a product apex are the same tree
LIVE_DEDUPE_M = 1.5
#: live fallback is skipped when the uncovered part is smaller than this
LIVE_MIN_AREA_M2 = 625.0
ALGO_VERSION = "3.0.0"

_manifest_lock = threading.Lock()
_manifest_cache: tuple[float, dict] | None = None


# --------------------------------------------------------------------------
# manifest / coverage
# --------------------------------------------------------------------------

def manifest_entries() -> dict:
    """``entries`` of the Zenodo manifest, re-read when the file changes."""
    global _manifest_cache
    try:
        mtime = MANIFEST_PATH.stat().st_mtime
    except OSError:
        return {}
    with _manifest_lock:
        if _manifest_cache and _manifest_cache[0] == mtime:
            return _manifest_cache[1]
        try:
            ent = json.loads(MANIFEST_PATH.read_text()).get("entries", {}) or {}
        except Exception as e:  # noqa: BLE001
            log.warning("trees_v3: manifest unreadable: %s", e)
            ent = {}
        _manifest_cache = (mtime, ent)
        return ent


def product_entry(code: str, entries: dict | None = None) -> dict | None:
    """The ``_light_gpkg_v2`` manifest entry iff it is a 2.1 product."""
    import v21_products
    e = (entries if entries is not None else manifest_entries()).get(f"{code}_light_gpkg_v2")
    if not e or str(e.get("version") or "") != v21_products.MANIFEST_VERSION:
        return None
    if not (e.get("bucket_url") and e.get("filename")) and not e.get("link"):
        return None
    return e


def parents_for_bbox_wgs(w: float, s: float, e: float, n: float) -> list[dict]:
    """Parent KGs whose WGS84 bbox intersects — ``[{kg_code, min_lon, …}]``."""
    import search_index as si
    c = si.get_index()._conn()
    rows = c.execute(
        """SELECT k.kg_code, k.min_lon, k.min_lat, k.max_lon, k.max_lat, k.processed
           FROM kg k JOIN kg_rtree r ON r.id = k.rowid
           WHERE r.max_lon >= ? AND r.min_lon <= ? AND r.max_lat >= ? AND r.min_lat <= ?
           ORDER BY k.kg_code""", (w, e, s, n)).fetchall()
    return [dict(r) for r in rows]


def _bbox_wgs_to_3035(bb) -> "box":
    import tile_index as ti
    return ti.geometry_to_3035(box(*bb)).envelope


_LC_CACHE: dict[str, tuple[float, dict | None]] = {}
_LC_CACHE_MAX = 64


def landcover_grid(code: str) -> dict | None:
    """Decoded ``landcover.grid25`` for a product code (small process-local
    cache keyed by the store's ``ingested_at``-independent code; None when the
    product has no grid25 — monster KGs)."""
    import kg_v2_store
    import v21_products
    hit = _LC_CACHE.get(code)
    if hit is not None:
        return hit[1]
    g = kg_v2_store.get_grid25(code)
    lc = None
    if g and g.get("landcover"):
        try:
            lc = v21_products.decode_landcover_grid25(g["landcover"])
        except Exception as e:  # noqa: BLE001
            log.warning("trees_v3: landcover grid25 %s: %s", code, e)
    if len(_LC_CACHE) >= _LC_CACHE_MAX:
        _LC_CACHE.pop(next(iter(_LC_CACHE)))
    _LC_CACHE[code] = (0.0, lc)
    return lc


def product_footprint(lc: dict | None, bbox_3035, geom_3035):
    """Where a product actually has data: the classified ``landcover.grid25``
    cells (v2 segments exist only inside the KG polygon), dilated one cell and
    hole-filled, clipped to *geom_3035*.  Falls back to the bbox when the
    product has no grid25."""
    if lc is None:
        return bbox_3035.intersection(geom_3035)
    from scipy import ndimage
    from rasterio.features import shapes
    from rasterio.transform import Affine
    from shapely.geometry import shape as shp_shape
    cell = lc["cell_m"]
    minx, miny, maxx, maxy = geom_3035.bounds
    rows, cols = lc["cls"].shape
    c0 = max(0, int((minx - lc["x0"]) // cell) - 2); c1 = min(cols, int((maxx - lc["x0"]) // cell) + 3)
    r0 = max(0, int((lc["y0"] - maxy) // cell) - 2); r1 = min(rows, int((lc["y0"] - miny) // cell) + 3)
    if c1 <= c0 or r1 <= r0:
        return GeometryCollection()
    m = lc["cls"][r0:r1, c0:c1] > 0
    if not m.any():
        return GeometryCollection()
    m = ndimage.binary_fill_holes(ndimage.binary_dilation(m, iterations=1))
    tf = Affine(cell, 0, lc["x0"] + c0 * cell, 0, -cell, lc["y0"] - r0 * cell)
    polys = [shp_shape(g) for g, v in shapes(m.astype(np.uint8), mask=m, transform=tf) if v]
    if not polys:
        return GeometryCollection()
    return unary_union(polys).buffer(0).intersection(geom_3035)


def resolve_coverage(geom_3035) -> dict:
    """Split an AOI into product-covered and live parts.

    Returns ``{parents, product: [{code, parent, entry, bbox_3035, footprint}],
    live_parents, product_geom, live_geom, product_frac}``.  A product's
    footprint is its classified grid25 cells (≈ the KG polygon; bbox when
    the product has no grid25).  ``live_geom`` = AOI minus every footprint,
    so neighbouring KGs without a 2.1 product are covered live even where a
    2.1 bbox overlaps them; ``live_parents`` lists parents intersecting the
    live part."""
    import kg_v2_store
    import tile_index as ti
    wgs = ti.geometry_from_3035(geom_3035.envelope)
    w, s, e, n = wgs.bounds
    parents = parents_for_bbox_wgs(w, s, e, n)
    entries = manifest_entries()
    product, no_product = [], []
    for p in parents:
        pbox = _bbox_wgs_to_3035((p["min_lon"], p["min_lat"], p["max_lon"], p["max_lat"]))
        got = []
        for code in kg_v2_store.codes_for_parent(p["kg_code"]) or []:
            ent = product_entry(code, entries)
            if not ent:
                continue
            bb = kg_v2_store.get_bbox(code)
            cb = _bbox_wgs_to_3035(bb) if bb else pbox
            if not cb.intersects(geom_3035):
                continue
            fp = product_footprint(landcover_grid(code), cb, geom_3035)
            if fp.is_empty:
                continue
            got.append({"code": code, "parent": p["kg_code"], "entry": ent,
                        "bbox_3035": cb, "footprint": fp})
        if got:
            product.extend(got)
        else:
            no_product.append((p["kg_code"], pbox))
    product_geom = unary_union([g["footprint"] for g in product]) if product else GeometryCollection()
    live_geom = geom_3035.difference(product_geom) if product else geom_3035
    if live_geom.area < LIVE_MIN_AREA_M2:
        live_geom = GeometryCollection()
    live_parents = [c for c, pb in no_product if not live_geom.is_empty and pb.intersects(live_geom)]
    frac = product_geom.area / geom_3035.area if geom_3035.area > 0 else 0.0
    return {"parents": [p["kg_code"] for p in parents], "product": product,
            "live_parents": live_parents, "product_geom": product_geom,
            "live_geom": live_geom, "product_frac": round(float(min(frac, 1.0)), 4)}


# --------------------------------------------------------------------------
# light GPKG + apices
# --------------------------------------------------------------------------

def gpkg_path(code: str, entry: dict) -> str | None:
    """Local path to ``<code>_light_v2.gpkg`` (processor output, cache, or
    Bearer-authenticated Zenodo download)."""
    import search_index as si
    local = Path(f"data/austria_processor/gpkg/{code}_light_v2.gpkg")
    if local.exists() and local.stat().st_size > 0:
        return str(local)
    cache = si.get_gpkg_cache()
    p = cache.get(code, GPKG_VARIANT)
    if p:
        return p
    from v2_ingest import _link
    url, hdr = _link(entry)
    if not url:
        return None
    return cache.download(code, GPKG_VARIANT, url, headers=hdr,
                          expected_size=int(entry.get("size") or 0))


def read_apices(path: str, geom_3035, code: str) -> list[dict]:
    """``tree_apices`` rows inside *geom_3035* as flat dicts (+ ``e``, ``n``)."""
    import fiona
    import v21_products
    out = []
    pg = prep(geom_3035)
    with fiona.open(path, layer=v21_products.TREE_APICES_LAYER) as src:
        for f in src.filter(bbox=geom_3035.bounds):
            x, y = f["geometry"]["coordinates"][:2]
            if not pg.contains(Point(x, y)):
                continue
            r = dict(f["properties"])
            r["e"], r["n"] = float(x), float(y)
            r["product_code"] = code
            out.append(r)
    return out


def collect_product_trees(cov: dict, geom_3035, progress=None) -> tuple[list[dict], list[str], list[str]]:
    """All product apices in the AOI, deduped by ``tree_id``.
    Returns (rows, codes_used, codes_failed)."""
    rows: dict[str, dict] = {}
    used, failed = [], []
    for i, prod in enumerate(cov["product"]):
        if progress:
            progress(f"Reading tree_apices {prod['code']} ({i + 1}/{len(cov['product'])})…")
        try:
            p = gpkg_path(prod["code"], prod["entry"])
            if not p:
                raise RuntimeError("light GPKG unavailable")
            for r in read_apices(p, prod["footprint"], prod["code"]):
                rows.setdefault(r["tree_id"], r)
            used.append(prod["code"])
        except Exception as e:  # noqa: BLE001
            log.warning("trees_v3: %s apices unavailable: %s", prod["code"], e)
            failed.append(prod["code"])
    return list(rows.values()), used, failed


# --------------------------------------------------------------------------
# grid25 canopy denominator
# --------------------------------------------------------------------------

class CanopyGrid:
    """Cell centres of the product ``landcover.grid25`` sections intersecting
    an AOI, with per-cell stand class — the canopy denominator provider."""

    def __init__(self, codes: list[str], geom_3035):
        import v21_products
        self.cell_m = float(v21_products.GRID_M)
        es, ns, cls, frac = [], [], [], []
        self.codes = []
        self.legend: dict[int, str] = {}
        minx, miny, maxx, maxy = geom_3035.bounds
        for code in codes:
            lc = landcover_grid(code)
            if lc is None:
                continue
            self.codes.append(code)
            self.legend.update({int(k): v for k, v in lc["legend"].items()})
            cell = lc["cell_m"]
            rows, cols = lc["cls"].shape
            c0 = max(0, int((minx - lc["x0"]) // cell) - 1)
            c1 = min(cols, int((maxx - lc["x0"]) // cell) + 2)
            r0 = max(0, int((lc["y0"] - maxy) // cell) - 1)
            r1 = min(rows, int((lc["y0"] - miny) // cell) + 2)
            if c1 <= c0 or r1 <= r0:
                continue
            rr, cc = np.mgrid[r0:r1, c0:c1]
            es.append(lc["x0"] + (cc.ravel() + 0.5) * cell)
            ns.append(lc["y0"] - (rr.ravel() + 0.5) * cell)
            cls.append(lc["cls"][r0:r1, c0:c1].ravel())
            frac.append(lc["cover_frac"][r0:r1, c0:c1].ravel())
        if es:
            self.e = np.concatenate(es); self.n = np.concatenate(ns)
            self.cls = np.concatenate(cls); self.frac = np.concatenate(frac)
            # cells from overlapping product grids: keep the first classified one
            key = np.round(self.e / self.cell_m).astype(np.int64) * 10_000_000 + \
                np.round(self.n / self.cell_m).astype(np.int64)
            order = np.lexsort((self.cls == 0, key))
            _, first = np.unique(key[order], return_index=True)
            keep = order[first]
            self.e, self.n, self.cls, self.frac = self.e[keep], self.n[keep], self.cls[keep], self.frac[keep]
        else:
            self.e = self.n = np.zeros(0); self.cls = np.zeros(0, np.uint8); self.frac = np.zeros(0, np.float32)
        self.stand_codes = {c for c, name in self.legend.items() if name in v21_products.STAND_CONTEXT_TYPES}
        self.tree_code = next((c for c, name in self.legend.items() if name == "tree"), None)

    @property
    def available(self) -> bool:
        return self.e.size > 0

    def canopy(self, geom_3035) -> dict:
        """Stand-cell areas (m²) whose centre lies in *geom_3035*."""
        if not self.available:
            return {"n_cells": 0, "stand_m2": 0.0, "stand_m2_weighted": 0.0, "forest_m2": 0.0}
        from shapely import contains_xy
        minx, miny, maxx, maxy = geom_3035.bounds
        pre = (self.e >= minx) & (self.e <= maxx) & (self.n >= miny) & (self.n <= maxy)
        if not pre.any():
            return {"n_cells": 0, "stand_m2": 0.0, "stand_m2_weighted": 0.0, "forest_m2": 0.0}
        idx = np.nonzero(pre)[0]
        inside = contains_xy(geom_3035, self.e[idx], self.n[idx])
        idx = idx[inside]
        a = self.cell_m ** 2
        cls = self.cls[idx]
        stand = np.isin(cls, list(self.stand_codes)) if self.stand_codes else np.zeros(cls.shape, bool)
        forest = (cls == self.tree_code) if self.tree_code is not None else np.zeros(cls.shape, bool)
        return {"n_cells": int(idx.size), "stand_m2": float(stand.sum() * a),
                "stand_m2_weighted": float((self.frac[idx][stand]).sum() * a),
                "forest_m2": float(forest.sum() * a)}


# --------------------------------------------------------------------------
# features / summary
# --------------------------------------------------------------------------

def product_row_to_feature(r: dict) -> dict:
    """``_tree_v2_feature``-shaped GeoJSON Feature (WGS84 apex point)."""
    import tile_index as ti
    wgs = ti.geometry_from_3035(Point(r["e"], r["n"]))
    props = {
        "tree_id": r.get("tree_id"),
        "apex_lon": round(wgs.x, 7), "apex_lat": round(wgs.y, 7),
        "height_m": r.get("h_m"),
        "crown_area_sqm": r.get("crown_area_m2"),
        "crown_radius_mean_m": r.get("crown_r_m"),
        "dbh_est_cm": r.get("dbh_est_cm"), "dbh_method": "heuristic_h_crown",
        "volume_m3_est": r.get("volume_m3_est"),
        "leaf_type": r.get("leaf_type_hint") or "unknown",
        "leaf_type_conf": r.get("leaf_type_conf"),
        "detection_source": r.get("detection_source"),
        "detection_conf": r.get("detection_conf"),
        "surface_class": r.get("surface_class"),
        "tree_likelihood": r.get("tree_likelihood"),
        "stand_context": r.get("stand_context") or "none",
        "segment_type": r.get("segment_type"),
        "segment_type_conf": r.get("segment_type_conf"),
        "dh_per_year_m": r.get("dh_per_year_m"),
        "als_year": r.get("als_year"),
        "ndvi": r.get("ndvi"),
        "source": "product", "product_code": r.get("product_code"),
    }
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "Point", "coordinates": [props["apex_lon"], props["apex_lat"]]}}


def live_tree_to_row(t) -> dict:
    """A ``tree_inventory.Tree`` from the live fallback as a v3 row."""
    return {
        "tree_id": t.tree_id, "e": t.apex_e, "n": t.apex_n, "h_m": t.height_m,
        "crown_r_m": t.crown_radius_mean_m, "crown_area_m2": t.crown_area_sqm,
        "detection_source": t.detection_source, "detection_conf": t.detection_conf,
        "surface_class": t.surface_class, "tree_likelihood": t.tree_likelihood,
        "stand_context": "unknown", "segment_type": None, "segment_type_conf": None,
        "dh_per_year_m": None, "als_year": None, "ndvi": None,
        "leaf_type_hint": t.leaf_type, "leaf_type_conf": t.leaf_type_conf,
        "dbh_est_cm": t.dbh_est_cm, "volume_m3_est": t.volume_m3_est,
        "product_code": None, "_source": "live", "_is_edge": t.is_edge,
    }


def row_to_feature(r: dict) -> dict:
    f = product_row_to_feature(r)
    if r.get("_source") == "live":
        f["properties"]["source"] = "live"
        f["properties"]["is_edge"] = bool(r.get("_is_edge"))
    return f


def dedupe_live(product_rows: list[dict], live_rows: list[dict],
                radius_m: float = LIVE_DEDUPE_M) -> tuple[list[dict], int]:
    """Drop live trees that duplicate a product apex.  Returns (kept, n_dropped)."""
    if not product_rows or not live_rows:
        return live_rows, 0
    from scipy.spatial import cKDTree
    kd = cKDTree(np.array([[r["e"], r["n"]] for r in product_rows]))
    pts = np.array([[r["e"], r["n"]] for r in live_rows])
    d, _ = kd.query(pts, k=1, distance_upper_bound=radius_m)
    keep = [r for r, dd in zip(live_rows, d) if not np.isfinite(dd)]
    return keep, len(live_rows) - len(keep)


def summarise(rows: list[dict], aoi_area_m2: float, canopy: dict | None,
              live_canopy_m2: float = 0.0, live_area_m2: float = 0.0) -> dict:
    """Explicit-denominator summary over v3 rows (product + live)."""
    hs = np.array([r["h_m"] for r in rows if r.get("h_m") is not None], np.float32)
    n = len(rows)
    n_live = sum(1 for r in rows if r.get("_source") == "live")
    stems = [r for r in rows if not r.get("_is_edge")]
    area_ha = aoi_area_m2 / 1e4
    canopy = canopy or {}
    canopy_m2 = float(canopy.get("stand_m2", 0.0)) + float(live_canopy_m2)
    canopy_ha = canopy_m2 / 1e4
    forest_ha = float(canopy.get("forest_m2", 0.0)) / 1e4
    forest_stems = [r for r in stems if r.get("stand_context") == "tree"]

    def _pct(p):
        return round(float(np.percentile(hs, p)), 2) if hs.size else 0.0

    n_dom = max(1, int(round(100 * max(canopy_ha, 0.01))))
    hs_sorted = np.sort(hs)[::-1]
    h_dom = float(np.mean(hs_sorted[:min(n_dom, hs_sorted.size)])) if hs.size else 0.0
    h_top100 = 0.0
    if rows:
        cell = {}
        for r in rows:
            if r.get("h_m") is None:
                continue
            cell.setdefault((int(r["e"] // 100), int(r["n"] // 100)), []).append(r["h_m"])
        tops = [max(v) for v in cell.values()]
        h_top100 = float(np.mean(tops)) if tops else 0.0
    hist = {}
    if hs.size:
        top = int(np.ceil(float(hs.max()) / 2.0) * 2)
        edges = np.arange(0, top + 2, 2)
        cnt, _ = np.histogram(hs, bins=edges)
        hist = {f"{int(edges[i])}-{int(edges[i + 1])}": int(c) for i, c in enumerate(cnt) if c}

    def _by(key, default="unknown"):
        out: dict[str, int] = {}
        for r in rows:
            k = r.get(key) or default
            out[str(k)] = out.get(str(k), 0) + 1
        return out

    dh = np.array([r["dh_per_year_m"] for r in rows if r.get("dh_per_year_m") is not None], np.float32)
    vol = float(sum((r.get("volume_m3_est") or 0.0) for r in rows))
    return {
        "n_trees": n, "n_trees_product": n - n_live, "n_trees_live": n_live,
        "n_trees_edge": n - len(stems),
        "area_ha_total": round(area_ha, 3),
        "area_ha_canopy": round(canopy_ha, 3),
        "area_ha_canopy_cover_weighted": round((float(canopy.get("stand_m2_weighted", 0.0)) + live_canopy_m2) / 1e4, 3),
        "area_ha_forest": round(forest_ha, 3),
        "area_ha_live_part": round(live_area_m2 / 1e4, 3),
        "canopy_denominator_note": (
            "product part: landcover.grid25 25 m cells (dominant v2 segment class in "
            "tree/orchard/vineyard/hedge/garden/shrub) whose centre lies in the AOI, "
            "x 625 m2; area_ha_forest = class 'tree' only; live part: nDSM >= "
            "min_tree_height canopy pixels"),
        "stems_per_ha_total": round(len(stems) / area_ha, 1) if area_ha > 0 else 0,
        "stems_per_ha_canopy": round(len(stems) / canopy_ha, 1) if canopy_ha > 0 else 0,
        "stems_per_ha_forest": round(len(forest_stems) / forest_ha, 1) if forest_ha > 0 else 0,
        "stems_per_ha_note": "non-edge trees only (product apices are never edge)",
        "h_mean_m": round(float(hs.mean()), 2) if hs.size else 0.0,
        "h_p50_m": _pct(50), "h_p90_m": _pct(90), "h_p95_m": _pct(95), "h_p99_m": _pct(99),
        "h_max_m": round(float(hs.max()), 2) if hs.size else 0.0,
        "h_dom_m": round(h_dom, 2), "h_dom_basis": "canopy",
        "h_top100_m": round(h_top100, 2),
        "height_histogram_2m": hist,
        "by_leaf_type": _by("leaf_type_hint"),
        "by_stand_context": _by("stand_context", "none"),
        "by_surface_class": _by("surface_class"),
        "by_detection": _by("detection_source"),
        "by_source": {"product": n - n_live, "live": n_live},
        "volume_m3_est_total": round(vol, 1),
        "volume_m3_est_per_ha_canopy": round(vol / canopy_ha, 1) if canopy_ha > 0 else None,
        "dh_per_year_m_median": round(float(np.median(dh)), 3) if dh.size else None,
        "n_with_dh": int(dh.size),
        "grid25_cells_in_aoi": int(canopy.get("n_cells", 0)),
    }


# --------------------------------------------------------------------------
# changes (epoch a = product) helpers
# --------------------------------------------------------------------------

def rows_to_trees(rows: list[dict], transform, shape) -> list:
    """Product rows → ``tree_inventory.Tree`` objects on a live raster grid
    (row/col clamped; ``label`` 0 — no crown mask, so crown-overlap matching
    is skipped)."""
    import tree_inventory as tv
    out = []
    for i, r in enumerate(rows):
        col, row = ~transform * (r["e"], r["n"])
        rr = int(min(max(row, 0), shape[0] - 1)); cc = int(min(max(col, 0), shape[1] - 1))
        out.append(tv.Tree(
            tree_id=r["tree_id"], seq=i, label=0, apex_e=float(r["e"]), apex_n=float(r["n"]),
            apex_row=rr, apex_col=cc, height_m=float(r.get("h_m") or 0.0),
            crown_area_sqm=float(r.get("crown_area_m2") or 0.0),
            crown_radius_mean_m=float(r.get("crown_r_m") or 0.0),
            crown_radius_max_m=float(r.get("crown_r_m") or 0.0), is_edge=False,
            dbh_est_cm=float(r.get("dbh_est_cm") or 0.0),
            volume_m3_est=float(r.get("volume_m3_est") or 0.0),
            leaf_type=r.get("leaf_type_hint") or "unknown",
            leaf_type_conf=float(r.get("leaf_type_conf") or 0.0),
            detection_source=r.get("detection_source") or "ndsm",
            detection_conf=float(r.get("detection_conf") or 0.0),
            surface_class=r.get("surface_class") or "tree",
            tree_likelihood=float(r.get("tree_likelihood") or 1.0)))
    return out


def crown_proxy_ndsm(rows: list[dict], transform, shape) -> np.ndarray:
    """Epoch-a nDSM proxy from product apices: every crown disc (radius
    ``crown_r_m``, ≥1 px) painted with ``h_m`` (max where discs overlap).
    Used ONLY for the 3 m-radius apex evidence in ``match_trees`` — the
    product carries no 1 m raster (disk rule)."""
    z = np.zeros(shape, np.float32)
    res = abs(transform.a)
    for r in rows:
        col, row = ~transform * (r["e"], r["n"])
        rad = max(1, int(round(float(r.get("crown_r_m") or 1.0) / res)))
        r0, r1 = int(row) - rad, int(row) + rad + 1
        c0, c1 = int(col) - rad, int(col) + rad + 1
        rr0, rr1 = max(r0, 0), min(r1, shape[0]); cc0, cc1 = max(c0, 0), min(c1, shape[1])
        if rr1 <= rr0 or cc1 <= cc0:
            continue
        yy, xx = np.mgrid[rr0:rr1, cc0:cc1]
        disc = (yy - row) ** 2 + (xx - col) ** 2 <= rad * rad
        h = float(r.get("h_m") or 0.0)
        sub = z[rr0:rr1, cc0:cc1]
        sub[disc] = np.maximum(sub[disc], h)
    return z


def product_als_year(rows: list[dict]) -> int | None:
    ys = [r["als_year"] for r in rows if r.get("als_year")]
    return int(np.median(ys)) if ys else None
