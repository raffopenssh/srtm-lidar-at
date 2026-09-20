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
ALGO_VERSION = "3.1.0"

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
    """The ``_light_gpkg_v2`` manifest entry iff it is a readable (≥2.1) product."""
    import v21_products
    e = (entries if entries is not None else manifest_entries()).get(f"{code}_light_gpkg_v2")
    if not e or str(e.get("version") or "") not in v21_products.READABLE_MANIFEST_VERSIONS:
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


def read_apices(path: str, geom_3035, code: str, version: str | None = None) -> list[dict]:
    """``tree_apices`` rows inside *geom_3035* as flat dicts (+ ``e``, ``n``),
    re-anchored to the BEV grid (:func:`canonicalize_rows`)."""
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
            r["product_version"] = version
            out.append(r)
    canonicalize_rows(out)
    return out


def canonicalize_rows(rows: list[dict]) -> dict:
    """Re-anchor product apices to the integer-metre BEV grid (FEEDBACK-5 §1).

    2.1 products were stitched on a fractional-origin grid: the pixel data
    are true BEV pixels but every reported centre is shifted by a constant
    (δe, δn) per product, |δ| < 1 m.  δ is recovered as the median fractional
    offset of the apices from the ``x.5`` centres, the coordinates are moved
    back onto the grid and ``tree_id`` is re-derived — so a 2.1 product, a
    2.2 product and a live call all name the same pixel the same way.  The
    original product id is kept as ``tree_id_product`` when it differs
    (clients that stored 3.0 ids can migrate).  Idempotent for 2.2 products
    (δ = 0).  Returns {de, dn, n_changed}.
    """
    if not rows:
        return {"de": 0.0, "dn": 0.0, "n_changed": 0}
    import tree_inventory as tv
    es = np.array([r["e"] for r in rows]); ns = np.array([r["n"] for r in rows])
    # Sign convention is FIXED by how rasterio floored the window: E was
    # reported too far east by frac(origin_e) ∈ [0,1), N too far south by
    # ceil(origin_n) - origin_n ∈ [0,1).  Positions alone cannot tell a
    # +0.3 from a -0.7 shift, so we do not guess — we apply the known sign.
    fe = np.mod(es - 0.5, 1.0)             # 0 when aligned
    fn = np.mod(0.5 - ns, 1.0)
    de, dn = float(np.median(fe)), float(np.median(fn))
    if de > 0.98:
        de = 0.0
    if dn > 0.98:
        dn = 0.0
    if de < 0.02 and dn < 0.02:
        return {"de": 0.0, "dn": 0.0, "n_changed": 0}
    n_changed = 0
    for r, e, n in zip(rows, es, ns):
        e_t = np.floor(e - de) + 0.5
        n_t = np.floor(n + dn) + 0.5
        new_id = tv._stable_tree_id(float(e_t), float(n_t))
        if new_id != r.get("tree_id"):
            r["tree_id_product"] = r.get("tree_id")
            r["tree_id"] = new_id
            n_changed += 1
        r["e"], r["n"] = float(e_t), float(n_t)
    return {"de": round(de, 3), "dn": round(dn, 3), "n_changed": n_changed}


def read_crowns(path: str, tree_ids: set[str]) -> dict:
    """{tree_id: shapely Polygon EPSG:3035} from the 2.2 ``tree_crowns``
    layer — {} when the product has none (2.1)."""
    import fiona
    import v21_products
    from shapely.geometry import shape as _shape
    out: dict = {}
    if not tree_ids:
        return out
    try:
        layers = fiona.listlayers(path)
    except Exception:  # noqa: BLE001
        return out
    if v21_products.TREE_CROWNS_LAYER not in layers:
        return out
    with fiona.open(path, layer=v21_products.TREE_CROWNS_LAYER) as src:
        for f in src:
            tid = (f["properties"] or {}).get("tree_id")
            if tid in tree_ids:
                out[tid] = _shape(f["geometry"])
    return out


def mark_edge(rows: list[dict], geom_3035) -> int:
    """Set ``_is_edge`` per row against the SUBMITTED AOI: a crown whose
    mean-radius disc crosses the AOI boundary is an edge tree, exactly as the
    live v2 engine reports it (FEEDBACK-5 §5).  Returns the edge count."""
    if not rows:
        return 0
    import shapely
    pts = shapely.points(np.array([[r["e"], r["n"]] for r in rows]))
    d = shapely.distance(geom_3035.boundary, pts)
    n = 0
    for r, dd in zip(rows, d):
        edge = bool(dd < float(r.get("crown_r_m") or 0.0))
        r["_is_edge"] = edge or bool(r.get("_is_edge"))
        n += int(r["_is_edge"])
    return n


def rank_vitality_in_aoi(rows: list[dict]) -> None:
    """Re-rank ``ndvi_percentile_in_aoi`` (and the relative stressed/vital
    verdict) over the AOI population per leaf class — the product ranked
    KG-wide.  ``dead`` (absolute) is untouched."""
    for cls in ("coniferous", "broadleaf", "unknown"):
        sel = [r for r in rows if r.get("vitality") not in (None, "dead", "unknown")
               and r.get("ndvi_mean") is not None
               and ((r.get("leaf_type_hint") == cls) or
                    (cls == "unknown" and r.get("leaf_type_hint") not in ("coniferous", "broadleaf", "dead")))]
        if len(sel) < 5:
            continue
        vals = np.array([r["ndvi_mean"] for r in sel], np.float32)
        order = np.argsort(np.argsort(vals))
        pct = (order + 0.5) / len(vals) * 100.0
        p10 = float(np.percentile(vals, 10))
        p50 = float(np.median(vals))
        for r, v, pc in zip(sel, vals, pct):
            r["ndvi_percentile_in_aoi"] = round(float(pc), 1)
            if v <= p10 and v < p50:
                r["vitality"] = "stressed"
                r["vitality_conf"] = round(min(0.8, 0.4 + (p10 - float(v)) * 3), 2)
            else:
                r["vitality"] = "vital"
                r["vitality_conf"] = round(min(0.9, float(pc) / 100.0 + 0.3), 2)


def collect_product_trees(cov: dict, geom_3035, progress=None, *, failed_reasons: dict | None = None,
                          product_versions: dict | None = None,
                          gpkg_paths: dict | None = None) -> tuple[list[dict], list[str], list[str]]:
    """All product apices in the AOI, deduped by ``tree_id``.
    Returns (rows, codes_used, codes_failed); optional dicts receive
    per-code failure reasons, manifest versions and local GPKG paths."""
    rows: dict[str, dict] = {}
    used, failed = [], []
    reasons: dict[str, str] = {}
    versions: dict[str, str] = {}
    paths: dict[str, str] = {}
    for i, prod in enumerate(cov["product"]):
        if progress:
            progress(f"Reading tree_apices {prod['code']} ({i + 1}/{len(cov['product'])})…")
        try:
            p = gpkg_path(prod["code"], prod["entry"])
            if not p:
                raise RuntimeError("light GPKG unavailable (not local, not cached, download failed)")
            ver = str(prod["entry"].get("version") or "")
            for r in read_apices(p, prod["footprint"], prod["code"], ver):
                rows.setdefault(r["tree_id"], r)
            used.append(prod["code"])
            versions[prod["code"]] = ver
            paths[prod["code"]] = p
        except Exception as e:  # noqa: BLE001
            log.warning("trees_v3: %s apices unavailable: %s", prod["code"], e)
            failed.append(prod["code"])
            reasons[prod["code"]] = f"{type(e).__name__}: {str(e)[:160]}"
    if failed_reasons is not None:
        failed_reasons.update(reasons)
    if product_versions is not None:
        product_versions.update(versions)
    if gpkg_paths is not None:
        gpkg_paths.update(paths)
    return list(rows.values()), used, failed


# --------------------------------------------------------------------------
# grid25 canopy denominator
# --------------------------------------------------------------------------

class CanopyGrid:
    """Cells of the product ``landcover.grid25`` sections intersecting an
    AOI, with per-cell stand class and (2.2) class / canopy fractions — the
    canopy denominator provider.

    ``canopy(geom)`` is AREA-WEIGHTED when the product carries fractions:
    Σ (cell ∩ AOI area) × frac, i.e. the v2 1 m pixel canopy aggregated onto
    25 m cells, with no centre-in-AOI rim loss (FEEDBACK-5 §6).  2.1 grids
    (no fractions) fall back to the centre + dominant-class rule and say so
    in ``method``.
    """

    def __init__(self, codes: list[str], geom_3035):
        import v21_products
        self.cell_m = float(v21_products.GRID_M)
        es, ns, cls, frac = [], [], [], []
        woody, treef, canf = [], [], []
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
            n_cells = rr.size
            es.append(lc["x0"] + (cc.ravel() + 0.5) * cell)
            ns.append(lc["y0"] - (rr.ravel() + 0.5) * cell)
            cls.append(lc["cls"][r0:r1, c0:c1].ravel())
            frac.append(lc["cover_frac"][r0:r1, c0:c1].ravel())
            for src, dst in ((lc.get("woody_frac"), woody), (lc.get("tree_frac"), treef),
                             (lc.get("canopy_frac"), canf)):
                dst.append(src[r0:r1, c0:c1].ravel() if src is not None
                           else np.full(n_cells, np.nan, np.float32))
        if es:
            self.e = np.concatenate(es); self.n = np.concatenate(ns)
            self.cls = np.concatenate(cls); self.frac = np.concatenate(frac)
            self.woody = np.concatenate(woody); self.treef = np.concatenate(treef)
            self.canf = np.concatenate(canf)
            # cells from overlapping product grids: keep the first classified one
            key = np.round(self.e / self.cell_m).astype(np.int64) * 10_000_000 + \
                np.round(self.n / self.cell_m).astype(np.int64)
            order = np.lexsort((self.cls == 0, key))
            _, first = np.unique(key[order], return_index=True)
            keep = order[first]
            for attr in ("e", "n", "cls", "frac", "woody", "treef", "canf"):
                setattr(self, attr, getattr(self, attr)[keep])
        else:
            self.e = self.n = np.zeros(0); self.cls = np.zeros(0, np.uint8); self.frac = np.zeros(0, np.float32)
            self.woody = self.treef = self.canf = np.zeros(0, np.float32)
        self.stand_codes = {c for c, name in self.legend.items() if name in v21_products.STAND_CONTEXT_TYPES}
        self.tree_code = next((c for c, name in self.legend.items() if name == "tree"), None)
        self.has_fractions = bool(self.e.size) and bool(np.isfinite(self.woody).all())
        self.has_canopy_frac = bool(self.e.size) and bool(np.isfinite(self.canf).all())

    @property
    def available(self) -> bool:
        return self.e.size > 0

    _EMPTY = {"n_cells": 0, "stand_m2": 0.0, "stand_m2_weighted": 0.0, "forest_m2": 0.0,
              "canopy_m2": None, "method": "none"}

    def canopy(self, geom_3035) -> dict:
        """Stand / forest / canopy areas (m²) inside *geom_3035*."""
        if not self.available:
            return dict(self._EMPTY)
        import shapely
        from shapely import contains_xy
        minx, miny, maxx, maxy = geom_3035.bounds
        h = self.cell_m / 2.0
        pre = (self.e >= minx - h) & (self.e <= maxx + h) & (self.n >= miny - h) & (self.n <= maxy + h)
        if not pre.any():
            return dict(self._EMPTY)
        idx = np.nonzero(pre)[0]
        a = self.cell_m ** 2
        if self.has_fractions:
            # area-weighted: (cell ∩ AOI) × class fraction
            boxes = shapely.box(self.e[idx] - h, self.n[idx] - h, self.e[idx] + h, self.n[idx] + h)
            inter = shapely.area(shapely.intersection(boxes, geom_3035))
            hit = inter > 0
            idx, inter = idx[hit], inter[hit]
            stand_w = float((inter * self.woody[idx]).sum())
            forest_w = float((inter * self.treef[idx]).sum())
            canopy_w = float((inter * self.canf[idx]).sum()) if self.has_canopy_frac else None
            return {"n_cells": int(idx.size), "stand_m2": stand_w, "stand_m2_weighted": stand_w,
                    "forest_m2": forest_w, "canopy_m2": canopy_w,
                    "method": "area_weighted_fractions" + ("+canopy_frac" if canopy_w is not None else "")}
        inside = contains_xy(geom_3035, self.e[idx], self.n[idx])
        idx = idx[inside]
        cls = self.cls[idx]
        stand = np.isin(cls, list(self.stand_codes)) if self.stand_codes else np.zeros(cls.shape, bool)
        forest = (cls == self.tree_code) if self.tree_code is not None else np.zeros(cls.shape, bool)
        return {"n_cells": int(idx.size), "stand_m2": float(stand.sum() * a),
                "stand_m2_weighted": float((self.frac[idx][stand]).sum() * a),
                "forest_m2": float(forest.sum() * a), "canopy_m2": None,
                "method": "centre_dominant_class (2.1 product)"}


# --------------------------------------------------------------------------
# features / summary
# --------------------------------------------------------------------------

def product_row_to_feature(r: dict, crown=None) -> dict:
    """``_tree_v2_feature``-shaped GeoJSON Feature (WGS84 apex point, or the
    crown polygon when *crown* (EPSG:3035) is given)."""
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
        "is_edge": bool(r.get("_is_edge")),
        # 2.2 per-crown BEV ortho RGBI spectra (None on 2.1 products / live ndsm-only)
        "vitality": r.get("vitality") or "unknown", "vitality_conf": r.get("vitality_conf"),
        "species_hint": r.get("species_hint") or "unknown", "species_conf": r.get("species_conf"),
        "ndvi_mean": r.get("ndvi_mean", r.get("ndvi")), "ndvi_p10": r.get("ndvi_p10"),
        "ndvi_percentile_in_aoi": r.get("ndvi_percentile_in_aoi"),
        "nir_mean": r.get("nir_mean"), "brightness_mean": r.get("brightness_mean"),
        "green_ratio_mean": r.get("green_ratio_mean"),
        "source": r.get("_source") or "product", "product_code": r.get("product_code"),
        "product_version": r.get("product_version"),
    }
    if r.get("tree_id_product"):
        props["tree_id_product"] = r["tree_id_product"]
    if r.get("tree_id_v2"):
        props["tree_id_v2"] = r["tree_id_v2"]
    geom = {"type": "Point", "coordinates": [props["apex_lon"], props["apex_lat"]]}
    if crown is not None:
        from shapely.geometry import mapping
        geom = mapping(ti.geometry_from_3035(crown))
        props["crown_geometry"] = "polygon"
    return {"type": "Feature", "properties": props, "geometry": geom}


def live_tree_to_row(t, source: str = "live") -> dict:
    """A ``tree_inventory.Tree`` from a live run as a v3 row (spectra carried
    when the live run had ortho)."""
    sp = t.spectral or {}
    return {
        "tree_id": t.tree_id, "e": t.apex_e, "n": t.apex_n, "h_m": t.height_m,
        "crown_r_m": t.crown_radius_mean_m, "crown_area_m2": t.crown_area_sqm,
        "detection_source": t.detection_source, "detection_conf": t.detection_conf,
        "surface_class": t.surface_class, "tree_likelihood": t.tree_likelihood,
        "stand_context": "unknown", "segment_type": None, "segment_type_conf": None,
        "dh_per_year_m": None, "als_year": None,
        "leaf_type_hint": t.leaf_type, "leaf_type_conf": t.leaf_type_conf,
        "vitality": t.vitality, "vitality_conf": t.vitality_conf or None,
        "species_hint": t.species_hint, "species_conf": t.species_conf or None,
        "ndvi_mean": sp.get("ndvi_mean"), "ndvi_p10": sp.get("ndvi_p10"),
        "ndvi_percentile_in_aoi": sp.get("ndvi_percentile_in_aoi"),
        "nir_mean": sp.get("nir_mean"), "brightness_mean": sp.get("brightness_mean"),
        "green_ratio_mean": sp.get("green_ratio_mean"),
        "dbh_est_cm": t.dbh_est_cm, "volume_m3_est": t.volume_m3_est,
        "product_code": None, "_source": source, "_is_edge": t.is_edge, "_label": t.label,
    }


def row_to_feature(r: dict, crown=None) -> dict:
    return product_row_to_feature(r, crown)


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
    n_live = sum(1 for r in rows if (r.get("_source") or "product") != "product")
    stems = [r for r in rows if not r.get("_is_edge")]
    area_ha = aoi_area_m2 / 1e4
    canopy = canopy or {}
    method = canopy.get("method", "none")
    if canopy.get("canopy_m2") is not None:
        # 2.2: the apex-inventory canopy mask (nDSM ≥ min_tree_height inside
        # a v2 segment, roofs/crop rejected), area-weighted per 25 m cell —
        # the same quantity the live part measures on 1 m pixels.
        canopy_m2 = float(canopy["canopy_m2"]) + float(live_canopy_m2)
        canopy_note = ("product part: landcover.grid25.canopy_frac (1 m apex-inventory canopy: "
                       "nDSM >= min_tree_height inside a v2 segment, roofs/crop rejected) x "
                       "(cell ∩ AOI) area; area_ha_forest = tree_frac x (cell ∩ AOI); "
                       "live part: nDSM >= min_tree_height canopy pixels")
    elif method.startswith("area_weighted"):
        canopy_m2 = float(canopy.get("stand_m2", 0.0)) + float(live_canopy_m2)
        canopy_note = ("product part: landcover.grid25.woody_frac (fraction of 1 m px in "
                       "tree/orchard/vineyard/hedge/garden/shrub segments) x (cell ∩ AOI) area; "
                       "area_ha_forest = tree_frac x (cell ∩ AOI); live part: nDSM >= "
                       "min_tree_height canopy pixels")
    else:
        canopy_m2 = float(canopy.get("stand_m2", 0.0)) + float(live_canopy_m2)
        canopy_note = ("product part (2.1 product, no fractions): landcover.grid25 25 m cells "
                       "(dominant v2 segment class in tree/orchard/vineyard/hedge/garden/shrub) "
                       "whose centre lies in the AOI, x 625 m2; area_ha_forest = class 'tree' "
                       "only; live part: nDSM >= min_tree_height canopy pixels")
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
        "area_ha_stand": round((float(canopy.get("stand_m2", 0.0)) + live_canopy_m2) / 1e4, 3),
        "canopy_denominator_note": canopy_note,
        "canopy_method": method,
        "stems_per_ha_total": round(len(stems) / area_ha, 1) if area_ha > 0 else 0,
        "stems_per_ha_canopy": round(len(stems) / canopy_ha, 1) if canopy_ha > 0 else 0,
        "stems_per_ha_forest": round(len(forest_stems) / forest_ha, 1) if forest_ha > 0 else 0,
        "stems_per_ha_note": ("non-edge trees only; is_edge is judged against the submitted AOI "
                              "(crown disc crosses the boundary) for product and live apices alike"),
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
        "by_vitality": _by("vitality"),
        "by_species_hint": _by("species_hint"),
        "by_source": _by("_source", "product"),
        "volume_m3_est_total": round(vol, 1),
        "volume_m3_est_per_ha_canopy": round(vol / canopy_ha, 1) if canopy_ha > 0 else None,
        "dh_per_year_m_median": round(float(np.median(dh)), 3) if dh.size else None,
        "n_with_dh": int(dh.size),
        "grid25_cells_in_aoi": int(canopy.get("n_cells", 0)),
    }


# --------------------------------------------------------------------------
# changes (epoch a = product) helpers
# --------------------------------------------------------------------------

def rows_to_trees(rows: list[dict], transform, shape, with_labels: bool = True) -> list:
    """Product rows → ``tree_inventory.Tree`` objects on a live raster grid
    (row/col clamped).  ``label`` = index+1, matching
    :func:`crown_label_raster`, so crown-overlap (pass 2) matching works."""
    import tree_inventory as tv
    out = []
    for i, r in enumerate(rows):
        col, row = ~transform * (r["e"], r["n"])
        rr = int(min(max(row, 0), shape[0] - 1)); cc = int(min(max(col, 0), shape[1] - 1))
        out.append(tv.Tree(
            tree_id=r["tree_id"], seq=i, label=(i + 1) if with_labels else 0,
            apex_e=float(r["e"]), apex_n=float(r["n"]),
            apex_row=rr, apex_col=cc, height_m=float(r.get("h_m") or 0.0),
            crown_area_sqm=float(r.get("crown_area_m2") or 0.0),
            crown_radius_mean_m=float(r.get("crown_r_m") or 0.0),
            crown_radius_max_m=float(r.get("crown_r_m") or 0.0), is_edge=bool(r.get("_is_edge")),
            dbh_est_cm=float(r.get("dbh_est_cm") or 0.0),
            volume_m3_est=float(r.get("volume_m3_est") or 0.0),
            leaf_type=r.get("leaf_type_hint") or "unknown",
            leaf_type_conf=float(r.get("leaf_type_conf") or 0.0),
            vitality=r.get("vitality") or "unknown",
            vitality_conf=float(r.get("vitality_conf") or 0.0),
            species_hint=r.get("species_hint") or "unknown",
            species_conf=float(r.get("species_conf") or 0.0),
            detection_source=r.get("detection_source") or "ndsm",
            detection_conf=float(r.get("detection_conf") or 0.0),
            surface_class=r.get("surface_class") or "tree",
            tree_likelihood=float(r.get("tree_likelihood") or 1.0)))
    return out


def crown_label_raster(rows: list[dict], crowns: dict, transform, shape) -> tuple[np.ndarray, int]:
    """Epoch-a crown label raster on the live grid: row i → label i+1.
    2.2 crown polygons where available, else the mean-radius disc (≥1 px).
    Returns (labels_a int32, n_polygon_crowns)."""
    from rasterio.features import rasterize
    from shapely.geometry import Point as _P
    shapes = []
    n_poly = 0
    res = abs(transform.a)
    for i, r in enumerate(rows):
        g = crowns.get(r["tree_id"]) if crowns else None
        if g is not None and not g.is_empty:
            n_poly += 1
        else:
            g = _P(r["e"], r["n"]).buffer(max(res * 0.75, float(r.get("crown_r_m") or 1.0)))
        shapes.append((g, i + 1))
    if not shapes:
        return np.zeros(shape, np.int32), 0
    # taller crowns painted last so an overlap belongs to the dominant tree
    shapes.sort(key=lambda sv: float(rows[sv[1] - 1].get("h_m") or 0.0))
    lab = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32", all_touched=False)
    return lab, n_poly


def crown_proxy_ndsm(rows: list[dict], transform, shape, labels_a: np.ndarray | None = None) -> np.ndarray:
    """Epoch-a nDSM proxy from product apices: every crown (2.2 polygon or
    mean-radius disc) painted with its ``h_m``.  Used for the 3 m-radius apex
    evidence in ``match_trees`` and for the coarse ``felling_patches`` — the
    product carries no 1 m raster (disk rule)."""
    if labels_a is None:
        labels_a, _ = crown_label_raster(rows, {}, transform, shape)
    h = np.zeros(len(rows) + 1, np.float32)
    for i, r in enumerate(rows):
        h[i + 1] = float(r.get("h_m") or 0.0)
    return h[np.clip(labels_a, 0, len(rows))]


def felling_patches_from_crowns(rows: list[dict], labels_a: np.ndarray, ndsm_b: np.ndarray,
                                transform, min_drop_m: float, min_patch_sqm: float = 25.0,
                                pmask: np.ndarray | None = None) -> tuple[list[dict], dict]:
    """Crown-level felling patches for the product→live change path.

    The product has no epoch-a raster, so a pixel drop against an h_m-painted
    proxy would flag every crown margin (apex height − sloping live crown).
    Instead each epoch-a crown footprint is judged as a whole: live nDSM p90
    inside the footprint vs the crown's apex height.  Crowns with
    ``h_m − p90_live ≥ min_drop_m`` are felled; adjacent felled footprints
    merge into patches (≥ min_patch_sqm).  Returns (patches, stats) with
    patches = [{geometry_3035, area_sqm, drop_mean_m, drop_max_m,
    height_a_mean_m, n_crowns}].
    """
    from scipy import ndimage
    from rasterio.features import shapes
    from shapely.geometry import shape as _shape
    n = len(rows)
    if n == 0 or labels_a.max() == 0:
        return [], {"n_crowns_judged": 0, "n_crowns_felled": 0}
    zb = np.nan_to_num(ndsm_b, nan=0.0).astype(np.float32)
    lab = labels_a if pmask is None else np.where(pmask, labels_a, 0)
    idx = np.arange(1, n + 1)
    p90 = np.asarray(ndimage.labeled_comprehension(
        zb, lab, idx, lambda v: float(np.percentile(v, 90)) if v.size else np.nan, float, np.nan))
    h = np.array([float(r.get("h_m") or 0.0) for r in rows], np.float32)
    drop = h - p90
    felled = np.isfinite(p90) & (drop >= min_drop_m)
    n_judged = int(np.isfinite(p90).sum())
    if not felled.any():
        return [], {"n_crowns_judged": n_judged, "n_crowns_felled": 0}
    fl = np.zeros(n + 1, bool); fl[1:] = felled
    fmask = fl[np.clip(lab, 0, n)]
    comp, ncomp = ndimage.label(fmask, structure=np.ones((3, 3)))
    px = abs(transform.a * transform.e)
    out = []
    for geom, val in shapes(comp.astype(np.int32), mask=comp > 0, transform=transform, connectivity=8):
        cid = int(val)
        sel = comp == cid
        area = float(sel.sum()) * px
        if area < min_patch_sqm:
            continue
        labs = np.unique(lab[sel]); labs = labs[labs > 0] - 1
        poly = _shape(geom).simplify(0.5, preserve_topology=True)
        out.append({"geometry_3035": poly, "area_sqm": round(area, 1),
                    "drop_mean_m": round(float(drop[labs].mean()), 2) if labs.size else None,
                    "drop_max_m": round(float(drop[labs].max()), 2) if labs.size else None,
                    "height_a_mean_m": round(float(h[labs].mean()), 2) if labs.size else None,
                    "n_crowns": int(labs.size)})
    return out, {"n_crowns_judged": n_judged, "n_crowns_felled": int(felled.sum())}


def product_als_year(rows: list[dict]) -> int | None:
    ys = [r["als_year"] for r in rows if r.get("als_year")]
    return int(np.median(ys)) if ys else None
