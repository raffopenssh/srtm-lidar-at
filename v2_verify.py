"""v2_verify — integrity + improvement gates for v2 KG products.

Two call sites, two gates:

1. **Peer, before upload** (``verify_before_upload``): the freshly built
   ``<code>_v2.json.gz`` + ``<code>_light_v2.gpkg`` against the v1 JSON of
   the same code (downloaded from Zenodo, ~MB).  Checks the codec round
   trip, structural completeness, **tile stitching** (every LiDAR-bearing
   tile segmented, segmented area ≥ v1, no all-zero holes in the
   ``segment_type`` raster inside the parcel union), parcel/building
   counts vs. cadastre + v1, coverage percentages ≥ v1 (−tolerance),
   unclassified share ≤ v1 (+tolerance), light GPKG ``integrity_check``
   + layer inventory + feature counts consistent with the JSON.
   A failed gate means **nothing is uploaded** and the KG is reported
   as ``v2_verify_failed`` (the parent defers/retries it; the director
   sees it in the merged log as ``v2verify:``).

2. **Primary, before ingest** (``verify_for_ingest``): the downloaded blob
   against the v1 row already in the search index (or the v1 JSON on
   disk).  Same document-level checks (no GPKG — we do not pull light
   GPKGs to the primary; the manifest must merely show a committed
   ``_light_gpkg_v2`` of non-zero size).  Only after this passes does the
   primary swap the index row to v2, store the blob and delete the v1
   file.

Every check appends to ``report['checks']`` as ``(name, ok, detail)`` so
the operator can see *which* gate tripped.  Tolerances are deliberately
modest — v2 must be at least as complete as v1; it may legitimately have
more segments (finer classes) or fewer (merged classes).
"""
from __future__ import annotations

import io
import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

import kg_json_v2

log = logging.getLogger("austria_processor.v2verify")

REQUIRED_SECTIONS = ("version", "kg_code", "bbox", "area_summary", "landscape", "terrain",
                     "parcels", "building_footprints", "coverage", "data_quality", "model")
LIGHT_LAYERS_REQUIRED = ("segments", "segment_points", "parcels", "buildings",
                         "segment_type", "segment_height")
LIGHT_LAYERS_V2 = ("parcel_outline_z",)
# product 2.1 (docs/v2.1-product-spec.md) — checked by check_v21_document /
# check_v21_light_gpkg; product_version is fatal, the 5 m / apex layers are
# fatal only when the JSON claims them.
PRODUCT_VERSION = "2.1"
LIGHT_LAYERS_V21 = ("terrain_coarse_dtm", "terrain_coarse_slope")

TOL = {
    "parcel_count_rel": 0.01,      # cadastre may change a little between runs
    "coverage_pp": 1.0,            # percentage points
    "building_cov_pp": 2.0,
    "seg_area_rel": 0.02,          # v2 segmented area ≥ 0.98 × v1
    "unclassified_pp": 2.0,        # v2 unclassified share ≤ v1 + 2 pp
    "segments_rel_min": 0.5,       # sanity — v2 segments ≥ half of v1's
    "raster_hole_frac": 0.005,     # zero segment_type inside parcel union
    "outline_z_min_frac": 0.5,     # ≥ 50 %% of parcels with geometry carry outline_z
}


class Report(dict):
    def __init__(self):
        super().__init__(ok=True, checks=[], errors=[], warnings=[])

    def check(self, name, ok, detail="", *, fatal=True):
        self["checks"].append((name, bool(ok), detail))
        if not ok:
            (self["errors"] if fatal else self["warnings"]).append(f"{name}: {detail}")
            if fatal:
                self["ok"] = False
        return ok

    def summary(self) -> str:
        n_ok = sum(1 for _, ok, _ in self["checks"] if ok)
        s = f"{'PASS' if self['ok'] else 'FAIL'} {n_ok}/{len(self['checks'])} checks"
        if self["errors"]:
            s += " | " + "; ".join(self["errors"][:4])
        if self["warnings"]:
            s += " | warn: " + "; ".join(self["warnings"][:3])
        return s


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _num(v, default=None):
    try:
        f = float(v)
        return default if f != f else f
    except (TypeError, ValueError):
        return default


def _unclassified_pct(doc) -> float:
    a = _g(doc, "area_summary", default={}) or {}
    tot = 0.0
    unc = 0.0
    for t, v in a.items():
        if not isinstance(v, dict):
            continue
        s = _num(v.get("area_sqm"), 0.0)
        tot += s
        if t == "unclassified":
            unc += s
    return 100.0 * unc / tot if tot > 0 else 0.0


def _active_tiles(doc) -> tuple[int, int, list]:
    """(n_tiles_with_lidar, n_tiles_segmented, unsegmented_tile_idx)."""
    tiles = _g(doc, "data_quality", "tiles", default=[]) or []
    lidar = seg = 0
    bad = []
    for t in tiles:
        if not isinstance(t, dict) or not t.get("dtm"):
            continue
        if t.get("outside_austria") or int(t.get("valid_pixels") or 0) < 100:
            continue
        lidar += 1
        if t.get("segmentation") and int(t.get("n_objects") or 0) > 0:
            seg += 1
        else:
            bad.append(t.get("tile_index"))
    return lidar, seg, bad


# --------------------------------------------------------------------------
# document-level checks (shared by both gates)
# --------------------------------------------------------------------------

def check_document(v2: dict, v1: dict | None, rep: Report, *, code: str | None = None,
                   expected_parcels: int | None = None) -> None:
    rep.check("sections", all(k in v2 for k in REQUIRED_SECTIONS),
              "missing: " + ",".join(k for k in REQUIRED_SECTIONS if k not in v2))
    rep.check("version", str(v2.get("version")) == "v2", f"version={v2.get('version')}")
    if code:
        rep.check("kg_code", str(v2.get("kg_code")) == str(code),
                  f"{v2.get('kg_code')} != {code}")
    clf = str(_g(v2, "model", "classifier", default=""))
    rep.check("model_v2", clf.startswith("lgbm_v2"), f"classifier={clf!r}")
    by_tile = _g(v2, "model", "classifier_by_tile")
    if isinstance(by_tile, dict):
        n_v1 = sum(int(n) for c, n in by_tile.items() if not str(c).startswith("lgbm_v2"))
        rep.check("model_all_tiles_v2", n_v1 == 0,
                  f"{n_v1} tile(s) fell back to v1 RF", fatal=False)

    n_seg = int(_num(_g(v2, "landscape", "n_segments"), 0) or 0)
    rep.check("segments_present", n_seg > 0, f"n_segments={n_seg}")

    # tile stitching — every LiDAR tile segmented.  A v1 baseline that
    # itself carried gaps (unsegmented tiles, upstream-failed tiles, DTM
    # holes) caps what an upgrade regenerated from that same full GPKG can
    # achieve, so against a baseline these are *non-regression* checks;
    # without one (fresh KG) they are advisory — v1 completed such KGs as
    # "partial" too, and a fatal gate here would only loop the KG.
    lidar, seg, bad = _active_tiles(v2)
    l1, s1, _ = _active_tiles(v1) if v1 is not None else (0, 0, [])
    n_up = int(_num(_g(v2, "data_quality", "n_upstream_failed_tiles"), 0) or 0)
    n_up1 = int(_num(_g(v1, "data_quality", "n_upstream_failed_tiles"), 0) or 0) if v1 else 0
    rep.check("tiles_segmented", lidar > 0 and seg > 0, f"{seg}/{lidar} lidar tiles segmented")
    rep.check("tiles_all_segmented", seg == lidar,
              f"{seg}/{lidar} lidar tiles segmented; unsegmented={bad[:10]}",
              fatal=(v1 is not None and seg < s1))
    rep.check("no_upstream_failed_tiles", n_up == 0,
              f"{n_up} upstream-failed tile(s)" + (f" (v1: {n_up1})" if v1 else ""),
              fatal=(v1 is not None and n_up > n_up1))

    # parcels
    pd = _g(v2, "parcels", "details", default=[]) or []
    pc = int(_num(_g(v2, "parcels", "count"), 0) or 0)
    rep.check("parcels_count_matches_details", pc == len(pd), f"count={pc} details={len(pd)}")
    if expected_parcels is not None and expected_parcels > 0:
        rel = abs(pc - expected_parcels) / expected_parcels
        rep.check("parcels_vs_cadastre", rel <= TOL["parcel_count_rel"],
                  f"{pc} vs cadastre {expected_parcels}")
    if pd:
        with_geom = sum(1 for p in pd if p.get("vertex_heights"))
        with_oz = sum(1 for p in pd if p.get("outline_z"))
        rep.check("parcel_outline_z", with_geom == 0 or with_oz >= TOL["outline_z_min_frac"] * with_geom,
                  f"{with_oz}/{with_geom} parcels carry outline_z", fatal=False)
        n_elev = sum(1 for p in pd if _num(p.get("elevation_m")) is not None)
        pd1 = (_g(v1, "parcels", "details", default=[]) or []) if v1 else []
        n_elev1 = sum(1 for p in pd1 if isinstance(p, dict) and _num(p.get("elevation_m")) is not None)
        _elev_ok = n_elev >= 0.9 * len(pd)
        # fatal: below half, or regressed vs a v1 baseline that had them
        _elev_fatal = n_elev < 0.5 * len(pd) or (pd1 and n_elev < 0.9 * n_elev1)
        rep.check("parcel_elevations", _elev_ok,
                  f"{n_elev}/{len(pd)} parcels with elevation" + (f" (v1: {n_elev1}/{len(pd1)})" if pd1 else ""),
                  fatal=bool(_elev_fatal))

    check_v21_document(v2, rep)

    cov = _g(v2, "coverage", default={}) or {}
    cov1 = (_g(v1, "coverage", default={}) or {}) if v1 else {}
    for k in ("parcel_elevation_coverage_pct", "parcel_segmentation_coverage_pct"):
        v = _num(cov.get(k))
        v1v = _num(cov1.get(k)) if v1 else None
        # a baseline that had 0 / no coverage figure cannot be regressed
        rep.check(f"coverage.{k}", v is not None and v > 0, f"{k}={v}",
                  fatal=not (v1 is not None and not v1v))

    if v1 is None:
        return
    # ---- improvement / non-regression vs v1 --------------------------------
    pc1 = int(_num(_g(v1, "parcels", "count"), 0) or 0)
    if pc1 > 0:
        rel = abs(pc - pc1) / pc1
        rep.check("parcels_vs_v1", rel <= TOL["parcel_count_rel"], f"v2={pc} v1={pc1}")
    cov1 = _g(v1, "coverage", default={}) or {}
    for k, tol in (("parcel_elevation_coverage_pct", TOL["coverage_pp"]),
                   ("parcel_segmentation_coverage_pct", TOL["coverage_pp"]),
                   ("building_height_coverage_pct", TOL["building_cov_pp"])):
        a, b = _num(cov.get(k)), _num(cov1.get(k))
        if a is None or b is None:
            continue
        rep.check(f"{k}_ge_v1", a >= b - tol, f"v2={a:.1f} v1={b:.1f}")
    sa, sb = _num(cov.get("total_segmented_area_sqm")), _num(cov1.get("total_segmented_area_sqm"))
    if sa is not None and sb and sb > 0:
        rep.check("segmented_area_ge_v1", sa >= (1 - TOL["seg_area_rel"]) * sb,
                  f"v2={sa:.0f} v1={sb:.0f} m²")
    rep.check("lidar_tiles_ge_v1", lidar >= l1, f"v2={lidar} v1={l1} lidar tiles")
    n1 = int(_num(_g(v1, "landscape", "n_segments"), 0) or 0)
    if n1 > 0:
        rep.check("segments_vs_v1", n_seg >= TOL["segments_rel_min"] * n1,
                  f"v2={n_seg} v1={n1}", fatal=False)
    u2, u1 = _unclassified_pct(v2), _unclassified_pct(v1)
    rep.check("unclassified_le_v1", u2 <= u1 + TOL["unclassified_pp"], f"v2={u2:.1f}%% v1={u1:.1f}%%")
    nb2 = int(_num(_g(v2, "new_buildings", "count"), 0) or 0)
    nb1 = int(_num(_g(v1, "new_buildings", "count"), 0) or 0)
    rep["delta"] = {"n_segments": n_seg - n1, "unclassified_pp": round(u2 - u1, 2),
                    "new_buildings": nb2 - nb1,
                    "seg_area_rel": round(sa / sb, 4) if sa is not None and sb else None}


def _bbox_3035(v2: dict):
    bb = v2.get("bbox") or {}
    try:
        from pyproj import Transformer
        t = Transformer.from_crs(4326, 3035, always_xy=True)
        xs, ys = zip(*[t.transform(x, y) for x, y in (
            (bb["min_lon"], bb["min_lat"]), (bb["max_lon"], bb["min_lat"]),
            (bb["min_lon"], bb["max_lat"]), (bb["max_lon"], bb["max_lat"]))])
        return min(xs), min(ys), max(xs), max(ys)
    except Exception:  # noqa: BLE001
        return None


def check_v21_document(v2: dict, rep: Report) -> None:
    """Product-2.1 JSON checks (JSON subset — also run on the primary)."""
    pv = str(v2.get("product_version") or "")
    rep.check("product_version", pv == PRODUCT_VERSION, f"product_version={pv!r}")
    g = _g(v2, "terrain", "grid25")
    if not rep.check("grid25_present", isinstance(g, dict) and bool(g.get("elev")),
                     "terrain.grid25 missing (tile arrays freed?)", fatal=False):
        return
    try:
        import v21_products as v21
        d = v21.decode_terrain_grid25(g)
    except Exception as e:  # noqa: BLE001
        rep.check("grid25_decodes", False, str(e)[:160])
        return
    rep.check("grid25_decodes", True)
    cell = d["cell_m"]
    cols, rows = d["cols"], d["rows"]
    bb = _bbox_3035(v2)
    if bb:
        # the grid is built from the (overlapping, bbox-overshooting) tile
        # rectangles snapped to 25 m, so it must COVER the bbox and may
        # exceed it by up to ~1 tile (1.6 km) per side — not "±1 cell".
        x0, y0 = d["x0"], d["y0"]
        x1, y1 = x0 + cols * cell, y0 - rows * cell
        covers = (x0 <= bb[0] + cell and y1 <= bb[1] + cell
                  and x1 >= bb[2] - cell and y0 >= bb[3] - cell)
        slack = 1700.0
        sane = (bb[0] - x0 <= slack and x1 - bb[2] <= slack
                and bb[1] - y1 <= slack and y0 - bb[3] <= slack)
        rep.check("grid25_dims", covers and sane,
                  f"grid {cols}x{rows}@{cell:.0f}m x[{x0:.0f},{x1:.0f}] y[{y1:.0f},{y0:.0f}] "
                  f"vs bbox x[{bb[0]:.0f},{bb[2]:.0f}] y[{bb[1]:.0f},{bb[3]:.0f}]")
    n_valid = int(np.isfinite(d["elev"]).sum())
    frac = n_valid / max(cols * rows, 1)
    rep.check("grid25_finite", frac >= 0.4, f"{n_valid}/{cols * rows} cells valid ({100 * frac:.0f}%%)")
    emin, emax = _num(_g(v2, "terrain", "elevation_min_m")), _num(_g(v2, "terrain", "elevation_max_m"))
    if n_valid and emin is not None and emax is not None:
        gmin, gmax = float(np.nanmin(d["elev"])), float(np.nanmax(d["elev"]))
        ok5 = gmin >= emin - 5 and gmax <= emax + 5
        rep.check("grid25_range", gmin >= emin - 50 and gmax <= emax + 50,
                  f"grid {gmin:.0f}..{gmax:.0f} m vs terrain {emin:.0f}..{emax:.0f} m")
        if not ok5:
            rep.check("grid25_range_drift", False,
                      f"grid {gmin:.0f}..{gmax:.0f} m outside terrain ±5 m", fatal=False)
    lc = _g(v2, "landcover", "grid25")
    if isinstance(lc, dict):
        rep.check("landcover_grid25_dims", int(lc.get("cols", 0)) == cols and int(lc.get("rows", 0)) == rows,
                  f"landcover {lc.get('cols')}x{lc.get('rows')} vs terrain {cols}x{rows}")
    ts = _g(v2, "tree_stats", default={}) or {}
    rep.check("tree_stats_v21", "n_apices" in ts and "det_mode" in ts,
              f"keys={sorted(ts)[:8]}", fatal=False)


def check_v21_light_gpkg(c: sqlite3.Connection, layers: dict, v2: dict, rep: Report) -> None:
    """Product-2.1 light GPKG checks (peer only; *c* is an open ro connection)."""
    missing = [l for l in LIGHT_LAYERS_V21 if l not in layers]
    has_grid = bool(_g(v2, "terrain", "grid25"))
    rep.check("terrain_coarse_layers", not missing, "missing: " + ",".join(missing),
              fatal=has_grid)
    ts = _g(v2, "tree_stats", default={}) or {}
    n_ap = int(_num(ts.get("n_apices"), 0) or 0)
    claims = ts.get("apices_layer") == "tree_apices"
    if "tree_apices" not in layers:
        rep.check("apices_layer", not claims and n_ap == 0,
                  f"tree_apices layer missing (json n_apices={n_ap})", fatal=claims)
        return
    try:
        n = c.execute('SELECT COUNT(*) FROM "tree_apices"').fetchone()[0]
    except Exception as e:  # noqa: BLE001
        rep.check("apices_layer", False, str(e)[:120])
        return
    rtree = c.execute("SELECT COUNT(*) FROM gpkg_extensions WHERE table_name='tree_apices' "
                      "AND extension_name='gpkg_rtree_index'").fetchone()[0] > 0
    rep.check("apices_layer", rtree and (n_ap == 0 or abs(n - n_ap) <= 0.2 * n_ap),
              f"layer={n} json={n_ap} rtree={rtree}")


# --------------------------------------------------------------------------
# light GPKG checks (peer only)
# --------------------------------------------------------------------------

def _segment_type_nonzero(path: str, union_geom_3035=None) -> tuple[int, float | None]:
    """(non-zero px, hole fraction inside union or None).  Reads the palette
    PNG tiles directly (GDAL expands them to RGBA)."""
    from PIL import Image
    import rasterio
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = c.execute("SELECT MAX(zoom_level) FROM gpkg_tile_matrix WHERE table_name='segment_type'").fetchone()
        if not row or row[0] is None:
            return 0, None
        z = row[0]
        tw, th, mw, mh = c.execute(
            "SELECT tile_width, tile_height, matrix_width, matrix_height FROM gpkg_tile_matrix "
            "WHERE table_name='segment_type' AND zoom_level=?", (z,)).fetchone()
        H, W = mh * th, mw * tw
        full = np.zeros((H, W), bool)
        nz = 0
        for col, rw, blob in c.execute("SELECT tile_column, tile_row, tile_data FROM segment_type WHERE zoom_level=?", (z,)):
            im = Image.open(io.BytesIO(blob))
            a = np.array(im if im.mode in ("P", "L") else im.convert("L"), dtype=np.uint8)
            m = a > 0
            nz += int(m.sum())
            r0, c0 = rw * th, col * tw
            hh, ww = min(th, H - r0), min(tw, W - c0)
            if hh > 0 and ww > 0:
                full[r0:r0 + hh, c0:c0 + ww] = m[:hh, :ww]
        hole = None
        if union_geom_3035 is not None and not union_geom_3035.is_empty:
            from rasterio.features import geometry_mask
            with rasterio.open(f"GPKG:{path}:segment_type") as ds:
                tf = ds.transform
            try:
                inside = ~geometry_mask([union_geom_3035], out_shape=(H, W), transform=tf,
                                        invert=False, all_touched=False)
                n_in = int(inside.sum())
                if n_in > 0:
                    hole = float((inside & ~full).sum()) / n_in
            except Exception as e:  # noqa: BLE001
                log.debug("segment_type hole check skipped: %s", e)
        return nz, hole
    finally:
        c.close()


def check_light_gpkg(path: str, v2: dict, rep: Report, union_geom_3035=None) -> None:
    p = Path(path)
    if not rep.check("light_gpkg_exists", p.exists() and p.stat().st_size > 0, str(path)):
        return
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        ic = c.execute("PRAGMA integrity_check").fetchone()[0]
        rep.check("light_gpkg_integrity", ic == "ok", ic[:120])
        app_id = c.execute("PRAGMA application_id").fetchone()[0]
        rep.check("light_gpkg_app_id", app_id == 0x47504B47, hex(app_id))
        layers = {n: t for n, t in c.execute("SELECT table_name, data_type FROM gpkg_contents")}
        missing = [l for l in LIGHT_LAYERS_REQUIRED if l not in layers]
        rep.check("light_gpkg_layers", not missing, "missing: " + ",".join(missing))
        missing2 = [l for l in LIGHT_LAYERS_V2 if l not in layers]
        rep.check("light_gpkg_v2_layers", not missing2, "missing: " + ",".join(missing2), fatal=False)

        def _count(t):
            try:
                return c.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            except Exception:
                return -1
        n_seg = int(_num(_g(v2, "landscape", "n_segments"), 0) or 0)
        # ``segments`` holds one polygon per connected part of a segment
        # (a label may split into several parts), so count distinct ids.
        try:
            ns = c.execute('SELECT COUNT(DISTINCT id) FROM "segments"').fetchone()[0]
        except Exception:
            ns = _count("segments")
        # The polygon layer is vectorised from the *stitched* label raster
        # inside the KG mask, so objects whose pixels were overwritten in a
        # tile overlap (or clipped by the block mask) have a JSON row and a
        # ``segment_points`` row but no polygon.  That is a v1-era trait, not
        # a v2 defect: 45410-east (v1, 2026-09-18) drops 2.0 %, fresh v2
        # blocks 2.3–2.7 %.  A 2 % fatal tolerance therefore failed every
        # fresh v2 KG fleet-wide on 2026-09-18 (files discarded, no ``_json``,
        # KG re-picked in a loop).  Fatal only past 10 %; 2–10 % is a
        # non-fatal drift warning so the rate stays visible in the log.
        _d = abs(ns - n_seg)
        rep.check("segments_layer_count", ns > 0 and (n_seg == 0 or _d <= max(20, 0.10 * n_seg)),
                  f"layer={ns} json={n_seg}")
        if n_seg and ns > 0 and _d > max(5, 0.02 * n_seg):
            rep.check("segments_layer_drift", False,
                      f"layer={ns} json={n_seg} ({100.0 * _d / n_seg:.1f}% objects without polygon)",
                      fatal=False)
        npx = _count("segment_points")
        rep.check("segment_points_count", abs(npx - ns) <= max(5, 0.02 * ns),
                  f"points={npx} segments={ns}", fatal=False)
        pc = int(_num(_g(v2, "parcels", "count"), 0) or 0)
        npar = _count("parcels")
        rep.check("parcels_layer_count", pc == 0 or abs(npar - pc) <= max(2, 0.01 * pc),
                  f"layer={npar} json={pc}")
        nb = int(_num(_g(v2, "building_footprints", "count"), 0) or 0)
        nbl = _count("buildings")
        rep.check("buildings_layer_count", nb == 0 or abs(nbl - nb) <= max(2, 0.01 * nb),
                  f"layer={nbl} json={nb}")
        try:
            check_v21_light_gpkg(c, layers, v2, rep)
        except Exception as e:  # noqa: BLE001
            rep.check("v21_gpkg_checks", False, str(e)[:160], fatal=False)
        try:
            n_styles = c.execute("SELECT COUNT(DISTINCT f_table_name) FROM layer_styles").fetchone()[0]
            rep.check("layer_styles", n_styles >= len(LIGHT_LAYERS_REQUIRED) - 1,
                      f"{n_styles} styled layers", fatal=False)
        except Exception:
            pass
    finally:
        c.close()
    # raster stitching: nonzero segment_type px vs. segmented area, holes in union
    try:
        nz, hole = _segment_type_nonzero(path, union_geom_3035)
        sa = _num(_g(v2, "coverage", "total_segmented_area_sqm"))
        if sa and sa > 0:
            # ``total_segmented_area_sqm`` is Σ per-tile valid px over the
            # *overlapping* 1.5 km grid (0.1 km overlap, see
            # ``_compute_tile_grid``) — every seam strip is counted twice,
            # while the stitched raster holds each px once.  For an n×m grid
            # the deduped/summed ratio tends to (1.4/1.5)² ≈ 0.87, so a flat
            # 0.95 floor failed every multi-tile fresh v2 KG on 2026-09-18
            # (45631-northeast 4 tiles → 0.934, 72010-north → 0.90) while
            # single-tile 19570 sat at 0.99999.  Real stitching gaps are
            # caught by ``segment_type_no_holes`` below; here fatal only
            # below the overlap-corrected floor, non-fatal drift above it.
            _nt = int(_num(_g(v2, "coverage", "n_tiles"), 1) or 1)
            _floor = 0.95 * ((1.4 / 1.5) ** 2 if _nt > 1 else 1.0)
            _r = nz / sa
            rep.check("segment_type_raster_area", _r >= _floor,
                      f"nonzero px={nz} json segmented m²={sa:.0f} ratio={_r:.3f} floor={_floor:.3f} tiles={_nt}")
            if _floor <= _r < 0.95:
                rep.check("segment_type_raster_area_drift", False,
                          f"ratio={_r:.3f} (tile-overlap double count, {_nt} tiles)", fatal=False)
        if hole is not None:
            # Holes are expected when the doc itself declares a gap (an
            # unsegmented or upstream-failed tile) — inherited from the v1
            # baseline / its full GPKG; fatal only for an *undeclared* hole.
            _l, _s, _ = _active_tiles(v2)
            _nup = int(_num(_g(v2, "data_quality", "n_upstream_failed_tiles"), 0) or 0)
            _declared_gap = (_s < _l) or _nup > 0
            rep.check("segment_type_no_holes", hole <= TOL["raster_hole_frac"],
                      f"{100*hole:.2f}%% of parcel union has no segment class"
                      + (" (declared tile gap)" if _declared_gap else ""),
                      fatal=not _declared_gap)
        rep["segment_type_nonzero_px"] = nz
        rep["segment_type_hole_frac"] = hole
    except Exception as e:  # noqa: BLE001
        rep.check("segment_type_readable", False, str(e)[:160], fatal=False)


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def verify_before_upload(code: str, json_v2_path: str, light_gpkg_path: str | None,
                         v1_doc: dict | None, *, expected_parcels: int | None = None,
                         union_geom_3035=None) -> Report:
    rep = Report()
    p = Path(json_v2_path)
    if not rep.check("json_v2_exists", p.exists() and p.stat().st_size > 0, str(p)):
        return rep
    blob = p.read_bytes()
    rep["json_v2_bytes"] = len(blob)
    rep.check("json_v2_gzip", blob[:2] == b"\x1f\x8b", "not gzip")
    try:
        v2 = kg_json_v2.decode(blob)
    except Exception as e:  # noqa: BLE001
        rep.check("json_v2_decodes", False, str(e)[:160])
        return rep
    rep.check("json_v2_decodes", True)
    ok, why = kg_json_v2.roundtrip_ok(v2)
    rep.check("codec_roundtrip", ok, why)
    check_document(v2, v1_doc, rep, code=code, expected_parcels=expected_parcels)
    if light_gpkg_path is not None:
        check_light_gpkg(light_gpkg_path, v2, rep, union_geom_3035)
    if v1_doc is not None:
        try:
            v1_bytes = len(json.dumps(v1_doc, indent=2).encode())
            rep["size_ratio_v1_pretty_to_v2_gz"] = round(v1_bytes / max(len(blob), 1), 1)
        except Exception:
            pass
    return rep


def verify_for_ingest(code: str, blob: bytes, v1_doc: dict | None, manifest_entries: dict | None,
                      *, v1_row: dict | None = None) -> tuple[Report, dict | None]:
    """Primary gate.  Returns (report, decoded_doc_or_None)."""
    rep = Report()
    rep.check("blob_gzip", blob[:2] == b"\x1f\x8b", "not gzip")
    try:
        v2 = kg_json_v2.decode(blob)
    except Exception as e:  # noqa: BLE001
        rep.check("blob_decodes", False, str(e)[:160])
        return rep, None
    rep.check("blob_decodes", True)
    if manifest_entries is not None:
        e = manifest_entries.get(f"{code}_light_gpkg_v2")
        rep.check("manifest_light_gpkg_v2", isinstance(e, dict) and int(e.get("size") or 0) > 0
                  and bool(e.get("uploaded_at")), "no committed _light_gpkg_v2 entry")
    check_document(v2, v1_doc, rep, code=code)
    if v1_doc is None and v1_row:
        # Fall back to the flat index row for the few comparable numbers.
        pc = int(_num(_g(v2, "parcels", "count"), 0) or 0)
        pc1 = int(_num(v1_row.get("n_parcels"), 0) or 0)
        if pc1 > 0:
            rep.check("parcels_vs_v1_row", abs(pc - pc1) / pc1 <= TOL["parcel_count_rel"],
                      f"v2={pc} v1={pc1}")
        n1 = int(_num(v1_row.get("n_segments"), 0) or 0)
        n2 = int(_num(_g(v2, "landscape", "n_segments"), 0) or 0)
        if n1 > 0:
            rep.check("segments_vs_v1_row", n2 >= TOL["segments_rel_min"] * n1,
                      f"v2={n2} v1={n1}", fatal=False)
    return rep, v2
