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
try:
    from v21_products import PRODUCT_VERSION
except Exception:  # noqa: BLE001
    PRODUCT_VERSION = "2.3"
LIGHT_LAYERS_V21 = ("terrain_coarse_dtm", "terrain_coarse_slope")

TOL = {
    "parcel_count_rel": 0.01,      # cadastre may change a little between runs
    "coverage_pp": 1.0,            # percentage points (elevation; fatal)
    "seg_coverage_pp_warn": 1.0,   # segmentation coverage: 1–2.5 pp drop = non-fatal drift
    "seg_coverage_pp": 2.5,        # … drift past 2.5 pp is reported loudly (segv2 drops
                                   # path/earthwork objects, so a few single-object
                                   # parcels lose their area_summary — a class-set
                                   # consequence, not a bug; 2026-09-21: 28 upgrades
                                   # struck at −1.1…−1.7 pp)
    "seg_coverage_pp_fatal": 8.0,  # … fatal only when BOTH the raw figure and the
                                   # ≥300 m² figure drop past this (a tile that came
                                   # back empty).  2026-09-22: 217 strike events / 35
                                   # parents struck out at −2.5…−7.5 pp (≥300 m²) with
                                   # raw drops of 0…−5.4 pp — every one deterministic
                                   # (identical figures on two peers), i.e. a segv2 vs
                                   # v1 class-set difference on 3–7 % of parcels, not a
                                   # data hole.  Empty tiles are caught by
                                   # segment_type_no_holes / lidar_tiles_ge_v1 /
                                   # segmented_area_ge_v1 independently.
    "parcel_count_abs": 2,         # … or ±2 parcels, whichever is larger (tiny blocks)
    "building_cov_pp": 2.0,
    "seg_area_rel": 0.02,          # v2 segmented area ≥ 0.98 × v1
    "unclassified_pp": 2.0,        # v2 unclassified share ≤ v1 + 2 pp
    "segments_rel_min": 0.5,       # sanity — v2 segments ≥ half of v1's
    "raster_hole_frac": 0.005,     # zero segment_type inside parcel union
    "outline_z_min_frac": 0.5,     # ≥ 50 % of parcels with geometry carry outline_z
}


# Operator hints per fatal check — appended to ``Report.summary()`` (and so
# to the merged fleet log / ``/process.txt?q=v2verify``) so a FAIL row says
# *where to look* without re-deriving the pipeline.  Keep each ≤ ~90 chars.
HINTS = {
    "unclassified_le_v1": "few huge 'unclassified' objs = cross-tile anchor-weak merge demotion → "
                          "grep peer log 'Anchor-weak' (v2 skips those merges since 2026-09-21)",
    "parcel_segmentation_coverage_pct_ge_v1": "raw AND ≥300 m² coverage both dropped >8 pp → a tile was not "
                                              "segmented; check data_quality.tiles (smaller gaps are _drift warnings)",
    "segment_type_no_holes": "segment_type==0 inside parcel union → unsegmented tile / BEV read hole; "
                             "grep 'gpkg_full' + 'deferred' for that KG",
    "parcels_vs_v1": "cadastre changed between runs (±1–2 is normal for old v1) — compare parcels_vs_cadastre",
    "segmented_area_ge_v1": "v2 covers less LiDAR area than v1 on the same grid → missing DTM tile / BEV outage",
    "lidar_tiles_ge_v1": "a DTM tile v1 had is missing now → BEV read failure, see bev_pause / Anchor",
    "bbox_covers_v1": "v2 AOI shrunk vs v1 → KG split/definition changed; compare kg_splitter output",
    "grid25_dims": "grid25 raster does not match bbox → v21_products grid anchor bug (docs/v2.1-product-spec.md)",
    "light_gpkg_integrity": "sqlite integrity_check failed → disk full / interrupted write on peer",
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
        hints = []
        for e in self["errors"]:
            h = HINTS.get(e.split(":", 1)[0])
            if h and h not in hints:
                hints.append(h)
        if hints:
            s += " | hint: " + " / ".join(hints[:2])
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


SEG_COV_MIN_AREA_SQM = 300.0   # sliver parcels below this are excluded from the fatal coverage gate
SEG_COV_MIN_PARCELS = 20       # … need this many big parcels for the filtered figure to mean anything


def _seg_coverage_min_area(doc, min_area: float = SEG_COV_MIN_AREA_SQM) -> float | None:
    """``parcel_segmentation_coverage_pct`` recomputed over parcels ≥ *min_area*
    (needs ``parcels.details``); None when unavailable / too few parcels."""
    det = _g(doc, "parcels", "details", default=None)
    if not isinstance(det, list):
        return None
    big = [p for p in det if isinstance(p, dict) and (_num(p.get("area_sqm"), 0) or 0) >= min_area]
    if len(big) < SEG_COV_MIN_PARCELS:
        return None
    return 100.0 * sum(1 for p in big if p.get("area_summary")) / len(big)


def _lost_parcels_diag(v2, v1, min_area: float = SEG_COV_MIN_AREA_SQM) -> str:
    """Forensic tail for a failed ``parcel_segmentation_coverage_pct_ge_v1``:
    which parcels ≥ *min_area* had an ``area_summary`` in v1 but none in v2
    (and the reverse), summarised by v1 dominant type, area and whether the
    v2 doc has *any* segmentation on the parcel's tile.  The failing v2 JSON
    is not uploaded, so this line is the only evidence that survives — it
    tells apart "v2 class-set drops a type v1 had" (fix: map that type) from
    "a tile came back empty" (fix: BEV/tile) without reprocessing."""
    try:
        d1 = _g(v1, "parcels", "details", default=None) or []
        d2 = _g(v2, "parcels", "details", default=None) or []
        if not isinstance(d1, list) or not isinstance(d2, list):
            return ""
        def _key(p):
            return p.get("parcel_id") or p.get("gst_nr") or p.get("id") or p.get("kg_gst")
        m2 = {_key(p): p for p in d2 if isinstance(p, dict) and _key(p) is not None}
        lost, gained = [], 0
        for p in d1:
            if not isinstance(p, dict) or (_num(p.get("area_sqm"), 0) or 0) < min_area:
                continue
            q = m2.get(_key(p))
            if q is None:
                continue
            if p.get("area_summary") and not q.get("area_summary"):
                lost.append((p, q))
            elif q.get("area_summary") and not p.get("area_summary"):
                gained += 1
        if not lost:
            return f" | lost=0 gained={gained} (key mismatch? v1={len(d1)} v2={len(d2)} details)"
        from collections import Counter
        types = Counter()
        for p, _q in lost:
            as_ = p.get("area_summary") or {}
            types[p.get("dominant_type") or (next(iter(as_)) if as_ else "?")] += 1
        areas = sorted((_num(p.get("area_sqm"), 0) or 0) for p, _ in lost)
        med = areas[len(areas) // 2]
        # v2-side signal: does the lost parcel still carry elevation (tile
        # was read) and what does the light seg raster say?
        elev = sum(1 for _p, q in lost if q.get("elevation_m") is not None)
        types_s = ",".join(f"{t}={n}" for t, n in types.most_common(4))
        ex = ",".join(str(_key(p)) for p, _ in lost[:3])
        return (f" | lost={len(lost)} gained={gained} v1_types[{types_s}] "
                f"area_med={med:.0f}m² max={areas[-1]:.0f}m² v2_has_elev={elev}/{len(lost)} e.g. {ex}")
    except Exception as exc:  # noqa: BLE001
        return f" | lost-diag failed: {exc}"


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


def _nodata_tile_union_3035(doc):
    """Union (EPSG:3035) of tile bboxes that have **no DTM at all**
    (``dtm=False`` / ``valid_pixels=0``): open water (Bodensee, Rhine) or
    outside BEV ALS coverage.  The cadastral parcel union routinely
    extends over such areas (91109 Gaissau: tiles 1/11/12 on Lake
    Constance, ``parcel_segmentation_coverage_pct=80.4``), so a hole
    there is permanent and not a stitching defect.  None if no such tile."""
    tiles = _g(doc, "data_quality", "tiles", default=[]) or []
    boxes = []
    for t in tiles:
        if not isinstance(t, dict) or t.get("dtm") or int(t.get("valid_pixels") or 0) >= 100:
            continue
        bb = t.get("bbox_wgs")
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            boxes.append([float(x) for x in bb])
    if not boxes:
        return None
    try:
        from shapely.geometry import box
        from shapely.ops import transform as _shp_transform, unary_union
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform
        return unary_union([_shp_transform(tr, box(*b)) for b in boxes])
    except Exception as e:  # noqa: BLE001
        log.debug("nodata tile union skipped: %s", e)
        return None


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
    check_v23_top_ranking(v2, rep)

    cov = _g(v2, "coverage", default={}) or {}
    cov1 = (_g(v1, "coverage", default={}) or {}) if v1 else {}
    for k in ("parcel_elevation_coverage_pct", "parcel_segmentation_coverage_pct"):
        v = _num(cov.get(k))
        v1v = _num(cov1.get(k)) if v1 else None
        # a baseline that had 0 / no coverage figure cannot be regressed;
        # a parcel-less block (57210-southeast: alpine quadrant whose huge
        # parcels all have their centroid in a sibling block, 0 parcels /
        # 0 buildings) has nothing to cover — 0.0 is the correct figure.
        rep.check(f"coverage.{k}", v is not None and v > 0,
                  f"{k}={v}" + (" (no parcels in KG)" if pc == 0 else ""),
                  fatal=pc > 0 and not (v1 is not None and not v1v))

    if v1 is None:
        return
    # ---- improvement / non-regression vs v1 --------------------------------
    pc1 = int(_num(_g(v1, "parcels", "count"), 0) or 0)
    if pc1 > 0:
        _d = abs(pc - pc1)
        rep.check("parcels_vs_v1", _d <= max(TOL["parcel_count_abs"], TOL["parcel_count_rel"] * pc1),
                  f"v2={pc} v1={pc1}")
    cov1 = _g(v1, "coverage", default={}) or {}
    for k, tol in (("parcel_elevation_coverage_pct", TOL["coverage_pp"]),
                   ("parcel_segmentation_coverage_pct", TOL["seg_coverage_pp"]),
                   ("building_height_coverage_pct", TOL["building_cov_pp"])):
        a, b = _num(cov.get(k)), _num(cov1.get(k))
        if a is None or b is None:
            continue
        if k == "parcel_segmentation_coverage_pct":
            # Small KGs (19423: 164 parcels, 19732: 38) flip a handful of
            # sliver parcels (70–250 m², smaller than the object covering
            # them) between runs — 1–6 parcels = 2.6–3.6 pp, past the 2.5 pp
            # gate although nothing is wrong (2026-09-21, 5 parents struck
            # after 0d4f717).  Judge the fatal gate on parcels ≥ 300 m² where
            # both docs carry details; the raw figure stays a drift warning.
            # The ≥300 m² figure has a small denominator on medium KGs (50–200
            # parcels): losing 3–6 parcels = 3–7 pp although the raw figure
            # moved 0–2 pp.  Fatal only when *both* figures drop past
            # seg_coverage_pp_fatal — that is what a lost tile looks like;
            # anything smaller is reported as non-fatal drift with the
            # lost-parcel forensic tail so the class-set gap stays visible.
            fatal_tol = TOL["seg_coverage_pp_fatal"]
            fa, fb = _seg_coverage_min_area(v2), _seg_coverage_min_area(v1)
            if fa is not None and fb is not None:
                _det = f"v2={fa:.1f} v1={fb:.1f} (parcels ≥{SEG_COV_MIN_AREA_SQM:.0f} m²; raw {a:.1f}/{b:.1f})"
                big_drop = fa < fb - fatal_tol and a < b - fatal_tol
                if fa < fb - tol or big_drop:
                    _det += _lost_parcels_diag(v2, v1)
                rep.check(f"{k}_ge_v1", not big_drop, _det)
                if not big_drop and (fa < fb - tol or a < b - TOL["seg_coverage_pp_warn"]):
                    rep.check(f"{k}_drift", False,
                              f"v2={a:.1f} v1={b:.1f} ({a - b:+.1f} pp raw; "
                              f"{fa - fb:+.1f} pp ≥{SEG_COV_MIN_AREA_SQM:.0f} m²)", fatal=False)
                continue
            # no per-parcel details on one side: judge the raw figure alone
            rep.check(f"{k}_ge_v1", a >= b - fatal_tol, f"v2={a:.1f} v1={b:.1f}")
            if b - fatal_tol <= a < b - TOL["seg_coverage_pp_warn"]:
                rep.check(f"{k}_drift", False, f"v2={a:.1f} v1={b:.1f} ({a - b:+.1f} pp)", fatal=False)
            continue
        if k == "building_height_coverage_pct":
            # Percentage over a *small* denominator: 40114-northeast-1 has
            # 4 buildings (75.0 vs 100.0 = one building), 63323-south 17
            # (70.6 vs 75.0 = one), 90014-northwest-2 ~22 (76.7 vs 81.8 =
            # one) — all struck on a 2 pp gate on 2026-09-23 although a
            # single edge building flipping its nDSM sample is run-to-run
            # noise, not a lost tile.  Fatal only when ≥3 buildings (or
            # 5 %) lost height; the pp figure stays a drift warning.
            nb = int(_num(_g(v2, "building_footprints", "count"), 0) or 0)
            nb1 = int(_num(_g(v1, "building_footprints", "count"), 0) or 0)
            n = max(nb, nb1, 1)
            lost = max(0.0, (b - a) * n / 100.0)
            fatal_lost = max(3.0, 0.05 * n)
            rep.check(f"{k}_ge_v1", lost <= fatal_lost,
                      f"v2={a:.1f} v1={b:.1f} (≈{lost:.1f} of {n} buildings lost height; fatal >{fatal_lost:.0f})")
            if lost <= fatal_lost and a < b - tol:
                rep.check(f"{k}_drift", False, f"v2={a:.1f} v1={b:.1f} ({a - b:+.1f} pp, ≈{lost:.0f} buildings)",
                          fatal=False)
            continue
        rep.check(f"{k}_ge_v1", a >= b - tol, f"v2={a:.1f} v1={b:.1f}")
    # Tile-grid comparisons.  ``total_segmented_area_sqm`` is the SUM of
    # per-tile valid px over the overlapping 1.5 km grid and ``n_tiles``
    # depends on how the bbox happened to fall on the grid when the v1 ran
    # (19185: v1 4 tiles / 9.06 km², v2 2 tiles / 4.53 km² for a bbox that
    # v2 covers *more* of).  Neither is comparable across a grid change, so
    # a differing grid is judged on AOI coverage: fatal only when the v2
    # bbox fails to cover the v1 bbox (a shrunk AOI); otherwise a warning.
    sa, sb = _num(cov.get("total_segmented_area_sqm")), _num(cov1.get("total_segmented_area_sqm"))
    nt2 = int(_num(cov.get("n_tiles"), 0) or 0)
    nt1 = int(_num(cov1.get("n_tiles"), 0) or 0)
    same_grid = nt1 == 0 or nt2 == 0 or nt1 == nt2
    bb2, bb1 = _bbox_3035(v2), _bbox_3035(v1)
    aoi_covered = True
    if bb2 and bb1:
        _sl = 25.0
        aoi_covered = (bb2[0] <= bb1[0] + _sl and bb2[1] <= bb1[1] + _sl
                       and bb2[2] >= bb1[2] - _sl and bb2[3] >= bb1[3] - _sl)
        rep.check("bbox_covers_v1", aoi_covered,
                  f"v2 x[{bb2[0]:.0f},{bb2[2]:.0f}] y[{bb2[1]:.0f},{bb2[3]:.0f}] "
                  f"v1 x[{bb1[0]:.0f},{bb1[2]:.0f}] y[{bb1[1]:.0f},{bb1[3]:.0f}]")
    if sa is not None and sb and sb > 0:
        rep.check("segmented_area_ge_v1", sa >= (1 - TOL["seg_area_rel"]) * sb,
                  f"v2={sa:.0f} v1={sb:.0f} m²" + ("" if same_grid else f" (grid {nt2} vs {nt1} tiles)"),
                  fatal=same_grid or not aoi_covered)
    rep.check("lidar_tiles_ge_v1", lidar >= l1, f"v2={lidar} v1={l1} lidar tiles"
              + ("" if same_grid else f" (grid {nt2} vs {nt1} tiles)"),
              fatal=same_grid or not aoi_covered)
    n1 = int(_num(_g(v1, "landscape", "n_segments"), 0) or 0)
    if n1 > 0:
        rep.check("segments_vs_v1", n_seg >= TOL["segments_rel_min"] * n1,
                  f"v2={n_seg} v1={n1}", fatal=False)
    u2, u1 = _unclassified_pct(v2), _unclassified_pct(v1)
    # Object count + mean size of the v2 unclassified share: "n=2 avg=117k m²"
    # is one demoted cross-tile blob (pipeline), "n=800 avg=60 m²" is the
    # classifier genuinely unsure (model/threshold) — very different fixes.
    _ua = (_g(v2, "area_summary", "unclassified", default={}) or {})
    _un = int(_num(_ua.get("n_objects"), 0) or 0)
    _ushape = (f" (n={_un} avg={_num(_ua.get('area_sqm'), 0.0) / _un:.0f}m²)" if _un else "")
    rep.check("unclassified_le_v1", u2 <= u1 + TOL["unclassified_pp"],
              f"v2={u2:.1f}%{_ushape} v1={u1:.1f}%")
    nb2 = int(_num(_g(v2, "new_buildings", "count"), 0) or 0)
    nb1 = int(_num(_g(v1, "new_buildings", "count"), 0) or 0)
    rep["delta"] = {"n_segments": n_seg - n1, "unclassified_pp": round(u2 - u1, 2),
                    "new_buildings": nb2 - nb1,
                    "seg_area_rel": round(sa / sb, 4) if sa is not None and sb else None}


def _bbox_3035(v2: dict):
    bb = (v2 or {}).get("bbox") or {}
    try:
        from pyproj import Transformer
        t = Transformer.from_crs(4326, 3035, always_xy=True)
        xs, ys = zip(*[t.transform(x, y) for x, y in (
            (bb["min_lon"], bb["min_lat"]), (bb["max_lon"], bb["min_lat"]),
            (bb["min_lon"], bb["max_lat"]), (bb["max_lon"], bb["max_lat"]))])
        return min(xs), min(ys), max(xs), max(ys)
    except Exception:  # noqa: BLE001
        return None


def _lidar_extent_3035(v2: dict):
    """EPSG:3035 envelope of the tiles that actually carry a DTM, or None.

    Border KGs (80110-southeast-2, Tirol/Italy) have whole tile rows outside
    Austria: no BEV data, nothing stitched, so the 25 m grid legitimately
    starts a tile row inside the doc bbox."""
    tiles = _g(v2, "data_quality", "tiles", default=[]) or []
    bbs = [t.get("bbox_wgs") for t in tiles
           if isinstance(t, dict) and t.get("dtm") and not t.get("outside_austria")
           and int(t.get("valid_pixels") or 0) >= 100 and t.get("bbox_wgs")]
    if not bbs or len(bbs) == len([t for t in tiles if isinstance(t, dict) and t.get("bbox_wgs")]):
        return None
    try:
        from pyproj import Transformer
        t = Transformer.from_crs(4326, 3035, always_xy=True)
        xs, ys = [], []
        for w, s, e, n in bbs:
            for x, y in ((w, s), (e, s), (w, n), (e, n)):
                px, py = t.transform(x, y)
                xs.append(px); ys.append(py)
        return min(xs), min(ys), max(xs), max(ys)
    except Exception:  # noqa: BLE001
        return None


def check_v21_document(v2: dict, rep: Report) -> None:
    """Product-2.1 JSON checks (JSON subset — also run on the primary)."""
    pv = str(v2.get("product_version") or "")
    # peers verify their own output (must be the current version); the primary
    # ingests whatever the fleet uploads during a rollout — an older readable
    # product line is a warning there, not a rejection.
    readable = {"2.1", "2.2", "2.3"}
    if pv == PRODUCT_VERSION:
        rep.check("product_version", True, f"product_version={pv!r}")
    else:
        rep.check("product_version", pv in readable, f"product_version={pv!r} (current {PRODUCT_VERSION})",
                  fatal=pv not in readable)
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
    _lx = _lidar_extent_3035(v2)
    if bb and _lx:
        # only the tiles with lidar can be gridded — judge against their
        # envelope (clipped to the doc bbox) instead of the whole bbox.
        bb = (max(bb[0], _lx[0]), max(bb[1], _lx[1]), min(bb[2], _lx[2]), min(bb[3], _lx[3]))
    if bb:
        # the grid is built from the (overlapping, bbox-overshooting) tile
        # rectangles snapped to 25 m, so it must COVER the bbox and may
        # exceed it by up to ~1 tile (1.6 km) per side — not "±1 cell".
        x0, y0 = d["x0"], d["y0"]
        x1, y1 = x0 + cols * cell, y0 - rows * cell
        covers = (x0 <= bb[0] + cell and y1 <= bb[1] + cell
                  and x1 >= bb[2] - cell and y0 >= bb[3] - cell)
        # 19565 (2026-09-21): 2×2 tiles for a 1.3×2.0 km bbox, east
        # overshoot 1759 m = one 1575 m tile + the 100 m read buffer + snap.
        slack = 1800.0
        sane = (bb[0] - x0 <= slack and x1 - bb[2] <= slack
                and bb[1] - y1 <= slack and y0 - bb[3] <= slack)
        _gd = (f"grid {cols}x{rows}@{cell:.0f}m x[{x0:.0f},{x1:.0f}] y[{y1:.0f},{y0:.0f}] "
               f"vs bbox x[{bb[0]:.0f},{bb[2]:.0f}] y[{bb[1]:.0f},{bb[3]:.0f}]")
        # Fatal only when the grid fails to COVER the AOI. Overshoot past
        # ``slack`` is a size cost, not a correctness defect: 09048
        # (2026-09-21, 958×719 m bbox straddling a tile row boundary) got
        # a 2-tile-tall grid, 2034 m above the bbox, and struck out on it.
        rep.check("grid25_dims", covers, _gd)
        if covers and not sane:
            rep.check("grid25_overshoot", False, _gd + f" (> {slack:.0f} m past bbox)", fatal=False)
    n_valid = int(np.isfinite(d["elev"]).sum())
    frac = n_valid / max(cols * rows, 1)
    rep.check("grid25_finite", frac >= 0.4, f"{n_valid}/{cols * rows} cells valid ({100 * frac:.0f}%)")
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



def _gpkg_point_xy(blob: bytes):
    """(x, y) from a GeoPackage binary Point blob (no spatialite needed)."""
    import struct
    if not blob or blob[:2] != b"GP":
        return None
    flags = blob[3]
    env = (flags >> 1) & 0x07
    env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(env)
    if env_len is None:
        return None
    off = 8 + env_len
    bo = "<" if blob[off] == 1 else ">"
    gtype = struct.unpack(bo + "I", blob[off + 1:off + 5])[0]
    if gtype % 1000 != 1:
        return None
    x, y = struct.unpack(bo + "dd", blob[off + 5:off + 21])
    return float(x), float(y)

def check_v23_top_ranking(v2: dict, rep: Report) -> None:
    """Product-2.3: top-N lists are robust-ranked and plausibility-filtered.

    Fatal on a peer producing the current version (a 2.3 doc whose
    ``top_10_trees`` still carries an 80 m tree is the bug this version
    exists to fix); a warning for readable older lines on the primary.
    """
    pv = str(v2.get("product_version") or "")
    fatal = pv == PRODUCT_VERSION
    if not fatal and pv in ("2.1", "2.2"):
        return
    objs = v2.get("top_10_objects") or []
    trees = v2.get("top_10_trees") or []
    mm = v2.get("top_manmade_objects")
    rep.check("top_manmade_present", isinstance(mm, list),
              "top_manmade_objects missing", fatal=fatal)
    rows = list(objs) + list(trees) + list(mm or [])
    missing = [r for r in rows if not isinstance(r.get("height_robust_m"), (int, float))]
    rep.check("top_height_robust", not missing,
              f"{len(missing)}/{len(rows)} top rows lack height_robust_m", fatal=fatal)
    # robust ≤ max, lists non-increasing in robust height
    bad = [r for r in rows if isinstance(r.get("height_robust_m"), (int, float))
           and isinstance(r.get("height_max_m"), (int, float))
           and r["height_robust_m"] > r["height_max_m"] + 0.011]
    rep.check("top_robust_le_max", not bad, f"{len(bad)} rows robust > max", fatal=fatal)
    for name, lst in (("top_10_objects", objs), ("top_10_trees", trees)):
        hs = [r.get("height_robust_m") for r in lst if isinstance(r.get("height_robust_m"), (int, float))]
        rep.check(f"{name}_sorted", all(a >= b - 1e-6 for a, b in zip(hs, hs[1:])),
                  f"{name} not sorted by height_robust_m", fatal=fatal)
    try:
        import quality_flags as _qf
        T = _qf.THRESHOLDS
        spikes = [r for r in trees if isinstance(r.get("height_max_m"), (int, float))
                  and r["height_max_m"] >= T["tree_max_height_m"]["critical"]]
        spikes += [r for r in objs if r.get("type") in _qf.GROUND_TYPES
                   and isinstance(r.get("height_robust_m"), (int, float))
                   and r["height_robust_m"] >= T["flat_type_max_height_m"]["high"]]
        spikes += [r for r in objs if r.get("type") in ("water", "waterbody")
                   and isinstance(r.get("height_robust_m"), (int, float))
                   and r["height_robust_m"] >= T["water_max_height_m"]["high"]]
        rep.check("top_no_implausible", not spikes,
                  f"{len(spikes)} implausible rows in top lists "
                  f"(e.g. {spikes[0].get('type')} {spikes[0].get('height_max_m')} m)" if spikes else "",
                  fatal=fatal)
    except ImportError:
        pass


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
    # 2.2: apex grid anchor (pixel centres at x.5 on the integer-metre BEV
    # grid) — a fractional-origin regression would silently break tree_id
    # stability between product and live (FEEDBACK-5 §1).
    pts = []
    try:
        gcol = c.execute("SELECT column_name FROM gpkg_geometry_columns WHERE table_name='tree_apices'").fetchone()[0]
        for (blob,) in c.execute(f'SELECT "{gcol}" FROM "tree_apices" LIMIT 50'):
            xy = _gpkg_point_xy(blob)
            if xy:
                pts.append(xy)
    except Exception:  # noqa: BLE001
        pts = []
    if pts:
        off = [abs(((x - 0.5) % 1.0)) + abs(((y - 0.5) % 1.0)) for x, y in pts]
        bad = sum(1 for o in off if min(o, 2.0 - o) > 0.02)
        # fatal since 2026-09-20: KG 63330 shipped 50/50 apices at x.9 (restored
        # pre-2.2 tile checkpoints with a fractional origin) — tree_id unstable vs live.
        rep.check("apex_grid_anchor", bad == 0, f"{bad}/{len(pts)} apices off the x.5 grid")
    n_cr = int(_num(ts.get("n_crowns"), 0) or 0)
    if ts.get("crowns_layer") == "tree_crowns":
        try:
            m = c.execute('SELECT COUNT(*) FROM "tree_crowns"').fetchone()[0]
        except Exception as e:  # noqa: BLE001
            m = -1
            rep.check("crowns_layer", False, str(e)[:120], fatal=False)
        if m >= 0:
            rep.check("crowns_layer", abs(m - n_cr) <= 0.2 * max(n_cr, 1) and m <= n,
                      f"layer={m} json={n_cr} apices={n}", fatal=False)


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
        # a block whose centroid filter kept no parcels / no buildings has
        # nothing to write into those layers (47325-southeast: 1502 → 0
        # parcels) — required only when the JSON reports any.
        _pc0 = int(_num(_g(v2, "parcels", "count"), 0) or 0)
        _nb0 = int(_num(_g(v2, "building_footprints", "count"), 0) or 0)
        _optional = ({"parcels"} if _pc0 == 0 else set()) | ({"buildings"} if _nb0 == 0 else set())
        missing = [l for l in LIGHT_LAYERS_REQUIRED if l not in layers and l not in _optional]
        rep.check("light_gpkg_layers", not missing, "missing: " + ",".join(missing))
        _opt_missing = [l for l in _optional if l not in layers]
        if _opt_missing:
            rep.check("light_gpkg_layers_empty", False,
                      "absent (no features in KG): " + ",".join(sorted(_opt_missing)), fatal=False)
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
        _nd = _nodata_tile_union_3035(v2)
        _measure_union = union_geom_3035
        if _nd is not None and union_geom_3035 is not None and not union_geom_3035.is_empty:
            try:
                _measure_union = union_geom_3035.difference(_nd)
            except Exception:  # noqa: BLE001
                _measure_union = union_geom_3035
        nz, hole = _segment_type_nonzero(path, _measure_union)
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
            # The grid floor is exact only for a rectangular KG made of
            # full tiles (2×2 → 0.934, 2×3 → 0.924 fleet-wide).  For a
            # large irregular KG it is wrong in *both* directions:
            # ``total_segmented_area_sqm`` counts every valid DTM px in
            # the tile bboxes, while the stitched raster only keeps px of
            # *kept* objects — segments <50 % inside the parcel union are
            # dropped by the KG-mask filter and the centroid dedup drops
            # overlap duplicates.  A KG whose parcel union fills only part
            # of its 15-tile bbox therefore legitimately reads 0.75–0.83
            # (47106 → 0.827 vs floor 0.828, 57210-south → 0.745/0.780,
            # 2026-09-23: both struck, v2 products dropped for a healthy
            # raster).  Real stitching loss is what ``segment_type_no_holes``
            # measures *inside the union*, so whenever that check can run
            # the area ratio is only a drift warning.  Fatal only when no
            # union is available (legacy callers) or the raster is empty.
            _shape_aware = hole is not None
            # A parcel-less block has no union to measure holes in and no
            # KG-mask to explain the ratio; the raster only has to exist.
            _no_parcels = int(_num(_g(v2, "parcels", "count"), 0) or 0) == 0
            _ok = nz > 0 and (_shape_aware or _no_parcels or _r >= _floor)
            rep.check("segment_type_raster_area", _ok,
                      f"nonzero px={nz} json segmented m²={sa:.0f} ratio={_r:.3f} floor={_floor:.3f} tiles={_nt}"
                      + (" (hole check authoritative)" if _shape_aware
                         else " (no parcels in KG — ratio advisory)" if _no_parcels else ""))
            if _ok and _r < 0.95:
                rep.check("segment_type_raster_area_drift", False,
                          f"ratio={_r:.3f} (tile-overlap double count / KG-mask filter, {_nt} tiles)",
                          fatal=False)
        if hole is not None:
            # Holes are expected when the doc itself declares a gap (an
            # unsegmented or upstream-failed tile) — inherited from the v1
            # baseline / its full GPKG; fatal only for an *undeclared* hole.
            _l, _s, _ = _active_tiles(v2)
            _nup = int(_num(_g(v2, "data_quality", "n_upstream_failed_tiles"), 0) or 0)
            _declared_gap = (_s < _l) or _nup > 0
            rep.check("segment_type_no_holes", hole <= TOL["raster_hole_frac"],
                      f"{100*hole:.2f}% of parcel union has no segment class"
                      + (" (declared tile gap)" if _declared_gap else "")
                      + (" (no-DTM tiles excluded from union)" if _nd is not None else ""),
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
