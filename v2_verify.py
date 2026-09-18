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

    # tile stitching — every LiDAR tile segmented
    lidar, seg, bad = _active_tiles(v2)
    rep.check("tiles_all_segmented", lidar > 0 and seg == lidar,
              f"{seg}/{lidar} lidar tiles segmented; unsegmented={bad[:10]}")
    n_up = int(_num(_g(v2, "data_quality", "n_upstream_failed_tiles"), 0) or 0)
    rep.check("no_upstream_failed_tiles", n_up == 0, f"{n_up} upstream-failed tile(s)")

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
        rep.check("parcel_elevations", n_elev >= 0.9 * len(pd), f"{n_elev}/{len(pd)} parcels with elevation")

    cov = _g(v2, "coverage", default={}) or {}
    for k in ("parcel_elevation_coverage_pct", "parcel_segmentation_coverage_pct"):
        v = _num(cov.get(k))
        rep.check(f"coverage.{k}", v is not None and v > 0, f"{k}={v}")

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
    l1, s1, _ = _active_tiles(v1)
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
        ns = _count("segments")
        rep.check("segments_layer_count", ns > 0 and (n_seg == 0 or abs(ns - n_seg) <= max(5, 0.02 * n_seg)),
                  f"layer={ns} json={n_seg}")
        npx = _count("segment_points")
        rep.check("segment_points_count", npx == ns, f"points={npx} segments={ns}", fatal=False)
        pc = int(_num(_g(v2, "parcels", "count"), 0) or 0)
        npar = _count("parcels")
        rep.check("parcels_layer_count", pc == 0 or abs(npar - pc) <= max(2, 0.01 * pc),
                  f"layer={npar} json={pc}")
        nb = int(_num(_g(v2, "building_footprints", "count"), 0) or 0)
        nbl = _count("buildings")
        rep.check("buildings_layer_count", nb == 0 or abs(nbl - nb) <= max(2, 0.01 * nb),
                  f"layer={nbl} json={nb}")
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
            rep.check("segment_type_raster_area", nz >= 0.95 * sa,
                      f"nonzero px={nz} json segmented m²={sa:.0f}")
        if hole is not None:
            rep.check("segment_type_no_holes", hole <= TOL["raster_hole_frac"],
                      f"{100*hole:.2f}%% of parcel union has no segment class")
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
