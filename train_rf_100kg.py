#!/usr/bin/env python3
"""Train Random Forest on 100 random KGs using full cadastre ground truth.

For each KG:
1. Fetch bbox from cadastre API
2. Fetch all three cadastre layers:
   - parcels (with landuse_summary polygons)
   - building_footprints (cm-precision polygons)
   - landuse (sub-parcel landuse polygons)
3. Read LIDAR DTM/DSM
4. Read ortho
5. Fetch Copernicus (NDVI, WorldCover, SAR, harmonics)
6. Run watershed segmentation
7. Match segments to cadastre ground truth via spatial overlap
8. Accumulate features + labels
9. Train RF on all accumulated data

Run: python3 train_rf_100kg.py 2>&1 | tee /tmp/rf_train_100kg.log
"""
import json
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import requests
from shapely.geometry import box, shape as shapely_shape, Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    stream=sys.stdout,
)
log = logging.getLogger('rf_train')

# Suppress noisy loggers
for name in ['rasterio', 'urllib3', 'botocore', 'PIL', 'fiona']:
    logging.getLogger(name).setLevel(logging.WARNING)

CADASTRE_BASE = "https://cadastre-process-api.exe.xyz/api/v1"
RESULTS_DIR = Path("/tmp/rf_train_100kg")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# WGS84 <-> EPSG:3035 transformers
_tx_to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
_tx_to_wgs = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


def transform_to_3035(geom):
    return shapely_transform(_tx_to_3035.transform, geom)


def transform_to_wgs(geom):
    return shapely_transform(_tx_to_wgs.transform, geom)


def get_random_kgs(n: int = 100) -> list[dict]:
    """Fetch KGs that overlap LIDAR coverage and pick n random ones.

    Strategy: query KGs from multiple districts across Austria,
    filter to those with building_footprints (indicates processed KG),
    then random sample.
    """
    # Get all states' districts
    states = [
        "Steiermark", "Kärnten", "Niederösterreich", "Oberösterreich",
        "Salzburg", "Tirol", "Vorarlberg", "Burgenland", "Wien",
    ]
    all_kgs = []
    for state in states:
        try:
            resp = requests.get(
                f"{CADASTRE_BASE}/search/district",
                params={"state": state},
                timeout=30,
            )
            resp.raise_for_status()
            districts = resp.json().get("data", [])
            for d in districts:
                code = d.get("district_code")
                if not code:
                    continue
                try:
                    resp2 = requests.get(
                        f"{CADASTRE_BASE}/search/kg",
                        params={"district": d["district_name"], "limit": 200},
                        timeout=30,
                    )
                    resp2.raise_for_status()
                    kgs = resp2.json().get("data", [])
                    for kg in kgs:
                        if kg.get("kg_code") and kg.get("parcel_count", 0) > 10:
                            all_kgs.append(kg)
                except Exception as e:
                    log.warning("Failed to fetch KGs for district %s: %s", code, e)
        except Exception as e:
            log.warning("Failed to fetch districts for %s: %s", state, e)

    log.info("Found %d KGs total across Austria", len(all_kgs))

    # Filter to KGs that have bbox within LIDAR coverage
    # LIDAR covers roughly lon 9.06-17.78, lat 45.98-49.12 (all of Austria)
    # Further filter: prefer KGs with buildings (more diverse training data)
    with_buildings = [kg for kg in all_kgs if kg.get("building_count", 0) > 5]
    log.info("%d KGs have >5 buildings", len(with_buildings))

    # Random sample
    random.seed(42)  # reproducible
    if len(with_buildings) >= n:
        selected = random.sample(with_buildings, n)
    else:
        selected = with_buildings
        if len(selected) < n and all_kgs:
            extra = random.sample(
                [k for k in all_kgs if k not in selected],
                min(n - len(selected), len(all_kgs) - len(selected))
            )
            selected.extend(extra)

    log.info("Selected %d KGs for training", len(selected))
    return selected


def fetch_cadastre_layers(kg_code: str) -> dict:
    """Fetch all three cadastre layers for a KG.

    Returns dict with:
      - parcels: list of (Polygon_3035, landuse_code, landuse_summary)
      - building_footprints: list of Polygon_3035
      - landuse: list of (Polygon_3035, landuse_code, landuse_abbr)
    """
    result = {"parcels": [], "building_footprints": [], "landuse": []}

    try:
        resp = requests.get(
            f"{CADASTRE_BASE}/export/geojson",
            params={
                "kg": kg_code,
                "layers": "parcels,building_footprints,landuse",
                "include_geometry": "true",
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("KG %s: cadastre fetch failed: %s", kg_code, e)
        return result

    # Parse parcels (polygons with landuse_summary)
    parcels_fc = data.get("parcels", {}).get("features", [])
    for f in parcels_fc:
        try:
            geom = shapely_shape(f["geometry"])
            if geom.is_empty or not isinstance(geom, (Polygon, MultiPolygon)):
                continue
            geom_3035 = transform_to_3035(geom)
            props = f.get("properties", {})
            lu_summary = props.get("landuse_summary", {})
            # Extract dominant landuse code from summary
            dominant_code = None
            if lu_summary:
                # Format: {"Wald - W": 5, "Baufläche (befestigt) - B(bf)": 2}
                # We need the numeric code. Parse from landuse_on_parcel or infer.
                dominant_code = _extract_dominant_code(lu_summary)
            result["parcels"].append({
                "geometry": geom_3035,
                "landuse_code": dominant_code,
                "landuse_summary": lu_summary,
                "area_sqm": props.get("area_sqm", 0),
                "parcel_id": props.get("parcel_id", ""),
            })
        except Exception:
            continue

    # Parse building footprints (cm-precision polygons)
    bfp_fc = data.get("building_footprints", {}).get("features", [])
    for f in bfp_fc:
        try:
            geom = shapely_shape(f["geometry"])
            if geom.is_empty or not isinstance(geom, (Polygon, MultiPolygon)):
                continue
            geom_3035 = transform_to_3035(geom)
            result["building_footprints"].append(geom_3035)
        except Exception:
            continue

    # Parse landuse polygons (sub-parcel landuse areas)
    lu_fc = data.get("landuse", {}).get("features", [])
    for f in lu_fc:
        try:
            geom = shapely_shape(f["geometry"])
            if geom.is_empty:
                continue
            props = f.get("properties", {})
            code_str = props.get("landuse_code", "")
            abbr = props.get("landuse_abbr", "")
            try:
                code = int(code_str)
            except (ValueError, TypeError):
                code = None
            # Landuse may be Points (centroids) if polygons not available
            if isinstance(geom, (Polygon, MultiPolygon)):
                geom_3035 = transform_to_3035(geom)
                result["landuse"].append({
                    "geometry": geom_3035,
                    "code": code,
                    "abbr": abbr,
                })
            else:
                # Point geometry — still useful for labelling via nearest
                geom_3035 = transform_to_3035(geom)
                result["landuse"].append({
                    "geometry": geom_3035,
                    "code": code,
                    "abbr": abbr,
                    "is_point": True,
                })
        except Exception:
            continue

    log.info("KG %s: %d parcels, %d building_footprints, %d landuse features",
             kg_code, len(result["parcels"]), len(result["building_footprints"]),
             len(result["landuse"]))
    return result


# Mapping from landuse description to numeric code
_LANDUSE_DESC_TO_CODE = {
    "B": 42, "B(bf)": 41, "B(bg)": 40,
    "V": 48, "W": 56, "LN": 52, "LN(W)": 52, "LN(Hu)": 53,
    "A": 51, "WG": 63, "GA": 64, "OG": 65, "Alpe": 54,
    "GW": 70, "Öd": 80, "Fe": 81,
    # Longer descriptions
    "Wald": 56, "Acker": 51, "Wiese": 52, "Weide": 55,
    "Obstgarten": 65, "Weingarten": 63, "Gewässer": 70,
}


def _extract_dominant_code(lu_summary: dict) -> int | None:
    """Extract dominant landuse code from landuse_summary dict.

    Keys are like "Wald - W" or "Baufläche (befestigt) - B(bf)".
    Value is count of NS symbols.
    """
    if not lu_summary:
        return None
    # Find highest-count entry
    best_key = max(lu_summary, key=lu_summary.get)
    # Extract abbreviation after " - "
    if " - " in best_key:
        abbr = best_key.split(" - ")[-1]
        if abbr in _LANDUSE_DESC_TO_CODE:
            return _LANDUSE_DESC_TO_CODE[abbr]
    # Try matching description words
    for word, code in _LANDUSE_DESC_TO_CODE.items():
        if word in best_key:
            return code
    return None


def match_segment_to_cadastre(
    feat: dict,
    cadastre_data: dict,
    transform,
) -> tuple[int | None, str]:
    """Match a segment to cadastre ground truth.

    Priority:
    1. Building footprint overlap → code 42 (roof)
    2. Landuse polygon overlap → use landuse code
    3. Parcel polygon overlap → use dominant parcel landuse code

    Returns (landuse_code, source) where source is
    'building_footprint', 'landuse_polygon', 'parcel', or None.
    """
    ce = feat.get("centroid_e", 0)
    cn = feat.get("centroid_n", 0)
    if ce == 0 and cn == 0:
        return None, ""

    from shapely.geometry import Point
    pt = Point(ce, cn)
    seg_area = feat.get("area", 0)

    # Build approximate segment polygon from bbox if available
    seg_bbox = feat.get("bbox")
    if seg_bbox and len(seg_bbox) == 4:
        seg_poly = box(seg_bbox[0], seg_bbox[1], seg_bbox[2], seg_bbox[3])
    else:
        # Use a small buffer around centroid
        r = max(1.0, (seg_area / 3.14159) ** 0.5) if seg_area > 0 else 5.0
        seg_poly = pt.buffer(r)

    # 1. Check building footprints (cm precision — best ground truth)
    for bfp in cadastre_data["building_footprints"]:
        try:
            if bfp.intersects(seg_poly):
                overlap = bfp.intersection(seg_poly).area
                if overlap > 0.3 * seg_poly.area or overlap > 0.3 * bfp.area:
                    return 42, "building_footprint"
        except Exception:
            continue

    # 2. Check landuse polygons (sub-parcel level)
    best_lu_code = None
    best_lu_overlap = 0
    for lu in cadastre_data["landuse"]:
        code = lu.get("code")
        if code is None:
            continue
        geom = lu["geometry"]
        if lu.get("is_point"):
            # Point — check distance
            try:
                d = pt.distance(geom)
                if d < 20:  # within 20m
                    if best_lu_code is None:
                        best_lu_code = code
            except Exception:
                continue
        else:
            try:
                if geom.intersects(seg_poly):
                    overlap = geom.intersection(seg_poly).area
                    if overlap > best_lu_overlap:
                        best_lu_overlap = overlap
                        best_lu_code = code
            except Exception:
                continue

    if best_lu_code is not None and best_lu_overlap > 0.2 * seg_poly.area:
        return best_lu_code, "landuse_polygon"

    # 3. Fall back to parcel polygon
    best_parcel_code = None
    best_parcel_overlap = 0
    for p in cadastre_data["parcels"]:
        code = p.get("landuse_code")
        if code is None:
            continue
        try:
            geom = p["geometry"]
            if geom.intersects(seg_poly):
                overlap = geom.intersection(seg_poly).area
                if overlap > best_parcel_overlap:
                    best_parcel_overlap = overlap
                    best_parcel_code = code
        except Exception:
            continue

    if best_parcel_code is not None:
        return best_parcel_code, "parcel"

    # 4. If landuse point was close enough
    if best_lu_code is not None:
        return best_lu_code, "landuse_point"

    return None, ""


def process_one_kg(
    kg: dict,
    include_copernicus: bool = True,
) -> tuple[list[dict], list[str], dict]:
    """Process one KG: segment + match to cadastre.

    Returns (features, labels, stats).
    """
    import raster_io
    import tile_index as ti
    import object_segmentation as oc
    import learned_classifier as lc

    kg_code = kg["kg_code"]
    stats = {"kg_code": kg_code, "kg_name": kg.get("kg_name", "")}

    # Get KG bbox — bbox is nested: kg["bbox"]["min_lon"] etc.
    kg_bbox = kg.get("bbox", {})
    if isinstance(kg_bbox, dict) and all(k in kg_bbox for k in ["min_lon", "min_lat", "max_lon", "max_lat"]):
        west, south = kg_bbox["min_lon"], kg_bbox["min_lat"]
        east, north = kg_bbox["max_lon"], kg_bbox["max_lat"]
    else:
        # Fetch bbox from cadastre
        try:
            resp = requests.get(
                f"{CADASTRE_BASE}/search/kg",
                params={"code": kg_code},
                timeout=30,
            )
            resp.raise_for_status()
            kgs = resp.json().get("data", [])
            if not kgs:
                stats["error"] = "KG not found"
                return [], [], stats
            k = kgs[0]
            bb = k.get("bbox", {})
            west, south = bb["min_lon"], bb["min_lat"]
            east, north = bb["max_lon"], bb["max_lat"]
        except Exception as e:
            stats["error"] = f"bbox fetch: {e}"
            return [], [], stats

    # Limit KG size — skip very large ones (>5km in any direction)
    dx_km = (east - west) * 111 * np.cos(np.radians((south + north) / 2))
    dy_km = (north - south) * 111
    if dx_km > 5 or dy_km > 5:
        # Take center 2km x 2km
        cx, cy = (west + east) / 2, (south + north) / 2
        half = 0.01  # ~1km
        west, south, east, north = cx - half, cy - half, cx + half, cy + half
        log.info("KG %s: large (%.1f x %.1f km), cropping to center 2km",
                 kg_code, dx_km, dy_km)

    stats["bbox"] = [west, south, east, north]
    geom_wgs = box(west, south, east, north)
    geom_3035 = transform_to_3035(geom_wgs)

    # 1. Fetch cadastre layers
    t0 = time.time()
    cadastre_data = fetch_cadastre_layers(kg_code)
    stats["cadastre_time"] = round(time.time() - t0, 1)
    stats["n_parcels"] = len(cadastre_data["parcels"])
    stats["n_footprints"] = len(cadastre_data["building_footprints"])
    stats["n_landuse"] = len(cadastre_data["landuse"])

    if stats["n_parcels"] == 0 and stats["n_footprints"] == 0:
        stats["error"] = "no cadastre data"
        return [], [], stats

    # 2. Read LIDAR
    t0 = time.time()
    try:
        data = raster_io.read_dtm_dsm(geom_3035, ti.DEFAULT_DATASET)
    except Exception as e:
        stats["error"] = f"LIDAR: {e}"
        return [], [], stats
    stats["lidar_time"] = round(time.time() - t0, 1)
    stats["shape"] = list(data["shape"])

    # 3. Read ortho
    t0 = time.time()
    spectral = None
    try:
        import ortho_io
        rgb, nir = ortho_io.read_ortho_for_als(data)
        spectral = ortho_io.compute_spectral_indices(rgb, nir=nir)
        if rgb is not None:
            spectral["red"] = rgb[0].astype(np.float32)
            spectral["green"] = rgb[1].astype(np.float32)
            spectral["blue"] = rgb[2].astype(np.float32)
        if nir is not None:
            spectral["nir"] = nir.astype(np.float32)
    except Exception as e:
        log.warning("KG %s: ortho failed: %s", kg_code, e)
    stats["ortho_time"] = round(time.time() - t0, 1)
    stats["has_ortho"] = spectral is not None

    # 4. Copernicus data (with per-layer timeout via threading)
    copernicus_data = None
    if include_copernicus:
        t0 = time.time()
        try:
            import copernicus
            import concurrent.futures
            bbox_dict = {"west": west, "south": south, "east": east, "north": north}
            cop = {}

            def _fetch_ndvi():
                d = copernicus.get_ndvi_composite(bbox_dict)
                return {"ndvi": d["ndvi"], "transform": d["transform"], "crs": d["crs"]}

            def _fetch_landcover():
                return copernicus.get_land_cover(bbox_dict)

            def _fetch_sar():
                d = copernicus.get_sar_backscatter(bbox_dict, "2023-06-01", "2023-09-30")
                return {"vv": d["vv"], "vh": d["vh"], "sar_transform": d["transform"], "sar_crs": d["crs"]}

            def _fetch_harmonics():
                import ndvi_harmonics
                return ndvi_harmonics.get_harmonic_features(bbox_dict)

            # Fetch NDVI, landcover, SAR with 3-min timeout each
            # Harmonics with 4-min timeout (it often needs batch jobs)
            COP_TIMEOUT = 180  # 3 min per layer
            HARM_TIMEOUT = 240  # 4 min for harmonics

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
                for name, func, timeout in [
                    ("ndvi", _fetch_ndvi, COP_TIMEOUT),
                    ("landcover", _fetch_landcover, COP_TIMEOUT),
                    ("sar", _fetch_sar, COP_TIMEOUT),
                    ("harmonics", _fetch_harmonics, HARM_TIMEOUT),
                ]:
                    try:
                        fut = exe.submit(func)
                        result = fut.result(timeout=timeout)
                        if result is not None:
                            if name == "ndvi":
                                cop.update(result)
                            elif name == "landcover":
                                cop["landcover"] = result
                            elif name == "sar":
                                cop.update(result)
                            elif name == "harmonics":
                                cop["harmonics"] = result
                            log.info("KG %s: %s OK", kg_code, name)
                    except concurrent.futures.TimeoutError:
                        log.warning("KG %s: %s timed out after %ds, skipping",
                                    kg_code, name, timeout)
                    except Exception as e:
                        log.debug("KG %s: %s failed: %s", kg_code, name, e)

            copernicus_data = cop if cop else None
        except Exception as e:
            log.warning("KG %s: Copernicus: %s", kg_code, e)
        stats["copernicus_time"] = round(time.time() - t0, 1)
        stats["has_copernicus"] = copernicus_data is not None
        if copernicus_data:
            stats["cop_layers"] = list(copernicus_data.keys())

    # 5. Segment
    t0 = time.time()
    try:
        result = oc.segment_and_classify(
            data["dtm"], data["dsm"], data["mask"], data["transform"],
            spectral=spectral, copernicus=copernicus_data,
        )
    except Exception as e:
        stats["error"] = f"segmentation: {e}"
        return [], [], stats
    stats["segment_time"] = round(time.time() - t0, 1)

    features_list = [obj.features for obj in result["objects"]]
    stats["n_segments"] = len(features_list)

    if not features_list:
        stats["error"] = "no segments"
        return [], [], stats

    # 6. Match segments to cadastre ground truth
    t0 = time.time()
    train_features = []
    train_labels = []
    source_counts = {"building_footprint": 0, "landuse_polygon": 0,
                     "parcel": 0, "landuse_point": 0, "unmatched": 0}

    for feat in features_list:
        code, source = match_segment_to_cadastre(feat, cadastre_data, data["transform"])
        if code is not None and code in lc.CADASTRE_TO_TYPE:
            train_features.append(feat)
            train_labels.append(lc.CADASTRE_TO_TYPE[code])
            source_counts[source] = source_counts.get(source, 0) + 1
        else:
            source_counts["unmatched"] += 1

    stats["match_time"] = round(time.time() - t0, 1)
    stats["n_labelled"] = len(train_features)
    stats["source_counts"] = source_counts
    stats["label_distribution"] = {}
    for lbl in train_labels:
        stats["label_distribution"][lbl] = stats["label_distribution"].get(lbl, 0) + 1

    log.info("KG %s (%s): %d segments, %d labelled (%s)",
             kg_code, kg.get("kg_name", ""),
             len(features_list), len(train_features),
             ", ".join(f"{k}={v}" for k, v in source_counts.items() if v > 0))

    return train_features, train_labels, stats


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("RF Training: 100 random KGs with include_copernicus=True")
    log.info("=" * 70)

    # Get random KGs
    log.info("Fetching KG list from cadastre API...")
    kgs = get_random_kgs(100)
    log.info("Selected %d KGs", len(kgs))

    # Save KG list
    kg_list_path = RESULTS_DIR / "kg_list.json"
    with open(kg_list_path, "w") as f:
        json.dump([{"kg_code": k["kg_code"], "kg_name": k.get("kg_name", "")} for k in kgs], f, indent=2)

    # Process each KG
    all_features = []
    all_labels = []
    all_stats = []
    n_success = 0
    n_fail = 0

    for i, kg in enumerate(kgs):
        log.info("-" * 50)
        log.info("[%d/%d] Processing KG %s (%s)",
                 i + 1, len(kgs), kg["kg_code"], kg.get("kg_name", ""))

        try:
            features, labels, stats = process_one_kg(kg, include_copernicus=True)
            stats["index"] = i
            all_stats.append(stats)

            if features:
                all_features.extend(features)
                all_labels.extend(labels)
                n_success += 1
                log.info("  → +%d samples (total: %d)", len(features), len(all_features))
            else:
                n_fail += 1
                log.warning("  → FAILED: %s", stats.get("error", "no labelled segments"))
        except Exception as e:
            n_fail += 1
            log.error("  → EXCEPTION: %s", traceback.format_exc())
            all_stats.append({"kg_code": kg["kg_code"], "error": str(e), "index": i})

        # Save progress every 10 KGs
        if (i + 1) % 10 == 0:
            progress = {
                "completed": i + 1,
                "total": len(kgs),
                "success": n_success,
                "fail": n_fail,
                "total_samples": len(all_features),
                "elapsed_min": round((time.time() - t_start) / 60, 1),
            }
            with open(RESULTS_DIR / "progress.json", "w") as f:
                json.dump(progress, f, indent=2)
            log.info("Progress: %d/%d done, %d samples, %.1f min elapsed",
                     i + 1, len(kgs), len(all_features),
                     (time.time() - t_start) / 60)

    # Save all stats
    with open(RESULTS_DIR / "kg_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)

    log.info("=" * 70)
    log.info("Collection complete: %d KGs succeeded, %d failed", n_success, n_fail)
    log.info("Total training samples: %d", len(all_features))

    # Label distribution
    label_dist = {}
    for lbl in all_labels:
        label_dist[lbl] = label_dist.get(lbl, 0) + 1
    log.info("Label distribution:")
    for lbl, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        log.info("  %-15s %6d (%.1f%%)", lbl, cnt, 100 * cnt / len(all_labels))

    # Source distribution
    source_dist = {}
    for s in all_stats:
        for src, cnt in s.get("source_counts", {}).items():
            source_dist[src] = source_dist.get(src, 0) + cnt
    log.info("Ground truth sources:")
    for src, cnt in sorted(source_dist.items(), key=lambda x: -x[1]):
        log.info("  %-20s %6d", src, cnt)

    if len(all_features) < 20:
        log.error("Not enough samples to train (%d < 20)", len(all_features))
        return

    # Train RF
    log.info("=" * 70)
    log.info("Training Random Forest on %d samples...", len(all_features))
    import learned_classifier as lc
    clf = lc.LearnedClassifier()
    try:
        train_stats = clf.train(all_features, all_labels)
        log.info("Training complete!")
        log.info("  OOB score: %.4f", train_stats["oob_score"])
        log.info("  Classes: %d", train_stats["n_classes"])
        log.info("  Samples: %d", train_stats["n_train"])
        log.info("  Top features:")
        for feat_name, imp in sorted(train_stats["top_features"].items(), key=lambda x: -x[1]):
            log.info("    %-25s %.4f", feat_name, imp)

        # Save training report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_kgs_processed": n_success,
            "n_kgs_failed": n_fail,
            "n_total_samples": len(all_features),
            "training_stats": train_stats,
            "label_distribution": label_dist,
            "source_distribution": source_dist,
            "elapsed_minutes": round((time.time() - t_start) / 60, 1),
        }
        with open(RESULTS_DIR / "training_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        log.info("Report saved to %s", RESULTS_DIR / "training_report.json")
    except Exception as e:
        log.error("Training failed: %s", traceback.format_exc())

    elapsed = time.time() - t_start
    log.info("Total time: %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
