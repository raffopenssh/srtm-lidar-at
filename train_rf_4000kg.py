#!/usr/bin/env python3
"""Train Random Forest on 4000 random KGs using cadastre + OSM ground truth.

Improvements over train_rf_100kg.py:
- 4000 KGs (was 100)
- OSM roads/paths/landcover as supplementary ground truth
- Model checkpoint saved every 10 successful KGs
- n_kgs passed to clf.train() for metadata

Run: python3 train_rf_4000kg.py 2>&1 | tee /tmp/rf_train_4000kg.log
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

N_KGS = 4000
MODEL_CHECKPOINT_INTERVAL = 10  # train & save model every N successful KGs

CADASTRE_BASE = "https://cadastre-process-api.exe.xyz/api/v1"
RESULTS_DIR = Path("/tmp/rf_train_4000kg")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Permanent storage — survives /tmp cleanup
PERMANENT_DIR = Path("/home/exedev/srtm-lidar/rf_training_data")
PERMANENT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PERMANENT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# WGS84 <-> EPSG:3035 transformers
_tx_to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
_tx_to_wgs = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


def transform_to_3035(geom):
    return shapely_transform(_tx_to_3035.transform, geom)


def transform_to_wgs(geom):
    return shapely_transform(_tx_to_wgs.transform, geom)


def get_random_kgs(n: int = N_KGS) -> list[dict]:
    """Fetch KGs that overlap LIDAR coverage and pick n random ones."""
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
                if not code or not d.get("district_name", "").strip():
                    continue
                try:
                    resp2 = requests.get(
                        f"{CADASTRE_BASE}/search/kg",
                        params={"district": d["district_name"], "limit": 500},
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

    with_buildings = [kg for kg in all_kgs if kg.get("building_count", 0) > 5]
    log.info("%d KGs have >5 buildings", len(with_buildings))

    random.seed(42)
    if len(with_buildings) >= n:
        selected = random.sample(with_buildings, n)
    else:
        selected = list(with_buildings)
        if len(selected) < n and all_kgs:
            remaining = [k for k in all_kgs if k not in selected]
            extra = random.sample(remaining, min(n - len(selected), len(remaining)))
            selected.extend(extra)

    log.info("Selected %d KGs for training", len(selected))
    return selected


def fetch_cadastre_layers(kg_code: str) -> dict:
    """Fetch all three cadastre layers for a KG."""
    result = {"parcels": [], "building_footprints": [], "landuse": []}

    try:
        resp = requests.get(
            f"{CADASTRE_BASE}/export/geojson",
            params={
                "kg": kg_code,
                "layers": "parcels,building_footprints,landuse_polygons",
                "include_geometry": "true",
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("KG %s: cadastre fetch failed: %s", kg_code, e)
        return result

    parcels_fc = data.get("parcels", {}).get("features", [])
    for f in parcels_fc:
        try:
            geom = shapely_shape(f["geometry"])
            if geom.is_empty or not isinstance(geom, (Polygon, MultiPolygon)):
                continue
            geom_3035 = transform_to_3035(geom)
            props = f.get("properties", {})
            lu_summary = props.get("landuse_summary", {})
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

    lu_fc = data.get("landuse_polygons", {}).get("features", [])
    for f in lu_fc:
        try:
            geom = shapely_shape(f["geometry"])
            if geom.is_empty or not isinstance(geom, (Polygon, MultiPolygon)):
                continue
            props = f.get("properties", {})
            code_str = props.get("landuse_code", "")
            abbr = props.get("landuse_abbr", "")
            try:
                code = int(code_str)
            except (ValueError, TypeError):
                code = None
            geom_3035 = transform_to_3035(geom)
            result["landuse"].append({
                "geometry": geom_3035,
                "code": code,
                "abbr": abbr,
                "area_sqm": props.get("area_sqm", 0),
            })
        except Exception:
            continue

    log.info("KG %s: %d parcels, %d building_footprints, %d landuse features",
             kg_code, len(result["parcels"]), len(result["building_footprints"]),
             len(result["landuse"]))
    return result


_LANDUSE_DESC_TO_CODE = {
    "B": 42, "B(bf)": 41, "B(bg)": 40,
    "V": 48, "W": 56, "LN": 52, "LN(W)": 52, "LN(Hu)": 53,
    "A": 51, "WG": 63, "GA": 64, "OG": 65, "Alpe": 54,
    "GW": 70, "Öd": 80, "Fe": 81,
    "Wald": 56, "Acker": 51, "Wiese": 52, "Weide": 55,
    "Obstgarten": 65, "Weingarten": 63, "Gewässer": 70,
}


def _extract_dominant_code(lu_summary: dict) -> int | None:
    if not lu_summary:
        return None
    best_key = max(lu_summary, key=lu_summary.get)
    if " - " in best_key:
        abbr = best_key.split(" - ")[-1]
        if abbr in _LANDUSE_DESC_TO_CODE:
            return _LANDUSE_DESC_TO_CODE[abbr]
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
    """
    ce = feat.get("centroid_e", 0)
    cn = feat.get("centroid_n", 0)
    if ce == 0 and cn == 0:
        return None, ""

    from shapely.geometry import Point
    pt = Point(ce, cn)
    seg_area = feat.get("area", 0)

    seg_bbox = feat.get("bbox")
    if seg_bbox and len(seg_bbox) == 4:
        seg_poly = box(seg_bbox[0], seg_bbox[1], seg_bbox[2], seg_bbox[3])
    else:
        r = max(1.0, (seg_area / 3.14159) ** 0.5) if seg_area > 0 else 5.0
        seg_poly = pt.buffer(r)

    # 1. Building footprints
    for bfp in cadastre_data["building_footprints"]:
        try:
            if bfp.intersects(seg_poly):
                overlap = bfp.intersection(seg_poly).area
                if overlap > 0.3 * seg_poly.area or overlap > 0.3 * bfp.area:
                    return 42, "building_footprint"
        except Exception:
            continue

    # 2. Landuse polygons
    best_lu_code = None
    best_lu_overlap = 0
    for lu in cadastre_data["landuse"]:
        code = lu.get("code")
        if code is None:
            continue
        geom = lu["geometry"]
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

    # 3. Parcel polygon
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

    if best_lu_code is not None:
        return best_lu_code, "landuse_polygon_weak"

    return None, ""


def match_segment_to_osm(
    feat: dict,
    osm_labels: np.ndarray,
    transform,
) -> tuple[str | None, str]:
    """Match a segment to OSM ground truth via its centroid pixel.

    Returns (label_string_or_None, source_string).
    """
    ce = feat.get("centroid_e", 0)
    cn = feat.get("centroid_n", 0)
    if ce == 0 and cn == 0:
        return None, ""

    # Convert 3035 coordinate to pixel row/col
    import rasterio
    try:
        col, row = ~transform * (ce, cn)
        row, col = int(round(row)), int(round(col))
        if 0 <= row < osm_labels.shape[0] and 0 <= col < osm_labels.shape[1]:
            lbl = str(osm_labels[row, col])
            if lbl and lbl != "":
                return lbl, "osm"
    except Exception:
        pass
    return None, ""


def process_one_kg(
    kg: dict,
    include_copernicus: bool = True,
    include_osm: bool = True,
) -> tuple[list[dict], list[str], dict]:
    """Process one KG: segment + match to cadastre + OSM.

    Returns (features, labels, stats).
    """
    import raster_io
    import tile_index as ti
    import object_segmentation as oc
    import learned_classifier as lc

    kg_code = kg["kg_code"]
    stats = {"kg_code": kg_code, "kg_name": kg.get("kg_name", "")}

    # Get KG bbox
    kg_bbox = kg.get("bbox", {})
    if isinstance(kg_bbox, dict) and all(k in kg_bbox for k in ["min_lon", "min_lat", "max_lon", "max_lat"]):
        west, south = kg_bbox["min_lon"], kg_bbox["min_lat"]
        east, north = kg_bbox["max_lon"], kg_bbox["max_lat"]
    else:
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

    # Limit KG size to center 3km
    dx_km = (east - west) * 111 * np.cos(np.radians((south + north) / 2))
    dy_km = (north - south) * 111
    if dx_km > 3 or dy_km > 3:
        cx, cy = (west + east) / 2, (south + north) / 2
        half = 0.0135
        west, south, east, north = cx - half, cy - half, cx + half, cy + half
        log.info("KG %s: large (%.1f x %.1f km), cropping to center 3km",
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

    # Observation year — drives Copernicus/SAR/Hansen year scoping
    obs_year = ti.dataset_to_year(ti.DEFAULT_DATASET)

    # 2. Read LIDAR
    t0 = time.time()
    try:
        data = raster_io.read_dtm_dsm(geom_3035, ti.DEFAULT_DATASET)
    except Exception as e:
        stats["error"] = f"LIDAR: {e}"
        return [], [], stats
    stats["lidar_time"] = round(time.time() - t0, 1)
    stats["shape"] = list(data["shape"])

    # 2b. Multi-date DTM/DSM
    dtm_dates = None
    dsm_dates = None
    try:
        other_dates = sorted(d for d in ti.DATASETS if d != ti.DEFAULT_DATASET)
        if other_dates:
            dtm_dates = {}
            dsm_dates = {}
            ref_h, ref_w = data["shape"]
            for date_key in other_dates:
                try:
                    d2 = raster_io.read_dtm_dsm(geom_3035, date_key)
                    mh = min(ref_h, d2["shape"][0])
                    mw = min(ref_w, d2["shape"][1])
                    dtm_dates[date_key] = d2["dtm"][:mh, :mw]
                    dsm_dates[date_key] = d2["dsm"][:mh, :mw]
                except Exception as e:
                    log.warning("KG %s: multi-date %s failed: %s", kg_code, date_key, e)
            if dtm_dates:
                mh = min(ref_h, *(a.shape[0] for a in dtm_dates.values()))
                mw = min(ref_w, *(a.shape[1] for a in dtm_dates.values()))
                dtm_dates[ti.DEFAULT_DATASET] = data["dtm"][:mh, :mw]
                dsm_dates[ti.DEFAULT_DATASET] = data["dsm"][:mh, :mw]
                for dk in list(dtm_dates):
                    dtm_dates[dk] = dtm_dates[dk][:mh, :mw]
                    dsm_dates[dk] = dsm_dates[dk][:mh, :mw]
                log.info("KG %s: loaded %d temporal dates: %s",
                         kg_code, len(dtm_dates), sorted(dtm_dates))
            else:
                dtm_dates = None
                dsm_dates = None
    except Exception as e:
        log.warning("KG %s: multi-date read failed: %s", kg_code, e)
        dtm_dates = None
        dsm_dates = None
    stats["has_temporal"] = dtm_dates is not None and len(dtm_dates) >= 2

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

    # 4. Copernicus data
    copernicus_data = None
    if include_copernicus:
        t0 = time.time()
        try:
            import copernicus
            import concurrent.futures
            bbox_dict = {"west": west, "south": south, "east": east, "north": north}
            cop = {}

            def _fetch_ndvi():
                d = copernicus.get_ndvi_composite(bbox_dict, year=obs_year)
                return {"ndvi": d["ndvi"], "transform": d["transform"], "crs": d["crs"]}

            def _fetch_landcover():
                return copernicus.get_land_cover(bbox_dict)

            def _fetch_sar():
                sar_start = f"{obs_year}-06-01"
                sar_end   = f"{obs_year}-09-30"
                d = copernicus.get_sar_backscatter(bbox_dict, sar_start, sar_end)
                return {"vv": d["vv"], "vh": d["vh"], "sar_transform": d["transform"], "sar_crs": d["crs"]}

            def _fetch_harmonics():
                import ndvi_harmonics
                return ndvi_harmonics.get_harmonic_features(bbox_dict, year=obs_year)

            COP_TIMEOUT = 180
            HARM_TIMEOUT = 900

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

    # 5. Hansen Global Forest Change
    hansen_data = None
    t0 = time.time()
    try:
        import hansen
        hansen_data = hansen.get_forest_prior(
            (west, south, east, north),
            data["transform"], data["shape"],
        )
        stats["has_hansen"] = True
        log.info("KG %s: Hansen OK (forest=%d px, loss=%d px)",
                 kg_code,
                 int(hansen_data["current_forest"].sum()),
                 int((hansen_data["loss_year"] > 0).sum()))
    except Exception as e:
        log.warning("KG %s: Hansen failed: %s", kg_code, e)
        stats["has_hansen"] = False
    stats["hansen_time"] = round(time.time() - t0, 1)

    # 5b. OSM ground truth (roads, paths, landcover)
    osm_labels = None
    if include_osm:
        t0 = time.time()
        try:
            import osm_features
            osm_result = osm_features.fetch_osm_ground_truth(
                (west, south, east, north),
                data["transform"],
                data["shape"],
            )
            osm_labels = osm_result["labels"]
            stats["osm_road_px"] = osm_result["n_road_px"]
            stats["osm_landcover_px"] = osm_result["n_landcover_px"]
            stats["has_osm"] = osm_result["n_road_px"] + osm_result["n_landcover_px"] > 0
            log.info("KG %s: OSM OK (road=%d px, lc=%d px)",
                     kg_code, osm_result["n_road_px"], osm_result["n_landcover_px"])
        except Exception as e:
            log.warning("KG %s: OSM failed: %s", kg_code, e)
            stats["has_osm"] = False
        stats["osm_time"] = round(time.time() - t0, 1)

    # 6. Segment
    t0 = time.time()
    try:
        result = oc.segment_and_classify(
            data["dtm"], data["dsm"], data["mask"], data["transform"],
            dtm_dates=dtm_dates, dsm_dates=dsm_dates,
            spectral=spectral, copernicus=copernicus_data,
            hansen=hansen_data,
            observation_year=obs_year,
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

    # 7. Match segments to ground truth: cadastre first, OSM as fallback
    t0 = time.time()
    train_features = []
    train_labels = []
    source_counts = {
        "building_footprint": 0, "landuse_polygon": 0,
        "landuse_polygon_weak": 0, "parcel": 0,
        "osm": 0, "unmatched": 0,
    }

    for feat in features_list:
        # Try cadastre first (higher precision)
        code, source = match_segment_to_cadastre(feat, cadastre_data, data["transform"])
        if code is not None and code in lc.CADASTRE_TO_TYPE:
            train_features.append(feat)
            train_labels.append(lc.CADASTRE_TO_TYPE[code])
            source_counts[source] = source_counts.get(source, 0) + 1
            continue

        # Fall back to OSM ground truth
        if osm_labels is not None:
            osm_lbl, osm_src = match_segment_to_osm(feat, osm_labels, data["transform"])
            if osm_lbl and osm_lbl in lc.TYPE_CLASSES:
                train_features.append(feat)
                train_labels.append(osm_lbl)
                source_counts["osm"] += 1
                continue

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


def _clear_downloaded_caches():
    """Delete cached .npz/.tif files from /tmp to reclaim memory."""
    import shutil
    cleared = 0
    for cache_dir in [
        Path("/tmp/copernicus_cache"),
        Path("/tmp/hansen_cache"),
    ]:
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                try:
                    if f.is_dir():
                        shutil.rmtree(f)
                    else:
                        f.unlink()
                    cleared += 1
                except Exception:
                    pass
    if cleared:
        log.info("Cleared %d cached raster entries to free memory", cleared)


def train_and_save_model(all_features, all_labels, n_kgs, tag="checkpoint"):
    """Train RF model and save to disk. Returns training stats or None on failure."""
    import learned_classifier as lc

    if len(all_features) < 20:
        log.warning("Not enough samples (%d) for model %s", len(all_features), tag)
        return None

    log.info("Training RF model [%s] on %d samples from %d KGs...",
             tag, len(all_features), n_kgs)
    clf = lc.LearnedClassifier()
    try:
        train_stats = clf.train(all_features, all_labels, n_kgs=n_kgs)
        log.info("Model [%s]: OOB=%.4f, %d classes, %d samples",
                 tag, train_stats["oob_score"], train_stats["n_classes"],
                 train_stats["n_train"])
        return train_stats
    except Exception as e:
        log.error("Model training [%s] failed: %s", tag, traceback.format_exc())
        return None


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("RF Training: %d random KGs with cadastre + OSM ground truth", N_KGS)
    log.info("=" * 70)

    # Get random KGs
    log.info("Fetching KG list from cadastre API...")
    kgs = get_random_kgs(N_KGS)
    log.info("Selected %d KGs", len(kgs))

    # Save KG list
    kg_list_data = [{"kg_code": k["kg_code"], "kg_name": k.get("kg_name", "")} for k in kgs]
    for dest in [RESULTS_DIR / "kg_list.json", PERMANENT_DIR / "kg_list.json"]:
        with open(dest, "w") as f:
            json.dump(kg_list_data, f, indent=2)

    # --- Resume from checkpoint ---
    all_features = []
    all_labels = []
    all_stats = []
    n_success = 0
    n_fail = 0
    completed_kgs = set()

    for ckpt_file in sorted(CHECKPOINT_DIR.glob("kg_*.npz")):
        try:
            ckpt = np.load(ckpt_file, allow_pickle=True)
            kg_code = str(ckpt["kg_code"])
            feats = ckpt["features"].tolist()
            lbls = ckpt["labels"].tolist()
            all_features.extend(feats)
            all_labels.extend(lbls)
            completed_kgs.add(kg_code)
            n_success += 1
        except Exception as e:
            log.warning("Failed to load checkpoint %s: %s", ckpt_file, e)

    stats_ckpt = PERMANENT_DIR / "kg_stats_partial.json"
    if stats_ckpt.exists():
        try:
            all_stats = json.loads(stats_ckpt.read_text())
        except Exception:
            pass

    if completed_kgs:
        log.info("RESUMING: loaded %d KGs (%d samples) from checkpoints",
                 len(completed_kgs), len(all_features))

    # Track when we last trained a model checkpoint
    last_model_at_n_success = (n_success // MODEL_CHECKPOINT_INTERVAL) * MODEL_CHECKPOINT_INTERVAL

    # Process each KG
    for i, kg in enumerate(kgs):
        kg_code = kg["kg_code"]

        if kg_code in completed_kgs:
            log.info("[%d/%d] KG %s — already checkpointed, skipping",
                     i + 1, len(kgs), kg_code)
            continue

        log.info("-" * 50)
        log.info("[%d/%d] Processing KG %s (%s)",
                 i + 1, len(kgs), kg_code, kg.get("kg_name", ""))

        try:
            features, labels, stats = process_one_kg(
                kg, include_copernicus=True, include_osm=True)
            stats["index"] = i
            all_stats.append(stats)

            if features:
                all_features.extend(features)
                all_labels.extend(labels)
                n_success += 1
                log.info("  → +%d samples (total: %d from %d KGs)",
                         len(features), len(all_features), n_success)

                # Per-KG checkpoint
                ckpt_path = CHECKPOINT_DIR / f"kg_{kg_code}.npz"
                np.savez_compressed(
                    ckpt_path,
                    kg_code=kg_code,
                    features=np.array(features, dtype=object),
                    labels=np.array(labels, dtype=object),
                )
                log.info("  Checkpoint saved: %s (%d samples)", ckpt_path.name, len(features))

                # Model checkpoint every N successful KGs
                if n_success >= last_model_at_n_success + MODEL_CHECKPOINT_INTERVAL:
                    train_stats = train_and_save_model(
                        all_features, all_labels, n_success,
                        tag=f"checkpoint_{n_success}kg")
                    last_model_at_n_success = n_success
                    if train_stats:
                        log.info("  ✓ Model checkpoint saved at %d KGs (OOB=%.4f)",
                                 n_success, train_stats["oob_score"])
                    # Clear downloaded cache data to prevent memory buildup
                    _clear_downloaded_caches()
            else:
                n_fail += 1
                log.warning("  → FAILED: %s", stats.get("error", "no labelled segments"))
        except Exception as e:
            n_fail += 1
            log.error("  → EXCEPTION: %s", traceback.format_exc())
            all_stats.append({"kg_code": kg_code, "error": str(e), "index": i})

        # Save progress
        progress = {
            "completed": n_success + n_fail,
            "total": len(kgs),
            "success": n_success,
            "fail": n_fail,
            "total_samples": len(all_features),
            "elapsed_min": round((time.time() - t_start) / 60, 1),
            "completed_kgs": sorted(completed_kgs | {kg_code}),
        }
        for dest in [RESULTS_DIR / "progress.json", PERMANENT_DIR / "progress.json"]:
            with open(dest, "w") as f:
                json.dump(progress, f, indent=2)

        with open(stats_ckpt, "w") as f:
            json.dump(all_stats, f, indent=2, default=str)

        completed_kgs.add(kg_code)

        import gc
        gc.collect()

        if (i + 1) % 5 == 0 or len(completed_kgs) % 5 == 0:
            log.info("Progress: %d/%d done (%d success), %d samples, %.1f min elapsed",
                     len(completed_kgs), len(kgs), n_success, len(all_features),
                     (time.time() - t_start) / 60)

    # Final stats
    for dest in [RESULTS_DIR / "kg_stats.json", PERMANENT_DIR / "kg_stats.json"]:
        with open(dest, "w") as f:
            json.dump(all_stats, f, indent=2, default=str)

    log.info("=" * 70)
    log.info("Collection complete: %d KGs succeeded, %d failed", n_success, n_fail)
    log.info("Total training samples: %d", len(all_features))

    label_dist = {}
    for lbl in all_labels:
        label_dist[lbl] = label_dist.get(lbl, 0) + 1
    log.info("Label distribution:")
    for lbl, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        log.info("  %-15s %6d (%.1f%%)", lbl, cnt, 100 * cnt / max(len(all_labels), 1))

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

    # Final model training
    log.info("=" * 70)
    log.info("Training FINAL Random Forest on %d samples from %d KGs...",
             len(all_features), n_success)
    train_stats = train_and_save_model(all_features, all_labels, n_success, tag="final")

    if train_stats:
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
        for dest in [RESULTS_DIR / "training_report.json", PERMANENT_DIR / "training_report.json"]:
            with open(dest, "w") as f:
                json.dump(report, f, indent=2, default=str)

        log.info("Report saved to %s", PERMANENT_DIR / "training_report.json")

    elapsed = time.time() - t_start
    log.info("Total time: %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
