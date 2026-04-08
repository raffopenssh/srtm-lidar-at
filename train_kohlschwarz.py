"""Train RF classifier on Kohlschwarz KG (63330) cadastre ground truth.

Processes multiple tiles across the KG, extracts segment features,
matches them to cadastre landuse polygons, and trains the RF.
"""
import json
import logging
import time
import sys
import numpy as np
from collections import Counter
from shapely.geometry import shape, box, Point
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
from rasterio.features import rasterize as rio_rasterize

import tile_index as ti
import raster_io
import object_segmentation as oc
import learned_classifier as lc

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

TF_4326_TO_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


def load_landuse_polygons(path="/tmp/kohlschwarz_landuse.json"):
    """Load landuse polygons, transform to EPSG:3035."""
    d = json.load(open(path))
    polys = []
    for f in d.get("features", []):
        geom_json = f.get("geometry")
        if not geom_json or geom_json.get("type") != "Polygon":
            continue
        code = f["properties"].get("landuse_code")
        if not code:
            continue
        code = int(code)
        if code not in lc.CADASTRE_TO_TYPE:
            continue
        try:
            geom = shape(geom_json)
            if geom.is_empty or not geom.is_valid:
                geom = geom.buffer(0)
            geom_3035 = shapely_transform(TF_4326_TO_3035.transform, geom)
            polys.append({"geometry": geom_3035, "code": code, "type": lc.CADASTRE_TO_TYPE[code]})
        except Exception:
            continue
    log.info("Loaded %d landuse polygons", len(polys))
    code_counts = Counter(p["type"] for p in polys)
    for t, n in code_counts.most_common():
        log.info("  %s: %d polygons", t, n)
    return polys


def rasterize_landuse(polygons, transform, shape_hw):
    """Rasterize landuse polygons to a code raster."""
    pairs = [(p["geometry"], p["code"]) for p in polygons if not p["geometry"].is_empty]
    if not pairs:
        return np.zeros(shape_hw, dtype=np.int32)
    raster = rio_rasterize(
        pairs, out_shape=shape_hw, transform=transform,
        fill=0, dtype=np.int32, all_touched=True,
    )
    return raster


def match_segments_to_landuse(features, labels, landuse_raster):
    """Match each segment to its dominant landuse code."""
    matched = []
    for feat in features:
        lbl = feat["label"]
        seg = labels == lbl
        if seg.sum() < 5:
            continue
        codes_in_seg = landuse_raster[seg]
        codes_in_seg = codes_in_seg[codes_in_seg > 0]
        if len(codes_in_seg) == 0:
            continue
        dominant_code = int(np.bincount(codes_in_seg).argmax())
        # Require at least 50% of segment to have the dominant code
        dominance = np.sum(codes_in_seg == dominant_code) / len(codes_in_seg)
        if dominance < 0.5:
            continue
        if dominant_code not in lc.CADASTRE_TO_TYPE:
            continue
        matched.append((feat, lc.CADASTRE_TO_TYPE[dominant_code]))
    return matched


def process_tile(center_lon, center_lat, radius_m, landuse_polys, dataset="20240915"):
    """Process one tile: segment, extract features, match to landuse."""
    try:
        geom_wgs = Point(center_lon, center_lat)
        geom_3035 = ti.geometry_to_3035(geom_wgs)
        e, n = geom_3035.coords[0]
        bbox_3035 = box(e - radius_m, n - radius_m, e + radius_m, n + radius_m)

        # Read LIDAR
        data = raster_io.read_dtm_dsm(bbox_3035, dataset)
        h, w = data["shape"]
        log.info("Tile (%.4f, %.4f): %dx%d", center_lon, center_lat, w, h)

        # Read ortho
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
            log.warning("Ortho failed: %s", e)
            spectral = None

        # Segment
        result = oc.segment_and_classify(
            data["dtm"], data["dsm"], data["mask"], data["transform"],
            spectral=spectral,
        )
        labels = result["labels"]
        features_list = [obj.features for obj in result["objects"]]

        # Rasterize landuse
        landuse_raster = rasterize_landuse(landuse_polys, data["transform"], (h, w))
        n_labelled_px = np.sum(landuse_raster > 0)
        log.info("Landuse coverage: %d/%d pixels (%.0f%%)",
                 n_labelled_px, h * w, 100 * n_labelled_px / (h * w))

        # Match
        matched = match_segments_to_landuse(features_list, labels, landuse_raster)
        log.info("Matched %d / %d segments to landuse", len(matched), len(features_list))
        return matched

    except Exception as e:
        log.warning("Tile (%.4f, %.4f) failed: %s", center_lon, center_lat, e)
        return []


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("Training RF classifier on Kohlschwarz KG (63330)")
    log.info("=" * 60)

    # Load landuse polygons
    landuse_polys = load_landuse_polygons()

    # Tile grid across Kohlschwarz — sorted by expected landuse density.
    # Landuse polygons are densest around the center of the KG.
    # Use 500m tiles, sorted by distance from centroid of landuse polys.
    tile_radius = 250  # meters
    step_deg_lon = 0.005  # ~400m at this latitude
    step_deg_lat = 0.004  # ~440m

    # Compute centroid of landuse polygons in WGS84
    poly_lons = [p["geometry"].centroid.x for p in landuse_polys if not p["geometry"].is_empty]
    poly_lats = [p["geometry"].centroid.y for p in landuse_polys if not p["geometry"].is_empty]
    # These are in EPSG:3035, convert back
    from pyproj import Transformer
    tf_back = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    centroid_3035_e = np.mean(poly_lons)
    centroid_3035_n = np.mean(poly_lats)
    centroid_lon, centroid_lat = tf_back.transform(centroid_3035_e, centroid_3035_n)
    log.info("Landuse centroid: %.4f, %.4f", centroid_lon, centroid_lat)

    centers = []
    lon = 15.055
    while lon < 15.148:
        lat = 47.095
        while lat < 47.165:
            centers.append((lon, lat))
            lat += step_deg_lat
        lon += step_deg_lon

    # Sort by distance from centroid (process dense areas first)
    centers.sort(key=lambda c: (c[0] - centroid_lon)**2 + (c[1] - centroid_lat)**2)
    log.info("Processing %d tiles (sorted by landuse density)", len(centers))

    all_matched = []
    for i, (lon, lat) in enumerate(centers):
        log.info("--- Tile %d/%d (%.4f, %.4f) ---", i + 1, len(centers), lon, lat)
        matched = process_tile(lon, lat, tile_radius, landuse_polys)
        all_matched.extend(matched)
        # Progress
        types_so_far = Counter(m[1] for m in all_matched)
        log.info("Running total: %d samples, %d types", len(all_matched), len(types_so_far))

        # Early stop if we have enough
        if len(all_matched) > 3000 and len(types_so_far) >= 6:
            log.info("Enough data: %d samples, %d types — stopping", len(all_matched), len(types_so_far))
            break
        # Skip tiles with zero matches after we've processed enough tiles
        if i > 50 and len(all_matched) < 50:
            log.warning("Only %d matches after %d tiles, something wrong", len(all_matched), i)
            break

    log.info("\nTotal matched segments: %d", len(all_matched))
    type_counts = Counter(m[1] for m in all_matched)
    for t, n in type_counts.most_common():
        log.info("  %s: %d", t, n)

    if len(all_matched) < 20:
        log.error("Not enough training data!")
        sys.exit(1)

    # Train RF
    features = [m[0] for m in all_matched]
    labels = [m[1] for m in all_matched]

    clf = lc.LearnedClassifier()
    stats = clf.train(features, labels)

    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)
    log.info("OOB score: %.3f", stats["oob_score"])
    log.info("Classes: %d", stats["n_classes"])
    log.info("Samples: %d", stats["n_train"])
    log.info("Top features:")
    for k, v in sorted(stats["top_features"].items(), key=lambda x: -x[1]):
        log.info("  %s: %.4f", k, v)
    log.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
