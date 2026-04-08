"""Cadastre building footprint fetcher and ground-truth evaluator.

Fetches building polygons from the Austrian Cadastre API, transforms them
to EPSG:3035, rasterizes onto our 1 m LIDAR grid, and provides evaluation
metrics against our object classifier output.

API base: https://cadastre-process-api.exe.xyz/api/v1
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import shape as shapely_shape, mapping
from shapely.ops import transform as shapely_transform

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CADASTRE_BASE = "https://cadastre-process-api.exe.xyz/api/v1"
CACHE_DIR = Path("/tmp/cadastre_cache")
MAX_PER_REQUEST = 100_000
REQUEST_TIMEOUT = 60  # seconds

# Reusable CRS transformer (thread-safe after creation)
_TRANSFORMER_4326_TO_3035 = Transformer.from_crs(
    "EPSG:4326", "EPSG:3035", always_xy=True
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_hash(bbox: tuple[float, float, float, float]) -> str:
    """Deterministic short hash for a WGS84 bounding box."""
    key = f"{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _cache_path(bbox: tuple[float, float, float, float]) -> Path:
    return _ensure_cache_dir() / f"buildings_{_bbox_hash(bbox)}.geojson"


def _load_cached(bbox: tuple[float, float, float, float]) -> Optional[dict]:
    """Load cached GeoJSON response, if fresh (< 24 h)."""
    p = _cache_path(bbox)
    if not p.exists():
        return None
    age_h = (time.time() - p.stat().st_mtime) / 3600
    if age_h > 24:
        log.debug("Cache expired for bbox %s (%.1f h old)", bbox, age_h)
        return None
    try:
        data = json.loads(p.read_text())
        log.debug("Cache hit for bbox %s (%d features)", bbox, len(data.get("features", [])))
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Corrupt cache file %s: %s", p, exc)
        return None


def _save_cache(bbox: tuple[float, float, float, float], data: dict) -> None:
    try:
        _cache_path(bbox).write_text(json.dumps(data))
    except OSError as exc:
        log.warning("Failed to write cache: %s", exc)


def _transform_geom_to_3035(geom):
    """Transform a shapely geometry from EPSG:4326 → EPSG:3035."""
    return shapely_transform(_TRANSFORMER_4326_TO_3035.transform, geom)


# ---------------------------------------------------------------------------
# 1. Fetch building footprints
# ---------------------------------------------------------------------------

def _find_kgs_for_bbox(bbox_wgs84: tuple[float, float, float, float]) -> list[str]:
    """Find KG codes covering a WGS84 bounding box via the cadastre lookup."""
    west, south, east, north = bbox_wgs84
    # Sample points across the bbox to discover KG codes
    import itertools
    lons = [west, (west + east) / 2, east]
    lats = [south, (south + north) / 2, north]
    points = [{"lon": lo, "lat": la, "id": f"{lo:.5f}_{la:.5f}"}
              for lo, la in itertools.product(lons, lats)]
    try:
        resp = requests.post(
            f"{CADASTRE_BASE}/spatial/points",
            json={"points": points},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        kg_codes = set()
        for r in data.get("results", []):
            kc = r.get("kg_code")
            if kc:
                kg_codes.add(kc)
        return sorted(kg_codes)
    except Exception as e:
        log.warning("KG lookup failed: %s", e)
        return []


def fetch_building_footprints(
    bbox_wgs84: tuple[float, float, float, float],
    *,
    use_cache: bool = True,
    limit: int = MAX_PER_REQUEST,
) -> Optional[list]:
    """Fetch building footprint POLYGONS from the Austrian Cadastre API.

    Uses the /export/geojson endpoint with layers=building_footprints to get
    actual polygon geometries from KG source files (mm-precision).
    Falls back to /spatial/bbox (Point centroids) if export fails.

    Parameters
    ----------
    bbox_wgs84 : (west, south, east, north)
        Bounding box in WGS84 / EPSG:4326.
    use_cache : bool
        If True, cache responses in /tmp/cadastre_cache/.

    Returns
    -------
    list[shapely.geometry.Polygon | MultiPolygon] in EPSG:3035,
    or None if the API is unreachable / returns an error.
    """
    west, south, east, north = bbox_wgs84
    log.info(
        "Fetching cadastre building footprints for bbox [%.4f, %.4f, %.4f, %.4f]",
        west, south, east, north,
    )

    # --- try cache ---
    geojson = None
    if use_cache:
        geojson = _load_cached(bbox_wgs84)

    # --- HTTP fetch: export/geojson with building_footprints layer ---
    if geojson is None:
        kg_codes = _find_kgs_for_bbox(bbox_wgs84)
        if not kg_codes:
            log.warning("No KG codes found for bbox, cannot fetch footprints")
            return []

        all_features = []
        for kg in kg_codes:
            try:
                t0 = time.time()
                resp = requests.get(
                    f"{CADASTRE_BASE}/export/geojson",
                    params={
                        "kg": kg,
                        "layers": "building_footprints",
                        "include_geometry": "true",
                    },
                    timeout=REQUEST_TIMEOUT * 2,  # KG export can be slower
                )
                elapsed = time.time() - t0
                resp.raise_for_status()
                kg_data = resp.json()
                feats = _extract_features(kg_data)
                log.info(
                    "KG %s: %d building footprints in %.1fs",
                    kg, len(feats), elapsed,
                )
                all_features.extend(feats)
            except Exception as e:
                log.warning("KG %s footprint export failed: %s", kg, e)

        if all_features:
            geojson = {"type": "FeatureCollection", "features": all_features}
            if use_cache:
                _save_cache(bbox_wgs84, geojson)
        else:
            log.warning("No footprints from export, returning empty")
            return []

    # --- parse GeoJSON -> shapely polygons in EPSG:3035 ---
    features = _extract_features(geojson)
    if not features:
        log.warning("No building features returned for bbox %s", bbox_wgs84)
        return []

    from shapely.geometry import box as shapely_box
    clip_box = shapely_box(west, south, east, north)

    polygons_3035 = []
    n_skipped = 0
    for feat in features:
        geom_json = feat.get("geometry")
        if geom_json is None:
            n_skipped += 1
            continue
        try:
            geom_4326 = shapely_shape(geom_json)
            if geom_4326.is_empty or geom_4326.geom_type == "Point":
                n_skipped += 1
                continue
            # Clip to bbox (KG export returns all footprints in the KG)
            if not geom_4326.intersects(clip_box):
                continue
            geom_clipped = geom_4326.intersection(clip_box)
            if geom_clipped.is_empty:
                continue
            geom_3035 = _transform_geom_to_3035(geom_clipped)
            polygons_3035.append(geom_3035)
        except Exception as exc:
            n_skipped += 1
            log.debug("Skipping invalid feature geometry: %s", exc)

    log.info(
        "Parsed %d building footprints (%d skipped) -> EPSG:3035",
        len(polygons_3035), n_skipped,
    )
    return polygons_3035


def _extract_features(geojson: dict) -> list[dict]:
    """Robustly extract feature list from various GeoJSON shapes."""
    if not isinstance(geojson, dict):
        return []
    gtype = geojson.get("type", "")
    if gtype == "FeatureCollection":
        return geojson.get("features", [])
    if gtype == "Feature":
        return [geojson]
    # Some APIs return a bare geometry or a dict of layers
    if "features" in geojson:
        val = geojson["features"]
        if isinstance(val, list):
            return val
    # Nested layer dict? e.g. {"buildings": {"type": "FeatureCollection", ...}}
    for key in ("buildings", "building_footprints"):
        sub = geojson.get(key)
        if isinstance(sub, dict):
            return _extract_features(sub)
        if isinstance(sub, list):
            return sub
    return []


# ---------------------------------------------------------------------------
# 2. Rasterize footprints onto a grid
# ---------------------------------------------------------------------------

def rasterize_buildings(
    footprints_3035: list,
    transform,
    shape: tuple[int, int],
) -> np.ndarray:
    """Create a boolean raster where True = building pixel.

    Parameters
    ----------
    footprints_3035 : list of shapely Polygon/MultiPolygon in EPSG:3035
    transform : rasterio Affine transform for the output grid
    shape : (rows, cols) of the output grid

    Returns
    -------
    np.ndarray of bool, shape = *shape*
    """
    from rasterio.features import rasterize

    if not footprints_3035:
        log.info("No footprints to rasterize – returning empty mask")
        return np.zeros(shape, dtype=bool)

    # rasterio.features.rasterize expects (geometry, value) pairs
    geom_value_pairs = [(geom, 1) for geom in footprints_3035 if not geom.is_empty]

    if not geom_value_pairs:
        return np.zeros(shape, dtype=bool)

    log.info("Rasterizing %d building footprints onto %s grid", len(geom_value_pairs), shape)
    raster = rasterize(
        geom_value_pairs,
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=True,
    )
    mask = raster.astype(bool)
    n_px = int(mask.sum())
    log.info(
        "Building mask: %d pixels (%.2f%% of grid)",
        n_px, 100 * n_px / (shape[0] * shape[1]) if shape[0] * shape[1] > 0 else 0,
    )
    return mask


# ---------------------------------------------------------------------------
# 3. Convenience: fetch + rasterize
# ---------------------------------------------------------------------------

def get_building_mask(
    bbox_wgs84: tuple[float, float, float, float],
    transform,
    shape: tuple[int, int],
    *,
    use_cache: bool = True,
) -> Optional[np.ndarray]:
    """Fetch building footprints and rasterize them in one call.

    Parameters
    ----------
    bbox_wgs84 : (west, south, east, north)
    transform : rasterio Affine (EPSG:3035)
    shape : (rows, cols)
    use_cache : bool

    Returns
    -------
    np.ndarray of bool, or None if the fetch failed.
    """
    footprints = fetch_building_footprints(bbox_wgs84, use_cache=use_cache)
    if footprints is None:
        return None
    return rasterize_buildings(footprints, transform, shape)


# ---------------------------------------------------------------------------
# 4. Evaluate classifier against cadastre ground truth
# ---------------------------------------------------------------------------

# Building-related type codes from object_classifier
_BUILDING_CODES = {9, 10, 17, 18}  # building, structure, solar_panel, greenhouse


def evaluate_classification(
    predicted_types: np.ndarray,
    building_mask: np.ndarray,
    *,
    building_codes: set[int] | None = None,
) -> dict:
    """Compare predicted object-type raster against cadastre building mask.

    Parameters
    ----------
    predicted_types : np.ndarray uint8  (type_code per pixel)
        From ``object_classifier.make_type_raster()``.
    building_mask : np.ndarray bool
        Ground truth from ``get_building_mask()`` or ``rasterize_buildings()``.
    building_codes : set[int], optional
        Type codes that count as "building" in the prediction.
        Default: {9 (building), 10 (structure), 17 (solar_panel), 18 (greenhouse)}.

    Returns
    -------
    dict with keys:
        tp, fp, fn, tn   – confusion matrix counts
        precision         – TP / (TP + FP)  (what fraction of predicted buildings
                            are actually buildings)
        recall            – TP / (TP + FN)  (what fraction of real buildings
                            we detected)
        f1                – harmonic mean of precision & recall
        iou               – intersection over union (Jaccard index)
        accuracy          – (TP + TN) / total
        building_pixels_truth  – total building pixels in ground truth
        building_pixels_pred   – total building pixels in prediction
    """
    if building_codes is None:
        building_codes = _BUILDING_CODES

    if predicted_types.shape != building_mask.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted_types.shape} "
            f"vs building_mask {building_mask.shape}"
        )

    pred_bld = np.isin(predicted_types, list(building_codes))
    truth = building_mask.astype(bool)

    tp = int(np.sum(pred_bld & truth))
    fp = int(np.sum(pred_bld & ~truth))
    fn = int(np.sum(~pred_bld & truth))
    tn = int(np.sum(~pred_bld & ~truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "accuracy": round(accuracy, 4),
        "building_pixels_truth": int(truth.sum()),
        "building_pixels_pred": int(pred_bld.sum()),
    }

    log.info(
        "Evaluation: P=%.3f R=%.3f F1=%.3f IoU=%.3f  "
        "(TP=%d FP=%d FN=%d TN=%d)",
        precision, recall, f1, iou, tp, fp, fn, tn,
    )
    return result


# ---------------------------------------------------------------------------
# 5. Bonus: per-class breakdown
# ---------------------------------------------------------------------------

def classification_confusion_by_type(
    predicted_types: np.ndarray,
    building_mask: np.ndarray,
) -> dict[str, dict]:
    """For each predicted type code, count pixels inside/outside buildings.

    Returns dict mapping type_name → {"inside_building": int, "outside_building": int}.
    Useful for seeing which types are confused with buildings.
    """
    # Import locally to avoid circular dependency
    try:
        from object_classifier import OBJECT_TYPE_NAMES
    except ImportError:
        OBJECT_TYPE_NAMES = {}

    truth = building_mask.astype(bool)
    codes = np.unique(predicted_types)
    result = {}
    for code in codes:
        code_mask = predicted_types == code
        name = OBJECT_TYPE_NAMES.get(int(code), f"type_{code}")
        result[name] = {
            "inside_building": int(np.sum(code_mask & truth)),
            "outside_building": int(np.sum(code_mask & ~truth)),
        }
    return result
