"""Hansen Global Forest Change (GFC-2024-v1.12) integration.

Reads forest cover, loss, and gain data from Google's Hansen GFC tiles
via HTTP range requests. Used to calibrate and validate forest loss
(tree_loss) detection from BEV LIDAR time series.

Data:
  - treecover2000: % canopy cover in 2000 (uint8, 0-100)
  - lossyear: year of loss (uint8, 0=none, 1-24 = 2001-2024)
  - gain: forest gain 2000-2012 (uint8, 0/1)
  - datamask: 0=nodata, 1=land, 2=water
  - first/last: first/last Landsat year with observations

All tiles are WGS84 (EPSG:4326) at ~30m resolution.
Tile 50N_010E covers Austria.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS

log = logging.getLogger(__name__)

CACHE_DIR = Path("/tmp/hansen_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12"
TILE = "50N_010E"

LAYER_URLS = {
    "treecover2000": f"{BASE_URL}/Hansen_GFC-2024-v1.12_treecover2000_{TILE}.tif",
    "gain":          f"{BASE_URL}/Hansen_GFC-2024-v1.12_gain_{TILE}.tif",
    "lossyear":      f"{BASE_URL}/Hansen_GFC-2024-v1.12_lossyear_{TILE}.tif",
    "datamask":      f"{BASE_URL}/Hansen_GFC-2024-v1.12_datamask_{TILE}.tif",
    "first":         f"{BASE_URL}/Hansen_GFC-2024-v1.12_first_{TILE}.tif",
    "last":          f"{BASE_URL}/Hansen_GFC-2024-v1.12_last_{TILE}.tif",
}

GDAL_ENV = {
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "10000000",
}


def _cache_key(layer: str, bbox: tuple) -> str:
    h = hashlib.md5(f"{layer}_{bbox}".encode()).hexdigest()[:12]
    return f"{layer}_{h}"


def _read_layer_window(layer: str, bbox_wgs84: tuple) -> tuple[np.ndarray, rasterio.Affine]:
    """Read a windowed portion of a Hansen layer.

    Parameters
    ----------
    layer : one of treecover2000, gain, lossyear, datamask, first, last
    bbox_wgs84 : (west, south, east, north) in EPSG:4326

    Returns
    -------
    (array, transform) - 2D array + window transform
    """
    west, south, east, north = bbox_wgs84
    cache_path = CACHE_DIR / f"{_cache_key(layer, bbox_wgs84)}.npz"

    if cache_path.exists():
        cached = np.load(str(cache_path), allow_pickle=True)
        tf_flat = cached["transform"]
        tf = rasterio.Affine(*tf_flat[:6])
        return cached["data"], tf

    url = LAYER_URLS[layer]
    vsicurl = f"/vsicurl/{url}"
    log.info("Hansen: reading %s for bbox %s", layer, bbox_wgs84)

    for k, v in GDAL_ENV.items():
        os.environ[k] = v

    from bev_retry import read_with_retry
    data, tf = read_with_retry(
        vsicurl,
        read_fn=lambda src: (
            src.read(1, window=from_bounds(west, south, east, north, src.transform)),
            src.window_transform(from_bounds(west, south, east, north, src.transform)),
        ),
        caller=f"Hansen {layer}",
    )

    # Cache
    tf_flat = np.array([tf.a, tf.b, tf.c, tf.d, tf.e, tf.f])
    np.savez_compressed(str(cache_path), data=data, transform=tf_flat)
    log.info("Hansen: cached %s (%dx%d)", layer, data.shape[1], data.shape[0])

    return data, tf


def read_hansen_window(
    bbox_wgs84: tuple,
    layers: list[str] | None = None,
) -> dict:
    """Read windowed Hansen data for a bounding box.

    Parameters
    ----------
    bbox_wgs84 : (west, south, east, north)
    layers : which layers to read (default: all)

    Returns
    -------
    dict with keys per layer (array), plus 'transform' and 'shape'
    """
    if layers is None:
        layers = list(LAYER_URLS.keys())

    result = {}
    tf = None
    for layer in layers:
        data, layer_tf = _read_layer_window(layer, bbox_wgs84)
        result[layer] = data
        if tf is None:
            tf = layer_tf

    first_arr = result[layers[0]]
    result["transform"] = tf
    result["shape"] = first_arr.shape
    result["crs"] = CRS.from_epsg(4326)
    return result


def resample_to_target(
    hansen_data: dict,
    target_transform,
    target_shape: tuple[int, int],
) -> dict:
    """Resample Hansen 30m WGS84 data to a 1m EPSG:3035 target grid.

    Returns dict with same layer keys, resampled.
    """
    src_crs = CRS.from_epsg(4326)
    dst_crs = CRS.from_epsg(3035)
    result = {}

    for key in ["treecover2000", "lossyear", "gain", "datamask", "first", "last"]:
        src = hansen_data.get(key)
        if src is None:
            continue
        dst = np.zeros(target_shape, dtype=src.dtype)
        reproject(
            source=src,
            destination=dst,
            src_transform=hansen_data["transform"],
            src_crs=src_crs,
            dst_transform=target_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        result[key] = dst

    return result


def get_forest_prior(
    bbox_wgs84: tuple,
    target_transform,
    target_shape: tuple[int, int],
    min_treecover: int = 25,
) -> dict:
    """Get Hansen forest prior resampled to target grid.

    Returns
    -------
    dict with:
        was_forest_2000 : bool mask
        loss_year : uint8 (0=no loss, 1-24 = 2001-2024)
        gain : bool mask
        current_forest : bool - estimated current forest
        treecover2000 : uint8 (0-100)
    """
    hansen = read_hansen_window(
        bbox_wgs84,
        layers=["treecover2000", "lossyear", "gain", "datamask"],
    )
    resampled = resample_to_target(hansen, target_transform, target_shape)

    tc = resampled.get("treecover2000", np.zeros(target_shape, dtype=np.uint8))
    ly = resampled.get("lossyear", np.zeros(target_shape, dtype=np.uint8))
    gn = resampled.get("gain", np.zeros(target_shape, dtype=np.uint8))
    dm = resampled.get("datamask", np.ones(target_shape, dtype=np.uint8))

    was_forest = (tc >= min_treecover) & (dm == 1)
    lost = ly > 0
    gained = gn > 0
    current_forest = (was_forest & ~lost) | gained

    log.info(
        "Hansen prior: was_forest=%d px, lost=%d px, gained=%d px, current=%d px",
        int(was_forest.sum()), int(lost.sum()), int(gained.sum()), int(current_forest.sum()),
    )

    return {
        "was_forest_2000": was_forest,
        "loss_year": ly,
        "gain": gained,
        "current_forest": current_forest,
        "treecover2000": tc,
    }


def get_forest_loss_mask(
    bbox_wgs84: tuple,
    target_transform,
    target_shape: tuple[int, int],
    min_treecover: int = 25,
    loss_year_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """Boolean mask at 1m resolution indicating Hansen forest loss.

    Parameters
    ----------
    loss_year_range : (start, end) inclusive, years 1-24 (2001-2024).
                      None = any year.
    """
    prior = get_forest_prior(bbox_wgs84, target_transform, target_shape, min_treecover)
    ly = prior["loss_year"]
    was_f = prior["was_forest_2000"]

    if loss_year_range is not None:
        start, end = loss_year_range
        loss = was_f & (ly >= start) & (ly <= end)
    else:
        loss = was_f & (ly > 0)

    return loss


def calibrate_tree_loss(
    objects: list,
    labels: np.ndarray,
    hansen_prior: dict,
    observation_year: int | None = None,
) -> list:
    """Improve tree_loss detection using Hansen forest loss data.

    Modifies objects in-place and returns the list.

    Parameters
    ----------
    observation_year : int or None
        Calendar year of the LIDAR observation (e.g. 2024).  "Recent" loss
        is scoped to the 5 years ending at this year.  Defaults to 2024.

    Logic:
    1. tree_loss objects overlapping Hansen loss → boost confidence
    2. tree/shrub/grass objects on recent Hansen loss → reclassify to tree_loss
    3. tree_loss objects with NO Hansen support → reduce confidence
    """
    _obs = observation_year or 2024
    _recent_start = max(_obs - 2000 - 5, 1)  # Hansen code for obs_year-5
    _recent_end   = min(_obs - 2000, 24)      # Hansen code for obs_year
    loss_mask = hansen_prior["loss_year"] > 0
    was_forest = hansen_prior["was_forest_2000"]
    # Recent loss scoped to [obs_year-5 .. obs_year]
    recent_loss = (
        (hansen_prior["loss_year"] >= _recent_start)
        & (hansen_prior["loss_year"] <= _recent_end)
        & was_forest
    )

    modified = 0
    for obj in objects:
        seg_mask = labels == obj.obj_id
        seg_pixels = int(seg_mask.sum())
        if seg_pixels < 4:
            continue

        loss_overlap = int((seg_mask & loss_mask).sum())
        loss_frac = loss_overlap / seg_pixels

        recent_overlap = int((seg_mask & recent_loss).sum())
        recent_frac = recent_overlap / seg_pixels

        forest_overlap = int((seg_mask & was_forest).sum())
        forest_frac = forest_overlap / seg_pixels

        if obj.obj_type == "tree_loss":
            if loss_frac > 0.15:
                # Hansen confirms forest loss - boost confidence
                obj.confidence = min(obj.confidence + 0.15, 0.95)
                modified += 1
            elif forest_frac < 0.1:
                # Not even forest area according to Hansen → downgrade
                obj.confidence = max(obj.confidence - 0.2, 0.2)
                modified += 1
            # If was forest but no Hansen loss: could be very recent,
            # keep existing confidence (our LIDAR is more recent than Hansen)

        elif obj.obj_type in ("tree", "shrub", "grass", "bare_soil", "crop"):
            # Vegetation/ground on recent Hansen loss area → possible tree_loss
            if recent_frac > 0.15 and forest_frac > 0.15:
                # Hansen loss signal on former forest.
                # Evidence of clearing: temporal change OR low current canopy
                # (temporal data may be absent if only one LIDAR date exists).
                h_change_abs = abs(obj.height_change)
                has_temporal_evidence = (
                    h_change_abs > 0.5 or obj.temporal_stability < 0.6
                )
                has_height_evidence = obj.height_mean < 5.0  # young regrowth can be up to 5m
                if has_temporal_evidence or has_height_evidence:
                    from object_segmentation import OBJECT_TYPES
                    obj.obj_type = "tree_loss"
                    obj.type_code = OBJECT_TYPES["tree_loss"]
                    obj.confidence = min(0.6 + recent_frac * 0.2, 0.85)
                    obj.is_manmade = True
                    modified += 1
                elif recent_frac > 0.5:
                    # Majority Hansen loss — reclassify with lower confidence
                    from object_segmentation import OBJECT_TYPES
                    obj.obj_type = "tree_loss"
                    obj.type_code = OBJECT_TYPES["tree_loss"]
                    obj.confidence = min(0.45 + recent_frac * 0.15, 0.70)
                    obj.is_manmade = True
                    modified += 1

    log.info("Hansen calibration: modified %d objects", modified)
    return objects


def evaluate_forest_loss(
    objects: list,
    labels: np.ndarray,
    hansen_prior: dict,
    observation_year: int | None = None,
) -> dict:
    """Compare tree_loss detections against Hansen loss.

    Parameters
    ----------
    observation_year : int or None
        Calendar year of the observation.  Only Hansen loss *up to* this
        year is used as reference.  Defaults to 2024.

    Returns precision, recall, F1 metrics.
    """
    _obs = observation_year or 2024
    _max_code = min(_obs - 2000, 24)
    h, w = labels.shape
    loss_mask = (
        (hansen_prior["loss_year"] > 0)
        & (hansen_prior["loss_year"] <= _max_code)
        & hansen_prior["was_forest_2000"]
    )

    # Build our detection mask
    our_loss = np.zeros((h, w), dtype=bool)
    for obj in objects:
        if obj.obj_type == "tree_loss":
            our_loss[labels == obj.obj_id] = True

    # Crop to common shape
    mh = min(h, loss_mask.shape[0])
    mw = min(w, loss_mask.shape[1])
    loss_ref = loss_mask[:mh, :mw]
    loss_det = our_loss[:mh, :mw]

    tp = int((loss_ref & loss_det).sum())
    fp = int((~loss_ref & loss_det).sum())
    fn = int((loss_ref & ~loss_det).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    iou = tp / max(tp + fp + fn, 1)

    result = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "iou": round(iou, 3),
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "hansen_loss_pixels": int(loss_ref.sum()),
        "our_clearcut_pixels": int(loss_det.sum()),
    }
    log.info(
        "Hansen eval: P=%.3f R=%.3f F1=%.3f IoU=%.3f (tp=%d fp=%d fn=%d)",
        precision, recall, f1, iou, tp, fp, fn,
    )
    return result
