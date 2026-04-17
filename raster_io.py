"""Windowed reading of remote BEV ALS rasters via /vsicurl/.

Reads only the pixels needed for a given geometry, avoiding downloading
entire 12 GB tiles.  Supports reading from multiple tiles when a geometry
spans tile boundaries.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.windows import from_bounds, Window
from rasterio.transform import from_bounds as transform_from_bounds
from shapely.geometry import box, mapping
import shapely

import hashlib
from pathlib import Path

import tile_index as ti

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disk cache for windowed reads — avoids repeated HTTP range requests to BEV
# for the same (layer, tile, window).  Especially useful for the processor
# where retries / restarts would otherwise re-fetch everything.
# ---------------------------------------------------------------------------

BEV_CACHE_DIR = Path("data/austria_processor/bev_tile_cache")
BEV_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(layer: str, tile: tuple, min_e: float, min_n: float,
               max_e: float, max_n: float, dataset: str) -> str:
    """Deterministic cache key for a windowed tile read."""
    raw = f"{layer}|N{tile[0]}E{tile[1]}|{dataset}|{min_e:.0f},{min_n:.0f},{max_e:.0f},{max_n:.0f}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return BEV_CACHE_DIR / f"{key}.npz"

# GDAL env settings for efficient remote reads
GDAL_ENV = {
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_CONNECTTIMEOUT": "10",
    "GDAL_HTTP_MAX_RETRY": "3",        # don't let GDAL retry forever internally
    "GDAL_HTTP_RETRY_DELAY": "2",       # 2s between GDAL-level retries
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(128 * 1024 * 1024),  # 128 MB cache
    "GDAL_CACHEMAX": "512",
}


def _apply_gdal_env():
    for k, v in GDAL_ENV.items():
        os.environ.setdefault(k, v)

_apply_gdal_env()


def read_window_bbox(
    layer: str,
    min_e: float, min_n: float, max_e: float, max_n: float,
    dataset: str = ti.DEFAULT_DATASET,
    pad: float = 5.0,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Read raster data for a bbox in EPSG:3035.

    Returns (data_2d, transform, crs).  Data is float32, nodata replaced with NaN.
    If bbox spans multiple tiles, they are mosaicked.
    """
    min_e -= pad
    min_n -= pad
    max_e += pad
    max_n += pad

    tiles = ti.find_tiles_for_bbox(min_e, min_n, max_e, max_n)
    if not tiles:
        raise ValueError(f"No tiles cover bbox E[{min_e},{max_e}] N[{min_n},{max_n}]")

    if len(tiles) == 1:
        return _read_single_tile(layer, tiles[0], min_e, min_n, max_e, max_n, dataset)
    else:
        return _read_multi_tile(layer, tiles, min_e, min_n, max_e, max_n, dataset)


def _read_single_tile(
    layer: str, tile: tuple[int, int],
    min_e: float, min_n: float, max_e: float, max_n: float,
    dataset: str,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    # --- Check disk cache first ---
    ck = _cache_key(layer, tile, min_e, min_n, max_e, max_n, dataset)
    cp = _cache_path(ck)
    if cp.exists():
        try:
            cached = np.load(str(cp), allow_pickle=False)
            tf_arr = cached["transform"]
            tf = rasterio.transform.Affine(*tf_arr[:6])
            log.info("%s N%dE%d window [%.0f,%.0f]-[%.0f,%.0f] → cache hit",
                     layer, tile[0], tile[1], min_e, min_n, max_e, max_n)
            return cached["data"], tf, rasterio.crs.CRS.from_epsg(3035)
        except Exception as e:
            log.warning("Corrupt BEV cache %s: %s", cp.name, e)
            cp.unlink(missing_ok=True)

    from bev_retry import open_with_retry
    url = ti.get_tile_url(layer, tile[0], tile[1], dataset)
    label = f"{layer} N{tile[0]}E{tile[1]}"
    log.info(f"Reading {label} window [{min_e:.0f},{min_n:.0f}]-[{max_e:.0f},{max_n:.0f}]")

    with open_with_retry(url, caller=label) as ds:
        window = from_bounds(min_e, min_n, max_e, max_n, ds.transform)
        # Clamp window to dataset bounds
        window = window.intersection(Window(0, 0, ds.width, ds.height))
        data = ds.read(1, window=window).astype(np.float32)
        transform = ds.window_transform(window)
        nodata = ds.nodata
        if nodata is not None:
            data[data == nodata] = np.nan

        # --- Save to disk cache ---
        try:
            tf_arr = np.array([transform.a, transform.b, transform.c,
                               transform.d, transform.e, transform.f])
            np.savez_compressed(str(cp), data=data, transform=tf_arr)
        except Exception as e:
            log.warning("Failed to cache %s: %s", label, e)

        return data, transform, ds.crs


def _read_multi_tile(
    layer: str, tiles: list[tuple[int, int]],
    min_e: float, min_n: float, max_e: float, max_n: float,
    dataset: str,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Mosaic multiple tiles into one array for the given bbox."""
    width = int(np.ceil(max_e - min_e))
    height = int(np.ceil(max_n - min_n))
    out = np.full((height, width), np.nan, dtype=np.float32)
    out_transform = rasterio.transform.from_origin(min_e, max_n, ti.TILE_RES, ti.TILE_RES)
    crs = None

    for tile in tiles:
        try:
            data, tf, tile_crs = _read_single_tile(
                layer, tile, min_e, min_n, max_e, max_n, dataset
            )
            crs = tile_crs
            # Calculate offset in output array
            # The tile data transform tells us where it sits
            col_off = int(round((tf.c - out_transform.c) / ti.TILE_RES))
            row_off = int(round((out_transform.f - tf.f) / ti.TILE_RES))
            h, w = data.shape
            # Clamp to output bounds
            src_row = max(0, -row_off)
            src_col = max(0, -col_off)
            dst_row = max(0, row_off)
            dst_col = max(0, col_off)
            copy_h = min(h - src_row, height - dst_row)
            copy_w = min(w - src_col, width - dst_col)
            if copy_h > 0 and copy_w > 0:
                patch = data[src_row:src_row+copy_h, src_col:src_col+copy_w]
                target = out[dst_row:dst_row+copy_h, dst_col:dst_col+copy_w]
                mask = ~np.isnan(patch)
                target[mask] = patch[mask]
        except Exception as e:
            log.warning(f"Failed to read tile N{tile[0]}E{tile[1]}: {e}")

    return out, out_transform, crs or rasterio.crs.CRS.from_epsg(3035)


def read_masked(
    layer: str,
    geom_3035: shapely.geometry.base.BaseGeometry,
    dataset: str = ti.DEFAULT_DATASET,
    pad: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Read raster data clipped to a geometry in EPSG:3035.

    Returns (data, mask, transform, crs) where mask is True for valid pixels
    inside the geometry.
    """
    bounds = geom_3035.bounds  # (minx, miny, maxx, maxy) = (min_e, min_n, max_e, max_n)
    data, transform, crs = read_window_bbox(
        layer, bounds[0], bounds[1], bounds[2], bounds[3], dataset, pad
    )

    # Create mask from geometry
    rows, cols = data.shape
    from rasterio.features import geometry_mask
    gmask = geometry_mask(
        [mapping(geom_3035)],
        out_shape=(rows, cols),
        transform=transform,
        invert=True,  # True = inside geometry
    )
    valid = gmask & ~np.isnan(data)
    return data, valid, transform, crs


def read_dtm_dsm(
    geom_3035: shapely.geometry.base.BaseGeometry,
    dataset: str = ti.DEFAULT_DATASET,
    pad: float = 5.0,
) -> dict:
    """Read both DTM and DSM for a geometry, compute nDSM (normalised heights).

    Returns dict with dtm, dsm, ndsm arrays, mask, transform, crs.
    """
    bounds = geom_3035.bounds
    dtm, tf, crs = read_window_bbox("DTM", bounds[0], bounds[1], bounds[2], bounds[3], dataset, pad)
    dsm, _, _ = read_window_bbox("DSM", bounds[0], bounds[1], bounds[2], bounds[3], dataset, pad)

    # Ensure same shape
    min_h = min(dtm.shape[0], dsm.shape[0])
    min_w = min(dtm.shape[1], dsm.shape[1])
    dtm = dtm[:min_h, :min_w]
    dsm = dsm[:min_h, :min_w]

    # Geometry mask
    from rasterio.features import geometry_mask
    gmask = geometry_mask(
        [mapping(geom_3035)],
        out_shape=(min_h, min_w),
        transform=tf,
        invert=True,
    )
    valid = gmask & ~np.isnan(dtm) & ~np.isnan(dsm)

    # Normalised DSM = DSM - DTM (object heights above ground)
    ndsm = np.where(valid, dsm - dtm, np.nan)
    # Clamp small negative values to 0 (noise)
    ndsm = np.where((ndsm < 0) & valid, 0.0, ndsm)

    return {
        "dtm": dtm, "dsm": dsm, "ndsm": ndsm,
        "mask": valid, "transform": tf, "crs": crs,
        "shape": (min_h, min_w),
    }


def read_multi_date_ndsm(
    geom_3035: shapely.geometry.base.BaseGeometry,
    dates: list[str] | None = None,
    pad: float = 5.0,
) -> dict:
    """Read nDSM for multiple ALS dates, aligned to a common grid.

    Loads DTM+DSM for each date, computes nDSM, and aligns all arrays
    to the smallest common extent.  This enables multi-temporal stability
    analysis: pixels that maintain the same height across 2-3 years are
    almost certainly built structures, while changing pixels are vegetation.

    Parameters
    ----------
    geom_3035 : shapely geometry (EPSG:3035)
        Area of interest.
    dates : list of str, optional
        Dataset date keys to load.  Default: all 3 available dates.
    pad : float
        Padding in metres around the geometry.

    Returns
    -------
    dict with keys:
        ndsm_by_date  – {date_str: 2D float32 array}
        dtm           – DTM from the latest date (reference)
        dsm           – DSM from the latest date (reference)
        mask          – boolean valid-data mask (intersection of all dates)
        transform     – rasterio Affine
        crs           – rasterio CRS
        shape         – (rows, cols)
        temporal_std  – 2D float32: per-pixel std of nDSM across dates
        temporal_range – 2D float32: per-pixel max-min of nDSM across dates
        dates_loaded  – list of date strings actually loaded
    """
    if dates is None:
        dates = sorted(ti.DATASETS.keys())
    for d in dates:
        if d not in ti.DATASETS:
            raise ValueError(f"Unknown dataset {d!r}. Available: {sorted(ti.DATASETS)}")

    # Load all dates
    all_data = {}
    for d in dates:
        log.info("Reading DTM+DSM for temporal analysis, date %s", d)
        try:
            all_data[d] = read_dtm_dsm(geom_3035, dataset=d, pad=pad)
        except Exception as e:
            log.warning("Failed to load date %s: %s", d, e)

    if not all_data:
        raise ValueError("Could not load any LIDAR dates")
    if len(all_data) < 2:
        log.warning("Only %d date(s) loaded — temporal analysis needs ≥2", len(all_data))

    # Find common shape (smallest extent)
    loaded_dates = sorted(all_data.keys())
    shapes = [all_data[d]["shape"] for d in loaded_dates]
    min_h = min(s[0] for s in shapes)
    min_w = min(s[1] for s in shapes)

    # Use latest date as reference
    ref_date = loaded_dates[-1]
    ref = all_data[ref_date]

    # Align and extract nDSM
    ndsm_by_date = {}
    combined_mask = np.ones((min_h, min_w), dtype=bool)
    for d in loaded_dates:
        dd = all_data[d]
        ndsm_d = dd["ndsm"][:min_h, :min_w].copy()
        mask_d = dd["mask"][:min_h, :min_w]
        ndsm_d[~mask_d] = np.nan
        ndsm_by_date[d] = ndsm_d
        combined_mask &= mask_d

    # Stack for temporal statistics
    ndsm_stack = np.stack([ndsm_by_date[d] for d in loaded_dates], axis=0)  # (N, H, W)

    with np.errstate(all='ignore'):
        temporal_std = np.where(
            combined_mask,
            np.nanstd(ndsm_stack, axis=0),
            np.nan
        )
        temporal_range = np.where(
            combined_mask,
            np.nanmax(ndsm_stack, axis=0) - np.nanmin(ndsm_stack, axis=0),
            np.nan
        )

    return {
        "ndsm_by_date": ndsm_by_date,
        "dtm": ref["dtm"][:min_h, :min_w],
        "dsm": ref["dsm"][:min_h, :min_w],
        "ndsm": ref["ndsm"][:min_h, :min_w],
        "mask": combined_mask,
        "transform": ref["transform"],
        "crs": ref["crs"],
        "shape": (min_h, min_w),
        "temporal_std": temporal_std,
        "temporal_range": temporal_range,
        "dates_loaded": loaded_dates,
    }
