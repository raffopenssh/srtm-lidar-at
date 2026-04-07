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

import tile_index as ti

log = logging.getLogger(__name__)

# GDAL env settings for efficient remote reads
GDAL_ENV = {
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
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
    url = ti.get_tile_url(layer, tile[0], tile[1], dataset)
    log.info(f"Reading {layer} tile N{tile[0]}E{tile[1]} window [{min_e:.0f},{min_n:.0f}]-[{max_e:.0f},{max_n:.0f}]")

    with rasterio.open(url) as ds:
        window = from_bounds(min_e, min_n, max_e, max_n, ds.transform)
        # Clamp window to dataset bounds
        window = window.intersection(Window(0, 0, ds.width, ds.height))
        data = ds.read(1, window=window).astype(np.float32)
        transform = ds.window_transform(window)
        nodata = ds.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
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
