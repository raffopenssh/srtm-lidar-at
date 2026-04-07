"""Windowed reading of remote BEV orthophoto (DOP) rasters via /vsicurl/.

Reads only the pixels needed for a given bounding box from Austria's national
orthophoto tiles (~8 GB each, 250k×250k pixels at 20 cm), avoiding full
downloads.  Supports multi-tile mosaicking and resampling from native 20 cm
resolution to any target (typically 1 m to match ALS grids).

Two data families are supported:

1. **DOP 50 km tiles** – RGB (uint8), EPSG:3035, same 50 km grid as ALS.
   URL pattern:  ``DOP_CRS3035RES50000mN{n}E{e}_{date}.tif``

2. **RGBI Operate** – separate RGB and NIR files per survey area, various
   Austrian CRS (EPSG:31254/31255/31256).  These are irregularly shaped and
   *not* on the 50 km grid.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin as transform_from_origin
from rasterio.windows import Window, from_bounds

import tile_index as ti

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GDAL environment (shared with raster_io)
# ---------------------------------------------------------------------------

GDAL_ENV = {
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(128 * 1024 * 1024),  # 128 MB
    "GDAL_CACHEMAX": "512",
}


def _apply_gdal_env() -> None:
    for k, v in GDAL_ENV.items():
        os.environ.setdefault(k, v)


_apply_gdal_env()

# ---------------------------------------------------------------------------
# Orthophoto dataset registry
# ---------------------------------------------------------------------------

#: Native ground sampling distance of the DOP tiles (metres).
DOP_NATIVE_RES = 0.2

#: Pixel count per axis for a 50 km tile at 20 cm resolution.
DOP_TILE_PIXELS = 250_000

#: DOP 50 km tile datasets keyed by publication date.
ORTHO_DATASETS: dict[str, dict] = {
    "20220128": {
        "base_url": "https://data.bev.gv.at/download/DOP/20220128/",
        "filename_pattern": "DOP_CRS3035RES50000mN{n}E{e}_20220128.tif",
        "crs": "EPSG:3035",
        "bands": 3,       # R, G, B
        "dtype": "uint8",
        "resolution": DOP_NATIVE_RES,
    },
}

DEFAULT_ORTHO_DATASET = "20220128"

#: RGBI operate series – keyed by series publication date.
RGBI_SERIES: dict[str, dict] = {
    "20221027": {
        "base_url": "https://data.bev.gv.at/download/DOP/20221027/",
        "description": "Operate from 2018-2021",
    },
    "20240625": {
        "base_url": "https://data.bev.gv.at/download/DOP/20240625/",
        "description": "Operate from 2021-2023",
    },
    "20250415": {
        "base_url": "https://data.bev.gv.at/download/DOP/20250415/",
        "description": "Operate from 2022-2024",
    },
}

# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def get_dop_tile_url(
    northing: int,
    easting: int,
    dataset: str = DEFAULT_ORTHO_DATASET,
) -> str:
    """Return a ``/vsicurl/`` URL for a DOP 50 km tile.

    Parameters
    ----------
    northing, easting:
        Tile origin in EPSG:3035 metres (must be in ``TILE_COORDS``).
    dataset:
        Publication-date key into :data:`ORTHO_DATASETS`.

    Raises
    ------
    ValueError
        If dataset or tile coordinates are invalid.
    """
    if dataset not in ORTHO_DATASETS:
        raise ValueError(
            f"Unknown ortho dataset {dataset!r}. "
            f"Available: {sorted(ORTHO_DATASETS)}"
        )
    if (northing, easting) not in ti.TILE_COORDS:
        raise ValueError(
            f"No tile at N{northing}E{easting}. "
            f"Use tile_index.find_tiles_for_bbox() to discover valid tiles."
        )

    ds = ORTHO_DATASETS[dataset]
    filename = ds["filename_pattern"].format(n=northing, e=easting)
    return f"/vsicurl/{ds['base_url']}{filename}"


def get_rgbi_url(
    operat: str,
    band_type: str = "RGB",
    series: str = "20250415",
) -> str:
    """Return a ``/vsicurl/`` URL for an RGBI operate mosaic.

    Parameters
    ----------
    operat:
        Operate identifier, e.g. ``"2024470"``.
    band_type:
        ``"RGB"`` or ``"NIR"``.
    series:
        Publication-date key into :data:`RGBI_SERIES`.
    """
    band_type = band_type.upper()
    if band_type not in ("RGB", "NIR"):
        raise ValueError(f"band_type must be 'RGB' or 'NIR', got {band_type!r}")
    if series not in RGBI_SERIES:
        raise ValueError(
            f"Unknown RGBI series {series!r}. Available: {sorted(RGBI_SERIES)}"
        )

    base = RGBI_SERIES[series]["base_url"]
    filename = f"{operat}_Mosaik_{band_type}.tif"
    return f"/vsicurl/{base}{filename}"


# ---------------------------------------------------------------------------
# Internal: pick the best overview level for a target resolution
# ---------------------------------------------------------------------------


def _best_overview_level(
    ds: rasterio.DatasetReader,
    target_res: float,
) -> Optional[int]:
    """Return the best overview index for *target_res*, or None for full-res.

    Picks the coarsest overview whose resolution is still ≤ target_res so that
    GDAL reads from a smaller pyramid level, saving bandwidth.
    """
    if not ds.overviews(1):
        return None

    native_res = abs(ds.transform.a)  # pixel size in CRS units
    best: Optional[int] = None
    best_factor: int = 1

    for idx, factor in enumerate(ds.overviews(1)):
        overview_res = native_res * factor
        if overview_res <= target_res:
            if factor > best_factor:
                best = idx
                best_factor = factor

    return best_factor if best is not None else None


# ---------------------------------------------------------------------------
# Single-tile reader (internal)
# ---------------------------------------------------------------------------


def _read_ortho_single_tile(
    tile: tuple[int, int],
    min_e: float,
    min_n: float,
    max_e: float,
    max_n: float,
    resolution: float,
    dataset: str,
    resampling: Resampling,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Read RGB bands from one DOP tile, resampled to *resolution*.

    Returns
    -------
    (data, transform, crs)
        *data* has shape ``(bands, H, W)`` as uint8.
    """
    url = get_dop_tile_url(tile[0], tile[1], dataset)
    log.info(
        "Reading DOP tile N%dE%d window [%.0f,%.0f]-[%.0f,%.0f] @ %.2fm",
        tile[0], tile[1], min_e, min_n, max_e, max_n, resolution,
    )

    with rasterio.open(url) as ds:
        native_res = abs(ds.transform.a)
        num_bands = ds.count

        # Compute the pixel window at native resolution
        window = from_bounds(min_e, min_n, max_e, max_n, ds.transform)
        window = window.intersection(Window(0, 0, ds.width, ds.height))

        if window.width < 1 or window.height < 1:
            raise ValueError(
                f"Empty window for tile N{tile[0]}E{tile[1]} with bbox "
                f"[{min_e},{min_n}]-[{max_e},{max_n}]"
            )

        # Compute output size at target resolution
        # Use the *actual* geographic extent of the clamped window
        win_transform = ds.window_transform(window)
        win_width_m = window.width * native_res
        win_height_m = window.height * native_res
        out_cols = max(1, int(round(win_width_m / resolution)))
        out_rows = max(1, int(round(win_height_m / resolution)))

        # Read with out_shape for on-the-fly decimation (GDAL uses overviews
        # automatically when reading with an out_shape smaller than window).
        data = ds.read(
            list(range(1, num_bands + 1)),
            window=window,
            out_shape=(num_bands, out_rows, out_cols),
            resampling=resampling,
        )

        # Build the output transform anchored at the window's top-left corner
        out_transform = transform_from_origin(
            win_transform.c,   # left  (easting)
            win_transform.f,   # top   (northing)
            resolution,
            resolution,
        )

        return data, out_transform, ds.crs


# ---------------------------------------------------------------------------
# Multi-tile mosaicking (internal)
# ---------------------------------------------------------------------------


def _read_ortho_multi_tile(
    tiles: list[tuple[int, int]],
    min_e: float,
    min_n: float,
    max_e: float,
    max_n: float,
    resolution: float,
    dataset: str,
    resampling: Resampling,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Mosaic multiple DOP tiles into one array for the given bbox."""
    out_cols = max(1, int(round((max_e - min_e) / resolution)))
    out_rows = max(1, int(round((max_n - min_n) / resolution)))
    n_bands = ORTHO_DATASETS.get(dataset, {}).get("bands", 3)
    out = np.zeros((n_bands, out_rows, out_cols), dtype=np.uint8)
    out_transform = transform_from_origin(min_e, max_n, resolution, resolution)
    crs: Optional[rasterio.crs.CRS] = None

    for tile in tiles:
        try:
            data, tf, tile_crs = _read_ortho_single_tile(
                tile, min_e, min_n, max_e, max_n,
                resolution, dataset, resampling,
            )
            crs = tile_crs

            # Calculate where this patch sits in the output mosaic
            col_off = int(round((tf.c - out_transform.c) / resolution))
            row_off = int(round((out_transform.f - tf.f) / resolution))
            _, h, w = data.shape

            # Clamp source / destination ranges
            src_row = max(0, -row_off)
            src_col = max(0, -col_off)
            dst_row = max(0, row_off)
            dst_col = max(0, col_off)
            copy_h = min(h - src_row, out_rows - dst_row)
            copy_w = min(w - src_col, out_cols - dst_col)

            if copy_h > 0 and copy_w > 0:
                patch = data[
                    :, src_row : src_row + copy_h, src_col : src_col + copy_w
                ]
                target = out[
                    :, dst_row : dst_row + copy_h, dst_col : dst_col + copy_w
                ]
                # Overwrite with non-zero (black = nodata for ortho)
                mask = np.any(patch > 0, axis=0)
                target[:, mask] = patch[:, mask]

        except Exception:
            log.warning(
                "Failed to read DOP tile N%dE%d", tile[0], tile[1],
                exc_info=True,
            )

    return out, out_transform, crs or rasterio.crs.CRS.from_epsg(3035)


# ---------------------------------------------------------------------------
# Public API – DOP 50 km grid reads
# ---------------------------------------------------------------------------


def read_ortho_window(
    min_e: float,
    min_n: float,
    max_e: float,
    max_n: float,
    resolution: float = 1.0,
    dataset: str = DEFAULT_ORTHO_DATASET,
    pad: float = 0.0,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Read RGB orthophoto for a bbox in EPSG:3035.

    This is the main entry point for reading DOP 50 km tiles.  The native
    20 cm pixels are resampled on-the-fly to *resolution* (default 1 m to
    match the ALS grid) using GDAL overview levels where available.

    Parameters
    ----------
    min_e, min_n, max_e, max_n:
        Bounding box in EPSG:3035 metres.
    resolution:
        Target pixel size in metres (default 1.0).
    dataset:
        Publication-date key (default ``"20220128"``).
    pad:
        Extra padding in metres added around the bbox.
    resampling:
        Rasterio resampling method (default ``Resampling.bilinear``).

    Returns
    -------
    (rgb, transform, crs)
        *rgb* is ``uint8`` with shape ``(3, H, W)``.
    """
    min_e -= pad
    min_n -= pad
    max_e += pad
    max_n += pad

    tiles = ti.find_tiles_for_bbox(min_e, min_n, max_e, max_n)
    if not tiles:
        raise ValueError(
            f"No DOP tiles cover bbox E[{min_e:.0f},{max_e:.0f}] "
            f"N[{min_n:.0f},{max_n:.0f}]"
        )

    log.info(
        "read_ortho_window: bbox [%.0f,%.0f]-[%.0f,%.0f] → %d tile(s) @ %.2fm",
        min_e, min_n, max_e, max_n, len(tiles), resolution,
    )

    if len(tiles) == 1:
        return _read_ortho_single_tile(
            tiles[0], min_e, min_n, max_e, max_n,
            resolution, dataset, resampling,
        )
    else:
        return _read_ortho_multi_tile(
            tiles, min_e, min_n, max_e, max_n,
            resolution, dataset, resampling,
        )


def read_ortho_rgb(
    min_e: float,
    min_n: float,
    max_e: float,
    max_n: float,
    dataset: str = DEFAULT_ORTHO_DATASET,
    pad: float = 0.0,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Convenience wrapper: read RGB resampled to 1 m (ALS grid).

    Equivalent to ``read_ortho_window(..., resolution=1.0)``.

    Returns
    -------
    (rgb, transform, crs)
        *rgb* is ``uint8`` with shape ``(3, H, W)``.
    """
    return read_ortho_window(
        min_e, min_n, max_e, max_n,
        resolution=1.0,
        dataset=dataset,
        pad=pad,
    )


# ---------------------------------------------------------------------------
# Public API – RGBI operate reads
# ---------------------------------------------------------------------------


def read_rgbi_operate(
    operat: str,
    min_e: float,
    min_n: float,
    max_e: float,
    max_n: float,
    series: str = "20250415",
    resolution: float = 1.0,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, Optional[np.ndarray], rasterio.transform.Affine, rasterio.crs.CRS]:
    """Read RGB and NIR bands from an RGBI operate for a bbox.

    The bbox coordinates must be in the operate's native CRS (one of
    EPSG:31254, 31255, 31256).  Inspect the file first if the CRS is unknown.

    Parameters
    ----------
    operat:
        Operate ID, e.g. ``"2024470"``.
    min_e, min_n, max_e, max_n:
        Bounding box in the operate's native CRS.
    series:
        Series date key into :data:`RGBI_SERIES`.
    resolution:
        Target pixel size in CRS units (metres).
    resampling:
        Rasterio resampling method.

    Returns
    -------
    (rgb, nir, transform, crs)
        *rgb* has shape ``(3, H, W)`` uint8.  *nir* has shape ``(H, W)`` uint8
        or ``None`` if the NIR file could not be read.
    """
    rgb_url = get_rgbi_url(operat, "RGB", series)
    nir_url = get_rgbi_url(operat, "NIR", series)

    log.info(
        "Reading RGBI operate %s (series %s) window [%.0f,%.0f]-[%.0f,%.0f]",
        operat, series, min_e, min_n, max_e, max_n,
    )

    rgb, transform, crs = _read_generic_window(
        rgb_url, min_e, min_n, max_e, max_n, resolution, resampling,
    )

    nir: Optional[np.ndarray] = None
    try:
        nir_data, _, _ = _read_generic_window(
            nir_url, min_e, min_n, max_e, max_n, resolution, resampling,
        )
        # NIR file has 1 band
        nir = nir_data[0] if nir_data.ndim == 3 else nir_data
    except Exception:
        log.warning(
            "Could not read NIR for operate %s (series %s)",
            operat, series, exc_info=True,
        )

    return rgb, nir, transform, crs


def _read_generic_window(
    url: str,
    min_e: float,
    min_n: float,
    max_e: float,
    max_n: float,
    resolution: float,
    resampling: Resampling,
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Read an arbitrary /vsicurl/ GeoTIFF window, resampled to *resolution*."""
    log.debug("Opening %s", url)

    with rasterio.open(url) as ds:
        native_res = abs(ds.transform.a)
        num_bands = ds.count

        window = from_bounds(min_e, min_n, max_e, max_n, ds.transform)
        window = window.intersection(Window(0, 0, ds.width, ds.height))

        if window.width < 1 or window.height < 1:
            raise ValueError(
                f"Empty window for {url} with bbox "
                f"[{min_e},{min_n}]-[{max_e},{max_n}]"
            )

        win_transform = ds.window_transform(window)
        win_width_m = window.width * native_res
        win_height_m = window.height * native_res
        out_cols = max(1, int(round(win_width_m / resolution)))
        out_rows = max(1, int(round(win_height_m / resolution)))

        data = ds.read(
            list(range(1, num_bands + 1)),
            window=window,
            out_shape=(num_bands, out_rows, out_cols),
            resampling=resampling,
        )

        out_transform = transform_from_origin(
            win_transform.c,
            win_transform.f,
            resolution,
            resolution,
        )

        return data, out_transform, ds.crs


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------


def compute_ndvi(
    rgb: np.ndarray,
    nir: np.ndarray,
) -> np.ndarray:
    """Normalised Difference Vegetation Index.

    NDVI = (NIR − R) / (NIR + R)

    Parameters
    ----------
    rgb:
        Shape ``(3, H, W)`` uint8 with bands [R, G, B].
    nir:
        Shape ``(H, W)`` uint8.

    Returns
    -------
    ndvi:
        Shape ``(H, W)`` float32 in [-1, 1].  NaN where denominator is zero.
    """
    red = rgb[0].astype(np.float32)
    nir_f = nir.astype(np.float32)
    denom = nir_f + red
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(denom > 0, (nir_f - red) / denom, np.nan)
    return ndvi.astype(np.float32)


def compute_spectral_indices(
    rgb: np.ndarray,
    nir: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Compute a suite of spectral indices from orthophoto bands.

    Parameters
    ----------
    rgb:
        Shape ``(3, H, W)`` uint8 with bands [R, G, B].
    nir:
        Optional shape ``(H, W)`` uint8.

    Returns
    -------
    dict
        Keys are index names, values are float32 arrays of shape ``(H, W)``.

        - ``"brightness"`` – mean of R, G, B  (0–255 range)
        - ``"green_ratio"`` – G / (R + G + B)  (0–1, NaN where sum == 0)
        - ``"rg_index"`` – (R − G) / (R + G)  (−1 to 1, NaN where sum == 0)
        - ``"ndvi"`` – (NIR − R) / (NIR + R)  (only if *nir* is provided)
    """
    r = rgb[0].astype(np.float32)
    g = rgb[1].astype(np.float32)
    b = rgb[2].astype(np.float32)

    rgb_sum = r + g + b
    rg_sum = r + g

    indices: dict[str, np.ndarray] = {}

    # Brightness
    indices["brightness"] = (rgb_sum / 3.0).astype(np.float32)

    # Green ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        green_ratio = np.where(rgb_sum > 0, g / rgb_sum, np.nan)
    indices["green_ratio"] = green_ratio.astype(np.float32)

    # Red-green index
    with np.errstate(divide="ignore", invalid="ignore"):
        rg_index = np.where(rg_sum > 0, (r - g) / rg_sum, np.nan)
    indices["rg_index"] = rg_index.astype(np.float32)

    # NDVI (requires NIR)
    if nir is not None:
        indices["ndvi"] = compute_ndvi(rgb, nir)

    return indices


# ---------------------------------------------------------------------------
# Convenience: read ortho aligned to an ALS result dict
# ---------------------------------------------------------------------------


def read_ortho_for_als(
    als_result: dict,
    dataset: str = DEFAULT_ORTHO_DATASET,
) -> np.ndarray:
    """Read an RGB orthophoto aligned to an ALS raster result.

    Parameters
    ----------
    als_result:
        Dict as returned by :func:`raster_io.read_dtm_dsm`, containing at
        least ``"transform"``, ``"crs"``, and ``"shape"`` keys.
    dataset:
        Ortho dataset key.

    Returns
    -------
    rgb:
        Shape ``(3, H, W)`` uint8, exactly matching *als_result["shape"]*.
    """
    tf = als_result["transform"]
    h, w = als_result["shape"]
    res = abs(tf.a)

    # Derive bbox from ALS transform + shape
    min_e = tf.c
    max_n = tf.f
    max_e = min_e + w * res
    min_n = max_n - h * res

    rgb, _, _ = read_ortho_window(
        min_e, min_n, max_e, max_n,
        resolution=res,
        dataset=dataset,
    )

    # Ensure exact shape match (rounding can cause ±1 pixel differences)
    if rgb.shape[1] != h or rgb.shape[2] != w:
        out = np.zeros((rgb.shape[0], h, w), dtype=rgb.dtype)
        copy_h = min(rgb.shape[1], h)
        copy_w = min(rgb.shape[2], w)
        out[:, :copy_h, :copy_w] = rgb[:, :copy_h, :copy_w]
        rgb = out

    return rgb
