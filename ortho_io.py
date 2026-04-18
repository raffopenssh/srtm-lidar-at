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

import hashlib
from pathlib import Path

import tile_index as ti

# === SECTION: Disk cache for ortho windowed reads ===
ORTHO_CACHE_DIR = Path("data/austria_processor/ortho_tile_cache")
ORTHO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)

# === SECTION: GDAL environment ===

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

# === SECTION: BEV retry import ===
from bev_retry import open_with_retry


def _apply_gdal_env() -> None:
    for k, v in GDAL_ENV.items():
        os.environ.setdefault(k, v)


_apply_gdal_env()

# === SECTION: Orthophoto dataset registry (URLs, tile grids) ===

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

# === SECTION: RGBI operate index ===
# bbox in WGS84 [lat_min, lon_min, lat_max, lon_max].
# Discovered from BEV CSW catalogue + verified HTTP availability.
# Each operate has separate RGB + NIR files. Newest first per area.

RGBI_OPERATES: dict[str, dict] = {
    # --- series 20250415 (2024 flights, newest) ---
    "2024150": {"series": "20250415", "crs": "EPSG:31256", "bbox_wgs84": [48.49597, 14.9363, 49.03636, 16.01072]},
    "2024250": {"series": "20250415", "crs": "EPSG:31255", "bbox_wgs84": [47.51278, 13.0169, 47.66768, 14.01495]},
    "2024260": {"series": "20250415", "crs": "EPSG:31255", "bbox_wgs84": [47.3285, 12.1186, 47.75278, 13.01837]},
    "2024350": {"series": "20250415", "crs": "EPSG:31256", "bbox_wgs84": [46.60296, 14.99683, 47.17265, 16.05465]},
    "2024450": {"series": "20250415", "crs": "EPSG:31256", "bbox_wgs84": [48.45779, 15.97477, 48.81885, 16.97557]},
    "2024460": {"series": "20250415", "crs": "EPSG:31256", "bbox_wgs84": [48.09893, 15.97708, 48.47719, 16.98824]},
    "2024470": {"series": "20250415", "crs": "EPSG:31255", "bbox_wgs84": [47.22059, 12.125, 47.35244, 13.01902]},
    # --- series 20240625 (2023 flights) ---
    "2023150": {"series": "20240625", "crs": "EPSG:31255", "bbox_wgs84": [48.09734, 12.72798, 48.78187, 14.07954]},
    "2023160": {"series": "20240625", "crs": "EPSG:31254", "bbox_wgs84": [46.74871, 10.05247, 47.21787, 11.14116]},
    "2023250": {"series": "20240625", "crs": "EPSG:31256", "bbox_wgs84": [48.00149, 14.96165, 48.52169, 16.01384]},
    "2023260": {"series": "20240625", "crs": "EPSG:31254", "bbox_wgs84": [47.19706, 10.05199, 47.59562, 11.14475]},
    "2023270": {"series": "20240625", "crs": "EPSG:31254", "bbox_wgs84": [46.84868, 9.50695, 47.60354, 10.25028]},
    "2023350": {"series": "20240625", "crs": "EPSG:31255", "bbox_wgs84": [48.17708, 13.98818, 48.80079, 15.02073]},
    "2023360": {"series": "20240625", "crs": "EPSG:31256", "bbox_wgs84": [47.50696, 14.97485, 48.02701, 16.01688]},
    "2023370": {"series": "20240625", "crs": "EPSG:31254", "bbox_wgs84": [46.90936, 11.10419, 47.24789, 11.67044]},
    "2023450": {"series": "20240625", "crs": "EPSG:31256", "bbox_wgs84": [47.14727, 14.98769, 47.53242, 16.01903]},
    "2023460": {"series": "20240625", "crs": "EPSG:31255", "bbox_wgs84": [47.64769, 12.72819, 48.11736, 14.02085]},
    "2023470": {"series": "20240625", "crs": "EPSG:31255", "bbox_wgs84": [47.60124, 13.981, 47.74681, 15.02956]},
    # --- series 20221027 (2018-2021 flights, oldest) ---
    "2021150": {"series": "20221027", "crs": "EPSG:31256", "bbox_wgs84": [48.49597, 14.93629, 49.03635, 16.01071]},
    "2021160": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [46.58761, 12.09967, 47.03762, 13.02276]},
    "2021250": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [47.33291, 12.98421, 47.66768, 14.01494]},
    "2021260": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [47.32849, 12.11857, 47.75278, 13.01837]},
    "2021350": {"series": "20221027", "crs": "EPSG:31256", "bbox_wgs84": [46.60296, 14.99683, 47.17264, 16.05465]},
    "2021360": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [47.01811, 12.98517, 47.35288, 14.0109]},
    "2021370": {"series": "20221027", "crs": "EPSG:31256", "bbox_wgs84": [47.50695, 14.98423, 47.66733, 16.01687]},
    "2021450": {"series": "20221027", "crs": "EPSG:31256", "bbox_wgs84": [48.45779, 15.97476, 48.81885, 16.97556]},
    "2021460": {"series": "20221027", "crs": "EPSG:31256", "bbox_wgs84": [48.09893, 15.97707, 48.47719, 16.98823]},
    "2021480": {"series": "20221027", "crs": "EPSG:31256", "bbox_wgs84": [47.14727, 14.9877, 47.53242, 16.01904]},
    "2020150": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [48.09734, 12.72797, 48.78187, 14.07954]},
    "2020160": {"series": "20221027", "crs": "EPSG:31254", "bbox_wgs84": [46.74871, 10.05296, 47.1729, 11.14091]},
    "2020250": {"series": "20221027", "crs": "EPSG:31256", "bbox_wgs84": [48.00148, 14.96165, 48.52169, 16.01383]},
    "2020260": {"series": "20221027", "crs": "EPSG:31254", "bbox_wgs84": [47.15208, 10.11819, 47.59561, 11.14474]},
    "2020350": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [48.17707, 13.98818, 48.80078, 15.02073]},
    "2020360": {"series": "20221027", "crs": "EPSG:31256", "bbox_wgs84": [47.64183, 14.97485, 48.02708, 16.01605]},
    "2020460": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [47.64769, 12.72818, 48.11737, 14.02084]},
    "2020550": {"series": "20221027", "crs": "EPSG:31254", "bbox_wgs84": [46.85768, 9.52348, 47.60346, 10.25027]},
    "2019150": {"series": "20221027", "crs": "EPSG:31256", "bbox_wgs84": [47.67523, 15.97958, 48.27028, 17.16991]},
    "2019160": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [46.47841, 12.98724, 47.03806, 14.00692]},
    "2019250": {"series": "20221027", "crs": "EPSG:31256", "bbox_wgs84": [46.82174, 15.98205, 47.75772, 16.73073]},
    "2019260": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [47.00736, 13.97381, 47.57593, 15.05049]},
    "2019350": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [47.86219, 13.98422, 48.20545, 15.03075]},
    "2019360": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [46.69242, 13.9701, 47.03628, 15.05914]},
    "2019370": {"series": "20221027", "crs": "EPSG:31254", "bbox_wgs84": [46.90935, 11.10419, 47.60652, 11.67955]},
    "2019450": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [47.54729, 13.98033, 47.8907, 15.03393]},
    "2019460": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [46.36541, 13.96748, 46.72147, 15.08178]},
    "2019470": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [46.95451, 11.61763, 47.63369, 12.16423]},
    "2018470": {"series": "20221027", "crs": "EPSG:31255", "bbox_wgs84": [47.01357, 12.11569, 47.35244, 13.02024]},
}


#: Map publication year hint → series key(s).  The ~2020 slot should match
#: any operate in the 20221027 series (covers flights 2018-2021).
YEAR_TO_SERIES: dict[int, list[str]] = {
    2024: ["20250415"],
    2023: ["20240625"],
    2020: ["20221027"],  # "~2020" UI slot covers 2018-2021 flights
}


def find_rgbi_operates(
    lat_min: float, lon_min: float, lat_max: float, lon_max: float,
    *, newest_first: bool = True, year: int | None = None,
) -> list[str]:
    """Return operate IDs whose WGS84 bbox overlaps the query bbox.

    *year* acts as a *series* selector when it matches a key in
    ``YEAR_TO_SERIES`` (e.g. year=2020 returns all operates from the
    20221027 series — flights 2018-2021).  Otherwise it filters by the
    4-digit year prefix of the operate ID.

    Ordered newest-first by default so callers can pick the freshest imagery.
    """
    hits: list[str] = []
    for opid, info in RGBI_OPERATES.items():
        bb = info["bbox_wgs84"]  # [lat_min, lon_min, lat_max, lon_max]
        if bb[0] <= lat_max and bb[2] >= lat_min and bb[1] <= lon_max and bb[3] >= lon_min:
            hits.append(opid)
    if year is not None:
        allowed_series = YEAR_TO_SERIES.get(year)
        if allowed_series:
            hits = [o for o in hits if RGBI_OPERATES[o]["series"] in allowed_series]
        else:
            hits = [o for o in hits if o.startswith(str(year))]
    if newest_first:
        hits.sort(key=lambda o: int(o[:4]), reverse=True)
    return hits


# === SECTION: URL helpers (tile URL construction) ===


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


# === SECTION: Overview level selection ===


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


# === SECTION: Single-tile reader ===


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
    with open_with_retry(url, caller=f"DOP N{tile[0]}E{tile[1]}") as ds:
        native_res = abs(ds.transform.a)
        num_bands = ds.count
        positive_y = ds.transform.e > 0  # south-up raster (BEV DOP tiles)

        if positive_y:
            # Non-standard south-up transform: pixel row increases = northing
            # increases.  rasterio.from_bounds() doesn't support this, so
            # compute the window manually.
            origin_e = ds.transform.c
            origin_n = ds.transform.f
            col_off = (min_e - origin_e) / ds.transform.a
            col_end = (max_e - origin_e) / ds.transform.a
            row_off = (min_n - origin_n) / ds.transform.e
            row_end = (max_n - origin_n) / ds.transform.e
            # Clamp to valid tile extent to avoid negative indices
            # (bbox may extend beyond this tile when KG straddles a
            # tile boundary).
            row_start_c = max(0, int(round(row_off)))
            row_end_c = min(ds.height, int(round(row_end)))
            col_start_c = max(0, int(round(col_off)))
            col_end_c = min(ds.width, int(round(col_end)))
            if row_end_c <= row_start_c or col_end_c <= col_start_c:
                raise ValueError(
                    f"Empty window for tile N{tile[0]}E{tile[1]} with bbox "
                    f"[{min_e},{min_n}]-[{max_e},{max_n}] (clamped to zero)"
                )
            window = Window.from_slices(
                (row_start_c, row_end_c),
                (col_start_c, col_end_c),
            )
        else:
            window = from_bounds(min_e, min_n, max_e, max_n, ds.transform)

        window = window.intersection(Window(0, 0, ds.width, ds.height))

        if window.width < 1 or window.height < 1:
            raise ValueError(
                f"Empty window for tile N{tile[0]}E{tile[1]} with bbox "
                f"[{min_e},{min_n}]-[{max_e},{max_n}]"
            )

        # Compute output size at target resolution
        # Use the *actual* geographic extent of the clamped window
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

        if positive_y:
            # Flip rows so the output is north-up (standard orientation)
            data = data[:, ::-1, :].copy()

        # Compute the geographic extent of the window we actually read
        win_transform = ds.window_transform(window)
        if positive_y:
            # For south-up rasters the window top-left is the SW corner;
            # after flipping, the output top-left is the NW corner.
            actual_min_e = win_transform.c
            actual_min_n = win_transform.f  # south edge
            actual_max_n = actual_min_n + win_height_m
            out_transform = transform_from_origin(
                actual_min_e, actual_max_n, resolution, resolution,
            )
        else:
            out_transform = transform_from_origin(
                win_transform.c,   # left  (easting)
                win_transform.f,   # top   (northing)
                resolution,
                resolution,
            )

        return data, out_transform, ds.crs


# === SECTION: Multi-tile mosaicking ===


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


# === SECTION: Public API — DOP 50km grid reads ===


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


# === SECTION: Public API — RGBI operate reads ===


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

    with open_with_retry(url, caller=url.rsplit('/', 1)[-1]) as ds:
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


# === SECTION: Spectral indices (NDVI, CIR) ===


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


# === SECTION: Convenience — read ortho aligned to ALS result ===


def _ensure_shape(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Pad or crop *arr* so spatial dims match *(h, w)*."""
    if arr.ndim == 3:
        if arr.shape[1] == h and arr.shape[2] == w:
            return arr
        out = np.zeros((arr.shape[0], h, w), dtype=arr.dtype)
        ch = min(arr.shape[1], h)
        cw = min(arr.shape[2], w)
        out[:, :ch, :cw] = arr[:, :ch, :cw]
        return out
    # 2-D (single band like NIR)
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    out = np.zeros((h, w), dtype=arr.dtype)
    ch = min(arr.shape[0], h)
    cw = min(arr.shape[1], w)
    out[:ch, :cw] = arr[:ch, :cw]
    return out


def _try_read_rgbi_for_bbox(
    min_e: float, min_n: float, max_e: float, max_n: float,
    resolution: float, h: int, w: int,
    dst_transform=None,
    year: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Try to read RGB + NIR from an RGBI operate covering the EPSG:3035 bbox.

    Returns (rgb, nir) or (None, None) if no operate covers the area.
    Both arrays are reprojected and aligned to the target EPSG:3035 grid
    (h, w) at *resolution*.
    """
    # --- Check disk cache first ---
    _ck_raw = f"rgbi|{min_e:.0f},{min_n:.0f},{max_e:.0f},{max_n:.0f}|{resolution}|{year}"
    _ck = hashlib.md5(_ck_raw.encode()).hexdigest()
    _cp = ORTHO_CACHE_DIR / f"{_ck}.npz"
    if _cp.exists():
        try:
            cached = np.load(str(_cp), allow_pickle=False)
            rgb = cached["rgb"]
            nir = cached["nir"] if "nir" in cached and cached["nir"].shape[0] > 0 else None
            log.info("Ortho RGBI cache hit for [%.0f,%.0f]-[%.0f,%.0f] @ %.1fm",
                     min_e, min_n, max_e, max_n, resolution)
            return rgb, nir
        except Exception as e:
            log.warning("Corrupt ortho cache %s: %s", _cp.name, e)
            _cp.unlink(missing_ok=True)

    import pyproj
    from rasterio.warp import reproject as rio_reproject, Resampling as RioResampling
    from rasterio.crs import CRS

    # Convert bbox corners to WGS84 for operate lookup
    tf_to_wgs = pyproj.Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon_min, lat_min = tf_to_wgs.transform(min_e, min_n)
    lon_max, lat_max = tf_to_wgs.transform(max_e, max_n)

    operates = find_rgbi_operates(lat_min, lon_min, lat_max, lon_max, year=year)
    if not operates:
        return None, None

    # Build the target EPSG:3035 transform if not provided
    dst_crs = CRS.from_epsg(3035)
    if dst_transform is None:
        dst_transform = transform_from_origin(min_e, max_n, resolution, resolution)

    for opid in operates:
        info = RGBI_OPERATES[opid]
        series = info["series"]
        op_crs_str = info["crs"]
        src_crs = CRS.from_user_input(op_crs_str)
        try:
            # Transform bbox to the operate's native CRS (with margin
            # for the rotation between CRSes)
            tf_to_op = pyproj.Transformer.from_crs(
                "EPSG:3035", op_crs_str, always_xy=True
            )
            # Transform all 4 corners to handle rotation
            corners_3035 = [
                (min_e, min_n), (max_e, min_n),
                (max_e, max_n), (min_e, max_n),
            ]
            corners_op = [tf_to_op.transform(e, n) for e, n in corners_3035]
            oe_min = min(c[0] for c in corners_op) - 10  # small margin
            oe_max = max(c[0] for c in corners_op) + 10
            on_min = min(c[1] for c in corners_op) - 10
            on_max = max(c[1] for c in corners_op) + 10

            rgb_url = get_rgbi_url(opid, "RGB", series)
            nir_url = get_rgbi_url(opid, "NIR", series)

            # Read a window from the source in its native CRS
            with open_with_retry(rgb_url, caller=f"RGBI {opid} RGB") as ds:
                win = from_bounds(oe_min, on_min, oe_max, on_max, ds.transform)
                win = win.intersection(Window(0, 0, ds.width, ds.height))
                if win.width < 1 or win.height < 1:
                    continue
                src_data = ds.read([1, 2, 3], window=win)
                src_transform = ds.window_transform(win)

            # Reproject RGB to EPSG:3035 target grid
            rgb = np.zeros((3, h, w), dtype=np.uint8)
            for band in range(3):
                rio_reproject(
                    source=src_data[band],
                    destination=rgb[band],
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=RioResampling.bilinear,
                )

            # Read and reproject NIR
            nir = None
            try:
                with open_with_retry(nir_url, caller=f"RGBI {opid} NIR") as ds:
                    win = from_bounds(oe_min, on_min, oe_max, on_max, ds.transform)
                    win = win.intersection(Window(0, 0, ds.width, ds.height))
                    if win.width >= 1 and win.height >= 1:
                        src_nir = ds.read(1, window=win)
                        src_nir_tf = ds.window_transform(win)
                        nir = np.zeros((h, w), dtype=np.uint8)
                        rio_reproject(
                            source=src_nir,
                            destination=nir,
                            src_transform=src_nir_tf,
                            src_crs=src_crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=RioResampling.bilinear,
                        )
            except Exception as e:
                log.warning("NIR read failed for operate %s: %s", opid, e)

            has_data = np.any(rgb > 0, axis=0)
            coverage = has_data.sum() / (h * w) if h * w > 0 else 0
            log.info(
                "Read RGBI operate %s (series %s) reprojected to %dx%d @ %.1fm, "
                "NIR=%s, coverage=%.0f%%",
                opid, series, w, h, resolution,
                "yes" if nir is not None else "no",
                coverage * 100,
            )
            # --- Save to disk cache ---
            try:
                save_dict = {"rgb": rgb}
                save_dict["nir"] = nir if nir is not None else np.array([], dtype=np.uint8)
                np.savez_compressed(str(_cp), **save_dict)
            except Exception as e:
                log.warning("Failed to cache ortho: %s", e)
            return rgb, nir

        except Exception as e:
            log.warning("RGBI operate %s failed: %s", opid, e)
            continue

    return None, None


def read_ortho_for_als(
    als_result: dict,
    dataset: str = DEFAULT_ORTHO_DATASET,
    year: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read orthophoto aligned to an ALS raster result.

    Tries RGBI operates first (RGB + NIR with real NDVI).  Falls back to
    DOP 50 km RGB tiles if no operate covers the area.

    Parameters
    ----------
    als_result:
        Dict as returned by :func:`raster_io.read_dtm_dsm`, containing at
        least ``"transform"``, ``"crs"``, and ``"shape"`` keys.
    dataset:
        DOP 50 km tile dataset key (used as fallback).

    Returns
    -------
    (rgb, nir)
        *rgb* has shape ``(3, H, W)`` uint8.
        *nir* has shape ``(H, W)`` uint8, or ``None`` if only RGB available.
    """
    tf = als_result["transform"]
    h, w = als_result["shape"]
    res = abs(tf.a)

    # Derive EPSG:3035 bbox from ALS transform + shape
    min_e = tf.c
    max_n = tf.f
    max_e = min_e + w * res
    min_n = max_n - h * res

    # 1. Try RGBI operates (RGB + NIR)
    rgb, nir = _try_read_rgbi_for_bbox(min_e, min_n, max_e, max_n, res, h, w,
                                        dst_transform=tf, year=year)
    if rgb is not None:
        return rgb, nir

    # 2. Fallback: DOP 50 km tiles (RGB only, no NIR)
    log.info("No RGBI operate found, falling back to DOP 50km tiles")
    rgb, _, _ = read_ortho_window(
        min_e, min_n, max_e, max_n,
        resolution=res,
        dataset=dataset,
    )
    rgb = _ensure_shape(rgb, h, w)
    return rgb, None
