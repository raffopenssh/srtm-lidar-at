"""Copernicus openEO interface for Sentinel-2 NDVI, ESA WorldCover, and Sentinel-1 SAR.

Provides cloud-free NDVI composites, monthly NDVI time series, land cover
classification, and SAR backscatter data via the Copernicus Data Space
openEO API.  Results are cached locally in /tmp/copernicus_cache/.

Usage::

    from copernicus import get_ndvi_composite, get_ndvi_timeseries
    result = get_ndvi_composite({"west": 16.3, "south": 48.2, "east": 16.4, "north": 48.3})
    ndvi = result["ndvi"]  # np.ndarray (H, W)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import tempfile
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import openeo
except ImportError:
    openeo = None  # type: ignore[assignment]

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.crs import CRS
except ImportError:
    rasterio = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Credentials & configuration
# ---------------------------------------------------------------------------
CLIENT_ID = "sh-19061cbb-c6f9-4464-bba6-006e7fa17435"
CLIENT_SECRET = "<REDACTED_SECRET>"
OPENEO_URL = "openeo.dataspace.copernicus.eu"

CACHE_DIR = pathlib.Path("/tmp/copernicus_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Maximum bbox extent in degrees (~10 km at mid-latitudes ≈ 0.09°)
MAX_BBOX_SPAN_DEG = 0.12

# Synchronous download size threshold (area in sq-degrees).
# Below this we use direct download(); above we use batch jobs.
SYNC_AREA_THRESHOLD = 0.008  # ~0.09° × 0.09°

# ESA WorldCover class legend
WORLDCOVER_CLASSES: Dict[int, str] = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
    70: "snow_ice",
    80: "permanent_water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_connection: Optional[Any] = None


def _get_connection() -> Any:
    """Return a cached, authenticated openEO connection."""
    global _connection
    if _connection is not None:
        return _connection

    if openeo is None:
        raise ImportError("The 'openeo' package is required. Install with: pip install openeo")

    logger.info("Connecting to openEO backend at %s", OPENEO_URL)
    conn = openeo.connect(OPENEO_URL)
    conn.authenticate_oidc_client_credentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    logger.info("Authenticated successfully")
    _connection = conn
    return conn


def _bbox_hash(bbox: Dict[str, float], **extra: Any) -> str:
    """Deterministic hash for a bbox + extra parameters (for cache keys)."""
    payload = json.dumps({"bbox": bbox, **extra}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_path(prefix: str, bbox: Dict[str, float], **extra: Any) -> pathlib.Path:
    """Return the cache file path for a given request."""
    h = _bbox_hash(bbox, **extra)
    return CACHE_DIR / f"{prefix}_{h}.tif"


def _validate_bbox(bbox: Dict[str, float]) -> Dict[str, float]:
    """Validate and normalise a WGS-84 bounding box dict."""
    required = {"west", "south", "east", "north"}
    if not required.issubset(bbox.keys()):
        raise ValueError(f"bbox must contain keys {required}, got {set(bbox.keys())}")

    w, s, e, n = bbox["west"], bbox["south"], bbox["east"], bbox["north"]
    if w >= e or s >= n:
        raise ValueError(f"Invalid bbox extents: west={w} >= east={e} or south={s} >= north={n}")

    span_lon = e - w
    span_lat = n - s
    if span_lon > MAX_BBOX_SPAN_DEG or span_lat > MAX_BBOX_SPAN_DEG:
        logger.warning(
            "Bbox span (%.4f° × %.4f°) exceeds recommended max %.4f°. "
            "Large requests may be slow or fail.",
            span_lon, span_lat, MAX_BBOX_SPAN_DEG,
        )

    return {"west": w, "south": s, "east": e, "north": n}


def _bbox_area_deg(bbox: Dict[str, float]) -> float:
    return (bbox["east"] - bbox["west"]) * (bbox["north"] - bbox["south"])


def _read_geotiff(path: Union[str, pathlib.Path]) -> Tuple[np.ndarray, Any, Any]:
    """Read a GeoTIFF and return (data, transform, crs).

    Returns data with shape (bands, H, W) or (H, W) if single-band.
    """
    if rasterio is None:
        raise ImportError("The 'rasterio' package is required. Install with: pip install rasterio")

    path = pathlib.Path(path)
    # If path is a directory (batch job output), find the first .tif inside
    if path.is_dir():
        tifs = sorted(path.glob("*.tif")) + sorted(path.glob("*.tiff"))
        if not tifs:
            raise FileNotFoundError(f"No GeoTIFF files found in {path}")
        path = tifs[0]
        logger.debug("Using GeoTIFF from directory: %s", path)

    with rasterio.open(str(path)) as ds:
        data = ds.read()  # (bands, H, W)
        transform = ds.transform
        crs = ds.crs

    if data.shape[0] == 1:
        data = data[0]  # squeeze to (H, W)

    return data, transform, crs


def _run_datacube(
    datacube: Any,
    output_path: pathlib.Path,
    title: str = "copernicus_job",
    format: str = "GTiff",
) -> pathlib.Path:
    """Download a datacube result to *output_path*.

    Uses synchronous ``download()`` for small cubes and batch-job
    ``execute_batch()`` for larger ones.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try synchronous download first (faster for small areas)
    logger.info("Downloading datacube synchronously → %s", output_path)
    try:
        datacube.download(str(output_path), format=format)
        logger.info("Synchronous download complete: %s", output_path)
        return output_path
    except Exception as exc:
        # If sync fails (e.g. too large), fall back to batch
        logger.warning("Synchronous download failed (%s), falling back to batch job", exc)

    # Batch job fallback
    logger.info("Submitting batch job: %s", title)
    output_dir = output_path.parent / f"{output_path.stem}_batch"
    output_dir.mkdir(parents=True, exist_ok=True)

    job = datacube.execute_batch(
        outputfile=str(output_dir),
        out_format=format,
        title=title,
        max_poll_interval=30,
        print=lambda msg: logger.info("[batch] %s", msg),
    )
    logger.info("Batch job %s finished", job.job_id)

    # Find the result file
    tifs = sorted(output_dir.glob("*.tif")) + sorted(output_dir.glob("*.tiff"))
    if tifs:
        # Copy/rename to expected path
        import shutil
        shutil.copy2(str(tifs[0]), str(output_path))
    elif output_dir.is_file():
        import shutil
        shutil.copy2(str(output_dir), str(output_path))

    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_ndvi_composite(
    bbox_wgs84: Dict[str, float],
    year: int = 2023,
) -> Dict[str, Any]:
    """Fetch a cloud-free NDVI composite for a bounding box.

    Uses the growing season (April–September) of *year* and computes
    the temporal **median** NDVI after cloud masking with SCL dilation.

    Parameters
    ----------
    bbox_wgs84 : dict
        ``{"west": float, "south": float, "east": float, "north": float}``
        in EPSG:4326.
    year : int
        Year to process (default 2023).

    Returns
    -------
    dict
        ``{"ndvi": np.ndarray (H,W), "transform": Affine, "crs": CRS,
        "date_range": str}``
    """
    bbox = _validate_bbox(bbox_wgs84)
    start_date = f"{year}-04-01"
    end_date = f"{year}-09-30"
    date_range = f"{start_date}/{end_date}"

    cache_file = _cache_path("ndvi_composite", bbox, year=year)
    if cache_file.exists():
        logger.info("Cache hit for NDVI composite: %s", cache_file)
        data, transform, crs = _read_geotiff(cache_file)
        return {
            "ndvi": data.astype(np.float32),
            "transform": transform,
            "crs": crs,
            "date_range": date_range,
        }

    logger.info("Fetching NDVI composite for bbox=%s, year=%d", bbox, year)
    conn = _get_connection()

    # Load Sentinel-2 L2A with B04 (Red), B08 (NIR), and SCL (Scene Classification)
    s2 = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=[start_date, end_date],
        bands=["B04", "B08", "SCL"],
    )

    # Cloud masking using SCL dilation (removes clouds, cloud shadows, etc.)
    s2_masked = s2.process(
        "mask_scl_dilation",
        data=s2,
        scl_band_name="SCL",
    )

    # Compute NDVI: (B08 - B04) / (B08 + B04)
    ndvi_cube = s2_masked.ndvi(nir="B08", red="B04")

    # Temporal median composite
    ndvi_composite = ndvi_cube.reduce_dimension(
        dimension="t",
        reducer="median",
    )

    # Download
    try:
        _run_datacube(ndvi_composite, cache_file, title=f"NDVI composite {year}")
    except Exception as exc:
        logger.error("Failed to download NDVI composite: %s", exc)
        raise RuntimeError(f"NDVI composite download failed: {exc}") from exc

    data, transform, crs = _read_geotiff(cache_file)
    return {
        "ndvi": data.astype(np.float32),
        "transform": transform,
        "crs": crs,
        "date_range": date_range,
    }


def get_ndvi_timeseries(
    bbox_wgs84: Dict[str, float],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Fetch monthly NDVI aggregates over a period.

    Clouds are filtered using SCL-based masking.  Each month is aggregated
    with the **median** reducer.

    Parameters
    ----------
    bbox_wgs84 : dict
        Bounding box in EPSG:4326.
    start_date, end_date : str
        ISO date strings, e.g. ``"2023-01-01"`` and ``"2023-12-31"``.

    Returns
    -------
    dict
        ``{"monthly_ndvi": {"2023-01": ndarray, ...},
        "transform": Affine, "crs": CRS}``
    """
    bbox = _validate_bbox(bbox_wgs84)

    cache_file = _cache_path("ndvi_ts", bbox, start=start_date, end=end_date)
    if cache_file.exists():
        logger.info("Cache hit for NDVI time series: %s", cache_file)
        return _parse_timeseries_tiff(cache_file, start_date, end_date)

    logger.info(
        "Fetching NDVI time series for bbox=%s, %s → %s",
        bbox, start_date, end_date,
    )
    conn = _get_connection()

    s2 = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=[start_date, end_date],
        bands=["B04", "B08", "SCL"],
    )

    # Cloud mask
    s2_masked = s2.process(
        "mask_scl_dilation",
        data=s2,
        scl_band_name="SCL",
    )

    # NDVI
    ndvi_cube = s2_masked.ndvi(nir="B08", red="B04")

    # Monthly aggregation
    ndvi_monthly = ndvi_cube.aggregate_temporal_period(
        period="month",
        reducer="median",
    )

    try:
        _run_datacube(ndvi_monthly, cache_file, title="NDVI time series")
    except Exception as exc:
        logger.error("Failed to download NDVI time series: %s", exc)
        raise RuntimeError(f"NDVI time series download failed: {exc}") from exc

    return _parse_timeseries_tiff(cache_file, start_date, end_date)


def _parse_timeseries_tiff(
    path: pathlib.Path,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Parse a multi-band GeoTIFF where each band is a monthly aggregate."""
    if rasterio is None:
        raise ImportError("rasterio is required")

    with rasterio.open(str(path)) as ds:
        data = ds.read()  # (bands, H, W)
        transform = ds.transform
        crs = ds.crs
        band_count = ds.count
        # Try to get band descriptions/dates from tags
        descriptions = [ds.descriptions[i] if ds.descriptions[i] else None for i in range(band_count)]

    # Build month labels from date range
    from datetime import datetime, timedelta
    import calendar

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        # Advance to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    monthly_ndvi: Dict[str, np.ndarray] = {}
    for i in range(min(band_count, len(months))):
        label = descriptions[i] if descriptions[i] else months[i]
        monthly_ndvi[label] = data[i].astype(np.float32)

    # If more bands than expected months, use numeric fallback
    if band_count > len(months):
        for i in range(len(months), band_count):
            label = descriptions[i] if descriptions[i] else f"band_{i+1}"
            monthly_ndvi[label] = data[i].astype(np.float32)

    return {
        "monthly_ndvi": monthly_ndvi,
        "transform": transform,
        "crs": crs,
    }


def get_land_cover(
    bbox_wgs84: Dict[str, float],
) -> Dict[str, Any]:
    """Fetch ESA WorldCover 10 m land-use classification.

    Parameters
    ----------
    bbox_wgs84 : dict
        Bounding box in EPSG:4326.

    Returns
    -------
    dict
        ``{"map": np.ndarray (H,W), "transform": Affine, "crs": CRS,
        "classes": dict}``
    """
    bbox = _validate_bbox(bbox_wgs84)

    cache_file = _cache_path("landcover", bbox)
    if cache_file.exists():
        logger.info("Cache hit for land cover: %s", cache_file)
        data, transform, crs = _read_geotiff(cache_file)
        return {
            "map": data.astype(np.uint8),
            "transform": transform,
            "crs": crs,
            "classes": WORLDCOVER_CLASSES.copy(),
        }

    logger.info("Fetching ESA WorldCover for bbox=%s", bbox)
    conn = _get_connection()

    # ESA WorldCover 10m 2021 v2 — single band "MAP"
    # Temporal extent is required by openEO even for static datasets
    lc = conn.load_collection(
        "ESA_WORLDCOVER_10M_2021_V2",
        spatial_extent=bbox,
        temporal_extent=["2021-01-01", "2021-12-31"],
        bands=["MAP"],
    )

    # Reduce the (trivial) temporal dimension
    lc_flat = lc.reduce_dimension(dimension="t", reducer="first")

    try:
        _run_datacube(lc_flat, cache_file, title="ESA WorldCover")
    except Exception as exc:
        logger.error("Failed to download land cover: %s", exc)
        raise RuntimeError(f"Land cover download failed: {exc}") from exc

    data, transform, crs = _read_geotiff(cache_file)
    return {
        "map": data.astype(np.uint8),
        "transform": transform,
        "crs": crs,
        "classes": WORLDCOVER_CLASSES.copy(),
    }


def get_sar_backscatter(
    bbox_wgs84: Dict[str, float],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Fetch Sentinel-1 SAR VV+VH backscatter composite.

    SAR penetrates clouds and can distinguish built structures from
    vegetation regardless of season.

    Parameters
    ----------
    bbox_wgs84 : dict
        Bounding box in EPSG:4326.
    start_date, end_date : str
        ISO date strings.

    Returns
    -------
    dict
        ``{"vv": np.ndarray (H,W), "vh": np.ndarray (H,W),
        "vv_vh_ratio": np.ndarray (H,W),
        "transform": Affine, "crs": CRS,
        "date_range": str}``
    """
    bbox = _validate_bbox(bbox_wgs84)

    cache_file = _cache_path("sar", bbox, start=start_date, end=end_date)
    if cache_file.exists():
        logger.info("Cache hit for SAR backscatter: %s", cache_file)
        return _parse_sar_tiff(cache_file, start_date, end_date)

    logger.info(
        "Fetching SAR backscatter for bbox=%s, %s → %s",
        bbox, start_date, end_date,
    )
    conn = _get_connection()

    s1 = conn.load_collection(
        "SENTINEL1_GRD",
        spatial_extent=bbox,
        temporal_extent=[start_date, end_date],
        bands=["VV", "VH"],
    )

    # Apply SAR backscatter processing (terrain correction)
    s1_processed = s1.sar_backscatter(
        coefficient="sigma0-ellipsoid",
    )

    # Temporal median composite
    sar_composite = s1_processed.reduce_dimension(
        dimension="t",
        reducer="median",
    )

    try:
        _run_datacube(sar_composite, cache_file, title="SAR backscatter")
    except Exception as exc:
        logger.error("Failed to download SAR backscatter: %s", exc)
        raise RuntimeError(f"SAR backscatter download failed: {exc}") from exc

    return _parse_sar_tiff(cache_file, start_date, end_date)


def _parse_sar_tiff(
    path: pathlib.Path,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Parse a 2-band SAR GeoTIFF (VV, VH)."""
    if rasterio is None:
        raise ImportError("rasterio is required")

    with rasterio.open(str(path)) as ds:
        data = ds.read()  # (bands, H, W)
        transform = ds.transform
        crs = ds.crs

    # Bands: VV=0, VH=1
    vv = data[0].astype(np.float32) if data.ndim == 3 and data.shape[0] >= 1 else data.astype(np.float32)
    vh = data[1].astype(np.float32) if data.ndim == 3 and data.shape[0] >= 2 else np.zeros_like(vv)

    # VV/VH ratio (useful for distinguishing land cover types)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(vh != 0, vv / vh, 0.0).astype(np.float32)

    return {
        "vv": vv,
        "vh": vh,
        "vv_vh_ratio": ratio,
        "transform": transform,
        "crs": crs,
        "date_range": f"{start_date}/{end_date}",
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def clear_cache() -> int:
    """Remove all cached files. Returns the number of files deleted."""
    count = 0
    for f in CACHE_DIR.iterdir():
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    logger.info("Cleared %d cached files", count)
    return count


def bbox_from_center(
    lon: float,
    lat: float,
    size_m: float = 5000.0,
) -> Dict[str, float]:
    """Create a WGS-84 bbox centred on (lon, lat) with *size_m* half-width.

    Useful for quick queries around a point of interest.
    """
    import math

    # Approximate degree offsets
    dlat = size_m / 111_320.0
    dlon = size_m / (111_320.0 * math.cos(math.radians(lat)))
    return {
        "west": lon - dlon,
        "south": lat - dlat,
        "east": lon + dlon,
        "north": lat + dlat,
    }


def ndvi_quality_mask(
    ndvi: np.ndarray,
    min_val: float = -1.0,
    max_val: float = 1.0,
) -> np.ndarray:
    """Return a boolean mask where NDVI values are within a valid range."""
    return (ndvi >= min_val) & (ndvi <= max_val) & np.isfinite(ndvi)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Small test area in Vienna
    test_bbox = bbox_from_center(16.37, 48.21, size_m=2000)
    print(f"Test bbox: {test_bbox}")

    print("\n--- NDVI Composite ---")
    try:
        result = get_ndvi_composite(test_bbox, year=2023)
        print(f"  Shape: {result['ndvi'].shape}")
        print(f"  NDVI range: [{np.nanmin(result['ndvi']):.3f}, {np.nanmax(result['ndvi']):.3f}]")
        print(f"  CRS: {result['crs']}")
        print(f"  Date range: {result['date_range']}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n--- Land Cover ---")
    try:
        lc = get_land_cover(test_bbox)
        print(f"  Shape: {lc['map'].shape}")
        unique, counts = np.unique(lc["map"], return_counts=True)
        for u, c in zip(unique, counts):
            name = lc["classes"].get(int(u), "unknown")
            print(f"  Class {u:3d} ({name:25s}): {c:6d} px")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n--- SAR Backscatter ---")
    try:
        sar = get_sar_backscatter(test_bbox, "2023-06-01", "2023-08-31")
        print(f"  VV shape: {sar['vv'].shape}")
        print(f"  VH shape: {sar['vh'].shape}")
    except Exception as e:
        print(f"  Error: {e}")
