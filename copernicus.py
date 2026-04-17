"""Copernicus openEO interface for Sentinel-2 NDVI, ESA WorldCover, and Sentinel-1 SAR.

Provides cloud-free NDVI composites, monthly NDVI time series, land cover
classification, and SAR backscatter data via the Copernicus Data Space
openEO API.  Results are cached locally with LRU eviction (max 2 GB).

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
# Credentials — multiple accounts for rotation when rate-limited (429) or overloaded.
# OLD (expired 2026-04): CLIENT_ID = "sh-19061cbb-c6f9-4464-bba6-006e7fa17435"
# OLD (expired 2026-04): CLIENT_SECRET = "<REDACTED_SECRET>"
# OLD (account 1, out of credits): CLIENT_ID = "sh-187c6dab-6b27-4ce8-afa8-b73f38e640f3"
# OLD (account 1, out of credits): CLIENT_SECRET = "<REDACTED_SECRET>"
_CREDENTIALS = [
    ("sh-f36653c6-5d8c-48a1-b86d-476c50eb389c", "<REDACTED_SECRET>"),  # fresh 2026-04
    ("sh-8d8c685f-df36-4536-b949-666532d08414", "<REDACTED_SECRET>"),  # renews 2026-05-01
    ("sh-2ed25dbb-857d-4e99-b070-e1954a99a980", "<REDACTED_SECRET>"),  # renews 2026-05-01
]
_credential_index = 0  # current credential pair
CLIENT_ID = _CREDENTIALS[0][0]
CLIENT_SECRET = _CREDENTIALS[0][1]
OPENEO_URL = "openeo.dataspace.copernicus.eu"

# Permanent cache survives /tmp cleanup and service restarts.
# LRU eviction keeps total size under CACHE_MAX_BYTES.
CACHE_DIR = pathlib.Path("/home/exedev/srtm-lidar/rf_training_data/copernicus_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB

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

# Per-credential connection pool — keyed by credential index.
# Used by _get_connection_for_cred() for parallel workers that each
# need their own openEO session (1 sync download per client_id).
_connections: Dict[int, Any] = {}

# Global flag: set when Copernicus returns 402 PaymentRequired.
# Callers (e.g. rf_train) can check this to pause gracefully.
credits_exhausted: bool = False
_credits_exhausted_at: Optional[str] = None  # ISO timestamp


class CreditsExhaustedError(Exception):
    """Raised when Copernicus returns 402 PaymentRequired."""
    pass


def _check_credits_error(exc: Exception) -> None:
    """If *exc* is a 402 PaymentRequired, set the global flag and re-raise
    as CreditsExhaustedError so callers can handle it distinctly."""
    global credits_exhausted, _credits_exhausted_at
    msg = str(exc)
    if '402' in msg and 'PaymentRequired' in msg:
        credits_exhausted = True
        _credits_exhausted_at = __import__('datetime').datetime.utcnow().isoformat()
        logger.error("Copernicus credits exhausted — set credits_exhausted flag")
        raise CreditsExhaustedError(msg) from exc


def rotate_credentials() -> bool:
    """Switch to the next credential pair. Returns True if rotated, False if exhausted all."""
    global _credential_index, _connection, CLIENT_ID, CLIENT_SECRET
    old_idx = _credential_index
    _credential_index = (_credential_index + 1) % len(_CREDENTIALS)
    if _credential_index == old_idx and len(_CREDENTIALS) == 1:
        return False  # only one set of credentials
    CLIENT_ID, CLIENT_SECRET = _CREDENTIALS[_credential_index]
    _connection = None  # force re-auth with new credentials
    logger.info("Rotated to credential set %d/%d (client_id=%s)",
                _credential_index + 1, len(_CREDENTIALS), CLIENT_ID[:16] + "...")
    return _credential_index != old_idx  # True unless we wrapped all the way around


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
    logger.info("Authenticated successfully (client_id=%s)", CLIENT_ID[:16] + "...")
    _connection = conn
    return conn


def _get_connection_for_cred(cred_index: int) -> Any:
    """Return a cached connection for a specific credential index.

    Each credential gets its own openEO session, allowing parallel sync
    downloads (openEO limits 1 concurrent sync job per client_id).
    """
    if cred_index in _connections:
        return _connections[cred_index]

    if openeo is None:
        raise ImportError("The 'openeo' package is required.")

    cid, csecret = _CREDENTIALS[cred_index]
    logger.info("Connecting to openEO for cred %d/%d (client_id=%s)",
                cred_index + 1, len(_CREDENTIALS), cid[:16] + "...")
    conn = openeo.connect(OPENEO_URL)
    conn.authenticate_oidc_client_credentials(
        client_id=cid, client_secret=csecret,
    )
    logger.info("Authenticated cred %d (client_id=%s)", cred_index + 1, cid[:16] + "...")
    _connections[cred_index] = conn
    return conn


def _bbox_hash(bbox: Dict[str, float], **extra: Any) -> str:
    """Deterministic hash for a bbox + extra parameters (for cache keys)."""
    payload = json.dumps({"bbox": bbox, **extra}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_path(prefix: str, bbox: Dict[str, float], **extra: Any) -> pathlib.Path:
    """Return the cache file path for a given request."""
    h = _bbox_hash(bbox, **extra)
    return CACHE_DIR / f"{prefix}_{h}.tif"


def _touch_cache(path: pathlib.Path):
    """Update atime/mtime on a cache file (for LRU tracking)."""
    try:
        path.touch()
    except Exception:
        pass


def _enforce_cache_limit():
    """Evict oldest cache files if total size exceeds CACHE_MAX_BYTES."""
    try:
        files = []
        for f in CACHE_DIR.iterdir():
            if f.is_file():
                st = f.stat()
                files.append((st.st_mtime, st.st_size, f))
        total = sum(s for _, s, _ in files)
        if total <= CACHE_MAX_BYTES:
            return
        # Sort oldest first, evict until under limit
        files.sort()
        for mtime, size, f in files:
            if total <= CACHE_MAX_BYTES:
                break
            try:
                f.unlink()
                total -= size
                logger.debug("Cache evict: %s (%d KB)", f.name, size // 1024)
            except Exception:
                pass
    except Exception:
        pass


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
    # Retry once on 429 Too Many Requests
    # Use a thread with timeout to avoid blocking workers indefinitely
    import time as _time
    import concurrent.futures
    SYNC_DOWNLOAD_TIMEOUT = 180  # 3 minutes max for synchronous download

    for attempt in range(2):
        logger.info("Downloading datacube synchronously → %s (timeout=%ds)",
                    output_path, SYNC_DOWNLOAD_TIMEOUT)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(datacube.download, str(output_path), format)
                future.result(timeout=SYNC_DOWNLOAD_TIMEOUT)
            logger.info("Synchronous download complete: %s", output_path)
            return output_path
        except concurrent.futures.TimeoutError:
            logger.warning("Synchronous download timed out after %ds, falling back to batch job",
                          SYNC_DOWNLOAD_TIMEOUT)
            break
        except Exception as exc:
            _check_credits_error(exc)  # raises CreditsExhaustedError on 402
            exc_str = str(exc)
            if ("429" in exc_str or "503" in exc_str or "max connections" in exc_str) and attempt == 0:
                logger.warning("Rate limited/overloaded (%s), rotating credentials and retrying...",
                              '429' if '429' in exc_str else '503')
                rotate_credentials()
                _time.sleep(5)
                # Rebuild the datacube with new connection on retry
                continue
            # If sync fails (e.g. too large), fall back to batch
            logger.warning("Synchronous download failed (%s), falling back to batch job", exc)
            break

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

    _enforce_cache_limit()
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_ndvi_composite(
    bbox_wgs84: Dict[str, float],
    year: int = 2023,
    _conn: Any = None,
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
        _touch_cache(cache_file)
        data, transform, crs = _read_geotiff(cache_file)
        return {
            "ndvi": data.astype(np.float32),
            "transform": transform,
            "crs": crs,
            "date_range": date_range,
        }

    logger.info("Fetching NDVI composite for bbox=%s, year=%d", bbox, year)
    conn = _conn or _get_connection()

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

    Downloads one NDVI composite per month and stacks them locally.
    This avoids the openEO aggregate_temporal_period bug where
    sync download collapses multi-band temporal output to 1 band.

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
    from datetime import datetime
    import calendar

    bbox = _validate_bbox(bbox_wgs84)

    # Check for stacked cache (new format)
    cache_file = _cache_path("ndvi_ts_v2", bbox, start=start_date, end=end_date)
    if cache_file.exists():
        logger.info("Cache hit for NDVI time series v2: %s", cache_file)
        _touch_cache(cache_file)
        return _parse_timeseries_tiff(cache_file, start_date, end_date)

    logger.info(
        "Fetching NDVI time series (per-month) for bbox=%s, %s → %s",
        bbox, start_date, end_date,
    )

    # Build month list — skip winter months (Nov-Feb) which often have
    # zero cloud-free scenes in Austria, causing openEO EmptyBounds errors
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    SKIP_MONTHS = {11, 12, 1, 2}  # Nov-Feb: snow/clouds, no usable NDVI
    months = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        if current.month not in SKIP_MONTHS:
            months.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    logger.info("NDVI months to fetch: %s", [m.strftime('%Y-%m') for m in months])

    # Fetch each month as a separate NDVI composite — parallel downloads
    import concurrent.futures

    monthly_ndvi: Dict[str, np.ndarray] = {}
    transform = None
    crs = None

    conn = _get_connection()

    # Build download tasks: (label, cache_path, datacube_or_None)
    tasks = []
    for m in months:
        label = m.strftime("%Y-%m")
        last_day = calendar.monthrange(m.year, m.month)[1]
        m_start = m.strftime("%Y-%m-%d")
        m_end = m.replace(day=last_day).strftime("%Y-%m-%d")
        month_cache = _cache_path("ndvi_month", bbox, start=m_start, end=m_end)
        tasks.append((label, m_start, m_end, month_cache))

    # Check which months need downloading
    to_download = []
    for label, m_start, m_end, month_cache in tasks:
        if month_cache.exists():
            logger.debug("Cache hit for %s: %s", label, month_cache)
            _touch_cache(month_cache)
        else:
            to_download.append((label, m_start, m_end, month_cache))

    # Download missing months in parallel (2 concurrent — openEO rate limit)
    def _download_month(args):
        import re as _re
        import time as _time
        label, m_start, m_end, month_cache = args
        logger.info("Fetching NDVI for %s (%s → %s)", label, m_start, m_end)
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                c = _get_connection()
                s2 = c.load_collection(
                    "SENTINEL2_L2A",
                    spatial_extent=bbox,
                    temporal_extent=[m_start, m_end],
                    bands=["B04", "B08", "SCL"],
                )
                s2_masked = s2.process(
                    "mask_scl_dilation", data=s2, scl_band_name="SCL",
                )
                ndvi_cube = s2_masked.ndvi(nir="B08", red="B04")
                ndvi_median = ndvi_cube.reduce_dimension(
                    dimension="t", reducer="median",
                )
                # Use sync-only download for monthly NDVI — don't fall back to
                # batch on EmptyBounds (no data = skip month, batch won't help)
                month_cache.parent.mkdir(parents=True, exist_ok=True)
                ndvi_median.download(str(month_cache), format="GTiff")
                logger.info("NDVI %s downloaded OK", label)
                return label, None
            except Exception as exc:
                exc_str = str(exc)
                if ("429" in exc_str or "503" in exc_str or "max connections" in exc_str) and attempt < max_retries:
                    # Parse Retry-After from exception string if available
                    retry_match = _re.search(r'Retry-After[":\s]+(\d+)', exc_str, _re.IGNORECASE)
                    if retry_match:
                        wait_secs = int(retry_match.group(1))
                    else:
                        wait_secs = 10 * (attempt + 1)  # 10s, 20s, 30s
                    logger.warning(
                        "NDVI %s rate limited/overloaded, retry %d/%d in %ds (rotating credentials)...",
                        label, attempt + 1, max_retries, wait_secs,
                    )
                    rotate_credentials()
                    _time.sleep(wait_secs)
                    continue
                _check_credits_error(exc)  # raises CreditsExhaustedError on 402
                logger.warning("NDVI %s failed: %s — skipping month", label, exc)
                return label, exc
        # Should not be reached, but guard anyway
        return label, Exception(f"NDVI {label}: retries exhausted")

    if to_download:
        logger.info("Downloading %d NDVI months (2 parallel)...", len(to_download))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(_download_month, t): t[0] for t in to_download}
            for fut in concurrent.futures.as_completed(futures):
                label, exc = fut.result()
                if exc:
                    logger.debug("Month %s failed", label)
                else:
                    logger.info("Month %s done", label)

    # Read all cached months
    for label, m_start, m_end, month_cache in tasks:
        if not month_cache.exists():
            continue
        try:
            with rasterio.open(str(month_cache)) as ds:
                data = ds.read(1).astype(np.float32)
                if transform is None:
                    transform = ds.transform
                    crs = ds.crs
                monthly_ndvi[label] = data
        except Exception as exc:
            logger.warning("Failed to read NDVI %s: %s", label, exc)

    if not monthly_ndvi:
        raise RuntimeError("No monthly NDVI data retrieved")

    logger.info("NDVI time series: %d/%d months retrieved", len(monthly_ndvi), len(months))

    # Stack into multi-band cache for future use
    try:
        ref_shape = next(iter(monthly_ndvi.values())).shape
        sorted_labels = sorted(monthly_ndvi.keys())
        stack = np.stack([monthly_ndvi[l] for l in sorted_labels], axis=0)
        with rasterio.open(
            str(cache_file), "w", driver="GTiff",
            height=ref_shape[0], width=ref_shape[1],
            count=len(sorted_labels), dtype="float32",
            crs=crs, transform=transform,
        ) as dst:
            for i, label in enumerate(sorted_labels):
                dst.write(stack[i], i + 1)
                dst.set_band_description(i + 1, label)
        logger.info("Saved stacked NDVI TS cache: %s (%d bands)", cache_file, len(sorted_labels))
    except Exception as exc:
        logger.warning("Failed to write stacked cache: %s", exc)

    return {
        "monthly_ndvi": monthly_ndvi,
        "transform": transform,
        "crs": crs,
    }


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
        descriptions = [ds.descriptions[i] if ds.descriptions[i] else None for i in range(band_count)]

    # Build month labels from date range
    from datetime import datetime

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    monthly_ndvi: Dict[str, np.ndarray] = {}
    for i in range(band_count):
        # Prefer band description (set by our stacked cache writer)
        if descriptions[i]:
            label = descriptions[i]
        elif i < len(months):
            label = months[i]
        else:
            label = f"band_{i+1}"
        monthly_ndvi[label] = data[i].astype(np.float32)

    return {
        "monthly_ndvi": monthly_ndvi,
        "transform": transform,
        "crs": crs,
    }


def get_land_cover(
    bbox_wgs84: Dict[str, float],
    _conn: Any = None,
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
        _touch_cache(cache_file)
        data, transform, crs = _read_geotiff(cache_file)
        return {
            "map": data.astype(np.uint8),
            "transform": transform,
            "crs": crs,
            "classes": WORLDCOVER_CLASSES.copy(),
        }

    logger.info("Fetching ESA WorldCover for bbox=%s", bbox)
    conn = _conn or _get_connection()

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
    _conn: Any = None,
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
        _touch_cache(cache_file)
        return _parse_sar_tiff(cache_file, start_date, end_date)

    logger.info(
        "Fetching SAR backscatter for bbox=%s, %s → %s",
        bbox, start_date, end_date,
    )
    conn = _conn or _get_connection()

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
