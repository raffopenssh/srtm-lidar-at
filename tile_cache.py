"""Grid-snapped pre-caching for remote data sources.

Snaps bounding boxes to a regular grid so adjacent KGs share cached tiles.
Used by austria_processor to avoid per-KG downloads from Copernicus, Hansen, etc.

Grid sizes (tuned per data source resolution + API limits):
  - Copernicus (10m): 0.1° tiles (~10×7km) — fits openEO sync limit
  - Hansen (30m):     0.5° tiles (~50×35km) — one /vsicurl/ range read
  - ESA WorldCover:   0.1° tiles (same as Copernicus)

Usage::

    from tile_cache import CopernicusTileCache, HansenTileCache

    cop_cache = CopernicusTileCache()
    ndvi = cop_cache.get_ndvi(kg_bbox, year=2024)     # cache hit or fetch
    lc   = cop_cache.get_landcover(kg_bbox)            # same tile, instant

    hansen_cache = HansenTileCache()
    hdata = hansen_cache.get_forest_prior(kg_bbox, transform, shape)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class CacheMissError(RuntimeError):
    """Raised by tile caches in *forbid_remote* mode when a tile is
    missing both locally and from the Zenodo persistent cache.

    Used by *cache-only* peers to refuse processing KGs that would
    require Copernicus or Hansen API calls.  The caller (austria_processor)
    catches this and re-queues the KG for the frontier (primary) peer.
    """


# Process-global flag — when True, all tile caches refuse remote fetches
# and raise CacheMissError on any miss.  Set via env COPERNICUS_FORBIDDEN=1
# or by austria_processor.main() when started with --cache-only.
FORBID_REMOTE: bool = os.environ.get("COPERNICUS_FORBIDDEN", "").lower() in ("1", "true", "yes")


def set_forbid_remote(enabled: bool) -> None:
    """Toggle cache-only mode globally for this process."""
    global FORBID_REMOTE
    FORBID_REMOTE = bool(enabled)
    log.info("tile_cache: forbid_remote=%s (cache-only mode)", FORBID_REMOTE)


def _write_tile_meta(npz_path: Path, product: str,
                     w: float, s: float, e: float, n: float,
                     **extra):
    """Write sidecar .meta.json for zenodo_cache reverse index."""
    try:
        from zenodo_cache import write_tile_meta
        write_tile_meta(npz_path, product, w, s, e, n, **extra)
    except Exception:
        pass  # Non-critical — don't break cache writes


def _atomic_savez(path: Path, **arrays) -> None:
    """Write an .npz file atomically via a temp file + rename.

    Prevents corrupt 0-byte cache files if the process is interrupted.
    np.savez_compressed appends '.npz' if the path doesn't already end
    with '.npz', so we use a '.tmp.npz' suffix to keep numpy happy and
    then rename to the final path.
    """
    # Use .tmp.npz so numpy doesn't append another .npz
    tmp = path.with_suffix(".tmp.npz")
    try:
        np.savez_compressed(str(tmp), **arrays)
        tmp.rename(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# === SECTION: Grid-snapping utilities ===

def snap_bbox_to_grid(west: float, south: float, east: float, north: float,
                      step: float) -> Tuple[float, float, float, float]:
    """Snap a bbox outward to a regular grid.

    Returns the smallest grid-aligned bbox that fully contains the input.
    """
    w = math.floor(west / step) * step
    s = math.floor(south / step) * step
    e = math.ceil(east / step) * step
    n = math.ceil(north / step) * step
    return (w, s, e, n)


def tile_key(prefix: str, w: float, s: float, e: float, n: float,
             **extra) -> str:
    """Deterministic cache key for a grid tile."""
    payload = f"{prefix}_{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}"
    if extra:
        payload += "_" + json.dumps(extra, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def crop_raster_to_bbox(data: np.ndarray, transform, target_bbox_wgs: tuple,
                        ) -> Tuple[np.ndarray, Any]:
    """Crop a raster array to a sub-bbox. Both in same CRS.

    data: 2D array, transform: rasterio Affine for data
    target_bbox_wgs: (west, south, east, north)
    Returns: (cropped_data, cropped_transform)
    """
    import rasterio
    from rasterio.windows import from_bounds

    w, s, e, n = target_bbox_wgs
    window = from_bounds(w, s, e, n, transform)
    # Clamp to array bounds
    row_off = max(0, int(window.row_off))
    col_off = max(0, int(window.col_off))
    row_end = min(data.shape[0], int(window.row_off + window.height))
    col_end = min(data.shape[1], int(window.col_off + window.width))

    if row_end <= row_off or col_end <= col_off:
        return data, transform

    cropped = data[row_off:row_end, col_off:col_end]
    new_tf = rasterio.transform.from_origin(
        transform.c + col_off * transform.a,
        transform.f + row_off * transform.e,
        abs(transform.a), abs(transform.e),
    )
    return cropped, new_tf


# === SECTION: CopernicusTileCache ===

# === SECTION: Tile bbox index (for dashboard map overlay) ===

_TILE_INDEX_PATH = Path("data/austria_processor/tile_bbox_index.json")
_TILE_INDEX_LOCK = Path("data/austria_processor/tile_bbox_index.lock")


def _record_tile_bbox(source: str, w: float, s: float, e: float, n: float,
                      status: str = "cached"):
    """Record a tile bbox in the index file for map visualisation.

    Uses a lockfile for cross-process safety (subprocess writes,
    gunicorn reads).  Fast path: skip if key already present.
    """
    import fcntl
    key = f"{source}_{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}"
    entry = {"source": source, "w": round(w, 5), "s": round(s, 5),
             "e": round(e, 5), "n": round(n, 5), "status": status,
             "ts": time.time()}
    try:
        _TILE_INDEX_LOCK.parent.mkdir(parents=True, exist_ok=True)
        with open(_TILE_INDEX_LOCK, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                idx = {}
                if _TILE_INDEX_PATH.exists():
                    try:
                        idx = json.loads(_TILE_INDEX_PATH.read_text())
                    except Exception:
                        idx = {}
                if key in idx and idx[key].get("status") == status:
                    return  # already recorded
                idx[key] = entry
                tmp = str(_TILE_INDEX_PATH) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(idx, f)
                os.replace(tmp, _TILE_INDEX_PATH)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
    except (BlockingIOError, OSError):
        pass  # another process holds the lock — skip this update


def get_tile_bbox_index() -> dict:
    """Return the tile bbox index for API/map consumption.

    Returns dict with 'copernicus' and 'hansen' lists of
    {w, s, e, n, status} bboxes.
    """
    try:
        if _TILE_INDEX_PATH.exists():
            raw = json.loads(_TILE_INDEX_PATH.read_text())
            # Group by source
            result = {"copernicus": [], "hansen": []}
            for k, v in raw.items():
                src = v.get("source", "")
                if src in result:
                    result[src].append(v)
            return result
    except Exception:
        pass
    return {"copernicus": [], "hansen": []}


def rebuild_tile_bbox_index():
    """No-op fallback.  The index is maintained incrementally by
    ``_record_tile_bbox`` which is called from ``_tile_path`` on every
    cache access (hit or miss).
    """
    pass


# === SECTION: Copernicus retry config ===
_SERVER_ERROR_MAX_RETRIES = 4        # up to 4 retries (5 total attempts)
_SERVER_ERROR_BACKOFF_SECS = [30, 60, 120, 240]  # exponential backoff delays


def _is_server_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a transient HTTP 5xx server error.

    Matches error messages containing '[500]', '[502]', '[503]',
    'Internal Server Error', 'Bad Gateway', 'Service Unavailable',
    'EjrApiError', or the generic 'Server error' pattern that the
    Copernicus openEO backend emits during maintenance windows.
    """
    msg = str(exc)
    # Also check chained cause
    cause_msg = str(exc.__cause__) if exc.__cause__ else ""
    combined = f"{msg} {cause_msg}"
    server_patterns = (
        "[500]", "[502]", "[503]",
        "Internal Server Error", "Bad Gateway", "Service Unavailable",
        "Server error", "EjrApiError",
        "server error", "502 Bad Gateway", "503 Service",
        # Batch job failures — openEO server-side errors, worth retrying
        "didn't finish successfully",
        "Status: error",
        "JobFailedException",
    )
    return any(pat in combined for pat in server_patterns)


class CopernicusTileCache:
    """Grid-snapped cache for Copernicus openEO data.

    Snaps requests to 0.1° tiles (~10×7km at Austrian latitudes).
    Caches NDVI composites, ESA WorldCover, and SAR data.

    On local cache miss, tries Zenodo persistent cache before falling
    back to the expensive openEO API.  See ``zenodo_cache.py``.
    """

    GRID_STEP = 0.1  # degrees
    CACHE_DIR = Path("data/austria_processor/copernicus_tiles")

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._stats = {"hits": 0, "misses": 0, "errors": 0}
        self._zenodo_cache = None  # lazy init
        self._zenodo_tried = False

    def _try_zenodo(self, product: str, tw: float, ts: float, te: float, tn: float,
                    **extra) -> bool:
        """Try to restore a tile from Zenodo.  Returns True if restored."""
        try:
            if self._zenodo_cache is None and not self._zenodo_tried:
                self._zenodo_tried = True
                from zenodo_cache import ZenodoCache, CacheManifest
                manifest = CacheManifest()
                if manifest.depo_id or manifest.all_files():
                    self._zenodo_cache = ZenodoCache()
            if self._zenodo_cache is None:
                return False
            result = self._zenodo_cache.fetch_copernicus(
                product, tw, ts, te, tn, dest_dir=self.CACHE_DIR, **extra)
            return result is not None
        except Exception as e:
            log.debug("Zenodo cache fetch failed for %s: %s", product, e)
            return False

    def _tile_path(self, prefix: str, w: float, s: float, e: float, n: float,
                   **extra) -> Path:
        key = tile_key(prefix, w, s, e, n, **extra)
        # Record individual 0.1° cells for map overlay (not the full snapped bbox,
        # which can span multiple cells and renders as oversized rectangles).
        step = self.GRID_STEP
        cx = w
        while cx < e - 1e-9:
            cy = s
            while cy < n - 1e-9:
                _record_tile_bbox("copernicus",
                                  round(cx, 5), round(cy, 5),
                                  round(cx + step, 5), round(cy + step, 5),
                                  "cached")
                cy += step
            cx += step
        return self.CACHE_DIR / f"{prefix}_{key}.npz"

    def _snap(self, bbox: dict) -> Tuple[float, float, float, float]:
        return snap_bbox_to_grid(
            bbox["west"], bbox["south"], bbox["east"], bbox["north"],
            self.GRID_STEP)

    def _iter_cells(self, bbox_wgs: dict):
        """Yield (w, s, e, n) for each 0.1° grid cell covering *bbox_wgs*."""
        tw, ts, te, tn = self._snap(bbox_wgs)
        step = self.GRID_STEP
        w = tw
        while w < te - 1e-9:
            s_ = ts
            while s_ < tn - 1e-9:
                yield (round(w, 4), round(s_, 4),
                       round(w + step, 4), round(s_ + step, 4))
                s_ += step
            w += step

    def has_cached(self, bbox_wgs: dict, *, ndvi: bool = True,
                   landcover: bool = True, sar: bool = False,
                   harmonics: bool = False, year: int = 2024) -> bool:
        """Check if all requested products are cached (local or Zenodo).

        Iterates over individual 0.1° grid cells — the same granularity
        that the Austria processor stores.  Does NOT download anything;
        only checks file existence and Zenodo ZIP indices.
        """
        products = []
        if ndvi:
            products.append(("ndvi", {"year": year}))
        if landcover:
            products.append(("worldcover", {}))
        if sar:
            products.append(("sar", {"year": year}))
        if harmonics:
            products.append(("harmonics", {"year": year}))
        if not products:
            return True

        # Lazy-init Zenodo index
        try:
            if self._zenodo_cache is None and not self._zenodo_tried:
                self._zenodo_tried = True
                from zenodo_cache import ZenodoCache, CacheManifest
                manifest = CacheManifest()
                if manifest.depo_id or manifest.all_files():
                    self._zenodo_cache = ZenodoCache()
        except Exception:
            pass

        for cw, cs, ce, cn in self._iter_cells(bbox_wgs):
            for product, extra in products:
                path = self._tile_path(product, cw, cs, ce, cn, **extra)
                if path.exists():
                    continue
                # Check Zenodo ZIP index (no download)
                if self._zenodo_cache is None:
                    return False
                try:
                    from zenodo_cache import (_cell_for_bbox, _zip_filename,
                                              _npz_entry_name,
                                              _legacy_strip_zip_for)
                    bs, bn, bw, be = _cell_for_bbox(cs, cw)
                    candidates = [
                        _zip_filename(product, bs, bn, bw, be),
                        _legacy_strip_zip_for(product, bs),
                    ]
                    seen_c = set()
                    candidates = [c for c in candidates
                                  if not (c in seen_c or seen_c.add(c))]
                    entry_name = _npz_entry_name(product, cw, cs, ce, cn, **extra)
                    found = False
                    for zip_name in candidates:
                        idx = self._zenodo_cache._get_zip_index(zip_name)
                        if idx is None:
                            continue
                        if idx.has_entry(entry_name):
                            found = True
                            break
                        if product == "worldcover":
                            alt = _npz_entry_name(product, cw, cs, ce, cn)
                            if idx.has_entry(alt):
                                found = True
                                break
                    if not found:
                        return False
                except Exception:
                    return False
        return True

    def _read_cell_npz(self, product: str, cw: float, cs: float,
                       ce: float, cn: float, **extra) -> Optional[Path]:
        """Ensure a single 0.1° cell NPZ is on local disk (from Zenodo if needed)."""
        path = self._tile_path(product, cw, cs, ce, cn, **extra)
        if path.exists():
            return path
        if self._try_zenodo(product, cw, cs, ce, cn, **extra) and path.exists():
            return path
        return None

    def _try_legacy_cache(self, product: str, bbox_wgs: dict, **extra) -> Optional[Path]:
        """Check for an old-style cache file keyed by the full snapped bbox.

        Before the per-cell refactor, cache files were stored under one key
        covering the entire snapped region.  This finds those files so they
        still count as hits.
        """
        tw, ts, te, tn = self._snap(bbox_wgs)
        path = self._tile_path(product, tw, ts, te, tn, **extra)
        if path.exists():
            return path
        return None

    def _mosaic_raster_cells(self, cell_results: list,
                             data_keys: list[str]) -> Optional[dict]:
        """Mosaic multiple per-cell raster results into one.

        Each element of *cell_results* is a dict with the data arrays
        listed in *data_keys*, plus ``transform`` (Affine) and ``crs``.

        Returns a single dict with the same keys, covering the bounding
        extent of all cells.
        """
        if len(cell_results) == 1:
            return cell_results[0]

        import rasterio
        from rasterio.merge import merge as rio_merge
        import tempfile
        import os

        # For each data key, mosaic independently
        first_crs = str(cell_results[0].get("crs", "EPSG:4326"))
        mosaic_result = {"crs": first_crs}
        mosaic_tf = None

        # Adjacent Copernicus cells can come back in different UTM zones
        # (zone 32 vs 33 across Austria). rasterio.merge insists on a
        # uniform CRS, so reproject any cells whose CRS differs from the
        # first cell's into ``first_crs`` before writing the temp TIFF.
        from rasterio.warp import (calculate_default_transform,
                                     reproject, Resampling)
        from rasterio.transform import array_bounds
        for dk in data_keys:
            tmp_paths = []
            datasets = []
            try:
                for cr in cell_results:
                    arr = cr[dk]
                    tf = cr["transform"]
                    crs_str = str(cr.get("crs", "EPSG:4326"))
                    tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                    tmp_paths.append(tmp.name)
                    if crs_str != first_crs:
                        h, w = arr.shape
                        src_bounds = array_bounds(h, w, tf)
                        dst_tf, dst_w, dst_h = calculate_default_transform(
                            crs_str, first_crs, w, h, *src_bounds)
                        dst_arr = np.zeros((dst_h, dst_w), dtype=arr.dtype)
                        resamp = (Resampling.nearest
                                  if np.issubdtype(arr.dtype, np.integer)
                                  else Resampling.bilinear)
                        reproject(
                            source=arr, destination=dst_arr,
                            src_transform=tf, src_crs=crs_str,
                            dst_transform=dst_tf, dst_crs=first_crs,
                            resampling=resamp,
                        )
                        arr = dst_arr
                        tf = dst_tf
                    with rasterio.open(
                        tmp.name, "w", driver="GTiff",
                        height=arr.shape[0], width=arr.shape[1], count=1,
                        dtype=arr.dtype, crs=first_crs, transform=tf,
                    ) as dst:
                        dst.write(arr, 1)
                    datasets.append(rasterio.open(tmp.name))
                mosaic_arr, mt = rio_merge(datasets)
                mosaic_result[dk] = mosaic_arr[0]
                if mosaic_tf is None:
                    mosaic_tf = mt
            finally:
                for ds in datasets:
                    ds.close()
                for p in tmp_paths:
                    os.unlink(p)

        mosaic_result["transform"] = mosaic_tf
        return mosaic_result

    def read_cached_product(self, bbox_wgs: dict, product: str,
                            year: int = 2024) -> Optional[dict]:
        """Read a product from per-cell cache and mosaic if needed.

        Only reads from local files + Zenodo.  Never calls the live API.
        Caller should verify ``has_cached()`` first.

        Returns dict in the same format as ``get_ndvi()`` / ``get_sar()``
        etc., or None on failure.
        """
        extra = {"year": year} if product in ("ndvi", "sar", "harmonics") else {}
        cells = list(self._iter_cells(bbox_wgs))

        if len(cells) == 1:
            cw, cs, ce, cn = cells[0]
            path = self._read_cell_npz(product, cw, cs, ce, cn, **extra)
            if path is None:
                return None
            try:
                cached = np.load(str(path), allow_pickle=True)
                result = {}
                for k in cached.files:
                    if k in ("transform", "crs"):
                        continue
                    result[k] = cached[k]
                if "transform" in cached:
                    result["transform"] = _arr_to_affine(cached["transform"])
                if "crs" in cached:
                    result["crs"] = str(cached["crs"])
                return result
            except Exception as e:
                log.warning("Cache read failed %s: %s", path.name, e)
                return None

        # Multiple cells — mosaic with rasterio
        try:
            import rasterio
            from rasterio.merge import merge as rio_merge
            from rasterio.transform import from_bounds
            import tempfile, os

            tmp_paths = []
            datasets = []
            for cw, cs, ce, cn in cells:
                path = self._read_cell_npz(product, cw, cs, ce, cn, **extra)
                if path is None:
                    return None
                cached = np.load(str(path), allow_pickle=True)
                data_keys = [k for k in cached.files if k not in ("transform", "crs")]
                if not data_keys:
                    return None
                arr = cached[data_keys[0]]
                tf = _arr_to_affine(cached["transform"])
                crs_str = str(cached["crs"])
                # Write to temp GeoTIFF for rasterio merge
                tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                tmp_paths.append(tmp.name)
                with rasterio.open(
                    tmp.name, "w", driver="GTiff",
                    height=arr.shape[0], width=arr.shape[1], count=1,
                    dtype=arr.dtype, crs=crs_str, transform=tf,
                ) as dst:
                    dst.write(arr, 1)
                datasets.append(rasterio.open(tmp.name))

            mosaic, mosaic_tf = rio_merge(datasets)
            for ds in datasets:
                ds.close()
            for p in tmp_paths:
                os.unlink(p)

            result = {data_keys[0]: mosaic[0]}
            result["transform"] = mosaic_tf
            result["crs"] = crs_str
            return result
        except Exception as e:
            log.warning("Mosaic of cached cells failed: %s", e)
            return None

    def _load_ndvi_npz(self, path: Path) -> Optional[dict]:
        """Load an NDVI result dict from an NPZ file."""
        try:
            cached = np.load(str(path), allow_pickle=True)
            return {
                "ndvi": cached["ndvi"],
                "transform": _arr_to_affine(cached["transform"]),
                "crs": str(cached["crs"]),
            }
        except Exception as e:
            log.warning("Corrupt Copernicus NDVI cache %s: %s", path.name, e)
            path.unlink(missing_ok=True)
            return None

    def _fetch_ndvi_cell(self, cw, cs, ce, cn, year, cred_index):
        """Fetch a single 0.1° NDVI cell from the API and cache it."""
        if FORBID_REMOTE:
            raise CacheMissError(
                f"NDVI cell {cw:.2f},{cs:.2f} (year={year}) not cached (forbid_remote)")
        cell_bbox = {"west": cw, "south": cs, "east": ce, "north": cn}
        path = self._tile_path("ndvi", cw, cs, ce, cn, year=year)
        from copernicus import CreditsExhaustedError, IPThrottledError
        last_exc = None
        for attempt in range(_SERVER_ERROR_MAX_RETRIES + 1):
            try:
                import copernicus
                _conn = copernicus._get_connection_for_cred(cred_index) if cred_index is not None else None
                result = copernicus.get_ndvi_composite(cell_bbox, year=year, _conn=_conn)
                tf = result["transform"]
                _atomic_savez(
                    path,
                    ndvi=result["ndvi"],
                    transform=np.array([tf.a, tf.b, tf.c, tf.d, tf.e, tf.f]),
                    crs=str(result.get("crs", "EPSG:4326")),
                )
                _write_tile_meta(path, "ndvi", cw, cs, ce, cn, year=year)
                log.info("Copernicus NDVI cell cached: %.2f,%.2f → %.2f,%.2f (%dx%d)",
                         cw, cs, ce, cn, result["ndvi"].shape[1], result["ndvi"].shape[0])
                return result
            except Exception as e:
                if isinstance(e, (CreditsExhaustedError, IPThrottledError)) or \
                   isinstance(e.__cause__, (CreditsExhaustedError, IPThrottledError)):
                    self._stats["errors"] += 1
                    log.error("Copernicus NDVI: credits exhausted / IP-throttled — pausing")
                    raise
                last_exc = e
                if _is_server_error(e) and attempt < _SERVER_ERROR_MAX_RETRIES:
                    delay = _SERVER_ERROR_BACKOFF_SECS[attempt]
                    log.warning(
                        "Copernicus NDVI cell fetch failed (server error, attempt %d/%d), "
                        "retrying in %ds: %s",
                        attempt + 1, _SERVER_ERROR_MAX_RETRIES + 1, delay, e,
                    )
                    time.sleep(delay)
                    continue
                break
        # Check if failure was due to IP throttling
        if last_exc and 'IP-throttled' in str(last_exc):
            from copernicus import IPThrottledError as _IPT
            raise _IPT(str(last_exc))
        self._stats["errors"] += 1
        log.warning("Copernicus NDVI cell fetch failed (%.4f,%.4f): %s", cw, cs, last_exc)
        return None

    def get_ndvi(self, bbox_wgs: dict, year: int = 2024, cred_index: int = None) -> Optional[dict]:
        """Get NDVI composite, using per-cell grid-snapped cache.

        Iterates 0.1° cells covering *bbox_wgs*.  For each cell, checks
        local cache → Zenodo → API.  Mosaics when bbox spans 2+ cells.
        Also checks for legacy (full-snapped-bbox) cache files.
        """
        # --- Legacy check: old-style full-bbox cache file ---
        legacy = self._try_legacy_cache("ndvi", bbox_wgs, year=year)
        if legacy is not None:
            result = self._load_ndvi_npz(legacy)
            if result is not None:
                self._stats["hits"] += 1
                return result

        # --- Per-cell iteration ---
        cells = list(self._iter_cells(bbox_wgs))
        cell_results = []
        for cw, cs, ce, cn in cells:
            # Local cache?
            path = self._read_cell_npz("ndvi", cw, cs, ce, cn, year=year)
            if path is not None:
                loaded = self._load_ndvi_npz(path)
                if loaded is not None:
                    self._stats["hits"] += 1
                    cell_results.append(loaded)
                    continue
            # Fetch from API
            self._stats["misses"] += 1
            fetched = self._fetch_ndvi_cell(cw, cs, ce, cn, year, cred_index)
            if fetched is None:
                return None  # one cell failed → whole product fails
            cell_results.append(fetched)

        if len(cell_results) == 1:
            return cell_results[0]

        # Mosaic multiple cells
        try:
            return self._mosaic_raster_cells(cell_results, ["ndvi"])
        except Exception as e:
            log.warning("NDVI mosaic failed: %s", e)
            return None

    def _load_worldcover_npz(self, path: Path) -> Optional[dict]:
        """Load a WorldCover result dict from an NPZ file."""
        try:
            cached = np.load(str(path), allow_pickle=True)
            return cached["data"].item()  # dict stored via allow_pickle
        except Exception as e:
            log.warning("Corrupt WorldCover cache %s: %s", path.name, e)
            path.unlink(missing_ok=True)
            return None

    def _fetch_worldcover_cell(self, cw, cs, ce, cn, cred_index):
        """Fetch a single 0.1° WorldCover cell from the API and cache it."""
        if FORBID_REMOTE:
            raise CacheMissError(
                f"WorldCover cell {cw:.2f},{cs:.2f} not cached (forbid_remote)")
        cell_bbox = {"west": cw, "south": cs, "east": ce, "north": cn}
        path = self._tile_path("worldcover", cw, cs, ce, cn)
        from copernicus import CreditsExhaustedError, IPThrottledError
        last_exc = None
        for attempt in range(_SERVER_ERROR_MAX_RETRIES + 1):
            try:
                import copernicus
                _conn = copernicus._get_connection_for_cred(cred_index) if cred_index is not None else None
                result = copernicus.get_land_cover(cell_bbox, _conn=_conn)
                _atomic_savez(path, data=np.array(result, dtype=object))
                _write_tile_meta(path, "worldcover", cw, cs, ce, cn)
                log.info("Copernicus WorldCover cell cached: %.2f,%.2f → %.2f,%.2f",
                         cw, cs, ce, cn)
                return result
            except Exception as e:
                if isinstance(e, (CreditsExhaustedError, IPThrottledError)) or \
                   isinstance(e.__cause__, (CreditsExhaustedError, IPThrottledError)):
                    self._stats["errors"] += 1
                    log.error("Copernicus WorldCover: credits exhausted / IP-throttled — pausing")
                    raise
                last_exc = e
                if _is_server_error(e) and attempt < _SERVER_ERROR_MAX_RETRIES:
                    delay = _SERVER_ERROR_BACKOFF_SECS[attempt]
                    log.warning(
                        "Copernicus WorldCover cell fetch failed (server error, attempt %d/%d), "
                        "retrying in %ds: %s",
                        attempt + 1, _SERVER_ERROR_MAX_RETRIES + 1, delay, e,
                    )
                    time.sleep(delay)
                    continue
                break
        # Check if failure was due to IP throttling
        if last_exc and 'IP-throttled' in str(last_exc):
            from copernicus import IPThrottledError as _IPT
            raise _IPT(str(last_exc))
        self._stats["errors"] += 1
        log.warning("Copernicus WorldCover cell fetch failed (%.4f,%.4f): %s", cw, cs, last_exc)
        return None

    def get_landcover(self, bbox_wgs: dict, cred_index: int = None) -> Optional[dict]:
        """Get ESA WorldCover, using per-cell grid-snapped cache.

        Iterates 0.1° cells covering *bbox_wgs*.  For each cell, checks
        local cache → Zenodo → API.  Mosaics when bbox spans 2+ cells.
        Also checks for legacy (full-snapped-bbox) cache files.
        """
        # --- Legacy check: old-style full-bbox cache file ---
        legacy = self._try_legacy_cache("worldcover", bbox_wgs)
        if legacy is not None:
            result = self._load_worldcover_npz(legacy)
            if result is not None:
                self._stats["hits"] += 1
                return result

        # --- Per-cell iteration ---
        cells = list(self._iter_cells(bbox_wgs))
        cell_results = []
        for cw, cs, ce, cn in cells:
            # Local cache?
            path = self._read_cell_npz("worldcover", cw, cs, ce, cn)
            if path is not None:
                loaded = self._load_worldcover_npz(path)
                if loaded is not None:
                    self._stats["hits"] += 1
                    cell_results.append(loaded)
                    continue
            # Fetch from API
            self._stats["misses"] += 1
            fetched = self._fetch_worldcover_cell(cw, cs, ce, cn, cred_index)
            if fetched is None:
                return None
            cell_results.append(fetched)

        if len(cell_results) == 1:
            return cell_results[0]

        # Mosaic multiple cells — WorldCover result has {map, transform, crs, classes}
        try:
            mosaic = self._mosaic_raster_cells(cell_results, ["map"])
            mosaic["classes"] = cell_results[0].get("classes", {})
            return mosaic
        except Exception as e:
            log.warning("WorldCover mosaic failed: %s", e)
            return None

    def _load_sar_npz(self, path: Path) -> Optional[dict]:
        """Load a SAR result dict from an NPZ file."""
        try:
            cached = np.load(str(path), allow_pickle=True)
            return {
                "vv": cached["vv"], "vh": cached["vh"],
                "transform": _arr_to_affine(cached["transform"]),
                "crs": str(cached["crs"]),
            }
        except Exception as e:
            log.warning("Corrupt SAR cache %s: %s", path.name, e)
            path.unlink(missing_ok=True)
            return None

    def _fetch_sar_cell(self, cw, cs, ce, cn, year, cred_index):
        """Fetch a single 0.1° SAR cell from the API and cache it."""
        if FORBID_REMOTE:
            raise CacheMissError(
                f"SAR cell {cw:.2f},{cs:.2f} (year={year}) not cached (forbid_remote)")
        cell_bbox = {"west": cw, "south": cs, "east": ce, "north": cn}
        path = self._tile_path("sar", cw, cs, ce, cn, year=year)
        from copernicus import CreditsExhaustedError, IPThrottledError
        last_exc = None
        for attempt in range(_SERVER_ERROR_MAX_RETRIES + 1):
            try:
                import copernicus
                _conn = copernicus._get_connection_for_cred(cred_index) if cred_index is not None else None
                result = copernicus.get_sar_backscatter(
                    cell_bbox, f"{year}-06-01", f"{year}-09-30", _conn=_conn)
                tf = result["transform"]
                _atomic_savez(
                    path,
                    vv=result["vv"], vh=result["vh"],
                    transform=np.array([tf.a, tf.b, tf.c, tf.d, tf.e, tf.f]),
                    crs=str(result.get("crs", "EPSG:4326")),
                )
                _write_tile_meta(path, "sar", cw, cs, ce, cn, year=year)
                log.info("Copernicus SAR cell cached: %.2f,%.2f → %.2f,%.2f",
                         cw, cs, ce, cn)
                return result
            except Exception as e:
                if isinstance(e, (CreditsExhaustedError, IPThrottledError)) or \
                   isinstance(e.__cause__, (CreditsExhaustedError, IPThrottledError)):
                    self._stats["errors"] += 1
                    log.error("Copernicus SAR: credits exhausted / IP-throttled — pausing")
                    raise
                last_exc = e
                if _is_server_error(e) and attempt < _SERVER_ERROR_MAX_RETRIES:
                    delay = _SERVER_ERROR_BACKOFF_SECS[attempt]
                    log.warning(
                        "Copernicus SAR cell fetch failed (server error, attempt %d/%d), "
                        "retrying in %ds: %s",
                        attempt + 1, _SERVER_ERROR_MAX_RETRIES + 1, delay, e,
                    )
                    time.sleep(delay)
                    continue
                break
        # Check if failure was due to IP throttling
        if last_exc and 'IP-throttled' in str(last_exc):
            from copernicus import IPThrottledError as _IPT
            raise _IPT(str(last_exc))
        self._stats["errors"] += 1
        log.warning("Copernicus SAR cell fetch failed (%.4f,%.4f): %s", cw, cs, last_exc)
        return None

    def get_sar(self, bbox_wgs: dict, year: int = 2024, cred_index: int = None) -> Optional[dict]:
        """Get SAR backscatter, using per-cell grid-snapped cache.

        Iterates 0.1° cells covering *bbox_wgs*.  For each cell, checks
        local cache → Zenodo → API.  Mosaics when bbox spans 2+ cells.
        Also checks for legacy (full-snapped-bbox) cache files.
        """
        # --- Legacy check: old-style full-bbox cache file ---
        legacy = self._try_legacy_cache("sar", bbox_wgs, year=year)
        if legacy is not None:
            result = self._load_sar_npz(legacy)
            if result is not None:
                self._stats["hits"] += 1
                return result

        # --- Per-cell iteration ---
        cells = list(self._iter_cells(bbox_wgs))
        cell_results = []
        for cw, cs, ce, cn in cells:
            # Local cache?
            path = self._read_cell_npz("sar", cw, cs, ce, cn, year=year)
            if path is not None:
                loaded = self._load_sar_npz(path)
                if loaded is not None:
                    self._stats["hits"] += 1
                    cell_results.append(loaded)
                    continue
            # Fetch from API
            self._stats["misses"] += 1
            fetched = self._fetch_sar_cell(cw, cs, ce, cn, year, cred_index)
            if fetched is None:
                return None
            cell_results.append(fetched)

        if len(cell_results) == 1:
            return cell_results[0]

        # Mosaic multiple cells
        try:
            return self._mosaic_raster_cells(cell_results, ["vv", "vh"])
        except Exception as e:
            log.warning("SAR mosaic failed: %s", e)
            return None

    _HARMONIC_KEYS = ["h_mean", "h_amplitude", "h_phase", "h_rmse"]

    def _load_harmonics_npz(self, path: Path) -> Optional[dict]:
        """Load a harmonics result dict from an NPZ file."""
        try:
            cached = np.load(str(path), allow_pickle=True)
            result = {}
            for k in self._HARMONIC_KEYS:
                if k in cached:
                    result[k] = cached[k]
            if "transform" in cached:
                result["transform"] = _arr_to_affine(cached["transform"])
            if "crs" in cached:
                result["crs"] = str(cached["crs"])
            return result
        except Exception as e:
            log.warning("Corrupt harmonics cache %s: %s", path.name, e)
            path.unlink(missing_ok=True)
            return None

    def _fetch_harmonics_cell(self, cw, cs, ce, cn, year, progress_fn):
        """Fetch a single 0.1° harmonics cell from the API and cache it."""
        if FORBID_REMOTE:
            raise CacheMissError(
                f"Harmonics cell {cw:.2f},{cs:.2f} (year={year}) not cached (forbid_remote)")
        cell_bbox = {"west": cw, "south": cs, "east": ce, "north": cn}
        path = self._tile_path("harmonics", cw, cs, ce, cn, year=year)
        from copernicus import CreditsExhaustedError, IPThrottledError
        last_exc = None
        for attempt in range(_SERVER_ERROR_MAX_RETRIES + 1):
            try:
                import ndvi_harmonics
                result = ndvi_harmonics.get_harmonic_features(
                    cell_bbox, year, progress_fn=progress_fn)
                if result is None:
                    self._stats["errors"] += 1
                    return None
                # Save to cache
                save_kw = {}
                for k in self._HARMONIC_KEYS:
                    if k in result:
                        save_kw[k] = result[k]
                tf = result.get("transform")
                if tf:
                    save_kw["transform"] = np.array(
                        [tf.a, tf.b, tf.c, tf.d, tf.e, tf.f])
                save_kw["crs"] = str(result.get("crs", "EPSG:4326"))
                _atomic_savez(path, **save_kw)
                _write_tile_meta(path, "harmonics", cw, cs, ce, cn, year=year)
                log.info("Copernicus harmonics cell cached: %.2f,%.2f → %.2f,%.2f",
                         cw, cs, ce, cn)
                return result
            except Exception as e:
                if isinstance(e, (CreditsExhaustedError, IPThrottledError)) or \
                   isinstance(e.__cause__, (CreditsExhaustedError, IPThrottledError)):
                    self._stats["errors"] += 1
                    raise
                last_exc = e
                if _is_server_error(e) and attempt < _SERVER_ERROR_MAX_RETRIES:
                    delay = _SERVER_ERROR_BACKOFF_SECS[attempt]
                    log.warning(
                        "Copernicus harmonics cell fetch failed (attempt %d/%d), "
                        "retrying in %ds: %s",
                        attempt + 1, _SERVER_ERROR_MAX_RETRIES + 1, delay, e)
                    time.sleep(delay)
                    continue
                break
        # Check if failure was due to IP throttling
        if last_exc and 'IP-throttled' in str(last_exc):
            from copernicus import IPThrottledError as _IPT
            raise _IPT(str(last_exc))
        self._stats["errors"] += 1
        log.warning("Copernicus harmonics cell fetch failed (%.4f,%.4f): %s", cw, cs, last_exc)
        return None

    def get_harmonics(self, bbox_wgs: dict, year: int = 2024,
                      cred_index: int = None,
                      progress_fn=None) -> Optional[dict]:
        """Get NDVI harmonic features, using per-cell grid-snapped cache.

        Fetches monthly NDVI time series and computes harmonic fit
        (mean, amplitude, phase, rmse) for each 0.1° grid cell.
        Subsequent requests within the same cell are instant cache hits.
        Mosaics when bbox spans 2+ cells.
        """
        # --- Legacy check: old-style full-bbox cache file ---
        legacy = self._try_legacy_cache("harmonics", bbox_wgs, year=year)
        if legacy is not None:
            result = self._load_harmonics_npz(legacy)
            if result is not None:
                self._stats["hits"] += 1
                return result

        # --- Per-cell iteration ---
        cells = list(self._iter_cells(bbox_wgs))
        cell_results = []
        for cw, cs, ce, cn in cells:
            # Local cache?
            path = self._read_cell_npz("harmonics", cw, cs, ce, cn, year=year)
            if path is not None:
                loaded = self._load_harmonics_npz(path)
                if loaded is not None:
                    self._stats["hits"] += 1
                    cell_results.append(loaded)
                    continue
            # Fetch from API
            self._stats["misses"] += 1
            fetched = self._fetch_harmonics_cell(
                cw, cs, ce, cn, year, progress_fn)
            if fetched is None:
                return None
            cell_results.append(fetched)

        if len(cell_results) == 1:
            return cell_results[0]

        # Mosaic multiple cells
        try:
            # Only mosaic keys that are present in the first result
            mosaic_keys = [k for k in self._HARMONIC_KEYS
                           if k in cell_results[0]]
            return self._mosaic_raster_cells(cell_results, mosaic_keys)
        except Exception as e:
            log.warning("Harmonics mosaic failed: %s", e)
            return None

    @property
    def stats(self) -> dict:
        n_files = sum(1 for _ in self.CACHE_DIR.glob("*.npz"))
        total_bytes = sum(f.stat().st_size for f in self.CACHE_DIR.glob("*.npz"))
        return {**self._stats, "files": n_files,
                "size_mb": round(total_bytes / 1024 / 1024, 1)}


# === SECTION: HansenTileCache ===

class HansenTileCache:
    """Grid-snapped cache for Hansen Global Forest Change data.

    Snaps requests to 0.5° tiles (~50×35km).
    Hansen GFC is one global tile per 10° square (50N_010E covers all Austria),
    read via /vsicurl/ HTTP range requests. Caching avoids redundant range reads.

    On local cache miss, tries Zenodo persistent cache before falling
    back to the slow UMD servers.  See ``zenodo_cache.py``.
    """

    GRID_STEP = 0.5  # degrees — larger tiles since Hansen is 30m
    CACHE_DIR = Path("data/austria_processor/hansen_tiles")

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._stats = {"hits": 0, "misses": 0, "errors": 0}
        self._zenodo_cache = None  # lazy init
        self._zenodo_tried = False

    def _tile_path(self, w: float, s: float, e: float, n: float) -> Path:
        key = tile_key("hansen", w, s, e, n)
        # Record individual 0.5° cells for map overlay (not the full snapped bbox)
        step = self.GRID_STEP
        cx = w
        while cx < e - 1e-9:
            cy = s
            while cy < n - 1e-9:
                _record_tile_bbox("hansen",
                                  round(cx, 5), round(cy, 5),
                                  round(cx + step, 5), round(cy + step, 5),
                                  "cached")
                cy += step
            cx += step
        return self.CACHE_DIR / f"hansen_{key}.npz"

    def _snap(self, bbox_wgs: tuple) -> Tuple[float, float, float, float]:
        w, s, e, n = bbox_wgs
        return snap_bbox_to_grid(w, s, e, n, self.GRID_STEP)

    def has_cached(self, bbox_wgs: tuple) -> bool:
        """Check if Hansen tile is cached (local or Zenodo).  No downloads.

        Iterates 0.5° cells covering *bbox_wgs* and verifies each is on
        local disk or recorded in the Zenodo cache manifest's ZIP index.
        """
        try:
            if self._zenodo_cache is None and not self._zenodo_tried:
                self._zenodo_tried = True
                from zenodo_cache import ZenodoCache, CacheManifest
                manifest = CacheManifest()
                if manifest.depo_id or manifest.all_files():
                    self._zenodo_cache = ZenodoCache()
        except Exception:
            pass
        tw, ts, te, tn = self._snap(bbox_wgs)
        step = self.GRID_STEP
        cw = tw
        while cw < te - 1e-9:
            cs = ts
            while cs < tn - 1e-9:
                ce = round(cw + step, 5)
                cn = round(cs + step, 5)
                path = self._tile_path(cw, cs, ce, cn)
                if path.exists():
                    cs += step
                    continue
                if self._zenodo_cache is None:
                    return False
                try:
                    from zenodo_cache import (_cell_for_bbox, _zip_filename,
                                              _npz_entry_name,
                                              _legacy_strip_zip_for)
                    bs, bn, bw, be = _cell_for_bbox(cs, cw)
                    candidates = [
                        _zip_filename("hansen", bs, bn, bw, be),
                        _legacy_strip_zip_for("hansen", bs),
                    ]
                    seen_c = set()
                    candidates = [c for c in candidates
                                  if not (c in seen_c or seen_c.add(c))]
                    entry_name = _npz_entry_name("hansen", cw, cs, ce, cn)
                    found = False
                    for zip_name in candidates:
                        idx = self._zenodo_cache._get_zip_index(zip_name)
                        if idx is not None and idx.has_entry(entry_name):
                            found = True
                            break
                    if not found:
                        return False
                except Exception:
                    return False
                cs += step
            cw += step
        return True

    def get_raw(self, bbox_wgs: tuple) -> Optional[dict]:
        """Get raw Hansen layers for a snapped tile bbox."""
        tw, ts, te, tn = self._snap(bbox_wgs)
        path = self._tile_path(tw, ts, te, tn)

        if path.exists():
            self._stats["hits"] += 1
            try:
                cached = np.load(str(path), allow_pickle=True)
                cached_shape = tuple(cached["shape"])
                # Check for degenerate cached data
                if any(d == 0 for d in cached_shape):
                    log.warning("Degenerate Hansen cache %s (shape %s), removing",
                                path.name, cached_shape)
                    path.unlink(missing_ok=True)
                    # Fall through to re-fetch below
                else:
                    result = {}
                    for layer in ["treecover2000", "lossyear", "gain", "datamask"]:
                        if layer in cached:
                            result[layer] = cached[layer]
                    result["transform"] = _arr_to_affine(cached["transform"])
                    result["shape"] = cached_shape
                    return result
            except Exception as e:
                log.warning("Corrupt Hansen cache %s: %s", path.name, e)
                path.unlink(missing_ok=True)

        # Try Zenodo persistent cache before remote fetch
        self._stats["misses"] += 1
        try:
            if self._zenodo_cache is None and not self._zenodo_tried:
                self._zenodo_tried = True
                from zenodo_cache import ZenodoCache, CacheManifest
                manifest = CacheManifest()
                if manifest.depo_id or manifest.all_files():
                    self._zenodo_cache = ZenodoCache()
        except Exception:
            pass
        if self._zenodo_cache is not None:
            try:
                restored = self._zenodo_cache.fetch_hansen(
                    tw, ts, te, tn, dest_dir=self.CACHE_DIR)
                if restored and path.exists():
                    self._stats["hits"] += 1
                    self._stats["misses"] -= 1
                    cached = np.load(str(path), allow_pickle=True)
                    result = {}
                    for layer in ["treecover2000", "lossyear", "gain", "datamask"]:
                        if layer in cached:
                            result[layer] = cached[layer]
                    result["transform"] = _arr_to_affine(cached["transform"])
                    result["shape"] = tuple(cached["shape"])
                    return result
            except Exception as e:
                log.debug("Zenodo Hansen fetch failed: %s", e)

        if FORBID_REMOTE:
            raise CacheMissError(
                f"Hansen tile {tw:.2f},{ts:.2f} not cached (forbid_remote)")

        try:
            import hansen
            tile_bbox = (tw, ts, te, tn)
            raw = hansen.read_hansen_window(
                tile_bbox,
                layers=["treecover2000", "lossyear", "gain", "datamask"],
            )
            tf = raw["transform"]
            save_dict = {
                "transform": np.array([tf.a, tf.b, tf.c, tf.d, tf.e, tf.f]),
                "shape": np.array(raw["shape"]),
            }
            for layer in ["treecover2000", "lossyear", "gain", "datamask"]:
                if layer in raw:
                    save_dict[layer] = raw[layer]
            _atomic_savez(path, **save_dict)
            _write_tile_meta(path, "hansen", tw, ts, te, tn)
            log.info("Hansen tile cached: %.2f,%.2f → %.2f,%.2f (%dx%d)",
                     tw, ts, te, tn, raw["shape"][1], raw["shape"][0])
            return raw
        except Exception as e:
            self._stats["errors"] += 1
            log.warning("Hansen tile fetch failed: %s", e)
            return None

    def get_forest_prior(self, bbox_wgs: tuple, target_transform,
                         target_shape: tuple) -> Optional[dict]:
        """Get Hansen forest prior resampled to target grid, tile-cached."""
        raw = self.get_raw(bbox_wgs)
        if raw is None:
            return None

        try:
            import hansen
            # Resample the cached tile data to target grid
            resampled = hansen.resample_to_target(raw, target_transform, target_shape)

            tc = resampled.get("treecover2000", np.zeros(target_shape, np.uint8))
            ly = resampled.get("lossyear", np.zeros(target_shape, np.uint8))
            gn = resampled.get("gain", np.zeros(target_shape, np.uint8))
            dm = resampled.get("datamask", np.ones(target_shape, np.uint8))

            was_forest = (tc >= 25) & (dm == 1)
            current_forest = (was_forest & ~(ly > 0)) | (gn > 0)

            return {
                "was_forest_2000": was_forest,
                "loss_year": ly,
                "gain": gn > 0,
                "current_forest": current_forest,
                "treecover2000": tc,
            }
        except Exception as e:
            log.warning("Hansen resample failed: %s", e)
            return None

    @property
    def stats(self) -> dict:
        n_files = sum(1 for _ in self.CACHE_DIR.glob("*.npz"))
        total_bytes = sum(f.stat().st_size for f in self.CACHE_DIR.glob("*.npz"))
        return {**self._stats, "files": n_files,
                "size_mb": round(total_bytes / 1024 / 1024, 1)}


# === SECTION: Geographic sorting (nearest-neighbor KG ordering) ===

def sort_kgs_geographically(kgs: list[dict], step: float = 0.1) -> list[dict]:
    """Sort KGs in serpentine (boustrophedon) scan for maximum cache reuse.

    Groups KGs into latitude bands of width ``step`` degrees, then sorts
    alternating left→right / right→left within each band.  This serpentine
    pattern ensures:

    - Adjacent KGs in the list share BEV LiDAR tiles (1m DTM/DSM)
    - Copernicus 0.1° tiles are fully reused across each band
    - Hansen 0.5° tiles (~5 bands high) stay in cache for many bands
    - Segmentation tile 100m overlap is reused for neighbors
    - Band-to-band transitions are short (just one step° in latitude)

    Compared to simple grid-cell sort, the serpentine avoids large jumps
    at the east/west edge of each row.

    Parameters
    ----------
    kgs : list[dict]
        Each dict must have 'lat', 'lon' (centroid) and optionally 'bbox'.
    step : float
        Latitude band height in degrees.  Default 0.1° (~11km) matches
        the Copernicus tile grid so each band's KGs share the same tiles.

    Returns
    -------
    list[dict]  – KGs in serpentine scan order.
    """
    if not kgs:
        return []

    def _get_coords(kg):
        """Extract centroid, falling back to bbox center."""
        lat = kg.get("lat")
        lon = kg.get("lon")
        if lat is not None and lon is not None:
            return (lat, lon)
        bb = kg.get("bbox", {})
        if isinstance(bb, dict) and "min_lon" in bb:
            return (
                (bb["min_lat"] + bb["max_lat"]) / 2,
                (bb["min_lon"] + bb["max_lon"]) / 2,
            )
        return (47.0, 13.0)  # center of Austria fallback

    # Assign each KG to a latitude band
    decorated = []
    for kg in kgs:
        lat, lon = _get_coords(kg)
        band = math.floor(lat / step)
        decorated.append((band, lon, lat, kg))

    # Sort by band (south→north), then by longitude within band
    decorated.sort(key=lambda x: (x[0], x[1]))

    # Group by band
    bands: dict[int, list] = {}
    for band, lon, lat, kg in decorated:
        bands.setdefault(band, []).append((lon, lat, kg))

    # Serpentine: even bands left→right, odd bands right→left
    result = []
    for i, band_key in enumerate(sorted(bands.keys())):
        band_kgs = bands[band_key]
        if i % 2 == 1:
            band_kgs = band_kgs[::-1]
        result.extend(kg for _, _, kg in band_kgs)

    return result


def _nn_order_indices(points: np.ndarray) -> list[int]:
    """Greedy nearest-neighbor ordering on a 2D point array.

    Returns list of indices into *points*, starting from the
    westernmost point (smallest x).
    """
    from scipy.spatial import KDTree
    n = len(points)
    if n <= 1:
        return list(range(n))
    tree = KDTree(points)
    visited = np.zeros(n, dtype=bool)
    current = int(np.argmin(points[:, 0]))
    order = []
    for _ in range(n):
        order.append(current)
        visited[current] = True
        if len(order) == n:
            break
        k = min(20, n)
        dists, idxs = tree.query(points[current], k=k)
        if np.ndim(dists) == 0:
            dists, idxs = [dists], [idxs]
        found = False
        for d, idx in zip(dists, idxs):
            if not visited[idx]:
                current = int(idx)
                found = True
                break
        if not found:
            remaining = np.where(~visited)[0]
            if len(remaining) == 0:
                break
            rem_d = np.linalg.norm(points[remaining] - points[current], axis=1)
            current = int(remaining[np.argmin(rem_d)])
    return order


def order_kgs_nearest_neighbor(kgs: list[dict],
                               start_code: str = None) -> list[dict]:
    """Two-level nearest-neighbor ordering for maximum cache reuse.

    Level 1: Group KGs into 0.1° Copernicus grid cells.  Order cells
             by nearest-neighbor — this minimises Copernicus tile
             fetches (each 0.1° tile is fetched once for all KGs in
             the cell).

    Level 2: Within each cell, order KGs by nearest-neighbor — this
             maximises BEV LiDAR HTTP range-read cache hits (adjacent
             KGs read overlapping 1m tiles).

    Result: ~1003 Copernicus tile transitions (one per unique cell),
    ~235 Hansen transitions, and short spatial hops within each cell.

    Parameters
    ----------
    kgs : list[dict]
        Each dict must have 'bbox' with min/max lat/lon.
    start_code : str, optional
        KG code to start from (for resuming).  If ``None``, starts
        from the westernmost Copernicus cell (Vorarlberg).
    """
    if not kgs:
        return []
    if len(kgs) == 1:
        return list(kgs)

    from collections import defaultdict

    COP_STEP = CopernicusTileCache.GRID_STEP  # 0.1°

    def _centroid(kg):
        bb = kg.get("bbox", {})
        if isinstance(bb, dict) and "min_lon" in bb:
            return ((bb["min_lon"] + bb["max_lon"]) / 2,
                    (bb["min_lat"] + bb["max_lat"]) / 2)
        return (kg.get("lon", 13.0), kg.get("lat", 47.0))

    # --- Group by Copernicus cell ---
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for kg in kgs:
        lon, lat = _centroid(kg)
        cell = (math.floor(lon / COP_STEP) * COP_STEP,
                math.floor(lat / COP_STEP) * COP_STEP)
        cells[cell].append(kg)

    cell_keys = list(cells.keys())
    mean_lat = sum(c[1] for c in cell_keys) / len(cell_keys)
    cos_lat = math.cos(math.radians(mean_lat))

    # --- NN order on cells ---
    cell_pts = np.array([(c[0] * cos_lat * 111, c[1] * 111)
                         for c in cell_keys])
    cell_order = _nn_order_indices(cell_pts)

    # If resuming, rotate so the cell containing start_code comes first
    if start_code:
        for ci, ck in enumerate(cell_keys):
            if any(kg.get("kg_code") == start_code for kg in cells[ck]):
                try:
                    pos = cell_order.index(ci)
                    cell_order = cell_order[pos:] + cell_order[:pos]
                except ValueError:
                    pass
                break

    # --- NN within each cell ---
    result = []
    for ci in cell_order:
        cell_kgs = cells[cell_keys[ci]]
        if len(cell_kgs) <= 1:
            result.extend(cell_kgs)
        else:
            kg_pts = np.array([
                (lon * cos_lat * 111, lat * 111)
                for lon, lat in (_centroid(kg) for kg in cell_kgs)
            ])
            inner = _nn_order_indices(kg_pts)
            result.extend(cell_kgs[i] for i in inner)

    # Log stats
    total_dist = 0.0
    for i in range(1, len(result)):
        lon1, lat1 = _centroid(result[i - 1])
        lon2, lat2 = _centroid(result[i])
        dx = (lon2 - lon1) * cos_lat * 111
        dy = (lat2 - lat1) * 111
        total_dist += math.sqrt(dx * dx + dy * dy)
    avg_step = total_dist / max(len(result) - 1, 1)
    log.info("NN KG ordering: %d KGs in %d cells, path %.0f km, avg step %.1f km",
             len(result), len(cell_keys), total_dist, avg_step)

    return result




# === SECTION: Helpers ===

def _arr_to_affine(arr):
    """Convert a 6-element array back to rasterio Affine."""
    import rasterio
    return rasterio.Affine(float(arr[0]), float(arr[1]), float(arr[2]),
                           float(arr[3]), float(arr[4]), float(arr[5]))


def cache_summary() -> dict:
    """Return summary of all tile caches (for status endpoint)."""
    result = {}
    for name, path in [("copernicus", CopernicusTileCache.CACHE_DIR),
                       ("hansen", HansenTileCache.CACHE_DIR)]:
        if path.exists():
            files = list(path.glob("*.npz"))
            result[name] = {
                "files": len(files),
                "size_mb": round(sum(f.stat().st_size for f in files) / 1024 / 1024, 1),
            }
        else:
            result[name] = {"files": 0, "size_mb": 0}
    return result


# === SECTION: KG cache coverage predicate ===

def is_kg_fully_cached(bbox_wgs: dict, year: int = 2024,
                      check_hansen: bool = True,
                      cop_cache: "CopernicusTileCache | None" = None,
                      hansen_cache: "HansenTileCache | None" = None) -> bool:
    """True iff all Copernicus + Hansen tiles for *bbox_wgs* are cached.

    Used by the Peer Director to identify KGs a *cache-only* peer can
    process without burning Copernicus credentials.  Checks local disk
    and the Zenodo persistent cache index — never downloads anything.
    """
    cop = cop_cache or CopernicusTileCache()
    if not cop.has_cached(bbox_wgs, ndvi=True, landcover=True,
                          sar=True, harmonics=True, year=year):
        return False
    if check_hansen:
        hc = hansen_cache or HansenTileCache()
        try:
            tup = (bbox_wgs["west"], bbox_wgs["south"],
                   bbox_wgs["east"], bbox_wgs["north"])
        except Exception:
            return False
        if not hc.has_cached(tup):
            return False
    return True
