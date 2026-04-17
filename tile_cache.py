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

# ---------------------------------------------------------------------------
# Grid-snapping utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Copernicus Tile Cache
# ---------------------------------------------------------------------------

class CopernicusTileCache:
    """Grid-snapped cache for Copernicus openEO data.

    Snaps requests to 0.1° tiles (~10×7km at Austrian latitudes).
    Caches NDVI composites, ESA WorldCover, and SAR data.
    """

    GRID_STEP = 0.1  # degrees
    CACHE_DIR = Path("data/austria_processor/copernicus_tiles")

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._stats = {"hits": 0, "misses": 0, "errors": 0}

    def _tile_path(self, prefix: str, w: float, s: float, e: float, n: float,
                   **extra) -> Path:
        key = tile_key(prefix, w, s, e, n, **extra)
        return self.CACHE_DIR / f"{prefix}_{key}.npz"

    def _snap(self, bbox: dict) -> Tuple[float, float, float, float]:
        return snap_bbox_to_grid(
            bbox["west"], bbox["south"], bbox["east"], bbox["north"],
            self.GRID_STEP)

    def get_ndvi(self, bbox_wgs: dict, year: int = 2024) -> Optional[dict]:
        """Get NDVI composite, using grid-snapped cache."""
        tw, ts, te, tn = self._snap(bbox_wgs)
        tile_bbox = {"west": tw, "south": ts, "east": te, "north": tn}
        path = self._tile_path("ndvi", tw, ts, te, tn, year=year)

        if path.exists():
            self._stats["hits"] += 1
            try:
                cached = np.load(str(path), allow_pickle=True)
                return {
                    "ndvi": cached["ndvi"],
                    "transform": _arr_to_affine(cached["transform"]),
                    "crs": str(cached["crs"]),
                }
            except Exception as e:
                log.warning("Corrupt Copernicus cache %s: %s", path.name, e)
                path.unlink(missing_ok=True)

        # Fetch from openEO for the full tile
        self._stats["misses"] += 1
        try:
            import copernicus
            result = copernicus.get_ndvi_composite(tile_bbox, year=year)
            # Save to cache
            tf = result["transform"]
            np.savez_compressed(
                str(path),
                ndvi=result["ndvi"],
                transform=np.array([tf.a, tf.b, tf.c, tf.d, tf.e, tf.f]),
                crs=str(result.get("crs", "EPSG:4326")),
            )
            log.info("Copernicus NDVI tile cached: %.2f,%.2f → %.2f,%.2f (%dx%d)",
                     tw, ts, te, tn, result["ndvi"].shape[1], result["ndvi"].shape[0])
            return result
        except Exception as e:
            self._stats["errors"] += 1
            log.warning("Copernicus NDVI tile fetch failed: %s", e)
            return None

    def get_landcover(self, bbox_wgs: dict) -> Optional[dict]:
        """Get ESA WorldCover, grid-snapped."""
        tw, ts, te, tn = self._snap(bbox_wgs)
        tile_bbox = {"west": tw, "south": ts, "east": te, "north": tn}
        path = self._tile_path("worldcover", tw, ts, te, tn)

        if path.exists():
            self._stats["hits"] += 1
            try:
                cached = np.load(str(path), allow_pickle=True)
                return cached["data"].item()  # dict stored via allow_pickle
            except Exception:
                path.unlink(missing_ok=True)

        self._stats["misses"] += 1
        try:
            import copernicus
            result = copernicus.get_land_cover(tile_bbox)
            np.savez_compressed(str(path), data=np.array(result, dtype=object))
            log.info("Copernicus WorldCover tile cached: %.2f,%.2f → %.2f,%.2f",
                     tw, ts, te, tn)
            return result
        except Exception as e:
            self._stats["errors"] += 1
            log.warning("Copernicus WorldCover tile fetch failed: %s", e)
            return None

    def get_sar(self, bbox_wgs: dict, year: int = 2024) -> Optional[dict]:
        """Get SAR backscatter, grid-snapped."""
        tw, ts, te, tn = self._snap(bbox_wgs)
        tile_bbox = {"west": tw, "south": ts, "east": te, "north": tn}
        path = self._tile_path("sar", tw, ts, te, tn, year=year)

        if path.exists():
            self._stats["hits"] += 1
            try:
                cached = np.load(str(path), allow_pickle=True)
                return {
                    "vv": cached["vv"], "vh": cached["vh"],
                    "transform": _arr_to_affine(cached["transform"]),
                    "crs": str(cached["crs"]),
                }
            except Exception:
                path.unlink(missing_ok=True)

        self._stats["misses"] += 1
        try:
            import copernicus
            result = copernicus.get_sar_backscatter(
                tile_bbox, f"{year}-06-01", f"{year}-09-30")
            tf = result["transform"]
            np.savez_compressed(
                str(path),
                vv=result["vv"], vh=result["vh"],
                transform=np.array([tf.a, tf.b, tf.c, tf.d, tf.e, tf.f]),
                crs=str(result.get("crs", "EPSG:4326")),
            )
            log.info("Copernicus SAR tile cached: %.2f,%.2f → %.2f,%.2f",
                     tw, ts, te, tn)
            return result
        except Exception as e:
            self._stats["errors"] += 1
            log.warning("Copernicus SAR tile fetch failed: %s", e)
            return None

    @property
    def stats(self) -> dict:
        n_files = sum(1 for _ in self.CACHE_DIR.glob("*.npz"))
        total_bytes = sum(f.stat().st_size for f in self.CACHE_DIR.glob("*.npz"))
        return {**self._stats, "files": n_files,
                "size_mb": round(total_bytes / 1024 / 1024, 1)}


# ---------------------------------------------------------------------------
# Hansen Tile Cache
# ---------------------------------------------------------------------------

class HansenTileCache:
    """Grid-snapped cache for Hansen Global Forest Change data.

    Snaps requests to 0.5° tiles (~50×35km).
    Hansen GFC is one global tile per 10° square (50N_010E covers all Austria),
    read via /vsicurl/ HTTP range requests. Caching avoids redundant range reads.
    """

    GRID_STEP = 0.5  # degrees — larger tiles since Hansen is 30m
    CACHE_DIR = Path("data/austria_processor/hansen_tiles")

    def __init__(self):
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._stats = {"hits": 0, "misses": 0, "errors": 0}

    def _tile_path(self, w: float, s: float, e: float, n: float) -> Path:
        key = tile_key("hansen", w, s, e, n)
        return self.CACHE_DIR / f"hansen_{key}.npz"

    def _snap(self, bbox_wgs: tuple) -> Tuple[float, float, float, float]:
        w, s, e, n = bbox_wgs
        return snap_bbox_to_grid(w, s, e, n, self.GRID_STEP)

    def get_raw(self, bbox_wgs: tuple) -> Optional[dict]:
        """Get raw Hansen layers for a snapped tile bbox."""
        tw, ts, te, tn = self._snap(bbox_wgs)
        path = self._tile_path(tw, ts, te, tn)

        if path.exists():
            self._stats["hits"] += 1
            try:
                cached = np.load(str(path), allow_pickle=True)
                result = {}
                for layer in ["treecover2000", "lossyear", "gain", "datamask"]:
                    if layer in cached:
                        result[layer] = cached[layer]
                result["transform"] = _arr_to_affine(cached["transform"])
                result["shape"] = tuple(cached["shape"])
                return result
            except Exception as e:
                log.warning("Corrupt Hansen cache %s: %s", path.name, e)
                path.unlink(missing_ok=True)

        # Fetch from remote
        self._stats["misses"] += 1
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
            np.savez_compressed(str(path), **save_dict)
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


# ---------------------------------------------------------------------------
# Geographic sorting for cache locality
# ---------------------------------------------------------------------------

def sort_kgs_geographically(kgs: list[dict], step: float = 0.5) -> list[dict]:
    """Sort KGs in geographic tile order for maximum cache reuse.

    Groups KGs by 0.5° grid cell, then processes each cell's KGs together.
    This ensures all KGs in a Copernicus/Hansen tile are processed consecutively,
    so the tile is fetched once and reused for all KGs in the cell.
    """
    def _sort_key(kg):
        bb = kg.get("bbox", {})
        if isinstance(bb, dict) and "min_lon" in bb:
            lon = (bb["min_lon"] + bb["max_lon"]) / 2
            lat = (bb["min_lat"] + bb["max_lat"]) / 2
        else:
            lon, lat = 13.0, 47.0  # default center of Austria
        # Snap to grid cell, then sort west→east, south→north
        cell_lon = math.floor(lon / step) * step
        cell_lat = math.floor(lat / step) * step
        return (cell_lat, cell_lon, lat, lon)

    return sorted(kgs, key=_sort_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
