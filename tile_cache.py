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

# ---------------------------------------------------------------------------
# Tile bbox index — records WGS84 bboxes of all cached tiles for map overlay
# ---------------------------------------------------------------------------

_TILE_INDEX_PATH = Path("data/austria_processor/tile_bbox_index.json")
_tile_index_lock = threading.Lock() if 'threading' in dir() else None

try:
    import threading as _threading
    _tile_index_lock = _threading.Lock()
except Exception:
    pass


def _record_tile_bbox(source: str, w: float, s: float, e: float, n: float,
                      status: str = "cached"):
    """Record a tile bbox in the index file for map visualisation.

    source: 'copernicus', 'hansen', 'bev'
    status: 'cached' or 'evicted'
    """
    key = f"{source}_{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}"
    entry = {"source": source, "w": round(w, 5), "s": round(s, 5),
             "e": round(e, 5), "n": round(n, 5), "status": status,
             "ts": time.time()}
    try:
        if _tile_index_lock:
            _tile_index_lock.acquire()
        idx = {}
        if _TILE_INDEX_PATH.exists():
            try:
                idx = json.loads(_TILE_INDEX_PATH.read_text())
            except Exception:
                idx = {}
        idx[key] = entry
        _TILE_INDEX_PATH.write_text(json.dumps(idx))
    except Exception:
        pass
    finally:
        if _tile_index_lock:
            try:
                _tile_index_lock.release()
            except Exception:
                pass


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
    """Scan cache directories and rebuild the tile bbox index.

    Copernicus tiles are in EPSG:32633 — we reconstruct WGS84 bbox
    from transform + shape.  Hansen tiles are already in WGS84.
    """
    from pyproj import Transformer
    tx_to_wgs = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)

    idx = {}

    # --- Copernicus tiles ---
    cop_dir = CopernicusTileCache.CACHE_DIR
    if cop_dir.exists():
        for f in cop_dir.glob("*.npz"):
            try:
                cached = np.load(str(f), allow_pickle=True)
                t = cached["transform"]
                # Find a data array to get shape
                for arr_key in ["ndvi", "vv", "data"]:
                    if arr_key in cached:
                        shape = cached[arr_key].shape
                        break
                else:
                    continue
                crs = str(cached.get("crs", "EPSG:32633"))
                west_m = float(t[2])
                north_m = float(t[5])
                east_m = west_m + shape[1] * float(t[0])
                south_m = north_m + shape[0] * float(t[4])
                if "4326" in crs:
                    w, s, e, n = west_m, south_m, east_m, north_m
                else:
                    w, s = tx_to_wgs.transform(west_m, south_m)
                    e, n = tx_to_wgs.transform(east_m, north_m)
                key = f"copernicus_{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}"
                idx[key] = {"source": "copernicus", "w": round(w, 5),
                            "s": round(s, 5), "e": round(e, 5),
                            "n": round(n, 5), "status": "cached",
                            "ts": f.stat().st_mtime,
                            "file": f.name}
            except Exception as exc:
                log.debug("Skip copernicus cache %s: %s", f.name, exc)

    # --- Hansen tiles ---
    han_dir = HansenTileCache.CACHE_DIR
    if han_dir.exists():
        for f in han_dir.glob("*.npz"):
            try:
                cached = np.load(str(f), allow_pickle=True)
                t = cached["transform"]
                shape = tuple(cached["shape"])
                w = float(t[2])
                n = float(t[5])
                e = w + shape[1] * float(t[0])
                s = n + shape[0] * float(t[4])
                key = f"hansen_{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}"
                idx[key] = {"source": "hansen", "w": round(w, 5),
                            "s": round(s, 5), "e": round(e, 5),
                            "n": round(n, 5), "status": "cached",
                            "ts": f.stat().st_mtime,
                            "file": f.name}
            except Exception as exc:
                log.debug("Skip hansen cache %s: %s", f.name, exc)

    try:
        _TILE_INDEX_PATH.write_text(json.dumps(idx))
        log.info("Tile bbox index rebuilt: %d entries", len(idx))
    except Exception as exc:
        log.warning("Failed to write tile bbox index: %s", exc)

    return idx


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
            _record_tile_bbox("copernicus", tw, ts, te, tn, "cached")
            return result
        except Exception as e:
            self._stats["errors"] += 1
            from copernicus import CreditsExhaustedError
            if isinstance(e, CreditsExhaustedError) or isinstance(e.__cause__, CreditsExhaustedError):
                log.error("Copernicus NDVI: credits exhausted — pausing")
                raise
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
            _record_tile_bbox("copernicus", tw, ts, te, tn, "cached")
            return result
        except Exception as e:
            self._stats["errors"] += 1
            from copernicus import CreditsExhaustedError
            if isinstance(e, CreditsExhaustedError) or isinstance(e.__cause__, CreditsExhaustedError):
                log.error("Copernicus WorldCover: credits exhausted — pausing")
                raise
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
            _record_tile_bbox("copernicus", tw, ts, te, tn, "cached")
            return result
        except Exception as e:
            self._stats["errors"] += 1
            from copernicus import CreditsExhaustedError
            if isinstance(e, CreditsExhaustedError) or isinstance(e.__cause__, CreditsExhaustedError):
                log.error("Copernicus SAR: credits exhausted — pausing")
                raise
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
            _record_tile_bbox("hansen", tw, ts, te, tn, "cached")
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
