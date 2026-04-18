#!/usr/bin/env python3
"""Austria Landscape Processor — segment every KG, upload to Zenodo.

Produces per-KG:
  1. Full GPKG: all rasters (DTM, DSM, nDSM, ortho, CIR, segmentation) + legend
  2. Light GPKG: segmentation raster+vector, parcels w/ DTM heights, buildings
     w/ object heights, vectorised new buildings, vectorised infrastructure
  3. JSON summary: area summary, landscape characterisation, tallest objects/trees,
     terrain stats, NDVI, Hansen loss, new buildings, infrastructure

Usage:
  python3 austria_processor.py                     # process all KGs
  python3 austria_processor.py --kg 63349          # process single KG
  python3 austria_processor.py --state Steiermark  # process one state
  python3 austria_processor.py --retry-failed      # retry failed KGs only
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import gzip
import hashlib
import json
import logging
import multiprocessing
import os
import signal
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import box, shape as shapely_shape, Point, Polygon, MultiPolygon, mapping
from shapely.ops import transform as shapely_transform

# === SECTION: Config ===

CADASTRE_BASE = "https://cadastre-process-api.exe.xyz/api/v1"
ZENODO_TOKEN = "2dnLSA2YYTc8jt3a1X0qDZUBb1hyOIpGJ44UoJr8N69wdePODgq4cjbJ0DJa"
VERSION = "v1"

DATA_DIR = Path("data/austria_processor")
MANIFEST_PATH = DATA_DIR / "zenodo_manifest.json"
JSON_DIR = DATA_DIR / "json"
GPKG_DIR = DATA_DIR / "gpkg"  # persistent storage — survives /tmp cleanup
LOG_DIR = DATA_DIR / "logs"
PROGRESS_FILE = DATA_DIR / "progress.json"
KG_LIST_FILE = DATA_DIR / "kg_list.json"
FAILED_KGS_FILE = DATA_DIR / "failed_kgs.json"
IN_PROGRESS_FILE = DATA_DIR / "in_progress_kg.txt"
CIRCUIT_BREAKER_FILE = DATA_DIR / "openeo_circuit.json"
TILE_HISTORY_FILE = DATA_DIR / "tile_history.json"
COPERNICUS_PAUSE_FILE = DATA_DIR / "copernicus_paused"

MAX_KG_PIXELS = 4_000_000
KG_TIMEOUT_SECONDS = 30 * 60
JSON_DIR_MAX_BYTES = 4 * 1024 ** 3  # 4GB
MAX_KG_AREA_KM = 1.5  # crop KG bbox if wider
DISK_MIN_FREE_GB = 5.0  # trigger cache cleanup
DISK_CRITICAL_FREE_GB = 3.0  # pause processor if still low after cleanup

# Tile caches — initialised lazily in subprocess
_cop_cache = None
_hansen_cache = None

def _get_cop_cache():
    global _cop_cache
    if _cop_cache is None:
        from tile_cache import CopernicusTileCache
        _cop_cache = CopernicusTileCache()
    return _cop_cache

def _get_hansen_cache():
    global _hansen_cache
    if _hansen_cache is None:
        from tile_cache import HansenTileCache
        _hansen_cache = HansenTileCache()
    return _hansen_cache

# CRS transformers
_tx_to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
_tx_to_wgs = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


# === SECTION: Disk cache management (LRU cleanup, free-space checks) ===

def check_disk_space(current_kg_code: str = "") -> bool:
    """Check free disk space and clean caches if below threshold.

    Returns True if processing can continue, False if disk is critically low.
    """
    import shutil
    usage = shutil.disk_usage("/")
    free_gb = usage.free / (1024 ** 3)

    if free_gb >= DISK_MIN_FREE_GB:
        return True

    _log = logging.getLogger("disk_cleanup")
    _log.warning("Disk free: %.1f GB (threshold: %.1f GB) — starting cache cleanup",
                 free_gb, DISK_MIN_FREE_GB)
    freed_bytes = 0

    def _lru_delete(directory: Path, target_bytes: int, label: str,
                    exclude_prefix: str = "") -> int:
        """Delete oldest files in directory by mtime until target_bytes freed."""
        nonlocal freed_bytes
        if not directory.exists():
            return 0
        files = []
        for f in directory.iterdir():
            if f.is_file() and not f.name.startswith("."):
                if exclude_prefix and f.name.startswith(exclude_prefix):
                    continue
                try:
                    files.append((f, f.stat().st_mtime, f.stat().st_size))
                except OSError:
                    continue
        files.sort(key=lambda x: x[1])  # oldest first
        deleted = 0
        n_deleted = 0
        for fpath, _, fsize in files:
            if deleted >= target_bytes:
                break
            try:
                fpath.unlink()
                deleted += fsize
                freed_bytes += fsize
                n_deleted += 1
            except OSError:
                continue
        if deleted > 0:
            _log.info("  %s: freed %.1f MB (%d files)",
                      label, deleted / 1e6, n_deleted)
        return deleted

    target_2gb = 2 * 1024 ** 3

    # 1. Delete old temp GPKG files (>1 hour old)
    if GPKG_DIR.exists():
        import time as _t
        now = _t.time()
        for f in GPKG_DIR.iterdir():
            if f.is_file() and (now - f.stat().st_mtime) > 3600:
                try:
                    sz = f.stat().st_size
                    f.unlink()
                    freed_bytes += sz
                    _log.info("  Deleted old GPKG: %s (%.1f MB)", f.name, sz / 1e6)
                except OSError:
                    pass

    # 2. BEV tile cache (LRU)
    bev_cache = DATA_DIR / "bev_tile_cache"
    _lru_delete(bev_cache, target_2gb, "bev_tile_cache")

    # 3. Ortho tile cache (LRU)
    ortho_cache = DATA_DIR / "ortho_tile_cache"
    _lru_delete(ortho_cache, target_2gb, "ortho_tile_cache")

    # 4. Copernicus tiles (LRU)
    cop_cache = DATA_DIR / "copernicus_tiles"
    _lru_delete(cop_cache, target_2gb, "copernicus_tiles")

    # 5. Hansen tiles (LRU)
    hansen_cache = DATA_DIR / "hansen_tiles"
    _lru_delete(hansen_cache, target_2gb, "hansen_tiles")

    # 6. RF training Copernicus cache (LRU)
    rf_cop_cache = Path("rf_training_data/copernicus_cache")
    _lru_delete(rf_cop_cache, target_2gb, "rf_copernicus_cache")

    _log.info("Disk cleanup freed %.1f MB total", freed_bytes / 1e6)

    # Re-check
    usage = shutil.disk_usage("/")
    free_gb = usage.free / (1024 ** 3)
    if free_gb < DISK_CRITICAL_FREE_GB:
        _log.error("Disk still critically low: %.1f GB free after cleanup. "
                   "Pausing processor.", free_gb)
        return False

    _log.info("Disk free after cleanup: %.1f GB — OK to continue", free_gb)
    return True


# === SECTION: Logging ===

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    stream=sys.stdout,
)
log = logging.getLogger('austria_processor')
for name in ['rasterio', 'urllib3', 'botocore', 'PIL', 'fiona']:
    logging.getLogger(name).setLevel(logging.WARNING)

# === SECTION: Ensure directories ===

for d in [DATA_DIR, JSON_DIR, GPKG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === SECTION: ProgressTracker (JSON-backed state, read by dashboard API) ===

_progress_lock = None  # set in main()

class ProgressTracker:
    """Thread-safe progress state, persisted to disk."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = __import__('threading').Lock()
        self._state = {
            "state": "idle",
            "total_kgs": 0,
            "completed": 0,
            "success": 0,
            "failed": 0,
            "uploaded": 0,
            "upload_size_bytes": 0,
            "current_kg": None,
            "rate_kgs_per_hour": 0,
            "avg_seconds_per_kg": 0,
            "elapsed_seconds": 0,
            "eta_seconds": 0,
            "recent_log": [],
            "failed_kgs": [],
            "parcels_total": 0,
            "buildings_total": 0,
            "started_at": None,
            "last_kg_code": None,
            "last_kg_seconds": 0,
            "n_new_buildings_total": 0,
            "n_infrastructure_total": 0,
            "kg_centroids": [],
        }
        self._load()

    def _load(self):
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                self._state.update(data)
        except Exception:
            pass

    def save(self):
        with self._lock:
            tmp = str(self.path) + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(self._state, f, indent=2, default=str)
            os.rename(tmp, str(self.path))

    def get(self) -> dict:
        with self._lock:
            return dict(self._state)

    def update(self, **kwargs):
        with self._lock:
            self._state.update(kwargs)

    def set_current_kg(self, kg_code: str, kg_name: str, state: str, step: str):
        with self._lock:
            self._state["current_kg"] = {
                "code": kg_code, "name": kg_name, "state": state,
                "step": step, "started_at": datetime.now(timezone.utc).isoformat(),
            }

    def set_step(self, step: str, detail: str = ""):
        with self._lock:
            if self._state["current_kg"]:
                now = datetime.now(timezone.utc).isoformat()
                prev_step = self._state["current_kg"].get("step", "")
                prev_start = self._state["current_kg"].get("step_started_at")
                # Record elapsed time for previous step
                if prev_step and prev_start:
                    try:
                        from datetime import datetime as _dt
                        t0 = _dt.fromisoformat(prev_start)
                        t1 = _dt.fromisoformat(now)
                        elapsed = (t1 - t0).total_seconds()
                        steps_done = self._state["current_kg"].setdefault("step_times", {})
                        steps_done[prev_step] = round(elapsed, 1)
                    except Exception:
                        pass
                self._state["current_kg"]["step"] = step
                self._state["current_kg"]["step_started_at"] = now
                if detail:
                    self._state["current_kg"]["step_detail"] = detail
                else:
                    self._state["current_kg"].pop("step_detail", None)

    def add_log(self, level: str, msg: str, kg: str = ""):
        with self._lock:
            entry = {"ts": datetime.now(timezone.utc).isoformat(),
                     "level": level, "msg": msg, "kg": kg}
            self._state["recent_log"].append(entry)
            self._state["recent_log"] = self._state["recent_log"][-200:]

    def add_failure(self, code: str, name: str, error: str, step: str):
        with self._lock:
            self._state["failed_kgs"].append({
                "code": code, "name": name, "error": error,
                "step": step, "ts": datetime.now(timezone.utc).isoformat(),
            })
            self._state["failed"] += 1
            self._state["completed"] += 1

    def record_success(self, parcels: int = 0, buildings: int = 0, upload_bytes: int = 0,
                        last_kg_code: str = None, last_kg_seconds: float = 0,
                        n_new_buildings: int = 0, n_infrastructure: int = 0):
        with self._lock:
            self._state["success"] += 1
            self._state["completed"] += 1
            self._state["uploaded"] += 1
            self._state["upload_size_bytes"] += upload_bytes
            self._state["parcels_total"] += parcels
            self._state["buildings_total"] += buildings
            if last_kg_code is not None:
                self._state["last_kg_code"] = last_kg_code
            self._state["last_kg_seconds"] = last_kg_seconds
            self._state["n_new_buildings_total"] += n_new_buildings
            self._state["n_infrastructure_total"] += n_infrastructure

    def update_rates(self, started_at: float):
        elapsed = time.time() - started_at
        with self._lock:
            n = self._state["completed"]
            self._state["elapsed_seconds"] = int(elapsed)
            if n > 0:
                avg = elapsed / n
                self._state["avg_seconds_per_kg"] = round(avg, 1)
                self._state["rate_kgs_per_hour"] = round(3600 / avg, 2)
                remaining = self._state["total_kgs"] - n
                self._state["eta_seconds"] = int(remaining * avg)

    def update_system_metrics(self):
        """Collect system metrics: RAM, CPU, disk, caches."""
        system = {}
        try:
            # System RAM
            with open("/proc/meminfo") as f:
                mi = f.read()
            for line in mi.splitlines():
                if line.startswith("MemTotal:"):
                    system["ram_total_mb"] = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    system["ram_avail_mb"] = int(line.split()[1]) // 1024
            if "ram_total_mb" in system and "ram_avail_mb" in system:
                used = system["ram_total_mb"] - system["ram_avail_mb"]
                system["ram_used_mb"] = used
                system["ram_pct"] = round(100 * used / max(system["ram_total_mb"], 1), 1)
        except Exception:
            pass

        try:
            # CPU load
            load1 = os.getloadavg()[0]
            n_cpu = os.cpu_count() or 1
            system["load_1m"] = round(load1, 2)
            system["cpu_pct"] = round(100 * load1 / n_cpu, 1)
        except Exception:
            pass

        try:
            # Disk free
            st = os.statvfs("/")
            free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
            total_gb = (st.f_blocks * st.f_frsize) / (1024 ** 3)
            system["disk_free_gb"] = round(free_gb, 1)
            system["disk_used_pct"] = round(100 * (1 - free_gb / total_gb), 1)
        except Exception:
            pass

        try:
            # Process RSS
            status = open("/proc/self/status").read()
            for line in status.splitlines():
                if line.startswith("VmRSS:"):
                    system["proc_ram_mb"] = int(line.split()[1]) // 1024
                    break
            system["proc_pid"] = os.getpid()
        except Exception:
            pass

        try:
            # JSON cache stats
            json_files = list(JSON_DIR.glob("*.json"))
            system["json_cache_files"] = len(json_files)
            system["json_cache_mb"] = round(
                sum(f.stat().st_size for f in json_files) / 1024 / 1024, 1)
        except Exception:
            pass

        try:
            # GPKG temp dir
            gpkg_files = list(GPKG_DIR.glob("*.gpkg"))
            system["gpkg_temp_files"] = len(gpkg_files)
            system["gpkg_temp_mb"] = round(
                sum(f.stat().st_size for f in gpkg_files) / 1024 / 1024, 1)
        except Exception:
            pass

        try:
            # Tile cache stats
            from tile_cache import cache_summary
            system["tile_caches"] = cache_summary()
        except Exception:
            pass

        try:
            # Circuit breaker state
            cb = _read_circuit_breaker()
            system["copernicus_circuit"] = (
                "open" if cb["consecutive_failures"] >= 3
                and (time.time() - cb["last_failure"]) < cb["cooldown"]
                else "closed"
            )
        except Exception:
            pass

        with self._lock:
            self._state["system"] = system


# === SECTION: Circuit breaker (openEO rate-limit protection) ===

def _read_circuit_breaker() -> dict:
    try:
        if CIRCUIT_BREAKER_FILE.exists():
            return json.loads(CIRCUIT_BREAKER_FILE.read_text())
    except Exception:
        pass
    return {"consecutive_failures": 0, "last_failure": 0.0, "cooldown": 120}

def _write_circuit_breaker(state: dict):
    try:
        CIRCUIT_BREAKER_FILE.write_text(json.dumps(state))
    except Exception:
        pass


# === SECTION: Geometry helpers (CRS transforms) ===

def transform_to_3035(geom):
    return shapely_transform(_tx_to_3035.transform, geom)

def transform_to_wgs(geom):
    return shapely_transform(_tx_to_wgs.transform, geom)


# === SECTION: KG list (fetch + cache all ~8440 KGs) ===

def get_all_kgs(state_filter: str = None) -> list[dict]:
    """Fetch all KGs from cadastre API. Caches to kg_list.json."""
    # Use cache if <24h old and no filter
    if not state_filter and KG_LIST_FILE.exists():
        age = time.time() - KG_LIST_FILE.stat().st_mtime
        if age < 86400:
            kgs = json.loads(KG_LIST_FILE.read_text())
            log.info("Loaded %d KGs from cache (%s)", len(kgs), KG_LIST_FILE)
            return kgs

    states = ["Steiermark", "Kärnten", "Niederösterreich", "Oberösterreich",
              "Salzburg", "Tirol", "Vorarlberg", "Burgenland", "Wien"]
    if state_filter:
        states = [s for s in states if s.lower() == state_filter.lower()]
        if not states:
            states = [state_filter]  # try exact match

    all_kgs = []
    for state in states:
        try:
            resp = requests.get(f"{CADASTRE_BASE}/search/district",
                                params={"state": state}, timeout=30)
            resp.raise_for_status()
            districts = resp.json().get("data", [])
            for d in districts:
                code = d.get("district_code")
                if not code:
                    continue
                try:
                    resp2 = requests.get(f"{CADASTRE_BASE}/search/kg",
                                         params={"district": d["district_name"], "limit": 500},
                                         timeout=30)
                    resp2.raise_for_status()
                    kgs = resp2.json().get("data", [])
                    for kg in kgs:
                        if kg.get("kg_code"):
                            kg["state_name"] = state
                            all_kgs.append(kg)
                except Exception as e:
                    log.warning("Failed to fetch KGs for district %s: %s", code, e)
        except Exception as e:
            log.warning("Failed to fetch districts for %s: %s", state, e)

    log.info("Fetched %d KGs across %s", len(all_kgs), states)

    # Cache
    if not state_filter:
        KG_LIST_FILE.write_text(json.dumps(all_kgs, indent=1, default=str))

    return all_kgs


# === SECTION: Cadastre data fetching (REST API calls) ===

def fetch_cadastre_data(kg_code: str) -> dict:
    """Fetch parcels, building footprints, landuse from cadastre API."""
    result = {"parcels": [], "building_footprints": [], "landuse": [],
              "parcels_geojson": [], "buildings_geojson": []}
    try:
        resp = requests.get(
            f"{CADASTRE_BASE}/export/geojson",
            params={"kg": kg_code,
                    "layers": "parcels,building_footprints,landuse_polygons",
                    "include_geometry": "true"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("KG %s: cadastre fetch failed: %s", kg_code, e)
        return result

    # Parse parcels
    parcels_fc = data.get("parcels", {}).get("features", [])
    for f in parcels_fc:
        try:
            geom = shapely_shape(f["geometry"])
            if geom.is_empty:
                continue
            geom_3035 = transform_to_3035(geom)
            props = f.get("properties", {})
            result["parcels"].append({
                "geometry": geom_3035,
                "geometry_wgs": geom,
                "properties": props,
                "parcel_id": props.get("parcel_id", ""),
                "area_sqm": props.get("area_sqm", 0),
            })
            result["parcels_geojson"].append(f)
        except Exception:
            continue

    # Parse building footprints
    bfp_fc = data.get("building_footprints", {}).get("features", [])
    for f in bfp_fc:
        try:
            geom = shapely_shape(f["geometry"])
            if geom.is_empty:
                continue
            geom_3035 = transform_to_3035(geom)
            result["building_footprints"].append({
                "geometry": geom_3035,
                "geometry_wgs": geom,
                "properties": f.get("properties", {}),
            })
            result["buildings_geojson"].append(f)
        except Exception:
            continue

    # Parse landuse
    lu_fc = data.get("landuse_polygons", {}).get("features", [])
    for f in lu_fc:
        try:
            geom = shapely_shape(f["geometry"])
            if geom.is_empty:
                continue
            geom_3035 = transform_to_3035(geom)
            props = f.get("properties", {})
            result["landuse"].append({
                "geometry": geom_3035,
                "code": props.get("landuse_code"),
                "abbr": props.get("landuse_abbr", ""),
            })
        except Exception:
            continue

    # --- Filter buildings to KG boundary (parcels define the KG extent) ---
    n_before = len(result["building_footprints"])
    if result["parcels"] and result["building_footprints"]:
        try:
            from shapely.ops import unary_union as _uu
            from shapely.validation import make_valid as _mv
            parcel_geoms = []
            for p in result["parcels"]:
                g = p["geometry"]
                try:
                    parcel_geoms.append(_mv(g) if not g.is_valid else g)
                except Exception:
                    parcel_geoms.append(g.buffer(0))
            kg_boundary = _uu(parcel_geoms).buffer(50)  # 50m buffer
            filtered_bfp = []
            filtered_bfp_geo = []
            for i, b in enumerate(result["building_footprints"]):
                if kg_boundary.contains(b["geometry"].centroid):
                    filtered_bfp.append(b)
                    if i < len(result["buildings_geojson"]):
                        filtered_bfp_geo.append(result["buildings_geojson"][i])
            n_removed = n_before - len(filtered_bfp)
            if n_removed > 0:
                log.info("KG %s: filtered %d/%d buildings outside KG boundary",
                         kg_code, n_removed, n_before)
            result["building_footprints"] = filtered_bfp
            result["buildings_geojson"] = filtered_bfp_geo
        except Exception as e:
            log.warning("KG %s: building filter failed: %s", kg_code, e)

    log.info("KG %s: %d parcels, %d footprints (was %d), %d landuse",
             kg_code, len(result["parcels"]), len(result["building_footprints"]),
             n_before, len(result["landuse"]))
    return result


# === SECTION: Height enrichment (DTM/DSM → parcel + building heights) ===

def enrich_parcels_with_heights(parcels: list, dtm: np.ndarray,
                                transform) -> list[dict]:
    """Add DTM elevation at each parcel vertex + centroid."""
    enriched = []
    for p in parcels:
        geom_3035 = p["geometry"]
        geom_wgs = p["geometry_wgs"]
        props = dict(p.get("properties", {}))

        # Centroid height
        try:
            c3 = geom_3035.centroid
            col = int((c3.x - transform.c) / transform.a)
            row = int((transform.f - c3.y) / abs(transform.e))
            h, w = dtm.shape
            if 0 <= row < h and 0 <= col < w:
                props["centroid_dtm_m"] = round(float(dtm[row, col]), 2)
            c_wgs = transform_to_wgs(c3)
            props["centroid_lon"] = round(c_wgs.x, 7)
            props["centroid_lat"] = round(c_wgs.y, 7)
        except Exception:
            pass

        # Vertex heights
        try:
            coords = []
            if geom_wgs.geom_type == 'Polygon':
                rings = [geom_wgs.exterior] + list(geom_wgs.interiors)
            elif geom_wgs.geom_type == 'MultiPolygon':
                rings = []
                for poly in geom_wgs.geoms:
                    rings.append(poly.exterior)
                    rings.extend(poly.interiors)
            else:
                rings = []

            for ring in rings:
                ring_coords = []
                for x, y in ring.coords:
                    pt_3035 = _tx_to_3035.transform(x, y)
                    col2 = int((pt_3035[0] - transform.c) / transform.a)
                    row2 = int((transform.f - pt_3035[1]) / abs(transform.e))
                    dtm_val = None
                    if 0 <= row2 < dtm.shape[0] and 0 <= col2 < dtm.shape[1]:
                        dtm_val = round(float(dtm[row2, col2]), 2)
                    ring_coords.append({"lon": round(x, 7), "lat": round(y, 7),
                                        "dtm_m": dtm_val})
                coords.append(ring_coords)
            props["vertex_heights"] = coords
        except Exception:
            pass

        enriched.append({"parcel_id": p["parcel_id"], "area_sqm": p["area_sqm"],
                         "properties": props, "geometry_wgs": geom_wgs})
    return enriched


def enrich_buildings_with_heights(buildings: list, dtm: np.ndarray,
                                  dsm: np.ndarray, ndsm: np.ndarray,
                                  transform) -> list[dict]:
    """Add object height stats for each building footprint."""
    from rasterio.features import rasterize as rio_rasterize
    enriched = []
    h, w = dtm.shape

    for b in buildings:
        geom_3035 = b["geometry"]
        geom_wgs = b["geometry_wgs"]
        props = dict(b.get("properties", {}))

        try:
            # Rasterize this single footprint
            fp_mask = rio_rasterize(
                [(geom_3035, 1)], out_shape=(h, w), transform=transform,
                fill=0, dtype=np.uint8, all_touched=True,
            ).astype(bool)

            obj_heights = ndsm[fp_mask]
            dsm_vals = dsm[fp_mask]

            if len(obj_heights) > 0:
                max_h = float(np.nanmax(obj_heights))
                mean_h = float(np.nanmean(obj_heights))
                std_h = float(np.nanstd(dsm_vals))  # DSM std for roof type

                props["max_height_m"] = round(max_h, 2)
                props["mean_height_m"] = round(mean_h, 2)
                props["dsm_std"] = round(std_h, 2)
                props["roof_type_hint"] = "flat" if std_h < 1.5 else "pitched"
                props["stories_est"] = max(1, round(max_h / 3.0))
                props["footprint_area_sqm"] = round(float(geom_3035.area), 1)
        except Exception:
            pass

        # Centroid
        try:
            c3 = geom_3035.centroid
            c_wgs = transform_to_wgs(c3)
            props["centroid_lon"] = round(c_wgs.x, 7)
            props["centroid_lat"] = round(c_wgs.y, 7)
        except Exception:
            pass

        # Vertex heights
        try:
            coords = []
            if geom_wgs.geom_type == 'Polygon':
                rings_w = [geom_wgs.exterior]
            elif geom_wgs.geom_type == 'MultiPolygon':
                rings_w = [p.exterior for p in geom_wgs.geoms]
            else:
                rings_w = []
            for ring in rings_w:
                rc = []
                for x, y in ring.coords:
                    pt_3035 = _tx_to_3035.transform(x, y)
                    col2 = int((pt_3035[0] - transform.c) / transform.a)
                    row2 = int((transform.f - pt_3035[1]) / abs(transform.e))
                    oh = None
                    if 0 <= row2 < h and 0 <= col2 < w:
                        oh = round(float(ndsm[row2, col2]), 2)
                    rc.append({"lon": round(x, 7), "lat": round(y, 7),
                               "object_height_m": oh})
                coords.append(rc)
            props["vertex_heights"] = coords
        except Exception:
            pass

        enriched.append({"properties": props, "geometry_wgs": geom_wgs,
                         "geometry_3035": geom_3035})
    return enriched


# === SECTION: Vectorise unmatched segments (new buildings, infrastructure) ===

def _segment_touches_edge(seg_mask: np.ndarray) -> bool:
    """Return True if any pixel in *seg_mask* lies on the raster boundary.

    Edge-touching segments are likely truncated by the segmentation window
    and their area/shape metrics are unreliable.
    """
    h, w = seg_mask.shape
    if h < 2 or w < 2:
        return True
    return bool(
        seg_mask[0, :].any() or seg_mask[-1, :].any() or
        seg_mask[:, 0].any() or seg_mask[:, -1].any()
    )


def vectorise_unmatched_buildings(objects: list, labels: np.ndarray,
                                  mask: np.ndarray, transform,
                                  cadastre_fp_mask: np.ndarray,
                                  ndsm: np.ndarray) -> list[dict]:
    """Find roof/building segments that don't overlap cadastre footprints."""
    from rasterio.features import shapes as rasterize_shapes

    results = []
    obj_map = {o.obj_id: o for o in objects}

    # Segments classified as roof/building
    building_objs = [o for o in objects if o.obj_type == 'roof']
    if not building_objs:
        return results

    building_ids = {o.obj_id for o in building_objs}
    label_int = labels.astype(np.int32)

    for geom_dict, val in rasterize_shapes(
        label_int, mask=mask & np.isin(label_int, list(building_ids)),
        transform=transform, connectivity=4,
    ):
        oid = int(val)
        obj = obj_map.get(oid)
        if obj is None:
            continue

        # Check overlap with cadastre footprints
        seg_mask = labels == oid
        if cadastre_fp_mask is not None:
            overlap = np.sum(seg_mask & cadastre_fp_mask)
            total = np.sum(seg_mask)
            if total > 0 and overlap / total > 0.3:
                continue  # matches existing cadastre footprint

        # This is a "new" building
        from shapely.geometry import shape as s_shape
        try:
            poly_3035 = s_shape(geom_dict)
            # Skip tiny polygon fragments (< 20 sqm) — these are
            # segmentation noise at edges, not real buildings.
            if poly_3035.area < 20:
                continue
            poly_wgs = transform_to_wgs(poly_3035)
            c_wgs = poly_wgs.centroid
            at_edge = _segment_touches_edge(seg_mask)

            # Height stats from nDSM
            seg_h = ndsm[seg_mask]
            max_h = float(np.nanmax(seg_h)) if len(seg_h) > 0 else 0
            mean_h = float(np.nanmean(seg_h)) if len(seg_h) > 0 else 0

            results.append({
                "type": obj.obj_type,
                "area_sqm": round(float(poly_3035.area), 1),
                "max_height_m": round(max_h, 2),
                "mean_height_m": round(mean_h, 2),
                "stories_est": max(1, round(max_h / 3.0)),
                "roof_type_hint": "flat" if (np.nanstd(seg_h) if len(seg_h) > 0 else 0) < 1.5 else "pitched",
                "centroid_lon": round(c_wgs.x, 7),
                "centroid_lat": round(c_wgs.y, 7),
                "geometry_wgs": mapping(poly_wgs),
                "confidence": round(obj.confidence, 2),
                "edge_clipped": at_edge,
            })
        except Exception:
            continue

    return results


def vectorise_infrastructure(objects: list, labels: np.ndarray,
                             mask: np.ndarray, transform,
                             ndsm: np.ndarray, dtm: np.ndarray,
                             dtm_dates: dict = None) -> list[dict]:
    """Vectorise parking, solar panels, earthworks, clear cuts, etc."""
    from rasterio.features import shapes as rasterize_shapes

    INFRA_TYPES = {
        'parking': {'min_area': 50, 'fields': ['area_sqm', 'est_parking_spots']},
        'solar_panel': {'min_area': 10, 'fields': ['area_sqm']},
        'excavation': {'min_area': 20, 'fields': ['volume_m3']},
        'fill': {'min_area': 20, 'fields': ['volume_m3']},
        'tree_loss': {'min_area': 5000, 'fields': ['area_sqm']},  # 0.5ha = 5000m2
        'bridge': {'min_area': 20, 'fields': ['area_sqm', 'max_height_m']},
        'mast': {'min_area': 1, 'fields': ['max_height_m']},
        'fence': {'min_area': 5, 'fields': ['length_m']},
        'wall': {'min_area': 5, 'fields': ['length_m']},
    }

    results = []
    obj_map = {o.obj_id: o for o in objects}

    for obj in objects:
        if obj.obj_type not in INFRA_TYPES:
            continue
        spec = INFRA_TYPES[obj.obj_type]

        seg_mask = labels == obj.obj_id
        if not seg_mask.any():
            continue
        at_edge = _segment_touches_edge(seg_mask)

        # Area filter — relax for edge-clipped segments since their
        # measured area is a lower bound (the real feature continues
        # beyond the segmentation window).
        effective_area = obj.area_sqm
        if at_edge and obj.obj_type == 'tree_loss':
            # Edge-clipped tree_loss: only require 500m² (the full
            # clearing might be huge, we just see the edge of it)
            if effective_area < 500:
                continue
        elif obj.obj_type == 'tree_loss':
            if effective_area < 5000:
                continue  # interior clear cuts > 0.5ha only
        else:
            if effective_area < spec['min_area']:
                continue

        try:
            label_single = np.where(seg_mask, 1, 0).astype(np.int32)
            from shapely.geometry import shape as s_shape

            # Collect all polygons for this segment and pick the largest
            all_polys = [
                s_shape(gd)
                for gd, val in rasterize_shapes(
                    label_single, mask=seg_mask, transform=transform, connectivity=4,
                )
                if val != 0
            ]
            if not all_polys:
                continue
            poly_3035 = max(all_polys, key=lambda p: p.area)

            # Apply min_area filter on the actual polygon area (not segment-level)
            if poly_3035.area < spec['min_area']:
                continue

            poly_wgs = transform_to_wgs(poly_3035)
            c_wgs = poly_wgs.centroid

            seg_h = ndsm[seg_mask]
            seg_dtm = dtm[seg_mask]

            feature = {
                "type": obj.obj_type,
                "area_sqm": round(float(poly_3035.area), 1),
                "centroid_lon": round(c_wgs.x, 7),
                "centroid_lat": round(c_wgs.y, 7),
                "geometry_wgs": mapping(poly_wgs),
                "confidence": round(obj.confidence, 2),
                "edge_clipped": at_edge,
            }

            if obj.obj_type == 'parking':
                # ~12.5 m2 per parking spot (standard)
                feature["est_parking_spots"] = max(1, round(poly_3035.area / 12.5))

            if obj.obj_type in ('excavation', 'fill'):
                # Volume estimate: mean absolute height change * area
                mean_h = abs(float(np.nanmean(seg_h))) if len(seg_h) > 0 else 0
                volume = mean_h * poly_3035.area
                feature["volume_m3"] = round(volume, 1)
                if volume < 10 and not at_edge:
                    continue  # skip tiny earthworks (unless edge-clipped)

            if obj.obj_type in ('bridge', 'mast'):
                feature["max_height_m"] = round(float(np.nanmax(seg_h)), 2) if len(seg_h) > 0 else 0

            if obj.obj_type in ('fence', 'wall'):
                # Length estimate from perimeter / 2 (rectangular assumption)
                feature["length_m"] = round(poly_3035.length / 2, 1)
                feature["max_height_m"] = round(float(np.nanmax(seg_h)), 2) if len(seg_h) > 0 else 0

            results.append(feature)
        except Exception:
            continue

    return results


# === SECTION: Resolve edge-clipped features (tile boundary fixup) ===

def resolve_edge_clipped_features(
    new_buildings: list,
    infrastructure: list,
    full_ndsm: np.ndarray,
    full_dsm: np.ndarray,
    full_mask: np.ndarray,
    full_transform,
) -> tuple[list, list]:
    """Re-measure edge-clipped buildings & infra using the full-KG nDSM.

    For each feature with ``edge_clipped=True`` whose type has height above
    ground (roofs, solar, mast, excavation, fill, bridge), we rasterize the
    clipped polygon into the *full-KG* nDSM, then flood-fill outwards from
    that seed into contiguous pixels that match the height profile.  The
    resulting polygon replaces the clipped one and the flag is cleared to
    ``edge_resolved``.

    Tree-loss keeps its flag because it depends on temporal differencing which
    only exists inside the segmentation window.

    Returns (new_buildings, infrastructure) — mutated copies.
    """
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as s_shape
    from scipy.ndimage import binary_dilation

    fh, fw = full_ndsm.shape
    RESOLVABLE_TYPES = {'roof', 'greenhouse', 'solar_panel', 'parking',
                        'excavation', 'fill', 'bridge', 'mast'}

    def _try_resolve(feat: dict) -> dict:
        """Attempt to grow a clipped feature into full-KG nDSM."""
        if not feat.get('edge_clipped'):
            return feat
        if feat.get('type') not in RESOLVABLE_TYPES:
            return feat

        try:
            geom_wgs = feat['geometry_wgs']
            if isinstance(geom_wgs, dict):
                poly_wgs = s_shape(geom_wgs)
            else:
                poly_wgs = geom_wgs
            poly_3035 = transform_to_3035(poly_wgs)

            # Rasterize seed into full-KG grid
            seed = rio_rasterize(
                [(poly_3035, 1)], out_shape=(fh, fw), transform=full_transform,
                fill=0, dtype=np.uint8, all_touched=True).astype(bool)
            if seed.sum() < 2:
                return feat

            # Height profile of the seed region
            seed_h = full_ndsm[seed & full_mask]
            seed_h = seed_h[np.isfinite(seed_h)]
            if len(seed_h) < 2:
                return feat
            h_lo = float(np.percentile(seed_h, 10)) - 2.0
            h_hi = float(np.percentile(seed_h, 90)) + 2.0

            # Height-compatible mask in full-KG
            compatible = full_mask & np.isfinite(full_ndsm) & \
                         (full_ndsm >= h_lo) & (full_ndsm <= h_hi)

            # Iterative binary dilation from seed, bounded by compatible
            grown = seed.copy()
            for _ in range(200):  # max ~200m growth
                expanded = binary_dilation(grown, iterations=1) & compatible
                if expanded.sum() == grown.sum():
                    break
                grown = expanded

            if grown.sum() <= seed.sum():
                return feat  # no growth

            # Vectorize the grown mask
            grown_int = grown.astype(np.int32)
            polys = []
            for geom_dict, val in rio_shapes(grown_int, mask=grown,
                                             transform=full_transform):
                if val == 1:
                    polys.append(s_shape(geom_dict))
            if not polys:
                return feat

            from shapely.ops import unary_union as _uu
            merged_3035 = _uu(polys)
            merged_wgs = transform_to_wgs(merged_3035)

            # Recompute stats on grown region
            grown_h = full_ndsm[grown]
            grown_h = grown_h[np.isfinite(grown_h)]
            max_h = float(np.nanmax(grown_h)) if len(grown_h) > 0 else feat.get('max_height_m', 0)
            mean_h = float(np.nanmean(grown_h)) if len(grown_h) > 0 else feat.get('mean_height_m', 0)

            grown_dsm = full_dsm[grown]
            grown_dsm = grown_dsm[np.isfinite(grown_dsm)]
            dsm_std = float(np.std(grown_dsm)) if len(grown_dsm) > 0 else 0.0

            resolved = dict(feat)
            resolved['geometry_wgs'] = mapping(merged_wgs)
            resolved['area_sqm'] = round(float(merged_3035.area), 1)
            resolved['max_height_m'] = round(max_h, 2)
            if 'mean_height_m' in resolved:
                resolved['mean_height_m'] = round(mean_h, 2)
            c_wgs = merged_wgs.centroid
            resolved['centroid_lon'] = round(c_wgs.x, 7)
            resolved['centroid_lat'] = round(c_wgs.y, 7)
            resolved['edge_clipped'] = False
            resolved['edge_resolved'] = True

            # Type-specific updates
            if feat.get('type') in ('roof', 'greenhouse', 'solar_panel'):
                resolved['stories_est'] = max(1, round(max_h / 3.0))
                resolved['roof_type_hint'] = 'flat' if dsm_std < 1.5 else 'pitched'
            if feat.get('type') == 'parking':
                resolved['est_parking_spots'] = max(1, round(merged_3035.area / 12.5))
            if feat.get('type') in ('excavation', 'fill'):
                resolved['volume_m3'] = round(abs(mean_h) * merged_3035.area, 1)

            log.info("  edge_resolved %s: %s m² → %s m²",
                     feat.get('type'), feat.get('area_sqm'), resolved['area_sqm'])
            return resolved
        except Exception as e:
            log.debug("  edge_resolve failed for %s: %s", feat.get('type'), e)
            return feat

    resolved_nb = [_try_resolve(nb) for nb in new_buildings]
    resolved_infra = [_try_resolve(inf) for inf in infrastructure]
    return resolved_nb, resolved_infra


# === SECTION: GPKG CRS fix + style + vector writers (shared by full + light builders) ===

def _fix_gpkg_raster_crs(gpkg_path: str):
    """Ensure all raster layers in a GPKG have valid CRS registration.

    GDAL's GPKG driver registers uint8 rasters as data_type='tiles' instead of
    '2d-gridded-coverage'. QGIS may not read CRS from 'tiles'-type layers,
    showing a "Layer has no CRS" warning. This function:
      1. Converts any 'tiles' entries in gpkg_contents to '2d-gridded-coverage'
      2. Ensures the gpkg_2d_gridded_coverage_ancillary table + rows exist
      3. Ensures the gpkg_2d_gridded_tile_ancillary table exists
      4. Registers the gridded coverage extension in gpkg_extensions
    """
    import sqlite3
    if not os.path.exists(gpkg_path) or os.path.getsize(gpkg_path) == 0:
        return
    conn = sqlite3.connect(gpkg_path)
    try:
        # Find raster layers registered as 'tiles' that should be '2d-gridded-coverage'
        tile_rows = conn.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type = 'tiles'"
        ).fetchall()
        if not tile_rows:
            return  # nothing to fix

        tile_names = [r[0] for r in tile_rows]
        log.info("Fixing GPKG raster CRS for %d 'tiles' layers: %s",
                 len(tile_names), tile_names)

        # 1. Update data_type
        conn.execute(
            "UPDATE gpkg_contents SET data_type = '2d-gridded-coverage' "
            "WHERE data_type = 'tiles'"
        )

        # 2. Ensure ancillary tables exist
        conn.execute('''CREATE TABLE IF NOT EXISTS gpkg_2d_gridded_coverage_ancillary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tile_matrix_set_name TEXT NOT NULL,
            datatype TEXT NOT NULL DEFAULT 'integer',
            scale REAL NOT NULL DEFAULT 1.0,
            "offset" REAL NOT NULL DEFAULT 0.0,
            precision REAL DEFAULT 1.0,
            data_null REAL,
            grid_cell_encoding TEXT DEFAULT 'grid-value-is-center',
            uom TEXT,
            field_name TEXT DEFAULT 'Height',
            quantity_definition TEXT DEFAULT 'Height'
        )''')
        conn.execute('''CREATE TABLE IF NOT EXISTS gpkg_2d_gridded_tile_ancillary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tpudt_name TEXT NOT NULL,
            tpudt_id INTEGER NOT NULL,
            scale REAL NOT NULL DEFAULT 1.0,
            "offset" REAL NOT NULL DEFAULT 0.0,
            min REAL DEFAULT NULL,
            max REAL DEFAULT NULL,
            mean REAL DEFAULT NULL,
            std_dev REAL DEFAULT NULL
        )''')

        # 3. Add ancillary rows for newly-converted layers
        for tname in tile_names:
            existing = conn.execute(
                "SELECT 1 FROM gpkg_2d_gridded_coverage_ancillary "
                "WHERE tile_matrix_set_name = ?", (tname,)
            ).fetchone()
            if not existing:
                conn.execute(
                    'INSERT INTO gpkg_2d_gridded_coverage_ancillary '
                    '(tile_matrix_set_name, datatype, scale, "offset") '
                    'VALUES (?, ?, ?, ?)',
                    (tname, 'integer', 1.0, 0.0)
                )

        # 4. Register the gridded coverage extension
        try:
            conn.execute('''CREATE TABLE IF NOT EXISTS gpkg_extensions (
                table_name TEXT,
                column_name TEXT,
                extension_name TEXT NOT NULL,
                definition TEXT NOT NULL,
                scope TEXT NOT NULL,
                CONSTRAINT ge_tce UNIQUE (table_name, column_name, extension_name)
            )''')
            for tname in tile_names:
                conn.execute(
                    'INSERT OR IGNORE INTO gpkg_extensions '
                    '(table_name, column_name, extension_name, definition, scope) '
                    'VALUES (?, NULL, ?, ?, ?)',
                    (tname, 'gpkg_2d_gridded_coverage',
                     'http://www.geopackage.org/18-000.html', 'read-write')
                )
        except Exception:
            pass  # extension table may already have constraints

        conn.commit()
    except Exception as e:
        log.warning("GPKG CRS fix failed for %s: %s", gpkg_path, e)
    finally:
        conn.close()

SEGMENT_COLORS = {
    "tree":         (0, 100, 0, 180),
    "shrub":        (34, 139, 34, 180),
    "grass":        (124, 252, 0, 150),
    "hedge":        (46, 139, 87, 170),
    "water":        (30, 144, 255, 180),
    "roof":         (220, 20, 60, 200),
    "greenhouse":   (255, 105, 180, 180),
    "solar_panel":  (65, 105, 225, 200),
    "fence":        (160, 82, 45, 170),
    "wall":         (139, 69, 19, 170),
    "mast":         (64, 64, 64, 200),
    "wind_turbine": (21, 101, 192, 200),
    "substation":   (255, 111, 0, 200),
    "road":         (128, 128, 128, 160),
    "path":         (169, 169, 169, 150),
    "parking":      (105, 105, 105, 160),
    "bridge":       (112, 128, 144, 170),
    "crop":         (218, 165, 32, 160),
    "orchard":      (107, 142, 35, 170),
    "vineyard":     (147, 112, 219, 170),
    "garden":       (60, 179, 113, 160),
    "bare_soil":    (210, 180, 140, 140),
    "rock":         (139, 134, 130, 160),
    "excavation":   (139, 0, 0, 200),
    "fill":         (255, 140, 0, 200),
    "tree_loss":    (255, 0, 255, 200),
    "construction": (255, 69, 0, 200),
    "unclassified": (190, 190, 190, 120),
}


def _to_multi(geom_dict: dict) -> dict:
    """Promote a Polygon geometry dict to MultiPolygon for fiona schema compat."""
    if geom_dict and geom_dict.get('type') == 'Polygon':
        return {'type': 'MultiPolygon', 'coordinates': [geom_dict['coordinates']]}
    return geom_dict


def _height_class(h):
    """Classify height (m) into a forestry-relevant height class string."""
    if h is None or h < 0.5:
        return 'ground'
    if h < 2:
        return 'low (<2m)'
    if h < 5:
        return 'shrub (2-5m)'
    if h < 10:
        return 'young (5-10m)'
    if h < 15:
        return 'pole (10-15m)'
    if h < 20:
        return 'mid (15-20m)'
    if h < 25:
        return 'mature (20-25m)'
    if h < 30:
        return 'tall (25-30m)'
    return 'emergent (30m+)'


def _viridis_rgb(t):
    """Return (R,G,B) for t in [0,1] on the viridis scale."""
    VIRIDIS = [
        (68,1,84),(72,35,116),(64,67,135),(52,94,141),(41,120,142),
        (32,144,140),(34,167,132),(68,190,112),(121,209,81),(189,222,38),(253,231,37)
    ]
    t = max(0.0, min(1.0, t))
    idx = t * (len(VIRIDIS) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(VIRIDIS) - 1)
    f = idx - lo
    return tuple(int(VIRIDIS[lo][c] + f * (VIRIDIS[hi][c] - VIRIDIS[lo][c])) for c in range(3))


def _write_gpkg_categorized_style(gpkg_path: str, layer_name: str,
                                   color_mode: str = 'type'):
    """Write a QGIS-compatible layer_styles table for auto-rendering."""
    import sqlite3
    color_field = 'color_height' if color_mode == 'height' else 'color'
    qml = (
        '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
        '<qgis version="3.34">'
        '<renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0">'
        '<symbols>'
        '<symbol type="fill" name="0" clip_to_extent="1" alpha="0.7">'
        '<layer class="SimpleFill" enabled="1" locked="0" pass="0">'
        '<Option type="Map">'
        '<Option type="QString" value="solid" name="style"/>'
        '<Option type="QString" value="0.35,0.35,0.35,255,rgb:0,0,0,1" name="outline_color"/>'
        '<Option type="QString" value="0.2" name="outline_width"/>'
        '</Option>'
        f'<data_defined_properties><Property><Option type="Map">'
        f'<Option type="Map" name="properties"><Option type="Map" name="fillColor">'
        f'<Option type="bool" value="true" name="active"/>'
        f'<Option type="QString" value="&quot;{color_field}&quot;" name="expression"/>'
        f'<Option type="int" value="3" name="type"/>'
        f'</Option></Option></Option></Property></data_defined_properties>'
        '</layer></symbol></symbols></renderer-v2></qgis>'
    )
    conn = sqlite3.connect(gpkg_path)
    try:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS layer_styles ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'f_table_catalog TEXT DEFAULT \'\','
            'f_table_schema TEXT DEFAULT \'\','
            'f_table_name TEXT,'
            'f_geometry_column TEXT,'
            'styleName TEXT,'
            'styleQML TEXT,'
            'styleSLD TEXT,'
            'useAsDefault BOOLEAN,'
            'description TEXT,'
            'owner TEXT,'
            'ui TEXT,'
            'update_time TIMESTAMP DEFAULT (strftime(\'%Y-%m-%dT%H:%M:%fZ\',\'now\'))'
            ')'
        )
        conn.execute(
            'INSERT INTO layer_styles '
            '(f_table_name, f_geometry_column, styleName, styleQML, useAsDefault, description) '
            'VALUES (?, ?, ?, ?, 1, ?)',
            (layer_name, 'geom', f'Segment {color_mode}', qml,
             f'Auto-generated colour-by-{color_mode} style'),
        )
        conn.commit()
    finally:
        conn.close()


def _write_segment_vectors(gpkg_path: str, labels: np.ndarray,
                           objects: list, mask: np.ndarray,
                           transform, layer_name: str = 'segments',
                           obs_year: int = 0):
    """Write vectorised segment polygons with full attributes to a GPKG.

    Includes all SegmentedObject attributes: height, shape, spectral,
    temporal change, texture, SAR, harmonics, phenology, and
    observation period metadata.
    """
    from rasterio.features import shapes as rasterize_shapes
    import fiona
    from fiona.crs import from_epsg

    obj_map = {o.obj_id: o for o in objects}
    label_int = labels.astype(np.int32)

    schema = {
        'geometry': 'MultiPolygon',
        'properties': [
            # Identity
            ('id', 'int'),
            ('type', 'str'),
            ('type_code', 'int'),
            ('group_type', 'str'),
            ('height_class', 'str'),
            # Geometry
            ('area_sqm', 'float'),
            ('perimeter_m', 'float'),
            ('compactness', 'float'),
            ('elongation', 'float'),
            ('solidity', 'float'),
            ('extent', 'float'),
            # Height
            ('height_max_m', 'float'),
            ('height_mean_m', 'float'),
            ('height_p90_m', 'float'),
            ('height_std_m', 'float'),
            # Surface
            ('slope_mean_deg', 'float'),
            ('roughness', 'float'),
            ('dsm_edge_strength', 'float'),
            # Spectral
            ('ndvi_mean', 'float'),
            ('ndvi_std', 'float'),
            ('ndvi_fused', 'float'),
            ('brightness_mean', 'float'),
            ('nir_mean', 'float'),
            # Temporal change
            ('height_change_m', 'float'),
            ('dtm_change_m', 'float'),
            ('temporal_stability', 'float'),
            ('volume_change_m3', 'float'),
            ('volume_change_abs_m3', 'float'),
            ('dtm_change_max_m', 'float'),
            # Texture (GLCM)
            ('glcm_entropy', 'float'),
            ('glcm_homogeneity', 'float'),
            ('texture_complexity', 'float'),
            # SAR
            ('sar_vv', 'float'),
            ('sar_vh', 'float'),
            # Phenology
            ('harm_amplitude', 'float'),
            ('harm_phase', 'float'),
            ('phenology_class', 'str'),
            # Classification
            ('confidence', 'float'),
            ('is_manmade', 'int'),
            # Rendering
            ('color', 'str'),
            ('color_height', 'str'),
            # Observation period
            ('obs_year', 'int'),
        ],
    }
    with fiona.open(gpkg_path, 'w', driver='GPKG', layer=layer_name,
                    schema=schema, crs=from_epsg(3035)) as dst:
        written = 0
        for geom_dict, val in rasterize_shapes(
            label_int, mask=mask, transform=transform, connectivity=4,
        ):
            oid = int(val)
            obj = obj_map.get(oid)
            if obj is None:
                continue
            tc = SEGMENT_COLORS.get(obj.obj_type, (128, 128, 128, 120))
            hex_type = '#{:02X}{:02X}{:02X}'.format(tc[0], tc[1], tc[2])
            hv = _viridis_rgb(min(1.0, (max(0, obj.height_max) / 45.0) ** 0.5))
            hex_height = '#{:02X}{:02X}{:02X}'.format(*hv)
            dst.write({
                'geometry': _to_multi(geom_dict),
                'properties': {
                    'id': oid,
                    'type': obj.obj_type,
                    'type_code': obj.type_code,
                    'group_type': obj.group_type or '',
                    'height_class': _height_class(obj.height_max),
                    'area_sqm': round(obj.area_sqm, 1),
                    'perimeter_m': round(obj.perimeter_m, 1),
                    'compactness': round(obj.compactness, 3),
                    'elongation': round(obj.elongation, 2),
                    'solidity': round(obj.solidity, 3),
                    'extent': round(obj.extent, 3),
                    'height_max_m': round(obj.height_max, 2),
                    'height_mean_m': round(obj.height_mean, 2),
                    'height_p90_m': round(obj.height_p90, 2),
                    'height_std_m': round(obj.height_std, 2),
                    'slope_mean_deg': round(obj.slope_mean, 1),
                    'roughness': round(obj.roughness, 3),
                    'dsm_edge_strength': round(obj.dsm_edge_strength, 3),
                    'ndvi_mean': round(obj.ndvi_mean, 4),
                    'ndvi_std': round(obj.ndvi_std, 4),
                    'ndvi_fused': round(obj.ndvi_fused, 4),
                    'brightness_mean': round(obj.brightness_mean, 1),
                    'nir_mean': round(obj.nir_mean, 1),
                    'height_change_m': round(obj.height_change, 3),
                    'dtm_change_m': round(obj.dtm_change, 3),
                    'temporal_stability': round(obj.temporal_stability, 3),
                    'volume_change_m3': round(obj.volume_change_m3, 1),
                    'volume_change_abs_m3': round(obj.volume_change_abs_m3, 1),
                    'dtm_change_max_m': round(obj.dtm_change_max, 3),
                    'glcm_entropy': round(obj.glcm_entropy, 4),
                    'glcm_homogeneity': round(obj.glcm_homogeneity, 4),
                    'texture_complexity': round(obj.texture_complexity, 4),
                    'sar_vv': round(obj.sar_vv, 4),
                    'sar_vh': round(obj.sar_vh, 4),
                    'harm_amplitude': round(obj.harm_amplitude, 4),
                    'harm_phase': round(obj.harm_phase, 1),
                    'phenology_class': obj.phenology_class or '',
                    'confidence': round(obj.confidence, 3),
                    'is_manmade': int(obj.is_manmade) if obj.is_manmade else 0,
                    'color': hex_type,
                    'color_height': hex_height,
                    'obs_year': obs_year or 0,
                },
            })
            written += 1
    log.info("GPKG vector layer '%s': %d polygons", layer_name, written)

    try:
        _write_gpkg_categorized_style(gpkg_path, layer_name, 'type')
    except Exception as e:
        log.warning('GPKG style table failed: %s', e)


def _write_segment_points(gpkg_path: str, objects: list,
                          layer_name: str = 'segment_points',
                          obs_year: int = 0):
    """Write segment centroid points with full attributes to a GPKG.

    This mirrors the GeoJSON Point features that the API returns,
    enabling the same map visualisation from the GPKG/Zenodo store.
    """
    import fiona
    from fiona.crs import from_epsg

    schema = {
        'geometry': 'Point',
        'properties': [
            # Identity
            ('id', 'int'),
            ('type', 'str'),
            ('type_code', 'int'),
            ('group_type', 'str'),
            ('height_class', 'str'),
            # Geometry
            ('area_sqm', 'float'),
            ('perimeter_m', 'float'),
            ('compactness', 'float'),
            ('elongation', 'float'),
            ('solidity', 'float'),
            ('extent', 'float'),
            # Height
            ('height_max_m', 'float'),
            ('height_mean_m', 'float'),
            ('height_p90_m', 'float'),
            ('height_std_m', 'float'),
            # Surface
            ('slope_mean_deg', 'float'),
            ('roughness', 'float'),
            ('dsm_edge_strength', 'float'),
            # Spectral
            ('ndvi_mean', 'float'),
            ('ndvi_fused', 'float'),
            ('brightness_mean', 'float'),
            ('nir_mean', 'float'),
            # Temporal
            ('height_change_m', 'float'),
            ('dtm_change_m', 'float'),
            ('temporal_stability', 'float'),
            ('volume_change_m3', 'float'),
            ('volume_change_abs_m3', 'float'),
            ('dtm_change_max_m', 'float'),
            # Texture
            ('glcm_entropy', 'float'),
            ('glcm_homogeneity', 'float'),
            ('texture_complexity', 'float'),
            # SAR
            ('sar_vv', 'float'),
            ('sar_vh', 'float'),
            # Phenology
            ('harm_amplitude', 'float'),
            ('harm_phase', 'float'),
            ('phenology_class', 'str'),
            # Classification
            ('confidence', 'float'),
            ('is_manmade', 'int'),
            # Rendering
            ('color', 'str'),
            ('color_height', 'str'),
            # Observation
            ('obs_year', 'int'),
        ],
    }

    with fiona.open(gpkg_path, 'w', driver='GPKG', layer=layer_name,
                    schema=schema, crs=from_epsg(4326)) as dst:
        written = 0
        for obj in objects:
            # Convert centroid from EPSG:3035 to WGS84
            try:
                lon, lat = _tx_to_wgs.transform(obj.centroid_e, obj.centroid_n)
            except Exception:
                continue
            tc = SEGMENT_COLORS.get(obj.obj_type, (128, 128, 128, 120))
            hex_type = '#{:02X}{:02X}{:02X}'.format(tc[0], tc[1], tc[2])
            hv = _viridis_rgb(min(1.0, (max(0, obj.height_max) / 45.0) ** 0.5))
            hex_height = '#{:02X}{:02X}{:02X}'.format(*hv)
            dst.write({
                'geometry': {'type': 'Point', 'coordinates': [round(lon, 7), round(lat, 7)]},
                'properties': {
                    'id': obj.obj_id,
                    'type': obj.obj_type,
                    'type_code': obj.type_code,
                    'group_type': obj.group_type or '',
                    'height_class': _height_class(obj.height_max),
                    'area_sqm': round(obj.area_sqm, 1),
                    'perimeter_m': round(obj.perimeter_m, 1),
                    'compactness': round(obj.compactness, 3),
                    'elongation': round(obj.elongation, 2),
                    'solidity': round(obj.solidity, 3),
                    'extent': round(obj.extent, 3),
                    'height_max_m': round(obj.height_max, 2),
                    'height_mean_m': round(obj.height_mean, 2),
                    'height_p90_m': round(obj.height_p90, 2),
                    'height_std_m': round(obj.height_std, 2),
                    'slope_mean_deg': round(obj.slope_mean, 1),
                    'roughness': round(obj.roughness, 3),
                    'dsm_edge_strength': round(obj.dsm_edge_strength, 3),
                    'ndvi_mean': round(obj.ndvi_mean, 4),
                    'ndvi_fused': round(obj.ndvi_fused, 4),
                    'brightness_mean': round(obj.brightness_mean, 1),
                    'nir_mean': round(obj.nir_mean, 1),
                    'height_change_m': round(obj.height_change, 3),
                    'dtm_change_m': round(obj.dtm_change, 3),
                    'temporal_stability': round(obj.temporal_stability, 3),
                    'volume_change_m3': round(obj.volume_change_m3, 1),
                    'volume_change_abs_m3': round(obj.volume_change_abs_m3, 1),
                    'dtm_change_max_m': round(obj.dtm_change_max, 3),
                    'glcm_entropy': round(obj.glcm_entropy, 4),
                    'glcm_homogeneity': round(obj.glcm_homogeneity, 4),
                    'texture_complexity': round(obj.texture_complexity, 4),
                    'sar_vv': round(obj.sar_vv, 4),
                    'sar_vh': round(obj.sar_vh, 4),
                    'harm_amplitude': round(obj.harm_amplitude, 4),
                    'harm_phase': round(obj.harm_phase, 1),
                    'phenology_class': obj.phenology_class or '',
                    'confidence': round(obj.confidence, 3),
                    'is_manmade': int(obj.is_manmade) if obj.is_manmade else 0,
                    'color': hex_type,
                    'color_height': hex_height,
                    'obs_year': obs_year or 0,
                },
            })
            written += 1
    log.info("GPKG point layer '%s': %d points", layer_name, written)

    # Add point style
    try:
        _write_gpkg_point_style(gpkg_path, layer_name)
    except Exception as e:
        log.warning('GPKG point style failed: %s', e)


def _write_gpkg_point_style(gpkg_path: str, layer_name: str):
    """Write QGIS-compatible point style using data-defined colour."""
    import sqlite3
    qml = (
        '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
        '<qgis version="3.34">'
        '<renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0">'
        '<symbols>'
        '<symbol type="marker" name="0" clip_to_extent="1" alpha="0.8">'
        '<layer class="SimpleMarker" enabled="1" locked="0" pass="0">'
        '<Option type="Map">'
        '<Option type="QString" value="circle" name="name"/>'
        '<Option type="QString" value="3" name="size"/>'
        '<Option type="QString" value="0.35,0.35,0.35,255" name="outline_color"/>'
        '<Option type="QString" value="0.2" name="outline_width"/>'
        '</Option>'
        '<data_defined_properties><Property><Option type="Map">'
        '<Option type="Map" name="properties"><Option type="Map" name="fillColor">'
        '<Option type="bool" value="true" name="active"/>'
        '<Option type="QString" value="&quot;color&quot;" name="expression"/>'
        '<Option type="int" value="3" name="type"/>'
        '</Option></Option></Option></Property></data_defined_properties>'
        '</layer></symbol></symbols></renderer-v2></qgis>'
    )
    conn = sqlite3.connect(gpkg_path)
    try:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS layer_styles ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'f_table_catalog TEXT DEFAULT \'\','
            'f_table_schema TEXT DEFAULT \'\','
            'f_table_name TEXT,'
            'f_geometry_column TEXT,'
            'styleName TEXT,'
            'styleQML TEXT,'
            'styleSLD TEXT,'
            'useAsDefault BOOLEAN,'
            'description TEXT,'
            'owner TEXT,'
            'ui TEXT,'
            'update_time TIMESTAMP DEFAULT (strftime(\'%Y-%m-%dT%H:%M:%fZ\',\'now\'))'
            ')'
        )
        conn.execute(
            'INSERT INTO layer_styles '
            '(f_table_name, f_geometry_column, styleName, styleQML, useAsDefault, description) '
            'VALUES (?, ?, ?, ?, 1, ?)',
            (layer_name, 'geom', 'Segment points by type', qml,
             'Auto-generated colour-by-type point style'),
        )
        conn.commit()
    finally:
        conn.close()


def _write_gpkg_all_styles(gpkg_path: str, has_segments: bool = True,
                           has_points: bool = True, has_parcels: bool = False,
                           has_buildings: bool = False, has_new_buildings: bool = False,
                           has_infrastructure: bool = False):
    """Write QGIS styles for all vector and raster layers in a GPKG."""
    import sqlite3
    conn = sqlite3.connect(gpkg_path)
    try:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS layer_styles ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'f_table_catalog TEXT DEFAULT \'\','
            'f_table_schema TEXT DEFAULT \'\','
            'f_table_name TEXT,'
            'f_geometry_column TEXT,'
            'styleName TEXT,'
            'styleQML TEXT,'
            'styleSLD TEXT,'
            'useAsDefault BOOLEAN,'
            'description TEXT,'
            'owner TEXT,'
            'ui TEXT,'
            'update_time TIMESTAMP DEFAULT (strftime(\'%Y-%m-%dT%H:%M:%fZ\',\'now\'))'
            ')'
        )

        def _add_style(table, geom_col, name, qml, desc):
            conn.execute(
                'INSERT INTO layer_styles '
                '(f_table_name, f_geometry_column, styleName, styleQML, useAsDefault, description) '
                'VALUES (?, ?, ?, ?, 1, ?)',
                (table, geom_col, name, qml, desc),
            )

        # --- Vector layer styles ---

        # Parcels: yellow outline, transparent fill
        if has_parcels:
            _add_style('parcels', 'geom', 'Parcel boundaries',
                '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
                '<qgis version="3.34"><renderer-v2 type="singleSymbol">'
                '<symbols><symbol type="fill" name="0" alpha="0.3">'
                '<layer class="SimpleFill"><Option type="Map">'
                '<Option type="QString" value="255,255,200,50" name="color"/>'
                '<Option type="QString" value="200,180,0,255" name="outline_color"/>'
                '<Option type="QString" value="0.4" name="outline_width"/>'
                '</Option></layer></symbol></symbols></renderer-v2></qgis>',
                'Yellow outline parcel boundaries')

        # Buildings: red fill, dark outline
        if has_buildings:
            _add_style('buildings', 'geom', 'Buildings',
                '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
                '<qgis version="3.34"><renderer-v2 type="singleSymbol">'
                '<symbols><symbol type="fill" name="0" alpha="0.6">'
                '<layer class="SimpleFill"><Option type="Map">'
                '<Option type="QString" value="220,20,60,160" name="color"/>'
                '<Option type="QString" value="100,10,30,255" name="outline_color"/>'
                '<Option type="QString" value="0.3" name="outline_width"/>'
                '</Option></layer></symbol></symbols></renderer-v2></qgis>',
                'Red building footprints')

        # New buildings: magenta, dashed
        if has_new_buildings:
            _add_style('new_buildings', 'geom', 'New buildings',
                '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
                '<qgis version="3.34"><renderer-v2 type="singleSymbol">'
                '<symbols><symbol type="fill" name="0" alpha="0.7">'
                '<layer class="SimpleFill"><Option type="Map">'
                '<Option type="QString" value="255,0,255,180" name="color"/>'
                '<Option type="QString" value="180,0,180,255" name="outline_color"/>'
                '<Option type="QString" value="0.5" name="outline_width"/>'
                '<Option type="QString" value="dash" name="outline_style"/>'
                '</Option></layer></symbol></symbols></renderer-v2></qgis>',
                'Magenta dashed new building detections')

        # Infrastructure: orange
        if has_infrastructure:
            _add_style('infrastructure', 'geom', 'Infrastructure',
                '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
                '<qgis version="3.34"><renderer-v2 type="singleSymbol">'
                '<symbols><symbol type="fill" name="0" alpha="0.6">'
                '<layer class="SimpleFill"><Option type="Map">'
                '<Option type="QString" value="255,140,0,160" name="color"/>'
                '<Option type="QString" value="180,100,0,255" name="outline_color"/>'
                '<Option type="QString" value="0.3" name="outline_width"/>'
                '</Option></layer></symbol></symbols></renderer-v2></qgis>',
                'Orange infrastructure features')

        # --- Raster layer styles ---
        # Detect which raster tables exist in this GPKG
        raster_tables = set()
        try:
            rows = conn.execute(
                "SELECT table_name FROM gpkg_contents "
                "WHERE data_type IN ('tiles', '2d-gridded-coverage')"
            ).fetchall()
            raster_tables = {r[0] for r in rows}
        except Exception:
            pass

        # Build segment_type categorized colour map from SEGMENT_COLORS
        if 'segment_type' in raster_tables:
            _cat_items = []
            for type_name, type_code in sorted(SEGMENT_COLORS.items(), key=lambda x: x[1]):
                from object_segmentation import OBJECT_TYPES as _OT
                code = _OT.get(type_name)
                if code is None:
                    continue
                r, g, b, a = SEGMENT_COLORS[type_name]
                _cat_items.append(
                    f'<item value="{code}" label="{type_name}" '
                    f'color="{r},{g},{b}" alpha="{a}"/>'
                )
            seg_qml = (
                '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
                '<qgis version="3.34">'
                '<pipe><rasterrenderer type="paletted" band="1" opacity="0.8">'
                '<colorPalette>' + ''.join(_cat_items) + '</colorPalette>'
                '</rasterrenderer></pipe></qgis>'
            )
            _add_style('segment_type', '', 'Object types', seg_qml,
                       'Categorized colour map by object type code')

        # DTM/DSM: terrain colour ramp (green-brown-white)
        dtm_qml = (
            '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
            '<qgis version="3.34"><pipe>'
            '<rasterrenderer type="singlebandpseudocolor" band="1" opacity="1">'
            '<rastershader><colorrampshader colorRampType="INTERPOLATED" '
            'classificationMode="2" minimumValue="0" maximumValue="3000">'
            '<item value="0" color="#1a9641" label="0m"/>'
            '<item value="500" color="#a6d96a" label="500m"/>'
            '<item value="1000" color="#ffffbf" label="1000m"/>'
            '<item value="2000" color="#a0522d" label="2000m"/>'
            '<item value="3000" color="#ffffff" label="3000m"/>'
            '</colorrampshader></rastershader></rasterrenderer></pipe></qgis>'
        )
        if 'DTM' in raster_tables:
            _add_style('DTM', '', 'Elevation', dtm_qml,
                       'Terrain elevation colour ramp')
        if 'DSM' in raster_tables:
            _add_style('DSM', '', 'Surface model', dtm_qml,
                       'Digital surface model colour ramp')

        # nDSM / segment_height: height colour ramp (viridis-style)
        height_qml = (
            '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
            '<qgis version="3.34"><pipe>'
            '<rasterrenderer type="singlebandpseudocolor" band="1" opacity="1">'
            '<rastershader><colorrampshader colorRampType="INTERPOLATED" '
            'classificationMode="2" minimumValue="0" maximumValue="30">'
            '<item value="0" color="#440154" label="0m"/>'
            '<item value="5" color="#31688e" label="5m"/>'
            '<item value="10" color="#35b779" label="10m"/>'
            '<item value="20" color="#90d743" label="20m"/>'
            '<item value="30" color="#fde725" label="30m"/>'
            '</colorrampshader></rastershader></rasterrenderer></pipe></qgis>'
        )
        if 'nDSM' in raster_tables:
            _add_style('nDSM', '', 'Object heights', height_qml,
                       'Normalised DSM height colour ramp')
        if 'segment_height' in raster_tables:
            _add_style('segment_height', '', 'Segment heights', height_qml,
                       'Segment height colour ramp')

        # Multi-band RGB rendering (Ortho + CIR)
        _rgb_qml = (
            '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
            '<qgis version="3.34"><pipe>'
            '<rasterrenderer type="multibandcolor" opacity="1" '
            'redBand="1" greenBand="2" blueBand="3">'
            '</rasterrenderer></pipe></qgis>'
        )
        for tbl in raster_tables:
            if tbl.startswith('Ortho_'):
                _add_style(tbl, '', 'True colour', _rgb_qml,
                           'RGB orthophoto rendering')
            elif tbl.startswith('CIR_'):
                _add_style(tbl, '', 'CIR false colour', _rgb_qml,
                           'Colour Infrared: NIR→R, Red→G, Green→B')
            elif tbl.startswith('DTM_'):
                _add_style(tbl, '', 'Elevation', dtm_qml,
                           f'Terrain elevation {tbl}')
            elif tbl.startswith('DSM_'):
                _add_style(tbl, '', 'Surface model', dtm_qml,
                           f'Digital surface model {tbl}')

        # --- Sentinel-2 NDVI: green-yellow-brown diverging ramp ---
        if 'NDVI' in raster_tables:
            ndvi_qml = (
                '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
                '<qgis version="3.34"><pipe>'
                '<rasterrenderer type="singlebandpseudocolor" band="1" opacity="1">'
                '<rastershader><colorrampshader colorRampType="INTERPOLATED" '
                'classificationMode="2" minimumValue="-0.2" maximumValue="1.0">'
                '<item value="-0.2" color="#d73027" label="-0.2 (bare/water)"/>'
                '<item value="0.0" color="#f46d43" label="0.0"/>'
                '<item value="0.1" color="#fdae61" label="0.1"/>'
                '<item value="0.2" color="#fee08b" label="0.2"/>'
                '<item value="0.3" color="#ffffbf" label="0.3"/>'
                '<item value="0.4" color="#d9ef8b" label="0.4"/>'
                '<item value="0.5" color="#a6d96a" label="0.5"/>'
                '<item value="0.6" color="#66bd63" label="0.6"/>'
                '<item value="0.7" color="#1a9850" label="0.7"/>'
                '<item value="0.8" color="#006837" label="0.8 (dense veg)"/>'
                '<item value="1.0" color="#004529" label="1.0"/>'
                '</colorrampshader></rastershader></rasterrenderer></pipe></qgis>'
            )
            _add_style('NDVI', '', 'Sentinel-2 NDVI', ndvi_qml,
                       'NDVI composite: red = bare/water, green = vegetation')

        # --- ESA WorldCover: official class colour scheme ---
        if 'WorldCover' in raster_tables:
            wc_qml = (
                '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
                '<qgis version="3.34"><pipe>'
                '<rasterrenderer type="paletted" band="1" opacity="0.85">'
                '<colorPalette>'
                '<item value="10" color="#006400" label="Tree cover" alpha="255"/>'
                '<item value="20" color="#ffbb22" label="Shrubland" alpha="255"/>'
                '<item value="30" color="#ffff4c" label="Grassland" alpha="255"/>'
                '<item value="40" color="#f096ff" label="Cropland" alpha="255"/>'
                '<item value="50" color="#fa0000" label="Built-up" alpha="255"/>'
                '<item value="60" color="#b4b4b4" label="Bare / sparse veg" alpha="255"/>'
                '<item value="70" color="#f0f0f0" label="Snow and ice" alpha="255"/>'
                '<item value="80" color="#0064c8" label="Permanent water" alpha="255"/>'
                '<item value="90" color="#0096a0" label="Herbaceous wetland" alpha="255"/>'
                '<item value="95" color="#00cf75" label="Mangroves" alpha="255"/>'
                '<item value="100" color="#fae6a0" label="Moss and lichen" alpha="255"/>'
                '</colorPalette>'
                '</rasterrenderer></pipe></qgis>'
            )
            _add_style('WorldCover', '', 'ESA WorldCover 2021', wc_qml,
                       'Land cover classes with official ESA colour scheme')

        # --- Sentinel-1 SAR (VV+VH): grey ramp for radar backscatter (dB) ---
        if 'SAR' in raster_tables:
            sar_qml = (
                '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
                '<qgis version="3.34"><pipe>'
                '<rasterrenderer type="multibandcolor" opacity="1" '
                'redBand="1" greenBand="2" blueBand="1">'
                '<redContrastEnhancement><minValue>-25</minValue>'
                '<maxValue>0</maxValue>'
                '<algorithm>StretchToMinimumMaximum</algorithm>'
                '</redContrastEnhancement>'
                '<greenContrastEnhancement><minValue>-30</minValue>'
                '<maxValue>-5</maxValue>'
                '<algorithm>StretchToMinimumMaximum</algorithm>'
                '</greenContrastEnhancement>'
                '<blueContrastEnhancement><minValue>-25</minValue>'
                '<maxValue>0</maxValue>'
                '<algorithm>StretchToMinimumMaximum</algorithm>'
                '</blueContrastEnhancement>'
                '</rasterrenderer></pipe></qgis>'
            )
            _add_style('SAR', '', 'SAR VV-VH composite', sar_qml,
                       'Sentinel-1 SAR: Red/Blue=VV, Green=VH (stretched dB)')

        # --- Hansen tree cover 2000: green ramp 0-100% ---
        if 'Hansen_treecover' in raster_tables:
            tc_qml = (
                '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
                '<qgis version="3.34"><pipe>'
                '<rasterrenderer type="singlebandpseudocolor" band="1" opacity="0.9">'
                '<rastershader><colorrampshader colorRampType="INTERPOLATED" '
                'classificationMode="2" minimumValue="0" maximumValue="100">'
                '<item value="0" color="#f7f7f7" label="0% (no trees)"/>'
                '<item value="10" color="#d9f0a3" label="10%"/>'
                '<item value="25" color="#addd8e" label="25%"/>'
                '<item value="50" color="#78c679" label="50%"/>'
                '<item value="75" color="#31a354" label="75%"/>'
                '<item value="100" color="#006837" label="100% (full canopy)"/>'
                '</colorrampshader></rastershader></rasterrenderer></pipe></qgis>'
            )
            _add_style('Hansen_treecover', '', 'Tree cover 2000', tc_qml,
                       'Hansen Global Forest: tree canopy density in year 2000')

        # --- Hansen loss year: diverging yellow→red timeline ---
        if 'Hansen_lossyear' in raster_tables:
            ly_qml = (
                '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
                '<qgis version="3.34"><pipe>'
                '<rasterrenderer type="paletted" band="1" opacity="0.9">'
                '<colorPalette>'
                '<item value="0" color="#ffffff" label="No loss" alpha="0"/>'
                '<item value="1" color="#ffffcc" label="2001" alpha="255"/>'
                '<item value="5" color="#fed976" label="2005" alpha="255"/>'
                '<item value="10" color="#fd8d3c" label="2010" alpha="255"/>'
                '<item value="15" color="#e31a1c" label="2015" alpha="255"/>'
                '<item value="20" color="#b10026" label="2020" alpha="255"/>'
                '<item value="23" color="#800026" label="2023" alpha="255"/>'
                '</colorPalette>'
                '</rasterrenderer></pipe></qgis>'
            )
            _add_style('Hansen_lossyear', '', 'Forest loss year', ly_qml,
                       'Hansen Global Forest: year of forest loss (1=2001 … 23=2023)')

        conn.commit()
    finally:
        conn.close()



def _compute_tile_grid(west, south, east, north, tile_km=1.5, overlap_km=0.1):
    """Compute overlapping tiles covering the full KG bbox. Returns [(w,s,e,n)]."""
    cos_lat = np.cos(np.radians((south + north) / 2))
    dx_deg = tile_km / (111 * cos_lat)
    dy_deg = tile_km / 111
    step_x = (tile_km - overlap_km) / (111 * cos_lat)
    step_y = (tile_km - overlap_km) / 111
    tiles = []
    y = south
    while y < north:
        x = west
        while x < east:
            tiles.append((x, y, min(x + dx_deg, east + dx_deg), min(y + dy_deg, north + dy_deg)))
            x += step_x
        y += step_y
    return tiles


def _merge_terrain_stats(stats_list):
    """Merge terrain stats from tiles via pixel-weighted averages."""
    if not stats_list:
        return {}
    if len(stats_list) == 1:
        return stats_list[0][0]
    total_px = sum(n for _, n in stats_list)
    if total_px == 0:
        return stats_list[0][0]

    def _wmean(kp):
        vals = []
        for s, n in stats_list:
            v = s
            for k in kp:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None: break
            if v is not None and isinstance(v, (int, float)):
                vals.append((v, n))
        return round(sum(v*w for v,w in vals)/sum(w for _,w in vals), 3) if vals else None

    def _extreme(kp, fn):
        vals = []
        for s, _ in stats_list:
            v = s
            for k in kp:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None: break
            if v is not None and isinstance(v, (int, float)):
                vals.append(v)
        return round(fn(vals), 2) if vals else None

    merged = {
        "elevation": {
            "min": _extreme(["elevation","min"], min),
            "max": _extreme(["elevation","max"], max),
            "mean": _wmean(["elevation","mean"]),
            "std": _wmean(["elevation","std"]),
            "range": None,
            "p10": _wmean(["elevation","p10"]),
            "p50": _wmean(["elevation","p50"]),
            "p90": _wmean(["elevation","p90"]),
        },
        "slope_deg": {
            "min": _extreme(["slope_deg","min"], min),
            "max": _extreme(["slope_deg","max"], max),
            "mean": _wmean(["slope_deg","mean"]),
            "std": _wmean(["slope_deg","std"]),
        },
        "slope_classes_pct": {}, "aspect_distribution_pct": {},
        "ruggedness_tri": {
            "mean": _wmean(["ruggedness_tri","mean"]),
            "max": _extreme(["ruggedness_tri","max"], max),
            "classification": None,
        },
        "area_sqm": total_px,
        "area_ha": round(total_px / 10000, 2),
    }
    em, ex = merged["elevation"]["min"], merged["elevation"]["max"]
    if em is not None and ex is not None:
        merged["elevation"]["range"] = round(ex - em, 2)
    for cls_key in ["slope_classes_pct", "aspect_distribution_pct"]:
        all_keys = set()
        for s, _ in stats_list:
            all_keys.update(s.get(cls_key, {}).keys())
        for k in all_keys:
            merged[cls_key][k] = round(
                sum(s.get(cls_key, {}).get(k, 0) * n for s, n in stats_list) / total_px, 1)
    tri_mean = merged["ruggedness_tri"]["mean"]
    if tri_mean is not None:
        for thr, lbl in [(0.1,"level"),(0.3,"nearly level"),(1.0,"slightly rugged"),
                         (3.0,"intermediately rugged"),(10.0,"moderately rugged")]:
            if tri_mean < thr:
                merged["ruggedness_tri"]["classification"] = lbl
                break
        else:
            merged["ruggedness_tri"]["classification"] = "highly rugged"
    return merged


def _find_tile_for_point(e3035, n3035, tile_seg_results):
    """Find tile containing point (e, n) EPSG:3035."""
    for tr in tile_seg_results:
        left, bottom, right, top = tr["bounds_3035"]
        if left <= e3035 <= right and bottom <= n3035 <= top:
            return tr
    return None


def _read_dtm_for_tile(tr):
    """Re-read DTM/DSM for a tile (BEV cache makes this fast)."""
    import raster_io as _rio
    import tile_index as _ti
    return _rio.read_dtm_dsm(box(*tr["bounds_3035"]), _ti.DEFAULT_DATASET)


# === SECTION: Boundary segment merging (cross-tile stitching) ===

def _merge_boundary_segments(
    labels_full: np.ndarray,
    seg_type_full: np.ndarray,
    all_objects: list,
    tile_seg_results: list,
    full_left: float,
    full_top: float,
    res: float,
    ndsm_full: np.ndarray | None = None,
    rf_only: bool = False,
):
    """Merge compatible segments that touch across tile boundaries.

    Segments split by tile boundaries often have too few pixels for reliable
    classification.  This function:
      1. Computes the overlap zones from tile_seg_results bounds.
      2. Scans overlap zones for adjacent segment-pairs from different tiles.
      3. Merges compatible pairs (union-find) using the same _MERGE_RULES.
      4. Remaps labels_full / seg_type_full so the merged segment has one ID.
      5. Recomputes area-weighted stats for the surviving SegmentedObject.
      6. Re-classifies merged segments.

    Modifies labels_full, seg_type_full, and all_objects **in place**.
    Returns the number of merges performed.
    """
    from object_segmentation import _MERGE_RULES, OBJECT_TYPES, classify_object

    full_h, full_w = labels_full.shape
    obj_map = {o.obj_id: o for o in all_objects}
    if len(obj_map) < 2:
        return 0

    # ------------------------------------------------------------------
    # 1. Build a tile-ownership raster to identify cross-tile adjacency.
    #    Each pixel gets the index of the tile that "owns" it (highest
    #    feather weight).  Adjacent pixels with different tile owners
    #    AND different labels are cross-tile boundary candidates.
    # ------------------------------------------------------------------
    tile_owner = np.full((full_h, full_w), -1, dtype=np.int16)
    tile_best_w = np.zeros((full_h, full_w), dtype=np.float32)

    def _fw(rows, cols, margin=100):
        wy = np.ones(rows, dtype=np.float32)
        wx = np.ones(cols, dtype=np.float32)
        m = min(margin, rows // 2, cols // 2)
        if m > 0:
            ramp = np.linspace(0.0, 1.0, m, endpoint=False, dtype=np.float32)
            wy[:m] = np.minimum(wy[:m], ramp)
            wy[-m:] = np.minimum(wy[-m:], ramp[::-1])
            wx[:m] = np.minimum(wx[:m], ramp)
            wx[-m:] = np.minimum(wx[-m:], ramp[::-1])
        return wy[:, None] * wx[None, :]

    for ti_idx, tr in enumerate(tile_seg_results):
        th, tw = tr["shape"]
        tile_left, tile_bottom, tile_right, tile_top = tr["bounds_3035"]
        col_off = int(round((tile_left - full_left) / res))
        row_off = int(round((full_top - tile_top) / res))
        r_end = min(row_off + th, full_h)
        c_end = min(col_off + tw, full_w)
        th_eff = r_end - row_off
        tw_eff = c_end - col_off
        if th_eff <= 0 or tw_eff <= 0:
            continue
        fw = _fw(th_eff, tw_eff, margin=100)
        wins = fw > tile_best_w[row_off:r_end, col_off:c_end]
        tile_owner[row_off:r_end, col_off:c_end] = np.where(
            wins, ti_idx, tile_owner[row_off:r_end, col_off:c_end])
        tile_best_w[row_off:r_end, col_off:c_end] = np.maximum(
            tile_best_w[row_off:r_end, col_off:c_end], fw)
    del tile_best_w

    # ------------------------------------------------------------------
    # 2. Find adjacent segment pairs across tile boundaries.
    #    Scan horizontal and vertical neighbours; keep pairs where
    #    tile_owner differs AND labels differ AND both labels are valid.
    # ------------------------------------------------------------------
    adj_pairs: set[tuple[int, int]] = set()

    def _collect_cross_pairs(owner_a, owner_b, label_a, label_b):
        """Find unique (min,max) label pairs across tile boundaries."""
        cross = ((owner_a != owner_b) & (owner_a >= 0) & (owner_b >= 0)
                 & (label_a != label_b) & (label_a > 0) & (label_b > 0))
        if not cross.any():
            return
        rc = np.nonzero(cross)
        la = label_a[rc].astype(np.int64)
        lb = label_b[rc].astype(np.int64)
        # Vectorised unique-pair extraction
        mn = np.minimum(la, lb)
        mx = np.maximum(la, lb)
        # Pack into single int64 for fast unique
        packed = mn * (np.int64(labels_full.max()) + 1) + mx
        for p in np.unique(packed):
            divisor = int(labels_full.max()) + 1
            a_id = int(p // divisor)
            b_id = int(p % divisor)
            adj_pairs.add((a_id, b_id))

    # Vertical adjacency: (r, c) vs (r, c+1)
    _collect_cross_pairs(
        tile_owner[:, :-1], tile_owner[:, 1:],
        labels_full[:, :-1], labels_full[:, 1:])
    # Horizontal adjacency: (r, c) vs (r+1, c)
    _collect_cross_pairs(
        tile_owner[:-1, :], tile_owner[1:, :],
        labels_full[:-1, :], labels_full[1:, :])
    del tile_owner

    if not adj_pairs:
        return 0

    # ------------------------------------------------------------------
    # 3. Filter to compatible pairs and build union-find
    # ------------------------------------------------------------------
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Larger area survives as root
        oa = obj_map.get(ra)
        ob = obj_map.get(rb)
        area_a = oa.area_sqm if oa else 0
        area_b = ob.area_sqm if ob else 0
        if area_a >= area_b:
            parent[rb] = ra
        else:
            parent[ra] = rb

    n_compat = 0
    for id_a, id_b in adj_pairs:
        oa = obj_map.get(id_a)
        ob = obj_map.get(id_b)
        if oa is None or ob is None:
            continue
        # Check compatibility: same type or merge-rule compatible
        if oa.obj_type == ob.obj_type:
            union(id_a, id_b)
            n_compat += 1
        else:
            compat_a = _MERGE_RULES.get(oa.obj_type, set())
            if ob.obj_type in compat_a:
                union(id_a, id_b)
                n_compat += 1

    if n_compat == 0:
        return 0

    # ------------------------------------------------------------------
    # 4. Build merge groups
    # ------------------------------------------------------------------
    groups: dict[int, list[int]] = {}  # root_id → [member_ids]
    all_ids = set()
    for id_a, id_b in adj_pairs:
        all_ids.add(id_a)
        all_ids.add(id_b)
    for oid in all_ids:
        root = find(oid)
        groups.setdefault(root, set()).add(oid)
    # Only keep groups with actual merges
    merge_groups = {root: members for root, members in groups.items()
                    if len(members) > 1}
    if not merge_groups:
        return 0

    # ------------------------------------------------------------------
    # 5. Remap labels and recompute stats
    # ------------------------------------------------------------------
    n_merges = 0
    remap = {}  # old_id → new_id (survivor)
    remove_ids = set()

    for root_id, member_ids in merge_groups.items():
        survivor = obj_map.get(root_id)
        if survivor is None:
            continue
        absorbed = [obj_map[mid] for mid in member_ids
                    if mid != root_id and mid in obj_map]
        if not absorbed:
            continue

        # Area-weighted merge of stats
        total_area = survivor.area_sqm + sum(o.area_sqm for o in absorbed)
        if total_area <= 0:
            continue

        def _wmean(attr):
            """Area-weighted mean of an attribute."""
            val = getattr(survivor, attr) * survivor.area_sqm
            for o in absorbed:
                val += getattr(o, attr) * o.area_sqm
            return val / total_area

        # Recompute survivor stats
        survivor.area_sqm = total_area
        survivor.perimeter_m = survivor.perimeter_m + sum(o.perimeter_m for o in absorbed)  # approximate
        survivor.height_max = max(survivor.height_max, *(o.height_max for o in absorbed))
        survivor.height_mean = _wmean('height_mean')
        survivor.height_p90 = max(survivor.height_p90, *(o.height_p90 for o in absorbed))
        survivor.height_std = _wmean('height_std')
        survivor.slope_mean = _wmean('slope_mean')
        survivor.roughness = _wmean('roughness')
        survivor.ndvi_mean = _wmean('ndvi_mean')
        survivor.ndvi_std = _wmean('ndvi_std')
        survivor.ndvi_fused = _wmean('ndvi_fused')
        survivor.brightness_mean = _wmean('brightness_mean')
        survivor.nir_mean = _wmean('nir_mean')
        survivor.height_change = _wmean('height_change')
        survivor.dtm_change = _wmean('dtm_change')
        survivor.temporal_stability = _wmean('temporal_stability')
        survivor.volume_change_m3 = survivor.volume_change_m3 + sum(o.volume_change_m3 for o in absorbed)
        survivor.volume_change_abs_m3 = survivor.volume_change_abs_m3 + sum(o.volume_change_abs_m3 for o in absorbed)
        survivor.dtm_change_max = max(survivor.dtm_change_max, *(o.dtm_change_max for o in absorbed))
        survivor.glcm_entropy = _wmean('glcm_entropy')
        survivor.glcm_homogeneity = _wmean('glcm_homogeneity')
        survivor.texture_complexity = _wmean('texture_complexity')
        survivor.sar_vv = _wmean('sar_vv')
        survivor.sar_vh = _wmean('sar_vh')
        survivor.harm_amplitude = _wmean('harm_amplitude')
        survivor.harm_phase = _wmean('harm_phase')
        # Recompute compactness (approximate)
        if survivor.perimeter_m > 0:
            survivor.compactness = round(
                4 * 3.14159 * survivor.area_sqm / (survivor.perimeter_m ** 2), 4)

        # Recompute centroid from pixel coordinates
        # (done after label remap below)

        # Build remap
        for o in absorbed:
            remap[o.obj_id] = root_id
            remove_ids.add(o.obj_id)
        n_merges += len(absorbed)

    if not remap:
        return 0

    # Save pre-merge types for anchor-density check in step 7
    _pre_merge_types = {o.obj_id: o.obj_type for o in all_objects}
    for oid in remove_ids:
        o = obj_map.get(oid)
        if o:
            _pre_merge_types[oid] = o.obj_type

    # ------------------------------------------------------------------
    # 6. Apply label remap to the full rasters (vectorised)
    # ------------------------------------------------------------------
    # Build a lookup array for fast remapping
    max_label = int(labels_full.max())
    lut = np.arange(max_label + 1, dtype=np.int32)
    for old_id, new_id in remap.items():
        if old_id <= max_label:
            lut[old_id] = new_id
    labels_full[:] = lut[labels_full]

    # Update seg_type_full for remapped labels
    for root_id in merge_groups:
        survivor = obj_map.get(root_id)
        if survivor is None:
            continue
        seg_type_full[labels_full == root_id] = survivor.type_code

    # ------------------------------------------------------------------
    # 7. Re-classify merged segments with anchor-density validation
    #
    # "Anchor density" = how many original same-type segments support
    # the merged polygon.  A 10 ha polygon classified as "roof" from
    # one tiny segment is very weak; 10 ha of "tree" from 50 segments
    # is solid.  When area-per-anchor exceeds the threshold, we force
    # re-classification and penalise confidence.
    # ------------------------------------------------------------------
    _ANCHOR_MAX_AREA = {
        # Manmade: individual structures rarely > these sizes
        'roof': 2000, 'wall': 500, 'fence': 500, 'mast': 200,
        'greenhouse': 2000, 'solar_panel': 1000, 'substation': 3000,
        'wind_turbine': 5000,
        # Transport: roads are elongated but segments stay bounded
        'road': 5000, 'path': 3000, 'parking': 3000, 'bridge': 1000,
        # Vegetation: canopy segments are naturally larger
        'tree': 20000, 'shrub': 15000, 'hedge': 5000,
        'grass': 30000, 'crop': 50000, 'garden': 10000,
        'orchard': 20000, 'vineyard': 15000,
        # Natural / disturbance
        'water': 50000, 'bare_soil': 30000, 'rock': 30000,
        'excavation': 10000, 'fill': 10000, 'construction': 5000,
        'tree_loss': 20000, 'earthwork': 10000,
    }
    _MANMADE_TYPES = {
        'roof', 'wall', 'fence', 'mast', 'greenhouse', 'solar_panel',
        'road', 'path', 'parking', 'bridge', 'wind_turbine', 'substation'}

    try:
        from learned_classifier import get_classifier
        rf = get_classifier()
    except Exception:
        rf = None

    n_reclassified = 0
    n_anchor_weak = 0
    for root_id, member_ids in merge_groups.items():
        survivor = obj_map.get(root_id)
        if survivor is None:
            continue

        # --- Anchor-density check ---
        current_type = survivor.obj_type
        # Count how many original (pre-merge) segments had a compatible type
        n_anchors = 0
        compat_set = _MERGE_RULES.get(current_type, set()) | {current_type}
        for mid in member_ids:
            orig_type = _pre_merge_types.get(mid, '')
            if orig_type in compat_set:
                n_anchors += 1
        n_anchors = max(n_anchors, 1)

        area_per_anchor = survivor.area_sqm / n_anchors
        max_area = _ANCHOR_MAX_AREA.get(current_type, 10000)
        anchor_weak = area_per_anchor > max_area

        if anchor_weak:
            n_anchor_weak += 1
            log.info("  Anchor-weak: id=%d type=%s area=%.0fm² anchors=%d "
                     "(%.0fm²/anchor, limit=%d)",
                     root_id, current_type, survivor.area_sqm,
                     n_anchors, area_per_anchor, max_area)

        # --- Build features for re-classification ---
        feat = {
            'area_sqm': survivor.area_sqm,
            'height_max': survivor.height_max,
            'height_mean': survivor.height_mean,
            'height_p90': survivor.height_p90,
            'height_std': survivor.height_std,
            'slope_mean': survivor.slope_mean,
            'roughness': survivor.roughness,
            'ndvi_mean': survivor.ndvi_mean,
            'ndvi_std': survivor.ndvi_std,
            'ndvi_fused': survivor.ndvi_fused,
            'brightness_mean': survivor.brightness_mean,
            'nir_mean': survivor.nir_mean,
            'height_change': survivor.height_change,
            'dtm_change': survivor.dtm_change,
            'temporal_stability': survivor.temporal_stability,
            'compactness': survivor.compactness,
            'elongation': survivor.elongation,
            'solidity': survivor.solidity,
            'extent': survivor.extent,
            'dsm_edge_strength': survivor.dsm_edge_strength,
            'volume_change_m3': survivor.volume_change_m3,
            'volume_change_abs_m3': survivor.volume_change_abs_m3,
            'dtm_change_max': survivor.dtm_change_max,
            'glcm_entropy': survivor.glcm_entropy,
            'glcm_homogeneity': survivor.glcm_homogeneity,
            'texture_complexity': survivor.texture_complexity,
            'sar_vv': survivor.sar_vv,
            'sar_vh': survivor.sar_vh,
            'harm_amplitude': survivor.harm_amplitude,
            'harm_phase': survivor.harm_phase,
            'perimeter_m': survivor.perimeter_m,
        }
        if survivor.features:
            for k, v in survivor.features.items():
                if k not in feat:
                    feat[k] = v

        new_type = None
        new_conf = survivor.confidence
        if rf and rf.model is not None:
            try:
                new_type, new_conf = rf.predict(feat)
            except Exception:
                pass
        if new_type is None:
            try:
                new_type, _, new_conf, _ = classify_object(
                    feat, has_spectral=bool(survivor.ndvi_mean))
            except Exception:
                new_type = None

        # Anchor-weak: if the re-classified type is also implausible
        # for the anchor count, mark as unclassified.
        if anchor_weak and new_type:
            new_max = _ANCHOR_MAX_AREA.get(new_type, 10000)
            new_area_per = survivor.area_sqm / n_anchors
            if new_area_per > new_max:
                # Re-classified type is also too large per anchor → unclassified
                log.info("    → unclassified (re-class %s also exceeds %.0f/anchor)",
                         new_type, new_max)
                new_type = 'unclassified'
                new_conf = min(new_conf * 0.3, 0.2)
            else:
                # Re-classified type fits the anchor density → accept but penalise
                new_conf = min(new_conf * 0.7, 0.5)

        if new_type and new_type in OBJECT_TYPES:
            if new_type != current_type:
                n_reclassified += 1
            survivor.obj_type = new_type
            survivor.type_code = OBJECT_TYPES[new_type]
            survivor.confidence = new_conf
            survivor.is_manmade = new_type in _MANMADE_TYPES
            seg_type_full[labels_full == root_id] = survivor.type_code

    # ------------------------------------------------------------------
    # 8. Recompute centroids for merged segments
    # ------------------------------------------------------------------
    for root_id in merge_groups:
        survivor = obj_map.get(root_id)
        if survivor is None:
            continue
        pix = np.argwhere(labels_full == root_id)
        if len(pix) > 0:
            mean_row = pix[:, 0].mean()
            mean_col = pix[:, 1].mean()
            survivor.centroid_e = full_left + (mean_col + 0.5) * res
            survivor.centroid_n = full_top - (mean_row + 0.5) * res

    # ------------------------------------------------------------------
    # 9. Remove absorbed objects from all_objects
    # ------------------------------------------------------------------
    all_objects[:] = [o for o in all_objects if o.obj_id not in remove_ids]

    log.info("  Boundary merge: %d pairs found, %d segments merged into %d groups, "
             "%d reclassified, %d anchor-weak",
             len(adj_pairs), n_merges, len(merge_groups),
             n_reclassified, n_anchor_weak)
    return n_merges


# === SECTION: Tiled GPKG + JSON builders (build_full/light_gpkg, build_json_summary) ===

def build_full_gpkg_tiled(kg_code, tile_seg_results, all_objects, obs_year, rf_only=False):
    """Full GPKG: stitched raster layers (DTM, DSM, nDSM, segment_type,
    segment_height) covering the full KG bbox, plus unified segment vectors."""
    import rasterio
    import rasterio.transform
    out_path = str(GPKG_DIR / f"{kg_code}_full.gpkg")
    if os.path.exists(out_path): os.unlink(out_path)
    if not tile_seg_results:
        open(out_path, 'w').close()
        return out_path

    table_count = 0
    _GPKG_NODATA = -9999.0  # NaN not supported in GPKG ancillary table

    def _write_table(name, arrays, h, w, tf, dtype='float32', descs=None):
        nonlocal table_count
        opts = dict(driver='GPKG', width=w, height=h, count=len(arrays),
                    dtype=dtype, crs='EPSG:3035', transform=tf,
                    RASTER_TABLE=name, RASTER_IDENTIFIER=name)
        if dtype == 'float32': opts['nodata'] = _GPKG_NODATA
        if table_count > 0: opts['APPEND_SUBDATASET'] = 'YES'
        with rasterio.open(out_path, 'w', **opts) as dst:
            for i, arr in enumerate(arrays, 1):
                band = arr[:h, :w]
                if dtype == 'float32':
                    band = np.where(np.isnan(band), _GPKG_NODATA, band)
                dst.write(band, i)
                if descs and i <= len(descs): dst.set_band_description(i, descs[i-1])
        table_count += 1

    # ------------------------------------------------------------------
    # Compute union bounding box of all tiles
    # ------------------------------------------------------------------
    all_lefts = [tr["bounds_3035"][0] for tr in tile_seg_results]
    all_bottoms = [tr["bounds_3035"][1] for tr in tile_seg_results]
    all_rights = [tr["bounds_3035"][2] for tr in tile_seg_results]
    all_tops = [tr["bounds_3035"][3] for tr in tile_seg_results]
    full_left = min(all_lefts)
    full_bottom = min(all_bottoms)
    full_right = max(all_rights)
    full_top = max(all_tops)

    res = 1.0  # 1m resolution
    full_w = int(round((full_right - full_left) / res))
    full_h = int(round((full_top - full_bottom) / res))
    full_tf = rasterio.transform.from_bounds(
        full_left, full_bottom, full_right, full_top, full_w, full_h)

    log.info("  Full-KG raster: %d x %d px (%.0f x %.0f m), %d tiles",
             full_w, full_h, full_right - full_left, full_top - full_bottom,
             len(tile_seg_results))

    # ------------------------------------------------------------------
    # Allocate full-KG rasters (weighted accumulation for seamless blending)
    # ------------------------------------------------------------------
    # For continuous rasters (DTM/DSM/nDSM), use weighted accumulation
    # to feather-blend overlapping tiles and eliminate stitching lines.
    dtm_sum = np.zeros((full_h, full_w), dtype=np.float64)
    dsm_sum = np.zeros((full_h, full_w), dtype=np.float64)
    ndsm_sum = np.zeros((full_h, full_w), dtype=np.float64)
    weight_sum = np.zeros((full_h, full_w), dtype=np.float64)
    seg_type_full = np.zeros((full_h, full_w), dtype=np.uint8)
    seg_height_full = np.full((full_h, full_w), np.nan, dtype=np.float32)
    labels_full = np.zeros((full_h, full_w), dtype=np.int32)
    # For categorical rasters (labels/type), track best tile per pixel
    # so overlap zones use the tile whose centre is closest (= highest feather weight).
    best_cat_weight = np.zeros((full_h, full_w), dtype=np.float32)

    obj_map = {o.obj_id: o for o in all_objects}

    def _feather_weight(rows, cols, margin=100):
        """Build a 2D weight array that tapers from 1.0 at centre to 0.0 at edges.

        margin is the feather width in pixels (= overlap_km * 1000 = 100m @ 1m res).
        Pixels beyond margin from any edge get weight 1.0.
        """
        wy = np.ones(rows, dtype=np.float32)
        wx = np.ones(cols, dtype=np.float32)
        m = min(margin, rows // 2, cols // 2)
        if m > 0:
            ramp = np.linspace(0.0, 1.0, m, endpoint=False, dtype=np.float32)
            wy[:m] = np.minimum(wy[:m], ramp)
            wy[-m:] = np.minimum(wy[-m:], ramp[::-1])
            wx[:m] = np.minimum(wx[:m], ramp)
            wx[-m:] = np.minimum(wx[-m:], ramp[::-1])
        return wy[:, None] * wx[None, :]  # outer product → 2D

    # ------------------------------------------------------------------
    # Paint each tile into the full-KG rasters
    # ------------------------------------------------------------------
    for ti_idx, tr in enumerate(tile_seg_results):
        th, tw = tr["shape"]
        tile_left, tile_bottom, tile_right, tile_top = tr["bounds_3035"]

        # Compute pixel offset of this tile within the full-KG raster
        col_off = int(round((tile_left - full_left) / res))
        row_off = int(round((full_top - tile_top) / res))

        # Clamp to full raster bounds (safety)
        r_end = min(row_off + th, full_h)
        c_end = min(col_off + tw, full_w)
        th_eff = r_end - row_off
        tw_eff = c_end - col_off
        if th_eff <= 0 or tw_eff <= 0:
            continue

        # DTM / DSM / nDSM — feather-blended
        try:
            tdata = _read_dtm_for_tile(tr)
            tile_dtm = tdata["dtm"][:th_eff, :tw_eff].astype(np.float32)
            tile_dsm = tdata["dsm"][:th_eff, :tw_eff].astype(np.float32)
            tile_ndsm = tdata["ndsm"][:th_eff, :tw_eff].astype(np.float32)
            del tdata

            fw = _feather_weight(th_eff, tw_eff, margin=100)
            valid = ~np.isnan(tile_dtm)
            w = np.where(valid, fw, 0.0).astype(np.float64)

            dtm_sum[row_off:r_end, col_off:c_end] += np.where(valid, tile_dtm, 0.0).astype(np.float64) * w
            dsm_sum[row_off:r_end, col_off:c_end] += np.where(valid, tile_dsm, 0.0).astype(np.float64) * w
            ndsm_sum[row_off:r_end, col_off:c_end] += np.where(valid, tile_ndsm, 0.0).astype(np.float64) * w
            weight_sum[row_off:r_end, col_off:c_end] += w
        except Exception as e:
            log.warning("GPKG tile %d DTM re-read failed: %s", ti_idx + 1, e)

        # Segment type + segment height + labels (categorical — highest-weight-wins)
        # Only overwrite a pixel if this tile has a higher feather weight
        # (i.e. the pixel is closer to this tile's centre than any previous tile).
        if tr.get("labels") is not None:
            labels_tile = tr["labels"][:th_eff, :tw_eff].astype(np.int32)

            type_tile = np.zeros((th_eff, tw_eff), dtype=np.uint8)
            for uid in np.unique(labels_tile):
                if uid == 0: continue
                obj = obj_map.get(int(uid))
                if obj: type_tile[labels_tile == uid] = obj.type_code

            cat_fw = _feather_weight(th_eff, tw_eff, margin=100)
            wins = cat_fw > best_cat_weight[row_off:r_end, col_off:c_end]
            labels_full[row_off:r_end, col_off:c_end] = np.where(
                wins, labels_tile, labels_full[row_off:r_end, col_off:c_end])
            seg_type_full[row_off:r_end, col_off:c_end] = np.where(
                wins, type_tile, seg_type_full[row_off:r_end, col_off:c_end])

            if tr.get("ndsm") is not None:
                tile_sh = tr["ndsm"][:th_eff, :tw_eff].astype(np.float32)
                seg_height_full[row_off:r_end, col_off:c_end] = np.where(
                    wins, tile_sh, seg_height_full[row_off:r_end, col_off:c_end])

            best_cat_weight[row_off:r_end, col_off:c_end] = np.maximum(
                best_cat_weight[row_off:r_end, col_off:c_end], cat_fw)

    # ------------------------------------------------------------------
    # Normalise weighted sums → final blended rasters
    # ------------------------------------------------------------------
    has_data = weight_sum > 0
    dtm_full = np.where(has_data, dtm_sum / weight_sum, np.nan).astype(np.float32)
    dsm_full = np.where(has_data, dsm_sum / weight_sum, np.nan).astype(np.float32)
    ndsm_full = np.where(has_data, ndsm_sum / weight_sum, np.nan).astype(np.float32)
    del dtm_sum, dsm_sum, ndsm_sum, weight_sum, best_cat_weight

    # ------------------------------------------------------------------
    # Merge boundary segments across tiles (before writing rasters)
    # ------------------------------------------------------------------
    try:
        _merge_boundary_segments(
            labels_full, seg_type_full, all_objects,
            tile_seg_results, full_left, full_top, res,
            ndsm_full=ndsm_full, rf_only=rf_only)
    except Exception as e:
        log.warning("Full GPKG boundary merge failed: %s", e)

    # ------------------------------------------------------------------
    # Write stitched raster layers
    # ------------------------------------------------------------------
    _write_table('DTM', [dtm_full], full_h, full_w, full_tf,
                 descs=['Digital Terrain Model (m)'])
    del dtm_full

    _write_table('DSM', [dsm_full], full_h, full_w, full_tf,
                 descs=['Digital Surface Model (m)'])
    del dsm_full

    _write_table('nDSM', [ndsm_full], full_h, full_w, full_tf,
                 descs=['Normalised DSM (m)'])
    del ndsm_full

    _write_table('segment_type', [seg_type_full], full_h, full_w, full_tf,
                 dtype='uint8', descs=['Object type code'])
    del seg_type_full

    _write_table('segment_height', [seg_height_full], full_h, full_w, full_tf,
                 descs=['Segment height (m)'])
    del seg_height_full

    # ------------------------------------------------------------------
    # Multi-date DTM/DSM: one layer per additional date
    # ------------------------------------------------------------------
    import tile_index as _ti
    import raster_io as _rio
    other_dates = sorted(d for d in _ti.DATASETS if d != _ti.DEFAULT_DATASET)
    for date_key in other_dates:
        year = _ti.dataset_to_year(date_key)
        dtm_sum2 = np.zeros((full_h, full_w), dtype=np.float64)
        dsm_sum2 = np.zeros((full_h, full_w), dtype=np.float64)
        wsum2 = np.zeros((full_h, full_w), dtype=np.float64)
        any_data = False
        for ti_idx, tr in enumerate(tile_seg_results):
            th, tw = tr["shape"]
            tile_left = tr["bounds_3035"][0]
            tile_top = tr["bounds_3035"][3]
            col_off = int(round((tile_left - full_left) / res))
            row_off = int(round((full_top - tile_top) / res))
            r_end = min(row_off + th, full_h)
            c_end = min(col_off + tw, full_w)
            th_eff = r_end - row_off
            tw_eff = c_end - col_off
            if th_eff <= 0 or tw_eff <= 0:
                continue
            try:
                d2 = _rio.read_dtm_dsm(box(*tr["bounds_3035"]), date_key)
                td = d2["dtm"][:th_eff, :tw_eff].astype(np.float32)
                ts2 = d2["dsm"][:th_eff, :tw_eff].astype(np.float32)
                del d2
                fw = _feather_weight(th_eff, tw_eff, margin=100)
                v = ~np.isnan(td)
                w = np.where(v, fw, 0.0).astype(np.float64)
                dtm_sum2[row_off:r_end, col_off:c_end] += np.where(v, td, 0.0).astype(np.float64) * w
                dsm_sum2[row_off:r_end, col_off:c_end] += np.where(v, ts2, 0.0).astype(np.float64) * w
                wsum2[row_off:r_end, col_off:c_end] += w
                any_data = True
            except Exception:
                pass
        if any_data:
            hd2 = wsum2 > 0
            dtm_extra = np.where(hd2, dtm_sum2 / wsum2, np.nan).astype(np.float32)
            dsm_extra = np.where(hd2, dsm_sum2 / wsum2, np.nan).astype(np.float32)
            _write_table(f'DTM_{year}', [dtm_extra], full_h, full_w, full_tf,
                         descs=[f'DTM {year} (m)'])
            _write_table(f'DSM_{year}', [dsm_extra], full_h, full_w, full_tf,
                         descs=[f'DSM {year} (m)'])
            log.info("  FULL_GPKG: DTM/DSM %d written", year)
            del dtm_extra, dsm_extra
        del dtm_sum2, dsm_sum2, wsum2

    # ------------------------------------------------------------------
    # Ortho + CIR: one layer per available ortho year
    # ------------------------------------------------------------------
    import ortho_io as _oio
    import concurrent.futures as _cf
    # Ortho year slots: map ortho_year → read_ortho_for_als year param
    _ORTHO_YEARS = {2024: 2024, 2023: 2023, 2020: 2020}

    def _stitch_ortho_for_year(o_year):
        """Read ortho per tile for a given year, feather-blend into full-KG."""
        ortho_sum = np.zeros((4, full_h, full_w), dtype=np.float64)
        ortho_w = np.zeros((full_h, full_w), dtype=np.float64)
        got_rgb = False
        got_nir = False
        for ti_idx, tr in enumerate(tile_seg_results):
            th, tw = tr["shape"]
            tile_left = tr["bounds_3035"][0]
            tile_top = tr["bounds_3035"][3]
            col_off = int(round((tile_left - full_left) / res))
            row_off = int(round((full_top - tile_top) / res))
            r_end = min(row_off + th, full_h)
            c_end = min(col_off + tw, full_w)
            th_eff = r_end - row_off
            tw_eff = c_end - col_off
            if th_eff <= 0 or tw_eff <= 0:
                continue
            try:
                tdata = _read_dtm_for_tile(tr)
                with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                    fut = _ex.submit(_oio.read_ortho_for_als, tdata, year=o_year)
                    rgb_t, nir_t = fut.result(timeout=180)
                del tdata
                if rgb_t is not None:
                    got_rgb = True
                    fw = _feather_weight(th_eff, tw_eff, margin=100).astype(np.float64)
                    for b in range(3):
                        ortho_sum[b, row_off:r_end, col_off:c_end] += \
                            rgb_t[b][:th_eff, :tw_eff].astype(np.float64) * fw
                    if nir_t is not None:
                        got_nir = True
                        ortho_sum[3, row_off:r_end, col_off:c_end] += \
                            nir_t[:th_eff, :tw_eff].astype(np.float64) * fw
                    ortho_w[row_off:r_end, col_off:c_end] += fw
            except Exception as e:
                log.warning("GPKG tile %d ortho yr=%d failed: %s", ti_idx + 1, o_year, e)
        # Normalise
        has = ortho_w > 0
        ortho_arr = np.zeros((4, full_h, full_w), dtype=np.uint8)
        for b in range(4):
            ortho_arr[b] = np.where(has,
                np.clip(ortho_sum[b] / np.where(has, ortho_w, 1.0), 0, 255),
                0).astype(np.uint8)
        return ortho_arr, got_rgb, got_nir

    for o_year in sorted(_ORTHO_YEARS.keys(), reverse=True):  # newest first
        try:
            ortho_full, got_rgb, got_nir = _stitch_ortho_for_year(_ORTHO_YEARS[o_year])
            if got_rgb:
                bands = [ortho_full[0], ortho_full[1], ortho_full[2]]
                descs = ['Red', 'Green', 'Blue']
                if got_nir:
                    bands.append(ortho_full[3])
                    descs.append('NIR')
                _write_table(f'Ortho_{o_year}', bands, full_h, full_w, full_tf,
                             dtype='uint8', descs=descs)
                if got_nir:
                    _write_table(f'CIR_{o_year}',
                                 [ortho_full[3], ortho_full[0], ortho_full[1]],
                                 full_h, full_w, full_tf, dtype='uint8',
                                 descs=['NIR\u2192Red', 'Red\u2192Green', 'Green\u2192Blue'])
                log.info("  FULL_GPKG: Ortho+CIR %d written (NIR=%s)", o_year, got_nir)
            else:
                log.info("  FULL_GPKG: no ortho data for year %d", o_year)
            del ortho_full
        except Exception as e:
            log.warning("FULL_GPKG: ortho year %d failed: %s", o_year, e)

    # ------------------------------------------------------------------
    # Copernicus NDVI (10m, resampled to 1m grid)
    # ------------------------------------------------------------------
    try:
        cop_cache = _get_cop_cache()
        ndvi_sum = np.zeros((full_h, full_w), dtype=np.float64)
        ndvi_wsum = np.zeros((full_h, full_w), dtype=np.float64)
        any_ndvi = False
        for ti_idx, tr in enumerate(tile_seg_results):
            bbox_wgs = tr["bbox_wgs"]
            tw_t, ts_t, te_t, tn_t = bbox_wgs
            bbox_dict = {"west": tw_t, "south": ts_t, "east": te_t, "north": tn_t}
            nd = cop_cache.get_ndvi(bbox_dict, year=obs_year)
            if nd and nd.get("ndvi") is not None:
                from rasterio.warp import reproject, Resampling
                src_arr = nd["ndvi"].astype(np.float32)
                src_tf = nd["transform"]
                src_crs = nd.get("crs", "EPSG:4326")
                th_t, tw_t2 = tr["shape"]
                tile_left = tr["bounds_3035"][0]
                tile_top = tr["bounds_3035"][3]
                col_off = int(round((tile_left - full_left) / res))
                row_off = int(round((full_top - tile_top) / res))
                r_end = min(row_off + th_t, full_h)
                c_end = min(col_off + tw_t2, full_w)
                th_eff = r_end - row_off
                tw_eff = c_end - col_off
                if th_eff <= 0 or tw_eff <= 0:
                    continue
                dst_tf = rasterio.transform.from_bounds(
                    tile_left, tr["bounds_3035"][1],
                    tr["bounds_3035"][2], tile_top, tw_t2, th_t)
                dst_arr = np.full((th_t, tw_t2), np.nan, dtype=np.float32)
                reproject(
                    src_arr, dst_arr,
                    src_transform=src_tf, src_crs=src_crs,
                    dst_transform=dst_tf, dst_crs='EPSG:3035',
                    resampling=Resampling.bilinear,
                )
                tile_data = dst_arr[:th_eff, :tw_eff]
                fw = _feather_weight(th_eff, tw_eff, margin=100)
                v = ~np.isnan(tile_data)
                w = np.where(v, fw, 0.0).astype(np.float64)
                ndvi_sum[row_off:r_end, col_off:c_end] += np.where(v, tile_data, 0.0).astype(np.float64) * w
                ndvi_wsum[row_off:r_end, col_off:c_end] += w
                any_ndvi = True
        if any_ndvi:
            hd = ndvi_wsum > 0
            ndvi_full = np.where(hd, ndvi_sum / ndvi_wsum, np.nan).astype(np.float32)
            _write_table('NDVI', [ndvi_full], full_h, full_w, full_tf,
                         descs=['Sentinel-2 NDVI composite'])
            log.info("  FULL_GPKG: NDVI written")
            del ndvi_full
        del ndvi_sum, ndvi_wsum
    except Exception as e:
        log.warning("FULL_GPKG: NDVI layer failed: %s", e)

    # ------------------------------------------------------------------
    # ESA WorldCover (10m, resampled to 1m grid)
    # ------------------------------------------------------------------
    try:
        cop_cache = _get_cop_cache()
        wc_full = np.zeros((full_h, full_w), dtype=np.uint8)
        any_wc = False
        for ti_idx, tr in enumerate(tile_seg_results):
            bbox_wgs = tr["bbox_wgs"]
            tw_t, ts_t, te_t, tn_t = bbox_wgs
            bbox_dict = {"west": tw_t, "south": ts_t, "east": te_t, "north": tn_t}
            lc = cop_cache.get_landcover(bbox_dict)
            if lc and lc.get("map") is not None:
                from rasterio.warp import reproject, Resampling
                src_arr = lc["map"].astype(np.uint8)
                src_tf = lc["transform"]
                src_crs = lc.get("crs", "EPSG:4326")
                th_t, tw_t2 = tr["shape"]
                tile_left = tr["bounds_3035"][0]
                tile_top = tr["bounds_3035"][3]
                col_off = int(round((tile_left - full_left) / res))
                row_off = int(round((full_top - tile_top) / res))
                r_end = min(row_off + th_t, full_h)
                c_end = min(col_off + tw_t2, full_w)
                th_eff = r_end - row_off
                tw_eff = c_end - col_off
                if th_eff <= 0 or tw_eff <= 0:
                    continue
                dst_tf = rasterio.transform.from_bounds(
                    tile_left, tr["bounds_3035"][1],
                    tr["bounds_3035"][2], tile_top, tw_t2, th_t)
                dst_arr = np.zeros((th_t, tw_t2), dtype=np.uint8)
                reproject(
                    src_arr, dst_arr,
                    src_transform=src_tf, src_crs=src_crs,
                    dst_transform=dst_tf, dst_crs='EPSG:3035',
                    resampling=Resampling.nearest,
                )
                wc_full[row_off:r_end, col_off:c_end] = dst_arr[:th_eff, :tw_eff]
                any_wc = True
        if any_wc:
            _write_table('WorldCover', [wc_full], full_h, full_w, full_tf,
                         dtype='uint8', descs=['ESA WorldCover 2021 class'])
            log.info("  FULL_GPKG: WorldCover written")
        del wc_full
    except Exception as e:
        log.warning("FULL_GPKG: WorldCover layer failed: %s", e)

    # ------------------------------------------------------------------
    # Sentinel-1 SAR (VV + VH, 10m, resampled to 1m grid)
    # ------------------------------------------------------------------
    try:
        cop_cache = _get_cop_cache()
        vv_full = np.full((full_h, full_w), np.nan, dtype=np.float32)
        vh_full = np.full((full_h, full_w), np.nan, dtype=np.float32)
        any_sar = False
        for ti_idx, tr in enumerate(tile_seg_results):
            bbox_wgs = tr["bbox_wgs"]
            tw_t, ts_t, te_t, tn_t = bbox_wgs
            bbox_dict = {"west": tw_t, "south": ts_t, "east": te_t, "north": tn_t}
            sar = cop_cache.get_sar(bbox_dict, year=obs_year)
            if sar:
                from rasterio.warp import reproject, Resampling
                th_t, tw_t2 = tr["shape"]
                tile_left = tr["bounds_3035"][0]
                tile_top = tr["bounds_3035"][3]
                col_off = int(round((tile_left - full_left) / res))
                row_off = int(round((full_top - tile_top) / res))
                r_end = min(row_off + th_t, full_h)
                c_end = min(col_off + tw_t2, full_w)
                th_eff = r_end - row_off
                tw_eff = c_end - col_off
                if th_eff <= 0 or tw_eff <= 0:
                    continue
                dst_tf = rasterio.transform.from_bounds(
                    tile_left, tr["bounds_3035"][1],
                    tr["bounds_3035"][2], tile_top, tw_t2, th_t)
                src_tf = sar["transform"]
                src_crs = sar.get("crs", "EPSG:4326")
                for band_name, full_arr in [("vv", vv_full), ("vh", vh_full)]:
                    src_arr = sar.get(band_name)
                    if src_arr is not None:
                        src_arr = src_arr.astype(np.float32)
                        dst_arr = np.full((th_t, tw_t2), np.nan, dtype=np.float32)
                        reproject(
                            src_arr, dst_arr,
                            src_transform=src_tf, src_crs=src_crs,
                            dst_transform=dst_tf, dst_crs='EPSG:3035',
                            resampling=Resampling.bilinear,
                        )
                        full_arr[row_off:r_end, col_off:c_end] = dst_arr[:th_eff, :tw_eff]
                        any_sar = True
        if any_sar:
            _write_table('SAR', [vv_full, vh_full], full_h, full_w, full_tf,
                         descs=['Sentinel-1 VV (dB)', 'Sentinel-1 VH (dB)'])
            log.info("  FULL_GPKG: SAR (VV+VH) written")
        del vv_full, vh_full
    except Exception as e:
        log.warning("FULL_GPKG: SAR layer failed: %s", e)

    # ------------------------------------------------------------------
    # Hansen Global Forest Change (30m, resampled to 1m grid)
    # ------------------------------------------------------------------
    try:
        hc = _get_hansen_cache()
        tc_full = np.zeros((full_h, full_w), dtype=np.uint8)
        ly_full = np.zeros((full_h, full_w), dtype=np.uint8)
        any_hansen = False
        for ti_idx, tr in enumerate(tile_seg_results):
            bbox_wgs = tr["bbox_wgs"]
            tw_t, ts_t, te_t, tn_t = bbox_wgs
            th_t, tw_t2 = tr["shape"]
            tile_left = tr["bounds_3035"][0]
            tile_top = tr["bounds_3035"][3]
            col_off = int(round((tile_left - full_left) / res))
            row_off = int(round((full_top - tile_top) / res))
            r_end = min(row_off + th_t, full_h)
            c_end = min(col_off + tw_t2, full_w)
            th_eff = r_end - row_off
            tw_eff = c_end - col_off
            if th_eff <= 0 or tw_eff <= 0:
                continue
            dst_tf = rasterio.transform.from_bounds(
                tile_left, tr["bounds_3035"][1],
                tr["bounds_3035"][2], tile_top, tw_t2, th_t)
            hd = hc.get_forest_prior(
                (tw_t, ts_t, te_t, tn_t), dst_tf, (th_t, tw_t2))
            if hd:
                tc = hd.get("treecover2000")
                ly = hd.get("loss_year")
                if tc is not None:
                    tc_full[row_off:r_end, col_off:c_end] = tc[:th_eff, :tw_eff]
                    any_hansen = True
                if ly is not None:
                    ly_full[row_off:r_end, col_off:c_end] = ly[:th_eff, :tw_eff]
        if any_hansen:
            _write_table('Hansen_treecover', [tc_full], full_h, full_w, full_tf,
                         dtype='uint8', descs=['Tree cover 2000 (%)'])
            _write_table('Hansen_lossyear', [ly_full], full_h, full_w, full_tf,
                         dtype='uint8', descs=['Forest loss year (0=none, 1=2001, ..., 23=2023)'])
            log.info("  FULL_GPKG: Hansen (treecover + lossyear) written")
        del tc_full, ly_full
    except Exception as e:
        log.warning("FULL_GPKG: Hansen layer failed: %s", e)

    # ------------------------------------------------------------------
    # Segment vectors — one unified layer from stitched labels
    # ------------------------------------------------------------------
    if all_objects and tile_seg_results:
        try:
            mask_full = labels_full > 0
            _write_segment_vectors(
                out_path, labels_full, all_objects,
                mask_full, full_tf, layer_name='segments', obs_year=obs_year)
        except Exception as e:
            log.warning("Full GPKG segment vectors failed: %s", e)
    del labels_full

    # Segment centroid points — one layer with all objects
    if all_objects:
        try:
            _write_segment_points(out_path, all_objects,
                                  layer_name='segment_points', obs_year=obs_year)
        except Exception as e:
            log.warning("Full GPKG segment points failed: %s", e)

    # Write QGIS styles for all layers (raster + vector)
    try:
        _write_gpkg_all_styles(
            out_path,
            has_segments=bool(all_objects),
            has_points=bool(all_objects),
        )
    except Exception as e:
        log.warning("Full GPKG styles failed: %s", e)

    # Fix CRS for uint8 raster layers (GDAL registers as 'tiles' not '2d-gridded-coverage')
    _fix_gpkg_raster_crs(out_path)

    fsize = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    log.info("  FULL_GPKG: %.1f MB, %d tables, %d tiles",
             fsize / 1e6, table_count, len(tile_seg_results))
    return out_path


def build_light_gpkg_tiled(kg_code, tile_seg_results, all_objects,
                           cadastre_data, new_buildings, infrastructure,
                           obs_year=0, rf_only=False):
    """Light GPKG: stitched segment rasters + enriched cadastre vectors."""
    import rasterio, fiona
    import rasterio.transform
    from fiona.crs import from_epsg

    out_path = str(GPKG_DIR / f"{kg_code}_light.gpkg")
    if os.path.exists(out_path): os.unlink(out_path)

    table_count = 0
    _GPKG_NODATA_L = -9999.0

    def _write_raster(name, arrays, h, w, tf, dtype='uint8', descs=None):
        nonlocal table_count
        opts = dict(driver='GPKG', width=w, height=h, count=len(arrays),
                    dtype=dtype, crs='EPSG:3035', transform=tf,
                    RASTER_TABLE=name, RASTER_IDENTIFIER=name)
        if dtype == 'float32': opts['nodata'] = _GPKG_NODATA_L
        if table_count > 0: opts['APPEND_SUBDATASET'] = 'YES'
        with rasterio.open(out_path, 'w', **opts) as dst:
            for i, arr in enumerate(arrays, 1):
                band = arr[:h, :w]
                if dtype == 'float32':
                    band = np.where(np.isnan(band), _GPKG_NODATA_L, band)
                dst.write(band, i)
                if descs and i <= len(descs): dst.set_band_description(i, descs[i-1])
        table_count += 1

    obj_map = {o.obj_id: o for o in all_objects}

    # ------------------------------------------------------------------
    # Stitch segment rasters across all tiles into full-KG extent
    # ------------------------------------------------------------------
    if tile_seg_results:
        all_lefts = [tr["bounds_3035"][0] for tr in tile_seg_results]
        all_bottoms = [tr["bounds_3035"][1] for tr in tile_seg_results]
        all_rights = [tr["bounds_3035"][2] for tr in tile_seg_results]
        all_tops = [tr["bounds_3035"][3] for tr in tile_seg_results]
        full_left = min(all_lefts)
        full_bottom = min(all_bottoms)
        full_right = max(all_rights)
        full_top = max(all_tops)

        res = 1.0  # 1m resolution
        full_w = int(round((full_right - full_left) / res))
        full_h = int(round((full_top - full_bottom) / res))
        full_tf = rasterio.transform.from_bounds(
            full_left, full_bottom, full_right, full_top, full_w, full_h)

        # Allocate full-KG rasters for segments
        seg_type_full = np.zeros((full_h, full_w), dtype=np.uint8)
        seg_height_full = np.full((full_h, full_w), np.nan, dtype=np.float32)
        labels_full = np.zeros((full_h, full_w), dtype=np.int32)
        best_cat_weight = np.zeros((full_h, full_w), dtype=np.float32)

        def _feather_weight_light(rows, cols, margin=100):
            wy = np.ones(rows, dtype=np.float32)
            wx = np.ones(cols, dtype=np.float32)
            m = min(margin, rows // 2, cols // 2)
            if m > 0:
                ramp = np.linspace(0.0, 1.0, m, endpoint=False, dtype=np.float32)
                wy[:m] = np.minimum(wy[:m], ramp)
                wy[-m:] = np.minimum(wy[-m:], ramp[::-1])
                wx[:m] = np.minimum(wx[:m], ramp)
                wx[-m:] = np.minimum(wx[-m:], ramp[::-1])
            return wy[:, None] * wx[None, :]

        # Paint each tile (highest-weight-wins for categorical data)
        for ti_idx, tr in enumerate(tile_seg_results):
            th, tw = tr["shape"]
            tile_left, tile_bottom, tile_right, tile_top = tr["bounds_3035"]

            col_off = int(round((tile_left - full_left) / res))
            row_off = int(round((full_top - tile_top) / res))

            r_end = min(row_off + th, full_h)
            c_end = min(col_off + tw, full_w)
            th_eff = r_end - row_off
            tw_eff = c_end - col_off
            if th_eff <= 0 or tw_eff <= 0:
                continue

            if tr.get("labels") is not None:
                labels_tile = tr["labels"][:th_eff, :tw_eff].astype(np.int32)

                type_tile = np.zeros((th_eff, tw_eff), dtype=np.uint8)
                for uid in np.unique(labels_tile):
                    if uid == 0: continue
                    obj = obj_map.get(int(uid))
                    if obj: type_tile[labels_tile == uid] = obj.type_code

                cat_fw = _feather_weight_light(th_eff, tw_eff, margin=100)
                wins = cat_fw > best_cat_weight[row_off:r_end, col_off:c_end]
                labels_full[row_off:r_end, col_off:c_end] = np.where(
                    wins, labels_tile, labels_full[row_off:r_end, col_off:c_end])
                seg_type_full[row_off:r_end, col_off:c_end] = np.where(
                    wins, type_tile, seg_type_full[row_off:r_end, col_off:c_end])

                if tr.get("ndsm") is not None:
                    tile_sh = tr["ndsm"][:th_eff, :tw_eff].astype(np.float32)
                    seg_height_full[row_off:r_end, col_off:c_end] = np.where(
                        wins, tile_sh, seg_height_full[row_off:r_end, col_off:c_end])

                best_cat_weight[row_off:r_end, col_off:c_end] = np.maximum(
                    best_cat_weight[row_off:r_end, col_off:c_end], cat_fw)

        del best_cat_weight

        # Merge boundary segments across tiles
        try:
            _merge_boundary_segments(
                labels_full, seg_type_full, all_objects,
                tile_seg_results, full_left, full_top, res,
                rf_only=rf_only)
        except Exception as e:
            log.warning("Light GPKG boundary merge failed: %s", e)

        # Write stitched segment rasters
        _write_raster('segment_type', [seg_type_full], full_h, full_w, full_tf,
                      descs=['Object type code'])
        del seg_type_full

        _write_raster('segment_height', [seg_height_full], full_h, full_w, full_tf,
                      dtype='float32', descs=['Segment height (m)'])
        del seg_height_full

        # Segment vectors — one unified layer from stitched labels
        if all_objects:
            try:
                mask_full = labels_full > 0
                _write_segment_vectors(
                    out_path, labels_full, all_objects,
                    mask_full, full_tf, layer_name='segments', obs_year=obs_year)
            except Exception as e:
                log.warning("Light GPKG segment vectors failed: %s", e)
        del labels_full

    # --- Pre-load DTM data once per tile for GPKG parcel/building enrichment ---
    _gpkg_tile_data = {}  # ti_idx → tdata
    for ti_idx, tr in enumerate(tile_seg_results):
        try:
            _gpkg_tile_data[ti_idx] = _read_dtm_for_tile(tr)
        except Exception:
            pass

    def _gpkg_find_tile(e3035, n3035):
        for ti_idx, tr in enumerate(tile_seg_results):
            left, bottom, right, top = tr["bounds_3035"]
            if left <= e3035 <= right and bottom <= n3035 <= top:
                td = _gpkg_tile_data.get(ti_idx)
                if td is not None:
                    return tr, td
        return None, None

    # --- Parcels (enriched from tiles) ---
    parcels = cadastre_data["parcels"]
    if parcels:
        schema_p = {'geometry': 'MultiPolygon', 'properties': [
            ('parcel_id', 'str'), ('area_sqm', 'float'),
            ('centroid_dtm_m', 'float'), ('centroid_lon', 'float'), ('centroid_lat', 'float')]}
        with fiona.open(out_path, 'w', driver='GPKG', layer='parcels',
                        schema=schema_p, crs=from_epsg(4326)) as dst:
            for p in parcels:
                geom_wgs = p["geometry_wgs"]
                props = {"parcel_id": p["parcel_id"], "area_sqm": p["area_sqm"],
                         "centroid_dtm_m": None, "centroid_lon": None, "centroid_lat": None}
                try:
                    c3 = p["geometry"].centroid
                    c_wgs = transform_to_wgs(c3)
                    props["centroid_lon"] = round(c_wgs.x, 7)
                    props["centroid_lat"] = round(c_wgs.y, 7)
                    tr, tdata = _gpkg_find_tile(c3.x, c3.y)
                    if tr and tdata:
                        tf = tdata["transform"]
                        col = int((c3.x - tf.c) / tf.a)
                        row = int((tf.f - c3.y) / abs(tf.e))
                        dh, dw = tdata["dtm"].shape
                        if 0 <= row < dh and 0 <= col < dw:
                            val = float(tdata["dtm"][row, col])
                            if np.isfinite(val): props["centroid_dtm_m"] = round(val, 2)
                except Exception:
                    pass
                dst.write({'geometry': _to_multi(mapping(geom_wgs)), 'properties': props})

    # --- Buildings (enriched from tiles) ---
    bldgs = cadastre_data["building_footprints"]
    if bldgs:
        from rasterio.features import rasterize as rio_rasterize
        schema_b = {'geometry': 'MultiPolygon', 'properties': [
            ('max_height_m', 'float'), ('mean_height_m', 'float'),
            ('dsm_std', 'float'), ('roof_type_hint', 'str'),
            ('stories_est', 'int'), ('footprint_area_sqm', 'float'),
            ('centroid_lon', 'float'), ('centroid_lat', 'float')]}
        with fiona.open(out_path, 'w', driver='GPKG', layer='buildings',
                        schema=schema_b, crs=from_epsg(4326)) as dst:
            for b in bldgs:
                geom_3035 = b["geometry"]
                geom_wgs = b["geometry_wgs"]
                props = {"max_height_m": None, "mean_height_m": None, "dsm_std": None,
                         "roof_type_hint": "", "stories_est": None,
                         "footprint_area_sqm": round(float(geom_3035.area), 1),
                         "centroid_lon": None, "centroid_lat": None}
                try:
                    c3 = geom_3035.centroid
                    c_wgs = transform_to_wgs(c3)
                    props["centroid_lon"] = round(c_wgs.x, 7)
                    props["centroid_lat"] = round(c_wgs.y, 7)
                    tr, tdata = _gpkg_find_tile(c3.x, c3.y)
                    if tr and tdata:
                        dh, dw = tdata["shape"]
                        fp = rio_rasterize(
                            [(geom_3035, 1)], out_shape=(dh, dw),
                            transform=tdata["transform"],
                            fill=0, dtype=np.uint8, all_touched=True).astype(bool)
                        oh = tdata["ndsm"][fp]; oh = oh[np.isfinite(oh)]
                        dv = tdata["dsm"][fp]; dv = dv[np.isfinite(dv)]
                        if len(oh) > 0:
                            mh = float(np.nanmax(oh))
                            props["max_height_m"] = round(mh, 2)
                            props["mean_height_m"] = round(float(np.nanmean(oh)), 2)
                            props["dsm_std"] = round(float(np.std(dv)), 2) if len(dv) > 0 else 0.0
                            props["roof_type_hint"] = "flat" if props["dsm_std"] < 1.5 else "pitched"
                            props["stories_est"] = max(1, round(mh / 3.0))
                except Exception:
                    pass
                dst.write({'geometry': _to_multi(mapping(geom_wgs)), 'properties': props})
    del _gpkg_tile_data  # free memory

    # --- New buildings ---
    if new_buildings:
        schema_nb = {'geometry': 'MultiPolygon', 'properties': [
            ('type', 'str'), ('area_sqm', 'float'), ('max_height_m', 'float'),
            ('stories_est', 'int'), ('roof_type_hint', 'str'), ('confidence', 'float'),
            ('centroid_lon', 'float'), ('centroid_lat', 'float'), ('edge_clipped', 'bool')]}
        with fiona.open(out_path, 'w', driver='GPKG', layer='new_buildings',
                        schema=schema_nb, crs=from_epsg(4326)) as dst:
            for nb in new_buildings:
                dst.write({'geometry': _to_multi(nb["geometry_wgs"]), 'properties': {
                    'type': nb.get('type','roof'), 'area_sqm': nb.get('area_sqm',0),
                    'max_height_m': nb.get('max_height_m',0), 'stories_est': nb.get('stories_est',1),
                    'roof_type_hint': nb.get('roof_type_hint',''), 'confidence': nb.get('confidence',0),
                    'centroid_lon': nb.get('centroid_lon'), 'centroid_lat': nb.get('centroid_lat'),
                    'edge_clipped': nb.get('edge_clipped', False)}})

    # --- Infrastructure ---
    if infrastructure:
        schema_i = {'geometry': 'MultiPolygon', 'properties': [
            ('type', 'str'), ('area_sqm', 'float'), ('volume_m3', 'float'),
            ('max_height_m', 'float'), ('est_parking_spots', 'int'),
            ('confidence', 'float'), ('centroid_lon', 'float'), ('centroid_lat', 'float'),
            ('edge_clipped', 'bool')]}
        with fiona.open(out_path, 'w', driver='GPKG', layer='infrastructure',
                        schema=schema_i, crs=from_epsg(4326)) as dst:
            for inf in infrastructure:
                dst.write({'geometry': _to_multi(inf["geometry_wgs"]), 'properties': {
                    'type': inf.get('type',''), 'area_sqm': inf.get('area_sqm',0),
                    'volume_m3': inf.get('volume_m3'), 'max_height_m': inf.get('max_height_m'),
                    'est_parking_spots': inf.get('est_parking_spots'),
                    'confidence': inf.get('confidence',0),
                    'centroid_lon': inf.get('centroid_lon'), 'centroid_lat': inf.get('centroid_lat'),
                    'edge_clipped': inf.get('edge_clipped', False)}})

    # Segment centroid points — one layer with all objects
    if all_objects:
        try:
            _write_segment_points(out_path, all_objects,
                                  layer_name='segment_points', obs_year=obs_year)
        except Exception as e:
            log.warning("Light GPKG segment points failed: %s", e)

    # Write QGIS styles for all layers
    try:
        _write_gpkg_all_styles(
            out_path,
            has_segments=bool(all_objects),
            has_points=bool(all_objects),
            has_parcels=bool(cadastre_data.get("parcels")),
            has_buildings=bool(cadastre_data.get("building_footprints")),
            has_new_buildings=bool(new_buildings),
            has_infrastructure=bool(infrastructure),
        )
    except Exception as e:
        log.warning("Light GPKG styles failed: %s", e)

    # Fix CRS for uint8 raster layers (GDAL registers as 'tiles' not '2d-gridded-coverage')
    _fix_gpkg_raster_crs(out_path)

    return out_path


# === SECTION: Data quality scoring ===

# Weight each data source by its contribution to segmentation quality.
# DTM+DSM are essential (segmentation fails without them).
# Ortho and Copernicus NDVI provide spectral features for RF classification.
# SAR, harmonics, worldcover, and hansen are supplementary.
_QUALITY_WEIGHTS = {
    "dtm":              0.15,
    "dsm":              0.15,
    "ortho":            0.15,
    "copernicus_ndvi":  0.12,
    "sar":              0.08,
    "worldcover":       0.06,
    "harmonics":        0.06,
    "hansen":           0.05,
    "multi_date_dtm":   0.05,
    "terrain":          0.03,
    "cadastre_buildings": 0.05,
    "infrastructure":   0.02,
    "segmentation":     0.03,
}


def compute_data_quality(tile_data_availability: list[dict]) -> dict:
    """Compute data quality score and summary from per-tile availability.

    Returns a dict with:
      - quality_score: 0.0–1.0 (pixel-weighted across tiles)
      - quality_grade: A/B/C/D/F
      - layers_summary: per-layer fraction of tiles that had it
      - tiles: per-tile availability list
      - missing_layers: layers that were absent from ALL tiles
      - available_layers: layers present in ALL tiles
      - partial_layers: layers present in some but not all tiles
    """
    if not tile_data_availability:
        return {
            "quality_score": 0.0, "quality_grade": "F",
            "layers_summary": {}, "tiles": [],
            "missing_layers": list(_QUALITY_WEIGHTS.keys()),
            "available_layers": [], "partial_layers": [],
        }

    layer_keys = list(_QUALITY_WEIGHTS.keys())
    total_weight = sum(_QUALITY_WEIGHTS.values())

    # Pixel-weighted quality: tiles with more valid pixels count more
    total_pixels = sum(t.get("valid_pixels", 0) for t in tile_data_availability)
    if total_pixels == 0:
        total_pixels = len(tile_data_availability)  # fallback: equal weight

    weighted_score = 0.0
    layer_counts = {k: 0 for k in layer_keys}
    n_active_tiles = 0  # tiles that actually had data (valid_pixels > 0 or segmentation)

    for t in tile_data_availability:
        px = t.get("valid_pixels", 0)
        if px == 0 and not t.get("segmentation"):
            continue  # skip empty/skipped tiles for scoring
        n_active_tiles += 1
        tile_weight = px / total_pixels if total_pixels > 0 else 1.0 / len(tile_data_availability)
        tile_score = 0.0
        for k in layer_keys:
            if t.get(k):
                tile_score += _QUALITY_WEIGHTS[k]
                layer_counts[k] += 1
        weighted_score += (tile_score / total_weight) * tile_weight

    n_tiles_denom = max(n_active_tiles, 1)
    layers_summary = {k: round(v / n_tiles_denom, 3) for k, v in layer_counts.items()}

    # Classify layers
    missing = [k for k, v in layers_summary.items() if v == 0]
    available = [k for k, v in layers_summary.items() if v >= 1.0]
    partial = [k for k, v in layers_summary.items() if 0 < v < 1.0]

    score = round(min(1.0, weighted_score), 3)
    if score >= 0.9:
        grade = "A"
    elif score >= 0.75:
        grade = "B"
    elif score >= 0.55:
        grade = "C"
    elif score >= 0.35:
        grade = "D"
    else:
        grade = "F"

    return {
        "quality_score": score,
        "quality_grade": grade,
        "quality_weights": {k: round(v / total_weight, 3) for k, v in _QUALITY_WEIGHTS.items()},
        "layers_summary": layers_summary,
        "tiles": tile_data_availability,
        "n_active_tiles": n_active_tiles,
        "n_total_tiles": len(tile_data_availability),
        "missing_layers": missing,
        "available_layers": available,
        "partial_layers": partial,
    }


def build_json_summary_tiled(kg_code, kg_info, tile_seg_results, all_objects,
                             cadastre_data, terrain_stats, spectral_info,
                             copernicus_info, hansen_info, new_buildings,
                             infrastructure, obs_year, n_tiles=1, tile_km=1.5,
                             total_seg_pixels=0, tile_data_availability=None):
    """Build JSON summary from tiled segmentation results."""
    import tile_index as ti
    objects = all_objects
    summary = {
        "version": VERSION,
        "kg_code": kg_code,
        "kg_name": kg_info.get("kg_name", ""),
        "state": kg_info.get("state_name", ""),
        "gemeinde": kg_info.get("gemeinde_name", ""),
        "district": kg_info.get("district_name", ""),
        "bbox": kg_info.get("bbox", {}),
        "total_area_sqm": total_seg_pixels,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observation_period": {
            "primary_year": obs_year, "start": f"{obs_year}-01-01", "end": f"{obs_year}-12-31",
            "lidar_dataset": ti.DEFAULT_DATASET,
            "all_lidar_datasets": sorted(ti.DATASETS.keys()),
            "sentinel2": f"Sentinel-2 L2A {obs_year}",
            "hansen": "Hansen GFC-2024-v1.12 (2000-2024)",
            "cadastre": "BEV INSPIRE cadastre (current)",
        },
    }
    # --- Area summary ---
    type_counts = Counter()
    type_heights = defaultdict(list)
    for obj in objects:
        type_counts[obj.obj_type] += int(obj.area_sqm)
        type_heights[obj.obj_type].append(round(obj.height_max, 2))
    total_px = max(total_seg_pixels, 1)
    summary["area_summary"] = {
        t: {"pixels": px, "area_sqm": px, "fraction": round(px/total_px, 4),
            "n_objects": len(type_heights[t]),
            "observation_period": f"{obs_year}-01-01 to {obs_year}-12-31"}
        for t, px in type_counts.most_common()
    }
    summary["height_distribution"] = {
        t: {"min": round(min(hs),2), "max": round(max(hs),2),
            "mean": round(sum(hs)/len(hs),2),
            "p90": round(float(np.percentile(hs,90)),2), "count": len(hs)}
        for t, hs in type_heights.items() if hs
    }
    # --- Landscape ---
    landscape = {}
    if terrain_stats: landscape["terrain"] = terrain_stats
    if objects:
        total_perim = sum(getattr(o,'perimeter_m', getattr(o,'perimeter',0)) or 0 for o in objects)
        landscape["edge_density"] = round(total_perim / total_px, 4)
        landscape["n_segments"] = len(objects)
        landscape["mean_segment_area_sqm"] = round(total_px / len(objects), 1)
        tf_ = [c/total_px for c in type_counts.values()]
        landscape["shannon_diversity"] = round(float(-sum(f*np.log(f) for f in tf_ if f > 0)), 3)
        if type_counts: landscape["dominant_type"] = type_counts.most_common(1)[0][0]
        veg = {'tree','shrub','grass','hedge','crop','orchard','vineyard','garden'}
        vp = sum(v for k,v in type_counts.items() if k in veg)
        landscape["vegetated_fraction"] = round(vp/total_px, 4)
        landscape["is_vegetated"] = vp/total_px > 0.5
    summary["landscape"] = landscape
    # --- Top 10 objects/trees ---
    if objects:
        summary["top_10_objects"] = []
        for o in sorted(objects, key=lambda o: o.height_max, reverse=True)[:10]:
            c = None
            try:
                lon, lat = _tx_to_wgs.transform(o.centroid_e, o.centroid_n)
                c = {"lon": round(lon,7), "lat": round(lat,7)}
            except Exception: pass
            summary["top_10_objects"].append({
                "type": o.obj_type, "height_max_m": round(o.height_max,2),
                "height_mean_m": round(o.height_mean,2), "area_sqm": round(o.area_sqm,1),
                "coordinate": c, "confidence": round(o.confidence,3),
                "is_manmade": o.is_manmade, "observation_year": obs_year})
        trees = [o for o in objects if o.obj_type == 'tree']
        summary["top_10_trees"] = []
        for t in sorted(trees, key=lambda o: o.height_max, reverse=True)[:10]:
            c = None
            try:
                lon, lat = _tx_to_wgs.transform(t.centroid_e, t.centroid_n)
                c = {"lon": round(lon,7), "lat": round(lat,7)}
            except Exception: pass
            summary["top_10_trees"].append({
                "height_m": round(t.height_max,2), "canopy_height_m": round(t.height_mean,2),
                "height_p90_m": round(t.height_p90,2), "coordinate": c,
                "area_sqm": round(t.area_sqm,1), "ndvi_mean": round(t.ndvi_mean,4),
                "ndvi_fused": round(t.ndvi_fused,4), "height_change_m": round(t.height_change,3),
                "phenology_class": t.phenology_class or '', "observation_year": obs_year})
        if trees:
            summary["tree_stats"] = {
                "count": len(trees),
                "total_canopy_sqm": round(sum(t.area_sqm for t in trees), 1),
                "mean_height_m": round(sum(t.height_max for t in trees)/len(trees), 2),
                "est_stem_volume_m3": round(sum(0.3*t.area_sqm*t.height_max/3 for t in trees), 1)}
    # --- Terrain ---
    summary["terrain"] = {}
    if terrain_stats:
        ts = terrain_stats; elev = ts.get("elevation",{}); sl = ts.get("slope_deg",{})
        tri = ts.get("ruggedness_tri",{}); ad = ts.get("aspect_distribution_pct",{})
        summary["terrain"] = {
            "steepness_mean_deg": sl.get("mean"), "steepness_max_deg": sl.get("max"),
            "aspect_dominant": max(ad, key=ad.get) if ad else "",
            "aspect_distribution_pct": ad, "slope_classes_pct": ts.get("slope_classes_pct",{}),
            "roughness_mean": tri.get("mean"), "curvature_mean": None,
            "elevation_min_m": elev.get("min"), "elevation_max_m": elev.get("max"),
            "elevation_range_m": elev.get("range"), "elevation_mean_m": elev.get("mean"),
            "method": "DTM slope/aspect/TRI, BEV ALS 1m, tiled"}
    # --- NDVI ---
    ndvi_info = {}
    bv = spectral_info.get("_vals",[])
    if bv: ndvi_info["bev_nir_mean"] = round(float(np.mean(bv)),4); ndvi_info["bev_nir_std"] = round(float(np.std(bv)),4); ndvi_info["method_bev"] = f"BEV DOP RGBI 0.2m, {obs_year}"
    cv = copernicus_info.get("cop_ndvi_vals",[])
    if cv: ndvi_info["copernicus_mean"] = round(float(np.mean(cv)),4); ndvi_info["method_copernicus"] = f"Sentinel-2 L2A, {obs_year}"
    summary["ndvi"] = ndvi_info
    # --- SAR ---
    sar_info = {}
    for band in ['vv','vh']:
        vs = copernicus_info.get(f"{band}_vals",[])
        if vs: sar_info[f"{band}_mean_db"] = round(float(np.mean(vs)),2); sar_info[f"{band}_std_db"] = round(float(np.std(vs)),2)
    if sar_info: sar_info["method"] = f"Sentinel-1 IW GRD, VV+VH, summer {obs_year}"
    summary["sar"] = sar_info
    # --- Harmonics ---
    harm_info = {}
    for hk in ['h_mean','h_amplitude','h_phase','h_rmse']:
        vs = copernicus_info.get(f"{hk}_vals",[])
        if vs: harm_info[hk.replace('h_','')+'_mean'] = round(float(np.mean(vs)),4)
    if harm_info: harm_info["method"] = f"Harmonic fit to monthly Sentinel-2 NDVI, {obs_year}"
    summary["ndvi_harmonics"] = harm_info
    # --- Temporal change ---
    temp = {}
    if objects:
        dc = [o.dtm_change for o in objects if abs(o.dtm_change) > 0.01]
        hc = [o.height_change for o in objects if abs(o.height_change) > 0.01]
        if dc: temp["dtm_change_mean_m"] = round(float(np.mean(dc)),3); temp["dtm_change_max_abs_m"] = round(float(np.max(np.abs(dc))),3); temp["n_changed_segments"] = len(dc)
        if hc: temp["height_change_mean_m"] = round(float(np.mean(hc)),3)
        temp["net_volume_change_m3"] = round(sum(o.volume_change_m3 for o in objects),1)
        temp["total_disturbed_volume_m3"] = round(sum(o.volume_change_abs_m3 for o in objects),1)
        stab = [o.temporal_stability for o in objects]
        if stab: temp["mean_stability"] = round(float(np.mean(stab)),3)
        temp["datasets_compared"] = sorted(ti.DATASETS.keys())
        temp["method"] = "DTM/DSM differencing across BEV ALS dates"
    summary["temporal_change"] = temp
    # --- Phenology ---
    if objects:
        pc = Counter(o.phenology_class for o in objects if o.phenology_class)
        if pc: summary["phenology"] = {"distribution": dict(pc.most_common()), "method": "harmonic fit"}
    # --- Hansen ---
    hs = {}
    if hansen_info:
        loss_years = hansen_info.get("loss_years",[])
        if loss_years:
            per_year = {}
            for ly in loss_years:
                for yr in range(1,25):
                    n = int((ly==yr).sum())
                    if n > 0:
                        k = f"{2000+yr}"
                        per_year.setdefault(k, {"pixels":0,"area_sqm":0})
                        per_year[k]["pixels"] += n; per_year[k]["area_sqm"] += n*900
            if per_year: hs["loss_by_year"] = per_year; hs["total_loss_pixels"] = sum(v["pixels"] for v in per_year.values())
        tc = hansen_info.get("tc_vals",[])
        if tc: hs["mean_treecover2000_pct"] = round(float(np.mean(tc)),1)
        cf = hansen_info.get("cf_sum",[0])
        if cf[0] > 0: hs["current_forest_pixels"] = cf[0]
        if hs: hs["method"] = "Hansen GFC-2024-v1.12, 30m"
    summary["hansen"] = hs
    # --- New buildings + infrastructure ---
    summary["new_buildings"] = {
        "count": len(new_buildings),
        "features": [{"type": nb.get("type"), "area_sqm": nb.get("area_sqm"),
                      "max_height_m": nb.get("max_height_m"), "stories_est": nb.get("stories_est"),
                      "roof_type_hint": nb.get("roof_type_hint"),
                      "centroid_lon": nb.get("centroid_lon"), "centroid_lat": nb.get("centroid_lat"),
                      "confidence": nb.get("confidence"), "edge_clipped": nb.get("edge_clipped", False)}
                     for nb in new_buildings]}
    ibt = defaultdict(list)
    for inf in infrastructure: ibt[inf["type"]].append(inf)
    summary["infrastructure"] = {
        "total": len(infrastructure),
        "by_type": {t: {"count": len(items), "total_area_sqm": round(sum(i.get("area_sqm",0) for i in items),1),
                        "features": [{k:v for k,v in i.items() if k != "geometry_wgs"} for i in items]}
                   for t, items in ibt.items()}}
    # --- Pre-load DTM data once per tile for parcel/building enrichment ---
    obj_map = {o.obj_id: o for o in objects} if objects else {}
    _tile_data_cache = {}  # tile_idx → tdata dict
    for ti_idx, tr in enumerate(tile_seg_results):
        try:
            _tile_data_cache[ti_idx] = _read_dtm_for_tile(tr)
        except Exception as e:
            log.warning("JSON: failed to pre-load DTM for tile %d: %s", ti_idx, e)
    log.info("JSON: pre-loaded DTM for %d/%d tiles", len(_tile_data_cache), len(tile_seg_results))

    def _find_tile_with_data(e3035, n3035):
        """Find tile containing point AND return its cached tdata."""
        for ti_idx, tr in enumerate(tile_seg_results):
            left, bottom, right, top = tr["bounds_3035"]
            if left <= e3035 <= right and bottom <= n3035 <= top:
                tdata = _tile_data_cache.get(ti_idx)
                if tdata is not None:
                    return tr, tdata
        return None, None

    def _point_elevation_fallback(e3035, n3035):
        """Direct DTM point read as fallback when no tile covers the point."""
        try:
            import raster_io as _rio
            import tile_index as _ti
            from shapely.geometry import box as _box
            # Read a tiny window around the point (10×10m)
            tdata = _rio.read_dtm_dsm(_box(e3035 - 5, n3035 - 5, e3035 + 5, n3035 + 5),
                                      _ti.DEFAULT_DATASET)
            dtm = tdata["dtm"]; tf = tdata["transform"]
            col = int((e3035 - tf.c) / tf.a); row = int((tf.f - n3035) / abs(tf.e))
            dh, dw = dtm.shape
            if 0 <= row < dh and 0 <= col < dw:
                val = float(dtm[row, col])
                if np.isfinite(val):
                    return round(val, 2)
            # Try patch average
            v = dtm[np.isfinite(dtm)]
            if len(v) > 0:
                return round(float(np.nanmean(v)), 2)
        except Exception:
            pass
        return None

    # --- Per-parcel detail ---
    parcel_details = []
    n_parcel_no_tile = 0
    for p in cadastre_data["parcels"]:
        pd = {"parcel_id": p["parcel_id"], "area_sqm": round(p.get("area_sqm",0), 1)}
        geom_3035 = p["geometry"]
        try:
            cw = transform_to_wgs(geom_3035.centroid)
            pd["centroid"] = {"lon": round(cw.x,7), "lat": round(cw.y,7)}
        except Exception: pass
        try:
            c3 = geom_3035.centroid
            tr, tdata = _find_tile_with_data(c3.x, c3.y)
            if tr and tdata:
                dtm = tdata["dtm"]; tf = tdata["transform"]
                col = int((c3.x - tf.c)/tf.a); row = int((tf.f - c3.y)/abs(tf.e))
                dh, dw = dtm.shape
                if 0 <= row < dh and 0 <= col < dw:
                    val = float(dtm[row,col])
                    if np.isfinite(val): pd["elevation_m"] = round(val,2)
                    else:
                        patch = dtm[max(0,row-2):min(dh,row+3), max(0,col-2):min(dw,col+3)]
                        v = patch[np.isfinite(patch)]
                        if len(v) > 0: pd["elevation_m"] = round(float(np.nanmean(v)),2)
            else:
                n_parcel_no_tile += 1
                # Fallback: direct DTM point read
                elev = _point_elevation_fallback(c3.x, c3.y)
                if elev is not None:
                    pd["elevation_m"] = elev
                if tr.get("labels") is not None and objects:
                    from rasterio.features import rasterize as rio_rasterize
                    pm = rio_rasterize([(geom_3035,1)], out_shape=tr["shape"],
                                       transform=tr["transform"], fill=0, dtype=np.uint8,
                                       all_touched=True).astype(bool)
                    pl = tr["labels"][pm]; pn = tdata["ndsm"][pm]
                    tc_ = Counter(); th_ = defaultdict(list)
                    for lbl in np.unique(pl):
                        obj = obj_map.get(int(lbl))
                        if obj: npx = int((pl==lbl).sum()); tc_[obj.obj_type] += npx; th_[obj.obj_type].append(obj.height_max)
                    if tc_:
                        pd["area_summary"] = {t: {"area_sqm": px, "fraction": round(px/max(int(pm.sum()),1),4)} for t, px in tc_.most_common()}
                        pd["height_distribution"] = {t: {"min": round(min(hs_),2), "max": round(max(hs_),2), "mean": round(sum(hs_)/len(hs_),2)} for t, hs_ in th_.items() if hs_}
                    veg = {'tree','shrub','grass','hedge','crop','orchard','vineyard','garden'}
                    vpx = sum(v for k,v in tc_.items() if k in veg); tpx = max(int(pm.sum()),1)
                    pd["vegetated_fraction"] = round(vpx/tpx, 4); pd["is_vegetated"] = vpx/tpx > 0.5
                    vh = pn[np.isfinite(pn)]
                    if len(vh) > 0: pd["ndsm_max_m"] = round(float(np.max(vh)),2); pd["ndsm_mean_m"] = round(float(np.mean(vh)),2)
        except Exception as e:
            log.debug("JSON: parcel %s enrichment failed: %s", p.get("parcel_id"), e)
        parcel_details.append(pd)
    if n_parcel_no_tile > 0:
        log.warning("JSON: %d/%d parcels have no matching tile (centroid outside all tile bounds)",
                    n_parcel_no_tile, len(cadastre_data["parcels"]))
    summary["parcels"] = {
        "count": len(cadastre_data["parcels"]),
        "total_area_sqm": round(sum(p.get("area_sqm",0) for p in cadastre_data["parcels"]), 1),
        "details": parcel_details}
    # --- Per-building detail ---
    bld_details = []
    n_bld_no_tile = 0
    for b in cadastre_data["building_footprints"]:
        bd = {}; geom_3035 = b["geometry"]; props = b.get("properties",{})
        bd["building_id"] = props.get("building_id", props.get("id",""))
        bd["footprint_area_sqm"] = round(float(geom_3035.area), 1)
        try:
            cw = transform_to_wgs(geom_3035.centroid)
            bd["centroid"] = {"lon": round(cw.x,7), "lat": round(cw.y,7)}
        except Exception: pass
        try:
            c3 = geom_3035.centroid
            tr, tdata = _find_tile_with_data(c3.x, c3.y)
            if tr and tdata:
                from rasterio.features import rasterize as rio_rasterize
                dh, dw = tdata["shape"]
                bm = rio_rasterize([(geom_3035,1)], out_shape=(dh,dw), transform=tdata["transform"],
                                   fill=0, dtype=np.uint8, all_touched=True).astype(bool)
                oh = tdata["ndsm"][bm]; oh = oh[np.isfinite(oh)]
                dv = tdata["dsm"][bm]; dv = dv[np.isfinite(dv)]
                if len(oh) > 0:
                    mh = float(np.nanmax(oh))
                    bd["max_height_m"] = round(mh,2); bd["mean_height_m"] = round(float(np.nanmean(oh)),2)
                    bd["dsm_std"] = round(float(np.std(dv)),2) if len(dv) > 0 else 0.0
                    bd["roof_type_hint"] = "flat" if bd["dsm_std"] < 1.5 else "pitched"
                    bd["stories_est"] = max(1, round(mh/3.0))
                if tr.get("labels") is not None and objects:
                    bl = tr["labels"][bm]; tc_ = Counter()
                    for lbl in np.unique(bl):
                        obj = obj_map.get(int(lbl))
                        if obj: tc_[obj.obj_type] += int((bl==lbl).sum())
                    if tc_: bd["segment_types"] = {t:px for t,px in tc_.most_common()}
            else:
                n_bld_no_tile += 1
                # Fallback: at least get elevation for the centroid
                elev = _point_elevation_fallback(c3.x, c3.y)
                if elev is not None:
                    bd["centroid_dtm_m"] = elev
        except Exception as e:
            log.debug("JSON: building %s enrichment failed: %s", bd.get("building_id"), e)
        bld_details.append(bd)
    if n_bld_no_tile > 0:
        log.warning("JSON: %d/%d buildings have no matching tile", n_bld_no_tile, len(cadastre_data["building_footprints"]))
    # Free tile data cache
    del _tile_data_cache
    summary["building_footprints"] = {"count": len(cadastre_data["building_footprints"]), "details": bld_details}
    # --- Coverage ---
    nwe = sum(1 for p in parcel_details if p.get("elevation_m") is not None)
    nwa = sum(1 for p in parcel_details if p.get("area_summary"))
    nbh = sum(1 for b in bld_details if b.get("max_height_m") is not None)
    summary["coverage"] = {
        "n_tiles": n_tiles, "tile_km": tile_km, "total_segmented_area_sqm": total_seg_pixels,
        "parcel_elevation_coverage_pct": round(100*nwe/max(len(parcel_details),1), 1),
        "parcel_segmentation_coverage_pct": round(100*nwa/max(len(parcel_details),1), 1),
        "building_height_coverage_pct": round(100*nbh/max(len(bld_details),1), 1),
        "note": "Full KG tiled segmentation; every parcel/building has elevation + segmentation data."}
    # --- Data quality ---
    dq = compute_data_quality(tile_data_availability or [])
    summary["data_quality"] = dq
    # --- Segment summary (counts only; full points are in GPKG) ---
    seg_type_counts = Counter(obj.obj_type for obj in objects)
    summary["segments"] = {
        "total": len(objects),
        "by_type": dict(seg_type_counts.most_common()),
    }
    # --- Methods ---
    summary["methods"] = {
        "segmentation": f"Felzenszwalb+RAG on {n_tiles} overlapping {tile_km}km tiles, centroid-dedup",
        "classification": "Random Forest (44 features, cadastre+OSM trained) + rule-based fallback",
        "calibration": "Cadastre footprints for confidence boosting",
        "height": "BEV ALS DTM/DSM 1m, nDSM = DSM - DTM",
        "temporal_change": "DTM/DSM differencing, " + ", ".join(sorted(ti.DATASETS.keys())),
        "ortho": "BEV DOP RGBI 0.2m", "ndvi_bev": "(NIR-Red)/(NIR+Red) from BEV DOP",
        "ndvi_copernicus": "Sentinel-2 L2A, openEO, 10m",
        "ndvi_harmonics": "Harmonic fit to monthly Sentinel-2 NDVI",
        "sar": "Sentinel-1 IW GRD, VV+VH, openEO",
        "terrain": "Slope/aspect/TRI from DTM, tiled",
        "texture": "GLCM from ortho greyscale",
        "hansen": "Hansen GFC-2024-v1.12, 30m",
        "infrastructure": "austria-power API",
        "roof_type": "DSM std < 1.5m = flat", "stories_est": "max_h/3m",
        "stem_volume": "0.3*canopy_area*height/3", "parking_spots": "area/12.5m\u00b2",
        "earthwork_volume": "mean(|nDSM|)*area",
        "fragmentation": "Shannon diversity", "edge_density": "perimeter/area",
        "phenology": "harmonic amplitude+phase \u2192 class",
        "cadastre_source": "BEV INSPIRE cadastre via cadastre-process-api.exe.xyz",
        "data_sources": [
            "BEV ALS DTM/DSM 1m (2022-2024)", "BEV DOP RGBI 0.2m",
            "Sentinel-2 L2A 10m NDVI (openEO)", "ESA WorldCover 10m",
            "Sentinel-1 SAR IW GRD 10m (openEO)", "Hansen GFC-2024-v1.12 30m",
            "Austrian Cadastre (BEV INSPIRE)", "Austria Power Infrastructure API"]}
    return summary


# === SECTION: process_one_kg() — core per-KG pipeline (runs in subprocess) ===

def _compute_tile_grid(west, south, east, north, tile_km=1.5, overlap_km=0.1):
    """Compute a grid of overlapping tiles covering the full KG bbox.

    Returns list of (w, s, e, n) bboxes in WGS84.
    Tiles overlap by *overlap_km* so edge objects are fully contained
    in at least one tile.
    """
    cos_lat = np.cos(np.radians((south + north) / 2))
    dx_deg = tile_km / (111 * cos_lat)
    dy_deg = tile_km / 111
    step_x = (tile_km - overlap_km) / (111 * cos_lat)
    step_y = (tile_km - overlap_km) / 111

    tiles = []
    y = south
    while y < north:
        x = west
        while x < east:
            tw = x
            ts = y
            te = min(x + dx_deg, east + dx_deg)  # allow overshoot
            tn = min(y + dy_deg, north + dy_deg)
            tiles.append((tw, ts, te, tn))
            x += step_x
        y += step_y
    return tiles


def _merge_terrain_stats(stats_list: list[tuple[dict, int]]) -> dict:
    """Merge terrain stats from multiple tiles.

    Each entry is (stats_dict, n_valid_pixels).  Numeric stats are
    merged via pixel-weighted averages; min/max take the extremes;
    percentile-based fields use weighted means (approximate but fine
    for summary stats).
    """
    if not stats_list:
        return {}
    if len(stats_list) == 1:
        return stats_list[0][0]

    total_px = sum(n for _, n in stats_list)
    if total_px == 0:
        return stats_list[0][0]

    def _wmean(key_path):
        vals = []
        for s, n in stats_list:
            v = s
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None and isinstance(v, (int, float)):
                vals.append((v, n))
        if not vals:
            return None
        return round(sum(v * w for v, w in vals) / sum(w for _, w in vals), 3)

    def _wmin(key_path):
        vals = []
        for s, n in stats_list:
            v = s
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None and isinstance(v, (int, float)):
                vals.append(v)
        return round(min(vals), 2) if vals else None

    def _wmax(key_path):
        vals = []
        for s, n in stats_list:
            v = s
            for k in key_path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None and isinstance(v, (int, float)):
                vals.append(v)
        return round(max(vals), 2) if vals else None

    merged = {
        "elevation": {
            "min": _wmin(["elevation", "min"]),
            "max": _wmax(["elevation", "max"]),
            "mean": _wmean(["elevation", "mean"]),
            "std": _wmean(["elevation", "std"]),
            "range": None,  # recompute
            "p10": _wmean(["elevation", "p10"]),
            "p50": _wmean(["elevation", "p50"]),
            "p90": _wmean(["elevation", "p90"]),
        },
        "slope_deg": {
            "min": _wmin(["slope_deg", "min"]),
            "max": _wmax(["slope_deg", "max"]),
            "mean": _wmean(["slope_deg", "mean"]),
            "std": _wmean(["slope_deg", "std"]),
        },
        "slope_classes_pct": {},
        "aspect_distribution_pct": {},
        "ruggedness_tri": {
            "mean": _wmean(["ruggedness_tri", "mean"]),
            "max": _wmax(["ruggedness_tri", "max"]),
            "classification": None,
        },
        "area_sqm": total_px,
        "area_ha": round(total_px / 10000, 2),
    }
    emin = merged["elevation"]["min"]
    emax = merged["elevation"]["max"]
    if emin is not None and emax is not None:
        merged["elevation"]["range"] = round(emax - emin, 2)

    # Weighted-average slope classes and aspect distribution
    for cls_key in ["slope_classes_pct", "aspect_distribution_pct"]:
        all_keys = set()
        for s, _ in stats_list:
            all_keys.update(s.get(cls_key, {}).keys())
        for k in all_keys:
            merged[cls_key][k] = round(
                sum(s.get(cls_key, {}).get(k, 0) * n for s, n in stats_list) / total_px, 1)

    # TRI classification from merged mean
    tri_mean = merged["ruggedness_tri"]["mean"]
    if tri_mean is not None:
        if tri_mean < 0.1:
            merged["ruggedness_tri"]["classification"] = "level"
        elif tri_mean < 0.3:
            merged["ruggedness_tri"]["classification"] = "nearly level"
        elif tri_mean < 1.0:
            merged["ruggedness_tri"]["classification"] = "slightly rugged"
        elif tri_mean < 3.0:
            merged["ruggedness_tri"]["classification"] = "intermediately rugged"
        elif tri_mean < 10.0:
            merged["ruggedness_tri"]["classification"] = "moderately rugged"
        else:
            merged["ruggedness_tri"]["classification"] = "highly rugged"

    return merged


def process_one_kg(kg: dict, include_copernicus: bool = True, max_km: float = None,
                   failed_tiles: dict = None, rf_only: bool = False) -> dict:
    """Process a single KG with tiled segmentation for full coverage.

    The full KG is divided into overlapping 1.5km tiles.  Each tile
    undergoes the full pipeline (multi-date LiDAR, ortho, Copernicus,
    Hansen, segmentation, classification).  Results are merged into
    one set of vectors/stats covering the entire KG.

    Parameters
    ----------
    failed_tiles : dict[int, float] | None
        Mapping of original tile index → sub-tile km.  On a retry
        after timeout, only the tile that was in-progress is
        sub-divided at the given resolution.  Already-completed
        tiles are restored from checkpoint and keep their original
        quality.  Example: ``{3: 0.5}`` means "sub-divide original
        tile 3 into 0.5 km sub-tiles".

    This function runs in a subprocess for memory isolation.
    """
    import raster_io
    import tile_index as ti
    import object_segmentation as oc
    import terrain_analysis as ta

    kg_code = kg["kg_code"]
    result = {"kg_code": kg_code, "success": False, "step": "init", "files": {}}

    # --- Tile checkpoint cache for resume-on-restart ---
    _tile_ckpt_dir = DATA_DIR / "tile_checkpoints" / kg_code
    _tile_ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _save_tile_checkpoint(tile_idx, seg_result_entry, core_objects,
                              new_buildings, infra_objects, tile_avail_entry,
                              terrain_entry, spectral_accum, cop_accum,
                              hansen_accum, seg_pixels):
        """Persist a completed tile to disk for crash recovery."""
        import pickle as _pkl
        ckpt = {
            "seg_result": seg_result_entry,
            "core_objects": core_objects,
            "new_buildings": new_buildings,
            "infrastructure": infra_objects,
            "tile_avail": tile_avail_entry,
            "terrain": terrain_entry,      # (stats, n_px) or None
            "spectral_accum": spectral_accum,  # partial spectral_info entries
            "copernicus_accum": cop_accum,      # partial copernicus_info entries
            "hansen_accum": hansen_accum,        # partial hansen_info entries
            "seg_pixels": seg_pixels,
        }
        tmp = _tile_ckpt_dir / f"tile_{tile_idx}.pkl.tmp"
        dst = _tile_ckpt_dir / f"tile_{tile_idx}.pkl"
        with open(tmp, "wb") as f:
            _pkl.dump(ckpt, f, protocol=_pkl.HIGHEST_PROTOCOL)
        tmp.rename(dst)

    def _load_tile_checkpoint(tile_idx):
        """Load a cached tile result. Returns dict or None."""
        import pickle as _pkl
        dst = _tile_ckpt_dir / f"tile_{tile_idx}.pkl"
        if not dst.exists():
            return None
        try:
            with open(dst, "rb") as f:
                return _pkl.load(f)
        except Exception:
            dst.unlink(missing_ok=True)
            return None

    def _clear_tile_checkpoints():
        """Remove all checkpoints for this KG (after successful completion)."""
        import shutil
        shutil.rmtree(_tile_ckpt_dir, ignore_errors=True)

    _prev_step = [None, None]  # [step_name, start_time]
    _step_times = {}           # step_name → seconds

    _tile_progress = [0, 0]  # [current_tile_idx, n_tiles]

    # --- Subprocess warning/error relay ---
    # Write WARNING+ log records to a JSONL file so the parent process
    # step-monitor thread can relay them to progress.add_log().
    _warnings_file = DATA_DIR / "subprocess_warnings.jsonl"
    try:
        _warnings_file.write_text("")  # truncate on start
    except Exception:
        pass

    class _WarningRelayHandler(logging.Handler):
        """Logging handler that appends WARNING+ records to a JSONL file."""
        def __init__(self, path):
            super().__init__(level=logging.WARNING)
            self._path = path
        # Known-harmless messages to suppress entirely
        _SUPPRESS = (
            "DeprecationWarning: 'Memory' driver is deprecated",
            "Failed to parse API error response",
            "NaN nodata value cannot be recorded",
        )
        # Messages to downgrade from error → warning (non-fatal)
        _DOWNGRADE = (
            "Preflight process graph validation failed",
        )
        def emit(self, record):
            try:
                msg = self.format(record)
                if any(s in msg for s in self._SUPPRESS):
                    return
                lvl = "error" if record.levelno >= logging.ERROR else "warning"
                # Downgrade non-fatal errors to warnings
                if lvl == "error" and any(s in msg for s in self._DOWNGRADE):
                    lvl = "warning"
                import json as _j
                entry = _j.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "level": lvl,
                    "msg": msg,
                    "step": _prev_step[0] or "",
                }) + "\n"
                with open(self._path, "a") as f:
                    f.write(entry)
            except Exception:
                pass

    _relay_handler = _WarningRelayHandler(str(_warnings_file))
    _relay_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.getLogger().addHandler(_relay_handler)

    _subtile_progress = [None]  # [dict | None] — set when processing a sub-tile
    _tile_statuses = []  # [{lat, lng, status, issues, subtile_info}] — per-tile dots for map

    def _report_step(step, detail=""):
        try:
            now = time.time()
            now_iso = datetime.now(timezone.utc).isoformat()
            # Record elapsed time for previous step
            if _prev_step[0] is not None and _prev_step[1] is not None:
                elapsed = round(now - _prev_step[1], 1)
                _step_times[_prev_step[0]] = round(
                    _step_times.get(_prev_step[0], 0) + elapsed, 1)
            _prev_step[0] = step
            _prev_step[1] = now
            step_file = DATA_DIR / "current_step.json"
            import json as _json
            payload = {"step": step, "detail": detail,
                       "ts": now_iso, "step_times": _step_times,
                       "current_tile": _tile_progress[0],
                       "n_tiles": _tile_progress[1]}
            if _subtile_progress[0] is not None:
                payload["subtile"] = _subtile_progress[0]
            if _tile_statuses:
                payload["tile_statuses"] = _tile_statuses
            _json.dump(payload, open(str(step_file) + ".tmp", "w"))
            os.rename(str(step_file) + ".tmp", str(step_file))
        except Exception:
            pass

    try:
        # --- Determine bbox ---
        result["step"] = "bbox"
        bb = kg.get("bbox", {})
        if isinstance(bb, dict) and "min_lon" in bb:
            west, south = bb["min_lon"], bb["min_lat"]
            east, north = bb["max_lon"], bb["max_lat"]
        else:
            resp = requests.get(f"{CADASTRE_BASE}/search/kg",
                                params={"code": kg_code}, timeout=30)
            resp.raise_for_status()
            kgs = resp.json().get("data", [])
            if not kgs:
                result["error"] = "KG not found"
                return result
            bb = kgs[0].get("bbox", {})
            west, south = bb["min_lon"], bb["min_lat"]
            east, north = bb["max_lon"], bb["max_lat"]

        full_west, full_south, full_east, full_north = west, south, east, north
        obs_year = ti.dataset_to_year(ti.DEFAULT_DATASET)

        # --- 1. Cadastre ---
        result["step"] = "cadastre"
        _report_step("cadastre")
        cadastre_data = fetch_cadastre_data(kg_code)
        result["n_parcels"] = len(cadastre_data["parcels"])
        result["n_buildings"] = len(cadastre_data["building_footprints"])
        _report_step("cadastre", f"{len(cadastre_data['parcels'])} parcels, "
                     f"{len(cadastre_data['building_footprints'])} buildings")

        # --- 1b. Refine bbox from cadastre geometry union ---
        all_cad_geoms = ([p["geometry"] for p in cadastre_data["parcels"]]
                         + [b["geometry"] for b in cadastre_data["building_footprints"]])
        if all_cad_geoms:
            try:
                from shapely.ops import unary_union as _unary_union
                from shapely.validation import make_valid
                valid_geoms = []
                for g in all_cad_geoms:
                    try:
                        valid_geoms.append(make_valid(g) if not g.is_valid else g)
                    except Exception:
                        valid_geoms.append(g.buffer(0))
                cad_union_3035 = _unary_union(valid_geoms)
                cad_union_wgs = transform_to_wgs(cad_union_3035)
                cb = cad_union_wgs.bounds
                full_west = min(full_west, cb[0])
                full_south = min(full_south, cb[1])
                full_east = max(full_east, cb[2])
                full_north = max(full_north, cb[3])
                log.info("KG %s: full bbox from cadastre union: %.4f,%.4f → %.4f,%.4f",
                         kg_code, full_west, full_south, full_east, full_north)
            except Exception as e:
                log.warning("KG %s: cadastre union failed, using API bbox: %s",
                            kg_code, e)

        # --- 2. Compute tile grid ---
        tile_km = max_km if max_km is not None else MAX_KG_AREA_KM
        base_tiles_wgs = _compute_tile_grid(
            full_west, full_south, full_east, full_north,
            tile_km=tile_km, overlap_km=0.1)

        # --- 2b. Expand failed tiles into sub-tiles ---
        # failed_tiles maps original tile_idx → sub_tile_km.
        # Replace that tile's bbox with a finer grid; keep others.
        _ft = failed_tiles or {}
        tiles_wgs = []      # final flat list of bboxes
        _subtile_info = {}  # maps new flat idx → (orig_tile_idx, sub_idx, n_sub, sub_km)
        _orig_to_flat = {}  # maps orig_idx → new flat_idx (non-expanded tiles only)
        for orig_idx, bbox in enumerate(base_tiles_wgs):
            if orig_idx in _ft:
                sub_km = _ft[orig_idx]
                ow, os_, oe, on = bbox
                subs = _compute_tile_grid(ow, os_, oe, on,
                                          tile_km=sub_km, overlap_km=0.05)
                log.info("KG %s: tile %d sub-divided into %d sub-tiles @ %.1fkm",
                         kg_code, orig_idx, len(subs), sub_km)
                for si, sb in enumerate(subs):
                    flat_idx = len(tiles_wgs)
                    _subtile_info[flat_idx] = (orig_idx, si, len(subs), sub_km)
                    tiles_wgs.append(sb)
            else:
                _orig_to_flat[orig_idx] = len(tiles_wgs)
                tiles_wgs.append(bbox)

        n_tiles = len(tiles_wgs)

        # --- 2c. Remap checkpoint files when indices shifted ---
        # Previous run saved checkpoints keyed by the *old* flat
        # index (= orig_idx when there was no expansion).  With
        # failed-tile expansion the flat indices shift.  We rename
        # files so _load_tile_checkpoint finds the right data, and
        # delete stale files that belong to a previous (now
        # superseded) expansion of a failed tile.
        if _ft and _orig_to_flat:
            existing = sorted(_tile_ckpt_dir.glob("tile_*.pkl"))
            # Parse old flat indices from filenames
            old_idxs = {}
            for p in existing:
                try:
                    old_idxs[int(p.stem.split("_")[1])] = p
                except (IndexError, ValueError):
                    continue
            # 1) Delete any checkpoint whose old flat idx is NOT a
            #    valid orig tile (i.e. it was a sub-tile from the
            #    previous expansion and is now stale).
            for old_idx, path in list(old_idxs.items()):
                if old_idx not in _orig_to_flat:
                    # Not a surviving original tile — stale sub-tile
                    # or the failed tile itself.
                    path.unlink(missing_ok=True)
                    log.debug("KG %s: removed stale checkpoint %s",
                              kg_code, path.name)
                    del old_idxs[old_idx]
            # 2) Rename surviving checkpoints to new flat indices.
            #    Rename in descending order to avoid collisions.
            renames = []
            for old_idx, path in old_idxs.items():
                new_idx = _orig_to_flat[old_idx]
                if new_idx != old_idx:
                    renames.append((path, _tile_ckpt_dir / f"tile_{new_idx}.pkl"))
            for src, dst in sorted(renames, key=lambda p: -int(p[0].stem.split("_")[1])):
                if src.exists() and not dst.exists():
                    src.rename(dst)
                    log.debug("KG %s: renamed checkpoint %s → %s",
                              kg_code, src.name, dst.name)
        _n_subtiled = len(_ft)
        if _n_subtiled:
            log.info("KG %s: %d tiles total (%d original tiles sub-divided)",
                     kg_code, n_tiles, _n_subtiled)
        log.info("KG %s: %d tiles (%.1fkm base) covering %.4f,%.4f → %.4f,%.4f",
                 kg_code, n_tiles, tile_km,
                 full_west, full_south, full_east, full_north)
        _tile_progress[1] = n_tiles
        _report_step("tiles", f"{n_tiles} tiles @ {tile_km}km")

        # --- Accumulators for merging tile results ---
        all_objects = []          # SegmentedObject list
        all_new_buildings = []    # vectorised new buildings
        all_infrastructure = []   # vectorised infrastructure
        terrain_stats_list = []   # (stats, n_px) for merging
        all_spectral_info = {}    # for JSON ndvi section
        all_copernicus_info = {}  # for JSON sar/harmonics
        all_hansen_info = {}      # for JSON hansen section
        tile_seg_results = []     # (tile_bbox_3035, labels, objects, transform, shape, ndsm, mask)
        tile_data_availability = []  # per-tile data source tracking
        total_seg_pixels = 0
        next_obj_id = 1           # global unique obj_id counter

        # --- 3. Process each tile ---
        for tile_idx, (tw, ts, te, tn) in enumerate(tiles_wgs):
            _tile_progress[0] = tile_idx + 1
            # Build label that includes sub-tile info when applicable
            _sti = _subtile_info.get(tile_idx)
            if _sti:
                _orig_idx, _si, _nsub, _skm = _sti
                tile_label = (f"tile {tile_idx+1}/{n_tiles} "
                              f"(sub-tile {_si+1}/{_nsub} of orig {_orig_idx+1} @ {_skm}km)")
                _subtile_progress[0] = {"orig_tile": _orig_idx + 1,
                                        "sub_idx": _si + 1,
                                        "n_sub": _nsub,
                                        "sub_km": _skm}
            else:
                tile_label = f"tile {tile_idx+1}/{n_tiles}"
                _subtile_progress[0] = None
            result["step"] = f"tile_{tile_idx+1}"
            # Track tile centroid + status for dashboard map
            _tile_lat = (ts + tn) / 2
            _tile_lng = (tw + te) / 2
            _tile_entry = {"lat": round(_tile_lat, 5), "lng": round(_tile_lng, 5),
                          "status": "current", "issues": {},
                          "idx": tile_idx + 1}
            if _sti:
                _tile_entry["subtile"] = {"orig": _orig_idx + 1, "sub": _si + 1,
                                          "n_sub": _nsub, "km": _skm}
            _tile_statuses.append(_tile_entry)

            # --- Check for cached tile checkpoint ---
            cached_tile = _load_tile_checkpoint(tile_idx)
            if cached_tile is not None:
                log.info("KG %s %s: restored from checkpoint", kg_code, tile_label)
                if cached_tile["seg_result"] is not None:
                    tile_seg_results.append(cached_tile["seg_result"])
                all_objects.extend(cached_tile.get("core_objects", []))
                # Remap obj_ids to avoid collisions with fresh tiles
                for obj in cached_tile["core_objects"]:
                    obj.obj_id = next_obj_id
                    next_obj_id += 1
                all_new_buildings.extend(cached_tile.get("new_buildings", []))
                all_infrastructure.extend(cached_tile.get("infrastructure", []))
                tile_data_availability.append(cached_tile["tile_avail"])
                t_terrain_entry = cached_tile.get("terrain")
                if t_terrain_entry is not None:
                    terrain_stats_list.append(t_terrain_entry)
                total_seg_pixels += cached_tile.get("seg_pixels", 0)
                # Merge accumulated info dicts
                for k, v in cached_tile.get("spectral_accum", {}).items():
                    all_spectral_info.setdefault(k, []).extend(v)
                for k, v in cached_tile.get("copernicus_accum", {}).items():
                    all_copernicus_info.setdefault(k, []).extend(v)
                for k, v in cached_tile.get("hansen_accum", {}).items():
                    if k in ("loss_years",):
                        all_hansen_info.setdefault(k, []).extend(v)
                    elif k == "cf_sum":
                        all_hansen_info.setdefault("cf_sum", [0])
                        all_hansen_info["cf_sum"][0] += v[0] if v else 0
                    else:
                        all_hansen_info.setdefault(k, []).extend(v)
                # Remap labels in seg_result to match new obj_ids
                # (not needed — labels are per-tile and used by
                # build_*_tiled with their own obj_map)
                # Mark tile as done-from-cache on dashboard map
                _tile_avail = cached_tile.get("tile_avail", {})
                _tile_entry["status"] = "done"
                _tile_entry["issues"] = {
                    k: ("ok" if _tile_avail.get(k) else "miss")
                    for k in ["dtm", "ortho", "copernicus_ndvi", "worldcover", "sar", "hansen"]
                }
                _tile_entry["n_objects"] = len(cached_tile.get("core_objects", []))
                continue

            _report_step(f"tile_{tile_idx+1}", f"processing {tile_label}")

            # Per-tile data availability tracker
            tile_avail = {
                "tile_index": tile_idx + 1,
                "bbox_wgs": [round(tw, 5), round(ts, 5), round(te, 5), round(tn, 5)],
                "dtm": False, "dsm": False, "multi_date_dtm": False,
                "ortho": False, "copernicus_ndvi": False,
                "worldcover": False, "sar": False, "harmonics": False,
                "hansen": False, "cadastre_buildings": False,
                "infrastructure": False, "terrain": False,
                "segmentation": False, "n_objects": 0,
                "valid_pixels": 0,
            }

            # Per-tile accumulator snapshots (for checkpoint)
            _tile_spectral = {}
            _tile_copernicus = {}
            _tile_hansen = {}

            tile_geom_wgs = box(tw, ts, te, tn)
            tile_geom_3035 = transform_to_3035(tile_geom_wgs)

            # --- 3a. LiDAR (default date) ---
            _report_step("lidar", f"{tile_label} — reading DTM/DSM")
            try:
                tdata = raster_io.read_dtm_dsm(tile_geom_3035, ti.DEFAULT_DATASET)
            except Exception as e:
                log.warning("KG %s %s: LiDAR read failed: %s", kg_code, tile_label, e)
                tile_data_availability.append(tile_avail)
                _tile_entry["status"] = "error"
                _tile_entry["issues"]["lidar"] = "fail"
                continue
            th, tw_ = tdata["shape"]
            tvalid = int(tdata["mask"].sum())
            if tvalid < 100:
                log.info("KG %s %s: skipping (only %d valid px)", kg_code, tile_label, tvalid)
                tile_data_availability.append(tile_avail)
                # Cache empty tiles — deterministic, no point retrying
                _save_tile_checkpoint(tile_idx, None, [], [], [], tile_avail, None, {}, {}, {}, 0)
                _tile_entry["status"] = "skip"
                continue

            t_transform = tdata["transform"]
            t_mask = tdata["mask"]
            t_ndsm = tdata["ndsm"]
            tile_avail["dtm"] = True
            tile_avail["dsm"] = True
            tile_avail["valid_pixels"] = tvalid

            # --- 3b. Terrain stats for this tile ---
            _report_step("terrain", tile_label)
            _tile_terrain_entry = None
            try:
                t_terrain = ta.characterise_terrain(tdata["dtm"], t_mask)
                _tile_terrain_entry = (t_terrain, tvalid)
                terrain_stats_list.append(_tile_terrain_entry)
                tile_avail["terrain"] = True
            except Exception as e:
                log.warning("KG %s %s: terrain failed: %s", kg_code, tile_label, e)

            # --- 3c. Multi-date DTM/DSM ---
            _report_step("lidar", f"{tile_label} — multi-date")
            dtm_dates = None
            dsm_dates = None
            try:
                other_dates = sorted(d for d in ti.DATASETS if d != ti.DEFAULT_DATASET)
                if other_dates:
                    dtm_dates = {}
                    dsm_dates = {}
                    for date_key in other_dates:
                        try:
                            d2 = raster_io.read_dtm_dsm(tile_geom_3035, date_key)
                            mh = min(th, d2["shape"][0])
                            mw = min(tw_, d2["shape"][1])
                            dtm_dates[date_key] = d2["dtm"][:mh, :mw]
                            dsm_dates[date_key] = d2["dsm"][:mh, :mw]
                        except Exception:
                            pass
                    if dtm_dates:
                        mh = min(th, *(a.shape[0] for a in dtm_dates.values()))
                        mw = min(tw_, *(a.shape[1] for a in dtm_dates.values()))
                        dtm_dates[ti.DEFAULT_DATASET] = tdata["dtm"][:mh, :mw]
                        dsm_dates[ti.DEFAULT_DATASET] = tdata["dsm"][:mh, :mw]
                        for dk in list(dtm_dates):
                            dtm_dates[dk] = dtm_dates[dk][:mh, :mw]
                            dsm_dates[dk] = dsm_dates[dk][:mh, :mw]
                    else:
                        dtm_dates = dsm_dates = None
            except Exception:
                dtm_dates = dsm_dates = None
            if dtm_dates and len(dtm_dates) > 1:
                tile_avail["multi_date_dtm"] = True

            # --- 3d. Orthophoto ---
            _report_step("ortho", tile_label)
            spectral = None
            try:
                import ortho_io
                import concurrent.futures
                ORTHO_TIMEOUT = 180
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
                    fut = exe.submit(ortho_io.read_ortho_for_als, tdata)
                    try:
                        rgb, nir = fut.result(timeout=ORTHO_TIMEOUT)
                        spectral = ortho_io.compute_spectral_indices(rgb, nir=nir)
                        if rgb is not None:
                            spectral["red"] = rgb[0].astype(np.float32)
                            spectral["green"] = rgb[1].astype(np.float32)
                            spectral["blue"] = rgb[2].astype(np.float32)
                        if nir is not None:
                            spectral["nir"] = nir.astype(np.float32)
                    except concurrent.futures.TimeoutError:
                        log.warning("KG %s %s: ortho timed out", kg_code, tile_label)
            except Exception as e:
                log.warning("KG %s %s: ortho failed: %s", kg_code, tile_label, e)
            if spectral is not None:
                tile_avail["ortho"] = True

            # Accumulate spectral info for JSON
            if spectral and spectral.get("ndvi") is not None:
                ndvi_arr = spectral["ndvi"]
                v = ndvi_arr[t_mask[:ndvi_arr.shape[0], :ndvi_arr.shape[1]]]
                v = v[np.isfinite(v)]
                if len(v) > 0:
                    subsample = v[::max(1, len(v)//5000)].tolist()
                    all_spectral_info.setdefault("_vals", []).extend(subsample)
                    _tile_spectral.setdefault("_vals", []).extend(subsample)

            # --- 3e. Copernicus ---
            _report_step("copernicus", f"{tile_label} — NDVI")
            copernicus_data = None
            if include_copernicus:
                c_breaker = _read_circuit_breaker()
                if c_breaker["consecutive_failures"] < 3 or \
                   (time.time() - c_breaker["last_failure"]) >= c_breaker["cooldown"]:
                    try:
                        bbox_dict = {"west": tw, "south": ts, "east": te, "north": tn}
                        cop_cache = _get_cop_cache()
                        cop = {}
                        nd = cop_cache.get_ndvi(bbox_dict, year=obs_year)
                        if nd and nd.get("ndvi") is not None:
                            cop["ndvi"] = nd["ndvi"]
                            cop["transform"] = nd.get("transform")
                            cop["crs"] = nd.get("crs")
                        _report_step("copernicus", f"{tile_label} — WorldCover")
                        lc = cop_cache.get_landcover(bbox_dict)
                        if lc:
                            cop["landcover"] = lc
                        _report_step("copernicus", f"{tile_label} — SAR")
                        sar = cop_cache.get_sar(bbox_dict, year=obs_year)
                        if sar:
                            cop.update({k: sar[k] for k in ["vv", "vh"] if k in sar})
                            if "transform" in sar:
                                cop["sar_transform"] = sar["transform"]
                        if cop:
                            try:
                                _report_step("copernicus", f"{tile_label} — harmonics")
                                import ndvi_harmonics
                                import concurrent.futures as _cf
                                with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                                    _hfut = _ex.submit(
                                        ndvi_harmonics.get_harmonic_features,
                                        bbox_dict, obs_year)
                                    try:
                                        harm = _hfut.result(timeout=300)
                                        if harm is not None:
                                            cop["harmonics"] = harm
                                    except (_cf.TimeoutError, Exception):
                                        pass
                            except Exception:
                                pass
                        copernicus_data = cop if cop else None
                        # Update circuit breaker
                        if copernicus_data:
                            c_breaker["consecutive_failures"] = 0
                        else:
                            c_breaker["consecutive_failures"] += 1
                            c_breaker["last_failure"] = time.time()
                            c_breaker["cooldown"] = min(600, 60 * (2 ** min(c_breaker["consecutive_failures"], 4)))
                        _write_circuit_breaker(c_breaker)
                    except Exception as e:
                        from copernicus import CreditsExhaustedError
                        if isinstance(e, CreditsExhaustedError) or \
                           isinstance(e.__cause__, CreditsExhaustedError):
                            log.error("KG %s: Copernicus credits exhausted", kg_code)
                            result["copernicus_exhausted"] = True
                            COPERNICUS_PAUSE_FILE.write_text(
                                f"Credits exhausted at {datetime.now(timezone.utc).isoformat()}\n")
                        else:
                            log.warning("KG %s %s: Copernicus failed: %s",
                                        kg_code, tile_label, e)
                        c_breaker["consecutive_failures"] += 1
                        c_breaker["last_failure"] = time.time()
                        _write_circuit_breaker(c_breaker)

            # Accumulate Copernicus info + track availability
            if copernicus_data:
                if copernicus_data.get("ndvi") is not None:
                    tile_avail["copernicus_ndvi"] = True
                if copernicus_data.get("landcover") is not None:
                    tile_avail["worldcover"] = True
                if copernicus_data.get("vv") is not None or copernicus_data.get("vh") is not None:
                    tile_avail["sar"] = True
                if copernicus_data.get("harmonics") is not None:
                    tile_avail["harmonics"] = True
                for band in ['vv', 'vh']:
                    arr = copernicus_data.get(band)
                    if arr is not None:
                        v = arr[np.isfinite(arr)]
                        if len(v) > 0:
                            ss = v[::max(1, len(v)//2000)].tolist()
                            all_copernicus_info.setdefault(f"{band}_vals", []).extend(ss)
                            _tile_copernicus.setdefault(f"{band}_vals", []).extend(ss)
                if copernicus_data.get("ndvi") is not None:
                    cv = copernicus_data["ndvi"]
                    cv = cv[np.isfinite(cv)]
                    if len(cv) > 0:
                        ss = cv[::max(1, len(cv)//2000)].tolist()
                        all_copernicus_info.setdefault("cop_ndvi_vals", []).extend(ss)
                        _tile_copernicus.setdefault("cop_ndvi_vals", []).extend(ss)
                if copernicus_data.get("harmonics"):
                    harm = copernicus_data["harmonics"]
                    for hk in ['h_mean', 'h_amplitude', 'h_phase', 'h_rmse']:
                        arr = harm.get(hk)
                        if arr is not None:
                            v = arr[np.isfinite(arr)]
                            if len(v) > 0:
                                ss = v[::max(1, len(v)//2000)].tolist()
                                all_copernicus_info.setdefault(f"{hk}_vals", []).extend(ss)
                                _tile_copernicus.setdefault(f"{hk}_vals", []).extend(ss)

            # --- 3f. Hansen ---
            _report_step("hansen", tile_label)
            hansen_data = None
            try:
                hc = _get_hansen_cache()
                hansen_data = hc.get_forest_prior(
                    (tw, ts, te, tn), t_transform, (th, tw_))
            except Exception as e:
                log.warning("KG %s %s: Hansen failed: %s", kg_code, tile_label, e)
            # Accumulate Hansen
            if hansen_data:
                tile_avail["hansen"] = True
                ly = hansen_data.get("loss_year")
                if ly is not None:
                    all_hansen_info.setdefault("loss_years", []).append(ly)
                    _tile_hansen.setdefault("loss_years", []).append(ly)
                tc = hansen_data.get("treecover2000")
                if tc is not None:
                    ss = tc[np.isfinite(tc)].ravel()[::max(1, tc.size//2000)].tolist()
                    all_hansen_info.setdefault("tc_vals", []).extend(ss)
                    _tile_hansen.setdefault("tc_vals", []).extend(ss)
                cf = hansen_data.get("current_forest")
                if cf is not None:
                    cf_val = int(cf.sum())
                    all_hansen_info.setdefault("cf_sum", [0])
                    all_hansen_info["cf_sum"][0] += cf_val
                    _tile_hansen["cf_sum"] = [cf_val]

            # --- 3g. Segmentation ---
            _report_step("segment", f"{tile_label} — rasterise cadastre")

            building_fp_mask = None
            if cadastre_data["building_footprints"]:
                try:
                    from rasterio.features import rasterize as rio_rasterize
                    pairs = [(b["geometry"], 1) for b in cadastre_data["building_footprints"]
                             if not b["geometry"].is_empty]
                    if pairs:
                        building_fp_mask = rio_rasterize(
                            pairs, out_shape=(th, tw_), transform=t_transform,
                            fill=0, dtype=np.uint8, all_touched=True,
                        ).astype(bool)
                        tile_avail["cadastre_buildings"] = True
                except Exception:
                    pass

            infra = None
            try:
                from infrastructure_lookup import InfrastructureLookup
                infra = InfrastructureLookup.for_bbox(tw, ts, te, tn)
                if infra and len(infra) > 0:
                    tile_avail["infrastructure"] = True
            except Exception:
                pass

            _report_step("segment", f"{tile_label} — classify")
            try:
                seg_result = oc.segment_and_classify(
                    tdata["dtm"], tdata["dsm"], t_mask, t_transform,
                    dtm_dates=dtm_dates, dsm_dates=dsm_dates,
                    spectral=spectral, copernicus=copernicus_data,
                    building_footprints=building_fp_mask,
                    hansen=hansen_data,
                    observation_year=obs_year,
                    infra_lookup=infra,
                    rf_only=rf_only,
                )
            except Exception as e:
                log.warning("KG %s %s: segmentation failed: %s", kg_code, tile_label, e)
                tile_data_availability.append(tile_avail)
                continue

            t_objects = seg_result["objects"]
            t_labels = seg_result["labels"]
            tile_avail["segmentation"] = True
            tile_avail["n_objects"] = len(t_objects)

            # Remap obj_ids to global unique range
            id_remap = {}
            for obj in t_objects:
                old_id = obj.obj_id
                obj.obj_id = next_obj_id
                id_remap[old_id] = next_obj_id
                next_obj_id += 1
            # Remap label array
            new_labels = np.zeros_like(t_labels)
            for old_id, new_id in id_remap.items():
                new_labels[t_labels == old_id] = new_id
            t_labels = new_labels

            # Store tile segmentation result for parcel enrichment
            tile_bounds_3035 = raster_io.read_window_bbox.__wrapped__ if False else None
            # Compute tile bounds in EPSG:3035 from transform + shape
            import rasterio.transform
            t_bounds_3035 = rasterio.transform.array_bounds(th, tw_, t_transform)
            tile_seg_results.append({
                "bounds_3035": t_bounds_3035,  # (left, bottom, right, top)
                "labels": t_labels,
                "objects": t_objects,
                "transform": t_transform,
                "shape": (th, tw_),
                "ndsm": t_ndsm,
                "mask": t_mask,
                "bbox_wgs": (tw, ts, te, tn),
            })
            total_seg_pixels += tvalid

            # Filter objects: keep only those whose centroid is inside
            # the non-overlap core of this tile (avoid double-counting
            # at tile boundaries).
            # Core = tile shrunk by half the overlap on each side.
            core_shrink = 50  # 50m = half of 100m overlap
            core_left = t_bounds_3035[0] + core_shrink
            core_bottom = t_bounds_3035[1] + core_shrink
            core_right = t_bounds_3035[2] - core_shrink
            core_top = t_bounds_3035[3] - core_shrink
            # For edge tiles (first/last row/col), don't shrink outward
            if tile_idx == 0 or tw <= full_west + 0.0001:
                core_left = t_bounds_3035[0]
            if tile_idx == 0 or ts <= full_south + 0.0001:
                core_bottom = t_bounds_3035[1]
            if te >= full_east - 0.0001:
                core_right = t_bounds_3035[2]
            if tn >= full_north - 0.0001:
                core_top = t_bounds_3035[3]

            core_objects = []
            for obj in t_objects:
                if (core_left <= obj.centroid_e <= core_right and
                        core_bottom <= obj.centroid_n <= core_top):
                    core_objects.append(obj)
            all_objects.extend(core_objects)

            # --- 3h. Vectorise new buildings & infrastructure (this tile) ---
            _report_step("vectorise", f"{tile_label} — new buildings")
            _tile_nb = []
            _tile_iv = []
            try:
                _tile_nb = vectorise_unmatched_buildings(
                    t_objects, t_labels, t_mask, t_transform,
                    building_fp_mask, t_ndsm)
                all_new_buildings.extend(_tile_nb)
            except Exception as e:
                log.warning("KG %s %s: new buildings failed: %s", kg_code, tile_label, e)
            _report_step("vectorise", f"{tile_label} — infrastructure")
            try:
                _tile_iv = vectorise_infrastructure(
                    t_objects, t_labels, t_mask, t_transform, t_ndsm, tdata["dtm"])
                all_infrastructure.extend(_tile_iv)
            except Exception as e:
                log.warning("KG %s %s: infrastructure failed: %s", kg_code, tile_label, e)

            # Record tile data availability
            tile_data_availability.append(tile_avail)

            # Save tile checkpoint for crash recovery
            _save_tile_checkpoint(
                tile_idx,
                seg_result_entry=tile_seg_results[-1],
                core_objects=core_objects,
                new_buildings=_tile_nb,
                infra_objects=_tile_iv,
                tile_avail_entry=tile_avail,
                terrain_entry=_tile_terrain_entry,
                spectral_accum=_tile_spectral,
                cop_accum=_tile_copernicus,
                hansen_accum=_tile_hansen,
                seg_pixels=tvalid,
            )

            # Free tile memory
            del tdata, t_labels, t_objects, seg_result, spectral
            del copernicus_data, hansen_data, dtm_dates, dsm_dates
            del building_fp_mask, t_mask, t_ndsm
            gc.collect()

            log.info("KG %s %s: done (%d objects, %d core)",
                     kg_code, tile_label, len(id_remap), len(core_objects))
            # Update tile status for dashboard map
            _tile_entry["status"] = "done"
            _tile_entry["issues"] = {
                k: ("ok" if tile_avail.get(k) else "miss")
                for k in ["dtm", "ortho", "copernicus_ndvi", "worldcover", "sar", "hansen"]
            }
            _tile_entry["n_objects"] = len(core_objects)
            _report_step(f"tile_{tile_idx+1}",
                         f"done: {len(core_objects)} objects")

        # --- 4. Merge terrain stats ---
        terrain_stats = _merge_terrain_stats(terrain_stats_list)
        result["n_segments"] = len(all_objects)
        result["n_new_buildings"] = len(all_new_buildings)
        result["n_infrastructure"] = len(all_infrastructure)
        log.info("KG %s: merged %d objects from %d tiles, %d new buildings, %d infra",
                 kg_code, len(all_objects), n_tiles,
                 len(all_new_buildings), len(all_infrastructure))

        # --- 5. Build full GPKG ---
        result["step"] = "gpkg_full"
        _report_step("gpkg_full", f"{len(tile_seg_results)} tiles, {len(all_objects)} objects")
        full_gpkg = build_full_gpkg_tiled(
            kg_code, tile_seg_results, all_objects, obs_year, rf_only=rf_only)
        result["files"]["full_gpkg"] = full_gpkg

        # --- 6. Build light GPKG ---
        result["step"] = "gpkg_light"
        _report_step("gpkg_light", f"{len(cadastre_data['parcels'])} parcels, {len(all_new_buildings)} new bldg")
        light_gpkg = build_light_gpkg_tiled(
            kg_code, tile_seg_results, all_objects,
            cadastre_data, all_new_buildings, all_infrastructure,
            obs_year=obs_year, rf_only=rf_only)
        result["files"]["light_gpkg"] = light_gpkg

        # --- 7. Build JSON summary ---
        result["step"] = "json"
        _report_step("json", "building summary")
        json_summary = build_json_summary_tiled(
            kg_code, kg, tile_seg_results, all_objects,
            cadastre_data, terrain_stats,
            all_spectral_info, all_copernicus_info, all_hansen_info,
            all_new_buildings, all_infrastructure, obs_year,
            n_tiles=n_tiles, tile_km=tile_km,
            total_seg_pixels=total_seg_pixels,
            tile_data_availability=tile_data_availability)

        # Store quality info in result for logging / Zenodo
        dq = json_summary.get("data_quality", {})
        result["quality_score"] = dq.get("quality_score", 0)
        result["quality_grade"] = dq.get("quality_grade", "F")
        result["available_layers"] = dq.get("available_layers", [])
        result["missing_layers"] = dq.get("missing_layers", [])

        json_path = str(JSON_DIR / f"{kg_code}.json")
        with open(json_path, 'w') as f:
            json.dump(json_summary, f, indent=2, default=str)
        result["files"]["json"] = json_path

        result["success"] = True
        result["step"] = "done"
        # Clean up tile checkpoints on success
        _clear_tile_checkpoints()

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        log.error("KG %s failed at step %s: %s", kg_code, result["step"], e)

    # Clean up the warning relay handler
    try:
        logging.getLogger().removeHandler(_relay_handler)
    except Exception:
        pass

    gc.collect()
    return result

# === SECTION: Output validation ===

def validate_kg_outputs(kg_code: str, files: dict) -> list[str]:
    """Validate all KG output products. Returns list of issues (empty = all OK)."""
    issues = []

    # --- Full GPKG ---
    full_path = files.get("full_gpkg", "")
    if not full_path or not os.path.exists(full_path):
        issues.append("FULL_GPKG: missing")
    else:
        try:
            import re as _re
            expected_raster = ["DTM", "DSM", "nDSM", "segment_type", "segment_height"]
            found_layers = set()
            try:
                import sqlite3
                conn = sqlite3.connect(full_path)
                tables = [r[0] for r in conn.execute(
                    "SELECT table_name FROM gpkg_contents "
                    "WHERE data_type IN ('tiles', '2d-gridded-coverage')").fetchall()]
                vector_tabs = [r[0] for r in conn.execute(
                    "SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchall()]
                conn.close()
                found_layers = set(tables)
            except Exception:
                vector_tabs = []
            # Check required raster layers (stitched single layers)
            for layer in expected_raster:
                if layer not in found_layers:
                    issues.append(f"FULL_GPKG: missing raster layer '{layer}'")
            # Ortho is optional but note if missing
            has_ortho = any(t.startswith('Ortho_') for t in found_layers)
            if not has_ortho:
                log.info("  FULL_GPKG: no ortho layer (may not be available for this KG)")
            # Check segments vector layer
            if 'segments' not in vector_tabs:
                issues.append("FULL_GPKG: missing vector layer 'segments'")
            fsize = os.path.getsize(full_path)
            if fsize < 10_000:
                issues.append(f"FULL_GPKG: suspiciously small ({fsize} bytes)")
            log.info("  FULL_GPKG: %.1f MB, layers=%s", fsize/1e6, sorted(found_layers))
        except Exception as e:
            issues.append(f"FULL_GPKG: cannot open: {e}")

    # --- Light GPKG ---
    light_path = files.get("light_gpkg", "")
    if not light_path or not os.path.exists(light_path):
        issues.append("LIGHT_GPKG: missing")
    else:
        try:
            import sqlite3
            import re as _re
            conn = sqlite3.connect(light_path)
            # Raster tables
            raster_tables = [r[0] for r in conn.execute(
                "SELECT table_name FROM gpkg_contents "
                    "WHERE data_type IN ('tiles', '2d-gridded-coverage')").fetchall()]
            # Vector tables
            vector_tables = [r[0] for r in conn.execute(
                "SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchall()]
            conn.close()

            # segment_type and segment_height rasters (stitched single layers)
            if 'segment_type' not in raster_tables:
                issues.append("LIGHT_GPKG: missing segment_type raster")
            if 'segment_height' not in raster_tables:
                issues.append("LIGHT_GPKG: missing segment_height raster")

            # segments vector (single stitched layer)
            if 'segments' not in vector_tables:
                issues.append("LIGHT_GPKG: missing vector layer 'segments'")

            # parcels is always a single (non-tiled) layer
            if "parcels" not in vector_tables:
                issues.append("LIGHT_GPKG: missing vector layer 'parcels'")

            # Count features in vector layers
            try:
                import fiona
                for vt in vector_tables:
                    with fiona.open(light_path, layer=vt) as src:
                        n = len(src)
                        log.info("  LIGHT_GPKG: layer '%s' → %d features", vt, n)
                        if n == 0:
                            issues.append(f"LIGHT_GPKG: layer '{vt}' is empty")
            except Exception as e:
                issues.append(f"LIGHT_GPKG: cannot read vector layers: {e}")

            fsize = os.path.getsize(light_path)
            log.info("  LIGHT_GPKG: %.1f MB, rasters=%s, vectors=%s",
                     fsize/1e6, raster_tables, vector_tables)
        except Exception as e:
            issues.append(f"LIGHT_GPKG: cannot open: {e}")

    # --- JSON summary ---
    json_path = files.get("json", "")
    if not json_path or not os.path.exists(json_path):
        issues.append("JSON: missing")
    else:
        try:
            with open(json_path) as f:
                js = json.load(f)

            # Required top-level keys
            required_keys = [
                "version", "kg_code", "kg_name", "bbox", "total_area_sqm",
                "observation_period", "area_summary", "height_distribution",
                "landscape", "top_10_objects", "top_10_trees", "tree_stats",
                "terrain", "ndvi", "sar", "ndvi_harmonics",
                "temporal_change", "phenology",
                "hansen", "new_buildings", "infrastructure",
                "parcels", "building_footprints", "methods", "coverage",
                "data_quality",
            ]
            for k in required_keys:
                if k not in js:
                    issues.append(f"JSON: missing top-level key '{k}'")

            # Check area_summary not empty
            area_sum = js.get("area_summary", {})
            if not area_sum:
                issues.append("JSON: area_summary is empty")

            # Check total_area_sqm > 0
            if js.get("total_area_sqm", 0) <= 0:
                issues.append("JSON: total_area_sqm is zero or negative")

            # Check parcels have details
            parcels = js.get("parcels", {})
            p_details = parcels.get("details", [])
            if parcels.get("count", 0) > 0 and not p_details:
                issues.append("JSON: parcels.details is empty despite count>0")
            else:
                # Spot-check parcels
                if p_details:
                    n_with_elev = sum(1 for p in p_details if p.get("elevation_m") is not None)
                    n_with_area = sum(1 for p in p_details if p.get("area_summary"))
                    log.info("  JSON: %d/%d parcels with elevation, %d/%d with area_summary",
                             n_with_elev, len(p_details), n_with_area, len(p_details))
                    if n_with_elev == 0:
                        issues.append("JSON: no parcels have elevation_m")
                    elif n_with_elev < len(p_details) * 0.8:
                        issues.append(f"JSON: only {n_with_elev}/{len(p_details)} parcels have elevation_m (expect >80% with full-KG DTM)")
                    if n_with_area == 0 and len(p_details) > 5:
                        # area_summary only available for parcels inside segmentation window
                        log.info("  JSON: no parcels have area_summary (may be outside seg window)")
                    # Check largest parcel has full data
                    biggest = max(p_details, key=lambda p: p.get("area_sqm", 0))
                    if not biggest.get("parcel_id"):
                        issues.append("JSON: largest parcel missing parcel_id")
                    if not biggest.get("centroid"):
                        issues.append("JSON: largest parcel missing centroid")

            # Check building_footprints have details
            bfp = js.get("building_footprints", {})
            bf_details = bfp.get("details", [])
            if bfp.get("count", 0) > 0 and not bf_details:
                issues.append("JSON: building_footprints.details empty despite count>0")
            else:
                if bf_details:
                    b0 = bf_details[0]
                    if not b0.get("centroid"):
                        issues.append("JSON: first building missing centroid")
                    if b0.get("max_height_m") is None:
                        issues.append("JSON: first building missing max_height_m")

            # Check for NaN/None in critical numeric fields
            for t, vals in js.get("height_distribution", {}).items():
                for fk in ["min", "max", "mean"]:
                    v = vals.get(fk)
                    if v is None or (isinstance(v, float) and (v != v)):  # NaN check
                        issues.append(f"JSON: height_distribution[{t}].{fk} is NaN/None")

            # Check terrain populated
            terrain = js.get("terrain", {})
            if not terrain or terrain.get("steepness_mean_deg") is None:
                issues.append("JSON: terrain stats missing/empty")

            # Check new enriched sections (SAR, harmonics, temporal, phenology)
            sar = js.get("sar", {})
            if not sar:
                issues.append("JSON: sar section missing/empty (Copernicus may have been skipped)")
            else:
                if sar.get("vv_mean_db") is None and sar.get("method"):
                    issues.append("JSON: sar has method but no vv_mean_db")

            harmonics = js.get("ndvi_harmonics", {})
            if not harmonics:
                issues.append("JSON: ndvi_harmonics section missing/empty")
            else:
                for hk in ["mean_mean", "amplitude_mean", "phase_mean"]:
                    v = harmonics.get(hk)
                    if v is not None and isinstance(v, float) and v != v:
                        issues.append(f"JSON: ndvi_harmonics.{hk} is NaN")

            temporal = js.get("temporal_change", {})
            if not temporal:
                issues.append("JSON: temporal_change section missing/empty")

            phenology = js.get("phenology", {})
            # Phenology may be empty if no vegetated objects — not an error, just note
            if not phenology:
                log.info("  JSON: phenology section empty (may be ok for non-vegetated KGs)")

            # Check tree_stats
            tree_stats = js.get("tree_stats", {})
            if not tree_stats:
                issues.append("JSON: tree_stats section missing")

            # Check observation_period
            obs = js.get("observation_period", {})
            if not obs or not obs.get("start"):
                issues.append("JSON: observation_period missing/empty")

            # Deep NaN scan on all numeric values
            def _scan_nan(obj, path=""):
                nan_paths = []
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        nan_paths.extend(_scan_nan(v, f"{path}.{k}"))
                elif isinstance(obj, list):
                    for i, v in enumerate(obj[:5]):  # sample first 5
                        nan_paths.extend(_scan_nan(v, f"{path}[{i}]"))
                elif isinstance(obj, float) and obj != obj:
                    nan_paths.append(path)
                return nan_paths
            nan_paths = _scan_nan(js)
            if nan_paths:
                issues.append(f"JSON: NaN found at {len(nan_paths)} path(s): {', '.join(nan_paths[:5])}")

            # Check methods present
            methods = js.get("methods", {})
            if len(methods) < 5:
                issues.append(f"JSON: methods section sparse ({len(methods)} entries)")
            # Check new method keys exist
            for mk in ["sar", "ndvi_harmonics", "temporal_change", "phenology"]:
                if mk not in methods:
                    issues.append(f"JSON: methods missing '{mk}' key")

            # Summary stats
            n_parcel_details = len(p_details)
            n_bfp_details = len(bf_details)
            n_types = len(area_sum)
            log.info("  JSON: %.1f KB, %d types, %d parcel details, %d building details",
                     os.path.getsize(json_path)/1024, n_types,
                     n_parcel_details, n_bfp_details)

        except json.JSONDecodeError as e:
            issues.append(f"JSON: parse error: {e}")
        except Exception as e:
            issues.append(f"JSON: validation error: {e}")

    return issues


def log_kg_stats_from_json(kg_code: str, json_path: str, elapsed: float):
    """Read the output JSON and log key stats. Keeps log concise."""
    try:
        with open(json_path) as f:
            js = json.load(f)
    except Exception as e:
        log.warning("KG %s: cannot read JSON for stats: %s", kg_code, e)
        return

    parcels = js.get("parcels", {})
    bfp = js.get("building_footprints", {})
    landscape = js.get("landscape", {})
    terrain = js.get("terrain", {})
    area_sum = js.get("area_summary", {})
    tree_stats = js.get("tree_stats", {})
    top_obj = js.get("top_10_objects", [{}])
    top_tree = js.get("top_10_trees", [{}])
    hansen = js.get("hansen", {})
    ndvi = js.get("ndvi", {})
    new_b = js.get("new_buildings", {})
    infra = js.get("infrastructure", {})

    n_seg = landscape.get("n_segments", 0)
    n_par = parcels.get("count", 0)
    n_bld = bfp.get("count", 0)
    n_new = new_b.get("count", 0)
    n_inf = infra.get("total", 0)

    log.info("KG %s: SUCCESS in %.0fs", kg_code, elapsed)
    log.info("  %d segments | %d parcels | %d buildings | %d new buildings | %d infrastructure",
             n_seg, n_par, n_bld, n_new, n_inf)
    log.info("  area=%.0f m\u00b2 | dominant=%s | vegetated=%.0f%% | shannon=%.2f",
             js.get("total_area_sqm", 0),
             landscape.get("dominant_type", "?"),
             (landscape.get("vegetated_fraction", 0) or 0) * 100,
             landscape.get("shannon_diversity", 0) or 0)
    top5 = list(area_sum.items())[:5]
    if top5:
        log.info("  top types: %s",
                 ", ".join(f"{t}={v.get('area_sqm',0)}m\u00b2" for t, v in top5))
    log.info("  tallest=%.1fm | tallest_tree=%.1fm | trees=%d | stem_vol=%.0f m\u00b3",
             top_obj[0].get("height_max_m", 0) if top_obj else 0,
             top_tree[0].get("height_m", 0) if top_tree else 0,
             tree_stats.get("count", 0),
             tree_stats.get("est_stem_volume_m3", 0) or 0)
    log.info("  elev=[%.0f,%.0f]m range=%.0fm | slope=%.1f\u00b0 | aspect=%s",
             terrain.get("elevation_min_m") or 0,
             terrain.get("elevation_max_m") or 0,
             terrain.get("elevation_range_m") or 0,
             terrain.get("steepness_mean_deg") or 0,
             terrain.get("aspect_dominant", "?"))
    log.info("  ndvi: bev=%.3f cop=%.3f | hansen_loss=%d px",
             ndvi.get("bev_nir_mean", 0) or 0,
             ndvi.get("copernicus_mean", 0) or 0,
             hansen.get("total_loss_pixels", 0) or 0)
    # New enriched sections
    sar = js.get("sar", {})
    if sar:
        log.info("  sar: VV=%.1fdB VH=%.1fdB",
                 sar.get("vv_mean_db", 0) or 0,
                 sar.get("vh_mean_db", 0) or 0)
    else:
        log.info("  sar: (empty)")
    harm = js.get("ndvi_harmonics", {})
    if harm:
        log.info("  harmonics: mean=%.4f amp=%.4f phase=%.4f",
                 harm.get("mean_mean", 0) or 0,
                 harm.get("amplitude_mean", 0) or 0,
                 harm.get("phase_mean", 0) or 0)
    else:
        log.info("  harmonics: (empty)")
    temporal = js.get("temporal_change", {})
    if temporal:
        log.info("  temporal: %d event types, dtm_dates=%s",
                 len(temporal.get("events", [])),
                 temporal.get("dtm_dates", []))
    else:
        log.info("  temporal: (empty)")
    pheno = js.get("phenology", {})
    if pheno:
        log.info("  phenology: %s", dict(pheno))
    # Data quality
    dq = js.get("data_quality", {})
    if dq:
        log.info("  quality: %.1f%% grade=%s | available=%s | missing=%s",
                 (dq.get("quality_score", 0)) * 100,
                 dq.get("quality_grade", "?"),
                 ",".join(dq.get("available_layers", [])) or "none",
                 ",".join(dq.get("missing_layers", [])) or "none")


# === SECTION: Zenodo upload helpers ===

def upload_kg_to_zenodo(kg_code: str, kg_name: str, files: dict,
                        manifest, quality_score: float = 0.0,
                        quality_grade: str = "",
                        available_layers: list = None,
                        missing_layers: list = None) -> dict:
    """Upload KG files to Zenodo, verify, delete local GPKGs."""
    from zenodo_client import Client, landscape_metadata

    client = Client(token=ZENODO_TOKEN)
    upload_stats = {"uploaded": [], "errors": [], "total_bytes": 0}

    _avail = available_layers or []
    _miss = missing_layers or []

    for file_key, local_path in files.items():
        if not local_path or not os.path.exists(local_path):
            continue

        zenodo_key = f"{kg_code}_{file_key}"
        file_type = "gpkg" if file_key.endswith("gpkg") else "json"

        meta_func = lambda k, fn, v, _kg=kg_code, _name=kg_name, _ft=file_type, \
            _qs=quality_score, _qg=quality_grade, _al=_avail, _ml=_miss: \
            landscape_metadata(_kg, _name, v, _ft,
                               quality_score=_qs, quality_grade=_qg,
                               available_layers=_al, missing_layers=_ml)

        try:
            fsize = os.path.getsize(local_path)
            client.upload(zenodo_key, local_path, VERSION, meta_func, manifest)

            # Verify
            status = client.head_file(zenodo_key, manifest)
            if status < 200 or status >= 300:
                upload_stats["errors"].append({
                    "key": zenodo_key, "error": f"verify failed: HTTP {status}"})
                continue

            upload_stats["uploaded"].append(zenodo_key)
            upload_stats["total_bytes"] += fsize

            # Delete GPKG locally (keep JSON)
            if file_key != "json" and os.path.exists(local_path):
                os.unlink(local_path)
                log.info("KG %s: deleted local %s after verified upload", kg_code, file_key)

        except Exception as e:
            upload_stats["errors"].append({
                "key": zenodo_key, "error": str(e)})
            log.error("KG %s: Zenodo upload failed for %s: %s",
                      kg_code, file_key, e)

    return upload_stats


# === SECTION: JSON dir cleanup (4GB cap, LRU eviction) ===

def cleanup_json_dir():
    """Delete oldest JSON files if dir exceeds 4GB."""
    total = sum(f.stat().st_size for f in JSON_DIR.glob("*.json") if f.is_file())
    if total <= JSON_DIR_MAX_BYTES:
        return

    files = sorted(JSON_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime)
    while total > JSON_DIR_MAX_BYTES * 0.8 and files:  # trim to 80%
        f = files.pop(0)
        total -= f.stat().st_size
        f.unlink()
        log.info("JSON cleanup: deleted %s", f.name)


def _save_tile_history(kg_code: str, tile_statuses: list, status: str = "completed"):
    """Append completed/failed KG tile positions to tile_history.json."""
    try:
        history = {}
        if TILE_HISTORY_FILE.exists():
            history = json.loads(TILE_HISTORY_FILE.read_text())
        if status == "completed" and tile_statuses:
            history[kg_code] = [[round(t["lat"], 5), round(t["lng"], 5)]
                                for t in tile_statuses if "lat" in t and "lng" in t]
        else:
            history[kg_code] = status  # "failed"
        tmp = str(TILE_HISTORY_FILE) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(history, f)
        os.rename(tmp, str(TILE_HISTORY_FILE))
    except Exception as e:
        log.warning("Failed to save tile history for %s: %s", kg_code, e)


# === SECTION: main() — KG iteration, retry ladder, subprocess management ===

# Per-tile retry ladder: on timeout, sub-divide ONLY the timed-out
# tile into smaller sub-tiles.  Each step is a tile_km for the sub-tiles.
# The first entry (0.5km) is the first sub-division of a 1.5km tile;
# 0.2km is used if a 0.5km sub-tile itself times out.
SUB_TILE_LADDER = [0.5, 0.2]   # km — per-tile subdivision sizes

# Increased timeout: allow extra time for sub-tiled retries.
KG_TIMEOUT_SECONDS_RETRY = 90 * 60   # 90 min for retry runs

RETRIED_KGS_FILE = DATA_DIR / "retried_kgs.json"

# Graceful shutdown flag
_shutdown_requested = False


def _load_failed_kgs() -> set:
    """Load permanently-failed KG codes from file."""
    try:
        if FAILED_KGS_FILE.exists():
            return set(json.loads(FAILED_KGS_FILE.read_text()))
    except Exception:
        pass
    return set()


def _save_failed_kgs(codes: set):
    """Save permanently-failed KG codes."""
    try:
        FAILED_KGS_FILE.write_text(json.dumps(sorted(codes), indent=2))
    except Exception:
        pass


def _load_retried_kgs() -> set:
    """Load KG codes that already got a retry pass (so we don't retry again)."""
    try:
        if RETRIED_KGS_FILE.exists():
            return set(json.loads(RETRIED_KGS_FILE.read_text()))
    except Exception:
        pass
    return set()


def _save_retried_kgs(codes: set):
    try:
        RETRIED_KGS_FILE.write_text(json.dumps(sorted(codes), indent=2))
    except Exception:
        pass


def _copernicus_probe() -> bool:
    """Try a tiny Copernicus request to check if credits are back."""
    try:
        import copernicus
        copernicus.credits_exhausted = False
        copernicus._connection = None
        for k in list(copernicus._connections.keys()):
            copernicus._connections.pop(k, None)
        conn = copernicus._get_connection()
        cube = conn.load_collection(
            'SENTINEL2_L2A',
            spatial_extent={'west': 15, 'south': 47,
                            'east': 15.01, 'north': 47.01},
            temporal_extent=['2024-06-01', '2024-06-15'],
            bands=['B04'],
        )
        cube.max_time().download()
        return True
    except Exception:
        return False


def main():
    global _shutdown_requested

    parser = argparse.ArgumentParser(description="Austria Landscape Processor")
    parser.add_argument("--kg", help="Process single KG code")
    parser.add_argument("--state", help="Process KGs in one state")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry only previously failed KGs")
    parser.add_argument("--no-copernicus", action="store_true",
                        help="Skip Copernicus data (faster)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List KGs without processing")
    parser.add_argument("--rf-only", action="store_true",
                        help="Only emit RF classes; low-confidence segments become 'unclassified'")
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    # --- Signal handling for graceful shutdown ---
    def _handle_signal(signum, frame):
        global _shutdown_requested
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        log.warning("Received %s — will finish current KG then exit", sig_name)
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # --- GDAL config for large raster I/O ---
    # Suppress GDAL 3.11 deprecation warning for 'Memory' driver (used internally by rasterio)
    os.environ.setdefault('GDAL_DEPRECATION_WARNING_THRESHOLD', '99999')
    os.environ.setdefault('GDAL_CACHEMAX', '256')
    os.environ.setdefault('VSI_CACHE', 'TRUE')
    os.environ.setdefault('VSI_CACHE_SIZE', '33554432')  # 32 MB
    os.environ.setdefault('GDAL_HTTP_MULTIPLEX', 'YES')
    os.environ.setdefault('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES')
    os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif,.tiff')

    log.info("=" * 70)
    log.info("🇦🇹 Austria Landscape Processor starting")
    log.info("=" * 70)

    # --- Load/create manifest ---
    from zenodo_client import Manifest
    manifest = Manifest(str(MANIFEST_PATH))
    log.info("Zenodo manifest: %d entries", len(manifest))

    # --- Load progress tracker (reset stale state from previous run) ---
    progress = ProgressTracker(PROGRESS_FILE)
    progress.update(
        failed_kgs=[],
        failed=0,
        completed=0,
        success=0,
        uploaded=0,
        upload_size_bytes=0,
        last_kg_code=None,
        last_kg_seconds=0,
        n_new_buildings_total=0,
        n_infrastructure_total=0,
        parcels_total=0,
        buildings_total=0,
        recent_log=[],
        current_kg=None,
    )
    progress.save()

    # --- Get KG list ---
    if args.kg:
        kgs = []
        try:
            resp = requests.get(f"{CADASTRE_BASE}/search/kg",
                                params={"code": args.kg}, timeout=30)
            resp.raise_for_status()
            kgs = resp.json().get("data", [])
        except Exception as e:
            log.error("Failed to fetch KG %s: %s", args.kg, e)
            return
        if not kgs:
            log.error("KG %s not found", args.kg)
            return
    else:
        kgs = get_all_kgs(state_filter=args.state)

    if not kgs:
        log.error("No KGs to process")
        return

    # --- Filter ---
    if args.retry_failed:
        failed_keys = set()
        for key in manifest.keys():
            if key.endswith("_error"):
                failed_keys.add(key.replace("_error", ""))
        kgs = [kg for kg in kgs if kg["kg_code"] in failed_keys]
        log.info("Retry mode: %d failed KGs to reprocess", len(kgs))

    # --- Determine already-completed KGs (Zenodo manifest + local JSON) ---
    completed_codes = set()
    for key in manifest.keys():
        if key.endswith("_json"):
            completed_codes.add(key.replace("_json", ""))
    # Also count local JSONs not yet uploaded (e.g. upload failed but JSON exists)
    for jf in JSON_DIR.glob("*.json"):
        completed_codes.add(jf.stem)
    log.info("Already completed: %d (manifest + local JSON)", len(completed_codes))

    # --- Load failed KGs + handle crash recovery ---
    failed_kgs = _load_failed_kgs()
    retried_kgs = _load_retried_kgs()

    # On restart, give previously-failed KGs one fresh attempt.
    # KGs that already got a retry pass stay permanently skipped.
    prev_failed = failed_kgs.copy()
    to_retry = prev_failed - retried_kgs
    if to_retry:
        log.info("Clearing %d previously-failed KGs for retry: %s",
                 len(to_retry), sorted(to_retry)[:10])
        retried_kgs |= to_retry
        failed_kgs -= to_retry
        _save_retried_kgs(retried_kgs)
        _save_failed_kgs(failed_kgs)

    # Crash recovery: check IN_PROGRESS_FILE for interrupted KG
    if IN_PROGRESS_FILE.exists():
        interrupted_kg = IN_PROGRESS_FILE.read_text().strip()
        if interrupted_kg:
            log.info("Previous run interrupted during KG %s — will retry "
                     "(not marking as failed)", interrupted_kg)
            # Make sure it's not in the failed set so it gets retried
            failed_kgs.discard(interrupted_kg)
            completed_codes.discard(interrupted_kg)
        IN_PROGRESS_FILE.unlink()

    if failed_kgs:
        log.info("Skipping %d permanently-failed KGs: %s",
                 len(failed_kgs), sorted(failed_kgs)[:20])

    pending = [kg for kg in kgs
               if kg["kg_code"] not in completed_codes
               and kg["kg_code"] not in failed_kgs]

    # Sort by nearest-neighbor traversal for maximum cache reuse
    from tile_cache import order_kgs_nearest_neighbor
    # Resume from last completed KG if available
    resume_from = None
    if completed_codes:
        # Find the last completed KG that has a local JSON
        json_files = sorted(JSON_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime)
        if json_files:
            resume_from = json_files[-1].stem
            log.info("Resuming nearest-neighbor traversal from last KG: %s", resume_from)
    pending = order_kgs_nearest_neighbor(pending, start_code=resume_from)
    log.info("KGs ordered by nearest-neighbor traversal for cache locality")

    log.info("Total KGs: %d, completed: %d, failed (permanent): %d, pending: %d",
             len(kgs), len(completed_codes), len(failed_kgs), len(pending))

    if args.dry_run:
        for kg in pending[:20]:
            log.info("  Would process: %s (%s)", kg["kg_code"], kg.get("kg_name", ""))
        if len(pending) > 20:
            log.info("  ... and %d more", len(pending) - 20)
        return

    progress.update(
        state="running",
        total_kgs=len(kgs),
        completed=len(completed_codes),
        success=len(completed_codes),
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    progress.save()

    t_start = time.time()
    include_cop = not args.no_copernicus
    rf_only = args.rf_only

    for i, kg in enumerate(pending):
        # --- Check graceful shutdown ---
        if _shutdown_requested:
            log.info("Shutdown requested — stopping after %d KGs", i)
            break

        kg_code = kg["kg_code"]
        kg_name = kg.get("kg_name", "")
        kg_state = kg.get("state_name", "")

        log.info("-" * 50)
        log.info("[%d/%d] Processing KG %s (%s, %s)",
                 i + 1, len(pending), kg_code, kg_name, kg_state)

        # Memory check
        gc.collect()
        try:
            rss_mb = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1024
            log.info("  Parent RSS: %.0f MB", rss_mb)
        except Exception:
            pass

        # --- Disk space check before starting KG ---
        if not check_disk_space(kg_code):
            progress.add_log("error", f"Disk critically low — pausing processor", kg_code)
            progress.update(state="paused_disk")
            progress.save()
            log.error("Disk critically low after cleanup — pausing")
            break

        progress.set_current_kg(kg_code, kg_name, kg_state, "starting")
        progress.add_log("info", f"Starting KG {kg_code} ({kg_name})", kg_code)
        progress.save()

        IN_PROGRESS_FILE.write_text(kg_code)

        # Start background thread to monitor subprocess step reporting
        import threading
        _step_monitor_stop = threading.Event()

        def _monitor_step_file(_stop=_step_monitor_stop, _code=kg_code):
            step_file = DATA_DIR / "current_step.json"
            warnings_file = DATA_DIR / "subprocess_warnings.jsonl"
            last_step = ""
            _warn_offset = 0  # track read position in warnings file
            # Track per-step worst issue level: {step: "warning"|"error"}
            _step_issues = {}
            _last_detail = ""  # track detail changes for save
            while not _stop.is_set():
                try:
                    if step_file.exists():
                        sd = json.loads(step_file.read_text())
                        s = sd.get("step", "")
                        detail = sd.get("detail", "")
                        step_changed = bool(s and s != last_step)
                        detail_changed = (detail != _last_detail)
                        # Always merge step_times + tile progress from subprocess
                        sub_times = sd.get("step_times", {})
                        with progress._lock:
                            ckg = progress._state.get("current_kg")
                            if ckg:
                                if sub_times:
                                    ckg["step_times"] = sub_times
                                ct = sd.get("current_tile", 0)
                                nt = sd.get("n_tiles", 0)
                                if nt > 0:
                                    ckg["current_tile"] = ct
                                    ckg["n_tiles"] = nt
                                # Relay sub-tile info for dashboard
                                st_info = sd.get("subtile")
                                if st_info:
                                    ckg["subtile"] = st_info
                                else:
                                    ckg.pop("subtile", None)
                                # Pass detail through for display
                                if detail:
                                    ckg["step_detail"] = detail
                                    if detail_changed:
                                        ckg["step_detail_ts"] = sd.get("ts") or datetime.now(timezone.utc).isoformat()
                                else:
                                    ckg.pop("step_detail", None)
                                    ckg.pop("step_detail_ts", None)
                                # Expose step issues to frontend
                                ckg["step_issues"] = dict(_step_issues)
                                # Relay per-tile statuses for map dots
                                ts_list = sd.get("tile_statuses")
                                if ts_list is not None:
                                    ckg["tile_statuses"] = ts_list
                        if step_changed:
                            prev_step = last_step
                            last_step = s
                            progress.set_step(s, detail)
                            # Log the NEW step starting
                            if detail:
                                progress.add_log("info", f"KG {_code}: {s} \u2014 {detail}", _code)
                            # If the PREVIOUS step had issues, log its completion
                            # at that level so the log matches the tag badge
                            if prev_step and prev_step in _step_issues:
                                issue_lvl = _step_issues[prev_step]
                                step_label = prev_step
                                step_time = sd.get("step_times", {}).get(prev_step)
                                time_str = f" ({step_time}s)" if step_time else ""
                                progress.add_log(
                                    issue_lvl,
                                    f"KG {_code}: {step_label}{time_str} completed with {issue_lvl}s",
                                    _code,
                                )
                        # Save whenever step or detail changed
                        if step_changed or detail_changed:
                            _last_detail = detail
                            progress.save()
                except Exception:
                    pass
                # --- Relay subprocess warnings/errors to dashboard ---
                try:
                    if warnings_file.exists():
                        with open(warnings_file, "r") as wf:
                            wf.seek(_warn_offset)
                            new_lines = wf.readlines()
                            _warn_offset = wf.tell()
                        for line in new_lines:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                                lvl = entry.get("level", "warning")
                                progress.add_log(
                                    lvl,
                                    f"KG {_code}: {entry.get('msg', '')}",
                                    _code,
                                )
                                # Attribute warning/error to the step that emitted it
                                warn_step = entry.get("step") or last_step
                                if warn_step:
                                    cur = _step_issues.get(warn_step, "")
                                    if lvl == "error" or cur != "error":
                                        _step_issues[warn_step] = lvl
                            except json.JSONDecodeError:
                                pass
                        if new_lines:
                            progress.save()
                except Exception:
                    pass
                _stop.wait(2)

        _step_monitor_stop.clear()
        step_thread = threading.Thread(target=_monitor_step_file, daemon=True)
        step_thread.start()

        t_kg = time.time()
        result = None
        kg_succeeded = False

        try:
            # ---- Per-tile retry: on timeout, sub-divide ONLY the
            #      timed-out tile, not the entire KG. ----
            failed_tiles = {}         # orig tile_idx → sub_km
            _tile_ladder_pos = {}     # orig tile_idx → next index in SUB_TILE_LADDER
            # Allow enough attempts: each tile can independently
            # step through the full ladder, plus the initial run.
            max_attempts = len(SUB_TILE_LADDER) * 4 + 1

            for attempt_idx in range(max_attempts):
                if attempt_idx > 0:
                    gc.collect()
                    log.info("  → Retrying KG %s: sub-tiling %d tile(s)",
                             kg_code, len(failed_tiles))
                    progress.add_log("info",
                                     f"Retry KG {kg_code}: sub-tiling {len(failed_tiles)} tile(s)",
                                     kg_code)
                    progress.save()

                # On retries, disable Copernicus for sub-tiles ≤ 0.2km
                # (10m resolution not useful at 200m tiles).
                min_sub_km = min(failed_tiles.values()) if failed_tiles else MAX_KG_AREA_KM
                use_cop = include_cop and min_sub_km >= 0.5

                timeout = KG_TIMEOUT_SECONDS if attempt_idx == 0 else KG_TIMEOUT_SECONDS_RETRY

                pool = multiprocessing.Pool(processes=1)
                try:
                    async_result = pool.apply_async(
                        process_one_kg, args=(kg,),
                        kwds={"include_copernicus": use_cop,
                              "failed_tiles": failed_tiles if failed_tiles else None,
                              "rf_only": rf_only})
                    try:
                        result = async_result.get(timeout=timeout)
                        break  # success — exit retry loop
                    except multiprocessing.TimeoutError:
                        _step_monitor_stop.set()
                        step_thread.join(timeout=3)
                        # Read which tile was in progress from current_step.json
                        last_step = "unknown"
                        timed_out_tile = None  # always an original tile index
                        try:
                            sd = json.loads((DATA_DIR / "current_step.json").read_text())
                            last_step = sd.get("step", "unknown")
                            # If we're already processing sub-tiles, the
                            # subtile field gives us the *original* tile idx.
                            st_info = sd.get("subtile")
                            if st_info and "orig_tile" in st_info:
                                timed_out_tile = st_info["orig_tile"] - 1  # 0-based
                            else:
                                ct = sd.get("current_tile", 0)
                                if ct > 0:
                                    timed_out_tile = ct - 1  # 0-based
                        except Exception:
                            pass
                        pool.terminate()
                        pool.join()

                        # Only sub-tile if the timeout occurred during
                        # actual tile processing.  Post-tile steps
                        # (GPKG build, JSON, validate, upload) should
                        # NOT trigger sub-tiling — just retry the
                        # whole KG with the same tiles.
                        _TILE_STEPS = {"lidar", "terrain", "ortho",
                                       "copernicus", "hansen", "segment",
                                       "vectorise"}
                        _is_tile_step = (last_step in _TILE_STEPS or
                                         last_step.startswith("tile_"))

                        if timed_out_tile is not None and _is_tile_step:
                            # Per-tile ladder: each tile can step
                            # through SUB_TILE_LADDER independently.
                            t_lp = _tile_ladder_pos.get(timed_out_tile, 0)
                            if t_lp >= len(SUB_TILE_LADDER):
                                # This tile has exhausted all sizes
                                log.error("KG %s: TIMEOUT at step %s — tile %d exhausted all retries",
                                          kg_code, last_step, timed_out_tile)
                                failed_kgs.add(kg_code)
                                _save_failed_kgs(failed_kgs)
                                progress.add_failure(kg_code, kg_name,
                                                     f"timeout at {last_step} tile {timed_out_tile} (all retries)",
                                                     last_step)
                                progress.add_log("error",
                                                 f"KG {kg_code} tile {timed_out_tile} timed out after all retries",
                                                 kg_code)
                                progress.save()
                                # Persist tile dots for timed-out KG
                                with progress._lock:
                                    _ckg = progress._state.get("current_kg") or {}
                                    _ts = _ckg.get("tile_statuses", [])
                                _save_tile_history(kg_code, _ts, "failed")
                                result = None
                                break

                            sub_km = SUB_TILE_LADDER[t_lp]
                            _tile_ladder_pos[timed_out_tile] = t_lp + 1

                            # Mark tile for sub-division.  Stale
                            # checkpoint cleanup is handled inside
                            # process_one_kg (section 2c) which has
                            # full knowledge of the old-→new index
                            # mapping.
                            failed_tiles[timed_out_tile] = sub_km
                            log.warning("  → TIMEOUT after %d min at step %s — "
                                        "sub-dividing tile %d at %.1fkm",
                                        timeout // 60, last_step,
                                        timed_out_tile, sub_km)
                        elif not _is_tile_step:
                            # Timeout during post-tile step (GPKG build,
                            # JSON, validate, etc.) — retry the whole KG
                            # with the same tiles, don't sub-tile.
                            log.warning("  → TIMEOUT after %d min during post-tile step '%s' — "
                                        "retrying without sub-tiling",
                                        timeout // 60, last_step)
                            # Don't modify failed_tiles — keep whatever
                            # was there (usually empty on first attempt).
                        else:
                            # Can't identify the tile — treat as permanent
                            log.error("KG %s: TIMEOUT at step %s — can't identify tile",
                                      kg_code, last_step)
                            failed_kgs.add(kg_code)
                            _save_failed_kgs(failed_kgs)
                            progress.add_failure(kg_code, kg_name,
                                                 f"timeout at {last_step} (unknown tile)",
                                                 last_step)
                            progress.add_log("error",
                                             f"KG {kg_code} timed out at {last_step}, can't identify tile",
                                             kg_code)
                            progress.save()
                            # Persist tile dots for timed-out KG
                            with progress._lock:
                                _ckg = progress._state.get("current_kg") or {}
                                _ts = _ckg.get("tile_statuses", [])
                            _save_tile_history(kg_code, _ts, "failed")
                            result = None
                            break

                        # Reset step monitor for next attempt
                        _step_monitor_stop.clear()
                        step_thread = threading.Thread(target=_monitor_step_file, daemon=True)
                        step_thread.start()
                        continue
                finally:
                    pool.close()
                    pool.join()

            # Stop step monitor
            _step_monitor_stop.set()
            step_thread.join(timeout=3)

            if result is None:
                # All retries exhausted or permanent failure — already logged
                pass
            elif result.get("success"):
                elapsed_kg = time.time() - t_kg
                kg_succeeded = True

                # --- Validate outputs ---
                progress.set_step("validate")
                progress.save()
                issues = validate_kg_outputs(kg_code, result["files"])
                if issues:
                    for iss in issues:
                        log.warning("KG %s VALIDATION: %s", kg_code, iss)
                    progress.add_log("warning",
                                     f"KG {kg_code}: {len(issues)} validation issue(s)",
                                     kg_code)
                else:
                    log.info("KG %s: all outputs validated OK", kg_code)

                # --- Log stats from JSON ---
                json_path = result["files"].get("json", "")
                if json_path and os.path.exists(json_path):
                    log_kg_stats_from_json(kg_code, json_path, elapsed_kg)

                # --- Upload to Zenodo ---
                progress.set_step("upload")
                progress.save()

                upload_stats = upload_kg_to_zenodo(
                    kg_code, kg_name, result["files"], manifest,
                    quality_score=result.get("quality_score", 0),
                    quality_grade=result.get("quality_grade", ""),
                    available_layers=result.get("available_layers"),
                    missing_layers=result.get("missing_layers"))

                if upload_stats["errors"]:
                    for err in upload_stats["errors"]:
                        log.warning("KG %s upload error: %s", kg_code, err)

                progress.record_success(
                    parcels=result.get("n_parcels", 0),
                    buildings=result.get("n_buildings", 0),
                    upload_bytes=upload_stats["total_bytes"],
                    last_kg_code=kg_code,
                    last_kg_seconds=elapsed_kg,
                    n_new_buildings=result.get("n_new_buildings", 0),
                    n_infrastructure=result.get("n_infrastructure", 0),
                )
                _qg = result.get('quality_grade', '')
                _qs = result.get('quality_score', 0)
                progress.add_log(
                    "success",
                    f"KG {kg_code} done in {elapsed_kg:.0f}s "
                    f"({result.get('n_segments', 0)} segs, "
                    f"{result.get('n_parcels', 0)} par, "
                    f"{result.get('n_buildings', 0)} bldg, "
                    f"quality={_qs:.0%} {_qg})",
                    kg_code,
                )
                # Persist tile dots for completed KG
                with progress._lock:
                    _ckg = progress._state.get("current_kg") or {}
                    _ts = _ckg.get("tile_statuses", [])
                _save_tile_history(kg_code, _ts, "completed")
            else:
                # --- Copernicus credits exhausted? Don't mark as permanently failed ---
                is_credits_issue = (result.get("copernicus_exhausted")
                                    or '402' in str(result.get("error", ""))
                                    or 'PaymentRequired' in str(result.get("error", "")))
                if is_credits_issue:
                    log.warning("KG %s: Copernicus credits issue — will retry after credits restored", kg_code)
                    progress.add_log("warning",
                                     f"KG {kg_code}: credits exhausted — not marking failed",
                                     kg_code)
                else:
                    progress.add_failure(
                        kg_code, kg_name,
                        result.get("error", "unknown"),
                        result.get("step", "unknown"),
                    )
                    progress.add_log(
                        "error",
                        f"KG {kg_code} failed at {result.get('step')}: {result.get('error')}",
                        kg_code,
                    )
                    # Record failure in manifest for retry tracking
                    from zenodo_client import Entry
                    manifest.set(f"{kg_code}_error", Entry(
                        key=f"{kg_code}_error",
                        depo_id=0, bucket_url="", filename="",
                        size=0, checksum="",
                        uploaded_at=datetime.now(timezone.utc).isoformat(),
                        version=json.dumps({
                            "error": result.get("error", ""),
                            "step": result.get("step", ""),
                            "traceback": result.get("traceback", "")[:500],
                        }),
                    ))
                    manifest.save()

                    log.warning("KG %s: FAILED at %s: %s",
                                kg_code, result.get("step"), result.get("error"))
                    # Persist tile dots for failed KG
                    with progress._lock:
                        _ckg = progress._state.get("current_kg") or {}
                        _ts = _ckg.get("tile_statuses", [])
                    _save_tile_history(kg_code, _ts, "failed")

        except Exception as e:
            progress.add_failure(kg_code, kg_name, str(e), "exception")
            progress.add_log("error", f"KG {kg_code} exception: {e}", kg_code)
            log.error("KG %s: EXCEPTION: %s", kg_code, traceback.format_exc())
            # Persist tile dots for exceptioned KG
            with progress._lock:
                _ckg = progress._state.get("current_kg") or {}
                _ts = _ckg.get("tile_statuses", [])
            _save_tile_history(kg_code, _ts, "failed")

        # Cleanup
        if IN_PROGRESS_FILE.exists():
            IN_PROGRESS_FILE.unlink()

        progress.update_rates(t_start)
        progress.update_system_metrics()
        progress.save()
        cleanup_json_dir()

        gc.collect()

        # Explicitly free large subprocess data
        del result
        gc.collect()

        # --- Copernicus pause: auto-probe every 15 min instead of just waiting ---
        if COPERNICUS_PAUSE_FILE.exists():
            log.warning("⏸ Copernicus credits exhausted — PAUSING.")
            progress.update(state="paused_copernicus")
            progress.add_log("warning",
                             "Paused: Copernicus credits exhausted. "
                             "Will auto-probe every 15 min or delete pause file to resume.", "")
            progress.save()

            # Try credential rotation first
            try:
                import copernicus
                if copernicus.rotate_credentials():
                    log.info("Rotated to next Copernicus credential set")
            except Exception:
                pass

            probe_count = 0
            while COPERNICUS_PAUSE_FILE.exists():
                probe_count += 1
                time.sleep(900)  # 15 min
                if _shutdown_requested:
                    break
                log.info("Credits paused — probe #%d: testing Copernicus...", probe_count)
                if _copernicus_probe():
                    log.info("Credits restored! Removing pause file and resuming.")
                    try:
                        COPERNICUS_PAUSE_FILE.unlink()
                    except Exception:
                        pass
                    break
                else:
                    log.info("Still no credits — will retry in 15 min")
                    # Try rotating to another credential on each probe
                    try:
                        import copernicus
                        copernicus.rotate_credentials()
                    except Exception:
                        pass

            # Reset copernicus module for fresh connections
            try:
                import copernicus
                copernicus.credits_exhausted = False
                copernicus._connection = None
                for k in list(copernicus._connections.keys()):
                    copernicus._connections.pop(k, None)
            except Exception:
                pass
            log.info("▶ Copernicus resumed.")
            progress.update(state="running")
            progress.add_log("info", "Resumed after Copernicus pause", "")
            progress.save()

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            s = progress.get()
            log.info("Progress: %d/%d done (%d success, %d fail), "
                     "%.1f KGs/hr, ETA %.1fh",
                     s["completed"], s["total_kgs"],
                     s["success"], s["failed"],
                     s["rate_kgs_per_hour"],
                     s["eta_seconds"] / 3600 if s["eta_seconds"] else 0)

    # Done
    if _shutdown_requested:
        progress.update(state="stopped", current_kg=None)
        log.info("Graceful shutdown complete.")
    else:
        progress.update(state="complete", current_kg=None)
    progress.save()

    elapsed = time.time() - t_start
    s = progress.get()
    log.info("=" * 70)
    log.info("Processing %s: %d success, %d failed in %.1f hours",
             "stopped" if _shutdown_requested else "complete",
             s["success"], s["failed"], elapsed / 3600)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
