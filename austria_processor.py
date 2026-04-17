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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CADASTRE_BASE = "https://cadastre-process-api.exe.xyz/api/v1"
ZENODO_TOKEN = "2dnLSA2YYTc8jt3a1X0qDZUBb1hyOIpGJ44UoJr8N69wdePODgq4cjbJ0DJa"
VERSION = "v1"

DATA_DIR = Path("data/austria_processor")
MANIFEST_PATH = DATA_DIR / "zenodo_manifest.json"
JSON_DIR = DATA_DIR / "json"
GPKG_DIR = Path("/tmp/austria_processor/gpkg")
LOG_DIR = DATA_DIR / "logs"
PROGRESS_FILE = DATA_DIR / "progress.json"
KG_LIST_FILE = DATA_DIR / "kg_list.json"
FAILED_KGS_FILE = DATA_DIR / "failed_kgs.json"
IN_PROGRESS_FILE = DATA_DIR / "in_progress_kg.txt"
CIRCUIT_BREAKER_FILE = DATA_DIR / "openeo_circuit.json"
COPERNICUS_PAUSE_FILE = DATA_DIR / "copernicus_paused"

MAX_KG_PIXELS = 15_000_000
KG_TIMEOUT_SECONDS = 30 * 60
JSON_DIR_MAX_BYTES = 4 * 1024 ** 3  # 4GB
MAX_KG_AREA_KM = 3.0  # crop KG bbox if wider

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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    stream=sys.stdout,
)
log = logging.getLogger('austria_processor')
for name in ['rasterio', 'urllib3', 'botocore', 'PIL', 'fiona']:
    logging.getLogger(name).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Ensure directories
# ---------------------------------------------------------------------------

for d in [DATA_DIR, JSON_DIR, GPKG_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Progress tracking (written to disk, read by /api/v1/processing/status)
# ---------------------------------------------------------------------------

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

    def set_step(self, step: str):
        with self._lock:
            if self._state["current_kg"]:
                self._state["current_kg"]["step"] = step

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

    def record_success(self, parcels: int = 0, buildings: int = 0, upload_bytes: int = 0):
        with self._lock:
            self._state["success"] += 1
            self._state["completed"] += 1
            self._state["uploaded"] += 1
            self._state["upload_size_bytes"] += upload_bytes
            self._state["parcels_total"] += parcels
            self._state["buildings_total"] += buildings

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


# ---------------------------------------------------------------------------
# Circuit breaker (same pattern as rf_train)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def transform_to_3035(geom):
    return shapely_transform(_tx_to_3035.transform, geom)

def transform_to_wgs(geom):
    return shapely_transform(_tx_to_wgs.transform, geom)


# ---------------------------------------------------------------------------
# KG list
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cadastre data fetching
# ---------------------------------------------------------------------------

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

    log.info("KG %s: %d parcels, %d footprints, %d landuse",
             kg_code, len(result["parcels"]), len(result["building_footprints"]),
             len(result["landuse"]))
    return result


# ---------------------------------------------------------------------------
# Height enrichment for parcels and buildings
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Vectorise segments not matched to cadastre (new buildings, infrastructure)
# ---------------------------------------------------------------------------

def vectorise_unmatched_buildings(objects: list, labels: np.ndarray,
                                  mask: np.ndarray, transform,
                                  cadastre_fp_mask: np.ndarray,
                                  ndsm: np.ndarray) -> list[dict]:
    """Find roof/building segments that don't overlap cadastre footprints."""
    from rasterio.features import shapes as rasterize_shapes

    results = []
    obj_map = {o.obj_id: o for o in objects}

    # Segments classified as roof/building
    building_objs = [o for o in objects if o.obj_type in ('roof', 'greenhouse', 'solar_panel')]
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
            poly_wgs = transform_to_wgs(poly_3035)
            c_wgs = poly_wgs.centroid

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
        if obj.area_sqm < spec['min_area']:
            continue
        if obj.obj_type == 'tree_loss' and obj.area_sqm < 5000:
            continue  # clear cuts > 0.5ha only

        seg_mask = labels == obj.obj_id
        if not seg_mask.any():
            continue

        try:
            label_single = np.where(seg_mask, 1, 0).astype(np.int32)
            for geom_dict, val in rasterize_shapes(
                label_single, mask=seg_mask, transform=transform, connectivity=4,
            ):
                if val == 0:
                    continue
                from shapely.geometry import shape as s_shape
                poly_3035 = s_shape(geom_dict)
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
                }

                if obj.obj_type == 'parking':
                    # ~12.5 m2 per parking spot (standard)
                    feature["est_parking_spots"] = max(1, round(poly_3035.area / 12.5))

                if obj.obj_type in ('excavation', 'fill'):
                    # Volume estimate: mean absolute height change * area
                    mean_h = abs(float(np.nanmean(seg_h))) if len(seg_h) > 0 else 0
                    volume = mean_h * poly_3035.area
                    feature["volume_m3"] = round(volume, 1)
                    if volume < 10:
                        continue  # skip tiny earthworks

                if obj.obj_type in ('bridge', 'mast'):
                    feature["max_height_m"] = round(float(np.nanmax(seg_h)), 2) if len(seg_h) > 0 else 0

                if obj.obj_type in ('fence', 'wall'):
                    # Length estimate from perimeter / 2 (rectangular assumption)
                    feature["length_m"] = round(poly_3035.length / 2, 1)
                    feature["max_height_m"] = round(float(np.nanmax(seg_h)), 2) if len(seg_h) > 0 else 0

                results.append(feature)
                break  # one polygon per object
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# GPKG builders
# ---------------------------------------------------------------------------

def build_full_gpkg(kg_code: str, data: dict, spectral: dict,
                    labels: np.ndarray, objects: list,
                    mask: np.ndarray, transform) -> str:
    """Build full GeoPackage with all raster layers."""
    import rasterio

    out_path = str(GPKG_DIR / f"{kg_code}_full.gpkg")
    if os.path.exists(out_path):
        os.unlink(out_path)

    h, w = data["shape"]
    dtm = data["dtm"]
    dsm = data["dsm"]
    ndsm = data["ndsm"]
    table_count = 0

    def _write_table(name, arrays, dtype='float32', descriptions=None):
        nonlocal table_count
        opts = dict(
            driver='GPKG', width=w, height=h, count=len(arrays),
            dtype=dtype, crs='EPSG:3035', transform=transform,
            RASTER_TABLE=name, RASTER_IDENTIFIER=name,
        )
        if dtype == 'float32':
            opts['nodata'] = float('nan')
        if table_count > 0:
            opts['APPEND_SUBDATASET'] = 'YES'
        with rasterio.open(out_path, 'w', **opts) as dst:
            for i, arr in enumerate(arrays, 1):
                out = arr[:h, :w] if arr.shape[0] >= h and arr.shape[1] >= w else arr
                dst.write(out, i)
                if descriptions and i <= len(descriptions):
                    dst.set_band_description(i, descriptions[i - 1])
        table_count += 1

    # Core DTM/DSM/nDSM
    _write_table('DTM', [dtm.astype(np.float32)])
    _write_table('DSM', [dsm.astype(np.float32)])
    _write_table('nDSM', [ndsm.astype(np.float32)])

    # Ortho
    if spectral is not None:
        bands = []
        descs = []
        for ch in ['red', 'green', 'blue']:
            if ch in spectral:
                bands.append(spectral[ch].astype(np.uint8) if spectral[ch].max() > 1
                             else (spectral[ch] * 255).astype(np.uint8))
                descs.append(ch.capitalize())
        if 'nir' in spectral and spectral['nir'] is not None:
            nir = spectral['nir']
            bands.append(nir.astype(np.uint8) if nir.max() > 1
                         else (nir * 255).astype(np.uint8))
            descs.append('NIR')
        if bands:
            _write_table('Ortho', bands, dtype='uint8', descriptions=descs)
            # CIR = NIR, Red, Green
            if len(bands) >= 4:
                _write_table('CIR', [bands[3], bands[0], bands[1]],
                             dtype='uint8', descriptions=['NIR', 'Red', 'Green'])

    # Segmentation rasters
    if labels is not None and objects:
        obj_map = {o.obj_id: o for o in objects}
        type_raster = np.zeros((h, w), dtype=np.uint8)
        height_raster = np.clip(ndsm, 0, 255).astype(np.float32)
        for obj in objects:
            type_raster[labels == obj.obj_id] = obj.type_code
        _write_table('segment_type', [type_raster], dtype='uint8',
                     descriptions=['Object type code'])
        _write_table('segment_height', [height_raster],
                     descriptions=['Object height (m)'])

    return out_path


def build_light_gpkg(kg_code: str, data: dict, labels: np.ndarray,
                     objects: list, mask: np.ndarray, transform,
                     cadastre_data: dict, ndsm: np.ndarray,
                     new_buildings: list, infrastructure: list) -> str:
    """Build lightweight GeoPackage with segmentation + enriched cadastre."""
    import rasterio
    import fiona
    from fiona.crs import from_epsg

    out_path = str(GPKG_DIR / f"{kg_code}_light.gpkg")
    if os.path.exists(out_path):
        os.unlink(out_path)

    h, w = data["shape"]
    table_count = 0

    def _write_raster(name, arrays, dtype='uint8', descriptions=None):
        nonlocal table_count
        opts = dict(
            driver='GPKG', width=w, height=h, count=len(arrays),
            dtype=dtype, crs='EPSG:3035', transform=transform,
            RASTER_TABLE=name, RASTER_IDENTIFIER=name,
        )
        if dtype == 'float32':
            opts['nodata'] = float('nan')
        if table_count > 0:
            opts['APPEND_SUBDATASET'] = 'YES'
        with rasterio.open(out_path, 'w', **opts) as dst:
            for i, arr in enumerate(arrays, 1):
                out = arr[:h, :w]
                dst.write(out, i)
                if descriptions and i <= len(descriptions):
                    dst.set_band_description(i, descriptions[i - 1])
        table_count += 1

    # Segmentation raster
    if labels is not None and objects:
        obj_map = {o.obj_id: o for o in objects}
        type_raster = np.zeros((h, w), dtype=np.uint8)
        for obj in objects:
            type_raster[labels == obj.obj_id] = obj.type_code
        _write_raster('segment_type', [type_raster], descriptions=['Object type code'])

    # Segmentation vector
    if labels is not None and objects:
        from rasterio.features import shapes as rasterize_shapes
        obj_map = {o.obj_id: o for o in objects}
        label_int = labels.astype(np.int32)

        schema = {
            'geometry': 'Polygon',
            'properties': [
                ('id', 'int'), ('type', 'str'), ('group_type', 'str'),
                ('height_max_m', 'float'), ('height_mean_m', 'float'),
                ('area_sqm', 'float'), ('confidence', 'float'),
                ('ndvi_mean', 'float'),
            ],
        }
        with fiona.open(out_path, 'w', driver='GPKG', layer='segments',
                        schema=schema, crs=from_epsg(3035)) as dst:
            for geom_dict, val in rasterize_shapes(
                label_int, mask=mask, transform=transform, connectivity=4,
            ):
                oid = int(val)
                obj = obj_map.get(oid)
                if obj is None:
                    continue
                dst.write({
                    'geometry': geom_dict,
                    'properties': {
                        'id': oid, 'type': obj.obj_type,
                        'group_type': obj.group_type or '',
                        'height_max_m': round(obj.height_max, 2),
                        'height_mean_m': round(obj.height_mean, 2),
                        'area_sqm': round(obj.area_sqm, 1),
                        'confidence': round(obj.confidence, 2),
                        'ndvi_mean': round(obj.ndvi_mean, 3) if obj.ndvi_mean else 0.0,
                    },
                })

    # Parcels with heights
    dtm = data["dtm"]
    enriched_parcels = enrich_parcels_with_heights(
        cadastre_data["parcels"], dtm, transform)
    if enriched_parcels:
        schema_p = {
            'geometry': 'Polygon',
            'properties': [
                ('parcel_id', 'str'), ('area_sqm', 'float'),
                ('centroid_dtm_m', 'float'),
                ('centroid_lon', 'float'), ('centroid_lat', 'float'),
            ],
        }
        with fiona.open(out_path, 'w', driver='GPKG', layer='parcels',
                        schema=schema_p, crs=from_epsg(4326)) as dst:
            for ep in enriched_parcels:
                geom_wgs = ep["geometry_wgs"]
                p = ep["properties"]
                dst.write({
                    'geometry': mapping(geom_wgs),
                    'properties': {
                        'parcel_id': ep["parcel_id"],
                        'area_sqm': ep["area_sqm"],
                        'centroid_dtm_m': p.get('centroid_dtm_m'),
                        'centroid_lon': p.get('centroid_lon'),
                        'centroid_lat': p.get('centroid_lat'),
                    },
                })

    # Building footprints with heights
    dsm = data["dsm"]
    enriched_bldgs = enrich_buildings_with_heights(
        cadastre_data["building_footprints"], dtm, dsm, ndsm, transform)
    if enriched_bldgs:
        schema_b = {
            'geometry': 'Polygon',
            'properties': [
                ('max_height_m', 'float'), ('mean_height_m', 'float'),
                ('dsm_std', 'float'), ('roof_type_hint', 'str'),
                ('stories_est', 'int'), ('footprint_area_sqm', 'float'),
                ('centroid_lon', 'float'), ('centroid_lat', 'float'),
            ],
        }
        with fiona.open(out_path, 'w', driver='GPKG', layer='buildings',
                        schema=schema_b, crs=from_epsg(4326)) as dst:
            for eb in enriched_bldgs:
                p = eb["properties"]
                dst.write({
                    'geometry': mapping(eb["geometry_wgs"]),
                    'properties': {
                        'max_height_m': p.get('max_height_m'),
                        'mean_height_m': p.get('mean_height_m'),
                        'dsm_std': p.get('dsm_std'),
                        'roof_type_hint': p.get('roof_type_hint', ''),
                        'stories_est': p.get('stories_est'),
                        'footprint_area_sqm': p.get('footprint_area_sqm'),
                        'centroid_lon': p.get('centroid_lon'),
                        'centroid_lat': p.get('centroid_lat'),
                    },
                })

    # New buildings (not in cadastre)
    if new_buildings:
        schema_nb = {
            'geometry': 'Polygon',
            'properties': [
                ('type', 'str'), ('area_sqm', 'float'),
                ('max_height_m', 'float'), ('stories_est', 'int'),
                ('roof_type_hint', 'str'), ('confidence', 'float'),
                ('centroid_lon', 'float'), ('centroid_lat', 'float'),
            ],
        }
        with fiona.open(out_path, 'w', driver='GPKG', layer='new_buildings',
                        schema=schema_nb, crs=from_epsg(4326)) as dst:
            for nb in new_buildings:
                dst.write({
                    'geometry': nb["geometry_wgs"],
                    'properties': {
                        'type': nb.get('type', 'roof'),
                        'area_sqm': nb.get('area_sqm', 0),
                        'max_height_m': nb.get('max_height_m', 0),
                        'stories_est': nb.get('stories_est', 1),
                        'roof_type_hint': nb.get('roof_type_hint', ''),
                        'confidence': nb.get('confidence', 0),
                        'centroid_lon': nb.get('centroid_lon'),
                        'centroid_lat': nb.get('centroid_lat'),
                    },
                })

    # Infrastructure
    if infrastructure:
        schema_infra = {
            'geometry': 'Polygon',
            'properties': [
                ('type', 'str'), ('area_sqm', 'float'),
                ('volume_m3', 'float'), ('max_height_m', 'float'),
                ('est_parking_spots', 'int'), ('confidence', 'float'),
                ('centroid_lon', 'float'), ('centroid_lat', 'float'),
            ],
        }
        with fiona.open(out_path, 'w', driver='GPKG', layer='infrastructure',
                        schema=schema_infra, crs=from_epsg(4326)) as dst:
            for inf in infrastructure:
                dst.write({
                    'geometry': inf["geometry_wgs"],
                    'properties': {
                        'type': inf.get('type', ''),
                        'area_sqm': inf.get('area_sqm', 0),
                        'volume_m3': inf.get('volume_m3'),
                        'max_height_m': inf.get('max_height_m'),
                        'est_parking_spots': inf.get('est_parking_spots'),
                        'confidence': inf.get('confidence', 0),
                        'centroid_lon': inf.get('centroid_lon'),
                        'centroid_lat': inf.get('centroid_lat'),
                    },
                })

    return out_path


# ---------------------------------------------------------------------------
# JSON summary builder
# ---------------------------------------------------------------------------

def build_json_summary(kg_code: str, kg_info: dict, data: dict,
                       labels: np.ndarray, objects: list,
                       cadastre_data: dict, terrain_stats: dict,
                       spectral: dict, hansen_data: dict,
                       copernicus_data: dict, new_buildings: list,
                       infrastructure: list, ndsm: np.ndarray,
                       obs_year: int) -> dict:
    """Build comprehensive JSON summary for a KG."""
    import tile_index as ti

    mask = data["mask"]
    dtm = data["dtm"]
    h, w = data["shape"]

    # --- KG info ---
    summary = {
        "version": VERSION,
        "kg_code": kg_code,
        "kg_name": kg_info.get("kg_name", ""),
        "state": kg_info.get("state_name", ""),
        "gemeinde": kg_info.get("gemeinde_name", ""),
        "district": kg_info.get("district_name", ""),
        "bbox": kg_info.get("bbox", {}),
        "total_area_sqm": int(mask.sum()),  # valid pixels = m2 at 1m res
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observation_period": {
            "start": f"{obs_year}-01-01",
            "end": f"{obs_year}-12-31",
            "lidar_dataset": ti.DEFAULT_DATASET,
        },
    }

    # --- Area summary by type ---
    type_counts = Counter()
    type_heights = defaultdict(list)
    if objects:
        obj_map = {o.obj_id: o for o in objects}
        for obj in objects:
            seg_px = int((labels == obj.obj_id).sum())
            type_counts[obj.obj_type] += seg_px
            type_heights[obj.obj_type].append({
                "h_max": round(obj.height_max, 2),
                "h_mean": round(obj.height_mean, 2),
            })

    area_summary = {}
    for t, px in type_counts.most_common():
        heights = [h["h_max"] for h in type_heights[t]]
        area_summary[t] = {
            "pixels": px,
            "area_sqm": px,  # 1m resolution → 1 pixel = 1 m²
            "fraction": round(px / max(int(mask.sum()), 1), 4),
            "n_objects": len(type_heights[t]),
            "observation_period": f"{obs_year}-01-01 to {obs_year}-12-31",
        }
    summary["area_summary"] = area_summary

    # --- Height distribution per type ---
    height_dist = {}
    for t, hlist in type_heights.items():
        heights = [h["h_max"] for h in hlist]
        if heights:
            height_dist[t] = {
                "min": round(min(heights), 2),
                "max": round(max(heights), 2),
                "mean": round(sum(heights) / len(heights), 2),
                "p90": round(float(np.percentile(heights, 90)), 2),
                "count": len(heights),
            }
    summary["height_distribution"] = height_dist

    # --- Landscape characterisation ---
    landscape = {}
    if terrain_stats:
        landscape["terrain"] = terrain_stats

    # Fragmentation: edge density (perimeter of all segments / total area)
    if objects:
        total_perimeter = sum(getattr(o, 'perimeter', 0) or 0 for o in objects)
        total_area = max(int(mask.sum()), 1)
        landscape["edge_density"] = round(total_perimeter / total_area, 4)
        landscape["n_segments"] = len(objects)
        landscape["mean_segment_area_sqm"] = round(total_area / len(objects), 1)

        # Shannon diversity of types
        type_fracs = [c / total_area for c in type_counts.values()]
        shannon = -sum(f * np.log(f) for f in type_fracs if f > 0)
        landscape["shannon_diversity"] = round(float(shannon), 3)

        # Dominant type
        if type_counts:
            landscape["dominant_type"] = type_counts.most_common(1)[0][0]

        # % vegetated
        veg_types = {'tree', 'shrub', 'grass', 'hedge', 'crop', 'orchard',
                     'vineyard', 'garden'}
        veg_px = sum(v for k, v in type_counts.items() if k in veg_types)
        landscape["vegetated_fraction"] = round(veg_px / total_area, 4)
        landscape["is_vegetated"] = veg_px / total_area > 0.5

    summary["landscape"] = landscape

    # --- Top 10 highest objects ---
    if objects:
        sorted_by_h = sorted(objects, key=lambda o: o.height_max, reverse=True)[:10]
        summary["top_10_objects"] = []
        for o in sorted_by_h:
            c_wgs = None
            try:
                ce, cn = o.centroid_e, o.centroid_n
                lon, lat = _tx_to_wgs.transform(ce, cn)
                c_wgs = {"lon": round(lon, 7), "lat": round(lat, 7)}
            except Exception:
                pass
            summary["top_10_objects"].append({
                "type": o.obj_type,
                "height_m": round(o.height_max, 2),
                "coordinate": c_wgs,
            })

    # --- Top 10 tallest trees ---
    if objects:
        trees = [o for o in objects if o.obj_type == 'tree']
        sorted_trees = sorted(trees, key=lambda o: o.height_max, reverse=True)[:10]
        summary["top_10_trees"] = []
        for t in sorted_trees:
            c_wgs = None
            try:
                lon, lat = _tx_to_wgs.transform(t.centroid_e, t.centroid_n)
                c_wgs = {"lon": round(lon, 7), "lat": round(lat, 7)}
            except Exception:
                pass
            summary["top_10_trees"].append({
                "height_m": round(t.height_max, 2),
                "canopy_height_m": round(t.height_mean, 2),
                "coordinate": c_wgs,
                "area_sqm": round(t.area_sqm, 1),
            })

        # Tree stats
        if trees:
            summary["tree_stats"] = {
                "count": len(trees),
                "total_canopy_sqm": round(sum(t.area_sqm for t in trees), 1),
                "mean_height_m": round(sum(t.height_max for t in trees) / len(trees), 2),
                "est_stem_volume_m3": round(
                    sum(0.3 * t.area_sqm * t.height_max / 3 for t in trees), 1
                ),  # rough cone estimate
            }

    # --- Terrain ---
    summary["terrain"] = {}
    if terrain_stats:
        ts = terrain_stats
        summary["terrain"] = {
            "steepness_mean_deg": ts.get("slope_mean_deg"),
            "steepness_max_deg": ts.get("slope_max_deg"),
            "aspect_dominant": ts.get("dominant_aspect", ""),
            "roughness_mean": ts.get("roughness_mean"),
            "curvature_mean": ts.get("curvature_mean"),
            "elevation_min_m": ts.get("elevation_min_m"),
            "elevation_max_m": ts.get("elevation_max_m"),
            "elevation_range_m": ts.get("elevation_range_m"),
            "method": "DTM Sobel slope + aspect, BEV ALS 1m",
        }

    # --- NDVI ---
    ndvi_info = {}
    if spectral and spectral.get("ndvi") is not None:
        ndvi = spectral["ndvi"]
        valid = ndvi[mask[:ndvi.shape[0], :ndvi.shape[1]]] if mask is not None else ndvi.ravel()
        valid = valid[np.isfinite(valid)]
        if len(valid) > 0:
            ndvi_info["bev_nir_mean"] = round(float(np.nanmean(valid)), 4)
            ndvi_info["bev_nir_std"] = round(float(np.nanstd(valid)), 4)
            ndvi_info["method_bev"] = f"BEV DOP RGBI 0.2m, {obs_year}"
    if copernicus_data and copernicus_data.get("ndvi") is not None:
        cop_ndvi = copernicus_data["ndvi"]
        valid_c = cop_ndvi[np.isfinite(cop_ndvi)]
        if len(valid_c) > 0:
            ndvi_info["copernicus_mean"] = round(float(np.nanmean(valid_c)), 4)
            ndvi_info["method_copernicus"] = f"Sentinel-2 L2A NDVI composite, {obs_year}"
    summary["ndvi"] = ndvi_info

    # --- Hansen forest loss ---
    hansen_summary = {}
    if hansen_data:
        loss_year = hansen_data.get("loss_year")
        if loss_year is not None:
            per_year = {}
            for yr in range(1, 25):  # 1=2001 ... 24=2024
                n = int((loss_year == yr).sum())
                if n > 0:
                    per_year[f"{2000 + yr}"] = {"pixels": n, "area_sqm": n * 900}  # 30m res
            hansen_summary["loss_by_year"] = per_year
            hansen_summary["total_loss_pixels"] = int((loss_year > 0).sum())
        tc2000 = hansen_data.get("treecover2000")
        if tc2000 is not None:
            hansen_summary["mean_treecover2000_pct"] = round(float(np.nanmean(tc2000)), 1)
        cf = hansen_data.get("current_forest")
        if cf is not None:
            hansen_summary["current_forest_pixels"] = int(cf.sum())
        hansen_summary["method"] = "Hansen GFC-2024-v1.12, 30m, University of Maryland"
    summary["hansen"] = hansen_summary

    # --- New buildings ---
    summary["new_buildings"] = {
        "count": len(new_buildings),
        "features": [{
            "type": nb.get("type"),
            "area_sqm": nb.get("area_sqm"),
            "max_height_m": nb.get("max_height_m"),
            "stories_est": nb.get("stories_est"),
            "roof_type_hint": nb.get("roof_type_hint"),
            "centroid_lon": nb.get("centroid_lon"),
            "centroid_lat": nb.get("centroid_lat"),
            "confidence": nb.get("confidence"),
        } for nb in new_buildings],
    }

    # --- Infrastructure ---
    infra_by_type = defaultdict(list)
    for inf in infrastructure:
        infra_by_type[inf["type"]].append(inf)
    summary["infrastructure"] = {
        "total": len(infrastructure),
        "by_type": {
            t: {
                "count": len(items),
                "total_area_sqm": round(sum(i.get("area_sqm", 0) for i in items), 1),
                "features": [{
                    k: v for k, v in i.items()
                    if k != "geometry_wgs"
                } for i in items],
            }
            for t, items in infra_by_type.items()
        },
    }

    # --- Parcels summary ---
    summary["parcels"] = {
        "count": len(cadastre_data["parcels"]),
        "total_area_sqm": round(sum(p.get("area_sqm", 0) for p in cadastre_data["parcels"]), 1),
    }

    # --- Methods ---
    summary["methods"] = {
        "segmentation": "Felzenszwalb over-segmentation + RAG merge on fused gradient (DTM+DSM+RGBI+NDVI), 1m resolution",
        "classification": "Random Forest (44 features, cadastre+OSM trained) with rule-based fallback",
        "height": "BEV ALS DTM/DSM 1m, nDSM = DSM - DTM",
        "ortho": "BEV DOP RGBI 0.2m, resampled to 1m for spectral indices",
        "ndvi_bev": "(NIR - Red) / (NIR + Red) from BEV DOP RGBI",
        "ndvi_copernicus": "Sentinel-2 L2A B08/B04, openEO, 10m resampled to 1m",
        "terrain": "Slope (Sobel), aspect, TRI, TPI, curvature from DTM",
        "hansen": "Hansen GFC-2024-v1.12, 30m, treecover2000 + lossyear + gain",
        "roof_type": "flat = DSM std < 1.5m within footprint, pitched otherwise",
        "stories_est": "max_object_height / 3m, rounded",
        "stem_volume": "Rough cone estimate: 0.3 * canopy_area * height / 3",
        "parking_spots": "Area / 12.5 m² (standard parking spot size)",
        "earthwork_volume": "mean(|nDSM|) * polygon_area",
        "fragmentation": "Shannon diversity index on segment type fractions",
        "edge_density": "Total segment perimeter / total area",
        "cadastre_source": "BEV INSPIRE cadastre via cadastre-process-api.exe.xyz",
        "data_sources": [
            "BEV ALS DTM/DSM 1m (2022-2024)",
            "BEV DOP RGBI 0.2m (2022-2024)",
            "Sentinel-2 L2A 10m (openEO)",
            "ESA WorldCover 10m",
            "Sentinel-1 SAR 10m (openEO)",
            "Hansen GFC-2024-v1.12 30m",
            "Austrian Cadastre (BEV INSPIRE)",
        ],
    }

    return summary


# ---------------------------------------------------------------------------
# Core per-KG processing (runs in subprocess)
# ---------------------------------------------------------------------------

def process_one_kg(kg: dict, include_copernicus: bool = True) -> dict:
    """Process a single KG. Returns dict with file paths + stats.

    This function runs in a subprocess for memory isolation.
    """
    import raster_io
    import tile_index as ti
    import object_segmentation as oc
    import terrain_analysis as ta

    kg_code = kg["kg_code"]
    result = {"kg_code": kg_code, "success": False, "step": "init", "files": {}}

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

        # Limit size
        dx_km = (east - west) * 111 * np.cos(np.radians((south + north) / 2))
        dy_km = (north - south) * 111
        if dx_km > MAX_KG_AREA_KM or dy_km > MAX_KG_AREA_KM:
            cx, cy = (west + east) / 2, (south + north) / 2
            half = (MAX_KG_AREA_KM / 2) / 111
            west, south, east, north = cx - half, cy - half, cx + half, cy + half
            log.info("KG %s: cropped to %.1fkm window", kg_code, MAX_KG_AREA_KM)

        geom_wgs = box(west, south, east, north)
        geom_3035 = transform_to_3035(geom_wgs)
        obs_year = ti.dataset_to_year(ti.DEFAULT_DATASET)

        # --- 1. Cadastre ---
        result["step"] = "cadastre"
        cadastre_data = fetch_cadastre_data(kg_code)
        result["n_parcels"] = len(cadastre_data["parcels"])
        result["n_buildings"] = len(cadastre_data["building_footprints"])

        # --- 2. LiDAR ---
        result["step"] = "lidar"
        data = raster_io.read_dtm_dsm(geom_3035, ti.DEFAULT_DATASET)
        h, w = data["shape"]
        valid_px = int(data["mask"].sum())
        if valid_px > MAX_KG_PIXELS:
            result["error"] = f"too large: {valid_px} px"
            return result
        if valid_px < 100:
            result["error"] = f"too few valid pixels: {valid_px}"
            return result

        ndsm = data["ndsm"]
        mask = data["mask"]
        transform = data["transform"]

        # --- 3. Orthophoto ---
        result["step"] = "ortho"
        spectral = None
        try:
            import ortho_io
            rgb, nir = ortho_io.read_ortho_for_als(data)
            spectral = ortho_io.compute_spectral_indices(rgb, nir=nir)
            if rgb is not None:
                spectral["red"] = rgb[0].astype(np.float32)
                spectral["green"] = rgb[1].astype(np.float32)
                spectral["blue"] = rgb[2].astype(np.float32)
            if nir is not None:
                spectral["nir"] = nir.astype(np.float32)
        except Exception as e:
            log.warning("KG %s: ortho failed: %s", kg_code, e)

        # --- 4. Copernicus (tile-cached) ---
        copernicus_data = None
        if include_copernicus:
            result["step"] = "copernicus"
            cb = _read_circuit_breaker()
            if cb["consecutive_failures"] >= 3 and \
               (time.time() - cb["last_failure"]) < cb["cooldown"]:
                log.info("KG %s: Copernicus circuit breaker OPEN, skipping", kg_code)
            else:
                try:
                    bbox_dict = {"west": west, "south": south,
                                 "east": east, "north": north}
                    cop_cache = _get_cop_cache()
                    cop = {}
                    # NDVI (grid-snapped tile cache)
                    nd = cop_cache.get_ndvi(bbox_dict, year=obs_year)
                    if nd and nd.get("ndvi") is not None:
                        cop["ndvi"] = nd["ndvi"]
                        cop["transform"] = nd.get("transform")
                        cop["crs"] = nd.get("crs")
                    # WorldCover (grid-snapped tile cache)
                    lc = cop_cache.get_landcover(bbox_dict)
                    if lc:
                        cop["landcover"] = lc
                    # SAR (grid-snapped tile cache)
                    sar = cop_cache.get_sar(bbox_dict, year=obs_year)
                    if sar:
                        cop.update({k: sar[k] for k in ["vv", "vh"] if k in sar})
                        if "transform" in sar:
                            cop["sar_transform"] = sar["transform"]

                    copernicus_data = cop if cop else None

                    # Update circuit breaker
                    if copernicus_data:
                        cb["consecutive_failures"] = 0
                    else:
                        cb["consecutive_failures"] += 1
                        cb["last_failure"] = time.time()
                        cb["cooldown"] = min(600, 60 * (2 ** min(cb["consecutive_failures"], 4)))
                    _write_circuit_breaker(cb)
                except Exception as e:
                    from copernicus import CreditsExhaustedError
                    if isinstance(e, CreditsExhaustedError) or isinstance(e.__cause__, CreditsExhaustedError):
                        log.error("KG %s: Copernicus credits exhausted — signalling pause", kg_code)
                        result["copernicus_exhausted"] = True
                        # Write pause file so main loop can detect it
                        COPERNICUS_PAUSE_FILE.write_text(
                            f"Credits exhausted at {datetime.now(timezone.utc).isoformat()}\n"
                            f"Provide new credentials in copernicus.py and delete this file to resume.\n"
                        )
                    else:
                        log.warning("KG %s: Copernicus failed: %s", kg_code, e)
                    cb["consecutive_failures"] += 1
                    cb["last_failure"] = time.time()
                    _write_circuit_breaker(cb)

        # --- 5. Hansen (tile-cached) ---
        result["step"] = "hansen"
        hansen_data = None
        try:
            hc = _get_hansen_cache()
            hansen_data = hc.get_forest_prior(
                (west, south, east, north), transform, (h, w))
        except Exception as e:
            log.warning("KG %s: Hansen failed: %s", kg_code, e)

        # --- 6. Segmentation ---
        result["step"] = "segment"

        # Building footprint mask
        building_fp_mask = None
        if cadastre_data["building_footprints"]:
            try:
                from rasterio.features import rasterize as rio_rasterize
                pairs = [(b["geometry"], 1) for b in cadastre_data["building_footprints"]
                         if not b["geometry"].is_empty]
                if pairs:
                    building_fp_mask = rio_rasterize(
                        pairs, out_shape=(h, w), transform=transform,
                        fill=0, dtype=np.uint8, all_touched=True,
                    ).astype(bool)
            except Exception:
                pass

        seg_result = oc.segment_and_classify(
            data["dtm"], data["dsm"], mask, transform,
            spectral=spectral, copernicus=copernicus_data,
            building_footprints=building_fp_mask,
            hansen=hansen_data,
            observation_year=obs_year,
        )
        objects = seg_result["objects"]
        labels = seg_result["labels"]
        result["n_segments"] = len(objects)

        # --- 7. Terrain ---
        result["step"] = "terrain"
        terrain_stats = {}
        try:
            terrain_stats = ta.characterise_terrain(data["dtm"], mask)
        except Exception as e:
            log.warning("KG %s: terrain failed: %s", kg_code, e)

        # --- 8. Vectorise unmatched buildings & infrastructure ---
        result["step"] = "vectorise"
        new_buildings = []
        infrastructure_vec = []
        try:
            new_buildings = vectorise_unmatched_buildings(
                objects, labels, mask, transform, building_fp_mask, ndsm)
        except Exception as e:
            log.warning("KG %s: new buildings vectorise failed: %s", kg_code, e)
        try:
            infrastructure_vec = vectorise_infrastructure(
                objects, labels, mask, transform, ndsm, data["dtm"])
        except Exception as e:
            log.warning("KG %s: infrastructure vectorise failed: %s", kg_code, e)

        result["n_new_buildings"] = len(new_buildings)
        result["n_infrastructure"] = len(infrastructure_vec)

        # --- 9. Build full GPKG ---
        result["step"] = "gpkg_full"
        full_gpkg = build_full_gpkg(
            kg_code, data, spectral, labels, objects, mask, transform)
        result["files"]["full_gpkg"] = full_gpkg

        # --- 10. Build light GPKG ---
        result["step"] = "gpkg_light"
        light_gpkg = build_light_gpkg(
            kg_code, data, labels, objects, mask, transform,
            cadastre_data, ndsm, new_buildings, infrastructure_vec)
        result["files"]["light_gpkg"] = light_gpkg

        # --- 11. Build JSON summary ---
        result["step"] = "json"
        json_summary = build_json_summary(
            kg_code, kg, data, labels, objects, cadastre_data,
            terrain_stats, spectral, hansen_data, copernicus_data,
            new_buildings, infrastructure_vec, ndsm, obs_year)

        json_path = str(JSON_DIR / f"{kg_code}.json")
        with open(json_path, 'w') as f:
            json.dump(json_summary, f, indent=2, default=str)
        result["files"]["json"] = json_path

        result["success"] = True
        result["step"] = "done"

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        log.error("KG %s failed at step %s: %s", kg_code, result["step"], e)

    # Force GC
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Zenodo upload
# ---------------------------------------------------------------------------

def upload_kg_to_zenodo(kg_code: str, kg_name: str, files: dict,
                        manifest) -> dict:
    """Upload KG files to Zenodo, verify, delete local GPKGs."""
    from zenodo_client import Client, landscape_metadata

    client = Client(token=ZENODO_TOKEN)
    upload_stats = {"uploaded": [], "errors": [], "total_bytes": 0}

    for file_key, local_path in files.items():
        if not local_path or not os.path.exists(local_path):
            continue

        zenodo_key = f"{kg_code}_{file_key}"
        file_type = "gpkg" if file_key.endswith("gpkg") else "json"

        meta_func = lambda k, fn, v, _kg=kg_code, _name=kg_name, _ft=file_type: \
            landscape_metadata(_kg, _name, v, _ft)

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


# ---------------------------------------------------------------------------
# JSON dir cleanup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Austria Landscape Processor")
    parser.add_argument("--kg", help="Process single KG code")
    parser.add_argument("--state", help="Process KGs in one state")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry only previously failed KGs")
    parser.add_argument("--no-copernicus", action="store_true",
                        help="Skip Copernicus data (faster)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List KGs without processing")
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    log.info("=" * 70)
    log.info("🇦🇹 Austria Landscape Processor starting")
    log.info("=" * 70)

    # --- Load/create manifest ---
    from zenodo_client import Manifest
    manifest = Manifest(str(MANIFEST_PATH))
    log.info("Zenodo manifest: %d entries", len(manifest))

    # --- Load progress tracker ---
    progress = ProgressTracker(PROGRESS_FILE)

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
        # Only process KGs that have errors in manifest
        failed_keys = set()
        for key in manifest.keys():
            entry = manifest.get(key)
            # We'll store failure info with key "KG_error"
            if key.endswith("_error"):
                failed_keys.add(key.replace("_error", ""))
        kgs = [kg for kg in kgs if kg["kg_code"] in failed_keys]
        log.info("Retry mode: %d failed KGs to reprocess", len(kgs))

    # Skip already completed
    completed_codes = set()
    for key in manifest.keys():
        if key.endswith("_json"):
            completed_codes.add(key.replace("_json", ""))
    pending = [kg for kg in kgs if kg["kg_code"] not in completed_codes]

    # Sort geographically for tile-cache locality
    from tile_cache import sort_kgs_geographically
    pending = sort_kgs_geographically(pending)
    log.info("KGs sorted geographically for cache locality")

    log.info("Total KGs: %d, already completed: %d, pending: %d",
             len(kgs), len(completed_codes), len(pending))

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

    for i, kg in enumerate(pending):
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

        progress.set_current_kg(kg_code, kg_name, kg_state, "starting")
        progress.add_log("info", f"Starting KG {kg_code} ({kg_name})", kg_code)
        progress.save()

        IN_PROGRESS_FILE.write_text(kg_code)

        t_kg = time.time()
        try:
            pool = multiprocessing.Pool(processes=1)
            try:
                async_result = pool.apply_async(
                    process_one_kg, args=(kg,),
                    kwds={"include_copernicus": include_cop})
                result = async_result.get(timeout=KG_TIMEOUT_SECONDS)
            except multiprocessing.TimeoutError:
                log.error("KG %s: TIMEOUT after %d min", kg_code,
                          KG_TIMEOUT_SECONDS // 60)
                pool.terminate()
                pool.join()
                progress.add_failure(kg_code, kg_name,
                                     "timeout", "unknown")
                progress.add_log("error", f"KG {kg_code} timed out", kg_code)
                progress.save()
                continue
            finally:
                pool.close()
                pool.join()

            elapsed_kg = time.time() - t_kg

            if result.get("success"):
                # Upload to Zenodo
                progress.set_step("upload")
                progress.save()

                upload_stats = upload_kg_to_zenodo(
                    kg_code, kg_name, result["files"], manifest)

                if upload_stats["errors"]:
                    for err in upload_stats["errors"]:
                        log.warning("KG %s upload error: %s", kg_code, err)

                progress.record_success(
                    parcels=result.get("n_parcels", 0),
                    buildings=result.get("n_buildings", 0),
                    upload_bytes=upload_stats["total_bytes"],
                )
                progress.add_log(
                    "success",
                    f"KG {kg_code} completed in {elapsed_kg:.0f}s "
                    f"({result.get('n_segments', 0)} segments, "
                    f"{result.get('n_new_buildings', 0)} new buildings)",
                    kg_code,
                )
                log.info("KG %s: SUCCESS in %.0fs (%d segments)",
                         kg_code, elapsed_kg, result.get("n_segments", 0))
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

        except Exception as e:
            progress.add_failure(kg_code, kg_name, str(e), "exception")
            progress.add_log("error", f"KG {kg_code} exception: {e}", kg_code)
            log.error("KG %s: EXCEPTION: %s", kg_code, traceback.format_exc())

        # Cleanup
        if IN_PROGRESS_FILE.exists():
            IN_PROGRESS_FILE.unlink()

        progress.update_rates(t_start)
        progress.update_system_metrics()
        progress.save()
        cleanup_json_dir()

        gc.collect()

        # --- Copernicus pause check ---
        if COPERNICUS_PAUSE_FILE.exists():
            log.warning("⏸ Copernicus credits exhausted — PAUSING.")
            log.warning("  Update credentials in copernicus.py and delete %s to resume.",
                        COPERNICUS_PAUSE_FILE)
            progress.update(state="paused_copernicus")
            progress.add_log("warning",
                             "Paused: Copernicus credits exhausted. "
                             "Provide new creds & delete pause file.", "")
            progress.save()
            while COPERNICUS_PAUSE_FILE.exists():
                time.sleep(30)
            # Reset copernicus module flag so fresh creds are tried
            try:
                import copernicus
                copernicus.credits_exhausted = False
                copernicus._connection = None
                for k in list(copernicus._connections.keys()):
                    copernicus._connections.pop(k, None)
            except Exception:
                pass
            log.info("▶ Copernicus pause file removed — RESUMING.")
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
    progress.update(state="complete", current_kg=None)
    progress.save()

    elapsed = time.time() - t_start
    s = progress.get()
    log.info("=" * 70)
    log.info("Processing complete: %d success, %d failed in %.1f hours",
             s["success"], s["failed"], elapsed / 3600)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
