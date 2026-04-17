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
GPKG_DIR = DATA_DIR / "gpkg"  # persistent storage — survives /tmp cleanup
LOG_DIR = DATA_DIR / "logs"
PROGRESS_FILE = DATA_DIR / "progress.json"
KG_LIST_FILE = DATA_DIR / "kg_list.json"
FAILED_KGS_FILE = DATA_DIR / "failed_kgs.json"
IN_PROGRESS_FILE = DATA_DIR / "in_progress_kg.txt"
CIRCUIT_BREAKER_FILE = DATA_DIR / "openeo_circuit.json"
COPERNICUS_PAUSE_FILE = DATA_DIR / "copernicus_paused"

MAX_KG_PIXELS = 4_000_000
KG_TIMEOUT_SECONDS = 30 * 60
JSON_DIR_MAX_BYTES = 4 * 1024 ** 3  # 4GB
MAX_KG_AREA_KM = 1.5  # crop KG bbox if wider

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
                break  # one polygon per object
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Resolve edge-clipped features against full-KG rasters
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GPKG builders
# ---------------------------------------------------------------------------

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


def build_full_gpkg(kg_code: str, data: dict, spectral: dict,
                    labels: np.ndarray, objects: list,
                    mask: np.ndarray, transform,
                    obs_year: int = 0) -> str:
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

        # Vector segments with full attributes + QGIS style
        try:
            _write_segment_vectors(out_path, labels, objects, mask, transform,
                                   obs_year=obs_year)
        except Exception as e:
            log.warning("Full GPKG vector segments failed: %s", e)

    # Validate
    found_layers = []
    try:
        import rasterio
        with rasterio.open(out_path) as ds:
            found_layers = ds.descriptions or []
        fsize = os.path.getsize(out_path)
        log.info("  FULL_GPKG: %.1f MB, %d tables", fsize / 1e6, table_count)
    except Exception:
        pass

    return out_path


def build_light_gpkg(kg_code: str, data: dict, labels: np.ndarray,
                     objects: list, mask: np.ndarray, transform,
                     cadastre_data: dict, ndsm: np.ndarray,
                     new_buildings: list, infrastructure: list,
                     obs_year: int = 0,
                     full_data: dict = None) -> str:
    """Build lightweight GeoPackage with segmentation + enriched cadastre.

    When *full_data* is provided (full-KG DTM/DSM/nDSM), parcels and buildings
    are enriched against the full-KG raster so every feature gets heights,
    even those outside the segmentation window.
    """
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

    # Segmentation vector (full attributes + QGIS style)
    if labels is not None and objects:
        try:
            _write_segment_vectors(out_path, labels, objects, mask, transform,
                                   obs_year=obs_year)
        except Exception as e:
            log.warning("Light GPKG vector segments failed: %s", e)

    # Parcels with heights — use full-KG DTM if available
    enrich_dtm = full_data["dtm"] if full_data else data["dtm"]
    enrich_tf = full_data["transform"] if full_data else transform
    enriched_parcels = enrich_parcels_with_heights(
        cadastre_data["parcels"], enrich_dtm, enrich_tf)
    if enriched_parcels:
        schema_p = {
            'geometry': 'MultiPolygon',
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
                    'geometry': _to_multi(mapping(geom_wgs)),
                    'properties': {
                        'parcel_id': ep["parcel_id"],
                        'area_sqm': ep["area_sqm"],
                        'centroid_dtm_m': p.get('centroid_dtm_m'),
                        'centroid_lon': p.get('centroid_lon'),
                        'centroid_lat': p.get('centroid_lat'),
                    },
                })

    # Building footprints with heights — use full-KG rasters if available
    enrich_dsm = full_data["dsm"] if full_data else data["dsm"]
    enrich_ndsm = full_data["ndsm"] if full_data else ndsm
    enriched_bldgs = enrich_buildings_with_heights(
        cadastre_data["building_footprints"], enrich_dtm, enrich_dsm, enrich_ndsm, enrich_tf)
    if enriched_bldgs:
        schema_b = {
            'geometry': 'MultiPolygon',
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
                    'geometry': _to_multi(mapping(eb["geometry_wgs"])),
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
            'geometry': 'MultiPolygon',
            'properties': [
                ('type', 'str'), ('area_sqm', 'float'),
                ('max_height_m', 'float'), ('stories_est', 'int'),
                ('roof_type_hint', 'str'), ('confidence', 'float'),
                ('centroid_lon', 'float'), ('centroid_lat', 'float'),
                ('edge_clipped', 'bool'),
            ],
        }
        with fiona.open(out_path, 'w', driver='GPKG', layer='new_buildings',
                        schema=schema_nb, crs=from_epsg(4326)) as dst:
            for nb in new_buildings:
                dst.write({
                    'geometry': _to_multi(nb["geometry_wgs"]),
                    'properties': {
                        'type': nb.get('type', 'roof'),
                        'area_sqm': nb.get('area_sqm', 0),
                        'max_height_m': nb.get('max_height_m', 0),
                        'stories_est': nb.get('stories_est', 1),
                        'roof_type_hint': nb.get('roof_type_hint', ''),
                        'confidence': nb.get('confidence', 0),
                        'centroid_lon': nb.get('centroid_lon'),
                        'centroid_lat': nb.get('centroid_lat'),
                        'edge_clipped': nb.get('edge_clipped', False),
                    },
                })

    # Infrastructure
    if infrastructure:
        schema_infra = {
            'geometry': 'MultiPolygon',
            'properties': [
                ('type', 'str'), ('area_sqm', 'float'),
                ('volume_m3', 'float'), ('max_height_m', 'float'),
                ('est_parking_spots', 'int'), ('confidence', 'float'),
                ('centroid_lon', 'float'), ('centroid_lat', 'float'),
                ('edge_clipped', 'bool'),
            ],
        }
        with fiona.open(out_path, 'w', driver='GPKG', layer='infrastructure',
                        schema=schema_infra, crs=from_epsg(4326)) as dst:
            for inf in infrastructure:
                dst.write({
                    'geometry': _to_multi(inf["geometry_wgs"]),
                    'properties': {
                        'type': inf.get('type', ''),
                        'area_sqm': inf.get('area_sqm', 0),
                        'volume_m3': inf.get('volume_m3'),
                        'max_height_m': inf.get('max_height_m'),
                        'est_parking_spots': inf.get('est_parking_spots'),
                        'confidence': inf.get('confidence', 0),
                        'centroid_lon': inf.get('centroid_lon'),
                        'centroid_lat': inf.get('centroid_lat'),
                        'edge_clipped': inf.get('edge_clipped', False),
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
                       obs_year: int,
                       full_data: dict = None,
                       seg_window_km: float = None) -> dict:
    """Build comprehensive JSON summary for a KG.

    When *full_data* is provided, parcel/building enrichment uses the full-KG
    DTM/DSM so every feature gets elevation, even those outside the
    segmentation window.  *seg_window_km* records the window size used.
    """
    import tile_index as ti
    import rasterio

    mask = data["mask"]
    dtm = data["dtm"]
    h, w = data["shape"]

    # Full-KG rasters for enrichment (fall back to windowed data)
    full_dtm = full_data["dtm"] if full_data else dtm
    full_dsm = full_data["dsm"] if full_data else data["dsm"]
    full_ndsm = full_data["ndsm"] if full_data else ndsm
    full_mask = full_data["mask"] if full_data else mask
    full_tf = full_data["transform"] if full_data else data["transform"]
    full_h, full_w = full_data["shape"] if full_data else (h, w)

    # --- KG info ---
    summary = {
        "version": VERSION,
        "kg_code": kg_code,
        "kg_name": kg_info.get("kg_name", ""),
        "state": kg_info.get("state_name", ""),
        "gemeinde": kg_info.get("gemeinde_name", ""),
        "district": kg_info.get("district_name", ""),
        "bbox": kg_info.get("bbox", {}),
        "total_area_sqm": int(full_mask.sum()),  # full-KG valid pixels = m2 at 1m res
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observation_period": {
            "primary_year": obs_year,
            "start": f"{obs_year}-01-01",
            "end": f"{obs_year}-12-31",
            "lidar_dataset": ti.DEFAULT_DATASET,
            "all_lidar_datasets": sorted(ti.DATASETS.keys()),
            "sentinel2": f"Sentinel-2 L2A {obs_year}",
            "hansen": "Hansen GFC-2024-v1.12 (2000-2024)",
            "cadastre": "BEV INSPIRE cadastre (current)",
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
                "height_max_m": round(o.height_max, 2),
                "height_mean_m": round(o.height_mean, 2),
                "area_sqm": round(o.area_sqm, 1),
                "coordinate": c_wgs,
                "confidence": round(o.confidence, 3),
                "is_manmade": o.is_manmade,
                "slope_mean_deg": round(o.slope_mean, 1),
                "ndvi_mean": round(o.ndvi_mean, 4),
                "height_change_m": round(o.height_change, 3),
                "observation_year": obs_year,
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
                "height_p90_m": round(t.height_p90, 2),
                "coordinate": c_wgs,
                "area_sqm": round(t.area_sqm, 1),
                "ndvi_mean": round(t.ndvi_mean, 4),
                "ndvi_fused": round(t.ndvi_fused, 4),
                "height_change_m": round(t.height_change, 3),
                "phenology_class": t.phenology_class or '',
                "observation_year": obs_year,
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
        elev = ts.get("elevation", {})
        slope = ts.get("slope_deg", {})
        tri = ts.get("ruggedness_tri", {})
        curv = ts.get("curvature", {})
        aspect_dist = ts.get("aspect_distribution_pct", {})
        # Find dominant aspect
        dominant_aspect = max(aspect_dist, key=aspect_dist.get) if aspect_dist else ""
        summary["terrain"] = {
            "steepness_mean_deg": slope.get("mean"),
            "steepness_max_deg": slope.get("max"),
            "aspect_dominant": dominant_aspect,
            "aspect_distribution_pct": aspect_dist,
            "slope_classes_pct": ts.get("slope_classes_pct", {}),
            "roughness_mean": tri.get("mean"),
            "curvature_mean": curv.get("mean") if isinstance(curv, dict) else None,
            "elevation_min_m": elev.get("min"),
            "elevation_max_m": elev.get("max"),
            "elevation_range_m": elev.get("range"),
            "elevation_mean_m": elev.get("mean"),
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

    # --- SAR backscatter ---
    sar_info = {}
    if copernicus_data:
        for band in ['vv', 'vh']:
            arr = copernicus_data.get(band)
            if arr is not None:
                valid = arr[np.isfinite(arr)]
                if len(valid) > 0:
                    sar_info[f"{band}_mean_db"] = round(float(np.nanmean(valid)), 2)
                    sar_info[f"{band}_std_db"] = round(float(np.nanstd(valid)), 2)
        if sar_info:
            sar_info["method"] = f"Sentinel-1 IW GRD, VV+VH, summer {obs_year}"
    summary["sar"] = sar_info

    # --- NDVI harmonics (phenology) ---
    harmonics_info = {}
    if copernicus_data and copernicus_data.get("harmonics"):
        harm = copernicus_data["harmonics"]
        for key in ['h_mean', 'h_amplitude', 'h_phase', 'h_rmse']:
            arr = harm.get(key)
            if arr is not None:
                valid = arr[np.isfinite(arr)]
                if len(valid) > 0:
                    harmonics_info[key.replace('h_', '') + '_mean'] = round(float(np.nanmean(valid)), 4)
        harmonics_info["method"] = f"1st-order harmonic fit to monthly Sentinel-2 NDVI, {obs_year}"
    summary["ndvi_harmonics"] = harmonics_info

    # --- Temporal change summary ---
    temporal_info = {}
    if objects:
        changes = [(o.dtm_change, o.height_change, o.volume_change_m3,
                    o.volume_change_abs_m3, o.temporal_stability) for o in objects]
        dtm_ch = [c[0] for c in changes if abs(c[0]) > 0.01]
        h_ch = [c[1] for c in changes if abs(c[1]) > 0.01]
        vol_net = sum(c[2] for c in changes)
        vol_abs = sum(c[3] for c in changes)
        stab = [c[4] for c in changes]
        if dtm_ch:
            temporal_info["dtm_change_mean_m"] = round(float(np.mean(dtm_ch)), 3)
            temporal_info["dtm_change_max_abs_m"] = round(float(np.max(np.abs(dtm_ch))), 3)
            temporal_info["n_changed_segments"] = len(dtm_ch)
        if h_ch:
            temporal_info["height_change_mean_m"] = round(float(np.mean(h_ch)), 3)
        temporal_info["net_volume_change_m3"] = round(vol_net, 1)
        temporal_info["total_disturbed_volume_m3"] = round(vol_abs, 1)
        if stab:
            temporal_info["mean_stability"] = round(float(np.mean(stab)), 3)
        temporal_info["datasets_compared"] = sorted(ti.DATASETS.keys())
        temporal_info["method"] = "DTM/DSM difference across BEV ALS dates (2022-2024)"
    summary["temporal_change"] = temporal_info

    # --- Phenology class distribution ---
    if objects:
        pheno_counts = Counter(o.phenology_class for o in objects if o.phenology_class)
        if pheno_counts:
            summary["phenology"] = {
                "distribution": {k: v for k, v in pheno_counts.most_common()},
                "method": "1st-order harmonic fit: mean+amplitude+phase → class",
            }

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
            "edge_clipped": nb.get("edge_clipped", False),
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

    # --- Per-parcel detail ---
    # Use full-KG DTM for elevation so ALL parcels get heights.
    # Segmentation labels only cover the windowed area.
    seg_transform = data["transform"]
    parcel_details = []
    obj_map = {o.obj_id: o for o in objects} if objects else {}
    seg_geom_3035 = None
    if seg_window_km is not None:
        # Build segmentation window polygon for inside/outside test
        seg_bounds = rasterio.transform.array_bounds(h, w, seg_transform)
        seg_geom_3035 = box(*seg_bounds)
    n_inside_window = 0
    for p in cadastre_data["parcels"]:
        pd = {
            "parcel_id": p["parcel_id"],
            "area_sqm": round(p.get("area_sqm", 0), 1),
        }
        geom_3035 = p["geometry"]
        # Centroid
        try:
            c_wgs = transform_to_wgs(geom_3035.centroid)
            pd["centroid"] = {"lon": round(c_wgs.x, 7), "lat": round(c_wgs.y, 7)}
        except Exception:
            pass
        # DTM elevation at centroid — use full-KG DTM (with neighborhood fallback)
        try:
            c3 = geom_3035.centroid
            col = int((c3.x - full_tf.c) / full_tf.a)
            row = int((full_tf.f - c3.y) / abs(full_tf.e))
            if 0 <= row < full_h and 0 <= col < full_w:
                val = float(full_dtm[row, col])
                if np.isfinite(val):
                    pd["elevation_m"] = round(val, 2)
                else:
                    # Try 5x5 neighborhood
                    r0, r1 = max(0, row-2), min(full_h, row+3)
                    c0, c1 = max(0, col-2), min(full_w, col+3)
                    patch = full_dtm[r0:r1, c0:c1]
                    valid = patch[np.isfinite(patch)]
                    if len(valid) > 0:
                        pd["elevation_m"] = round(float(np.nanmean(valid)), 2)
        except Exception:
            pass
        # Check if parcel is inside the segmentation window.
        # Use overlap fraction, not bare intersects, so parcels with
        # only a sliver inside the window don't get misleading area_summary.
        parcel_in_window = True
        seg_coverage_frac = 1.0
        if seg_geom_3035 is not None:
            try:
                if not seg_geom_3035.intersects(geom_3035):
                    parcel_in_window = False
                    seg_coverage_frac = 0.0
                else:
                    isect = seg_geom_3035.intersection(geom_3035)
                    parcel_area = max(geom_3035.area, 1e-6)
                    seg_coverage_frac = isect.area / parcel_area
                    # Require ≥50% of parcel inside window for segmentation data
                    parcel_in_window = seg_coverage_frac >= 0.5
            except Exception:
                parcel_in_window = False
                seg_coverage_frac = 0.0
        pd["segmented"] = parcel_in_window
        pd["seg_coverage_frac"] = round(seg_coverage_frac, 3)
        if parcel_in_window:
            n_inside_window += 1
        # Segment type breakdown within parcel (only if ≥50% inside window)
        if parcel_in_window and labels is not None and objects:
            try:
                from rasterio.features import rasterize as rio_rasterize
                p_mask = rio_rasterize(
                    [(geom_3035, 1)], out_shape=(h, w), transform=seg_transform,
                    fill=0, dtype=np.uint8, all_touched=True).astype(bool)
                p_labels = labels[p_mask]
                p_ndsm = ndsm[p_mask]
                tc = Counter()
                th = defaultdict(list)
                for lbl in np.unique(p_labels):
                    obj = obj_map.get(int(lbl))
                    if obj is None:
                        continue
                    n_px = int((p_labels == lbl).sum())
                    tc[obj.obj_type] += n_px
                    th[obj.obj_type].append(obj.height_max)
                if tc:
                    pd["area_summary"] = {
                        t: {"area_sqm": px, "fraction": round(px / max(int(p_mask.sum()), 1), 4)}
                        for t, px in tc.most_common()
                    }
                    pd["height_distribution"] = {
                        t: {"min": round(min(hs), 2), "max": round(max(hs), 2),
                            "mean": round(sum(hs)/len(hs), 2)}
                        for t, hs in th.items() if hs
                    }
                # Vegetated fraction
                veg_types = {'tree','shrub','grass','hedge','crop','orchard','vineyard','garden'}
                veg_px = sum(v for k, v in tc.items() if k in veg_types)
                total_px = max(int(p_mask.sum()), 1)
                pd["vegetated_fraction"] = round(veg_px / total_px, 4)
                pd["is_vegetated"] = veg_px / total_px > 0.5
                # Elevation stats from nDSM
                valid_h = p_ndsm[np.isfinite(p_ndsm)]
                if len(valid_h) > 0:
                    pd["ndsm_max_m"] = round(float(np.max(valid_h)), 2)
                    pd["ndsm_mean_m"] = round(float(np.mean(valid_h)), 2)
            except Exception:
                pass
        parcel_details.append(pd)

    summary["parcels"] = {
        "count": len(cadastre_data["parcels"]),
        "total_area_sqm": round(sum(p.get("area_sqm", 0) for p in cadastre_data["parcels"]), 1),
        "parcels_in_seg_window": n_inside_window,
        "details": parcel_details,
    }

    # --- Per-building-footprint detail ---
    # Use full-KG rasters for height so ALL buildings get data.
    building_details = []
    for b in cadastre_data["building_footprints"]:
        bd = {}
        geom_3035 = b["geometry"]
        props = b.get("properties", {})
        bd["building_id"] = props.get("building_id", props.get("id", ""))
        bd["footprint_area_sqm"] = round(float(geom_3035.area), 1)
        # Centroid
        try:
            c_wgs = transform_to_wgs(geom_3035.centroid)
            bd["centroid"] = {"lon": round(c_wgs.x, 7), "lat": round(c_wgs.y, 7)}
        except Exception:
            pass
        # Height stats from full-KG nDSM
        try:
            from rasterio.features import rasterize as rio_rasterize
            b_mask_full = rio_rasterize(
                [(geom_3035, 1)], out_shape=(full_h, full_w), transform=full_tf,
                fill=0, dtype=np.uint8, all_touched=True).astype(bool)
            oh = full_ndsm[b_mask_full]
            oh = oh[np.isfinite(oh)]
            dsm_vals = full_dsm[b_mask_full]
            dsm_vals = dsm_vals[np.isfinite(dsm_vals)]
            if len(oh) > 0:
                max_h = float(np.max(oh))
                bd["max_height_m"] = round(max_h, 2)
                bd["mean_height_m"] = round(float(np.mean(oh)), 2)
                bd["dsm_std"] = round(float(np.std(dsm_vals)), 2) if len(dsm_vals) > 0 else 0.0
                bd["roof_type_hint"] = "flat" if bd["dsm_std"] < 1.5 else "pitched"
                bd["stories_est"] = max(1, round(max_h / 3.0))
        except Exception:
            pass
        # Check if building is inside segmentation window (same 50% rule)
        bld_in_window = True
        if seg_geom_3035 is not None:
            try:
                if not seg_geom_3035.intersects(geom_3035):
                    bld_in_window = False
                else:
                    isect = seg_geom_3035.intersection(geom_3035)
                    bld_in_window = isect.area / max(geom_3035.area, 1e-6) >= 0.5
            except Exception:
                bld_in_window = False
        bd["segmented"] = bld_in_window
        # Segment types within footprint (only if inside window)
        if bld_in_window and labels is not None and objects:
            try:
                b_mask_seg = rio_rasterize(
                    [(geom_3035, 1)], out_shape=(h, w), transform=seg_transform,
                    fill=0, dtype=np.uint8, all_touched=True).astype(bool)
                b_labels = labels[b_mask_seg]
                tc = Counter()
                for lbl in np.unique(b_labels):
                    obj = obj_map.get(int(lbl))
                    if obj:
                        tc[obj.obj_type] += int((b_labels == lbl).sum())
                if tc:
                    bd["segment_types"] = {t: px for t, px in tc.most_common()}
            except Exception:
                pass
        building_details.append(bd)

    summary["building_footprints"] = {
        "count": len(cadastre_data["building_footprints"]),
        "details": building_details,
    }

    # --- Coverage ---
    # Segmentation window vs full KG extent
    seg_area_sqm = int(mask.sum())
    full_area_sqm = int(full_mask.sum())
    seg_coverage_pct = round(100 * seg_area_sqm / max(full_area_sqm, 1), 1)
    n_parcels_total = len(cadastre_data["parcels"])
    n_parcels_with_elev = sum(1 for pd in parcel_details if pd.get("elevation_m") is not None)
    n_buildings_total = len(cadastre_data["building_footprints"])
    n_buildings_with_h = sum(1 for bd in building_details if bd.get("max_height_m") is not None)
    summary["coverage"] = {
        "segmentation_window_km": seg_window_km,
        "segmentation_area_sqm": seg_area_sqm,
        "full_kg_area_sqm": full_area_sqm,
        "segmentation_coverage_pct": seg_coverage_pct,
        "parcel_coverage_pct": round(100 * n_parcels_with_elev / max(n_parcels_total, 1), 1),
        "building_coverage_pct": round(100 * n_buildings_with_h / max(n_buildings_total, 1), 1),
        "parcels_in_seg_window": n_inside_window,
        "parcels_total": n_parcels_total,
        "buildings_total": n_buildings_total,
        "note": "Segmentation+classification covers the central window; DTM elevation covers all parcels/buildings in the full KG.",
    }

    # --- Methods ---
    summary["methods"] = {
        "segmentation": "Felzenszwalb over-segmentation + RAG merge on fused gradient (DTM+DSM+RGBI+NDVI), 1m resolution",
        "classification": "Random Forest (44 features, cadastre+OSM trained) with rule-based fallback + cadastre calibration",
        "calibration": "Building footprints from cadastre used for confidence boosting and missed-building reclassification",
        "height": "BEV ALS DTM/DSM 1m, nDSM = DSM - DTM",
        "temporal_change": "DTM/DSM differencing across all available BEV ALS dates (" + ", ".join(sorted(ti.DATASETS.keys())) + ")",
        "ortho": "BEV DOP RGBI 0.2m, resampled to 1m for spectral indices",
        "ndvi_bev": "(NIR - Red) / (NIR + Red) from BEV DOP RGBI",
        "ndvi_copernicus": "Sentinel-2 L2A B08/B04, openEO, 10m resampled to 1m",
        "ndvi_harmonics": "1st-order harmonic fit (mean + amplitude·cos(2πt/12 - phase)) to monthly Sentinel-2 NDVI",
        "sar": "Sentinel-1 IW GRD, VV+VH polarisation, summer composite via openEO",
        "terrain": "Slope (Sobel), aspect, TRI, TPI, curvature from full-KG DTM",
        "texture": "GLCM contrast/homogeneity/entropy from BEV ortho greyscale",
        "hansen": "Hansen GFC-2024-v1.12, 30m, treecover2000 + lossyear + gain",
        "infrastructure": "austria-power API (wind turbines, solar, substations, masts)",
        "roof_type": "flat = DSM std < 1.5m within footprint, pitched otherwise",
        "stories_est": "max_object_height / 3m, rounded",
        "stem_volume": "Rough cone estimate: 0.3 * canopy_area * height / 3",
        "parking_spots": "Area / 12.5 m² (standard parking spot size)",
        "earthwork_volume": "mean(|nDSM|) * polygon_area",
        "fragmentation": "Shannon diversity index on segment type fractions",
        "edge_density": "Total segment perimeter / total area",
        "phenology": "Harmonic amplitude+phase → crop/deciduous/evergreen/bare classes",
        "top_10_objects": "Tallest objects within the segmentation window (not full KG)",
        "coverage": "Full-KG DTM read for parcel/building elevation; segmentation limited to central window to prevent timeouts",
        "cadastre_source": "BEV INSPIRE cadastre via cadastre-process-api.exe.xyz",
        "data_sources": [
            "BEV ALS DTM/DSM 1m (2022, 2023, 2024)",
            "BEV DOP RGBI 0.2m (2022-2024)",
            "Sentinel-2 L2A 10m NDVI composites + monthly time series (openEO)",
            "ESA WorldCover 10m",
            "Sentinel-1 SAR IW GRD 10m VV/VH (openEO)",
            "Hansen GFC-2024-v1.12 30m",
            "Austrian Cadastre (BEV INSPIRE)",
            "Austria Power Infrastructure API",
        ],
    }

    return summary


# ---------------------------------------------------------------------------
# Core per-KG processing (runs in subprocess)
# ---------------------------------------------------------------------------

def process_one_kg(kg: dict, include_copernicus: bool = True, max_km: float = None) -> dict:
    """Process a single KG. Returns dict with file paths + stats.

    This function runs in a subprocess for memory isolation.
    """
    import raster_io
    import tile_index as ti
    import object_segmentation as oc
    import terrain_analysis as ta

    kg_code = kg["kg_code"]
    result = {"kg_code": kg_code, "success": False, "step": "init", "files": {}}

    def _report_step(step, detail=""):
        """Write current step to temp file for parent to read."""
        try:
            step_file = DATA_DIR / "current_step.json"
            import json as _json
            _json.dump({"step": step, "detail": detail, "ts": datetime.now(timezone.utc).isoformat()},
                       open(str(step_file) + ".tmp", "w"))
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

        # Store the API bbox as the "full KG" bbox (may be refined below)
        full_west, full_south, full_east, full_north = west, south, east, north
        obs_year = ti.dataset_to_year(ti.DEFAULT_DATASET)

        # --- 1. Cadastre ---
        result["step"] = "cadastre"
        _report_step("cadastre")
        cadastre_data = fetch_cadastre_data(kg_code)
        result["n_parcels"] = len(cadastre_data["parcels"])
        result["n_buildings"] = len(cadastre_data["building_footprints"])
        _report_step("cadastre", f"{len(cadastre_data['parcels'])} parcels, {len(cadastre_data['building_footprints'])} buildings")

        # --- 1b. Compute full KG bbox from cadastre geometry union ---
        # The API bbox can be inaccurate; use the actual geometry extent.
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
                cb = cad_union_wgs.bounds  # (minx, miny, maxx, maxy)
                # Use the wider of API bbox and cadastre bbox
                full_west = min(full_west, cb[0])
                full_south = min(full_south, cb[1])
                full_east = max(full_east, cb[2])
                full_north = max(full_north, cb[3])
                log.info("KG %s: full KG bbox from cadastre union: %.4f,%.4f → %.4f,%.4f",
                         kg_code, full_west, full_south, full_east, full_north)
            except Exception as e:
                log.warning("KG %s: cadastre union failed, using API bbox: %s", kg_code, e)

        full_geom_wgs = box(full_west, full_south, full_east, full_north)
        full_geom_3035 = transform_to_3035(full_geom_wgs)

        # --- 2. Full-KG LiDAR (default date only, for height enrichment) ---
        result["step"] = "lidar_full"
        _report_step("lidar_full", "reading full-KG DTM/DSM")
        full_data = raster_io.read_dtm_dsm(full_geom_3035, ti.DEFAULT_DATASET)
        full_h, full_w = full_data["shape"]
        full_valid_px = int(full_data["mask"].sum())
        if full_valid_px < 100:
            result["error"] = f"too few valid pixels in full KG: {full_valid_px}"
            return result
        full_dtm = full_data["dtm"]
        full_dsm = full_data["dsm"]
        full_ndsm = full_data["ndsm"]
        full_mask = full_data["mask"]
        full_transform = full_data["transform"]
        log.info("KG %s: full-KG DTM/DSM %dx%d, %d valid px",
                 kg_code, full_h, full_w, full_valid_px)
        _report_step("lidar_full", f"full KG {full_h}x{full_w}, {full_valid_px} px")

        # --- 2b. Terrain analysis on full-KG DTM ---
        result["step"] = "terrain_full"
        _report_step("terrain_full", "full-KG terrain analysis")
        terrain_stats = {}
        try:
            terrain_stats = ta.characterise_terrain(full_dtm, full_mask)
        except Exception as e:
            log.warning("KG %s: full-KG terrain failed: %s", kg_code, e)
        _report_step("terrain_full", "done")

        # --- 3. Crop to segmentation window for expensive operations ---
        crop_km = max_km if max_km is not None else MAX_KG_AREA_KM
        dx_km = (full_east - full_west) * 111 * np.cos(np.radians((full_south + full_north) / 2))
        dy_km = (full_north - full_south) * 111
        seg_cropped = False
        if dx_km > crop_km or dy_km > crop_km:
            cx, cy = (full_west + full_east) / 2, (full_south + full_north) / 2
            half = (crop_km / 2) / 111
            west, south, east, north = cx - half, cy - half, cx + half, cy + half
            log.info("KG %s: segmentation window cropped to %.1fkm", kg_code, crop_km)
            seg_cropped = True
        else:
            west, south, east, north = full_west, full_south, full_east, full_north

        geom_wgs = box(west, south, east, north)
        geom_3035 = transform_to_3035(geom_wgs)

        # --- 3b. Windowed LiDAR for segmentation ---
        result["step"] = "lidar"
        _report_step("lidar", "reading windowed DTM/DSM for segmentation")
        if seg_cropped:
            data = raster_io.read_dtm_dsm(geom_3035, ti.DEFAULT_DATASET)
        else:
            # No crop needed — reuse full-KG data
            data = full_data
        h, w = data["shape"]
        valid_px = int(data["mask"].sum())
        # Scale pixel limit with crop window (smaller window = fewer pixels needed)
        effective_max_px = MAX_KG_PIXELS if max_km is None else int((max_km * 1000) ** 2)
        if valid_px > effective_max_px:
            result["error"] = f"too large: {valid_px} px (limit {effective_max_px})"
            return result
        if valid_px < 100:
            result["error"] = f"too few valid pixels: {valid_px}"
            return result
        _report_step("lidar", f"{h}x{w}, {valid_px} valid px")

        ndsm = data["ndsm"]
        mask = data["mask"]
        transform = data["transform"]

        # --- 2b. Multi-date DTM/DSM (temporal change features) ---
        # The RF model uses temporal features (14.7% importance) so we
        # need all available dates for accurate classification.
        dtm_dates = None
        dsm_dates = None
        try:
            other_dates = sorted(d for d in ti.DATASETS if d != ti.DEFAULT_DATASET)
            if other_dates:
                dtm_dates = {}
                dsm_dates = {}
                ref_h, ref_w = h, w
                for date_key in other_dates:
                    try:
                        d2 = raster_io.read_dtm_dsm(geom_3035, date_key)
                        mh = min(ref_h, d2["shape"][0])
                        mw = min(ref_w, d2["shape"][1])
                        dtm_dates[date_key] = d2["dtm"][:mh, :mw]
                        dsm_dates[date_key] = d2["dsm"][:mh, :mw]
                    except Exception as e:
                        log.warning("KG %s: multi-date %s failed: %s",
                                    kg_code, date_key, e)
                if dtm_dates:
                    # Align all arrays to the smallest common extent
                    mh = min(ref_h, *(a.shape[0] for a in dtm_dates.values()))
                    mw = min(ref_w, *(a.shape[1] for a in dtm_dates.values()))
                    dtm_dates[ti.DEFAULT_DATASET] = data["dtm"][:mh, :mw]
                    dsm_dates[ti.DEFAULT_DATASET] = data["dsm"][:mh, :mw]
                    for dk in list(dtm_dates):
                        dtm_dates[dk] = dtm_dates[dk][:mh, :mw]
                        dsm_dates[dk] = dsm_dates[dk][:mh, :mw]
                    log.info("KG %s: loaded %d temporal dates: %s",
                             kg_code, len(dtm_dates), sorted(dtm_dates))
                    _report_step("lidar",
                                 f"{h}x{w}, {valid_px} px, {len(dtm_dates)} dates")
                else:
                    dtm_dates = None
                    dsm_dates = None
        except Exception as e:
            log.warning("KG %s: multi-date read failed: %s", kg_code, e)
            dtm_dates = None
            dsm_dates = None

        # --- 3. Orthophoto (with timeout protection) ---
        result["step"] = "ortho"
        _report_step("ortho")
        ORTHO_TIMEOUT = 180  # 3 min max
        spectral = None
        try:
            import ortho_io
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
                fut = exe.submit(ortho_io.read_ortho_for_als, data)
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
                    log.warning("KG %s: ortho timed out after %ds",
                                kg_code, ORTHO_TIMEOUT)
        except Exception as e:
            log.warning("KG %s: ortho failed: %s", kg_code, e)
        if spectral:
            _report_step("ortho", f"RGBI loaded, {len(spectral)} bands")
        else:
            _report_step("ortho", "skipped/failed")

        # --- 4. Copernicus (tile-cached) ---
        copernicus_data = None
        if include_copernicus:
            result["step"] = "copernicus"
            _report_step("copernicus")
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

                    # NDVI harmonics (6.7% of RF feature importance)
                    # Only attempt if at least one Copernicus layer succeeded
                    if cop:
                        try:
                            import ndvi_harmonics
                            HARM_TIMEOUT = 300  # 5 min
                            import concurrent.futures as _cf
                            with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                                _hfut = _ex.submit(
                                    ndvi_harmonics.get_harmonic_features,
                                    bbox_dict, obs_year)
                                try:
                                    harm = _hfut.result(timeout=HARM_TIMEOUT)
                                    if harm is not None:
                                        cop["harmonics"] = harm
                                        log.info("KG %s: harmonics OK", kg_code)
                                except _cf.TimeoutError:
                                    log.warning("KG %s: harmonics timed out after %ds",
                                                kg_code, HARM_TIMEOUT)
                                except Exception as he:
                                    log.debug("KG %s: harmonics failed: %s", kg_code, he)
                        except Exception:
                            pass

                    copernicus_data = cop if cop else None

                    # Update circuit breaker
                    if copernicus_data:
                        cb["consecutive_failures"] = 0
                    else:
                        cb["consecutive_failures"] += 1
                        cb["last_failure"] = time.time()
                        cb["cooldown"] = min(600, 60 * (2 ** min(cb["consecutive_failures"], 4)))
                        # Rotate credentials for next attempt
                        try:
                            import copernicus as _cop_mod
                            _cop_mod.rotate_credentials()
                        except Exception:
                            pass
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

        if copernicus_data:
            bands = [k for k in copernicus_data.keys() if k not in ('transform','crs','sar_transform')]
            _report_step("copernicus", f"loaded: {', '.join(bands)}")
        else:
            _report_step("copernicus", "skipped (circuit breaker or failed)")

        # --- 5. Hansen (tile-cached) ---
        result["step"] = "hansen"
        _report_step("hansen")
        hansen_data = None
        try:
            hc = _get_hansen_cache()
            hansen_data = hc.get_forest_prior(
                (west, south, east, north), transform, (h, w))
        except Exception as e:
            log.warning("KG %s: Hansen failed: %s", kg_code, e)
        if hansen_data:
            _report_step("hansen", "loaded")
        else:
            _report_step("hansen", "skipped/failed")

        # --- 6. Segmentation ---
        result["step"] = "segment"
        _report_step("segment")

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

        # Infrastructure lookup for rule-based detection of
        # wind turbines, solar panels, substations, masts
        infra = None
        try:
            from infrastructure_lookup import InfrastructureLookup
            infra = InfrastructureLookup.for_bbox(west, south, east, north)
            if len(infra) > 0:
                log.info("KG %s: %d infrastructure features loaded",
                         kg_code, len(infra))
        except Exception as e:
            log.debug("KG %s: infrastructure lookup failed: %s", kg_code, e)

        seg_result = oc.segment_and_classify(
            data["dtm"], data["dsm"], mask, transform,
            dtm_dates=dtm_dates, dsm_dates=dsm_dates,
            spectral=spectral, copernicus=copernicus_data,
            building_footprints=building_fp_mask,
            hansen=hansen_data,
            observation_year=obs_year,
            infra_lookup=infra,
        )
        objects = seg_result["objects"]
        labels = seg_result["labels"]
        result["n_segments"] = len(objects)
        _report_step("segment", f"{len(objects)} objects, {result.get('n_segments',0)} segments")

        # --- 7. Terrain (already done on full-KG DTM in step 2b) ---

        # --- 8. Vectorise unmatched buildings & infrastructure ---
        result["step"] = "vectorise"
        _report_step("vectorise")
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
        _report_step("vectorise", f"{len(new_buildings)} new buildings, {len(infrastructure_vec)} infrastructure")

        # --- 8b. Resolve edge-clipped features against full-KG nDSM ---
        if seg_cropped and (new_buildings or infrastructure_vec):
            result["step"] = "resolve_edges"
            _report_step("resolve_edges")
            try:
                new_buildings, infrastructure_vec = resolve_edge_clipped_features(
                    new_buildings, infrastructure_vec,
                    full_ndsm, full_dsm, full_mask, full_transform)
                n_resolved = (sum(1 for x in new_buildings if x.get('edge_resolved'))
                              + sum(1 for x in infrastructure_vec if x.get('edge_resolved')))
                _report_step("resolve_edges", f"{n_resolved} features resolved")
            except Exception as e:
                log.warning("KG %s: edge resolution failed: %s", kg_code, e)

        # --- 9. Build full GPKG ---
        result["step"] = "gpkg_full"
        _report_step("gpkg_full")
        full_gpkg = build_full_gpkg(
            kg_code, data, spectral, labels, objects, mask, transform,
            obs_year=obs_year)
        result["files"]["full_gpkg"] = full_gpkg

        # --- 10. Build light GPKG ---
        result["step"] = "gpkg_light"
        _report_step("gpkg_light")
        light_gpkg = build_light_gpkg(
            kg_code, data, labels, objects, mask, transform,
            cadastre_data, ndsm, new_buildings, infrastructure_vec,
            obs_year=obs_year, full_data=full_data)
        result["files"]["light_gpkg"] = light_gpkg

        # --- 11. Build JSON summary ---
        result["step"] = "json"
        _report_step("json")
        json_summary = build_json_summary(
            kg_code, kg, data, labels, objects, cadastre_data,
            terrain_stats, spectral, hansen_data, copernicus_data,
            new_buildings, infrastructure_vec, ndsm, obs_year,
            full_data=full_data,
            seg_window_km=crop_km if seg_cropped else None)

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

    # Explicitly free large arrays before returning to parent
    for _var in ('data', 'full_data', 'full_dtm', 'full_dsm', 'full_ndsm',
                 'labels', 'objects', 'ndsm', 'spectral',
                 'copernicus_data', 'hansen_data', 'cadastre_data',
                 'new_buildings', 'infrastructure_vec', 'dtm_dates', 'dsm_dates',
                 'building_fp_mask', 'seg_result', 'terrain_stats'):
        try:
            del locals()[_var]  # noqa
        except (KeyError, NameError):
            pass
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------

def validate_kg_outputs(kg_code: str, files: dict) -> list[str]:
    """Validate all KG output products. Returns list of issues (empty = all OK)."""
    issues = []

    # --- Full GPKG ---
    full_path = files.get("full_gpkg", "")
    if not full_path or not os.path.exists(full_path):
        issues.append("FULL_GPKG: missing")
    else:
        try:
            import rasterio
            expected_layers = ["DTM", "DSM", "nDSM", "segment_type", "segment_height"]
            found_layers = set()
            # List all GPKG raster tables (tiles + 2d-gridded-coverage)
            try:
                import sqlite3
                conn = sqlite3.connect(full_path)
                tables = [r[0] for r in conn.execute(
                    "SELECT table_name FROM gpkg_contents "
                    "WHERE data_type IN ('tiles', '2d-gridded-coverage')").fetchall()]
                conn.close()
                found_layers = set(tables)
            except Exception:
                pass
            for layer in expected_layers:
                if layer not in found_layers:
                    issues.append(f"FULL_GPKG: missing raster layer '{layer}'")
            if "Ortho" not in found_layers:
                issues.append("FULL_GPKG: missing Ortho layer (ortho read may have failed)")
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
            conn = sqlite3.connect(light_path)
            # Raster tables
            raster_tables = [r[0] for r in conn.execute(
                "SELECT table_name FROM gpkg_contents "
                    "WHERE data_type IN ('tiles', '2d-gridded-coverage')").fetchall()]
            # Vector tables
            vector_tables = [r[0] for r in conn.execute(
                "SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchall()]
            conn.close()

            if "segment_type" not in raster_tables:
                issues.append("LIGHT_GPKG: missing segment_type raster")

            expected_vectors = ["segments", "parcels"]
            for v in expected_vectors:
                if v not in vector_tables:
                    issues.append(f"LIGHT_GPKG: missing vector layer '{v}'")

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

# Retry ladder: on timeout, shrink the crop window and retry.
# Disable Copernicus on tiny windows (10m resolution useless at 200m).
RETRY_LADDER = [1.5, 0.5, 0.2]  # km — first attempt uses MAX_KG_AREA_KM
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

    # Sort geographically for tile-cache locality
    from tile_cache import sort_kgs_geographically
    pending = sort_kgs_geographically(pending)
    log.info("KGs sorted geographically for cache locality")

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

        progress.set_current_kg(kg_code, kg_name, kg_state, "starting")
        progress.add_log("info", f"Starting KG {kg_code} ({kg_name})", kg_code)
        progress.save()

        IN_PROGRESS_FILE.write_text(kg_code)

        # Start background thread to monitor subprocess step reporting
        import threading
        _step_monitor_stop = threading.Event()

        def _monitor_step_file(_stop=_step_monitor_stop, _code=kg_code):
            step_file = DATA_DIR / "current_step.json"
            last_step = ""
            while not _stop.is_set():
                try:
                    if step_file.exists():
                        sd = json.loads(step_file.read_text())
                        s = sd.get("step", "")
                        detail = sd.get("detail", "")
                        if s and s != last_step:
                            last_step = s
                            progress.set_step(s)
                            if detail:
                                progress.add_log("info", f"KG {_code}: {s} \u2014 {detail}", _code)
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
            # ---- Retry ladder: 1.5km (default) → 0.5km → 0.2km ----
            # On timeout, shrink window and retry. Disable Copernicus
            # on tiny windows (10m resolution not useful at 200m).
            attempt_windows = [None] + RETRY_LADDER  # None = use MAX_KG_AREA_KM
            for attempt_idx, attempt_km in enumerate(attempt_windows):
                if attempt_idx > 0:
                    gc.collect()
                    log.info("  → Retrying KG %s with %.0fm window",
                             kg_code, attempt_km * 1000)
                    progress.add_log("info",
                                     f"Retry KG {kg_code} at {attempt_km*1000:.0f}m window",
                                     kg_code)
                    progress.save()

                # Skip slow Copernicus on tiny windows — 10m data
                # isn't useful at 200m and API timeouts eat the budget.
                use_cop = include_cop and (attempt_km is None or attempt_km >= 0.5)

                pool = multiprocessing.Pool(processes=1)
                try:
                    async_result = pool.apply_async(
                        process_one_kg, args=(kg,),
                        kwds={"include_copernicus": use_cop,
                              "max_km": attempt_km})
                    try:
                        result = async_result.get(timeout=KG_TIMEOUT_SECONDS)
                        break  # success — exit retry ladder
                    except multiprocessing.TimeoutError:
                        _step_monitor_stop.set()
                        step_thread.join(timeout=3)
                        # Read last step from file
                        last_step = "unknown"
                        try:
                            sd = json.loads((DATA_DIR / "current_step.json").read_text())
                            last_step = sd.get("step", "unknown")
                        except Exception:
                            pass
                        pool.terminate()
                        pool.join()

                        if attempt_idx >= len(attempt_windows) - 1:
                            # Exhausted all retries
                            log.error("KG %s: TIMEOUT at step %s after all retries — permanent fail",
                                      kg_code, last_step)
                            failed_kgs.add(kg_code)
                            _save_failed_kgs(failed_kgs)
                            progress.add_failure(kg_code, kg_name,
                                                 f"timeout at {last_step} (all retries)",
                                                 last_step)
                            progress.add_log("error",
                                             f"KG {kg_code} timed out at {last_step} after all retries",
                                             kg_code)
                            progress.save()
                            result = None
                            break
                        else:
                            next_km = attempt_windows[attempt_idx + 1]
                            log.warning("  → TIMEOUT after %d min at step %s (%.1fkm) — will retry at %.0fm",
                                        KG_TIMEOUT_SECONDS // 60, last_step,
                                        attempt_km or MAX_KG_AREA_KM, next_km * 1000)
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
                    kg_code, kg_name, result["files"], manifest)

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
                progress.add_log(
                    "success",
                    f"KG {kg_code} done in {elapsed_kg:.0f}s "
                    f"({result.get('n_segments', 0)} segs, "
                    f"{result.get('n_parcels', 0)} par, "
                    f"{result.get('n_buildings', 0)} bldg)",
                    kg_code,
                )
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
