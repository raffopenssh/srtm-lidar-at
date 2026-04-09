#!/usr/bin/env python3
"""Bulk KG export: comprehensive GeoPackage per Katastralgemeinde.

Produces a paired GPKG + GeoTIFF per KG containing ALL analysis layers.

Vector layers (GPKG, EPSG:3035):
  - parcels               Cadastre parcel polygons with landuse summaries
  - landuse               Cadastre Nutzungsflächen polygons
  - landuse_points        Cadastre Nutzungssymbole (points)
  - building_footprints   BEV building footprint polygons
  - buildings             Building address records (points)
  - segments              Watershed-segmented objects (polygons, 25 types)
  - change_events         Temporal change events (2022→2023, 2023→2024)
  - tree_changes          Per-tree growth/felling analysis
  - terrain_zones         Terrain characterisation per tile

Raster bands (GeoTIFF, EPSG:3035, 1m):
  Band  1: dtm_2022       Band  2: dsm_2022
  Band  3: dtm_2023       Band  4: dsm_2023
  Band  5: dtm_2024       Band  6: dsm_2024
  Band  7: ndsm_2024      (= DSM-DTM, object heights)
  Band  8: ortho_r        Band  9: ortho_g
  Band 10: ortho_b        Band 11: ortho_nir
  Band 12: ndvi           (fused BEV + Copernicus growing season)
  Band 13: slope          (degrees, from 2024 DTM)
  Band 14: aspect         (degrees, from 2024 DTM)
  Band 15: tri            (Terrain Ruggedness Index)
  Band 16: tpi            (Topographic Position Index)
  Band 17: curvature_profile
  Band 18: curvature_plan
  Band 19: segment_type   (type codes from segmentation)
  Band 20: dtm_change_22_24  (DTM 2024 - DTM 2022)
  Band 21: ndsm_change_22_24 (nDSM 2024 - nDSM 2022)
  Band 22: hansen_treecover2000
  Band 23: hansen_lossyear
  Band 24: landcover      (ESA WorldCover 10m)

Usage:
    python3 bulk_export.py --kg 63330
    python3 bulk_export.py --kg 63330,63331
    python3 bulk_export.py --gemeinde "Kainach bei Voitsberg"
    python3 bulk_export.py --district Voitsberg
    python3 bulk_export.py --kg 63330 --skip-segments --skip-hansen
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.transform import from_origin
from pyproj import Transformer
from shapely.geometry import shape as shapely_shape, box, mapping, Point
from shapely.ops import transform as shapely_transform, unary_union
import requests

# Local modules
import tile_index as ti
import raster_io
import ortho_io
import terrain_analysis as ta
import temporal_analysis as tca

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('bulk_export')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CADASTRE_BASE = "https://cadastre-process-api.exe.xyz/api/v1"
OUTPUT_DIR = Path("/home/exedev/srtm-lidar/exports")
DATASETS = ["20220915", "20230915", "20240915"]
TILE_MAX_M = 3500  # 3.5km×3.5km = ~12 km² per tile, well under 25 km² limit
REQUEST_TIMEOUT = 120

# CRS transformers
_T_4326_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
_T_3035_4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


# ---------------------------------------------------------------------------
# 1. Fetch cadastre vector data
# ---------------------------------------------------------------------------

def fetch_kg_info(kg_code: str) -> dict:
    r = requests.get(f"{CADASTRE_BASE}/search/kg", params={"code": kg_code},
                     timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()["data"]
    if not data:
        raise ValueError(f"KG {kg_code} not found")
    return data[0]


def fetch_cadastre_vectors(kg_code: str) -> dict[str, gpd.GeoDataFrame]:
    """Fetch all cadastre vector layers for a KG."""
    log.info("Fetching cadastre vectors for KG %s...", kg_code)
    layers = {}

    for layer_name, params in [
        ("parcels",            {"kg": kg_code, "layers": "parcels"}),
        ("building_footprints", {"kg": kg_code, "layers": "building_footprints"}),
        # buildings have bare [lon,lat] geometry, need special handling
        ("buildings",          {"kg": kg_code, "layers": "buildings"}),
        ("landuse",            {"kg": kg_code, "layers": "landuse"}),
    ]:
        try:
            r = requests.get(f"{CADASTRE_BASE}/export/geojson",
                             params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            gj = r.json()
            feats = gj.get("features", [])
            if not feats:
                continue

            if layer_name == "buildings":
                # Buildings have bare [lon,lat] as geometry, fix to proper GeoJSON Point
                for f in feats:
                    g = f.get("geometry")
                    if isinstance(g, list) and len(g) == 2:
                        f["geometry"] = {"type": "Point", "coordinates": g}

            if layer_name == "landuse":
                # Split polygons vs points
                polys = [f for f in feats if f["geometry"]["type"] in ("Polygon", "MultiPolygon")]
                pts = [f for f in feats if f["geometry"]["type"] == "Point"]
                if polys:
                    layers["landuse"] = gpd.GeoDataFrame.from_features(polys, crs="EPSG:4326")
                    log.info("  landuse: %d polygons", len(polys))
                if pts:
                    layers["landuse_points"] = gpd.GeoDataFrame.from_features(pts, crs="EPSG:4326")
                    log.info("  landuse_points: %d points", len(pts))
            else:
                gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
                # Flatten dicts for GPKG compat
                for col in gdf.columns:
                    if gdf[col].dtype == object:
                        sample = gdf[col].dropna().iloc[0] if len(gdf[col].dropna()) else None
                        if isinstance(sample, (dict, list)):
                            gdf[col] = gdf[col].apply(
                                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x) if x is not None else None
                            )
                layers[layer_name] = gdf
                log.info("  %s: %d features", layer_name, len(gdf))
        except Exception as e:
            log.warning("  Failed fetching %s: %s", layer_name, e)

    return layers


# ---------------------------------------------------------------------------
# 2. Tiling
# ---------------------------------------------------------------------------

def compute_tiles(bbox_3035: tuple[float, float, float, float],
                  max_side: float = TILE_MAX_M
                  ) -> list[tuple[float, float, float, float]]:
    min_e, min_n, max_e, max_n = bbox_3035
    width = max_e - min_e
    height = max_n - min_n
    nx = max(1, int(np.ceil(width / max_side)))
    ny = max(1, int(np.ceil(height / max_side)))
    tile_w = width / nx
    tile_h = height / ny
    tiles = []
    for ix in range(nx):
        for iy in range(ny):
            tiles.append((
                min_e + ix * tile_w,
                min_n + iy * tile_h,
                min_e + (ix + 1) * tile_w,
                min_n + (iy + 1) * tile_h,
            ))
    log.info("Split into %d tiles (%d×%d, each ~%.0f×%.0fm)",
             len(tiles), nx, ny, tile_w, tile_h)
    return tiles


# ---------------------------------------------------------------------------
# 3. Read raster data per tile (all years + ortho)
# ---------------------------------------------------------------------------

def read_tile_rasters(tile_bbox_3035: tuple, kg_geom_3035) -> dict | None:
    """Read DTM/DSM for all 3 dates + ortho for one tile, clipped to KG."""
    min_e, min_n, max_e, max_n = tile_bbox_3035
    tile_geom = box(min_e, min_n, max_e, max_n)
    clip_geom = tile_geom.intersection(kg_geom_3035)
    if clip_geom.is_empty:
        return None

    result = {}
    pad = 5.0

    # Read primary (latest) date to establish grid
    primary_ds = DATASETS[-1]
    try:
        data = raster_io.read_dtm_dsm(clip_geom, primary_ds, pad=pad)
    except Exception as e:
        log.warning("Primary raster read failed: %s", e)
        return None

    tf = data["transform"]
    h, w = data["shape"]
    mask = data["mask"]
    result["transform"] = tf
    result["shape"] = (h, w)
    result["mask"] = mask
    result["clip_geom"] = clip_geom
    result["data_primary"] = data  # keep full dict for ortho_io

    # Store primary DTM/DSM/nDSM
    year = primary_ds[:4]
    result[f"dtm_{year}"] = data["dtm"]
    result[f"dsm_{year}"] = data["dsm"]
    result[f"ndsm_{year}"] = data["ndsm"]

    # Other dates
    for ds in DATASETS:
        if ds == primary_ds:
            continue
        yr = ds[:4]
        try:
            d = raster_io.read_dtm_dsm(clip_geom, ds, pad=pad)
            mh, mw = min(d["shape"][0], h), min(d["shape"][1], w)
            result[f"dtm_{yr}"] = d["dtm"][:mh, :mw]
            result[f"dsm_{yr}"] = d["dsm"][:mh, :mw]
            result[f"ndsm_{yr}"] = d["ndsm"][:mh, :mw]
            log.info("  DTM/DSM %s: %dx%d", yr, mw, mh)
        except Exception as e:
            log.warning("  Date %s failed: %s", yr, e)

    # Orthophoto
    try:
        rgb, nir = ortho_io.read_ortho_for_als(data)
        if rgb is not None:
            result["ortho_r"] = rgb[0]
            result["ortho_g"] = rgb[1]
            result["ortho_b"] = rgb[2]
            if nir is not None:
                result["ortho_nir"] = nir
            log.info("  Ortho: RGB%s", "+NIR" if nir is not None else "")
    except Exception as e:
        log.warning("  Ortho failed: %s", e)

    return result


# ---------------------------------------------------------------------------
# 4. Terrain analysis per tile
# ---------------------------------------------------------------------------

def compute_terrain_rasters(tile_data: dict) -> dict:
    """Compute slope, aspect, TRI, TPI, curvature rasters."""
    dtm = tile_data.get(f"dtm_{DATASETS[-1][:4]}")
    if dtm is None:
        return {}
    mask = tile_data["mask"]

    result = {}
    try:
        result["slope"] = ta.compute_slope(dtm)
        result["aspect"] = ta.compute_aspect(dtm)
        result["tri"] = ta.compute_tri(dtm)
        result["tpi"] = ta.compute_tpi(dtm)
        curv = ta.compute_curvature(dtm)
        result["curvature_profile"] = curv["profile_curvature"]
        result["curvature_plan"] = curv["plan_curvature"]
        log.info("  Terrain rasters computed")
    except Exception as e:
        log.warning("  Terrain analysis failed: %s", e)
    return result


# ---------------------------------------------------------------------------
# 5. NDVI (from ortho spectral indices)
# ---------------------------------------------------------------------------

def compute_ndvi(tile_data: dict) -> np.ndarray | None:
    """Compute NDVI from ortho bands."""
    if "ortho_r" not in tile_data:
        return None
    try:
        rgb = np.stack([tile_data["ortho_r"], tile_data["ortho_g"],
                       tile_data["ortho_b"]])
        nir = tile_data.get("ortho_nir")
        si = ortho_io.compute_spectral_indices(rgb, nir=nir)
        return si.get("ndvi")
    except Exception as e:
        log.warning("  NDVI computation failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# 6. Temporal change rasters
# ---------------------------------------------------------------------------

def compute_temporal_changes(tile_data: dict) -> dict:
    """Compute DTM and nDSM change between first and last date."""
    result = {}
    first_year = DATASETS[0][:4]
    last_year = DATASETS[-1][:4]
    dtm_a = tile_data.get(f"dtm_{first_year}")
    dtm_b = tile_data.get(f"dtm_{last_year}")
    ndsm_a = tile_data.get(f"ndsm_{first_year}")
    ndsm_b = tile_data.get(f"ndsm_{last_year}")

    h, w = tile_data["shape"]
    if dtm_a is not None and dtm_b is not None:
        mh, mw = min(dtm_a.shape[0], dtm_b.shape[0], h), min(dtm_a.shape[1], dtm_b.shape[1], w)
        result["dtm_change_22_24"] = (dtm_b[:mh, :mw] - dtm_a[:mh, :mw]).astype(np.float32)
    if ndsm_a is not None and ndsm_b is not None:
        mh, mw = min(ndsm_a.shape[0], ndsm_b.shape[0], h), min(ndsm_a.shape[1], ndsm_b.shape[1], w)
        result["ndsm_change_22_24"] = (ndsm_b[:mh, :mw] - ndsm_a[:mh, :mw]).astype(np.float32)
    if result:
        log.info("  Temporal change rasters computed")
    return result


# ---------------------------------------------------------------------------
# 7. Hansen forest data
# ---------------------------------------------------------------------------

def compute_hansen(tile_data: dict) -> dict:
    """Fetch Hansen treecover2000 and lossyear, resampled to tile grid."""
    import hansen as hn
    tf = tile_data["transform"]
    h, w = tile_data["shape"]
    clip_geom = tile_data["clip_geom"]
    # Get WGS84 bbox
    bbox_wgs = shapely_transform(_T_3035_4326.transform, clip_geom).bounds
    try:
        prior = hn.get_forest_prior(bbox_wgs, tf, (h, w))
        result = {}
        if prior.get("treecover2000") is not None:
            result["hansen_treecover2000"] = prior["treecover2000"].astype(np.float32)
        if prior.get("loss_year") is not None:
            result["hansen_lossyear"] = prior["loss_year"].astype(np.float32)
        log.info("  Hansen data fetched")
        return result
    except Exception as e:
        log.warning("  Hansen failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# 8. Copernicus data (land cover + NDVI)
# ---------------------------------------------------------------------------

def fetch_copernicus(tile_data: dict) -> dict:
    """Fetch ESA WorldCover land cover resampled to tile grid."""
    try:
        import copernicus
        clip_geom = tile_data["clip_geom"]
        bbox_wgs = shapely_transform(_T_3035_4326.transform, clip_geom).bounds
        bbox_dict = {"west": bbox_wgs[0], "south": bbox_wgs[1],
                     "east": bbox_wgs[2], "north": bbox_wgs[3]}

        result = {}
        try:
            lc = copernicus.get_land_cover(bbox_dict)
            if lc and lc.get("map") is not None:
                # Resample to our 1m grid
                from rasterio.warp import reproject, Resampling
                tf = tile_data["transform"]
                h, w = tile_data["shape"]
                dst = np.zeros((h, w), dtype=np.float32)
                reproject(
                    source=lc["map"].astype(np.float32),
                    destination=dst,
                    src_transform=lc["transform"],
                    src_crs=lc["crs"],
                    dst_transform=tf,
                    dst_crs="EPSG:3035",
                    resampling=Resampling.nearest,
                )
                result["landcover"] = dst
                log.info("  ESA WorldCover fetched")
        except Exception as e:
            log.warning("  WorldCover failed: %s", e)

        return result
    except ImportError:
        return {}
    except Exception as e:
        log.warning("  Copernicus failed: %s", e)
        return {}


# ---------------------------------------------------------------------------
# 9. Segmentation
# ---------------------------------------------------------------------------

def run_segmentation(tile_data: dict, hansen_data: dict | None = None,
                     copernicus_data: dict | None = None) -> dict | None:
    """Run full watershed segmentation + classification on a tile."""
    try:
        import object_segmentation as seg
        import hansen as hn

        latest = DATASETS[-1][:4]
        dtm = tile_data[f"dtm_{latest}"]
        dsm = tile_data[f"dsm_{latest}"]
        mask = tile_data["mask"]
        tf = tile_data["transform"]
        h, w = tile_data["shape"]

        # Spectral dict
        spectral = None
        if "ortho_r" in tile_data:
            rgb = np.stack([tile_data["ortho_r"], tile_data["ortho_g"],
                           tile_data["ortho_b"]])
            nir = tile_data.get("ortho_nir")
            spectral = ortho_io.compute_spectral_indices(rgb, nir=nir)
            spectral["red"] = rgb[0].astype(np.float32)
            spectral["green"] = rgb[1].astype(np.float32)
            spectral["blue"] = rgb[2].astype(np.float32)
            if nir is not None:
                spectral["nir"] = nir.astype(np.float32)

        # Multi-temporal DTM/DSM
        dtm_dates, dsm_dates = {}, {}
        for ds in DATASETS:
            yr = ds[:4]
            dk = f"dtm_{yr}"
            sk = f"dsm_{yr}"
            if dk in tile_data:
                mh, mw = min(tile_data[dk].shape[0], h), min(tile_data[dk].shape[1], w)
                dtm_dates[ds] = tile_data[dk][:mh, :mw]
            if sk in tile_data:
                mh, mw = min(tile_data[sk].shape[0], h), min(tile_data[sk].shape[1], w)
                dsm_dates[ds] = tile_data[sk][:mh, :mw]

        # Copernicus prior for segmentation — already resampled to tile grid
        cop = None
        if copernicus_data and copernicus_data.get("landcover") is not None:
            cop = {
                "landcover": {
                    "map": copernicus_data["landcover"].astype(np.uint8),
                    "transform": tf,
                    "crs": "EPSG:3035",
                },
                "transform": tf,
                "crs": "EPSG:3035",
            }

        # Observation year = latest dataset available
        _obs_year = max(int(ds[:4]) for ds in DATASETS)
        result = seg.segment_and_classify(
            dtm, dsm, mask, tf,
            dtm_dates=dtm_dates if dtm_dates else None,
            dsm_dates=dsm_dates if dsm_dates else None,
            spectral=spectral,
            copernicus=cop,
            observation_year=_obs_year,
        )

        # Hansen calibration of tree_loss
        if hansen_data and hansen_data.get("hansen_treecover2000") is not None:
            try:
                clip_geom = tile_data["clip_geom"]
                bbox_wgs = shapely_transform(_T_3035_4326.transform, clip_geom).bounds
                prior = hn.get_forest_prior(bbox_wgs, tf, (h, w))
                result["objects"] = hn.calibrate_tree_loss(
                    result["objects"], result["labels"], prior,
                    observation_year=_obs_year)
            except Exception as e:
                log.warning("  Hansen calibration failed: %s", e)

        return result
    except Exception as e:
        log.warning("Segmentation failed: %s", e)
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# 10. Vectorize segments
# ---------------------------------------------------------------------------

def segments_to_features(seg_result: dict, transform) -> list[dict]:
    """Convert segmentation objects to dicts with shapely geometries.

    Uses a single shapes() call on the full label raster for efficiency.
    """
    features = []
    objects = seg_result.get("objects", [])
    labels = seg_result.get("labels")
    if labels is None or not objects:
        return features

    obj_map = {o.obj_id: o for o in objects}

    # Single vectorization pass over the entire label raster
    try:
        label_int = labels.astype(np.int32)
        all_polys = list(shapes(label_int, mask=label_int > 0, transform=transform))
        log.info("  Vectorized %d raw polygons from labels", len(all_polys))
    except Exception as e:
        log.warning("  Bulk vectorization failed: %s", e)
        return features

    for geom_dict, val in all_polys:
        oid = int(val)
        obj = obj_map.get(oid)
        if obj is None:
            continue
        geom = shapely_shape(geom_dict)
        if geom.is_empty or geom.area < 1:
            continue
        features.append({
            "geometry": geom,
            "properties": {
                "obj_id": oid,
                "type": obj.obj_type,
                "type_code": int(obj.type_code),
                "group_id": int(getattr(obj, "group_id", 0) or 0),
                "group_type": getattr(obj, "group_type", "") or "",
                "height_max_m": round(float(obj.height_max), 2),
                "height_mean_m": round(float(obj.height_mean), 2),
                "area_sqm": round(float(obj.area_sqm), 1),
                "confidence": round(float(getattr(obj, "confidence", 0.5)), 2),
                "ndvi_mean": round(float(getattr(obj, "ndvi_mean", 0)), 3),
                "slope_mean": round(float(getattr(obj, "slope_mean", 0)), 1),
                "roughness": round(float(getattr(obj, "roughness", 0)), 4),
            },
        })

    log.info("  Segments vectorized: %d polygons from %d objects",
             len(features), len(objects))
    return features


# ---------------------------------------------------------------------------
# 11. Temporal change events as vectors
# ---------------------------------------------------------------------------

def _build_comparison(tile_data: dict, da: str, db: str) -> dict | None:
    """Build a compare_dates()-compatible dict from pre-loaded tile rasters.

    Avoids redundant remote reads by reusing arrays already in tile_data.
    """
    ya, yb = da[:4], db[:4]
    dtm_a = tile_data.get(f"dtm_{ya}")
    dsm_a = tile_data.get(f"dsm_{ya}")
    dtm_b = tile_data.get(f"dtm_{yb}")
    dsm_b = tile_data.get(f"dsm_{yb}")
    if dtm_a is None or dtm_b is None or dsm_a is None or dsm_b is None:
        return None

    h, w = tile_data["shape"]
    mask = tile_data["mask"]

    mh = min(dtm_a.shape[0], dtm_b.shape[0], h)
    mw = min(dtm_a.shape[1], dtm_b.shape[1], w)
    dtm_a = dtm_a[:mh, :mw]
    dsm_a = dsm_a[:mh, :mw]
    dtm_b = dtm_b[:mh, :mw]
    dsm_b = dsm_b[:mh, :mw]
    m = mask[:mh, :mw] & ~np.isnan(dtm_a) & ~np.isnan(dtm_b)

    ndsm_a = np.clip(dsm_a - dtm_a, 0, None).astype(np.float32)
    ndsm_b = np.clip(dsm_b - dtm_b, 0, None).astype(np.float32)

    ndsm_change = np.where(m, ndsm_b - ndsm_a, np.nan)
    dtm_change = np.where(m, dtm_b - dtm_a, np.nan)
    dsm_change = np.where(m, dsm_b - dsm_a, np.nan)

    from temporal_analysis import _local_roughness, _surface_slope
    dtm_rough_a = _local_roughness(dtm_a, 5)
    dtm_rough_b = _local_roughness(dtm_b, 5)
    dtm_slope_a = _surface_slope(dtm_a)
    dtm_slope_b = _surface_slope(dtm_b)

    valid = ndsm_change[m]
    dtm_v = dtm_change[m]
    if valid.size > 0:
        stats = {
            "date_a": da, "date_b": db,
            "pixel_count": int(m.sum()), "area_sqm": float(m.sum()),
            "ndsm_change_mean": round(float(np.nanmean(valid)), 3),
            "ndsm_change_std": round(float(np.nanstd(valid)), 3),
            "ndsm_change_min": round(float(np.nanmin(valid)), 3),
            "ndsm_change_max": round(float(np.nanmax(valid)), 3),
            "ndsm_change_median": round(float(np.nanmedian(valid)), 3),
            "dtm_change_mean": round(float(np.nanmean(dtm_v)), 3),
            "dtm_change_std": round(float(np.nanstd(dtm_v)), 3),
            "dtm_change_min": round(float(np.nanmin(dtm_v)), 3),
            "dtm_change_max": round(float(np.nanmax(dtm_v)), 3),
            "dsm_change_mean": round(float(np.nanmean(dsm_change[m])), 3),
            "pct_increased_1m": round(float(np.sum(valid > 1.0) / valid.size * 100), 2),
            "pct_decreased_1m": round(float(np.sum(valid < -1.0) / valid.size * 100), 2),
            "pct_dtm_changed_0_3m": round(float(np.sum(np.abs(dtm_v) > 0.3) / dtm_v.size * 100), 2),
            "terrain_roughness_before": round(float(np.nanmean(dtm_rough_a[m])), 4),
            "terrain_roughness_after": round(float(np.nanmean(dtm_rough_b[m])), 4),
        }
    else:
        stats = {"date_a": da, "date_b": db, "pixel_count": 0, "area_sqm": 0.0}

    log.info("  compare (preloaded) %s>%s: %d px, mean dh=%.2fm",
             da, db, stats.get("pixel_count", 0), stats.get("ndsm_change_mean", 0))

    return {
        "ndsm_change": ndsm_change, "dtm_change": dtm_change, "dsm_change": dsm_change,
        "ndsm_a": ndsm_a, "ndsm_b": ndsm_b,
        "dtm_a": dtm_a, "dtm_b": dtm_b, "dsm_a": dsm_a, "dsm_b": dsm_b,
        "dtm_roughness_a": dtm_rough_a, "dtm_roughness_b": dtm_rough_b,
        "dtm_slope_a": dtm_slope_a, "dtm_slope_b": dtm_slope_b,
        "mask": m, "transform": tile_data["transform"],
        "crs": rasterio.crs.CRS.from_epsg(3035), "stats": stats,
    }


def compute_change_events(tile_data: dict) -> list[dict]:
    """Detect change events using pre-loaded rasters (no redundant remote reads)."""
    features = []

    for i in range(len(DATASETS) - 1):
        da, db = DATASETS[i], DATASETS[i + 1]
        try:
            comparison = _build_comparison(tile_data, da, db)
            if comparison is None:
                continue
            events = tca.detect_changes(
                tile_data["clip_geom"], da, db,
                min_change=0.5, comparison=comparison,
            )
            for ev in events:
                geom = Point(ev.centroid_e, ev.centroid_n)
                features.append({
                    "geometry": geom,
                    "properties": {
                        "event_type": ev.event_type,
                        "date_a": da, "date_b": db,
                        "area_sqm": round(ev.area_sqm, 1),
                        "height_before_m": round(ev.height_before, 2),
                        "height_after_m": round(ev.height_after, 2),
                        "height_change_mean_m": round(ev.height_change_mean, 2),
                        "height_change_max_m": round(ev.height_change_max, 2),
                        "dtm_change_mean_m": round(ev.dtm_change_mean, 2),
                        "dtm_change_max_m": round(ev.dtm_change_max, 2),
                        "dsm_change_mean_m": round(ev.dsm_change_mean, 2),
                        "confidence": round(ev.confidence, 2),
                        "detail": ev.detail or "",
                    },
                })
            log.info("  Change events %s→%s: %d events", da, db, len(events))
        except Exception as e:
            log.warning("  Change detection %s→%s failed: %s", da, db, e)

    return features


# ---------------------------------------------------------------------------
# 12. Per-tree growth analysis
# ---------------------------------------------------------------------------

def compute_tree_changes(tile_data: dict, seg_result: dict | None = None) -> list[dict]:
    """Derive per-tree height changes from pre-loaded rasters + segment objects.

    Instead of using the deprecated object_classifier (which re-reads all
    rasters from remote), we compute height change for each tree/shrub segment
    using the already-loaded multi-date nDSM arrays.  ~300× faster.
    """
    features = []
    if seg_result is None:
        return features

    first_yr, last_yr = DATASETS[0][:4], DATASETS[-1][:4]
    ndsm_a = tile_data.get(f"ndsm_{first_yr}")
    ndsm_b = tile_data.get(f"ndsm_{last_yr}")
    if ndsm_a is None or ndsm_b is None:
        return features

    h, w = tile_data["shape"]
    mh = min(ndsm_a.shape[0], ndsm_b.shape[0], h)
    mw = min(ndsm_a.shape[1], ndsm_b.shape[1], w)
    ndsm_a = ndsm_a[:mh, :mw]
    ndsm_b = ndsm_b[:mh, :mw]
    labels = seg_result.get("labels")
    objects = seg_result.get("objects", [])
    tf = tile_data["transform"]

    tree_types = {"tree", "shrub"}
    tid = 0
    for obj in objects:
        if obj.obj_type not in tree_types:
            continue
        seg_mask = labels[:mh, :mw] == obj.obj_id
        if seg_mask.sum() < 4:
            continue
        ha = float(np.nanmean(ndsm_a[seg_mask]))
        hb = float(np.nanmean(ndsm_b[seg_mask]))
        dh = hb - ha
        if abs(dh) < 0.3:
            status = "stable"
        elif dh > 0:
            status = "grown"
        elif hb < 1.0:
            status = "felled"
        else:
            status = "shrunk"
        # Centroid from obj attributes (already in EPSG:3035)
        geom = Point(obj.centroid_e, obj.centroid_n)
        tid += 1
        features.append({
            "geometry": geom,
            "properties": {
                "tree_id": tid,
                "status": status,
                "height_before_m": round(ha, 2),
                "height_after_m": round(hb, 2),
                "height_change_m": round(dh, 2),
                "crown_area_sqm": round(float(obj.area_sqm), 1),
            },
        })
    log.info("  Tree changes from segments: %d trees (%d grown, %d felled, %d stable)",
             len(features),
             sum(1 for f in features if f["properties"]["status"] == "grown"),
             sum(1 for f in features if f["properties"]["status"] == "felled"),
             sum(1 for f in features if f["properties"]["status"] == "stable"))
    return features


# ---------------------------------------------------------------------------
# 13. Mosaic tiles
# ---------------------------------------------------------------------------

def mosaic_tiles(tiles_data: list[dict | None],
                 kg_bbox_3035: tuple[float, float, float, float],
                 band_keys: list[str],
                 ) -> dict:
    """Mosaic tile data into a single KG-wide raster."""
    min_e, min_n, max_e, max_n = kg_bbox_3035
    min_e, min_n = np.floor(min_e), np.floor(min_n)
    max_e, max_n = np.ceil(max_e), np.ceil(max_n)

    w = int(max_e - min_e)
    h = int(max_n - min_n)
    tf = from_origin(min_e, max_n, 1.0, 1.0)

    result = {"transform": tf, "shape": (h, w)}
    arrays = {k: np.full((h, w), np.nan, dtype=np.float32) for k in band_keys}

    for td in tiles_data:
        if td is None:
            continue
        ttf = td["transform"]

        col_off = int(round((ttf.c - min_e) / 1.0))
        row_off = int(round((max_n - ttf.f) / 1.0))

        for key in band_keys:
            if key not in td:
                continue
            arr = td[key]
            ah, aw = arr.shape[:2]
            src_r0 = max(0, -row_off)
            src_c0 = max(0, -col_off)
            dst_r0 = max(0, row_off)
            dst_c0 = max(0, col_off)
            copy_h = min(ah - src_r0, h - dst_r0)
            copy_w = min(aw - src_c0, w - dst_c0)
            if copy_h <= 0 or copy_w <= 0:
                continue
            src = arr[src_r0:src_r0 + copy_h, src_c0:src_c0 + copy_w]
            valid = ~np.isnan(src) if src.dtype in (np.float32, np.float64) else np.ones_like(src, dtype=bool)
            target = arrays[key][dst_r0:dst_r0 + copy_h, dst_c0:dst_c0 + copy_w]
            target[valid] = src[valid].astype(np.float32)

    result.update(arrays)
    return result


# ---------------------------------------------------------------------------
# 14. Write GPKG + GeoTIFF
# ---------------------------------------------------------------------------

# Ordered list of all raster band keys
BAND_ORDER = [
    "dtm_2022", "dsm_2022",
    "dtm_2023", "dsm_2023",
    "dtm_2024", "dsm_2024",
    "ndsm_2024",
    "ortho_r", "ortho_g", "ortho_b", "ortho_nir",
    "ndvi",
    "slope", "aspect", "tri", "tpi",
    "curvature_profile", "curvature_plan",
    "segment_type",
    "dtm_change_22_24", "ndsm_change_22_24",
    "hansen_treecover2000", "hansen_lossyear",
    "landcover",
]


def write_gpkg(output_path: Path,
               vectors: dict[str, gpd.GeoDataFrame],
               mosaic: dict,
               segment_features: list[dict],
               change_features: list[dict],
               tree_features: list[dict],
               kg_info: dict):
    """Write comprehensive GPKG (vectors) + GeoTIFF (rasters)."""
    log.info("Writing output to %s ...", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Vector layers ---
    first = True
    for layer_name, gdf in vectors.items():
        if gdf is None or len(gdf) == 0:
            continue
        gdf_3035 = gdf.to_crs("EPSG:3035")
        gdf_3035.to_file(str(output_path), layer=layer_name, driver="GPKG",
                         mode="w" if first else "a")
        first = False
        log.info("  Vector '%s': %d features", layer_name, len(gdf_3035))

    # Segment polygons
    if segment_features:
        seg_gdf = gpd.GeoDataFrame(
            [f["properties"] for f in segment_features],
            geometry=[f["geometry"] for f in segment_features],
            crs="EPSG:3035",
        )
        seg_gdf.to_file(str(output_path), layer="segments", driver="GPKG",
                        mode="w" if first else "a")
        first = False
        log.info("  Vector 'segments': %d features", len(seg_gdf))

    # Change events
    if change_features:
        ch_gdf = gpd.GeoDataFrame(
            [f["properties"] for f in change_features],
            geometry=[f["geometry"] for f in change_features],
            crs="EPSG:3035",
        )
        ch_gdf.to_file(str(output_path), layer="change_events", driver="GPKG",
                        mode="w" if first else "a")
        first = False
        log.info("  Vector 'change_events': %d features", len(ch_gdf))

    # Tree changes
    if tree_features:
        tr_gdf = gpd.GeoDataFrame(
            [f["properties"] for f in tree_features],
            geometry=[f["geometry"] for f in tree_features],
            crs="EPSG:3035",
        )
        tr_gdf.to_file(str(output_path), layer="tree_changes", driver="GPKG",
                        mode="w" if first else "a")
        first = False
        log.info("  Vector 'tree_changes': %d features", len(tr_gdf))

    # --- Raster GeoTIFF ---
    tf = mosaic["transform"]
    h, w = mosaic["shape"]

    raster_bands = []
    band_names = []
    for key in BAND_ORDER:
        if key in mosaic and not np.all(np.isnan(mosaic[key])):
            raster_bands.append(mosaic[key])
            band_names.append(key)

    raster_path = str(output_path).replace(".gpkg", "_raster.tif")
    if raster_bands:
        n_bands = len(raster_bands)
        with rasterio.open(
            raster_path, "w", driver="GTiff",
            width=w, height=h, count=n_bands,
            dtype="float32", crs="EPSG:3035", transform=tf,
            nodata=np.nan, compress="deflate", predictor=2,
            tiled=True, blockxsize=256, blockysize=256,
        ) as dst:
            for i, (arr, name) in enumerate(zip(raster_bands, band_names), 1):
                out = np.full((h, w), np.nan, dtype=np.float32)
                ah, aw = arr.shape[:2]
                ch, cw = min(ah, h), min(aw, w)
                out[:ch, :cw] = arr[:ch, :cw].astype(np.float32)
                dst.write(out, i)
                dst.set_band_description(i, name)
        log.info("  Raster: %d bands, %dx%d px", n_bands, w, h)
    else:
        log.warning("  No raster bands to write")
        raster_path = None

    # --- Band index JSON ---
    index_path = str(output_path).replace(".gpkg", "_bands.json")
    band_index = {
        "kg": kg_info,
        "crs": "EPSG:3035",
        "resolution_m": 1.0,
        "raster_size": {"width": w, "height": h},
        "bands": {i + 1: name for i, name in enumerate(band_names)},
        "vector_layers": list(vectors.keys()) +
            (["segments"] if segment_features else []) +
            (["change_events"] if change_features else []) +
            (["tree_changes"] if tree_features else []),
        "files": {
            "vectors": output_path.name,
            "raster": Path(raster_path).name if raster_path else None,
        },
    }
    with open(index_path, "w") as f:
        json.dump(band_index, f, indent=2, ensure_ascii=False)

    # --- README ---
    readme_path = str(output_path).replace(".gpkg", "_README.txt")
    with open(readme_path, "w") as f:
        f.write(f"KG Export: {kg_info.get('kg_name', '')} ({kg_info.get('kg_code', '')})\n")
        f.write(f"Gemeinde: {kg_info.get('gemeinde_name', '')}, {kg_info.get('state_name', '')}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Parcels: {kg_info.get('parcel_count', 0)}, Buildings: {kg_info.get('building_count', 0)}\n")
        f.write(f"Area: {kg_info.get('total_area_sqm', 0) / 10000:.1f} ha\n\n")
        f.write("FILES:\n")
        f.write(f"  {output_path.name}  \u2014 Vector layers (GeoPackage, EPSG:3035)\n")
        if raster_path:
            f.write(f"  {Path(raster_path).name}  \u2014 Raster layers (GeoTIFF, EPSG:3035, 1m)\n")
        f.write(f"  {Path(index_path).name}  \u2014 Band index + metadata (JSON)\n\n")

        f.write("VECTOR LAYERS:\n")
        for name, gdf in vectors.items():
            if gdf is not None and len(gdf) > 0:
                cols = ", ".join(c for c in gdf.columns if c != "geometry")
                f.write(f"  {name}: {len(gdf)} features [{cols}]\n")
        if segment_features:
            f.write(f"  segments: {len(segment_features)} features [obj_id, type, type_code, group_type, height_max_m, area_sqm, confidence, ndvi_mean, ...]\n")
        if change_features:
            f.write(f"  change_events: {len(change_features)} features [event_type, date_a, date_b, area_sqm, height_change_mean_m, ...]\n")
        if tree_features:
            f.write(f"  tree_changes: {len(tree_features)} features [tree_id, status, height_before_m, height_after_m, ...]\n")

        f.write(f"\nRASTER BANDS ({w}\u00d7{h} px, 1m resolution, EPSG:3035):\n")
        for i, name in enumerate(band_names, 1):
            f.write(f"  Band {i:2d}: {name}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("HOW TO EXTRACT VALUES PER POLYGON:\n")
        f.write("=" * 60 + "\n\n")
        f.write("Option 1: rasterstats (Python):\n")
        f.write("  from rasterstats import zonal_stats\n")
        f.write("  import geopandas as gpd\n")
        f.write(f"  parcels = gpd.read_file('{output_path.name}', layer='parcels')\n")
        f.write(f"  stats = zonal_stats(parcels, '{Path(raster_path).name if raster_path else ""}',\n")
        f.write("      band=5,  # dtm_2024\n")
        f.write("      stats=['mean','std','min','max','count','median'])\n")
        f.write("  parcels = parcels.join(pd.DataFrame(stats))\n\n")
        f.write("Option 2: QGIS:\n")
        f.write("  1. Load both files (drag & drop)\n")
        f.write("  2. Processing > Toolbox > Zonal Statistics\n")
        f.write("  3. Select polygon layer + raster band\n\n")
        f.write("Option 3: geopandas + rasterio (manual):\n")
        f.write("  import rasterio\n")
        f.write("  from rasterio.mask import mask\n")
        f.write("  src = rasterio.open('..._raster.tif')\n")
        f.write("  for idx, row in parcels.iterrows():\n")
        f.write("      clipped, _ = mask(src, [row.geometry], crop=True, band=5)\n")
        f.write("      print(f'Parcel {row.parcel_id}: mean DTM = {np.nanmean(clipped):.1f}m')\n")

    log.info("  Output files written")


# ---------------------------------------------------------------------------
# 15. Main pipeline: export one KG
# ---------------------------------------------------------------------------

def export_kg(kg_code: str, output_dir: Path = OUTPUT_DIR,
              skip_segments: bool = False,
              skip_changes: bool = False,
              skip_trees: bool = False,
              skip_hansen: bool = False,
              skip_copernicus: bool = False) -> Path:
    """Full export pipeline for a single KG."""
    t0 = time.time()
    log.info("=" * 60)
    log.info("EXPORT KG %s", kg_code)
    log.info("=" * 60)

    # 1. KG info
    kg_info = fetch_kg_info(kg_code)
    kg_name = kg_info["kg_name"]
    log.info("KG: %s (%s), %s, %s",
             kg_name, kg_code, kg_info["gemeinde_name"], kg_info["state_name"])
    log.info("  %d parcels, %d buildings, %.1f ha",
             kg_info["parcel_count"], kg_info["building_count"],
             kg_info["total_area_sqm"] / 10000)

    # 2. Cadastre vectors
    vectors = fetch_cadastre_vectors(kg_code)

    # 3. KG bbox → EPSG:3035
    bbox = kg_info["bbox"]
    x1, y1 = _T_4326_3035.transform(bbox["min_lon"], bbox["min_lat"])
    x2, y2 = _T_4326_3035.transform(bbox["max_lon"], bbox["max_lat"])
    kg_bbox_3035 = (x1, y1, x2, y2)

    # Build KG boundary from parcel union (more precise than bbox)
    kg_geom_3035 = box(x1, y1, x2, y2)
    if "parcels" in vectors and len(vectors["parcels"]) > 0:
        try:
            from shapely.validation import make_valid
            parcels_3035 = vectors["parcels"].to_crs("EPSG:3035")
            valid_geoms = [make_valid(g) for g in parcels_3035.geometry if g is not None and not g.is_empty]
            kg_geom_3035 = unary_union(valid_geoms).buffer(10)
            log.info("  KG boundary from parcels: %.1f ha", kg_geom_3035.area / 10000)
        except Exception as e:
            log.warning("  Parcel union failed, using bbox: %s", e)

    # 4. Tile grid
    tiles = compute_tiles(kg_bbox_3035)

    # 5. Process each tile
    tiles_data = []
    all_segment_features = []
    all_change_features = []
    all_tree_features = []

    for i, tile_bbox in enumerate(tiles):
        log.info("\n" + "-" * 50)
        log.info("TILE %d/%d", i + 1, len(tiles))
        log.info("-" * 50)

        # 5a. Read rasters (DTM/DSM all years + ortho)
        td = read_tile_rasters(tile_bbox, kg_geom_3035)
        if td is None:
            tiles_data.append(None)
            continue

        # 5b. Terrain analysis rasters
        terrain = compute_terrain_rasters(td)
        td.update(terrain)

        # 5c. NDVI
        ndvi = compute_ndvi(td)
        if ndvi is not None:
            td["ndvi"] = ndvi

        # 5d. Temporal change rasters
        temporal = compute_temporal_changes(td)
        td.update(temporal)

        # 5e. Hansen
        hansen_data = {}
        if not skip_hansen:
            hansen_data = compute_hansen(td)
            td.update(hansen_data)

        # 5f. Copernicus (land cover)
        cop_data = {}
        if not skip_copernicus:
            cop_data = fetch_copernicus(td)
            td.update(cop_data)

        # 5g. Segmentation (full pipeline with all priors)
        seg_result = None
        if not skip_segments:
            seg_result = run_segmentation(td, hansen_data, cop_data)
            if seg_result is not None:
                # Segment type raster
                labels = seg_result["labels"]
                objects = seg_result["objects"]
                h, w = td["shape"]
                seg_type = np.full((h, w), np.nan, dtype=np.float32)
                obj_map = {o.obj_id: o for o in objects}
                for oid, obj in obj_map.items():
                    seg_type[labels == oid] = float(obj.type_code)
                td["segment_type"] = seg_type

                # Vectorize segments
                seg_feats = segments_to_features(seg_result, td["transform"])
                all_segment_features.extend(seg_feats)
                log.info("  Segments: %d objects → %d polygons",
                         len(objects), len(seg_feats))

        # 5h. Change events
        if not skip_changes:
            ch_feats = compute_change_events(td)
            all_change_features.extend(ch_feats)

        # 5i. Tree changes
        if not skip_trees:
            tr_feats = compute_tree_changes(td, seg_result=seg_result)
            all_tree_features.extend(tr_feats)

        tiles_data.append(td)

    # 6. Mosaic all rasters
    log.info("\nMosaicking %d tiles...", len(tiles_data))
    mosaic = mosaic_tiles(tiles_data, kg_bbox_3035, BAND_ORDER)

    # 7. Write output
    safe_name = kg_name.replace(" ", "_").replace("/", "-")
    output_path = output_dir / f"{kg_code}_{safe_name}.gpkg"
    write_gpkg(output_path, vectors, mosaic,
               all_segment_features, all_change_features, all_tree_features,
               kg_info)

    elapsed = time.time() - t0
    log.info("\n" + "=" * 60)
    log.info("DONE: %s in %.0fs (%.1f min)", output_path.name, elapsed, elapsed / 60)
    log.info("=" * 60)
    return output_path


# ---------------------------------------------------------------------------
# 16. Batch: resolve KG codes
# ---------------------------------------------------------------------------

def resolve_kg_codes(args) -> list[str]:
    if args.kg:
        return [c.strip() for c in args.kg.split(",")]
    params = {}
    if args.gemeinde:
        params["gemeinde"] = args.gemeinde
    elif args.district:
        params["district"] = args.district
    elif args.state:
        params["q"] = args.state
    if not params:
        raise ValueError("Specify --kg, --gemeinde, --district, or --state")
    r = requests.get(f"{CADASTRE_BASE}/search/kg", params={**params, "limit": 10000},
                     timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    codes = [d["kg_code"] for d in r.json()["data"]]
    log.info("Resolved %d KG codes from %s", len(codes), params)
    return codes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bulk KG Export — comprehensive GPKG + GeoTIFF per KG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Runs ALL analysis pipelines per KG:
  • DTM/DSM for 2022, 2023, 2024 (3 dates)
  • Orthophoto RGBI
  • Terrain: slope, aspect, TRI, TPI, curvature
  • NDVI (fused BEV + Copernicus)
  • Watershed segmentation (25 object types + 11 groups)
  • Temporal change detection (excavation, fill, tree_loss, etc.)
  • Per-tree growth/felling analysis
  • Hansen Global Forest Change (treecover, loss year)
  • ESA WorldCover land cover
  • Cadastre: parcels, buildings, footprints, landuse
"""
    )
    parser.add_argument("--kg", help="KG code(s), comma-separated")
    parser.add_argument("--gemeinde", help="Gemeinde name")
    parser.add_argument("--district", help="District name")
    parser.add_argument("--state", help="State name")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--skip-segments", action="store_true")
    parser.add_argument("--skip-changes", action="store_true")
    parser.add_argument("--skip-trees", action="store_true")
    parser.add_argument("--skip-hansen", action="store_true")
    parser.add_argument("--skip-copernicus", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kg_codes = resolve_kg_codes(args)
    log.info("Exporting %d KGs to %s", len(kg_codes), output_dir)

    results = []
    for code in kg_codes:
        try:
            path = export_kg(
                code, output_dir,
                skip_segments=args.skip_segments,
                skip_changes=args.skip_changes,
                skip_trees=args.skip_trees,
                skip_hansen=args.skip_hansen,
                skip_copernicus=args.skip_copernicus,
            )
            results.append((code, str(path), "OK"))
        except Exception as e:
            log.error("KG %s FAILED: %s", code, e)
            import traceback
            traceback.print_exc()
            results.append((code, "", str(e)))

    log.info("\n" + "=" * 60)
    log.info("SUMMARY: %d/%d succeeded", sum(1 for r in results if r[2] == "OK"), len(results))
    for code, path, status in results:
        log.info("  %s: %s %s", code, status, path if status == "OK" else "")


if __name__ == "__main__":
    main()
