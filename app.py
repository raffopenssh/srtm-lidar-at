"""Austrian LIDAR & Orthophoto Analysis API.

Endpoints:
  1. POST /api/v1/elevation        — Enrich features with DSM/DTM elevation
  2. POST /api/v1/terrain          — Terrain characterisation (slope, ruggedness, …)
  3. POST /api/v1/objects          — Object detection & classification summary
  4. POST /api/v1/objects/raster   — Classified object raster (GeoTIFF)
  5. POST /api/v1/changes          — Temporal change detection between ALS dates
  6. POST /api/v1/changes/trees    — Per-tree growth / felling analysis
  7. POST /api/v1/changes/summary  — Multi-epoch change summary
  8. GET  /api/v1/info             — Datasets, object types, event types
  9. GET  /api/v1/docs/llm.txt     — Machine-readable API reference
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from flask import Flask, request, jsonify, send_file, Response
from shapely.geometry import mapping, shape, Point, LineString as SLineString

import tile_index as ti
import raster_io
import terrain_analysis as ta
import landscape_classifier as oc  # new landscape-focused classifier
import object_segmentation as seg  # watershed-based segmentation
import hansen  # Hansen Global Forest Change calibration
import temporal_analysis as tca
import geo_parse

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')

MAX_AREA_SQM = 25_000_000  # 25 km²

# ---------------------------------------------------------------------------
# Segment progress tracking  (file-backed so all gunicorn workers can read)
# ---------------------------------------------------------------------------
_PROGRESS_DIR = Path('/tmp/segment_progress')
_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

def _progress_set(task_id: str, step: str, detail: str = ""):
    """Update progress for a running segment task."""
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    try:
        t0 = 0.0
        if p.exists():
            try:
                t0 = json.loads(p.read_text()).get('t0', 0.0)
            except Exception:
                pass
        p.write_text(json.dumps(dict(step=step, detail=detail, t0=t0,
                                     updated=time.time())))
    except Exception:
        pass

def _progress_start(task_id: str):
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    p.write_text(json.dumps(dict(step='starting', detail='',
                                 t0=time.time(), updated=time.time())))

def _progress_end(task_id: str):
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

@app.route('/api/v1/segment/progress')
def segment_progress():
    """Poll progress of a running segment task."""
    task_id = request.args.get('task_id', '')
    p = _PROGRESS_DIR / f"{task_id}.json"
    if not task_id or not p.exists():
        return jsonify(dict(active=False, step='', detail='', elapsed=0))
    try:
        info = json.loads(p.read_text())
        return jsonify(dict(active=True, step=info.get('step', ''),
                            detail=info.get('detail', ''),
                            elapsed=round(time.time() - info.get('t0', time.time()), 1)))
    except Exception:
        return jsonify(dict(active=False, step='', detail='', elapsed=0))


@app.route('/api/v1/training/status')
def training_status():
    """Return RF training job status: running, current KG, model info, resource usage."""
    import subprocess, re, pathlib

    result = dict(running=False, current_kg=None, progress=None,
                  model=None, pid=None, cpu_pct=None, ram_mb=None)

    # Find the training process
    try:
        ps = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=5)
        for line in ps.stdout.splitlines():
            if 'python3 train_rf_4000kg.py' in line and 'grep' not in line and 'bash' not in line and 'tee' not in line:
                parts = line.split()
                result['running'] = True
                result['pid'] = int(parts[1])
                result['cpu_pct'] = float(parts[2])
                result['ram_mb'] = round(int(parts[5]) / 1024)  # RSS in KB → MB
                break
    except Exception:
        pass

    # Parse last log lines for current KG and progress
    log_path = pathlib.Path('/tmp/rf_train_4000kg.log')
    if log_path.exists():
        try:
            # Read last 32KB of log (enough to find current KG line)
            with open(log_path, 'rb') as f:
                f.seek(max(0, f.seek(0, 2) - 32768))
                tail = f.read().decode('utf-8', errors='replace')
            lines = tail.strip().splitlines()
            # Find last "Processing KG" line
            for line in reversed(lines):
                m = re.search(r'\[(\d+)/(\d+)\]\s+Processing KG (\d+)\s+\(([^)]+)\)', line)
                if m:
                    result['current_kg'] = dict(
                        index=int(m.group(1)), total=int(m.group(2)),
                        kg_code=m.group(3), kg_name=m.group(4))
                    result['progress'] = f"{m.group(1)}/{m.group(2)}"
                    break
            # Find last successful checkpoint count
            for line in reversed(lines):
                m = re.search(r'already checkpointed, skipping', line)
                if not m:
                    m2 = re.search(r'Checkpoint saved: (\S+)', line)
                    if m2:
                        break
        except Exception:
            pass

    # Checkpoint count
    ckpt_dir = pathlib.Path('/home/exedev/srtm-lidar/rf_training_data/checkpoints')
    if ckpt_dir.exists():
        result['n_checkpoints'] = len(list(ckpt_dir.glob('kg_*.npz')))

    # Model info
    meta_path = pathlib.Path('/tmp/learned_classifier/rf_meta.json')
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            result['model'] = dict(
                oob_score=round(meta.get('oob_score', 0), 4),
                n_train=meta.get('n_train', 0),
                n_kgs=meta.get('n_kgs', 0),
                n_classes=len(meta.get('classes', [])),
                trained_at=meta.get('trained_at', ''))
        except Exception:
            pass

    return jsonify(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_geometry():
    """Extract geometry from request. Supports JSON body, form data, file upload."""
    if 'file' in request.files:
        f = request.files['file']
        content = f.read().decode('utf-8')
        return geo_parse.parse_input(content)
    if request.is_json:
        body = request.get_json()
        if 'geometry' in body:
            geom_input = body['geometry']
        elif 'type' in body:
            geom_input = body
        else:
            raise ValueError("JSON body must contain 'geometry' or be a valid GeoJSON")
        return geo_parse.parse_input(geom_input)
    if request.form.get('geometry'):
        return geo_parse.parse_input(request.form['geometry'])
    data = request.get_data(as_text=True)
    if data:
        return geo_parse.parse_input(data)
    raise ValueError("No geometry provided. Send GeoJSON, KML, or coordinates.")


def _get_params():
    params = {}
    if request.is_json:
        body = request.get_json()
        params = {k: v for k, v in body.items()
                  if k not in ('geometry', 'type', 'features', 'coordinates')}
    for key in ('dataset', 'date_a', 'date_b', 'dates',
                'min_height', 'max_height', 'min_area', 'min_change',
                'object_types', 'resolution', 'format',
                'include_ortho', 'include_temporal',
                'include_copernicus', 'include_cadastre',
                'include_hansen', 'color_mode', 'types',
                'ortho_year', 'min_object_size',
                'felz_scale', 'rag_threshold', 'groups'):
        val = request.args.get(key)
        if val is not None:
            params[key] = val
    return params


def _validate_area(geom_3035):
    area = geom_3035.area
    if area > MAX_AREA_SQM:
        raise ValueError(
            f"Area too large: {area/1e6:.1f} km² (max {MAX_AREA_SQM/1e6:.0f} km²). "
            f"Use a smaller geometry."
        )


def _error(msg, code=400):
    return jsonify({"error": str(msg)}), code


def _rf_model_meta() -> dict:
    """Return RF model version info for response metadata."""
    try:
        import learned_classifier as lc
        clf = lc.get_classifier()
        if clf.is_trained:
            return {
                "rf_trained_at": clf.trained_at,
                "rf_n_kgs": clf.n_kgs,
                "rf_oob": round(clf.oob_score, 4),
                "rf_n_train": clf.n_train,
            }
    except Exception:
        pass
    return {}


def _try_read_ortho(data: dict) -> tuple:
    """Attempt to read RGB+NIR ortho aligned to ALS data.

    Returns (rgb, spectral) or (None, None).  *spectral* will include
    an ``"ndvi"`` key when NIR is available from an RGBI operate.
    """
    try:
        import ortho_io
        rgb, nir = ortho_io.read_ortho_for_als(data)
        spectral = ortho_io.compute_spectral_indices(rgb, nir=nir)
        # Add raw bands for object_segmentation fused gradient + classification
        if rgb is not None:
            spectral["red"] = rgb[0].astype(np.float32)
            spectral["green"] = rgb[1].astype(np.float32)
            spectral["blue"] = rgb[2].astype(np.float32)
        if nir is not None:
            spectral["nir"] = nir.astype(np.float32)
        return rgb, spectral
    except Exception as e:
        log.warning("Ortho read failed (non-fatal): %s", e)
        return None, None


def _try_copernicus(geom_wgs84, *, ndvi=True, landcover=True, sar=False,
                    harmonics=False, year: int = 2023) -> dict | None:
    """Attempt to fetch Copernicus data for a geometry.

    Parameters
    ----------
    year : int
        Observation year.  NDVI composite and SAR backscatter are fetched
        for the growing season (Apr–Sep) of this year.
    """
    try:
        import copernicus
        bbox = geom_wgs84.bounds  # (minx, miny, maxx, maxy) = (west, south, east, north)
        bbox_dict = {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]}
        result = {}
        if ndvi:
            try:
                ndvi_data = copernicus.get_ndvi_composite(bbox_dict, year=year)
                result["ndvi"] = ndvi_data["ndvi"]
                result["transform"] = ndvi_data["transform"]
                result["crs"] = ndvi_data["crs"]
            except Exception as e:
                log.warning("Copernicus NDVI failed: %s", e)
        if landcover:
            try:
                lc = copernicus.get_land_cover(bbox_dict)
                result["landcover"] = lc
            except Exception as e:
                log.warning("Copernicus land cover failed: %s", e)
        if sar:
            try:
                sar_start = f"{year}-06-01"
                sar_end   = f"{year}-09-30"
                sar_data = copernicus.get_sar_backscatter(bbox_dict, sar_start, sar_end)
                result["vv"] = sar_data["vv"]
                result["vh"] = sar_data["vh"]
                result["sar_transform"] = sar_data["transform"]
                result["sar_crs"] = sar_data["crs"]
            except Exception as e:
                log.warning("Copernicus SAR failed: %s", e)
        if harmonics:
            try:
                import ndvi_harmonics
                harm = ndvi_harmonics.get_harmonic_features(bbox_dict, year=year)
                if harm:
                    result["harmonics"] = harm
                    log.info("NDVI harmonics: mean amp=%.3f",
                             float(np.nanmean(harm.get("h_amplitude", [0]))))
            except Exception as e:
                log.warning("NDVI harmonics failed: %s", e)
        return result if result else None
    except ImportError:
        log.info("Copernicus module not available")
        return None
    except Exception as e:
        log.warning("Copernicus data failed: %s", e)
        return None


def _try_cadastre(geom_wgs84, transform, shape) -> np.ndarray | None:
    """Attempt to fetch building footprints from cadastre."""
    try:
        import cadastre
        bbox = geom_wgs84.bounds
        return cadastre.get_building_mask(bbox, transform, shape)
    except ImportError:
        log.info("Cadastre module not available")
        return None
    except Exception as e:
        log.warning("Cadastre fetch failed: %s", e)
        return None


def _try_hansen(geom_wgs84, transform, shape) -> dict | None:
    """Attempt to load Hansen forest prior for segment_and_classify."""
    try:
        bbox = geom_wgs84.bounds
        return hansen.get_forest_prior(bbox, transform, shape)
    except Exception as e:
        log.warning("Hansen prior failed: %s", e)
        return None


def _clear_raster_caches():
    """Delete cached .npz / .tif / batch dirs to reclaim memory after training."""
    import pathlib, shutil
    cleared = 0
    for cache_dir in [
        pathlib.Path("/tmp/copernicus_cache"),
        pathlib.Path("/tmp/hansen_cache"),
    ]:
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                try:
                    if f.is_dir():
                        shutil.rmtree(f)
                    else:
                        f.unlink()
                    cleared += 1
                except Exception:
                    pass
    if cleared:
        log.info("Cleared %d cached raster entries after training", cleared)


# ---------------------------------------------------------------------------
# 1. ELEVATION
# ---------------------------------------------------------------------------

@app.route('/api/v1/elevation', methods=['POST'])
def elevation():
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)

        result_features = []
        for feat in features:
            geom = feat['geometry']
            geom_3035 = ti.geometry_to_3035(geom)

            if geom.geom_type == 'Point':
                e, n = geom_3035.coords[0][:2]
                bounds = (e - 5, n - 5, e + 5, n + 5)
                dsm_data, tf, _ = raster_io.read_window_bbox('DSM', *bounds, dataset, pad=0)
                dtm_data, _, _ = raster_io.read_window_bbox('DTM', *bounds, dataset, pad=0)
                row = max(0, min(int((tf.f - n) / abs(tf.e)), dsm_data.shape[0]-1))
                col = max(0, min(int((e - tf.c) / tf.a), dsm_data.shape[1]-1))
                dsm_val = round(float(dsm_data[row, col]), 2)
                dtm_val = round(float(dtm_data[row, col]), 2)
                props = dict(feat.get('properties', {}))
                props['dsm_elevation_m'] = dsm_val
                props['dtm_elevation_m'] = dtm_val
                props['object_height_m'] = round(float(dsm_val - dtm_val), 2)
                # Add DSM altitude as Z coordinate
                geom_z = Point(geom.x, geom.y, dsm_val)
                result_features.append({"type": "Feature", "properties": props, "geometry": mapping(geom_z)})

            elif geom.geom_type in ('LineString', 'MultiLineString'):
                _validate_area(geom_3035.buffer(10))
                coords_3035 = list(geom_3035.coords) if geom.geom_type == 'LineString' else \
                    [c for ls in geom_3035.geoms for c in ls.coords]
                bounds = geom_3035.bounds
                dsm_data, tf, _ = raster_io.read_window_bbox('DSM', *bounds, dataset)
                dtm_data, _, _ = raster_io.read_window_bbox('DTM', *bounds, dataset)
                enriched_coords = []
                coords_wgs_3d = []
                for e, n in coords_3035:
                    row = max(0, min(int((tf.f - n) / abs(tf.e)), dsm_data.shape[0]-1))
                    col = max(0, min(int((e - tf.c) / tf.a), dsm_data.shape[1]-1))
                    dsm_val = round(float(dsm_data[row, col]), 2)
                    dtm_val = round(float(dtm_data[row, col]), 2)
                    pt_wgs = ti.geometry_from_3035(Point(e, n))
                    enriched_coords.append({
                        "lon": round(pt_wgs.x, 8), "lat": round(pt_wgs.y, 8),
                        "dsm_elevation_m": dsm_val,
                        "dtm_elevation_m": dtm_val,
                        "object_height_m": round(float(dsm_val - dtm_val), 2),
                    })
                    coords_wgs_3d.append((round(pt_wgs.x, 8), round(pt_wgs.y, 8), dsm_val))
                props = dict(feat.get('properties', {}))
                props['elevation_profile'] = enriched_coords
                props['dsm_elevation_min'] = min(p['dsm_elevation_m'] for p in enriched_coords)
                props['dsm_elevation_max'] = max(p['dsm_elevation_m'] for p in enriched_coords)
                # Return geometry with DSM altitude as Z coordinate
                geom_z = SLineString(coords_wgs_3d)
                result_features.append({"type": "Feature", "properties": props, "geometry": mapping(geom_z)})

            else:
                _validate_area(geom_3035)
                data = raster_io.read_dtm_dsm(geom_3035, dataset)
                dtm_valid = data['dtm'][data['mask']]
                dsm_valid = data['dsm'][data['mask']]
                ndsm_valid = data['ndsm'][data['mask']]
                props = dict(feat.get('properties', {}))
                for name, arr in [('dsm_elevation', dsm_valid), ('dtm_elevation', dtm_valid), ('object_heights', ndsm_valid)]:
                    props[name] = {'min': round(float(np.nanmin(arr)), 2), 'max': round(float(np.nanmax(arr)), 2), 'mean': round(float(np.nanmean(arr)), 2)}
                props['area_sqm'] = int(np.sum(data['mask']))
                # Add mean DSM altitude as Z coordinate on polygon exterior
                dsm_mean = round(float(np.nanmean(dsm_valid)), 2)
                geom_dict = mapping(geom)
                if geom_dict.get('type') == 'Polygon' and geom_dict.get('coordinates'):
                    geom_dict['coordinates'] = tuple(
                        tuple((x, y, dsm_mean) for x, y, *_ in ring)
                        for ring in geom_dict['coordinates']
                    )
                result_features.append({"type": "Feature", "properties": props, "geometry": geom_dict})

        return jsonify({"type": "FeatureCollection", "features": result_features,
                        "meta": {"dataset": dataset, "processing_time_s": round(time.time()-t0, 2)}})
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 2. TERRAIN
# ---------------------------------------------------------------------------

@app.route('/api/v1/terrain', methods=['POST'])
def terrain():
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        results = []
        for feat in features:
            geom = feat['geometry']
            geom_3035 = ti.geometry_to_3035(geom)
            _validate_area(geom_3035.buffer(10) if geom.geom_type == 'Point' else geom_3035)
            if geom.geom_type == 'Point':
                geom_3035 = geom_3035.buffer(50)
            dtm_data, mask, tf, crs = raster_io.read_masked('DTM', geom_3035, dataset)
            terrain_stats = ta.characterise_terrain(dtm_data, mask)
            props = dict(feat.get('properties', {}))
            props['terrain'] = terrain_stats
            results.append({"type": "Feature", "properties": props, "geometry": mapping(feat['geometry'])})
        return jsonify({"type": "FeatureCollection", "features": results,
                        "meta": {"dataset": dataset, "processing_time_s": round(time.time()-t0, 2)}})
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 3. OBJECTS
# ---------------------------------------------------------------------------

@app.route('/api/v1/objects', methods=['POST'])
def objects_summary():
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        min_height = float(params.get('min_height', 0.2))
        min_area = int(params.get('min_area', 2))
        include_ortho = str(params.get('include_ortho', 'false')).lower() in ('true', '1', 'yes')
        include_temporal = str(params.get('include_temporal', 'false')).lower() in ('true', '1', 'yes')
        include_copernicus = str(params.get('include_copernicus', 'false')).lower() in ('true', '1', 'yes')
        include_cadastre = str(params.get('include_cadastre', 'false')).lower() in ('true', '1', 'yes')
        object_types_filter = params.get('object_types', None)
        if isinstance(object_types_filter, str):
            object_types_filter = [t.strip() for t in object_types_filter.split(',')]

        all_objects = []
        for feat in features:
            geom = feat['geometry']
            geom_3035 = ti.geometry_to_3035(geom)
            if geom.geom_type == 'Point':
                geom_3035 = geom_3035.buffer(100)
            _validate_area(geom_3035)

            # Multi-temporal LIDAR: loads DTM+DSM for all dates
            dtm_dates, dsm_dates = None, None
            temporal_std, temporal_range, n_dates = None, None, 1
            if include_temporal:
                try:
                    multi = raster_io.read_multi_date_ndsm(geom_3035)
                    data = {
                        'dtm': multi['dtm'], 'dsm': multi['dsm'],
                        'ndsm': multi['ndsm'], 'mask': multi['mask'],
                        'transform': multi['transform'], 'crs': multi['crs'],
                        'shape': multi['shape'],
                    }
                    temporal_std = multi['temporal_std']
                    temporal_range = multi['temporal_range']
                    n_dates = len(multi['dates_loaded'])
                    # Build per-date DTM/DSM dicts for landscape classifier
                    dtm_dates = {}
                    dsm_dates = {}
                    for d in multi['dates_loaded']:
                        try:
                            dd = raster_io.read_dtm_dsm(geom_3035, dataset=d)
                            mh = min(dd['shape'][0], data['shape'][0])
                            mw = min(dd['shape'][1], data['shape'][1])
                            dtm_dates[d] = dd['dtm'][:mh, :mw]
                            dsm_dates[d] = dd['dsm'][:mh, :mw]
                        except Exception as e:
                            log.warning("Failed to load date %s for time series: %s", d, e)
                except Exception as e:
                    log.warning("Multi-temporal load failed, falling back to single date: %s", e)
                    data = raster_io.read_dtm_dsm(geom_3035, dataset)
            else:
                data = raster_io.read_dtm_dsm(geom_3035, dataset)

            rgb, spectral = (None, None)
            if include_ortho:
                rgb, spectral = _try_read_ortho(data)

            # Copernicus data (Sentinel-2 NDVI, land cover, SAR)
            copernicus_data = None
            if include_copernicus:
                copernicus_data = _try_copernicus(geom, sar=True, harmonics=True, year=ti.dataset_to_year(dataset))

            # Cadastre building footprints (ground truth)
            building_footprints = None
            if include_cadastre:
                building_footprints = _try_cadastre(
                    geom, data['transform'], data['shape'],
                )

            # Use new landscape classifier
            result = oc.classify_landscape(
                data['dtm'], data['dsm'], data['mask'], data['transform'],
                dtm_dates=dtm_dates,
                dsm_dates=dsm_dates,
                copernicus=copernicus_data,
                building_footprints=building_footprints,
                min_height=min_height,
                min_area=min_area,
                spectral=spectral,
                rgb=rgb,
                temporal_std=temporal_std,
                temporal_range=temporal_range,
                n_temporal_dates=n_dates,
            )
            objects = result['objects']

            if object_types_filter:
                objects = [o for o in objects if o.obj_type in object_types_filter]
            max_height = params.get('max_height')
            if max_height:
                objects = [o for o in objects if o.height_max <= float(max_height)]
            all_objects.extend(objects)

        summary = oc.summarise_objects(all_objects)
        obj_features = []
        for obj in all_objects:
            centroid_wgs = ti.geometry_from_3035(Point(obj.centroid_e, obj.centroid_n))
            props = {
                "id": obj.obj_id, "type": obj.obj_type, "type_code": obj.type_code,
                "height_max_m": obj.height_max, "height_mean_m": obj.height_mean,
                "height_p90_m": obj.height_p90, "area_sqm": obj.area_sqm,
                "compactness": obj.compactness, "elongation": obj.elongation,
                "solidity": obj.solidity, "extent": obj.extent,
                "dsm_edge_strength": obj.dsm_edge_strength,
                "height_class": obj.height_class,
                "is_manmade": obj.is_manmade,
                "confidence": obj.confidence,
                "linear_feature": obj.linear_feature,
                "machinery_trace": obj.machinery_trace,
            }
            if include_ortho or include_copernicus:
                props["ndvi_mean"] = obj.ndvi_mean
            if include_temporal:
                props["temporal_std"] = obj.temporal_std
                props["temporal_stable"] = obj.temporal_stable
                props["temporal_signal"] = obj.temporal_signal
            obj_features.append({"type": "Feature", "properties": props, "geometry": mapping(centroid_wgs)})

        return jsonify({
            "summary": summary, "type": "FeatureCollection", "features": obj_features,
            "meta": {"dataset": dataset, "min_height": min_height, "min_area": min_area,
                     "include_ortho": include_ortho,
                     "include_temporal": include_temporal,
                     "include_copernicus": include_copernicus,
                     "include_cadastre": include_cadastre,
                     "n_temporal_dates": n_dates if include_temporal else 1,
                     "classifier": "landscape_v2",
                     "object_types_filter": object_types_filter,
                     "processing_time_s": round(time.time()-t0, 2)},
        })
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 3b. WATERSHED SEGMENTATION (new — Felzenszwalb + RAG)
# ---------------------------------------------------------------------------

@app.route('/api/v1/segment', methods=['POST'])
def segment_objects():
    """Watershed-based object segmentation and classification.

    Fused gradient → Felzenszwalb → RAG merge → per-object features → classify → group.
    Returns individual objects (tree, roof, road_surface, …) AND groups (forest, building, …).
    """
    task_id = request.args.get('task_id', '')
    def _prog(step, detail=''):
        if task_id:
            _progress_set(task_id, step, detail)
    try:
        t0 = time.time()
        if task_id:
            _progress_start(task_id)
        _prog('Parsing geometry')
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        min_object_size = int(params.get('min_object_size', 30))
        felz_scale = float(params.get('felz_scale', 150))
        rag_threshold = float(params.get('rag_threshold', 0.12))
        include_ortho = str(params.get('include_ortho', 'true')).lower() in ('true', '1', 'yes')
        include_temporal = str(params.get('include_temporal', 'false')).lower() in ('true', '1', 'yes')
        include_copernicus = str(params.get('include_copernicus', 'false')).lower() in ('true', '1', 'yes')
        include_cadastre = str(params.get('include_cadastre', 'false')).lower() in ('true', '1', 'yes')
        include_hansen = str(params.get('include_hansen', 'false')).lower() in ('true', '1', 'yes')
        type_filter = params.get('types', None)
        if isinstance(type_filter, str):
            type_filter = [t.strip() for t in type_filter.split(',')]
        group_filter = params.get('groups', None)
        if isinstance(group_filter, str):
            group_filter = [g.strip() for g in group_filter.split(',')]

        all_objects = []
        all_stats = None
        all_evaluation = None
        all_labels = None
        all_transform = None
        all_shape = None
        all_mask = None
        hansen_evaluation = None

        for feat in features:
            geom = feat['geometry']
            geom_3035 = ti.geometry_to_3035(geom)
            if geom.geom_type == 'Point':
                geom_3035 = geom_3035.buffer(100)
            _validate_area(geom_3035)

            # Load DTM/DSM
            _prog('Loading DTM/DSM', 'remote raster reads')
            dtm_dates, dsm_dates = None, None
            if include_temporal:
                _prog('Loading DTM/DSM', 'multi-temporal (3 dates)')
                try:
                    multi = raster_io.read_multi_date_ndsm(geom_3035)
                    data = {
                        'dtm': multi['dtm'], 'dsm': multi['dsm'],
                        'ndsm': multi['ndsm'], 'mask': multi['mask'],
                        'transform': multi['transform'], 'crs': multi['crs'],
                        'shape': multi['shape'],
                    }
                    dtm_dates, dsm_dates = {}, {}
                    for d in multi['dates_loaded']:
                        try:
                            _prog('Loading DTM/DSM', f'date {d}')
                            dd = raster_io.read_dtm_dsm(geom_3035, dataset=d)
                            mh = min(dd['shape'][0], data['shape'][0])
                            mw = min(dd['shape'][1], data['shape'][1])
                            dtm_dates[d] = dd['dtm'][:mh, :mw]
                            dsm_dates[d] = dd['dsm'][:mh, :mw]
                        except Exception as e:
                            log.warning("Date %s load failed: %s", d, e)
                except Exception as e:
                    log.warning("Multi-temporal failed, single date: %s", e)
                    data = raster_io.read_dtm_dsm(geom_3035, dataset)
            else:
                data = raster_io.read_dtm_dsm(geom_3035, dataset)

            rgb, spectral = (None, None)
            if include_ortho:
                _prog('Loading orthophoto', 'RGBI 20cm')
                rgb, spectral = _try_read_ortho(data)

            copernicus_data = None
            if include_copernicus:
                _prog('Loading Copernicus', 'NDVI + landcover + SAR + harmonics')
                copernicus_data = _try_copernicus(geom, sar=True, harmonics=True, year=ti.dataset_to_year(dataset))

            building_footprints = None
            if include_cadastre:
                _prog('Loading cadastre', 'building footprints')
                building_footprints = _try_cadastre(
                    geom, data['transform'], data['shape'],
                )

            hansen_data = None
            if include_hansen:
                _prog('Loading Hansen', 'forest change data')
                hansen_data = _try_hansen(geom, data['transform'], data['shape'])

            # Run segmentation pipeline
            _prog('Segmenting & classifying', 'watershed + classification')
            obs_year = ti.dataset_to_year(dataset)
            result = seg.segment_and_classify(
                data['dtm'], data['dsm'], data['mask'], data['transform'],
                dtm_dates=dtm_dates,
                dsm_dates=dsm_dates,
                spectral=spectral,
                copernicus=copernicus_data,
                building_footprints=building_footprints,
                hansen=hansen_data,
                min_object_size=min_object_size,
                felz_scale=felz_scale,
                rag_threshold=rag_threshold,
                observation_year=obs_year,
            )

            objects = result['objects']
            labels = result['labels']

            # Hansen forest loss calibration
            hansen_evaluation = None
            if include_hansen and hansen_data:
                try:
                    objects = hansen.calibrate_tree_loss(objects, labels, hansen_data, observation_year=obs_year)
                    hansen_evaluation = hansen.evaluate_forest_loss(objects, labels, hansen_data, observation_year=obs_year)
                except Exception as e:
                    log.warning("Hansen calibration failed: %s", e)

            # Populate seg_cache so overlay/gpkg endpoints can reuse results
            seg_cache_key = f"{geom_3035.bounds}_{dataset}_{include_ortho}_{include_copernicus}_{include_cadastre}_{include_hansen}_temporal"
            _seg_cache.update({
                "labels": labels, "objects": objects,
                "mask": data['mask'], "transform": data['transform'],
                "shape": data['shape'], "ndsm": data.get('ndsm'), "key": seg_cache_key,
            })
            # Populate raster data cache so overlay endpoints don't re-fetch
            raster_cache_key = f"{geom_3035.bounds}_{dataset}"
            _raster_cache.update({"key": raster_cache_key, "data": data})
            if rgb is not None:
                _raster_cache.update({"ortho": rgb, "ortho_key": raster_cache_key})
            log.info("segment: cached results for overlay reuse")

            # Filters
            if type_filter:
                objects = [o for o in objects if o.obj_type in type_filter]
            if group_filter:
                objects = [o for o in objects if o.group_type in group_filter]

            all_objects.extend(objects)
            all_stats = result.get('stats')
            all_labels = labels
            all_transform = data['transform']
            all_shape = data['shape']
            all_mask = data['mask']
            if result.get('evaluation'):
                all_evaluation = result['evaluation']

        # Build GeoJSON response
        obj_features = []
        for obj in all_objects:
            centroid_wgs = ti.geometry_from_3035(Point(obj.centroid_e, obj.centroid_n))
            props = {
                "id": obj.obj_id,
                "type": obj.obj_type,
                "type_code": obj.type_code,
                "group_id": obj.group_id,
                "group_type": obj.group_type,
                "height_max_m": obj.height_max,
                "height_mean_m": obj.height_mean,
                "height_p90_m": obj.height_p90,
                "area_sqm": obj.area_sqm,
                "compactness": obj.compactness,
                "elongation": obj.elongation,
                "solidity": obj.solidity,
                "extent": obj.extent,
                "dsm_edge_strength": obj.dsm_edge_strength,
                "slope_mean": obj.slope_mean,
                "roughness": obj.roughness,
                "is_manmade": obj.is_manmade,
                "confidence": obj.confidence,
            }
            if include_ortho or include_copernicus:
                props["ndvi_mean"] = obj.ndvi_mean
                props["ndvi_fused"] = obj.ndvi_fused
                props["brightness_mean"] = obj.brightness_mean
                props["nir_mean"] = obj.nir_mean
            if include_temporal:
                props["height_change"] = obj.height_change
                props["dtm_change"] = obj.dtm_change
                props["temporal_stability"] = obj.temporal_stability
            # Texture features
            if obj.glcm_entropy > 0:
                props["glcm_entropy"] = obj.glcm_entropy
                props["glcm_homogeneity"] = obj.glcm_homogeneity
                props["texture_complexity"] = obj.texture_complexity
            # SAR features
            if obj.sar_vv > 0:
                props["sar_vv"] = obj.sar_vv
                props["sar_vh"] = obj.sar_vh
            # Phenology features
            if obj.harm_amplitude > 0:
                props["harm_amplitude"] = obj.harm_amplitude
                props["harm_phase"] = obj.harm_phase
                props["phenology_class"] = obj.phenology_class
            obj_features.append({
                "type": "Feature",
                "properties": props,
                "geometry": mapping(centroid_wgs),
            })

        resp = {
            "type": "FeatureCollection",
            "features": obj_features,
            "stats": all_stats,
            "meta": {
                "classifier": "watershed_v1",
                "pipeline": "Sobel→Felzenszwalb→RAG→classify→group",
                "dataset": dataset,
                "min_object_size": min_object_size,
                "felz_scale": felz_scale,
                "rag_threshold": rag_threshold,
                "include_ortho": include_ortho,
                "include_temporal": include_temporal,
                "include_copernicus": include_copernicus,
                "include_cadastre": include_cadastre,
                "include_hansen": include_hansen,
                "processing_time_s": round(time.time() - t0, 2),
                **_rf_model_meta(),
            },
        }
        if all_evaluation:
            resp["cadastre_evaluation"] = all_evaluation
        if hansen_evaluation:
            resp["hansen_evaluation"] = hansen_evaluation

        _progress_end(task_id)
        return jsonify(resp)
    except ValueError as e:
        _progress_end(task_id)
        return _error(str(e))
    except Exception as e:
        _progress_end(task_id)
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 3c. SEGMENT RASTER OVERLAY
# ---------------------------------------------------------------------------

# Cache for last segmentation result so legend filter re-renders are instant
_seg_cache = {"labels": None, "objects": None, "mask": None,
              "transform": None, "shape": None, "ndsm": None, "key": None}

# Cache for raster data (DTM/DSM/nDSM/ortho) keyed by (bounds, dataset)
# so overlay endpoints don't re-fetch from remote after segment has loaded them
_raster_cache = {"key": None, "data": None, "ortho": None, "ortho_key": None}

# Segment type → RGBA colour (matches frontend TYPE_SHAPES)
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


def _diverging_rgb(t):
    """Blue-white-red diverging scale, t in [-1,1] mapped to [0,1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        # Blue to white
        f = t * 2
        return (int(33 + f * 222), int(102 + f * 153), int(172 + f * 83))
    else:
        # White to red
        f = (t - 0.5) * 2
        return (int(255 - f * 37), int(255 - f * 192), int(255 - f * 192))


def _segment_rgba(labels, objects, mask, type_filter=None, color_mode='type', ndsm=None):
    """Render segmentation labels as RGBA image.

    color_mode: 'type' = categorical colors, 'height' = viridis by height
    """
    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    obj_map = {o.obj_id: o for o in objects}

    if color_mode == 'height' and ndsm is not None:
        # Per-pixel viridis coloring from actual nDSM values
        included = np.zeros((h, w), dtype=bool)
        for obj_id, obj in obj_map.items():
            if type_filter and obj.obj_type not in type_filter:
                continue
            included |= (labels == obj_id)
        # Build viridis LUT (256 entries)
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            lut[i] = _viridis_rgb(i / 255.0)
        idx = np.clip((np.clip(ndsm / 35.0, 0, 1) * 255).astype(np.uint8), 0, 255)
        for c in range(3):
            rgba[:, :, c] = lut[idx, c]
        rgba[:, :, 3] = np.where(included & mask, np.where(ndsm > 0.3, 180, 60).astype(np.uint8), 0)
    else:
        for obj_id, obj in obj_map.items():
            if type_filter and obj.obj_type not in type_filter:
                continue
            seg_mask = labels == obj_id
            color = SEGMENT_COLORS.get(obj.obj_type, (128, 128, 128, 120))
            for c in range(4):
                rgba[:, :, c][seg_mask] = color[c]

    # Transparent where no data
    rgba[:, :, 3][~mask] = 0
    return rgba


def _render_seg_overlay(labels, objects, mask, transform, shape_hw, type_filter=None, color_mode='type', ndsm=None):
    """Render segmentation as RGBA, reproject to WGS84, return overlay response."""
    from rasterio.warp import calculate_default_transform, reproject as rp, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds

    rgba_3035 = _segment_rgba(labels, objects, mask, type_filter, color_mode=color_mode, ndsm=ndsm)

    src_crs = CRS.from_epsg(3035)
    dst_crs = CRS.from_epsg(4326)
    h, w = shape_hw
    dst_tf, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, w, h, *array_bounds(h, w, transform),
    )
    rgba_wgs = np.zeros((4, dst_h, dst_w), dtype=np.uint8)
    for band in range(4):
        rp(
            source=rgba_3035[:, :, band],
            destination=rgba_wgs[band],
            src_transform=transform,
            src_crs=src_crs,
            dst_transform=dst_tf,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    rgba_out = np.transpose(rgba_wgs, (1, 2, 0))  # (H,W,4)
    bounds = array_bounds(dst_h, dst_w, dst_tf)
    bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])  # south,west,north,east
    return _send_rgba_overlay(rgba_out, bounds_wgs)


@app.route('/api/v1/segment/overlay', methods=['POST'])
def segment_overlay():
    """Return segment classification as a coloured PNG overlay (reprojected to WGS84).

    First call runs full segmentation and caches the result.
    Subsequent calls with only ?types= changed use the cache for instant re-renders.
    """
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        include_ortho = str(params.get('include_ortho', 'true')).lower() in ('true', '1', 'yes')
        include_copernicus = str(params.get('include_copernicus', 'false')).lower() in ('true', '1', 'yes')
        include_cadastre = str(params.get('include_cadastre', 'false')).lower() in ('true', '1', 'yes')
        include_hansen = str(params.get('include_hansen', 'false')).lower() in ('true', '1', 'yes')
        type_filter_str = params.get('types', None)
        type_filter = None
        if type_filter_str:
            type_filter = set(t.strip() for t in type_filter_str.split(','))
        color_mode = params.get('color_mode', 'type')  # 'type' or 'height'

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(100)
        _validate_area(geom_3035)

        # Build a cache key from geometry bounds + dataset + analysis options
        cache_key = f"{geom_3035.bounds}_{dataset}_{include_ortho}_{include_copernicus}_{include_cadastre}_{include_hansen}_temporal"

        if _seg_cache["key"] == cache_key:
            # Re-render from cache — instant
            log.info("segment overlay: re-render from cache (filter=%s)", type_filter)
            return _render_seg_overlay(
                _seg_cache["labels"], _seg_cache["objects"],
                _seg_cache["mask"], _seg_cache["transform"],
                _seg_cache["shape"], type_filter, color_mode,
                ndsm=_seg_cache.get("ndsm"),
            )

        # Full segmentation pipeline
        # Always load temporal data for stability-based building detection
        dtm_dates, dsm_dates = None, None
        try:
            multi = raster_io.read_multi_date_ndsm(geom_3035)
            data = {
                'dtm': multi['dtm'], 'dsm': multi['dsm'],
                'ndsm': multi['ndsm'], 'mask': multi['mask'],
                'transform': multi['transform'], 'crs': multi['crs'],
                'shape': multi['shape'],
            }
            dtm_dates, dsm_dates = {}, {}
            for d in multi['dates_loaded']:
                try:
                    dd = raster_io.read_dtm_dsm(geom_3035, dataset=d)
                    mh = min(dd['shape'][0], data['shape'][0])
                    mw = min(dd['shape'][1], data['shape'][1])
                    dtm_dates[d] = dd['dtm'][:mh, :mw]
                    dsm_dates[d] = dd['dsm'][:mh, :mw]
                except Exception as e:
                    log.warning("overlay: date %s load failed: %s", d, e)
        except Exception as e:
            log.warning("overlay: multi-temporal failed, single date: %s", e)
            data = raster_io.read_dtm_dsm(geom_3035, dataset)

        rgb, spectral = (None, None)
        if include_ortho:
            rgb, spectral = _try_read_ortho(data)

        copernicus_data = None
        if include_copernicus:
            copernicus_data = _try_copernicus(geom, sar=True, harmonics=True, year=ti.dataset_to_year(dataset))

        building_footprints = None
        if include_cadastre:
            building_footprints = _try_cadastre(geom, data['transform'], data['shape'])

        hansen_data = None
        if include_hansen:
            hansen_data = _try_hansen(geom, data['transform'], data['shape'])

        obs_year = ti.dataset_to_year(dataset)
        result = seg.segment_and_classify(
            data['dtm'], data['dsm'], data['mask'], data['transform'],
            dtm_dates=dtm_dates,
            dsm_dates=dsm_dates,
            spectral=spectral,
            copernicus=copernicus_data,
            building_footprints=building_footprints,
            hansen=hansen_data,
            observation_year=obs_year,
        )

        objects = result['objects']
        labels = result['labels']

        # Hansen calibration
        if include_hansen and hansen_data:
            try:
                objects = hansen.calibrate_tree_loss(objects, labels, hansen_data, observation_year=obs_year)
            except Exception as e:
                log.warning("Hansen calibration failed in overlay: %s", e)

        # Store in cache for fast re-renders with different type filters
        _seg_cache.update({
            "labels": labels, "objects": objects,
            "mask": data['mask'], "transform": data['transform'],
            "shape": data['shape'], "ndsm": data['ndsm'], "key": cache_key,
        })
        log.info("segment overlay: full pipeline %.1fs, cached for re-renders", time.time() - t0)

        return _render_seg_overlay(labels, objects, data['mask'],
                                   data['transform'], data['shape'], type_filter, color_mode,
                                   ndsm=data['ndsm'])
    except Exception as e:
        log.error("segment overlay: %s", traceback.format_exc())
        return _error(str(e))


# ---------------------------------------------------------------------------
# 3d. GEOPACKAGE EXPORT
# ---------------------------------------------------------------------------

@app.route('/api/v1/export/geopackage', methods=['POST'])
def export_geopackage():
    """Export all current map data as a GeoPackage (DTM, DSM, ortho, segments, raster)."""
    try:
        import fiona
        from fiona.crs import from_epsg
    except ImportError:
        # fiona not available, use rasterio-only approach
        pass

    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        include_ortho = str(params.get('include_ortho', 'true')).lower() in ('true', '1', 'yes')
        include_segments = str(params.get('include_segments', 'true')).lower() in ('true', '1', 'yes')
        type_filter_str = params.get('types', None)
        type_filter = None
        if type_filter_str:
            type_filter = set(t.strip() for t in type_filter_str.split(','))

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(100)
        _validate_area(geom_3035)

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        dtm = data['dtm']
        dsm = data['dsm']
        ndsm = data['ndsm']
        tf = data['transform']
        h, w = data['shape']
        mask = data['mask']

        tmp = tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False)
        tmp_path = tmp.name
        tmp.close()

        # Band count: DTM, DSM, nDSM + optional ortho RGB(I) + optional segment type
        bands = ['DTM', 'DSM', 'nDSM']
        arrays = [dtm.astype(np.float32), dsm.astype(np.float32), ndsm.astype(np.float32)]

        # Ortho
        rgb = None
        if include_ortho:
            try:
                import ortho_io
                rgb_arr, nir = ortho_io.read_ortho_for_als(data)
                if rgb_arr is not None:
                    for i, name in enumerate(['Red', 'Green', 'Blue']):
                        bands.append(name)
                        arrays.append(rgb_arr[i].astype(np.float32))
                    if nir is not None:
                        bands.append('NIR')
                        arrays.append(nir.astype(np.float32))
                    rgb = rgb_arr
            except Exception as e:
                log.warning("Ortho for gpkg failed: %s", e)

        # Segments — use cache if available, otherwise run pipeline
        if include_segments:
            try:
                # Check if cached segmentation matches this geometry
                cache_key_check = f"{geom_3035.bounds}_{dataset}"
                if _seg_cache["key"] and cache_key_check in _seg_cache["key"]:
                    log.info("GeoPackage: using cached segmentation")
                    labels = _seg_cache["labels"]
                    objects = _seg_cache["objects"]
                else:
                    spectral = None
                    if rgb is not None:
                        import ortho_io
                        _, nir_arr = ortho_io.read_ortho_for_als(data)
                        spectral = ortho_io.compute_spectral_indices(rgb, nir=nir_arr)
                        if rgb is not None:
                            spectral["red"] = rgb[0].astype(np.float32)
                            spectral["green"] = rgb[1].astype(np.float32)
                            spectral["blue"] = rgb[2].astype(np.float32)
                        if nir_arr is not None:
                            spectral["nir"] = nir_arr.astype(np.float32)
                    result = seg.segment_and_classify(
                        dtm, dsm, mask, tf, spectral=spectral,
                        observation_year=ti.dataset_to_year(dataset),
                    )
                    objects = result['objects']
                    labels = result['labels']

                # Type filter
                if type_filter:
                    filtered_ids = {o.obj_id for o in objects if o.obj_type in type_filter}
                else:
                    filtered_ids = {o.obj_id for o in objects}

                type_raster = np.zeros((h, w), dtype=np.float32)
                obj_map = {o.obj_id: o for o in objects}
                for oid in filtered_ids:
                    if oid in obj_map:
                        seg_mask = labels == oid
                        type_raster[seg_mask] = float(obj_map[oid].type_code)
                # Per-pixel nDSM height (not per-segment mean)
                height_raster = np.where(ndsm > 0, ndsm, 0).astype(np.float32)

                bands.append('segment_type')
                arrays.append(type_raster)
                bands.append('segment_height')
                arrays.append(height_raster)
            except Exception as e:
                log.warning("Segments for gpkg failed: %s", e)

        n_bands = len(arrays)
        with rasterio.open(
            tmp_path, 'w', driver='GPKG', width=w, height=h,
            count=n_bands, dtype='float32', crs='EPSG:3035',
            transform=tf, nodata=np.nan,
        ) as dst:
            for i, (arr, name) in enumerate(zip(arrays, bands), 1):
                # Ensure shape matches
                out = arr[:h, :w] if arr.shape[0] >= h and arr.shape[1] >= w else arr
                dst.write(out, i)
                dst.set_band_description(i, name)

        log.info("GeoPackage export: %d bands, %.1fs", n_bands, time.time() - t0)
        return send_file(
            tmp_path, mimetype='application/geopackage+sqlite3',
            as_attachment=True, download_name='landscape_export.gpkg',
        )
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error("geopackage export: %s", traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 4. OBJECT RASTER
# ---------------------------------------------------------------------------

@app.route('/api/v1/objects/raster', methods=['POST'])
def objects_raster():
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        resolution = float(params.get('resolution', 1.0))
        min_height = float(params.get('min_height', 0.2))
        min_area = int(params.get('min_area', 2))
        include_ortho = str(params.get('include_ortho', 'false')).lower() in ('true', '1', 'yes')
        include_temporal = str(params.get('include_temporal', 'false')).lower() in ('true', '1', 'yes')

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(100)
        _validate_area(geom_3035)

        temporal_std, temporal_range, n_dates = None, None, 1
        if include_temporal:
            try:
                multi = raster_io.read_multi_date_ndsm(geom_3035)
                data = {
                    'dtm': multi['dtm'], 'dsm': multi['dsm'],
                    'ndsm': multi['ndsm'], 'mask': multi['mask'],
                    'transform': multi['transform'], 'crs': multi['crs'],
                    'shape': multi['shape'],
                }
                temporal_std = multi['temporal_std']
                temporal_range = multi['temporal_range']
                n_dates = len(multi['dates_loaded'])
            except Exception as e:
                log.warning("Multi-temporal load failed: %s", e)
                data = raster_io.read_dtm_dsm(geom_3035, dataset)
        else:
            data = raster_io.read_dtm_dsm(geom_3035, dataset)

        rgb, spectral = (None, None)
        if include_ortho:
            rgb, spectral = _try_read_ortho(data)

        objects = oc.classify_objects(
            data['ndsm'], data['dtm'], data['mask'], data['transform'],
            min_height=min_height, min_area=min_area,
            dsm=data['dsm'], rgb=rgb, spectral=spectral,
            temporal_std=temporal_std, temporal_range=temporal_range,
            n_temporal_dates=n_dates,
        )

        type_band, height_band, out_tf = oc.create_classified_raster(
            data['ndsm'], data['dtm'], data['dsm'], data['mask'],
            data['transform'], objects, output_resolution=resolution,
            spectral=spectral,
        )

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        h, w = type_band.shape
        with rasterio.open(tmp_path, 'w', driver='GTiff', width=w, height=h, count=2,
                           dtype='float32', crs='EPSG:3035', transform=out_tf, nodata=-9999) as dst:
            dst.write(type_band.astype(np.float32), 1)
            dst.write(height_band, 2)
            dst.set_band_description(1, 'object_type_code')
            dst.set_band_description(2, 'object_height_m')
            dst.update_tags(1, **{f'OBJECT_TYPE_{v}': k for k, v in oc.OBJECT_TYPES.items()})

        return send_file(tmp_path, mimetype='image/tiff', as_attachment=True,
                         download_name=f'objects_{dataset}.tif')
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 5. TEMPORAL CHANGE DETECTION
# ---------------------------------------------------------------------------

@app.route('/api/v1/changes', methods=['POST'])
def changes():
    """Detect changes between two ALS dates. Returns classified change events."""
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        date_a = params.get('date_a', '20220915')
        date_b = params.get('date_b', '20240915')
        min_change = float(params.get('min_change', 1.0))

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(200)
        _validate_area(geom_3035)

        comparison = tca.compare_dates(geom_3035, date_a, date_b)
        events = tca.detect_changes(geom_3035, date_a, date_b,
                                    min_change=min_change, comparison=comparison)

        event_features = []
        for ev in events:
            centroid_wgs = ti.geometry_from_3035(Point(ev.centroid_e, ev.centroid_n))
            props = {
                "event_type": ev.event_type,
                "area_sqm": ev.area_sqm,
                "height_before_m": ev.height_before,
                "height_after_m": ev.height_after,
                "height_change_mean_m": ev.height_change_mean,
                "height_change_max_m": ev.height_change_max,
                "dtm_change_mean_m": ev.dtm_change_mean,
                "dtm_change_max_m": ev.dtm_change_max,
                "dsm_change_mean_m": ev.dsm_change_mean,
                "confidence": ev.confidence,
                "detail": ev.detail,
            }
            event_features.append({"type": "Feature", "properties": props,
                                   "geometry": mapping(centroid_wgs)})

        # Summarise by type
        by_type = {}
        for ev in events:
            t2 = ev.event_type
            by_type.setdefault(t2, {"count": 0, "total_area_sqm": 0})
            by_type[t2]["count"] += 1
            by_type[t2]["total_area_sqm"] += ev.area_sqm
        for v in by_type.values():
            v["total_area_sqm"] = round(v["total_area_sqm"], 1)

        return jsonify({
            "type": "FeatureCollection", "features": event_features,
            "summary": {"total_events": len(events), "by_type": by_type},
            "comparison_stats": comparison["stats"],
            "meta": {"date_a": date_a, "date_b": date_b, "min_change_m": min_change,
                     "processing_time_s": round(time.time()-t0, 2)},
        })
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 6. PER-TREE GROWTH ANALYSIS
# ---------------------------------------------------------------------------

@app.route('/api/v1/changes/trees', methods=['POST'])
def changes_trees():
    """Per-tree growth / felling analysis between two dates."""
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        date_a = params.get('date_a', '20220915')
        date_b = params.get('date_b', '20240915')

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(200)
        _validate_area(geom_3035)

        tree_changes = tca.detect_tree_growth(geom_3035, date_a, date_b)

        tree_features = []
        for tc in tree_changes:
            centroid_wgs = ti.geometry_from_3035(Point(tc.centroid_e, tc.centroid_n))
            tree_features.append({"type": "Feature", "properties": {
                "tree_id": tc.tree_id, "status": tc.status,
                "height_before_m": tc.height_before, "height_after_m": tc.height_after,
                "height_change_m": tc.height_change,
                "crown_area_before_sqm": tc.crown_area_before,
                "crown_area_after_sqm": tc.crown_area_after,
            }, "geometry": mapping(centroid_wgs)})

        by_status = {}
        for tc in tree_changes:
            by_status.setdefault(tc.status, {"count": 0, "mean_dh": []})
            by_status[tc.status]["count"] += 1
            by_status[tc.status]["mean_dh"].append(tc.height_change)
        for v in by_status.values():
            dh_list = v.pop("mean_dh")
            v["height_change_mean_m"] = round(float(np.mean(dh_list)), 2) if dh_list else 0

        return jsonify({
            "type": "FeatureCollection", "features": tree_features,
            "summary": {"total_trees": len(tree_changes), "by_status": by_status},
            "meta": {"date_a": date_a, "date_b": date_b,
                     "processing_time_s": round(time.time()-t0, 2)},
        })
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 7. MULTI-EPOCH SUMMARY
# ---------------------------------------------------------------------------

@app.route('/api/v1/changes/summary', methods=['POST'])
def changes_summary():
    """Multi-epoch change summary across all available dates."""
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dates = params.get('dates', None)
        if isinstance(dates, str):
            dates = [d.strip() for d in dates.split(',')]

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(200)
        _validate_area(geom_3035)

        result = tca.temporal_summary(geom_3035, dates=dates)
        result["meta"] = {"processing_time_s": round(time.time()-t0, 2)}
        return jsonify(result)
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# INFO
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LIDAR / ORTHO OVERLAY & DOWNLOAD
# ---------------------------------------------------------------------------

def _extract_single_geom(features_or_geom):
    """Extract a single shapely geometry from _get_geometry() output."""
    if isinstance(features_or_geom, list):
        if not features_or_geom:
            raise ValueError("No geometry provided.")
        feat = features_or_geom[0]
        if isinstance(feat, dict) and 'geometry' in feat:
            return feat['geometry']
        return shape(feat) if isinstance(feat, dict) else feat
    return features_or_geom


def _geometry_to_3035_bbox(features_or_geom):
    """Convert WGS84 geometry to EPSG:3035 and return (geom_3035, bbox_3035, bbox_wgs84)."""
    geom_wgs84 = _extract_single_geom(features_or_geom)
    geom_3035 = ti.geometry_to_3035(geom_wgs84)
    _validate_area(geom_3035)
    b = geom_3035.bounds  # (minx, miny, maxx, maxy)
    bw = geom_wgs84.bounds
    return geom_3035, b, bw


def _get_cached_raster(geom_3035, dataset):
    """Return DTM/DSM data from cache if available, else read from remote."""
    cache_key = f"{geom_3035.bounds}_{dataset}"
    if _raster_cache["key"] == cache_key and _raster_cache["data"] is not None:
        log.info("raster cache hit for %s", cache_key)
        return _raster_cache["data"]
    log.info("raster cache miss, reading from remote")
    data = raster_io.read_dtm_dsm(geom_3035, dataset)
    _raster_cache.update({"key": cache_key, "data": data})
    return data


def _hillshade(elevation, azimuth=315, altitude=45, z_factor=1.0):
    """Compute hillshade from an elevation array."""
    dy, dx = np.gradient(elevation, 1.0)
    dx *= z_factor
    dy *= z_factor
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    hs = (np.sin(alt_rad) * np.cos(slope) +
          np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return np.clip(hs, 0, 1).astype(np.float32)


def _dtm_rgba(dtm, mask):
    """Render DTM as hillshade relief with hypsometric tinting.

    Shows actual terrain: valleys, ridges, slopes — the relief you see on
    a good topographic map.
    """
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    h, w = dtm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # --- Multi-directional hillshade for rich relief ---
    hs1 = _hillshade(dtm, azimuth=315, altitude=40)
    hs2 = _hillshade(dtm, azimuth=90, altitude=55)
    hs_combined = np.clip(0.65 * hs1 + 0.35 * hs2, 0, 1)

    # --- Hypsometric tint based on DTM elevation ---
    vmin = float(np.nanpercentile(dtm[mask], 2)) if mask.any() else 0
    vmax = float(np.nanpercentile(dtm[mask], 98)) if mask.any() else 1000
    if vmax - vmin < 10:
        vmax = vmin + 10

    hypso_colors = [
        (0.0, '#2d6a2e'),   # valley: deep green
        (0.15, '#5a9e3c'),  # lower slopes: green
        (0.3, '#8ebb4a'),   # mid-low: yellow-green
        (0.45, '#c4b85c'),  # mid: olive/tan
        (0.6, '#b8956a'),   # upper mid: brown
        (0.75, '#a08070'),  # upper slopes: grey-brown
        (0.9, '#c8c0b8'),   # near summit: light grey
        (1.0, '#f0ece8'),   # summit: near-white
    ]
    cmap_hypso = LinearSegmentedColormap.from_list('hypso', hypso_colors)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    hypso = cmap_hypso(norm(np.clip(dtm, vmin, vmax)))[:, :, :3]  # (H,W,3) float

    # Modulate by hillshade
    for c in range(3):
        hypso[:, :, c] = hypso[:, :, c] * (0.25 + 0.75 * hs_combined)

    rgba[:, :, :3] = (np.clip(hypso, 0, 1) * 255).astype(np.uint8)
    rgba[:, :, 3] = np.where(mask, 255, 0)
    return rgba


def _ndsm_rgba(ndsm, dsm, mask, vmax=35):
    """Render nDSM as viridis height coloring with DSM hillshade for 3D effect.

    Ground (ndsm < 0.3) is transparent so it can layer over the DTM relief.
    """
    import matplotlib
    import matplotlib.cm as cm

    h, w = ndsm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    hs_dsm = _hillshade(dsm, azimuth=315, altitude=40)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    colormap = cm.get_cmap('viridis')
    ndsm_clamped = np.clip(ndsm, 0, vmax)
    rgb_float = colormap(norm(ndsm_clamped))[:, :, :3]  # (H,W,3)

    # Modulate by DSM hillshade
    for c in range(3):
        rgb_float[:, :, c] = rgb_float[:, :, c] * (0.3 + 0.7 * hs_dsm)

    elevated = mask & (ndsm > 0.3)
    rgba[:, :, :3] = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    rgba[:, :, 3] = np.where(elevated, 220, 0)
    return rgba


def _reproject_rasters_to_wgs84(arrays_3035, transform_3035, shape_3035, mask_3035=None):
    """Reproject raw float32 rasters from EPSG:3035 to EPSG:4326.

    Parameters
    ----------
    arrays_3035 : dict[str, np.ndarray]
        Named 2D float32 arrays in EPSG:3035 (e.g. dtm, dsm, ndsm).
    transform_3035 : Affine
        Source rasterio transform.
    shape_3035 : (int, int)
        Source (rows, cols).
    mask_3035 : np.ndarray or None
        Boolean mask. Reprojected as nearest-neighbour.

    Returns
    -------
    (arrays_wgs, mask_wgs, transform_wgs, bounds_wgs)
        arrays_wgs: dict with same keys, reprojected.
        mask_wgs: boolean mask in WGS84 grid.
        transform_wgs: rasterio Affine for the WGS84 grid.
        bounds_wgs: (south, west, north, east) for Leaflet.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds

    src_crs = CRS.from_epsg(3035)
    dst_crs = CRS.from_epsg(4326)
    h, w = shape_3035

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, w, h, *array_bounds(h, w, transform_3035),
    )

    arrays_wgs = {}
    for name, src in arrays_3035.items():
        dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=src.astype(np.float32),
            destination=dst,
            src_transform=transform_3035,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        arrays_wgs[name] = dst

    # Reproject mask
    if mask_3035 is not None:
        mask_src = mask_3035.astype(np.uint8)
        mask_dst = np.zeros((dst_height, dst_width), dtype=np.uint8)
        reproject(
            source=mask_src,
            destination=mask_dst,
            src_transform=transform_3035,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        mask_wgs = mask_dst > 0
    else:
        # Derive mask from any reprojected array (non-NaN)
        first = next(iter(arrays_wgs.values()))
        mask_wgs = ~np.isnan(first)

    bounds = array_bounds(dst_height, dst_width, dst_transform)
    # (left, bottom, right, top) = (west, south, east, north)
    bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])  # south, west, north, east

    return arrays_wgs, mask_wgs, dst_transform, bounds_wgs


def _reproject_rgb_to_wgs84(rgb_3035, transform_3035, shape_3035):
    """Reproject a (3,H,W) uint8 RGB array from EPSG:3035 to EPSG:4326.

    Returns (rgb_wgs, mask_wgs, bounds_wgs).
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds

    src_crs = CRS.from_epsg(3035)
    dst_crs = CRS.from_epsg(4326)
    h, w = shape_3035

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, w, h, *array_bounds(h, w, transform_3035),
    )

    rgb_wgs = np.zeros((3, dst_height, dst_width), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=rgb_3035[band],
            destination=rgb_wgs[band],
            src_transform=transform_3035,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    mask_wgs = (rgb_wgs[0] > 0) | (rgb_wgs[1] > 0) | (rgb_wgs[2] > 0)

    bounds = array_bounds(dst_height, dst_width, dst_transform)
    bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])

    return rgb_wgs, mask_wgs, bounds_wgs


def _send_rgba_overlay(rgba, bounds_wgs):
    """Encode RGBA array as PNG and return Flask response with bounds header."""
    from PIL import Image
    img = Image.fromarray(rgba, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, 'PNG', optimize=True)
    buf.seek(0)
    resp = send_file(buf, mimetype='image/png')
    south, west, north, east = bounds_wgs
    resp.headers['X-Bounds'] = f'{south},{west},{north},{east}'
    resp.headers['Access-Control-Expose-Headers'] = 'X-Bounds'
    return resp


@app.route('/api/v1/dtm/overlay', methods=['POST'])
def dtm_overlay():
    """Return DTM hillshade relief as a PNG overlay (reprojected to WGS84)."""
    try:
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        data = _get_cached_raster(geom_3035, dataset)

        # Reproject raw elevation to WGS84 *before* computing hillshade
        arrays_wgs, mask_wgs, tf_wgs, bounds_wgs = _reproject_rasters_to_wgs84(
            {'dtm': data['dtm']},
            data['transform'], data['shape'], data['mask'],
        )
        dtm_wgs = arrays_wgs['dtm']
        # Fill NaN for hillshade computation
        dtm_wgs = np.nan_to_num(dtm_wgs, nan=float(np.nanmedian(dtm_wgs[mask_wgs])) if mask_wgs.any() else 0)

        rgba = _dtm_rgba(dtm_wgs, mask_wgs)
        return _send_rgba_overlay(rgba, bounds_wgs)
    except Exception as e:
        log.error("dtm overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/lidar/overlay', methods=['POST'])
def lidar_overlay():
    """Return nDSM as a PNG overlay (viridis + hillshade, reprojected to WGS84)."""
    try:
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        data = _get_cached_raster(geom_3035, dataset)

        # Reproject raw elevation to WGS84 before rendering
        arrays_wgs, mask_wgs, tf_wgs, bounds_wgs = _reproject_rasters_to_wgs84(
            {'ndsm': data['ndsm'], 'dsm': data['dsm']},
            data['transform'], data['shape'], data['mask'],
        )
        ndsm_wgs = np.nan_to_num(arrays_wgs['ndsm'], nan=0)
        dsm_wgs = np.nan_to_num(arrays_wgs['dsm'], nan=0)

        rgba = _ndsm_rgba(ndsm_wgs, dsm_wgs, mask_wgs)
        return _send_rgba_overlay(rgba, bounds_wgs)
    except Exception as e:
        log.error("lidar overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/ortho/overlay', methods=['POST'])
def ortho_overlay():
    """Return orthophoto as a PNG overlay (reprojected to WGS84)."""
    try:
        import ortho_io
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        ortho_year = params.get('ortho_year')
        if ortho_year:
            ortho_year = int(ortho_year)
        data = _get_cached_raster(geom_3035, dataset)

        # Use cached ortho if available and matches (same geometry, default year)
        raster_cache_key = f"{geom_3035.bounds}_{dataset}"
        if (not ortho_year and _raster_cache.get("ortho") is not None
                and _raster_cache.get("ortho_key") == raster_cache_key):
            log.info("ortho overlay: using cached ortho")
            rgb = _raster_cache["ortho"]
            nir = None  # nir not needed for RGB overlay
        else:
            rgb, nir = ortho_io.read_ortho_for_als(data, year=ortho_year)

        # Reproject RGB to WGS84
        rgb_wgs, mask_wgs, bounds_wgs = _reproject_rgb_to_wgs84(
            rgb, data['transform'], data['shape'],
        )

        h, w = rgb_wgs.shape[1], rgb_wgs.shape[2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = rgb_wgs[0]
        rgba[:, :, 1] = rgb_wgs[1]
        rgba[:, :, 2] = rgb_wgs[2]
        rgba[:, :, 3] = np.where(mask_wgs, 255, 0)

        return _send_rgba_overlay(rgba, bounds_wgs)
    except Exception as e:
        log.error("ortho overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/hansen/overlay', methods=['POST'])
def hansen_overlay():
    """Return Hansen forest loss as a coloured PNG overlay.

    Green = current forest, magenta = loss (brighter = more recent),
    cyan = forest gain.
    """
    try:
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        geom_wgs = _extract_single_geom(geom_wgs84)
        bbox_wgs = geom_wgs.bounds

        prior = hansen.get_forest_prior(
            bbox_wgs, data['transform'], data['shape'],
        )

        h, w = data['shape']
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Current forest: green
        forest = prior['current_forest']
        rgba[:, :, 0][forest] = 20
        rgba[:, :, 1][forest] = 120
        rgba[:, :, 2][forest] = 20
        rgba[:, :, 3][forest] = 140

        # Forest gain: cyan
        gain = prior['gain']
        rgba[:, :, 0][gain] = 0
        rgba[:, :, 1][gain] = 200
        rgba[:, :, 2][gain] = 200
        rgba[:, :, 3][gain] = 180

        # Loss: magenta, brightness by recency (year 1=2001 dark, 24=2024 bright)
        ly = prior['loss_year']
        loss = ly > 0
        brightness = np.clip(80 + (ly.astype(np.float32) / 24.0) * 175, 80, 255).astype(np.uint8)
        rgba[:, :, 0][loss] = brightness[loss]
        rgba[:, :, 1][loss] = 0
        rgba[:, :, 2][loss] = brightness[loss]
        rgba[:, :, 3][loss] = 200

        # Transparent where no data
        rgba[:, :, 3][~data['mask']] = 0

        from rasterio.warp import calculate_default_transform, reproject as rp, Resampling
        from rasterio.crs import CRS
        from rasterio.transform import array_bounds

        src_crs = CRS.from_epsg(3035)
        dst_crs = CRS.from_epsg(4326)
        dst_tf, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, w, h, *array_bounds(h, w, data['transform']),
        )
        rgba_wgs = np.zeros((4, dst_h, dst_w), dtype=np.uint8)
        for band in range(4):
            rp(
                source=rgba[:, :, band],
                destination=rgba_wgs[band],
                src_transform=data['transform'],
                src_crs=src_crs,
                dst_transform=dst_tf,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
        rgba_out = np.transpose(rgba_wgs, (1, 2, 0))
        bounds = array_bounds(dst_h, dst_w, dst_tf)
        bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])
        return _send_rgba_overlay(rgba_out, bounds_wgs)
    except Exception as e:
        log.error("hansen overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/lidar/geotiff', methods=['POST'])
def lidar_geotiff():
    """Download nDSM as a georeferenced GeoTIFF."""
    try:
        geom_wgs84 = _extract_single_geom(_get_geometry())
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035 = ti.geometry_to_3035(geom_wgs84)
        _validate_area(geom_3035)

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        ndsm = data['ndsm']
        dtm = data['dtm']
        dsm = data['dsm'] if 'dsm' in data else dtm + ndsm
        tf = data['transform']
        h, w = data['shape']

        tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        with rasterio.open(tmp.name, 'w', driver='GTiff', width=w, height=h,
                           count=3, dtype='float32', crs='EPSG:3035',
                           transform=tf, compress='deflate') as dst:
            dst.write(dtm.astype(np.float32), 1)
            dst.write(dsm.astype(np.float32), 2)
            dst.write(ndsm.astype(np.float32), 3)
            dst.set_band_description(1, 'DTM')
            dst.set_band_description(2, 'DSM')
            dst.set_band_description(3, 'nDSM')

        return send_file(tmp.name, mimetype='image/tiff', as_attachment=True,
                         download_name='lidar_dtm_dsm_ndsm.tif')
    except Exception as e:
        log.error("lidar geotiff: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/ortho/geotiff', methods=['POST'])
def ortho_geotiff():
    """Download orthophoto as a georeferenced GeoTIFF (RGB + NIR if available)."""
    try:
        import ortho_io
        geom_wgs84 = _extract_single_geom(_get_geometry())
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035 = ti.geometry_to_3035(geom_wgs84)
        _validate_area(geom_3035)

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        rgb, nir = ortho_io.read_ortho_for_als(data)
        tf = data['transform']
        h, w = data['shape']

        n_bands = 4 if nir is not None else 3
        tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        with rasterio.open(tmp.name, 'w', driver='GTiff', width=w, height=h,
                           count=n_bands, dtype='uint8', crs='EPSG:3035',
                           transform=tf, compress='deflate') as dst:
            dst.write(rgb[0], 1)
            dst.write(rgb[1], 2)
            dst.write(rgb[2], 3)
            dst.set_band_description(1, 'Red')
            dst.set_band_description(2, 'Green')
            dst.set_band_description(3, 'Blue')
            if nir is not None:
                dst.write(nir, 4)
                dst.set_band_description(4, 'NIR')

        return send_file(tmp.name, mimetype='image/tiff', as_attachment=True,
                         download_name='orthophoto_rgbi.tif')
    except Exception as e:
        log.error("ortho geotiff: %s", traceback.format_exc())
        return _error(str(e))


# ---------------------------------------------------------------------------
# RF CLASSIFIER TRAINING
# ---------------------------------------------------------------------------

@app.route('/api/v1/classifier/train', methods=['POST'])
def train_classifier():
    """Train RF classifier from cadastre ground truth over a bbox.

    Params: geometry (bbox/geojson), dataset, include_ortho, include_copernicus,
            include_temporal, include_hansen.
    Fetches segment features + cadastre parcel codes, trains RF model.
    All data sources (ortho, copernicus, hansen, temporal) are included by
    default so the RF sees the same features it will use at inference time.
    """
    try:
        import learned_classifier as lc
        import object_segmentation as oc
        import cadastre

        params = _parse_params()
        geom, geom_3035 = _parse_geometry(params)
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        obs_year = ti.dataset_to_year(dataset)

        include_temporal = str(params.get('include_temporal', 'true')).lower() in ('true', '1', 'yes')
        include_hansen = str(params.get('include_hansen', 'true')).lower() in ('true', '1', 'yes')

        # Read LIDAR
        data = raster_io.read_dtm_dsm(geom_3035, dataset)

        # Multi-temporal DTM/DSM
        dtm_dates, dsm_dates = None, None
        if include_temporal:
            try:
                multi = raster_io.read_multi_date_ndsm(geom_3035)
                dtm_dates, dsm_dates = {}, {}
                for d in multi['dates_loaded']:
                    try:
                        dd = raster_io.read_dtm_dsm(geom_3035, dataset=d)
                        mh = min(dd['shape'][0], data['shape'][0])
                        mw = min(dd['shape'][1], data['shape'][1])
                        dtm_dates[d] = dd['dtm'][:mh, :mw]
                        dsm_dates[d] = dd['dsm'][:mh, :mw]
                    except Exception as e:
                        log.warning("Train: date %s load failed: %s", d, e)
            except Exception as e:
                log.warning("Train: multi-temporal failed: %s", e)

        # Read ortho
        rgb, spectral = _try_read_ortho(data)

        # Copernicus (NDVI, land cover, SAR, harmonics)
        copernicus_data = _try_copernicus(geom, sar=True, harmonics=True, year=ti.dataset_to_year(dataset))

        # Hansen forest prior
        hansen_data = None
        if include_hansen:
            hansen_data = _try_hansen(geom, data['transform'], data['shape'])

        # Building footprints from cadastre (for calibration features)
        building_footprints = _try_cadastre(geom, data['transform'], data['shape'])

        # Segment (feature extraction) — pass ALL data sources
        result = oc.segment_and_classify(
            data['dtm'], data['dsm'], data['mask'], data['transform'],
            dtm_dates=dtm_dates,
            dsm_dates=dsm_dates,
            spectral=spectral,
            copernicus=copernicus_data,
            building_footprints=building_footprints,
            hansen=hansen_data,
            ortho_year=obs_year,
            observation_year=obs_year,
        )
        objects = result['objects']
        labels_arr = result['labels']

        # Hansen tree-loss calibration
        if include_hansen and hansen_data:
            try:
                objects = hansen.calibrate_tree_loss(
                    objects, labels_arr, hansen_data,
                    observation_year=obs_year,
                )
            except Exception as e:
                log.warning("Train: Hansen calibration failed: %s", e)

        features = [obj.features for obj in objects]

        # Fetch cadastre parcel codes
        bbox_wgs = geom.bounds
        try:
            parcels = cadastre.fetch_parcel_land_use(
                (bbox_wgs[0], bbox_wgs[1], bbox_wgs[2], bbox_wgs[3]))
        except Exception:
            parcels = None

        if parcels is None:
            return _error("Could not fetch cadastre parcel codes")

        # Match segments to cadastre labels
        train_features = []
        train_labels = []
        for feat in features:
            code = _dominant_cadastre_code(feat, parcels)
            if code and code in lc.CADASTRE_TO_TYPE:
                train_features.append(feat)
                train_labels.append(lc.CADASTRE_TO_TYPE[code])

        if len(train_features) < 20:
            return _error(f"Only {len(train_features)} labelled segments, need >= 20")

        clf = lc.LearnedClassifier()
        stats = clf.train(train_features, train_labels)

        # Clear cached raster data to reclaim memory
        _clear_raster_caches()

        return jsonify({
            "status": "trained",
            "training_stats": stats,
            "n_segments_total": len(features),
            "n_segments_labelled": len(train_features),
            "dataset": dataset,
            "observation_year": obs_year,
            "data_sources": {
                "temporal": dtm_dates is not None and len(dtm_dates) >= 2,
                "ortho": spectral is not None,
                "copernicus": copernicus_data is not None,
                "hansen": hansen_data is not None,
                "building_footprints": building_footprints is not None,
            },
        })

    except Exception as e:
        log.error("classifier train: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/classifier/status', methods=['GET'])
def classifier_status():
    """Check if a trained RF model exists."""
    try:
        import learned_classifier as lc
        clf = lc.get_classifier()
        return jsonify({
            "trained": clf.is_trained,
            "trained_at": clf.trained_at,
            "n_kgs": clf.n_kgs,
            "n_train": clf.n_train,
            "oob_score": clf.oob_score,
            "n_classes": len(clf.classes),
            "classes": clf.classes,
            "top_features": dict(sorted(
                clf.feature_importances.items(),
                key=lambda x: -x[1]
            )[:15]) if clf.feature_importances else {},
        })
    except Exception as e:
        return _error(str(e))


def _dominant_cadastre_code(feat, parcels):
    """Find the most common cadastre code overlapping a segment."""
    # parcels is expected to be a list of {geometry, code} or similar
    # For now, find parcels whose centroid falls within segment bbox
    ce = feat.get("centroid_e", 0)
    cn = feat.get("centroid_n", 0)
    if not parcels:
        return None
    # Simple: find nearest parcel
    best = None
    best_dist = float('inf')
    for p in parcels:
        pc = p.get("centroid")
        code = p.get("code")
        if pc and code:
            d = ((pc[0] - ce)**2 + (pc[1] - cn)**2)**0.5
            if d < best_dist:
                best_dist = d
                best = code
    return best


# ---------------------------------------------------------------------------
# INFO & DOCS
# ---------------------------------------------------------------------------

@app.route('/api/v1/info', methods=['GET'])
def info():
    return jsonify({
        "name": "Austrian LIDAR & Orthophoto Analysis API",
        "version": "3.0.0",
        "description": "Landscape transformation analysis: LIDAR DTM/DSM time series + Sentinel-2 + cadastre",
        "classifier": "landscape_v2 — 10 types focused on human transformation of terrain",
        "source": "data.bev.gv.at",
        "resolution_lidar": "1m",
        "resolution_ortho": "0.2m (1m for analysis, 0.5m for GLCM texture)",
        "crs": "EPSG:3035",
        "datasets_als": {k: {"dtm": True, "dsm": True} for k in sorted(ti.DATASETS.keys())},
        "datasets_ortho": ["20220128 (RGB 50km tiles)"],
        "tiles": len(ti.TILE_COORDS),
        "landscape_types": oc.LANDSCAPE_TYPES,
        "data_sources": {
            "bev_als_dtm_dsm": "1m resolution, 3 dates (2022/2023/2024)",
            "bev_dop_rgbi": "0.2m orthophoto, 47 RGBI operates",
            "copernicus_sentinel2": "10m NDVI growing-season composite (via openEO)",
            "copernicus_worldcover": "10m ESA land cover classification",
            "copernicus_sentinel1_sar": "SAR backscatter (VV+VH)",
            "cadastre_footprints": "mm-precision building polygons (ground truth)",
        },
        "change_event_types": tca.EVENT_TYPES,
        "endpoints": {
            "POST /api/v1/elevation": "Enrich features with DSM/DTM elevation",
            "POST /api/v1/terrain": "Terrain characterisation (slope, ruggedness, etc.)",
            "POST /api/v1/objects": "Object detection and classification (10 landscape types)",
            "POST /api/v1/segment": "Watershed segmentation: 25 object types + 11 group types (Felzenszwalb+RAG)",
            "POST /api/v1/objects/raster": "Classified object raster download (GeoTIFF)",
            "POST /api/v1/changes": "Temporal change detection (earthworks, trees, buildings, roads)",
            "POST /api/v1/changes/trees": "Per-tree growth / felling analysis",
            "POST /api/v1/changes/summary": "Multi-epoch change summary (2022→2023→2024)",
            "GET /api/v1/info": "This endpoint",
            "GET /api/v1/docs/llm.txt": "Machine-readable API reference",
        },
        "max_area_sqkm": MAX_AREA_SQM / 1e6,
    })


@app.route('/api/v1/docs/llm.txt', methods=['GET'])
def llm_docs():
    p = Path(__file__).parent / 'llm.txt'
    if p.exists():
        return Response(p.read_text(), mimetype='text/plain')
    return Response("Documentation not yet generated.", mimetype='text/plain')


# ============ PARSE GEOMETRY FILE ============

@app.route('/api/v1/parse-geometry', methods=['POST'])
def parse_geometry_file():
    """Parse an uploaded geometry file (Shapefile ZIP, GeoPackage, GeoJSON, KML, GPX, WKT, etc).
    Returns GeoJSON FeatureCollection with all features.
    """
    import fiona
    import fiona.io
    import zipfile
    from shapely.geometry import shape as shp_shape, mapping as shp_mapping
    from shapely.ops import unary_union
    try:
        if 'file' not in request.files:
            return _error('No file uploaded', 400)
        f = request.files['file']
        fname = (f.filename or '').lower()
        raw = f.read()
        if not raw:
            return _error('Empty file', 400)

        features = []

        # Try text-based formats first
        if fname.endswith(('.geojson', '.json')):
            gj = json.loads(raw)
            return jsonify(gj if gj.get('type') == 'FeatureCollection' else {
                'type': 'FeatureCollection',
                'features': gj.get('features', [{'type': 'Feature', 'geometry': gj, 'properties': {}}])
            })

        if fname.endswith(('.kml', '.xml')):
            parsed = geo_parse.parse_input(raw.decode('utf-8', errors='replace'))
            return jsonify(geo_parse.features_to_geojson(parsed))

        if fname.endswith('.wkt'):
            from shapely import wkt
            geom = wkt.loads(raw.decode('utf-8', errors='replace').strip())
            return jsonify({'type': 'FeatureCollection', 'features': [
                {'type': 'Feature', 'geometry': shp_mapping(geom), 'properties': {}}
            ]})

        # Binary formats via fiona
        tmp_dir = tempfile.mkdtemp(prefix='geo_upload_')
        try:
            # Shapefile in ZIP
            if fname.endswith('.zip'):
                zpath = os.path.join(tmp_dir, 'upload.zip')
                with open(zpath, 'wb') as wf:
                    wf.write(raw)
                # Find .shp inside zip
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(tmp_dir)
                # Find shp file
                shp_files = []
                for root, dirs, files in os.walk(tmp_dir):
                    for fn in files:
                        if fn.lower().endswith('.shp'):
                            shp_files.append(os.path.join(root, fn))
                if not shp_files:
                    # Try as GPKG or other format inside zip
                    for root, dirs, files in os.walk(tmp_dir):
                        for fn in files:
                            if fn.lower().endswith(('.gpkg', '.geojson', '.json', '.kml', '.gpx')):
                                shp_files.append(os.path.join(root, fn))
                if not shp_files:
                    return _error('No shapefile (.shp) or supported file found in ZIP', 400)
                src_path = shp_files[0]
            else:
                # Single file (gpkg, gpx, shp, etc)
                ext = os.path.splitext(fname)[1] or '.gpkg'
                src_path = os.path.join(tmp_dir, 'upload' + ext)
                with open(src_path, 'wb') as wf:
                    wf.write(raw)

            with fiona.open(src_path) as src:
                src_crs = src.crs
                # Reproject to WGS84 if needed
                need_reproject = False
                if src_crs and str(src_crs).upper() not in ('EPSG:4326', '{"INIT": "EPSG:4326"}'):
                    try:
                        from pyproj import CRS, Transformer
                        c = CRS(src_crs)
                        if c.to_epsg() != 4326:
                            need_reproject = True
                            transformer = Transformer.from_crs(c, CRS.from_epsg(4326), always_xy=True)
                    except Exception:
                        pass

                for feat in src:
                    geom = shp_shape(feat['geometry'])
                    if need_reproject:
                        from shapely.ops import transform
                        geom = transform(transformer.transform, geom)
                    props = dict(feat.get('properties', {}))
                    # Convert non-serializable values
                    for k, v in list(props.items()):
                        if v is not None and not isinstance(v, (str, int, float, bool)):
                            props[k] = str(v)
                    features.append({
                        'type': 'Feature',
                        'geometry': shp_mapping(geom),
                        'properties': props,
                    })
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if not features:
            return _error('No features found in file', 400)

        log.info("parse-geometry: %s → %d features", fname, len(features))
        return jsonify({'type': 'FeatureCollection', 'features': features})

    except Exception as e:
        log.error("parse-geometry: %s", traceback.format_exc())
        return _error(f'Failed to parse file: {e}')


# ============ SHARE ============

SHARE_DIR = Path('data/shares')
SHARE_DIR.mkdir(parents=True, exist_ok=True)
SHARE_MAX_BYTES = 1_000_000_000  # 1 GB


def _share_evict():
    """Remove oldest share files until total size < SHARE_MAX_BYTES."""
    files = sorted(SHARE_DIR.glob('*.json.gz'), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    while total > SHARE_MAX_BYTES and files:
        victim = files.pop(0)
        total -= victim.stat().st_size
        victim.unlink(missing_ok=True)
        log.info("share: evicted %s (total was %d MB)", victim.name, total // 1_000_000)


@app.route('/api/v1/share', methods=['POST'])
def share_save():
    """Save analysis result + UI state for sharing. Returns {id, url}.
    
    Content-hash dedup: if payload matches an existing share, reuse its ID.
    Client can also send {reuse_id: "abc123"} to update an existing share in-place.
    """
    try:
        import hashlib
        payload = request.get_json(force=True)
        if not payload:
            return _error('Empty payload')
        
        host = request.headers.get('X-Forwarded-Host', request.host)
        proto = request.headers.get('X-Forwarded-Proto', 'https')
        
        # Extract and remove reuse_id before hashing/storing
        reuse_id = payload.pop('reuse_id', None)
        
        data_json = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        data = gzip.compress(data_json.encode())
        content_hash = hashlib.sha256(data_json.encode()).hexdigest()[:24]
        
        # Check if client wants to reuse an existing share ID
        
        if reuse_id and re.match(r'^[a-f0-9]{12}$', reuse_id):
            existing = SHARE_DIR / f'{reuse_id}.json.gz'
            if existing.exists():
                existing.write_bytes(data)
                existing.touch()
                url = f'{proto}://{host}/?share={reuse_id}'
                log.info("share: updated existing %s (%d KB)", reuse_id, len(data) // 1024)
                return jsonify({'id': reuse_id, 'url': url, 'reused': True})
        
        # Content-hash dedup: check all existing shares for identical content
        for existing_file in SHARE_DIR.glob('*.json.gz'):
            try:
                existing_json = gzip.decompress(existing_file.read_bytes()).decode()
                existing_hash = hashlib.sha256(existing_json.encode()).hexdigest()[:24]
                if existing_hash == content_hash:
                    share_id = existing_file.stem.split('.')[0]
                    existing_file.touch()  # Keep alive (LRU)
                    url = f'{proto}://{host}/?share={share_id}'
                    log.info("share: dedup hit %s", share_id)
                    return jsonify({'id': share_id, 'url': url, 'reused': True})
            except Exception:
                continue
        
        # New share
        share_id = uuid.uuid4().hex[:12]
        (SHARE_DIR / f'{share_id}.json.gz').write_bytes(data)
        _share_evict()
        url = f'{proto}://{host}/?share={share_id}'
        log.info("share: saved %s (%d KB)", share_id, len(data) // 1024)
        return jsonify({'id': share_id, 'url': url, 'reused': False})
    except Exception as e:
        log.error("share save: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/share/<share_id>', methods=['GET'])
def share_load(share_id):
    """Retrieve a saved share."""
    try:
        if not re.match(r'^[a-f0-9]{12}$', share_id):
            return _error('Invalid share ID', 400)
        p = SHARE_DIR / f'{share_id}.json.gz'
        if not p.exists():
            return _error('Share not found', 404)
        data = gzip.decompress(p.read_bytes())
        # Touch file to keep it alive (LRU)
        p.touch()
        return Response(data, mimetype='application/json')
    except Exception as e:
        log.error("share load: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
