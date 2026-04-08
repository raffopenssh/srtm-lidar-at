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

import io
import json
import logging
import os
import tempfile
import time
import traceback
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
                'include_copernicus', 'include_cadastre'):
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


def _try_copernicus(geom_wgs84, *, ndvi=True, landcover=True, sar=False) -> dict | None:
    """Attempt to fetch Copernicus data for a geometry."""
    try:
        import copernicus
        bbox = geom_wgs84.bounds  # (minx, miny, maxx, maxy) = (west, south, east, north)
        bbox_dict = {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]}
        result = {}
        if ndvi:
            try:
                ndvi_data = copernicus.get_ndvi_composite(bbox_dict)
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
                sar_data = copernicus.get_sar_backscatter(bbox_dict, "2023-06-01", "2023-09-30")
                result["vv"] = sar_data["vv"]
                result["vh"] = sar_data["vh"]
            except Exception as e:
                log.warning("Copernicus SAR failed: %s", e)
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
                copernicus_data = _try_copernicus(geom)

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
    try:
        t0 = time.time()
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
            dtm_dates, dsm_dates = None, None
            if include_temporal:
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
                            log.warning("Date %s load failed: %s", d, e)
                except Exception as e:
                    log.warning("Multi-temporal failed, single date: %s", e)
                    data = raster_io.read_dtm_dsm(geom_3035, dataset)
            else:
                data = raster_io.read_dtm_dsm(geom_3035, dataset)

            rgb, spectral = (None, None)
            if include_ortho:
                rgb, spectral = _try_read_ortho(data)

            copernicus_data = None
            if include_copernicus:
                copernicus_data = _try_copernicus(geom)

            building_footprints = None
            if include_cadastre:
                building_footprints = _try_cadastre(
                    geom, data['transform'], data['shape'],
                )

            # Run segmentation pipeline
            result = seg.segment_and_classify(
                data['dtm'], data['dsm'], data['mask'], data['transform'],
                dtm_dates=dtm_dates,
                dsm_dates=dsm_dates,
                spectral=spectral,
                copernicus=copernicus_data,
                building_footprints=building_footprints,
                min_object_size=min_object_size,
                felz_scale=felz_scale,
                rag_threshold=rag_threshold,
            )

            objects = result['objects']
            labels = result['labels']

            # Hansen forest loss calibration
            hansen_evaluation = None
            if include_hansen:
                try:
                    bbox_wgs = geom.bounds  # WGS84
                    hansen_prior = hansen.get_forest_prior(
                        bbox_wgs, data['transform'], data['shape'],
                    )
                    objects = hansen.calibrate_clear_cut(objects, labels, hansen_prior)
                    hansen_evaluation = hansen.evaluate_forest_loss(objects, labels, hansen_prior)
                except Exception as e:
                    log.warning("Hansen calibration failed: %s", e)

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
            },
        }
        if all_evaluation:
            resp["cadastre_evaluation"] = all_evaluation
        if hansen_evaluation:
            resp["hansen_evaluation"] = hansen_evaluation

        return jsonify(resp)
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 3c. SEGMENT RASTER OVERLAY
# ---------------------------------------------------------------------------

# Cache for last segmentation result so legend filter re-renders are instant
_seg_cache = {"labels": None, "objects": None, "mask": None,
              "transform": None, "shape": None, "key": None}

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
    "clear_cut":    (255, 0, 255, 200),
    "construction": (255, 69, 0, 200),
}


def _segment_rgba(labels, objects, mask, type_filter=None):
    """Render segmentation labels as RGBA image."""
    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    obj_map = {o.obj_id: o for o in objects}
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


def _render_seg_overlay(labels, objects, mask, transform, shape_hw, type_filter=None):
    """Render segmentation as RGBA, reproject to WGS84, return overlay response."""
    from rasterio.warp import calculate_default_transform, reproject as rp, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds

    rgba_3035 = _segment_rgba(labels, objects, mask, type_filter)

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

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(100)
        _validate_area(geom_3035)

        # Build a cache key from geometry bounds + dataset + analysis options
        cache_key = f"{geom_3035.bounds}_{dataset}_{include_ortho}_{include_copernicus}_{include_cadastre}_{include_hansen}"

        if _seg_cache["key"] == cache_key:
            # Re-render from cache — instant
            log.info("segment overlay: re-render from cache (filter=%s)", type_filter)
            return _render_seg_overlay(
                _seg_cache["labels"], _seg_cache["objects"],
                _seg_cache["mask"], _seg_cache["transform"],
                _seg_cache["shape"], type_filter,
            )

        # Full segmentation pipeline
        data = raster_io.read_dtm_dsm(geom_3035, dataset)

        rgb, spectral = (None, None)
        if include_ortho:
            rgb, spectral = _try_read_ortho(data)

        copernicus_data = None
        if include_copernicus:
            copernicus_data = _try_copernicus(geom)

        building_footprints = None
        if include_cadastre:
            building_footprints = _try_cadastre(geom, data['transform'], data['shape'])

        result = seg.segment_and_classify(
            data['dtm'], data['dsm'], data['mask'], data['transform'],
            spectral=spectral,
            copernicus=copernicus_data,
            building_footprints=building_footprints,
        )

        objects = result['objects']
        labels = result['labels']

        # Hansen calibration
        if include_hansen:
            try:
                bbox_wgs = geom.bounds
                hansen_prior = hansen.get_forest_prior(
                    bbox_wgs, data['transform'], data['shape'],
                )
                objects = hansen.calibrate_clear_cut(objects, labels, hansen_prior)
            except Exception as e:
                log.warning("Hansen calibration failed in overlay: %s", e)

        # Store in cache for fast re-renders with different type filters
        _seg_cache.update({
            "labels": labels, "objects": objects,
            "mask": data['mask'], "transform": data['transform'],
            "shape": data['shape'], "key": cache_key,
        })
        log.info("segment overlay: full pipeline %.1fs, cached for re-renders", time.time() - t0)

        return _render_seg_overlay(labels, objects, data['mask'],
                                   data['transform'], data['shape'], type_filter)
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

        # Segments
        seg_type_band = None
        if include_segments:
            try:
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
                        type_raster[labels == oid] = float(obj_map[oid].type_code)

                bands.append('segment_type')
                arrays.append(type_raster)
                seg_type_band = type_raster
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

        data = raster_io.read_dtm_dsm(geom_3035, dataset)

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

        data = raster_io.read_dtm_dsm(geom_3035, dataset)

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

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        rgb, nir = ortho_io.read_ortho_for_als(data)

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
        "resolution_ortho": "0.2m (resampled to 1m for analysis)",
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


@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
