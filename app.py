"""Austrian LIDAR Analysis API.

Provides 4 endpoints for geographic feature analysis using BEV ALS DTM/DSM data:
1. /api/v1/elevation - Enrich features with DSM elevation
2. /api/v1/terrain - Terrain characterisation (slope, ruggedness, etc.)
3. /api/v1/objects - Object detection and classification summary
4. /api/v1/objects/raster - Classified object raster (GeoPackage)
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from flask import Flask, request, jsonify, send_file, Response
from shapely.geometry import mapping, shape, Point

import tile_index as ti
import raster_io
import terrain_analysis as ta
import object_classifier as oc
import geo_parse

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')

MAX_AREA_SQM = 25_000_000  # 25 km² safety limit


def _get_geometry():
    """Extract geometry from request. Supports JSON body, form data, file upload."""
    # Check for file upload
    if 'file' in request.files:
        f = request.files['file']
        content = f.read().decode('utf-8')
        return geo_parse.parse_input(content)

    # Check JSON body
    if request.is_json:
        body = request.get_json()
        # If body has 'geometry' key, parse just that
        if 'geometry' in body:
            geom_input = body['geometry']
        elif 'type' in body:
            geom_input = body
        else:
            raise ValueError("JSON body must contain 'geometry' or be a valid GeoJSON")
        return geo_parse.parse_input(geom_input)

    # Check form data
    if request.form.get('geometry'):
        return geo_parse.parse_input(request.form['geometry'])

    # Check raw body
    data = request.get_data(as_text=True)
    if data:
        return geo_parse.parse_input(data)

    raise ValueError("No geometry provided. Send GeoJSON, KML, or coordinates.")


def _get_params():
    """Extract common query parameters."""
    params = {}
    if request.is_json:
        body = request.get_json()
        params = {k: v for k, v in body.items() if k != 'geometry' and k != 'type' and k != 'features'}

    # Query params override JSON body
    for key in ('dataset', 'min_height', 'max_height', 'min_area',
                'object_types', 'resolution', 'format'):
        val = request.args.get(key)
        if val is not None:
            params[key] = val

    return params


def _validate_area(geom_3035):
    area = geom_3035.area
    if area > MAX_AREA_SQM:
        raise ValueError(
            f"Area too large: {area/1e6:.1f} km² (max {MAX_AREA_SQM/1e6:.0f} km²). "
            f"Use a smaller geometry or the bbox parameter."
        )


def _error(msg, code=400):
    return jsonify({"error": str(msg)}), code


# ---------------------------------------------------------------------------
# 1. ELEVATION ENRICHMENT
# ---------------------------------------------------------------------------

@app.route('/api/v1/elevation', methods=['POST'])
def elevation():
    """Enrich feature points with DSM elevation values."""
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
                # Single point - read DSM value
                e, n = geom_3035.coords[0][:2]
                bounds = (e - 5, n - 5, e + 5, n + 5)
                dsm_data, tf, _ = raster_io.read_window_bbox('DSM', *bounds, dataset, pad=0)
                dtm_data, _, _ = raster_io.read_window_bbox('DTM', *bounds, dataset, pad=0)
                # Get center pixel
                row = int((tf.f - n) / abs(tf.e))
                col = int((e - tf.c) / tf.a)
                row = max(0, min(row, dsm_data.shape[0]-1))
                col = max(0, min(col, dsm_data.shape[1]-1))

                props = dict(feat.get('properties', {}))
                props['dsm_elevation_m'] = round(float(dsm_data[row, col]), 2)
                props['dtm_elevation_m'] = round(float(dtm_data[row, col]), 2)
                props['object_height_m'] = round(float(dsm_data[row, col] - dtm_data[row, col]), 2)
                result_features.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": mapping(geom),
                })

            elif geom.geom_type in ('LineString', 'MultiLineString'):
                # Sample points along line
                _validate_area(geom_3035.buffer(10))
                coords_3035 = list(geom_3035.coords) if geom.geom_type == 'LineString' else \
                    [c for ls in geom_3035.geoms for c in ls.coords]

                bounds = geom_3035.bounds
                dsm_data, tf, _ = raster_io.read_window_bbox('DSM', *bounds, dataset)
                dtm_data, _, _ = raster_io.read_window_bbox('DTM', *bounds, dataset)

                enriched_coords = []
                for e, n in coords_3035:
                    row = int((tf.f - n) / abs(tf.e))
                    col = int((e - tf.c) / tf.a)
                    row = max(0, min(row, dsm_data.shape[0]-1))
                    col = max(0, min(col, dsm_data.shape[1]-1))
                    dsm_val = float(dsm_data[row, col])
                    dtm_val = float(dtm_data[row, col])
                    # Convert back to WGS84
                    pt_wgs = ti.geometry_from_3035(Point(e, n))
                    enriched_coords.append({
                        "lon": round(pt_wgs.x, 8),
                        "lat": round(pt_wgs.y, 8),
                        "dsm_elevation_m": round(dsm_val, 2),
                        "dtm_elevation_m": round(dtm_val, 2),
                        "object_height_m": round(dsm_val - dtm_val, 2),
                    })

                props = dict(feat.get('properties', {}))
                props['elevation_profile'] = enriched_coords
                props['dsm_elevation_min'] = min(p['dsm_elevation_m'] for p in enriched_coords)
                props['dsm_elevation_max'] = max(p['dsm_elevation_m'] for p in enriched_coords)
                result_features.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": mapping(geom),
                })

            else:
                # Polygon - summarise elevation
                _validate_area(geom_3035)
                data = raster_io.read_dtm_dsm(geom_3035, dataset)
                dtm_valid = data['dtm'][data['mask']]
                dsm_valid = data['dsm'][data['mask']]
                ndsm_valid = data['ndsm'][data['mask']]

                props = dict(feat.get('properties', {}))
                props['dsm_elevation'] = {
                    'min': round(float(np.nanmin(dsm_valid)), 2),
                    'max': round(float(np.nanmax(dsm_valid)), 2),
                    'mean': round(float(np.nanmean(dsm_valid)), 2),
                }
                props['dtm_elevation'] = {
                    'min': round(float(np.nanmin(dtm_valid)), 2),
                    'max': round(float(np.nanmax(dtm_valid)), 2),
                    'mean': round(float(np.nanmean(dtm_valid)), 2),
                }
                props['object_heights'] = {
                    'min': round(float(np.nanmin(ndsm_valid)), 2),
                    'max': round(float(np.nanmax(ndsm_valid)), 2),
                    'mean': round(float(np.nanmean(ndsm_valid)), 2),
                }
                props['area_sqm'] = int(np.sum(data['mask']))
                result_features.append({
                    "type": "Feature",
                    "properties": props,
                    "geometry": mapping(geom),
                })

        elapsed = time.time() - t0
        return jsonify({
            "type": "FeatureCollection",
            "features": result_features,
            "meta": {"dataset": dataset, "processing_time_s": round(elapsed, 2)},
        })
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 2. TERRAIN CHARACTERISATION
# ---------------------------------------------------------------------------

@app.route('/api/v1/terrain', methods=['POST'])
def terrain():
    """Terrain characterisation: slope, aspect, ruggedness, curvature."""
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
                # Buffer point for terrain analysis
                geom_3035 = geom_3035.buffer(50)

            dtm_data, mask, tf, crs = raster_io.read_masked('DTM', geom_3035, dataset)
            terrain_stats = ta.characterise_terrain(dtm_data, mask)

            props = dict(feat.get('properties', {}))
            props['terrain'] = terrain_stats
            results.append({
                "type": "Feature",
                "properties": props,
                "geometry": mapping(feat['geometry']),
            })

        elapsed = time.time() - t0
        return jsonify({
            "type": "FeatureCollection",
            "features": results,
            "meta": {"dataset": dataset, "processing_time_s": round(elapsed, 2)},
        })
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 3. OBJECT DETECTION SUMMARY
# ---------------------------------------------------------------------------

@app.route('/api/v1/objects', methods=['POST'])
def objects_summary():
    """Detect and classify above-ground objects. Returns summary + GeoJSON."""
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        min_height = float(params.get('min_height', 0.3))
        min_area = int(params.get('min_area', 2))
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

            data = raster_io.read_dtm_dsm(geom_3035, dataset)
            objects = oc.classify_objects(
                data['ndsm'], data['dtm'], data['mask'],
                data['transform'],
                min_height=min_height,
                min_area=min_area,
            )

            # Apply filters
            if object_types_filter:
                objects = [o for o in objects if o.obj_type in object_types_filter]

            max_height = params.get('max_height')
            if max_height:
                objects = [o for o in objects if o.height_max <= float(max_height)]

            all_objects.extend(objects)

        # Generate summary
        summary = oc.summarise_objects(all_objects)

        # Convert objects to GeoJSON features (centroids in WGS84)
        obj_features = []
        for obj in all_objects:
            centroid_wgs = ti.geometry_from_3035(Point(obj.centroid_e, obj.centroid_n))
            obj_features.append({
                "type": "Feature",
                "properties": {
                    "id": obj.obj_id,
                    "type": obj.obj_type,
                    "type_code": obj.type_code,
                    "height_max_m": obj.height_max,
                    "height_mean_m": obj.height_mean,
                    "height_p90_m": obj.height_p90,
                    "area_sqm": obj.area_sqm,
                    "compactness": obj.compactness,
                    "elongation": obj.elongation,
                    "crown_shape": obj.crown_shape,
                    "height_class": obj.height_class,
                },
                "geometry": mapping(centroid_wgs),
            })

        elapsed = time.time() - t0
        return jsonify({
            "summary": summary,
            "type": "FeatureCollection",
            "features": obj_features,
            "meta": {
                "dataset": dataset,
                "min_height": min_height,
                "min_area": min_area,
                "object_types_filter": object_types_filter,
                "processing_time_s": round(elapsed, 2),
            },
        })
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# 4. OBJECT RASTER
# ---------------------------------------------------------------------------

@app.route('/api/v1/objects/raster', methods=['POST'])
def objects_raster():
    """Generate classified object raster as GeoTIFF or GeoPackage.

    Band 1: Object type code (uint8)
    Band 2: Object height in metres (float32)
    """
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        resolution = float(params.get('resolution', 1.0))
        min_height = float(params.get('min_height', 0.3))
        min_area = int(params.get('min_area', 2))
        out_format = params.get('format', 'geotiff').lower()
        object_types_filter = params.get('object_types', None)
        if isinstance(object_types_filter, str):
            object_types_filter = [t.strip() for t in object_types_filter.split(',')]

        # Use first feature
        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(100)
        _validate_area(geom_3035)

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        objects = oc.classify_objects(
            data['ndsm'], data['dtm'], data['mask'],
            data['transform'],
            min_height=min_height,
            min_area=min_area,
        )

        if object_types_filter:
            objects = [o for o in objects if o.obj_type in object_types_filter]

        type_band, height_band, out_tf = oc.create_classified_raster(
            data['ndsm'], data['mask'], data['transform'],
            objects, output_resolution=resolution,
        )

        # Write to temp file
        suffix = '.tif' if out_format in ('geotiff', 'tif', 'tiff') else '.gpkg'
        driver = 'GTiff' if suffix == '.tif' else 'GPKG'

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name

        h, w = type_band.shape
        with rasterio.open(
            tmp_path, 'w', driver=driver,
            width=w, height=h, count=2,
            dtype='float32',
            crs='EPSG:3035',
            transform=out_tf,
            nodata=-9999,
        ) as dst:
            dst.write(type_band.astype(np.float32), 1)
            dst.write(height_band, 2)
            dst.set_band_description(1, 'object_type_code')
            dst.set_band_description(2, 'object_height_m')
            # Write color table for object types
            dst.update_tags(1, **{
                'OBJECT_TYPE_0': 'ground',
                'OBJECT_TYPE_1': 'low_vegetation',
                'OBJECT_TYPE_2': 'shrub',
                'OBJECT_TYPE_3': 'tree_coniferous',
                'OBJECT_TYPE_4': 'tree_broadleaf',
                'OBJECT_TYPE_5': 'tree_unclassified',
                'OBJECT_TYPE_6': 'building',
                'OBJECT_TYPE_7': 'structure',
                'OBJECT_TYPE_8': 'mast_pole',
                'OBJECT_TYPE_9': 'wall_fence',
                'OBJECT_TYPE_10': 'unclassified',
            })

        elapsed = time.time() - t0
        mime = 'image/tiff' if suffix == '.tif' else 'application/geopackage+sqlite3'
        download_name = f'objects_{dataset}{suffix}'

        return send_file(
            tmp_path,
            mimetype=mime,
            as_attachment=True,
            download_name=download_name,
        )
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# ---------------------------------------------------------------------------
# INFO ENDPOINTS
# ---------------------------------------------------------------------------

@app.route('/api/v1/info', methods=['GET'])
def info():
    return jsonify({
        "name": "Austrian LIDAR Analysis API",
        "version": "1.0.0",
        "description": "Analysis of BEV ALS DTM/DSM LIDAR data for Austria",
        "source": "data.bev.gv.at",
        "resolution": "1m",
        "crs": "EPSG:3035",
        "datasets": list(ti.DATASETS.keys()),
        "tiles": len(ti.TILE_COORDS),
        "object_types": oc.OBJECT_TYPES,
        "endpoints": {
            "POST /api/v1/elevation": "Enrich features with DSM/DTM elevation",
            "POST /api/v1/terrain": "Terrain characterisation (slope, ruggedness, etc.)",
            "POST /api/v1/objects": "Object detection and classification summary",
            "POST /api/v1/objects/raster": "Classified object raster download",
            "GET /api/v1/info": "This endpoint",
            "GET /api/v1/docs/llm.txt": "LLM-readable documentation",
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
