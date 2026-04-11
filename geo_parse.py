"""Parse geographic features from KML, GeoJSON, or inline coordinates."""

from __future__ import annotations

import json
import re
from typing import Optional
from xml.etree import ElementTree as ET

from shapely.geometry import shape, Point, LineString, Polygon, mapping, MultiPolygon, box
from shapely.ops import unary_union
import pyproj
import shapely

KML_NS = {
    'kml': 'http://www.opengis.net/kml/2.2',
    'gx': 'http://www.google.com/kml/ext/2.2',
}


def parse_input(data: str | dict, content_type: str = "") -> list[dict]:
    """Parse input geometry. Returns list of {"name": ..., "geometry": shapely_geom_wgs84}.

    Accepts:
    - GeoJSON string or dict (Feature, FeatureCollection, or bare geometry)
    - KML string
    - Simple coordinate string "lon,lat" or "lon,lat;lon,lat;..."
    """
    if isinstance(data, dict):
        return _parse_geojson(data)

    data = data.strip()
    if not data:
        raise ValueError("Empty input")

    # Try GeoJSON
    if data.startswith('{') or data.startswith('['):
        try:
            return _parse_geojson(json.loads(data))
        except (json.JSONDecodeError, Exception):
            pass

    # Try KML
    if '<?xml' in data[:100] or '<kml' in data[:500]:
        return _parse_kml(data)

    # Try simple coordinates
    return _parse_coords(data)


def _parse_geojson(obj: dict) -> list[dict]:
    features = []
    if obj.get("type") == "FeatureCollection":
        for f in obj.get("features", []):
            geom = shape(f.get("geometry", {}))
            name = f.get("properties", {}).get("name", f"feature_{len(features)+1}")
            features.append({"name": name, "geometry": geom, "properties": f.get("properties", {})})
    elif obj.get("type") == "Feature":
        geom = shape(obj.get("geometry", {}))
        name = obj.get("properties", {}).get("name", "feature_1")
        features.append({"name": name, "geometry": geom, "properties": obj.get("properties", {})})
    elif obj.get("type") in ("Point", "LineString", "Polygon", "MultiPoint",
                              "MultiLineString", "MultiPolygon", "GeometryCollection"):
        geom = shape(obj)
        features.append({"name": "geometry_1", "geometry": geom, "properties": {}})
    else:
        raise ValueError(f"Unsupported GeoJSON type: {obj.get('type')}")
    return features


def _parse_kml(kml_str: str) -> list[dict]:
    root = ET.fromstring(kml_str)
    features = []

    for pm in root.iter('{http://www.opengis.net/kml/2.2}Placemark'):
        name_el = pm.find('kml:name', KML_NS)
        name = name_el.text if name_el is not None else f"placemark_{len(features)+1}"

        geom = _kml_placemark_to_geom(pm)
        if geom is not None:
            features.append({"name": name, "geometry": geom, "properties": {"name": name}})

    if not features:
        raise ValueError("No placemarks with geometry found in KML")
    return features


def _kml_placemark_to_geom(pm) -> Optional[shapely.geometry.base.BaseGeometry]:
    ns = 'http://www.opengis.net/kml/2.2'

    # Point
    pt = pm.find(f'{{{ns}}}Point/{{{ns}}}coordinates')
    if pt is not None:
        coords = _parse_kml_coords(pt.text)
        if coords:
            return Point(coords[0][:2])

    # LineString
    ls = pm.find(f'{{{ns}}}LineString/{{{ns}}}coordinates')
    if ls is not None:
        coords = _parse_kml_coords(ls.text)
        if coords and len(coords) >= 2:
            return LineString([(c[0], c[1]) for c in coords])

    # Polygon
    poly = pm.find(f'{{{ns}}}Polygon')
    if poly is not None:
        outer = poly.find(f'{{{ns}}}outerBoundaryIs/{{{ns}}}LinearRing/{{{ns}}}coordinates')
        if outer is not None:
            coords = _parse_kml_coords(outer.text)
            if coords and len(coords) >= 4:
                ring = [(c[0], c[1]) for c in coords]
                # Parse inner rings (holes)
                holes = []
                for inner in poly.findall(f'{{{ns}}}innerBoundaryIs/{{{ns}}}LinearRing/{{{ns}}}coordinates'):
                    hole_coords = _parse_kml_coords(inner.text)
                    if hole_coords and len(hole_coords) >= 4:
                        holes.append([(c[0], c[1]) for c in hole_coords])
                return Polygon(ring, holes)

    # MultiGeometry
    mg = pm.find(f'{{{ns}}}MultiGeometry')
    if mg is not None:
        geoms = []
        for child in mg:
            # Wrap in a fake placemark
            fake = ET.Element('Placemark')
            fake.append(child)
            g = _kml_placemark_to_geom(fake)
            if g:
                geoms.append(g)
        if geoms:
            return unary_union(geoms)

    return None


def _parse_kml_coords(text: str) -> list[tuple[float, ...]]:
    """Parse KML coordinate string 'lon,lat,alt lon,lat,alt ...'"""
    coords = []
    for part in text.strip().split():
        parts = part.strip().split(',')
        if len(parts) >= 2:
            try:
                coords.append(tuple(float(p) for p in parts))
            except ValueError:
                continue
    return coords


def _parse_coords(text: str) -> list[dict]:
    """Parse simple coordinate formats."""
    # Try "lon,lat" or "lat,lon" single point
    parts = re.split(r'[;\n]+', text.strip())
    points = []
    for part in parts:
        nums = re.findall(r'[-+]?\d*\.?\d+', part)
        if len(nums) >= 2:
            lon, lat = float(nums[0]), float(nums[1])
            # Heuristic: if first number looks like lat (>40), swap
            if abs(lon) > 90 and abs(lat) <= 180:
                lon, lat = lat, lon
            points.append((lon, lat))

    if not points:
        raise ValueError("Could not parse coordinates from input")

    if len(points) == 1:
        geom = Point(points[0])
    elif len(points) == 2:
        geom = LineString(points)
    else:
        # Close polygon if needed
        if points[0] != points[-1]:
            points.append(points[0])
        geom = Polygon(points)

    return [{"name": "input", "geometry": geom, "properties": {}}]


# ---------------------------------------------------------------------------
# Austria bounds validation
# ---------------------------------------------------------------------------

# Rough bounding box of Austria in WGS84
_AUSTRIA_BBOX = box(9.5, 46.3, 17.2, 49.0)


def validate_austria_bounds(geom) -> None:
    """Check that a WGS84 geometry intersects Austria's bounding box.

    Raises ValueError if the geometry is entirely outside Austria.
    """
    if not geom.intersects(_AUSTRIA_BBOX):
        raise ValueError(
            "Geometry is outside Austria. This service only covers Austrian territory."
        )


# ---------------------------------------------------------------------------
# Multi-feature union
# ---------------------------------------------------------------------------

def union_features(features: list[dict]) -> list[dict]:
    """Union multiple features into a single feature with a convex hull boundary.

    - Points are buffered by ~100 m (≈0.001° at Austrian latitudes) before union.
    - Mixed geometry types (points, lines, polygons) are supported.
    - If there is only one feature, it is returned as-is.

    Returns a list containing a single feature dict.
    """
    if len(features) <= 1:
        return features

    # Approximate 100 m buffer in WGS84 degrees at ~47°N
    POINT_BUFFER_DEG = 0.001  # ~100 m

    geoms = []
    for f in features:
        g = f["geometry"]
        if g.geom_type == "Point":
            g = g.buffer(POINT_BUFFER_DEG)
        elif g.geom_type == "MultiPoint":
            g = unary_union([p.buffer(POINT_BUFFER_DEG) for p in g.geoms])
        geoms.append(g)

    merged = unary_union(geoms).convex_hull

    # Combine properties from all features
    names = [f.get("name", "") for f in features if f.get("name")]
    combined_name = "; ".join(names) if names else "union"

    return [{
        "name": combined_name,
        "geometry": merged,
        "properties": {"name": combined_name, "source_feature_count": len(features)},
    }]


def features_to_geojson(features: list[dict], properties_extra: dict = None) -> dict:
    """Convert list of feature dicts back to GeoJSON FeatureCollection."""
    gj_features = []
    for f in features:
        props = dict(f.get("properties", {}))
        if properties_extra:
            props.update(properties_extra)
        gj_features.append({
            "type": "Feature",
            "properties": props,
            "geometry": mapping(f["geometry"]),
        })
    return {
        "type": "FeatureCollection",
        "features": gj_features,
    }
