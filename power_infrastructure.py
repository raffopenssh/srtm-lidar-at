"""Austrian energy infrastructure training data from austria-power API + OSM.

Provides ground-truth labels for:
- wind_turbine: AustroControl + igwindkraft wind turbines
- mast: antennas, towers, poles, stacks
- solar_panel: large ground-mount solar farms (>1000m² from OSM polygons)
- substation: transformer stations (from OSM polygons)

Used by the RF training pipeline alongside cadastre and OSM landcover labels.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform

log = logging.getLogger("power_infrastructure")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POWER_API_URL = "https://austria-power.exe.xyz:8000/api/infrastructure"

OVERPASS_URLS = [
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
USER_AGENT = "srtm-lidar-at/1.0 (raffaelhickisch+exedev@gmail.com)"

CACHE_DIR = Path("/tmp/power_infra_cache")
CACHE_MAX_AGE_S = 24 * 3600  # 24 hours

MAX_RETRIES = 3
RETRY_SLEEP = 10
REQUEST_TIMEOUT = 120
INTER_QUERY_PAUSE = 5  # seconds between sequential Overpass queries

# CRS transformers
_T_4326_TO_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
_T_3035_TO_4326 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

# Types to skip from AustroControl obstacle data
_SKIP_TYPES = {"Crane", "Cable car", "Cableway", "Cable Car"}

# AustroControl obstacle type → (label, default_radius_m)
_OBSTACLE_TYPE_MAP = {
    "Antenna":        ("mast", 5.0),
    "Mast":           ("mast", 3.0),
    "Tower":          ("mast", 3.0),
    "Pole":           ("mast", 3.0),
    "Stack":          ("mast", 5.0),
    "Building":       ("roof", None),   # radius computed from height
}

# (category, type) → label for non-obstacle features
_CATEGORY_TYPE_MAP = {
    ("wind_energy", "Windmill farm"):       "wind_turbine",
    ("wind_energy", "Windpower plant"):     "wind_turbine",
    ("wind_energy", "windpark"):            "wind_turbine",
    ("solar_energy", "solar"):              "solar_panel",
    ("substation", "transformer_station"):  "substation",
    ("substation", "substation"):           "substation",
    ("substation", "substation_380kv"):     "substation",
}

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(prefix: str, bbox: tuple) -> str:
    raw = f"{prefix}_{bbox[0]:.5f},{bbox[1]:.5f},{bbox[2]:.5f},{bbox[3]:.5f}"
    return prefix + "_" + hashlib.md5(raw.encode()).hexdigest()[:16]


def _read_cache(prefix: str, bbox: tuple) -> Optional[dict | list]:
    """Read cached JSON if it exists and is not expired."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{_cache_key(prefix, bbox)}.json"
    if p.exists():
        try:
            age = time.time() - p.stat().st_mtime
            if age > CACHE_MAX_AGE_S:
                log.debug("Cache expired (%ds old): %s", int(age), p.name)
                return None
            data = json.loads(p.read_text())
            log.info("Cache hit (%s): %s", prefix, p.name)
            return data
        except Exception:
            pass
    return None


def _write_cache(prefix: str, bbox: tuple, data) -> None:
    """Write data to cache as JSON."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{_cache_key(prefix, bbox)}.json"
    try:
        p.write_text(json.dumps(data))
        log.debug("Cache written: %s", p.name)
    except Exception as e:
        log.warning("Failed to write cache: %s", e)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _to_3035_point(lon: float, lat: float) -> tuple[float, float]:
    """Convert WGS84 lon/lat to EPSG:3035 easting/northing."""
    return _T_4326_TO_3035.transform(lon, lat)


def _to_3035_geom(geom):
    """Transform a shapely geometry from WGS84 to EPSG:3035."""
    return shapely_transform(_T_4326_TO_3035.transform, geom)


# ---------------------------------------------------------------------------
# Austria Power API
# ---------------------------------------------------------------------------

def _fetch_power_api(bbox_wgs84: tuple[float, float, float, float]) -> Optional[dict]:
    """Fetch infrastructure data from austria-power API.

    Parameters
    ----------
    bbox_wgs84 : (west, south, east, north)

    Returns
    -------
    GeoJSON FeatureCollection dict, or None on failure.
    """
    cached = _read_cache("power_api", bbox_wgs84)
    if cached is not None:
        return cached

    west, south, east, north = bbox_wgs84
    params = {
        "bbox": f"{west},{south},{east},{north}",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                POWER_API_URL,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                _write_cache("power_api", bbox_wgs84, data)
                n_feat = len(data.get("features", []))
                log.info("Power API: %d features for bbox %s", n_feat, bbox_wgs84)
                return data
            if resp.status_code in (429, 502, 503, 504):
                sleep_s = RETRY_SLEEP * attempt
                log.warning(
                    "Power API HTTP %d (attempt %d/%d), sleeping %ds",
                    resp.status_code, attempt, MAX_RETRIES, sleep_s,
                )
                time.sleep(sleep_s)
                continue
            log.error("Power API HTTP %d: %s", resp.status_code, resp.text[:200])
            return None
        except requests.RequestException as exc:
            log.error("Power API request error (attempt %d/%d): %s",
                      attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)

    log.error("Power API: all %d attempts failed", MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# Parse Power API response → infrastructure point list
# ---------------------------------------------------------------------------

def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _parse_power_features(geojson: dict) -> list[dict]:
    """Parse GeoJSON FeatureCollection into normalized infrastructure point dicts.

    Returns list of dicts with keys:
        lon, lat, label, height_agl_m, radius_m, confidence, source,
        year (optional), uncertain (optional)
    """
    results = []
    features = geojson.get("features", [])

    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})

        # Extract coordinates (centroid for all geometry types)
        coords = geom.get("coordinates")
        if coords is None:
            continue
        geom_type = geom.get("type", "")
        if geom_type == "Point":
            lon, lat = coords[0], coords[1]
        elif geom_type == "LineString":
            mid = len(coords) // 2
            lon, lat = coords[mid][0], coords[mid][1]
        elif geom_type == "Polygon":
            ring = coords[0]
            lon = sum(c[0] for c in ring) / len(ring)
            lat = sum(c[1] for c in ring) / len(ring)
        else:
            continue

        source = props.get("source", "")
        category = props.get("category", "")
        feat_type = props.get("type", "")
        height = _safe_float(props.get("height_agl_m"))
        area_sqm = _safe_float(props.get("area_sqm"))
        rotor_diameter = _safe_float(props.get("rotor_diameter_m"))
        hub_height = _safe_float(props.get("hub_height_m"))
        year = _safe_int(props.get("year"))

        # --- Skip temporary / non-useful ---
        if feat_type in _SKIP_TYPES:
            continue

        # --- Look up (category, type) in the clean mapping ---
        label = _CATEGORY_TYPE_MAP.get((category, feat_type))

        if label == "wind_turbine":
            radius = max(rotor_diameter / 2.0, 15.0) if rotor_diameter else 15.0
            h = hub_height or height
            pt = {
                "lon": lon, "lat": lat,
                "label": "wind_turbine",
                "height_agl_m": h,
                "radius_m": radius,
                "confidence": 0.9,
                "source": source or "igwindkraft",
            }
            if year is not None:
                pt["year"] = year
                if year < 2010:
                    pt["uncertain"] = True
                    pt["confidence"] = 0.7
            results.append(pt)
            continue

        if label == "solar_panel":
            # Include ALL solar plants from the API as training data.
            # Even smaller installations (~100-1000 m²) are valuable for
            # teaching the RF model the spectral/height signature of solar
            # panels.  Very tiny entries (<50 m²) are skipped as noise.
            effective_area = area_sqm if area_sqm and area_sqm > 50 else 200.0
            radius = math.sqrt(effective_area / math.pi)
            pt = {
                "lon": lon, "lat": lat,
                "label": "solar_panel",
                "height_agl_m": height or 2.5,
                "radius_m": max(radius, 8.0),
                "confidence": 0.85 if (area_sqm and area_sqm > 1000) else 0.7,
                "source": "osm_power",
            }
            results.append(pt)
            continue

        if label == "substation":
            pt = {
                "lon": lon, "lat": lat,
                "label": "substation",
                "height_agl_m": height or 10.0,
                "radius_m": 30.0,
                "confidence": 0.8,
                "source": source or "power_api",
            }
            results.append(pt)
            continue

        # --- AustroControl obstacles (Antenna, Mast, Tower, etc.) ---
        if feat_type in _OBSTACLE_TYPE_MAP:
            obs_label, default_radius = _OBSTACLE_TYPE_MAP[feat_type]

            if obs_label == "roof":
                if area_sqm and area_sqm > 0:
                    radius = math.sqrt(area_sqm) / 2.0
                elif height and height > 0:
                    radius = max(height * 1.0, 10.0)
                else:
                    radius = 15.0
            else:
                radius = default_radius

            pt = {
                "lon": lon, "lat": lat,
                "label": obs_label,
                "height_agl_m": height,
                "radius_m": radius,
                "confidence": 0.85 if height else 0.7,
                "source": "austrocontrol",
            }
            results.append(pt)
            continue

        # Unrecognised — skip silently
        log.debug("Skipping unrecognised feature: category=%s type=%s source=%s",
                  category, feat_type, source)

    log.info("Parsed %d infrastructure points from power API", len(results))
    return results


def fetch_power_infrastructure(
    bbox_wgs84: tuple[float, float, float, float],
) -> list[dict]:
    """Fetch power infrastructure points for a given WGS84 bbox.

    Parameters
    ----------
    bbox_wgs84 : (west, south, east, north)

    Returns
    -------
    List of dicts, each with keys:
        lon, lat, label, height_agl_m, radius_m, confidence, source
    """
    geojson = _fetch_power_api(bbox_wgs84)
    if geojson is None:
        log.warning("No data from power API")
        return []
    return _parse_power_features(geojson)


# ---------------------------------------------------------------------------
# Overpass: power-related polygons (solar farms, substations)
# ---------------------------------------------------------------------------

def _overpass_post(query: str, tag: str = "") -> Optional[dict]:
    """POST a single query to Overpass with retries across multiple endpoints.

    Same pattern as osm_features.py.
    """
    for url in OVERPASS_URLS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    url,
                    data={"data": query},
                    headers={"User-Agent": USER_AGENT},
                    timeout=REQUEST_TIMEOUT,
                )
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 504):
                    reason = "rate-limited" if resp.status_code == 429 else "timeout"
                    sleep_s = RETRY_SLEEP * attempt
                    log.warning(
                        "Overpass [%s] %s %s (attempt %d/%d), sleeping %ds",
                        tag, url.split('/')[2], reason, attempt, MAX_RETRIES, sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                log.error("Overpass [%s] HTTP %d from %s",
                          tag, resp.status_code, url.split('/')[2])
                break  # try next URL
            except requests.RequestException as exc:
                log.error("Overpass [%s] request error (attempt %d/%d): %s",
                          tag, attempt, MAX_RETRIES, exc)
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_SLEEP)
        # Exhausted retries on this URL, try next
    return None


def _parse_way_polygon(el: dict) -> Optional[Polygon]:
    """Parse an Overpass `out geom;` way element into a shapely Polygon in EPSG:3035."""
    geom_pts = el.get("geometry", [])
    if not geom_pts:
        return None
    coords = [(pt["lon"], pt["lat"]) for pt in geom_pts]
    if len(coords) < 4:
        return None
    # Ensure closed ring
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        poly = _to_3035_geom(Polygon(coords))
        if poly.is_valid and not poly.is_empty and poly.area > 1.0:
            return poly
    except Exception:
        pass
    return None


def _parse_relation_polygons(el: dict) -> list[Polygon]:
    """Parse an Overpass `out geom;` relation into shapely Polygons in EPSG:3035."""
    polys = []
    members = el.get("members", [])
    for m in members:
        if m.get("type") != "way":
            continue
        geom_pts = m.get("geometry", [])
        if not geom_pts:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in geom_pts]
        role = m.get("role", "outer")
        if role == "outer" and len(coords) >= 4:
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            try:
                poly = _to_3035_geom(Polygon(coords))
                if poly.is_valid and not poly.is_empty and poly.area > 1.0:
                    polys.append(poly)
            except Exception:
                pass
    return polys


def fetch_osm_power_polygons(
    bbox_wgs84: tuple[float, float, float, float],
) -> list[tuple]:
    """Fetch power-related polygons from Overpass API.

    Returns list of (shapely_polygon_3035, label) tuples.
    Labels are 'solar_panel' or 'substation'.
    """
    cached = _read_cache("osm_power", bbox_wgs84)
    if cached is not None:
        # Reconstruct from cached raw Overpass responses
        return _parse_osm_power_elements(cached)

    west, south, east, north = bbox_wgs84
    bb = f"[bbox:{south},{west},{north},{east}]"

    # Two separate queries to respect Overpass rate limits
    queries = {
        "solar": (
            f'[out:json][timeout:60]{bb};'
            f'(way["power"="plant"]["plant:source"="solar"];'
            f'way["power"="generator"]["generator:source"="solar"];'
            f'relation["power"="plant"]["plant:source"="solar"];'
            f'relation["power"="generator"]["generator:source"="solar"];);'
            f'out geom;'
        ),
        "substation": (
            f'[out:json][timeout:60]{bb};'
            f'(way["power"="substation"];'
            f'relation["power"="substation"];);'
            f'out geom;'
        ),
    }

    raw_results = {}
    for tag, query in queries.items():
        log.info("Overpass [power]: fetching %s for bbox %s", tag, bbox_wgs84)
        data = _overpass_post(query, tag=f"power_{tag}")
        if data is not None:
            raw_results[tag] = data
            n = len(data.get("elements", []))
            log.info("Overpass [power]: %s → %d elements", tag, n)
        else:
            log.warning("Overpass [power]: %s query failed, skipping", tag)
            raw_results[tag] = {"elements": []}
        time.sleep(INTER_QUERY_PAUSE)

    _write_cache("osm_power", bbox_wgs84, raw_results)
    return _parse_osm_power_elements(raw_results)


def _parse_osm_power_elements(raw_results: dict) -> list[tuple]:
    """Parse cached Overpass results into (polygon_3035, label) tuples."""
    polygons = []

    # Solar farms — include all polygons >50m² (even small rooftop arrays
    # are valuable training samples for the solar_panel class)
    for el in raw_results.get("solar", {}).get("elements", []):
        if el.get("type") == "way":
            poly = _parse_way_polygon(el)
            if poly is not None and poly.area > 50:
                polygons.append((poly, "solar_panel"))
        elif el.get("type") == "relation":
            for poly in _parse_relation_polygons(el):
                if poly.area > 50:
                    polygons.append((poly, "solar_panel"))

    # Substations
    for el in raw_results.get("substation", {}).get("elements", []):
        if el.get("type") == "way":
            poly = _parse_way_polygon(el)
            if poly is not None:
                polygons.append((poly, "substation"))
        elif el.get("type") == "relation":
            for poly in _parse_relation_polygons(el):
                polygons.append((poly, "substation"))

    log.info("Parsed %d power polygons from OSM (solar: %d, substation: %d)",
             len(polygons),
             sum(1 for _, l in polygons if l == "solar_panel"),
             sum(1 for _, l in polygons if l == "substation"))
    return polygons


# ---------------------------------------------------------------------------
# Match infrastructure to segments
# ---------------------------------------------------------------------------

def match_infrastructure_to_segments(
    infra_points: list[dict],
    osm_polygons: list[tuple],
    features_list: list[dict],
    seg_labels: np.ndarray,
    transform,
    ndsm: Optional[np.ndarray] = None,
) -> dict[int, str]:
    """Match infrastructure features to image segments.

    Strategy:
    1. For each infra point: find segments whose centroid is within
       radius_m + 20m buffer.
    2. If height_agl_m is available and nDSM exists: check that segment's
       h_mean or h_max is within 30% of known height (for structures >10m).
    3. For OSM polygons: rasterize and do majority-vote per segment.

    Parameters
    ----------
    infra_points : list of dicts from fetch_power_infrastructure()
    osm_polygons : list of (shapely_polygon_3035, label) from fetch_osm_power_polygons()
    features_list : per-segment feature dicts (with centroid_e, centroid_n, h_mean, etc.)
    seg_labels : segment label raster (h, w), same shape as ndsm
    transform : rasterio Affine transform (EPSG:3035)
    ndsm : normalised DSM array (h, w), optional — used for height agreement check

    Returns
    -------
    dict mapping segment index (position in features_list) → label string
    """
    matched = {}  # seg_index → label
    match_confidences = {}  # seg_index → confidence (for conflict resolution)

    if not features_list:
        return matched

    # Build spatial index of segment centroids in EPSG:3035
    seg_centroids_e = np.array([f.get("centroid_e", 0.0) for f in features_list])
    seg_centroids_n = np.array([f.get("centroid_n", 0.0) for f in features_list])

    # --- 1. Match point features by proximity + height ---
    n_point_matched = 0
    for pt in infra_points:
        pt_e, pt_n = _to_3035_point(pt["lon"], pt["lat"])
        search_radius = pt.get("radius_m", 15.0) + 20.0
        pt_height = pt.get("height_agl_m")
        pt_label = pt["label"]
        pt_conf = pt.get("confidence", 0.7)

        # Find all segments within search radius
        dist = np.sqrt((seg_centroids_e - pt_e) ** 2 + (seg_centroids_n - pt_n) ** 2)
        candidates = np.where(dist < search_radius)[0]

        if len(candidates) == 0:
            continue

        for seg_idx in candidates:
            seg_idx = int(seg_idx)
            feat = features_list[seg_idx]

            # Height agreement check for tall structures
            if pt_height is not None and pt_height > 10.0:
                seg_h_max = feat.get("h_max", 0.0)
                seg_h_mean = feat.get("h_mean", 0.0)
                # Use h_max for tall structures (mast/wind_turbine may only
                # occupy part of a segment)
                seg_h = max(seg_h_max, seg_h_mean)
                if seg_h > 0:
                    # Check within 30% tolerance
                    ratio = seg_h / pt_height
                    if ratio < 0.7 or ratio > 1.3:
                        # Height mismatch — reduce confidence but don't
                        # completely reject (nDSM might clip tall structures)
                        if ratio < 0.3:
                            continue  # too far off, skip
                        pt_conf_adj = pt_conf * 0.6
                    else:
                        pt_conf_adj = pt_conf
                else:
                    # No height info for segment — use point data as-is
                    pt_conf_adj = pt_conf * 0.8
            else:
                pt_conf_adj = pt_conf

            # Prefer closer segments and higher confidence
            d = dist[seg_idx]
            distance_penalty = 1.0 - (d / search_radius) * 0.3
            effective_conf = pt_conf_adj * distance_penalty

            # Only update if better than existing match
            if seg_idx not in matched or effective_conf > match_confidences.get(seg_idx, 0):
                matched[seg_idx] = pt_label
                match_confidences[seg_idx] = effective_conf
                n_point_matched += 1

    log.info("Point matching: %d segments matched from %d infrastructure points",
             len([k for k in matched]), len(infra_points))

    # --- 2. Match OSM polygon features by rasterization + majority vote ---
    n_poly_matched = 0
    if osm_polygons and seg_labels is not None:
        try:
            from rasterio.features import rasterize

            shape = seg_labels.shape

            # Encode labels: solar_panel=1, substation=2
            _poly_code = {"solar_panel": 1, "substation": 2}
            _code_to_label = {1: "solar_panel", 2: "substation"}

            pairs = []
            for poly, label in osm_polygons:
                code = _poly_code.get(label)
                if code and not poly.is_empty:
                    pairs.append((poly, code))

            if pairs:
                poly_raster = rasterize(
                    pairs, out_shape=shape, transform=transform,
                    fill=0, dtype=np.uint8, all_touched=False,
                )

                for seg_idx, feat in enumerate(features_list):
                    seg_id = feat.get("label")
                    if seg_id is None:
                        continue

                    seg_mask = seg_labels == seg_id
                    seg_px = int(seg_mask.sum())
                    if seg_px < 2:
                        continue

                    codes_in_seg = poly_raster[seg_mask]
                    nonzero = codes_in_seg[codes_in_seg != 0]
                    if len(nonzero) == 0:
                        continue

                    unique, counts = np.unique(nonzero, return_counts=True)
                    best_code = int(unique[counts.argmax()])
                    best_frac = counts.max() / seg_px

                    if best_frac >= 0.15:  # at least 15% overlap
                        poly_label = _code_to_label.get(best_code)
                        if poly_label:
                            poly_conf = 0.85 * best_frac
                            # Polygon labels are higher-quality for area features
                            if seg_idx not in matched or poly_conf > match_confidences.get(seg_idx, 0):
                                matched[seg_idx] = poly_label
                                match_confidences[seg_idx] = poly_conf
                                n_poly_matched += 1
        except ImportError:
            log.warning("rasterio not available — skipping polygon matching")
        except Exception as exc:
            log.error("Polygon matching failed: %s", exc)

    log.info("Polygon matching: %d additional segments matched from %d polygons",
             n_poly_matched, len(osm_polygons))

    return matched


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def get_infrastructure_training_labels(
    bbox_wgs84: tuple[float, float, float, float],
    features_list: list[dict],
    seg_labels: np.ndarray,
    transform,
    ndsm: Optional[np.ndarray] = None,
) -> tuple[dict[int, str], dict]:
    """Get infrastructure training labels for segments.

    Top-level function called by the RF training pipeline. Fetches data from
    the austria-power API and OSM, then matches to segments.

    Parameters
    ----------
    bbox_wgs84 : (west, south, east, north)
    features_list : per-segment feature dicts
    seg_labels : segment label raster (h, w)
    transform : rasterio Affine transform (EPSG:3035)
    ndsm : normalised DSM (h, w), optional

    Returns
    -------
    (labels_dict, stats_dict)
        labels_dict : {segment_index: label_string}
        stats_dict  : {source_counts, n_matched, ...}
    """
    stats = {
        "n_api_points": 0,
        "n_osm_polygons": 0,
        "n_matched": 0,
        "source_counts": {},
        "label_counts": {},
        "errors": [],
    }

    # Fetch infrastructure points from austria-power API
    infra_points = []
    try:
        infra_points = fetch_power_infrastructure(bbox_wgs84)
        stats["n_api_points"] = len(infra_points)
        # Count by source
        for pt in infra_points:
            src = pt.get("source", "unknown")
            stats["source_counts"][src] = stats["source_counts"].get(src, 0) + 1
    except Exception as exc:
        log.error("Failed to fetch power infrastructure: %s", exc)
        stats["errors"].append(f"power_api: {exc}")

    # Fetch OSM power polygons
    osm_polygons = []
    try:
        osm_polygons = fetch_osm_power_polygons(bbox_wgs84)
        stats["n_osm_polygons"] = len(osm_polygons)
    except Exception as exc:
        log.error("Failed to fetch OSM power polygons: %s", exc)
        stats["errors"].append(f"osm_power: {exc}")

    if not infra_points and not osm_polygons:
        log.info("No infrastructure data found for bbox %s", bbox_wgs84)
        return {}, stats

    # Match to segments
    try:
        labels_dict = match_infrastructure_to_segments(
            infra_points, osm_polygons, features_list, seg_labels,
            transform, ndsm=ndsm,
        )
    except Exception as exc:
        log.error("Infrastructure segment matching failed: %s", exc)
        stats["errors"].append(f"matching: {exc}")
        return {}, stats

    stats["n_matched"] = len(labels_dict)

    # Count matched labels
    for label in labels_dict.values():
        stats["label_counts"][label] = stats["label_counts"].get(label, 0) + 1

    log.info(
        "Infrastructure training labels: %d matched segments "
        "(from %d API points + %d OSM polygons) — %s",
        len(labels_dict), len(infra_points), len(osm_polygons),
        stats["label_counts"],
    )

    return labels_dict, stats
