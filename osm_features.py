"""OpenStreetMap ground truth fetcher for RF classifier training.

Fetches road/path, waterway, and land-cover data from the Overpass API,
reprojects to EPSG:3035, and rasterises onto a 1m LIDAR grid for use as
training labels alongside cadastre building footprints.

Key design choices:
- Uses `out geom;` format (inline coordinates) to avoid expensive node recursion
- Separate sequential queries per tag category to stay within Overpass limits
- z.overpass-api.de endpoint with proper User-Agent
- 5s pause between queries to respect rate limits
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import LineString, Polygon
from shapely.ops import transform as shapely_transform

log = logging.getLogger("osm_features")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERPASS_URLS = [
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]
USER_AGENT = "srtm-lidar-at/1.0 (raffaelhickisch+exedev@gmail.com)"

CACHE_DIR = Path("/tmp/osm_cache")
MAX_RETRIES = 3
RETRY_SLEEP = 10
REQUEST_TIMEOUT = 120
INTER_QUERY_PAUSE = 5  # seconds between sequential queries

_T_4326_TO_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

# ---------------------------------------------------------------------------
# Highway → label mapping
# ---------------------------------------------------------------------------

_HIGHWAY_ROAD = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "service",
    "unclassified", "pedestrian", "living_street",
}

_HIGHWAY_PATH = {
    "track", "footway", "cycleway", "bridleway", "path", "steps",
}

# ---------------------------------------------------------------------------
# Waterway linestrings → buffer radii in metres (EPSG:3035)
# ---------------------------------------------------------------------------

_WATERWAY_BUFFER = {
    "river": 12.0,
    "canal": 8.0,
    "stream": 3.0,
    "drain": 2.0,
    "ditch": 1.5,
}

# ---------------------------------------------------------------------------
# Landcover tag → label mapping
# ---------------------------------------------------------------------------

_LANDCOVER_MAP: dict[tuple[str, str], str] = {}

# tree / forest
for v in ("forest", "wood"):
    _LANDCOVER_MAP[("landuse", v)] = "tree"
_LANDCOVER_MAP[("natural", "wood")] = "tree"

# crop
for v in ("farmland", "farm"):
    _LANDCOVER_MAP[("landuse", v)] = "crop"

# grass
for v in ("meadow", "grass"):
    _LANDCOVER_MAP[("landuse", v)] = "grass"
_LANDCOVER_MAP[("natural", "grassland")] = "grass"

# orchard / vineyard
_LANDCOVER_MAP[("landuse", "orchard")] = "orchard"
_LANDCOVER_MAP[("landuse", "vineyard")] = "vineyard"

# residential/commercial → roof (approximate)
for v in ("residential", "commercial", "industrial", "retail"):
    _LANDCOVER_MAP[("landuse", v)] = "roof"

# excavation
_LANDCOVER_MAP[("landuse", "quarry")] = "excavation"

# water (polygons)
_LANDCOVER_MAP[("natural", "water")] = "water"
_LANDCOVER_MAP[("waterway", "riverbank")] = "water"
for v in ("reservoir", "basin"):
    _LANDCOVER_MAP[("landuse", v)] = "water"

# rock
for v in ("bare_rock", "scree"):
    _LANDCOVER_MAP[("natural", v)] = "rock"

# bare_soil
for v in ("sand", "beach"):
    _LANDCOVER_MAP[("natural", v)] = "bare_soil"

# garden
for v in ("cemetery", "allotments"):
    _LANDCOVER_MAP[("landuse", v)] = "garden"

# shrub
for v in ("scrub", "heath"):
    _LANDCOVER_MAP[("natural", v)] = "shrub"

# power infrastructure
_LANDCOVER_MAP[("power", "generator")] = "solar_panel"  # needs solar check
_LANDCOVER_MAP[("power", "substation")] = "substation"
_LANDCOVER_MAP[("power", "plant")] = "substation"


# ---------------------------------------------------------------------------
# Landcover rasterization codes
# ---------------------------------------------------------------------------

_LC_LABELS = [
    "",           # 0 = unmatched
    "tree",       # 1
    "crop",       # 2
    "grass",      # 3
    "orchard",    # 4
    "vineyard",   # 5
    "roof",       # 6
    "excavation", # 7
    "water",      # 8
    "rock",       # 9
    "bare_soil",  # 10
    "garden",     # 11
    "shrub",      # 12
]

_LC_LABEL_TO_CODE = {lbl: i for i, lbl in enumerate(_LC_LABELS) if lbl}
_LC_CODE_TO_LABEL = {i: lbl for i, lbl in enumerate(_LC_LABELS)}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(bbox: tuple) -> str:
    raw = f"{bbox[0]:.5f},{bbox[1]:.5f},{bbox[2]:.5f},{bbox[3]:.5f}"
    return "osm_" + hashlib.md5(raw.encode()).hexdigest()[:16]


def _read_cache(bbox: tuple) -> Optional[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{_cache_key(bbox)}.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            log.info("OSM cache hit: %s (%d elements)", p.name,
                     sum(len(v.get("elements", [])) for v in data.values()))
            return data
        except Exception:
            pass
    return None


def _write_cache(bbox: tuple, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{_cache_key(bbox)}.json"
    try:
        p.write_text(json.dumps(data))
    except Exception as e:
        log.warning("Failed to write OSM cache: %s", e)


# ---------------------------------------------------------------------------
# Overpass queries — separate sequential queries with `out geom;`
# ---------------------------------------------------------------------------

def _overpass_post(query: str, tag: str = "") -> Optional[dict]:
    """POST a single query to Overpass with retries across multiple endpoints."""
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
                    log.warning("Overpass [%s] %s %s (attempt %d/%d), sleeping %ds",
                                tag, url.split('/')[2], reason, attempt, MAX_RETRIES, sleep_s)
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
        # If we exhausted retries on this URL, try next
    return None


def _query_all(bbox: tuple[float, float, float, float]) -> Optional[dict]:
    """Fetch OSM data as separate sequential queries. Returns merged result dict."""
    cached = _read_cache(bbox)
    if cached is not None:
        return cached

    west, south, east, north = bbox
    bb = f"[bbox:{south},{west},{north},{east}]"

    # Four separate queries, each using `out geom;` (inline coords, no recursion)
    queries = {
        "highway":  f'[out:json][timeout:60]{bb};way["highway"];out geom;',
        "waterway": f'[out:json][timeout:60]{bb};way["waterway"];out geom;',
        "landuse":  f'[out:json][timeout:60]{bb};(way["landuse"];relation["landuse"];);out geom;',
        "natural":  f'[out:json][timeout:60]{bb};(way["natural"];relation["natural"];);out geom;',
        "power":    f'[out:json][timeout:60]{bb};(way["power"];node["power"="generator"];node["power"="tower"];);out geom;',
    }

    results = {}
    for tag, query in queries.items():
        log.info("Overpass: fetching %s for bbox %s", tag, bbox)
        data = _overpass_post(query, tag=tag)
        if data is not None:
            results[tag] = data
            n = len(data.get("elements", []))
            log.info("Overpass: %s → %d elements", tag, n)
        else:
            log.warning("Overpass: %s query failed, skipping", tag)
            results[tag] = {"elements": []}
        time.sleep(INTER_QUERY_PAUSE)

    _write_cache(bbox, results)
    return results


# ---------------------------------------------------------------------------
# Parse `out geom;` elements → shapely geometries
# ---------------------------------------------------------------------------

def _to_3035(geom):
    return shapely_transform(_T_4326_TO_3035.transform, geom)


def _way_coords(el: dict) -> list[tuple[float, float]]:
    """Extract (lon, lat) coordinate list from an `out geom;` way element."""
    geom = el.get("geometry", [])
    if not geom:
        return []
    return [(pt["lon"], pt["lat"]) for pt in geom]


def _match_landcover(tags: dict) -> Optional[str]:
    for key in ("landuse", "natural", "waterway"):
        val = tags.get(key)
        if val:
            label = _LANDCOVER_MAP.get((key, val))
            if label:
                return label
    return None


def _parse_all(data: dict) -> tuple[list, list]:
    """Parse merged query results into road and landcover geometries.

    Returns
    -------
    roads : list of (shapely_geom_3035, label)  "road" or "path"
    landcover : list of (shapely_geom_3035, label)
    """
    roads: list[tuple] = []
    landcover: list[tuple] = []

    # --- Highway ways ---
    for el in data.get("highway", {}).get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {})
        highway = tags.get("highway", "")
        coords = _way_coords(el)
        if len(coords) < 2:
            continue
        label = None
        if highway in _HIGHWAY_ROAD:
            label = "road"
        elif highway in _HIGHWAY_PATH:
            label = "path"
        if label:
            try:
                line = _to_3035(LineString(coords))
                if not line.is_empty:
                    roads.append((line, label))
            except Exception:
                pass

    # --- Waterway ways ---
    for el in data.get("waterway", {}).get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {})
        ww = tags.get("waterway", "")
        coords = _way_coords(el)
        if len(coords) < 2:
            continue

        # Riverbank polygons (closed ways)
        if ww == "riverbank" and len(coords) >= 4 and coords[0] == coords[-1]:
            try:
                poly = _to_3035(Polygon(coords))
                if poly.is_valid and not poly.is_empty:
                    landcover.append((poly, "water"))
            except Exception:
                pass
            continue

        # Linestring waterways → buffer to polygon
        buf_r = _WATERWAY_BUFFER.get(ww)
        if buf_r:
            try:
                line = _to_3035(LineString(coords))
                if not line.is_empty:
                    buffered = line.buffer(buf_r, cap_style=2)
                    if not buffered.is_empty:
                        landcover.append((buffered, "water"))
            except Exception:
                pass

    # --- Landuse + natural ways/relations ---
    for section in ("landuse", "natural"):
        for el in data.get(section, {}).get("elements", []):
            tags = el.get("tags", {})
            lc_label = _match_landcover(tags)
            if not lc_label:
                continue

            if el.get("type") == "way":
                coords = _way_coords(el)
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    try:
                        poly = _to_3035(Polygon(coords))
                        if poly.is_valid and not poly.is_empty:
                            landcover.append((poly, lc_label))
                    except Exception:
                        pass

            elif el.get("type") == "relation":
                # `out geom;` on relations includes member geometries
                members = el.get("members", [])
                outers = []
                for m in members:
                    if m.get("type") != "way":
                        continue
                    geom = m.get("geometry", [])
                    if not geom:
                        continue
                    coords = [(pt["lon"], pt["lat"]) for pt in geom]
                    role = m.get("role", "outer")
                    if role == "outer" and len(coords) >= 4:
                        if coords[0] == coords[-1]:
                            try:
                                poly = _to_3035(Polygon(coords))
                                if poly.is_valid and not poly.is_empty:
                                    landcover.append((poly, lc_label))
                            except Exception:
                                pass

    log.info("Parsed %d road/path + %d landcover geometries from OSM",
             len(roads), len(landcover))
    return roads, landcover


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def _rasterize_roads(roads, transform, shape):
    """Buffer and rasterize roads/paths. Returns uint8: 0=none, 1=road, 2=path."""
    from rasterio.features import rasterize
    if not roads:
        return np.zeros(shape, dtype=np.uint8)

    pairs = []
    for geom, label in roads:
        try:
            buf = 3.0 if label == "road" else 1.5
            buffered = geom.buffer(buf, cap_style=2)
            if not buffered.is_empty:
                pairs.append((buffered, 1 if label == "road" else 2))
        except Exception:
            pass
    if not pairs:
        return np.zeros(shape, dtype=np.uint8)

    # Roads (1) win over paths (2): put paths first
    pairs.sort(key=lambda x: x[1], reverse=True)
    return rasterize(pairs, out_shape=shape, transform=transform,
                     fill=0, dtype=np.uint8, all_touched=True)


def _rasterize_landcover(landcover, transform, shape):
    """Rasterize landcover polygons. Returns uint8 coded grid."""
    from rasterio.features import rasterize
    if not landcover:
        return np.zeros(shape, dtype=np.uint8)

    pairs = []
    for geom, label in landcover:
        code = _LC_LABEL_TO_CODE.get(label, 0)
        if code and not geom.is_empty:
            pairs.append((geom, code))
    if not pairs:
        return np.zeros(shape, dtype=np.uint8)

    return rasterize(pairs, out_shape=shape, transform=transform,
                     fill=0, dtype=np.uint8, all_touched=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_osm_ground_truth(
    bbox_wgs84: tuple[float, float, float, float],
    transform,
    shape: tuple[int, int],
) -> dict:
    """Fetch OSM data and rasterize road/path + landcover labels.

    Parameters
    ----------
    bbox_wgs84 : (west, south, east, north)
    transform  : rasterio Affine transform in EPSG:3035
    shape      : (rows, cols)

    Returns
    -------
    dict with keys: labels (ndarray), source, n_road_px, n_landcover_px
    """
    empty = {
        "labels": np.full(shape, "", dtype="<U12"),
        "source": "osm",
        "n_road_px": 0,
        "n_landcover_px": 0,
    }

    data = _query_all(bbox_wgs84)
    if data is None:
        return empty

    roads, landcover = _parse_all(data)
    if not roads and not landcover:
        log.info("No relevant OSM features found in bbox")
        return empty

    labels = np.full(shape, "", dtype="<U12")

    # Rasterize landcover first
    n_lc = 0
    try:
        lc_grid = _rasterize_landcover(landcover, transform, shape)
        for code, lbl in _LC_CODE_TO_LABEL.items():
            if code == 0 or not lbl:
                continue
            mask = lc_grid == code
            labels[mask] = lbl
        n_lc = int(np.count_nonzero(lc_grid))
    except Exception as exc:
        log.error("Landcover rasterization failed: %s", exc)

    # Roads override landcover (more precise ground truth)
    n_road = 0
    try:
        road_grid = _rasterize_roads(roads, transform, shape)
        labels[road_grid == 1] = "road"
        labels[road_grid == 2] = "path"
        n_road = int(np.count_nonzero(road_grid))
    except Exception as exc:
        log.error("Road rasterization failed: %s", exc)

    log.info("OSM ground truth: %d road/path px, %d landcover px in %s grid",
             n_road, n_lc, shape)

    return {
        "labels": labels,
        "source": "osm",
        "n_road_px": n_road,
        "n_landcover_px": n_lc,
    }
