"""OpenStreetMap ground truth fetcher for RF classifier training.

Fetches road/path and land-cover polygons from the Overpass API, reprojects
them to EPSG:3035, and rasterises them onto a 1 m LIDAR grid so they can be
used as training labels alongside cadastre building footprints.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from pyproj import Transformer
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    shape as shapely_shape,
)
from shapely.ops import transform as shapely_transform

log = logging.getLogger("osm_features")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CACHE_DIR = Path("/tmp/osm_cache")
MAX_RETRIES = 3
RETRY_SLEEP = 10  # seconds
REQUEST_TIMEOUT = 120  # seconds (must exceed Overpass server timeout)

# CRS transformer WGS-84 → EPSG:3035 (thread-safe after creation)
_T_4326_TO_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

# ---------------------------------------------------------------------------
# Highway → label mapping
# ---------------------------------------------------------------------------

_HIGHWAY_ROAD = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "residential", "service",
    "unclassified",
    "pedestrian", "living_street",
}

_HIGHWAY_PATH = {
    "track", "footway", "cycleway", "bridleway", "path", "steps",
}

# ---------------------------------------------------------------------------
# Landcover tag → label mapping
# ---------------------------------------------------------------------------

_LANDCOVER_MAP: dict[tuple[str, str], str] = {}

# tree
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

# roof (approximate for built-up)
for v in ("residential", "commercial", "industrial", "retail"):
    _LANDCOVER_MAP[("landuse", v)] = "roof"

# excavation
_LANDCOVER_MAP[("landuse", "quarry")] = "excavation"

# water (polygons)
_LANDCOVER_MAP[("natural", "water")] = "water"
_LANDCOVER_MAP[("waterway", "riverbank")] = "water"
for v in ("reservoir", "basin"):
    _LANDCOVER_MAP[("landuse", v)] = "water"

# water (linestrings — rivers, streams, canals → buffered during rasterization)
_WATERWAY_LINES = {
    "river": 12.0,     # buffer radius in metres (3035)
    "canal": 8.0,
    "stream": 3.0,
    "drain": 2.0,
    "ditch": 1.5,
}

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


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _bbox_cache_key(bbox: tuple[float, float, float, float]) -> str:
    """Deterministic short hash for a rounded WGS-84 bounding box."""
    rounded = tuple(round(c, 4) for c in bbox)
    key = ",".join(f"{c:.4f}" for c in rounded)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _cache_path(bbox: tuple[float, float, float, float]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"osm_{_bbox_cache_key(bbox)}.json"


def _read_cache(bbox: tuple[float, float, float, float]) -> Optional[dict]:
    p = _cache_path(bbox)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            log.debug("OSM cache hit: %s", p.name)
            return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("OSM cache read failed: %s", exc)
    return None


def _write_cache(bbox: tuple[float, float, float, float], data: dict) -> None:
    try:
        _cache_path(bbox).write_text(json.dumps(data))
    except OSError as exc:
        log.warning("Failed to write OSM cache: %s", exc)


# ---------------------------------------------------------------------------
# Overpass API query
# ---------------------------------------------------------------------------

def _build_overpass_query(bbox: tuple[float, float, float, float]) -> str:
    """Build a single Overpass QL query fetching roads + landcover.

    bbox is (west, south, east, north) in WGS-84.
    Overpass expects (south, west, north, east).
    """
    west, south, east, north = bbox
    header = f"[out:json][timeout:90][bbox:{south},{west},{north},{east}];"

    # Single grouped query: roads, waterways (lines), landcover (polygons)
    query = (
        f"{header}\n"
        f"(\n"
        # Roads & paths
        f'  way["highway"];\n'
        # Waterway linestrings (rivers, streams, canals)
        f'  way["waterway"~"^(river|stream|canal|drain|ditch)$"];\n'
        # Landcover polygons
        f'  way["landuse"];\n'
        f'  relation["landuse"];\n'
        f'  way["natural"];\n'
        f'  relation["natural"];\n'
        # Waterway area polygons (riverbank)
        f'  way["waterway"="riverbank"];\n'
        f'  relation["waterway"="riverbank"];\n'
        f");\n"
        f"out body;\n"
        f">;\n"
        f"out skel qt;\n"
    )
    return query


def _query_overpass(bbox: tuple[float, float, float, float]) -> Optional[dict]:
    """Execute an Overpass query with retries on rate-limiting."""
    cached = _read_cache(bbox)
    if cached is not None:
        return cached

    query = _build_overpass_query(bbox)
    log.info("Querying Overpass API for bbox %s", bbox)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                _write_cache(bbox, data)
                return data
            if resp.status_code == 429:
                log.warning(
                    "Overpass rate-limited (attempt %d/%d), sleeping %ds",
                    attempt, MAX_RETRIES, RETRY_SLEEP,
                )
                time.sleep(RETRY_SLEEP)
                continue
            log.error("Overpass HTTP %d: %s", resp.status_code, resp.text[:200])
        except requests.RequestException as exc:
            log.error("Overpass request failed (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)

    log.error("Overpass query failed after %d attempts", MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# OSM JSON → Shapely geometries
# ---------------------------------------------------------------------------

def _parse_elements(data: dict) -> tuple[list, list]:
    """Parse Overpass JSON into road geometries and landcover geometries.

    Returns
    -------
    roads : list of (shapely_geom_3035, label)  where label is "road" or "path"
    landcover : list of (shapely_geom_3035, label)
    """
    # Build node lookup  {id: (lon, lat)}
    nodes: dict[int, tuple[float, float]] = {}
    ways: dict[int, dict] = {}
    relations: list[dict] = []

    for el in data.get("elements", []):
        t = el.get("type")
        if t == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
        elif t == "way":
            ways[el["id"]] = el
        elif t == "relation":
            relations.append(el)

    roads: list[tuple] = []
    landcover: list[tuple] = []

    # --- Process ways -------------------------------------------------------
    for wid, w in ways.items():
        tags = w.get("tags", {})
        nds = w.get("nodes", [])
        if not tags or len(nds) < 2:
            continue

        coords = [nodes[n] for n in nds if n in nodes]
        if len(coords) < 2:
            continue

        highway = tags.get("highway")
        if highway:
            # Build linestring for road/path
            label = None
            if highway in _HIGHWAY_ROAD:
                label = "road"
            elif highway in _HIGHWAY_PATH:
                label = "path"
            if label:
                line = LineString(coords)
                line_3035 = _to_3035(line)
                if not line_3035.is_empty:
                    roads.append((line_3035, label))
            # Don't also treat it as landcover even if it has landuse tags
            continue

        # Waterway linestrings (rivers, streams, canals, drains, ditches)
        waterway = tags.get("waterway")
        if waterway and waterway in _WATERWAY_LINES:
            line = LineString(coords)
            line_3035 = _to_3035(line)
            if not line_3035.is_empty:
                # Store as (geometry, "water", buffer_radius)
                buf = _WATERWAY_LINES[waterway]
                buffered = line_3035.buffer(buf, cap_style=2)
                if not buffered.is_empty:
                    landcover.append((buffered, "water"))
            continue

        # Landcover: closed ways (polygons)
        lc_label = _match_landcover(tags)
        if lc_label and len(coords) >= 4 and coords[0] == coords[-1]:
            try:
                poly = Polygon(coords)
                if poly.is_valid and not poly.is_empty:
                    poly_3035 = _to_3035(poly)
                    if not poly_3035.is_empty:
                        landcover.append((poly_3035, lc_label))
            except Exception:
                pass

    # --- Process relations (multipolygons) ----------------------------------
    for rel in relations:
        tags = rel.get("tags", {})
        lc_label = _match_landcover(tags)
        if not lc_label:
            continue
        if tags.get("type") != "multipolygon":
            continue

        outers: list[list] = []
        inners: list[list] = []
        for member in rel.get("members", []):
            if member.get("type") != "way":
                continue
            wid = member["ref"]
            w = ways.get(wid)
            if not w:
                continue
            coords = [nodes[n] for n in w.get("nodes", []) if n in nodes]
            if len(coords) < 2:
                continue
            role = member.get("role", "outer")
            if role == "inner":
                inners.append(coords)
            else:
                outers.append(coords)

        # Assemble outer rings (may need merging)
        merged_outers = _merge_rings(outers)
        merged_inners = _merge_rings(inners)

        for outer_ring in merged_outers:
            if len(outer_ring) < 4:
                continue
            # Close ring if needed
            if outer_ring[0] != outer_ring[-1]:
                outer_ring.append(outer_ring[0])
            # Find inners that belong to this outer
            try:
                outer_poly = Polygon(outer_ring)
            except Exception:
                continue
            holes = []
            for inner_ring in merged_inners:
                if len(inner_ring) < 4:
                    continue
                if inner_ring[0] != inner_ring[-1]:
                    inner_ring.append(inner_ring[0])
                try:
                    ip = Polygon(inner_ring)
                    if outer_poly.contains(ip.representative_point()):
                        holes.append(inner_ring)
                except Exception:
                    pass
            try:
                poly = Polygon(outer_ring, holes)
                if poly.is_valid and not poly.is_empty:
                    poly_3035 = _to_3035(poly)
                    if not poly_3035.is_empty:
                        landcover.append((poly_3035, lc_label))
            except Exception:
                pass

    log.info(
        "Parsed %d road/path geometries and %d landcover geometries from OSM",
        len(roads), len(landcover),
    )
    return roads, landcover


def _merge_rings(segments: list[list]) -> list[list]:
    """Try to merge way segments that share endpoints into closed rings."""
    if not segments:
        return []

    # Already closed single-way rings
    closed = [s for s in segments if len(s) >= 4 and s[0] == s[-1]]
    open_segs = [s for s in segments if not (len(s) >= 4 and s[0] == s[-1])]

    if not open_segs:
        return closed

    # Greedy merge: join segments sharing endpoints
    merged: list[list] = []
    remaining = list(open_segs)

    while remaining:
        current = list(remaining.pop(0))
        changed = True
        while changed:
            changed = False
            for i, seg in enumerate(remaining):
                if not seg:
                    continue
                if current[-1] == seg[0]:
                    current.extend(seg[1:])
                    remaining.pop(i)
                    changed = True
                    break
                elif current[-1] == seg[-1]:
                    current.extend(reversed(seg[:-1]))
                    remaining.pop(i)
                    changed = True
                    break
                elif current[0] == seg[-1]:
                    current = seg[:-1] + current
                    remaining.pop(i)
                    changed = True
                    break
                elif current[0] == seg[0]:
                    current = list(reversed(seg[1:])) + current
                    remaining.pop(i)
                    changed = True
                    break
        merged.append(current)

    return closed + merged


def _match_landcover(tags: dict) -> Optional[str]:
    """Match OSM tags to our landcover label set."""
    for key in ("landuse", "natural", "waterway"):
        val = tags.get(key)
        if val:
            label = _LANDCOVER_MAP.get((key, val))
            if label:
                return label
    return None


def _to_3035(geom):
    """Transform a shapely geometry from EPSG:4326 → EPSG:3035."""
    return shapely_transform(_T_4326_TO_3035.transform, geom)


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def _rasterize_roads(
    roads: list[tuple],
    transform,
    shape: tuple[int, int],
) -> np.ndarray:
    """Buffer road/path linestrings and rasterize into a label grid.

    Returns an array of uint8 codes: 0=no road, 1=road, 2=path.
    """
    from rasterio.features import rasterize

    if not roads:
        return np.zeros(shape, dtype=np.uint8)

    geom_value_pairs = []
    for geom, label in roads:
        try:
            buf = 3.0 if label == "road" else 1.5
            buffered = geom.buffer(buf, cap_style=2)  # flat caps
            if not buffered.is_empty:
                code = 1 if label == "road" else 2
                geom_value_pairs.append((buffered, code))
        except Exception:
            pass

    if not geom_value_pairs:
        return np.zeros(shape, dtype=np.uint8)

    # Roads (code=1) should win over paths (code=2), so put paths first
    geom_value_pairs.sort(key=lambda x: x[1], reverse=True)

    return rasterize(
        geom_value_pairs,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )


def _rasterize_landcover(
    landcover: list[tuple],
    transform,
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize landcover polygons into a coded grid.

    Label encoding:
      0 = unmatched
      1..N = index into _LC_LABELS
    """
    from rasterio.features import rasterize

    if not landcover:
        return np.zeros(shape, dtype=np.uint8)

    geom_value_pairs = []
    for geom, label in landcover:
        code = _LC_LABEL_TO_CODE.get(label, 0)
        if code and not geom.is_empty:
            geom_value_pairs.append((geom, code))

    if not geom_value_pairs:
        return np.zeros(shape, dtype=np.uint8)

    return rasterize(
        geom_value_pairs,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )


# Landcover label → code mapping
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
    bbox_wgs84 : (west, south, east, north) in WGS-84
    transform : rasterio Affine transform for the output grid (EPSG:3035)
    shape : (rows, cols) of the output grid

    Returns
    -------
    dict with keys:
        labels : np.ndarray (rows, cols) of dtype '<U12' with string labels
                 or "" for unmatched pixels.
        source : "osm"
        n_road_px : int  – number of pixels labelled road or path
        n_landcover_px : int  – number of pixels with a landcover label
    """
    rows, cols = shape
    empty_result = {
        "labels": np.full(shape, "", dtype="<U12"),
        "source": "osm",
        "n_road_px": 0,
        "n_landcover_px": 0,
    }

    # Fetch data from Overpass
    try:
        data = _query_overpass(bbox_wgs84)
    except Exception as exc:
        log.error("Failed to query Overpass API: %s", exc)
        return empty_result

    if data is None:
        return empty_result

    # Parse elements
    try:
        roads, landcover = _parse_elements(data)
    except Exception as exc:
        log.error("Failed to parse OSM elements: %s", exc)
        return empty_result

    if not roads and not landcover:
        log.info("No relevant OSM features found in bbox")
        return empty_result

    # Rasterize
    labels = np.full(shape, "", dtype="<U12")

    try:
        lc_grid = _rasterize_landcover(landcover, transform, shape)
        # Write landcover labels into the string grid
        for code, lbl in _LC_CODE_TO_LABEL.items():
            if code == 0 or not lbl:
                continue
            mask = lc_grid == code
            labels[mask] = lbl
        n_lc = int(np.count_nonzero(lc_grid))
    except Exception as exc:
        log.error("Landcover rasterization failed: %s", exc)
        n_lc = 0

    try:
        road_grid = _rasterize_roads(roads, transform, shape)
        # Roads override landcover (more precise ground truth)
        labels[road_grid == 1] = "road"
        labels[road_grid == 2] = "path"
        n_road = int(np.count_nonzero(road_grid))
    except Exception as exc:
        log.error("Road rasterization failed: %s", exc)
        n_road = 0

    log.info(
        "OSM ground truth: %d road/path pixels, %d landcover pixels in %s grid",
        n_road, n_lc, shape,
    )

    return {
        "labels": labels,
        "source": "osm",
        "n_road_px": n_road,
        "n_landcover_px": n_lc,
    }
