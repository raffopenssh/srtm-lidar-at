"""BEV ALS tile index for Austria's LIDAR data.

Manages the grid of 50km x 50km tiles (EPSG:3035, 1m resolution) published by
the Austrian Federal Office of Metrology and Surveying (BEV) for both DTM
(Digital Terrain Model) and DSM (Digital Surface Model) layers.

Files are ~12 GB each but support HTTP range requests, so we access them via
GDAL's /vsicurl/ virtual filesystem for efficient windowed reads.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pyproj
import shapely.ops

if TYPE_CHECKING:
    import shapely.geometry

# ---------------------------------------------------------------------------
# Dataset registry – keyed by date string.  Each entry carries base URLs for
# the DTM and DSM layers.
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, str]] = {
    "20220915": {
        "dtm_base": "https://data.bev.gv.at/download/ALS/DTM/20220915/",
        "dsm_base": "https://data.bev.gv.at/download/ALS/DSM/20220915/",
    },
    "20230915": {
        "dtm_base": "https://data.bev.gv.at/download/ALS/DTM/20230915/",
        "dsm_base": "https://data.bev.gv.at/download/ALS/DSM/20230915/",
    },
    "20240915": {
        "dtm_base": "https://data.bev.gv.at/download/ALS/DTM/20240915/",
        "dsm_base": "https://data.bev.gv.at/download/ALS/DSM/20240915/",
    },
}

DEFAULT_DATASET = "20240915"

# ---------------------------------------------------------------------------
# Tile geometry constants
# ---------------------------------------------------------------------------

TILE_SIZE = 50_000          # metres – each tile spans 50 km in both axes
TILE_PIXELS = 50_001        # 1 m resolution → 50001 pixels per axis
TILE_RES = 1.0              # ground sampling distance in metres

# ---------------------------------------------------------------------------
# Complete tile list – 55 tiles, given as (northing, easting) in EPSG:3035.
# Both DTM and DSM share the same grid positions.
# ---------------------------------------------------------------------------

TILE_COORDS: set[tuple[int, int]] = {
    # Row N2550000
    (2_550_000, 4_650_000),
    # Row N2600000
    (2_600_000, 4_300_000),
    (2_600_000, 4_350_000),
    (2_600_000, 4_400_000),
    (2_600_000, 4_450_000),
    (2_600_000, 4_500_000),
    (2_600_000, 4_550_000),
    (2_600_000, 4_600_000),
    (2_600_000, 4_650_000),
    (2_600_000, 4_700_000),
    (2_600_000, 4_750_000),
    # Row N2650000
    (2_650_000, 4_250_000),
    (2_650_000, 4_300_000),
    (2_650_000, 4_350_000),
    (2_650_000, 4_400_000),
    (2_650_000, 4_450_000),
    (2_650_000, 4_500_000),
    (2_650_000, 4_550_000),
    (2_650_000, 4_600_000),
    (2_650_000, 4_650_000),
    (2_650_000, 4_700_000),
    (2_650_000, 4_750_000),
    (2_650_000, 4_800_000),
    # Row N2700000
    (2_700_000, 4_250_000),
    (2_700_000, 4_300_000),
    (2_700_000, 4_350_000),
    (2_700_000, 4_400_000),
    (2_700_000, 4_450_000),
    (2_700_000, 4_500_000),
    (2_700_000, 4_550_000),
    (2_700_000, 4_600_000),
    (2_700_000, 4_650_000),
    (2_700_000, 4_700_000),
    (2_700_000, 4_750_000),
    (2_700_000, 4_800_000),
    # Row N2750000
    (2_750_000, 4_500_000),
    (2_750_000, 4_550_000),
    (2_750_000, 4_600_000),
    (2_750_000, 4_650_000),
    (2_750_000, 4_700_000),
    (2_750_000, 4_750_000),
    (2_750_000, 4_800_000),
    (2_750_000, 4_850_000),
    # Row N2800000
    (2_800_000, 4_500_000),
    (2_800_000, 4_550_000),
    (2_800_000, 4_600_000),
    (2_800_000, 4_650_000),
    (2_800_000, 4_700_000),
    (2_800_000, 4_750_000),
    (2_800_000, 4_800_000),
    # Row N2850000
    (2_850_000, 4_600_000),
    (2_850_000, 4_650_000),
    (2_850_000, 4_700_000),
    (2_850_000, 4_750_000),
    (2_850_000, 4_800_000),
}

assert len(TILE_COORDS) == 55, f"Expected 55 tiles, got {len(TILE_COORDS)}"

# ---------------------------------------------------------------------------
# Cached pyproj Transformers
# ---------------------------------------------------------------------------


@functools.cache
def _transformer_wgs84_to_3035() -> pyproj.Transformer:
    return pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


@functools.cache
def _transformer_3035_to_wgs84() -> pyproj.Transformer:
    return pyproj.Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tile_url(
    layer: str,
    northing: int,
    easting: int,
    dataset: str = DEFAULT_DATASET,
) -> str:
    """Return a ``/vsicurl/`` URL for a specific tile.

    Parameters
    ----------
    layer:
        ``'DTM'`` or ``'DSM'``.
    northing, easting:
        Tile origin in EPSG:3035 metres (e.g. 2700000, 4500000).
    dataset:
        Date key into :data:`DATASETS` (default ``"20240915"``).

    Raises
    ------
    ValueError
        If the layer, dataset, or tile coordinates are invalid.
    """
    layer = layer.upper()
    if layer not in ("DTM", "DSM"):
        raise ValueError(f"layer must be 'DTM' or 'DSM', got {layer!r}")
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {sorted(DATASETS)}"
        )
    if (northing, easting) not in TILE_COORDS:
        raise ValueError(
            f"No tile at N{northing}E{easting}. "
            f"Use find_tiles_for_bbox() to discover valid tiles."
        )

    base_key = f"{layer.lower()}_base"
    base_url = DATASETS[dataset][base_key]
    filename = f"ALS_{layer}_CRS3035RES50000mN{northing}E{easting}.tif"
    return f"/vsicurl/{base_url}{filename}"


def find_tiles_for_bbox(
    min_e: float,
    min_n: float,
    max_e: float,
    max_n: float,
) -> list[tuple[int, int]]:
    """Return tile coordinates whose 50 km cell intersects the given bbox.

    Parameters
    ----------
    min_e, min_n, max_e, max_n:
        Bounding box in EPSG:3035 (easting / northing in metres).

    Returns
    -------
    list of (northing, easting)
        Sorted list of tile origins that overlap the query bbox.
    """
    hits: list[tuple[int, int]] = []
    for n, e in TILE_COORDS:
        # Tile covers [e, e+TILE_SIZE] × [n, n+TILE_SIZE]
        if e + TILE_SIZE < min_e or e > max_e:
            continue
        if n + TILE_SIZE < min_n or n > max_n:
            continue
        hits.append((n, e))
    hits.sort()
    return hits


def bbox_wgs84_to_3035(
    west: float,
    south: float,
    east: float,
    north: float,
) -> tuple[float, float, float, float]:
    """Convert a WGS 84 bbox to an EPSG:3035 bbox.

    Parameters
    ----------
    west, south, east, north:
        Longitude / latitude bounds in decimal degrees (EPSG:4326).

    Returns
    -------
    (min_easting, min_northing, max_easting, max_northing)
        Bounding box in EPSG:3035 metres.  The output bbox is the *envelope*
        of the four transformed corner points, which is conservative for the
        moderate extents of Austria.
    """
    tx = _transformer_wgs84_to_3035()
    # Transform all four corners and take the envelope.
    xs, ys = tx.transform(
        [west, east, east, west],
        [south, south, north, north],
    )
    return (min(xs), min(ys), max(xs), max(ys))


def geometry_to_3035(geom_wgs84: shapely.geometry.base.BaseGeometry) -> shapely.geometry.base.BaseGeometry:
    """Transform a Shapely geometry from WGS 84 (EPSG:4326) to EPSG:3035.

    Coordinates are interpreted as (longitude, latitude) on input and
    (easting, northing) on output.
    """
    tx = _transformer_wgs84_to_3035()
    return shapely.ops.transform(tx.transform, geom_wgs84)


def geometry_from_3035(geom_3035: shapely.geometry.base.BaseGeometry) -> shapely.geometry.base.BaseGeometry:
    """Transform a Shapely geometry from EPSG:3035 to WGS 84 (EPSG:4326).

    Coordinates are interpreted as (easting, northing) on input and
    (longitude, latitude) on output.
    """
    tx = _transformer_3035_to_wgs84()
    return shapely.ops.transform(tx.transform, geom_3035)
