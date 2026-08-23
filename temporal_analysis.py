"""Temporal change detection from multi-date BEV ALS LIDAR data.

Compares DTM + DSM across acquisition dates (20220915, 20230915, 20240915)
to detect and classify landscape changes such as tree growth, felling,
new construction, demolition, earthworks, and vegetation dynamics.

Uses connected-component analysis (scipy.ndimage) to delineate change
regions, then classifies each region based on height profiles, surface
texture, and geometric properties.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

import raster_io
import tile_index as ti

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ChangeEvent:
    """A spatially contiguous region of significant height change."""

    event_type: str
    area_sqm: float
    height_before: float       # mean nDSM in date_a
    height_after: float        # mean nDSM in date_b
    height_change_mean: float
    height_change_max: float
    dtm_change_mean: float     # terrain change (cut/fill)
    dtm_change_max: float      # max terrain change in region
    dsm_change_mean: float     # raw surface change
    centroid_e: float          # EPSG:3035 easting
    centroid_n: float          # EPSG:3035 northing
    bbox: tuple[float, float, float, float]  # (min_e, min_n, max_e, max_n)
    confidence: float          # 0.0 – 1.0
    detail: dict = field(default_factory=dict)  # subtype, measurements etc.


@dataclass
class TreeChange:
    """Per-tree change record between two dates."""

    tree_id: int
    status: str                # "grown", "new", "felled", "stable"
    height_before: float       # peak nDSM date_a  (0 if new)
    height_after: float        # peak nDSM date_b  (0 if felled)
    height_change: float
    crown_area_before: float   # sq m
    crown_area_after: float    # sq m
    centroid_e: float
    centroid_n: float
    label_a: int = 0           # segment label in date_a raster (0 = none)
    label_b: int = 0           # segment label in date_b raster (0 = none)


# ---------------------------------------------------------------------------
# Event-type constants (for programmatic consumers)
# ---------------------------------------------------------------------------

EVENT_TYPES = [
    # Vegetation
    "tree_growth",
    "tree_felling",
    "new_tree",
    "forest_clearcut",
    "vegetation_growth",
    "vegetation_loss",
    # Built environment
    "new_building",
    "demolition",
    "construction",
    # Earthworks (terrain modifications)
    "earthwork_fill",        # terrain raised: dam, embankment, landfill, levelling
    "earthwork_cut",         # terrain lowered: excavation, trench, quarry, grading
    "earthwork_grading",     # terrain smoothed/flattened (roughness reduced)
    "earthwork_dam",         # linear elevated earthwork (levee, dam, embankment)
    "earthwork_trench",      # linear depressed earthwork (drainage, utility trench)
    "earthwork_pond",        # new depression that may hold water
    # Road / surface
    "road_new",              # new road or path where none existed
    "road_resurfaced",       # existing road with surface change (smoother/different DTM)
    "road_widened",          # road corridor expanded
    # General
    "surface_change",        # DSM surface texture changed without height change
    "unclassified_change",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _align_grids(data_a: dict, data_b: dict) -> tuple[dict, dict]:
    """Trim both raster dicts to the overlapping pixel extent.

    Both dictionaries come from :func:`raster_io.read_dtm_dsm` which may
    return arrays that differ by a pixel or two due to float→int rounding.
    We simply crop to the minimum common shape.
    """
    min_h = min(data_a["shape"][0], data_b["shape"][0])
    min_w = min(data_a["shape"][1], data_b["shape"][1])
    for d in (data_a, data_b):
        for key in ("dtm", "dsm", "ndsm", "mask"):
            d[key] = d[key][:min_h, :min_w]
        d["shape"] = (min_h, min_w)
    return data_a, data_b


def _local_roughness(arr: np.ndarray, size: int = 5) -> np.ndarray:
    """Fast local standard-deviation (proxy for surface texture / roughness)."""
    a = arr.astype(np.float64)
    mean = ndimage.uniform_filter(a, size=size, mode="nearest")
    mean_sq = ndimage.uniform_filter(a ** 2, size=size, mode="nearest")
    var = np.clip(mean_sq - mean ** 2, 0, None)
    return np.sqrt(var).astype(np.float32)


def _surface_slope(arr: np.ndarray) -> np.ndarray:
    """Slope in degrees from a 2-D raster."""
    dy, dx = np.gradient(arr, 1.0)
    return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2))).astype(np.float32)


# Backward-compat aliases
_ndsm_roughness = _local_roughness
_ndsm_slope = _surface_slope


def _label_to_map_coords(
    label_arr: np.ndarray,
    label_id: int,
    transform,
) -> tuple[float, float, tuple[float, float, float, float]]:
    """Return (centroid_e, centroid_n, bbox) in EPSG:3035 for *label_id*."""
    rows, cols = np.where(label_arr == label_id)
    if len(rows) == 0:
        return 0.0, 0.0, (0.0, 0.0, 0.0, 0.0)
    mean_col = float(np.mean(cols))
    mean_row = float(np.mean(rows))
    ce = transform.c + mean_col * transform.a
    cn = transform.f + mean_row * transform.e
    min_col, max_col = int(cols.min()), int(cols.max())
    min_row, max_row = int(rows.min()), int(rows.max())
    bbox = (
        round(transform.c + min_col * transform.a, 1),
        round(transform.f + max_row * transform.e, 1),
        round(transform.c + (max_col + 1) * transform.a, 1),
        round(transform.f + min_row * transform.e, 1),
    )
    return round(ce, 1), round(cn, 1), bbox


def _classify_change_region(
    ndsm_a: np.ndarray,
    ndsm_b: np.ndarray,
    dtm_a: np.ndarray,
    dtm_b: np.ndarray,
    dsm_a: np.ndarray,
    dsm_b: np.ndarray,
    region_mask: np.ndarray,
    dtm_roughness_a: np.ndarray | None = None,
    dtm_roughness_b: np.ndarray | None = None,
    dtm_slope_a: np.ndarray | None = None,
    dtm_slope_b: np.ndarray | None = None,
) -> tuple[str, float, dict]:
    """Classify a single connected-component change region.

    Returns
    -------
    (event_type, confidence, detail_dict)
        *detail_dict* provides extra context about the change (e.g.
        "subtype", "direction", etc.).
    """
    ha = ndsm_a[region_mask]  # object heights before
    hb = ndsm_b[region_mask]  # object heights after
    da = dtm_a[region_mask]   # terrain before
    db = dtm_b[region_mask]   # terrain after
    sa = dsm_a[region_mask]   # surface before
    sb = dsm_b[region_mask]   # surface after

    mean_a = float(np.nanmean(ha))
    mean_b = float(np.nanmean(hb))
    dh = mean_b - mean_a                      # nDSM change
    dtm_dh = float(np.nanmean(db - da))       # terrain change
    dtm_dh_max = float(np.nanmax(db - da))    # max terrain rise
    dtm_dh_min = float(np.nanmin(db - da))    # max terrain drop
    dtm_dh_std = float(np.nanstd(db - da))    # terrain change variability
    dsm_dh = float(np.nanmean(sb - sa))       # raw surface change
    area = int(region_mask.sum())

    # Surface texture before and after (nDSM std)
    std_a = float(np.nanstd(ha)) if len(ha) > 2 else 0.0
    std_b = float(np.nanstd(hb)) if len(hb) > 2 else 0.0

    # DTM roughness before and after (if precomputed)
    if dtm_roughness_a is not None and dtm_roughness_b is not None:
        rough_a_vals = dtm_roughness_a[region_mask]
        rough_b_vals = dtm_roughness_b[region_mask]
        dtm_rough_a = float(np.nanmean(rough_a_vals))
        dtm_rough_b = float(np.nanmean(rough_b_vals))
    else:
        dtm_rough_a = float(np.nanstd(da)) if len(da) > 2 else 0.0
        dtm_rough_b = float(np.nanstd(db)) if len(db) > 2 else 0.0

    # DTM slope (mean over region)
    if dtm_slope_a is not None:
        slope_a = float(np.nanmean(dtm_slope_a[region_mask]))
    else:
        slope_a = 0.0
    if dtm_slope_b is not None:
        slope_b = float(np.nanmean(dtm_slope_b[region_mask]))
    else:
        slope_b = 0.0

    rough_before = std_a > 1.5
    rough_after = std_b > 1.5
    flat_before = std_a < 1.0
    flat_after = std_b < 1.0

    # Elongation of the change region (approximate from bounding box)
    rows, cols = np.where(region_mask)
    if len(rows) > 1:
        row_span = rows.max() - rows.min() + 1
        col_span = cols.max() - cols.min() + 1
        elong = max(row_span, col_span) / max(min(row_span, col_span), 1)
        region_width = min(row_span, col_span)
    else:
        elong = 1.0
        region_width = 1

    detail: dict = {}

    # =====================================================================
    # EARTHWORKS — terrain-level changes (DTM changed significantly)
    # =====================================================================
    # DTM change is the primary signal; nDSM change should be small
    # (objects on top didn't change much, the ground itself moved)
    dtm_dominant = abs(dtm_dh) > 0.5 and abs(dtm_dh) > abs(dh - dtm_dh) * 0.7

    if dtm_dominant and abs(dtm_dh) > 0.3:
        roughness_change = dtm_rough_b - dtm_rough_a

        # --- Terrain grading / flattening: roughness significantly reduced ---
        if (roughness_change < -0.15 and abs(dtm_dh) < 2.0
                and dtm_rough_b < dtm_rough_a * 0.6):
            detail["subtype"] = "grading"
            detail["roughness_before"] = round(dtm_rough_a, 3)
            detail["roughness_after"] = round(dtm_rough_b, 3)
            # Check if this is road construction (smooth, flat, elongated)
            if (elong > 4 and dtm_rough_b < 0.1 and slope_b < 5
                    and region_width < 20):
                detail["subtype"] = "road_construction"
                return "road_new", min(0.6 + elong / 20, 0.92), detail
            return "earthwork_grading", min(0.55 + abs(roughness_change) * 2, 0.92), detail

        # --- Dam / embankment: linear raised terrain ---
        if (dtm_dh > 0.5 and elong > 3 and region_width < 30):
            detail["subtype"] = "embankment" if dtm_dh < 3 else "dam"
            detail["height_gain"] = round(dtm_dh, 2)
            return "earthwork_dam", min(0.6 + dtm_dh / 10 + elong / 20, 0.93), detail

        # --- Trench / drainage channel: linear depressed terrain ---
        if (dtm_dh < -0.5 and elong > 3 and region_width < 15):
            detail["subtype"] = "trench" if abs(dtm_dh) < 3 else "channel"
            detail["depth"] = round(abs(dtm_dh), 2)
            return "earthwork_trench", min(0.6 + abs(dtm_dh) / 10 + elong / 20, 0.93), detail

        # --- Pond / basin: compact depression ---
        if (dtm_dh < -0.5 and elong < 3 and dtm_rough_b < 0.1
                and slope_b < 5 and area > 10):
            detail["subtype"] = "pond" if abs(dtm_dh) > 1.5 else "basin"
            detail["depth"] = round(abs(dtm_dh), 2)
            return "earthwork_pond", min(0.55 + abs(dtm_dh) / 8, 0.90), detail

        # --- Fill (terrain raised): landfill, platform, levelling up ---
        if dtm_dh > 0.5:
            detail["subtype"] = "fill"
            detail["height_gain"] = round(dtm_dh, 2)
            if dtm_rough_b < dtm_rough_a * 0.7:
                detail["subtype"] = "levelling_fill"  # filled AND smoothed
            return "earthwork_fill", min(0.6 + dtm_dh / 10, 0.93), detail

        # --- Cut (terrain lowered): excavation, quarry, grading down ---
        if dtm_dh < -0.5:
            detail["subtype"] = "cut"
            detail["depth"] = round(abs(dtm_dh), 2)
            if area > 500:
                detail["subtype"] = "quarry_or_pit"
            return "earthwork_cut", min(0.6 + abs(dtm_dh) / 10, 0.93), detail

    # =====================================================================
    # ROAD SURFACE CHANGES
    # =====================================================================
    # Road resurfacing: ground level, elongated, surface became smoother,
    # small DTM change (new tarmac layer is typically 3-10cm)
    ground_level = mean_a < 0.5 and mean_b < 0.5
    if ground_level:
        # Road resurfacing: DTM rose slightly (new asphalt layer), surface smoother
        if (0.02 < dtm_dh < 0.3 and elong > 3
                and dtm_rough_b < dtm_rough_a * 0.8
                and region_width < 30):
            detail["subtype"] = "resurfacing"
            detail["layer_thickness_m"] = round(dtm_dh, 3)
            return "road_resurfaced", min(0.5 + elong / 30, 0.88), detail

        # Road widening: adjacent ground became smoother + slightly raised
        if (abs(dtm_dh) < 0.5 and dtm_rough_b < 0.1 and slope_b < 5
                and elong > 3 and area > 20):
            # Check if the region is adjacent to already-smooth terrain
            if dtm_rough_a > dtm_rough_b * 1.5:
                detail["subtype"] = "widening"
                return "road_widened", min(0.45 + elong / 30, 0.85), detail

        # New road/path: terrain graded flat, elongated, ground level
        if (elong > 4 and dtm_rough_b < 0.12 and slope_b < 8
                and region_width < 25 and abs(dtm_dh) > 0.1):
            detail["subtype"] = "new_road"
            return "road_new", min(0.5 + elong / 20, 0.90), detail

    # =====================================================================
    # DEMOLITION
    # =====================================================================
    if dh < -3.0 and flat_before and mean_b < 1.0:
        detail["subtype"] = "demolition"
        return "demolition", min(0.6 + abs(dh) / 20.0, 0.95), detail

    # =====================================================================
    # FOREST / TREE CHANGES
    # =====================================================================
    # Forest clearcut: large area, nDSM dropped from >5m to <2m
    if dh < -3.0 and mean_a > 5.0 and mean_b < 2.0 and area > 100:
        detail["subtype"] = "clearcut"
        return "forest_clearcut", min(0.7 + area / 5000.0, 0.95), detail

    # Tree felling: nDSM decreased >3m where canopy existed
    if dh < -3.0 and mean_a > 3.0:
        if rough_before or mean_a > 5.0:
            detail["subtype"] = "felling"
            return "tree_felling", min(0.65 + abs(dh) / 30.0, 0.95), detail

    # =====================================================================
    # NEW CONSTRUCTION
    # =====================================================================
    # New building: flat-topped rise >3m on formerly ground area
    if dh > 3.0 and mean_a < 1.0 and flat_after:
        detail["subtype"] = "new_building"
        return "new_building", min(0.6 + dh / 20.0, 0.95), detail

    # New tree: rose from <1m to >3m
    if dh > 2.0 and mean_a < 1.0 and mean_b > 3.0:
        detail["subtype"] = "new_tree"
        return "new_tree", min(0.5 + dh / 20.0, 0.9), detail

    # Construction: new elevated structure, not tree-like
    if dh > 3.0 and mean_a < 2.0 and flat_after:
        detail["subtype"] = "construction"
        return "construction", min(0.5 + dh / 20.0, 0.9), detail

    # =====================================================================
    # SURFACE TEXTURE CHANGES (DSM changed but not DTM or nDSM much)
    # =====================================================================
    dsm_rough_change = abs(dsm_dh) > 0.05 and abs(dh) < 0.3 and abs(dtm_dh) < 0.3
    if dsm_rough_change:
        detail["subtype"] = "surface_texture"
        return "surface_change", 0.4, detail

    # =====================================================================
    # VEGETATION DYNAMICS
    # =====================================================================
    # Tree growth: canopy increased moderately (0.5 – 5m)
    if 0.5 <= dh <= 5.0 and mean_a > 2.0:
        detail["subtype"] = "tree_growth"
        return "tree_growth", min(0.5 + dh / 10.0, 0.9), detail

    # General vegetation gain
    if 0.3 <= dh <= 2.0:
        detail["subtype"] = "vegetation_growth"
        return "vegetation_growth", min(0.4 + dh / 5.0, 0.85), detail

    # General vegetation loss
    if -2.0 <= dh <= -0.3:
        detail["subtype"] = "vegetation_loss"
        return "vegetation_loss", min(0.4 + abs(dh) / 5.0, 0.85), detail

    return "unclassified_change", 0.3, detail


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_dates(
    geom_3035,
    date_a: str,
    date_b: str,
    pad: float = 5.0,
) -> dict:
    """Pixel-level height comparison between two ALS dates.

    Parameters
    ----------
    geom_3035 : shapely geometry (EPSG:3035)
        Area of interest.
    date_a, date_b : str
        Dataset date keys (e.g. ``"20220915"``, ``"20240915"``).
    pad : float
        Padding in metres around the geometry.

    Returns
    -------
    dict with keys:
        ndsm_change   – 2-D float32 array (nDSM_b − nDSM_a)
        dtm_change    – 2-D float32 array (DTM_b − DTM_a)
        ndsm_a, ndsm_b, dtm_a, dtm_b – source arrays
        mask          – boolean valid-data mask
        transform     – rasterio Affine
        crs           – rasterio CRS
        stats         – summary statistics dict
    """
    for d in (date_a, date_b):
        if d not in ti.DATASETS:
            raise ValueError(
                f"Unknown dataset {d!r}. Available: {sorted(ti.DATASETS)}"
            )

    log.info("Reading DTM+DSM for date %s", date_a)
    data_a = raster_io.read_dtm_dsm(geom_3035, dataset=date_a, pad=pad)
    log.info("Reading DTM+DSM for date %s", date_b)
    data_b = raster_io.read_dtm_dsm(geom_3035, dataset=date_b, pad=pad)

    data_a, data_b = _align_grids(data_a, data_b)

    mask = data_a["mask"] & data_b["mask"]

    ndsm_change = np.where(mask, data_b["ndsm"] - data_a["ndsm"], np.nan)
    dtm_change = np.where(mask, data_b["dtm"] - data_a["dtm"], np.nan)
    dsm_change = np.where(mask, data_b["dsm"] - data_a["dsm"], np.nan)

    # Pre-compute terrain descriptors for earthworks / road detection
    dtm_rough_a = _local_roughness(data_a["dtm"], 5)
    dtm_rough_b = _local_roughness(data_b["dtm"], 5)
    dtm_slope_a = _surface_slope(data_a["dtm"])
    dtm_slope_b = _surface_slope(data_b["dtm"])

    valid = ndsm_change[mask]
    dtm_v = dtm_change[mask]
    dsm_v = dsm_change[mask]
    stats = {}
    if valid.size > 0:
        stats = {
            "date_a": date_a,
            "date_b": date_b,
            "pixel_count": int(mask.sum()),
            "area_sqm": float(mask.sum()),
            "ndsm_change_mean": round(float(np.nanmean(valid)), 3),
            "ndsm_change_std": round(float(np.nanstd(valid)), 3),
            "ndsm_change_min": round(float(np.nanmin(valid)), 3),
            "ndsm_change_max": round(float(np.nanmax(valid)), 3),
            "ndsm_change_median": round(float(np.nanmedian(valid)), 3),
            "dtm_change_mean": round(float(np.nanmean(dtm_v)), 3),
            "dtm_change_std": round(float(np.nanstd(dtm_v)), 3),
            "dtm_change_min": round(float(np.nanmin(dtm_v)), 3),
            "dtm_change_max": round(float(np.nanmax(dtm_v)), 3),
            "dsm_change_mean": round(float(np.nanmean(dsm_v)), 3),
            "pct_increased_1m": round(
                float(np.sum(valid > 1.0) / valid.size * 100), 2
            ),
            "pct_decreased_1m": round(
                float(np.sum(valid < -1.0) / valid.size * 100), 2
            ),
            "pct_dtm_changed_0_3m": round(
                float(np.sum(np.abs(dtm_v) > 0.3) / dtm_v.size * 100), 2
            ),
            "terrain_roughness_before": round(float(np.nanmean(dtm_rough_a[mask])), 4),
            "terrain_roughness_after": round(float(np.nanmean(dtm_rough_b[mask])), 4),
        }
    else:
        stats = {
            "date_a": date_a,
            "date_b": date_b,
            "pixel_count": 0,
            "area_sqm": 0.0,
            "note": "no overlapping valid pixels",
        }

    log.info(
        "compare_dates %s\u2192%s: %d px, mean \u0394h=%.2fm, mean \u0394DTM=%.3fm",
        date_a, date_b, stats.get("pixel_count", 0),
        stats.get("ndsm_change_mean", 0),
        stats.get("dtm_change_mean", 0),
    )

    return {
        "ndsm_change": ndsm_change,
        "dtm_change": dtm_change,
        "dsm_change": dsm_change,
        "ndsm_a": data_a["ndsm"],
        "ndsm_b": data_b["ndsm"],
        "dtm_a": data_a["dtm"],
        "dtm_b": data_b["dtm"],
        "dsm_a": data_a["dsm"],
        "dsm_b": data_b["dsm"],
        "dtm_roughness_a": dtm_rough_a,
        "dtm_roughness_b": dtm_rough_b,
        "dtm_slope_a": dtm_slope_a,
        "dtm_slope_b": dtm_slope_b,
        "mask": mask,
        "transform": data_a["transform"],
        "crs": data_a["crs"],
        "stats": stats,
    }


def detect_changes(
    geom_3035,
    date_a: str,
    date_b: str,
    min_change: float = 1.0,
    min_area: int = 4,
    comparison: Optional[dict] = None,
) -> list[ChangeEvent]:
    """Identify and classify significant change regions.

    Parameters
    ----------
    geom_3035 : shapely geometry (EPSG:3035)
        Area of interest.
    date_a, date_b : str
        Dataset date keys.
    min_change : float
        Minimum absolute height change (metres) to seed regions.
    min_area : int
        Minimum region size in pixels (= sq m at 1 m resolution).
    comparison : dict, optional
        Pre-computed output from :func:`compare_dates`.  If *None*, it will
        be computed automatically.

    Returns
    -------
    list[ChangeEvent]
        Detected change events sorted by descending absolute mean change.
    """
    if comparison is None:
        comparison = compare_dates(geom_3035, date_a, date_b)

    ndsm_change = comparison["ndsm_change"]
    dtm_change = comparison["dtm_change"]
    dsm_change = comparison.get("dsm_change", dtm_change)  # fallback
    ndsm_a = comparison["ndsm_a"]
    ndsm_b = comparison["ndsm_b"]
    dtm_a = comparison["dtm_a"]
    dtm_b = comparison["dtm_b"]
    dsm_a = comparison.get("dsm_a", dtm_a + np.where(comparison["mask"], ndsm_a, 0))
    dsm_b = comparison.get("dsm_b", dtm_b + np.where(comparison["mask"], ndsm_b, 0))
    dtm_rough_a = comparison.get("dtm_roughness_a")
    dtm_rough_b = comparison.get("dtm_roughness_b")
    dtm_slope_a = comparison.get("dtm_slope_a")
    dtm_slope_b = comparison.get("dtm_slope_b")
    mask = comparison["mask"]
    transform = comparison["transform"]

    # --- Build binary mask of significant change pixels ---
    # Use BOTH nDSM change and DTM change — earthworks may not show in nDSM
    sig_ndsm = np.abs(ndsm_change) > min_change
    sig_dtm = np.abs(dtm_change) > max(min_change * 0.5, 0.3)  # lower threshold for terrain
    significant = mask & (sig_ndsm | sig_dtm)

    # Light morphological cleaning: remove isolated pixels, bridge tiny gaps
    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    significant = ndimage.binary_opening(significant, structure=struct, iterations=1)
    significant = ndimage.binary_closing(significant, structure=struct, iterations=1)

    # --- Label connected components ---
    labels, n_labels = ndimage.label(significant, structure=struct)
    log.info(
        "detect_changes: %d connected components (min_change=%.1fm)",
        n_labels, min_change,
    )

    events: list[ChangeEvent] = []
    for label_id in range(1, n_labels + 1):
        region_mask = labels == label_id
        area = int(region_mask.sum())
        if area < min_area:
            continue

        # Pixel values in this region
        ha = ndsm_a[region_mask]
        hb = ndsm_b[region_mask]
        dh = ndsm_change[region_mask]
        dt = dtm_change[region_mask]
        ds = dsm_change[region_mask]

        mean_a = float(np.nanmean(ha))
        mean_b = float(np.nanmean(hb))
        dh_mean = float(np.nanmean(dh))
        dh_max = (
            float(np.nanmax(dh)) if dh_mean > 0 else float(np.nanmin(dh))
        )
        dtm_dh_mean = float(np.nanmean(dt))
        dtm_dh_max = float(np.nanmax(np.abs(dt)))
        dsm_dh_mean = float(np.nanmean(ds))

        # Classify with full context
        event_type, confidence, detail = _classify_change_region(
            ndsm_a, ndsm_b, dtm_a, dtm_b, dsm_a, dsm_b,
            region_mask,
            dtm_roughness_a=dtm_rough_a,
            dtm_roughness_b=dtm_rough_b,
            dtm_slope_a=dtm_slope_a,
            dtm_slope_b=dtm_slope_b,
        )

        centroid_e, centroid_n, bbox = _label_to_map_coords(
            labels, label_id, transform,
        )

        events.append(
            ChangeEvent(
                event_type=event_type,
                area_sqm=float(area),
                height_before=round(mean_a, 2),
                height_after=round(mean_b, 2),
                height_change_mean=round(dh_mean, 2),
                height_change_max=round(dh_max, 2),
                dtm_change_mean=round(dtm_dh_mean, 2),
                dtm_change_max=round(dtm_dh_max, 2),
                dsm_change_mean=round(dsm_dh_mean, 2),
                centroid_e=centroid_e,
                centroid_n=centroid_n,
                bbox=bbox,
                confidence=round(confidence, 3),
                detail=detail,
            )
        )

    # Sort by absolute mean change descending
    events.sort(key=lambda e: abs(e.height_change_mean), reverse=True)
    log.info(
        "detect_changes: %d events classified (%s)",
        len(events),
        ", ".join(
            f"{t}={sum(1 for e in events if e.event_type == t)}"
            for t in sorted({e.event_type for e in events})
        ),
    )
    return events


def temporal_summary(
    geom_3035,
    dates: Optional[Sequence[str]] = None,
) -> dict:
    """Multi-epoch change summary across all consecutive date pairs.

    Parameters
    ----------
    geom_3035 : shapely geometry (EPSG:3035)
    dates : sequence of date strings, optional
        Defaults to all available dates in chronological order.

    Returns
    -------
    dict
        ``pairs``    – per-pair comparison stats and detected events
        ``totals``   – aggregated area changed per event type
        ``trends``   – mean annual height change and per-category trends
    """
    if dates is None:
        dates = sorted(ti.DATASETS.keys())
    dates = list(dates)

    if len(dates) < 2:
        raise ValueError(
            f"Need ≥2 dates for temporal summary; got {dates!r}"
        )

    pairs: list[dict] = []
    all_events: list[ChangeEvent] = []

    for i in range(len(dates) - 1):
        da, db = dates[i], dates[i + 1]
        log.info("temporal_summary: processing pair %s → %s", da, db)
        comp = compare_dates(geom_3035, da, db)
        events = detect_changes(geom_3035, da, db, comparison=comp)
        all_events.extend(events)

        pairs.append({
            "date_a": da,
            "date_b": db,
            "stats": comp["stats"],
            "event_count": len(events),
            "events_by_type": _events_by_type(events),
        })

    # --- Aggregate totals ---
    totals: dict[str, dict] = {}
    for ev in all_events:
        t = ev.event_type
        if t not in totals:
            totals[t] = {"count": 0, "total_area_sqm": 0.0, "mean_dh": []}
        totals[t]["count"] += 1
        totals[t]["total_area_sqm"] += ev.area_sqm
        totals[t]["mean_dh"].append(ev.height_change_mean)

    for t, info in totals.items():
        dh_list = info.pop("mean_dh")
        info["total_area_sqm"] = round(info["total_area_sqm"], 1)
        info["height_change_mean"] = round(float(np.mean(dh_list)), 3)

    # --- Trend: overall mean annual nDSM change ---
    # Parse dates to compute year span
    first_year = int(dates[0][:4])
    last_year = int(dates[-1][:4])
    span_years = max(last_year - first_year, 1)

    total_dh = sum(
        p["stats"].get("ndsm_change_mean", 0) for p in pairs
    )
    trends = {
        "date_range": f"{dates[0]} → {dates[-1]}",
        "span_years": span_years,
        "cumulative_mean_ndsm_change_m": round(total_dh, 3),
        "annual_mean_ndsm_change_m": round(total_dh / span_years, 3),
    }

    log.info(
        "temporal_summary: %d pairs, %d total events, span=%dy",
        len(pairs), len(all_events), span_years,
    )
    return {"pairs": pairs, "totals": totals, "trends": trends}


def detect_tree_growth(
    geom_3035,
    date_a: str,
    date_b: str,
    min_tree_height: float = 3.0,
    crown_min_area: int = 4,
    return_rasters: bool = False,
) -> list[TreeChange]:
    """Per-tree change analysis between two dates.

    Uses the :mod:`object_classifier` tree detection on each date
    independently, then matches trees by spatial proximity.

    Parameters
    ----------
    geom_3035 : shapely geometry (EPSG:3035)
    date_a, date_b : str
        Dataset date keys.
    min_tree_height : float
        Minimum nDSM peak to consider a tree.
    crown_min_area : int
        Minimum crown footprint in pixels.

    Returns
    -------
    list[TreeChange]
        Per-tree change records.
    """
    import object_classifier as oc

    log.info("detect_tree_growth: reading date %s", date_a)
    data_a = raster_io.read_dtm_dsm(geom_3035, dataset=date_a)
    log.info("detect_tree_growth: reading date %s", date_b)
    data_b = raster_io.read_dtm_dsm(geom_3035, dataset=date_b)

    log.info("detect_tree_growth: classifying objects for date %s", date_a)
    objs_a, labels_a = oc.classify_objects(
        data_a["ndsm"], data_a["dtm"], data_a["mask"], data_a["transform"],
        min_height=min_tree_height, min_area=crown_min_area,
        dsm=data_a["dsm"], return_labels=True,
    )
    log.info("detect_tree_growth: classifying objects for date %s", date_b)
    objs_b, labels_b = oc.classify_objects(
        data_b["ndsm"], data_b["dtm"], data_b["mask"], data_b["transform"],
        min_height=min_tree_height, min_area=crown_min_area,
        dsm=data_b["dsm"], return_labels=True,
    )

    trees_a = [o for o in objs_a if "tree" in o.obj_type or o.obj_type == "shrub_bush"]
    trees_b = [o for o in objs_b if "tree" in o.obj_type or o.obj_type == "shrub_bush"]

    log.info(
        "detect_tree_growth: %d trees in %s, %d trees in %s",
        len(trees_a), date_a, len(trees_b), date_b,
    )

    # --- Match trees by nearest centroid ---
    match_radius = 5.0  # metres – maximum centroid shift to be "same tree"
    matched_b: set[int] = set()
    changes: list[TreeChange] = []
    tid = 0

    for ta in trees_a:
        best_dist = float("inf")
        best_tb: Optional[oc.DetectedObject] = None
        best_idx = -1
        for idx, tb in enumerate(trees_b):
            if idx in matched_b:
                continue
            dist = np.sqrt(
                (ta.centroid_e - tb.centroid_e) ** 2
                + (ta.centroid_n - tb.centroid_n) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best_tb = tb
                best_idx = idx

        if best_tb is not None and best_dist <= match_radius:
            matched_b.add(best_idx)
            dh = best_tb.height_max - ta.height_max
            if abs(dh) < 0.3:
                status = "stable"
            else:
                status = "grown" if dh > 0 else "felled"  # partial loss
            tid += 1
            changes.append(
                TreeChange(
                    tree_id=tid,
                    status=status,
                    height_before=round(ta.height_max, 2),
                    height_after=round(best_tb.height_max, 2),
                    height_change=round(dh, 2),
                    crown_area_before=round(ta.area_sqm, 1),
                    crown_area_after=round(best_tb.area_sqm, 1),
                    centroid_e=round(
                        (ta.centroid_e + best_tb.centroid_e) / 2, 1
                    ),
                    centroid_n=round(
                        (ta.centroid_n + best_tb.centroid_n) / 2, 1
                    ),
                    label_a=ta.label,
                    label_b=best_tb.label,
                )
            )
        else:
            # Tree in date_a with no match in date_b → felled
            tid += 1
            changes.append(
                TreeChange(
                    tree_id=tid,
                    status="felled",
                    height_before=round(ta.height_max, 2),
                    height_after=0.0,
                    height_change=round(-ta.height_max, 2),
                    crown_area_before=round(ta.area_sqm, 1),
                    crown_area_after=0.0,
                    centroid_e=round(ta.centroid_e, 1),
                    centroid_n=round(ta.centroid_n, 1),
                    label_a=ta.label,
                )
            )

    # Unmatched trees in date_b → new trees
    for idx, tb in enumerate(trees_b):
        if idx not in matched_b:
            tid += 1
            changes.append(
                TreeChange(
                    tree_id=tid,
                    status="new",
                    height_before=0.0,
                    height_after=round(tb.height_max, 2),
                    height_change=round(tb.height_max, 2),
                    crown_area_before=0.0,
                    crown_area_after=round(tb.area_sqm, 1),
                    centroid_e=round(tb.centroid_e, 1),
                    centroid_n=round(tb.centroid_n, 1),
                    label_b=tb.label,
                )
            )

    # Sort by absolute height change descending
    changes.sort(key=lambda c: abs(c.height_change), reverse=True)
    log.info(
        "detect_tree_growth: %d tree records (%s)",
        len(changes),
        ", ".join(
            f"{s}={sum(1 for c in changes if c.status == s)}"
            for s in sorted({c.status for c in changes})
        ),
    )
    if return_rasters:
        return changes, {
        "labels_a": labels_a, "transform_a": data_a["transform"],
        "labels_b": labels_b, "transform_b": data_b["transform"],
        }
    return changes


# ---------------------------------------------------------------------------
# Helpers for summarisation
# ---------------------------------------------------------------------------


def _events_by_type(events: list[ChangeEvent]) -> dict[str, dict]:
    """Group events by type with counts and total area."""
    groups: dict[str, dict] = {}
    for ev in events:
        t = ev.event_type
        if t not in groups:
            groups[t] = {"count": 0, "total_area_sqm": 0.0, "dh_values": []}
        groups[t]["count"] += 1
        groups[t]["total_area_sqm"] += ev.area_sqm
        groups[t]["dh_values"].append(ev.height_change_mean)
    for t, info in groups.items():
        dh_list = info.pop("dh_values")
        info["total_area_sqm"] = round(info["total_area_sqm"], 1)
        info["height_change_mean"] = round(float(np.mean(dh_list)), 3)
    return groups
