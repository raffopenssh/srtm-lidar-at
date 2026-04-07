"""Object detection and classification from BEV ALS DTM + DSM.

Two-phase approach:
  Phase 1 — Pixel-level surface classification using 3D shape analysis
  Phase 2 — Watershed segmentation of elevated pixels into individual objects,
            classified by aggregating pixel-level labels + morphometric features.

Surface signatures (empirically derived from Austrian ALS 1m data):
  - Ground (road/path):  nDSM < 0.3m, DTM smooth (std3 < 0.1)
  - Ground (meadow):     nDSM < 0.3m, DTM normal (std3 0.1–0.5)
  - Ground (rough/rock):  nDSM < 0.3m, DTM rough (std3 > 0.5)
  - Low vegetation:       nDSM 0.3–2m
  - Shrub/bush:           nDSM 2–4m
  - Building roof:        nDSM > 3m, nDSM surface slope < 15°, local std5 < 1.5m
  - Tree canopy:          nDSM > 4m, nDSM surface slope > 40°, roughness > 1.0
  - Coniferous crown:     peak with steep radial dropoff (>5m in 5m radius)
  - Broadleaf crown:      dome with gentle dropoff (<4m in 5m radius)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage import measure, morphology, segmentation, feature

log = logging.getLogger(__name__)

# Object type codes for raster output
OBJECT_TYPES = {
    "ground": 0,
    "road_path": 1,
    "meadow_field": 2,
    "rough_ground": 3,        # rocks, scree, uneven terrain
    "low_vegetation": 4,      # 0.3–2m
    "shrub_bush": 5,          # 2–4m
    "tree_coniferous": 6,
    "tree_broadleaf": 7,
    "tree_unclassified": 8,
    "building": 9,
    "structure": 10,          # small non-building structures
    "mast_pole": 11,
    "wall_fence": 12,
    "water": 13,
    "unclassified": 14,
}

OBJECT_TYPE_NAMES = {v: k for k, v in OBJECT_TYPES.items()}

# Logarithmic height class boundaries (metres) up to 80m
_HEIGHT_BREAKS = [0.5, 1, 2, 4, 8, 15, 25, 40, 60, 80]


def _height_class(h: float) -> str:
    prev = 0
    for brk in _HEIGHT_BREAKS:
        if h < brk:
            return f"{prev}-{brk}m"
        prev = brk
    return f">{_HEIGHT_BREAKS[-1]}m"


@dataclass
class DetectedObject:
    """A detected above-ground object."""
    obj_id: int
    obj_type: str
    type_code: int
    height_max: float
    height_mean: float
    height_p90: float
    area_sqm: float
    perimeter_m: float
    compactness: float
    elongation: float
    height_std: float
    centroid_e: float         # EPSG:3035
    centroid_n: float
    bbox: tuple[float, float, float, float]
    crown_shape: str = ""
    height_class: str = ""


# ---------------------------------------------------------------------------
# Phase 1: Pixel-level surface classification
# ---------------------------------------------------------------------------

def _classify_pixels(
    ndsm: np.ndarray,
    dtm: np.ndarray,
    dsm: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Classify every pixel by its 3D surface properties.

    Returns uint8 array of type codes.
    """
    h, w = ndsm.shape
    px = np.full((h, w), OBJECT_TYPES["unclassified"], dtype=np.uint8)
    n = np.where(mask, ndsm, 0.0).astype(np.float32)

    # --- Compute surface descriptors (fast: use uniform_filter for std) ---

    # nDSM surface slope
    ndy, ndx = np.gradient(n, 1.0)
    ndsm_slope = np.degrees(np.arctan(np.sqrt(ndx**2 + ndy**2)))

    # Fast local std via sqrt(E[x^2] - E[x]^2)
    def _fast_local_std(arr, size):
        a = arr.astype(np.float64)
        mean = ndimage.uniform_filter(a, size=size, mode='nearest')
        mean_sq = ndimage.uniform_filter(a**2, size=size, mode='nearest')
        var = np.clip(mean_sq - mean**2, 0, None)
        return np.sqrt(var).astype(np.float32)

    # DTM local roughness (3×3)
    dtm_std3 = _fast_local_std(dtm, 3)

    # nDSM local height variation (5×5)
    ndsm_std5 = _fast_local_std(n, 5)

    # DSM surface roughness (3×3)
    dsm_std3 = _fast_local_std(dsm, 3)

    # DTM slope
    ddy, ddx = np.gradient(dtm, 1.0)
    dtm_slope = np.degrees(np.arctan(np.sqrt(ddx**2 + ddy**2)))

    # --- Classify ground (nDSM < 0.3m) ---
    ground = mask & (n < 0.3)
    # Water: very smooth DTM + very low DSM roughness + flat
    water = ground & (dtm_std3 < 0.03) & (dsm_std3 < 0.03) & (dtm_slope < 3)
    px[water] = OBJECT_TYPES["water"]
    # Road/path: smooth ground
    road = ground & ~water & (dtm_std3 < 0.15) & (dsm_std3 < 0.15)
    px[road] = OBJECT_TYPES["road_path"]
    # Rough ground (rocks, scree)
    rough = ground & (dtm_std3 > 0.5)
    px[rough] = OBJECT_TYPES["rough_ground"]
    # Meadow/field: everything else at ground level
    meadow = ground & ~water & ~road & ~rough
    px[meadow] = OBJECT_TYPES["meadow_field"]

    # --- Low vegetation (0.3–2m) ---
    low_veg = mask & (n >= 0.3) & (n < 2.0)
    px[low_veg] = OBJECT_TYPES["low_vegetation"]

    # --- Shrubs/bushes (2–4m) ---
    shrub = mask & (n >= 2.0) & (n < 4.0)
    px[shrub] = OBJECT_TYPES["shrub_bush"]

    # --- Elevated objects (≥4m): building roofs vs tree canopy ---
    elevated = mask & (n >= 4.0)

    # Building roof signature: flat DSM surface locally
    # nDSM_slope < 15°, nDSM_std5 < 1.5m, dsm_std3 < 0.8m
    flat_surface = elevated & (ndsm_slope < 15) & (ndsm_std5 < 1.5) & (dsm_std3 < 0.8)
    # Also require gentle terrain underneath (buildings aren't on 30° slopes)
    building_px = flat_surface & (dtm_slope < 25)
    px[building_px] = OBJECT_TYPES["building"]

    # Tree canopy: rough, steep nDSM surface
    tree_canopy = elevated & ~building_px & (ndsm_slope > 20)
    px[tree_canopy] = OBJECT_TYPES["tree_unclassified"]  # refined later

    # Remaining elevated pixels that are neither clearly building nor tree
    other_elevated = elevated & ~building_px & ~tree_canopy
    px[other_elevated] = OBJECT_TYPES["tree_unclassified"]

    return px


# ---------------------------------------------------------------------------
# Phase 2: Segment into individual objects
# ---------------------------------------------------------------------------

def _segment_objects(
    ndsm: np.ndarray,
    mask: np.ndarray,
    pixel_classes: np.ndarray,
    min_height: float,
) -> np.ndarray:
    """Watershed segmentation splitting canopy into individual objects."""
    n = np.where(mask, ndsm, 0.0).astype(np.float32)
    elevated = (n >= min_height) & mask

    # --- Low objects (<4m): simple connected components ---
    low = elevated & (n < 4.0)
    low_clean = morphology.binary_opening(low, morphology.disk(1))
    low_labels, n_low = ndimage.label(low_clean)

    # --- Tall objects (≥4m): watershed ---
    tall = elevated & (n >= 4.0)
    tall_labels = np.zeros_like(n, dtype=np.int32)
    if np.any(tall):
        tall_labels = _watershed_tall(n, tall)

    # --- Buildings: separate segmentation so they don't merge with trees ---
    building_px = (pixel_classes == OBJECT_TYPES["building"]) & elevated
    if np.any(building_px):
        building_clean = morphology.binary_opening(building_px, morphology.square(2))
        bld_labels, n_bld = ndimage.label(building_clean)
        # Override tall_labels where we have building pixels
        max_tall = int(tall_labels.max()) if np.any(tall_labels) else 0
        bld_mask = bld_labels > 0
        tall_labels[bld_mask] = bld_labels[bld_mask] + max_tall

    # --- Merge ---
    combined = np.zeros_like(n, dtype=np.int32)
    combined[tall_labels > 0] = tall_labels[tall_labels > 0]
    offset = int(combined.max()) if np.any(combined > 0) else 0
    low_active = (low_labels > 0) & (combined == 0)
    combined[low_active] = low_labels[low_active] + offset

    # --- Ground-level features: large connected regions ---
    ground_classes = {OBJECT_TYPES["road_path"], OBJECT_TYPES["water"]}
    for gc in ground_classes:
        gc_mask = (pixel_classes == gc) & (combined == 0)
        if np.any(gc_mask):
            gc_clean = morphology.binary_opening(gc_mask, morphology.disk(2))
            gc_labels, _ = ndimage.label(gc_clean)
            offset2 = int(combined.max()) if np.any(combined > 0) else 0
            active = gc_labels > 0
            combined[active] = gc_labels[active] + offset2

    return combined


def _watershed_tall(ndsm: np.ndarray, tall_mask: np.ndarray) -> np.ndarray:
    """Marker-controlled watershed with height-adaptive local maxima."""
    smooth = ndimage.gaussian_filter(ndsm * tall_mask, sigma=2.0)
    markers = np.zeros(ndsm.shape, dtype=np.int32)
    marker_id = 0

    # Scale-dependent peak detection:
    # Tall trees (>20m) have ~8m crown radius
    # Medium trees (10-20m) have ~5m crown radius
    # Short trees (4-10m) have ~3m crown radius
    for thresh, min_dist in [(20, 8), (10, 5), (4, 3)]:
        lmax = feature.peak_local_max(
            smooth, min_distance=min_dist, threshold_abs=thresh,
            labels=tall_mask.astype(np.uint8),
        )
        for r, c in lmax:
            if markers[r, c] == 0:
                marker_id += 1
                markers[r, c] = marker_id

    if marker_id == 0:
        labels, _ = ndimage.label(tall_mask)
        return labels

    elev = (-smooth * 1000).astype(np.int32)
    elev = np.where(tall_mask, elev, 0)
    return segmentation.watershed(elev, markers=markers, mask=tall_mask, compactness=0.5)


# ---------------------------------------------------------------------------
# Phase 3: Classify each segment using aggregated pixel labels + morphometrics
# ---------------------------------------------------------------------------

def classify_objects(
    ndsm: np.ndarray,
    dtm: np.ndarray,
    mask: np.ndarray,
    transform,
    min_height: float = 0.3,
    min_area: int = 1,
    max_objects: int = 50000,
    dsm: np.ndarray | None = None,
) -> list[DetectedObject]:
    """Full pipeline: pixel classification → segmentation → object classification."""
    if dsm is None:
        dsm = dtm + np.where(mask, ndsm, 0)

    # Phase 1: pixel-level classification
    pixel_classes = _classify_pixels(ndsm, dtm, dsm, mask)
    log.info("Pixel classification complete")

    # Phase 2: segmentation
    labels = _segment_objects(ndsm, mask, pixel_classes, min_height)
    log.info(f"Segmented into {labels.max()} regions")

    # Phase 3: classify each segment
    n = np.where(mask, ndsm, 0.0).astype(np.float32)

    # Pre-compute surface descriptors for crown shape analysis
    smooth = ndimage.gaussian_filter(n, sigma=2.0)

    # DTM slope
    ddy, ddx = np.gradient(dtm, 1.0)
    dtm_slope = np.degrees(np.arctan(np.sqrt(ddx**2 + ddy**2)))

    regions = measure.regionprops(labels, intensity_image=n)
    log.info(f"Analysing {len(regions)} regions")

    objects = []
    for reg in regions:
        if reg.area < min_area or len(objects) >= max_objects:
            continue

        obj_mask = labels == reg.label
        obj_pixels = n[obj_mask]
        h_max = float(np.nanmax(obj_pixels))
        h_mean = float(np.nanmean(obj_pixels))
        h_std = float(np.nanstd(obj_pixels))
        h_p90 = float(np.nanpercentile(obj_pixels, 90))

        if h_max < min_height and h_max > 0.01:  # allow ground features (h≈0)
            pass  # keep ground features
        elif h_max < 0.01:
            # Pure ground segment — classify by dominant pixel class
            pass

        area = float(reg.area)
        perimeter = float(reg.perimeter) if reg.perimeter > 0 else 1
        compactness = (4 * np.pi * area / perimeter**2)
        elongation = (reg.major_axis_length / reg.minor_axis_length
                      if reg.minor_axis_length > 0 else 999)

        # Centroid in map coords
        row, col = reg.centroid
        e = transform.c + col * transform.a
        n_coord = transform.f + row * transform.e

        min_row, min_col, max_row, max_col = reg.bbox
        bbox = (
            round(transform.c + min_col * transform.a, 1),
            round(transform.f + max_row * transform.e, 1),
            round(transform.c + max_col * transform.a, 1),
            round(transform.f + min_row * transform.e, 1),
        )

        # --- Determine object type from pixel class majority + shape ---
        px_in_obj = pixel_classes[obj_mask]
        class_counts = np.bincount(px_in_obj, minlength=max(OBJECT_TYPES.values()) + 1)
        dominant_class = int(np.argmax(class_counts))

        # Terrain slope under this object
        slope_under = float(np.nanmean(dtm_slope[obj_mask]))

        obj_type, crown_shape = _classify_segment(
            h_max, h_mean, h_std, h_p90, area, compactness, elongation,
            dominant_class, class_counts, slope_under,
            smooth, obj_mask, int(row), int(col),
        )

        # Skip pure ground unless it's a road or water feature
        if obj_type in ("meadow_field", "ground", "rough_ground") and area < 100:
            continue

        obj = DetectedObject(
            obj_id=len(objects) + 1,
            obj_type=obj_type,
            type_code=OBJECT_TYPES.get(obj_type, 14),
            height_max=round(h_max, 2),
            height_mean=round(h_mean, 2),
            height_p90=round(h_p90, 2),
            area_sqm=round(area, 1),
            perimeter_m=round(perimeter, 1),
            compactness=round(min(compactness, 2.0), 3),
            elongation=round(min(elongation, 999), 2),
            height_std=round(h_std, 2),
            centroid_e=round(e, 1),
            centroid_n=round(n_coord, 1),
            bbox=bbox,
            crown_shape=crown_shape,
            height_class=_height_class(h_max),
        )
        objects.append(obj)

    log.info(f"Classified {len(objects)} objects")
    return objects


def _classify_segment(
    h_max, h_mean, h_std, h_p90, area, compactness, elongation,
    dominant_px_class, class_counts, slope_under,
    smooth_ndsm, obj_mask, center_row, center_col,
):
    """Classify a segment using pixel-class majority + 3D shape analysis."""

    # --- Ground-level features: use pixel classification directly ---
    if h_max < 0.3:
        name = OBJECT_TYPE_NAMES.get(dominant_px_class, "ground")
        return name, ""

    if h_max < 2.0:
        return "low_vegetation", ""

    if h_max < 4.0:
        if elongation > 6 and area < 300:
            return "wall_fence", ""
        return "shrub_bush", ""

    # --- Elevated objects (≥4m) ---

    # Mast/pole: tiny footprint, very tall
    if area < 25 and h_max > 10:
        return "mast_pole", ""

    # Check pixel-class composition
    total_px = max(class_counts.sum(), 1)
    building_frac = class_counts[OBJECT_TYPES["building"]] / total_px

    # --- Building: majority of pixels classified as flat-roof ---
    if building_frac > 0.5 and area >= 20:
        # Extra validation: buildings should be on gentle terrain
        if slope_under < 25:
            if area >= 30:
                return "building", ""
            return "structure", ""

    # Building with shaped (gabled/hipped) roof:
    # Moderate surface slope (10-35°), low roughness relative to trees,
    # and rectangular/compact shape on gentle terrain
    if (slope_under < 20 and h_std < 2.0 and 20 < area < 1000
            and compactness > 0.3 and elongation < 5 and h_max < 18):
        return "building", ""

    # --- Tree crown shape analysis ---
    # Analyse radial height profile from the segment's peak
    crown_shape = _analyse_crown_shape(smooth_ndsm, obj_mask, center_row, center_col)

    p90_ratio = h_p90 / max(h_max, 0.1)
    peak_ratio = h_max / max(h_mean, 0.1)

    if crown_shape == "conical":
        return "tree_coniferous", "conical"
    elif crown_shape == "dome":
        return "tree_broadleaf", "rounded"
    elif crown_shape == "columnar":
        return "tree_broadleaf", "columnar"
    else:
        # Fallback heuristics
        if peak_ratio > 1.4 and p90_ratio < 0.85:
            return "tree_coniferous", "conical"
        elif peak_ratio < 1.25 and p90_ratio > 0.88:
            return "tree_broadleaf", "rounded"
        elif h_max > 20:
            return "tree_coniferous", "irregular"
        return "tree_unclassified", "irregular"


def _analyse_crown_shape(
    smooth_ndsm: np.ndarray,
    obj_mask: np.ndarray,
    center_row: int,
    center_col: int,
) -> str:
    """Analyse the radial height profile of a tree crown.

    Returns: 'conical', 'dome', 'columnar', or 'unknown'.

    Conical (spruce/fir): steep dropoff from peak, >5m drop in 5m radius
    Dome (beech/oak): gentle dropoff, <3m drop in 5m, broad flat top
    Columnar (cypress/poplar): narrow, rapid lateral dropoff
    """
    h, w = smooth_ndsm.shape
    # Find the actual peak within the segment
    seg_vals = smooth_ndsm.copy()
    seg_vals[~obj_mask] = 0
    peak_loc = np.unravel_index(np.argmax(seg_vals), seg_vals.shape)
    pr, pc = peak_loc
    peak_h = smooth_ndsm[pr, pc]

    if peak_h < 4:
        return "unknown"

    # Sample heights at radial distances 2, 4, 6, 8m from peak
    # Average over 8 cardinal + diagonal directions
    radii = [2, 4, 6, 8]
    profile = []
    for radius in radii:
        ring_heights = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            rr = pr + int(dr * radius)
            cc = pc + int(dc * radius)
            if 0 <= rr < h and 0 <= cc < w and obj_mask[rr, cc]:
                ring_heights.append(smooth_ndsm[rr, cc])
        if ring_heights:
            profile.append(float(np.mean(ring_heights)))
        else:
            profile.append(0.0)

    if not profile or profile[0] == 0:
        return "unknown"

    # Dropoff from peak to radius=4m
    drop_4m = peak_h - profile[1] if len(profile) > 1 else 0
    # Dropoff from peak to radius=8m
    drop_8m = peak_h - profile[3] if len(profile) > 3 else 0
    # Ratio of height at radius=4m to peak (flatness of top)
    top_ratio = profile[1] / max(peak_h, 0.1)

    # Conical: steep dropoff
    if drop_4m > 4 and drop_8m > 8:
        return "conical"
    # Dome: gentle, broad crown
    if drop_4m < 2.5 and top_ratio > 0.85:
        return "dome"
    # Columnar: narrow, steep sides but not as peaked
    if drop_4m > 3 and drop_8m > 6 and top_ratio < 0.7:
        return "columnar"
    # Moderate conical
    if drop_4m > 3 and top_ratio < 0.8:
        return "conical"
    # Moderate dome
    if drop_4m < 3 and top_ratio > 0.8:
        return "dome"

    return "unknown"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise_objects(objects: list[DetectedObject]) -> dict:
    if not objects:
        return {"total_objects": 0, "by_type": {}, "by_height_class": {}}

    by_type: dict[str, dict] = {}
    by_height: dict[str, dict] = {}
    tree_heights = []

    for obj in objects:
        t = obj.obj_type
        by_type.setdefault(t, {"count": 0, "total_area_sqm": 0, "height_max": 0, "heights": []})
        by_type[t]["count"] += 1
        by_type[t]["total_area_sqm"] += obj.area_sqm
        by_type[t]["height_max"] = max(by_type[t]["height_max"], obj.height_max)
        by_type[t]["heights"].append(obj.height_max)

        hc = obj.height_class
        by_height.setdefault(hc, {"count": 0, "types": {}})
        by_height[hc]["count"] += 1
        by_height[hc]["types"].setdefault(t, 0)
        by_height[hc]["types"][t] += 1

        if "tree" in t:
            tree_heights.append(obj.height_max)

    for t, info in by_type.items():
        heights = info.pop("heights")
        info["total_area_sqm"] = round(info["total_area_sqm"], 1)
        info["height_mean"] = round(float(np.mean(heights)), 2)
        info["height_std"] = round(float(np.std(heights)), 2) if len(heights) > 1 else 0
        info["height_max"] = round(info["height_max"], 2)

    crown_types: dict[str, int] = {}
    for obj in objects:
        if "tree" in obj.obj_type and obj.crown_shape:
            crown_types.setdefault(obj.crown_shape, 0)
            crown_types[obj.crown_shape] += 1

    result = {
        "total_objects": len(objects),
        "by_type": by_type,
        "by_height_class": by_height,
    }
    if tree_heights:
        result["trees"] = {
            "total_count": len(tree_heights),
            "height_mean": round(float(np.mean(tree_heights)), 2),
            "height_max": round(float(np.max(tree_heights)), 2),
            "crown_type_distribution": crown_types,
        }
    return result


# ---------------------------------------------------------------------------
# Classified raster output
# ---------------------------------------------------------------------------

def create_classified_raster(
    ndsm: np.ndarray,
    dtm: np.ndarray,
    dsm: np.ndarray,
    mask: np.ndarray,
    transform,
    objects: list[DetectedObject],
    output_resolution: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Create 2-band raster: band 1 = type code, band 2 = height."""
    h, w = ndsm.shape

    # Pixel-level classification for base layer
    pixel_classes = _classify_pixels(ndsm, dtm, dsm, mask)
    type_band = pixel_classes.copy()
    height_band = np.where(mask, ndsm, -9999).astype(np.float32)

    # Override with segment-level classification where we have objects
    labels = _segment_objects(ndsm, mask, pixel_classes, 0.3)
    regions = measure.regionprops(labels)

    obj_centroids = np.array([[o.centroid_e, o.centroid_n] for o in objects]) if objects else np.empty((0, 2))
    obj_codes = np.array([o.type_code for o in objects], dtype=np.uint8) if objects else np.empty(0, dtype=np.uint8)

    for reg in regions:
        row, col = reg.centroid
        ce = transform.c + col * transform.a
        cn = transform.f + row * transform.e
        if len(obj_centroids) > 0:
            dists = (obj_centroids[:, 0] - ce)**2 + (obj_centroids[:, 1] - cn)**2
            code = obj_codes[np.argmin(dists)]
        else:
            code = OBJECT_TYPES["unclassified"]
        type_band[labels == reg.label] = code

    out_tf = transform
    if output_resolution != 1.0:
        new_h = max(1, int(h / output_resolution))
        new_w = max(1, int(w / output_resolution))
        height_full = np.where(mask, ndsm, 0).astype(np.float32)
        height_band = ndimage.zoom(height_full, (new_h / h, new_w / w), order=1).astype(np.float32)
        mask_r = ndimage.zoom(mask.astype(np.float32), (new_h / h, new_w / w), order=0) > 0.5
        height_band[~mask_r] = -9999
        type_band = ndimage.zoom(type_band.astype(np.float32), (new_h / h, new_w / w), order=0).astype(np.uint8)
        import rasterio.transform
        out_tf = rasterio.transform.from_origin(transform.c, transform.f, output_resolution, output_resolution)

    return type_band, height_band, out_tf
