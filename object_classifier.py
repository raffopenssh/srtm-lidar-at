"""Object detection and classification from normalised DSM (nDSM = DSM - DTM).

Classifies above-ground objects into categories:
- trees (coniferous, broadleaf), shrubs
- buildings, structures
- masts, poles
- walls, fences
- unclassified elevated objects

The classification uses morphological analysis of the nDSM:
- Height: object height above ground
- Shape: compactness, elongation, crown shape
- Texture: height variance within objects (smooth=building, rough=tree canopy)
- Context: slope of underlying terrain
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage
from skimage import measure, morphology

log = logging.getLogger(__name__)

# Object type codes for raster output
OBJECT_TYPES = {
    "ground": 0,
    "low_vegetation": 1,      # 0.3-2m
    "shrub": 2,               # 2-4m
    "tree_coniferous": 3,     # >4m, pointed crown
    "tree_broadleaf": 4,      # >4m, rounded crown
    "tree_unclassified": 5,   # >4m, unclear crown type
    "building": 6,
    "structure": 7,           # non-building structures
    "mast_pole": 8,           # tall narrow objects
    "wall_fence": 9,          # linear low features
    "unclassified": 10,
}

OBJECT_TYPE_NAMES = {v: k for k, v in OBJECT_TYPES.items()}


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
    compactness: float        # 4π·area/perimeter²  (1=circle, 0=elongated)
    elongation: float         # major_axis/minor_axis
    height_std: float         # internal height variation
    centroid_e: float         # EPSG:3035 easting
    centroid_n: float         # EPSG:3035 northing
    bbox: tuple[float, float, float, float]  # min_e, min_n, max_e, max_n
    crown_shape: str = ""     # for trees: "conical", "rounded", "columnar", "irregular"
    height_class: str = ""    # logarithmic: 0-0.5m, 0.5-1m, 1-2m, 2-4m, 4-8m, 8-15m, 15-25m, 25-40m, 40-60m, 60-80m, >80m


def classify_objects(
    ndsm: np.ndarray,
    dtm: np.ndarray,
    mask: np.ndarray,
    transform,
    min_height: float = 0.3,
    min_area: int = 1,
    max_objects: int = 50000,
) -> list[DetectedObject]:
    """Detect and classify above-ground objects from nDSM.

    Parameters
    ----------
    ndsm : 2D float array - normalised surface heights (DSM - DTM)
    dtm : 2D float array - terrain heights (for slope context)
    mask : 2D bool array - valid area mask
    transform : affine transform for coordinate conversion
    min_height : minimum object height in metres
    min_area : minimum object area in pixels (=sqm at 1m res)
    max_objects : safety limit
    """
    # Work on masked nDSM
    work = np.where(mask, ndsm, 0.0).astype(np.float32)

    # Threshold: pixels above min_height
    elevated = (work >= min_height) & mask

    if not np.any(elevated):
        return []

    # Label connected components
    # First, light morphological opening to reduce noise
    elevated_clean = morphology.binary_opening(elevated, morphology.disk(1))
    labels, n_labels = ndimage.label(elevated_clean)
    log.info(f"Found {n_labels} raw object regions")

    if n_labels > max_objects * 2:
        # Too many tiny objects - increase min area
        log.warning(f"Too many regions ({n_labels}), filtering small objects")

    # Compute region properties
    regions = measure.regionprops(labels, intensity_image=work)

    # Compute terrain slope for context
    from terrain_analysis import compute_slope
    slope = compute_slope(dtm)

    objects = []
    for i, reg in enumerate(regions):
        if reg.area < min_area:
            continue
        if len(objects) >= max_objects:
            log.warning(f"Reached max_objects limit ({max_objects})")
            break

        # Height stats within this object
        obj_pixels = work[labels == reg.label]
        h_max = float(np.nanmax(obj_pixels))
        h_mean = float(np.nanmean(obj_pixels))
        h_std = float(np.nanstd(obj_pixels))
        h_p90 = float(np.nanpercentile(obj_pixels, 90))

        if h_max < min_height:
            continue

        area = float(reg.area)  # sqm at 1m resolution
        perimeter = float(reg.perimeter)

        # Compactness (Polsby-Popper)
        compactness = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

        # Elongation
        if reg.minor_axis_length > 0:
            elongation = reg.major_axis_length / reg.minor_axis_length
        else:
            elongation = float('inf')

        # Convert centroid to map coordinates
        row, col = reg.centroid
        e = transform.c + col * transform.a
        n = transform.f + row * transform.e

        # Bbox in map coords
        min_row, min_col, max_row, max_col = reg.bbox
        bbox_min_e = transform.c + min_col * transform.a
        bbox_max_e = transform.c + max_col * transform.a
        bbox_max_n = transform.f + min_row * transform.e
        bbox_min_n = transform.f + max_row * transform.e

        # Classify
        obj_type, crown_shape = _classify_single_object(
            h_max, h_mean, h_std, area, compactness, elongation,
            obj_pixels, labels == reg.label, slope
        )

        height_class = _height_class(h_max)

        obj = DetectedObject(
            obj_id=i + 1,
            obj_type=obj_type,
            type_code=OBJECT_TYPES.get(obj_type, 10),
            height_max=round(h_max, 2),
            height_mean=round(h_mean, 2),
            height_p90=round(h_p90, 2),
            area_sqm=round(area, 1),
            perimeter_m=round(perimeter, 1),
            compactness=round(compactness, 3),
            elongation=round(elongation, 2),
            height_std=round(h_std, 2),
            centroid_e=round(e, 1),
            centroid_n=round(n, 1),
            bbox=(round(bbox_min_e, 1), round(bbox_min_n, 1),
                  round(bbox_max_e, 1), round(bbox_max_n, 1)),
            crown_shape=crown_shape,
            height_class=height_class,
        )
        objects.append(obj)

    log.info(f"Classified {len(objects)} objects")
    return objects


def _classify_single_object(
    h_max, h_mean, h_std, area, compactness, elongation,
    pixels, obj_mask, slope
):
    """Classify a single detected object. Returns (type_name, crown_shape)."""
    crown_shape = ""

    # Very small footprint, tall = mast/pole
    if area < 20 and h_max > 8 and elongation > 3:
        return "mast_pole", ""

    # Narrow elongated low features = wall/fence
    if h_max < 4 and elongation > 8 and area < 200:
        return "wall_fence", ""

    # Low vegetation (0.3 - 2m)
    if h_max < 2.0:
        return "low_vegetation", ""

    # Shrubs (2-4m, typically irregular)
    if h_max < 4.0:
        if compactness > 0.3:
            return "shrub", ""
        return "low_vegetation", ""

    # For objects > 4m, distinguish trees from buildings
    # Buildings: flat top (low h_std relative to height), rectangular, high compactness
    # Trees: variable canopy (high h_std), more circular

    # Coefficient of variation of height
    cv = h_std / max(h_mean, 0.1)

    # Height profile analysis for crown shape
    # Ratio of max height to mean height indicates crown shape
    peak_ratio = h_max / max(h_mean, 0.1)

    is_building = False

    # Building indicators
    if area > 15:  # Buildings are usually > 15 sqm
        if cv < 0.15:  # Very flat top
            is_building = True
        elif cv < 0.25 and compactness > 0.3 and elongation < 5:
            is_building = True
        elif cv < 0.3 and compactness > 0.5 and area > 50:
            is_building = True

    if is_building:
        if area > 200:
            return "building", ""
        elif area > 30:
            return "building", ""
        else:
            return "structure", ""

    # Trees - classify crown shape
    if peak_ratio > 1.6 and compactness > 0.2:
        crown_shape = "conical"  # Coniferous - spire shape
        return "tree_coniferous", crown_shape
    elif peak_ratio < 1.3 and compactness > 0.3:
        crown_shape = "rounded"  # Broadleaf - dome shape
        return "tree_broadleaf", crown_shape
    elif elongation < 2.0 and compactness > 0.4:
        crown_shape = "columnar"
        return "tree_broadleaf", crown_shape
    else:
        crown_shape = "irregular"
        # Try to distinguish by height variance
        if cv > 0.3 and peak_ratio > 1.4:
            return "tree_coniferous", "conical"
        elif cv < 0.25:
            return "tree_broadleaf", "rounded"
        return "tree_unclassified", crown_shape


# Logarithmic height class boundaries (metres) up to 80m
_HEIGHT_BREAKS = [0.5, 1, 2, 4, 8, 15, 25, 40, 60, 80]


def _height_class(h: float) -> str:
    """Assign a logarithmic height class label."""
    prev = 0
    for brk in _HEIGHT_BREAKS:
        if h < brk:
            return f"{prev}-{brk}m"
        prev = brk
    return f">{_HEIGHT_BREAKS[-1]}m"


def summarise_objects(objects: list[DetectedObject]) -> dict:
    """Generate a summary report of detected objects."""
    if not objects:
        return {"total_objects": 0, "by_type": {}, "by_height_class": {}}

    by_type = {}
    by_height = {}
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

    # Clean up - compute stats per type
    for t, info in by_type.items():
        heights = info.pop("heights")
        info["total_area_sqm"] = round(info["total_area_sqm"], 1)
        info["height_mean"] = round(float(np.mean(heights)), 2)
        info["height_std"] = round(float(np.std(heights)), 2) if len(heights) > 1 else 0
        info["height_max"] = round(info["height_max"], 2)

    # Crown type distribution for trees
    crown_types = {}
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


def create_classified_raster(
    ndsm: np.ndarray,
    mask: np.ndarray,
    transform,
    objects: list[DetectedObject],
    output_resolution: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Create a 2-band raster: band 1 = object type code, band 2 = height.

    If output_resolution != 1.0, resamples the output.
    Returns (type_band, height_band, transform).
    """
    h, w = ndsm.shape

    if output_resolution == 1.0:
        type_band = np.zeros((h, w), dtype=np.uint8)
        height_band = np.where(mask, ndsm, -9999).astype(np.float32)

        # Label connected regions and assign types
        elevated = (ndsm >= 0.3) & mask
        elevated_clean = morphology.binary_opening(elevated, morphology.disk(1))
        labels, _ = ndimage.label(elevated_clean)

        # Build lookup from centroid proximity to object
        # Match labels to classified objects
        regions = measure.regionprops(labels)
        label_to_type = {}
        for reg in regions:
            row, col = reg.centroid
            e = transform.c + col * transform.a
            n = transform.f + row * transform.e
            # Find closest classified object
            best_dist = float('inf')
            best_type = OBJECT_TYPES["unclassified"]
            for obj in objects:
                d = (obj.centroid_e - e)**2 + (obj.centroid_n - n)**2
                if d < best_dist:
                    best_dist = d
                    best_type = obj.type_code
            label_to_type[reg.label] = best_type

        for label_val, type_code in label_to_type.items():
            type_band[labels == label_val] = type_code

        out_transform = transform
    else:
        # Resample
        from scipy.ndimage import zoom
        scale = 1.0 / output_resolution
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

        # Height band - bilinear interpolation
        height_full = np.where(mask, ndsm, 0).astype(np.float32)
        height_band = zoom(height_full, (new_h / h, new_w / w), order=1).astype(np.float32)
        mask_resamp = zoom(mask.astype(np.float32), (new_h / h, new_w / w), order=0) > 0.5
        height_band[~mask_resamp] = -9999

        # Type band - nearest neighbour
        type_full = np.zeros((h, w), dtype=np.uint8)
        elevated = (ndsm >= 0.3) & mask
        elevated_clean = morphology.binary_opening(elevated, morphology.disk(1))
        labels, _ = ndimage.label(elevated_clean)
        regions = measure.regionprops(labels)
        label_to_type = {}
        for reg in regions:
            row, col = reg.centroid
            e = transform.c + col * transform.a
            n_coord = transform.f + row * transform.e
            best_dist = float('inf')
            best_type = OBJECT_TYPES["unclassified"]
            for obj in objects:
                d = (obj.centroid_e - e)**2 + (obj.centroid_n - n_coord)**2
                if d < best_dist:
                    best_dist = d
                    best_type = obj.type_code
            label_to_type[reg.label] = best_type
        for label_val, type_code in label_to_type.items():
            type_full[labels == label_val] = type_code

        type_band = zoom(type_full.astype(np.float32), (new_h / h, new_w / w), order=0).astype(np.uint8)

        # Update transform for new resolution
        import rasterio.transform
        out_transform = rasterio.transform.from_origin(
            transform.c, transform.f,
            output_resolution, output_resolution
        )

    return type_band, height_band, out_transform
