"""Landscape transformation classifier — detects how humans shaped terrain.

Philosophy: instead of 27 fragile object types, ask ONE question:
  "Is this man-made or natural, and what did humans do here?"

Primary signals (in order of reliability):
  1. DTM time series (3 dates × 1m) — where machinery reshaped terrain
  2. DSM surface morphology at 1m — linear features, engineered surfaces
  3. Copernicus Sentinel-2 growing-season NDVI composite (10m, reliable)
  4. nDSM height + stability across dates — buildings vs trees
  5. Cadastre footprints — ground truth for calibration only

Key innovations:
  - Hessian eigenvalue analysis for ridge/valley (embankment/ditch) detection
  - Multi-scale roughness ratio for engineered vs natural surface detection
  - DTM differencing with spatial coherence for machinery trace detection
  - Growing-season NDVI instead of single-date (eliminates seasonal bias)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter, label as ndi_label
from skimage import measure, morphology, segmentation

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type codes — 10 landscape types focused on human transformation
# ---------------------------------------------------------------------------

LANDSCAPE_TYPES = {
    "engineered_surface": 1,   # Roads, parking, paved — flat, smooth, linear/large
    "engineered_slope": 2,     # Embankments, cuttings, retaining walls
    "excavation": 3,           # Quarries, pits, construction sites — terrain removed
    "fill": 4,                 # Landfill, raised platforms, levees — terrain added
    "building": 5,             # Structures: height + stability + spectral
    "infrastructure": 6,       # Bridges, masts, fences, walls, other elevated
    "tree_canopy": 7,          # All trees (no conifer/broadleaf split)
    "vegetation": 8,           # Low veg, shrubs, meadows, crops
    "bare_natural": 9,         # Rock, cliff, bare soil, natural water
    "recent_disturbance": 10,  # Terrain modified between LIDAR dates
}

LANDSCAPE_TYPE_NAMES = {v: k for k, v in LANDSCAPE_TYPES.items()}

# Backward-compat alias (app.py uses OBJECT_TYPES / OBJECT_TYPE_NAMES)
OBJECT_TYPES = LANDSCAPE_TYPES
OBJECT_TYPE_NAMES = LANDSCAPE_TYPE_NAMES

# Height class boundaries
_HEIGHT_BREAKS = [0.5, 1, 2, 4, 8, 15, 25, 40, 60, 80]


def _height_class(h: float) -> str:
    prev = 0
    for brk in _HEIGHT_BREAKS:
        if h < brk:
            return f"{prev}-{brk}m"
        prev = brk
    return f">{_HEIGHT_BREAKS[-1]}m"


# ---------------------------------------------------------------------------
# DetectedObject — same interface as old classifier for API compatibility
# ---------------------------------------------------------------------------

@dataclass
class DetectedObject:
    """A detected landscape feature."""
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
    centroid_e: float
    centroid_n: float
    bbox: tuple[float, float, float, float]
    crown_shape: str = ""
    height_class: str = ""
    # Spectral
    ndvi_mean: float = 0.0
    ndvi_max: float = 0.0
    brightness_mean: float = 0.0
    spectral_class: str = ""
    # Temporal
    temporal_std: float = 0.0
    temporal_stable: bool = False
    temporal_signal: str = ""
    # New fields
    confidence: float = 0.5
    is_manmade: bool = False
    linear_feature: bool = False
    machinery_trace: bool = False


# ===================================================================
# CORE ANALYSIS: Multi-scale surface morphology
# ===================================================================

def _local_std(arr: np.ndarray, size: int = 3) -> np.ndarray:
    """Fast local standard deviation."""
    a = arr.astype(np.float64)
    mean = uniform_filter(a, size=size, mode="nearest")
    mean_sq = uniform_filter(a ** 2, size=size, mode="nearest")
    var = np.clip(mean_sq - mean ** 2, 0, None)
    return np.sqrt(var).astype(np.float32)


def _slope(arr: np.ndarray) -> np.ndarray:
    """Slope in degrees."""
    dy, dx = np.gradient(arr, 1.0)
    return np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2))).astype(np.float32)


# ===================================================================
# 1. HESSIAN-BASED LINEAR FEATURE DETECTION
# ===================================================================

def detect_linear_features(
    dtm: np.ndarray,
    dsm: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float = 2.0,
) -> dict:
    """Detect linear man-made features using Hessian eigenvalue analysis.

    Roads, ditches, embankments, walls are LINEAR — they have high curvature
    in one direction and low in the other.  The Hessian matrix eigenvalues
    capture this perfectly:
      - Ridge (embankment): lambda1 >> 0, |lambda2| small
      - Valley (ditch/road cut): lambda1 << 0, |lambda2| small
      - Blob (pit/mound): both eigenvalues large

    Returns dict with 2D arrays:
      ridge_strength, valley_strength, linearity, orientation_deg,
      engineered_mask (boolean)
    """
    h, w = dtm.shape

    # Smooth to suppress 1-pixel noise while preserving 3+ pixel features
    dtm_s = gaussian_filter(dtm.astype(np.float64), sigma=sigma)

    # Second derivatives (Hessian components)
    Hy = np.gradient(dtm_s, axis=0)
    Hx = np.gradient(dtm_s, axis=1)
    Hyy = np.gradient(Hy, axis=0)
    Hxx = np.gradient(Hx, axis=1)
    Hxy = np.gradient(Hx, axis=0)

    # Eigenvalues of 2x2 Hessian at each pixel
    trace = Hxx + Hyy
    det = Hxx * Hyy - Hxy ** 2
    discriminant = np.sqrt(np.clip(trace ** 2 - 4 * det, 0, None))
    lambda1 = (trace + discriminant) / 2  # larger eigenvalue
    lambda2 = (trace - discriminant) / 2  # smaller eigenvalue

    # Ridge strength: large positive lambda1, small |lambda2|
    # (convex ridge like embankment or road crown)
    abs1 = np.abs(lambda1) + 1e-10
    abs2 = np.abs(lambda2) + 1e-10
    ridge = np.where(
        lambda1 > 0.001,
        lambda1 * (1 - abs2 / abs1),
        0.0,
    ).astype(np.float32)

    # Valley strength: large negative lambda2, small |lambda1|
    # (concave valley like ditch or road cut)
    valley = np.where(
        lambda2 < -0.001,
        np.abs(lambda2) * (1 - abs1 / (abs2 + abs1)),
        0.0,
    ).astype(np.float32)

    # Linearity: how much the curvature is in one direction only
    # 1.0 = perfectly linear, 0.0 = isotropic (blob)
    linearity = np.where(
        abs1 + abs2 > 0.002,
        np.abs(abs1 - abs2) / (abs1 + abs2),
        0.0,
    ).astype(np.float32)

    # Orientation of the linear feature (angle of principal curvature)
    orientation = np.degrees(0.5 * np.arctan2(2 * Hxy, Hxx - Hyy)).astype(np.float32)

    # Also detect from DSM (walls, fences, hedges on top of terrain)
    ndsm = np.clip(dsm - dtm, 0, None)
    dsm_s = gaussian_filter(dsm.astype(np.float64), sigma=1.5)
    Dx = np.gradient(dsm_s, axis=1)
    Dy = np.gradient(dsm_s, axis=0)
    Dxx = np.gradient(Dx, axis=1)
    Dyy = np.gradient(Dy, axis=0)
    Dxy = np.gradient(Dx, axis=0)
    d_trace = Dxx + Dyy
    d_det = Dxx * Dyy - Dxy ** 2
    d_disc = np.sqrt(np.clip(d_trace ** 2 - 4 * d_det, 0, None))
    d_l1 = (d_trace + d_disc) / 2
    d_l2 = (d_trace - d_disc) / 2

    # Combine DTM + DSM linear signals
    dsm_ridge = np.where(d_l1 > 0.002, d_l1, 0.0).astype(np.float32)
    combined_ridge = np.maximum(ridge, dsm_ridge * 0.5)

    # Engineered linear feature mask:
    # High linearity AND (ridge OR valley) AND within valid area
    eng_linear = mask & (linearity > 0.6) & (
        (ridge > np.nanpercentile(ridge[mask], 90)) |
        (valley > np.nanpercentile(valley[mask], 90))
    )

    # Clean: remove isolated pixels, keep connected features
    eng_linear = morphology.remove_small_objects(eng_linear, min_size=10)
    # Thin to skeleton for true linear detection
    if eng_linear.any():
        skeleton = morphology.skeletonize(eng_linear)
    else:
        skeleton = np.zeros_like(eng_linear)

    return {
        "ridge_strength": ridge,
        "valley_strength": valley,
        "linearity": linearity,
        "orientation_deg": orientation,
        "dsm_ridge": dsm_ridge,
        "combined_ridge": combined_ridge,
        "engineered_linear_mask": eng_linear,
        "skeleton": skeleton,
    }


# ===================================================================
# 2. ENGINEERED SURFACE DETECTION (multi-scale roughness)
# ===================================================================

def detect_engineered_surfaces(
    dtm: np.ndarray,
    dsm: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Detect unnaturally smooth/flat terrain: roads, parking, foundations.

    Key insight: natural terrain has fractal roughness at all scales.
    Engineered surfaces are smooth at small scale but may have slope.

    Returns dict with 2D arrays:
      roughness_3, roughness_5, roughness_15,
      smoothness_ratio, engineered_surface_mask
    """
    # Multi-scale roughness (local std of DTM)
    rough_3 = _local_std(dtm, 3)    # 3m window — fine texture
    rough_5 = _local_std(dtm, 5)    # 5m window — medium texture
    rough_9 = _local_std(dtm, 9)    # 9m window — local terrain
    rough_15 = _local_std(dtm, 15)  # 15m window — broader terrain

    # Smoothness ratio: small-scale / large-scale roughness
    # Natural terrain: ratio ~0.3-0.8 (fractal)
    # Engineered: ratio < 0.15 (smooth at small scale, may have large-scale slope)
    smoothness_ratio = rough_3 / (rough_15 + 0.005)

    # Also check DSM roughness for ground-level areas
    ndsm = np.clip(dsm - dtm, 0, None)
    ground_level = ndsm < 0.3

    # Slope uniformity: engineered surfaces have consistent slope
    dtm_slope = _slope(dtm)
    slope_std = _local_std(dtm_slope, 7)  # std of slope = how variable the slope is

    # Engineered surface criteria (calibrated: meadows have rough_3 ~0.05-0.10):
    # Need asphalt-level smooth AND consistent slope grade
    engineered = mask & ground_level & (
        (rough_3 < 0.025)  # only truly engineered surfaces
    ) & (slope_std < 1.0)

    # Remove tiny fragments — roads/parking are at least 25m²
    engineered = morphology.remove_small_objects(engineered, min_size=25)

    return {
        "roughness_3": rough_3,
        "roughness_5": rough_5,
        "roughness_9": rough_9,
        "roughness_15": rough_15,
        "smoothness_ratio": smoothness_ratio,
        "slope_uniformity": slope_std,
        "dtm_slope": dtm_slope,
        "engineered_surface_mask": engineered,
    }


# ===================================================================
# 3. DTM TIME SERIES: MACHINERY TRACE DETECTION
# ===================================================================

def detect_machinery_traces(
    dtm_dates: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    min_dtm_change: float = 0.15,
) -> dict:
    """Detect where heavy machinery modified terrain between LIDAR dates.

    Excavators/bulldozers/graders leave distinctive DTM signatures:
    - Spatially coherent height changes (not random noise)
    - Natural→smooth roughness transition
    - Often paired cut/fill (excavate here, dump there)
    - Linear patterns (roads, ditches, embankments)

    Parameters
    ----------
    dtm_dates : {date_str: 2D float32}
        DTM arrays for each date, already aligned to same grid.
    mask : bool array
        Valid pixel mask.
    min_dtm_change : float
        Minimum DTM change to consider significant (metres).

    Returns
    -------
    dict with:
      disturbance_intensity: float32 array [0, 1]
      cut_mask, fill_mask: boolean arrays
      roughness_change: float32 array (negative = smoothed = machinery)
      total_disturbance_mask: boolean
      paired_cut_fill: boolean (cut and fill adjacent)
      per_pair: list of {date_a, date_b, dtm_change, ...}
    """
    dates = sorted(dtm_dates.keys())
    h, w = mask.shape

    if len(dates) < 2:
        return _empty_disturbance(h, w)

    # Accumulate disturbance signal across all consecutive pairs
    total_cut = np.zeros((h, w), dtype=np.float32)
    total_fill = np.zeros((h, w), dtype=np.float32)
    total_roughness_change = np.zeros((h, w), dtype=np.float32)
    any_disturbed = np.zeros((h, w), dtype=bool)
    per_pair = []

    for i in range(len(dates) - 1):
        da, db = dates[i], dates[i + 1]
        dtm_a = dtm_dates[da]
        dtm_b = dtm_dates[db]

        # Align shapes
        mh = min(dtm_a.shape[0], dtm_b.shape[0], h)
        mw = min(dtm_a.shape[1], dtm_b.shape[1], w)
        dtm_a = dtm_a[:mh, :mw]
        dtm_b = dtm_b[:mh, :mw]
        m = mask[:mh, :mw]

        dtm_diff = np.where(m, dtm_b - dtm_a, 0).astype(np.float32)

        # Roughness before and after
        rough_a = _local_std(dtm_a, 5)
        rough_b = _local_std(dtm_b, 5)
        rough_change = np.where(m, rough_b - rough_a, 0).astype(np.float32)

        # Significant changes (above noise floor)
        sig_change = m & (np.abs(dtm_diff) > min_dtm_change)

        # Spatial coherence filter: machinery makes spatially connected changes
        # Remove isolated changed pixels (noise), keep coherent regions
        struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
        sig_coherent = ndimage.binary_opening(sig_change, structure=struct, iterations=1)
        sig_coherent = ndimage.binary_closing(sig_coherent, structure=struct, iterations=1)
        sig_coherent = morphology.remove_small_objects(sig_coherent, min_size=8)

        # Cut (terrain lowered) and fill (terrain raised)
        cut = sig_coherent & (dtm_diff < -min_dtm_change)
        fill = sig_coherent & (dtm_diff > min_dtm_change)

        # Roughness decrease = terrain was smoothed = likely machinery
        smoothed = m & (rough_change < -0.05) & sig_coherent

        # Machinery confidence: combines height change + smoothing
        # Stronger signal = higher confidence
        change_magnitude = np.abs(dtm_diff)
        smoothing_signal = np.clip(-rough_change / 0.3, 0, 1)  # 0-1 scale
        height_signal = np.clip(change_magnitude / 2.0, 0, 1)   # 0-1 scale
        pair_intensity = np.where(
            sig_coherent,
            np.maximum(height_signal, smoothing_signal * 0.8),
            0.0,
        ).astype(np.float32)

        total_cut[:mh, :mw] += np.where(cut, np.abs(dtm_diff), 0)
        total_fill[:mh, :mw] += np.where(fill, dtm_diff, 0)
        total_roughness_change[:mh, :mw] += rough_change
        any_disturbed[:mh, :mw] |= sig_coherent

        # Look for paired cut-fill (excavate here → dump there)
        # Dilate cut and fill masks and check overlap
        cut_dilated = ndimage.binary_dilation(cut, structure=struct, iterations=5)
        fill_dilated = ndimage.binary_dilation(fill, structure=struct, iterations=5)
        paired = cut_dilated & fill_dilated & m

        per_pair.append({
            "date_a": da,
            "date_b": db,
            "dtm_change_mean": float(np.nanmean(dtm_diff[m])),
            "dtm_change_max": float(np.nanmax(np.abs(dtm_diff[m]))) if m.any() else 0,
            "pixels_cut": int(cut.sum()),
            "pixels_fill": int(fill.sum()),
            "pixels_smoothed": int(smoothed.sum()),
            "pixels_paired_cut_fill": int(paired.sum()),
            "area_disturbed_sqm": int(sig_coherent.sum()),
        })

    # Overall disturbance intensity (0-1)
    max_cut = np.nanmax(total_cut[mask]) if mask.any() and total_cut[mask].any() else 1.0
    max_fill = np.nanmax(total_fill[mask]) if mask.any() and total_fill[mask].any() else 1.0
    intensity = np.clip(
        (total_cut / max(max_cut, 0.1) + total_fill / max(max_fill, 0.1)) / 2,
        0, 1,
    ).astype(np.float32)

    return {
        "disturbance_intensity": intensity,
        "cut_mask": total_cut > min_dtm_change,
        "fill_mask": total_fill > min_dtm_change,
        "roughness_change": total_roughness_change,
        "total_disturbance_mask": any_disturbed,
        "per_pair": per_pair,
    }


def _empty_disturbance(h: int, w: int) -> dict:
    z = np.zeros((h, w), dtype=np.float32)
    return {
        "disturbance_intensity": z,
        "cut_mask": np.zeros((h, w), dtype=bool),
        "fill_mask": np.zeros((h, w), dtype=bool),
        "roughness_change": z,
        "total_disturbance_mask": np.zeros((h, w), dtype=bool),
        "per_pair": [],
    }


# ===================================================================
# 4. DSM TIME SERIES: SURFACE CHANGE DETECTION
# ===================================================================

def detect_dsm_changes(
    dsm_dates: dict[str, np.ndarray],
    dtm_dates: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict:
    """Detect surface changes from DSM time series.

    Complements DTM analysis: DSM captures above-ground changes
    (new buildings, tree growth/felling, new infrastructure).

    Returns dict with stability metrics and change masks.
    """
    dates = sorted(dsm_dates.keys())
    h, w = mask.shape

    if len(dates) < 2:
        return {
            "ndsm_temporal_std": np.zeros((h, w), dtype=np.float32),
            "ndsm_temporal_range": np.zeros((h, w), dtype=np.float32),
            "stable_elevated": np.zeros((h, w), dtype=bool),
            "growing": np.zeros((h, w), dtype=bool),
            "shrinking": np.zeros((h, w), dtype=bool),
        }

    # Compute nDSM for each date
    ndsm_stack = []
    for d in dates:
        dsm_d = dsm_dates[d]
        dtm_d = dtm_dates[d]
        mh = min(dsm_d.shape[0], dtm_d.shape[0], h)
        mw = min(dsm_d.shape[1], dtm_d.shape[1], w)
        ndsm_d = np.clip(dsm_d[:mh, :mw] - dtm_d[:mh, :mw], 0, None)
        # Pad to full size if needed
        full = np.zeros((h, w), dtype=np.float32)
        full[:mh, :mw] = ndsm_d
        ndsm_stack.append(full)

    stack = np.stack(ndsm_stack, axis=0)  # (N, H, W)

    with np.errstate(all="ignore"):
        temporal_std = np.where(mask, np.nanstd(stack, axis=0), 0).astype(np.float32)
        temporal_range = np.where(
            mask,
            np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0),
            0,
        ).astype(np.float32)

    # Stable elevated objects (buildings): height > 2.5m AND std < 0.5m
    latest_ndsm = ndsm_stack[-1]
    stable_elevated = mask & (latest_ndsm > 2.5) & (temporal_std < 0.5)

    # Growing: nDSM increased monotonically (trees)
    if len(ndsm_stack) >= 3:
        growing = mask & (ndsm_stack[-1] > ndsm_stack[0] + 0.5) & (
            ndsm_stack[1] >= ndsm_stack[0] - 0.2
        )
        shrinking = mask & (ndsm_stack[-1] < ndsm_stack[0] - 0.5) & (
            ndsm_stack[1] <= ndsm_stack[0] + 0.2
        )
    else:
        growing = mask & (ndsm_stack[-1] > ndsm_stack[0] + 0.5)
        shrinking = mask & (ndsm_stack[-1] < ndsm_stack[0] - 0.5)

    return {
        "ndsm_temporal_std": temporal_std,
        "ndsm_temporal_range": temporal_range,
        "stable_elevated": stable_elevated,
        "growing": growing,
        "shrinking": shrinking,
        "ndsm_latest": latest_ndsm,
        "ndsm_stack": stack,
    }


# ===================================================================
# 5. COPERNICUS DATA INTEGRATION
# ===================================================================

def _resample_copernicus_to_grid(
    cop_data: np.ndarray,
    cop_transform,
    cop_crs,
    target_transform,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Reproject + resample Copernicus raster (10m, EPSG:4326) to our 1m EPSG:3035 grid."""
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS

    dst_crs = CRS.from_epsg(3035)
    dst = np.full(target_shape, np.nan, dtype=np.float32)

    reproject(
        source=cop_data.astype(np.float32),
        destination=dst,
        src_transform=cop_transform,
        src_crs=cop_crs if cop_crs else CRS.from_epsg(4326),
        dst_transform=target_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def integrate_copernicus(
    copernicus: dict | None,
    target_transform,
    target_shape: tuple[int, int],
) -> dict:
    """Resample all Copernicus layers to the LIDAR grid.

    Returns dict with resampled arrays or None entries if unavailable.
    """
    result = {
        "ndvi": None,      # Growing-season composite, reliable
        "landcover": None,  # ESA WorldCover 10m classes
        "sar_vv": None,     # Sentinel-1 VV backscatter
        "sar_vh": None,     # Sentinel-1 VH backscatter
    }

    if copernicus is None:
        return result

    # NDVI composite
    if "ndvi" in copernicus and copernicus["ndvi"] is not None:
        try:
            result["ndvi"] = _resample_copernicus_to_grid(
                copernicus["ndvi"],
                copernicus.get("transform"),
                copernicus.get("crs"),
                target_transform,
                target_shape,
            )
            log.info("Copernicus NDVI resampled to %s", target_shape)
        except Exception as e:
            log.warning("Failed to resample Copernicus NDVI: %s", e)

    # Land cover
    if "landcover" in copernicus and copernicus["landcover"] is not None:
        try:
            lc_data = copernicus["landcover"]
            lc_info = lc_data if isinstance(lc_data, dict) else {"map": lc_data}
            from rasterio.warp import reproject, Resampling
            from rasterio.crs import CRS
            dst = np.zeros(target_shape, dtype=np.uint8)
            reproject(
                source=lc_info["map"].astype(np.uint8),
                destination=dst,
                src_transform=lc_info.get("transform", copernicus.get("transform")),
                src_crs=lc_info.get("crs", copernicus.get("crs")) or CRS.from_epsg(4326),
                dst_transform=target_transform,
                dst_crs=CRS.from_epsg(3035),
                resampling=Resampling.nearest,
            )
            result["landcover"] = dst
            log.info("Copernicus land cover resampled to %s", target_shape)
        except Exception as e:
            log.warning("Failed to resample land cover: %s", e)

    # SAR
    for band in ("sar_vv", "sar_vh"):
        key = band.replace("sar_", "")
        if key in copernicus and copernicus[key] is not None:
            try:
                result[band] = _resample_copernicus_to_grid(
                    copernicus[key],
                    copernicus.get("transform"),
                    copernicus.get("crs"),
                    target_transform,
                    target_shape,
                )
            except Exception as e:
                log.warning("Failed to resample SAR %s: %s", band, e)

    return result


# ===================================================================
# 6. PIXEL-LEVEL CLASSIFICATION
# ===================================================================

def _classify_pixels(
    dtm: np.ndarray,
    dsm: np.ndarray,
    mask: np.ndarray,
    *,
    linear: dict | None = None,
    surfaces: dict | None = None,
    disturbance: dict | None = None,
    dsm_changes: dict | None = None,
    cop: dict | None = None,
    building_truth: np.ndarray | None = None,
    spectral: dict | None = None,
) -> np.ndarray:
    """Assign each pixel a landscape type code.

    Combines all analysis layers into a single classification.
    Priority order (highest first):
      1. Recent disturbance (DTM changed between dates)
      2. Building (stable elevated + spectral confirmation)
      3. Engineered surface (roads, parking)
      4. Engineered slope (embankments, cuttings)
      5. Excavation / fill (from DTM time series)
      6. Infrastructure (elevated, non-building, non-tree)
      7. Tree canopy (elevated, growing/unstable, high NDVI)
      8. Low vegetation
      9. Bare natural
    """
    h, w = dtm.shape
    ndsm = np.clip(dsm - dtm, 0, None)
    px = np.zeros((h, w), dtype=np.uint8)  # Start as 0 = unclassified

    # --- Multi-scale surface descriptors (calibrated against BEV cadastre) ---
    dsm_std3 = _local_std(dsm, 3)          # surface roughness
    ndsm_clean = np.nan_to_num(ndsm, nan=0.0)  # clean for windowed ops
    ndsm_std5 = _local_std(ndsm_clean, 5)  # height uniformity within object
    dtm_slope = _slope(dtm)                 # terrain slope
    dtm_std5 = _local_std(dtm, 5)           # terrain roughness

    # ----- Layer 9: Bare natural (default for ground) -----
    ground = mask & (ndsm < 0.3)
    steep = dtm_slope > 35
    rough_ground = dtm_std5 > 0.4
    px[mask & ground & (steep | rough_ground)] = LANDSCAPE_TYPES["bare_natural"]

    # ----- Layer 8: Vegetation (ground level, normal roughness) -----
    px[mask & ground & ~steep & ~rough_ground] = LANDSCAPE_TYPES["vegetation"]

    # Low vegetation (0.3 - 2m)
    low_veg = mask & (ndsm >= 0.3) & (ndsm < 2.0)
    px[low_veg] = LANDSCAPE_TYPES["vegetation"]

    # Medium vegetation / low structures (2-4m) — default to vegetation
    mid_veg = mask & (ndsm >= 2.0) & (ndsm < 4.0)
    px[mid_veg] = LANDSCAPE_TYPES["vegetation"]

    # ----- Layer 7: Tree canopy (> 4m, default for tall objects) -----
    tall = mask & (ndsm >= 4.0)
    px[tall] = LANDSCAPE_TYPES["tree_canopy"]  # Default: tall = tree

    # ----- Layer 5: Building detection -----
    # Calibrated against BEV cadastre (KG 63332 Köflach, 696 footprints):
    #   dsm_std3:  buildings p50=0.70, trees p50=1.71
    #   ndsm_std5: buildings p50=1.28, trees p50=2.82
    #   dtm_slope: buildings p50=2.7°, trees p50=11.8°
    #
    # Strategy: multi-criteria scoring, not single threshold
    elevated = mask & (ndsm > 2.0)  # lowered from 2.5 to catch garages/sheds

    # Score each pixel for building-likeness (0 = tree, higher = building)
    bld_score = np.zeros((h, w), dtype=np.float32)

    # Surface smoothness: buildings have smoother DSM (gabled roofs < 1.2)
    bld_score += np.where(elevated & (dsm_std3 < 0.5), 2.0, 0)
    bld_score += np.where(elevated & (dsm_std3 < 1.0), 1.0, 0)
    bld_score += np.where(elevated & (dsm_std3 < 1.5), 0.5, 0)

    # Height uniformity: buildings have consistent height
    bld_score += np.where(elevated & (ndsm_std5 < 1.0), 1.5, 0)
    bld_score += np.where(elevated & (ndsm_std5 < 2.0), 0.8, 0)
    bld_score += np.where(elevated & (ndsm_std5 < 3.0), 0.3, 0)

    # Flat terrain: buildings are on flat ground
    bld_score += np.where(elevated & (dtm_slope < 5), 1.5, 0)
    bld_score += np.where(elevated & (dtm_slope < 10), 0.8, 0)
    bld_score += np.where(elevated & (dtm_slope < 20), 0.3, 0)

    # Height penalty: very tall (>25m) more likely tree or tower
    bld_score -= np.where(elevated & (ndsm > 25), 1.0, 0)

    # ----- BEV orthophoto NDVI from spectral dict (old-style single-date) -----
    bev_ndvi = None
    bev_brightness = None
    if spectral is not None:
        if "ndvi" in spectral and spectral["ndvi"] is not None:
            arr = np.asarray(spectral["ndvi"], dtype=np.float32)
            if arr.shape == (h, w):
                bev_ndvi = arr
        if "brightness" in spectral and spectral["brightness"] is not None:
            arr = np.asarray(spectral["brightness"], dtype=np.float32)
            if arr.shape == (h, w):
                bev_brightness = arr

    # NDVI refinement for building score
    if bev_ndvi is not None:
        valid_ndvi = np.isfinite(bev_ndvi)
        # Low NDVI on elevated pixels = strong building signal
        bld_score += np.where(elevated & valid_ndvi & (bev_ndvi < 0.15), 2.0, 0)
        bld_score += np.where(elevated & valid_ndvi & (bev_ndvi < 0.25), 1.0, 0)
        # High NDVI = vegetation, penalize building score
        bld_score -= np.where(elevated & valid_ndvi & (bev_ndvi > 0.30), 2.0, 0)
        bld_score -= np.where(elevated & valid_ndvi & (bev_ndvi > 0.20), 1.0, 0)
        if bev_brightness is not None:
            valid_bri = np.isfinite(bev_brightness)
            # Bright + low NDVI = very likely building
            bld_score += np.where(
                elevated & valid_ndvi & valid_bri &
                (bev_ndvi < 0.20) & (bev_brightness > 90), 1.5, 0
            )

    # Classify as building if score >= 5.0 (with NDVI) or 6.0 (without)
    score_thresh = 5.0 if bev_ndvi is not None else 6.0
    building_mask = elevated & (bld_score >= score_thresh)
    px[building_mask] = LANDSCAPE_TYPES["building"]

    # Low structures (2-4m): buildings if very flat and on flat ground
    low_bld = mask & (ndsm >= 2.0) & (ndsm < 4.0) & (dsm_std3 < 0.4) & \
              (ndsm_std5 < 1.0) & (dtm_slope < 8)
    px[low_bld] = LANDSCAPE_TYPES["building"]

    # ----- Temporal stability (strongest signal for building vs tree) -----
    if dsm_changes is not None and dsm_changes.get("stable_elevated") is not None:
        stable = dsm_changes["stable_elevated"]
        # Temporally stable + elevated = building (very high confidence)
        px[stable & elevated] = LANDSCAPE_TYPES["building"]
        # Growing = definitely tree
        if dsm_changes.get("growing") is not None:
            px[dsm_changes["growing"] & tall] = LANDSCAPE_TYPES["tree_canopy"]
        # Shrinking canopy = tree (felling/pruning)
        if dsm_changes.get("shrinking") is not None:
            px[dsm_changes["shrinking"] & tall] = LANDSCAPE_TYPES["tree_canopy"]

    # ----- Copernicus NDVI refinement (growing-season = reliable) -----
    if cop is not None and cop.get("ndvi") is not None:
        ndvi = cop["ndvi"]
        valid_ndvi = np.isfinite(ndvi)

        # Growing-season NDVI > 0.3 = definitely vegetation
        veg_from_ndvi = valid_ndvi & (ndvi > 0.3) & tall
        px[veg_from_ndvi] = LANDSCAPE_TYPES["tree_canopy"]

        # NDVI < 0.15 + elevated = building (reliable with growing-season composite)
        bld_from_ndvi = valid_ndvi & (ndvi < 0.15) & elevated & (bld_score >= 2.5)
        px[bld_from_ndvi] = LANDSCAPE_TYPES["building"]

    # ----- Copernicus land cover as soft prior -----
    if cop is not None and cop.get("landcover") is not None:
        lc_map = cop["landcover"]
        esa_built = (lc_map == 50)
        # Boost building where ESA says built-up AND geometry supports it
        px[esa_built & elevated & (bld_score >= 2.5)] = LANDSCAPE_TYPES["building"]

    # ----- Layer 4/3: Excavation / Fill (from DTM time series) -----
    if disturbance is not None:
        cut = disturbance.get("cut_mask")
        fill_m = disturbance.get("fill_mask")
        if cut is not None:
            # Cut on ground = excavation
            px[cut & ground] = LANDSCAPE_TYPES["excavation"]
        if fill_m is not None:
            # Fill on ground = fill
            px[fill_m & ground] = LANDSCAPE_TYPES["fill"]

    # ----- Layer 2: Engineered slopes -----
    if linear is not None:
        ridge = linear.get("ridge_strength")
        valley = linear.get("valley_strength")
        linearity = linear.get("linearity")
        if ridge is not None and linearity is not None:
            # Strong linear ridges at ground level = embankments
            ridge_thresh = np.nanpercentile(ridge[mask], 92) if mask.any() else 0.01
            eng_slope = mask & ground & (ridge > ridge_thresh) & (linearity > 0.5)
            px[eng_slope] = LANDSCAPE_TYPES["engineered_slope"]
            # Strong linear valleys = cuttings/ditches
            if valley is not None:
                valley_thresh = np.nanpercentile(valley[mask], 92) if mask.any() else 0.01
                eng_cut = mask & ground & (valley > valley_thresh) & (linearity > 0.5)
                px[eng_cut] = LANDSCAPE_TYPES["engineered_slope"]

    # ----- Layer 1: Engineered surfaces (roads, parking) -----
    if surfaces is not None:
        eng_surf = surfaces.get("engineered_surface_mask")
        if eng_surf is not None:
            px[eng_surf] = LANDSCAPE_TYPES["engineered_surface"]

    # ----- Layer 0: Recent disturbance (highest priority) -----
    if disturbance is not None:
        disturbed = disturbance.get("total_disturbance_mask")
        intensity = disturbance.get("disturbance_intensity")
        if disturbed is not None and intensity is not None:
            # Only mark as disturbance if intensity is significant
            # and it's not already classified as something more specific
            recent = disturbed & (intensity > 0.3) & (
                (px == LANDSCAPE_TYPES["vegetation"]) |
                (px == LANDSCAPE_TYPES["bare_natural"]) |
                (px == 0)
            )
            px[recent] = LANDSCAPE_TYPES["recent_disturbance"]

    # ----- Calibration against cadastre (if available) -----
    if building_truth is not None:
        # Use cadastre as soft correction:
        # Where cadastre says building but we said tree → check more carefully
        cadastre_building = building_truth.astype(bool)
        missed = cadastre_building & (px == LANDSCAPE_TYPES["tree_canopy"])
        # If elevated and flat-ish, trust cadastre
        px[missed & (ndsm > 2.0) & (dsm_std3 < 0.8)] = LANDSCAPE_TYPES["building"]

    return px


# ===================================================================
# 7. SEGMENT-LEVEL OBJECT EXTRACTION
# ===================================================================

def _segment_elevated(ndsm: np.ndarray, mask: np.ndarray, min_height: float = 0.3) -> np.ndarray:
    """Watershed segmentation of elevated pixels into individual objects."""
    elevated = mask & (ndsm >= min_height)
    if not elevated.any():
        return np.zeros_like(ndsm, dtype=np.int32)

    # Markers from local maxima
    from scipy.ndimage import maximum_filter, label as ndi_label
    local_max = maximum_filter(ndsm, size=7) == ndsm
    local_max &= elevated & (ndsm >= min_height + 0.5)

    markers, n_markers = ndi_label(local_max)
    if n_markers == 0:
        # Fall back to connected components
        labels, _ = ndi_label(elevated)
        return labels

    # Watershed from inverted nDSM
    from skimage.segmentation import watershed
    inv = -ndsm.copy()
    inv[~elevated] = 0
    labels = watershed(inv, markers=markers, mask=elevated)
    return labels


def _extract_objects(
    labels: np.ndarray,
    px: np.ndarray,
    ndsm: np.ndarray,
    dtm: np.ndarray,
    dsm: np.ndarray,
    mask: np.ndarray,
    transform,
    *,
    min_area: int = 4,
    cop: dict | None = None,
    dsm_changes: dict | None = None,
    linear: dict | None = None,
    disturbance: dict | None = None,
) -> list[DetectedObject]:
    """Extract DetectedObject list from segmented labels."""
    regions = measure.regionprops(labels, intensity_image=ndsm)
    objects = []
    obj_id = 0

    for reg in regions:
        area = reg.area
        if area < min_area:
            continue

        seg_mask = labels == reg.label
        seg_ndsm = ndsm[seg_mask]
        seg_px = px[seg_mask]

        h_max = float(np.nanmax(seg_ndsm))
        h_mean = float(np.nanmean(seg_ndsm))
        h_p90 = float(np.nanpercentile(seg_ndsm, 90))
        h_std = float(np.nanstd(seg_ndsm))

        # Morphometrics
        perimeter = reg.perimeter
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        bbox = reg.bbox  # (min_row, min_col, max_row, max_col)
        row_span = bbox[2] - bbox[0]
        col_span = bbox[3] - bbox[1]
        elongation = max(row_span, col_span) / max(min(row_span, col_span), 1)

        # Centroid in map coordinates
        cr, cc = reg.centroid
        ce = transform.c + cc * transform.a
        cn = transform.f + cr * transform.e
        map_bbox = (
            round(transform.c + bbox[1] * transform.a, 1),
            round(transform.f + bbox[2] * transform.e, 1),
            round(transform.c + bbox[3] * transform.a, 1),
            round(transform.f + bbox[0] * transform.e, 1),
        )

        # Dominant pixel class in segment
        type_counts = np.bincount(seg_px, minlength=11)
        dominant_type_code = int(np.argmax(type_counts[1:]) + 1) if type_counts[1:].max() > 0 else 0
        dominant_type = LANDSCAPE_TYPE_NAMES.get(dominant_type_code, "vegetation")

        # Segment-level refinement
        is_manmade = dominant_type in (
            "engineered_surface", "engineered_slope", "excavation",
            "fill", "building", "infrastructure", "recent_disturbance",
        )
        confidence = 0.5

        # ----- Building refinement -----
        if dominant_type_code == LANDSCAPE_TYPES["building"]:
            # Building should be: compact, not too elongated, reasonably sized
            if compactness > 0.2 and area > 10 and area < 10000:
                confidence = 0.7
                # Temporal stability boost
                if dsm_changes is not None and dsm_changes.get("stable_elevated") is not None:
                    stable_frac = float(dsm_changes["stable_elevated"][seg_mask].mean())
                    if stable_frac > 0.5:
                        confidence = min(confidence + 0.2, 0.95)
            else:
                # Too elongated or too small for building → infrastructure
                if elongation > 6 or area < 10:
                    dominant_type = "infrastructure"
                    dominant_type_code = LANDSCAPE_TYPES["infrastructure"]
                    is_manmade = True

        # ----- Tree refinement -----
        if dominant_type_code == LANDSCAPE_TYPES["tree_canopy"]:
            confidence = 0.6
            is_manmade = False
            if dsm_changes is not None and dsm_changes.get("growing") is not None:
                grow_frac = float(dsm_changes["growing"][seg_mask].mean())
                if grow_frac > 0.3:
                    confidence = min(confidence + 0.2, 0.9)

        # ----- Linear feature check -----
        is_linear = False
        if linear is not None and linear.get("engineered_linear_mask") is not None:
            lin_frac = float(linear["engineered_linear_mask"][seg_mask].mean())
            if lin_frac > 0.3:
                is_linear = True
                is_manmade = True

        # ----- Machinery trace check -----
        is_machinery = False
        if disturbance is not None and disturbance.get("total_disturbance_mask") is not None:
            dist_frac = float(disturbance["total_disturbance_mask"][seg_mask].mean())
            if dist_frac > 0.3:
                is_machinery = True
                is_manmade = True

        # Copernicus NDVI for segment
        ndvi_mean = 0.0
        if cop is not None and cop.get("ndvi") is not None:
            seg_ndvi = cop["ndvi"][seg_mask]
            valid = np.isfinite(seg_ndvi)
            if valid.any():
                ndvi_mean = float(np.nanmean(seg_ndvi[valid]))

        # Temporal signal
        temporal_std_val = 0.0
        temporal_stable = False
        temporal_signal = ""
        if dsm_changes is not None:
            tstd = dsm_changes.get("ndsm_temporal_std")
            if tstd is not None:
                temporal_std_val = float(np.nanmean(tstd[seg_mask]))
                temporal_stable = temporal_std_val < 0.5 and h_mean > 2.0
                if temporal_std_val < 0.3:
                    temporal_signal = "stable"
                elif temporal_std_val < 1.0:
                    temporal_signal = "minor_change"
                else:
                    temporal_signal = "major_change"

        obj_id += 1
        objects.append(DetectedObject(
            obj_id=obj_id,
            obj_type=dominant_type,
            type_code=dominant_type_code,
            height_max=round(h_max, 2),
            height_mean=round(h_mean, 2),
            height_p90=round(h_p90, 2),
            area_sqm=round(float(area), 1),
            perimeter_m=round(float(perimeter), 1),
            compactness=round(float(compactness), 3),
            elongation=round(float(elongation), 2),
            height_std=round(h_std, 2),
            centroid_e=round(ce, 1),
            centroid_n=round(cn, 1),
            bbox=map_bbox,
            height_class=_height_class(h_max),
            confidence=round(confidence, 3),
            is_manmade=is_manmade,
            linear_feature=is_linear,
            machinery_trace=is_machinery,
            ndvi_mean=round(ndvi_mean, 4),
            temporal_std=round(temporal_std_val, 3),
            temporal_stable=temporal_stable,
            temporal_signal=temporal_signal,
        ))

    objects.sort(key=lambda o: o.area_sqm, reverse=True)
    return objects


# ===================================================================
# 8. MAIN CLASSIFICATION ENTRY POINT
# ===================================================================

def classify_landscape(
    dtm: np.ndarray,
    dsm: np.ndarray,
    mask: np.ndarray,
    transform,
    *,
    dtm_dates: dict[str, np.ndarray] | None = None,
    dsm_dates: dict[str, np.ndarray] | None = None,
    copernicus: dict | None = None,
    building_footprints: np.ndarray | None = None,
    min_height: float = 0.3,
    min_area: int = 4,
    # Legacy compatibility
    spectral: dict | None = None,
    rgb: np.ndarray | None = None,
    temporal_std: np.ndarray | None = None,
    temporal_range: np.ndarray | None = None,
    n_temporal_dates: int = 1,
) -> dict:
    """Classify landscape into 10 types focused on human transformation.

    Parameters
    ----------
    dtm, dsm : 2D float32 arrays (EPSG:3035, 1m resolution)
    mask : boolean array (valid pixels)
    transform : rasterio Affine
    dtm_dates : {date_str: dtm_array} for multi-temporal DTM analysis
    dsm_dates : {date_str: dsm_array} for multi-temporal DSM analysis
    copernicus : dict from copernicus.py (ndvi, landcover, sar_vv, sar_vh, transform, crs)
    building_footprints : boolean raster from cadastre.py (for calibration only)
    min_height : minimum height for elevated objects
    min_area : minimum object area in pixels

    Returns
    -------
    dict with:
      type_map: 2D uint8 (type codes)
      confidence_map: 2D float32 [0,1]
      objects: list[DetectedObject]
      linear_features: dict from detect_linear_features()
      disturbance: dict from detect_machinery_traces()
      surfaces: dict from detect_engineered_surfaces()
      dsm_changes: dict from detect_dsm_changes()
      stats: summary dict
    """
    h, w = dtm.shape
    ndsm = np.clip(dsm - dtm, 0, None)
    log.info("classify_landscape: %dx%d, valid=%d px", w, h, int(mask.sum()))

    # --- Step 1: Linear feature detection from surface morphology ---
    log.info("Step 1: Linear feature detection (Hessian eigenvalues)")
    linear = detect_linear_features(dtm, dsm, mask)

    # --- Step 2: Engineered surface detection ---
    log.info("Step 2: Engineered surface detection (multi-scale roughness)")
    surfaces = detect_engineered_surfaces(dtm, dsm, mask)

    # --- Step 3: DTM time series machinery detection ---
    disturbance = None
    if dtm_dates is not None and len(dtm_dates) >= 2:
        log.info("Step 3: DTM time series machinery detection (%d dates)", len(dtm_dates))
        disturbance = detect_machinery_traces(dtm_dates, mask)
    else:
        log.info("Step 3: Skipped (no multi-temporal DTM)")

    # --- Step 4: DSM time series change detection ---
    dsm_changes = None
    if dsm_dates is not None and dtm_dates is not None and len(dsm_dates) >= 2:
        log.info("Step 4: DSM time series change detection (%d dates)", len(dsm_dates))
        dsm_changes = detect_dsm_changes(dsm_dates, dtm_dates, mask)
    elif temporal_std is not None:
        # Legacy: construct dsm_changes from old-style temporal data
        dsm_changes = {
            "ndsm_temporal_std": temporal_std,
            "ndsm_temporal_range": temporal_range if temporal_range is not None else np.zeros_like(temporal_std),
            "stable_elevated": mask & (ndsm > 2.5) & (temporal_std < 0.5) if temporal_std is not None else None,
            "growing": None,
            "shrinking": None,
            "ndsm_latest": ndsm,
        }
    else:
        log.info("Step 4: Skipped (no multi-temporal DSM)")

    # --- Step 5: Copernicus integration ---
    cop = None
    if copernicus is not None:
        log.info("Step 5: Copernicus data integration")
        cop = integrate_copernicus(copernicus, transform, (h, w))
    else:
        log.info("Step 5: Skipped (no Copernicus data)")

    # --- Step 6: Pixel-level classification ---
    log.info("Step 6: Pixel-level classification")
    px = _classify_pixels(
        dtm, dsm, mask,
        linear=linear,
        surfaces=surfaces,
        disturbance=disturbance,
        dsm_changes=dsm_changes,
        cop=cop,
        building_truth=building_footprints,
        spectral=spectral,
    )

    # --- Step 7: Segment-level object extraction ---
    log.info("Step 7: Segment-level object extraction")
    labels = _segment_elevated(ndsm, mask, min_height=min_height)
    objects = _extract_objects(
        labels, px, ndsm, dtm, dsm, mask, transform,
        min_area=min_area,
        cop=cop,
        dsm_changes=dsm_changes,
        linear=linear,
        disturbance=disturbance,
    )

    # --- Confidence map ---
    confidence_map = np.where(mask, 0.5, 0.0).astype(np.float32)
    # Boost confidence where multiple signals agree
    if disturbance is not None and disturbance.get("disturbance_intensity") is not None:
        confidence_map += disturbance["disturbance_intensity"] * 0.2
    if dsm_changes is not None and dsm_changes.get("ndsm_temporal_std") is not None:
        # Low temporal std = high confidence in classification
        tstd = dsm_changes["ndsm_temporal_std"]
        confidence_map += np.where(mask & (tstd < 0.3), 0.2, 0.0)
    if cop is not None and cop.get("ndvi") is not None:
        confidence_map += np.where(mask & np.isfinite(cop["ndvi"]), 0.1, 0.0)
    confidence_map = np.clip(confidence_map, 0, 1)

    # --- Stats ---
    stats = _compute_stats(px, mask, objects, disturbance, dsm_changes)

    log.info(
        "Classification complete: %d objects, %d man-made, %d natural",
        len(objects),
        sum(1 for o in objects if o.is_manmade),
        sum(1 for o in objects if not o.is_manmade),
    )

    return {
        "type_map": px,
        "confidence_map": confidence_map,
        "objects": objects,
        "linear_features": linear,
        "disturbance": disturbance,
        "surfaces": surfaces,
        "dsm_changes": dsm_changes,
        "stats": stats,
    }


def _compute_stats(px, mask, objects, disturbance, dsm_changes=None):
    """Compute summary statistics including land use change quantification."""
    total_px = int(mask.sum())
    type_counts = {}
    for name, code in LANDSCAPE_TYPES.items():
        count = int(np.sum(px[mask] == code))
        if count > 0:
            type_counts[name] = {
                "pixels": count,
                "area_sqm": count,  # 1m resolution
                "pct": round(100 * count / max(total_px, 1), 1),
            }

    manmade_codes = {1, 2, 3, 4, 5, 6, 10}
    manmade_px = sum(
        int(np.sum(px[mask] == c)) for c in manmade_codes
    )

    stats = {
        "total_area_sqm": total_px,
        "total_area_ha": round(total_px / 10000, 2),
        "by_type": type_counts,
        "manmade_area_sqm": manmade_px,
        "manmade_pct": round(100 * manmade_px / max(total_px, 1), 1),
        "natural_pct": round(100 * (total_px - manmade_px) / max(total_px, 1), 1),
        "total_objects": len(objects),
        "objects_manmade": sum(1 for o in objects if o.is_manmade),
        "objects_natural": sum(1 for o in objects if not o.is_manmade),
    }

    # --- Terrain disturbance quantification ---
    if disturbance is not None:
        dist_mask = disturbance.get("total_disturbance_mask", np.zeros(1, dtype=bool))
        cut_mask = disturbance.get("cut_mask", np.zeros(1, dtype=bool))
        fill_mask = disturbance.get("fill_mask", np.zeros(1, dtype=bool))
        stats["terrain_disturbance"] = {
            "total_disturbed_sqm": int(dist_mask.sum()),
            "total_disturbed_ha": round(int(dist_mask.sum()) / 10000, 3),
            "cut_area_sqm": int(cut_mask.sum()),
            "fill_area_sqm": int(fill_mask.sum()),
            "per_pair": disturbance.get("per_pair", []),
            "interpretation": _interpret_disturbance(disturbance),
        }

    # --- Surface change quantification (logging, growth, etc.) ---
    if dsm_changes is not None:
        growing = dsm_changes.get("growing")
        shrinking = dsm_changes.get("shrinking")
        ndsm_latest = dsm_changes.get("ndsm_latest")
        ndsm_stack = dsm_changes.get("ndsm_stack")

        surface_stats = {}
        if growing is not None:
            tree_growth = growing & (ndsm_latest > 4) if ndsm_latest is not None else growing
            surface_stats["tree_growth_sqm"] = int(tree_growth.sum())
        if shrinking is not None:
            # Logging/felling: was tall, now short
            if ndsm_stack is not None and len(ndsm_stack) >= 2:
                was_tall = ndsm_stack[0] > 4
                now_short = ndsm_stack[-1] < 2
                felled = mask & was_tall & now_short
                surface_stats["tree_felling_sqm"] = int(felled.sum())
                surface_stats["tree_felling_ha"] = round(int(felled.sum()) / 10000, 3)

                # Large clearcut areas (>100m² connected)
                if felled.any():
                    labels, n = ndimage.label(felled)
                    clearcuts = []
                    for i in range(1, n + 1):
                        area = int((labels == i).sum())
                        if area > 100:
                            clearcuts.append(area)
                    surface_stats["clearcut_areas_sqm"] = sorted(clearcuts, reverse=True)
                    surface_stats["clearcut_total_sqm"] = sum(clearcuts)

            surface_stats["vegetation_loss_sqm"] = int(shrinking.sum())

        # Excavation/mining from DTM (already in disturbance, but quantify volume)
        if disturbance is not None:
            cut = disturbance.get("cut_mask")
            fill_m = disturbance.get("fill_mask")
            if cut is not None and ndsm_stack is not None and len(ndsm_stack) >= 2:
                # Volume estimation: sum of DTM differences on cut/fill pixels
                # (rough estimate at 1m² × height change)
                per_pair = disturbance.get("per_pair", [])
                if per_pair:
                    total_cut_vol = sum(p.get("pixels_cut", 0) * abs(p.get("dtm_change_mean", 0))
                                        for p in per_pair)
                    total_fill_vol = sum(p.get("pixels_fill", 0) * abs(p.get("dtm_change_mean", 0))
                                         for p in per_pair)
                    surface_stats["excavation_volume_m3_est"] = round(total_cut_vol, 0)
                    surface_stats["fill_volume_m3_est"] = round(total_fill_vol, 0)

        stats["surface_changes"] = surface_stats

    return stats


def _interpret_disturbance(disturbance: dict) -> str:
    """Generate human-readable interpretation of terrain disturbance."""
    per_pair = disturbance.get("per_pair", [])
    if not per_pair:
        return "No terrain disturbance detected."

    parts = []
    for p in per_pair:
        da, db = p.get("date_a", "?"), p.get("date_b", "?")
        area = p.get("area_disturbed_sqm", 0)
        cut = p.get("pixels_cut", 0)
        fill = p.get("pixels_fill", 0)
        smoothed = p.get("pixels_smoothed", 0)
        paired = p.get("pixels_paired_cut_fill", 0)

        if area == 0:
            parts.append(f"{da}→{db}: no significant terrain change.")
            continue

        desc = f"{da}→{db}: {area}m² terrain modified"
        details = []
        if cut > 0:
            details.append(f"{cut}m² excavated")
        if fill > 0:
            details.append(f"{fill}m² filled")
        if smoothed > 0:
            details.append(f"{smoothed}m² graded/smoothed")
        if paired > 0:
            details.append(f"{paired}m² paired cut+fill (earthmoving)")
        if details:
            desc += " (" + ", ".join(details) + ")"
        parts.append(desc)

    return " ".join(parts)


# ===================================================================
# 9. BACKWARD-COMPATIBLE API
# ===================================================================

def classify_objects(
    ndsm: np.ndarray,
    dtm: np.ndarray,
    mask: np.ndarray,
    transform,
    *,
    min_height: float = 0.3,
    min_area: int = 4,
    dsm: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    spectral: dict | None = None,
    temporal_std: np.ndarray | None = None,
    temporal_range: np.ndarray | None = None,
    n_temporal_dates: int = 1,
) -> list[DetectedObject]:
    """Backward-compatible entry point matching old object_classifier API.

    Translates old-style arguments into classify_landscape() call.
    """
    if dsm is None:
        dsm = dtm + ndsm

    result = classify_landscape(
        dtm, dsm, mask, transform,
        min_height=min_height,
        min_area=min_area,
        spectral=spectral,
        rgb=rgb,
        temporal_std=temporal_std,
        temporal_range=temporal_range,
        n_temporal_dates=n_temporal_dates,
    )
    return result["objects"]


def summarise_objects(objects: list[DetectedObject]) -> dict:
    """Summarise detected objects (backward-compatible)."""
    if not objects:
        return {"total_objects": 0, "by_type": {}, "by_height_class": {}}

    by_type: dict[str, dict] = {}
    by_height: dict[str, dict] = {}

    for obj in objects:
        t = obj.obj_type
        by_type.setdefault(t, {
            "count": 0, "total_area_sqm": 0, "height_max": 0, "heights": [],
        })
        by_type[t]["count"] += 1
        by_type[t]["total_area_sqm"] += obj.area_sqm
        by_type[t]["height_max"] = max(by_type[t]["height_max"], obj.height_max)
        by_type[t]["heights"].append(obj.height_max)

        hc = obj.height_class
        by_height.setdefault(hc, {"count": 0, "types": {}})
        by_height[hc]["count"] += 1
        by_height[hc]["types"].setdefault(t, 0)
        by_height[hc]["types"][t] += 1

    for t, info in by_type.items():
        heights = info.pop("heights")
        info["total_area_sqm"] = round(info["total_area_sqm"], 1)
        info["height_mean"] = round(float(np.mean(heights)), 2)
        info["height_std"] = round(float(np.std(heights)), 2) if len(heights) > 1 else 0
        info["height_max"] = round(info["height_max"], 2)

    result = {
        "total_objects": len(objects),
        "by_type": by_type,
        "by_height_class": by_height,
        "manmade_count": sum(1 for o in objects if o.is_manmade),
        "natural_count": sum(1 for o in objects if not o.is_manmade),
        "manmade_pct": round(
            100 * sum(1 for o in objects if o.is_manmade) / max(len(objects), 1), 1
        ),
    }
    return result


def create_classified_raster(
    ndsm: np.ndarray,
    dtm: np.ndarray,
    dsm: np.ndarray,
    mask: np.ndarray,
    transform,
    objects: list[DetectedObject],
    output_resolution: float = 1.0,
    spectral: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Create 2-band raster: band 1 = type code, band 2 = height.

    Backward-compatible with old object_classifier API.
    """
    h, w = ndsm.shape

    # Run pixel classification
    result = classify_landscape(
        dtm, dsm, mask, transform,
        min_height=0.3,
        spectral=spectral,
    )
    type_band = result["type_map"].copy()
    height_band = np.where(mask, ndsm, -9999).astype(np.float32)

    out_tf = transform
    if output_resolution != 1.0:
        new_h = max(1, int(h / output_resolution))
        new_w = max(1, int(w / output_resolution))
        height_band = ndimage.zoom(
            np.where(mask, ndsm, 0).astype(np.float32),
            (new_h / h, new_w / w), order=1,
        ).astype(np.float32)
        mask_r = ndimage.zoom(
            mask.astype(np.float32), (new_h / h, new_w / w), order=0,
        ) > 0.5
        height_band[~mask_r] = -9999
        type_band = ndimage.zoom(
            type_band.astype(np.float32), (new_h / h, new_w / w), order=0,
        ).astype(np.uint8)
        import rasterio.transform
        out_tf = rasterio.transform.from_origin(
            transform.c, transform.f, output_resolution, output_resolution,
        )

    return type_band, height_band, out_tf
