"""Object detection and classification from BEV ALS DTM + DSM.

Two-phase approach:
  Phase 1 — Pixel-level surface classification using 3D shape analysis
  Phase 2 — Watershed segmentation of elevated pixels into individual objects,
            classified by aggregating pixel-level labels + morphometrics.

Optional spectral integration (Phase 1b):
  When RGB orthophoto data is available, spectral indices (NDVI, brightness,
  colour ratios) refine the geometry-based classification.  Without spectral
  data the classifier behaves identically to the pure-LIDAR version.

Surface signatures (empirically derived from Austrian ALS 1m data):
  - Ground (road/path):  nDSM < 0.2m, DTM smooth (std3 < 0.1)
  - Ground (meadow):     nDSM < 0.2m, DTM normal (std3 0.1–0.5)
  - Ground (rough/rock):  nDSM < 0.2m, DTM rough (std3 > 0.5)
  - Low vegetation:       nDSM 0.2–2m
  - Shrub/bush:           nDSM 2–4m
  - Building:             nDSM > 2m, low NDVI (<0.20), bright (>100), compact shape
  - Tree canopy:          nDSM > 4m, NDVI > 0.20, rough surface
  - Coniferous crown:     peak with steep radial dropoff (>5m in 5m radius)
  - Broadleaf crown:      dome with gentle dropoff (<4m in 5m radius)

Spectral refinements (when orthophoto available):
  - NDVI > 0.25 → vegetation;  < 0.12 → built / bare
  - NDVI < 0.0 AND brightness < 50 → water (very strict)
  - Blue-dominant → swimming pool
  - Brightness > 100 + low NDVI → road / parking lot
  - Tall + NDVI < 0.10 + brightness > 100 → building (not dead tree)
  - Regular low-veg pattern → vineyard / orchard

Calibrated against BEV cadastre ground truth from 100 sample areas across
Austria (urban, suburban, rural, alpine, forest, water) — 2026-04 calibration.

Key innovation: SCENE-ADAPTIVE THRESHOLDS.  Instead of fixed spectral
cutoffs, we compute the scene's NDVI/brightness distribution and set
thresholds relative to local conditions.  This handles winter orthophotos
(low NDVI everywhere), summer alpine (high contrast), and everything between.

Multi-temporal consensus (when available): objects stable across 2-3 LIDAR
dates are almost certainly buildings; growing/shrinking objects are vegetation.

  Water: ZERO pixels confirmed in Kohlschwarz — ultra-smooth ground is
  typically road surface, roof sheeting, or solar panels, not water.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage import measure, morphology, segmentation, feature

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Object type codes for raster output
# ---------------------------------------------------------------------------

OBJECT_TYPES = {
    # Ground types (0-3)
    "ground": 0,
    "road_path": 1,
    "meadow_field": 2,
    "rough_ground": 3,
    # Low / medium vegetation (4-5)
    "low_vegetation": 4,
    "shrub_bush": 5,
    # Trees (6-8)
    "tree_coniferous": 6,
    "tree_broadleaf": 7,
    "tree_unclassified": 8,
    # Buildings & structures (9-12)
    "building": 9,
    "structure": 10,
    "mast_pole": 11,
    "wall_fence": 12,
    # Water (13)
    "water": 13,
    "unclassified": 14,
    # --- NEW types ---
    "parking_lot": 15,
    "swimming_pool": 16,
    "solar_panel": 17,
    "greenhouse": 18,
    "bridge": 19,
    "power_line": 20,
    "hedge": 21,
    "tree_row": 22,
    "dead_tree": 23,
    "bare_soil": 24,
    "rock_cliff": 25,
    "vineyard_orchard": 26,
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


def _temporal_signal_label(seg_temporal: dict, h_max: float) -> str:
    """Derive a human-readable temporal signal from segment stats."""
    if seg_temporal is None:
        return ""
    t_std = seg_temporal["std_mean"]
    t_rng = seg_temporal["range_mean"]
    stable_frac = seg_temporal["stable_frac"]
    if t_std < 0.5 and stable_frac > 0.7:
        return "stable"
    if t_std > 2.0:
        if t_rng > 3.0:
            return "major_change"
        return "variable"
    if t_std > 1.0:
        return "moderate_change"
    return "minor_variation"


# ---------------------------------------------------------------------------
# Spectral helper — extract per-pixel index arrays from dict
# ---------------------------------------------------------------------------

def _get_spectral(spectral: dict | None, key: str) -> np.ndarray | None:
    """Safely retrieve a spectral index array, or *None*."""
    if spectral is None:
        return None
    arr = spectral.get(key)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Scene context — adaptive thresholds based on local conditions
# ---------------------------------------------------------------------------

@dataclass
class SceneContext:
    """Scene-adaptive thresholds computed from local NDVI / brightness.

    Instead of global constants (which fail across Austria's diverse
    landscape — winter alpine vs summer lowland), every threshold is
    derived from the statistical distribution of the query window.
    """
    # --- Raw scene statistics ---
    ndvi_p10: float = -0.05
    ndvi_p25: float = 0.05
    ndvi_median: float = 0.15
    ndvi_p75: float = 0.30
    ndvi_p90: float = 0.40
    bri_p10: float = 50.0
    bri_p25: float = 70.0
    bri_median: float = 90.0
    bri_p75: float = 120.0
    bri_p90: float = 150.0
    altitude_median: float = 500.0

    # --- Derived thresholds (set by _compute) ---
    # Building detection: NDVI below this AND brightness above bri_building
    ndvi_building: float = 0.15     # objects below this are non-vegetated
    bri_building: float = 85.0      # objects above this are bright enough
    # Tree recovery: building pixels above this NDVI → tree
    ndvi_tree_recovery: float = 0.22
    # Dead tree: VERY strict
    ndvi_dead_max: float = 0.05
    bri_dead_max: float = 65.0
    # Water
    ndvi_water_max: float = -0.05
    bri_water_max: float = 50.0
    # Scene type flag
    is_winter: bool = False        # True if scene NDVI is anomalously low
    is_dark: bool = False          # True if brightness is anomalously low
    # Multi-temporal stability (set when ndsm_dates provided)
    has_temporal: bool = False     # True if multi-date LIDAR available
    n_dates: int = 1               # Number of LIDAR dates loaded


def _compute_scene_context(
    ndsm: np.ndarray,
    dtm: np.ndarray,
    mask: np.ndarray,
    spectral: dict | None = None,
) -> SceneContext:
    """Compute scene-adaptive thresholds from the local raster window.

    The key insight: a winter orthophoto over alpine forest has NDVI~0.15
    for healthy conifers, while a summer lowland has NDVI~0.45.  Fixed
    thresholds (NDVI < 0.20 → building) work in one but fail in the other.

    Instead, we compute where buildings and trees ACTUALLY sit in this
    scene's distribution, using the bimodal structure of the NDVI histogram
    (built/bare vs vegetated) and the altitude to estimate season effects.
    """
    ctx = SceneContext()

    # Altitude → season estimate
    dtm_valid = dtm[mask]
    if len(dtm_valid) > 0:
        ctx.altitude_median = float(np.median(dtm_valid))

    ndvi = _get_spectral(spectral, "ndvi")
    brightness = _get_spectral(spectral, "brightness")

    if ndvi is not None:
        ndvi_valid = ndvi[mask]
        if len(ndvi_valid) > 100:
            ctx.ndvi_p10 = float(np.percentile(ndvi_valid, 10))
            ctx.ndvi_p25 = float(np.percentile(ndvi_valid, 25))
            ctx.ndvi_median = float(np.median(ndvi_valid))
            ctx.ndvi_p75 = float(np.percentile(ndvi_valid, 75))
            ctx.ndvi_p90 = float(np.percentile(ndvi_valid, 90))

    if brightness is not None:
        bri_valid = brightness[mask]
        if len(bri_valid) > 100:
            ctx.bri_p10 = float(np.percentile(bri_valid, 10))
            ctx.bri_p25 = float(np.percentile(bri_valid, 25))
            ctx.bri_median = float(np.median(bri_valid))
            ctx.bri_p75 = float(np.percentile(bri_valid, 75))
            ctx.bri_p90 = float(np.percentile(bri_valid, 90))

    # --- Detect winter/dark scene ---
    # Winter alpine: scene NDVI median < 0.20 (summer is typically 0.25-0.40)
    ctx.is_winter = ctx.ndvi_median < 0.20
    ctx.is_dark = ctx.bri_median < 75

    # --- Compute adaptive thresholds ---
    # The NDVI histogram of a mixed scene is bimodal:
    #   Mode 1 (left): built/bare surfaces (roads, buildings, bare soil)
    #   Mode 2 (right): vegetation (trees, grass, crops)
    # The building threshold should sit between the modes.
    #
    # Heuristic: building threshold = midpoint of p25 and p50, clamped.
    # In summer (median=0.35): threshold = (0.10 + 0.35)/2 = 0.22 → good
    # In winter (median=0.15): threshold = (0.02 + 0.15)/2 = 0.09 → tight, avoids trees
    ctx.ndvi_building = np.clip(
        (ctx.ndvi_p25 + ctx.ndvi_median) / 2,
        0.05,   # never below 0.05 (would miss everything)
        0.22,   # never above 0.22 (calibrated upper bound from 100 samples)
    )

    # Brightness threshold: buildings are brighter than trees.
    # Use scene p35 as floor (buildings should be above ~35th percentile).
    ctx.bri_building = np.clip(
        ctx.bri_p25 + 5,  # slightly above p25
        70.0,   # absolute floor
        110.0,  # cap: don't require TOO bright
    )

    # Tree recovery: building pixels with NDVI above this → reclassify as tree
    # Set above the building threshold but below vegetation center.
    # In winter: recovery at ~0.14.  In summer: at ~0.26.
    ctx.ndvi_tree_recovery = np.clip(
        ctx.ndvi_building + 0.06,
        0.12,
        0.28,
    )

    # Dead tree: must be well below the scene's vegetation floor AND dark.
    # In winter (ndvi_p10 = -0.02): dead tree < -0.02 → almost nothing qualifies
    # In summer (ndvi_p10 = 0.05): dead tree < 0.05
    ctx.ndvi_dead_max = np.clip(
        ctx.ndvi_p10,
        -0.10,
        0.08,
    )
    ctx.bri_dead_max = np.clip(
        ctx.bri_p10 + 5,
        40.0,
        70.0,
    )

    log.info(
        f"Scene context: alt={ctx.altitude_median:.0f}m "
        f"ndvi=[{ctx.ndvi_p10:.2f},{ctx.ndvi_median:.2f},{ctx.ndvi_p90:.2f}] "
        f"bri=[{ctx.bri_p10:.0f},{ctx.bri_median:.0f},{ctx.bri_p90:.0f}] "
        f"winter={ctx.is_winter} "
        f"thresholds: bld_ndvi<{ctx.ndvi_building:.2f} bld_bri>{ctx.bri_building:.0f} "
        f"tree_recov>{ctx.ndvi_tree_recovery:.2f} dead<{ctx.ndvi_dead_max:.2f}"
    )
    return ctx


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

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
    # --- spectral fields (populated only when orthophoto available) ---
    ndvi_mean: float = 0.0
    ndvi_max: float = 0.0
    brightness_mean: float = 0.0
    spectral_class: str = ""
    # --- temporal stability fields (populated when multi-date LIDAR available) ---
    temporal_std: float = 0.0       # mean std of nDSM across dates
    temporal_stable: bool = False   # True if object is temporally stable
    temporal_signal: str = ""       # "stable" / "growing" / "shrinking" / "new" / "removed"
    label: int = 0                  # raster segment label (for crown polygon lookup)
    # --- v2.3 non-forest surface verdict (set by app._trees_core via
    #     tree_inventory.surface_filter_labels; shared with /api/v2/trees) ---
    surface_class: str = ""
    tree_likelihood: float = 1.0


# ---------------------------------------------------------------------------
# Phase 1: Pixel-level surface classification
# ---------------------------------------------------------------------------

def _classify_pixels(
    ndsm: np.ndarray,
    dtm: np.ndarray,
    dsm: np.ndarray,
    mask: np.ndarray,
    spectral: dict | None = None,
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

    # --- Classify ground (nDSM < 0.2m) ---
    # Lowered from 0.3m to catch height changes as soon as they exceed 0.2m
    ground = mask & (n < 0.2)
    # Water: VERY strict geometry thresholds.
    # Calibration showed that ultra-smooth ground in Austria is almost always
    # road surface, roof sheeting, or solar panels — not water.  Water bodies
    # require spectral confirmation (Phase 1b) to be classified.
    water = ground & (dtm_std3 < 0.01) & (dsm_std3 < 0.01) & (dtm_slope < 1)
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

    # --- Low vegetation (0.2–2m) ---
    low_veg = mask & (n >= 0.2) & (n < 2.0)
    px[low_veg] = OBJECT_TYPES["low_vegetation"]

    # --- Shrubs/bushes (2–4m) ---
    shrub = mask & (n >= 2.0) & (n < 4.0)
    px[shrub] = OBJECT_TYPES["shrub_bush"]

    # --- Elevated objects (≥4m): building roofs vs tree canopy ---
    elevated = mask & (n >= 4.0)
    #
    # Calibration against cadastre (KG 63330) showed:
    #   - Real buildings: nDSM_slope median=43° (steep gabled roofs!),
    #     dsm_std3 median=0.74, nDSM_std5 median=1.47
    #   - Real trees: nDSM_slope median=66°, dsm_std3 median=1.99,
    #     nDSM_std5 median=3.17
    # Key insight: geometry alone cannot reliably separate buildings from
    # trees in Austrian alpine settings.  Use CONSERVATIVE geometry to mark
    # candidates, then SPECTRAL refinement (Phase 1b) does the real work.
    # Geometry candidates: very flat + smooth surfaces only (high confidence)
    ndsm_std3 = _fast_local_std(n, 3)
    building_surface = elevated & (ndsm_std5 < 1.5) & (dsm_std3 < 0.8) & (ndsm_std3 < 1.0)
    building_px = building_surface & (dtm_slope < 25)
    px[building_px] = OBJECT_TYPES["building"]

    # Everything else elevated defaults to tree (spectral phase will
    # promote non-green elevated pixels to building)
    tree_canopy = elevated & ~building_px
    px[tree_canopy] = OBJECT_TYPES["tree_unclassified"]  # refined later

    # --- Rock/cliff: steep DTM + rough ground, regardless of nDSM height ---
    rock_cliff = mask & (dtm_slope > 45) & (dtm_std3 > 0.8)
    px[rock_cliff] = OBJECT_TYPES["rock_cliff"]

    # --- Phase 1b: spectral refinement (optional, now scene-adaptive) ---
    if spectral is not None:
        ctx = _compute_scene_context(n, dtm, mask, spectral)
        px = _refine_with_spectral(px, spectral, n, mask, dtm_slope, dtm_std3,
                                   ndsm_slope, ndsm_std5, dsm_std3, ctx=ctx)

    return px


# ---------------------------------------------------------------------------
# Phase 1b: Spectral pixel-level refinement
# ---------------------------------------------------------------------------

def _refine_with_spectral(
    px: np.ndarray,
    spectral: dict,
    ndsm: np.ndarray,
    mask: np.ndarray,
    dtm_slope: np.ndarray,
    dtm_std3: np.ndarray,
    ndsm_slope: np.ndarray,
    ndsm_std5: np.ndarray,
    dsm_std3: np.ndarray,
    *,
    ctx: SceneContext | None = None,
) -> np.ndarray:
    """Refine geometry-based pixel classification using spectral indices.

    When *ctx* is provided, thresholds are scene-adaptive rather than fixed.
    Operates *in-place* on *px* and returns it.
    """
    if ctx is None:
        ctx = SceneContext()  # fallback to defaults
    ndvi = _get_spectral(spectral, "ndvi")
    brightness = _get_spectral(spectral, "brightness")
    green_ratio = _get_spectral(spectral, "green_ratio")
    rg_index = _get_spectral(spectral, "rg_index")
    blue_ratio = _get_spectral(spectral, "blue_ratio")

    # Bail if we got nothing useful
    if ndvi is None and brightness is None:
        return px

    # Make sure shapes match — spectral might have been resampled to the
    # LIDAR grid but safety-check anyway.
    target_shape = px.shape
    def _conform(arr):
        if arr is None:
            return None
        if arr.shape != target_shape:
            from scipy.ndimage import zoom
            factors = (target_shape[0] / arr.shape[0],
                       target_shape[1] / arr.shape[1])
            return zoom(arr, factors, order=1).astype(np.float32)
        return arr

    ndvi = _conform(ndvi)
    brightness = _conform(brightness)
    green_ratio = _conform(green_ratio)
    rg_index = _conform(rg_index)
    blue_ratio = _conform(blue_ratio)

    OT = OBJECT_TYPES  # shorthand

    # ------------------------------------------------------------------
    # 1. Bare soil: ground-level meadow pixels with very low NDVI
    # ------------------------------------------------------------------
    if ndvi is not None:
        bare = (
            mask
            & ((px == OT["meadow_field"]) | (px == OT["rough_ground"]))
            & (ndvi < 0.15)
        )
        if brightness is not None:
            # bare soil is typically medium brightness, not super dark
            # Shadow zones (brightness < 60) can have low NDVI but aren't bare
            bare = bare & (brightness > 60)
        px[bare] = OT["bare_soil"]

    # ------------------------------------------------------------------
    # 2. Road vs meadow disambiguation
    #    Geometry-only often confuses them.  Roads are bright/grey + low NDVI.
    # ------------------------------------------------------------------
    if ndvi is not None and brightness is not None:
        # Meadow pixels that look spectrally green → keep as meadow (no-op)
        # Meadow pixels with low NDVI + high brightness → reclassify as road
        meadow_to_road = (
            mask
            & (px == OT["meadow_field"])
            & (ndvi < 0.12)
            & (brightness > 100)
        )
        px[meadow_to_road] = OT["road_path"]

        # Road pixels that are actually green → reclassify as meadow
        road_to_meadow = (
            mask
            & (px == OT["road_path"])
            & (ndvi > 0.35)
        )
        px[road_to_meadow] = OT["meadow_field"]

    # ------------------------------------------------------------------
    # 3. Water refinement: VERY strict spectral confirmation required.
    #    Calibration showed that geometry-only "water" pixels are almost
    #    always road surfaces or smooth ground.  Real water needs:
    #    negative NDVI + very dark + ground level.
    # ------------------------------------------------------------------
    if ndvi is not None and brightness is not None:
        # Real water: ground-level, very dark, negative NDVI
        water_confirmed = (
            mask
            & (ndsm < 0.2)
            & (ndvi < -0.05)
            & (brightness < 50)
        )
        px[water_confirmed] = OT["water"]

        # Swimming pool: blue-dominant, small patches near ground level
        if blue_ratio is not None:
            pool = (
                mask
                & (ndsm < 1.0)
                & (blue_ratio > 0.38)
                & (ndvi < 0.05)
                & (brightness > 50)
                & (brightness < 200)
            )
            px[pool] = OT["swimming_pool"]

    # ------------------------------------------------------------------
    # 3b. Water correction: any geometry-based water that doesn't meet
    #     strict spectral criteria gets reclassified.
    #     Bright "water" → road_path (roof sheeting, smooth asphalt)
    #     Green "water" → meadow_field
    # ------------------------------------------------------------------
    if ndvi is not None:
        water_to_meadow = (
            mask
            & (px == OT["water"])
            & (ndvi > 0.10)
        )
        px[water_to_meadow] = OT["meadow_field"]

    if brightness is not None:
        water_to_road = (
            mask
            & (px == OT["water"])
            & (brightness > 80)
        )
        px[water_to_road] = OT["road_path"]

    # ------------------------------------------------------------------
    # 4. Dead tree: tall + very low NDVI (brown/grey canopy)
    #    Scene-adaptive: must be below scene ndvi_dead_max AND dark.
    #    Calibration (100 samples): 5,321 FPs were mostly dark building
    #    parts.  Require rough surface texture (vegetation, not smooth roof).
    # ------------------------------------------------------------------
    if ndvi is not None:
        dead_cond = (
            mask
            & ((px == OT["tree_unclassified"])
               | (px == OT["tree_coniferous"])
               | (px == OT["tree_broadleaf"]))
            & (ndvi < ctx.ndvi_dead_max)
            & (ndsm >= 4.0)
            & (ndsm < 25.0)          # dead trees don't reach 25m+
            & (dsm_std3 > 0.3)       # rough surface = vegetation, not smooth roof
        )
        if brightness is not None:
            dead_cond = dead_cond & (brightness < ctx.bri_dead_max)
        px[dead_cond] = OT["dead_tree"]

    # ------------------------------------------------------------------
    # 5. Parking lot: large flat area, low NDVI, bright, ground-to-low height
    #    Tightened: only mark very clear candidates at pixel level.
    #    Segment level (Phase 3) confirms by size (>500m²).
    # ------------------------------------------------------------------
    if ndvi is not None and brightness is not None:
        parking = (
            mask
            & (ndsm < 0.3)
            & (ndvi < ctx.ndvi_p25)       # below scene's 25th percentile
            & (brightness > ctx.bri_p75)  # above scene's 75th percentile
            & (dsm_std3 < 0.10)
            & (dtm_slope < 3)
        )
        px[parking] = OT["parking_lot"]

    # ------------------------------------------------------------------
    # 6. Solar panel: dark rectangular patches on building roofs
    #    Solar panels absorb NIR → strongly negative NDVI, very dark.
    # ------------------------------------------------------------------
    if ndvi is not None and brightness is not None:
        solar = (
            mask
            & (px == OT["building"])
            & (brightness < 50)
            & (ndvi < -0.05)  # strongly negative (absorbs NIR)
        )
        px[solar] = OT["solar_panel"]

    # ------------------------------------------------------------------
    # 7. Greenhouse: DISABLED at pixel level.
    #    Calibration: 16,183 false positives (bright roofs misclassified).
    #    Greenhouse detection moved to segment level (Phase 3) only,
    #    with strict shape + height + rural-context constraints.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 8. Rock/cliff spectral confirmation: steep + low NDVI + grey
    #    Only apply at ground level to avoid catching elevated objects.
    # ------------------------------------------------------------------
    if ndvi is not None:
        rock_extra = (
            mask
            & (ndsm < 0.5)          # ground level only
            & (dtm_slope > 40)
            & (dtm_std3 > 0.6)
            & (ndvi < ctx.ndvi_dead_max)  # scene-relative
        )
        px[rock_extra] = OT["rock_cliff"]

    # ------------------------------------------------------------------
    # 9. Building vs tree: SCENE-ADAPTIVE spectral separation.
    #
    #    The scene context provides thresholds tuned to local conditions:
    #    - ctx.ndvi_building: derived from scene NDVI distribution
    #      Summer lowland: ~0.22.  Winter alpine: ~0.09.
    #    - ctx.bri_building: derived from scene brightness distribution
    #    - ctx.ndvi_tree_recovery: building → tree correction threshold
    #
    #    This eliminates the core failure mode: fixed thresholds that
    #    work in summer but misclassify winter forest as buildings.
    # ------------------------------------------------------------------
    if ndvi is not None and brightness is not None:
        is_tree = (
            (px == OT["tree_unclassified"])
            | (px == OT["tree_broadleaf"])
            | (px == OT["tree_coniferous"])
        )
        # PRIMARY: tree pixels with low NDVI + bright surface → building
        tree_to_bld = (
            mask & is_tree
            & (ndvi < ctx.ndvi_building)
            & (brightness > ctx.bri_building)
            & (dtm_slope < 35)
        )
        px[tree_to_bld] = OT["building"]

        # SECONDARY: very low NDVI + moderate brightness (dark roofs)
        # Require brightness above scene p10+10 to avoid dark forest canopy
        tree_to_bld2 = (
            mask & is_tree
            & (ndvi < ctx.ndvi_building - 0.05)
            & (brightness > max(ctx.bri_p10 + 10, 70))
            & (ndsm >= 2.0)
            & (dtm_slope < 35)
        )
        px[tree_to_bld2] = OT["building"]

        # TERTIARY: shrub-height (2-4m) clearly non-green → building
        # Only when VERY clearly non-vegetated (below scene floor)
        shrub_to_bld = (
            mask
            & (px == OT["shrub_bush"])
            & (ndvi < ctx.ndvi_building - 0.05)
            & (brightness > ctx.bri_building)
            & (dsm_std3 < 0.5)
        )
        px[shrub_to_bld] = OT["building"]

        # RECOVERY: building pixels with vegetation NDVI → tree canopy
        bld_to_tree = (
            mask
            & (px == OT["building"])
            & (ndvi > ctx.ndvi_tree_recovery)
        )
        px[bld_to_tree] = OT["tree_unclassified"]

    elif ndvi is not None:
        # No brightness available — use NDVI alone at conservative threshold
        is_tree = (
            (px == OT["tree_unclassified"])
            | (px == OT["tree_broadleaf"])
            | (px == OT["tree_coniferous"])
        )
        tree_to_bld = (
            mask & is_tree
            & (ndvi < ctx.ndvi_building - 0.05)
            & (ndsm >= 2.0)
            & (dtm_slope < 35)
        )
        px[tree_to_bld] = OT["building"]

        bld_to_tree = (
            mask
            & (px == OT["building"])
            & (ndvi > ctx.ndvi_tree_recovery)
        )
        px[bld_to_tree] = OT["tree_unclassified"]

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
    ground_classes = {
        OBJECT_TYPES["road_path"],
        OBJECT_TYPES["water"],
        OBJECT_TYPES["parking_lot"],
        OBJECT_TYPES["swimming_pool"],
        OBJECT_TYPES["bare_soil"],
        OBJECT_TYPES["rock_cliff"],
    }
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
# Spectral statistics for a segment
# ---------------------------------------------------------------------------

def _segment_spectral_stats(
    obj_mask: np.ndarray,
    spectral: dict | None,
    rgb: np.ndarray | None,
) -> dict:
    """Compute per-segment spectral statistics.

    Returns a dict with keys: ndvi_mean, ndvi_max, brightness_mean,
    green_ratio_mean, blue_ratio_mean, rg_index_mean, spectral_class.
    All default to 0 / "" when spectral data is unavailable.
    """
    stats: dict = {
        "ndvi_mean": None,    # None = unavailable (no NIR band)
        "ndvi_max": None,
        "brightness_mean": 0.0,
        "green_ratio_mean": 0.0,
        "blue_ratio_mean": 0.0,
        "rg_index_mean": 0.0,
        "spectral_class": "",
    }
    if spectral is None and rgb is None:
        return stats

    npx = int(np.count_nonzero(obj_mask))
    if npx == 0:
        return stats

    # NDVI
    ndvi = _get_spectral(spectral, "ndvi")
    if ndvi is not None:
        safe = ndvi.shape == obj_mask.shape
        if safe:
            vals = ndvi[obj_mask]
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                stats["ndvi_mean"] = float(np.mean(finite))
                stats["ndvi_max"] = float(np.max(finite))

    # Brightness
    bri = _get_spectral(spectral, "brightness")
    if bri is not None and bri.shape == obj_mask.shape:
        stats["brightness_mean"] = float(np.nanmean(bri[obj_mask]))

    # Green ratio
    gr = _get_spectral(spectral, "green_ratio")
    if gr is not None and gr.shape == obj_mask.shape:
        stats["green_ratio_mean"] = float(np.nanmean(gr[obj_mask]))

    # Blue ratio (may or may not be in spectral dict)
    br = _get_spectral(spectral, "blue_ratio")
    if br is not None and br.shape == obj_mask.shape:
        stats["blue_ratio_mean"] = float(np.nanmean(br[obj_mask]))

    # RG index
    rg = _get_spectral(spectral, "rg_index")
    if rg is not None and rg.shape == obj_mask.shape:
        stats["rg_index_mean"] = float(np.nanmean(rg[obj_mask]))

    # Derive high-level spectral class
    nm = stats["ndvi_mean"]
    bm = stats["brightness_mean"]
    gm = stats["green_ratio_mean"]
    blm = stats["blue_ratio_mean"]

    if nm is not None:
        # Real NDVI available (NIR band present)
        # Water requires VERY strict criteria — dark + negative NDVI
        if nm < -0.05 and bm < 50:
            stats["spectral_class"] = "water"
        elif nm < 0.08 and bm > 100:
            stats["spectral_class"] = "bright_impervious"  # road / building
        elif nm < 0.08 and bm < 60:
            stats["spectral_class"] = "dark_impervious"     # shadow / dark roof
        elif nm < 0.12:
            stats["spectral_class"] = "bare_or_built"
        elif nm < 0.25:
            stats["spectral_class"] = "sparse_vegetation"
        elif nm < 0.40:
            stats["spectral_class"] = "moderate_vegetation"
        else:
            stats["spectral_class"] = "dense_vegetation"
        # Override for blue-dominant (pools)
        if blm > 0.38 and nm < 0.05:
            stats["spectral_class"] = "blue_water"
    elif bm > 0:
        # RGB only (no NIR) — classify using brightness + green ratio
        if gm > 0.38 and bm < 120:
            stats["spectral_class"] = "rgb_vegetation"
        elif bm > 140:
            stats["spectral_class"] = "rgb_bright"       # road / roof / snow
        elif bm < 45:
            stats["spectral_class"] = "rgb_dark"          # shadow / water
        else:
            stats["spectral_class"] = "rgb_mixed"

    return stats


# ---------------------------------------------------------------------------
# Phase 3: Classify each segment using aggregated pixel labels + morphometrics
# ---------------------------------------------------------------------------

def classify_objects(
    ndsm: np.ndarray,
    dtm: np.ndarray,
    mask: np.ndarray,
    transform,
    min_height: float = 0.2,
    min_area: int = 1,
    max_objects: int = 50000,
    dsm: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    spectral: dict | None = None,
    temporal_std: np.ndarray | None = None,
    temporal_range: np.ndarray | None = None,
    n_temporal_dates: int = 1,
    return_labels: bool = False,
) -> list[DetectedObject]:
    """Full pipeline: pixel classification → segmentation → object classification.

    Parameters
    ----------
    ndsm, dtm, mask, transform, min_height, min_area, max_objects, dsm
        Existing parameters — unchanged semantics.
    rgb : array [3, H, W] uint8, optional
        RGB orthophoto aligned to the LIDAR grid.
    spectral : dict, optional
        Pre-computed spectral indices from ``ortho_io.compute_spectral_indices``.
        Expected keys: ``ndvi``, ``green_ratio``, ``brightness``, ``rg_index``.
        May also contain ``blue_ratio``.
    temporal_std : array [H, W] float32, optional
        Per-pixel std of nDSM across multiple LIDAR dates.  Low values
        (< 0.5m) indicate temporally stable structures (buildings);
        high values (> 1.5m) indicate vegetation or change.
    temporal_range : array [H, W] float32, optional
        Per-pixel max−min nDSM across dates.  Complements temporal_std.
    n_temporal_dates : int
        Number of LIDAR dates used.  Temporal signals are more reliable
        with 3 dates than 2.
    """
    if dsm is None:
        dsm = dtm + np.where(mask, ndsm, 0)

    # If we have rgb but no spectral dict, derive a minimal brightness layer
    if spectral is None and rgb is not None:
        spectral = _spectral_from_rgb(rgb)

    # Compute scene context for adaptive thresholds
    scene_ctx = _compute_scene_context(ndsm, dtm, mask, spectral) if spectral else SceneContext()
    if temporal_std is not None:
        scene_ctx.has_temporal = True
        scene_ctx.n_dates = n_temporal_dates
        log.info("Multi-temporal stability enabled (%d dates)", n_temporal_dates)

    # Phase 1: pixel-level classification (with optional spectral)
    pixel_classes = _classify_pixels(ndsm, dtm, dsm, mask, spectral=spectral)
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
        _major = getattr(reg, 'axis_major_length', None) or reg.major_axis_length
        _minor = getattr(reg, 'axis_minor_length', None) or reg.minor_axis_length
        elongation = (_major / _minor if _minor > 0 else 999)

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

        # Spectral statistics for this segment
        seg_spectral = _segment_spectral_stats(obj_mask, spectral, rgb)

        # Temporal stability for this segment
        seg_temporal = None
        if temporal_std is not None and scene_ctx.has_temporal:
            t_std_vals = temporal_std[obj_mask]
            t_rng_vals = temporal_range[obj_mask] if temporal_range is not None else t_std_vals * 2
            valid_t = ~np.isnan(t_std_vals)
            if valid_t.any():
                seg_temporal = {
                    "std_mean": float(np.nanmean(t_std_vals)),
                    "std_median": float(np.nanmedian(t_std_vals)),
                    "std_max": float(np.nanmax(t_std_vals)),
                    "range_mean": float(np.nanmean(t_rng_vals)),
                    "range_max": float(np.nanmax(t_rng_vals)),
                    "stable_frac": float(np.sum(t_std_vals[valid_t] < 0.5) / valid_t.sum()),
                    "n_dates": scene_ctx.n_dates,
                }

        obj_type, crown_shape = _classify_segment(
            h_max, h_mean, h_std, h_p90, area, compactness, elongation,
            dominant_class, class_counts, slope_under,
            smooth, obj_mask, int(row), int(col),
            seg_spectral=seg_spectral,
            scene_ctx=scene_ctx,
            seg_temporal=seg_temporal,
        )

        # Skip pure ground unless it's a road or water feature or new ground type
        _keep_ground = {
            "road_path", "water", "parking_lot", "swimming_pool",
            "bare_soil", "rock_cliff",
        }
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
            ndvi_mean=round(seg_spectral["ndvi_mean"], 4) if seg_spectral["ndvi_mean"] is not None else 0.0,
            ndvi_max=round(seg_spectral["ndvi_max"], 4) if seg_spectral["ndvi_max"] is not None else 0.0,
            brightness_mean=round(seg_spectral["brightness_mean"], 2),
            spectral_class=seg_spectral["spectral_class"],
            temporal_std=round(seg_temporal["std_mean"], 3) if seg_temporal else 0.0,
            temporal_stable=(
                seg_temporal is not None
                and seg_temporal["std_mean"] < 0.5
                and seg_temporal["stable_frac"] > 0.7
            ),
            temporal_signal=(
                _temporal_signal_label(seg_temporal, h_max)
                if seg_temporal else ""
            ),
            label=int(reg.label),
        )
        objects.append(obj)

    log.info(f"Classified {len(objects)} objects")
    if return_labels:
        return objects, labels
    return objects


def crown_polygons(
    labels: np.ndarray,
    transform,
    wanted: set[int],
    simplify_m: float = 0.5,
) -> dict[int, "shapely.geometry.base.BaseGeometry"]:
    """Vectorise segment label rasters into crown polygons (EPSG:3035).

    Returns {label: shapely geometry} for labels in *wanted*.
    Polygons are simplified with tolerance *simplify_m* metres.
    """
    from rasterio import features as rio_features
    from shapely.geometry import shape as shp_shape
    from shapely.ops import unary_union

    out: dict[int, list] = {}
    lab32 = labels.astype(np.int32)
    mask = np.isin(lab32, list(wanted))
    for geom, val in rio_features.shapes(lab32, mask=mask, transform=transform):
        out.setdefault(int(val), []).append(shp_shape(geom))
    result = {}
    for lab, parts in out.items():
        g = parts[0] if len(parts) == 1 else unary_union(parts)
        if simplify_m:
            g = g.simplify(simplify_m, preserve_topology=True)
        result[lab] = g
    return result


def _spectral_from_rgb(rgb: np.ndarray) -> dict:
    """Derive minimal spectral indices from an RGB array [3, H, W] uint8.

    Without a true NIR band we cannot compute real NDVI, but we can
    approximate with the Excess-Green index and derive brightness /
    colour ratios that are still very useful.
    """
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        r, g, b = rgb[0], rgb[1], rgb[2]
    elif rgb.ndim == 3 and rgb.shape[2] == 3:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    else:
        return {}

    r_f = r.astype(np.float32)
    g_f = g.astype(np.float32)
    b_f = b.astype(np.float32)

    total = r_f + g_f + b_f + 1e-6  # avoid div-by-zero

    brightness = (r_f + g_f + b_f) / 3.0
    green_ratio = g_f / total
    blue_ratio = b_f / total
    rg_index = (r_f - g_f) / (r_f + g_f + 1e-6)

    # Pseudo-NDVI from Excess Green: 2g - r - b normalised to [-1, 1]
    exg = 2.0 * g_f - r_f - b_f
    ndvi_proxy = np.clip(exg / (total + 1e-6), -1.0, 1.0)

    return {
        "ndvi": ndvi_proxy,
        "brightness": brightness,
        "green_ratio": green_ratio,
        "blue_ratio": blue_ratio,
        "rg_index": rg_index,
    }


def _classify_segment(
    h_max, h_mean, h_std, h_p90, area, compactness, elongation,
    dominant_px_class, class_counts, slope_under,
    smooth_ndsm, obj_mask, center_row, center_col,
    *,
    seg_spectral: dict | None = None,
    scene_ctx: SceneContext | None = None,
    seg_temporal: dict | None = None,
):
    """Classify a segment using pixel-class majority + 3D shape analysis.

    When *seg_spectral* is provided the spectral statistics are used to
    refine / override the geometry-only decision.  *scene_ctx* provides
    scene-adaptive thresholds.  *seg_temporal* provides multi-temporal
    stability metrics from 2–3 LIDAR dates.

    Temporal stability logic:
      - std_mean < 0.5m → temporally stable (almost certainly built structure)
      - std_mean > 1.5m → temporally unstable (vegetation or change)
      - stable_frac > 0.8 → strong building evidence
      - Used to resolve ambiguous spectral/geometry cases
    """
    if scene_ctx is None:
        scene_ctx = SceneContext()

    # Extract temporal signals
    t_stable = False          # strong evidence of built structure
    t_unstable = False        # strong evidence of vegetation/change
    t_signal = ""             # "stable" / "growing" / "shrinking"
    if seg_temporal is not None:
        t_std = seg_temporal["std_mean"]
        t_rng = seg_temporal["range_mean"]
        t_stable_frac = seg_temporal["stable_frac"]
        n_dates = seg_temporal.get("n_dates", 2)

        # Stability thresholds — tighter with 3 dates, looser with 2
        stable_thr = 0.5 if n_dates >= 3 else 0.7
        unstable_thr = 1.5 if n_dates >= 3 else 2.0

        if t_std < stable_thr and t_stable_frac > 0.7:
            t_stable = True
            t_signal = "stable"
        elif t_std > unstable_thr:
            t_unstable = True
            # Determine direction: growing vs shrinking
            # Use range_max to detect if height increased or decreased
            if t_rng > 2.0:
                t_signal = "changing"
            else:
                t_signal = "variable"
    has_spectral = seg_spectral is not None and seg_spectral.get("spectral_class", "") != ""
    ndvi_m = seg_spectral["ndvi_mean"] if has_spectral else None
    ndvi_x = seg_spectral["ndvi_max"] if has_spectral else None
    bri_m = seg_spectral["brightness_mean"] if has_spectral else None
    sp_class = seg_spectral["spectral_class"] if has_spectral else ""
    blue_m = seg_spectral.get("blue_ratio_mean", 0.0) if has_spectral else 0.0

    # --- Ground-level features: use pixel classification directly ---
    if h_max < 0.2:
        name = OBJECT_TYPE_NAMES.get(dominant_px_class, "ground")

        # Spectral refinements for ground-level segments
        if has_spectral:
            # Swimming pool: blue dominant, small, ground level
            if sp_class == "blue_water" and area < 200:
                return "swimming_pool", ""
            # Parking lot: large flat impervious — require >500m²
            if name in ("road_path", "parking_lot") and area > 500:
                if ndvi_m is not None and ndvi_m < scene_ctx.ndvi_building:
                    if bri_m is not None and bri_m > scene_ctx.bri_building:
                        return "parking_lot", ""
            # Rock/cliff at ground level confirmed spectrally
            if name == "rough_ground" and ndvi_m is not None and ndvi_m < 0.08:
                if slope_under > 30:
                    return "rock_cliff", ""
            # Bare soil
            if name in ("meadow_field", "bare_soil"):
                if ndvi_m is not None and ndvi_m < 0.12:
                    return "bare_soil", ""
            # Water at ground level: override to road if bright
            if name == "water" and bri_m is not None and bri_m > 80:
                return "road_path", ""
            # Water: require VERY strict spectral confirmation
            # Calibration: zero actual water in Kohlschwarz test area;
            # all "water" was road/smooth ground.
            if sp_class == "water" and ndvi_m is not None and ndvi_m < -0.05:
                if bri_m is not None and bri_m < 50:
                    return "water", ""
                # Bright "water" is actually road or smooth surface
                return "road_path", ""

        return name, ""

    # --- Low vegetation: 0.2–2m ---
    if h_max < 2.0:
        # Low non-green objects: small built structures
        if has_spectral and ndvi_m is not None:
            if (ndvi_m < scene_ctx.ndvi_building - 0.05
                    and bri_m is not None and bri_m > scene_ctx.bri_building + 10
                    and area > 30 and compactness > 0.3):
                return "structure", ""
        return "low_vegetation", ""

    # --- 2–4m ---
    if h_max < 4.0:
        if elongation > 6 and area < 300:
            # Hedge: narrow linear vegetation
            if has_spectral and ndvi_m is not None and ndvi_m > 0.20:
                return "hedge", ""
            return "wall_fence", ""
        # Hedge: linear green feature 1-4m
        if elongation > 4 and has_spectral and ndvi_m is not None and ndvi_m > 0.25:
            return "hedge", ""
        # 2-4m non-green objects: built structures (walls, carports)
        if has_spectral and ndvi_m is not None:
            if (ndvi_m < scene_ctx.ndvi_building - 0.05
                    and bri_m is not None and bri_m > scene_ctx.bri_building):
                return "structure", ""
        return "shrub_bush", ""

    # --- Elevated objects (≥4m) ---

    # Power line: extremely thin, tall, elongated
    if area < 10 and h_max > 6 and elongation > 10:
        return "power_line", ""

    # Mast/pole: tiny footprint, very tall, non-vegetated
    # Calibration: 10,777 FPs — mostly building/tree edge fragments.
    # Drastically tightened: area < 8, height > 15, compact, non-green.
    if area < 8 and h_max > 15 and compactness > 0.4:
        if has_spectral and ndvi_m is not None:
            if ndvi_m < scene_ctx.ndvi_building - 0.05 and (bri_m is None or bri_m > 80):
                return "mast_pole", ""
        elif not has_spectral and compactness > 0.6 and area < 5:
            return "mast_pole", ""

    # Check pixel-class composition
    total_px = max(class_counts.sum(), 1)
    building_frac = class_counts[OBJECT_TYPES["building"]] / total_px
    # Also count solar panel pixels as part of building footprint
    solar_frac = class_counts[OBJECT_TYPES["solar_panel"]] / total_px if len(class_counts) > OBJECT_TYPES["solar_panel"] else 0
    greenhouse_frac = class_counts[OBJECT_TYPES["greenhouse"]] / total_px if len(class_counts) > OBJECT_TYPES["greenhouse"] else 0

    # --- Greenhouse: segment-level only (pixel-level disabled).
    #    Extremely strict: real greenhouses are rectangular, 3-6m, rural.
    if (has_spectral and ndvi_m is not None and ndvi_m < 0.05
            and bri_m is not None and bri_m > 150
            and 2.0 < h_max < 8.0 and 50 < area < 2000
            and compactness > 0.35 and elongation < 6
            and slope_under < 5 and h_std < 1.5):
        return "greenhouse", ""

    # --- Solar panel: dark patches on building roof ---
    if solar_frac > 0.5 and building_frac + solar_frac > 0.6:
        if has_spectral and bri_m is not None and bri_m < 55:
            if ndvi_m is not None and ndvi_m < -0.05:
                return "solar_panel", ""

    # --- TEMPORAL STABILITY: strong prior for building detection ---
    # Objects stable across 2-3 years of LIDAR are almost certainly built.
    # This resolves the biggest error mode: winter/alpine scenes where
    # spectral signals are unreliable.
    if t_stable and h_max >= 2 and area >= 15 and slope_under < 30:
        # Temporally stable elevated object on reasonable terrain → building
        # Still respect clear vegetation signal if available
        if has_spectral and ndvi_m is not None and ndvi_m > scene_ctx.ndvi_tree_recovery + 0.10:
            pass  # very green = definitely tree despite stability
        else:
            if area >= 25:
                return "building", "temporal_stable"
            return "structure", "temporal_stable"

    # --- Building: majority of pixels classified as building ---
    # No height cap!  Austrian cities have buildings up to 80m.
    # Instead, use scene-adaptive spectral + height-dependent confidence.
    if building_frac + solar_frac > 0.5 and area >= 15:
        height_ok = False
        if h_max < 25:
            # Standard buildings: just check terrain
            height_ok = (slope_under < 30)
        elif h_max < 80:
            # Tall buildings: require clear non-vegetation spectral signal
            if has_spectral and ndvi_m is not None:
                height_ok = (ndvi_m < scene_ctx.ndvi_building) and (slope_under < 25)
            elif t_stable:
                # Temporal stability compensates for missing/weak spectral
                height_ok = (slope_under < 25)
            else:
                height_ok = (compactness > 0.30) and (slope_under < 20)
        if height_ok:
            # Spectral override: if green (above tree recovery), it's canopy
            # But temporal stability overrides spectral for ambiguous cases
            if has_spectral and ndvi_m is not None and ndvi_m > scene_ctx.ndvi_tree_recovery:
                if t_stable:
                    # Stable despite green spectral — could be moss/lichen on roof
                    # or winter-to-summer seasonal variation.  Trust temporal.
                    if area >= 25:
                        return "building", "temporal_override"
                    return "structure", "temporal_override"
                pass  # fall through to tree classification
            else:
                if area >= 25:
                    return "building", ""
                return "structure", ""

    # Building with shaped (gabled/hipped) roof:
    # Height cap raised to 50m.  Scene-adaptive NDVI thresholds.
    if (slope_under < 25 and h_std < 3.5 and 15 < area < 3000
            and compactness > 0.15 and elongation < 8 and h_max < 50):
        if has_spectral and ndvi_m is not None and ndvi_m > scene_ctx.ndvi_tree_recovery:
            if t_stable:
                return "building", "temporal_override"
            pass  # fall through to tree
        elif has_spectral and ndvi_m is not None and ndvi_m < scene_ctx.ndvi_building:
            if bri_m is None or bri_m > scene_ctx.bri_building:
                return "building", ""
        elif not has_spectral:
            if t_stable or h_max < 25:
                return "building", "temporal_stable" if t_stable else ""

    # --- Bridge: elevated linear structure over depression/water ---
    if (elongation > 5 and h_max >= 4 and h_max < 20
            and compactness < 0.25 and area > 30):
        if has_spectral and ndvi_m is not None and ndvi_m < 0.10:
            return "bridge", ""
        # Without spectral, check if building-like but very elongated on low slope
        if slope_under < 10 and building_frac > 0.3 and elongation > 8:
            return "bridge", ""

    # --- Dead tree: tall, spectrally non-green, AND dark ---
    # Scene-adaptive: must be below scene dead thresholds AND rough.
    # Temporal stability: if stable across 2-3 years, it's NOT a dead tree
    # (dead trees fall/decompose within ~2 years; stable dark objects are
    # building parts, shadow artefacts, or dark roofs).
    if has_spectral and ndvi_m is not None and ndvi_m < scene_ctx.ndvi_dead_max and h_max >= 4:
        if (bri_m is not None and bri_m < scene_ctx.bri_dead_max
                and building_frac < 0.2 and h_max < 25
                and compactness < 0.6
                and not t_stable):  # stable objects are NOT dead trees
            return "dead_tree", "dead"

    # --- Tree row: linear arrangement of trees ---
    if elongation > 5 and area > 50 and h_max >= 4:
        if has_spectral and ndvi_m is not None and ndvi_m > 0.25:
            return "tree_row", "linear"
        elif not has_spectral:
            # Without spectral, use geometry: elongated canopy
            tree_frac = sum(
                class_counts[c] for c in (
                    OBJECT_TYPES["tree_coniferous"],
                    OBJECT_TYPES["tree_broadleaf"],
                    OBJECT_TYPES["tree_unclassified"],
                ) if c < len(class_counts)
            ) / total_px
            if tree_frac > 0.5:
                return "tree_row", "linear"

    # --- Vineyard/orchard: moderate height, regular pattern, moderate NDVI ---
    if has_spectral and ndvi_m is not None:
        if (0.15 < ndvi_m < 0.40 and h_max < 8 and area > 100
                and h_std > 0.5 and compactness > 0.2):
            return "vineyard_orchard", ""

    # --- Last-resort temporal check ---
    # If we've reached here (not matched as building/greenhouse/bridge/etc)
    # but the object is temporally stable + compact on low slope, it's likely
    # a building that didn't match the standard building path (e.g. unusual
    # pixel class mix, or spectral ambiguity in winter/alpine scenes).
    if (t_stable and h_max >= 2 and area >= 15
            and compactness > 0.15 and slope_under < 25
            and elongation < 10):
        # Only override if not clearly vegetated
        if not (has_spectral and ndvi_m is not None and ndvi_m > scene_ctx.ndvi_tree_recovery + 0.10):
            if area >= 25:
                return "building", "temporal_fallback"
            return "structure", "temporal_fallback"

    # --- Tree crown shape analysis ---
    crown_shape = _analyse_crown_shape(smooth_ndsm, obj_mask, center_row, center_col)

    p90_ratio = h_p90 / max(h_max, 0.1)
    peak_ratio = h_max / max(h_mean, 0.1)

    # Spectral can help distinguish coniferous (darker, lower NDVI in winter)
    # from broadleaf (brighter, higher NDVI in summer)
    spectral_tree_hint = ""
    if has_spectral and ndvi_m is not None and ndvi_m > 0.20:
        if ndvi_m > 0.40:
            spectral_tree_hint = "broadleaf"  # very green = broadleaf in summer
        elif ndvi_m < 0.28 and bri_m is not None and bri_m < 80:
            spectral_tree_hint = "coniferous"  # dark + lower green = conifer

    if crown_shape == "conical":
        return "tree_coniferous", "conical"
    elif crown_shape == "dome":
        return "tree_broadleaf", "rounded"
    elif crown_shape == "columnar":
        return "tree_broadleaf", "columnar"
    else:
        # Fallback heuristics — use spectral hint when available
        if spectral_tree_hint == "coniferous":
            return "tree_coniferous", "spectral"
        if spectral_tree_hint == "broadleaf":
            return "tree_broadleaf", "spectral"

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

        if "tree" in t or t == "dead_tree":
            tree_heights.append(obj.height_max)

    # Aggregate spectral stats per type
    spectral_by_type: dict[str, list[float]] = {}
    for obj in objects:
        if obj.ndvi_mean != 0.0:  # has spectral data
            spectral_by_type.setdefault(obj.obj_type, []).append(obj.ndvi_mean)

    for t, info in by_type.items():
        heights = info.pop("heights")
        info["total_area_sqm"] = round(info["total_area_sqm"], 1)
        info["height_mean"] = round(float(np.mean(heights)), 2)
        info["height_std"] = round(float(np.std(heights)), 2) if len(heights) > 1 else 0
        info["height_max"] = round(info["height_max"], 2)
        if t in spectral_by_type:
            ndvis = spectral_by_type[t]
            info["ndvi_mean"] = round(float(np.mean(ndvis)), 4)

    crown_types: dict[str, int] = {}
    for obj in objects:
        if ("tree" in obj.obj_type or obj.obj_type == "dead_tree") and obj.crown_shape:
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
    spectral: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Create 2-band raster: band 1 = type code, band 2 = height."""
    h, w = ndsm.shape

    # Pixel-level classification for base layer
    pixel_classes = _classify_pixels(ndsm, dtm, dsm, mask, spectral=spectral)
    type_band = pixel_classes.copy()
    height_band = np.where(mask, ndsm, -9999).astype(np.float32)

    # Override with segment-level classification where we have objects
    labels = _segment_objects(ndsm, mask, pixel_classes, 0.2)
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
