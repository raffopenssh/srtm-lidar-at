"""Watershed-based object segmentation and classification.

Philosophy: segment first, classify per-object.

  1. Fused gradient from DTM+DSM+nDSM+RGBI+NDVI (weighted per reference)
  2. Felzenszwalb over-segmentation on composite edge image
  3. RAG boundary merge to join weak-boundary neighbours
  4. Per-object feature vector computed ONCE per segment (fast!)
  5. Decision-tree classification with clear rules
  6. Hierarchical grouping: tree→forest, roof→building, etc.
  7. Cadastre calibration for building thresholds

References:
  - Copernicus Parcel Delineation: Sobel → Felzenszwalb → RAG merge
  - JRC NRT: EWMA/CuSum harmonic monitoring on per-segment time series
  - EuroSAT: 10-class Sentinel-2 LULC benchmark (Helber et al. 2019)
  - See docs/reference_algorithms_summary.md for full detail
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter, uniform_filter, label as ndi_label,
    maximum_filter, minimum_filter, distance_transform_edt,
)
from skimage import measure, morphology, graph
from skimage.segmentation import watershed, felzenszwalb
from skimage.filters import sobel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Object types — hierarchical: individual + group level
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Individual object types — only what our sensors can actually detect
#
# Detection basis per type:
#   DTM/DSM 1m : height, slope, roughness, shape, temporal change
#   RGBI 0.2m  : NDVI, brightness, NIR
#   Sentinel-2 : growing-season NDVI, ESA WorldCover prior
#
# Cadastre code references in comments for cross-validation.
# ---------------------------------------------------------------------------
OBJECT_TYPES = {
    # ---- Vegetation (NDVI + height + roughness) ----
    "tree": 1,           # nDSM>4m, rough DSM, high NDVI           [W 56]
    "shrub": 2,          # nDSM 0.5-4m, high NDVI                  [W(Kr) 57]
    "grass": 3,          # ground, moderate+ NDVI, smooth DTM      [LN 52-55,61]
    "hedge": 4,          # elongated shrub (length/width>4)        [linear veg]
    # ---- Water (very low NDVI + low NIR + flat) ----
    "water": 5,          # ESA water class, very low NDVI+NIR      [GW 70,71,96]
    # ---- Infrastructure: buildings (elevated+smooth+low NDVI) ----
    "roof": 10,          # compact elevated, smooth DSM, low NDVI  [B(Geb) 42]
    "greenhouse": 11,    # roof-like but high NIR transmittance     [B(Gwh) 45]
    "solar_panel": 12,   # very smooth, bright, low NDVI on roof   [detectable]
    # ---- Infrastructure: other (elevated + distinct shape) ----
    "fence": 15,         # low (0.5-2m), thin, elongated            [detectable]
    "wall": 16,          # narrow elevated, adjacent to roof        [detectable]
    "mast": 17,          # tiny footprint (<10m²), very tall (>15m) [detectable]
    # ---- Transportation (smooth DTM + elongated + low NDVI) ----
    "road": 20,          # smooth, elongated, low NDVI             [V 48,73]
    "path": 21,          # narrower road (<3m effective width)      [V(Weg) 74]
    "parking": 22,       # smooth, large, compact, low NDVI        [B(bf) 41]
    "bridge": 23,        # elevated road/path over gap              [Br 75]
    # ---- Agricultural (ground + NDVI + ESA prior) ----
    "crop": 30,          # flat, seasonal NDVI, ESA cropland        [A 51,62]
    "orchard": 31,       # regular tree spacing, <10m height        [OG 65]
    "vineyard": 32,      # low rows, <3m, row pattern in ortho      [WG 63]
    "garden": 33,        # mixed veg near buildings                 [GA 64]
    # ---- Terrain (slope + roughness + low NDVI) ----
    "bare_soil": 40,     # low NDVI, flat-moderate slope            [Öd 59,90]
    "rock": 41,          # steep + very rough DTM + low NDVI        [Fe 83,84]
    # ---- Disturbance (DTM temporal change) ----
    "excavation": 50,    # DTM lowered between dates                [Ab 80,93]
    "fill": 51,          # DTM raised between dates                 [Dep 81]
    "tree_loss": 52,     # logging/timber harvest: was tall, now ground, terrain intact
    "construction": 53,  # new structure, or site clearing (tree_loss + earthworks)
}

# Group types (merge adjacent compatible individuals)
GROUP_TYPES = {
    # Vegetation
    "forest": 101,           # tree+tree+shrub  [W 56]
    "woodland": 102,         # sparse trees+shrubs  [W(Kr) 57]
    "hedgerow": 103,         # hedge+hedge  [linear]
    # Water
    "waterbody": 106,        # water+water  [GW 70,71,96]
    # Infrastructure
    "building": 110,         # roof+wall+solar_panel+greenhouse  [B 42-47]
    "road_network": 115,     # road+path+parking  [V 48,73,74]
    # Agricultural
    "cropland": 120,         # crop+crop  [A 51,62]
    "pasture": 121,          # grass+grass (large)  [LN 52-55]
    "orchard_grove": 122,    # orchard+vineyard  [OG 65, WG 63]
    # Disturbance
    "quarry": 130,           # excavation+fill  [Ab 80,93]
    "construction_site": 131,# construction+excavation+fill
}

ALL_TYPE_NAMES = {v: k for k, v in {**OBJECT_TYPES, **GROUP_TYPES}.items()}

# Map individual types → old 10-type landscape codes (backward compat)
_COMPAT_MAP = {
    # Vegetation
    "tree": 7, "shrub": 8, "grass": 8, "hedge": 8,
    # Water
    "water": 9,
    # Buildings
    "roof": 5, "greenhouse": 5, "solar_panel": 5, "wall": 6,
    # Infrastructure
    "fence": 6, "mast": 6,
    # Transportation
    "road": 1, "path": 1, "parking": 1, "bridge": 6,
    # Agricultural
    "crop": 8, "orchard": 7, "vineyard": 8, "garden": 8,
    # Terrain
    "bare_soil": 9, "rock": 9,
    # Disturbance
    "excavation": 3, "fill": 4, "tree_loss": 10, "construction": 10,
}

# Map cadastre land-use codes → our detectable types (for cross-validation)
CADASTRE_TO_TYPE = {
    40: "garden", 41: "parking", 42: "roof", 43: "roof", 44: "roof",
    45: "greenhouse", 46: "roof", 47: "roof",
    48: "road", 49: "road",
    50: "grass", 51: "crop", 52: "grass", 53: "grass", 54: "grass",
    55: "grass", 56: "tree", 57: "shrub", 58: "grass",
    59: "bare_soil", 60: "water", 61: "grass", 62: "crop",
    63: "vineyard", 64: "garden", 65: "orchard", 66: "garden",
    67: "crop",
    70: "water", 71: "water", 72: "water",
    73: "road", 74: "path", 75: "bridge", 76: "road",
    77: "grass", 78: "grass", 79: "grass",  # cemetery/sports/park → grass at 1m
    80: "excavation", 81: "fill", 82: "water",
    83: "rock", 84: "rock", 85: "rock",
    86: "road", 87: "road", 88: "road",
    90: "bare_soil", 91: "bare_soil", 92: "grass",
    93: "excavation", 94: "excavation", 95: "bare_soil",
    96: "water", 97: "road",
}


# ---------------------------------------------------------------------------
# SegmentedObject — the output unit
# ---------------------------------------------------------------------------

@dataclass
class SegmentedObject:
    """A watershed-segmented landscape object."""
    obj_id: int
    obj_type: str
    type_code: int
    # Geometry
    area_sqm: float
    perimeter_m: float
    compactness: float  # isoperimetric quotient: 4π·area/perimeter²
    elongation: float   # major_axis / minor_axis
    centroid_e: float   # EPSG:3035 easting
    centroid_n: float   # EPSG:3035 northing
    bbox: tuple[float, float, float, float]  # (min_e, min_n, max_e, max_n)
    # Height
    height_max: float
    height_mean: float
    height_p90: float
    height_std: float
    # Surface
    slope_mean: float
    roughness: float  # DSM local std within segment
    # Spectral (from ortho or Sentinel-2)
    ndvi_mean: float = 0.0
    ndvi_std: float = 0.0
    ndvi_fused: float = 0.0   # seasonal-corrected (BEV 1m + Cop growing-season)
    brightness_mean: float = 0.0
    nir_mean: float = 0.0
    # Temporal
    height_change: float = 0.0  # nDSM change across dates
    dtm_change: float = 0.0    # terrain change (machinery)
    temporal_stability: float = 1.0  # 0=volatile, 1=stable
    # Volume change (DTM-based, m³ at 1m resolution)
    volume_change_m3: float = 0.0  # net: +fill, -cut
    volume_change_abs_m3: float = 0.0  # total disturbed volume
    dtm_change_max: float = 0.0  # max absolute terrain shift in segment
    # Shape / boundary
    solidity: float = 0.0  # area / convex_hull_area
    extent: float = 0.0    # area / bbox_area
    dsm_edge_strength: float = 0.0  # mean DSM gradient at segment boundary
    # Classification
    confidence: float = 0.5
    is_manmade: bool = False
    # Group membership
    group_id: int = 0
    group_type: str = ""
    # Texture (GLCM from ortho)
    glcm_entropy: float = 0.0
    glcm_homogeneity: float = 0.0
    texture_complexity: float = 0.0
    # SAR backscatter
    sar_vv: float = 0.0
    sar_vh: float = 0.0
    # NDVI harmonics (phenology)
    harm_amplitude: float = 0.0
    harm_phase: float = 0.0
    phenology_class: str = ""
    # Features vector (for debugging / calibration)
    features: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_std(arr: np.ndarray, size: int = 3) -> np.ndarray:
    """Fast local standard deviation via uniform filter."""
    a = arr.astype(np.float64)
    m1 = uniform_filter(a, size=size, mode="nearest")
    m2 = uniform_filter(a * a, size=size, mode="nearest")
    return np.sqrt(np.clip(m2 - m1 * m1, 0, None)).astype(np.float32)


def _slope(arr: np.ndarray, res: float = 1.0) -> np.ndarray:
    """Slope in degrees."""
    dy, dx = np.gradient(arr, res)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2))).astype(np.float32)


# ===================================================================
# 1. FUSED GRADIENT COMPUTATION
#    Weights from reference: CHM 0.25, DTM 0.20, NDVI 0.20,
#    NIR 0.15, G 0.10, R 0.05, B 0.05
# ===================================================================

def compute_fused_gradient(
    dtm: np.ndarray,
    dsm: np.ndarray,
    ndsm: np.ndarray,
    mask: np.ndarray,
    *,
    spectral: dict | None = None,
    sigma: float = 0.8,
) -> np.ndarray:
    """Compute fused edge gradient from all available data layers.

    Uses Sobel on each layer (as per Parcel Delineation pipeline),
    then weighted sum.  Object boundaries are strong gradient ridges
    in ANY layer.

    Returns float32 gradient image in [0, 1].
    """
    h, w = dtm.shape
    gradients = []
    weights = []

    def _add_layer(arr, weight, name, sig=sigma):
        """Smooth, Sobel, mask, normalize, add."""
        s = gaussian_filter(np.nan_to_num(arr, 0).astype(np.float64), sigma=sig)
        g = sobel(s)
        g = np.where(mask, g, 0)
        gmax = np.nanmax(g[mask]) if mask.any() else 1.0
        if gmax > 1e-10:
            g = g / gmax
        gradients.append(g.astype(np.float32))
        weights.append(weight)

    # Canopy Height Model (nDSM) — strongest structural boundary
    _add_layer(ndsm, 0.25, "CHM")
    # DTM — terrain breaks (roads, embankments, ditches)
    _add_layer(dtm, 0.20, "DTM")
    # DSM — surface breaks (building edges, canopy edges)
    _add_layer(dsm, 0.10, "DSM")

    # Spectral layers (if available)
    if spectral is not None:
        for key, weight, sig in [
            ("ndvi", 0.20, sigma),
            ("nir", 0.15, sigma),
            ("green", 0.05, sigma / 2),
            ("red", 0.03, sigma / 2),
            ("blue", 0.02, sigma / 2),
        ]:
            arr = spectral.get(key)
            if arr is not None:
                arr = np.asarray(arr, dtype=np.float64)
                if arr.shape == (h, w):
                    _add_layer(arr, weight, key, sig=sig)

        # Brightness as fallback when individual bands missing
        if spectral.get("brightness") is not None and spectral.get("red") is None:
            arr = np.asarray(spectral["brightness"], dtype=np.float64)
            if arr.shape == (h, w):
                _add_layer(arr, 0.10, "brightness", sig=sigma / 2)

    # Weighted sum
    fused = np.zeros((h, w), dtype=np.float64)
    total_w = 0.0
    for g, wt in zip(gradients, weights):
        fused += g * wt
        total_w += wt
    if total_w > 0:
        fused /= total_w

    # Normalize to [0, 1]
    fmax = np.nanmax(fused[mask]) if mask.any() else 1.0
    if fmax > 1e-10:
        fused /= fmax

    return np.where(mask, fused, 1.0).astype(np.float32)


# ===================================================================
# 2. SEGMENTATION: Felzenszwalb → RAG boundary merge
#    Two-layer approach: ground & elevated segmented separately
# ===================================================================

def segment_landscape(
    gradient: np.ndarray,
    ndsm: np.ndarray,
    mask: np.ndarray,
    *,
    felz_scale: float = 150.0,
    felz_sigma: float = 0.5,
    felz_min_size: int = 30,
    rag_threshold: float = 0.12,
) -> np.ndarray:
    """Two-layer Felzenszwalb + RAG merge segmentation.

    Separates ground (nDSM < 0.3m) and elevated (>= 0.3m) to prevent
    tree crowns merging with adjacent ground.

    Parameters match reference guidance:
      felz_scale: 100-200 (higher = fewer, larger segments)
      felz_min_size: 30-100 px
      rag_threshold: 0.10-0.20

    Returns int32 label array (0 = background/invalid).
    """
    h, w = gradient.shape
    labels_out = np.zeros((h, w), dtype=np.int32)
    max_label = 0

    for layer_name, layer_mask, scale_mult, min_mult in [
        ("elevated", mask & (ndsm >= 0.3), 1.0, 1.0),
        ("ground", mask & (ndsm < 0.3), 1.2, 2.0),  # coarser for ground
    ]:
        n_px = int(layer_mask.sum())
        if n_px < felz_min_size:
            continue

        layer_labels = _segment_layer(
            gradient, layer_mask,
            felz_scale=felz_scale * scale_mult,
            felz_sigma=felz_sigma,
            felz_min_size=int(felz_min_size * min_mult),
            rag_threshold=rag_threshold,
        )

        valid = layer_labels > 0
        labels_out[valid] = layer_labels[valid] + max_label
        if valid.any():
            max_label = int(labels_out.max())

        n_seg = len(np.unique(layer_labels[valid]))
        log.info("  %s layer: %d px → %d segments", layer_name, n_px, n_seg)

    return labels_out


def _segment_layer(
    gradient: np.ndarray,
    layer_mask: np.ndarray,
    *,
    felz_scale: float,
    felz_sigma: float,
    felz_min_size: int,
    rag_threshold: float,
) -> np.ndarray:
    """Felzenszwalb → RAG merge on a single layer."""
    h, w = gradient.shape

    # Prepare input: gradient within mask, 1.0 outside
    grad_in = np.where(layer_mask, gradient, 1.0).astype(np.float64)

    # Step 1: Felzenszwalb over-segmentation
    segments = felzenszwalb(
        grad_in,
        scale=felz_scale,
        sigma=felz_sigma,
        min_size=felz_min_size,
        channel_axis=None,
    )

    # Zero out segments outside mask
    segments = np.where(layer_mask, segments + 1, 0)  # +1 so label 0 = background

    # Step 2: RAG boundary merge (join segments with weak boundaries)
    if len(np.unique(segments[segments > 0])) > 1:
        try:
            edges = sobel(grad_in)
            bgraph = graph.rag_boundary(segments, edges)
            segments = graph.cut_threshold(segments, bgraph, threshold=rag_threshold)
        except Exception as e:
            log.debug("RAG merge skipped: %s", e)

    # Re-zero background
    segments = np.where(layer_mask, segments, 0)

    return segments.astype(np.int32)


# ===================================================================
# 3. PER-OBJECT FEATURE EXTRACTION
# ===================================================================

def extract_object_features(
    labels: np.ndarray,
    dtm: np.ndarray,
    dsm: np.ndarray,
    ndsm: np.ndarray,
    mask: np.ndarray,
    transform,
    *,
    spectral: dict | None = None,
    cop: dict | None = None,
    dtm_dates: dict | None = None,
    dsm_dates: dict | None = None,
    hansen: dict | None = None,
    observation_year: int | None = None,
) -> list[dict]:
    """Extract a feature vector for each labelled segment.

    Parameters
    ----------
    observation_year : int or None
        Calendar year of the primary LIDAR/ortho observation (e.g. 2024).
        Used to scope Hansen "recent loss" to years *before* the observation.
        If None, defaults to 2024.
    """
    _obs_year = observation_year or 2024
    # Hansen lossyear is 1-24 → 2001-2024.  "Recent" = within 5 years
    # *before* the observation, i.e. [obs_year-5, obs_year] mapped to
    # Hansen codes [obs_year-2000-5, obs_year-2000].
    _hansen_recent_start = max(_obs_year - 2000 - 5, 1)   # e.g. 2024→19
    _hansen_recent_end   = min(_obs_year - 2000, 24)       # e.g. 2024→24
    h, w = dtm.shape
    slope_arr = _slope(dtm)
    dsm_rough = _local_std(dsm, 3)
    dtm_rough = _local_std(dtm, 3)

    # DSM edge magnitude (Sobel gradient) — precompute once for boundary sharpness
    dsm_grad_x = ndimage.sobel(dsm, axis=1)
    dsm_grad_y = ndimage.sobel(dsm, axis=0)
    dsm_edge_mag = np.sqrt(dsm_grad_x**2 + dsm_grad_y**2)

    # --- Pre-compute temporal data ---
    temporal_ndsm_std = None
    temporal_ndsm_change = None
    temporal_dtm_change = None
    if dtm_dates and dsm_dates and len(dtm_dates) >= 2:
        dates = sorted(dtm_dates.keys())
        ndsm_stack = []
        for d in dates:
            dd = dtm_dates[d]
            ds = dsm_dates[d]
            mh = min(dd.shape[0], ds.shape[0], h)
            mw = min(dd.shape[1], ds.shape[1], w)
            nd = np.clip(ds[:mh, :mw] - dd[:mh, :mw], 0, None)
            full = np.full((h, w), np.nan, dtype=np.float32)
            full[:mh, :mw] = nd
            ndsm_stack.append(full)

        stack = np.stack(ndsm_stack, axis=0)
        with np.errstate(all="ignore"):
            temporal_ndsm_std = np.nanstd(stack, axis=0)
            temporal_ndsm_change = np.where(
                np.isfinite(stack[0]) & np.isfinite(stack[-1]),
                stack[-1] - stack[0], 0,
            )

        d_first = dtm_dates[dates[0]]
        d_last = dtm_dates[dates[-1]]
        mh2 = min(d_first.shape[0], d_last.shape[0], h)
        mw2 = min(d_first.shape[1], d_last.shape[1], w)
        dtm_diff = np.zeros((h, w), dtype=np.float32)
        dtm_diff[:mh2, :mw2] = d_last[:mh2, :mw2] - d_first[:mh2, :mw2]
        temporal_dtm_change = dtm_diff

    # --- Spectral arrays (BEV ortho 1m) ---
    bev_ndvi = _get_spectral_layer(spectral, "ndvi", h, w)
    bev_brightness = _get_spectral_layer(spectral, "brightness", h, w)
    bev_nir = _get_spectral_layer(spectral, "nir", h, w)
    bev_red = _get_spectral_layer(spectral, "red", h, w)
    bev_green = _get_spectral_layer(spectral, "green", h, w)
    bev_green_ratio = _get_spectral_layer(spectral, "green_ratio", h, w)
    bev_rg_index = _get_spectral_layer(spectral, "rg_index", h, w)

    # --- Copernicus (resampled to 1m) ---
    cop_ndvi = _get_spectral_layer(cop, "ndvi", h, w) if cop else None

    # --- Fused NDVI: seasonal-corrected 1m ---
    # BEV ortho has 1m spatial detail but arbitrary capture date (could be
    # winter → all veg looks dead).  Copernicus growing-season median is the
    # seasonal truth but at 10m (bleeds across boundaries).
    #
    # Strategy: keep BEV spatial *contrast* (road edge vs grass), shift
    # absolute level toward Copernicus.  Where BEV is unavailable, use
    # Copernicus alone.
    fused_ndvi = _fuse_ndvi(bev_ndvi, cop_ndvi, mask)
    cop_lc = None
    if cop and cop.get("landcover") is not None:
        arr = np.asarray(cop["landcover"], dtype=np.uint8)
        if arr.shape == (h, w):
            cop_lc = arr

    # --- SAR backscatter (resampled to 1m) ---
    sar_vv = _get_spectral_layer(cop, "vv", h, w) if cop else None
    sar_vh = _get_spectral_layer(cop, "vh", h, w) if cop else None

    # --- NDVI harmonic parameters (resampled to 1m) ---
    harm = cop.get("harmonics") if cop else None
    harm_mean = _get_spectral_layer(harm, "h_mean", h, w) if harm else None
    harm_amp = _get_spectral_layer(harm, "h_amplitude", h, w) if harm else None
    harm_phase = _get_spectral_layer(harm, "h_phase", h, w) if harm else None
    harm_rmse = _get_spectral_layer(harm, "h_rmse", h, w) if harm else None

    # --- Hansen Global Forest Change (resampled to 1m) ---
    hansen_tc = hansen.get("treecover2000") if hansen else None
    hansen_loss = hansen.get("loss_year") if hansen else None
    hansen_gain = hansen.get("gain") if hansen else None
    hansen_current = hansen.get("current_forest") if hansen else None

    # --- Extract per-region ---
    regions = measure.regionprops(labels, intensity_image=ndsm)
    objects = []

    for reg in regions:
        if reg.area < 2:
            continue
        seg = labels == reg.label
        seg_v = seg & mask
        if seg_v.sum() < 2:
            continue

        f = {"label": reg.label, "area": int(reg.area)}

        # Height
        sv = ndsm[seg_v]
        f["h_max"] = float(np.nanmax(sv))
        f["h_mean"] = float(np.nanmean(sv))
        f["h_p90"] = float(np.nanpercentile(sv, 90))
        f["h_p10"] = float(np.nanpercentile(sv, 10))
        f["h_std"] = float(np.nanstd(sv))
        f["is_elevated"] = f["h_mean"] > 0.5
        f["is_ground"] = f["h_mean"] < 0.3

        # Slope / terrain
        ss = slope_arr[seg_v]
        f["slope_mean"] = float(np.nanmean(ss))
        f["slope_max"] = float(np.nanmax(ss))
        f["slope_std"] = float(np.nanstd(ss))

        # Roughness
        f["dsm_roughness"] = float(np.nanmean(dsm_rough[seg_v]))
        f["dtm_roughness"] = float(np.nanmean(dtm_rough[seg_v]))

        # Shape
        f["compactness"] = (4 * np.pi * reg.area) / (reg.perimeter ** 2 + 1e-6)
        f["perimeter"] = float(reg.perimeter)
        if hasattr(reg, 'axis_major_length') and hasattr(reg, 'axis_minor_length'):
            minor = max(reg.axis_minor_length, 1.0)
            f["elongation"] = reg.axis_major_length / minor
        else:
            bb = reg.bbox
            rs, cs = bb[2] - bb[0], bb[3] - bb[1]
            f["elongation"] = max(rs, cs) / max(min(rs, cs), 1)

        # Solidity and extent (rectangularity indicators)
        f["solidity"] = float(reg.solidity) if hasattr(reg, 'solidity') else 0.0
        f["extent"] = float(reg.extent) if hasattr(reg, 'extent') else 0.0

        # DSM edge strength: sharp elevation drops at segment boundary
        boundary = ndimage.binary_dilation(seg, iterations=1) & ~seg & mask
        if boundary.sum() > 0:
            f["dsm_edge_strength"] = float(np.nanmean(dsm_edge_mag[boundary]))
        else:
            f["dsm_edge_strength"] = 0.0

        # Centroid in map coords
        cr, cc = reg.centroid
        f["centroid_e"] = transform.c + cc * transform.a
        f["centroid_n"] = transform.f + cr * transform.e
        bb = reg.bbox
        f["bbox"] = (
            round(transform.c + bb[1] * transform.a, 1),
            round(transform.f + bb[2] * transform.e, 1),
            round(transform.c + bb[3] * transform.a, 1),
            round(transform.f + bb[0] * transform.e, 1),
        )

        # Spectral
        f["ndvi_mean"], f["ndvi_std"], f["ndvi_max"] = _seg_stats(bev_ndvi, seg_v)
        f["brightness_mean"] = _seg_mean(bev_brightness, seg_v)
        f["nir_mean"] = _seg_mean(bev_nir, seg_v)
        f["red_mean"] = _seg_mean(bev_red, seg_v)
        f["green_mean"] = _seg_mean(bev_green, seg_v)
        f["green_ratio"] = _seg_mean(bev_green_ratio, seg_v)
        f["rg_index"] = _seg_mean(bev_rg_index, seg_v)
        # NIR brightness ratio: coniferous trees reflect less NIR per unit
        # brightness than deciduous; useful for tree type discrimination.
        nir_v = f["nir_mean"]
        bri_v = f["brightness_mean"]
        f["nir_brightness_ratio"] = nir_v / max(bri_v, 1.0) if nir_v > 0 else 0.0
        f["cop_ndvi_mean"] = _seg_mean(cop_ndvi, seg_v)
        # Fused NDVI: seasonal-corrected 1m (best of both worlds)
        f["fused_ndvi_mean"], f["fused_ndvi_std"], _ = _seg_stats(fused_ndvi, seg_v)

        # SAR backscatter
        f["sar_vv"] = _seg_mean(sar_vv, seg_v)
        f["sar_vh"] = _seg_mean(sar_vh, seg_v)
        vv_val = f["sar_vv"]
        vh_val = f["sar_vh"]
        f["sar_ratio"] = vv_val / max(vh_val, 1e-6) if vv_val > 0 else 0

        # NDVI harmonic parameters (prefix 'harm_' to avoid collision with height)
        f["harm_mean"] = _seg_mean(harm_mean, seg_v)
        f["harm_amplitude"] = _seg_mean(harm_amp, seg_v)
        f["harm_phase"] = _seg_mean(harm_phase, seg_v)
        f["harm_rmse"] = _seg_mean(harm_rmse, seg_v)

        # ESA WorldCover
        if cop_lc is not None:
            lc_vals = cop_lc[seg_v]
            if lc_vals.size > 0:
                counts = np.bincount(lc_vals, minlength=101)
                f["esa_dominant_lc"] = int(np.argmax(counts))
                n = max(lc_vals.size, 1)
                f["esa_built_frac"] = float(np.sum(lc_vals == 50)) / n
                f["esa_tree_frac"] = float(np.sum(lc_vals == 10)) / n
                f["esa_crop_frac"] = float(np.sum(lc_vals == 40)) / n
                f["esa_grass_frac"] = float(np.sum(lc_vals == 30)) / n
                f["esa_water_frac"] = float(np.sum(lc_vals == 80)) / n
            else:
                f.update({k: 0 for k in ["esa_dominant_lc", "esa_built_frac",
                         "esa_tree_frac", "esa_crop_frac", "esa_grass_frac", "esa_water_frac"]})
        else:
            f.update({k: 0 for k in ["esa_dominant_lc", "esa_built_frac",
                     "esa_tree_frac", "esa_crop_frac", "esa_grass_frac", "esa_water_frac"]})

        # Temporal
        f["temporal_h_std"] = _seg_mean(temporal_ndsm_std, seg_v)
        f["h_change"] = _seg_mean(temporal_ndsm_change, seg_v)
        f["dtm_change"] = _seg_mean(temporal_dtm_change, seg_v)
        f["dtm_change_abs"] = _seg_mean_abs(temporal_dtm_change, seg_v)
        f["stability"] = 1.0 / (1.0 + f["temporal_h_std"] + f["dtm_change_abs"])
        # Volume-change features (at 1m res, sum of ΔZ ≈ m³)
        f["volume_change_m3"] = _seg_sum(temporal_dtm_change, seg_v)
        f["volume_change_abs_m3"] = _seg_sum(np.abs(temporal_dtm_change) if temporal_dtm_change is not None else None, seg_v)
        f["dtm_change_max"] = float(np.nanmax(np.abs(temporal_dtm_change[seg_v]))) if temporal_dtm_change is not None and seg_v.any() else 0.0
        f["dtm_change_frac_03m"] = _seg_frac_above(temporal_dtm_change, seg_v, 0.3)

        # Hansen Global Forest Change
        if hansen_tc is not None:
            n_pix = max(int(seg_v.sum()), 1)
            f["hansen_treecover2000"] = float(np.mean(hansen_tc[seg_v])) if n_pix > 0 else 0.0
            f["hansen_loss_frac"] = float(np.sum(hansen_loss[seg_v] > 0)) / n_pix if hansen_loss is not None else 0.0
            f["hansen_recent_loss_frac"] = float(np.sum((hansen_loss[seg_v] >= _hansen_recent_start) & (hansen_loss[seg_v] <= _hansen_recent_end))) / n_pix if hansen_loss is not None else 0.0
            f["hansen_gain_frac"] = float(np.sum(hansen_gain[seg_v] > 0)) / n_pix if hansen_gain is not None else 0.0
            f["hansen_current_forest_frac"] = float(np.sum(hansen_current[seg_v])) / n_pix if hansen_current is not None else 0.0
        else:
            f["hansen_treecover2000"] = 0.0
            f["hansen_loss_frac"] = 0.0
            f["hansen_recent_loss_frac"] = 0.0
            f["hansen_gain_frac"] = 0.0
            f["hansen_current_forest_frac"] = 0.0

        objects.append(f)

    log.info("Extracted features for %d objects", len(objects))
    return objects


def _get_spectral_layer(d: dict | None, key: str, h: int, w: int) -> np.ndarray | None:
    if d is None:
        return None
    arr = d.get(key)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    return arr if arr.shape == (h, w) else None


def _seg_mean(arr: np.ndarray | None, seg: np.ndarray) -> float:
    if arr is None:
        return 0.0
    sv = arr[seg]
    valid = np.isfinite(sv)
    return float(np.nanmean(sv[valid])) if valid.any() else 0.0


def _seg_mean_abs(arr: np.ndarray | None, seg: np.ndarray) -> float:
    if arr is None:
        return 0.0
    sv = np.abs(arr[seg])
    valid = np.isfinite(sv)
    return float(np.nanmean(sv[valid])) if valid.any() else 0.0


def _seg_sum(arr: np.ndarray | None, seg: np.ndarray) -> float:
    """Sum of valid pixel values within segment (e.g. for volume = Σ ΔZ m³ at 1m res)."""
    if arr is None:
        return 0.0
    sv = arr[seg]
    valid = np.isfinite(sv)
    return float(np.nansum(sv[valid])) if valid.any() else 0.0


def _seg_frac_above(arr: np.ndarray | None, seg: np.ndarray, threshold: float) -> float:
    """Fraction of segment pixels where abs(value) > threshold."""
    if arr is None:
        return 0.0
    sv = arr[seg]
    valid = np.isfinite(sv)
    if not valid.any():
        return 0.0
    return float(np.sum(np.abs(sv[valid]) > threshold)) / float(valid.sum())


def _fuse_ndvi(
    bev: np.ndarray | None,
    cop: np.ndarray | None,
    mask: np.ndarray,
) -> np.ndarray | None:
    """Fuse BEV 1m NDVI with Copernicus 10m growing-season NDVI.

    The BEV ortho can be captured any day of the year, so a deciduous
    forest in January has NDVI ≈ 0.1 while Copernicus growing-season
    median shows 0.7.  We want to keep the 1m spatial detail (road vs
    adjacent grass) but shift the absolute level toward Copernicus.

    Method (local histogram matching at ~30m blocks):
      1. Compute block-mean of BEV and Cop (30m blocks ≈ 3× Cop pixels).
      2. For each block: offset = cop_block − bev_block.  This is the
         seasonal bias at that location.
      3. Smooth the offset field (Gaussian σ=15m) so it doesn't inject
         the 10m staircase artefact.
      4. fused = clip(bev + offset, -1, 1).

    Where BEV is unavailable → Copernicus.  Where Cop is unavailable → BEV.
    Where both unavailable → None.
    """
    if bev is None and cop is None:
        return None
    if bev is None:
        return cop
    if cop is None:
        return bev

    h, w = bev.shape
    if cop.shape != (h, w):
        return bev  # shape mismatch, trust BEV

    bev_f = np.where(mask & np.isfinite(bev), bev, np.nan).astype(np.float64)
    cop_f = np.where(mask & np.isfinite(cop), cop, np.nan).astype(np.float64)

    # Block-average both at ~30m (30 px at 1m) using a uniform filter
    # that ignores NaN via a count trick.
    blk = 31  # odd kernel
    valid_bev = np.isfinite(bev_f).astype(np.float64)
    valid_cop = np.isfinite(cop_f).astype(np.float64)
    bev_fill = np.nan_to_num(bev_f, 0.0)
    cop_fill = np.nan_to_num(cop_f, 0.0)

    bev_sum = uniform_filter(bev_fill, blk, mode="nearest")
    bev_cnt = uniform_filter(valid_bev, blk, mode="nearest")
    cop_sum = uniform_filter(cop_fill, blk, mode="nearest")
    cop_cnt = uniform_filter(valid_cop, blk, mode="nearest")

    eps = 1e-6
    with np.errstate(divide='ignore', invalid='ignore'):
        bev_blk = np.where(bev_cnt > eps, bev_sum / bev_cnt, np.nan)
        cop_blk = np.where(cop_cnt > eps, cop_sum / cop_cnt, np.nan)

    # Seasonal offset = how much Copernicus is above BEV at this location
    both_valid = np.isfinite(bev_blk) & np.isfinite(cop_blk)
    offset = np.where(both_valid, cop_blk - bev_blk, 0.0)

    # Smooth the offset so it doesn't carry 10m staircase edges
    offset = gaussian_filter(offset, sigma=15)

    # Apply: shift BEV toward Copernicus seasonal level
    fused = np.where(
        np.isfinite(bev_f),
        np.clip(bev_f + offset, -1.0, 1.0),
        cop_f,  # BEV missing → fall back to Cop
    )
    fused = np.where(mask, fused, np.nan).astype(np.float32)

    log.info("NDVI fusion: BEV mean=%.3f, Cop mean=%.3f, Fused mean=%.3f, "
             "offset mean=%.3f",
             float(np.nanmean(bev_f[mask])) if valid_bev.any() else 0,
             float(np.nanmean(cop_f[mask])) if valid_cop.any() else 0,
             float(np.nanmean(fused[mask & np.isfinite(fused)])),
             float(np.nanmean(offset[both_valid])) if both_valid.any() else 0)
    return fused


def _seg_stats(arr: np.ndarray | None, seg: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, std, max) for array values in segment."""
    if arr is None:
        return 0.0, 0.0, 0.0
    sv = arr[seg]
    valid = np.isfinite(sv)
    if not valid.any():
        return 0.0, 0.0, 0.0
    v = sv[valid]
    return float(np.mean(v)), float(np.std(v)), float(np.max(v))


# ===================================================================
# 4. OBJECT CLASSIFICATION — decision tree on per-object features
# ===================================================================

def _phenology_class(feat: dict) -> str:
    """Derive phenology class label from harmonic features."""
    amp = feat.get("harm_amplitude", 0)
    mean_v = feat.get("harm_mean", 0)
    phase = feat.get("harm_phase", 0)
    if amp == 0 and mean_v == 0:
        return ""
    if mean_v < 0.15 and amp < 0.05:
        return "road_or_bare"
    if amp > 0.15 and 4 <= phase <= 8:
        return "crop"
    if mean_v > 0.5 and amp < 0.08:
        return "forest"
    if mean_v > 0.3 and amp < 0.15:
        return "pasture"
    if amp > 0.10:
        return "seasonal_vegetation"
    return "unknown"


def classify_object(feat: dict, *, has_spectral: bool = False) -> tuple[str, int, float, bool]:
    """Classify a single object from its feature vector.

    Returns (type_name, type_code, confidence, is_manmade).

    Decision tree ordered by discriminative power:
    1. Temporal disturbance (highest priority — recent changes)
    2. Height separates ground from elevated
    3. Among elevated: roughness + shape + NDVI separate building from tree
    4. Among ground: roughness + NDVI separate road from field from bare
    """
    h_mean = feat.get("h_mean", 0)
    h_max = feat.get("h_max", 0)
    h_std = feat.get("h_std", 0)
    slope_mean = feat.get("slope_mean", 0)
    dsm_rough = feat.get("dsm_roughness", 0)
    dtm_rough = feat.get("dtm_roughness", 0)
    compact = feat.get("compactness", 0)
    elong = feat.get("elongation", 1)
    area = feat.get("area", 0)
    ndvi = feat.get("ndvi_mean", 0)
    ndvi_std = feat.get("ndvi_std", 0)
    brightness = feat.get("brightness_mean", 0)
    nir = feat.get("nir_mean", 0)
    cop_ndvi = feat.get("cop_ndvi_mean", 0)
    fused_ndvi = feat.get("fused_ndvi_mean", 0)
    esa_built = feat.get("esa_built_frac", 0)
    esa_tree = feat.get("esa_tree_frac", 0)
    esa_crop = feat.get("esa_crop_frac", 0)
    esa_grass = feat.get("esa_grass_frac", 0)
    esa_water = feat.get("esa_water_frac", 0)
    h_change = feat.get("h_change", 0)
    dtm_change = feat.get("dtm_change", 0)
    dtm_change_abs = feat.get("dtm_change_abs", 0)
    stability = feat.get("stability", 1)
    temporal_h_std = feat.get("temporal_h_std", 0)
    volume_change = feat.get("volume_change_m3", 0)
    volume_change_abs = feat.get("volume_change_abs_m3", 0)
    dtm_change_max = feat.get("dtm_change_max", 0)
    dtm_change_frac = feat.get("dtm_change_frac_03m", 0)
    solidity = feat.get("solidity", 0)
    extent = feat.get("extent", 0)
    dsm_edge = feat.get("dsm_edge_strength", 0)

    # --- Texture features (GLCM from ortho) ---
    glcm_entropy = feat.get("glcm_entropy", 0)
    glcm_homogeneity = feat.get("glcm_homogeneity", 0)
    glcm_contrast = feat.get("glcm_contrast", 0)
    tex_complexity = feat.get("texture_complexity", 0)
    has_texture = glcm_entropy > 0 or glcm_homogeneity > 0

    # --- SAR backscatter features ---
    sar_vv = feat.get("sar_vv", 0)
    sar_vh = feat.get("sar_vh", 0)
    sar_ratio = feat.get("sar_ratio", 0)
    has_sar = sar_vv > 0 or sar_vh > 0

    # --- NDVI harmonic (phenology) features ---
    harm_mean = feat.get("harm_mean", 0)
    harm_amp = feat.get("harm_amplitude", 0)
    harm_phase = feat.get("harm_phase", 0)
    harm_rmse = feat.get("harm_rmse", 0)
    has_harmonics = harm_mean != 0 or harm_amp != 0

    # --- NDVI selection hierarchy ---
    #   1. Fused (BEV 1m spatial + Copernicus seasonal correction) — best
    #   2. BEV 1m alone — good spatial, possibly wrong season
    #   3. Copernicus 10m alone — right season, coarse (bleeds into roads)
    # "have_ndvi" = any NDVI source is available
    # "ndvi_is_coarse" = relying on 10m Copernicus without 1m spatial detail
    if fused_ndvi != 0:
        best_ndvi = fused_ndvi
        have_ndvi = True
        ndvi_is_coarse = False
    elif has_spectral and ndvi != 0:
        best_ndvi = ndvi
        have_ndvi = True
        ndvi_is_coarse = False
    elif cop_ndvi != 0:
        best_ndvi = cop_ndvi
        have_ndvi = True
        ndvi_is_coarse = True   # 10m, bleeds across boundaries
    else:
        best_ndvi = 0
        have_ndvi = False
        ndvi_is_coarse = False

    # ---------------------------------------------------------------
    # DISTURBANCE (highest priority — recent changes)
    # ---------------------------------------------------------------
    # Trees naturally grow 2-5m between LIDAR dates (2022→2024).
    # Exclude objects that look like growing vegetation.
    is_veg_like = best_ndvi > 0.35 and dsm_rough > 0.8

    # NDVI veto: high NDVI means the surface is currently vegetated.
    # Real earthworks (excavation, fill, construction) destroy vegetation,
    # so NDVI > 0.5 at 1m resolution rules out active earthworks.
    # Former clearings that have regrown are no longer "disturbance".
    is_green = have_ndvi and best_ndvi > 0.5

    # Size gates: earthworks are spatially large; individual tree death
    # can be a single crown (~15-30m² at 1m).  Tiny slivers (<8m²) at
    # layer boundaries are always noise.
    big_enough_earthwork = area >= 25  # excavation, fill, construction
    big_enough_tree_loss = area >= 8   # single tree crown

    # --- DTM change: terrain was physically moved (excavation/fill) ---
    # DTM noise under forest is ±0.15-0.30m (tree removal changes
    # ground-return interpolation).  Only trust DTM change where the
    # signal clearly exceeds forest DTM noise AND surface is bare.
    on_ground = h_mean < 2.0
    dtm_signal_strong = dtm_change_abs > 0.8  # unambiguous earthwork
    dtm_signal_mod = dtm_change_abs > 0.4     # moderate, needs corroboration

    # Volume-enhanced detection: dtm_change_frac tells us what portion of
    # the segment is disturbed; volume_change gives the net m³ moved.
    # dtm_change_max catches localised deep cuts even when mean is low.
    has_volume_signal = (volume_change_abs > 50 or dtm_change_max > 1.0 or dtm_change_frac > 0.3)

    if big_enough_earthwork and not is_green and (
        dtm_signal_strong or (dtm_signal_mod and on_ground) or
        (has_volume_signal and not is_veg_like and on_ground)
    ):
        conf = min(0.4 + dtm_change_abs + 0.1 * min(abs(volume_change) / 100, 1.0), 0.95)
        if dtm_change < -0.4 or volume_change < -50:
            return "excavation", OBJECT_TYPES["excavation"], conf, True
        if dtm_change > 0.4 or volume_change > 50:
            return "fill", OBJECT_TYPES["fill"], conf, True

    # --- nDSM change: something was built, removed, or felled ---
    if abs(h_change) > 2.0 and temporal_h_std > 1.0:

        # Height DROPPED → tree_loss / tree death / site clearing
        if h_change < -2.0 and h_mean < 2.0:
            if big_enough_earthwork and not is_green and dtm_change_abs > 0.4:
                # Trees removed AND terrain significantly reshaped → site clearing
                return "construction", OBJECT_TYPES["construction"], 0.8, True
            elif big_enough_tree_loss:
                # Trees removed, terrain intact → logging / tree death
                conf = 0.7 if area < 50 else 0.85
                return "tree_loss", OBJECT_TYPES["tree_loss"], conf, True

        # Height GREW and doesn't look like vegetation → new structure
        if h_change > 2.0 and not is_veg_like and not is_green and big_enough_earthwork:
            return "construction", OBJECT_TYPES["construction"], min(0.5 + abs(h_change) / 10, 0.9), True

    # Combined: moderate DTM + nDSM change on non-vegetation
    if big_enough_earthwork and not is_green and dtm_change_abs > 0.4 and abs(h_change) > 1.0 and not is_veg_like and on_ground:
        return "construction", OBJECT_TYPES["construction"], 0.65, True

    # High instability on ground with real DTM shift — not on vegetated areas
    if big_enough_earthwork and not is_green and stability < 0.3 and temporal_h_std > 1.5 and dtm_change_abs > 0.4 and on_ground:
        if dtm_change < 0:
            return "excavation", OBJECT_TYPES["excavation"], 0.5, True
        return "fill", OBJECT_TYPES["fill"], 0.5, True

    # ---------------------------------------------------------------
    # WATER (ESA WorldCover hint + low NDVI + low NIR + flat)
    # ---------------------------------------------------------------
    if esa_water > 0.5 and best_ndvi < 0.1 and slope_mean < 5:
        return "water", OBJECT_TYPES["water"], 0.8, False
    # Detect from spectral: very low NIR + very low NDVI + flat
    # Guard: nir must be valid (>0) to avoid nodata false positives
    if has_spectral and 0 < nir < 30 and ndvi < 0.0 and slope_mean < 3 and h_mean < 0.3:
        return "water", OBJECT_TYPES["water"], 0.65, False

    # ---------------------------------------------------------------
    # ELEVATED OBJECTS (nDSM > 0.5m)
    # ---------------------------------------------------------------
    if h_mean > 0.5:
        # --- Building score ---
        bld_score = 0.0

        # Surface smoothness (buildings have smoother DSM than trees)
        # Calibrated: buildings p50=0.70, trees p50=1.71
        if dsm_rough < 0.5:
            bld_score += 3.0
        elif dsm_rough < 0.8:
            bld_score += 2.0
        elif dsm_rough < 1.2:
            bld_score += 0.5
        else:
            bld_score -= 1.5  # rough = tree-like

        # Uniform height within object
        # Calibrated: buildings p50=1.28, trees p50=2.82
        if h_std < 0.8:
            bld_score += 2.0
        elif h_std < 1.5:
            bld_score += 1.0
        elif h_std > 3.0:
            bld_score -= 2.0  # very variable = tree/mixed
        elif h_std > 2.0:
            bld_score -= 1.0

        # Flat terrain underneath
        if slope_mean < 5:
            bld_score += 1.5
        elif slope_mean < 10:
            bld_score += 0.5
        elif slope_mean > 15:
            bld_score -= 1.0  # steep = unlikely building

        # Compact shape (buildings are blocky)
        if compact > 0.4:
            bld_score += 1.5
        elif compact > 0.2:
            bld_score += 0.5
        elif compact < 0.1:
            bld_score -= 1.0  # very irregular = tree canopy

        # Spectral: low NDVI = non-vegetated
        # Note: fused NDVI shifts values toward growing-season levels,
        # so bare roofs may show 0.10-0.20 from Copernicus bleed.
        if have_ndvi:
            if best_ndvi < -0.1:
                bld_score += 3.0  # strongly non-vegetated
            elif best_ndvi < 0.1:
                bld_score += 2.0
            elif best_ndvi < 0.2:
                bld_score += 1.0
            elif best_ndvi > 0.55:
                bld_score -= 6.0  # unambiguous vegetation at 1m
            elif best_ndvi > 0.45:
                bld_score -= 4.0  # certainly vegetation
            elif best_ndvi > 0.35:
                bld_score -= 3.0  # very likely vegetation
            elif best_ndvi > 0.25:
                bld_score -= 2.0
            elif best_ndvi > 0.20:
                bld_score -= 0.5  # mild penalty

        # Combined tree signal: rough DSM + green = tree canopy, not roof.
        # Only apply heavy penalty when BOTH signals are strong.
        # Roofs can show roughness ~0.8-1.0 (chimneys, edges) and
        # fused NDVI ~0.15-0.25 (Copernicus bleed), so don't penalise those.
        if dsm_rough > 1.5 and best_ndvi > 0.25:
            bld_score -= 3.0  # very rough + green = certainly tree
        elif dsm_rough > 1.2 and best_ndvi > 0.35:
            bld_score -= 2.5  # rough + clearly green

        # Bright surface + low NDVI
        if has_spectral and brightness > 100 and ndvi < 0.1:
            bld_score += 1.5

        # Temporal stability is the strongest building indicator:
        # buildings don't move between LIDAR dates; tree canopies sway
        # and grow.  When temporal data is available (stability != 1.0
        # default), lean heavily on it.
        has_temporal = stability != 1.0  # 1.0 = no temporal data loaded
        if has_temporal:
            if stability > 0.9:
                bld_score += 3.0  # rock solid = strong building signal
            elif stability > 0.7:
                bld_score += 1.5
            elif stability < 0.4:
                bld_score -= 2.0  # volatile = tree canopy
            elif stability < 0.6:
                bld_score -= 1.0
        else:
            # No temporal data — mild bonus for stability placeholder
            if stability > 0.8:
                bld_score += 1.0

        # Rectangularity: only use as mild tiebreaker, not primary signal.
        # In practice solidity/extent overlap too much between compact tree
        # crowns and small buildings to carry heavy weight.
        if solidity > 0.90 and extent > 0.65:
            bld_score += 1.0  # very boxy = mild building hint
        elif solidity < 0.55 and extent < 0.35:
            bld_score -= 0.5  # very irregular = mild tree hint

        # ESA built-up prior
        if esa_built > 0.3:
            bld_score += 1.0

        # Texture: buildings have lower entropy than tree canopy
        # Real data: roof median ~8.1, tree ~7.7 — heavy overlap, use mild signals
        if has_texture:
            if glcm_entropy < 5.5:
                bld_score += 1.5   # very uniform = roof material
            elif glcm_entropy < 6.5:
                bld_score += 0.5   # moderate = possible building
            elif glcm_entropy > 8.0 and glcm_homogeneity < 0.25:
                bld_score -= 1.0   # very complex + very non-uniform = canopy
            # Homogeneity is more discriminative than entropy for buildings
            if glcm_homogeneity > 0.45:
                bld_score += 1.0   # smooth surface = built
            elif glcm_homogeneity < 0.25:
                bld_score -= 0.5   # rough = vegetation

        # SAR: buildings are strong corner reflectors (high VV)
        if has_sar:
            if sar_vv > 0.1 and sar_ratio > 3.0:
                bld_score += 1.5  # strong VV + high VV/VH = corner reflector
            elif sar_vh > 0.05 and sar_ratio < 1.5:
                bld_score -= 1.0  # high VH + low ratio = volume scattering (trees)

        # Harmonics: buildings have ~zero seasonal amplitude
        if has_harmonics:
            if harm_amp < 0.03 and harm_mean < 0.15:
                bld_score += 1.0  # no seasonality + bare = likely built
            elif harm_amp > 0.10:
                bld_score -= 1.0  # seasonal = vegetation

        # Height penalty: very tall more likely tree/tower
        if h_mean > 25:
            bld_score -= 1.5

        # --- Decision ---
        bld_thresh = 5.5 if have_ndvi else 7.0

        # --- Mast: tiny footprint, very tall, isolated ---
        if area < 10 and h_mean > 15 and compact < 0.5:
            return "mast", OBJECT_TYPES["mast"], 0.6, True

        # --- Solar panel: building-like + very smooth + bright ---
        if (bld_score >= bld_thresh and dsm_rough < 0.3
                and h_std < 0.5 and has_spectral and brightness > 120):
            return "solar_panel", OBJECT_TYPES["solar_panel"], min(0.5 + bld_score / 20, 0.95), True

        # --- Greenhouse: roof-like but high NIR (light transmits) ---
        if (bld_score >= bld_thresh - 1 and has_spectral
                and nir > 80 and ndvi > 0.0 and dsm_rough < 0.8 and h_mean > 2):
            return "greenhouse", OBJECT_TYPES["greenhouse"], 0.55, True

        if bld_score >= bld_thresh:
            if h_mean > 2.0 and compact > 0.15 and area > 8:
                conf = min(0.4 + bld_score / 20, 0.95)
                return "roof", OBJECT_TYPES["roof"], conf, True
            elif elong > 5 and area < 20:
                return "wall", OBJECT_TYPES["wall"], 0.5, True
            elif elong > 4:
                return "fence", OBJECT_TYPES["fence"], 0.5, True
            else:
                return "roof", OBJECT_TYPES["roof"], min(0.4 + bld_score / 20, 0.9), True

        # --- Not building: tree / shrub / hedge ---
        if h_mean >= 4.0:
            # Hedge: tall, very elongated vegetation
            if elong > 4 and best_ndvi > 0.2 and area < 200:
                return "hedge", OBJECT_TYPES["hedge"], 0.55, False
            conf = 0.5
            if best_ndvi > 0.4:
                conf = 0.85
            elif best_ndvi > 0.3:
                conf = 0.75
            elif dsm_rough > 1.0:
                conf = 0.65
            elif esa_tree > 0.3:
                conf = 0.7
            if h_change > 0.5:  # growing
                conf = min(conf + 0.1, 0.95)
            # Texture boost: high entropy + low homogeneity = forest canopy
            if has_texture and glcm_entropy > 7.0 and glcm_homogeneity < 0.35:
                conf = min(conf + 0.1, 0.95)
            # SAR boost: high VH = volume scattering from canopy
            if has_sar and sar_vh > 0.04 and sar_ratio < 2.0:
                conf = min(conf + 0.1, 0.95)
            # Harmonics boost: low amplitude + high mean = evergreen/stable forest
            if has_harmonics and harm_amp < 0.08 and harm_mean > 0.5:
                conf = min(conf + 0.1, 0.95)
            return "tree", OBJECT_TYPES["tree"], conf, False

        if h_mean >= 1.0:
            # Hedge: medium height, elongated, green
            if elong > 4 and best_ndvi > 0.2:
                return "hedge", OBJECT_TYPES["hedge"], 0.5, False
            if best_ndvi > 0.3 or dsm_rough > 0.8:
                return "shrub", OBJECT_TYPES["shrub"], 0.6, False
            elif compact > 0.3 and dsm_rough < 0.5:
                return "roof", OBJECT_TYPES["roof"], 0.5, True  # low building
            else:
                return "shrub", OBJECT_TYPES["shrub"], 0.4, False

        # Low elevated (0.5-1.0m)
        if best_ndvi > 0.3:
            if elong > 4:
                return "hedge", OBJECT_TYPES["hedge"], 0.4, False
            return "shrub", OBJECT_TYPES["shrub"], 0.5, False
        return "fence", OBJECT_TYPES["fence"], 0.3, True

    # ---------------------------------------------------------------
    # GROUND OBJECTS (nDSM < 0.5m)
    # ---------------------------------------------------------------

    # Engineered surface: smooth terrain = roads, parking, foundations.
    # This MUST be checked before NDVI-based veg classification because
    # even fused NDVI can read high near vegetation edges.
    is_smooth = dtm_rough < 0.04 and slope_mean < 15
    is_very_smooth = dtm_rough < 0.02 and slope_mean < 10

    # Texture-based road score: roads have very low entropy, high homogeneity
    # Texture road score: calibrated from real data
    # road median: ent=5.9, hom=0.53 | grass: ent=6.6, hom=0.44 | tree: ent=7.7, hom=0.28
    tex_road_score = 0.0
    if has_texture:
        if glcm_entropy < 5.5 and glcm_homogeneity > 0.5:
            tex_road_score = 1.0   # uniform surface = road
        elif glcm_entropy < 6.2 and glcm_homogeneity > 0.45:
            tex_road_score = 0.5   # likely paved
        elif glcm_entropy > 7.0 and glcm_homogeneity < 0.3:
            tex_road_score = -0.5  # complex texture = tree canopy
        elif glcm_entropy > 6.5 and glcm_homogeneity < 0.4:
            tex_road_score = -0.3  # medium-high = grass/vegetation

    if is_smooth:
        # Very smooth terrain is almost certainly engineered.
        # With fused or 1m NDVI, also accept moderate NDVI (adjacent veg
        # bleeds a little even at 1m).  With coarse-only NDVI, always
        # trust DTM roughness over the 10m NDVI.
        ndvi_ok = best_ndvi < 0.30 or ndvi_is_coarse or is_very_smooth
        if ndvi_ok:
            if elong > 3 and area > 15:
                conf = 0.7 if is_very_smooth else 0.55
                if have_ndvi and best_ndvi < 0.10:
                    conf = 0.85
                elif have_ndvi and best_ndvi < 0.20:
                    conf = 0.75
                conf = min(conf + tex_road_score * 0.1, 0.95)
                return "road", OBJECT_TYPES["road"], conf, True
            elif area > 40 and compact > 0.25:
                conf = 0.55
                if have_ndvi and best_ndvi < 0.15 and brightness > 80:
                    conf = 0.75
                conf = min(conf + tex_road_score * 0.1, 0.95)
                return "parking", OBJECT_TYPES["parking"], conf, True
            elif area > 8:
                conf = 0.45 + tex_road_score * 0.1
                return "path", OBJECT_TYPES["path"], min(conf, 0.95), True

    # Garden: moderate NDVI, near buildings (small area)
    if best_ndvi > 0.2 and best_ndvi < 0.5 and area < 300 and esa_built > 0.1:
        # But not if terrain is smooth (would be paved yard/driveway)
        if dtm_rough > 0.03:
            return "garden", OBJECT_TYPES["garden"], 0.45, False

    # Texture override: very low entropy + high homogeneity on ground =
    # road/path even with moderate NDVI (ortho bleed near edges).
    if has_texture and tex_road_score >= 0.5 and is_smooth:
        conf = 0.55 + tex_road_score * 0.15
        if elong > 3 and area > 15:
            return "road", OBJECT_TYPES["road"], min(conf, 0.85), True
        elif area > 8:
            return "path", OBJECT_TYPES["path"], min(conf, 0.75), True

    # ---------------------------------------------------------------
    # HARMONIC PHENOLOGY OVERRIDE (strongest temporal signal)
    # If we have harmonic features, use them as primary discriminator
    # for crop vs pasture vs road — this is the step change.
    # ---------------------------------------------------------------
    if has_harmonics and harm_amp > 0:
        # Crop: high amplitude (>0.15), peak Jun-Aug
        if harm_amp > 0.15 and 4 <= harm_phase <= 8:
            conf = 0.75
            if esa_crop > 0.3:
                conf = 0.85
            if best_ndvi > 0.3:
                conf = min(conf + 0.05, 0.95)
            if has_texture and 5.5 <= glcm_entropy <= 7.5:
                conf = min(conf + 0.05, 0.95)
            return "crop", OBJECT_TYPES["crop"], conf, False

        # Road/bare: very low mean + very low amplitude
        if harm_mean < 0.15 and harm_amp < 0.05:
            conf = 0.6
            if is_smooth:
                conf = 0.75
            if has_texture and tex_road_score > 0:
                conf = min(conf + 0.1, 0.90)
            # SAR confirmation: low VH = specular (road), not volume (veg)
            if has_sar and sar_vh < 0.03:
                conf = min(conf + 0.1, 0.90)
            if elong > 3 and area > 15:
                return "road", OBJECT_TYPES["road"], conf, True
            elif area > 40 and compact > 0.25:
                return "parking", OBJECT_TYPES["parking"], conf, True
            elif area > 8:
                return "path", OBJECT_TYPES["path"], conf, True
            return "bare_soil", OBJECT_TYPES["bare_soil"], 0.5, False

        # Pasture: moderate mean, low-moderate amplitude
        if harm_mean > 0.3 and harm_amp < 0.15:
            conf = 0.65
            if esa_grass > 0.3:
                conf = 0.75
            if best_ndvi > 0.3:
                conf = min(conf + 0.05, 0.85)
            return "grass", OBJECT_TYPES["grass"], conf, False

    # ---------------------------------------------------------------
    # SAR-based ground classification (cloud-proof)
    # ---------------------------------------------------------------
    if has_sar and not has_harmonics:
        # Very low VH on ground = specular surface = road/parking
        if sar_vh < 0.02 and sar_vv > 0.03 and is_smooth:
            conf = 0.55
            if elong > 3:
                return "road", OBJECT_TYPES["road"], conf, True
            elif area > 40:
                return "parking", OBJECT_TYPES["parking"], conf, True
        # High VH relative to height = dense low vegetation
        if sar_vh > 0.04 and best_ndvi > 0.3:
            if esa_crop > 0.3:
                return "crop", OBJECT_TYPES["crop"], 0.6, False
            return "grass", OBJECT_TYPES["grass"], 0.55, False

    # Crop vs grass (use ESA WorldCover + NDVI — fallback when no harmonics)
    if best_ndvi > 0.4:
        if ndvi_is_coarse and is_smooth:
            # Coarse NDVI on smooth ground = likely road with 10m veg bleed
            return "road", OBJECT_TYPES["road"], 0.45, True
        if esa_crop > 0.3:
            conf = 0.7
            if has_texture and 5.5 <= glcm_entropy <= 7.5:
                conf = min(conf + 0.05, 0.85)
            return "crop", OBJECT_TYPES["crop"], conf, False
        if area > 500:
            conf = 0.65
            if has_texture and 5.5 <= glcm_entropy <= 7.5:
                conf = min(conf + 0.05, 0.80)
            return "grass", OBJECT_TYPES["grass"], conf, False
        return "grass", OBJECT_TYPES["grass"], 0.5, False

    if best_ndvi > 0.2:
        if ndvi_is_coarse and is_smooth:
            return "path", OBJECT_TYPES["path"], 0.4, True
        if esa_crop > 0.3:
            return "crop", OBJECT_TYPES["crop"], 0.55, False
        return "grass", OBJECT_TYPES["grass"], 0.4, False

    # Rock: steep + rough + low NDVI
    if slope_mean > 30 and dtm_rough > 0.3:
        return "rock", OBJECT_TYPES["rock"], 0.6, False

    # Bare soil vs dark paved
    if best_ndvi < 0.15 and has_spectral:
        if brightness > 100:
            return "bare_soil", OBJECT_TYPES["bare_soil"], 0.5, False
        if brightness < 40:
            return "road", OBJECT_TYPES["road"], 0.4, True  # dark = asphalt

    # Default
    if slope_mean > 20 or dtm_rough > 0.2:
        return "bare_soil", OBJECT_TYPES["bare_soil"], 0.3, False
    return "grass", OBJECT_TYPES["grass"], 0.3, False


# ===================================================================
# 5. HIERARCHICAL GROUPING
# ===================================================================

# Adjacency merge rules: which individual types can merge
_MERGE_RULES = {
    "tree": {"tree", "shrub", "hedge"},
    "shrub": {"tree", "shrub", "hedge"},
    "hedge": {"tree", "shrub", "hedge"},
    "roof": {"roof", "wall", "solar_panel", "greenhouse"},
    "wall": {"roof", "wall"},
    "solar_panel": {"roof", "solar_panel"},
    "greenhouse": {"roof", "greenhouse"},
    "road": {"road", "path", "parking"},
    "path": {"road", "path"},
    "parking": {"road", "parking"},
    "grass": {"grass", "crop", "garden"},
    "crop": {"grass", "crop"},
    "garden": {"grass", "garden"},
    "orchard": {"orchard", "vineyard"},
    "vineyard": {"orchard", "vineyard"},
    "excavation": {"excavation", "fill", "construction", "bare_soil"},
    "fill": {"excavation", "fill", "construction", "bare_soil"},
    "construction": {"excavation", "fill", "construction"},
    "tree_loss": {"tree_loss", "bare_soil"},
    "water": {"water"},
}

# Resolve group name from the set of member types.
# Checked in order; first match wins.
_GROUP_NAME_MAP: list[tuple[set[str], str]] = [
    # Vegetation
    ({"tree"}, "forest"),
    ({"tree", "shrub"}, "forest"),
    ({"tree", "shrub", "hedge"}, "forest"),
    ({"shrub"}, "woodland"),
    ({"shrub", "hedge"}, "woodland"),
    ({"hedge"}, "hedgerow"),
    # Water
    ({"water"}, "waterbody"),
    # Buildings
    ({"roof"}, "building"),
    ({"roof", "wall"}, "building"),
    ({"roof", "solar_panel"}, "building"),
    ({"roof", "wall", "solar_panel"}, "building"),
    ({"roof", "greenhouse"}, "building"),
    ({"roof", "wall", "solar_panel", "greenhouse"}, "building"),
    ({"greenhouse"}, "building"),
    # Transportation
    ({"road"}, "road_network"),
    ({"road", "path"}, "road_network"),
    ({"road", "parking"}, "road_network"),
    ({"road", "path", "parking"}, "road_network"),
    ({"path"}, "road_network"),
    ({"parking"}, "road_network"),
    # Agricultural
    ({"crop"}, "cropland"),
    ({"grass", "crop"}, "cropland"),
    ({"grass"}, "pasture"),
    ({"grass", "garden"}, "pasture"),
    ({"garden"}, "pasture"),
    ({"orchard"}, "orchard_grove"),
    ({"vineyard"}, "orchard_grove"),
    ({"orchard", "vineyard"}, "orchard_grove"),
    # Disturbance
    ({"excavation"}, "quarry"),
    ({"excavation", "fill"}, "quarry"),
    ({"excavation", "fill", "construction"}, "construction_site"),
    ({"construction"}, "construction_site"),
    ({"excavation", "fill", "construction", "bare_soil"}, "construction_site"),
    ({"tree_loss"}, "construction_site"),
    ({"tree_loss", "bare_soil"}, "construction_site"),
]


def group_objects(
    objects: list[SegmentedObject],
    labels: np.ndarray,
) -> list[SegmentedObject]:
    """Merge adjacent compatible objects into groups.

    tree+tree → forest, roof+roof → building, etc.
    """
    if not objects:
        return objects

    label_to_obj = {o.obj_id: o for o in objects}

    # Build adjacency in one pass — find all neighboring label pairs
    adjacency: dict[int, set[int]] = {o.obj_id: set() for o in objects}
    valid_ids = set(label_to_obj.keys())

    def _add_pairs(arr_a, arr_b):
        diff = arr_a != arr_b
        a_vals = arr_a[diff]
        b_vals = arr_b[diff]
        # Stack and get unique pairs
        pairs = np.unique(np.column_stack([a_vals, b_vals]), axis=0)
        for a, b in pairs:
            if a > 0 and b > 0 and a in valid_ids and b in valid_ids:
                adjacency[a].add(b)
                adjacency[b].add(a)

    _add_pairs(labels[:, :-1].ravel(), labels[:, 1:].ravel())   # horizontal
    _add_pairs(labels[:-1, :].ravel(), labels[1:, :].ravel())   # vertical

    # Union-Find
    parent = {o.obj_id: o.obj_id for o in objects}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for obj in objects:
        compatible = _MERGE_RULES.get(obj.obj_type, set())
        for nlbl in adjacency.get(obj.obj_id, set()):
            nbr = label_to_obj.get(nlbl)
            if nbr and nbr.obj_type in compatible:
                union(obj.obj_id, nlbl)

    # Assign groups
    groups: dict[int, list[SegmentedObject]] = {}
    for obj in objects:
        groups.setdefault(find(obj.obj_id), []).append(obj)

    gid = 1
    for root, members in groups.items():
        types_in = {m.obj_type for m in members}
        group_name = _resolve_group_name(types_in)
        for m in members:
            m.group_id = gid
            m.group_type = group_name
        gid += 1

    return objects


def _resolve_group_name(types: set[str]) -> str:
    """Find best group name for a set of member types."""
    for candidates, name in _GROUP_NAME_MAP:
        if types == candidates or types.issubset(candidates):
            return name
    # Fallback: use most common type
    for candidates, name in _GROUP_NAME_MAP:
        if candidates & types:
            return name
    return ""


# ===================================================================
# 6. CADASTRE CALIBRATION
# ===================================================================

def calibrate_with_cadastre(
    objects: list[SegmentedObject],
    labels: np.ndarray,
    building_mask: np.ndarray,
    *,
    overlap_threshold: float = 0.3,
) -> list[SegmentedObject]:
    """Refine building classification using cadastre ground truth.

    High overlap with cadastre but not classified as building → reclassify,
    UNLESS the object looks clearly like vegetation (high NDVI + rough DSM).
    Trees commonly overhang building footprints.

    Classified as roof but zero cadastre overlap → lower confidence.
    """
    if building_mask is None or not building_mask.any():
        return objects

    for obj in objects:
        seg = labels == obj.obj_id
        seg_area = int(seg.sum())
        if seg_area == 0:
            continue

        overlap = int((seg & building_mask).sum())
        precision = overlap / max(seg_area, 1)

        if precision > overlap_threshold:
            # Don't reclassify objects that are clearly tree canopy
            # overhanging a building footprint.  Key signal: DSM roughness.
            # Roofs rarely exceed roughness=1.5; tree crowns typically >1.5.
            # Be conservative — prefer to reclassify (helps recall) unless
            # the canopy signal is very strong.
            looks_like_canopy = (
                obj.roughness > 1.5  # very rough = definite tree texture
            ) or (
                obj.roughness > 1.0 and obj.ndvi_fused > 0.40
            ) or (
                obj.ndvi_fused > 0.55  # overwhelmingly vegetated
            )

            if obj.obj_type in ("roof", "wall", "solar_panel", "greenhouse", "fence"):
                # Already building — just boost confidence
                obj.confidence = min(obj.confidence + 0.15, 0.95)
            elif not looks_like_canopy and obj.height_mean > 1.5:
                # Reclassify non-veg elevated object to roof
                obj.obj_type = "roof"
                obj.type_code = OBJECT_TYPES["roof"]
                obj.is_manmade = True
                obj.confidence = max(obj.confidence, 0.7)
            # else: canopy overhanging cadastre footprint → leave as-is

        elif precision < 0.05:
            if obj.obj_type == "roof" and obj.confidence < 0.7:
                obj.confidence = max(obj.confidence - 0.15, 0.2)

    return objects


# ===================================================================
# 7. EVALUATION
# ===================================================================

def evaluate_against_cadastre(
    objects: list[SegmentedObject],
    labels: np.ndarray,
    building_mask: np.ndarray,
) -> dict:
    """Precision/recall/F1 for building detection vs cadastre."""
    if building_mask is None or not building_mask.any():
        return {"error": "no_cadastre_data"}

    h, w = labels.shape
    pred = np.zeros((h, w), dtype=bool)
    _BUILDING_TYPES = {"roof", "wall", "solar_panel", "greenhouse", "construction"}
    for obj in objects:
        if obj.obj_type in _BUILDING_TYPES:
            pred |= (labels == obj.obj_id)

    truth = building_mask.astype(bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))

    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(pr, 4), "recall": round(rc, 4),
        "f1": round(f1, 4), "iou": round(iou, 4),
        "building_pixels_truth": int(truth.sum()),
        "building_pixels_pred": int(pred.sum()),
    }


# ===================================================================
# 8. MAIN ENTRY POINT
# ===================================================================

def segment_and_classify(
    dtm: np.ndarray,
    dsm: np.ndarray,
    mask: np.ndarray,
    transform,
    *,
    dtm_dates: dict[str, np.ndarray] | None = None,
    dsm_dates: dict[str, np.ndarray] | None = None,
    spectral: dict | None = None,
    copernicus: dict | None = None,
    building_footprints: np.ndarray | None = None,
    hansen: dict | None = None,
    min_object_size: int = 30,
    felz_scale: float = 150.0,
    rag_threshold: float = 0.12,
    ortho_year: int | None = None,
    observation_year: int | None = None,
    features_only: bool = False,
) -> dict:
    """Full pipeline: gradient → Felzenszwalb → RAG merge → features → classify → group.

    If features_only=True, return after feature extraction (Steps 1-3b),
    skipping classification/calibration/grouping.  Used by RF training.

    Parameters
    ----------
    dtm, dsm : 2D float32, EPSG:3035, 1m resolution
    mask : bool, valid pixels
    transform : rasterio Affine
    dtm_dates, dsm_dates : {date: array} for temporal analysis
    spectral : {ndvi, brightness, nir, red, green, blue} from ortho
    copernicus : {ndvi, landcover, ...} from Sentinel-2
    building_footprints : bool mask from cadastre
    min_object_size : Felzenszwalb min_size (pixels = m² at 1m)
    felz_scale : Felzenszwalb scale (higher = fewer segments)
    rag_threshold : RAG merge threshold (lower = less merging)

    Returns
    -------
    dict with: objects, labels, gradient, stats, evaluation
    """
    h, w = dtm.shape
    ndsm = np.clip(dsm - dtm, 0, None).astype(np.float32)
    has_spectral = spectral is not None and spectral.get("ndvi") is not None
    log.info("segment_and_classify: %dx%d, valid=%d px, spectral=%s",
             w, h, int(mask.sum()), has_spectral)

    # --- Resample Copernicus to 1m grid ---
    cop_resampled = None
    if copernicus is not None:
        cop_resampled = _resample_copernicus(copernicus, transform, (h, w))

    # --- Step 1: Fused gradient ---
    log.info("Step 1: Fused gradient (Sobel on DTM/DSM/CHM/spectral)")
    gradient = compute_fused_gradient(dtm, dsm, ndsm, mask, spectral=spectral)

    # --- Step 2: Felzenszwalb + RAG segmentation ---
    log.info("Step 2: Felzenszwalb + RAG segmentation (scale=%.0f, rag=%.2f)",
             felz_scale, rag_threshold)
    labels = segment_landscape(
        gradient, ndsm, mask,
        felz_scale=felz_scale,
        felz_min_size=min_object_size,
        rag_threshold=rag_threshold,
    )
    n_seg = len(np.unique(labels[labels > 0]))
    log.info("  → %d total segments", n_seg)
    del gradient  # free ~10MB

    # --- Step 3: Feature extraction ---
    log.info("Step 3: Per-object feature extraction")
    features = extract_object_features(
        labels, dtm, dsm, ndsm, mask, transform,
        spectral=spectral, cop=cop_resampled,
        dtm_dates=dtm_dates, dsm_dates=dsm_dates,
        hansen=hansen,
        observation_year=observation_year,
    )

    # --- Step 3b: Texture features (GLCM from ortho) ---
    try:
        from texture_features import compute_texture_per_segment
        als_result_dict = {"transform": transform, "shape": (h, w)}
        tex_by_label = compute_texture_per_segment(
            labels, als_result_dict, year=ortho_year,
        )
        if tex_by_label:
            log.info("Step 3b: Merging texture features for %d segments", len(tex_by_label))
            for feat in features:
                lbl = feat["label"]
                tex = tex_by_label.get(lbl, {})
                for key in ("glcm_contrast", "glcm_homogeneity", "glcm_entropy",
                            "glcm_dissimilarity", "glcm_energy", "texture_complexity"):
                    feat[key] = tex.get(key, 0.0)
        else:
            log.info("Step 3b: No texture features available (ortho not found)")
    except Exception as e:
        log.warning("Step 3b: Texture computation failed: %s", e)

    # Free heavy intermediates before classification
    del cop_resampled
    import gc; gc.collect()

    # --- features_only: return segments + features without classification ---
    if features_only:
        log.info("features_only mode: returning %d features (skipping Steps 4-6)",
                 len(features))
        return {
            "objects": [],
            "features": features,
            "labels": labels,
            "stats": {"total_objects": 0, "n_features": len(features)},
            "evaluation": None,
        }

    # --- Step 4: Classification ---
    # Try learned RF classifier first, fall back to rules
    use_rf = False
    try:
        from learned_classifier import get_classifier
        rf_clf = get_classifier()
        if rf_clf.is_trained:
            use_rf = True
            log.info("Step 4: Object classification (RF classifier, OOB=%.3f)",
                     rf_clf.oob_score)
        else:
            log.info("Step 4: Object classification (rule-based, no RF model)")
    except Exception:
        log.info("Step 4: Object classification (rule-based)")

    # Batch RF prediction (much faster than per-segment)
    rf_results = {}  # label -> (type_name, type_code, conf, is_mm)
    if use_rf:
        import warnings as _w
        from learned_classifier import classify_with_rf_batch
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            rf_results = classify_with_rf_batch(
                features, has_spectral=has_spectral,
            )
        log.info("Step 4: RF batch classified %d segments", len(rf_results))

    objects = []
    for feat in features:
        if use_rf and feat["label"] in rf_results:
            type_name, type_code, conf, is_mm = rf_results[feat["label"]]
        else:
            type_name, type_code, conf, is_mm = classify_object(
                feat, has_spectral=has_spectral,
            )
        obj = SegmentedObject(
            obj_id=feat["label"],
            obj_type=type_name,
            type_code=type_code,
            area_sqm=float(feat["area"]),
            perimeter_m=feat["perimeter"],
            compactness=round(feat["compactness"], 3),
            elongation=round(feat["elongation"], 2),
            centroid_e=round(feat["centroid_e"], 1),
            centroid_n=round(feat["centroid_n"], 1),
            bbox=feat["bbox"],
            height_max=round(feat["h_max"], 2),
            height_mean=round(feat["h_mean"], 2),
            height_p90=round(feat["h_p90"], 2),
            height_std=round(feat["h_std"], 2),
            slope_mean=round(feat["slope_mean"], 1),
            roughness=round(feat["dsm_roughness"], 3),
            ndvi_mean=round(feat.get("ndvi_mean", 0), 4),
            ndvi_std=round(feat.get("ndvi_std", 0), 4),
            ndvi_fused=round(feat.get("fused_ndvi_mean", 0), 4),
            brightness_mean=round(feat.get("brightness_mean", 0), 1),
            nir_mean=round(feat.get("nir_mean", 0), 1),
            height_change=round(feat.get("h_change", 0), 3),
            dtm_change=round(feat.get("dtm_change", 0), 3),
            temporal_stability=round(feat.get("stability", 1), 3),
            volume_change_m3=round(feat.get("volume_change_m3", 0), 1),
            volume_change_abs_m3=round(feat.get("volume_change_abs_m3", 0), 1),
            dtm_change_max=round(feat.get("dtm_change_max", 0), 3),
            solidity=round(feat.get("solidity", 0), 3),
            extent=round(feat.get("extent", 0), 3),
            dsm_edge_strength=round(feat.get("dsm_edge_strength", 0), 3),
            glcm_entropy=round(feat.get("glcm_entropy", 0), 4),
            glcm_homogeneity=round(feat.get("glcm_homogeneity", 0), 4),
            texture_complexity=round(feat.get("texture_complexity", 0), 4),
            sar_vv=round(feat.get("sar_vv", 0), 4),
            sar_vh=round(feat.get("sar_vh", 0), 4),
            harm_amplitude=round(feat.get("harm_amplitude", 0), 4),
            harm_phase=round(feat.get("harm_phase", 0), 1),
            phenology_class=_phenology_class(feat),
            confidence=round(conf, 3),
            is_manmade=is_mm,
            features=feat,
        )
        objects.append(obj)

    # --- Step 5: Cadastre calibration ---
    if building_footprints is not None:
        log.info("Step 5: Cadastre calibration")
        objects = calibrate_with_cadastre(objects, labels, building_footprints)

    # --- Step 6: Hierarchical grouping ---
    log.info("Step 6: Hierarchical grouping")
    objects = group_objects(objects, labels)

    objects.sort(key=lambda o: o.area_sqm, reverse=True)

    # --- Evaluation ---
    evaluation = None
    if building_footprints is not None:
        evaluation = evaluate_against_cadastre(objects, labels, building_footprints)
        if "precision" in evaluation:
            log.info("Cadastre eval: P=%.3f R=%.3f F1=%.3f IoU=%.3f",
                     evaluation["precision"], evaluation["recall"],
                     evaluation["f1"], evaluation["iou"])
        else:
            log.info("Cadastre eval: skipped (%s)", evaluation.get("error", "no data"))

    stats = _compute_stats(objects)
    log.info("Done: %d objects (%d man-made, %d natural), %d groups",
             len(objects),
             sum(1 for o in objects if o.is_manmade),
             sum(1 for o in objects if not o.is_manmade),
             len(set(o.group_id for o in objects)))

    return {
        "objects": objects,
        "labels": labels,
        "stats": stats,
        "evaluation": evaluation,
    }


def _resample_copernicus(copernicus: dict, target_transform, target_shape: tuple) -> dict:
    """Resample Copernicus data from 10m to 1m grid."""
    result = {"ndvi": None, "landcover": None, "vv": None, "vh": None,
              "harmonics": None}
    try:
        from rasterio.warp import reproject, Resampling
        from rasterio.crs import CRS
    except ImportError:
        return result

    def _reproject_layer(src_arr, src_tf, src_crs, dtype=np.float32,
                         resamp=Resampling.bilinear, fill=np.nan):
        dst = np.full(target_shape, fill, dtype=dtype)
        reproject(
            source=src_arr.astype(dtype), destination=dst,
            src_transform=src_tf,
            src_crs=src_crs or CRS.from_epsg(4326),
            dst_transform=target_transform,
            dst_crs=CRS.from_epsg(3035),
            resampling=resamp,
        )
        return dst

    cop_tf = copernicus.get("transform")
    cop_crs = copernicus.get("crs")

    if copernicus.get("ndvi") is not None:
        try:
            result["ndvi"] = _reproject_layer(
                np.asarray(copernicus["ndvi"]), cop_tf, cop_crs)
        except Exception as e:
            log.warning("Failed to resample Copernicus NDVI: %s", e)

    if copernicus.get("landcover") is not None:
        try:
            lc_data = copernicus["landcover"]
            lc_info = lc_data if isinstance(lc_data, dict) else {"map": lc_data}
            result["landcover"] = _reproject_layer(
                lc_info["map"].astype(np.uint8),
                lc_info.get("transform", cop_tf),
                lc_info.get("crs", cop_crs),
                dtype=np.uint8, resamp=Resampling.nearest, fill=0)
        except Exception as e:
            log.warning("Failed to resample land cover: %s", e)

    # SAR backscatter (VV, VH)
    sar_tf = copernicus.get("sar_transform", cop_tf)
    sar_crs = copernicus.get("sar_crs", cop_crs)
    for band in ("vv", "vh"):
        if copernicus.get(band) is not None:
            try:
                result[band] = _reproject_layer(
                    np.asarray(copernicus[band]), sar_tf, sar_crs)
            except Exception as e:
                log.warning("Failed to resample SAR %s: %s", band, e)

    # NDVI harmonics (h_mean, h_amplitude, h_phase, h_rmse)
    if copernicus.get("harmonics") is not None:
        harm = copernicus["harmonics"]
        harm_tf = harm.get("transform", cop_tf)
        harm_crs = harm.get("crs", cop_crs)
        h_resampled = {}
        for key in ("h_mean", "h_amplitude", "h_phase", "h_rmse"):
            if key in harm:
                try:
                    h_resampled[key] = _reproject_layer(
                        np.asarray(harm[key]), harm_tf, harm_crs)
                except Exception as e:
                    log.warning("Failed to resample harmonic %s: %s", key, e)
        if h_resampled:
            result["harmonics"] = h_resampled

    return result


def _compute_stats(objects: list[SegmentedObject]) -> dict:
    """Summary statistics."""
    if not objects:
        return {"total_objects": 0, "by_type": {}, "by_group": {}}

    type_counts = Counter()
    type_areas = Counter()
    group_counts = Counter()
    group_areas = Counter()

    for obj in objects:
        type_counts[obj.obj_type] += 1
        type_areas[obj.obj_type] += obj.area_sqm
        if obj.group_type:
            group_counts[obj.group_type] += 1
            group_areas[obj.group_type] += obj.area_sqm

    total_area = sum(o.area_sqm for o in objects)
    by_type = {t: {"count": type_counts[t], "area_sqm": round(type_areas[t], 1),
                   "pct": round(100 * type_areas[t] / max(total_area, 1), 1)}
               for t in type_counts}
    by_group = {g: {"count": group_counts[g], "area_sqm": round(group_areas[g], 1),
                    "pct": round(100 * group_areas[g] / max(total_area, 1), 1)}
                for g in group_counts}

    return {
        "total_objects": len(objects),
        "total_area_sqm": round(total_area, 1),
        "total_area_ha": round(total_area / 10000, 3),
        "by_type": by_type,
        "by_group": by_group,
        "n_groups": len(set(o.group_id for o in objects)),
        "manmade_count": sum(1 for o in objects if o.is_manmade),
        "natural_count": sum(1 for o in objects if not o.is_manmade),
        "manmade_area_pct": round(
            100 * sum(o.area_sqm for o in objects if o.is_manmade) / max(total_area, 1), 1),
        "mean_confidence": round(
            sum(o.confidence for o in objects) / max(len(objects), 1), 3),
    }


# ===================================================================
# 9. BACKWARD-COMPATIBLE API
# ===================================================================

def make_type_raster(
    labels: np.ndarray,
    objects: list[SegmentedObject],
    shape: tuple[int, int],
) -> np.ndarray:
    """Pixel-level type code raster (backward compat with old 10-type codes)."""
    type_map = np.zeros(shape, dtype=np.uint8)
    for obj in objects:
        type_map[labels == obj.obj_id] = _COMPAT_MAP.get(obj.obj_type, 8)
    return type_map


def summarise_objects(objects: list[SegmentedObject]) -> dict:
    """Summary compatible with old API."""
    return _compute_stats(objects)
