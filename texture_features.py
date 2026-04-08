"""GLCM texture features from high-resolution BEV orthophotos.

Extracts Grey-Level Co-occurrence Matrix (GLCM) texture descriptors
from 20cm orthophotos and aggregates them per 1m segment.  This
provides the discriminative power to separate roads from grass from
forest at ground level — something spectral means alone cannot do.

Key insight: at 20cm, a paved road has very low GLCM entropy and high
homogeneity.  Grass has medium entropy.  Tree canopy has high entropy
and high contrast.  These features are orthogonal to NDVI.

Usage::

    from texture_features import compute_texture_per_segment
    tex = compute_texture_per_segment(labels, rgb_1m, transform,
                                       als_result, ortho_resolution=0.5)
    # tex[label_id] = {"glcm_contrast": float, "glcm_entropy": float, ...}
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)

# We compute GLCM on a quantised grey image.  64 levels balances
# statistical stability (fewer empty bins) with discriminative power.
GLCM_LEVELS = 64
GLCM_DISTANCES = [1, 3]       # 1px and 3px offsets at working res
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 4 directions

# Working resolution for texture.  0.5m is a sweet spot:
# - 2.5× more detail than 1m (roads ~4-6px wide at 0.5m)
# - 25× less data than native 0.2m (manageable memory)
# - GLCM at d=1 captures 0.5m-scale texture (pavement grain, leaf clumps)
DEFAULT_TEXTURE_RES = 0.5


def compute_texture_layers(
    grey: np.ndarray,
    *,
    levels: int = GLCM_LEVELS,
    distances: list[int] | None = None,
    angles: list[float] | None = None,
    window_size: int = 11,
) -> dict[str, np.ndarray]:
    """Compute per-pixel GLCM texture over a sliding window.

    Parameters
    ----------
    grey : 2D uint8 array, quantised to [0, levels).
    levels : number of grey levels (default 64).
    distances : GLCM pixel offsets.
    angles : GLCM directions (radians).
    window_size : side length of the sliding window (must be odd).

    Returns
    -------
    dict of 2D float32 arrays, same shape as *grey*:
        - glcm_contrast : local intensity variation
        - glcm_homogeneity : inverse difference moment (high = uniform)
        - glcm_entropy : Shannon entropy of co-occurrence (high = complex)
        - glcm_dissimilarity : mean |i-j| over co-occurrence
        - glcm_energy : angular second moment (high = few dominant pairs)
    """
    from skimage.feature import graycomatrix, graycoprops

    if distances is None:
        distances = GLCM_DISTANCES
    if angles is None:
        angles = GLCM_ANGLES

    h, w = grey.shape
    pad = window_size // 2

    # Output arrays
    contrast = np.zeros((h, w), dtype=np.float32)
    homogeneity = np.zeros((h, w), dtype=np.float32)
    entropy = np.zeros((h, w), dtype=np.float32)
    dissimilarity = np.zeros((h, w), dtype=np.float32)
    energy = np.zeros((h, w), dtype=np.float32)

    # Pad image
    padded = np.pad(grey, pad, mode='reflect')

    # Stride: compute every 2nd pixel for speed, interpolate
    stride = 2
    rows = list(range(0, h, stride))
    cols = list(range(0, w, stride))

    for ri in rows:
        for ci in cols:
            patch = padded[ri:ri + window_size, ci:ci + window_size]
            glcm = graycomatrix(patch, distances=distances, angles=angles,
                                levels=levels, symmetric=True, normed=True)
            # Average over all distance/angle combinations
            contrast[ri, ci] = graycoprops(glcm, 'contrast').mean()
            homogeneity[ri, ci] = graycoprops(glcm, 'homogeneity').mean()
            dissimilarity[ri, ci] = graycoprops(glcm, 'dissimilarity').mean()
            energy[ri, ci] = graycoprops(glcm, 'energy').mean()

            # Entropy: -sum(p * log2(p))
            p = glcm.mean(axis=(2, 3))  # average over dist/angle
            p = p / (p.sum() + 1e-10)
            entropy[ri, ci] = -np.sum(p * np.log2(p + 1e-10))

    # Fill gaps from stride
    if stride > 1:
        from scipy.ndimage import zoom as ndi_zoom
        for name, arr in [("c", contrast), ("h", homogeneity),
                          ("e", entropy), ("d", dissimilarity),
                          ("n", energy)]:
            # Simple nearest-neighbour fill for strided gaps
            small = arr[::stride, ::stride]
            zoomed = ndi_zoom(small, (h / small.shape[0], w / small.shape[1]),
                              order=1)
            arr[:] = zoomed[:h, :w]

    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_entropy": entropy,
        "glcm_dissimilarity": dissimilarity,
        "glcm_energy": energy,
    }


def _quantise_grey(arr: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    """Convert float or uint8 image to quantised uint8 in [0, levels)."""
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - mn) / (mx - mn) * (levels - 1)
    return np.clip(scaled, 0, levels - 1).astype(np.uint8)


def read_ortho_for_texture(
    als_result: dict,
    resolution: float = DEFAULT_TEXTURE_RES,
    year: int | None = None,
) -> Optional[np.ndarray]:
    """Read orthophoto at texture resolution, return grey image.

    Returns a 2D uint8 array at *resolution* (default 0.5m), or None
    if no ortho is available.
    """
    try:
        import ortho_io
    except ImportError:
        log.warning("ortho_io not available for texture")
        return None

    tf = als_result["transform"]
    h1, w1 = als_result["shape"]
    res1 = abs(tf.a)  # 1m

    min_e = tf.c
    max_n = tf.f
    max_e = min_e + w1 * res1
    min_n = max_n - h1 * res1

    # Desired output size at texture resolution
    h_tex = int(round((max_n - min_n) / resolution))
    w_tex = int(round((max_e - min_e) / resolution))

    # Try RGBI operates first
    rgb, nir = ortho_io._try_read_rgbi_for_bbox(
        min_e, min_n, max_e, max_n,
        resolution, h_tex, w_tex,
        year=year,
    )
    if rgb is None:
        # Fallback: DOP tiles
        try:
            rgb, _, _ = ortho_io.read_ortho_window(
                min_e, min_n, max_e, max_n,
                resolution=resolution,
            )
        except Exception as e:
            log.warning("Texture ortho read failed: %s", e)
            return None

    if rgb is None:
        return None

    # Convert to greyscale: standard luminance weights
    r = rgb[0].astype(np.float32)
    g = rgb[1].astype(np.float32)
    b = rgb[2].astype(np.float32)
    grey = (0.299 * r + 0.587 * g + 0.114 * b)
    return grey.astype(np.float32)


def compute_texture_per_segment(
    labels_1m: np.ndarray,
    als_result: dict,
    *,
    resolution: float = DEFAULT_TEXTURE_RES,
    year: int | None = None,
    max_pixels: int = 2_000_000,
) -> dict[int, dict[str, float]]:
    """Compute GLCM texture features per labelled segment.

    Reads the ortho at *resolution*, computes texture layers, then
    aggregates per segment (upscaling labels to texture grid).

    Parameters
    ----------
    labels_1m : 2D int array of segment labels at 1m.
    als_result : dict with "transform", "shape" keys.
    resolution : texture working resolution (default 0.5m).
    year : ortho year override.
    max_pixels : skip texture if image exceeds this (memory guard).

    Returns
    -------
    dict mapping label → {"glcm_contrast", "glcm_homogeneity",
    "glcm_entropy", "glcm_dissimilarity", "glcm_energy"}.
    Empty dict if ortho unavailable.
    """
    grey = read_ortho_for_texture(als_result, resolution=resolution, year=year)
    if grey is None:
        log.info("No ortho available for texture features")
        return {}

    h_tex, w_tex = grey.shape
    total_px = h_tex * w_tex
    if total_px > max_pixels:
        # Downsample to stay within budget
        factor = (max_pixels / total_px) ** 0.5
        new_h = max(10, int(h_tex * factor))
        new_w = max(10, int(w_tex * factor))
        from scipy.ndimage import zoom as ndi_zoom
        grey = ndi_zoom(grey, (new_h / h_tex, new_w / w_tex), order=1)
        resolution = resolution / factor
        h_tex, w_tex = grey.shape
        log.info("Downsampled texture image to %dx%d (%.2fm)", w_tex, h_tex, resolution)

    log.info("Computing GLCM texture at %.2fm resolution (%dx%d)",
             resolution, w_tex, h_tex)

    # Quantise
    grey_q = _quantise_grey(grey, GLCM_LEVELS)

    # Compute texture layers
    tex_layers = compute_texture_layers(grey_q, levels=GLCM_LEVELS)

    # Upscale labels to texture grid
    scale = abs(als_result["transform"].a) / resolution  # e.g. 1.0/0.5 = 2
    from scipy.ndimage import zoom as ndi_zoom
    labels_tex = ndi_zoom(labels_1m.astype(np.float64),
                          (h_tex / labels_1m.shape[0], w_tex / labels_1m.shape[1]),
                          order=0).astype(np.int32)

    # Aggregate per segment
    unique_labels = np.unique(labels_tex)
    unique_labels = unique_labels[unique_labels > 0]

    result = {}
    for lbl in unique_labels:
        seg = labels_tex == lbl
        if seg.sum() < 4:
            continue
        entry = {}
        for key, layer in tex_layers.items():
            vals = layer[seg]
            entry[key] = float(np.nanmean(vals))
            entry[f"{key}_std"] = float(np.nanstd(vals))
        # Derived: texture heterogeneity ratio (entropy / homogeneity)
        hom = entry.get("glcm_homogeneity", 0.001)
        entry["texture_complexity"] = entry.get("glcm_entropy", 0) / max(hom, 0.001)
        result[int(lbl)] = entry

    log.info("Computed texture for %d segments", len(result))
    return result
