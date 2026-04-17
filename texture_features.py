"""GLCM texture features from high-resolution BEV orthophotos.

Extracts Grey-Level Co-occurrence Matrix (GLCM) texture descriptors
from orthophotos and aggregates them per segment.  Uses per-segment
GLCM computation (one GLCM per segment, not per pixel) for speed.

Key insight: at 0.5m, a paved road has very low GLCM entropy and high
homogeneity.  Grass has medium entropy.  Tree canopy has high entropy
and high contrast.  These features are orthogonal to NDVI.

Usage::

    from texture_features import compute_texture_per_segment
    tex = compute_texture_per_segment(labels, als_result, year=2024)
    # tex[label_id] = {"glcm_contrast": float, "glcm_entropy": float, ...}
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

GLCM_LEVELS = 32          # fewer levels = faster, still discriminative
GLCM_DISTANCES = [1, 2]   # pixel offsets at working resolution
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
DEFAULT_TEXTURE_RES = 0.5


def _quantise_grey(arr: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    """Convert image to quantised uint8 in [0, levels)."""
    arr = np.asarray(arr, dtype=np.float32)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - mn) / (mx - mn) * (levels - 1)
    return np.clip(scaled, 0, levels - 1).astype(np.uint8)


def _glcm_features_for_patch(patch: np.ndarray, levels: int = GLCM_LEVELS) -> dict[str, float]:
    """Compute GLCM features for a single image patch (one segment's pixels).

    Uses skimage graycomatrix on the bounding-box crop of the segment.
    Much faster than per-pixel sliding window.
    """
    from skimage.feature import graycomatrix, graycoprops

    if patch.size < 16:  # too small for meaningful texture
        return {"glcm_contrast": 0, "glcm_homogeneity": 0, "glcm_entropy": 0,
                "glcm_dissimilarity": 0, "glcm_energy": 0}

    glcm = graycomatrix(patch, distances=GLCM_DISTANCES, angles=GLCM_ANGLES,
                        levels=levels, symmetric=True, normed=True)

    contrast = float(graycoprops(glcm, 'contrast').mean())
    homogeneity = float(graycoprops(glcm, 'homogeneity').mean())
    dissimilarity = float(graycoprops(glcm, 'dissimilarity').mean())
    energy = float(graycoprops(glcm, 'energy').mean())

    # Entropy: -sum(p * log2(p)) averaged over dist/angle
    p = glcm.mean(axis=(2, 3))
    p = p / (p.sum() + 1e-10)
    entropy = float(-np.sum(p * np.log2(p + 1e-10)))

    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_entropy": entropy,
        "glcm_dissimilarity": dissimilarity,
        "glcm_energy": energy,
    }


def read_ortho_for_texture(
    als_result: dict,
    resolution: float = DEFAULT_TEXTURE_RES,
    year: int | None = None,
) -> Optional[np.ndarray]:
    """Read orthophoto at texture resolution, return grey float32 image."""
    try:
        import ortho_io
    except ImportError:
        log.warning("ortho_io not available for texture")
        return None

    tf = als_result["transform"]
    h1, w1 = als_result["shape"]
    res1 = abs(tf.a)

    min_e = tf.c
    max_n = tf.f
    max_e = min_e + w1 * res1
    min_n = max_n - h1 * res1

    h_tex = int(round((max_n - min_n) / resolution))
    w_tex = int(round((max_e - min_e) / resolution))

    rgb, nir = ortho_io._try_read_rgbi_for_bbox(
        min_e, min_n, max_e, max_n,
        resolution, h_tex, w_tex, year=year,
    )
    if rgb is None:
        try:
            rgb, _, _ = ortho_io.read_ortho_window(
                min_e, min_n, max_e, max_n, resolution=resolution,
            )
        except Exception as e:
            log.warning("Texture ortho read failed: %s", e)
            return None

    if rgb is None:
        return None

    r = rgb[0].astype(np.float32)
    g = rgb[1].astype(np.float32)
    b = rgb[2].astype(np.float32)
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def compute_texture_per_segment(
    labels_1m: np.ndarray,
    als_result: dict,
    *,
    resolution: float = DEFAULT_TEXTURE_RES,
    year: int | None = None,
    max_pixels: int = 4_000_000,
    grey_image: np.ndarray | None = None,
) -> dict[int, dict[str, float]]:
    """Compute GLCM texture features per labelled segment.

    For each segment: extract its bounding-box crop from the grey image,
    compute one GLCM, derive texture features.  Much faster than per-pixel.

    Parameters
    ----------
    labels_1m : 2D int array of segment labels at 1m.
    als_result : dict with "transform", "shape" keys.
    resolution : texture working resolution (default 0.5m).
    year : ortho year override.
    max_pixels : skip if image too large.
    grey_image : pre-loaded greyscale image (avoids re-reading ortho).

    Returns
    -------
    dict mapping label → texture feature dict.
    """
    if grey_image is not None:
        grey = grey_image
        log.info("Using pre-loaded grey image for texture (%dx%d)", grey.shape[1], grey.shape[0])
    else:
        grey = read_ortho_for_texture(als_result, resolution=resolution, year=year)
    if grey is None:
        log.info("No ortho available for texture features")
        return {}

    h_tex, w_tex = grey.shape
    total_px = h_tex * w_tex
    if total_px > max_pixels:
        factor = (max_pixels / total_px) ** 0.5
        new_h = max(10, int(h_tex * factor))
        new_w = max(10, int(w_tex * factor))
        from scipy.ndimage import zoom as ndi_zoom
        grey = ndi_zoom(grey, (new_h / h_tex, new_w / w_tex), order=1)
        resolution = resolution / factor
        h_tex, w_tex = grey.shape
        log.info("Downsampled texture to %dx%d (%.2fm)", w_tex, h_tex, resolution)

    log.info("Computing per-segment GLCM texture at %.2fm (%dx%d)",
             resolution, w_tex, h_tex)

    grey_q = _quantise_grey(grey, GLCM_LEVELS)

    # Upscale labels to texture grid (nearest-neighbour)
    from scipy.ndimage import zoom as ndi_zoom
    labels_tex = ndi_zoom(labels_1m.astype(np.float64),
                          (h_tex / labels_1m.shape[0], w_tex / labels_1m.shape[1]),
                          order=0).astype(np.int32)

    # Per-segment: extract bbox crop, compute GLCM
    from skimage import measure
    regions = measure.regionprops(labels_tex)

    result = {}
    for reg in regions:
        if reg.area < 16:  # need minimum pixels for GLCM
            continue
        lbl = reg.label
        # Bounding box crop
        r0, c0, r1, c1 = reg.bbox
        crop = grey_q[r0:r1, c0:c1].copy()
        # Mask out pixels not in this segment (set to 0)
        seg_crop = labels_tex[r0:r1, c0:c1] == lbl
        crop[~seg_crop] = 0

        entry = _glcm_features_for_patch(crop, GLCM_LEVELS)
        # Add std variants
        for key in list(entry.keys()):
            entry[f"{key}_std"] = 0.0  # per-segment GLCM has no spatial std
        # Derived: texture complexity
        hom = max(entry.get("glcm_homogeneity", 0.001), 0.001)
        entry["texture_complexity"] = entry.get("glcm_entropy", 0) / hom
        result[int(lbl)] = entry

    log.info("Computed texture for %d segments", len(result))
    return result
