"""Terrain characterisation from DTM: slope, aspect, ruggedness, curvature."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def compute_slope(dtm: np.ndarray, res: float = 1.0) -> np.ndarray:
    """Slope in degrees from a DTM grid."""
    dy, dx = np.gradient(dtm, res)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad)


def compute_aspect(dtm: np.ndarray, res: float = 1.0) -> np.ndarray:
    """Aspect in degrees (0=N, 90=E, 180=S, 270=W). -1 for flat areas."""
    dy, dx = np.gradient(dtm, res)
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    # Mark flat areas
    flat = (np.abs(dx) < 1e-6) & (np.abs(dy) < 1e-6)
    aspect[flat] = -1
    return aspect


def compute_tri(dtm: np.ndarray) -> np.ndarray:
    """Terrain Ruggedness Index (Riley et al. 1999).
    Mean absolute elevation difference to 8 neighbours."""
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    sum_sq_diff = np.zeros_like(dtm, dtype=np.float64)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            shifted = ndimage.shift(dtm, (dr, dc), mode='nearest')
            sum_sq_diff += (dtm - shifted) ** 2
    return np.sqrt(sum_sq_diff / 8.0).astype(np.float32)


def compute_tpi(dtm: np.ndarray, radius: int = 10) -> np.ndarray:
    """Topographic Position Index: elevation minus mean of neighbourhood."""
    kernel_size = 2 * radius + 1
    mean = ndimage.uniform_filter(dtm.astype(np.float64), size=kernel_size, mode='nearest')
    return (dtm - mean).astype(np.float32)


def compute_curvature(dtm: np.ndarray, res: float = 1.0) -> dict[str, np.ndarray]:
    """Profile and plan curvature."""
    dy, dx = np.gradient(dtm.astype(np.float64), res)
    dyy, dyx = np.gradient(dy, res)
    dxy, dxx = np.gradient(dx, res)

    p = dx**2 + dy**2
    q = p + 1

    # Profile curvature (in direction of steepest slope)
    profile = np.where(
        p > 1e-10,
        -(dx**2 * dxx + 2 * dx * dy * dxy + dy**2 * dyy) / (p * np.sqrt(q)**3),
        0.0
    ).astype(np.float32)

    # Plan curvature (perpendicular to slope)
    plan = np.where(
        p > 1e-10,
        -(dy**2 * dxx - 2 * dx * dy * dxy + dx**2 * dyy) / (p**1.5),
        0.0
    ).astype(np.float32)

    return {"profile_curvature": profile, "plan_curvature": plan}


def characterise_terrain(dtm: np.ndarray, mask: np.ndarray, res: float = 1.0) -> dict:
    """Full terrain characterisation. Returns stats dict."""
    slope = compute_slope(dtm, res)
    aspect = compute_aspect(dtm, res)
    tri = compute_tri(dtm)
    tpi = compute_tpi(dtm)

    valid = mask & ~np.isnan(dtm)
    elev_valid = dtm[valid]
    slope_valid = slope[valid]
    tri_valid = tri[valid]

    # Strip NaN values that propagate from DTM edge pixels through gradient ops
    tri_valid = tri_valid[np.isfinite(tri_valid)]
    slope_valid = slope_valid[np.isfinite(slope_valid)]
    elev_valid = elev_valid[np.isfinite(elev_valid)]

    def pct(arr, p):
        return float(np.nanpercentile(arr, p)) if len(arr) > 0 else None

    # Aspect distribution (8 cardinal directions)
    aspect_valid = aspect[valid]
    aspect_bins = {}
    dirs = [("N", 0, 45), ("NE", 45, 90), ("E", 90, 135), ("SE", 135, 180),
            ("S", 180, 225), ("SW", 225, 270), ("W", 270, 315), ("NW", 315, 360)]
    for name, lo, hi in dirs:
        count = int(np.sum((aspect_valid >= lo) & (aspect_valid < hi)))
        aspect_bins[name] = round(count / max(len(aspect_valid), 1) * 100, 1)

    # Slope classification
    slope_classes = {
        "flat_0_2deg": float(np.sum(slope_valid < 2) / max(len(slope_valid), 1) * 100),
        "gentle_2_5deg": float(np.sum((slope_valid >= 2) & (slope_valid < 5)) / max(len(slope_valid), 1) * 100),
        "moderate_5_15deg": float(np.sum((slope_valid >= 5) & (slope_valid < 15)) / max(len(slope_valid), 1) * 100),
        "steep_15_30deg": float(np.sum((slope_valid >= 15) & (slope_valid < 30)) / max(len(slope_valid), 1) * 100),
        "very_steep_30_45deg": float(np.sum((slope_valid >= 30) & (slope_valid < 45)) / max(len(slope_valid), 1) * 100),
        "extreme_above_45deg": float(np.sum(slope_valid >= 45) / max(len(slope_valid), 1) * 100),
    }

    return {
        "elevation": {
            "min": round(float(np.nanmin(elev_valid)), 2) if len(elev_valid) > 0 else None,
            "max": round(float(np.nanmax(elev_valid)), 2) if len(elev_valid) > 0 else None,
            "mean": round(float(np.nanmean(elev_valid)), 2) if len(elev_valid) > 0 else None,
            "std": round(float(np.nanstd(elev_valid)), 2) if len(elev_valid) > 0 else None,
            "range": round(float(np.nanmax(elev_valid) - np.nanmin(elev_valid)), 2) if len(elev_valid) > 0 else None,
            "p10": round(pct(elev_valid, 10), 2),
            "p50": round(pct(elev_valid, 50), 2),
            "p90": round(pct(elev_valid, 90), 2),
        },
        "slope_deg": {
            "min": round(float(np.nanmin(slope_valid)), 2) if len(slope_valid) > 0 else None,
            "max": round(float(np.nanmax(slope_valid)), 2) if len(slope_valid) > 0 else None,
            "mean": round(float(np.nanmean(slope_valid)), 2) if len(slope_valid) > 0 else None,
            "std": round(float(np.nanstd(slope_valid)), 2) if len(slope_valid) > 0 else None,
        },
        "slope_classes_pct": slope_classes,
        "aspect_distribution_pct": aspect_bins,
        "ruggedness_tri": {
            "mean": round(float(np.nanmean(tri_valid)), 3) if len(tri_valid) > 0 else None,
            "max": round(float(np.nanmax(tri_valid)), 3) if len(tri_valid) > 0 else None,
            "classification": _classify_tri(float(np.nanmean(tri_valid))) if len(tri_valid) > 0 else None,
        },
        "area_sqm": int(np.sum(valid)),
        "area_ha": round(int(np.sum(valid)) / 10000, 2),
    }


def _classify_tri(mean_tri: float) -> str:
    if mean_tri < 0.1:
        return "level"
    elif mean_tri < 0.3:
        return "nearly_level"
    elif mean_tri < 0.8:
        return "slightly_rugged"
    elif mean_tri < 1.5:
        return "intermediately_rugged"
    elif mean_tri < 3.0:
        return "moderately_rugged"
    elif mean_tri < 6.0:
        return "highly_rugged"
    else:
        return "extremely_rugged"
