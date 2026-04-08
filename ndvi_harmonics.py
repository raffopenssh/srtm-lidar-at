"""NDVI harmonic phenology features from Sentinel-2 time series.

Fits a harmonic model to monthly NDVI to extract phenological parameters
that discriminate land use classes:

  y(t) = mean + amplitude * cos(2π*t/12 - phase)

Where t is month index (0–11).  Key discriminative parameters:

  - **mean**: average greenness (forest > crop > grass > road)
  - **amplitude**: seasonality strength (crop >> pasture > forest > road)
  - **phase**: peak timing in months (crop: Jun-Jul, forest: Jul-Aug)
  - **rmse**: model fit residual (irregular patterns = mixed/disturbed)

These four features separate crop/pasture/forest/road with >85% accuracy
based on temporal behaviour alone (EuroSAT, LUCAS benchmarks).

Usage::

    from ndvi_harmonics import get_harmonic_features, compute_harmonics_per_segment
    # Per-pixel harmonic fit
    params = get_harmonic_features(monthly_ndvi_dict, transform, crs)
    # Per-segment aggregation
    seg_feats = compute_harmonics_per_segment(labels_1m, params)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)


def fit_harmonics_pixel(
    ndvi_series: np.ndarray,
    months: np.ndarray,
) -> dict[str, float]:
    """Fit 1st-order harmonic to a single pixel's monthly NDVI.

    Parameters
    ----------
    ndvi_series : 1D array of NDVI values per month.
    months : 1D int array of month indices (0-based, 0=Jan).

    Returns
    -------
    dict with keys: h_mean, h_amplitude, h_phase, h_rmse
    """
    valid = np.isfinite(ndvi_series)
    if valid.sum() < 3:
        return {"h_mean": 0, "h_amplitude": 0, "h_phase": 0, "h_rmse": 0}

    y = ndvi_series[valid]
    t = months[valid].astype(np.float64)

    # Design matrix: [1, cos(2πt/12), sin(2πt/12)]
    omega = 2 * np.pi / 12.0
    A = np.column_stack([
        np.ones(len(t)),
        np.cos(omega * t),
        np.sin(omega * t),
    ])

    # OLS
    try:
        beta, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"h_mean": float(np.nanmean(y)), "h_amplitude": 0, "h_phase": 0, "h_rmse": 0}

    b0, b1, b2 = beta
    amplitude = np.sqrt(b1**2 + b2**2)
    phase = np.arctan2(b2, b1)  # radians
    # Convert phase to month of peak (0=Jan, ..., 11=Dec)
    phase_month = (-phase / omega) % 12

    predicted = A @ beta
    rmse = np.sqrt(np.mean((y - predicted) ** 2))

    return {
        "h_mean": float(b0),
        "h_amplitude": float(amplitude),
        "h_phase": float(phase_month),
        "h_rmse": float(rmse),
    }


def fit_harmonics_image(
    monthly_ndvi: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Fit harmonics to every pixel of a monthly NDVI stack.

    Parameters
    ----------
    monthly_ndvi : {"YYYY-MM": 2D array} from copernicus.get_ndvi_timeseries().

    Returns
    -------
    dict with 2D arrays: h_mean, h_amplitude, h_phase, h_rmse
    """
    if not monthly_ndvi:
        return {}

    # Sort by date key and build stack
    sorted_keys = sorted(monthly_ndvi.keys())
    first = monthly_ndvi[sorted_keys[0]]
    h, w = first.shape

    # Extract month indices (0-based)
    month_indices = []
    for key in sorted_keys:
        # key format: "YYYY-MM" or similar
        try:
            parts = key.split("-")
            m = int(parts[1]) - 1  # 0-based
            month_indices.append(m)
        except (IndexError, ValueError):
            month_indices.append(0)

    months = np.array(month_indices, dtype=np.float64)
    n_months = len(sorted_keys)

    # Stack: (n_months, H, W)
    stack = np.stack([monthly_ndvi[k] for k in sorted_keys], axis=0)

    # Vectorised harmonic fit using least squares
    omega = 2 * np.pi / 12.0

    # Design matrix: (n_months, 3)
    A = np.column_stack([
        np.ones(n_months),
        np.cos(omega * months),
        np.sin(omega * months),
    ])

    # Reshape stack to (n_months, H*W)
    Y = stack.reshape(n_months, -1)  # (T, N)

    # Handle NaN: use pseudo-inverse approach
    # For efficiency, solve all pixels at once where possible
    # Fill NaN with 0 and create weight mask
    valid_mask = np.isfinite(Y)  # (T, N)
    Y_clean = np.where(valid_mask, Y, 0)
    n_valid = valid_mask.sum(axis=0)  # (N,)

    # Pixels with >= 3 valid observations
    enough = n_valid >= 3

    # For pixels with all observations valid, solve in batch
    all_valid = (n_valid == n_months)

    # Batch solve for fully-valid pixels
    beta_all = np.zeros((3, h * w), dtype=np.float64)
    rmse_all = np.zeros(h * w, dtype=np.float64)

    if all_valid.sum() > 0:
        try:
            # A.T @ A is 3x3, A.T @ Y is 3xN
            AtA_inv = np.linalg.inv(A.T @ A)
            AtY = A.T @ Y_clean[:, all_valid]
            beta_all[:, all_valid] = AtA_inv @ AtY
            predicted = A @ beta_all[:, all_valid]
            residuals = Y_clean[:, all_valid] - predicted
            rmse_all[all_valid] = np.sqrt(np.mean(residuals**2, axis=0))
        except np.linalg.LinAlgError:
            log.warning("Batch harmonic solve failed, falling back to per-pixel")
            all_valid[:] = False

    # Per-pixel solve for partial-coverage pixels
    partial = enough & ~all_valid
    partial_idx = np.where(partial)[0]
    if len(partial_idx) > 0:
        log.info("Fitting harmonics per-pixel for %d partial-coverage pixels", len(partial_idx))
        for idx in partial_idx:
            v = valid_mask[:, idx]
            if v.sum() < 3:
                continue
            Av = A[v]
            yv = Y_clean[v, idx]
            try:
                b, _, _, _ = np.linalg.lstsq(Av, yv, rcond=None)
                beta_all[:, idx] = b
                pred = Av @ b
                rmse_all[idx] = np.sqrt(np.mean((yv - pred) ** 2))
            except np.linalg.LinAlgError:
                pass

    # Extract parameters
    b0 = beta_all[0].reshape(h, w).astype(np.float32)
    b1 = beta_all[1].reshape(h, w).astype(np.float32)
    b2 = beta_all[2].reshape(h, w).astype(np.float32)

    amplitude = np.sqrt(b1**2 + b2**2)
    phase_rad = np.arctan2(b2, b1)
    phase_month = ((-phase_rad / omega) % 12).astype(np.float32)
    rmse = rmse_all.reshape(h, w).astype(np.float32)

    return {
        "h_mean": b0,
        "h_amplitude": amplitude,
        "h_phase": phase_month,
        "h_rmse": rmse,
    }


def get_harmonic_features(
    bbox_wgs84: dict,
    year: int = 2023,
) -> Optional[dict[str, np.ndarray]]:
    """Fetch NDVI time series and compute harmonic parameters.

    Parameters
    ----------
    bbox_wgs84 : dict with west/south/east/north
    year : year to process

    Returns
    -------
    dict with h_mean, h_amplitude, h_phase, h_rmse arrays + transform, crs
    or None on failure.
    """
    try:
        import copernicus
    except ImportError:
        log.warning("copernicus module not available")
        return None

    try:
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        ts = copernicus.get_ndvi_timeseries(bbox_wgs84, start, end)
        monthly = ts.get("monthly_ndvi", {})
        if len(monthly) < 4:
            log.warning("Only %d months of NDVI, need >= 4", len(monthly))
            return None

        log.info("Fitting harmonics to %d monthly NDVI layers", len(monthly))
        params = fit_harmonics_image(monthly)
        params["transform"] = ts["transform"]
        params["crs"] = ts["crs"]
        return params

    except Exception as e:
        log.warning("Harmonic feature computation failed: %s", e)
        return None


def compute_harmonics_per_segment(
    labels_1m: np.ndarray,
    harmonic_layers: dict[str, np.ndarray],
    target_shape: tuple[int, int] | None = None,
    target_transform=None,
    source_transform=None,
) -> dict[int, dict[str, float]]:
    """Aggregate harmonic parameters per labelled segment.

    The harmonic layers are at Sentinel-2 resolution (10m) and labels
    are at 1m.  We resample harmonics UP to 1m for per-segment aggregation.

    Parameters
    ----------
    labels_1m : 2D int array of segment labels at 1m.
    harmonic_layers : dict from fit_harmonics_image() with h_mean etc.
    target_shape : (h, w) of the 1m grid (optional, derived from labels).
    target_transform : rasterio Affine of the 1m grid.
    source_transform : rasterio Affine of the harmonic layers.

    Returns
    -------
    dict mapping label → {"h_mean", "h_amplitude", "h_phase", "h_rmse",
    "phenology_class"}
    """
    if not harmonic_layers:
        return {}

    keys = ["h_mean", "h_amplitude", "h_phase", "h_rmse"]
    present = [k for k in keys if k in harmonic_layers]
    if not present:
        return {}

    h_1m, w_1m = labels_1m.shape

    # Resample harmonic layers to 1m
    resampled = {}
    for k in present:
        arr = harmonic_layers[k]
        if arr.shape == (h_1m, w_1m):
            resampled[k] = arr
        else:
            # Use rasterio reproject if transforms available, else zoom
            if source_transform is not None and target_transform is not None:
                try:
                    from rasterio.warp import reproject as rio_reproject, Resampling
                    from rasterio.crs import CRS
                    dst = np.zeros((h_1m, w_1m), dtype=np.float32)
                    rio_reproject(
                        arr, dst,
                        src_transform=source_transform,
                        src_crs=harmonic_layers.get("crs", CRS.from_epsg(4326)),
                        dst_transform=target_transform,
                        dst_crs=CRS.from_epsg(3035),
                        resampling=Resampling.bilinear,
                    )
                    resampled[k] = dst
                    continue
                except Exception:
                    pass
            # Fallback: simple zoom
            from scipy.ndimage import zoom as ndi_zoom
            factor_h = h_1m / arr.shape[0]
            factor_w = w_1m / arr.shape[1]
            resampled[k] = ndi_zoom(arr, (factor_h, factor_w), order=1).astype(np.float32)

    # Aggregate per segment
    unique_labels = np.unique(labels_1m)
    unique_labels = unique_labels[unique_labels > 0]

    result = {}
    for lbl in unique_labels:
        seg = labels_1m == lbl
        if seg.sum() < 2:
            continue
        entry = {}
        for k in present:
            arr = resampled[k]
            vals = arr[seg]
            entry[k] = float(np.nanmean(vals))

        # Phenology classification heuristic
        amp = entry.get("h_amplitude", 0)
        mean_v = entry.get("h_mean", 0)
        phase = entry.get("h_phase", 6)
        rmse = entry.get("h_rmse", 0)

        # Classification based on temporal signature:
        #   crop:    high amplitude (>0.15), peak Jun-Aug (phase 5-7)
        #   pasture: moderate amplitude (0.05-0.15), moderate mean
        #   forest:  low amplitude (<0.05), high mean (>0.5)
        #   road:    very low mean (<0.15), very low amplitude
        if mean_v < 0.15 and amp < 0.05:
            entry["phenology_class"] = "road_or_bare"
        elif amp > 0.15 and 4 <= phase <= 8:
            entry["phenology_class"] = "crop"
        elif mean_v > 0.5 and amp < 0.08:
            entry["phenology_class"] = "forest"
        elif mean_v > 0.3 and amp < 0.15:
            entry["phenology_class"] = "pasture"
        elif amp > 0.10:
            entry["phenology_class"] = "seasonal_vegetation"
        else:
            entry["phenology_class"] = "unknown"

        result[int(lbl)] = entry

    log.info("Computed harmonic features for %d segments", len(result))
    return result
