# Reference Algorithm Summary for Watershed-Based Object Segmentation

Two references analyzed for algorithmic ideas applicable to landscape classification
using DTM, DSM, ortho imagery (RGBI), and Sentinel-2 NDVI.

---

## Reference 1: ec-jrc/nrt — Near Real-Time Change Detection

**Source**: https://github.com/ec-jrc/nrt

### Core Architecture

All five monitoring frameworks share a **two-phase** design:

1. **History Fitting Phase**: Fit a harmonic-trend regression model on a stable
   history period to define "normal" behavior
2. **Monitoring Phase**: For each new observation, compare observed vs. predicted
   values; accumulate evidence of divergence until a "break" is confirmed

### The Harmonic-Trend Model (Design Matrix)

The regression model fitted to each pixel's time series:

```
y(t) = β₀ + β₁·t + Σₖ [β₂ₖ·sin(2πk·t/T) + β₂ₖ₊₁·cos(2πk·t/T)] + ε(t)
```

Where:
- `β₀` = intercept
- `β₁·t` = linear trend (optional, controlled by `trend=True/False`)
- `k` = harmonic order (1, 2, or 3), `T` = annual period
- `ε(t)` = residual

This is implemented via `build_regressors()` which constructs design matrix `X`,
then solved by OLS, RIRLS, ROC, or CCDC-stable fitting methods.

### EWMA Algorithm (Brooks et al. 2014) — Most Relevant for Your Use Case

**Fitting:**
1. Fit harmonic model on history period using OLS
2. Screen outliers using Shewhart control chart (values > L×σ removed)
3. Compute σ (std of residuals) and control boundary:
   ```
   boundary = sensitivity × σ × sqrt(λ / (2 - λ))
   ```
4. Initialize EWMA process value by running recursion over history residuals

**Monitoring (per new observation):**
1. Predict expected value: `y_pred = X·β`
2. Compute residual: `r = y_observed - y_pred`
3. Screen extreme outliers: skip if `|r| > threshold × σ`
4. Update EWMA: `ewma_new = (1 - λ)·ewma_old + λ·r`
5. Detect break: `|ewma| ≥ boundary`

Key parameters:
- `λ = 0.3`: weight of current observation (0=full memory, 1=no memory)
- `sensitivity = 2`: lower = more sensitive
- `threshold_outlier = 10`: extreme outlier screening

### CCDC Algorithm (Zhu & Woodcock 2014)

**Fitting:**
1. Uses CCDC-stable fit with RIRLS (Reweighted Iteratively Reweighted Least Squares)
2. Outlier screening uses green and SWIR bands to identify clouds/shadows
3. Computes RMSE per pixel from residuals

**Monitoring:**
1. For each new observation, compute `|residual| / RMSE`
2. If ratio > sensitivity (default 3), mark as outlier
3. **Consecutive outlier counting**: `process = process × is_outlier + is_outlier`
   - This resets to 0 on any non-outlier observation
   - Increments by 1 on each consecutive outlier
4. Break confirmed when `process ≥ boundary` (default 3 consecutive outliers)

### CuSum and MoSum (bfast-style)

- **CuSum**: Accumulates sum of standardized residuals; boundary grows with √n
- **MoSum**: Moving window sum of residuals; tests against a bandwidth-dependent boundary
- Both use recursive/OLS-based structural change test statistics

### Key Takeaways for Your System

1. **Temporal modeling**: The harmonic model cleanly handles seasonal vegetation
   cycles. For your NDVI time series, fit `y = β₀ + β₁t + β₂sin(2πt/365) + β₃cos(2πt/365)` per-segment
2. **Change magnitude**: The RMSE-normalized residual from CCDC gives you a
   scale-invariant "surprise" metric per observation
3. **State persistence**: The EWMA process value can be serialized to NetCDF between
   observations — useful for operational systems
4. **Mask-based workflow**: Their 0/1/2/3/4 mask state machine is a clean pattern:
   not-monitored / monitored / unstable-history / confirmed-break / insufficient-data

---

## Reference 2: Copernicus Parcel Delineation (VITO)

**Source**: https://documentation.dataspace.copernicus.eu/.../Parcel%20delineation.html

### Pipeline Overview

Three-stage pipeline:
1. **NDVI Computation** from Sentinel-2 with cloud masking
2. **U-Net Neural Network** for boundary probability prediction
3. **Sobel + Felzenszwalb + RAG merging** for final segmentation

### Stage 1: NDVI Preparation

```python
# Cloud masking via SCL dilation
scl_dilation_mask(
    kernel1_size=17,   # erosion kernel for good pixels [2,4,5,6,7]
    kernel2_size=77,   # dilation kernel for bad pixels [3,8,9,10,11]
    erosion_kernel_size=3
)
# NDVI from B04 (Red) and B08 (NIR)
ndvi = (B08 - B04) / (B08 + B04)
```

### Stage 2: U-Net Boundary Detection

**Architecture:**
- Input: 128×128 pixel patches, 3 NDVI images as channels
  (reshaped to `[1, 128*128, 3]` — flattened spatial with 3 temporal channels)
- Three pre-trained U-Net models (ensemble)
- Each model predicts 4 times with different random temporal selections
- Final prediction = **median of all 12 predictions** (3 models × 4 runs)

**Preprocessing:**
```python
# Clamp NDVI to [-0.08, 0.92], then shift to [0, 1]
ndvi = clip(ndvi, -0.08, 0.92) + 0.08
# Fill NaN with 0
ndvi = fillna(ndvi, 0)
```

**Temporal selection:**
- From the year's worth of cloud-free observations, randomly select 3 dates
- Requires minimum 4 valid temporal images
- Warns if selected dates are not ≥1 week apart

**Output:** A single-band probability/boundary map where higher values = more
likely to be field interior, lower = boundary

### Stage 3: Sobel + Felzenszwalb Segmentation (THE KEY ALGORITHM)

This is the most directly relevant part for your watershed system:

```python
from skimage.filters import sobel
from skimage import segmentation, graph

# 1. Edge detection on U-Net output
edges = sobel(image_data)

# 2. Graph-based over-segmentation
segment = segmentation.felzenszwalb(
    image_data,
    scale=120,        # Higher = fewer, larger segments
    sigma=0.0,        # No Gaussian pre-smoothing (U-Net already smoothed)
    min_size=30,      # Minimum segment size in pixels
    channel_axis=None # Single-channel input
)

# 3. Region Adjacency Graph with boundary weights
bgraph = graph.rag_boundary(segment, edges)

# 4. Merge segments with similar boundaries
merged = graph.cut_threshold(segment, bgraph, threshold=0.15)

# 5. Post-filtering
result = merged.where(unet_output >= 0.3)  # Remove low-confidence regions
result = result.where(result >= 0)         # Remove negative values
```

**Felzenszwalb Algorithm Details:**
- Graph-based segmentation where each pixel is a node
- Edge weights = absolute intensity difference between adjacent pixels
- Greedy merging: merge two components C₁, C₂ if:
  `min_edge(C₁,C₂) ≤ min(Int(C₁) + τ(C₁), Int(C₂) + τ(C₂))`
  where `Int(C) = max internal edge weight`, `τ(C) = scale/|C|`
- `scale=120` controls the threshold: higher = more tolerant of differences = larger segments
- `min_size=30` merges any component smaller than 30 pixels into its neighbor

**RAG Boundary Merging Details:**
- `rag_boundary(segments, edges)`: builds a Region Adjacency Graph where edge
  weights between adjacent segments = mean Sobel edge magnitude along their
  shared boundary
- `cut_threshold(segments, rag, 0.15)`: merges any two adjacent segments
  whose boundary edge weight < 0.15 (weak boundary = should be same object)

### Key Takeaways for Your System

1. **Sobel on a pre-processed surface** rather than raw imagery gives cleaner edges
2. **Felzenszwalb → RAG merge** is a proven two-pass strategy: over-segment then merge
3. **The 0.3 threshold** for masking low-confidence regions prevents noisy segmentation in ambiguous areas
4. **Ensemble prediction** (median of multiple models/runs) is robust

---

## Concrete Implementation Plan for Your System

Combining ideas from both references for DTM/DSM/RGBI/NDVI landscape segmentation:

### 1. Multi-Layer Gradient Computation

```python
import numpy as np
from skimage.filters import sobel

# Compute gradient magnitude for each input layer
edge_dtm = sobel(dtm)           # Terrain breaks
edge_dsm = sobel(dsm)           # Structure boundaries (buildings, tree canopy)
edge_chm = sobel(dsm - dtm)     # Canopy height model edges
edge_ndvi = sobel(ndvi)         # Vegetation boundaries
edge_r = sobel(ortho[:,:,0])    # Red channel edges
edge_g = sobel(ortho[:,:,1])    # Green channel edges
edge_b = sobel(ortho[:,:,2])    # Blue channel edges
edge_nir = sobel(ortho[:,:,3])  # NIR channel edges

# Weighted composite gradient
composite_edge = (
    0.25 * edge_chm +    # Canopy height is strong structural boundary
    0.20 * edge_dtm +    # Terrain breaks
    0.20 * edge_ndvi +   # Vegetation type boundaries
    0.15 * edge_nir +    # NIR edges (vegetation health)
    0.10 * edge_g +      # Visible structure
    0.05 * edge_r +
    0.05 * edge_b
)
```

### 2. Watershed-Based Initial Segmentation

```python
from skimage.segmentation import watershed, felzenszwalb
from skimage.feature import peak_local_min
from scipy import ndimage

# Option A: Watershed on composite gradient
markers = peak_local_min(-composite_edge, min_distance=10, labels=mask)
marker_labels = ndimage.label(markers)[0]
segments = watershed(composite_edge, markers=marker_labels)

# Option B: Felzenszwalb (as in parcel delineation)
segments = felzenszwalb(composite_edge, scale=150, sigma=0.5, min_size=50)
```

### 3. RAG-Based Segment Merging

```python
from skimage import graph

# Build RAG weighted by edge strength along boundaries
rag = graph.rag_boundary(segments, composite_edge)
merged_segments = graph.cut_threshold(segments, rag, threshold=0.12)
```

### 4. Per-Segment Feature Extraction

```python
from scipy.ndimage import labeled_comprehension
import pandas as pd

def extract_segment_features(segments, dtm, dsm, ortho, ndvi):
    labels = np.unique(segments[segments > 0])
    features = []
    for label in labels:
        mask = segments == label
        features.append({
            'label': label,
            'area_px': mask.sum(),
            # Elevation features
            'dtm_mean': dtm[mask].mean(),
            'dtm_std': dtm[mask].std(),
            'dtm_range': dtm[mask].ptp(),
            'slope_mean': slope[mask].mean(),  # pre-computed from DTM
            # Height features
            'chm_mean': (dsm - dtm)[mask].mean(),
            'chm_max': (dsm - dtm)[mask].max(),
            'chm_std': (dsm - dtm)[mask].std(),
            # Spectral features
            'ndvi_mean': ndvi[mask].mean(),
            'ndvi_std': ndvi[mask].std(),
            'red_mean': ortho[:,:,0][mask].mean(),
            'green_mean': ortho[:,:,1][mask].mean(),
            'blue_mean': ortho[:,:,2][mask].mean(),
            'nir_mean': ortho[:,:,3][mask].mean(),
            # Shape features
            'perimeter': ...,  # from regionprops
            'compactness': ...,
        })
    return pd.DataFrame(features)
```

### 5. Temporal NDVI Monitoring (from nrt)

For each segment, apply the EWMA or CCDC approach:

```python
# Per-segment temporal monitoring
def fit_segment_harmonic(ndvi_timeseries, dates):
    """Fit harmonic model to segment-mean NDVI time series"""
    T = 365.25
    t_days = np.array([(d - dates[0]).days for d in dates], dtype=float)
    # Design matrix: [1, t, sin(2πt/T), cos(2πt/T)]
    X = np.column_stack([
        np.ones_like(t_days),
        t_days,
        np.sin(2 * np.pi * t_days / T),
        np.cos(2 * np.pi * t_days / T),
    ])
    beta, _, _, _ = np.linalg.lstsq(X, ndvi_timeseries, rcond=None)
    residuals = ndvi_timeseries - X @ beta
    rmse = np.sqrt(np.mean(residuals**2))
    return beta, rmse

def classify_temporal_stability(beta, rmse):
    """Use harmonic model parameters as classification features"""
    amplitude = np.sqrt(beta[2]**2 + beta[3]**2)  # seasonal amplitude
    phase = np.arctan2(beta[2], beta[3])           # peak timing
    trend = beta[1]                                 # long-term trend
    return {
        'ndvi_amplitude': amplitude,
        'ndvi_phase': phase,
        'ndvi_trend': trend,
        'ndvi_rmse': rmse,
        'ndvi_mean': beta[0],  # intercept ≈ mean level
    }
```

### 6. Classification Strategy

```python
from sklearn.ensemble import RandomForestClassifier

# Combine spatial + temporal features per segment
# Spatial: DTM stats, CHM stats, spectral means, shape
# Temporal: harmonic amplitude, phase, trend, RMSE
# Train RF classifier on labeled segments

classes = {
    'deciduous_forest': high CHM + high NDVI amplitude + late phase,
    'conifer_forest': high CHM + low NDVI amplitude + high mean NDVI,
    'cropland': low CHM + very high NDVI amplitude,
    'grassland': low CHM + moderate NDVI amplitude,
    'built_up': high DSM-DTM std + low NDVI + high spectral variance,
    'water': very low NDVI + low DTM std + low NIR,
    'bare_soil': low NDVI + low NDVI amplitude + moderate spectral values,
}
```

---

## Parameter Guidance

| Parameter | Parcel Delineation Value | Suggested for Landscape |
|-----------|------------------------|------------------------|
| Felzenszwalb `scale` | 120 | 100-200 (higher for coarser landscape units) |
| Felzenszwalb `min_size` | 30 px | 50-100 px (landscape objects are larger) |
| Felzenszwalb `sigma` | 0.0 | 0.5-1.0 (smooth noisy LiDAR/ortho) |
| RAG `cut_threshold` | 0.15 | 0.10-0.20 (tune per gradient normalization) |
| EWMA `lambda` | 0.3 | 0.3 (good default) |
| EWMA `sensitivity` | 2 | 2-3 (lower = more sensitive to change) |
| CCDC `boundary` | 3 consecutive | 3 (requires 3 consecutive outlier observations) |
| Harmonic order | 2 | 1-2 (order 1 usually sufficient for NDVI) |
