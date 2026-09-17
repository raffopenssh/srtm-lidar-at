# model v2 — model_G.joblib

hash `7089ca645ff5` · trained `2026-09-17T06:45:35Z` · n_train 1,234,249 · CV macro-F1 0.7808552894401568

classes (14): crop, garden, glacier, grass, orchard, parking, rail, road, rock, roof, shrub, tree, vineyard, water

merge: `{'excavation': 'earthwork', 'fill': 'earthwork', 'hedge': 'shrub', 'bare_soil': 'rock'}` · rule-only: `['earthwork', 'path']`

NDVI vetoes — min: `{'grass': 0.17, 'tree': 0.24, 'shrub': 0.19, 'orchard': 0.13, 'wetland': 0.12}` · max: `{'roof': 0.11, 'road': 0.19, 'parking': 0.2, 'rail': 0.14, 'water': 0.15, 'rock': 0.21, 'glacier': -0.05, 'bare_soil': 0.25, 'earthwork': 0.4}`

## Feature importance (gain)

| # | feature | gain % | splits |
|--:|---|--:|--:|
| 1 | ndvi_ndsm_coherence | 12.96 | 1967 |
| 2 | dist_building | 9.61 | 7154 |
| 3 | elevation_mean | 9.37 | 11284 |
| 4 | ndvi_mean | 7.37 | 3264 |
| 5 | dist_road | 6.07 | 6467 |
| 6 | dist_water | 5.50 | 8987 |
| 7 | esa_snow_frac | 4.72 | 233 |
| 8 | ndsm_frac_gt2 | 3.67 | 1498 |
| 9 | dist_rail | 3.28 | 2645 |
| 10 | esa_crop_frac | 3.22 | 1060 |
| 11 | tri_mean | 2.36 | 2241 |
| 12 | savi_mean | 2.05 | 2492 |
| 13 | cop_ndvi_mean | 1.81 | 6445 |
| 14 | ndvi_y_last | 1.63 | 622 |
| 15 | esa_dominant_lc | 1.48 | 449 |
| 16 | fused_ndvi_mean | 1.48 | 4837 |
| 17 | dsm_roughness | 1.13 | 2968 |
| 18 | h_mean | 1.12 | 1808 |
| 19 | nb_ndvi_mean | 1.02 | 5907 |
| 20 | h_p90 | 0.97 | 1793 |
| 21 | h_p50 | 0.97 | 2587 |
| 22 | ndvi_trend_per_year | 0.94 | 5397 |
| 23 | dist_parcel_edge | 0.92 | 8519 |
| 24 | esa_built_frac | 0.88 | 812 |
| 25 | ndvi_p10 | 0.84 | 3529 |
| 26 | nb_h_mean | 0.78 | 6410 |
| 27 | esa_grass_frac | 0.69 | 2050 |
| 28 | nir_year | 0.54 | 866 |
| 29 | ndvi_y_first | 0.52 | 4635 |
| 30 | esa_bare_frac | 0.51 | 328 |
| 31 | ndvi_tstd | 0.49 | 5689 |
| 32 | ndsm_frac_gt5 | 0.49 | 721 |
| 33 | dsm_year | 0.44 | 1762 |
| 34 | nir_lidar_gap | 0.44 | 2095 |
| 35 | sar_vh | 0.44 | 6797 |
| 36 | esa_tree_frac | 0.41 | 1550 |
| 37 | slope_mean | 0.40 | 2850 |
| 38 | sar_vv | 0.40 | 6200 |
| 39 | nir_mean | 0.38 | 5628 |
| 40 | dtm_change | 0.37 | 1316 |

### Gain by feature family

| family | gain % |
|---|--:|
| dist_* | 25.4 |
| nb_* | 2.2 |
| harm_* | 0.0 |
| sar_* | 1.0 |
| esa_* | 12.1 |
| hansen_* | 0.2 |
| glcm/texture | 0.1 |
| ndvi_* | 25.2 |

## Mean |SHAP| (TreeSHAP, 5000 sampled segments)

| # | feature | mean abs SHAP | top class |
|--:|---|--:|---|
| 1 | ndvi_mean | 0.821 | rock |
| 2 | elevation_mean | 0.530 | crop |
| 3 | ndsm_frac_gt2 | 0.388 | grass |
| 4 | dist_building | 0.300 | garden |
| 5 | h_p50 | 0.282 | tree |
| 6 | dist_road | 0.233 | garden |
| 7 | dist_parcel_edge | 0.181 | vineyard |
| 8 | ndvi_y_last | 0.174 | rock |
| 9 | ndvi_ndsm_coherence | 0.150 | tree |
| 10 | cop_ndvi_mean | 0.124 | crop |
| 11 | dist_water | 0.118 | water |
| 12 | savi_mean | 0.116 | rock |
| 13 | h_mean | 0.115 | tree |
| 14 | dtm_year | 0.091 | road |
| 15 | ndsm_frac_gt5 | 0.085 | tree |
| 16 | fused_ndvi_mean | 0.084 | rock |
| 17 | ndwi_mean | 0.083 | water |
| 18 | nb_ndvi_mean | 0.081 | rock |
| 19 | nb_h_mean | 0.078 | tree |
| 20 | ndvi_y_first | 0.077 | grass |
| 21 | h_iqr | 0.073 | rock |
| 22 | slope_mean | 0.072 | crop |
| 23 | tri_mean | 0.072 | rock |
| 24 | sar_vh | 0.071 | vineyard |
| 25 | sar_vv | 0.069 | crop |
| 26 | dsm_roughness | 0.069 | glacier |
| 27 | hansen_treecover2000 | 0.068 | orchard |
| 28 | aspect_cos | 0.067 | vineyard |
| 29 | dtm_range | 0.067 | parking |
| 30 | ndvi_tstd | 0.067 | glacier |
| 31 | ndvi_p10 | 0.066 | grass |
| 32 | nir_mean | 0.063 | orchard |
| 33 | green_ratio | 0.060 | water |
| 34 | rg_index | 0.060 | glacier |
| 35 | ndvi_p90 | 0.056 | parking |
| 36 | esa_grass_frac | 0.055 | vineyard |
| 37 | brightness_std | 0.054 | orchard |
| 38 | ndvi_trend_per_year | 0.052 | shrub |
| 39 | blue_mean | 0.051 | glacier |
| 40 | nir_lidar_gap | 0.051 | grass |

### Top-5 features per class (mean |SHAP|)

* **crop**: elevation_mean (1.68), cop_ndvi_mean (0.36), esa_crop_frac (0.35), slope_mean (0.33), dist_parcel_edge (0.32)
* **garden**: dist_building (1.57), dist_road (0.97), elevation_mean (0.56), dist_parcel_edge (0.52), ndsm_frac_gt2 (0.14)
* **glacier**: dist_water (0.32), esa_dominant_lc (0.22), ndvi_mean (0.21), rg_index (0.21), ndwi_mean (0.16)
* **grass**: ndvi_mean (2.44), ndsm_frac_gt2 (2.34), ndvi_y_last (0.60), elevation_mean (0.36), ndvi_p10 (0.21)
* **orchard**: elevation_mean (1.16), ndvi_mean (0.30), years_span (0.25), dist_road (0.24), nir_mean (0.21)
* **parking**: ndvi_mean (1.19), dist_building (0.50), ndsm_frac_gt2 (0.31), cop_ndvi_mean (0.26), savi_mean (0.26)
* **rail**: ndvi_y_first (0.13), ndvi_std (0.13), ndvi_mean (0.08), ndvi_y_last (0.08), curvature_mean (0.05)
* **road**: dist_road (0.87), ndvi_mean (0.56), dtm_year (0.44), dist_parcel_edge (0.29), dist_building (0.23)
* **rock**: ndvi_mean (3.18), elevation_mean (1.00), ndvi_y_last (0.72), savi_mean (0.61), dist_road (0.45)
* **roof**: ndsm_frac_gt2 (0.55), dist_building (0.47), h_max (0.33), ndsm_frac_gt5 (0.25), h_p50 (0.24)
* **shrub**: ndvi_mean (1.98), elevation_mean (0.45), h_p50 (0.43), ndvi_y_last (0.30), h_mean (0.24)
* **tree**: h_p50 (2.45), ndsm_frac_gt2 (1.80), ndvi_ndsm_coherence (1.58), h_mean (0.75), ndvi_mean (0.63)
* **vineyard**: elevation_mean (1.43), dist_parcel_edge (0.59), esa_grass_frac (0.29), dist_road (0.22), cop_ndvi_mean (0.21)
* **water**: ndvi_mean (0.66), dist_water (0.33), ndwi_mean (0.17), sar_vv (0.15), green_ratio (0.14)
