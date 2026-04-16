# Task: RF Model Cleanup — Drop Unlearnable Classes, Merge Rare Ones, Tighten Downsampling, Add Infrastructure Rules, Stop Training

## Context

We're training a Random Forest classifier for Austrian landscape segmentation from remote sensing data (LiDAR DTM/DSM, orthophotos, Sentinel-2 NDVI, ESA WorldCover, SAR, Hansen GFC). The model currently trains on cadastre + OSM ground truth across random Austrian Katastralgemeinden (KGs).

**Current state** (180 KGs, 460k raw samples, 20 classes):
- Live model OOB: **63.2%** — and the learning curve has flatlined from 130→180 KGs
- Downsampling (10× median cap on tree/grass) slowed the decline but didn't stop it
- The curve is now flat — more KGs won't help

**Root problem**: 3 classes are unlearnable (19/81/174 samples), 2 rare classes should merge, and the downsample cap is too generous.

## Changes Required (5 items)

### 1. Drop 3 unlearnable classes from RF training

Remove from `TYPE_CLASSES` in `learned_classifier.py` and exclude from training data:
- `wind_turbine` — 19 samples, 0% OOB
- `substation` — 81 samples, ~0% OOB
- `solar_panel` — 174 samples, ~20% OOB

These must remain as valid object types in the segmentation pipeline (`object_segmentation.py`) — they just won't be RF-predicted. Instead, they'll be detected by improved rule-based logic (see item 5 below).

**Files to edit:**
- `learned_classifier.py`: Remove from `TYPE_CLASSES` list (line ~132-136). In `train()` method, filter out samples with these labels before training. Keep them in `CADASTRE_TYPE_MAP` so they're still recognized during ground truth extraction.
- `train_rf_4000kg.py`: The `_label_segments()` function assigns labels. After assignment, any segment labeled as these 3 types should be excluded (not re-mapped — just dropped). The infrastructure KG prepending logic (`_find_infra_kgs()`) can stay since it also helps with nearby classes like `roof`, `road`, etc.

### 2. Merge excavation + fill → earthwork

These two classes have only 403 and 471 samples respectively, and they're distinguished purely by the sign of DTM change (negative = excavation, positive = fill). Combined as `earthwork` they'll have 874 samples — still small but more learnable.

**Files to edit:**
- `learned_classifier.py`:
  - `CADASTRE_TYPE_MAP`: Change codes 80, 81, 93 to map to `"earthwork"` instead of `"excavation"`/`"fill"`
  - `TYPE_CLASSES`: Replace `"excavation", "fill"` with `"earthwork"`
- `object_segmentation.py`:
  - `OBJECT_TYPES`: Add `"earthwork"` with a new code, keep `"excavation"` and `"fill"` as valid types for rule-based output
  - `classify_object()`: The rule-based path (lines ~955-1013) still returns `"excavation"` or `"fill"` based on DTM change sign — **keep this as-is**. Only the RF training label merges them.
  - `GROUP_COMPAT`, `MERGE_RULES`: Update references. The group `"quarry"` already merges excavation+fill — this stays.
  - Approach: When the RF predicts `"earthwork"`, the post-classification step in `_build_geojson_features()` should split it back into `"excavation"` or `"fill"` based on the segment's `dtm_change` sign (negative → excavation, positive → fill). This preserves the user-facing distinction.
- `train_rf_4000kg.py`: In `_label_segments()`, remap any `"excavation"` or `"fill"` labels to `"earthwork"`.

### 3. Tighten downsample cap from 10× to 5× median

In `learned_classifier.py`, function `_downsample()` (line ~145):
- Change `cap_multiplier` default from `10` to `5`
- This will cap tree/grass more aggressively, giving minority classes more relative weight

Current class distribution (460k total):
```
tree:        208,578 (45.3%) — currently capped to ~59k, will become ~30k
grass:       100,968 (21.9%) — currently capped to ~59k, will become ~30k  
roof:         33,192 (7.2%)
crop:         23,316 (5.1%)
road:         23,113 (5.0%)
vineyard:     17,973 (3.9%)
shrub:        13,394 (2.9%)
orchard:       9,161 (2.0%)
water:         7,695 (1.7%)
garden:        5,934 (1.3%) ← median
rock:          5,080 (1.1%)
tree_loss:     4,053 (0.9%)
path:          4,037 (0.9%)
bare_soil:     1,542 (0.3%)
earthwork:       874 (0.2%) ← merged excavation+fill
parking:         880 (0.2%)
```

After dropping 3 classes and merging, we go from 20 → 16 classes.

### 4. Stop the training job

The learning curve is flat from 130→180 KGs. More data won't help.

- Stop the `rf_train` systemd service: `sudo systemctl stop rf_train && sudo systemctl disable rf_train`
- The existing 188 checkpoints in `rf_training_data/checkpoints/` are sufficient
- After applying changes 1-3, trigger a single retrain using the existing checkpoints (call the `/api/v1/classifier/train` endpoint or invoke `learned_classifier.LearnedClassifier().train()` directly from a script)
- Also stop the curve eval cron if it exists (check crontab)

### 5. Improved rule-based detection for dropped classes

The 3 dropped classes need better rule-based detection in `classify_object()` (in `object_segmentation.py`). Currently:
- `solar_panel`: Detected at line ~1179 with a crude heuristic (building-like + smooth + bright). Works poorly.
- `wind_turbine`: **No rule-based detection at all** — only the RF predicted it.
- `substation`: **No rule-based detection at all** — only the RF predicted it.

**New approach: Use the austria-power.exe.xyz API for spatial lookup.**

The API at `https://austria-power.exe.xyz:8000/api/infrastructure` provides:
- 3,513 solar plants (points with `area_sqm`)
- ~3,000 wind turbines (points with `height_agl_m`, `rotor_diameter_m`, `hub_height_m`)
- 513 substations (points with `voltage_kv`)

**Implementation plan:**

Create a new module `infrastructure_lookup.py` that:

1. **On startup / first call**: Fetches all infrastructure from the API, builds a spatial index (rtree or scipy.spatial.cKDTree on WGS84 coords converted to EPSG:3035)
2. **Cache**: Store the full dataset in `/tmp/infrastructure_cache/all.json` with a 24h TTL
3. **Query function**: `find_nearby_infrastructure(bbox_3035, buffer_m=100) → list[dict]` returns infrastructure features whose coordinates fall within the buffered bbox
4. **Per-category queries**:
   - `find_solar(bbox_3035, buffer_m=50) → list[Point]`
   - `find_wind_turbines(bbox_3035, buffer_m=100) → list[Point]`  
   - `find_substations(bbox_3035, buffer_m=100) → list[Point]`

Then integrate into the segmentation pipeline in `object_segmentation.py`:

**For `solar_panel`**: After the RF classification (or in rule-based path), if a segment:
- Overlaps with a known solar plant location (within 50m)
- Has building-like characteristics (low NDVI, flat, smooth DSM)
- → Classify as `solar_panel` with high confidence

**For `wind_turbine`**: If a segment:
- Is within 100m of a known wind turbine location
- Has very tall height (h_mean > 50m or h_max > 80m)
- Has small footprint (area < 50m²)
- → Classify as `wind_turbine`
- Alternatively, even without API match: if h_mean > 60m AND area < 30m² AND compact < 0.3 → likely wind turbine (the mast rule at line ~1175 catches some of these but the threshold is too low at 15m)

**For `substation`**: If a segment:
- Is within 100m of a known substation location
- Has building-like characteristics OR is a fenced compound (low vegetation, structured)
- → Classify as `substation`

The infrastructure lookup should be called once per analysis bbox (not per segment) and the results cached for the duration of the analysis. Pass the nearby infrastructure list into `classify_object()` or apply as a post-classification overlay.

**Also enhance OSM queries** in `osm_features.py`: Add a `"power"` query to `_query_all()`:
```python
"power": f'[out:json][timeout:60]{bb};(way["power"];node["power"="generator"];node["power"="tower"];);out geom;',
```
And add to `_LANDCOVER_MAP`:
```python
_LANDCOVER_MAP[("power", "generator")] = "solar_panel"  # with solar check
_LANDCOVER_MAP[("power", "substation")] = "substation"
_LANDCOVER_MAP[("power", "plant")] = "substation"
```
This gives OSM-based ground truth for these types during training label assignment too (even though RF won't predict them, the rule-based path benefits).

## File Reference

| File | What to change |
|------|---------------|
| `learned_classifier.py` | Drop 3 classes from TYPE_CLASSES, merge excavation+fill→earthwork in CADASTRE_TYPE_MAP, change cap_multiplier 10→5 |
| `object_segmentation.py` | Add earthwork→excavation/fill post-split, integrate infrastructure_lookup for solar/wind/substation rule detection |
| `train_rf_4000kg.py` | Filter out dropped classes in _label_segments(), remap excavation/fill→earthwork |
| `infrastructure_lookup.py` | **NEW FILE** — spatial index over austria-power API data |
| `osm_features.py` | Add "power" query to _query_all(), add power→label mappings |

## Testing

After all changes:
1. Stop rf_train service
2. Retrain model once from existing 188 checkpoints: write a small script that loads all checkpoints, applies the label remapping (drop 3 + merge), calls `LearnedClassifier().train(X, y)` 
3. Check new model: should have 16 classes, OOB should be ~66-68%
4. Test an analysis on a known area with solar panels / wind turbines to verify rule-based detection works
5. Restart the srv service

## Important Notes

- **Do NOT delete the old model** — back up `/tmp/learned_classifier/rf_model.joblib` to `/tmp/learned_classifier/rf_model_20cls.joblib.bak` before retraining
- The eval curve cron job should be stopped too (check `crontab -l` and any systemd timers)
- The training status API (`/api/v1/training/status`) and UI (`training.html`) will show the job as stopped — that's expected
- The live model meta (`rf_meta.json`) will update after retrain with the new class list
- Keep `wind_turbine`, `substation`, `solar_panel` in `OBJECT_TYPES` dict and colour map — they're still valid output types from rule-based classification

## austria-power API Reference

```
Base: https://austria-power.exe.xyz:8000
GET /api/infrastructure?bbox=min_lon,min_lat,max_lon,max_lat&categories=solar_energy,wind_energy,substation

Returns GeoJSON FeatureCollection. Key properties:
- category: solar_energy | wind_energy | substation
- type: solar | Windmill farm | windpark | Windpower plant | transformer_station | substation_380kv
- height_agl_m, rotor_diameter_m, hub_height_m (wind)
- area_sqm (solar, wind)
- voltage_kv (substations)
- Coordinates in EPSG:4326
```

## Cadastre API Reference (for context)

```
Base: https://cadastre-process-api.exe.xyz
Building object_type codes relevant here:
- 80: Abbaufläche (quarry) → now "earthwork"
- 81: Deponie → now "earthwork"  
- 93: Abbaufläche → now "earthwork"
```
