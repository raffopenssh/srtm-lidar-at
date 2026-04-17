# Task: RF Model Cleanup — Drop Unlearnable Classes, Merge Rare Ones, Tighten Downsampling, Add Infrastructure Rules, Stop Training

## Context

We're training a Random Forest classifier for Austrian landscape segmentation from remote sensing data (LiDAR DTM/DSM, orthophotos, Sentinel-2 NDVI, ESA WorldCover, SAR, Hansen GFC). The model currently trains on cadastre + OSM ground truth across random Austrian Katastralgemeinden (KGs).

**Current state** (251 KGs, 647k raw samples, 20 classes):
- Live model OOB: **61.8%** at 250 KGs — curve still slowly declining (was 63.5% at 180 KGs)
- Downsampling (10× median cap on tree/grass) slowed the decline but didn't stop it
- Eval curve: 62.0-62.2% at 225-240 KGs — no sign of recovery

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

Current class distribution (647k total, 251 checkpoints):
```
tree:        301,280 (46.6%) — currently capped to ~93k, will become ~46k with 5x cap
grass:       134,189 (20.7%) — currently capped to ~93k, will become ~46k with 5x cap
roof:         44,227 (6.8%)
road:         41,799 (6.5%)
crop:         27,010 (4.2%)
vineyard:     22,758 (3.5%)
shrub:        17,910 (2.8%)
orchard:      12,565 (1.9%)
garden:       10,578 (1.6%)
water:         9,289 (1.4%) ← median
rock:          6,535 (1.0%)
tree_loss:     6,064 (0.9%)
path:          4,737 (0.7%)
parking:       3,918 (0.6%)
bare_soil:     2,919 (0.5%)
earthwork:     1,090 (0.2%) ← merged excavation(559)+fill(531)
```

With 5x median cap: 5 × 9,289 = **46,445**. Only tree and grass exceed this.
After dropping 3 classes and merging, we go from 20 → 16 classes.

### 4. Stop the training job

The learning curve is flat from 130→180 KGs. More data won't help.

- Stop the `rf_train` systemd service: `sudo systemctl stop rf_train && sudo systemctl disable rf_train`
- The existing 251 checkpoints in `rf_training_data/checkpoints/` are sufficient
- After applying changes 1-3, trigger a single retrain using the existing checkpoints (call the `/api/v1/classifier/train` endpoint or invoke `learned_classifier.LearnedClassifier().train()` directly from a script)
- Also stop the curve eval cron if it exists (check crontab)

### 5. Improved rule-based detection for dropped classes

The 3 dropped classes need better rule-based detection in `classify_object()` (`object_segmentation.py`). Currently:
- `solar_panel`: Detected at line ~1179 with a crude heuristic (building-like + smooth + bright). Works poorly.
- `wind_turbine`: **No rule-based detection at all** — only the RF predicted it.
- `substation`: **No rule-based detection at all** — only the RF predicted it.
- `mast`: Detected at line ~1174 with `area < 10 and h_mean > 15 and compact < 0.5`. Too simple — confuses with tree crowns.

**New approach: Spatial lookup from austria-power API + better physical heuristics.**

We have rich infrastructure data at `https://austria-power.exe.xyz:8000/api/infrastructure`. Query with `?bbox=min_lon,min_lat,max_lon,max_lat&categories=...` for per-tile enrichment. Data available:

| Category | Count | Key properties |
|----------|-------|---------|
| `solar_energy` | 3,513 | `area_sqm` (74–132k sqm; 803 rooftop <1k, 1148 ground 1k-10k, 1562 utility >10k) |
| `wind_energy` | 3,001 | `height_agl_m` (65–285m, median 186m), `hub_height_m` (31–161m, median 105m), `rotor_diameter_m` (20–163m, median 101m), `capacity_mw`, `year` |
| `substation` | 513 | `voltage_kv` (220–380kV), `capacity_mw` (0–88MW, median 6.7), 429 transformer_stations + 44 OSM + 40 380kV nodes |
| `telecom` | 1,713 | `height_agl_m` (22–164m, median 39m), all type "Antenna" |
| `structure` | 193 | Poles(70), Masts(30), Towers(16), Stacks(30), Buildings(68) — `height_agl_m` (1–251m, median 86m) |

#### Implementation: `infrastructure_lookup.py` (new module)

1. **On first call per analysis**: Fetch infrastructure for the analysis bbox (+500m buffer) from the API, convert coords to EPSG:3035, build a scipy `cKDTree` spatial index.
2. **Cache**: `/tmp/infrastructure_cache/` — cache the full nationwide dataset on first fetch (24h TTL), then query from it locally.
3. **API**: `find_nearby(centroid_3035, radius_m, categories) → list[dict]` — used per-segment in `classify_object()`.

#### Physical signatures & rules

**Wind turbine** — the most distinctive feature in LiDAR:
- In DSM: a single extremely tall point (65–285m AGL, median 186m) with tiny footprint
- The turbine tower is ~4–6m diameter → single 1m pixel column in DSM
- As a segment: h_mean 50–200m, h_max 80–285m, area 5–50m², very low compactness
- **Every turbine has surrounding infrastructure**: access road (gravel, ~4m wide, shows as `road` or `path`), assembly/crane pad (flat bare_soil, 30×50m), and cable trench to substation
- Rotor blades may appear as ghost artefacts in DSM (moving during capture)
- Rules:
  - IF near known wind turbine location (±150m) AND h_mean > 40m AND area < 100m² → `wind_turbine` (conf 0.95)
  - IF no API match but h_mean > 60m AND area < 50m² AND compact < 0.3 AND NDVI < 0.1 → `wind_turbine` (conf 0.7)
  - Current `mast` rule (h_mean > 15, area < 10) catches these at wrong threshold; raise to h_mean > 50 for wind_turbine, keep mast at 15–50m range

**Solar panel** — two very different signatures:

*Rooftop solar* (<1000 sqm, 803 known):
- Sits on existing building roof, 2–8m above ground
- In LiDAR: nearly identical to roof (h_mean 3–10m), very smooth DSM (roughness < 0.3), flat (h_std < 0.5)
- In orthophoto: dark blue/black, very low NDVI (<0.05), high brightness (reflective), uniform texture
- Distinguished from plain roof by: lower NDVI, smoother surface, more uniform colour
- Current rule (bld_score + smooth + bright) is on the right track but too crude
- Rules:
  - IF building-like (bld_score ≥ threshold) AND dsm_rough < 0.3 AND h_std < 0.5 AND NDVI < 0.05 AND near known solar location (±50m) → `solar_panel` (conf 0.85)
  - Without API match: same physical criteria but require brightness > 130 AND blue_mean > red_mean (solar panels are blue-ish) → `solar_panel` (conf 0.5)

*Ground-mounted solar farm* (1k–132k sqm, 2710 known):
- Ground level or on 1–3m stilts, vast area, extremely flat and uniform
- In LiDAR: h_mean 0.5–3m, near-zero h_std, near-zero slope, very low DSM roughness
- In orthophoto: dark/reflective rows, extremely low NDVI (<0.05), regular grid pattern
- GLCM texture: very low contrast, very high homogeneity (uniform panels)
- Distinguished from parking/road by: lower NDVI, more uniform, near-zero height
- Rules:
  - IF near known solar location (±100m) AND area > 500m² AND NDVI < 0.1 AND dsm_rough < 0.5 AND h_mean < 5m → `solar_panel` (conf 0.90)
  - Without API match: area > 1000m² AND NDVI < 0.05 AND dsm_rough < 0.3 AND h_std < 0.3 AND brightness > 100 → `solar_panel` (conf 0.4)

**Substation / transformer station** — fenced compound:
- 429 transformer stations (distribution level, 6.7 MW median) + 44 OSM + 40 major 380kV nodes
- Physical: rectangular fenced compound (50×50m to 200×200m), contains transformers (metal boxes 3–8m tall), busbars, gravel ground
- In LiDAR: mix of heights (0–10m), moderate DSM roughness (equipment), relatively flat DTM
- In orthophoto: grey/metallic, very low NDVI, visible geometric structure
- Distinguished from building by: larger footprint, lower compactness (not a solid rectangle), gravel ground between equipment
- Rules:
  - IF near known substation (±150m) AND area > 200m² AND NDVI < 0.15 AND h_mean 1–15m → `substation` (conf 0.85)
  - Without API match: very hard to detect — leave as `roof` (which groups into `building` anyway)

**Mast / antenna** — tall narrow structures:
- 1,713 telecom antennas (22–164m, median 39m) + 30 named masts + 70 poles + 16 towers
- Physical: steel lattice or concrete tower, 2–6m diameter, very tall
- In LiDAR: single tall spike, tiny footprint (<10m²), very low compactness
- Current rule works but threshold (h_mean > 15) is too low — many trees are 15–25m
- Rules:
  - IF near known telecom/structure location (±50m) AND h_mean > 20m AND area < 15m² → `mast` (conf 0.85)
  - IF no API match AND h_mean > 25m AND area < 10m² AND compact < 0.3 AND NDVI < 0.1 → `mast` (conf 0.6)
  - Separate from wind_turbine by height: mast typically 22–164m but area <15m²; wind turbine h_mean >60m
  - The overlap zone (25–60m, tiny footprint) should default to `mast` unless near a known wind turbine

#### Integration into pipeline

The infrastructure lookup should be called **once per analysis bbox** at the start of `segment_landscape()` in `object_segmentation.py`. Store the results as a list passed through to `classify_object()` via a new parameter `nearby_infra=None`. Each segment's centroid is checked against nearby infrastructure during classification.

The rules above integrate into the existing `classify_object()` flow — they go BEFORE the RF prediction (or replace it for these types, since RF no longer predicts them). The infrastructure check is an early-exit: if a segment is near known infrastructure and its physical signature matches, return immediately with the infrastructure type.

#### Also enhance OSM queries

In `osm_features.py`, add a `"power"` query to `_query_all()`:
```python
"power": f'[out:json][timeout:60]{bb};(way["power"];node["power"="generator"];node["power"="tower"];);out geom;',
```
And add to `_LANDCOVER_MAP`:
```python
_LANDCOVER_MAP[("power", "generator")] = "solar_panel"  # needs additional solar check
_LANDCOVER_MAP[("power", "substation")] = "substation"
_LANDCOVER_MAP[("power", "plant")] = "substation"
```
This provides OSM-based ground truth for rule-based detection during both training and inference.

## File Reference

| File | What to change |
|------|---------------|
| `learned_classifier.py` | Drop 3 classes from TYPE_CLASSES, merge excavation+fill→earthwork in CADASTRE_TYPE_MAP, change cap_multiplier 10→5 |
| `object_segmentation.py` | Add earthwork→excavation/fill post-split, integrate infrastructure_lookup into classify_object() for solar/wind/substation/mast rules, accept `nearby_infra` param, call lookup once in segment_landscape() |
| `train_rf_4000kg.py` | Filter out dropped classes in _label_segments(), remap excavation/fill→earthwork |
| `infrastructure_lookup.py` | **NEW FILE** — fetch + cache austria-power API, cKDTree spatial index, `find_nearby(centroid_3035, radius_m, categories)` |
| `osm_features.py` | Add "power" query to _query_all(), add power→label mappings |

## Testing

After all changes:
1. Stop rf_train service
2. Retrain model once from existing 251 checkpoints: write a small script that loads all checkpoints, applies the label remapping (drop 3 + merge), calls `LearnedClassifier().train(X, y)` 
3. Check new model: should have 16 classes, OOB should be ~66-68%
4. Test an analysis on a known area with solar panels / wind turbines to verify rule-based detection works
5. Restart the srv service

## Important Notes

- **Do NOT delete the old model** — back up `/tmp/learned_classifier/rf_model.joblib` to `/tmp/learned_classifier/rf_model_20cls_251kg.joblib.bak` before retraining
- The eval curve cron job should be stopped too (check `crontab -l` and any systemd timers)
- The training status API (`/api/v1/training/status`) and UI (`training.html`) will show the job as stopped — that's expected
- The live model meta (`rf_meta.json`) will update after retrain with the new class list
- Keep `wind_turbine`, `substation`, `solar_panel` in `OBJECT_TYPES` dict and colour map — they're still valid output types from rule-based classification

## austria-power API Reference

```
Base: https://austria-power.exe.xyz:8000
Docs: https://austria-power.exe.xyz:8000/llm.txt
GET /api/infrastructure?bbox=min_lon,min_lat,max_lon,max_lat&categories=...
GET /api/infrastructure/stats
```

Returns GeoJSON FeatureCollection. Filter by `categories` and/or `layers`.

**Categories relevant to us**: `solar_energy` (3513), `wind_energy` (3001), `substation` (513), `telecom` (1713), `structure` (193)

**Key properties by category**:
- `solar_energy`: `area_sqm` (74–132k; rooftop<1k, ground 1k-10k, utility >10k), `name`, `capacity_mw`
- `wind_energy`: `height_agl_m` (65–285m, med 186), `hub_height_m` (31–161, med 105), `rotor_diameter_m` (20–163, med 101), `capacity_mw` (0.1–7.5, med 3.0), `year`, `turbine_model`, `area_sqm`
- `substation`: `voltage_kv` (220–380), `capacity_mw` (0–88, med 6.7), `operator`, types: transformer_station(429)/substation(44)/substation_380kv(40)
- `telecom`: `height_agl_m` (22–164m, med 39m), all type "Antenna"
- `structure`: types Pole(70)/Mast(30)/Tower(16)/Stack(30)/Building(68), `height_agl_m` (1–251m, med 86m)

All coordinates in EPSG:4326. Use `?bbox=` for per-tile queries.  
Cache the full nationwide dataset locally (24h TTL) to avoid repeated large fetches.

## Cadastre API Reference (for context)

```
Base: https://cadastre-process-api.exe.xyz
Building object_type codes relevant here:
- 80: Abbaufläche (quarry) → now "earthwork"
- 81: Deponie → now "earthwork"  
- 93: Abbaufläche → now "earthwork"
```
