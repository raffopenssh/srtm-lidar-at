# RF Training (`train_rf_4000kg.py`) — Mental Model

Background job that trains the Random Forest classifier used by the
live segmentation pipeline. Runs as `rf_train.service` on the primary.
Log: `/tmp/rf_train_4000kg.log`. Status: `/api/v1/training/status`.

## What it does (in 5 lines)

1. Sample N random KGs across Austria.
2. For each KG: fetch cadastre + OSM ground truth, fetch all raster
   data (DTM/DSM/ortho/NDVI/SAR/Hansen), run segmentation, label
   each segment from ground truth.
3. Save labelled features as a checkpoint (`.npz`) to disk.
4. Every 10 KGs: load all checkpoints, train RF, save model to
   `/tmp/learned_classifier/`.
5. Skip KGs that crash (OOM, bad data) by recording them in a
   permanent `bad_kgs.json` so retries don't loop forever.

Checkpoints are durable. Restarting the service resumes from where
it stopped — missing KGs are filled in, model is retrained from full
checkpoint corpus.

## Storage layout (`rf_training_data/`)

```
rf_training_data/
  checkpoints/
    kg_<code>.npz         ← X, y, group (one row per segment)
  bad_kgs.json            ← permanently-skipped KGs
  copernicus_cache/        ← reused across runs
  hansen_cache/
  cadastre_cache/
  osm_cache/
```

## Ground-truth pipeline (the interesting bit)

Labelling segments is the hard part. Two sources, with sanity checks:

```
For each segment:
    ├─ 1. Try cadastre (rasterized at 1m, height-aware)
    │     └─ reject if pixel is ground-surface code (road/parking/
    │        bare soil) but nDSM > threshold (= probably tree canopy)
    │     └─ reject if cadastre code says ground-surface but NDVI > 0.25
    │        (= probably vegetation, cadastre stale)
    ├─ 2. Fall back to OSM ground-level types
    │     └─ reject if h_mean above per-type ceiling
    │        (don't label tree canopy as `road`)
    └─ 3. If neither matches confidently → unlabelled (skip from training)
```

Key functions:
- `rasterize_cadastre_labels()` — cadastre polygons → 1m raster, height-masked
- `match_segments_via_raster()` — segments → cadastre via majority pixel vote
- `match_segment_to_cadastre()` — fallback polygon-overlap matcher
- `match_segment_to_osm()` — OSM road/landcover matching

## Constants worth knowing

Top of file:
- `_GROUND_SURFACE_CODES` — cadastre codes that mean "bare ground".
  Trees/buildings overlapping these get masked.
- Per-code nDSM ceilings — above this height the cadastre label is
  probably wrong (tree above road).
- `_NDVI_MAX_CADASTRE` (0.25) — above this, cadastre ground-surface
  labels are vetoed.
- `_OSM_HEIGHT_CEILINGS` — same idea for OSM types.

Retuning these directly affects training set quality. Bump
conservatively and re-train.

## Circuit breaker

openEO 503/timeouts are common. `_read_circuit_breaker()` /
`_write_circuit_breaker()` use a file-backed cooldown so subprocesses
share state. When tripped, NDVI/SAR are skipped for the cooldown
window instead of burning 3×180s timeouts per KG.

## Model artefact

`/tmp/learned_classifier/model.joblib` (saved every 10 KGs). Loaded by
`learned_classifier.py` on next request. Feature order MUST match
`learned_classifier.FEATURE_KEYS` and
`object_segmentation.extract_object_features()`. Changing feature
count silently breaks inference — retrain after any feature change.

## Operations

```bash
sudo systemctl status rf_train
sudo systemctl restart rf_train
tail -f /tmp/rf_train_4000kg.log

# Status (via app.py)
curl localhost:8000/api/v1/training/status | python3 -m json.tool

# Force fresh training run on existing checkpoints
rm -f /tmp/learned_classifier/model.joblib
sudo systemctl restart rf_train   # picks up checkpoints, retrains

# Inspect checkpoint
python3 -c "
import numpy as np
d = np.load('rf_training_data/checkpoints/kg_91109.npz', allow_pickle=True)
print(d.files); print('X', d['X'].shape, 'y', d['y'].shape)
"

# Sister script for smaller runs
python3 train_rf_100kg.py        # earlier 100-KG version, superseded
```

## When to re-train

- Added/renamed an object type → retrain (label space changed).
- Added/removed an RF feature → retrain (feature dim changed).
- Tuned ground-truth filters above → retrain (label distribution changed).
- Cadastre or OSM data significantly updated upstream → retrain.

Don't retrain casually — 4000 KGs takes days even with cached rasters.

---

*See `AGENTS.md` for the project map.*
