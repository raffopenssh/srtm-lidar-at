# segv2 — handover (2026-09-12)

Read `segv2/README.md` first (fleet-safety contract, why-v2, product notes).
This file is the *state + next steps* for whoever continues.

## What exists (all committed, nothing wired into the fleet)

| File | Status |
|---|---|
| `segv2/gpkg_fetch.py` | done. `fetch(code,'full_gpkg')` → `/tmp/segv2_gpkg/…` (8-stream range dl, ~4.6 MB/s), audit → `data/segv2/audit.jsonl` (md5/size vs manifest, integrity_check, layer inventory, per-raster fill). `release()` deletes. |
| `segv2/gpkg_raster.py` | done. `FullGpkg`: reads DTM/DSM/nDSM/DTM_YYYY/DSM_YYYY (nodata→NaN), Ortho_YYYY RGB(+NIR), NDVI, WorldCover, SAR_VV/VH, Hansen_*; `read_segment_type()` decodes the v1 palette PNG tiles (edge tiles are RGBA → mapped via SEGMENT_COLORS). |
| `segv2/labels.py` | done. v2 label stack (see README). `LabelContext(kg, bbox3035)` fetches cadastre (export/geojson, gzip), OSM (`/osm/geometry?cat=road,rail,water,water_area`), INVEKOS (local GPKG `data/invekos/INSPIRE_SCHLAEGE_2024-1_POLYGON.gpkg`, 4.5 GB, EPSG:31287, bbox-indexed reads ~0.1 s). `rasterize(tf, shape, ndsm, ndvi)` → (label u8, source u8, weight f32) with height/NDVI vetoes. Verified visually on KG 19570 (roads = OSM pavement, verges → grass, fields = INVEKOS). |
| `segv2/features.py` | done. `pixel_layers()` (all layers from GPKG + harmonics from Zenodo tile cache, cache-only), `segment()` (v1 params; optional OSM hard edges), `extract()` → DataFrame with all 67 v1 `FEATURE_KEYS` + `V2_EXTRA_KEYS` (OSM distances, neighbour context, height distribution, NDWI/SAVI, flight-year gaps, per-year change). Vectorised (bincount/lexsort) — 1500² tile ≈ 5 s features, 11 s segmentation. |
| `segv2/build_dataset.py` | done, **running in tmux session `segv2build`** (`--n 150 --seed 1`, stratified by size band + 1°×2° cell). Output `data/segv2/dataset/<code>.parquet` + `.meta.json`, progress in `data/segv2/build_log.jsonl`, stdout `data/segv2/build_stdout.log` (ends with `FINISHED`). ~3.5 min/KG. KG mask = cadastre parcel union; tiles < 2 % coverage skipped. Columns: features, `y` (v2 label mode), `y_purity`, `y_src`, `y_weight`, `v1_type` (deployed pipeline's answer on the same pixels), `kg`, `tile`, `centroid_e/n`. |

Not started: `segv2/train.py` (the harness), v2 segmentation eval, promotion.

## Findings so far

* v1 `learned_classifier.CADASTRE_TO_TYPE` / `train_rf_4000kg._LANDUSE_DESC_TO_CODE` use the **wrong** NS code table (48→road etc.). Correct table: `labels.NS_CODE_TO_TYPE` (from `GET cadastre…/api/v1/landuse/codes`). The 253 old checkpoints in `rf_training_data/` are therefore not reusable.
* Harmonics: Zenodo cache has only 55 cells (35 for 2024) → `harm_*` will be 0 for most KGs. Options: (a) treat as missing (LightGBM handles NaN — set 0→NaN when `harmonics is None`), (b) drop harm features from v2. Don't fetch from openEO in this work-stream.
* Cache manifest lists 4 zip names that 404 on Zenodo (`copernicus_harmonics_cell_47.0_48.0_12.0_14.0.zip`, 2 sar cells, 1 worldcover cell) — worth a manifest cleanup ticket, separate from segv2.
* INVEKOS covers agricultural land only; forest labels come from cadastre 56 (W) and are still coarse (the 3 m nDSM MIN_H veto helps). Consider adding tree labels from `v1_type=='tree' & h_p90>8 & fused_ndvi>0.5` as a *weak* source only if forest recall suffers.
* Zenodo download from the primary saturates ~5 MB/s; if throughput becomes the bottleneck run `build_dataset.py` on a parked/idle peer (same repo, same manifest via sync) — never on a running frontier peer.
* pyarrow + lightgbm were pip-installed (`--user --break-system-packages`) on the primary only.

## Next steps (in order)

1. **`segv2/train.py`** — the harness:
   * load all parquets; keep rows `y!='' & y_purity>=0.6 & kg_frac>=0.5`; sample weights `y_weight × purity`.
   * class merge for training: `hedge→shrub`? (decide by count), `greenhouse` keep if ≥300 rows, `wetland/glacier/rail` keep if ≥300 else drop.
   * folds: `GroupKFold(5)` by parent `kg` (so split blocks stay together).
   * models: (A) **v1 baseline** = `data/best_model/rf_model.joblib` predicting from the 67 `FEATURE_KEYS` columns (map `excavation/fill→earthwork`), (B) RF same hyper-params on relabelled data, (C) `lightgbm.LGBMClassifier` on FEATURE_KEYS, (D) LGBM on ALL_KEYS (+context), (E) D + second-stage neighbour-probability features (stacking: mean predicted proba of adjacent segments — adjacency must be recomputed or stored in the build step).
   * metrics: macro-F1, per-class F1, weighted-F1, ECE (10-bin), confusion matrices; also **agreement of `v1_type` with `y`** as the "deployed pipeline" reference (that's what's on Zenodo today).
   * report → `data/segv2/report.md` + `report.json`. Promotion rule in README.
2. If D/E win: `segv2/model_v2.py` with a `predict(df)` API mirroring `LearnedClassifier.predict_batch`, saved to `data/model_v2/` (LGBM, ~10-50 MB). Feature-importance + SHAP summary in the report.
3. Segmentation v2 eval: rebuild 20 KGs with `--v2-edges` into a separate dir (add an `--out` flag) and compare boundary recall vs OSM lines + label purity distribution (higher mean purity = segments straddle fewer classes).
4. Only then (separate conversation): wiring plan — `object_segmentation.segment_and_classify` gets a `model='v2'` path; `austria_processor` product version `v2` (new deposits; `version` field already exists in manifest entries); `search_index` gains `kg_products(kg_code, version, product, depo_id, url, size, uploaded_at)` + `?version=` on the API; reprocessing reads rasters from the existing v1 full GPKGs (`gpkg_raster.py`) instead of BEV. Also implement the README "v2 product design notes" (5 m terrain layer, tree_apices layer, als_meta, GPKG overviews). Roll out via director only after the promotion criteria are met — **peers stay on v1 until then; beware the director auto-pushes `main` every tick, so any wiring commit must be behind a flag/version gate until promotion.**

## Ops

```bash
tmux attach -t segv2build                 # builder
tail -f data/segv2/build_stdout.log | grep segv2.build
python3 -c "import json;[print(json.loads(l).get('code'),json.loads(l).get('n_good'),json.loads(l).get('error')) for l in open('data/segv2/build_log.jsonl')]"
python3 segv2/build_dataset.py 12105 --keep-gpkg    # single KG
```
`data/segv2/dataset/*.parquet` and `data/invekos/` are gitignored (large); audit/build logs are committed.
