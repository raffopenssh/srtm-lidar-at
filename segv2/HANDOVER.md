# segv2 — handover (2026-09-13, status 2026-09-15 05:05 UTC: 167/173 parquets; two builder bugs fixed, last 5 KGs re-running in tmux `segv2build`)

Read `segv2/README.md` first (fleet-safety contract, why-v2, product notes).
This file is the *state + next steps* for whoever continues.

## What exists (all committed, nothing wired into the fleet)

| File | Status |
|---|---|
| `segv2/gpkg_fetch.py` | done. `fetch(code,'full_gpkg')` → `/tmp/segv2_gpkg/…` (8-stream range dl, ~4.6 MB/s), audit → `data/segv2/audit.jsonl` (md5/size vs manifest, integrity_check, layer inventory, per-raster fill). `release()` deletes. |
| `segv2/gpkg_raster.py` | done. `FullGpkg`: reads DTM/DSM/nDSM/DTM_YYYY/DSM_YYYY (nodata→NaN), Ortho_YYYY RGB(+NIR **only if `CIR_YYYY` exists** — `nir_years()`, `read_ortho_stack()`), NDVI, WorldCover, SAR_VV/VH, Hansen_*; `read_segment_type()` decodes the v1 palette PNG tiles. |
| `segv2/acquisition.py` | done (2026-09-13). Real flight dates: `als_year_rasters(tf, shape, mosaic_year)` per-pixel DSM/DTM flight year from BEV flight blocks (handles `DTM:2019, DSM:2009`, ranges, multi-block tiles); `ortho_flight_year(bounds, slot)` via operate id; `audit_kg(bounds)` gate. |
| `segv2/labels.py` | done. v2 label stack (see README). `LabelContext(kg, bbox3035)` fetches cadastre (export/geojson, gzip), OSM (`/osm/geometry?cat=road,rail,water,water_area`), INVEKOS (local GPKG `data/invekos/INSPIRE_SCHLAEGE_2024-1_POLYGON.gpkg`, 4.5 GB, EPSG:31287, bbox-indexed reads ~0.1 s). `rasterize(tf, shape, ndsm, ndvi)` → (label u8, source u8, weight f32) with height/NDVI vetoes. Verified visually on KG 19570 (roads = OSM pavement, verges → grass, fields = INVEKOS). |
| `segv2/features.py` | done. `pixel_layers()` (all layers from GPKG + harmonics from Zenodo tile cache, cache-only), `segment()` (v1 params; optional OSM hard edges), `extract()` → DataFrame with all 67 v1 `FEATURE_KEYS` + `V2_EXTRA_KEYS` (OSM distances, neighbour context, height distribution, NDWI/SAVI, flight-year gaps, per-year change). Vectorised (bincount/lexsort) — 1500² tile ≈ 5 s features, 11 s segmentation. |
| `segv2/build_dataset.py` | done; **rebuild running in tmux `segv2build`** (see audit section). Flags: `--n N` stratified, `--alpine N`, `--out DIR`, `--v2-edges`, `--keep-gpkg`. Output `data/segv2/dataset/<code>.parquet` + `.meta.json`, progress in `data/segv2/build_log.jsonl`, stdout `data/segv2/build_stdout.log` (ends with `FINISHED`). ~3.5 min/KG. KG mask = cadastre parcel union; tiles < 2 % coverage skipped. Columns: features, `y` (v2 label mode), `y_purity`, `y_src`, `y_weight`, `v1_type` (deployed pipeline's answer on the same pixels), `kg`, `tile`, `centroid_e/n`. |

| `segv2/train.py` | done (2026-09-13). Models A (v1 RF), P (v1_type on Zenodo), B (RF relabelled), C (LGBM FEATURE_KEYS), D (LGBM ALL_KEYS), E (D + kNN-neighbour OOF proba). GroupKFold by parent kg, macro/weighted F1, per-class, ECE, confusion, README promotion check. Report is rewritten after every model → `data/segv2/report.md|json`. `--save-final D` → `data/segv2/models/model_D.joblib`. **Full run in tmux `segv2train`**, log `data/segv2/train_full.log`, ends with `FINISHED`. Quick smoke (20 %, 2 folds): A macro-F1 0.307 / acc 0.67, P 0.246 / 0.54, C 0.621 / 0.86, ECE 0.153→0.079. Classes <300 rows dropped in the quick run (earthwork/water/rail/bare_soil/wetland/path/greenhouse) — the full run keeps earthwork/water/rail/bare_soil. |

v1 build (invalid, see below): 143 KGs ok, 9 errors. 1.10 M segments, 559 k passed the filter.

## ⚠ 2026-09-13 audit: dataset v1 was invalid — full rebuild running

Two defects were found in the first build (`data/segv2/dataset_v1_fakenir/`,
kept for reference; the harness numbers below are from it and are **not** to
be quoted):

1. **Fake NIR in 109/143 KGs.** The full GPKG writes `Ortho_YYYY` as 4-band
   only when NIR was fetched; otherwise band 4 is a GDAL alpha plane (255).
   `Ortho_2024` has no `CIR_2024` for ~75 % of Austria (BEV 20250415 RGBI
   series covers ~7 operates), so `read_ortho()` took alpha as NIR → `nir_mean`
   saturated at 255 in 99.97 % of those segments; `ndvi/ndwi/savi/nir_*` were
   `(255−R)/(255+R)`. The segmentation's fused gradient (ndvi .20 + nir .15
   weight) was affected too, so segment shapes differ from the deployed
   pipeline's. Fix: `gpkg_raster.nir_years()` (contract: real NIR iff
   `CIR_YYYY` exists), `read_ortho_stack()`; features use RGB from the newest
   year and NIR/NDVI from the newest *real* NIR year (`nir_year` feature).
   Every KG has real NIR from ≥1 year (`CIR_2020` in 148/148, `CIR_2023` 70,
   `CIR_2024` 34); the builder now hard-fails on `no_real_nir` and asserts NIR
   is not constant.
2. **Time was not factored in — and was wrong.** GPKG layer names are mosaic
   labels, not flight dates. From the vendored BEV flight blocks
   (`als_acquisition.py`, all 8440 KGs resolve): **64 % of KGs have the same
   flight in all three mosaics** → `h_change`, `dtm_change*`, `temporal_h_std`,
   `stability` (top-20 features in model C!) were resampling noise ÷ a
   fictitious 2-yr span; the other 36 % have real spans of 1–15 yr; 106 KGs
   have `DTM:2019, DSM:2009` splits; 1747 KGs straddle flight blocks. Ortho
   "2020" is the 2018–2021 series. Fix: `segv2/acquisition.py` paints
   per-pixel DSM/DTM flight years per mosaic, resolves the ortho operate's
   flight year; features: `dsm_year/dtm_year` real, `dsm_age = 2024 −
   flight`, `dtm_dsm_split`, `years_span` real, change features **NaN when
   span == 0** (LightGBM treats as missing), `_per_year` rates by real span,
   `ortho_lidar_gap`/`nir_lidar_gap` real, Hansen 3-yr loss anchored on the
   DSM flight year. New multi-year BEV spectral: `ndvi_y_first/last`,
   `ndvi_trend_per_year`, `ndvi_tstd`, `ndvi_years_span`,
   `ndvi_ndsm_coherence` (tall∧green).

Smaller fixes in the same pass: nDSM = clip(DSM−DTM) where both valid (GPKG
nDSM layer has ~7 % NaN holes); WorldCover 0/255 → NaN; label weights of
time-sensitive classes decay with `dsm_age` (`labels.AGE_DECAY`, since
INVEKOS 2024 / live cadastre+OSM label 2006–2024 pixels); builder writes an
`acquisition` audit block per KG and skips `als_flight_blocks_unknown`.

**Rebuild** (tmux `segv2build`, `data/segv2/build_stdout.log`, ends with
`FINISHED`; per-KG lines in `data/segv2/build_log.jsonl`): the 143 old codes +
30 alpine (`--alpine`, lowest building-density T/S/V/K parents for rock /
bare_soil support) = 173 codes in `data/segv2/rebuild_codes.txt`. ~3.5 min/KG
when Zenodo is healthy → **~10 h**; Zenodo has been returning 504 since
~07:57 UTC and the fetcher backs off (12 tries, ≤5 min each per range), so
add the outage length. **Check progress with** `ls data/segv2/dataset/*.parquet | wc -l` (target 173) — NOT `grep -c done: build_stdout.log`: stdout is rotated on every restart (`build_stdout_run1.log` 34, `run2.log` 105, current run 27+), so the per-file count plateaus. Per-KG time on the alpine/large KGs is 7–14 min, not 3.5.

**Transient-failure retry** (added 14:05 UTC): `build_dataset.main()` now runs a
second pass over every code whose meta had an `error` other than
`no_parcels`/`no_tiles` (so a 504 storm exhausting the fetcher's 12 tries →
`no_full_gpkg` is retried, not written off). The build was restarted at 14:06 UTC on the new code (`--skip-done` resumed
from the 34 parquets; first run's stdout kept as `build_stdout_run1.log`), so
the in-process retry pass applies; `FINISHED` is the final marker.
`segv2/retry_failed.sh` is kept as a manual chained-retry tool. Zenodo was fully unreachable 12:34–≥14:05 UTC (34 KGs done at that
point; 19713 and 22143 hit `no_full_gpkg` during the outage).

### 2026-09-15 03:20 UTC — run 3 finished, 6 KGs crashed, fix + re-run
Run 3 (`build_stdout_run3.log`) ended `FINISHED` with 166/173 parquets. The
6 missing (`73301-east 87102-north 72321 57004-west 73505-west 72010-south`)
all died in `features.extract` with `IndexError: arrays used as indices must
be of integer type` at `wgt = F["area"][bi]`: a tile with a single segment
has an empty adjacency list → `np.array([])` is float64. Fixed by forcing
`dtype=np.int64` on `ai`/`bi` (features.py:583); the in-process retry pass
re-hit the same bug so it could not help. After that fix the same 6 KGs failed on a **second** bug (runs 4/5):
`AssertionError: NIR is a constant alpha plane (rgb_sat=0.00)` on a tile
where the only CIR year (2020) has zero ortho coverage (73301-east tile 3:
Ortho_2020 fully black there, Ortho_2024 present) — the all-NaN NIR passed
the `nan_to_num(nan=255) >= 254` test and the rgb_sat was computed over the
black pixels. Fixes: `features.pixel_layers` only counts a year as a NIR year
for the window if >1 % of its RGB is non-black (else the tile has no NIR /
NDVI, NaN downstream); the builder check now looks at finite NIR pixels
only and rgb_sat over covered pixels only. 73301-east verified (80 k segs,
361 s). Remaining 5 re-running in tmux `segv2build` (`build_stdout.log`,
ends `FINISHED`; older stdouts `build_stdout_run3..5.log`). Expect 173.
**Caveat for training**: rows from such tiles have NaN `nir_*`/`ndvi_*`
(same as harmonics-missing rows) — LightGBM handles it; RF models A/B need
their existing NaN→0 fill.

### After the build (next conversation)
1. **Re-derive NDVI vetoes** from the real distribution: `labels.MIN_NDVI` and
   `train.MIN_NDVI_SEG/MAX_NDVI_SEG` were calibrated on the fake NDVI. Take
   ~5th percentile per vegetation class / 95th per sealed class of
   `ndvi_mean` on the new parquets, then run `train.py`.
2. `python3 segv2/train.py --models A,B,C,D,F,P --save-final D --report-suffix _v2`
   (~1 h). **F = D minus `dist_*`** is the honest recognition-only number for
   road/rail/water/roof (dist features share geometry with those labels). Also
   run a `harm_*`-free variant: harmonics exist for ~20 % of KGs only (58
   Zenodo cells) and will be missing at inference for most of Austria — don't
   ship a model that quietly leans on them.
3. Sanity on the new features before believing any number: `dsm_age` should
   vary 0–18 across KGs, `years_span` should be 0 for ~64 % of rows and NaN
   change features exactly there, `nir_mean` never ≥ 254 in bulk.

### Data completeness for the later fleet-wide v2 pass (all processed GPKGs)
* **No fetch needed for training.** Everything used is in the GPKG or local
  (INVEKOS gpkg, flight blocks, operate index) or Zenodo tile cache.
* **Harmonics**: real gap (crop/grass/orchard/vineyard phenology) but fetching
  = openEO credits via the director's credential pool; not for this stream.
  Keep NaN; measure the price with the harm-free variant.
* **ALS 2025 mosaic (20250915)** is NOT in the GPKGs (`DEFAULT_DATASET=20240915`).
  **794 KGs get a strictly newer flight in 2025** (e.g. Stmk 2009→2024) —
  a real two-date pair and a 15-yr-fresher DSM. Plan for the reprocessing
  pass: if `als_acquisition.lookup(bbox,'20250915')` is newer than 2024,
  read DTM/DSM 2025 from BEV and add `DTM_2025/DSM_2025`; features already
  handle arbitrary years/spans.
* **NIR fallback**: if a GPKG has no CIR layer at all, `ortho_io.read_ortho_for_als`
  can fetch RGBI live from BEV (opt-in `--fetch-gaps`, not default).
* **INVEKOS 2020 (or 2016) Schläge** from data.gv.at (~4 GB) would label
  old-flight KGs with contemporaneous field polygons — cheapest lever against
  the label-age problem for orchard/vineyard/crop. Second build.
* **`als_meta` table in the v2 GPKG is a must**, not a nice-to-have: the
  current GPKGs carry zero acquisition metadata; everything above was
  recoverable only from repo-vendored overlays.
* Hansen `gain` (2000–2012) not worth adding. S2/S1 predate no LiDAR flight
  problem is unfixable by fetching (S2 from 2015); `dsm_age` carries it.

## Harness results from the INVALID v1 dataset (reference only)

5-fold GroupKFold by parent KG, 559 k rows / 119 KGs, 15 classes:

| model | macro-F1 | weighted-F1 | acc | ECE |
|---|---:|---:|---:|---:|
| A v1 deployed RF | 0.240 | 0.641 | 0.666 | 0.151 |
| P v1 pipeline output (Zenodo `v1_type`) | 0.198 | 0.555 | 0.533 | — |
| B RF, v2 labels, 66 feats | 0.514 | 0.827 | 0.814 | 0.062 |
| C LGBM, 66 feats | 0.544 | 0.850 | 0.848 | 0.060 |
| D LGBM, ALL_KEYS (+context) | 0.655 | 0.883 | 0.888 | 0.056 |
| E D + kNN neighbour proba | 0.658 | 0.886 | 0.889 | 0.061 |

Quick 2-fold on v1 data with the 08:00 train.py changes (NDVI veto, sqrt
weights): D .661, **F (no dist_*) .605** — road .95→.56, garden .83→.76,
parking .66→.53; everything else within noise. So ~0.05 macro-F1 of D is
"agrees with OSM geometry". Rock/grass confusion root cause: all 14.6 k rock
rows came from 7 KGs (17 k from one), and ~5 % of alpine "grass" labels sat
on scree — hence the alpine picker + vegetation NDVI veto.

`train.py` state: models A,B,C,D,E,F,P; `--class-weight sqrt|balanced`
(default sqrt — `balanced` gave bare_soil 700× tree's weight);
`--report-suffix`; `--save-final` supports B/C/D/F; ECE, per-class F1,
confusion, README promotion check. `data/segv2/models/` not yet written.

## Findings so far

* v1 `learned_classifier.CADASTRE_TO_TYPE` / `train_rf_4000kg._LANDUSE_DESC_TO_CODE` use the **wrong** NS code table (48→road etc.). Correct table: `labels.NS_CODE_TO_TYPE` (from `GET cadastre…/api/v1/landuse/codes`). The 253 old checkpoints in `rf_training_data/` are therefore not reusable.
* Harmonics: Zenodo cache has only 55 cells (35 for 2024) → `harm_*` will be 0 for most KGs. Options: (a) treat as missing (LightGBM handles NaN — set 0→NaN when `harmonics is None`), (b) drop harm features from v2. Don't fetch from openEO in this work-stream.
* Cache manifest lists 4 zip names that 404 on Zenodo (`copernicus_harmonics_cell_47.0_48.0_12.0_14.0.zip`, 2 sar cells, 1 worldcover cell) — worth a manifest cleanup ticket, separate from segv2.
* INVEKOS covers agricultural land only; forest labels come from cadastre 56 (W) and are still coarse (the 3 m nDSM MIN_H veto helps). Consider adding tree labels from `v1_type=='tree' & h_p90>8 & fused_ndvi>0.5` as a *weak* source only if forest recall suffers.
* Zenodo download from the primary saturates ~5 MB/s; if throughput becomes the bottleneck run `build_dataset.py` on a parked/idle peer (same repo, same manifest via sync) — never on a running frontier peer.
* pyarrow + lightgbm were pip-installed (`--user --break-system-packages`) on the primary only.

## Next steps (in order)

1. ~~`segv2/train.py`~~ done — read `data/segv2/report.md`. Original spec kept for reference:
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
tmux attach -t segv2build                 # builder (rebuild, 173 codes)
grep -c "done:" data/segv2/build_stdout.log; tail -2 data/segv2/build_stdout.log
tail -f data/segv2/build_stdout.log | grep segv2.build
python3 -c "import json;[print(json.loads(l).get('code'),json.loads(l).get('n_good'),json.loads(l).get('error')) for l in open('data/segv2/build_log.jsonl')]"
python3 segv2/build_dataset.py 12105 --keep-gpkg    # single KG
```
`data/segv2/dataset/*.parquet` and `data/invekos/` are gitignored (large); audit/build logs are committed.

### ⏸ 2026-09-13 14:20 UTC — everything PAUSED for Zenodo recovery

Zenodo posted a status notice (slowness / intermittent outages from bot & AI
crawler traffic); our fetches were failing with HTTP 504 on every retry.
Operator asked to pause all machinery so Zenodo can recover:

* segv2 dataset rebuild: `tmux kill-session -t segv2build` (builder killed
  mid-run; `build_log.jsonl` is append-only, resume by rerunning
  `python3 segv2/build_dataset.py` for the codes not yet `done:`).
* Fleet: `POST /api/v1/director/stop` → director `mode=paused`, `active=-`,
  processors on at100/at106/at11/at68 hard-stopped (in-flight KGs 44206,
  45410, 57310-northeast, 86041-south will be re-picked from tile
  checkpoints when resumed).

**Resumed 2026-09-14 05:19 UTC** (Zenodo 200 but slow, ~14 s; range requests still
see intermittent 504s — fetch retries absorb them). Director `mode=auto`;
re-queued 23 oracle-incomplete hole KGs at position 0 (incl. the 4 interrupted
at stop); builder restarted in tmux `segv2build` (codes are positional, not
`--codes`), previous stdout kept as `build_stdout_run2.log`.
Note: at123 has been disk-full (1.5 GB free, untracked by diskstat) since
09-11 — predates 9240086; it exits `Processing complete` immediately when
activated, director then hands strips to other frontiers. at117 0 GB free
(json/ 4.4 GB role data).

**To resume** (once https://zenodo.org is healthy again):
```bash
TOKEN=$(cat data/admin_token)
curl -s -X POST -H "X-Admin-Token: $TOKEN" 'http://localhost:8000/api/v1/director/mode?mode=auto'
# then optionally: tmux new-session -d -s segv2build 'python3 segv2/build_dataset.py ... 2>&1 | tee -a data/segv2/build_stdout.log'
```
