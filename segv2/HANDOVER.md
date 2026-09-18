# segv2 — handover (state as of 2026-09-18 05:45 UTC)

Read `segv2/README.md` first (fleet-safety contract, why-v2, product notes). This file is
**current state + open work only**. The chronological log (dataset audit, fake-NIR / flight-
date defects, harness runs A–H, class-set decision, live-test bug hunt) lives in git:
`git log -p -- segv2/HANDOVER.md`.

## Where we are

* **Model shipped IN THE REPO**: `data/segv2/models/model_G.joblib` (36 MB, sha1 `7089ca645ff5…`)
  + `model_G.meta.json` — committed 2026-09-18 so every peer / new client gets it via `git pull`.
  Zenodo mirror as fallback (`model_v2.MODEL_ZENODO`, depo 22824068/22824064, `ensure_model()`). LightGBM, ALL_KEYS minus `harm_*`, 14 classes
  (crop garden glacier grass orchard parking rail road rock roof shrub tree vineyard water;
  `bare_soil→rock` merged, `earthwork`/`path` rule-only). 5-fold GroupKFold by parent KG on
  1.23 M segments / 146 KGs: **macro-F1 0.781 / wF1 0.921 / acc 0.925 / ECE 0.040** (v1 RF on
  the same rows: 0.283). Reports: `data/segv2/report_v2_final.md`, `report_model_v2.md`
  (gain/SHAP; regenerate with `python3 segv2/model_v2.py`).
* **v2 is wired but OFF fleet-wide**: every v2 path is gated on `SEG_MODEL_VERSION=v2` (env,
  default `v1`) or `model=v2` request param on `/api/v1/segment`. Peers produce v1 products.
* **WILHELM live test done**: `/?share=WILHELM` (v1) vs `/?share=WILHELMv2` (v2). Both ≈ 50 s
  wall; v2 inference is 3.9 s of it (1 s context fetch + LightGBM). v1: tree 1242 / grass 99 /
  crop 70 / tree_loss 55 / shrub 53 / hedge 18. v2: tree 803 / grass 414 / shrub 149 / rock 123
  / garden 20 / tree_loss 15 / crop 13, 10 INVEKOS overrides.

## Files (all committed)

| File | Role |
|---|---|
| `gpkg_fetch.py` | Zenodo full GPKG → `/tmp/segv2_gpkg/` (8-stream range dl), audit → `data/segv2/audit.jsonl`. 403/404/410 → `FileNotFoundError` (`gpkg_gone`). |
| `gpkg_raster.py` | `FullGpkg` reader. NIR is real **iff** `CIR_YYYY` exists (`nir_years()`, `read_ortho_stack()`); uint8 tables return band 0. |
| `acquisition.py` | Real ALS flight years per pixel from BEV flight blocks (`als_year_rasters`), ortho operate flight year, `dsm_flight_year(bbox)`. |
| `labels.py` | v2 label stack: cadastre (`NS_CODE_TO_TYPE`, corrected table) + OSM pavement/rail/water + INVEKOS (API `farm-subsidies-austria.exe.xyz`, `nearest_to=<flight yr>`; local 2024-1 GPKG fallback). `MIN_NDVI`/`MAX_H`/`AGE_DECAY`. |
| `features.py` | `pixel_layers()`, `segment()`, `extract()` → all v1 `FEATURE_KEYS` + `V2_EXTRA_KEYS` (dist_*, nb_*, multi-year NDVI, flight-year gaps; change features NaN when span==0). |
| `build_dataset.py` | per-KG parquet (`data/segv2/dataset/`, 172 KGs; 72321 has no full GPKG on Zenodo). Flags `--n --alpine --out --v2-edges --keep-gpkg --skip-done`; in-process retry of transient errors. |
| `train.py` | harness A/B/C/D/E/F/G/H/P, `derive_ndvi_vetoes`, `MERGE`, `DROP_CLASSES`, `--save-final X` → `models/model_X.joblib`. ~80 min per LGBM model on the primary. |
| `model_v2.py` | `ModelV2.load(letter|path)` (env `SEG_V2_MODEL`, default G), `predict(df)→(types, conf, proba)` with post-hoc NDVI class vetoes, `get_model()` singleton, `report()`. |
| `inference.py` | live-path adapter: `layers_from_arrays(...)`, `Context(bbox, flight_year, parcels, footprints)` (OSM + INVEKOS + cadastre fast path, once per KG/request), `classify()` = model + INVEKOS post-hoc (`frac≥0.7`, veg-like, `h_p90≤MAX_H` → Schlag type, conf 0.9, source `v2+invekos`), `run()` per tile. Feature parity with the training path verified on 19570. |
| `parcel_elevation.py` | parcel-outline DTM/nDSM profiles → JSON `parcels.details[].outline_z` (~550 B/parcel) + light-GPKG table `parcel_outline_z`; `decode()`, `outline_points()`. |
| `seg_live.py` | `python3 segv2/seg_live.py v2|v1` — POST WILHELM geometry to the running server, dump `/tmp/wilhelm_<m>.json`, print type/source counts. |

## Wiring (what v2 changes when enabled)

* `object_segmentation.segment_and_classify(model=, v2_context=, parcels=, footprints=)`: **Step
  3c** after v1 feature extraction runs `inference.run` on the same `labels`; results replace
  `rf_results`, so infra override / NDVI override / cadastre calibration / grouping are shared.
  Falls back to v1 RF with a warning on failure. `stats.model`, `stats.classifier`
  (`lgbm_v2:<hash>` / `rf_v1:<hash>`), `v2_classified`, `v2_invekos_override`.
* New `OBJECT_TYPES` `wetland=6 rail=24 glacier=42` synced in `app.py`, `austria_processor.py`,
  `static/index.html`, `parcel_compact.TYPE_LETTER` (`i I m`).
* `austria_processor`: `VERSION="v2"` iff env v2 (manifest/JSON/Zenodo metadata); one
  `inference.Context` per KG; `ParcelOutlineProfiler` per tile (+ top-up from raster sidecars
  / lazy tile cache for checkpoint-restored tiles); JSON `summary.model`, light GPKG
  `parcel_outline_z`.
* `app.py /api/v1/segment`: `model=` param; `meta.model`, `meta.model_classifier`,
  `meta.v2_trained_at/n_train/cv_macro_f1/classes`; per-feature `invekos_type/snar/frac`,
  `type2`, `type2_confidence`. `index.html` legend shows the v2 model line.

## Invariants / lessons (keep)

* Never ship a model that leans on `harm_*` — harmonics exist for ~20 % of KGs only and are not
  fetched in the v2 pass; G ≈ D (−0.002) proved they carry nothing at this coverage.
* `dist_*` (OSM/cadastre geometry) is worth ≈ +0.08 macro-F1, almost all in road/rail/water/
  parking/garden — these are partly *copied labels*. Honest recognition floor (H) ≈ 0.60.
* RGB and NIR must come from the **same ortho year** at inference (processor does this).
* `import app` in a script spawns peer-sync threads — test the app path via the running server.
* segv2 never writes to the Zenodo tile cache. Dead `cache_manifest.json` entries are handled
  in `zenodo_cache.py` (tombstones `size=0`, `ZipIndex` negative cache, `_RemoteFileLost`).
* Fleet-side leftover: 72321's `_full_gpkg` manifest entry points at an empty deposit
  (20492929); the coverage oracle only checks `_json`, so it will not be re-picked on its own.
  Worth a HEAD sweep over all `*_full_gpkg` bucket URLs.

## v2 rollout work-stream (started 2026-09-18, IN PROGRESS — see `docs/v2-upgrade.md` once written)

Goal: v2 = the product. Fresh KGs processed with v2; already-done v1 KGs *upgraded* from their
own Zenodo full GPKG (no BEV/openEO traffic, no credential → any idle peer can do it); v1 files
stay on Zenodo untouched; primary keeps v2 docs in a blob DB and drops the 22 GB of v1 JSON files.

**Decided design**
* Products: `_json_v2` = `<code>_v2.json.gz` (kgjson/2), `_light_gpkg_v2` = `<code>_light_v2.gpkg`.
  `_full_gpkg` unchanged (rasters identical; only fresh KGs write it). Fresh v2 KGs ALSO upload the
  legacy `_json` (compact, v2 content) — `_json` stays the fleet-wide completion marker (44 call
  sites depend on it; do not rename). Upgrades never touch `_json`/`_light_gpkg`/`_full_gpkg`.
* Triple-done logic (`app.py` zen_stall, ~4532/4959/18493) must accept `_light_gpkg_v2` as light.
* Primary: `data/kg_v2_store.db` (separate from the rebuildable search index) holds verified
  gz blobs; `search_index._select_kg_files_for_parent` reads the store first; then the v1
  `json/<code>.json` is deleted and `v1_bytes_freed` recorded (→ `process.txt` `v2:` line).
* Dispatch: upgrade candidates (v1 done, no `_json_v2`) fill the cache-only whitelist when
  `ready_kgs` is low; processor decides per KG (has `_json`+`_full_gpkg`, no `_json_v2` → upgrade).
  Peer downloads the full GPKG (few streams, disk-checked), runs `process_one_kg(source_gpkg=…)`,
  skips gpkg_full + early uploads, verifies, uploads light_v2 then json_v2, deletes the GPKG.

**Done (committed)**
| File | Status |
|---|---|
| `kg_json_v2.py` | codec; lossless round trip verified on 4 KGs; 21.6 MB pretty → 0.99 MB gz (18–22×) |
| `kg_v2_store.py` | primary blob store (WAL SQLite, stats(), v1_bytes_freed) |
| `v2_source.py` | GPKG raster shim: monkeypatches `raster_io.read_dtm_dsm`, `ortho_io.read_ortho_for_als/pick_rgbi_year_for_als`, processor globals `_fetch_copernicus_for_tile`/`_get_hansen_cache`; per-tile per-layer health check vs cadastre union (thresholds `THRESH`), fallback to BEV / cache-only tile cache; `summary()` → JSON `data_quality.v2_source`. Smoke-tested on `/tmp/segv2_gpkg/19570_full.gpkg`. |
| `v2_verify.py` | `verify_before_upload` (peer: codec, sections, model, tile stitching, parcels vs cadastre+v1, coverage ≥ v1, unclassified ≤ v1, light GPKG integrity/layers/counts, segment_type holes in union) + `verify_for_ingest` (primary). Not yet exercised on a real v2 product. |
| `segv2/model_v2.py` | `ensure_model()` / `available()` (Zenodo fallback) |

**Not done (next, in order)**
1. `austria_processor.py`: `MODEL_VERSION` default v2 when `model_v2.available()`; `process_one_kg(source_gpkg=, v1_doc=)` → install shim after cadastre (union_3035), skip oversize guard, skip gpkg_full + early uploads, write `_v2.json.gz` + `_light_v2.gpkg` (fresh v2: compact `<code>.json` too), run `v2_verify`, files `{"light_gpkg_v2","json_v2"}`; `upload_kg_to_zenodo` file_type `"gpkg" in key`, `delete_local = not key.startswith("json")`; `--v2-upgrade` arg / env → main(): completed = codes with `_json_v2`, pending = whitelist ∩ (has `_json`+`_full_gpkg`), kg dicts from v1 JSON bbox (block codes carry `_parent_kg_code`), GPKG download (Zenodo, disk check, ≤4 streams) then cleanup.
2. `app.py`: `/processing/start {v2_upgrade:true}`; peer-sync skips `_json` download when `_json_v2` present; v2 ingest thread (download blob → `verify_for_ingest` → `search_index.update_kg` from store → delete v1 file(s) → `kg_v2_store.put`); `process.txt` `v2:` line (`upgraded=N/8440 rate eta · fresh_v2=N · verify_fail=N · primary json_files=N disk=GB freed=GB avg/kg`); `/api/v1/model/v2`; `admin_update` pip-installs lightgbm if missing.
3. `search_index.py`: store-first in `_select_kg_files_for_parent`, `_kg_json_paths` users (3 sites) → data iterator; `product_version`, `zenodo_json_v2_url`, `zenodo_light_gpkg_v2_url` columns; `_by_parent` includes store parents.
4. `peer_director.py`: upgrade candidates into cache-only whitelist (cap `MAX_V2_UPGRADE_PEERS`, Zenodo Z-warn throttle), pass `v2_upgrade` in start payload.
5. Models page (`static/training.html`) + `static/docs.html` v2 section + kgjson/2 spec; `docs/v2-upgrade.md`; AGENTS.md link; `requirements.txt` + lightgbm.
6. Dry run one KG on an idle peer, then let the director roll out.

## Next steps (previous list, still valid after the above)

1. **Visual QA on 4 more KGs** (alpine, vineyard, urban, riparian): run `/api/v1/segment` with
   `model=v2` (adapt `seg_live.py`), POST result to `/api/v1/share` with `state.model='v2'`,
   rename to `<NAME>v2`, compare with the v1 share. Watch `journalctl -u srv` for `Step 3c` /
   `segv2 inference failed`.
2. **Processor v2 dry run** on ONE parked/idle peer (or the primary's processor in single-KG
   mode): systemd drop-in `Environment=SEG_MODEL_VERSION=v2` on `austria_processor`, **not**
   `srv`. Check JSON `model` block, `parcels.details[].outline_z`, light GPKG
   `parcel_outline_z`, manifest `version: v2`. The peer needs `model_G.joblib` —
   `data/segv2/models/` is gitignored: decide between a Zenodo deposit + download-on-first-use
   in `ModelV2.load`, or scp.
3. Seg-v2 edge eval: `build_dataset.py --v2-edges --out data/segv2/dataset_v2edges` on 20 KGs;
   compare mean `y_purity` and segment count vs the plain build.
4. Expose `outline_z` in the API (`/api/v1/kg/<code>`, `/parcel/<id>/detail`) once v2 JSONs
   exist (`parcel_elevation.decode`).
5. Fleet-wide v2 pass (separate conversation): README "v2 product design notes" (5 m terrain
   layer, `tree_apices`, `als_meta` table — a must, current GPKGs carry no acquisition
   metadata — GPKG overviews), `search_index` `kg_products(version…)`, reprocess from the v1
   full GPKGs via `gpkg_raster.py`, add `DTM/DSM_2025` where `als_acquisition.lookup(bbox,
   '20250915')` is newer (794 KGs). Roll out via the director only.
6. Later data levers: INVEKOS 2020/2016 Schläge for old-flight KGs; toponyms
   (`/toponyms?kg=`) as weak priors (vineyard/riparian/wetland); `ortho_io.read_ortho_for_als`
   NIR fallback for GPKGs without any CIR layer (`--fetch-gaps`).

## Ops

```bash
python3 segv2/seg_live.py v2                      # live v2 on WILHELM via running server
python3 segv2/model_v2.py                         # regenerate report_model_v2.md
python3 segv2/build_dataset.py 12105 --keep-gpkg  # single KG parquet
python3 segv2/train.py --models A,G --save-final G --report-suffix _x   # ~80 min, run in tmux
ls data/segv2/dataset/*.parquet | wc -l           # build progress (target 172)
```
`data/segv2/dataset/`, `data/segv2/models/`, `data/invekos/` are gitignored; audit/build/train
logs are committed. pyarrow + lightgbm are pip-installed on the primary only
(`--user --break-system-packages`).
