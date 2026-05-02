# AGENTS.md — srtm-lidar-at

## Quick Reference

- **Live**: https://srtm-lidar-at.exe.xyz:8000/
- **Stack**: Python 3.12 / Flask / gunicorn / Leaflet
- **Restart**: `sudo systemctl restart srv` (app) or `sudo systemctl restart rf_train` (training)
- **Austria Processor**: `sudo systemctl kill -s SIGKILL austria_processor && sleep 2 && sudo systemctl start austria_processor`
- **Processor log**: `tail -f data/austria_processor/logs/processor.log`
- **Processor dashboard**: https://srtm-lidar-at.exe.xyz:8000/process.html
- **Logs**: `journalctl -u srv -f` or `tail -f /tmp/rf_train_4000kg.log`
- **API docs**: `/api/v1/docs/llm.txt`
- **Cadastre API**: https://cadastre-process-api.exe.xyz/api/v1/docs/llm.txt

## What This Is

Flask API + Leaflet web UI that analyses Austrian landscape from remote sensing data.
Draws on 6 data sources (BEV LiDAR DTM/DSM, BEV orthophotos, Sentinel-2 NDVI,
ESA WorldCover, Sentinel-1 SAR, Austrian Cadastre). Segments landscape into
25 object types + 11 groups via watershed segmentation + Random Forest classifier.

## File Layout

### Core
| File | Lines | Purpose |
|------|------:|----------|
| `app.py` | ~5900 | Flask API — all endpoints, async task system, progress tracking, director API |
| `static/index.html` | ~3100 | Single-file Leaflet UI (all JS/CSS inline) |
| `static/query.html` | ~600 | Query Explorer — split-pane UI over `/api/v1/flags`, `/query`, `/query/parcels`, `/query/compound`, `/query/nature`, `/feedback`. Examples panel auto-parses `/api/v1/docs/llm.txt` (~90 curl examples) → grouped browseable list + datalist autocomplete. Deep-link `?obj_ref=&kg=&lon=&lat=&type=` (or `?endpoint=&params=`) pre-filters + centres + zooms the map; row auto-highlighted on load. |
| `static/flag.js` | ~620 | Flag widget — text-selection chip → matches a snippet to an object via `/api/v1/flags/match`, shows candidates + flags + feedback form. The popover header has an ↗ icon linking to `/query.html` with the matched obj_ref pre-filtered. `SrtmFlag.install()` / `SrtmFlag.openFor({obj_ref})`. |
| `object_segmentation.py` | ~2200 | Main analysis pipeline: Felzenszwalb+RAG → per-object classify |
| `learned_classifier.py` | ~560 | Random Forest classifier (44 features, cadastre-trained) |
| `peer_director.py` | ~770 | Peer Director — orchestrates processing across multiple VMs (see section below) |

### Search Index
| File | Purpose |
|------|----------|
| `search_index.py` | SQLite FTS5 + R-tree index over all 8440 KGs + per-parcel `kg_parcels` table. Spatial/text/admin/aggregate/parcel queries <50ms. Auto-rebuilds on new JSONs. |

### Cross-API Bridge
| File | Purpose |
|------|----------|
| `cadastre_bridge.py` | Joins cadastre API (parcels, legal refs, protected areas) with landscape analysis. Batch parcel enrichment, nature conservation scoring, query-based batch with all cadastre filter options + landscape post-filters. |

### Data I/O
| File | Purpose |
|------|----------|
| `raster_io.py` | Windowed reads from remote GeoTIFFs via `/vsicurl/` |
| `ortho_io.py` | BEV orthophoto reader (RGBI, 47 operates + DOP fallback) |
| `copernicus.py` | Sentinel-2 NDVI, ESA WorldCover, SAR, NDVI time series via openEO |
| `cadastre.py` | Building footprint fetcher + ground truth evaluator |
| `hansen.py` | Hansen Global Forest Change (GFC-2024-v1.12) |
| `osm_features.py` | OSM road/landcover via Overpass API |
| `bev_retry.py` | Retry wrapper for rasterio.open() — exponential backoff + proxy rotation |
| `tile_index.py` | 55-tile grid index, CRS transforms (EPSG:4326 ↔ EPSG:3035) |
| `geo_parse.py` | KML / GeoJSON / Shapefile / GPX / WKT parser |

### Feature Extraction
| File | Purpose |
|------|----------|
| `terrain_analysis.py` | Slope, aspect, TRI, TPI, curvature from DTM |
| `temporal_analysis.py` | Multi-date DTM comparison, 20 change event types |
| `texture_features.py` | GLCM texture from 20cm orthophoto (contrast/entropy/homogeneity) |
| `ndvi_harmonics.py` | Monthly NDVI → harmonic fit (mean, amplitude, phase) |

### Training
| File | Purpose |
|------|----------|
| `train_rf_4000kg.py` | Background RF training over 4000 KGs (runs as systemd service) |
| `train_rf_100kg.py` | Earlier 100-KG training script (superseded) |
| `calibrate.py` | Cadastre calibration utilities |

### Streamed GPKG Builder
| File | Purpose |
|------|----------|
| `gpkg_streamed.py` | Strip-streamed full-GPKG writer for large KGs (>100 Mpx). Keeps peak memory ~500 MB regardless of KG size. Called automatically by `build_full_gpkg_tiled()` when pixel count exceeds threshold. |
| `kg_splitter.py` | Splits large KGs (>28 tiles) into contiguous blocks with directional names. Parcel filtering at runtime. |

### Deprecated (kept for reference)
`landscape_classifier.py`, `object_classifier.py`, `scene_adaptive_classifier_patches.py`

## Services

| Unit | What | Config |
|------|------|--------|
| `srv.service` | gunicorn (2 workers, 4 threads, port 8000) + peer director thread | MemoryMax=3G, Restart=on-failure |
| `rf_train.service` | RF training background job (4000 KGs) | Restart=on-failure, RestartSec=30 |
| `austria_processor.service` | Austria processor (all KGs) | MemoryMax=8G, MemoryHigh=7G, Restart=on-failure |

All in `/etc/systemd/system/`. Source copies in repo root.

**On the primary**: `austria_processor.service` is disabled (director manages lifecycle).
The director thread inside `srv.service` starts/stops the processor via the REST API.

**On peers**: `austria_processor.service` is disabled (never auto-starts).
The director on the primary starts the processor via `POST /api/v1/processing/start`.
No `is_director` flag → no director loop in the peer’s gunicorn.

## API Endpoints

### Analysis
| Method | Path | Purpose |
|--------|------|----------|
| POST | `/api/v1/segment` | Main analysis — async when `async=true` |
| POST | `/api/v1/elevation` | Elevation enrichment |
| POST | `/api/v1/terrain` | Terrain characterisation |
| POST | `/api/v1/changes` | Temporal changes |
| POST | `/api/v1/changes/trees` | Tree growth analysis |

### Async Task System
| Method | Path | Purpose |
|--------|------|----------|
| GET | `/api/v1/segment/progress?task_id=` | Poll task progress |
| GET | `/api/v1/segment/result?task_id=` | Fetch completed result |
| POST | `/api/v1/segment/abort?task_id=` | Cancel running task |

### One-Stop URL
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/onestop?bbox=&format=` | Single-URL segment + download (async, queued) |

Tasks run in daemon threads. Progress tracked via JSON files in `/tmp/segment_progress/`.
Results stored gzipped in `/tmp/segment_results/`. Auto-cleaned after 4 hours.

**Auto-save**: When an async segment task completes, the result is automatically saved
as a share (`auto-<task_id[:8]>` in `data/shares/`). The `auto_share_id` is included
in the progress response so the frontend can recover results after page refresh.
The frontend stores the active `taskId` in `localStorage` and on reload resumes
polling or loads the auto-saved share.

### Overlays & Exports
| Method | Path | Purpose |
|--------|------|----------|
| POST | `/api/v1/segment/overlay` | Coloured PNG of segmentation |
| POST | `/api/v1/dtm/overlay`, `lidar/overlay`, `ortho/overlay`, `cir/overlay`, `hansen/overlay` | Tile overlays |
| POST | `/api/v1/export/geopackage` | Raster + vector layers in one GPKG (types/height filter) |
| POST | `/api/v1/export/kml` | Features as KML (types/height filter, group_by) |
| POST | `/api/v1/export/mbtiles` | Single layer as MBTiles (async) |
| POST | `/api/v1/lidar/geotiff`, `ortho/geotiff` | Raw GeoTIFF download |

### Shares & State
| Method | Path | Purpose |
|--------|------|----------|
| GET | `/api/v1/shares` | List saved shares (most recent first) |
| POST | `/api/v1/share` | Save share (dedup by content hash) |
| GET | `/api/v1/share/<id>` | Load share |
| POST | `/api/v1/share/<id>/rename` | Rename share ID |

Shares stored in `data/shares/` as `<id>.json.gz`. Max 1GB total, LRU eviction.
Contain: UI state + analysis result + cached overlay images.

### Utilities
| Method | Path | Purpose |
|--------|------|----------|
| GET | `/api/v1/layers?bbox=` | Layer availability for a bbox |
| GET | `/api/v1/info` | Server info |
| POST | `/api/v1/parse-geometry` | Parse uploaded geometry file |
| GET | `/api/v1/docs/llm.txt` | Machine-readable API docs |
| POST | `/api/v1/classifier/train` | Train RF on a bbox |
| GET | `/api/v1/classifier/status` | RF model status |
| GET | `/api/v1/training/status` | Background RF training progress |

### Search Index
| Method | Path | Purpose |
|--------|------|----------|
| GET | `/api/v1/index/status` | Index statistics (kg_count, processed, area, zenodo) |
| POST | `/api/v1/index/rebuild` | Rebuild index (~0.3s) |
| GET | `/api/v1/query` | Unified query — text/spatial/admin/type/hansen/buildings |
| GET | `/api/v1/kg/<code>` | KG JSON or index data with Zenodo links |
| GET | `/api/v1/parcel/<id>` | Parcel lookup via index + local JSON |

`/api/v1/query` params: `q=`, `kg=`, `parcel=`, `bbox=w,s,e,n`, `point=lon,lat`,
`state=`, `district=`, `gemeinde=`, `type=`, `hansen=true`, `new_buildings=true`,
`aggregate=true`, `processed_only=true`, `limit=`, `offset=`

### Cross-API Bridge (cadastre + landscape)
| Method | Path | Purpose |
|--------|------|----------|
| GET | `/api/v1/lookup` | Cadastre EDM lookup proxy (diacritics-insensitive) |
| GET | `/api/v1/query/parcels` | Fast SQL per-parcel index query — attribute + building + spatial filters, <50ms |
| GET\|POST | `/api/v1/parcels/batch` | Batch parcel enrichment — explicit IDs or query-based (GET uses query params with pf_ prefix for parcel filters) |
| GET | `/api/v1/parcels/landscape` | Query parcels with landscape filters (GET version of batch) |
| GET | `/api/v1/query/nature` | Nature conservation opportunity finder (conservation score 0-100) |
| GET | `/api/v1/parcel/<id>/detail` | Full combined parcel detail (both APIs) |
| GET | `/api/v1/kg/<code>/profile` | Combined KG profile (both APIs) |
| GET | `/api/v1/cadastre/legal/search` | Proxy: legal refs search |
| GET | `/api/v1/cadastre/protected_areas` | Proxy: WDPA protected areas |
| GET | `/api/v1/cadastre/landuse/distribution` | Proxy: landuse distribution |
| GET | `/api/v1/cadastre/landuse/codes` | Proxy: landuse reference codes |

`POST /api/v1/parcels/batch` supports three modes:
- **IDs mode**: `{"parcel_ids": ["63349-505/3", ...]}`  (max 200)
- **Cadastre query**: `{"query": {<cadastre /query params>}, "landscape_filters": {<post-filters>}}`
- **Compound (landscape-first)**: `{"compound": {<compound filters>}, "parcel_filters": {<per-parcel filters>}}`
  Starts from our landscape index → finds KGs → expands to parcels from KG JSONs → enriches with cadastre.
  This is the power query. Answers: "100 parcels with tree conf>0.8 area>800 + no buildings + SW aspect + rugged."

Compound filters: all `query_compound()` params — type_filters, landcover_filters, bbox, state,
district, gemeinde, aspect, dominant_type, phenology, terrain_class, quality_grade,
+ 70+ numeric min/max filters covering: terrain (slope, roughness, elevation, elevation_min/max,
elevation_range, steepness_max, tri), area (total_area, parcels, segments), buildings
(building_count, building_height, building_max_height, building_stories, building_stories_max,
building_pitched_pct, building_footprint, new_building_footprint/height/stories,
infrastructure, building_height_coverage), trees (tree_count, tree_height, tree_canopy_sqm,
tree_volume), vegetation (ndvi, vegetated_fraction, shannon_diversity), NDVI harmonics
(ndvi_amplitude, ndvi_harm_mean, ndvi_phase), SAR (sar_vv, sar_vh), temporal change
(dtm_change, volume_change, changed_segments, disturbed_volume, temporal_stability),
classification quality (confidence, rf_confidence, diverged_pct, rf_diverged_count,
rf_classified_pct, quality_score)

Parcel filters: `min_vegetated_fraction`, `max_vegetated_fraction`, `min_forested_fraction`,
`max_forested_fraction`, `dominant_type`, `types`, `min_type_fraction`, `min_ndsm_max`,
`min_elevation`, `max_elevation`, `min_slope`, `max_slope`, `min_parcel_area`,
`max_parcel_area`, `is_vegetated`, `min_rf_confidence`, `min_confidence`,
`min_hansen_recent_5yr`, `max_hansen_recent_5yr`, `min_hansen_total`, `max_hansen_total`,
`cadastre_has_buildings`, `cadastre_landuse`, `cadastre_min_area`, `roof_type`,
`min_stories`, `max_stories`, `sort`, `sort_dir`

**Per-parcel auto-classification** (`parcel_compact.classify_parcel`):
Every parcel gets an `auto_class` + `auto_subclass` + confidence at index
build. Reads frav, terrain (slope/tri/elev), top_trees mean height,
buildings (count/stories/footprint, attributed via PIP against
`vertex_heights`), and Hansen recent-5yr loss. 15-class taxonomy:
  forest (tall_forest|recently_thinned), young_forest (regenerating),
  wooded (open_woodland), meadow (rugged|orchard_meadow), alpine_meadow
  (pasture|high_alpine), cropland (fallow), vineyard, orchard, shrubland,
  built_up (apartments|multi_storey|house|dense), farmstead (with_house),
  infrastructure (road|mixed), water_body, disturbance (recent_clearfell|
  tree_loss|construction|earthwork), bare, mixed.
Query via `auto_class=`, `auto_subclass=`, `min_auto_class_confidence=`
on `/api/v1/query/parcels`.

**Per-parcel building rollup**: `kg_parcels` carries `building_count`,
`building_max_height_m`, `building_max_stories`,
`building_total_footprint_sqm` — spatially attributed at index build via
point-in-polygon (parcel `vertex_heights` → ray casting), with
nearest-centroid fallback. Filterable via `has_buildings=`,
`min_buildings/max_buildings`, `min_building_height/max_building_height`,
`min_building_stories/max_building_stories`.

Index: `data/search_index.db` (~5MB). SQLite FTS5 + R-tree. All queries <25ms.
Auto-rebuilt on startup and when new KG JSONs appear (60s poll).
Manual rebuild: `POST /api/v1/index/rebuild`.

Every KG record includes Zenodo download URLs when available:
- `zenodo_json_url` — JSON summary (~1MB)
- `zenodo_light_gpkg_url` — segments + enriched parcels/buildings (~100MB)
- `zenodo_full_gpkg_url` — all raster layers + vectors (~400MB)
- Also in `_links` object for convenience.

## Frontend (static/index.html)

Single HTML file (~2100 lines) with all CSS/JS inline. Key components:

- **Sidebar**: Endpoint selector, object type filter, option checkboxes, area input, Load/Analyse/Stop buttons
- **Map**: Leaflet with draw controls (polygon + rectangle only), layer panel, legend
- **Area input**: Compact display bar showing source (drawn/file/share), click to expand raw textarea. File drop/pick via 📎 button inside the bar.
- **Load dropdown**: `📂 Load` button fetches `/api/v1/shares` and shows recent shares + built-in Sample. Loads share via restoreShareResult().
- **Analyse**: Submits async task, shows ⏹ Stop button during processing, polls progress
- **Results**: Point markers on map, segment raster overlay, legend with type filtering, Download modal (Summary/JSON/GeoPackage/KML/MBTiles tabs)
- **Download/Share**: Visible as soon as geometry exists (draw/file/share). Download modal has type filter per tab.
- **GeoPackage tab**: Raster layers + vector segment polygons with height_class. Object type filter.
- **KML tab**: Features grouped by type or height_class, type filter, requires analysis results.
- **Area warning**: Soft hint when polygon > 100 ha.
- **Share**: 🔗 Share button saves state + result, generates permalink, supports renaming share ID
- **Clear**: Bin button (🗑) on draw toolbar clears everything with confirmation dialog
- **Layer panel**: Checkboxes + opacity sliders. Availability auto-checked via `/api/v1/layers`.
- **MBTiles tab**: Filters to only available layers using `/api/v1/layers` API. Hansen respects availability.

### Key JS Functions
- `getPostArgs()` — reads geometry textarea, returns `{ct, body}` for fetch
- `showResultOnMap(data)` — renders features + legend + overlay
- `restoreShareResult(data)` — restores full share state including overlays
- `clearEverything()` — resets all state to blank
- `updateGeoInputDisplay(label, source)` — updates the compact area input bar
- `checkLayerAvailability()` — debounced, hides unavailable layers in panel
- `buildMBTilesList()` — async, calls layers API to filter MBTiles options

### State Variables
- `lastResult` — last analysis JSON response
- `allFeatureData` — features array from result
- `overlays` — `{layerId: L.imageOverlay}` map
- `drawnItems` — L.FeatureGroup for drawn geometry
- `currentShareId`, `currentShareName` — active share tracking
- `_activeTaskId`, `_aborted` — async task abort control
- `hiddenTypes` — Set of types hidden via legend click
- `selectedTypes` — Set of types selected in dropdown filter

## Analysis Pipeline (object_segmentation.py)

1. Read DTM+DSM from raster_io (1m GeoTIFFs via HTTP range requests)
2. Optionally read: ortho (0.2m), NDVI, SAR, Hansen, cadastre
3. Compute fused gradient (Sobel on DTM/DSM/CHM/spectral)
4. Felzenszwalb over-segmentation (scale=150) + RAG boundary merge (threshold=0.12)
5. Per-segment feature extraction (44 features: height, shape, NDVI, texture, SAR, harmonics)
6. Classify via RF model if available, else rule-based `classify_object()`
7. Group adjacent compatible segments (tree→forest, roof→building, etc.)
8. Return GeoJSON features with properties

## 25 Object Types

| Category | Types |
|----------|-------|
| Vegetation | tree, shrub, grass, hedge |
| Water | water |
| Buildings | roof, greenhouse, solar_panel |
| Infrastructure | fence, wall, mast |
| Transportation | road, path, parking, bridge |
| Agricultural | crop, orchard, vineyard, garden |
| Terrain | bare_soil, rock |
| Disturbance | excavation, fill, tree_loss, construction |

## External Data Sources

| Source | Resolution | Access |
|--------|-----------|--------|
| BEV ALS DTM+DSM | 1m, 3 dates (2022/23/24) | HTTP range on remote GeoTIFF |
| BEV DOP RGBI | 0.2m, 47 operates | HTTP range on remote GeoTIFF |
| Sentinel-2 NDVI | 10m | openEO (client ID: `sh-19061cbb-...`) |
| ESA WorldCover | 10m | openEO |
| Sentinel-1 SAR | 10m | openEO |
| Hansen GFC | 30m | `/vsicurl/` on UMD servers |
| Austrian Cadastre | mm-precision | REST API (cadastre-process-api) |
| OSM | varies | Overpass API |

Caches: `/tmp/copernicus_cache/`, `/tmp/hansen_cache/`

### Zenodo Persistent Cache (`zenodo_cache.py`)

Local Copernicus/Hansen tile caches get evicted when disk approaches 5GB.
The Zenodo cache module persists these tiles on Zenodo so they survive eviction.

**Architecture**: Each local NPZ tile has a `.meta.json` sidecar (written by
`tile_cache._write_tile_meta()`) that records product type and grid coordinates.
The uploader uses these sidecars to group tiles into ZIP archives by product ×
0.5° latitude strip, then uploads to a single Zenodo deposit (depo 19650075).

**Upload flow** (`ZenodoCache.upload_all()`):
1. `_build_reverse_index()` reads `.meta.json` sidecars (+ fallback from `tile_bbox_index.json`)
2. Groups tiles by (product, lat strip)
3. For each group, compares local entry names vs cached remote ZIP central directory
4. If local ⊆ remote → skip. Otherwise, merge local + remote-only into new ZIP and upload.
5. Invalidates cached ZIP index after upload.

**Download**: On local cache miss, `tile_cache` calls `ZenodoCache.fetch_copernicus()`
or `fetch_hansen()`. Uses 2-3 HTTP range requests to read individual NPZ entries
from remote ZIP files via the cached central directory index.

**When uploads happen**:
- After each completed tile in the child subprocess (throttled to 30 min)
- After KG completion (forced)
- Before disk eviction of expensive tiles (forced)
- Before each KG in the parent process (throttled to 30 min)

**Key invariant**: Every `.npz` tile file MUST have a `.meta.json` sidecar.
Orphan tiles (no sidecar) are invisible to the uploader and waste disk.
`cleanup_orphan_tiles()` runs at processor startup to delete them.

**Manifest files** (don't confuse them):
- `data/austria_processor/cache_manifest.json` — Zenodo cache deposit (tiles)
- `data/austria_processor/zenodo_manifest.json` — KG product uploads (GPKGs, JSONs)

**Cached ZIP indices**: `data/austria_processor/zenodo_zip_index/*.json` — cached
central directories of remote ZIPs, keyed by MD5 of download URL. Invalidated
automatically after each upload. Stale indices cause false "local ⊆ remote" and
skip uploads — delete the directory to force re-fetch.

**Why not BEV/ortho?** BEV DTM/DSM/ortho are already COGs with efficient HTTP range
reads. At 1m resolution, all Austria = ~4TB (infeasible for Zenodo).

```bash
python3 zenodo_cache.py status      # show local + Zenodo tile counts
python3 zenodo_cache.py dry-run     # build ZIPs without uploading
python3 zenodo_cache.py upload       # upload local tiles to Zenodo
```

**Troubleshooting**:
- `Upload complete: 0 ZIPs, 0 tiles` — either no new tiles (normal when same
  Copernicus cells are reused across KGs), or all local tiles already on Zenodo.
  Check `python3 zenodo_cache.py status` for local vs remote counts.
- Tiles not uploading — check `.meta.json` sidecars exist alongside `.npz` files.
  Missing sidecars = orphans. Run `python3 -c "from zenodo_cache import cleanup_orphan_tiles; cleanup_orphan_tiles()"`
- Stale indices — `rm -rf data/austria_processor/zenodo_zip_index/` and re-flush.

## RF Training

Runs as `rf_train.service`. Script: `train_rf_4000kg.py`.

- Iterates 4000 random KGs, fetches cadastre+OSM ground truth + all raster data
- Checkpoints to `rf_training_data/checkpoints/kg_XXXXX.npz` — skips on restart
- Trains model every 10 KGs, saves to `/tmp/learned_classifier/`
- Status visible at `/api/v1/training/status` and in UI status bar
- Log: `/tmp/rf_train_4000kg.log`

## Developing

```bash
python3 app.py                        # Flask dev server on :8000
sudo systemctl restart srv             # restart gunicorn
journalctl -u srv -f                   # app logs
sudo systemctl restart rf_train        # restart RF training
tail -f /tmp/rf_train_4000kg.log       # training logs
systemctl status rf_train srv          # check both services
```

### Cross-Cutting Concerns (things that span multiple files)

These are the dangerous changes — they touch many files and are easy to break.

**Adding/changing an object type:**
1. `object_segmentation.py` → `OBJECT_TYPES` dict (canonical type→code mapping)
2. `object_segmentation.py` → `GROUP_TYPES` dict (if it belongs to a group)
3. `object_segmentation.py` → `classify_object()` (rule-based classification logic)
4. `app.py` → `SEGMENT_COLORS` dict (RGBA for overlays/exports)
5. `austria_processor.py` → `SEGMENT_COLORS` dict (**duplicate** of app.py's — must match)
6. `static/index.html` → JS `TYPE_COLORS` object (RGB for frontend legend — must match)
7. `learned_classifier.py` → `CADASTRE_TO_TYPE` (if cadastre has a matching land-use code)

**Adding/changing an RF feature:**
1. `learned_classifier.py` → `FEATURE_KEYS` list (canonical feature order, currently 57 keys)
2. `object_segmentation.py` → `extract_object_features()` (must populate the new key in feat dict)
3. Retrain the model: existing `.joblib` files become incompatible if feature count changes

**Changing the Copernicus credential set:**
1. Preferred: use the dashboard → 🔑 Credentials panel → “+ Add”, or
   `POST /api/v1/credentials {client_id, client_secret, label?, validate}`.
   New creds are validated (OIDC client-credentials probe), persisted to
   `data/austria_processor/copernicus_credentials.json`, and visible to
   the director on the next tick (no restart needed for new KGs).
2. The built-in seed list is `_BUILTIN_CREDENTIALS` in `copernicus.py`
   (~line 48). The runtime pool is the union of built-ins + the persisted
   store; user-added creds can be removed via the API, built-ins cannot.
3. `rm -f data/austria_processor/copernicus_paused` (removes throttle pause)
4. `rm -f data/austria_processor/openeo_circuit.json` (resets circuit breaker)
5. **Persistence**: `_exhausted_cred_indices` is mirrored to the credentials
   JSON (`exhausted=true`) so a fresh subprocess inherits the exhaustion
   state. The director treats `exhausted=true` and `last_status` in
   {`exhausted`,`invalid`} as “not valid” when computing
   `max_parallel_frontiers`. Call `POST /api/v1/credentials/validate`
   (no body) to re-probe everything, e.g. on the 1st of the month after
   credit renewals.
6. **Important**: the `@_retry_on_rotation` decorator tries `len(_CREDENTIALS)+1` attempts. Adding/removing credentials changes retry behaviour automatically.
7. **Per-credential usage telemetry**: every Copernicus call records
   success/error/rotated outcomes into
   `data/austria_processor/copernicus_credential_usage.json` (per-hour
   buckets, 7-day window). Hooked in `_run_datacube` (sync + batch) and
   `_download_month_sequential` (NDVI TS). `copernicus.list_credentials()`
   returns `usage:{success_7d, error_7d, rotated_7d, last_use, buckets,
   by_product}`; the dashboard 🔑 Credentials panel renders a stacked
   sparkline + totals + success rate per credential. To reset stats: `rm
   data/austria_processor/copernicus_credential_usage.json`. Multi-process
   safe via best-effort merge (coarse buckets tolerate occasional
   overwrites). When adding new code paths that hit Copernicus, call
   `copernicus.record_credential_usage(cred_index, kind, product)` on
   success/error so the dashboard stays accurate.

**Per-peer credential & lat-strip dedication (parallel frontiers):**
The director assigns disjoint slices of valid credentials and disjoint
cached lat strips to peers running frontier work in parallel.
* `min_creds_per_frontier` (default 2) in `peers.json` sets the minimum
  cred count per frontier peer. `max_parallel_frontiers = floor(valid /
  per)`.
* Capability-gated: only peers exposing `cred_subset_env` in
  `/api/v1/info→capabilities` get parallel work. Pre-upgrade peers run
  single-frontier as before (graceful upgrade).
* Pinning: each peer has an optional `pinned_role` in `peers.json`
  (`frontier`, `cache_only`, `idle`). Set via dashboard dropdown or
  `POST /api/v1/director/peers/<id>/pin {pinned_role}`.
* The processor honors `COPERNICUS_CRED_INDICES="0,2"` and
  `KG_LAT_STRIP_FILTER="[[47.0,47.5],[48.0,48.5]]"` env vars set by the
  director when starting the subprocess.

**Cache-only cache-miss avoidance:**
When a cache-only peer hits a missing tile, the processor records the KG
in `data/austria_processor/cache_miss_kgs.json` (on the primary, via
`POST /api/v1/processing/cache_misses`). The entry stores a fingerprint
of the relevant Zenodo cache strip; the director excludes the KG from
cache-only whitelists until the strip's manifest changes (i.e. new tiles
uploaded). Prevents re-hammering the API for KGs we already know are
incomplete in cache-only mode. Fingerprints are computed from
`updated_at` of the strip's `copernicus_*_strip_<S>_<N>.zip` and
`hansen_strip_<S>_<N>.zip` entries in `cache_manifest.json`.

**Changing tile grid / overlap:**
1. `austria_processor.py` → `_compute_tile_grid()` (tile_km, overlap_km params)
2. `tile_cache.py` → grid sizes per source (0.1° Copernicus, 0.5° Hansen)
3. Invalidate tile checkpoints: `rm -rf data/austria_processor/tile_checkpoints/`

### Navigation Cheatsheet

```bash
# Find any section in any file
grep -rn '# === SECTION' *.py

# Find a section in a specific file
grep -n '# ===' austria_processor.py

# Find all files that reference a type/feature/color
grep -rl 'SEGMENT_COLORS' *.py static/*.html

# Find where a function is defined
grep -n 'def process_one_kg' *.py

# Find the RF feature list
grep -A60 'FEATURE_KEYS = \[' learned_classifier.py

# Check which processor step is running
cat data/austria_processor/current_step.json | python3 -m json.tool

# Full project section index
grep -rn '# ===' *.py | sed 's/# === SECTION: //' | sed 's/ ===//' | column -t -s:
```

---

## app.py Code Map (~5700 lines)

All sections marked with `# === SECTION:` — use `grep -n '# ===' app.py`.

| Marker | Key contents |
|--------|---------------|
| `Processing queue` | `_TASK_SEMAPHORE` (max 2 concurrent heavy tasks) |
| `Async task system` | File-backed progress in `/tmp/segment_progress/`, result storage in `/tmp/segment_results/` |
| `Austria Processor endpoints` | Proxy to processor state: start/stop/pause/resume/status/log/manifest |
| `Geometry + parameter helpers` | `_get_geometry()`, `_get_params()`, `_validate_area()`, `_clean_polygon()` |
| `/api/v1/elevation` | DTM elevation enrichment |
| `/api/v1/terrain` | Terrain characterisation (slope, aspect, roughness) |
| `/api/v1/segment` | **Main analysis endpoint** — async segmentation pipeline, `_segment_core()`, `_segment_worker()` |
| `segment/overlay + raster rendering` | `_segment_rgba()`, `_render_seg_overlay()`, overlay cache |
| `export/geopackage` | GPKG export with raster layers + vectors, async via `_gpkg_worker()` |
| `export/kml` | KML export with grouping + styling |
| `export/mbtiles` | MBTiles export (async) |
| `/api/v1/changes` | Temporal change detection (multi-date DTM comparison) |
| `LiDAR/ortho overlay + download` | DTM/DSM/ortho/CIR/Hansen tile overlays + GeoTIFF download |
| `RF classifier training` | `/api/v1/classifier/train`, `/api/v1/classifier/status` |
| `/api/v1/docs + share` | Docs endpoint, share save/load/rename/list |
| `/api/v1/layers` | Layer availability check for a bbox |
| `/api/v1/onestop` | Single-URL segment + download (queued) |
| `/api/v1/parse-geometry` | Upload KML/GeoJSON/Shapefile/GPX/WKT |
| `/api/v1/share` | Save/load/rename/list shares (`data/shares/`, 1GB cap, LRU) |

---

## Austria Processor — Mental Model

**Read this section first when working on the processor. It will save you hours.**

The Austria Processor (`austria_processor.py`, 5131 lines, 80 functions) processes
all ~8440 Austrian Katastralgemeinden (KGs) into landscape analysis products and
uploads them to Zenodo. It runs for weeks as a background systemd service.

### The Big Picture (5-second version)

```
main()                           ← parent process, iterates KGs
  └─ for each KG:
       └─ multiprocessing.Pool(1)  ← child process (memory isolation)
            └─ process_one_kg()
                 ├── fetch cadastre → compute bbox
                 ├── tile grid (1.5km, 100m overlap)
                 ├── for each tile:
                 │     ├── read LiDAR (DTM/DSM, 3 dates)
                 │     ├── read ortho (RGBI)
                 │     ├── fetch Copernicus (NDVI/SAR/WorldCover)
                 │     ├── fetch Hansen (forest change)
                 │     ├── segment (Felzenszwalb + RF classify)
                 │     ├── vectorise new buildings + infrastructure
                 │     └── checkpoint tile to disk
                 ├── merge tile results (dedup at boundaries)
                 ├── build full GPKG (all raster layers + vectors)
                 ├── build light GPKG (segments + enriched parcels/buildings)
                 ├── build JSON summary (per-parcel stats)
                 ├── validate outputs
                 └── upload to Zenodo
```

### Architecture Decisions (why it's built this way)

1. **Subprocess per KG**: Each KG runs in `multiprocessing.Pool(1)` so memory
   is fully reclaimed between KGs. The parent only does orchestration + Zenodo upload.

2. **Tiling**: KGs vary from <1km to 27km across. Everything is processed as
   overlapping 1.5km tiles to cap memory at ~90MB/tile. Tiles overlap by 100m
   so edge objects aren't truncated. Dedup uses centroid-ownership (core zone =
   tile shrunk by 50m on overlap sides).

3. **Tile checkpoints**: Each completed tile is pickled to
   `data/austria_processor/tile_checkpoints/<kg>/tile_N.pkl`.
   On crash/restart, completed tiles are restored — only the interrupted tile
   is re-processed. Checkpoints are deleted after successful KG completion.

4. **Parent-child communication**: The subprocess writes step progress to
   `data/austria_processor/current_step.json` (atomic temp+rename). A parent
   thread (`_monitor_step_file`) polls this every 2s and feeds it into
   `ProgressTracker` → `progress.json` → dashboard API → `process.html`.

5. **Retry**: On timeout (90min), the parent retries once with the same
   tile grid — checkpoints restore completed tiles, only the interrupted
   tile is re-processed. Copernicus handles its own failures internally
   via `_quadrant_split()` (2×2 then 4×4). Tile grid is **never** subdivided.

6. **Grid-snapped caches**: Remote data (Copernicus, Hansen) is cached in a
   regular grid so adjacent KGs share cached tiles. See `tile_cache.py`.

### The Two Processes

| | Parent (`main()`) | Child (`process_one_kg()`) |
|---|---|---|
| **Runs in** | `austria_processor.service` | `multiprocessing.Pool(1)` |
| **PID** | Long-lived | New per KG |
| **Responsible for** | KG iteration, retry logic, Zenodo upload, progress tracking, signal handling | All data I/O, segmentation, GPKG/JSON building |
| **Communicates via** | `current_step.json`, `subprocess_warnings.jsonl`, return dict | Same files (writes them) |
| **Memory** | ~100MB | Up to 3GB (MemoryMax enforced by systemd) |
| **Timeout** | Enforces 30/90min via `async_result.get(timeout=)` | No awareness of timeout |

### Code Map of austria_processor.py

All sections are marked with `# === SECTION: ... ===` comments. Use `grep -n '# ===' austria_processor.py` to orient yourself.

| Marker | Key contents |
|--------|---------------|
| `Config` | `DATA_DIR`, `ZENODO_TOKEN`, `KG_TIMEOUT_SECONDS`, tile cache init |
| `Disk cache management` | `check_disk_space()`, `_lru_delete()` |
| `Logging` | File + stderr handlers |
| `ProgressTracker` | JSON-backed state class: `set_step()`, `add_log()`, `record_success()`, `update_rates()` |
| `Circuit breaker` | `_read_circuit_breaker()` — openEO rate-limit protection |
| `Geometry helpers` | `transform_to_3035/wgs()` |
| `KG list` | `get_all_kgs()` — fetch + cache all ~8440 KGs |
| `Cadastre data fetching` | `fetch_cadastre_data()` — REST calls to cadastre API |
| `Height enrichment` | `enrich_parcels_with_heights()`, `enrich_buildings_with_heights()` |
| `Vectorise unmatched segments` | `vectorise_unmatched_buildings()`, `vectorise_infrastructure()` |
| `Resolve edge-clipped features` | `resolve_edge_clipped_features()` — tile boundary fixup |
| `GPKG style + vector writers` | `_write_segment_vectors()`, `_write_segment_points()`, `_write_gpkg_all_styles()` |
| `Tiled GPKG + JSON builders` | **`build_full_gpkg_tiled()`**, **`build_light_gpkg_tiled()`**, **`build_json_summary_tiled()`** |
| `Data quality scoring` | `compute_data_quality()` |
| `process_one_kg()` | **The main per-KG pipeline** (runs in subprocess, see flow above) |
| `Output validation` | `validate_kg_outputs()` |
| `Zenodo upload helpers` | `upload_kg_to_zenodo()` |
| `JSON dir cleanup` | 4GB cap, LRU eviction |
| `main()` | **KG iteration, retry ladder, subprocess management** |

### Key Modules Called by the Processor

| Module | Size | What it does for the processor |
|--------|------|-------------------------------|
| `tile_cache.py` | ~950L | Grid-snapped 0.1° caches for Copernicus + Hansen. `CopernicusTileCache`, `HansenTileCache`, `order_kgs_nearest_neighbor()`. Zenodo fallback on miss. Re-raises `IPThrottledError`/`CreditsExhaustedError` (never swallows them). |
| `zenodo_cache.py` | ~1530L | Persistent cache on Zenodo. Uploads tiles as ZIP archives (one per product × lat strip). Downloads via HTTP range reads (2-3 requests per tile). Separate `cache_manifest.json`. |
| `object_segmentation.py` | 2218L | `segment_objects_in_area()` — Felzenszwalb + RAG + RF classify. Called once per tile. |
| `copernicus.py` | ~1300L | openEO client: NDVI, WorldCover, SAR, harmonics. 4-credential rotation, sync→batch fallback, `IPThrottledError`/`CreditsExhaustedError` propagation. See **Copernicus Throttle & Retry** section. |
| `ortho_io.py` | 992L | BEV orthophoto reader (RGBI, 47 Operates, DOP fallback). |
| `raster_io.py` | 359L | Windowed reads from BEV GeoTIFFs via `/vsicurl/`. |
| `hansen.py` | 453L | Hansen Global Forest Change data reader. |
| `cadastre.py` | 459L | Building footprints + parcel boundaries from cadastre API. |
| `terrain_analysis.py` | 157L | Slope, aspect, TRI, curvature from DTM. |
| `learned_classifier.py` | 559L | RF model loading + 44-feature classification. |
| `zenodo_client.py` | 841L | Zenodo deposit creation, file upload, publish. `Manifest` class for tracking. |
| `bev_retry.py` | 252L | Exponential backoff + proxy rotation for `rasterio.open()`. Proxy pool is free HTTPS proxies from GitHub lists — works for BEV GeoTIFFs, **not useful for Copernicus** (openEO auth is per-credential, not per-IP). |

### Data Flow Through a Single Tile

```
BEV servers ──→ raster_io / bev_retry ──→ DTM, DSM (1m)
                                           ↓
BEV servers ──→ ortho_io ──→ RGBI (0.2m)  │
                               ↓           ↓
openEO ──→ tile_cache ──→ NDVI/SAR/LC    terrain_analysis
                            ↓                ↓
UMD ──→ tile_cache ──→ Hansen           object_segmentation
                            ↓           ↙ ↓ ↘
cadastre API ──→ cadastre ──→ ground truth  features  labels
                                              ↓
                                    learned_classifier (RF)
                                              ↓
                                    classified objects
                                    ↙          ↓          ↘
                        vectorise_*     tile_seg_result    terrain_stats
```

### Persistence & Recovery

| File | Written by | Read by | Purpose |
|------|-----------|---------|--------|
| `progress.json` | Parent (ProgressTracker) | Dashboard API | Live state: current KG, step, rates, log |
| `current_step.json` | Child (`_report_step()`) | Parent (`_monitor_step_file` thread) | IPC: step name + detail + tile index |
| `subprocess_warnings.jsonl` | Child (`_WarningRelayHandler`) | Parent (`_monitor_step_file`) | WARNING/ERROR log relay |
| `in_progress_kg.txt` | Parent | Parent (on restart) | Crash recovery: re-process interrupted KG |
| `tile_checkpoints/<kg>/tile_N.pkl` | Child | Child (on retry) | Resume from last completed tile |
| `zenodo_manifest.json` | Parent (`Manifest`) | Parent + Dashboard | Upload tracking (success/error per KG) |
| `failed_kgs.json` | Parent | Parent (on restart) | Permanently-failed KGs to skip |
| `retry_queue.json` | API / transient handler | Parent (each iteration) | KG codes to insert next in queue (read + cleared) |
| `peer_urls.txt` | deploy.sh / director | app.py (`_sync_peer_data`), director | Peer URLs for data sync + director bandwidth polling |
| `peers.json` | Director API / manual | `peer_director.py` | Peer config: IDs, URLs, enabled, not_before, budget |
| `director_state.json` | Director loop | Director loop | Runtime: active peer, bandwidth per peer, mode |
| `director.lock` | Director loop | Director loop | fcntl lock — single director across gunicorn workers |
| `is_director` | Manual (primary only) | `app.py` (startup) | Flag: director loop only runs if this file exists |
| `kg_list.json` | Parent | Parent | Cached list of ~8440 KGs (avoid repeated API calls) |

### Operations

```bash
# --- Logs ---
tail -f data/austria_processor/logs/processor.log
grep -i "warning\|error\|failed" data/austria_processor/logs/processor.log | tail -20

# --- Status ---
curl -s http://localhost:8000/api/v1/processing/status | python3 -m json.tool

# --- Restart ---
sudo systemctl stop austria_processor && sudo systemctl start austria_processor

# --- Dashboard ---
# https://srtm-lidar-at.exe.xyz:8000/process.html
```

### Restarting at a Different KG

To stop the processor and restart it beginning with a specific KG (e.g. to
prioritise a particular area):

```bash
# 1. Stop the processor
sudo systemctl stop austria_processor

# 2. Clean up stale in-progress marker (prevents re-processing the interrupted KG)
rm -f data/austria_processor/in_progress_kg.txt

# 3. Edit the retry queue — items here are processed before the normal NN order
#    Put desired KG code(s) at the front of the list.
python3 -c "
import json
q = json.load(open('data/austria_processor/retry_queue.json'))
# Remove target KG if already in queue, then prepend
for code in ['TARGET_CODE']:
    if code in q: q.remove(code)
q = ['TARGET_CODE'] + q
json.dump(q, open('data/austria_processor/retry_queue.json', 'w'))
print('Queue now:', len(q), 'items, starts with', q[:3])
"

# 4. (Optional) Remove tile checkpoints if the KG was mid-processing
#    This forces a clean restart for that KG. Skip if the KG wasn't in progress.
rm -rf data/austria_processor/tile_checkpoints/TARGET_CODE/

# 5. Start the processor — it reads retry_queue.json first
sudo systemctl start austria_processor

# 6. Verify
tail -5 data/austria_processor/logs/processor.log
```

Replace `TARGET_CODE` with the KG code (e.g. `91109`). The retry queue is
consumed in order and prepended to the normal nearest-neighbor traversal.
Multiple KGs can be queued — just put them all at the front of the list.

### Re-processing a KG (after bad output or code fix)

```bash
# 1. Stop processor
sudo systemctl stop austria_processor

# 2. Delete Zenodo draft depositions (get depo_ids from manifest)
python3 -c "
import json, requests
TOKEN = '...'  # from austria_processor.py line 48
m = json.load(open('data/austria_processor/zenodo_manifest.json'))
entries = m.get('entries', m)
for k in list(entries.keys()):
    if 'KGCODE' in k:
        requests.delete(f'https://zenodo.org/api/deposit/depositions/{entries[k][\"depo_id\"]}',
                       params={'access_token': TOKEN})
        del entries[k]
json.dump(m, open('data/austria_processor/zenodo_manifest.json', 'w'), indent=2)
"

# 3. Remove local JSON + tile history
rm -f data/austria_processor/json/KGCODE.json
python3 -c "
import json
f = 'data/austria_processor/tile_history.json'
d = json.load(open(f))
d.pop('KGCODE', None)
json.dump(d, open(f, 'w'))
"

# 4. Clear search index entry
python3 -c "
import sqlite3
db = sqlite3.connect('data/search_index.db')
db.execute('UPDATE kg SET processed=0, zenodo_json_url=NULL, zenodo_json_size=NULL, zenodo_light_gpkg_url=NULL, zenodo_light_gpkg_size=NULL, zenodo_full_gpkg_url=NULL, zenodo_full_gpkg_size=NULL, zenodo_depo_id=NULL WHERE kg_code=?', ('KGCODE',))
db.execute('DELETE FROM kg_landcover WHERE kg_code=?', ('KGCODE',))
db.execute('DELETE FROM kg_hansen WHERE kg_code=?', ('KGCODE',))
db.commit()
"

# 5. Remove tile checkpoints if they exist
rm -rf data/austria_processor/tile_checkpoints/KGCODE/

# 6. Restart — KG will appear in pending queue
sudo systemctl start austria_processor
```

Replace `KGCODE` with the KG code (e.g. `91109`). Depositions must be
unpublished (draft) to delete via API. The KG will be re-queued via
nearest-neighbor ordering relative to the last completed KG.

### Common Failure Modes & Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `IPThrottledError` / all 4 creds 402 | Copernicus rate-limiting (transient, all probes pass) | **Automatic**: aborts KG, writes `copernicus_paused`, polls every 15 min, re-queues KG. See **Copernicus Throttle & Retry**. |
| `CreditsExhaustedError` | Genuine credit exhaustion (probe fails) | Update `_CREDENTIALS` in `copernicus.py`, delete `data/austria_processor/copernicus_paused` |
| Timeout on large KG | Copernicus slow (NDVI downloads) | Automatic: retry once with checkpoints, then deferred retry 5 KGs later if transient |
| OOM kill | KG exceeds 3GB MemoryMax | Automatic: systemd restarts, tile checkpoints preserved |
| `Synchronous download timed out` | Normal Copernicus behavior | No action — falls back to batch job automatically |
| `Hansen resample failed` | No Hansen data (western Vorarlberg) | Non-fatal, skipped. Hansen data sparse at AT borders |
| 0-byte cache files | Interrupted before atomic-write fix | Delete the 0-byte `.tif`/`.npz` files manually |
| Disk critically low | Caches filling disk | Automatic: `check_disk_space()` does LRU cleanup, pauses if <3GB |
| Stale `progress.json` | Service died without cleanup | Restart clears stale state in `main()` init block |
| Transient server error (500/503/timeout) | Remote service hiccup | Automatic: deferred retry 5 KGs later (up to 2×), then permanent fail |

### Per-Parcel Compact Layout (`parcels.details[i]`)

Storage at scale (1.6M parcels across 8000 KGs) means every byte counts.
The processor writes per-parcel object highlights as compact arrays + a
tiny FRaction Area Vector (frav). See `parcel_compact.py` for the schema.

```
parcel.frav        = { type_letter: area_sqm_int, ... }   always present when
                                                          any segments hit
parcel.top_objs[i] = [type_letter, hmax, hmean, area, lon, lat,
                       conf, rf_conf, manmade_int]   (5 entries, ~60 B each)
parcel.top_trees[i] = [hmax, hmean, hp90, area, lon, lat, ndvi_m, ndvi_f,
                        hchg, phen, conf, rf_conf]    (5 entries, ~95 B each)
```

Letter mapping (lowercase = natural, uppercase = man-made):
```
t=tree s=shrub g=grass h=hedge w=water R=roof G=greenhouse P=solar_panel
F=fence W=wall M=mast T=wind_turbine X=substation r=road p=path k=parking
b=bridge c=crop o=orchard v=vineyard a=garden B=bare_soil K=rock
E=excavation L=fill l=tree_loss C=construction e=earthwork u=unclassified
```

Legacy verbose forms (`top_10_objects`, `top_10_trees` per parcel) are
still accepted by readers but no longer emitted. quality_flags emits
`parcel_top_obj:<pid>:<i>` and `parcel_top_tree:<pid>:<i>` refs for both
formats. The `frav` is what backfill_parcel_top10.py guarantees on every
parcel — even tiny parcels with no per-segment top_objs.

**Auto-classification at index build**: every parcel JSON written to
`data/austria_processor/json/` is consumed by `search_index.py` on the next
`SearchIndex.build()`/`update()`. The kg_parcels schema includes the new
columns and `parcel_compact.classify_parcel(p)` runs once per parcel.
If you change the classifier, run
`python3 -c "from search_index import SearchIndex; SearchIndex().build()"`
to repopulate (~10s for ~30 KGs locally; ~30s once Austria is fully indexed).

Backfill (after a code change that adds new per-parcel info):
```bash
# Run on a peer (so the bandwidth comes out of the peer's quota)
curl -X POST https://srtm-lidar-at2.exe.xyz:8000/api/v1/admin/run_backfill \
  -H 'Content-Type: application/json' -d '{"force": true}'
curl https://srtm-lidar-at2.exe.xyz:8000/api/v1/admin/backfill_status
```
The script downloads each KG's light GPKG from Zenodo, joins
`segment_points` to parcel polygons, validates the result, and replaces
the JSON in place in the existing draft Zenodo deposition. The local
manifest is updated atomically; primary then re-syncs via the peer-sync
thread (which now compares uploaded_at vs local mtime to pick up
rewrites).

### Per-KG Outputs

1. **Full GPKG** (`{kg}_full.gpkg`): DTM/DSM/nDSM + ortho + segment_type rasters, segment vector polygons
2. **Light GPKG** (`{kg}_light.gpkg`): segment raster+vector, all parcels w/ DTM heights, all buildings w/ object heights, new buildings, infrastructure
3. **JSON summary** (`{kg}.json`): area summary, height distributions, landscape characterisation, top objects/trees, terrain, NDVI, Hansen loss, new buildings, infrastructure, coverage stats, methods

JSON `coverage` section: `n_tiles`, `tile_km`, `parcel_elevation_coverage_pct`, `parcel_segmentation_coverage_pct`, `building_height_coverage_pct`.

### API Endpoints (served by app.py, read processor state)

| Method | Path | Purpose |
|--------|------|----------|
| GET | `/api/v1/processing/status` | Processor progress (polled by process.html) |
| POST | `/api/v1/processing/start` | Start processor (optional: state=, kg=) |
| POST | `/api/v1/processing/pause` | Pause processor (SIGSTOP) |
| POST | `/api/v1/processing/resume` | Resume processor (SIGCONT) |
| POST | `/api/v1/processing/stop` | Stop processor (SIGTERM) |
| POST | `/api/v1/processing/single?kg=X` | Process single KG |
| POST | `/api/v1/processing/retry?kg=X` | Retry failed KG |
| GET\|POST | `/api/v1/processing/throttle` | Get/toggle bandwidth throttle (skip GPKG uploads) |
| GET | `/api/v1/processing/peers` | Peer coordination state (completed, current, priority, failed, manifest) |
| GET | `/api/v1/processing/peers/status` | Combined status across all peers (instances, combined_completed, rates) |
| GET | `/api/v1/processing/log` | Recent processor log lines |
| GET | `/api/v1/processing/manifest` | Zenodo manifest entries |
| GET\|PUT | `/api/v1/processing/cache_manifest` | Zenodo tile-cache manifest (shared across peers) |
| GET | `/api/v1/kg/<kg_code>` | KG JSON summary (local or Zenodo link) |
| GET | `/api/v1/parcel/<parcel_id>` | Parcel lookup via KG JSON |

### Files

| File | Size | Purpose |
|------|------|----------|
| `austria_processor.py` | 5131L | Main processor (this section documents it) |
| `zenodo_client.py` | 841L | Zenodo API client + `Manifest` class |
| `zenodo_cache.py` | ~1530L | Zenodo-backed persistent cache for Copernicus+Hansen tiles. See detailed section above. |
| `tile_cache.py` | 900L | Grid-snapped Copernicus + Hansen caching, Zenodo fallback on miss |
| `austria_processor.service` | — | systemd unit (MemoryMax=4G, Restart=on-failure). Disabled on peers. |
| `peer_director.py` | ~770L | Peer Director — bandwidth-based orchestration across VMs |
| `deploy.sh` | ~160L | One-command deployment for new peer VMs |
| `static/process.html` | ~2100L | Dashboard UI (status, map, log, Zenodo manifest, peer director cards) |
| `data/austria_processor/MONITOR.md` | — | Monitoring checklist + expected timelines |
| `gpkg_streamed.py` | ~500L | Strip-streamed full-GPKG builder for large KGs (auto-used >100 Mpx, skipped >200 Mpx) |
| `kg_splitter.py` | ~300L | KG block splitter — splits large KGs (>28 tiles) into contiguous blocks with directional names |

### KG Block Splitting

Large KGs (>28 tiles ≈ >42 km²) are automatically split into smaller contiguous
blocks for processing. Each block is named with a directional suffix:

```
49006 Innerbreitenau  →  49006-south  (28 tiles)
                         49006-north  (28 tiles)

80110 Sölden          →  80110-southwest-1, 80110-southeast-1, ... (16 blocks)
```

**How it works:**
- `kg_splitter.maybe_split_kg(kg)` checks tile count and splits bbox into grid blocks
- Splitting happens in `main()` when building the pending list — transparent to `process_one_kg`
- Each block processes only parcels whose centroid falls within its bbox
- Blocks are stored on Zenodo as `49006-south.json`, `49006-south_light.gpkg`, etc.
- The search index aggregates all blocks into one parent KG row (all parcels, all stats)
- The KG counter counts unique parent codes, not blocks
- Priority queue: putting `49006` in queue processes all its blocks

**Files affected by splitting:**
- `kg_splitter.py` — splitting logic, block code utilities
- `austria_processor.py` — block expansion in `main()`, parcel filtering in `process_one_kg()`
- `search_index.py` — `_enrich_kg_from_blocks()` aggregates block JSONs into parent row
- `app.py` — queue API resolves block codes via parent KG

### Where to Look When Debugging

| Problem area | Look at |
|---|---|
| KG fails during segmentation | `process_one_kg()` line ~3700, calls `object_segmentation.segment_objects_in_area()` |
| GPKG output wrong | `build_full_gpkg_tiled()` (line 1903) or `build_light_gpkg_tiled()` (line 2177) |
| JSON summary wrong | `build_json_summary_tiled()` (line 2537) |
| Parcel heights wrong | `enrich_parcels_with_heights()` (line 647) |
| Tile boundary artifacts | Centroid-ownership dedup in `process_one_kg()` ~line 3830, `_segment_touches_edge()` |
| Copernicus data missing | `tile_cache.py` → `copernicus.py` → credential rotation + `@_retry_on_rotation` decorator. See **Copernicus Throttle & Retry**. |
| BEV data read failure | `raster_io.py` → `bev_retry.py` (exponential backoff + proxy rotation) |
| Zenodo upload failure | `main()` ~line 4900, calls `zenodo_client.py` |
| Dashboard not updating | `app.py` `/api/v1/processing/status` reads `progress.json`; check parent thread alive |
| Copernicus quadrant fallback | `_quadrant_split()` in `_fetch_copernicus_for_tile()`. Skipped entirely on `IPThrottledError`. |
| Copernicus 402 / throttle | `copernicus.py` `_check_credits_error()` → `@_retry_on_rotation` (all 4 creds) → `IPThrottledError` → tile_cache re-raises → austria_processor writes pause file + aborts KG |


### Copernicus Throttle & Retry — Mental Model

**Read this before touching copernicus.py, tile_cache.py, or the Copernicus path in austria_processor.py.**

#### The Problem

Copernicus openEO returns HTTP 402 PaymentRequired when rate-limited. This is NOT always genuine credit exhaustion — often it's a transient rate-limit where the credential is still healthy (auth works, quota page shows credits remaining). We have 4 credentials. Sometimes only 1-2 are throttled; sometimes all 4 are. Recovery takes minutes to hours.

#### Three-Layer Architecture

```
copernicus.py          tile_cache.py           austria_processor.py
(API layer)            (cache layer)           (orchestrator)
─────────────          ──────────────          ────────────────────
402 detected           re-raises               writes pause file
  │                    IPThrottledError        aborts tile loop
  ├─ probe credential  and                     re-queues KG
  │  passes? → rotate  CreditsExhaustedError   polls every 15 min
  │  + CredentialRotatedError  (never returns     until probe passes
  │                             None for these)
  ├─ probe fails? → mark
  │  exhausted + rotate
  │  + CredentialRotatedError
  │
  └─ all exhausted?
     → CreditsExhaustedError

@_retry_on_rotation decorator
  catches CredentialRotatedError
  → rebuilds connection + datacube with next credential
  → retries the entire function
  → after len(_CREDENTIALS)+1 attempts:
      all probes passed → IPThrottledError (transient)
      some probes failed → CreditsExhaustedError (genuine)
```

#### Exception Types (copernicus.py)

| Exception | Meaning | Caught by |
|-----------|---------|----------|
| `CredentialRotatedError` | One credential got 402, rotated to next | `@_retry_on_rotation` decorator (retries with fresh cred) |
| `CreditsExhaustedError` | ALL credentials genuinely exhausted (probes failed) | tile_cache re-raises → austria_processor writes pause file |
| `IPThrottledError(RuntimeError)` | ALL credentials got transient 402 (probes passed) | tile_cache re-raises → austria_processor writes pause file |

**Critical rule**: `CredentialRotatedError`, `CreditsExhaustedError`, and `IPThrottledError` must NEVER be swallowed by generic `except Exception` blocks. The public functions (`get_ndvi_composite`, `get_land_cover`, `get_sar_backscatter`) have explicit `except (CredentialRotatedError, CreditsExhaustedError, IPThrottledError): raise` before their generic handler.

#### Credential Probing

`_check_credits_error(exc)` is called on every 402. It:
1. Authenticates the credential against the OIDC endpoint (not a data download)
2. If auth succeeds → "transient 402" → rotates + `CredentialRotatedError`
3. If auth fails with 402 → "genuinely exhausted" → marks credential, rotates + `CredentialRotatedError` (or `CreditsExhaustedError` if all gone)

**Why probes always pass during rate-limiting**: the auth endpoint is separate from the processing endpoint. A credential can authenticate fine but still get 402 on downloads. This is why we must try ALL credentials — the rate-limit may be per-credential or per-IP or timing-dependent.

#### Sync → Batch Fallback

Each product download tries sync first (3 min timeout), then batch job:
- Sync: `datacube.download()` — fast for 0.1° cells, often 402'd or times out
- Batch: `datacube.execute_batch()` — slower (5-15 min) but more reliable
- The `@_retry_on_rotation` decorator wraps the entire function, so a 402 on sync in credential 1 → retry the whole function with credential 2 (new sync attempt, then batch fallback)

#### Per-Product Retry Flow

**NDVI/WorldCover/SAR** (`get_ndvi_composite`, `get_land_cover`, `get_sar_backscatter`):
```
@_retry_on_rotation (up to 5 attempts with 4 creds)
  └─ build datacube with current credential
     └─ _run_datacube()
        ├─ sync download (1 attempt)
        │   402 → _check_credits_error → CredentialRotatedError → decorator retries
        │   timeout → fall through to batch
        └─ batch job
            402 → _check_credits_error → CredentialRotatedError → decorator retries
```

**NDVI Time Series** (`get_ndvi_timeseries` → `_download_month_sequential`):
- Downloads 8 months (Mar-Oct) sequentially
- Each month has its own retry loop with credential tracking (`tried_creds` set)
- After all 4 credentials fail for one month → returns `IPThrottledError` for that month
- Download loop: if `IPThrottledError` or `CreditsExhaustedError` returned → breaks immediately (cascade breaker)
- Also has per-month cooldown (`_FAILED_MONTH_COOLDOWNS`) — skips months that failed recently

#### tile_cache.py Bridge

All 4 `_fetch_*_cell` methods follow the same pattern:
```python
try:
    result = copernicus.get_*(cell_bbox, ...)
except (CreditsExhaustedError, IPThrottledError):
    raise   # NEVER swallowed — propagates to austria_processor
except server_error:
    retry with backoff
except other:
    return None  # soft failure for non-throttle errors
```

Additional safety: if `last_exc` contains "IP-throttled", raises `IPThrottledError` even from the `return None` path.

#### austria_processor.py Response

**`_try_fetch_single(bbox)`**: Early-bails if `copernicus.ip_throttled` flag is set.

**`_fetch_copernicus_for_tile()`**: On `IPThrottledError`/`CreditsExhaustedError`, re-raises immediately — no quadrant fallback. Quadrant fallback only triggers on timeouts/server errors.

**Tile loop** (inside `process_one_kg`):
```
try:
    copernicus_data = _fetch_copernicus_for_tile(...)
except (CreditsExhaustedError, IPThrottledError):
    result["copernicus_exhausted"] = True
    result["success"] = False
    COPERNICUS_PAUSE_FILE.write_text(...)   # data/austria_processor/copernicus_paused
    break   # ABORT tile loop
```

After tile loop, if `copernicus_exhausted + success=False`: return early (skip GPKG/JSON build).

**Parent process** (`main()`):
1. Subprocess returns with `copernicus_exhausted=True` → `is_credits_issue` check
2. KG added to `retry_queue.json` (tile checkpoints preserved)
3. Enters pause loop: sleeps 15 min → `_copernicus_probe()` → if OK, deletes pause file + resumes
4. `_copernicus_probe()` resets both `ip_throttled` and `credits_exhausted` flags, clears all cached connections, then tries a tiny NDVI download
5. On resume, the re-queued KG is processed next (tile checkpoints restore completed tiles)

#### Proxies — NOT Useful for Copernicus

`bev_proxy.py` manages a pool of free HTTPS proxies from GitHub lists. These are useful for BEV GeoTIFF range reads but **do not help with Copernicus 402s** because:
- openEO authentication is per-credential (OAuth client_credentials), not per-IP
- Rate-limiting is tied to the credential's account, not the source IP
- Free proxies are unreliable and slow for the data volumes openEO returns

Historical note: proxy rotation for Copernicus was tried and removed. The solution is credential rotation (try all 4), not IP rotation.

#### 4 Credentials (copernicus.py line ~48)

| Index | Client ID prefix | Notes |
|-------|-----------------|-------|
| 1 | `sh-f36653c6` | Fresh 2026-04 |
| 2 | `sh-8d8c685f` | Renews 2026-05-01 |
| 3 | `sh-2ed25dbb` | Renews 2026-05-01 |
| 4 | `sh-07af1740` | 30k credits |

All share the same CDSE quota pools (openEO, Sentinel Hub, COG, S3). Currently only openEO is used. Each account has 10k openEO credits/month.

#### Key Files & Flags

| File/Flag | Location | Purpose |
|-----------|----------|--------|
| `copernicus_paused` | `data/austria_processor/` | Pause file — parent polls every 15 min when present |
| `openeo_circuit.json` | `data/austria_processor/` | Circuit breaker — backs off on consecutive failures |
| `copernicus.ip_throttled` | Module global (per-process) | Fast-bail flag — set by decorator after all creds fail |
| `copernicus._exhausted_cred_indices` | Module global (per-process) | Set of credential indices confirmed genuinely exhausted |
| `copernicus._IP_THROTTLE_COOLDOWN` | 7200 (2 hours) | How long `ip_throttled` stays True before auto-reset |
| `_FAILED_MONTH_COOLDOWNS` | Module global dict | Per-(bbox,month) cooldown timestamps for NDVI TS |

**Process architecture note**: Module globals (`ip_throttled`, `_exhausted_cred_indices`, etc.) live in the subprocess (one per KG). They reset when a new KG starts in a fresh subprocess. The pause file is the cross-process communication mechanism.

#### Operational Commands

```bash
# Check if paused
cat data/austria_processor/copernicus_paused

# Force resume (probe will re-validate on next KG)
rm -f data/austria_processor/copernicus_paused

# Reset all throttle state
rm -f data/austria_processor/copernicus_paused data/austria_processor/openeo_circuit.json
sudo systemctl restart austria_processor

# Check which credential is active in the subprocess
grep 'Authenticated successfully\|Rotated to credential\|IP-throttled\|transient 402' \
  data/austria_processor/logs/processor.log | tail -20
```

---

### Peer Director — Multi-Instance Orchestration

**Read this section before touching peer_director.py, deploy.sh, or any director API endpoint.**

#### Architecture: One Frontier + Many Cache-Only Peers

The system uses a **single director** (the primary instance, `srtm-lidar-at`) to
orchestrate processing across multiple exe.dev VMs. Each VM has 100 GB/month
bandwidth.

**Two roles**:
- **Frontier** (one at a time): runs full processing including Copernicus + Hansen
  fetches.  Touches the shared Copernicus credentials, so only one frontier
  may run.  All credential rotation happens here.
- **Cache-only** (many in parallel): processor started with `--cache-only` /
  `COPERNICUS_FORBIDDEN=1`.  Refuses any Copernicus/Hansen API call — if a
  tile isn't in the local + Zenodo cache, it raises `CacheMissError` and
  the KG is re-queued for the frontier.  Peer is fed an explicit whitelist
  of fully-cached KGs computed by the director.

**Key invariants**:
- exactly one frontier peer running at any time (credential safety)
- up to `max_cache_only_peers` (default 8) cache-only peers running in parallel
- at least `min_reserve_peers` (default 0 — was 5; with cheap unlimited
  peers we no longer keep a reserve) enabled peers stay idle
- all Zenodo writes (KG uploads + tile-cache flushes) serialise through a
  single mutex broker on the primary (`/api/v1/zenodo/lock`)

```
┌─────────────────────────────────────────────────────┐
│  PRIMARY (srtm-lidar-at)                            │
│  ────────────────────────────────────                │
│  • Runs the Peer Director loop (peer_director.py)    │
│  • Has data/austria_processor/is_director flag        │
│  • Monitors bandwidth via vnstat on all peers         │
│  • Starts/stops processors on peers via REST API      │
│  • Syncs priority queue to the active peer             │
│  • Switches active peer when bandwidth < 2 GB         │
│  • Hosts the search index + combined dashboard         │
├─────────────────────────────────────────────────────┤
│  PEER at2 (srtm-lidar-at2)                          │
│  PEER at3 (srtm-lidar-at3)                          │
│  ────────────────────────────────────                │
│  • NO director loop (no is_director flag)             │
│  • NO systemd autostart for austria_processor         │
│  • Processor started/stopped ONLY by the director     │
│  • Shares Zenodo tile-cache deposit with primary        │
│  • Same codebase, same Copernicus credentials          │
└─────────────────────────────────────────────────────┘
```

**Critical invariant**: Only the primary runs the director loop. Peers are
passive workers. If a peer were to run its own director loop, it would
start/stop processors in conflict with the primary. This is enforced by the
`data/austria_processor/is_director` flag file (only exists on the primary).

#### Cache-only peers (parallel processing)

A peer started with `--cache-only` activates `tile_cache.FORBID_REMOTE`.
All `_fetch_*_cell` methods on `CopernicusTileCache` and `HansenTileCache.get_raw`
raise `CacheMissError` instead of calling the API.  The processor's tile loop
catches it, sets `result['cache_incomplete']=True`, and aborts the KG.  The
parent (`main()`) then re-queues the KG via `_append_retry_queue()` without
marking it failed — the frontier peer will pick it up later.

The director computes the cache-ready whitelist via `_compute_cache_ready_kgs()`:
1. Read `cache_manifest.json` and intersect lat-strip availability across
   `ndvi`, `sar`, `harmonics`, `worldcover`, `hansen` (cheap dict lookup).
2. For each candidate KG (bbox falls inside a covered strip), call
   `tile_cache.is_kg_fully_cached(bbox)` — walks per-cell index, no downloads.
3. Cache result for 5 minutes; the cache extends as the frontier fetches more.

When starting a cache-only peer, the director PUTs a slice of the whitelist
as the peer's priority queue and starts the processor with `cache_only=True`.
Different cache-only peers get different slices to reduce overlap.

#### Zenodo upload mutex

All Zenodo writes serialise through `/api/v1/zenodo/lock` on the primary:
- `upload_kg_to_zenodo()` wraps the entire upload in `zenodo_upload_lock()`
- `flush_tile_cache_to_zenodo()` wraps `ZenodoCache.upload_all()` similarly
- Lease has TTL 120s; a daemon thread renews via `/heartbeat` every 30s
- Stale leases (no heartbeat for >TTL) are auto-released
- Peers point at the broker via env `ZENODO_LOCK_URL`, set from
  `data/austria_processor/zenodo_lock_url.txt` (`deploy.sh` writes the
  primary's URL there)
- Primary uses `http://127.0.0.1:8000` automatically (set by app.py when
  spawning the processor on a host with the `is_director` flag)
- If the broker is unreachable, peers fail open (proceed without lease)
  to avoid deadlocking the fleet on a network blip

#### Per-peer config (`peers.json`)

New/relevant fields:
- `role: "frontier" | "cache_only"` — hint to the director.  If absent,
  treated as frontier (the director may still borrow it for cache-only work
  when the frontier is elsewhere).
- `min_reserve_peers` (top-level) — default 5.  Director never starts a
  cache-only peer if it would push the idle count below this.
- `max_cache_only_peers` (top-level) — default 8.  Cap on concurrent
  cache-only peers.

#### How the Director Works

`peer_director.py` (`PeerDirector` class) runs a background thread every 30s:

1. **Re-reads `peers.json`** from disk (handles cross-worker/cross-process updates)
2. **Polls bandwidth** on all peers via `GET /api/v1/bandwidth` (vnstat)
3. **Checks active peer**:
   - Budget exhausted (< 2 GB)? → Stop it, pick next peer with most bandwidth
   - Scheduled (`not_before` in future)? → Skip it
   - Stopped unexpectedly? → Restart it (if bandwidth remains)
   - Unreachable? → Deactivate, pick another
4. **Enforces single-active**: Stops any non-active peer found running
5. **Syncs priority queue** to the active peer every ~2.5 min
6. **Saves state** to `director_state.json`

**File lock**: Only one gunicorn worker runs the director loop (fcntl file lock
on `data/austria_processor/director.lock`). The other worker skips it.

#### Bandwidth Management

- Each exe.dev VM has 100 GB/month (resets on the 17th)
- Budget set to 95 GB (5 GB headroom) in `peers.json`
- When active peer drops below 2 GB remaining → director switches to the peer
  with the most remaining bandwidth
- When ALL peers are exhausted → director logs "no peers available" and waits
- After bandwidth reset (17th) → vnstat reports drop, peers become eligible again

#### Admin Token (cluster auth)

All mutating admin/director/processing/zenodo endpoints require
`X-Admin-Token: <token>`. Loopback (127.0.0.1, no XFF) is exempt so
the in-process director and on-box CLI work without plumbing it.

- Token lives in `data/admin_token` (gitignored, mode 0600). Auto-generated
  on first start of the primary.
- Peers must have the **same** token. `deploy.sh` accepts `ADMIN_TOKEN=...`
  to install it. Without it, peer registration and director-driven
  start/stop will 401.
- Dashboard prompts for the token on first 401 and stores it in
  `localStorage` (key `srtm_admin_token`). Reset with
  `srtmResetAdminToken()` in the JS console.
- The director re-reads `data/admin_token` on every outbound peer call
  (`peer_director._admin_headers()`), so token rotations propagate
  without a director restart.
- Rotate by writing a new value to `data/admin_token` on every peer +
  primary (e.g. via `for p in peers; do scp ...; done`); no service
  restart required.

```bash
# Get the token (on primary)
cat data/admin_token

# Use it from CLI
curl -H "X-Admin-Token: $(cat data/admin_token)" \
  -X POST https://srtm-lidar-at3.exe.xyz:8000/api/v1/admin/update
```

#### Deploying a New Peer

```bash
# On the new exe.dev VM:
#   ADMIN_TOKEN required so the peer can register with the director.
#   Get it via `cat data/admin_token` on the primary.
SELF_URL=https://srtm-lidar-at4.exe.xyz:8000 \
PEER_URL=https://srtm-lidar-at.exe.xyz:8000 \
ADMIN_TOKEN=<paste from primary> \
bash deploy.sh
```

`deploy.sh` does:
1. Clones repo, installs deps, decompresses RF model
2. Installs `srv.service` (gunicorn) and `austria_processor.service`
3. Starts the web server (`srv`) but does **NOT** enable/start the processor
4. Throttle mode is OFF by default (every peer uploads full + light GPKGs)
5. Auto-registers with the director via `POST /api/v1/director/peers/add`
6. The director will start the processor when it’s this peer’s turn

**What the peer does NOT have**:
- No `data/austria_processor/is_director` flag → no director loop
- `austria_processor.service` is disabled → no systemd auto-restart
- No `peers.json` with remote peers → default config only

**After deploy, make the VM public** (from the exe.dev shell):
```bash
share set-public srtm-lidar-at4
```

#### Updating All Peers

Click **⬆ Update Peers** on the dashboard, or:
```bash
curl -X POST http://localhost:8000/api/v1/director/update_peers
```

This calls `POST /api/v1/admin/update` on each peer, which does `git pull --ff-only`
then `sudo systemctl restart srv`. The timeout on the restart is expected (the
process serving the request dies). Peers come back up in ~10s.

**Important**: The update restarts `srv` (gunicorn) but does NOT restart the
processor. If the processor is running on the active peer, it keeps running —
only the web server restarts. Code changes to `austria_processor.py` take
effect when the processor finishes the current KG and is restarted by the
director.

To force a processor restart on the active peer:
```bash
curl -X POST https://<peer>/api/v1/processing/stop
# Director will restart it automatically on next tick (~30s)
```

#### Removing a Peer

Click the ✕ button on the peer card in the dashboard, or:
```bash
curl -X DELETE http://localhost:8000/api/v1/director/peers/<peer_id>
```

This stops the peer’s processor, removes it from `peers.json` and
`peer_urls.txt`, and clears its bandwidth state. The peer VM continues
running its web server but won’t receive any work.

#### Throttle Propagation

The 🔋 Throttle button in the dashboard toggles locally AND propagates
to all remote peers via `POST /api/v1/director/throttle`. This ensures
consistent throttle state across all instances.

```bash
# Set throttle on all peers
curl -X POST http://localhost:8000/api/v1/director/throttle \
  -H 'Content-Type: application/json' -d '{"throttle": true}'
```

#### Zenodo Cache on Peers (Shared Deposit)

All peers share the same Zenodo tile-cache deposit (depo 19650075) via
`cache_manifest.json` sync. The manifest is:
- **Pushed to the active peer** when the director starts its processor
  (via `PUT /api/v1/processing/cache_manifest`)
- **Synced bidirectionally** every 5 minutes by the peer-sync thread
  in `app.py` (`_sync_peer_data`)

This means a peer processing KGs near tiles the primary already cached
will fetch them from Zenodo (HTTP range reads, ~2-3 requests per tile)
instead of re-downloading from Copernicus. Saves openEO credits.

Concurrency safety: the director enforces single-active processing, so
only one peer writes to the deposit at a time. Tile cache uploads
(`flush_tile_cache_to_zenodo`) proceed regardless of the upload throttle
(throttle only blocks big GPKG uploads).

**If a peer has a stale manifest**: `rm -rf data/austria_processor/zenodo_zip_index/`
to force re-fetch of ZIP central directories. The next sync cycle will
push the latest manifest.

#### Director API Endpoints

| Method | Path | Purpose |
|--------|------|----------|
| GET | `/api/v1/bandwidth` | Local vnstat bandwidth for this instance |
| GET | `/api/v1/director/status` | Full director state: mode, active peer, bandwidth per peer |
| POST | `/api/v1/director/mode` | Set mode: `auto`, `manual`, `paused` |
| POST | `/api/v1/director/activate` | Manually activate a specific peer |
| POST | `/api/v1/director/stop` | Stop all peers and pause the director |
| GET\|POST | `/api/v1/director/peers` | Get/update full peers config |
| POST | `/api/v1/director/peers/add` | Add a new peer dynamically |
| DELETE | `/api/v1/director/peers/<id>` | Remove a peer (stops its processor) |
| GET\|POST | `/api/v1/director/throttle` | Get/propagate throttle state to all peers |
| GET | `/api/v1/director/proxy/status` | Proxy active peer’s processing status |
| GET | `/api/v1/director/proxy/log` | Proxy active peer’s processor log |
| POST | `/api/v1/director/update_peers` | Git pull + restart srv on all remote peers |
| POST | `/api/v1/admin/update` | Git pull + restart srv (called BY director) |
| POST | `/api/v1/admin/restart_processor` | Restart processor via systemd (fallback) |
| POST | `/api/v1/admin/disable_autostart` | Disable austria_processor systemd unit |

#### Director Files (all instance-specific, NOT in git)

| File | Purpose |
|------|----------|
| `data/austria_processor/is_director` | Flag file — director loop only runs if this exists |
| `data/austria_processor/peers.json` | Peer config: IDs, URLs, enabled, not_before |
| `data/austria_processor/director_state.json` | Runtime state: active peer, bandwidth, mode |
| `data/austria_processor/director.lock` | fcntl lock — ensures single director loop across workers |
| `data/austria_processor/peer_urls.txt` | Peer URLs for the data sync thread |

#### Server-Friendliness Throttle

When BEV (`data.bev.gv.at`), Zenodo, or Copernicus servers start emitting
warnings (HTTP 0 range-read drops, 429 / 503, openEO 402s) the director
automatically reduces the number of concurrent peers so we don't hammer
them. The mechanism:

1. `ProgressTracker.add_log()` (in `austria_processor.py`) classifies every
   warning/error into `bev` / `zenodo` / `copernicus` based on substring
   tokens, and keeps a 10-min sliding window. Per-minute rates are
   exposed in `progress.json → warning_rates` and propagate to the
   director via `/api/v1/processing/status`.
2. Each director tick (~30 s) `_capacity_factor()` takes the **max**
   per-kind 5-minute rate across all peers and maps it linearly to a
   sub-factor: 1.0 at zero warnings, `THROTTLE_MIN_FACTOR` (0.30) at
   `THROTTLE_SATURATION_RATE` warnings/min. The minimum sub-factor wins.
3. An EMA (`THROTTLE_EMA_ALPHA = 0.25`, half-life ~3 ticks) smooths the
   raw value, then a slow sinusoidal drift (±10 % over a 2-hour period,
   phase derived from the hostname) overlays an organic wobble.
4. `_orchestrate_parallel_frontiers()` and `_orchestrate_cache_only()`
   multiply their caps (`max_parallel_frontiers` and
   `max_cache_only_peers`) by the factor each tick. Frontiers always
   keep at least one slot — the active frontier is never pre-empted by
   the throttle. Cache-only count *can* drop to zero on sustained
   pressure.
5. Status payload exposes `capacity_factor` and `capacity_components`;
   dashboard shows a 🌿 pill (green/yellow/red) with the rates as
   tooltip.

Saturation thresholds (warnings per minute) live in
`peer_director.py → THROTTLE_SATURATION_RATE` (re-tuned 2026-04-23 for
the ~50-peer fleet — the previous values were too tight at this scale
and were causing the throttle to bite on ambient noise):
- bev: 5.0  (a few range-read retries are normal noise)
- zenodo: 1.5  (Zenodo rate-limits aggressively)
- copernicus: 0.4  (402s should be near zero in steady state)

A dead-zone (`THROTTLE_DEAD_ZONE_FRAC = 0.10`, i.e. 10 % of saturation)
is applied before the linear ramp — fleet-max rates below the dead-zone
read as zero so a single chatty peer can't drag capacity off 100 %.
Drift amplitude reduced from ±10 % to ±4 % (cosmetic-only, was loud
enough to flip dashboard colors on its own).

The fleet aggregator also filters out peers whose state is `stopped`
(clean exit — their warning window is stale immediately) and peers
that have been `unreachable` for more than 30 minutes. A peer that
just became unreachable is still trusted: it may be doing a long
GPKG upload and briefly not responding to `/processing/status`.
Last-seen-live timestamps live in `director_state.json` under
`peer_last_live_ts`.

Backoff timings are also more tender:
- `BANDWIDTH_BACKOFF_SECONDS = 900` (15 min after 3 failed bandwidth polls)
- `ZENODO_NETWORK_COOLDOWN_MIN = 60` (was 30)
- `HOLD_TENDENCY_WINDOW_HOURS = 6` (was 3) — longer memory for repeat offenders
- `HOLD_TENDENCY_MAX_MIN = 24 * 60` (was 12 h) — a bad peer can sit out a full day
- `THROTTLE_EMA_ALPHA = 0.15` (was 0.25) — slower recovery, kinder to upstreams
- `THROTTLE_MIN_FACTOR = 0.20` (was 0.30) — deeper cuts under heavy pressure

Tuning knobs: `THROTTLE_MIN_FACTOR`, `THROTTLE_EMA_ALPHA`,
`THROTTLE_DRIFT_PERIOD_S`, `THROTTLE_DRIFT_AMPLITUDE` at the top of
`peer_director.py`.

#### Director Modes

| Mode | Behaviour |
|------|----------|
| `auto` | Director picks peer with most bandwidth, starts/stops automatically |
| `manual` | Director keeps the manually-activated peer running, no auto-switch |
| `paused` | Director does nothing — all peers stay in current state |

#### Director High-Availability (`director_ha.py`)

Failover is automatic. Every VM (primary + peers) runs a watchdog
thread that pings the director's `GET /api/v1/director/heartbeat`
every 30 s. The director elects a *shadow* each tick — the most-reliable
peer that is **enabled, reachable, on the same git commit, has ≥ 5 GB
free disk and ≥ 10 GB remaining bandwidth**. Sticky: keeps the current
shadow unless the noise-score gap to the best alternative exceeds 0.3.
The director PUTs a full state snapshot to the shadow every 30 s.

**Snapshot contents** (small JSON, ~200 KB total): `director_state.json`,
`kg_strikes.json`, `failure_counts.json`, `cache_miss_kgs.json`,
`deferred_kgs.json`, `retry_queue.json`, `failed_kgs.json`,
`manifest_tombstones.json`, `copernicus_credentials.json`,
`peers.json`, `cache_manifest.json`, `peer_urls.txt`. Staged under
`data/austria_processor/shadow/`.

**Auto-failover**: shadow misses 3 consecutive heartbeats (90 s) →
promotes itself: installs staged snapshot, writes `is_director`,
restarts director loop in-process (singleton replaced so EMA /
capacity_history reload), broadcasts `POST /api/v1/director/announce`
to every peer. Peers flip `data/austria_processor/zenodo_lock_url.txt`
and `self.json:director_url`. Old director, if it ever comes back,
finds `stepped_down` flag and refuses to start its director loop —
lives on as a regular peer until manually re-promoted.

**Manual handover**: dashboard `⇋ Hand Over` button (next to `+ Add
Peer`) → `POST /api/v1/director/handover?to=<peer_id>` on the current
director. Director ships fresh snapshot inline to target via
`/api/v1/director/takeover`, target promotes itself, broadcasts
announce. Old director steps down proactively. Reload the dashboard
against the new director's URL afterwards.

**Identity** (`data/austria_processor/self.json`): `{id, url,
director_url}`. On the primary, `director_url=null` (it *is* the
director). Peers learn their identity at registration time (deploy.sh
or `+ Add Peer`); the director also broadcasts identity to all peers
at startup, and self-heals one peer per tick.

**HA endpoints** (all admin-token protected except heartbeat):

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/v1/director/heartbeat` | Liveness probe (public). 200 if director, 410 otherwise. |
| GET\|PUT | `/api/v1/director/snapshot` | Director GETs snapshot, shadow accepts staged PUT. |
| POST | `/api/v1/director/announce` | New director claims authority — peers flip pointer / step down. |
| POST | `/api/v1/director/step_down` | Voluntarily relinquish director role. |
| POST | `/api/v1/director/takeover` | Inbound takeover (manual handover or watchdog promotion). |
| POST | `/api/v1/director/handover?to=<id>` | Initiated by current director; ships state + steps down. |
| GET\|POST | `/api/v1/director/identity` | Read/set self.json. |

**Files**: `is_director` (this VM is director), `stepped_down`
(refused promotion, written by step-down), `self.json` (identity),
`zenodo_lock_url.txt` (peer's pointer to current director),
`shadow/` (staged snapshot), `shadow/meta.json` (origin + shadow_id
stamp — watchdog only takes over if `meta.shadow_id == self_id`).

**Disaster recovery (planned)**: just press `⇋ Hand Over` and pick a
peer. Or stop the primary and wait 90 s; the shadow takes over
automatically.


#### Cross-Cutting Concerns (director changes)

**Adding a peer**: Use the dashboard "+ Add Peer" button or API. This updates
`peers.json` on the primary only. The director re-reads it each tick.

**Removing a peer**: Use the ✕ button or DELETE API. Stops the processor,
removes from config and peer_urls.txt.

**Changing bandwidth budget**: Edit `peers.json` directly:
```bash
python3 -c "
import json
cfg = json.load(open('data/austria_processor/peers.json'))
cfg['budget_gb'] = 90  # more conservative
json.dump(cfg, open('data/austria_processor/peers.json', 'w'), indent=2)
"
```
Director picks it up on next tick (re-reads from disk).

**Scheduling a peer** (e.g. don’t use primary until next billing cycle):
```bash
python3 -c "
import json
cfg = json.load(open('data/austria_processor/peers.json'))
for p in cfg['peers']:
    if p['id'] == 'primary':
        p['not_before'] = '2026-05-17'  # skip until bandwidth resets
json.dump(cfg, open('data/austria_processor/peers.json', 'w'), indent=2)
"
```

**Disaster recovery**: If the primary goes down, peers stop receiving work but
don’t crash. To make a peer the new director:
1. Create `data/austria_processor/is_director` on the peer
2. Copy/create `peers.json` with all peer URLs
3. Restart `srv` on the peer

---

### Planned Refactor (next maintenance window)

Detailed prompt in `data/next-prompt.md`. Requires stopping the processor.

**Step 1 — Extract `segment_types.py`** (safe, processor can stay running):
- Move `SEGMENT_COLORS`, `_height_class()`, `_viridis_rgb()` out of `app.py` + `austria_processor.py` into a shared module.
- Fixes: `_height_class()` has already diverged between the two copies.

**Step 2 — Split `austria_processor.py`** (stop processor first):
- `austria_processor.py` (~2000L) — orchestration: `main()`, `process_one_kg()`, retry logic, Zenodo upload
- `kg_builders.py` (~1800L) — `build_full_gpkg_tiled()`, `build_light_gpkg_tiled()`, `build_json_summary_tiled()`, GPKG style/vector writers
- `kg_enrichment.py` (~800L) — `fetch_cadastre_data()`, height enrichment, vectorisation, edge-clip resolution

Gotchas: lazy imports inside `process_one_kg()` (subprocess boundary), pass `DATA_DIR`/`GPKG_DIR` as args to builders.

---

### Copernicus Speed Optimisation (next maintenance window)

Stop processor first: `sudo systemctl stop austria_processor`

**Context:** CDSE gives each account 5 independent quota pools (see
https://documentation.dataspace.copernicus.eu/Quotas.html). We only use openEO.
The others sit at 0% utilisation. Current Copernicus fetch is ~47 min/tile
(sequential, single credential). Target: ~8 min/tile via parallelism + offloading.

| Pool | Per account | 4 accounts | Current use |
|------|------------|------------|-------------|
| openEO credits | 10k/month | 40k | partial (sequential) |
| Sentinel Hub PU | 10k/month | 40k | **0%** |
| Sentinel Hub requests | 10k/month | 40k | **0%** |
| Direct COG HTTP | 50k/month | 200k | **0%** |
| S3 bandwidth | 12 TB/month | 48 TB | **0%** |

Current per-tile breakdown (all sequential on 1 credential):
- NDVI composite: ~10 min (1 openEO job)
- WorldCover: ~3 min (1 openEO job)
- SAR backscatter: ~5 min (1 openEO job)
- NDVI time series: ~29 min (8 openEO jobs, 1/month)

**Do these steps in order. Each step is independently deployable.**

#### Step 3 — WorldCover via direct AWS COG (easy win, ~30 min work)

ESA WorldCover v200 2021 is hosted as public COGs on AWS. No CDSE auth needed.
Already verified: `rasterio.open(url).read(window=...)` works.

Saves: ~3 min/tile + openEO credits. Effort: low.

**What to do:**

1. Create `worldcover_cog.py` (~80 lines). Single function:
   ```python
   def get_land_cover_cog(bbox_wgs84: dict) -> dict:
       """Fetch WorldCover via direct COG HTTP range read from AWS.
       Returns same format as copernicus.get_land_cover():
       {"map": np.ndarray(H,W, uint8), "transform": Affine, "crs": CRS,
        "classes": WORLDCOVER_CLASSES}
       """
   ```
   Tiles are 3°×3° COGs at:
   `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N{lat}E{lon:03d}_Map.tif`

   Austria needs 6 tiles: N45E009, N45E012, N45E015, N48E009, N48E012, N48E015.
   Each is 36000×36000 px, EPSG:4326, uint8.

   Implementation:
   - Compute which tile(s) the bbox falls in (floor lat to multiple of 3, same for lon)
   - `rasterio.open(url)` with `/vsicurl/` env or plain HTTPS (both work)
   - `ds.read(1, window=from_bounds(*bbox, ds.transform))` for windowed read
   - If bbox spans two tiles (rare, only at 12°E or 15°E), read both + mosaic
   - Return `{"map": data, "transform": window_transform, "crs": CRS.from_epsg(4326), "classes": WORLDCOVER_CLASSES}`
   - Cache in `rf_training_data/copernicus_cache/` using same hash scheme as `copernicus.py`
   - Retry with exponential backoff on HTTP errors (use `bev_retry.py` pattern)

2. In `tile_cache.py`, modify `CopernicusTileCache.get_landcover()` (~line 296):
   - Try `worldcover_cog.get_land_cover_cog(tile_bbox)` first
   - Fall back to `copernicus.get_land_cover()` on failure
   - The cache layer (`_atomic_savez`) stays the same — transparent to callers

3. In `copernicus.py`, update `get_land_cover()` similarly as fallback.
   `app.py` calls `copernicus.get_land_cover()` directly for interactive use.

Test: `python3 -c "from worldcover_cog import get_land_cover_cog; r = get_land_cover_cog({'west':15,'south':47.5,'east':15.1,'north':47.6}); print(r['map'].shape, set(r['map'].flatten()[:100]))"`

#### Step 4 — Parallel openEO across 4 credentials (biggest win, ~2 hours work)

openEO allows **2 concurrent processing jobs per account**. We have 4 accounts
= 8 concurrent slots. Currently `_fetch_copernicus_for_tile()` runs NDVI→WC→SAR→harmonics
sequentially through one credential. The NDVI time series alone (8 months) takes
~29 min because each month waits for the previous.

Saves: ~29 min → ~8 min for NDVI TS (3-4× speedup). Effort: medium.

We had parallel downloads before and they "often failed for large areas" — but
that was large-area batch jobs, not the 0.1° tiles we use now. At 0.1° tile size,
parallel sync downloads are safe.

**What to do:**

1. In `copernicus.py`, refactor `get_ndvi_timeseries()` (~line 669):
   - Currently: `for label, m_start, m_end, month_cache in to_download:` (sequential)
   - Change to: `ThreadPoolExecutor(max_workers=8)` submitting `_download_month_sequential()`
   - Each worker gets a dedicated `cred_index` via round-robin: `cred_index = i % len(FUNCTIONING_CREDENTIALS())`
   - Each worker calls `_get_connection_for_cred(cred_index)` — this already exists and creates per-credential sessions
   - Max 2 workers per credential (openEO concurrency limit)
   - On 402/CreditsExhausted from one credential, remove it from the pool, redistribute remaining work
   - Add 5-second stagger between submissions (openEO rate limit: 1 req/5s per account, but we use different accounts)

2. In `_download_month_sequential()` (~line 764), add `cred_index` parameter:
   - `c = _get_connection_for_cred(cred_index)` instead of `c = _get_connection()`
   - Error handling already works per-credential (402 detection, rotation)

3. Also parallelise the non-TS products within `_fetch_copernicus_for_tile()`
   in `austria_processor.py` (~line 4153):
   - `_try_fetch_single()` currently calls NDVI → WorldCover → SAR → harmonics sequentially
   - Refactor: submit NDVI, WorldCover (now COG, instant), and SAR as 3 concurrent futures
   - WorldCover goes to `worldcover_cog` (no credential needed)
   - NDVI composite uses credential A
   - SAR uses credential B
   - Then harmonics (NDVI TS) uses all credentials in parallel (step above)
   - Wait for all, merge results into `cop` dict

4. In `tile_cache.py`, add `cred_index` parameter to `get_ndvi()`, `get_sar()`,
   `get_harmonics()` — they already accept it, just ensure it's passed through.

Key constraint: openEO rate limit is 1 request per 5 seconds **per account**
(footnote 13 on quotas page). Different accounts can fire simultaneously.
So with 4 accounts: 4 requests per 5 seconds = 48/min.

Test: Process a single KG with `POST /api/v1/processing/single?kg=XXXXX` and
watch the processor log. NDVI months should appear interleaved across credentials
instead of sequential.

#### Step 5 — NDVI composite via Sentinel Hub Process API (separate quota pool, ~3 hours work)

The SH Process API uses Evalscripts (server-side JS) to compute NDVI with cloud
masking and return a GeoTIFF in ~10 seconds. Uses the **Sentinel Hub PU quota**
which is completely separate from openEO credits.

Saves: ~10 min → ~10 sec for NDVI composite. PU cost: ~19/tile.

Our credentials (`sh-*` client IDs) already work for Sentinel Hub — same OAuth.
Auth endpoint: `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`
Process endpoint: `https://sh.dataspace.copernicus.eu/api/v1/process`

**What to do:**

1. Create `sentinelhub_client.py` (~200 lines):

   ```python
   def get_ndvi_composite_sh(
       bbox_wgs84: dict,
       year: int = 2024,
       cred_index: int = 0,
   ) -> dict:
       """Fetch NDVI composite via Sentinel Hub Process API.
       Returns same format as copernicus.get_ndvi_composite():
       {"ndvi": np.ndarray(H,W, float32), "transform": Affine, "crs": CRS}
       """
   ```

   Evalscript for NDVI composite (cloud-masked temporal median):
   ```javascript
   //VERSION=3
   function setup() {
     return {
       input: [{ bands: ["B04", "B08", "SCL"], units: "DN" }],
       output: { bands: 1, sampleType: "FLOAT32" },
       mosaicking: Mosaicking.ORBIT
     };
   }
   function evaluatePixel(samples) {
     let validNDVI = [];
     for (let i = 0; i < samples.length; i++) {
       let scl = samples[i].SCL;
       // Skip clouds, shadows, snow, saturated
       if ([0,1,3,8,9,10,11].includes(scl)) continue;
       let b04 = samples[i].B04, b08 = samples[i].B08;
       if (b04 + b08 === 0) continue;
       validNDVI.push((b08 - b04) / (b08 + b04));
     }
     if (validNDVI.length === 0) return [NaN];
     validNDVI.sort((a, b) => a - b);
     let mid = Math.floor(validNDVI.length / 2);
     let median = validNDVI.length % 2 !== 0
       ? validNDVI[mid]
       : (validNDVI[mid - 1] + validNDVI[mid]) / 2;
     return [median];
   }
   ```

   Request body:
   ```json
   {
     "input": {
       "bounds": {"bbox": [west, south, east, north], "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
       "data": [{
         "type": "sentinel-2-l2a",
         "dataFilter": {"timeRange": {"from": "2024-04-01T00:00:00Z", "to": "2024-09-30T23:59:59Z"}}
       }]
     },
     "output": {
       "width": <pixels>,
       "height": <pixels>,
       "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
     },
     "evalscript": "<evalscript above>"
   }
   ```

   Compute width/height from bbox at 10m resolution:
   `width = round((east - west) * 111000 * cos(lat_mid_rad) / 10)`
   `height = round((north - south) * 111000 / 10)`

   Auth: POST to token endpoint with `grant_type=client_credentials`,
   `client_id=<sh-xxx>`, `client_secret=<secret>`. Token valid 10 min.
   Cache token per credential. Standard `requests` library, no `openeo` dependency.

   Parse response: raw TIFF bytes → `rasterio.MemoryFile` → read band 1.
   Construct Affine transform from bbox + pixel dimensions.

   PU cost estimate: ~19 PU per 0.1° tile (area_factor 3.13 × 1 band_factor × 6 samples × 1 INT16).
   At 40k PU/month across 4 accounts: ~2100 tiles/month from SH alone.

2. In `tile_cache.py`, modify `CopernicusTileCache.get_ndvi()` (~line 236):
   - Try `sentinelhub_client.get_ndvi_composite_sh(tile_bbox, year, cred_index)` first
   - Fall back to `copernicus.get_ndvi_composite()` on failure or if SH PU exhausted
   - Same cache layer — transparent to callers

3. In `copernicus.py`, update `get_ndvi_composite()` similarly for interactive use.

Test: `python3 -c "from sentinelhub_client import get_ndvi_composite_sh; r = get_ndvi_composite_sh({'west':15,'south':47.5,'east':15.1,'north':47.6}); print(r['ndvi'].shape, f'range=[{r[\"ndvi\"].min():.2f},{r[\"ndvi\"].max():.2f}]')"`

#### Step 6 — Pipeline all products concurrently (quick win after steps 3-5)

Once steps 3-5 are done, the products use different backends:
- WorldCover: AWS COG (no quota)
- NDVI composite: Sentinel Hub (SH PU quota)
- SAR: openEO (openEO credits)
- NDVI time series: openEO parallel (openEO credits, all 4 credentials)

They can all run simultaneously.

**What to do:**

Refactor `_try_fetch_single()` in `austria_processor.py` (~line 4175):
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _try_fetch_single(bbox, label=""):
    cop = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures["worldcover"] = pool.submit(worldcover_cog.get_land_cover_cog, bbox)
        futures["ndvi"] = pool.submit(cop_cache.get_ndvi, bbox, obs_year)  # → SH
        futures["sar"] = pool.submit(cop_cache.get_sar, bbox, obs_year)
        # Wait for these 3 (~seconds for WC+NDVI, ~5 min for SAR)
        for key, fut in futures.items():
            try:
                result = fut.result(timeout=600)
                # merge into cop dict...
            except Exception as e:
                log.warning("%s%s failed: %s", label, key, e)
    # Then harmonics (uses all credentials in parallel internally)
    if cop:
        harm = cop_cache.get_harmonics(bbox, year=obs_year, ...)
        if harm is not None:
            cop["harmonics"] = harm
    return cop if cop else None
```

Expected per-tile time after all steps:
- WorldCover: <2 sec (COG)
- NDVI composite: ~10 sec (SH) } all concurrent
- SAR: ~5 min (openEO)         } ← bottleneck
- NDVI time series: ~8 min (parallel openEO across 4 credentials)
- **Total: ~8 min/tile (down from ~47 min)**

#### Summary of expected impact

| Metric | Before | After |
|--------|--------|-------|
| Time per tile | ~47 min | ~8 min |
| Time per KG (3 tiles) | ~2.3 hours | ~25 min |
| Remaining 6000 KGs | ~583 days | ~104 days |
| openEO credits used | all 4 products | SAR + NDVI TS only |
| SH PU used | 0 | NDVI composite |
| COG requests used | 0 | WorldCover |
