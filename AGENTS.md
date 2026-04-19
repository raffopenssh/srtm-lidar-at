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
| `app.py` | ~5700 | Flask API — all endpoints, async task system, progress tracking |
| `static/index.html` | ~3100 | Single-file Leaflet UI (all JS/CSS inline) |
| `object_segmentation.py` | ~2200 | Main analysis pipeline: Felzenszwalb+RAG → per-object classify |
| `learned_classifier.py` | ~560 | Random Forest classifier (44 features, cadastre-trained) |

### Search Index
| File | Purpose |
|------|----------|
| `search_index.py` | SQLite FTS5 + R-tree index over all 8440 KGs. Spatial/text/admin/aggregate queries <25ms. Auto-rebuilds on new JSONs. |

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

### Deprecated (kept for reference)
`landscape_classifier.py`, `object_classifier.py`, `scene_adaptive_classifier_patches.py`

## Services

| Unit | What | Config |
|------|------|--------|
| `srv.service` | gunicorn (2 workers, 4 threads, port 8000) | MemoryMax=3G, Restart=on-failure |
| `rf_train.service` | RF training background job (4000 KGs) | Restart=on-failure, RestartSec=30 |

Both in `/etc/systemd/system/`. Source copies in repo root.

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
| POST | `/api/v1/parcels/batch` | Batch parcel enrichment — explicit IDs or query-based |
| GET | `/api/v1/parcels/landscape` | Query parcels with landscape filters (GET version of batch) |
| GET | `/api/v1/query/nature` | Nature conservation opportunity finder (conservation score 0-100) |
| GET | `/api/v1/parcel/<id>/detail` | Full combined parcel detail (both APIs) |
| GET | `/api/v1/kg/<code>/profile` | Combined KG profile (both APIs) |
| GET | `/api/v1/cadastre/legal/search` | Proxy: legal refs search |
| GET | `/api/v1/cadastre/protected_areas` | Proxy: WDPA protected areas |
| GET | `/api/v1/cadastre/landuse/distribution` | Proxy: landuse distribution |
| GET | `/api/v1/cadastre/landuse/codes` | Proxy: landuse reference codes |

`POST /api/v1/parcels/batch` supports two modes:
- **IDs mode**: `{"parcel_ids": ["63349-505/3", ...]}`  (max 200)
- **Query mode**: `{"query": {<any cadastre /query params>}, "landscape_filters": {<landscape post-filters>}}`

Landscape filters: `min_vegetated_fraction`, `min_ndvi`, `min_tree_canopy_sqm`,
`min_elevation`, `max_elevation`, `min_conservation_score`, `dominant_type`, `sort`, `sort_dir`

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

**Upload**: Bundles local NPZ tiles into ZIP archives (one per product × 0.5° lat strip),
uploads to a single Zenodo deposit. ~30 ZIPs, ~540 MB for all of Austria.
Uses `data/austria_processor/cache_manifest.json` (NOT `zenodo_manifest.json`).

**Download**: On local cache miss, tile_cache checks Zenodo before calling the
expensive API. Uses 2-3 HTTP range requests to read individual NPZ entries from
remote ZIP files. Writes restored tile to local cache dir.

**Why not BEV/ortho?** BEV DTM/DSM/ortho are already COGs with efficient HTTP range
reads. At 1m resolution, all Austria = ~4TB (infeasible for Zenodo). The per-KG
full GPKG (already on Zenodo) contains all BEV layers.

```bash
python3 zenodo_cache.py status      # show local + Zenodo tile counts
python3 zenodo_cache.py dry-run     # build ZIPs without uploading
python3 zenodo_cache.py upload       # upload local tiles to Zenodo
```

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
1. `copernicus.py` → `_CREDENTIALS` list
2. Delete `data/austria_processor/copernicus_paused` if it exists
3. Optionally reset circuit breaker: delete `data/austria_processor/openeo_circuit.json`

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
| `tile_cache.py` | 900L | Grid-snapped caches for Copernicus + Hansen. `CopernicusTileCache`, `HansenTileCache`, `order_kgs_nearest_neighbor()`. Zenodo fallback on miss. |
| `zenodo_cache.py` | 952L | Persistent cache on Zenodo. Uploads tiles as ZIP archives (one per product × lat strip). Downloads via HTTP range reads (2-3 requests per tile). Separate `cache_manifest.json`. |
| `object_segmentation.py` | 2218L | `segment_objects_in_area()` — Felzenszwalb + RAG + RF classify. Called once per tile. |
| `copernicus.py` | 1262L | openEO client: NDVI, WorldCover, SAR, harmonics. Has credential rotation + sync/batch fallback. |
| `ortho_io.py` | 992L | BEV orthophoto reader (RGBI, 47 Operates, DOP fallback). |
| `raster_io.py` | 359L | Windowed reads from BEV GeoTIFFs via `/vsicurl/`. |
| `hansen.py` | 453L | Hansen Global Forest Change data reader. |
| `cadastre.py` | 459L | Building footprints + parcel boundaries from cadastre API. |
| `terrain_analysis.py` | 157L | Slope, aspect, TRI, curvature from DTM. |
| `learned_classifier.py` | 559L | RF model loading + 44-feature classification. |
| `zenodo_client.py` | 841L | Zenodo deposit creation, file upload, publish. `Manifest` class for tracking. |
| `bev_retry.py` | 252L | Exponential backoff + proxy rotation for `rasterio.open()`. |

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
| `copernicus: credits exhausted` | All openEO credentials used up | Update `_CREDENTIALS` in `copernicus.py`, delete `data/austria_processor/copernicus_paused` |
| Timeout on large KG | Copernicus slow (NDVI downloads) | Automatic: retry once with checkpoints, then deferred retry 5 KGs later if transient |
| OOM kill | KG exceeds 3GB MemoryMax | Automatic: systemd restarts, tile checkpoints preserved |
| `Synchronous download timed out` | Normal Copernicus behavior | No action — falls back to batch job automatically |
| `Hansen resample failed` | No Hansen data (western Vorarlberg) | Non-fatal, skipped. Hansen data sparse at AT borders |
| 0-byte cache files | Interrupted before atomic-write fix | Delete the 0-byte `.tif`/`.npz` files manually |
| Disk critically low | Caches filling disk | Automatic: `check_disk_space()` does LRU cleanup, pauses if <3GB |
| Stale `progress.json` | Service died without cleanup | Restart clears stale state in `main()` init block |
| Transient server error (500/503/timeout) | Remote service hiccup | Automatic: deferred retry 5 KGs later (up to 2×), then permanent fail |

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
| GET | `/api/v1/processing/log` | Recent processor log lines |
| GET | `/api/v1/processing/manifest` | Zenodo manifest entries |
| GET | `/api/v1/kg/<kg_code>` | KG JSON summary (local or Zenodo link) |
| GET | `/api/v1/parcel/<parcel_id>` | Parcel lookup via KG JSON |

### Files

| File | Size | Purpose |
|------|------|----------|
| `austria_processor.py` | 5131L | Main processor (this section documents it) |
| `zenodo_client.py` | 841L | Zenodo API client + `Manifest` class |
| `zenodo_cache.py` | 952L | Zenodo-backed persistent cache: uploads/downloads Copernicus+Hansen tiles as ZIP archives |
| `tile_cache.py` | 900L | Grid-snapped Copernicus + Hansen caching, Zenodo fallback on miss |
| `austria_processor.service` | — | systemd unit (MemoryMax=4G, Restart=on-failure) |
| `static/process.html` | 1117L | Dashboard UI (status, map, log, Zenodo manifest) |
| `data/austria_processor/MONITOR.md` | — | Monitoring checklist + expected timelines |
| `gpkg_streamed.py` | ~500L | Strip-streamed full-GPKG builder for large KGs (auto-used >100 Mpx) |

### Where to Look When Debugging

| Problem area | Look at |
|---|---|
| KG fails during segmentation | `process_one_kg()` line ~3700, calls `object_segmentation.segment_objects_in_area()` |
| GPKG output wrong | `build_full_gpkg_tiled()` (line 1903) or `build_light_gpkg_tiled()` (line 2177) |
| JSON summary wrong | `build_json_summary_tiled()` (line 2537) |
| Parcel heights wrong | `enrich_parcels_with_heights()` (line 647) |
| Tile boundary artifacts | Centroid-ownership dedup in `process_one_kg()` ~line 3830, `_segment_touches_edge()` |
| Copernicus data missing | `tile_cache.py` → `copernicus.py` → credential rotation + circuit breaker |
| BEV data read failure | `raster_io.py` → `bev_retry.py` (exponential backoff + proxy rotation) |
| Zenodo upload failure | `main()` ~line 4900, calls `zenodo_client.py` |
| Dashboard not updating | `app.py` `/api/v1/processing/status` reads `progress.json`; check parent thread alive |
| Copernicus quadrant fallback | `_quadrant_split()` ~line 4132, called from `_fetch_copernicus_for_tile()` |


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
