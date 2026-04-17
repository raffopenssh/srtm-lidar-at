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
| `app.py` | 3223 | Flask API — all endpoints, async task system, progress tracking |
| `static/index.html` | ~2100 | Single-file Leaflet UI (all JS/CSS inline) |
| `object_segmentation.py` | 1939 | Main analysis pipeline: Felzenszwalb+RAG → per-object classify |
| `learned_classifier.py` | — | Random Forest classifier (44 features, cadastre-trained) |

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

### Austria Processor Operations

The processor runs as `austria_processor.service`. It processes KGs sequentially,
each tiled into 1.5km windows. It resumes from where it left off on restart
(no KG data is lost — incomplete KGs are retried).

```bash
# --- Logs (primary source of truth) ---
tail -f data/austria_processor/logs/processor.log     # live log
tail -100 data/austria_processor/logs/processor.log   # recent history
grep -i "warning\|error\|failed" data/austria_processor/logs/processor.log | tail -20  # problems

# --- Status ---
curl -s http://localhost:8000/api/v1/processing/status | python3 -m json.tool
# Key fields: state, completed, failed, current_kg.{code,step}, system.{ram_pct,disk_free_gb,proc_pid}

# --- Restart (SIGKILL needed — graceful stop waits for current tile to finish) ---
sudo systemctl kill -s SIGKILL austria_processor
sleep 2
sudo systemctl start austria_processor
# Note: RestartSec=60 in the unit file, so systemd waits 60s between stop and start.
# The `start` command may block ~60s while systemd enforces this.

# --- Dashboard ---
# https://srtm-lidar-at.exe.xyz:8000/process.html
# Shows: service card (PID/RAM), progress, rate, system resources, current KG
# pipeline steps, map of processed KGs, live log, Zenodo manifest.
```

**Common issues to check in logs:**
- `WARNING copernicus: Synchronous download timed out` → normal, falls back to batch job
- `WARNING tile_cache: Copernicus SAR tile fetch failed` → SAR download failed, skipped (non-fatal)
- `WARNING tile_cache: Hansen resample failed` → Hansen tile missing for bbox (western Vorarlberg)
- `ERROR copernicus: credits exhausted` → all Copernicus credentials used up, processor pauses
- `RuntimeWarning: Mean of empty slice` → should be fixed; if seen, check terrain_analysis.py

**Key files:**
- `data/austria_processor/logs/processor.log` — full log (stdout+stderr)
- `data/austria_processor/progress.json` — live state (read by dashboard API)
- `data/austria_processor/in_progress_kg.txt` — current KG code (for crash recovery)
- `data/austria_processor/zenodo_manifest.json` — upload tracking
- `data/austria_processor/copernicus_tiles/` — grid-snapped Copernicus cache (.npz)
- `data/austria_processor/hansen_tiles/` — grid-snapped Hansen cache (.npz)
- `data/austria_processor/bev_tile_cache/` — BEV DTM/DSM windowed read cache
- `rf_training_data/copernicus_cache/` — per-bbox Copernicus cache (.tif, from RF training)

**Cache corruption:** Downloads use atomic writes (temp file + rename). If you see
0-byte `.tif` or `.npz` files in cache dirs, delete them — they're leftovers from
interrupted downloads before the atomic write fix.

Deps: rasterio, pyproj, shapely, numpy, scipy, scikit-image, scikit-learn, flask, gunicorn, openeo, requests

### Austria Processor
| File | Purpose |
|------|----------|
| `austria_processor.py` | Main processor: iterates KGs, tiles, segments, builds GPKGs + JSON, uploads to Zenodo |
| `zenodo_client.py` | Python Zenodo API client (port of Go zenodo-mirror-go-pkg) |
| `austria_processor.service` | systemd unit for background processing |
| `static/process.html` | Processing dashboard (status, map, controls, Zenodo manifest) |

### Tiled Full-KG Processing

`process_one_kg()` tiles the **entire KG** into overlapping 1.5km windows, processes each
tile through the full pipeline, and merges results for complete KG coverage.

**Flow:**
1. Fetch cadastre → compute full KG bbox from geometry union (make_valid)
2. `_compute_tile_grid()`: 1.5km tiles, 100m overlap
3. **Per tile**: LiDAR (3 dates) + ortho + Copernicus + Hansen + Felzenszwalb segmentation + RF classify
4. Remap obj_ids to global unique range; dedup at tile boundaries via centroid-ownership (core zone = tile shrunk by 50m on overlap sides)
5. `_merge_terrain_stats()`: pixel-weighted merge across tiles
6. `build_full_gpkg_tiled()`: per-tile DTM/DSM/nDSM + segment_type rasters, segment vectors
7. `build_light_gpkg_tiled()`: segment rasters + all parcels/buildings enriched from their covering tile
8. `build_json_summary_tiled()`: every parcel gets elevation + area_summary + height_distribution

**Memory**: one 1.5km tile (~90MB) in memory at a time. Works for KGs up to 27km (Matrei: ~324 tiles).

**Retry ladder**: `RETRY_LADDER = [1.5, 0.5, 0.2]` km — on timeout, shrink tile size and retry.

**Key helpers:**
- `_find_tile_for_point(e, n, tiles)` — find which tile covers a point (for parcel enrichment)
- `_read_dtm_for_tile(tr)` — re-read DTM from BEV cache (instant, no HTTP)
- `_segment_touches_edge(seg_mask)` — detect truncated segments at tile boundary
- `edge_clipped` flag on new_buildings/infrastructure vectors

### Austria Processor Data
| Path | Purpose |
|------|----------|
| `data/austria_processor/zenodo_manifest.json` | Tracks all Zenodo uploads + failures |
| `data/austria_processor/json/` | Per-KG JSON summaries (kept under 4GB) |
| `data/austria_processor/progress.json` | Live progress state for dashboard |
| `data/austria_processor/kg_list.json` | Cached list of all ~8440 Austrian KGs |
| `data/austria_processor/bev_tile_cache/` | Cached BEV DTM/DSM windowed reads (fast re-read) |
| `data/austria_processor/gpkg/` | Temp GPKG files (deleted after Zenodo upload) |

### Austria Processor API Endpoints
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

### Per-KG Outputs
1. **Full GPKG** (`{kg}_full.gpkg`): per-tile DTM/DSM/nDSM + segment_type rasters, segment vector polygons
2. **Light GPKG** (`{kg}_light.gpkg`): segment raster+vector, **all** parcels w/ DTM heights, **all** buildings w/ object heights, new buildings, infrastructure
3. **JSON summary** (`{kg}.json`): area summary, height distributions, landscape characterisation, top objects/trees, terrain, NDVI, Hansen loss, new buildings, infrastructure, coverage stats, methods

JSON `coverage` section: `n_tiles`, `tile_km`, `parcel_elevation_coverage_pct`, `parcel_segmentation_coverage_pct`, `building_height_coverage_pct`.
