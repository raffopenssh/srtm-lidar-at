# Austria Processor — Mental Model

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

3. **Tile checkpoints** (two tiers, deployed dd08d74):

   **Metadata tier** — each completed tile is pickled to
   `data/austria_processor/tile_checkpoints/<kg>/tile_N.pkl`. Carries the
   full per-tile result (segmentation, classified objects, vector cadastre,
   NDVI/SAR/Hansen extracts — *everything except raw rasters*). KB–MB per
   tile. Restored on same-peer crash/restart.

   **Cross-peer registry** — when a KG is aborted/deferred mid-flight
   (BEV exhaustion, cred rotation, timeout, postpone, role eviction), the
   parent gzip+tars all tile pickles and uploads to the shared Zenodo
   cache deposit as `chkpt_<kg>.tar.gz`. The next peer that picks up the
   KG downloads + unpacks them inside `process_one_kg()` before the tile
   loop runs. So BEV-expensive aborts are not wasted across peers. See
   `tile_checkpoint_registry.py`. Fleet visibility: cache_manifest.json
   carries `chkpt_*` entries (size>0 = present, size=0 = tombstone
   = deleted on completion) and propagates fleet-wide every 5 min via the
   existing `_sync_peer_data` cycle — no new traffic.

   **Raster sidecars (Phase 1)** — per-peer local only, never uploaded.
   After each tile reads DTM/DSM/nDSM + multi-date DTM/DSM + ortho RGB+NIR,
   `tile_raster_sidecar.persist_dtm_dsm/persist_ortho` writes `.npy` files
   under `tile_checkpoints/<kg>/tile_N/raster/`. `build_full_gpkg_tiled`
   mmaps these instead of re-reading BEV, eliminating 6+ BEV passes per
   tile during the `gpkg_full` step (was the longest-running step
   pre-deploy at 5–7 h/peer). Gated on free disk ≥ `SIDECAR_MIN_FREE_GB`
   (env-overridable, default 8 GB) — silent no-op below that, with the
   BEV re-read fallback. Bulk-released at the existing tile-checkpoint
   free point after `gpkg_full` succeeds.

   All three tiers are deleted after successful KG completion (raster
   sidecars also released on Zenodo-upload-failure rebuild path). Audit
   trail in the merged 24h log via `?q=chkpt`.

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
| `tile_checkpoints/<kg>/tile_N.pkl` | Child | Child (on retry); other peers (via Zenodo chkpt registry) | Resume from last completed tile |
| `tile_checkpoints/<kg>/tile_N/raster/*.npy` | Child (per-tile loop) | Child (`gpkg_full` step, mmap) | Skip BEV DTM/DSM/ortho re-reads in `gpkg_full` (Phase 1, dd08d74) |
| `checkpoint_registry.json` | Parent (uploading peer only) | Primary `/process.txt` (read via `cache_manifest.json` mirror) | LRU registry of tile-tars uploaded to Zenodo — `{kg: {ts, n_tiles, bytes, name}}`, cap `MAX_REGISTRY_KGS=200` |
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
#    NOTE: this drops the metadata pickles AND the local raster sidecars
#    (Phase 1). If the same KG has a chkpt_*.tar.gz on Zenodo it will be
#    re-downloaded into the dir before the tile loop runs — if you want a
#    truly clean restart, also evict the Zenodo bundle (single-peer admin
#    op, normally only needed on schema bumps):
#      python3 -c "import tile_checkpoint_registry as r; r.delete_kg('TARGET_CODE')"
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



---

*See `AGENTS.md` for the project map.*
