# `app.py` — Mental Model

13k-line Flask app. Single process, two gunicorn workers × four threads.
Everything mutating goes through admin-token auth except loopback (127.0.0.1
without XFF). All long files use `# === SECTION:` markers —
`grep -n '# === SECTION' app.py`.

## Roles this single file plays

```
          ┌─────────────────────────────────────┐
          │           gunicorn (srv.service)         │
          │  worker A           │     worker B          │
          │   │   │   │           │     │   │   │   │   │
          │  thr thr thr thr   │    thr thr thr thr    │
          └──┬────────────────────┴───────────────────────┘
             │   one worker holds:
             ├─ director loop (fcntl lock on director.lock)
             ├─ status push thread
             ├─ peer-data sync thread
             ├─ role-data eviction loop
             └─ zenodo lock broker (in-memory mutex)
```

- **Sync HTTP**: most endpoints (geometry parse, query index, share I/O, shallow proxies)
- **Async tasks**: `/segment`, `/onestop`, large `/parcels/batch`, `/export/mbtiles` — spawn daemon threads, write file-backed progress
- **Background loops** (one per worker, gated by file lock so only one is real): director, status push, peer sync, role eviction
- **Proxies**: dashboard `process.html` and `query.html` call director + processor APIs through this file

## Section map (with line numbers as of this doc)

| L | Section | What lives here |
|---:|---|---|
| 261 | Copernicus credentials API | `/api/v1/credentials*` — add/list/validate creds. Pure cache reads on hot path; only director may probe. |
| 887 | Status push to director | Thread that POSTs `/processing/status` to current director (peers → director) |
| 997 | Role-data eviction | Demoted peers free `json/*.json` + `search_index.db` after grace |
| 1237 | Processing queue | `_TASK_SEMAPHORE` — max 2 concurrent heavy tasks (segment / export) |
| 1333 | Async task system | File-backed progress (`/tmp/segment_progress/`) + result storage (`/tmp/segment_results/`, gzip, 4 h TTL). Auto-saves to share. |
| 1902 | Austria Processor endpoints | Proxy `/processing/*` endpoints. Read `progress.json`, write retry queue, start/stop subprocess, broker cache misses. |
| 4183 | Bandwidth & Peer Director | Local vnstat reader + all `/director/*` endpoints. Director loop body. |
| 5258 | Zenodo upload mutex | In-process mutex used by primary's own subprocess |
| 5317 | Zenodo lock broker proxy | `/zenodo/lock` — cluster-wide TTL lease, peers point at primary |
| 5852 | Director high-availability | Heartbeat / shadow / handover / takeover / step-down / announce |
| 6104 | Search index API | `/index/*`, `/kg/<code>`, `/parcel/<id>`, `/query` |
| 6185 | GPKG detail lazy-load | Per-KG buildings / new_buildings / infrastructure / segments / segment_points (downloads light GPKG on demand) |
| 6342 | GET query param parsers | `_parse_compound_filters_get`, `_parse_parcel_filters_get` (pf_ prefix) |
| 6780 | Async query task support | `/parcels/batch` async mode |
| 7142 | Cross-API bridge | `/parcels/batch`, `/parcels/landscape`, `/query/nature`, `/parcel/<id>/detail`, `/kg/<code>/profile`, cadastre proxies |
| 7595 | Geometry + parameter helpers | `_get_geometry`, `_get_params`, `_validate_area`, `_clean_polygon` |
| 7975 | `/elevation` | DTM enrichment |
| 8066 | `/terrain` | Slope/aspect/roughness |
| 8099 | `/segment` | **Main analysis endpoint.** `_segment_core`, `_segment_worker` (async). |
| 8586 | `/segment/overlay` | RGBA raster rendering, overlay cache |
| 9164 | `/export/geopackage` | Async GPKG (raster + vector) |
| 9272 | `/export/kml` | Grouped/styled KML |
| 10210 | `/export/mbtiles` | Async single-layer MBTiles |
| 10483 | `/changes` | Multi-date DTM temporal analysis |
| 10550 | `/changes/trees` | Tree growth analysis |
| 10604 | Multi-epoch summary | Aggregated change summary |
| 10634 | `/info` | Server info, capabilities |
| 10636 | LiDAR/ortho/DTM/CIR/Hansen | Tile overlays + raw GeoTIFF download |
| 11180 | RF classifier training | `/classifier/train`, `/classifier/status` |
| 11388 | `/docs` + share | Docs endpoint, share helpers |
| 11451 | `/layers` | Per-bbox availability check |
| 11550 | `/onestop` | Single-URL segment + download (queued) |
| 12059 | Quality flags + feedback | `/flags/match`, `/feedback` |
| 12347 | `/parse-geometry` | KML/GeoJSON/Shapefile/GPX/WKT upload |
| 12502 | `/share` | Save/load/rename/list shares (1 GB cap, LRU, `data/shares/`) |

## Async task lifecycle

```
client POST /segment?async=true
    → returns task_id (uuid)
    → spawns daemon thread _segment_worker(task_id, ...)
          │
          ├─ writes /tmp/segment_progress/<task_id>.json (atomic temp+rename)
          ├─ reads aborted flag from same dir
          └─ on success: writes /tmp/segment_results/<task_id>.json.gz
                           and saves auto-share `auto-<task_id[:8]>`

client polls /segment/progress?task_id=<id>
    → reads progress JSON. When status=done, includes auto_share_id.

client GET /segment/result?task_id=<id>
    → streams gzipped JSON.

client POST /segment/abort?task_id=<id>
    → writes abort flag; worker checks at safe points and bails.
```

Frontend stores active `taskId` in `localStorage`; on reload either resumes
polling or loads the auto-share. Cleanup runs every hour, drops anything
>4 h old.

## Background threads (one is real, others fail file-lock)

| Thread | Purpose | Cadence | Section |
|---|---|---|---|
| Director loop | Reads `peers.json`, polls bandwidth, picks active peer, pushes plans, reissues cred slices, manages parallel frontiers + cache-only fleet | 30 s tick | 4183 |
| Status push | POSTs local `/processing/status` to current director | 30 s | 887 |
| Peer data sync | Pulls KG JSONs + manifest from peers, merges manifest, refreshes search index | 5 min | (utility) |
| Zenodo lock heartbeat | Renews lease on broker every 30 s while holding | per-lease | 5258 |
| Role-data eviction | Demoted peers — frees JSON+index after 1 h grace | 10 min | 997 |
| HA watchdog | Pings director heartbeat, promotes shadow on 6 misses | 30 s | 5852 |

All loops use `_DIRECTOR_FILE_LOCK` style fcntl locks so only one gunicorn
worker actually runs them — the other worker's loop sleeps.

## Authentication

`_require_admin_token()` (see top of file). Loopback (127.0.0.1, no XFF) is
exempt. Token at `data/admin_token` (mode 0600), re-read every call so
rotations propagate without restart.

Mutating routes that require it:
- `/admin/*`
- `/director/*` (except `heartbeat`, `identity` GET)
- `/processing/start|stop|pause|resume|single|retry|throttle|cache_misses|cache_manifest`
- `/credentials*` POST/DELETE
- `/zenodo/lock*` (peer→broker)

Dashboard prompts on first 401 and stores token in `localStorage` key
`srtm_admin_token`. Reset: `srtmResetAdminToken()`.

## Where to look (debug fast-path)

```bash
# Endpoint lives in which section?
grep -n '# === SECTION\|@app.route' app.py | grep -B1 'route("/api/v1/<path>"'

# A specific async task
ls /tmp/segment_progress/ /tmp/segment_results/
cat /tmp/segment_progress/<id>.json | python3 -m json.tool

# Director state
cat data/austria_processor/director_state.json | python3 -m json.tool

# Who holds the Zenodo lock?
cat data/austria_processor/zenodo_lock_state.json
```

## Critical invariants (re-stated; see AGENTS.md)

- Don't probe Copernicus credentials from the request path. Pure cache reads only.
- All Zenodo writes go through `/zenodo/lock`.
- `_require_admin_token()` on every mutating endpoint.
- File-lock the director loop before starting a daemon thread that mutates global state.
- Async tasks must call `_check_aborted()` at progress checkpoints — otherwise abort is a lie.

---

*See `AGENTS.md` for the project map.*
