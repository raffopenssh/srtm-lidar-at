# AGENTS.md — srtm-lidar-at

> **Token discipline**: this file is a navigation hub. Deep docs live in `docs/`.
> Don't load a `docs/*.md` unless your task touches that subsystem.

## ⚡ Fast read for agents: `/process.txt`

**Always start here when investigating cluster state.** Token-cheap,
text-only mirror of `/process.html` — full director / peer / log / Zenodo /
bandwidth snapshot. Per-worker render cache (10 s TTL, X-Cache header) +
director status cache (30 s, cross-worker via disk) so even concurrent
agent polls cost almost nothing on gunicorn. Add to your context with one
curl:

```bash
curl -s https://srtm-lidar-at.exe.xyz:8000/process.txt          # default 60 log lines, last 24h
curl -s 'https://srtm-lidar-at.exe.xyz:8000/process.txt?warn=1' # warnings + errors only
curl -s 'https://srtm-lidar-at.exe.xyz:8000/process.txt?peer=at3&log=200'
curl -s 'https://srtm-lidar-at.exe.xyz:8000/process.txt?q=cred&log=300'
curl -s 'https://srtm-lidar-at.exe.xyz:8000/process.txt?hidden=1' # also list stopped/idle peers
curl -s 'https://srtm-lidar-at.exe.xyz:8000/process.txt?hours=168&log=500&q=Diendorf' # 7d back, archive
```

Query params:
- `log=N` (default 60, max 500) — merged-log line count, newest first
- `warn=1` — restrict log to warnings + errors
- `peer=<substr>` — filter peer roster + log lines by id substring
- `q=<substr>` — substring filter on log message body
- `hidden=1` — also include stopped/idle/complete peers in roster
  (default: hidden; attention-state peers are always shown)
- **`hours=H`** — look back H hours into the merged log. Default 24
  (live ring only). Higher values transparently dip into the
  **long-term archive** at `data/log_archive/YYYY-MM-DD.jsonl.gz`
  (per-UTC-day gzipped JSONL, written when the live ring prunes).
  Use this to mine the full ~200-day forensic record — e.g.
  `?hours=2400&q=cred` for credential-rotation history.

For structured access, pair with `/api/v1/director/status` and
`/api/v1/director/log/history?hours=H` (both documented in
`/api/v1/docs/llm.txt`). The history endpoint also reads from the
per-day archive when `hours` exceeds the live ring — returns the
*most-recent* `limit` matches in range (sets `truncated:true` if more
exist), avoiding the prior bug that truncated the tail of the day.
Director orchestration events — peer auto-updates, Copernicus credential
revalidation/add/remove, frontier credential & cache-cell plan changes,
**bandwidth-wall auto-park / park-until-renewal** events — are emitted via
`app.director_event(…)` and appear inline in the merged 24h log
(`peer=director` for fleet-wide events, otherwise the affected peer id).

### Bandwidth fields in `/process.txt`

The text dashboard surfaces all the bandwidth telemetry needed to debug a
fleet that's canary-by-default (every peer sampled, slowdown auto-park
on throughput collapse, park-until-renewal on near/over budget):

* **Top-line `fleet_bw`** — `used_gb`, `budget_nominal_gb`, peers,
  parked, soonest `next_renew_in_days`, plus `wall~Xgb(min=Y,n=Z)`
  distilled from peers that auto-parked on throughput collapse.
* **Per-peer columns** — `bw%` (used/effective_budget), `used/bud`
  absolute GB, plus a free-form `bw_extras` tail with optional tokens:
  - `r=0.42` — canary recent/baseline throughput ratio (red <0.30)
  - `CANARY` — has an explicit `budget_gb` override
  - `cap=80G` — peer's persisted `observed_cap_gb` (it has hit a wall)
  - `parked→3.1d` / `parked→4.5h` — active `not_before` cooldown
  - `rd=15` — effective renew_day (override > first_seen day-of-month)
* **`fleet_bw` JSON block** in `/api/v1/director/status` for structured
  consumers — fields above plus `observed_cap_gb_min/median`,
  `peers_enabled`, `peers_parked`.

Thresholds (peer_director.py):
* `BANDWIDTH_LOW_WATER_GB = 4` — mid-KG triggers graceful stop +
  park-until-renewal so the in-flight KG finishes & uploads.
* `BANDWIDTH_HARD_DEPLETED_GB = 1` — hard stop, even mid-KG.
* `CANARY_SLOWDOWN_RATIO = 0.30`, baseline 30 min, 500 MB minimum bytes
  before a peer can be parked for throughput collapse ("soft" park).
* `CANARY_BASELINE_NETWORK_MBPS = 5.0` and `CANARY_RECENT_PARKED_MBPS
  = 0.5` — a soft park is upgraded to a **quality observation**
  (sets `observed_cap_gb` + counts toward fleet wall) only when the
  pre-collapse baseline was network-grade AND the peer truly stalled.
  Otherwise the peer was probably just Zenodo-upload-bound.
* `CANARY_QUALITY_PERSIST_S = 15 min` — the slowdown must be
  continuously observed for 15 min (streak resets above ratio 0.60)
  before counting as quality. Defends against transient network
  blips and short upstream outages.
* `FLEET_CONCURRENT_SLOWDOWN_FRAC = 0.30` — if ≥0.30 of canary-eligible
  peers are simultaneously in slowdown, treat as a fleet-wide upstream
  event (BEV/Zenodo/internet hiccup); soft park still fires defensively
  but no quality observation is recorded for any peer that tick.
  Surfaces in `/process.txt` as `slowdown N/M [FLEET-WIDE]`.
* `FLEET_WALL_MIN_QUALITY_OBS = 5` — fleet wall (`observed_cap_gb_min/median`)
  is suppressed in `fleet_bw` until ≥5 distinct quality obs across
  the fleet. Until then `process.txt` shows `wall=? (gathering, N
  quality obs)` instead of inventing a confident wall.
* `CANARY_PARK_COOLDOWN_S = 6h` — quality-grade parks (real shaping).
* `CANARY_PARK_COOLDOWN_SOFT_S = 1h` — soft parks (defensive, not
  network-grade or not persistent). Almost always upstream blips
  (Zenodo / openEO / transient internet); the short cooldown lets the
  peer rejoin quickly. If the issue is real, the peer will soft-park
  again next tick and eventually graduate to quality-grade.
* Park-until-renewal cooldown = peer's effective `renew_day` next
  occurrence (no need to track per-peer budgets separately; the existing
  `_peer_is_scheduled` gate makes the scheduler skip the peer).

### Throttle / director-efficiency sparkline

The Service-card sparkline (`/process.html` → `renderCapacityHistory`)
plots the rolling `capacity_history` ring (240 ticks ≈ 2h). Each tick
carries:

* `f`   — EMA-smoothed capacity factor in `[THROTTLE_MIN_FACTOR, 1.0]`
* `bev` / `zen` / `cop` — fleet warnings/min per upstream
* `stl` — **fleet CPU-steal median** across *running* peers (same
  population the cache-only ramp brake uses)
* `cpu` — derived `cpu_factor` ramp brake: `1.0` below 30% steal,
  gentle linear ramp, floor `0.55` at 60%+ steal. Mirrors the damping
  curve in `_max_cache_only_peers` so the chart shows *why* the
  director may be running fewer cache-only peers than the warning-rate
  ceiling alone would allow.

Low capacity_factor with steal high → hypervisor pool congested
(reducing peers won't recover cycles; LPT partition does the real
balancing). Low capacity_factor with steal low → a real upstream
is pushing back — check `B / Z / C` chips for which one.

History persists to `director_state.json` every tick so it survives
HA handover and gunicorn's two-worker swap (just like the BEV / Zen /
Cop sub-EMAs). Schema is a 7-tuple; the load path tolerates legacy
5-tuples written by pre-2026-05-19 directors.

`/process.txt` carries a `throttle:` block (window / cap_factor /
steal_med / cpu_factor / per-upstream warns-per-min, min/med/max over
the window) so forensic mining over the long-term archive can
correlate director efficiency with steal trends without parsing JSON.

### Role-based parks (who is parked, and why)

Only three reasons a peer is parked (`not_before` in the future):

1. **Primary** — `_enforce_primary_park`, every tick. Floor
   `not_before=2027-01-01`, `pinned_role=idle`. Manual reset won't
   survive (next tick re-extends). Primary hosts the public
   dashboard and director state; it must never process.
2. **Active director + shadow** — `_enforce_director_self_park`,
   every tick. **Rolling 2 h cooldown**, refreshed while the role
   is held. As soon as the peer stops being director/shadow, the
   stamp expires within 2 h and the peer rejoins rotation — no
   explicit release needed. (Primary is exempt; covered by #1.)
3. **Canary-evidenced BW wall** — `_park_peer_until_renewal`,
   only when the peer has an `observed_cap_gb` set by a quality-
   grade canary slowdown AND used_gb has reached that cap. We do
   NOT park on the nominal 95 GB budget; exe.dev's real limits and
   billing anchors are unknown. Soft canary auto-parks use a short
   1 h cooldown (`auto_park` event); quality parks 6 h.

The one-shot `_release_unverified_bw_parks` rescues peers that got
parked-until-renewal *without* an `observed_cap_gb` (legacy budget-
guess parks). It looks for our own `park_until_renewal` note tag,
so primary's 2027 stamp and canary `auto_park` cooldowns are
preserved.

**Operator note**: if you see an unexpected long `parked→Xd` on a
peer, check `peers.json` → `canary_notes` for the most recent
`event`. `role_park`/`park_until_renewal` are director-written;
`auto_park` is canary-written; anything else is manual.

## TL;DR

Flask + Leaflet app that segments Austrian landscape from BEV LiDAR + BEV ortho
+ Sentinel-2 + ESA WorldCover + Sentinel-1 + Cadastre. A background processor
(`austria_processor`) runs all ~8440 KGs and uploads to Zenodo. A peer director
on the primary VM orchestrates processing across multiple exe.dev VMs.

- **Live**: https://srtm-lidar-at.exe.xyz:8000/
- **Stack**: Python 3.12 / Flask / gunicorn / Leaflet
- **Dashboard**: `/process.html`  •  **Query UI**: `/query.html`  •  **API docs**: `/api/v1/docs/llm.txt`
- **Cadastre API**: https://cadastre-process-api.exe.xyz/api/v1/docs/llm.txt

## Deep-dive index (load on demand)

| Topic | File | When to read |
|---|---|---|
| `app.py` mental model | `docs/app.md` | touching `app.py` — section map, async task lifecycle, background threads, auth |
| Austria Processor mental model | `docs/austria-processor.md` | touching `austria_processor.py`, KG pipeline, tile checkpoints, GPKG/JSON builders, Zenodo upload |
| Peer Director (multi-VM orchestration) | `docs/peer-director.md` | touching `peer_director.py`, `deploy.sh`, parallel frontiers, HA, throttle, admin token, role eviction |
| Copernicus throttle & retry | `docs/copernicus-throttle.md` | touching `copernicus.py`, `tile_cache.py`, 402 handling, credential rotation |
| Search index | `docs/search-index.md` | touching `search_index.py`, schema, compound query, `kg_parcels`, auto-classification |
| RF training | `docs/rf-training.md` | touching `train_rf_4000kg.py`, ground-truth filters, retraining triggers |
| Zenodo persistent tile cache | `docs/zenodo-cache.md` | touching `zenodo_cache.py`, tile manifest, ZIP indices |
| Cross-cutting concerns | `docs/cross-cutting-concerns.md` | adding object types, RF features, tile grid, credential pool, navigation cheatsheet |
| Planned refactor + speed optimisation | `docs/planned-refactor.md` | next maintenance window work |
| Reference algorithms summary | `docs/reference_algorithms_summary.md` | segmentation/RF internals |

## Quick ops

```bash
sudo systemctl restart srv                # web app (gunicorn + director thread)
sudo systemctl restart rf_train           # RF training job
sudo systemctl kill -s SIGKILL austria_processor && sleep 2 && sudo systemctl start austria_processor
journalctl -u srv -f
tail -f data/austria_processor/logs/processor.log
tail -f /tmp/rf_train_4000kg.log
```

## Services

| Unit | Role | Notes |
|---|---|---|
| `srv.service` | gunicorn (2w × 4t, :8000) + director thread | MemoryMax=3G, on-failure |
| `austria_processor.service` | KG pipeline | **Disabled** on primary (director manages it) and on peers (director starts via API) |
| `rf_train.service` | background RF training (4000 KGs) | on-failure |

## File layout

### Core
| File | ~Lines | Purpose |
|---|---:|---|
| `app.py` | 5900 | Flask API, async tasks, progress, director API, share storage |
| `austria_processor.py` | 5100 | KG pipeline (parent + subprocess). See `docs/austria-processor.md` |
| `peer_director.py` | 770 | Multi-VM orchestration. See `docs/peer-director.md` |
| `object_segmentation.py` | 2200 | Felzenszwalb+RAG → per-object classify (44 features) |
| `learned_classifier.py` | 560 | RF classifier (`FEATURE_KEYS`, cadastre-trained) |
| `static/index.html` | 3100 | Single-file Leaflet UI |
| `static/process.html` | 2100 | Processor + director dashboard. Peer Director list uses unified compact strip (`.peer-card` + `.pm-card`) on desktop+mobile: donut · id (color-coded for attention) · role-tagged bar (FRONTIER/CACHE/PRIMARY/STOPPED/INTERRUPTED…) with KG inside · ⋯ menu · ▸ chevron expands legacy detail. `primary` aliased to `at1`. Sort: elapsed-on-current-KG (oldest first), running peers only. Live Log has range chip cycling `live`/`4h`/`24h` (default `4h` to keep payload small) → `/api/v1/director/log/history?hours=N`; warning filter re-renders from cache. |
| `static/query.html` | 600 | Query Explorer over `/api/v1/query*`, `/feedback` |
| `static/flag.js` | 620 | Flag widget (text-selection chip → `/api/v1/flags/match`) |

### Search & cross-API
| File | Purpose |
|---|---|
| `search_index.py` | SQLite FTS5 + R-tree over 8440 KGs + `kg_parcels`. Auto-rebuild on new JSONs. |
| `cadastre_bridge.py` | Joins cadastre API with landscape index (compound queries, nature scoring) |
| `parcel_compact.py` | Compact per-parcel layout (`frav`, `top_objs`, `top_trees`) + `classify_parcel` (15-class) |

### Data I/O
| File | Purpose |
|---|---|
| `raster_io.py` | Windowed reads from BEV via `/vsicurl/` |
| `ortho_io.py` | BEV orthophoto (RGBI, 47 operates + DOP fallback) |
| `copernicus.py` | openEO: NDVI / WC / SAR / harmonics (4-cred rotation). See `docs/copernicus-throttle.md` |
| `cadastre.py` | Building footprints + parcels |
| `hansen.py` | Hansen GFC (forest change) |
| `osm_features.py` | OSM via Overpass |
| `bev_retry.py` | Backoff + proxy rotation for `rasterio.open()` |
| `tile_index.py` | 55-tile grid, EPSG 4326 ↔ 3035 |
| `tile_cache.py` | Grid-snapped 0.1° caches, Zenodo fallback |
| `zenodo_cache.py` | Persistent tile cache on Zenodo. See `docs/zenodo-cache.md` |
| `zenodo_client.py` | Zenodo API + `Manifest` |
| `gpkg_streamed.py` | Strip-streamed full-GPKG for >100 Mpx KGs |
| `kg_splitter.py` | Splits KGs >28 tiles into directional blocks |
| `geo_parse.py` | KML/GeoJSON/Shapefile/GPX/WKT |

### Feature extraction & training
`terrain_analysis.py`, `temporal_analysis.py`, `texture_features.py`,
`ndvi_harmonics.py`, `train_rf_4000kg.py`, `calibrate.py`.

### Deprecated (kept for reference)
`landscape_classifier.py`, `object_classifier.py`, `scene_adaptive_classifier_patches.py`.

## API surface (high level)

Canonical machine-readable list: `GET /api/v1/docs/llm.txt`.
Groupings (see endpoint comments in `app.py`, search `# === SECTION:`):

- **Analysis**: `/api/v1/segment` (async), `/elevation`, `/terrain`, `/changes`, `/changes/trees`
- **Async tasks**: `/segment/progress|result|abort`, auto-saved as `auto-<task>` share
- **Overlays/exports**: `/segment/overlay`, `/{dtm,lidar,ortho,cir,hansen}/overlay`, `/export/{geopackage,kml,mbtiles}`, `/{lidar,ortho}/geotiff`
- **Shares**: `/shares`, `/share`, `/share/<id>`, `/share/<id>/rename` (1 GB cap, LRU, `data/shares/`)
- **One-stop**: `/api/v1/onestop?bbox=&format=`
- **Search index**: `/api/v1/query`, `/query/parcels`, `/kg/<code>`, `/parcel/<id>`, `/index/{status,rebuild}`
- **Cadastre bridge**: `/lookup`, `/parcels/batch` (ids / cadastre query / **compound**), `/parcels/landscape`, `/query/nature`, `/parcel/<id>/detail`, `/kg/<code>/profile`, `/cadastre/*` proxies
- **Processor (proxied)**: `/processing/{status,start,stop,pause,resume,single,retry,throttle,peers,peers/status,log,manifest,cache_manifest,cache_misses}`
- **Director (admin-token)**: `/director/{status,mode,activate,stop,peers,peers/add,peers/<id>,throttle,proxy/status,proxy/log,update_peers,heartbeat,snapshot,announce,step_down,takeover,handover,identity}`
- **Admin**: `/admin/{update,restart_processor,disable_autostart,run_backfill,backfill_status,diskstat,role_evict,proc_env}`
- **Credentials**: `/credentials` (POST add, list usage), `/credentials/validate`
- **Misc**: `/layers?bbox=`, `/info`, `/parse-geometry`, `/zenodo/lock` (mutex broker)

**Compound query** (the power query) — landscape-first:
`POST /api/v1/parcels/batch {"compound":{...}, "parcel_filters":{...}}`. 70+
numeric min/max filters across terrain, area, buildings, trees, vegetation,
NDVI harmonics, SAR, temporal change, classification quality. Per-parcel filters
cover vegetation, terrain, cadastre, auto_class. See `app.py` `# === SECTION:`
and `cadastre_bridge.py` for the full set.

## Frontend cheatsheet

- `getPostArgs()` → `{ct, body}` for fetch
- `showResultOnMap(data)` → renders features + legend + overlay
- `restoreShareResult(data)` → restore full state incl. overlays
- `clearEverything()` → reset to blank
- `checkLayerAvailability()` → debounced, hides unavailable layers
- State: `lastResult`, `allFeatureData`, `overlays`, `drawnItems`,
  `currentShareId/Name`, `_activeTaskId/_aborted`, `hiddenTypes`, `selectedTypes`
- `localStorage`: active `taskId` (resume after refresh), `srtm_admin_token`
  (admin-token for dashboard mutations, reset via `srtmResetAdminToken()`)

## Analysis pipeline (one-liner)

1. read DTM+DSM (`raster_io`) → 2. ortho/NDVI/SAR/Hansen/cadastre →
3. fused gradient → 4. Felzenszwalb (scale=150) + RAG merge (0.12) →
5. extract 44 features per segment → 6. RF classify (or rule-based) →
7. group adjacent compatible segments (tree→forest, roof→building) →
8. GeoJSON features. Details in `object_segmentation.py`.

## 25 object types (4 letters = man-made when capital)

```
Vegetation:    tree shrub grass hedge
Water:         water
Buildings:     roof greenhouse solar_panel
Infrastructure: fence wall mast (+ wind_turbine, substation)
Transport:     road path parking bridge
Agricultural:  crop orchard vineyard garden
Terrain:       bare_soil rock
Disturbance:   excavation fill tree_loss construction earthwork
```

Letter mapping for compact per-parcel arrays: see `parcel_compact.py`
(lowercase = natural, uppercase = man-made).

## External data sources

| Source | Resolution | Access |
|---|---|---|
| BEV ALS DTM/DSM | 1m, 3 dates (2022/23/24) | HTTP range on remote GeoTIFF |
| BEV DOP RGBI | 0.2m, 47 operates | HTTP range on remote GeoTIFF |
| Sentinel-2 NDVI / WorldCover / S1 SAR | 10m | openEO (4 CDSE creds) |
| Hansen GFC | 30m | `/vsicurl/` UMD |
| Austrian cadastre | mm | REST API |
| OSM | varies | Overpass |

Caches: `/tmp/copernicus_cache/`, `/tmp/hansen_cache/`. Persistent: Zenodo
depo 19650075 (see `docs/zenodo-cache.md`).

## Where to look (debug fast-path)

```bash
# Section markers
grep -n '# === SECTION' app.py austria_processor.py peer_director.py
# Project-wide section index
grep -rn '# ===' *.py | sed 's/# === SECTION: //' | sed 's/ ===//' | column -t -s:
# Color/type sync (must match across files)
grep -rl 'SEGMENT_COLORS' *.py static/*.html
# RF feature list
grep -A60 'FEATURE_KEYS = \[' learned_classifier.py
# Live processor step
cat data/austria_processor/current_step.json | python3 -m json.tool
```

More in `docs/cross-cutting-concerns.md`.

## Critical invariants (read before editing)

- **Object types & colors** are duplicated across `app.py`, `austria_processor.py`,
  `static/index.html` — must stay in sync. See `docs/cross-cutting-concerns.md`.
- **RF feature order** in `learned_classifier.py:FEATURE_KEYS` must match
  `object_segmentation.extract_object_features()`. Changing count invalidates
  the saved `.joblib`.
- **Only one director** runs at a time. Gated by `data/austria_processor/is_director`.
  Single-flight via `director.lock` (fcntl). HA failover in `director_ha.py`.
- **Only one frontier peer** at a time (Copernicus credential safety). Many
  cache-only peers in parallel are fine.
- **All Zenodo writes** serialise through `/api/v1/zenodo/lock` on the primary.
- **`CredentialRotatedError` / `CreditsExhaustedError` / `IPThrottledError`**
  must NEVER be swallowed by generic `except Exception`. See `docs/copernicus-throttle.md`.
- **Admin token** required for mutating endpoints. Loopback exempt. Lives in
  `data/admin_token` (gitignored, mode 0600). Same on all peers.
- **Process uid**: the processor MUST run as `exedev` (uid 1000), not root.
  See `docs/peer-director.md` (Processor uid section).
- **Don't probe Copernicus credentials from the request path.** Only
  `peer_director._refresh_credentials_if_due()` may. See `docs/copernicus-throttle.md`.

## Persistent state files (all `data/austria_processor/` unless noted)

| File | Purpose |
|---|---|
| `progress.json` | Live processor state (dashboard reads this) |
| `current_step.json` | Subprocess→parent IPC |
| `subprocess_warnings.jsonl` | Warning relay |
| `in_progress_kg.txt` | Crash-recovery marker |
| `tile_checkpoints/<kg>/tile_N.pkl` | Per-tile checkpoints (resume-on-retry) |
| `zenodo_manifest.json` | KG product uploads |
| `cache_manifest.json` | Zenodo tile-cache deposit (shared across peers) |
| `zenodo_zip_index/*.json` | Cached central directories of remote ZIPs |
| `failed_kgs.json` / `retry_queue.json` / `deferred_kgs.json` | Queue mgmt |
| `peers.json` / `director_state.json` / `director.lock` / `is_director` | Director |
| `peer_urls.txt` | Peer list for data-sync thread |
| `self.json` | Identity (id, url, director_url) |
| `copernicus_credentials.json` / `copernicus_credential_usage.json` | Creds + telemetry |
| `copernicus_paused` / `openeo_circuit.json` | Throttle state |
| `admin_token` (in `data/`) | Cluster auth |
| `shadow/` + `shadow/meta.json` | HA: staged state for shadow promotion |
| `zenodo_lock_url.txt` | Peer→broker pointer |
| `kg_strikes.json` / `cache_miss_kgs.json` | Reliability tracking |
| `data/search_index.db` | SQLite FTS5+R-tree (~5 MB, auto-rebuild) |
| `data/combined_log_24h.jsonl` | Live 24h merged log ring (pruned every ~10 min). NOT replicated to shadow — rebuilt from peers' own `recent_log` after takeover via `_combined_log_bootstrap_once`. |
| `data/log_archive/YYYY-MM-DD.jsonl.gz` | Long-term per-day archive (full 200-day run). Replicated to shadow via `PUT /api/v1/director/log_archive` on the shadow loop — today-only every hour, full sweep on shadow change, sha256-cached so steady-state traffic is ~0. |
| `data/shares/` | Share storage (1 GB cap, LRU) |

## Conventions for editing

- **Read the relevant `docs/*.md` first** when touching a subsystem. They
  contain hard-won invariants, failure modes, and recovery procedures that
  aren't obvious from the code.
- **Section markers**: every long file (`app.py`, `austria_processor.py`,
  `peer_director.py`) uses `# === SECTION: name ===`. Add markers when
  introducing new logical groupings.
- **Keep this file short.** New deep content → new file in `docs/` and link
  it here. Aim: AGENTS.md fits in a single screen of indexed content.
- **Restart discipline**: changes to `austria_processor.py` need a processor
  restart at the next KG boundary (or kill it). Changes to `app.py` /
  `peer_director.py` need `sudo systemctl restart srv` (the director thread
  reloads automatically — singleton replaced).
- **Push → restart → rollout ordering**. `_LOCAL_GIT_COMMIT` is frozen
  at `srv` import time and is what the director sends to peers as the
  rollout target (see `versions:` / `(target=...)` line in `/process.txt`).
  Commit *before* restarting srv, then `git push origin main`, then
  `sudo systemctl restart srv`. If you restart first and commit after,
  the director keeps pushing the old commit as target until the next
  restart and the fleet never picks up your change. (`_ensure_origin_synced`
  pushes local main to origin every tick — so a forgotten `git push` is
  recoverable, but a forgotten restart is not.)
