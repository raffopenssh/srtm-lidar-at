# Cross-Cutting Concerns (things that span multiple files)

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
cells (1°×2° lat×lon, matching the Zenodo cache bundle layout) to
peers running frontier work in parallel.
* `min_creds_per_frontier` (default 2) in `peers.json` is the FLOOR
  per-peer cred count. The adaptive planner (default ON via
  `adaptive_creds_per_frontier`) overrides it: when the smoothed
  Copernicus sub-factor is healthy it drops to per=1 so all 8 valid
  creds support 8 concurrent frontiers; under throttle it climbs back
  to the floor. See `_target_frontier_count()` and
  `_effective_creds_per_frontier()` in peer_director.py.
* `_max_parallel_frontiers()` = `min(target_frontier_count, len(_austria_cells()))`.
  With 20 Austria cells, cred capacity is the binding ceiling.
* Capability-gated: only peers exposing `cred_subset_env` in
  `/api/v1/info→capabilities` get parallel work. Pre-upgrade peers run
  single-frontier as before (graceful upgrade).
* Pinning: each peer has an optional `pinned_role` in `peers.json`
  (`frontier`, `cache_only`, `idle`). Set via dashboard dropdown or
  `POST /api/v1/director/peers/<id>/pin {pinned_role}`.
* The processor honors `COPERNICUS_CRED_INDICES="0,2"` and
  `KG_CELL_FILTER="[[s,n,w,e],...]"` env vars set by the director when
  starting the subprocess (legacy `KG_LAT_STRIP_FILTER="[[s,n],...]"`
  is still honoured for compat).

**Parallel-frontier ramp — plan-drift restart and cache-only ping-pong:**
The ramp from 1 → N parallel frontiers depends on three pieces of
bookkeeping working together. All three must be correct or the
dashboard plateaus at "running parallel: 1" despite plenty of capacity:
1. `_assign_cred_indices()` preserves the prior plan where it can.
   When the adaptive target rises (e.g. per=2 → per=1), oversized
   prior slices MUST be trimmed to `per_eff` rather than kept.
   Otherwise an early peer hangs on to creds=[0,1] and the leftover
   pool runs short for new peers. Drop indices that overlap with
   already-locked slices; defer under-allocated peers to the
   leftover-distribution pass which can top them up.
2. `_orchestrate_cache_only()` MUST skip both the active frontier AND
   any peer in `parallel_frontiers_active` / `frontier_cred_plan`.
   Otherwise the cache-only orchestrator hard-stops a freshly-promoted
   parallel peer (~18s after promotion) and re-starts it as cache-only,
   then the parallel orchestrator re-promotes the next tick. Net: only
   the active peer ever runs frontier.
3. `_orchestrate_parallel_frontiers()` MUST inspect
   `start_peer_processor()`'s return dict for `error`. Constrained
   starts (cache_only / cred_indices / lat_strips / queue_whitelist)
   skip the systemd fallback (see `is_constrained` branch in
   `start_peer_processor`) and return
   `{'error': 'api_start_failed', 'method': 'no_fallback_constrained'}`
   instead of raising. A peer that 500s on every start (e.g. a wedged
   gunicorn on the peer side) would otherwise hold its cred slice
   forever. Only append to `started` when the call actually succeeded.

**Persistence contract for parallel-frontier state (post-2026-05-06):**
All three of `parallel_frontiers_active`, `frontier_cred_plan`,
`frontier_strip_plan` survive worker / director restarts — they are
written to `director_state.json` by `_orchestrate_parallel_frontiers`
and reloaded by `PeerDirector.__init__`. The single-active-frontier
guard in `_check_and_switch` consults the **union** of
`parallel_frontiers_active` and the keys of `frontier_cred_plan`
before deciding to hard-stop a non-active running frontier. The guard
runs *before* `_orchestrate_parallel_frontiers` each tick, so without
this union a freshly-restarted director would kill every authorised
parallel frontier on its first tick (the original 2026-05-06 cascade:
at43 was active, at22 was an authorised parallel frontier, srv
restarted, the new worker's tick-1 hard-stopped at22 with `only at43
may run frontier`). The cache-only orchestrator already used this
union; the single-active guard now matches. The empty-`_austria_cells`
branch in `_orchestrate_parallel_frontiers` no longer wipes
`parallel_frontiers_active` — that wipe was the trigger for the
cascade and it gains nothing (the set is rebuilt later in the same
function body once cells are available). Transient (rebuilt every
tick, not persisted): `_frontier_restart_log`, `unreachable_count`,
`graceful_stop_sent`, `_target_frontier_count` (persisted, but as
an EMA seed only).

**Parallel-frontier slot grace across director takeover:**
`_orchestrate_parallel_frontiers()` keeps a peer in the authorised
set (`parallel_frontiers_active`) for up to
`UNREACHABLE_FAILOVER_THRESHOLD` (3) consecutive ticks of
`unreachable` instead of dropping it on first miss. Counters live in
`state['parallel_unreachable_count']` (persisted) and reset on the
first reachable poll. Without the grace, a director failover would
drain the parallel-frontier set: peers temporarily unreachable
during the takeover window (announce-flip, gunicorn worker swap,
heavy GPKG build) get omitted from `parallel_frontiers_active` on
the new director's first tick → the single-active guard hard-stops
them the next tick as non-authorised frontiers. The grace mirrors
the existing active-peer path. Plus: `ordered` includes
retained-unreachable peers so their cred/strip plan is reissued
(preserving prior slice) instead of triggering plan-drift restart
when they reappear. Symptom of the unfixed bug: dashboard shows
`5/8 frontiers` after a director swap, recovers only on the next
takeover.

**Plan-drift detection (no peer-side change needed for cred resizing):**
`_orchestrate_parallel_frontiers()` compares the current cred/strip
plan to `frontier_cred_plan` / `frontier_strip_plan` in director_state
and hard-stops any peer whose env vars don't match. The peer's next
start inherits the new `COPERNICUS_CRED_INDICES` / `KG_CELL_FILTER`.
Tile checkpoints survive (cost: one partial tile, ~3-10 min). No peer
code change is needed when the planner downsizes per (per=2 → per=1)
— just wait for the drift restart to fire on the next tick.

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

---

*See `AGENTS.md` for the project map.*


## Natura 2000 bridge (cadastre-process-api)

The cadastre API carries an indexed per-parcel Natura 2000 cache (353 AT
sites, ~896k parcels). We forward the `query` dict verbatim in
`/api/v1/parcels/batch` Mode 2 and `/api/v1/query/nature`, so **no SRTM
code change is needed** to bridge these.

**Filters on cadastre `/query` (passthrough via our Mode 2 `query`):**
- `has_natura2000=true|false` — parcel centroid in any N2K site
- `natura2000_site=<sitecode>` — restrict to one site (e.g. `AT1205A00` Wachau)
- Site types: `A`=Birds Directive (SPA), `B`=Habitats (pSCI/SCI/SAC), `C`=both

**Enrichment automatically present on returned parcels:**
- `in_natura2000` (bool), `natura2000_sites[]` of `{sitecode, sitename, sitetype, site_type_label, area_ha}`

**Direct cadastre endpoints** (use `https://cadastre-process-api.exe.xyz`):
`/api/v1/natura2000/{stats,search,site/<code>,site_parcels/<code>,point,parcel/<kg>/<gnr>,kg/<kg>}`

### Power-query examples (landscape × Natura 2000)

**Rule of thumb:** when expressing "what kind of land is this?", prefer
OUR observed land cover (`compound.type_filters`, `parcel_filters.types`,
`landscape_filters.min_tree_canopy_sqm`, NDVI, `dominant_type`) over the
cadastre's legal landuse code (`landuse=W`, `=LN`, ...). The cadastre tells
you what the parcel is *registered as*; our index tells you what's
*actually growing there* (BEV LiDAR + Sentinel-2 + ortho RGB-I, per-segment
RF confidence). Use cadastre landuse only as a coarse pre-filter or when
the legal designation itself is the thing you're querying for.

```bash
# Actually-forested N2K parcels in Tirol (RF tree conf ≥0.8, canopy ≥2000 m²),
# no buildings, ranked by conservation score. Mode 3 = landscape-first.
curl https://srtm-lidar-at.exe.xyz:8000/api/v1/parcels/batch \
  -H 'Content-Type: application/json' -d '{
    "compound": {"state":"Tirol",
      "type_filters":[{"type":"tree","min_confidence":0.8,"min_area_sqm":800}]},
    "parcel_filters": {"types":["tree"],"min_type_fraction":0.5,
      "min_ndvi":0.5,"cadastre_has_buildings":false,
      "sort":"conservation_score"},
    "query": {"has_natura2000":"true"},
    "cadastre_enrich": true, "limit": 50}'

# All parcels in one Habitats site enriched with our landscape index
curl 'https://srtm-lidar-at.exe.xyz:8000/api/v1/parcels/batch?natura2000_site=AT1205A00&limit=200'

# Steep south-facing alpine N2K grassland candidates (compound → parcels)
curl https://srtm-lidar-at.exe.xyz:8000/api/v1/parcels/batch \
  -H 'Content-Type: application/json' -d '{
    "compound": {"min_slope":20,"aspect":["S","SW","SE"],"min_elevation":1500,
      "type_filters":[{"type":"grass","min_confidence":0.7,"min_area_sqm":500}]},
    "parcel_filters": {"cadastre_has_buildings":false,"sort":"conservation_score"},
    "query": {"has_natura2000":"true"}, "cadastre_enrich": true, "limit": 100}'

# Cadastre says forest (W) but we DON'T see trees — deforestation / mis-class hunt
curl https://srtm-lidar-at.exe.xyz:8000/api/v1/parcels/batch \
  -H 'Content-Type: application/json' -d '{
    "query": {"has_natura2000":"true","landuse":"W"},
    "landscape_filters": {"max_tree_canopy_sqm":100,"sort":"conservation_score"},
    "limit": 50}'

# Edge cases: legally named in law but OUTSIDE polygon (RIS only)
curl 'https://srtm-lidar-at.exe.xyz:8000/api/v1/parcels/batch?has_natura2000=false&has_legal_refs=true&legal_context=nature_protection&limit=50'
```

**Performance**: cadastre's N2K filter is an indexed SQLite cache — typically
<100 ms even for hundreds of thousands of matches. Aggregate stats are
skipped by default when an N2K filter is active; pass `with_stats=true` to force.

## KG runtime limits (current state, May 2026)

`KG_TIMEOUT_SECONDS = 16 h` (first attempt) / `KG_RETRY_TIMEOUT_SECONDS = 12 h`
(retry). No timeout-induced failures in the warning archive. Long-running
KGs (e.g. `gpkg_full` at 11h) finish well inside the wall. Dominant warning
pattern is **openEO 503 / NDVI read-timeout** (transient upstream issues,
already handled with month-skip). **No need to extend KG runtime limits.**
Revisit only if `Maximum processing time ... exceeded` warnings appear in
`process.txt?warn=1`.
