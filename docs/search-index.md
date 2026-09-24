# Search Index — Mental Model

SQLite at `data/search_index.db` (~2.5 GB with 3.3 M parcel rows — the
`~5 MB` of early days is long gone; watch disk when adding columns). FTS5 +
R-tree + ~30 b-tree indexes.
All queries <50 ms. Auto-rebuilt at startup; auto-updated when new KG JSONs
appear (60 s poll). Manual rebuild: `POST /api/v1/index/rebuild`.

## Tables (overview)

```
kg                       ← one row per KG (parent KGs only)
├─ kg_landcover          ← per-type area summary
├─ kg_hansen             ← forest loss by year
├─ kg_classification     ← per-type stats (count, height, conf)
├─ kg_divergence         ← RF↔rule disagreement details
├─ kg_type_top           ← top-N segments per type per KG (for ranking)
├─ kg_buildings          ← per-building rollup (height, stories, roof)
├─ kg_parcels            ← per-parcel rollup (auto_class, terrain, buildings, hansen,
│                           frav / tree_h_* slim fields — see docs/siedler-contract.md)
└─ kg_trees              ← ≤5 tallest apices per parcel, h ≥ 20 m (siedler LID-2)

kg_rtree (virtual)       ← spatial index over kg.bbox
fts_kg (virtual)         ← full-text over name/state/district/gemeinde
index_meta               ← build timestamp, kg_count
```

## How a KG record is built (`_enrich_kg`)

```
data/austria_processor/json/<kg>.json   (or block aggregator)
      │
      │ _enrich_kg(c, code, data)
      │
      ├─→ kg row             (45+ columns: terrain, buildings, NDVI, SAR,
      │                       Hansen totals, RF stats, dominant_type, ...)
      ├─→ kg_landcover       (one row per object_type)
      ├─→ kg_hansen          (one row per loss_year)
      ├─→ kg_classification (per-type confidence + height stats)
      ├─→ kg_type_top       (top-N segments by area / volume / height / conf)
      ├─→ kg_buildings      (vertex_heights → polygon → stories/height/roof_type)
      ├─→ kg_parcels        (per-parcel terrain + frav-derived stats)
      │                       └─ building rollup via point-in-polygon
      │                          on `vertex_heights`, with centroid fallback
      │                       └─ auto_class via parcel_compact.classify_parcel
      └─→ fts_kg, kg_rtree   (one row each)
```

For split KGs (e.g. `49006-south`, `49006-north`): JSONs land per-block.
`_enrich_kg_from_blocks(parent_code, blocks)` aggregates by KG:
- weighted-average for area-weighted scalars
- sum for additive counts
- min/max for extrema
- union of nested arrays (top_10_*, top_by_type)

The parent code (`49006`) is the only thing visible to the API. Block codes
appear only in JSON filenames + Zenodo entries.

### File selection (`_select_kg_files_for_parent`)

Handles the messy reality of split + maybe-split retries. For each
parent KG it picks the freshest set of on-disk files using the
**Zenodo manifest's `uploaded_at`** as the source of truth (falls back
to `generated_at`, then mtime). Rules:

1. Files whose `<code>_error` manifest entry is newer than `<code>_json`
   are discarded (failed runs leave stale JSONs on disk).
2. If only the **plain** side has a committed `_json` upload, use the
   plain file.
3. If only the **block** side has committed uploads, use the blocks.
4. Otherwise pick whichever side has the most recent timestamp; the
   other side is dropped wholesale to avoid double-counting the same
   spatial region.

`/api/v1/kg/<code>` calls `merged_kg_json(...)` which uses the same
selector. For split KGs (no plain `<code>.json`) it returns the merged
JSON-shape dict so the dashboard sees `parcels.count`, `landscape.n_segments`,
`tree_stats`, `top_10_objects`, etc. — not just the flat index row.

### Surgical reindex / backfill endpoints

```
POST /api/v1/admin/reindex_split_kgs           # re-enrich every parent
                                                # with split / maybe-split
                                                # block files on disk
POST /api/v1/admin/backfill_jsons_from_manifest # download missing _json
                                                # uploads from Zenodo (skips
                                                # tombstoned + errored)
   body: {limit?: 200, codes?: [...], dry_run?: false}
```

## Auto-classification (per parcel, at index build)

`parcel_compact.classify_parcel(p)` runs once per parcel. Reads frav,
terrain (slope/tri/elev), top_trees mean height, building rollup, Hansen
recent-5yr loss → emits `auto_class` + `auto_subclass` + confidence.

15-class taxonomy (with subclasses):
```
forest (tall_forest | recently_thinned)
young_forest (regenerating)
wooded (open_woodland)
meadow (rugged | orchard_meadow)
alpine_meadow (pasture | high_alpine)
cropland (fallow)
vineyard
orchard
shrubland
built_up (apartments | multi_storey | house | dense)
farmstead (with_house)
infrastructure (road | mixed)
water_body
disturbance (recent_clearfell | tree_loss | construction | earthwork)
bare
mixed
```

Available as filters: `auto_class=`, `auto_subclass=`, `min_auto_class_confidence=`.

## Query taxonomy

### Single-KG / single-parcel (point lookup)
```
query_kg(code) → KG summary + Zenodo links
query_parcel(parcel_id) → parcel detail (lazy-loads from JSON)
```

### Spatial (R-tree)
```
query_bbox(min_lon, min_lat, max_lon, max_lat, ...)
query_point(lon, lat, radius_km)
```

### Administrative
```
query_admin(level='state'|'district'|'gemeinde', code= or name=)
aggregate_state / aggregate_district / aggregate_gemeinde / aggregate_country
```

### Text (FTS5)
```
query_text(q, limit, offset)   — FTS over name/admin
```

### Type-driven
```
query_type_ranking(object_type, metric='area'|'volume'|'height'|'rf_conf')
query_high_confidence_type(object_type, min_confidence=0.7)
query_low_confidence(max_confidence=0.5)
query_confidence_ranking(order='asc')
query_type_confidence(object_type)
```

### Domain-specific
```
query_hansen_loss(year_from, year_to, min_loss)
query_new_buildings(min_count)
query_processed(...)
query_divergence(min_pct, rf_type, final_type)
query_divergence_pairs()
```

### Per-parcel (`kg_parcels` table)
```
query_parcels_index(kg_code=, terrain_class=, dominant_type=,
                    aspect=, auto_class=, has_buildings=,
                    min_*=, max_*=, sort=, sort_dir=)
```
All filters are SQL b-tree-backed. <25 ms even on 1.6M parcels.

### Per-building (`kg_buildings` table)
```
query_buildings_index(min_height, max_height, roof_type, min_stories, ...)
```

### Compound (the power query)
```
query_compound(filters, limit, offset)
```
Accepts ~70 numeric min/max filters spanning terrain, area, buildings,
trees, vegetation, NDVI harmonics, SAR, temporal change, classification
quality. Used by `/api/v1/parcels/batch {compound: ...}` to start from
landscape index, find KGs, expand to parcels via JSONs, then enrich
via cadastre.

### Top features / ranking
```
query_top_features(feature_type, object_type=, ...)
     — buildings, new_buildings, infrastructure, segments, segment_points
query_segments(object_type=, min_rf_confidence=, ...)
```

## Schema mutation safety

`_migrate()` runs on every open. Adds missing columns idempotently using
`ALTER TABLE … ADD COLUMN …` guarded by PRAGMA introspection. To add a
new column:
1. Add it to the `CREATE TABLE` in `_schema_stmts`.
2. Add an `ALTER TABLE` block in `_migrate()`.
3. Populate it in `_enrich_kg` (or `_enrich_kg_from_blocks`).
4. Optionally add a `CREATE INDEX IF NOT EXISTS`.
5. Force rebuild: `python3 -c "from search_index import SearchIndex; SearchIndex().build()"`

Never drop a column — add a new one and migrate readers.

## GPKG cache (lazy detail loading)

`GpkgCache` (bottom of file) downloads light GPKGs from Zenodo on demand
for per-KG detail endpoints. Stored at `data/gpkg_cache/` (files
`<code>_<variant>.gpkg`; variant `light_v2` is the v2.x light GPKG shared
with the v3 tree service). LRU eviction at `GPKG_CACHE_MAX_BYTES`.
`_query_gpkg_layer(...)` reads via fiona/pyogrio.

`_resolve_gpkg(kg, variant, wait, prefer_v1)` resolves local processor
output → cache → Zenodo, taking the download link + byte size + md5 from
the **manifest** (`<code>_light_gpkg_v2` first, then `_light_gpkg`;
`prefer_v1` for layers only v1 has, e.g. `new_buildings`) and falling back
to the index's `zenodo_*_gpkg_url`. Auth is added by the fetch layer
(both `/api/files/<bucket>` and `/api/records/<id>/draft/...` are
owner-only — the old code 403'd on the latter and surfaced as 404).

### Downloads go through `zenodo_fetch.py` (progress-aware)

All request-path Zenodo product downloads use `zenodo_fetch.download()`:

* **Progress**: bytes done/total, EMA MB/s, ETA, attempt, status
  (`queued|connecting|downloading|verifying|done|error`). Per-thread hook
  (`zenodo_fetch.set_hook`) is bound by the async task workers
  (`app._bind_fetch_progress`) so any download inside a task lands in the
  task's progress file (`fetch` + `progress` fields of
  `/api/v1/segment/progress`).
* **Single-flight**: in-process waiters share one `FetchState`; across
  gunicorn workers an `fcntl` lock on `<dest>.lock` makes the sibling
  *relay* the sidecar progress (`<cache>/.fetch/<file>.json`) instead of
  downloading a second copy.
* **Robust to slow Zenodo**: 60 s per-read stall timeout, 4 attempts with
  HTTP `Range` resume from the partial `.tmp`, 429/503 `Retry-After`
  honoured, 4xx fail fast, size + md5 verified before the atomic rename.
  ≤ `MAX_PARALLEL=4` concurrent downloads per process.
* **Bounded blocking**: `download(wait=S)` raises `FetchInProgress(state)`
  after S seconds while the download continues in the background. The
  sync `GET /api/v1/kg/<code>/{buildings,new_buildings,infrastructure,
  segments,layers}` endpoints use this (`?wait=`, default 25 s) to answer
  `202 Accepted` + `Retry-After` + live state; the same URL returns 200
  once the file is local.
* **Parallel prefetch**: `prefetch_many(items, hook)` runs several
  downloads at once and folds them into one aggregate (`k/N files, X/Y
  MB, rate, ETA`) — `trees_v3.prefetch_gpkgs` pulls every covering GPKG
  of an AOI concurrently before reading apices.
* **Observability**: `GET /api/v1/zenodo/fetches` (in-flight from both
  workers via sidecars, recent completions from the shared
  `.fetch/_recent.json` ring, `stats.slow` = recent-hour median < 1 MB/s)
  and the `zenodo_fetch:` line in `/process.txt` (only printed when there
  is activity; flag `ZENODO-SLOW`).

Debug:
```bash
curl -s localhost:8000/api/v1/zenodo/fetches | jq .text
curl -s -i 'localhost:8000/api/v1/kg/<code>/buildings?wait=0&limit=1' | head -20   # 202 while cold
ls -la data/gpkg_cache/.fetch/            # live sidecars + _recent.json
```

## Where to look

```bash
# Schema
grep -n 'CREATE TABLE\|CREATE VIRTUAL\|CREATE INDEX' search_index.py

# All query methods
grep -n '    def query_\|    def aggregate_' search_index.py

# Inspect index live
sqlite3 data/search_index.db '.schema kg_parcels'
sqlite3 data/search_index.db 'select count(*) from kg_parcels group by auto_class;'
```

---

*See `AGENTS.md` for the project map.*
