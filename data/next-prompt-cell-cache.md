# Cell-grid Zenodo cache (replace 0.5° lat strips with 1° lat × 2° lon cells)

## Why

The Zenodo tile-cache deposit (depo 19650075) currently bundles tiles into
**latitude-only 0.5° strips**. The Peer Director uses these strips as a
write-coordination boundary: each frontier peer is pinned to one or more
disjoint lat strips so two peers never write the same Zenodo ZIP.

Result: with 7 strips covering Austria,
`max_parallel_frontiers = min(valid_creds=8, lat_strips=7) = 7`. Throttle
trims to 6. We have 8 valid Copernicus credentials sitting partially idle.

Several strips span huge longitudes:

| Strip | Lon span | KGs |
|-------|----------|-----|
| 46.5–47.0 | 9.87–16.29 (6.41°) | 1399 |
| 47.0–47.5 | 9.53–16.67 (7.13°) | 1621 |
| 47.5–48.0 | 9.68–17.12 (7.43°) | 1039 |
| 48.0–48.5 | 12.75–17.16 (4.41°) | 2591 |
| 48.5–49.0 | 13.44–16.95 (3.51°) | 1160 |
| 46.0–46.5 | 14.13–14.71 (0.58°) | 14 |
| 49.0–49.5 | (empty/tiny) | small |

If we split into **1° lat × 2° lon cells** we get ~12–14 non-empty cells
covering Austria — well above 8 creds — so credentials become the only
binding constraint. Throughput ceiling ~doubles for the same fleet.

Current strip ZIPs on Zenodo (depo 19650075, total 700 MB):

```
copernicus_harmonics_strip_46.5_47.0.zip   24 MB
copernicus_harmonics_strip_47.0_47.5.zip   75 MB
copernicus_harmonics_strip_47.5_48.0.zip  124 MB
copernicus_harmonics_strip_48.0_48.5.zip   80 MB
copernicus_harmonics_strip_48.5_49.0.zip   23 MB
copernicus_ndvi_strip_*.zip               5–42 MB each
copernicus_sar_strip_*.zip                6–85 MB each
copernicus_worldcover_strip_*.zip         0.1–2 MB each
hansen_strip_*.zip                        1–6 MB each
```

## Goal

1. **Add a longitude axis to the bundle key** — switch from
   `_strip_<S>_<N>.zip` to `_cell_<latS>_<latN>_<lonW>_<lonE>.zip` with
   1° lat × 2° lon (tunable via `STRIP_HEIGHT`/`STRIP_WIDTH` constants).
2. **Migrate existing 700 MB of cached tiles** into the new layout —
   re-bundle by reading the per-NPZ grid coords from each tile's bbox
   (the entry name encodes `s_w_n_e`), partitioning into the new cells,
   and uploading new ZIPs. Delete the old strip ZIPs from the deposit
   when the new ones land. Plan the migration so it's restart-safe and
   never corrupts the deposit.
3. **Update the Peer Director** to plan disjoint cell sets per frontier
   (currently disjoint lat strips). The existing peer-to-strip
   assignment logic in `peer_director.py` (`_assign_lat_strips`,
   `frontier_strip_plan`, `_austria_lat_strips`,
   `_max_parallel_frontiers`) becomes peer-to-cell.
4. **Update the processor's strip-filter env**
   (`KG_LAT_STRIP_FILTER=[[s,n],...]`) to support cells
   (`KG_BBOX_FILTER=[[s,n,w,e],...]` or extend the existing one).
5. **Update the cache-ready intersection logic**
   (`_compute_cache_ready_kgs` in `peer_director.py`,
   `is_kg_fully_cached` in `tile_cache.py`) to consume cell-keyed
   manifest entries.

## Design constraints

- **Backward compat during migration**: readers must accept both old
  strip names and new cell names until the migration completes. After
  migration, drop strip support and delete old ZIPs.
- **NPZ entry names inside ZIPs already encode bbox**
  (`{product}_{s:.4f}_{w:.4f}_{n:.4f}_{e:.4f}{_year?}.npz`) — no entry
  rename needed; only the ZIP container changes.
- **Single Zenodo deposit, atomic per-ZIP writes**: same upload mutex
  applies. The migration may need to disable the existing
  upload-throttle to push many new ZIPs in one go.
- **Cell size choice**: 1° × 2° gives ~14 non-empty cells across
  Austria. 0.5° × 1° would give ~28 (too granular — ZIPs become tiny
  and per-ZIP overhead dominates). 1° × 2° is a good first cut. Make
  it configurable via `STRIP_HEIGHT`/`STRIP_WIDTH` constants in
  `zenodo_cache.py`.
- **Empty cells**: don't create ZIP files for cells with no tiles.
  Manifest reflects only non-empty entries.
- **Cache-only peers must not be starved**: the cache-ready filter
  needs to refresh promptly after migration so cache-only fleet
  resumes work against new cell ZIPs.

## Files to read before starting

- `zenodo_cache.py` — bundle naming + per-strip upload/download
  (`_zip_filename`, `_lat_strips`, `_strip_for_lat`,
  `_npz_entry_name`, `upload_all`, `_build_reverse_index`, the cache
  manifest writer)
- `tile_cache.py` — local NPZ tiles + `is_kg_fully_cached` +
  `CacheMissError` flow
- `peer_director.py` — `_austria_lat_strips`, `_assign_lat_strips`,
  `_max_parallel_frontiers`, `_orchestrate_parallel_frontiers`,
  `_compute_cache_ready_kgs`, the per-peer env builder that produces
  `KG_LAT_STRIP_FILTER`
- `austria_processor.py` — the `KG_LAT_STRIP_FILTER` reader near the
  top of `main()` (search for `KG_LAT_STRIP_FILTER`), and the strip
  re-application after KG block expansion
- `app.py` (`/api/v1/processing/cache_manifest`) — the manifest sync
  endpoint, must accept old + new key formats during migration
- `AGENTS.md` — sections **Per-peer credential & lat-strip dedication**,
  **Zenodo Persistent Cache**, **Cache-only peers**

## Suggested approach

### Phase 1 — additive cell layout, dual-read

1. Add `STRIP_WIDTH = 2.0` (lon) and `STRIP_HEIGHT = 1.0` (lat) to
   `zenodo_cache.py`. Replace `_lat_strips()` with `_lat_lon_cells()`
   returning `[(s, n, w, e), ...]`.
2. New `_cell_for_bbox(lat, lon) -> (s, n, w, e)` helper.
3. Extend `_zip_filename(product, *cell)` to accept either
   `(s, n)` (old) or `(s, n, w, e)` (new); emit old format only behind
   a feature flag for migration. Default = new format.
4. Update `upload_all` and `fetch_*` to consume the new keys. Keep a
   compatibility shim that recognises old strip ZIPs in remote
   manifests (treat as a single cell spanning all lon for that lat
   strip).
5. Update Peer Director's strip plumbing: rename to cell plumbing.
   Frontier peers receive `cell_filter` env (list of
   `[s, n, w, e]`). KG centroid must fall inside one of the cells.
   Cells partition cleanly by id; assignment is round-robin like
   strips.
6. Update `austria_processor.py` `KG_LAT_STRIP_FILTER` reader to
   accept the new 4-tuple format (rename env to `KG_CELL_FILTER`,
   keep old name as alias for backward compat during rolling
   restart).
7. Frontiers keep working against old strip ZIPs (compat shim);
   new tiles fetched go into new cell ZIPs.

### Phase 2 — migration of existing 700 MB

1. Write a one-shot script `migrate_strips_to_cells.py` that:
   - Lists all ZIPs in the deposit via Zenodo API
   - For each strip ZIP: download, read its NPZ entries, partition by
     `(lat_centroid, lon_centroid)` into target cells, write new
     cell-named ZIPs locally
   - Upload new ZIPs to the deposit (under the upload mutex)
   - Refresh `cache_manifest.json` with new entries
   - Delete old strip ZIPs after verifying new ZIPs land
2. Run the script on the primary while the director is paused
   (`mode=paused`). Resume after migration.
3. Validate: every old NPZ entry appears in exactly one new cell ZIP.
   Spot-check by downloading a couple of NPZs through the new
   `fetch_*` path.

### Phase 3 — drop strip code

1. Remove the strip→cell compat shim (`_lat_strips`, strip parser).
2. Update AGENTS.md sections about strips → cells.

## Validation plan

- `python3 zenodo_cache.py status` should show ~14–28 cell ZIPs
  totalling roughly the same size as the 27 strip ZIPs (overhead
  dominates only for tiny products like worldcover).
- After migration, restart director and confirm
  `max_parallel_frontiers` rises to 8 (cred-bound rather than
  cell-bound); 8 frontiers actually start with disjoint cell
  assignments.
- Process one test KG end-to-end on each frontier — no
  `CacheMissError` for cells already covered.
- Verify `_compute_cache_ready_kgs` finds the same number of
  cache-ready KGs as before (the underlying tiles haven't changed,
  just their grouping).

## Out of scope

- Changing the per-NPZ grid step (`COP_STEP=0.1°`,
  `HANSEN_STEP=0.5°`). Tile granularity is unchanged.
- Switching deposits. Same depo 19650075 holds both strip and cell
  ZIPs during migration.
- The `austria_processor` block-splitter (`kg_splitter.py`) — that's
  orthogonal.

## Risk register

- **Half-migrated state**: if the script crashes after deleting old
  strip ZIPs but before all new cell ZIPs are uploaded, we lose
  cached tiles. Mitigation: never delete an old ZIP until *all* new
  ZIPs derived from it are confirmed uploaded + verified by
  central-directory readback.
- **Manifest divergence across peers**: the cache manifest is
  push/pull synced. After migration, primary's manifest is the truth.
  Force a manifest broadcast to all peers (existing
  `cache_manifest_sync` thread) before resuming the director.
- **Stale `zenodo_zip_index/` caches** on each peer: they cache
  central directories of remote ZIPs by URL hash. After migration,
  every peer needs `rm -rf data/austria_processor/zenodo_zip_index/`.
  Bake that into the migration script as a remote `POST /admin/...`
  helper, or document it as a manual step.
- **Concurrency safety during migration**: pause director
  (`mode=paused`) so no peer is writing tile uploads while migration
  reads + rewrites the deposit. The single-active-frontier invariant
  is normally enough, but the migration writes many ZIPs across all
  cells simultaneously — no frontier should be mid-upload.

## Acceptance criteria

- 8 frontiers can run concurrently in steady state (subject to throttle)
- `cache_manifest.json` contains only cell-keyed entries after migration
- No KG re-processing required
- Existing tile cache is preserved (every NPZ still fetchable through
  the new path)
- The Peer Director's `frontier_strip_plan` is renamed to
  `frontier_cell_plan` and exposes cells in `/director/status` for
  the dashboard

## Appendix — current state on disk

```
cache_manifest.json: depo_id=19650075, 27 ZIP files, 700 MB total
Latitude strips covering Austria: [(46.0, 46.5), (46.5, 47.0),
  (47.0, 47.5), (47.5, 48.0), (48.0, 48.5), (48.5, 49.0),
  (49.0, 49.5)]  # 7 strips
Proposed cells (1° lat × 2° lon, AT_WEST=9, AT_EAST=17.5):
  ~12–14 non-empty cells
Valid Copernicus credentials: 8 (all healthy)
Max parallel frontiers today: min(creds, strips)=7 → throttled to 6
Max parallel frontiers after migration: cred-bound = 8
```

## Reading order

1. AGENTS.md → "Per-peer credential & lat-strip dedication" + "Zenodo
   Persistent Cache" sections — get the mental model first.
2. `zenodo_cache.py` top 200 lines (constants + naming helpers).
3. `peer_director.py` lines around `_austria_lat_strips`,
   `_assign_lat_strips`, `_orchestrate_parallel_frontiers`.
4. `tile_cache.py` `is_kg_fully_cached` + `CacheMissError`.
5. Then start drafting Phase 1.
