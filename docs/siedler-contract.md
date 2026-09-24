# Siedler / sibling-service contract (`llm_api.py`)

Implements the srtm items of https://siedler-oesterreich.exe.xyz:8000/llm/ahead
(check: `…/llm/ahead/check/srtm`) and the shared per-KG spec from
`cadastre-process-api.exe.xyz/api/v1/docs/llm.txt?section=integration`.

**Design rule: everything answers from `data/search_index.db`.** No KG JSON,
no GPKG, no Zenodo/BEV fetch in the request path — the primary is
bandwidth-metered and the game polls per viewport tile. "Warm" == ingested
into the index; the only cold state is a KG whose `_json` is committed on
Zenodo but not yet ingested (`ready:false`, `retry_after_s:300`,
`pending_kgs`). `/llm/kg/<code>` returns 202 in that state.

| Item | Endpoint | Backing |
|---|---|---|
| ALL-1/1b | `GET /llm/kg/<code>[.json]` (`?parcels=1` embeds slim parcels) | `kg`, `kg_landcover`, `kg_hansen` |
| ALL-2 | `GET /llm/manifest.json`, `GET /llm/kgs[?codes=]` | `kg`, `index_meta` |
| ALL-3 | `GET /llm.txt` (root, text/plain, ~3 KB) | constant |
| ALL-4 | `POST /api/v1/prewarm?kgs=a,b` (≤50) | index state; triggers slim backfill + footprint match |
| LID-1 | `GET /api/v1/query/parcels?bbox=` gains `fracs`, `dom_terrain`, `tree_h`, `slope_deg`, `elev_m`, `aspect_deg`, `top_trees` (h≥25), `ready`, `kgs` | `kg_parcels.frav/tree_h_*`, `kg_trees` |
| LID-2 | `GET /api/v1/trees/bbox` | `kg_trees` |
| LID-3 | `GET /api/v1/query/buildings?bbox=` with `footprint_id` | `kg_buildings.footprint_id` via `building_footprint_match.py` |
| LID-4 | hillshade/nDSM XYZ tiles | **not implemented** (pre-rendering z12–17 for Austria = TBs + BEV egress; grid25 exists per KG if ever needed) |
| LID-5 | unauthenticated low-priority queue | **won't do** — fleet processes all 8440 KGs anyway |
| LID-6 | light artefact split | n/a — the v2 JSON blob already lands on the primary and is indexed; cold path is index ingest, not a download |

## Disk budget (the constraint)

Added to `search_index.db` (3.26 M parcels): `kg_parcels.frav` (~14 B/row
compact JSON `{letter: area_sqm}`), `tree_h_mean/max` (2 REAL),
`kg_trees` (≤5 tallest apices per parcel, **h ≥ 20 m only** →
`KG_TREES_MIN_H`, ~1.1 M rows) + one `(kg_code, lon)` index,
`kg_buildings.footprint_id` (~15 B/row). Total ≈ 200–250 MB. No parcel
R-tree: bbox queries go `kg_rtree` → KG codes → `p.kg_code IN (…)` +
centroid range (the old `JOIN kg` bbox predicate made SQLite SCAN the
parcel table on the sort index → 13 s/call).

## Backfill / migration

`_migrate()` adds the columns; `_enrich_kg` fills them for every new
ingest (named-column INSERT — `KG_PARCELS_COLS`). Legacy rows are filled
by `SearchIndex().backfill_slim_fields()` (UPDATE in place, ~0.2 s/KG,
idempotent — skips KGs that already carry `frav`). Run once after deploy
(2026-09-24: 2562 KGs in ~9 min, tmux `slimfill`). `prewarm` also
backfills any KG it finds with all-NULL `frav`.

`kg_trees` semantics: **sampled** (top-5 per cadastre parcel from the KG
JSON's `top_trees`), not the ~40k/KG apex layer in the light GPKG. Every
response says so (`source`, `complete:false`). `crown_d_m` is only set
when the tree segment is single-crown sized (≤400 m²).

## Footprint matching (LID-3)

Our `building_footprints` originate from the same BEV cadastre footprints
the cadastre API serves → centroids agree to ~0.2 m. `match_kg` pulls
`/spatial/footprints?bbox=<kg bbox>&geometry=0` (~1.4 MB/KG, once per
30 d), nearest-centroid within `max(3 m, 0.6·√area)` + area ratio 0.3–3,
writes `footprint_id` and a `kg_fp_match` marker. Single worker thread,
≤60 KGs/h (bandwidth guard). First call for an unmatched KG returns
`footprint_id:null` + `footprint_match:{kg:"warming"}`.

## CORS

`app._cors_preflight` / `_cors_public` add `*` on `llm_api.CORS_PREFIXES`
(and answer OPTIONS 204 before the admin-token gate). ETag is added for
sub-1 KiB bodies too (the generic hook only tags ≥1 KiB).
