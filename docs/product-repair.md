# Repairable layer gaps — `product_repair.py` (2026-09-23)

Load when a KG product is missing a *cache-fillable* layer (NDVI harmonics,
Copernicus NDVI / WorldCover / SAR, Hansen) and you want it fixed without a
from-scratch rerun, or when adding a new layer that comes from the shared
Zenodo tile cache.

## Why

Two independent gaps surfaced the same week:

* **openEO read-timeouts (09-22/23)** — the 240 s sync cap produced an error no
  breaker matched; ~200 tiles were baked with zero NDVI months → no harmonics
  (fixed upstream in `72f0585`, quadrant split).
* **v2 upgrades lose harmonics** — `v2_source._cache_only_fetch` can only take
  harmonics from the Zenodo tile cache (never openEO), and that cache holds ~40
  harmonics cells for all of Austria. The one-shot scan found **839 v2.3
  products / 147 v1 products (986 of 3548 docs) partial**, 948 of them
  harmonics-only, ~840 of them written 09-21…09-23 by the 2.3 re-upgrade wave.

Both are "the tile cache lacked a cell at bake time"; both are repaired by
(1) getting the cell into the cache and (2) re-upgrading from the KG's own
full GPKG. Nothing else about the product is wrong, so **no tombstone, no
requeue, no product deletion** — the partial product stays live and served
until the clean one replaces it in place.

## Mechanism (4 pieces)

| piece | where | what |
|---|---|---|
| **Label** | producer: `austria_processor._product_upload_version(fk, defects)`; `result["product_defects"]` set right after `compute_data_quality` | JSON manifest entries (`_json`, `_json_v2`) get `version = "<ver>-partial"` (`v2.3-partial`, `v1-partial`). GPKGs keep the plain label. `v21_products.v2_products_complete` compares `== MANIFEST_VERSION`, so a partial pair reads as *not current* → the normal re-upgrade machinery owns the fix. `base_version()` strips the suffix wherever the plain label is needed (`trees_v3.product_entry`, `search_index._version_key/_product_version`). |
| **Registry** | `data/austria_processor/product_repair.json` `{code: {defects, tiles, bbox, product_ts, version, attempts, state}}` | Fed by `observe()` at the two primary ingest points — fresh v1 JSON (`search_index.update_kg` → `_repair_observe`) and `_json_v2` (`v2_ingest`) — plus the one-shot corpus scan. Idempotent per `product_ts`. A clean product → `done`. |
| **Fixable gate** | `product_repair.fixable(code)` / `missing_cells(entry)`; used in `peer_director._compute_v2_upgrade_candidates` | Asks the Zenodo ZIP index (`CopernicusTileCache.has_cached(local_ok=False)`, `HansenTileCache.has_cached`) whether **every** missing product is cached for **every** 0.1° cell of the bbox. A partial code is ranked as an upgrade candidate only when fixable (else `n_partial_wait`). |
| **Cell fill** | director `_product_repair_plan` → `prewarm.repair_cells` + `prewarm.repair_codes` in `cache_manifest.json`; frontier `austria_processor._repair_cells_fill` (runs from `prewarm_cell_tiles` at every frontier KG end) | `sweep()` every 15 min (`PRODUCT_REPAIR_SWEEP_S`) refreshes states and emits up to 40 `(cell, products)` requests (`PRODUCT_REPAIR_MAX_CELLS`), oldest defects first. Cells are published only while Copernicus EMA is healthy **and** `app._openeo_health()` saw no read-timeouts / cascades in the last hour. Each frontier fetches ≤ `PREWARM_REPAIR_CELLS_PER_KG` (4) per KG end, only the listed products, skips cells another frontier already filled, then flushes to Zenodo. Same transport as the pre-warm flag: no restart, no endpoint, no new traffic. |

**Peer-side override.** Peers decide eligibility on *their* manifest copy
(`v2_upgrade_eligible`). Producer-labelled entries agree fleet-wide, but the
one-shot relabel on the primary never reaches peers (equal `uploaded_at` →
merge is a no-op), so `repair_codes` (the currently `fixable` codes, ≤300)
rides the same block and `_repair_override_codes()` lets those through even
when the peer's copy says "current v2.3".

## States and loop guard

`pending` (cells missing) → `fixable` (all cached; director event
`repair <code>: all missing cells cached — eligible for re-upgrade`) → `done`
(a clean product was ingested) or `stuck`. A re-upgrade that ran while the
code was `fixable` and still came back partial bumps `attempts`;
`MAX_ATTEMPTS = 2` → `stuck` (parked, shown in `/process.txt`, never
re-dispatched). Cells that vanish again (tombstoned ZIP) drop the code back to
`pending`. `done` entries are pruned after 7 days.

## `/process.txt`

```
repair:   pending=N fixable=F stuck=S done=D · by_defect[harmonics=… sar=…] (…) · <code>@<state>[<products>/<tiles>t/<missing cells>c/a<attempts>] … · plan: cells=C codes=K · debug: ?q=repair
```
`products:` / `v2:` lines count `v2.3-partial` as its own version bucket
(`stale_v2`, "re-upgrade pending") — that is the intended reading. Frontier
log lines: `repair: filled cell 14.00,47.50 harmonics`. Director events:
`?q=repair`.

## Ops

```bash
python3 product_repair.py                       # repair: line + JSON summary
python3 product_repair.py scan --list           # dry run over json/ + kg_v2_store (~5 min, 19 GB of JSON)
python3 product_repair.py scan --apply          # record + relabel primary manifest entries '<ver>-partial'
python3 product_repair.py sweep                 # one director sweep now (states + cell plan)
python3 product_repair.py reset 12029 45311-south   # stuck → pending, attempts=0
curl -s 'localhost:8000/process.txt?q=repair&log=200&hours=48'
```
Run `scan --apply` again only when a new defect class is added to
`REPAIRABLE_LAYERS`, or after a bug fix that changes what counts as a gap;
restart `srv` afterwards so both gunicorn workers reload the relabelled
manifest.

## Adding a repairable layer

1. Make sure the layer is a tile-cache product with a `has_cached`
   predicate and a `get_*` fetcher on `CopernicusTileCache` /
   `HansenTileCache`.
2. Add `tile_avail_key → product` to `REPAIRABLE_LAYERS`; teach
   `missing_cells()` / `_repair_cells_fill()` the `has_cached` kwarg and
   fetch call if it is not one of the existing five.
3. `python3 product_repair.py scan --apply`, restart srv.

Layers that do **not** belong here: anything from BEV (DTM/DSM/ortho) — those
are `upstream_fail` tiles handled by `partial_kgs` / the coverage oracle, and a
re-upgrade from the full GPKG cannot invent them.

## Invariants

* `-partial` is only ever appended to a JSON product's `version`; never to
  GPKG entries, never to `_json` semantics as the completion marker.
* The fixable gate is evidence-only (Zenodo ZIP index, `local_ok=False`) —
  the primary's disk says nothing about what a peer can fetch.
* Cell-fill requests are gated on openEO health; filling the very cells that
  timed out while openEO still times out just burns 240 s probes.
* The registry is primary-only state (like `v2_regen.json`); a temporary
  director elsewhere simply publishes an empty plan.
