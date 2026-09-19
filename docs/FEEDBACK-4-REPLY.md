# Reply to FEEDBACK-4 — product-backed tree inventory (`/api/v3/trees*`)

Status: **implemented**, `tree_algo_version = 3.0.0`; live on
https://srtm-lidar-at.exe.xyz:8000. Full contract in
`/api/v1/docs/llm.txt` → "v3 tree service".

## Why a v3

`/api/v2/trees` re-detects every apex from live BEV rasters on every call
(tens of seconds per km², BEV egress, Copernicus credits for ortho/NIR).
Since product 2.1 (2026-09-19) the Austria-wide processor already runs
exactly that detector once per KG and ships the result:

* `tree_apices` point layer in every `_light_v2.gpkg` — v2.3 engine,
  `ndsm_only` seeding on the stitched 1 m nDSM, cadastre building mask,
  per apex: `h_m, crown_r_m, crown_area_m2, surface_class,
  tree_likelihood, stand_context, segment_type(+conf), dh_per_year_m,
  als_year, ndvi, leaf_type_hint(+conf), dbh_est_cm, volume_m3_est`;
* `landcover.grid25` in the KG JSON — dominant v2 segment class per
  25 m cell (the canopy denominator).

v3 reads those instead of recomputing. Same tree IDs
(`t_<E_dm>_<N_dm>`), same allometry, same surface engine — a v3 answer
over a 2.1 KG is the v2 `ndsm_only` answer, frozen.

## What you get

| endpoint | in | out |
|---|---|---|
| `POST /api/v3/trees` | AOI (any geometry format we accept) | apex Features + explicit-denominator summary |
| `POST /api/v3/trees/by-polygons` | FeatureCollection of stands (`key_property`) | per-stand summaries (+`include_trees`) |
| `POST /api/v3/changes/trees` | AOI, `date_b` | product apices (epoch a) matched against a live nDSM (epoch b) |

New in the summary vs v2:

* `stems_per_ha_forest` — stems whose apex sits on a v2 `tree` segment,
  over the `tree`-class grid25 area (`area_ha_forest`). This is the
  "Waldfläche" density you asked for; `stems_per_ha_canopy` keeps the
  wider woody-stand denominator (tree/orchard/vineyard/hedge/garden/
  shrub) and `stems_per_ha_total` the AOI.
* `by_stand_context` — how many stems are forest vs orchard vs hedge vs
  garden trees. Filter server-side with `stand_context=tree`.
* `dh_per_year_m_median` / per-tree `dh_per_year_m` — apex height change
  between the oldest and newest BEV flight over the KG, normalised by
  the real flight-year gap (null when the KG has a single flight year).
* `als_year` per tree — the BEV block flight year, not the mosaic date.
* `by_source` / `meta.source_mix` — which trees came from a product and
  which from the live fallback.

## Coverage and honesty rules

* A product's footprint is its classified grid25 cells (≈ the KG
  polygon). Apices are read inside footprint ∩ AOI; where two products
  overlap, the same `tree_id` is counted once.
* The rest of the AOI (KGs not yet upgraded to 2.1) runs the live v2
  detector once (`fallback_live=true`, default), `ndsm_only` so both
  halves use the same seeding; live apices within 1.5 m of a product
  apex are dropped. `fallback_live=false` gives product-only answers
  and reports `meta.coverage.live_area_ha` so you know what is missing.
* Canopy area from a 25 m grid is coarser than the v2 pixel canopy
  (±1 cell per stand edge). `canopy_denominator_note` states exactly
  what was counted; `area_ha_canopy_cover_weighted` (× per-cell cover
  fraction) is the finer alternative.
* Product apices carry no `is_edge` (the inventory ran on the whole KG,
  so no AOI-edge crowns are truncated) — stems/ha in v3 therefore use
  every product apex; live-fallback edge trees are still excluded.

## Changes against the frozen inventory

`/api/v3/changes/trees` is built for the next BEV re-flight: epoch a is
the product (no re-detection, no drift), epoch b is live at `date_b`.
Limits, stated in the response: the product has no epoch-a 1 m raster,
so the 3 m apex evidence uses a crown-disc proxy and crown-overlap
(pass 2) matching is off — `unmatched_a_*` sub-classes are coarser;
`felling_patches` is null (use `/api/v2/changes/trees`). Growth rates
are only emitted when `date_b`'s flight year is later than the product
`als_year`; `epoch_dates.same_flight_epoch` flags republished data.

## Rollout

The fleet is upgrading KGs to 2.1 in the background (`v2:` line on
`/process.txt`). Until a KG is upgraded, v3 over it is 100 % live and
`meta.source_mix.product` is empty — same numbers, just slower. No
client change is needed as coverage grows.
