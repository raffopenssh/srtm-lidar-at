# Reply to FEEDBACK-5 — `/api/v3/trees` v3.1 + product 2.2

Status: **implemented 2026-09-20**, `tree_algo_version = 3.1.0` (v3), `2.4.0`
(live engine), product `2.2` (manifest `v2.2`). Contract in
`/api/v1/docs/llm.txt` → "v3 tree service". Point-by-point:

## 1 · tree_id stable between product and live — FIXED at the root

Root cause was not a half-pixel convention but a **grid-origin bug** in
`raster_io.read_window_bbox`: rasterio floored the fractional window
offset (so the pixel DATA were true BEV pixels) while the returned
transform kept the fractional AOI corner. Every apex was shifted by
`+frac(min_e)` E / `−(ceil(max_n)−max_n)` N — a constant per AOI (live)
and per KG (product), hence your (−0.1, −0.7).

* `read_window_bbox` now snaps the bbox outwards to whole metres. **All**
  1 m grids (live v2, live v3 fallback, fresh products, the v2-upgrade
  shim `v2_source._Gpkg`) are the integer-metre EPSG:3035 BEV grid; an
  apex is the pixel centre (`x.5`). `tree_id = t_<E_dm>_<N_dm>` of that
  centre — AOI-independent. Named in `meta.grid_anchor`
  (`apex_anchor: pixel_center`, origin, id rule).
* Existing 2.1 products are **re-anchored on read**
  (`trees_v3.canonicalize_rows`: the per-product δ is the median
  fractional offset, sign fixed by the known floor semantics). Measured
  on 63330: **99.1 % exact id equality** between the re-anchored 2.1
  product and a fresh live v2 call (the rest are watershed-context
  differences at the AOI edge). The id stored in the 2.1 file comes back
  as `tree_id_product` so ids captured from v3.0 can be migrated.
* `legacy_ids=1` (v2 and v3) adds `tree_id_v2` = the id a pre-2026-09-20
  `/api/v2/trees` call over **the same AOI** reported
  (`tree_inventory.legacy_tree_id`). Submit your old AOIs once, join on
  `tree_id_v2`, store `tree_id`. Your nearest-apex bridge becomes
  unnecessary.
* Caveat, stated in the docs: products built by the v2-upgrade path (from
  v1 full GPKGs) can carry ≤1 px tile misregistration inside the mosaic.

## 2 · ortho-only / fused stems — opt-in on the live path

Product apices stay `ndsm_only` (the processor never holds ≤0.5 m ortho).
`detection_mode=fused` on `/api/v3/trees` runs the live v2.3 fused
detector over the whole AOI (v2 cost), joins product rows by `tree_id`
(trivial now) and carries `stand_context`, `segment_type`,
`dh_per_year_m`, `als_year`, `product_code` onto the matched trees
(`source=fused_product`); ortho-only stems come back as `live_fused`;
product apices the fused run did not reproduce within 1.5 m are kept.
So one call gives your superset register with the product enrichment.

## 3 · vitality / species / per-crown spectra — IN THE PRODUCT (2.2)

The `ndvi` you saw was the 1 m BEV ortho NDVI point-sampled at the apex,
not Sentinel-2 — but a point, not a crown. 2.2 keeps the ortho R/G/B/NIR
bands per tile (uint8, checkpoint-only) and composes them per 2 km chunk
under the stitched apex inventory, so every apex now carries
**per-crown** `ndvi_mean`, `ndvi_p10`, `nir_mean`, `brightness_mean`,
`green_ratio_mean` and the derived `vitality` (+conf) and `species_hint`
(+conf), computed by the same `tree_inventory` code as the live engine
(`assign_vitality` / `assign_species_hint`, refactored out so the product
re-ranks **KG-wide** after all chunks). `dead` is absolute (NDVI < 0.15,
p10 < 0.10, h ≥ 5 m) — your standing-deadwood list. `stressed`/`vital` and
`ndvi_percentile_in_aoi` are **re-ranked over the submitted AOI on every
v3 call** (`rank_vitality_in_aoi`), so you re-cut exactly as in v2.
Anomaly rule tightened: bottom decile AND below the population median
(ties at the mode no longer flag a whole uniform stand).

## 4 · crown polygons — `tree_crowns` layer (2.2)

`crown_geometry=polygon` returns the watershed crown outline per apex
from the new `tree_crowns` layer (EPSG:3035, simplified 0.5 m, no R-tree,
`tree_id` only — ~250–600 B/crown, i.e. a few MB on a 30 MB light GPKG).
2.1 products / live rows fall back to the apex point;
`meta.crown_polygons` counts both.

## 5 · `is_edge` against the submitted AOI — done

`trees_v3.mark_edge`: crown disc (`crown_r_m`) crosses the AOI boundary →
`is_edge`, product and live alike; per stand in `by-polygons`. Densities
use non-edge stems; `include_edge=false` drops them from features too.

## 6 · canopy area — area-weighted 1 m canopy on the 25 m grid (2.2)

`landcover.grid25` gains three u8 fraction grids (~1 KB gz each):
`canopy_frac` (fraction of 1 m px in the apex-inventory canopy mask —
nDSM ≥ min_tree_height inside a v2 segment, roofs/crop rejected — i.e. the
v2 pixel canopy), `tree_frac`, `woody_frac`. `CanopyGrid.canopy` computes
Σ (cell ∩ AOI area) × frac — **no centre rule, no dominance rule, no rim
loss**. `area_ha_canopy` ← canopy_frac, `area_ha_forest` ← tree_frac,
`area_ha_stand` ← woody_frac; `summary.canopy_method` names the method,
2.1 products fall back to the old rule and say so.

## 7 · acquisition — `meta.acquisition`

Per product: flight blocks (`als_year_min/max`, `years_gap_median`,
`flight_dates`, `n_tiles`) from the product JSON, fleet roll-up, plus the
live dataset's block lookup; the note spells out that `meta.dataset` is
the mosaic release.

## 8 · change path — crown-overlap matching + felling patches

Epoch a is now a crown-footprint label raster (`tree_crowns` polygons, or
mean-radius discs for 2.1), so `match_trees` runs **both passes** (apex,
then crown overlap on the proxy labels). `felling_patches` are derived
**crown-level**: a crown is felled when `h_m − p90(live nDSM inside its
footprint) ≥ felling_min_drop_m`; adjacent felled footprints merge into
patches with `area_sqm`, `drop_mean_m`, `drop_max_m`,
`height_before_mean_m`, `n_crowns` — your fresh/old filter works
unchanged. (A pixel drop against an apex-height proxy would have flagged
every crown margin; the note explains the proxy.)

## Minor

* `GET /api/v3/*` → 405 with a hint (was the static catch-all's 404).
* `meta.product_codes_failed_reasons` per code.
* `meta.product_versions` per code — which fields to expect.

## Rollout

Product 2.2 re-upgrades the ~100 KGs already at 2.1 (eligibility is
`version == v2.2`; never-upgraded KGs still go first, 63330 is pinned via
`/api/v1/director/v2/priority`). Until a KG is re-upgraded, v3 over it
returns the 2.1 field set with re-anchored ids and centre-rule canopy.
