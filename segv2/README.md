# segv2 — segmentation/classification model v2 harness

**Fleet-safety contract**: nothing under `segv2/` is imported by `app.py`,
`austria_processor.py`, `object_segmentation.py` or `learned_classifier.py`.
Peers keep running the v1 model (`data/best_model/`) untouched. Only after the
harness shows a *significant* win (see "Promotion criteria") do we wire v2 into
the processor, bump the product version, and let the director roll it out.
`_ensure_origin_synced` pushes `main` every tick, so commits here *are*
visible to peers as code — but dead code until wired. Do not touch the
files above from this work-stream until promotion.

## Why v2 (short)

1. **Labels were wrong.** `CADASTRE_TO_TYPE` used the pre-Aug-2026 cadastre
   label table (code 48 = "road" — it is farmland, 3.76M parcels; 41/42
   swapped; 59 rivers = "bare_soil"; …). The v1 RF (OOB 0.70, 55 KGs) learnt
   from that. See `segv2/labels.py::NS_CODE_TO_TYPE` for the corrected map
   (BEV Katastralmappe SHP V2.9 Tabelle 8).
2. **Legal parcels ≠ physical surface.** Cadastre 95 (Straße) includes the
   green shoulder; 59/60 (Gewässer) include the banks. We now label pavement
   from OSM road centrelines buffered by `fclass`, water from OSM water
   areas/lines, rail from OSM rail; the cadastre remainder becomes verge /
   riparian grass+shrub.
3. **INVEKOS Schläge 2024-1** (AMA, CC-BY-AT 4.0, 2.95M field polygons with
   crop type) → precise, *current* arable/grassland/vineyard/orchard/hedge/
   pond labels, far better than the 30-yr-old cadastre Nutzung.
4. **Flight dates vary** (trees v2 lesson): DSM mosaics 2022/23/24 are
   stitched from 2010-2024 flights. v2 features carry the effective DSM year
   gap, per-year-normalised height change, ortho↔LiDAR year gap.
5. **Data comes from our own Zenodo full GPKGs** (all 1 m raster layers incl.
   multi-date DTM/DSM, ortho RGBI per year, NDVI, WorldCover, SAR, Hansen) +
   the Zenodo tile cache for harmonics. No BEV / openEO traffic. Every GPKG
   pulled is audited (md5 vs manifest, `integrity_check`, layer inventory)
   into `data/segv2/audit.jsonl`.

## Pipeline

```
segv2/gpkg_fetch.py   Zenodo full GPKG → /tmp (8 parallel range requests), audit
segv2/labels.py       cadastre + OSM + INVEKOS → 1 m label raster (priority stack)
segv2/features.py     GPKG rasters → tiles → object_segmentation(features_only)
                      + v2 extras (OSM distances, neighbour context, ALS dates)
segv2/build_dataset.py  per-KG parquet: features, label, label_source, purity, xy
segv2/train.py        harness: v1 baseline / RF-relabelled / HGB(+context);
                      grouped-by-KG spatial CV, macro-F1, per-class F1, ECE
```

## Label taxonomy (v2)

Existing 25 types + **new**: `rail`, `wetland`, `glacier`. Merged for
training as in v1 (`excavation`+`fill`→`earthwork`). Rule-only types
(wind_turbine, substation, solar_panel) stay rule-based.

## Promotion criteria

Grouped-KG CV on the same relabelled test folds: v2 must beat the v1 model
by ≥ +0.05 macro-F1 **and** not lose > 0.02 F1 on any of tree/roof/grass/crop/
water/road, with calibration (ECE) ≤ v1. Then: side-by-side visual QA on 5
KGs (dashboard overlay), and a reprocessing cost estimate.

## v2 product design notes (for the reprocessing pass)

Requested: downstream services (trees v2 `tree_inventory`, terrain endpoints
such as `/kg/<code>/heightfield`) must get much faster, **without** bloating
the per-KG JSON. Plan — precompute once during reprocessing, store in the
*GPKG* (cheap, range-readable) and keep only compact indexes in JSON:

| Need | Today | v2 |
|---|---|---|
| Terrain grids (heightfield, slope/aspect) | IDW from index points, or full GPKG download (GB) | `terrain_coarse` layer: DTM/DSM/nDSM at **5 m** + slope/aspect/TRI (int16, ~1/25 the pixels) in **light** GPKG; plus 25 m summary grid in JSON (`terrain.grid25`: elev/slope int16 base64, ≤ 40 KB/KG). Heightfield endpoint reads that instead of IDW. |
| Trees v2 | needs nDSM (full GPKG) + multi-date DSM | `tree_apices` vector layer (x, y, h, crown_r, dh_per_year, flight_year) in light GPKG; JSON keeps only per-parcel `top_trees` + KG-level histogram (already compact). Apex detection runs once on the same nDSM the segmenter used. |
| Flight-date normalisation | recomputed per request | per-tile `als_meta` table in light GPKG (`dtm_year, dsm_year, ortho_year, eff_date, years_gap`) + `acquisition` block in JSON (a few hundred bytes). |
| Segment lookup | full `segments` polygons | keep; add R-tree (GPKG standard `rtree_segments_geom`) so viewers/APIs range-read only what they need. |
| Raster layers | one PNG tile pyramid level | write **overviews** (zoom 0..N) in full GPKG so the dashboard/QGIS can stream at any zoom via HTTP range instead of downloading the file. |

JSON size budget stays ≈ today's (median 6 MB; 25 m terrain grid + acquisition
block add < 50 KB). Everything heavier lives in the light GPKG (already
downloaded by `GpkgCache` on demand and cached on the primary).

Performance rule carried from the v1 post-mortem: every per-tile step masks to
the KG's parcel union first; tiles with < 2 % KG coverage are skipped.
