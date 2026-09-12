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
