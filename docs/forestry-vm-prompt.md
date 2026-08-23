# Prompt for the new forestry-management VM

Copy everything between the fences into the first message on the new VM.
Prerequisites on that VM: GitHub access (this repo readable, its own new repo
writable) and network access to https://srtm-lidar-at.exe.xyz:8000.

---

```
Build a state-of-the-art forestry management system for Austrian forest
holdings, seeded with the pilot area "WILHELM"
(https://srtm-lidar-at.exe.xyz/?share=WILHELM — ~62 ha near 15.1188E 47.1371N,
76% forest, Styria). Create a new repo `forestry-at` on this VM.

## Division of labour (IMPORTANT — do not violate)

A separate cluster (srtm-lidar-at) already runs a generic remote-sensing
analysis API over Austrian BEV LiDAR + orthophoto + Sentinel + cadastre. It is
mid-way through an Austria-wide processing run and is resource-managed; treat
it as a READ-ONLY data provider:

- USE its API for: segmentation, per-tree change detection, terrain, temporal
  change, Austria-wide segment/parcel queries, cadastre + Natura2000 lookups,
  GPKG exports.
- Do NOT build or duplicate segmentation/classification there or here.
- Do NOT call any /api/v1/director/* or /api/v1/admin/* or /processing/*
  endpoints. Analysis endpoints only.
- Read raw rasters (DTM/DSM 1m, 3 epochs; ortho RGBI 0.2m) DIRECTLY from BEV
  via GDAL /vsicurl/ — they are public; do not proxy raster bytes through the
  API. URLs: https://data.bev.gv.at/download/ALS/{DTM,DSM}/{20220915,20230915,20240915}/
  and .../DOP/. Tile scheme: see GET /api/v1/layers?bbox=…
- Pull once, cache locally (the pilot AOI is small; cache all three epochs).
  Use async=true on slow endpoints and poll. One request at a time; this is a
  shared production system.

All forestry DOMAIN logic lives on THIS VM: your own DB, models, UI, reports.

## Upstream API — read these first

- https://srtm-lidar-at.exe.xyz:8000/api/v1/docs/llm.txt  (full API reference)
- GET /api/v1/share/WILHELM → full segmentation result for the pilot AOI
  (3386 segments with 40+ features each: height p90, NDVI harmonics,
  phenology_class, SAR, volume_change_m3, confidence…). Use its geometry as
  the AOI seed; GET /api/v1/share/WILHELM/download.gpkg for QGIS-ready layers.
- POST /api/v1/segment — re-run segmentation on any Austrian AOI ≤25 km²
  (async, ~2 min for 60 ha).
- POST /api/v1/changes/trees {date_a,date_b} — per-tree growth/felling
  matching between ALS epochs (2022/2023/2024).
- POST /api/v1/trees — single-epoch tree inventory: crown polygons
  (crown_geometry=polygon), crown_shape, leaf-type hint (coniferous/broadleaf/
  dead/unknown), and forestry summary (stems_per_ha, top_height_m, crown_cover_pct).
  Supports async=true (poll /api/v1/segment/progress, fetch /segment/result).
- POST /api/v1/changes/trees also supports async=true and crown_geometry=polygon.
- POST /api/v1/terrain — slope/aspect/TRI/TPI/curvature (for windthrow risk,
  harvest accessibility).
- POST /api/v1/changes — 20 change event types incl. forest_clearcut.
- GET /api/v1/query?segments=true&object_type=tree… — Austria-wide segment
  index (for later scaling beyond the pilot).
- GET /api/v1/parcels/batch, /api/v1/parcel/<id>/detail — cadastre ownership,
  parcel boundaries, protected-area overlaps (Natura 2000).

## What to build (MVP → iterate)

1. Ingestion: fetch WILHELM share + /changes/trees for 2022→2023→2024 +
   terrain; read DTM/DSM via /vsicurl/; persist to a local DB (SQLite +
   SpatiaLite or PostGIS). Schema: stands, trees (per-epoch observations,
   stable tree_id across epochs), parcels, events.
2. Stand delineation: cluster tree segments into stands using crown density,
   height class, phenology_class/leaf type, and parcel boundaries.
3. Inventory metrics per stand: stem count/ha (from tree detections),
   top height (h_dom), mean height, crown cover %, height distribution.
   Volume & biomass via Austrian allometric tariffs (e.g. Pollanschütz form
   factors; conifer vs broadleaf split from leaf-type hint); carbon from
   biomass (IPCC defaults). State assumptions explicitly.
4. Growth & change: per-tree height increment between epochs, felling
   detection, clearcut polygons (reconcile our tree_loss with Hansen),
   growth-rate anomalies (bark-beetle / drought candidates = NDVI harmonic
   amplitude drop + height stagnation).
5. Risk: windthrow index from terrain (slope, aspect vs prevailing W winds,
   stand height, edge exposure after neighbouring clearcuts).
6. Planning: thinning/harvest recommendations per stand (density vs yield
   table targets), accessibility (slope classes, distance to roads from OSM
   layer in the segmentation result).
7. UI: Leaflet web app (port 8000) — stand map, per-stand dossier, per-tree
   drill-down, epoch slider, printable management-plan report (PDF/HTML).
8. Keep an `AGENTS.md` from day one; commit early and often.

Start by reading llm.txt and the WILHELM share, design the DB schema, then
build ingestion. Show me the stand map for WILHELM as the first milestone.
```

---

## API additions made on THIS repo in support (shipped)

1. `POST /api/v1/trees` — single-date per-tree inventory: crown polygons,
   crown_shape, leaf-type hint, stems/ha, top height (h_dom), crown cover.
   Async-capable. Thin wrapper over `object_classifier.classify_objects`.
2. `/changes/trees`: `async=true` (reuses segment task framework) and
   `crown_geometry=polygon`, `min_tree_height` param.
3. Both documented in `llm.txt` (`### Analysis` section).

Rationale: tree detection stays canonical upstream; forestry domain logic
(allometrics, stands, planning, UI) stays on the forestry VM. The fleet is
mid-run on the 8440-KG job and the primary is dashboard-only — no forestry
app components belong on this cluster.
