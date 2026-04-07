# AGENTS.md — srtm-lidar-at

## What this is

Python/Flask API that analyses Austrian government LIDAR data (BEV ALS DTM + DSM, 1m resolution) and orthophotos (BEV DOP, 0.2m) on the fly via HTTP range requests — no local tile storage. Accepts any geometry (KML, GeoJSON, coordinates), returns elevation, terrain characterisation, object classification (27 types), temporal change detection (20 event types), and classified rasters.

Live: https://srtm-lidar-at.exe.xyz:8000/  
API docs: /api/v1/docs/llm.txt  
Cadastre integration: https://cadastre-process-api.exe.xyz/api/v1/docs/llm.txt

## Architecture

```
app.py               Flask API — 9 endpoints (4 existing + 3 temporal + info + docs)
tile_index.py         55-tile grid index, CRS transforms (WGS84 ↔ EPSG:3035)
                      3 ALS dates: 20220915, 20230915, 20240915
raster_io.py          Windowed reads from remote GeoTIFFs via /vsicurl/
ortho_io.py           Orthophoto reader (RGBI Operates preferred, DOP RGB 50km fallback)
                      47 RGBI operates with auto-discovery for real NDVI
                      Spectral indices: NDVI, brightness, green ratio, RG index
terrain_analysis.py   Slope, aspect, TRI, TPI, curvature
object_classifier.py  3-phase pipeline + spectral refinement (27 object types)
temporal_analysis.py  Multi-date comparison, change detection (20 event types)
                      Earthworks, road surfaces, tree growth, construction
geo_parse.py          KML / GeoJSON / coordinate string parser
static/index.html     Leaflet web UI
llm.txt               Machine-readable API reference
srv.service           systemd + gunicorn (2 workers, port 8000)
```

## Data Sources

### ALS LIDAR (DTM + DSM)
- **BEV ALS DTM/DSM** from data.bev.gv.at
- 3 dates: `20220915`, `20230915`, `20240915`
- 55 tiles, 50km×50km each, EPSG:3035, float32, ~12 GB per tile
- HTTP range requests — we read only the window we need
- URL: `https://data.bev.gv.at/download/ALS/{DTM|DSM}/{date}/ALS_{DTM|DSM}_CRS3035RES50000mN{n}E{e}.tif`

### Orthophoto (DOP)
- **BEV DOP RGB** 50km tiles, EPSG:3035, 0.2m resolution, 3-band uint8
- Date: `20220128`
- URL: `https://data.bev.gv.at/download/DOP/20220128/DOP_CRS3035RES50000mN{n}E{e}_20220128.tif`
- NOTE: These tiles have a non-standard south-up transform (positive Y). Handled in ortho_io.
- Resampled to 1m for analysis (matching ALS grid)

- **BEV DOP RGBI Operate** — separate RGB + NIR files per survey area (preferred for NDVI)
  - 47 operates covering all of Austria, indexed in `ortho_io.RGBI_OPERATES`
  - Series: 20221027 (2018-2021 flights), 20240625 (2023 flights), 20250415 (2024 flights)
  - Various CRS (EPSG:31254/31255/31256) depending on Meridianstreifen
  - URL: `https://data.bev.gv.at/download/DOP/{series}/{operat}_Mosaik_{RGB|NIR}.tif`
  - Auto-discovered by `ortho_io.find_rgbi_operates()` from WGS84 bbox
  - `read_ortho_for_als()` tries RGBI first, falls back to DOP 50km tiles

## Classification Pipeline (object_classifier.py)

**Phase 1 — Pixel-level.** Every pixel gets a type from 3D surface properties:
- nDSM surface slope, DSM roughness (std 3×3), nDSM height variation (std 5×5), DTM terrain slope
- Ground sub-types: water (ultra-smooth), road/path (smooth), meadow (normal), rough ground (rocky)
- Elevated: flat DSM surface + gentle terrain → building; rough steep surface → tree canopy
- Rock/cliff: steep DTM (>45°) + rough surface

**Phase 1b — Spectral refinement (optional, when orthophoto available).**
- NDVI to distinguish vegetation from built surfaces
- Brightness + color ratios for road/parking/pool/solar panel detection
- Dead tree detection (tall + low NDVI)
- Bare soil vs meadow, building vs tree corrections

**Phase 2 — Watershed segmentation.** Continuous canopy split into individual crowns.

**Phase 3 — Object classification.** Each segment classified by pixel-class majority + morphometrics + crown shape + spectral stats.

## 27 Object Types

| Code | Type | Signature |
|------|------|----------|
| 0 | ground | nDSM < 0.3m, uncategorised |
| 1 | road_path | nDSM < 0.3m, DTM std3 < 0.15 |
| 2 | meadow_field | nDSM < 0.3m, normal roughness |
| 3 | rough_ground | nDSM < 0.3m, DTM std3 > 0.5 |
| 4 | low_vegetation | 0.3–2m |
| 5 | shrub_bush | 2–4m |
| 6 | tree_coniferous | >4m, conical crown |
| 7 | tree_broadleaf | >4m, dome/rounded crown |
| 8 | tree_unclassified | >4m, ambiguous crown |
| 9 | building | flat DSM surface, gentle terrain, 20–2000 m² |
| 10 | structure | small non-building elevated flat object |
| 11 | mast_pole | tiny footprint (<25 m²), tall (>10m) |
| 12 | wall_fence | narrow elongated, <4m |
| 13 | water | ultra-smooth ground, DTM slope < 3° |
| 14 | unclassified | fallback |
| 15 | parking_lot | large flat paved, low NDVI (ortho) |
| 16 | swimming_pool | blue-dominant water, small (ortho) |
| 17 | solar_panel | dark rectangular rooftop (ortho) |
| 18 | greenhouse | bright, low NDVI, slightly elevated |
| 19 | bridge | elevated linear over depression |
| 20 | power_line | very thin, tall, elongated |
| 21 | hedge | linear vegetation 1–4m |
| 22 | tree_row | linear tree arrangement |
| 23 | dead_tree | tall, low NDVI (ortho) |
| 24 | bare_soil | ground, low NDVI (ortho) |
| 25 | rock_cliff | steep terrain, rough, no vegetation |
| 26 | vineyard_orchard | regular low/medium vegetation pattern |

## Temporal Change Detection (temporal_analysis.py)

3 ALS dates enable 2022→2023→2024 comparison. Detects:

### Earthworks (terrain-level DTM changes)
- **earthwork_fill** — terrain raised: landfill, platform, levelling
- **earthwork_cut** — terrain lowered: excavation, quarry, grading
- **earthwork_grading** — terrain smoothed/flattened (roughness reduced)
- **earthwork_dam** — linear raised terrain: levee, dam, embankment
- **earthwork_trench** — linear depression: drainage ditch, utility trench
- **earthwork_pond** — new compact depression: retention basin, pond

### Roads
- **road_new** — new road/path (terrain graded flat + elongated)
- **road_resurfaced** — existing road with slight DTM rise + smoother surface
- **road_widened** — road corridor expanded laterally

### Vegetation
- **tree_growth** / **tree_felling** / **new_tree** / **forest_clearcut**
- **vegetation_growth** / **vegetation_loss**

### Built environment
- **new_building** / **demolition** / **construction**

### Other
- **surface_change** — DSM texture changed without height change
- **unclassified_change**

## Key Conventions

- All geometry input in WGS84; internal processing in EPSG:3035
- nDSM = DSM − DTM (normalised object heights above ground)
- Max query area: 25 km²
- Orthophoto integration is optional (`include_ortho=true`)
- Raster output: 2 bands — type code (uint8) + height (float32)

## Developing

```bash
python3 app.py                    # Flask dev server on :8000
sudo systemctl restart srv        # gunicorn production
journalctl -u srv -f              # logs

# Test elevation
curl -X POST http://localhost:8000/api/v1/elevation \
  -H 'Content-Type: application/json' \
  -d '{"geometry": {"type": "Point", "coordinates": [15.115, 47.137]}}'

# Test change detection
curl -X POST http://localhost:8000/api/v1/changes \
  -H 'Content-Type: application/json' \
  -d '{"date_a": "20220915", "date_b": "20240915", "geometry": {"type": "Polygon", "coordinates": [[[15.4,47.07],[15.41,47.07],[15.41,47.08],[15.4,47.08],[15.4,47.07]]]}}'
```

Dependencies: rasterio, pyproj, shapely, numpy, scipy, scikit-image, flask, geopandas, fiona, lxml.
