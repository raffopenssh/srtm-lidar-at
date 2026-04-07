# AGENTS.md — srtm-lidar-at

## What this is

Python/Flask API that analyses Austrian government LIDAR data (BEV ALS DTM + DSM, 1m resolution) on the fly via HTTP range requests — no local tile storage. Accepts any geometry (KML, GeoJSON, coordinates), returns elevation, terrain characterisation, object classification, and classified rasters.

Live: https://srtm-lidar-at.exe.xyz:8000/  
API docs: /api/v1/docs/llm.txt  
Cadastre integration: https://cadastre-process-api.exe.xyz/api/v1/docs/llm.txt

## Architecture

```
app.py              Flask API — 4 POST endpoints + info/docs
tile_index.py       55-tile grid index, CRS transforms (WGS84 ↔ EPSG:3035)
raster_io.py        Windowed reads from remote GeoTIFFs via /vsicurl/
terrain_analysis.py Slope, aspect, TRI, TPI, curvature
object_classifier.py 3-phase pipeline: pixel classify → watershed segment → object classify
geo_parse.py        KML / GeoJSON / coordinate string parser
static/index.html   Leaflet web UI
llm.txt             Machine-readable API reference
srv.service         systemd + gunicorn (2 workers, port 8000)
```

## Data source

- **BEV ALS DTM/DSM** from data.bev.gv.at, date key `20240915` (Sep 2024)
- 55 tiles, 50km×50km each, EPSG:3035, float32, ~12 GB per tile
- HTTP range requests (`Accept-Ranges: bytes`) — we read only the window we need
- URL pattern: `https://data.bev.gv.at/download/ALS/{DTM|DSM}/20240915/ALS_{DTM|DSM}_CRS3035RES50000mN{n}E{e}.tif`
- Future: additional date keys for temporal comparison (structure is ready in `tile_index.DATASETS`)

## Classification pipeline (object_classifier.py)

**Phase 1 — Pixel-level.** Every pixel gets a type from 3D surface properties:
- nDSM surface slope, DSM roughness (std 3×3), nDSM height variation (std 5×5), DTM terrain slope
- Ground sub-types: water (ultra-smooth), road/path (smooth), meadow (normal), rough ground (rocky)
- Elevated: flat DSM surface + gentle terrain → building; rough steep surface → tree canopy

**Phase 2 — Watershed segmentation.** Continuous canopy split into individual crowns:
- Height-adaptive local maxima as seeds: 8m spacing for >20m trees, 5m for 10-20m, 3m for 4-10m
- Marker-controlled watershed on inverted smoothed nDSM
- Buildings segmented separately to avoid merging with adjacent trees

**Phase 3 — Object classification.** Each segment classified by:
- Dominant pixel class (majority vote from Phase 1)
- Morphometrics: area, compactness, elongation
- **Crown shape profiling**: radial height samples at 2/4/6/8m from peak
  - Conical (spruce/fir): >4m dropoff at 4m radius
  - Dome (beech/oak): <2.5m dropoff, broad top
  - Columnar: steep sides, not peaked

## 15 object types

| Code | Type | Signature |
|------|------|-----------|
| 0 | ground | nDSM < 0.3m, uncategorised |
| 1 | road_path | nDSM < 0.3m, DTM std3 < 0.15 |
| 2 | meadow_field | nDSM < 0.3m, normal roughness |
| 3 | rough_ground | nDSM < 0.3m, DTM std3 > 0.5 |
| 4 | low_vegetation | 0.3–2m |
| 5 | shrub_bush | 2–4m |
| 6 | tree_coniferous | >4m, conical crown |
| 7 | tree_broadleaf | >4m, dome/rounded crown |
| 8 | tree_unclassified | >4m, ambiguous crown |
| 9 | building | flat DSM surface, gentle terrain, 20–2000 sqm |
| 10 | structure | small non-building elevated flat object |
| 11 | mast_pole | tiny footprint (<25 sqm), tall (>10m) |
| 12 | wall_fence | narrow elongated, <4m |
| 13 | water | ultra-smooth ground, DTM slope < 3° |
| 14 | unclassified | fallback |

## Key conventions

- All geometry input in WGS84; internal processing in EPSG:3035
- nDSM = DSM − DTM (normalised object heights above ground)
- Height classes are logarithmic: 0–0.5, 0.5–1, 1–2, 2–4, 4–8, 8–15, 15–25, 25–40, 40–60, 60–80, >80m
- Max query area: 25 km² (safety limit for remote reads)
- Raster output: 2 bands — type code (uint8) + height (float32), with type legend in GeoTIFF tags

## Developing

```bash
# Run locally
python3 app.py                    # Flask dev server on :8000

# Production
sudo systemctl restart srv        # gunicorn, 2 workers
journalctl -u srv -f              # logs

# Test
curl -X POST http://localhost:8000/api/v1/elevation \
  -H 'Content-Type: application/json' \
  -d '{"geometry": {"type": "Point", "coordinates": [15.115, 47.137]}}'
```

Dependencies: rasterio, pyproj, shapely, numpy, scipy, scikit-image, flask, geopandas, fiona, lxml.

## Planned

- Temporal comparison across BEV dataset dates (tile_index.DATASETS ready for multiple dates)
- Integration with cadastre API for automatic parcel enrichment
- Object polygon output (currently centroids only)
