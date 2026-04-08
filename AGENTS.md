# AGENTS.md — srtm-lidar-at

## What this is

Python/Flask API that analyses Austrian landscape transformation using 6 data layers:
1. BEV ALS DTM+DSM (1m, 3 dates: 2022/2023/2024) via HTTP range requests
2. BEV DOP RGBI orthophotos (0.2m, 47 operates) via HTTP range requests
3. Copernicus Sentinel-2 NDVI growing-season composite (10m, via openEO)
4. Copernicus ESA WorldCover land cover (10m)
5. Copernicus Sentinel-1 SAR backscatter (10m)
6. Austrian Cadastre building footprint polygons (mm-precision, ground truth)

Classifies landscape into 10 types focused on human vs natural, detects
machinery traces from DTM time series, and identifies linear features
via Hessian eigenvalue analysis.

Live: https://srtm-lidar-at.exe.xyz:8000/
API docs: /api/v1/docs/llm.txt
Cadastre: https://cadastre-process-api.exe.xyz/api/v1/docs/llm.txt

## Architecture

```
app.py                   Flask API — all endpoints
hansen.py                Hansen Global Forest Change (GFC-2024-v1.12) integration
                          Reads treecover2000/lossyear/gain via /vsicurl/
                          calibrate_clear_cut() boosts/downgrades clear_cut confidence
                          evaluate_forest_loss() returns P/R/F1 vs Hansen reference
object_segmentation.py   NEW: Watershed-based segmentation + classification
                          Fused gradient (Sobel on DTM/DSM/CHM/RGBI/NDVI)
                          Felzenszwalb over-segmentation + RAG boundary merge
                          25 individual types + 11 group types
                          Hierarchical: tree→forest, roof→building, road→road_network
                          Cadastre-calibrated building detection (F1=0.74)
                          Endpoint: POST /api/v1/segment
landscape_classifier.py  DEPRECATED: 10-type pixel-level classifier
                          Superseded by object_segmentation.py
                          Endpoint: POST /api/v1/objects (legacy)
copernicus.py            Sentinel-2 NDVI, ESA WorldCover, SAR via openEO
cadastre.py              Building footprint fetcher + ground truth evaluator
tile_index.py            55-tile grid index, CRS transforms
raster_io.py             Windowed reads from remote GeoTIFFs via /vsicurl/
ortho_io.py              BEV orthophoto reader (RGBI operates + DOP fallback)
terrain_analysis.py      Slope, aspect, TRI, TPI, curvature
temporal_analysis.py     Multi-date comparison, 20 change event types
geo_parse.py             KML / GeoJSON / coordinate parser
static/index.html        Leaflet web UI
llm.txt                  Machine-readable API reference
srv.service              systemd + gunicorn (2 workers, port 8000)
docs/reference_algorithms_summary.md  Algorithm design notes

OLD (kept for reference):
object_classifier.py   Original 27-type classifier
```

## 10 Landscape Types (pixel-level, /api/v1/objects)

| Code | Type | How detected |
|------|------|-------------|
| 1 | engineered_surface | DTM roughness < 0.025m at 3m scale, slope uniformity < 1° |
| 2 | engineered_slope | Hessian ridge/valley strength + high linearity (>0.5) |
| 3 | excavation | DTM time series: terrain lowered + spatially coherent |
| 4 | fill | DTM time series: terrain raised + spatially coherent |
| 5 | building | Multi-criteria score: DSM std + nDSM std + slope + NDVI + stability |
| 6 | infrastructure | Elevated, non-building, non-tree: small/linear/irregular |
| 7 | tree_canopy | Elevated >4m, rough DSM, high NDVI, growing over time |
| 8 | vegetation | Low/medium height, not engineered |
| 9 | bare_natural | Steep + rough terrain, or bare soil |
| 10 | recent_disturbance | DTM changed >0.15m between dates + spatial coherence |

## 25 Object Types + 11 Groups (watershed, /api/v1/segment)

Individual objects detected per-segment after Felzenszwalb+RAG segmentation:

| Category | Types | How detected |
|----------|-------|-------------|
| Vegetation | tree, shrub, grass, hedge | nDSM height + NDVI + roughness + elongation |
| Water | water | ESA WorldCover + very low NDVI + low NIR + flat |
| Buildings | roof, greenhouse, solar_panel | Smooth DSM + low NDVI + compact + stable |
| Infrastructure | fence, wall, mast | Shape (elongated/tiny) + height + non-vegetated |
| Transportation | road, path, parking, bridge | Smooth DTM + elongated/compact + low NDVI |
| Agricultural | crop, orchard, vineyard, garden | NDVI + ESA cropland prior + area/spacing |
| Terrain | bare_soil, rock | Low NDVI + steep/rough (rock) or flat (soil) |
| Disturbance | excavation, fill, clear_cut, construction | DTM temporal change + nDSM temporal change |

Groups (adjacent compatible objects merged):

| Group | Members merged | Cadastre ref |
|-------|---------------|-------------|
| forest | tree+shrub+hedge | W 56, W(Kr) 57 |
| building | roof+wall+solar_panel+greenhouse | B 42-47 |
| road_network | road+path+parking | V 48,73,74 |
| cropland | crop+grass | A 51,62 |
| pasture | grass+garden | LN 52-55 |
| quarry | excavation+fill | Ab 80,93 |
| construction_site | construction+excavation+fill | recent |
| waterbody | water | GW 70,71 |
| woodland | shrub+hedge | sparse trees |
| hedgerow | hedge | linear veg |
| orchard_grove | orchard+vineyard | OG 65, WG 63 |

## Key Detection Methods

### Linear Feature Detection (Hessian eigenvalues)
DTM smoothed at σ=2m, Hessian matrix computed, eigenvalues λ1/λ2 extracted.
Ridges (embankments): λ1 >> 0, |λ2| small. Valleys (ditches): λ2 << 0.
Linearity = |λ1-λ2|/(λ1+λ2). Also applied to DSM for walls/fences.

### Machinery Trace Detection (DTM time series)
For each consecutive date pair: DTM differenced, filtered for spatial
coherence (8-connected opening+closing), small regions removed.
Roughness change computed (natural→smooth = machinery). Paired cut/fill
detected via dilation overlap.

### Building Detection (multi-criteria scoring)
Pixel score from 0-10+ combining:
- DSM roughness (std3 < 0.5: +2, < 1.0: +1, < 1.5: +0.5)
- Height uniformity (nDSM std5 < 1.0: +1.5, < 2.0: +0.8)
- Terrain flatness (slope < 5°: +1.5, < 10°: +0.8)
- NDVI < 0.15: +2.0 (if ortho available)
- Brightness > 90 + NDVI < 0.20: +1.5
- High NDVI > 0.30: -2.0 (penalize vegetation)
Threshold: 5.0 with ortho, 6.0 without.

## Copernicus Integration

OpenEO client credentials:
- Client ID: sh-19061cbb-c6f9-4464-bba6-006e7fa17435
- Backend: openeo.dataspace.copernicus.eu

Data fetched on-demand, cached in /tmp/copernicus_cache/.
NDVI composite uses April-September median (cloud-masked via SCL dilation).

## Cadastre Integration

Building footprints fetched from /api/v1/export/geojson?kg=...&layers=building_footprints.
KG codes discovered via point-in-polygon lookup on bbox corners.
Used for calibration/validation only, NOT as direct classification input.

## Developing

```bash
python3 app.py                    # Flask dev server on :8000
sudo systemctl restart srv        # gunicorn production
journalctl -u srv -f              # logs

# Calibrate against cadastre
python3 -c "
import raster_io, tile_index as ti, cadastre, landscape_classifier as lc
from shapely.geometry import box
data = raster_io.read_dtm_dsm(ti.geometry_to_3035(box(15.085,47.065,15.095,47.075)), '20240915')
fps = cadastre.fetch_building_footprints((15.085,47.065,15.095,47.075))
bldg = cadastre.rasterize_buildings(fps, data['transform'], data['shape'])
result = lc.classify_landscape(data['dtm'], data['dsm'], data['mask'], data['transform'])
print(cadastre.evaluate_classification(result['type_map'], bldg, building_codes={5}))
"
```

## Hansen Global Forest Change

GFC-2024-v1.12 tile 50N_010E (covers Austria). Layers:
- treecover2000: canopy cover % in year 2000
- lossyear: year of loss 1-24 (2001-2024), 0=no loss
- gain: forest gained 2000-2012
- datamask: 1=land, 2=water

Used by `hansen.py` to calibrate clear_cut detection:
- clear_cut on Hansen loss → confidence +0.15
- vegetation on recent Hansen loss + temporal instability → reclassify to clear_cut
- clear_cut on non-forest area → confidence -0.20

Cached in /tmp/hansen_cache/ as .npz files.

## GeoPackage Export

POST /api/v1/export/geopackage returns all layers in one GPKG:
- Band 1: DTM, Band 2: DSM, Band 3: nDSM
- Bands 4-6: RGB ortho (optional), Band 7: NIR (if available)
- Final band: segment_type (type codes from object_segmentation.py)
- Filter by ?types=roof,tree,road to include only specific segment types

## Segment Raster Overlay

POST /api/v1/segment/overlay returns coloured PNG (RGBA) showing segmentation.
Server caches last segmentation result — re-renders with different ?types= filter
are instant (no re-running pipeline). Legend filter in frontend toggles both
point markers AND segment raster overlay simultaneously.

Dependencies: rasterio, pyproj, shapely, numpy, scipy, scikit-image, flask, openeo, requests
