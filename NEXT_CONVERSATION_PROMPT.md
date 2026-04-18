# Austria Processor: GPKG Stitching, Segment Raster, NDVI Path Fix, Cache Management, Cadastre Bbox Filter

There are 5 issues to fix in the austria processor (`austria_processor.py`):

## 1. Stitch tiles into single raster layers in full GPKG and light GPKG

Currently `build_full_gpkg_tiled()` and `build_light_gpkg_tiled()` write **per-tile** raster layers (`DTM_t1`, `DTM_t2`, `segment_type_t1`, etc.) and per-tile vector layers (`segments_t1`, `segments_t2`, etc.). This produces dozens of layers in QGIS that are unusable.

**Target**: Match the output format of `/api/v1/export/geopackage` (see `_gpkg_core()` in `app.py` around line 2900). That endpoint produces:
- One `segment_type` raster (uint8, type code per pixel) covering the full bbox
- One `segment_height` raster (float32, nDSM height per pixel) covering the full bbox  
- One `segments` vector layer with all segment polygons
- One `parcels` vector layer, one `buildings` layer, etc.

**How to stitch**: For each KG, all tiles share the same CRS (EPSG:3035). Compute the union bounding box of all tiles, allocate a single full-KG raster, then paint each tile's data into it at the correct offset (using the tile's transform vs the full-KG transform). Same for segment_type and height rasters. For vectors, merge all per-tile segments into one layer.

For the **full GPKG**: stitch DTM, DSM, nDSM, segment_type, segment_height into single layers.
For the **light GPKG**: stitch segment_type + segment_height into a single 2-band raster, merge segment vectors into one layer.

The segment raster should be 2 bands: band 1 = type code (uint8), band 2 = height (float32). Use the same approach as `app.py` line ~3021.

## 2. Cadastre buildings/parcels outside KG bbox

For KG 91108 (Fussach), 1397/5453 buildings (25.6%) have centroids outside the KG bbox. Some are wildly off (lat 47.36 vs KG 47.45-47.51). The cadastre API returns features for the KG code, but some features extend far beyond the KG boundary.

**Fix**: In `fetch_cadastre_data()` (~line 427), after fetching parcels and building footprints, compute the KG boundary polygon from the union of all parcel geometries (parcels define the KG boundary). Then filter building_footprints to only those whose centroid falls within the KG boundary polygon (with a small buffer, e.g. 50m). This ensures the tile grid (computed from the KG bbox) covers all buildings.

Alternatively: compute the KG bbox from the cadastre data and use that to define the tile grid. But filtering is simpler and more correct.

## 3. Add NDVI component to path/road classification

The RF classifier over-classifies flat ground as "path" — for Fussach, 23.3% of area is classified as path, which is unrealistic. Paths and roads have very low NDVI (bare/paved surfaces), while grass/crop have moderate-high NDVI.

**Fix**: In `object_segmentation.py`, in the rule-based `classify_object()` fallback and in the feature extraction for RF, ensure there's a strong NDVI signal differentiating path from grass/crop. Specifically:
- In the rule-based classifier: if a ground-level segment has NDVI > 0.3 (Copernicus) or > 0.15 (BEV NIR), it should NOT be classified as path/road regardless of other features
- Check that the RF features include fused NDVI (they do — feature `ndvi_fused`), but the model may need the threshold enforced as a post-classification correction
- Add a post-classification NDVI override in `segment_and_classify()`: after RF classification, any segment classified as path/road with fused NDVI > 0.35 should be reclassified to grass (or crop if in agricultural context)

This is a targeted fix, not a full retraining.

## 4. Disk cache management when storage < 5GB free

The processor accumulates caches in:
- `data/austria_processor/bev_tile_cache/` — BEV DTM/DSM windowed reads
- `data/austria_processor/copernicus_tiles/` — Copernicus grid-snapped cache
- `data/austria_processor/hansen_tiles/` — Hansen grid-snapped cache  
- `rf_training_data/copernicus_cache/` — per-bbox Copernicus cache
- `data/austria_processor/gpkg/` — temp GPKG files (should be deleted after upload)

**Fix**: Add a `check_disk_space()` function called at the start of each KG processing. If free disk < 5GB:
1. Delete `data/austria_processor/gpkg/` temp files older than 1 hour
2. Delete oldest entries from `bev_tile_cache/` (LRU by mtime) until 2GB freed or cache empty
3. Delete oldest entries from `copernicus_tiles/` and `hansen_tiles/` (LRU)
4. Delete `rf_training_data/copernicus_cache/` entries (LRU)
5. Log what was deleted and how much space was freed
6. If still < 3GB free after cleanup, pause the processor with a clear error message

Use `shutil.disk_usage()` for checking. Don't delete files for the current KG being processed.

## 5. Validation issue: JSON parcels check

The current validation (`validate_kg_outputs`) warns when <80% of parcels have elevation. With the fixes from the previous conversation (pre-loading DTM per tile + fallback point reads), this should be nearly 100%. But if the cadastre bbox filter (issue #2) is also applied, fewer parcels will be outside tiles, further improving coverage. Keep the 80% threshold but log at INFO level how many used the fallback.

## Key files
- `austria_processor.py` — `build_full_gpkg_tiled()` (~line 1664), `build_light_gpkg_tiled()` (~line 1743), `fetch_cadastre_data()` (~line 427), `validate_kg_outputs()` (~line 3413)
- `app.py` — `_gpkg_core()` (~line 2900) for reference GPKG format
- `object_segmentation.py` — `classify_object()` and `segment_and_classify()` for NDVI override
- `tile_cache.py` — cache directories

## Testing
After changes, restart the processor (`sudo systemctl stop austria_processor; sleep 2; sudo systemctl start austria_processor`). It will re-process the current KG. Check:
- GPKG in QGIS should have single stitched layers, not `_t1`, `_t2` etc.
- Building count should be lower (filtered to KG boundary)
- Path percentage should decrease for flat KGs
- `df -h` should show cache cleanup working
