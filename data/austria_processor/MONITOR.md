# Austria Processor Monitor Instructions

Monitor the Austria landscape processor. Check every ~14 minutes, 6 cycles.

## Quick Commands

```bash
# Status
systemctl status austria_processor

# Recent log (last 100 lines)
tail -100 /home/exedev/srtm-lidar/data/austria_processor/logs/processor.log

# Progress JSON
cat data/austria_processor/progress.json | python3 -m json.tool

# Cache sizes
du -sh data/austria_processor/bev_tile_cache/ data/austria_processor/ortho_tile_cache/ data/austria_processor/copernicus_tiles/ data/austria_processor/hansen_tiles/ 2>/dev/null
ls data/austria_processor/bev_tile_cache/ 2>/dev/null | wc -l
ls data/austria_processor/ortho_tile_cache/ 2>/dev/null | wc -l

# Output files produced
ls -la data/austria_processor/json/*.json 2>/dev/null | tail -5
ls -la /tmp/austria_processor/gpkg/*.gpkg 2>/dev/null | tail -5

# Check a completed KG JSON
python3 -c "import json; js=json.load(open('data/austria_processor/json/KGCODE.json')); print(json.dumps({k:(v if not isinstance(v,dict) else '...' ) for k,v in js.items()}, indent=2))"
```

## What to Check Each Cycle

1. **Alive?** `systemctl status austria_processor` — active (running), memory < 3.5G
2. **Progress?** progress.json: `completed`, `success`, `failed`, `rate_kgs_per_hour`
3. **Errors/retries?** Grep log for `ERROR|WARNING|timeout|retry|exhausted|FAILED`:
   ```bash
   grep -cE 'ERROR|WARNING|TIMEOUT|FAILED' data/austria_processor/logs/processor.log
   grep -c 'SUCCESS' data/austria_processor/logs/processor.log
   ```
4. **Cache efficiency?** Look for `cache hit` vs remote reads in log
5. **Copernicus?** If paused: file `data/austria_processor/copernicus_paused` exists.
   Fix: update creds in `copernicus.py` `_CREDENTIALS`, then `rm data/austria_processor/copernicus_paused`
6. **Hansen?** Should show `cached` for repeat tiles. Failures on `datamask`/`gain` are non-fatal.
7. **Validation?** After SUCCESS, log shows `FULL_GPKG`, `LIGHT_GPKG`, `JSON` stats + any issues.
8. **Per-KG stats?** After SUCCESS, log shows segments/parcels/buildings/landscape/terrain/NDVI/Hansen.
9. **JSON compliance?** Spot-check a completed JSON:
   ```bash
   python3 -c "
   import json
   js=json.load(open('data/austria_processor/json/KGCODE.json'))
   p=js.get('parcels',{})
   b=js.get('building_footprints',{})
   print(f'parcels: {p.get("count")} total, {len(p.get("details",[]))} with detail')
   print(f'buildings: {b.get("count")} total, {len(b.get("details",[]))} with detail')
   d=p.get('details',[])
   if d: print(f'first parcel keys: {list(d[0].keys())}')
   bd=b.get('details',[])
   if bd: print(f'first building keys: {list(bd[0].keys())}')
   print(f'top-level keys: {list(js.keys())}')
   "
   ```

## Expected Per-KG Timeline (with caches warm)

| Step | Time |
|------|------|
| Cadastre fetch | 2-5s |
| DTM+DSM (cached) | <1s |
| DTM+DSM (remote) | 10-30s |
| Ortho (cached) | <1s |
| Ortho (remote) | 20-40s |
| Copernicus tiles | 5-30s (or skip if exhausted) |
| Hansen tiles | <1s cached, 30-60s remote |
| Segmentation | 30-50s |
| Feature extraction | 5-17min (scales with segment count) |
| Texture (from pre-loaded ortho) | included in above |
| Terrain | 2-5s |
| Vectorise | 2-10s |
| Full GPKG | 5-15s |
| Light GPKG | 5-15s |
| JSON summary | 5-30s (per-parcel rasterization) |
| Validation | <2s |
| Zenodo upload | 10-60s |
| **Total** | **8-25 min** |

## Failure Modes

- **Timeout (30min)**: Feature extraction on huge KGs (>6M pixels, >15k segments). Non-fatal, skips to next.
- **Copernicus 402**: Credits exhausted → processor pauses. Provide new creds.
- **BEV HTTP errors**: Retries with proxies. Usually transient. Check proxy pool.
- **Hansen Google Storage**: Direct-first, proxy on retry. `datamask`/`gain` fail sometimes. Non-fatal.
- **OOM**: MemoryMax=4G in systemd. Large KGs may OOM → auto-restart.

## Key Files

| Path | Purpose |
|------|----------|
| `data/austria_processor/logs/processor.log` | Main log |
| `data/austria_processor/progress.json` | Live progress |
| `data/austria_processor/zenodo_manifest.json` | Upload tracking |
| `data/austria_processor/json/*.json` | Per-KG JSON summaries |
| `data/austria_processor/copernicus_paused` | Pause flag (delete to resume) |
| `data/austria_processor/bev_tile_cache/` | Cached DTM/DSM windows |
| `data/austria_processor/ortho_tile_cache/` | Cached ortho windows |
| `/tmp/austria_processor/gpkg/` | Temp GPKGs (deleted after upload) |

## JSON Spec Compliance Checklist

Per the original spec, each KG JSON must contain:
- [x] `area_summary` — per-type pixels/m², fraction, observation period
- [x] `height_distribution` — per-type min/max/mean/p90
- [x] `landscape` — terrain, fragmentation, shannon diversity, dominant type, vegetated fraction
- [x] `top_10_objects` — height, type, coordinate
- [x] `top_10_trees` — height, canopy height, coordinate, area
- [x] `tree_stats` — count, canopy area, mean height, stem volume estimate
- [x] `terrain` — steepness, aspect, roughness, curvature, elevation range
- [x] `ndvi` — BEV NIR mean + Copernicus mean
- [x] `hansen` — loss_by_year, total_loss, treecover2000, method
- [x] `new_buildings` — count + features with area, height, stories, roof type, centroid
- [x] `infrastructure` — by_type with count, area, features
- [x] `parcels.details[]` — per-parcel: parcel_id, area, centroid, elevation, area_summary, height_dist, vegetated_fraction
- [x] `building_footprints.details[]` — per-building: area, centroid, max/mean height, roof type, stories, segment types
- [x] `methods` — full methodology for each value
- [x] `observation_period` — start/end dates, dataset
