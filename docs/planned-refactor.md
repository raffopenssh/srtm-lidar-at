# Planned Refactor (next maintenance window)

Detailed prompt in `data/next-prompt.md`. Requires stopping the processor.

**Step 1 — Extract `segment_types.py`** (safe, processor can stay running):
- Move `SEGMENT_COLORS`, `_height_class()`, `_viridis_rgb()` out of `app.py` + `austria_processor.py` into a shared module.
- Fixes: `_height_class()` has already diverged between the two copies.

**Step 2 — Split `austria_processor.py`** (stop processor first):
- `austria_processor.py` (~2000L) — orchestration: `main()`, `process_one_kg()`, retry logic, Zenodo upload
- `kg_builders.py` (~1800L) — `build_full_gpkg_tiled()`, `build_light_gpkg_tiled()`, `build_json_summary_tiled()`, GPKG style/vector writers
- `kg_enrichment.py` (~800L) — `fetch_cadastre_data()`, height enrichment, vectorisation, edge-clip resolution

Gotchas: lazy imports inside `process_one_kg()` (subprocess boundary), pass `DATA_DIR`/`GPKG_DIR` as args to builders.

---

### Copernicus Speed Optimisation (next maintenance window)

Stop processor first: `sudo systemctl stop austria_processor`

**Context:** CDSE gives each account 5 independent quota pools (see
https://documentation.dataspace.copernicus.eu/Quotas.html). We only use openEO.
The others sit at 0% utilisation. Current Copernicus fetch is ~47 min/tile
(sequential, single credential). Target: ~8 min/tile via parallelism + offloading.

| Pool | Per account | 4 accounts | Current use |
|------|------------|------------|-------------|
| openEO credits | 10k/month | 40k | partial (sequential) |
| Sentinel Hub PU | 10k/month | 40k | **0%** |
| Sentinel Hub requests | 10k/month | 40k | **0%** |
| Direct COG HTTP | 50k/month | 200k | **0%** |
| S3 bandwidth | 12 TB/month | 48 TB | **0%** |

Current per-tile breakdown (all sequential on 1 credential):
- NDVI composite: ~10 min (1 openEO job)
- WorldCover: ~3 min (1 openEO job)
- SAR backscatter: ~5 min (1 openEO job)
- NDVI time series: ~29 min (8 openEO jobs, 1/month)

**Do these steps in order. Each step is independently deployable.**

#### Step 3 — WorldCover via direct AWS COG (easy win, ~30 min work)

ESA WorldCover v200 2021 is hosted as public COGs on AWS. No CDSE auth needed.
Already verified: `rasterio.open(url).read(window=...)` works.

Saves: ~3 min/tile + openEO credits. Effort: low.

**What to do:**

1. Create `worldcover_cog.py` (~80 lines). Single function:
   ```python
   def get_land_cover_cog(bbox_wgs84: dict) -> dict:
       """Fetch WorldCover via direct COG HTTP range read from AWS.
       Returns same format as copernicus.get_land_cover():
       {"map": np.ndarray(H,W, uint8), "transform": Affine, "crs": CRS,
        "classes": WORLDCOVER_CLASSES}
       """
   ```
   Tiles are 3°×3° COGs at:
   `https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N{lat}E{lon:03d}_Map.tif`

   Austria needs 6 tiles: N45E009, N45E012, N45E015, N48E009, N48E012, N48E015.
   Each is 36000×36000 px, EPSG:4326, uint8.

   Implementation:
   - Compute which tile(s) the bbox falls in (floor lat to multiple of 3, same for lon)
   - `rasterio.open(url)` with `/vsicurl/` env or plain HTTPS (both work)
   - `ds.read(1, window=from_bounds(*bbox, ds.transform))` for windowed read
   - If bbox spans two tiles (rare, only at 12°E or 15°E), read both + mosaic
   - Return `{"map": data, "transform": window_transform, "crs": CRS.from_epsg(4326), "classes": WORLDCOVER_CLASSES}`
   - Cache in `rf_training_data/copernicus_cache/` using same hash scheme as `copernicus.py`
   - Retry with exponential backoff on HTTP errors (use `bev_retry.py` pattern)

2. In `tile_cache.py`, modify `CopernicusTileCache.get_landcover()` (~line 296):
   - Try `worldcover_cog.get_land_cover_cog(tile_bbox)` first
   - Fall back to `copernicus.get_land_cover()` on failure
   - The cache layer (`_atomic_savez`) stays the same — transparent to callers

3. In `copernicus.py`, update `get_land_cover()` similarly as fallback.
   `app.py` calls `copernicus.get_land_cover()` directly for interactive use.

Test: `python3 -c "from worldcover_cog import get_land_cover_cog; r = get_land_cover_cog({'west':15,'south':47.5,'east':15.1,'north':47.6}); print(r['map'].shape, set(r['map'].flatten()[:100]))"`

#### Step 4 — Parallel openEO across 4 credentials (biggest win, ~2 hours work)

openEO allows **2 concurrent processing jobs per account**. We have 4 accounts
= 8 concurrent slots. Currently `_fetch_copernicus_for_tile()` runs NDVI→WC→SAR→harmonics
sequentially through one credential. The NDVI time series alone (8 months) takes
~29 min because each month waits for the previous.

Saves: ~29 min → ~8 min for NDVI TS (3-4× speedup). Effort: medium.

We had parallel downloads before and they "often failed for large areas" — but
that was large-area batch jobs, not the 0.1° tiles we use now. At 0.1° tile size,
parallel sync downloads are safe.

**What to do:**

1. In `copernicus.py`, refactor `get_ndvi_timeseries()` (~line 669):
   - Currently: `for label, m_start, m_end, month_cache in to_download:` (sequential)
   - Change to: `ThreadPoolExecutor(max_workers=8)` submitting `_download_month_sequential()`
   - Each worker gets a dedicated `cred_index` via round-robin: `cred_index = i % len(FUNCTIONING_CREDENTIALS())`
   - Each worker calls `_get_connection_for_cred(cred_index)` — this already exists and creates per-credential sessions
   - Max 2 workers per credential (openEO concurrency limit)
   - On 402/CreditsExhausted from one credential, remove it from the pool, redistribute remaining work
   - Add 5-second stagger between submissions (openEO rate limit: 1 req/5s per account, but we use different accounts)

2. In `_download_month_sequential()` (~line 764), add `cred_index` parameter:
   - `c = _get_connection_for_cred(cred_index)` instead of `c = _get_connection()`
   - Error handling already works per-credential (402 detection, rotation)

3. Also parallelise the non-TS products within `_fetch_copernicus_for_tile()`
   in `austria_processor.py` (~line 4153):
   - `_try_fetch_single()` currently calls NDVI → WorldCover → SAR → harmonics sequentially
   - Refactor: submit NDVI, WorldCover (now COG, instant), and SAR as 3 concurrent futures
   - WorldCover goes to `worldcover_cog` (no credential needed)
   - NDVI composite uses credential A
   - SAR uses credential B
   - Then harmonics (NDVI TS) uses all credentials in parallel (step above)
   - Wait for all, merge results into `cop` dict

4. In `tile_cache.py`, add `cred_index` parameter to `get_ndvi()`, `get_sar()`,
   `get_harmonics()` — they already accept it, just ensure it's passed through.

Key constraint: openEO rate limit is 1 request per 5 seconds **per account**
(footnote 13 on quotas page). Different accounts can fire simultaneously.
So with 4 accounts: 4 requests per 5 seconds = 48/min.

Test: Process a single KG with `POST /api/v1/processing/single?kg=XXXXX` and
watch the processor log. NDVI months should appear interleaved across credentials
instead of sequential.

#### Step 5 — NDVI composite via Sentinel Hub Process API (separate quota pool, ~3 hours work)

The SH Process API uses Evalscripts (server-side JS) to compute NDVI with cloud
masking and return a GeoTIFF in ~10 seconds. Uses the **Sentinel Hub PU quota**
which is completely separate from openEO credits.

Saves: ~10 min → ~10 sec for NDVI composite. PU cost: ~19/tile.

Our credentials (`sh-*` client IDs) already work for Sentinel Hub — same OAuth.
Auth endpoint: `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token`
Process endpoint: `https://sh.dataspace.copernicus.eu/api/v1/process`

**What to do:**

1. Create `sentinelhub_client.py` (~200 lines):

   ```python
   def get_ndvi_composite_sh(
       bbox_wgs84: dict,
       year: int = 2024,
       cred_index: int = 0,
   ) -> dict:
       """Fetch NDVI composite via Sentinel Hub Process API.
       Returns same format as copernicus.get_ndvi_composite():
       {"ndvi": np.ndarray(H,W, float32), "transform": Affine, "crs": CRS}
       """
   ```

   Evalscript for NDVI composite (cloud-masked temporal median):
   ```javascript
   //VERSION=3
   function setup() {
     return {
       input: [{ bands: ["B04", "B08", "SCL"], units: "DN" }],
       output: { bands: 1, sampleType: "FLOAT32" },
       mosaicking: Mosaicking.ORBIT
     };
   }
   function evaluatePixel(samples) {
     let validNDVI = [];
     for (let i = 0; i < samples.length; i++) {
       let scl = samples[i].SCL;
       // Skip clouds, shadows, snow, saturated
       if ([0,1,3,8,9,10,11].includes(scl)) continue;
       let b04 = samples[i].B04, b08 = samples[i].B08;
       if (b04 + b08 === 0) continue;
       validNDVI.push((b08 - b04) / (b08 + b04));
     }
     if (validNDVI.length === 0) return [NaN];
     validNDVI.sort((a, b) => a - b);
     let mid = Math.floor(validNDVI.length / 2);
     let median = validNDVI.length % 2 !== 0
       ? validNDVI[mid]
       : (validNDVI[mid - 1] + validNDVI[mid]) / 2;
     return [median];
   }
   ```

   Request body:
   ```json
   {
     "input": {
       "bounds": {"bbox": [west, south, east, north], "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
       "data": [{
         "type": "sentinel-2-l2a",
         "dataFilter": {"timeRange": {"from": "2024-04-01T00:00:00Z", "to": "2024-09-30T23:59:59Z"}}
       }]
     },
     "output": {
       "width": <pixels>,
       "height": <pixels>,
       "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
     },
     "evalscript": "<evalscript above>"
   }
   ```

   Compute width/height from bbox at 10m resolution:
   `width = round((east - west) * 111000 * cos(lat_mid_rad) / 10)`
   `height = round((north - south) * 111000 / 10)`

   Auth: POST to token endpoint with `grant_type=client_credentials`,
   `client_id=<sh-xxx>`, `client_secret=<secret>`. Token valid 10 min.
   Cache token per credential. Standard `requests` library, no `openeo` dependency.

   Parse response: raw TIFF bytes → `rasterio.MemoryFile` → read band 1.
   Construct Affine transform from bbox + pixel dimensions.

   PU cost estimate: ~19 PU per 0.1° tile (area_factor 3.13 × 1 band_factor × 6 samples × 1 INT16).
   At 40k PU/month across 4 accounts: ~2100 tiles/month from SH alone.

2. In `tile_cache.py`, modify `CopernicusTileCache.get_ndvi()` (~line 236):
   - Try `sentinelhub_client.get_ndvi_composite_sh(tile_bbox, year, cred_index)` first
   - Fall back to `copernicus.get_ndvi_composite()` on failure or if SH PU exhausted
   - Same cache layer — transparent to callers

3. In `copernicus.py`, update `get_ndvi_composite()` similarly for interactive use.

Test: `python3 -c "from sentinelhub_client import get_ndvi_composite_sh; r = get_ndvi_composite_sh({'west':15,'south':47.5,'east':15.1,'north':47.6}); print(r['ndvi'].shape, f'range=[{r[\"ndvi\"].min():.2f},{r[\"ndvi\"].max():.2f}]')"`

#### Step 6 — Pipeline all products concurrently (quick win after steps 3-5)

Once steps 3-5 are done, the products use different backends:
- WorldCover: AWS COG (no quota)
- NDVI composite: Sentinel Hub (SH PU quota)
- SAR: openEO (openEO credits)
- NDVI time series: openEO parallel (openEO credits, all 4 credentials)

They can all run simultaneously.

**What to do:**

Refactor `_try_fetch_single()` in `austria_processor.py` (~line 4175):
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _try_fetch_single(bbox, label=""):
    cop = {}
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures["worldcover"] = pool.submit(worldcover_cog.get_land_cover_cog, bbox)
        futures["ndvi"] = pool.submit(cop_cache.get_ndvi, bbox, obs_year)  # → SH
        futures["sar"] = pool.submit(cop_cache.get_sar, bbox, obs_year)
        # Wait for these 3 (~seconds for WC+NDVI, ~5 min for SAR)
        for key, fut in futures.items():
            try:
                result = fut.result(timeout=600)
                # merge into cop dict...
            except Exception as e:
                log.warning("%s%s failed: %s", label, key, e)
    # Then harmonics (uses all credentials in parallel internally)
    if cop:
        harm = cop_cache.get_harmonics(bbox, year=obs_year, ...)
        if harm is not None:
            cop["harmonics"] = harm
    return cop if cop else None
```

Expected per-tile time after all steps:
- WorldCover: <2 sec (COG)
- NDVI composite: ~10 sec (SH) } all concurrent
- SAR: ~5 min (openEO)         } ← bottleneck
- NDVI time series: ~8 min (parallel openEO across 4 credentials)
- **Total: ~8 min/tile (down from ~47 min)**

#### Summary of expected impact

| Metric | Before | After |
|--------|--------|-------|
| Time per tile | ~47 min | ~8 min |
| Time per KG (3 tiles) | ~2.3 hours | ~25 min |
| Remaining 6000 KGs | ~583 days | ~104 days |
| openEO credits used | all 4 products | SAR + NDVI TS only |
| SH PU used | 0 | NDVI composite |
| COG requests used | 0 | WorldCover |

---

*See `AGENTS.md` for the project map.*
