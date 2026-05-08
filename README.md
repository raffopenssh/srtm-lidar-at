# srtm-lidar-at

Landscape segmentation for Austria from BEV LiDAR + ortho + Sentinel-2/1 +
ESA WorldCover + cadastre. Flask + Leaflet web app, plus a background
pipeline that processes all ~8440 Austrian cadastral communities (KGs) and
publishes results to Zenodo.

**Live**: https://srtm-lidar-at.exe.xyz:8000/

## What it does

- **Interactive UI** (`/`) — draw a polygon, get a per-object segmentation
  with 25 classes (trees, buildings, roads, crops, water, …) plus terrain,
  NDVI, SAR, Hansen forest-change overlays.
- **Query Explorer** (`/query.html`) — landscape-first compound queries over
  ~8440 KGs and millions of cadastral parcels. 70+ numeric filters
  (terrain, vegetation, NDVI harmonics, SAR, buildings, classification
  quality, …).
- **Processor dashboard** (`/process.html`) — live KG pipeline status,
  multi-VM peer orchestration, throttle/credentials, retry queues.
- **API** — `GET /api/v1/docs/llm.txt` for the canonical machine-readable
  endpoint list.

## Pipeline (per polygon)

1. Window-read DTM + DSM from BEV via `/vsicurl/`
2. Pull ortho RGBI, Sentinel-2 NDVI, ESA WorldCover, Sentinel-1 SAR,
   Hansen GFC, cadastre, OSM
3. Fused gradient → Felzenszwalb segmentation + RAG merge
4. 44 features per segment → Random Forest classify (cadastre-trained)
5. Group adjacent compatible segments (tree→forest, roof→building)
6. Emit GeoJSON + overlays + GeoPackage/KML/MBTiles export

Details: `object_segmentation.py`, `learned_classifier.py`.

## Stack

Python 3.12 · Flask · gunicorn · Leaflet · rasterio · scikit-learn ·
scikit-image · SQLite (FTS5 + R-tree) · openEO (Copernicus) · Zenodo.

## Services (systemd)

| Unit | Role |
|---|---|
| `srv.service` | gunicorn web app + peer-director thread (`:8000`) |
| `austria_processor.service` | KG pipeline (managed by director) |
| `rf_train.service` | Random Forest retraining |

```bash
sudo systemctl restart srv
journalctl -u srv -f
tail -f data/austria_processor/logs/processor.log
```

## Multi-VM orchestration

A peer director on the primary VM dispatches KGs to peer exe.dev VMs.
One frontier worker at a time (Copernicus credential safety); many
cache-only peers in parallel. HA failover via `director_ha.py`. All
Zenodo writes serialise through a mutex broker.

See `docs/peer-director.md`, `docs/austria-processor.md`,
`docs/copernicus-throttle.md`.

## Data sources

| Source | Resolution | Access |
|---|---|---|
| BEV ALS DTM/DSM | 1 m, 2022/23/24 | HTTP range on remote GeoTIFF |
| BEV DOP RGBI | 0.2 m | HTTP range on remote GeoTIFF |
| Sentinel-2 / WorldCover / Sentinel-1 | 10 m | openEO (CDSE) |
| Hansen GFC | 30 m | `/vsicurl/` UMD |
| Austrian cadastre | mm | REST API |
| OSM | varies | Overpass |

Persistent tile cache on Zenodo (deposit 19650075).

## Repo guide

Start with **`AGENTS.md`** — it's the navigation hub for the codebase, with
an index into `docs/*.md` for each subsystem.
