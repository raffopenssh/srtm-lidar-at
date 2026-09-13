# INVEKOS Schläge API — spec for the provider VM

Consumer: `srtm-lidar-at` (segv2 label builder now; later the fleet-wide v2
reprocessing over ~8440 KGs on ~50 peers). Written from the consumer side —
implement exactly this shape; anything extra is optional.

## What we do with it (so you understand the "why")

We rasterise field polygons onto a 1 m EPSG:3035 grid, one KG (Katastral-
gemeinde, 5–450 km²) at a time, tile by tile (1500×1500 m windows), and only
need **geometry + a crop-type string** per polygon. Each polygon is then
`buffer(-1 m)` shrunk and burned with a class derived from `SNAR_BEZEICHNUNG`
via regex (WEIN→vineyard, GLÖZ HECKE→hedge, ALM/WEIDE/WIESE→grass, …). We
read ~8440 KGs × ≤3 years; a peer needs one KG's polygons in < 2 s.

**Key requirement — multiple years.** Our LiDAR is flown 2006–2024 per
region; labelling a 2009 DSM with 2024 fields is wrong. We want to request the
Schläge year closest to the flight year. AMA publishes INSPIRE Schläge yearly
(`INSPIRE_SCHLAEGE_<year>-<release>_POLYGON`, ~2.95 M polys/yr, EPSG:31287) on
data.gv.at back to ~2015. Load every year you can get; expose which ones you
have.

## Endpoints

All responses `Content-Encoding: gzip` when `Accept-Encoding: gzip` (we always
send it). CORS irrelevant (server-to-server). Read-only, no auth beyond what
the exe.dev proxy does; but accept an optional `X-Api-Token` header so we can
lock it down later without changing the client.

### `GET /api/v1/invekos/years`
```json
{"years":[2016,2018,2020,2022,2023,2024],"default":2024,
 "releases":{"2024":"2024-1","2023":"2023-2"},"crs_native":"EPSG:31287"}
```

### `GET /api/v1/invekos/schlaege`  ← the hot path
Query params:
| param | required | notes |
|---|---|---|
| `bbox` | yes | `minx,miny,maxx,maxy` in `crs` (default **EPSG:3035**). We always pass 3035. Cap: 25 km × 25 km; 400 if larger. |
| `crs` | no | `3035` (default) or `31287`. Output geometry in this CRS. |
| `year` | no | one of `/years`; default = newest. |
| `nearest_to` | no | a flight year (e.g. `2009`); server picks the closest available year (ties → earlier). Mutually exclusive with `year`. Response header `X-Invekos-Year: 2016`. |
| `fields` | no | comma list from `snar,snar_code,area_ha,fs_kennung,geo_id,bio`; default `snar,snar_code`. Keep default small — we don't want 16 attributes. |
| `format` | no | `geojson` (default) or `fgb` (FlatGeobuf, preferred once we test it) or `wkb` (see below). |
| `simplify` | no | metres, Douglas–Peucker tolerance; default 0 (no simplification — we burn at 1 m). |

**GeoJSON response** — a FeatureCollection, geometry as Polygon/MultiPolygon
in the requested CRS, coordinates rounded to **0.1 m** (1 decimal), properties
only the requested fields, plus top-level:
```json
{"type":"FeatureCollection","year":2024,"release":"2024-1","crs":"EPSG:3035",
 "n":1834,"clipped":false,"features":[
   {"type":"Feature","geometry":{...},
    "properties":{"snar":"WEIN","snar_code":901}}]}
```
* `clipped:false` — return **whole polygons intersecting the bbox**, not
  clipped (we clip ourselves; clipping creates slivers and breaks the -1 m
  shrink at the window edge). Provide `clip=1` as an option if trivial.
* Don't dedupe across bbox tiles for us — we request per KG, not per tile.

**`format=wkb`** (fastest to parse on our side, optional but nice): a
newline-free binary stream: `uint32 n`, then per feature `uint16 snar_code`,
`uint32 wkb_len`, `wkb bytes`. Plus header `X-Invekos-Year`. Let us know if
you do this; else GeoJSON gz is fine.

### `GET /api/v1/invekos/kg/<kg_code>`  (convenience, optional)
Same as `/schlaege` but bbox resolved from the KG code (you can pull the
8440 KG bboxes once from
`https://srtm-lidar-at.exe.xyz:8000/api/v1/kg/<code>` or ask us for
`kg_list.json`). Saves us a round trip; not required.

### `GET /api/v1/invekos/snar_codes?year=`
`[{"snar_code":901,"snar":"WEIN","n":48211,"area_ha":41200.5}, …]` — full
vocabulary per year with counts. We use this once per year to extend our
regex→class rules (the vocabulary changes between years, e.g. GLÖZ types
appear in 2023).

### `GET /api/v1/invekos/stats?bbox=&year=` (optional)
Per-`snar_code` area within bbox. Cheap for us to have for QA dashboards; skip
if time-constrained.

### `GET /api/v1/docs/llm.txt`
Plain-text endpoint list + examples, same convention as our other APIs.

## Performance targets
* p50 < 500 ms, p95 < 2 s for a 10 km × 10 km bbox, one year, default fields
  (~2–6 k polygons, ~1–3 MB gz GeoJSON). Fleet load: ≤ 50 concurrent peers,
  each requesting ~1 KG per 20–40 min → negligible; but the initial
  training build hits ~200 KGs in a row from one client.
* Store per year in a **spatially indexed** store: PostGIS with GiST, or
  one GeoPackage per year with the R-tree (`pyogrio.read_dataframe(bbox=…)`
  on the GPKG already does 0.1 s reads locally — that's the floor to beat or
  match). Pre-transform a 3035 geometry column per year at load time so
  requests don't reproject 2.95 M polygons on the fly.
* Geometry validity: run `ST_MakeValid`/`buffer(0)` at import; report the
  count of fixed polygons in `/years`.

## Failure semantics
* 400 on bad bbox / unknown year (JSON `{"error":…}`).
* 413 if the bbox would return > 50 k features; we then split the bbox.
* Never 200 with a truncated list. If you must cap, set `"truncated":true`
  and 206.
* 5xx: we retry with backoff; keep them rare.

## Reference client (what we'll write)
```python
r = requests.get(f"{BASE}/api/v1/invekos/schlaege",
                 params={"bbox": ",".join(map(str, bbox3035)), "nearest_to": dsm_flight_year},
                 headers={"Accept-Encoding": "gzip"}, timeout=60)
year = int(r.headers["X-Invekos-Year"])
for f in r.json()["features"]:
    geom = shape(f["geometry"]); ty = invekos_type(f["properties"]["snar"])
```
