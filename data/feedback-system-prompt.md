# Implement: Ground-Truth Feedback & Quality-Flag System

You are working on **srtm-lidar-at** (read `/home/exedev/srtm-lidar/AGENTS.md` end-to-end before touching code — it has the full mental model: pipeline, RF features, Austria Processor, search index, cross-API bridge, peer director). Live at https://srtm-lidar-at.exe.xyz:8000/. Stack: Flask + gunicorn + SQLite FTS5/R-tree + Leaflet. Restart with `sudo systemctl restart srv`.

## Why

Our RF + rule-based classifier produces occasional outliers (e.g. a 102.2 m "tree" in KG `49006-south` Innerbreitenau — actually almost certainly a telecoms mast or a segmentation artefact spanning a cliff). We need a feedback loop that:

1. **Auto-flags implausible segments** without any human input or Copernicus calls (we have unlimited access to the GPKGs/JSONs locally and on Zenodo; Copernicus is rate-limited and reserved for the running Austria Processor).
2. **Lets students report corrections** from (a) the web UI (search results / map clicks), (b) QGIS (where they have the `*_light.gpkg` open), and (c) **bare coordinates without any segment ID** — e.g. "at lon=14.123, lat=47.456 there is a mast, not a tree".
3. **Transparently affects query results** — queries against `/api/v1/query`, `/api/v1/query/parcels`, `/api/v1/parcels/batch`, the cross-API bridge, KG profiles, etc. must reflect community corrections **without rewriting** the published GPKGs/JSONs on Zenodo (those stay frozen and reproducible).
4. **Becomes high-quality training data** for the next `train_rf_4000kg.py` run, weighted higher than cadastre/OSM labels.

No changes to the Austria Processor pipeline. No Copernicus calls. No GPKG/JSON rewrites.

## Hard constraints

- Austria Processor is running. **Do not** stop it, do not touch `austria_processor.py`, `copernicus.py`, `tile_cache.py`, `zenodo_cache.py`. Any feedback machinery must be passive metadata layered on top of frozen outputs.
- Search index DB (`data/search_index.db`) auto-rebuilds — do not put feedback rows there. Use a separate DB.
- The 25 object types and the 11 group types are defined in `object_segmentation.py` (`OBJECT_TYPES`, `GROUP_TYPES`). Reuse them; do not invent new ones (but allow a `not_a_feature` / `boundary_wrong` meta-label).
- Frontend is a single file `static/index.html` (~3100 lines, all CSS/JS inline). Keep it that way. Process dashboard is `static/process.html`.
- Feature engineering for RF lives in `learned_classifier.py` (`FEATURE_KEYS`, 57 keys) and `object_segmentation.py` (`extract_object_features`). Do not change these — only add a new label source.

## Deliverables (build them in this order; each is independently shippable)

### 1. `quality_flags.py` — automatic sanity-flagging (no humans, no network)

A module that scans a KG's light GPKG (or JSON summary) and emits flag rows. Run it across all already-processed KGs. Heuristics (tune thresholds against the existing data — show me histograms before committing thresholds):

| Flag code | Severity | Rule |
|---|---|---|
| `tree_height_implausible` | high | `type=tree` AND `ndsm_max > 60` (Austria's tallest verified ~57 m) |
| `tree_height_extreme` | critical | `type=tree` AND `ndsm_max > 80` |
| `building_height_implausible` | high | `type=roof` AND (`height > 200` OR (`height > 30` AND `footprint < 50 m²`)) |
| `mast_likely` | medium | `type=tree` AND `ndsm_max > 30` AND `footprint < 20 m²` AND `ndvi < 0.3` |
| `volume_implausible` | high | `tree.stem_vol > 50 m³` single segment |
| `class_cadastre_mismatch` | medium | predicted type contradicts cadastre landuse with both confidences high |
| `low_rf_confidence` | low | `rf_confidence < 0.4` |
| `rule_rf_disagreement` | medium | rule and RF disagree, both confident |
| `boundary_artifact` | low | segment touches tile edge AND has extreme stat (height, area, slope) |
| `ndvi_class_mismatch` | medium | `type=tree|forest|grass` AND `ndvi<0.2`; or `type=water` AND `ndvi>0.4` |
| `tiny_segment` | low | area < 4 m² (sub-pixel artefact) |
| `huge_building` | medium | `type=roof` AND footprint > 50 000 m² (verify against known industrial sites) |

Output schema (a new SQLite DB `data/feedback.sqlite`, completely separate from `search_index.db`):

```sql
CREATE TABLE flags (
  id INTEGER PRIMARY KEY,
  kg_code TEXT NOT NULL,
  segment_id INTEGER,        -- nullable; for KG-level or parcel-level flags
  parcel_id TEXT,            -- nullable
  flag_code TEXT NOT NULL,
  severity TEXT NOT NULL,    -- low|medium|high|critical
  predicted_type TEXT,
  predicted_attrs_json TEXT, -- height, area, ndvi, conf, etc.
  centroid_lon REAL,         -- always populated for spatial join
  centroid_lat REAL,
  geom_wkt TEXT,             -- segment polygon (for QGIS / map display)
  computed_at INTEGER NOT NULL,
  rule_version TEXT NOT NULL,
  UNIQUE(kg_code, segment_id, flag_code, rule_version)
);
CREATE INDEX flags_kg ON flags(kg_code);
CREATE INDEX flags_severity ON flags(severity);
CREATE INDEX flags_code ON flags(flag_code);
-- R-tree for spatial lookup by point
CREATE VIRTUAL TABLE flags_rtree USING rtree(id, min_lon, max_lon, min_lat, max_lat);
```

CLI:
```bash
python3 quality_flags.py scan-kg <kg_code>      # one KG
python3 quality_flags.py scan-all                # iterate all processed KGs (use Zenodo light GPKGs if not local)
python3 quality_flags.py stats                   # histogram of flag codes / severities
python3 quality_flags.py thresholds              # print histograms of stat distributions to help tune
```

Source preference: prefer local light GPKG; fall back to Zenodo URL from search index (`zenodo_light_gpkg_url`). Cache downloaded GPKGs in `/tmp/feedback_gpkg_cache/` with LRU eviction (cap 5 GB).

### 2. `feedback_db.py` — feedback storage + resolution

Same `data/feedback.sqlite`. Schema:

```sql
CREATE TABLE feedback (
  id INTEGER PRIMARY KEY,
  -- target: at LEAST one of (segment_ref, point) must be set
  kg_code TEXT,                -- nullable; resolved from coords if missing
  segment_id INTEGER,          -- nullable; resolved from coords if missing
  parcel_id TEXT,              -- nullable
  point_lon REAL,              -- always populated (clicked point or segment centroid)
  point_lat REAL,
  -- resolution
  resolved_segment_id INTEGER, -- after coord→segment lookup
  resolved_kg_code TEXT,
  resolved_at INTEGER,
  resolution_status TEXT,      -- pending | resolved | ambiguous | no_segment | outside_processed
  resolution_distance_m REAL,  -- distance from click to nearest segment centroid
  -- predicted (snapshot at submission time, for audit)
  predicted_type TEXT,
  predicted_attrs_json TEXT,
  -- corrected
  kind TEXT NOT NULL,          -- confirm | reject | correct_type | correct_attrs | report_missing
  corrected_type TEXT,
  corrected_attrs_json TEXT,   -- e.g. {"height": 35, "is_mast": true}
  -- author
  user_id TEXT NOT NULL,
  user_role TEXT,              -- student | trusted | admin
  confidence TEXT,             -- sure | likely | guess
  notes TEXT,
  source_app TEXT NOT NULL,    -- web_query | web_map | qgis_plugin | bulk_csv | review_queue
  created_at INTEGER NOT NULL,
  -- moderation
  status TEXT NOT NULL DEFAULT 'active', -- active | superseded | rejected_by_admin
  superseded_by INTEGER
);
CREATE INDEX fb_segment ON feedback(resolved_kg_code, resolved_segment_id);
CREATE INDEX fb_user ON feedback(user_id);
CREATE INDEX fb_created ON feedback(created_at);
CREATE VIRTUAL TABLE feedback_rtree USING rtree(id, min_lon, max_lon, min_lat, max_lat);

-- Consensus view
CREATE VIEW feedback_consensus AS
SELECT
  resolved_kg_code AS kg_code,
  resolved_segment_id AS segment_id,
  COUNT(*) FILTER (WHERE kind='confirm')                            AS n_confirms,
  COUNT(*) FILTER (WHERE kind='reject')                             AS n_rejects,
  COUNT(*) FILTER (WHERE kind='correct_type')                       AS n_corrections,
  -- mode of corrected_type weighted by user_role and confidence
  ...                                                               AS majority_corrected_type,
  AVG(CASE WHEN confidence='sure' THEN 1.0 WHEN confidence='likely' THEN 0.6 ELSE 0.3 END) AS avg_confidence,
  MAX(created_at)                                                   AS last_updated
FROM feedback
WHERE status='active' AND resolution_status='resolved'
GROUP BY resolved_kg_code, resolved_segment_id;
```

#### Coordinate→segment resolution (the key bit for the "no ID" case)

Function `resolve_point(lon, lat) -> {kg_code, segment_id, distance_m, confidence}`:

1. **KG lookup**: spatial query against `search_index.db` `kg_rtree` (already exists) — find which KG the point falls in. If none, status=`outside_processed`.
2. **Segment lookup**: open the KG's light GPKG (local first, else Zenodo) and run a point-in-polygon over the segments layer. Use rasterio/shapely with an R-tree spatial index (build it on first access, cache pickled at `/tmp/feedback_gpkg_cache/<kg>.rtree`).
3. If the point is inside a segment polygon → `resolved`. If within 5 m of a segment edge → also `resolved` with that segment, distance recorded. If between segments (gaps from raster→vector) → pick nearest within 10 m, status=`resolved`. If >10 m to any segment → `no_segment` (e.g. inside a hole; still record so we can report "missing object here").
4. If two segments equally close (boundary case) → `ambiguous`, store both candidates in `predicted_attrs_json.candidates`, surface to user for disambiguation.
5. Cache resolutions; re-resolve when the KG is re-processed (track `kg_processed_at` per resolution, invalidate if KG json mtime changes).

Always store `point_lon`/`point_lat` so feedback survives re-segmentation. A nightly job re-runs `resolve_point` for any feedback whose KG was reprocessed since last resolution.

#### `kind=report_missing`

User says "there's a building here that you missed" with just a coordinate. We store it with `resolution_status='no_segment'` and surface it as a special class of feedback for the next training run (negative samples for whatever was predicted there + positive sample of the missing class at that pixel).

### 3. API endpoints (add to `app.py`, new section `# === SECTION: Feedback ===`)

```
POST   /api/v1/feedback                    # single feedback (segment_id OR point)
POST   /api/v1/feedback/bulk               # CSV/GeoJSON upload
GET    /api/v1/feedback?kg=&user=&since=   # list, paginated
GET    /api/v1/feedback/<id>
POST   /api/v1/feedback/<id>/supersede     # user replaces own feedback
DELETE /api/v1/feedback/<id>               # admin or own (soft delete via status)
GET    /api/v1/feedback/resolve?lon=&lat=  # preview which segment a click would hit

GET    /api/v1/flags?kg=&severity=&code=&bbox=
GET    /api/v1/flags/stats                 # counts per code, per KG, per type
GET    /api/v1/flags/segment/<kg>/<seg>    # all flags for one segment
POST   /api/v1/flags/rebuild?kg=           # admin: re-run quality_flags on a KG

GET    /api/v1/review/queue                # active-learning queue (see deliverable 6)
POST   /api/v1/review/<id>/label           # one-click label from queue
```

`POST /api/v1/feedback` request body — accept ANY of these forms:

```json
// (a) by segment ref
{"kg_code":"49006-south","segment_id":12345,"kind":"correct_type","corrected_type":"mast","confidence":"sure"}

// (b) by coordinates only
{"point":{"lon":14.1234,"lat":47.4567},"kind":"correct_type","corrected_type":"mast","notes":"telecoms tower"}

// (c) coordinates + assertion that the area is unsegmented
{"point":{"lon":14.1234,"lat":47.4567},"kind":"report_missing","corrected_type":"roof","corrected_attrs":{"height":12}}

// (d) with predicted snapshot for audit (frontend includes this if available)
{"kg_code":"49006-south","segment_id":12345,"predicted_type":"tree","predicted_attrs":{"ndsm_max":102.2},"kind":"reject","notes":"this is a 35 m mast on a cliff"}
```

Resolution runs synchronously on POST (typically <100 ms with R-tree cache) so the response includes `{resolved_kg_code, resolved_segment_id, resolution_status, distance_m}`.

Auth: simple bearer-token header `X-Feedback-Token: <token>`. Tokens stored in `data/feedback_users.json` mapping token → `{user_id, role, name, email}`. Roles: `student` (default), `trusted` (weight 2×), `admin` (weight 5×, can supersede others). Generate tokens via CLI `python3 feedback_db.py mint-token --user student_42 --role student`.

Rate limit per token: 1000/day. Use existing patterns; if none, simple in-memory bucket per token.

### 4. Transparent query overrides

The entire system must prefer feedback consensus over original predictions in **all** read APIs **without** rewriting the underlying data.

- Add a SQL view `effective_segment` (in `feedback.sqlite`, ATTACH to `search_index.db` for joins) that computes per `(kg_code, segment_id)`:
  - `effective_type = COALESCE(majority_corrected_type WHERE consensus_score >= threshold, predicted_type)`
  - `effective_height = COALESCE(majority_corrected_height, predicted_height)`
  - `community_verified` boolean
  - `n_confirms`, `n_rejects`, `consensus_score`
- Consensus threshold: `(n_confirms + n_corrections) >= 2` AND `avg_confidence >= 0.6`, OR a single `trusted`/`admin` correction.
- All filter expressions in `query_compound()` (search_index.py) and the cross-API bridge (`cadastre_bridge.py`) must operate on `effective_*` when feedback exists for that segment. This requires touching `query_compound()` carefully — make sure aggregate queries (`aggregate=true`, KG profiles, conservation scores) recompute consistently.
- Response objects gain optional fields: `predicted_type`, `effective_type`, `community_verified`, `n_confirms`, `n_rejects`, `flags: ["tree_height_implausible", ...]`. Frontend renders a small badge.
- A query parameter `&include_unverified=false` lets a query exclude segments with active high-severity flags but no community confirmation — useful for student work where they want clean inputs.

Flags layer the same way: `effective_segment.flags` lists active flag codes for the segment.

**Crucial**: do not duplicate data. The view JOINs at query time. Feedback is sparse (we expect <0.1% of segments will ever get feedback). Performance budget: <50 ms added to a typical query — measure and confirm.

### 5. Frontend (web)

#### `static/index.html` — query results & map

- Each feature in the result list grows: `🚩 Report` button, `✓ Confirm` button, badge area for flags + community-verified.
- Click `Report` → modal:
  - Pre-filled with `kg_code`, `segment_id`, predicted type, predicted attributes
  - Dropdown of 25 types + `not_a_feature` + `boundary_wrong`
  - Optional attribute corrections (height, etc.)
  - Confidence radio (sure/likely/guess)
  - Notes textarea
  - Submit → POST /api/v1/feedback, optimistic UI update
- Right-click anywhere on the map → "Report what's here…" → modal with `point` mode (deliverable 2's coord-only path). The modal first calls `/api/v1/feedback/resolve?lon=&lat=` to show "This will attach to segment #12345 (tree, 102 m). Continue?" — user can override and pick `report_missing` instead.
- New legend filter: `Hide community-rejected`, `Show only flagged`.
- Token entry: a small "🔑 Feedback token" link in the sidebar that stores token in `localStorage`. Anonymous users can browse but not submit.
- Show flag badges on map markers (red dot for critical, orange for high).

#### `static/process.html` — quality dashboard

New section "Data Quality":
- Total flags by severity (cards)
- Top 20 KGs by flag rate
- Top flag codes (bar chart)
- Recent feedback (live feed)
- Per-class confusion matrix from feedback (predicted vs corrected)

#### `/review` — active-learning queue

New page (or section in process.html). Shows segments where:
- `rf_confidence ∈ [0.4, 0.7]` AND has at least one flag AND has no feedback yet
- Ranked by severity + geographic dispersion (so reviewers don't all label the same KG)

One-click label UI: predicted type pre-selected, photo (BEV ortho thumbnail via existing `/api/v1/ortho/overlay`), 4-button choice (Confirm / Wrong type → submenu / Not a feature / Skip). Logs as `source_app=review_queue`.

### 6. QGIS plugin

A minimal QGIS plugin in `tools/qgis_feedback_plugin/`:
- `metadata.txt`, `__init__.py`, `feedback.py`
- Adds a toolbar button "Report to srtm-lidar" + right-click action on layer features
- When triggered with a feature selected from a `*_light.gpkg` segments layer: reads `kg_code` (from filename) and `segment_id` (from feature) → opens a QDialog with the same fields as the web modal → POST to `/api/v1/feedback`
- When triggered with no feature (just a map click): captures coord, POST with `point` only
- Token stored in QSettings (`srtm-lidar/feedback_token`)
- Endpoint configurable (default https://srtm-lidar-at.exe.xyz:8000)

Provide a `README.md` with install instructions (drop into `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/` or use Plugin Builder format).

### 7. Bulk import

`POST /api/v1/feedback/bulk` accepts a CSV with columns `lon, lat, [kg_code], [segment_id], kind, corrected_type, [height], [confidence], [notes]`. Or a GeoJSON FeatureCollection where each feature's properties hold the same fields. Returns a per-row resolution report. Cap 10k rows per request.

Provide a Python helper `feedback_db.py import-csv path.csv --user student_42`.

### 8. Training integration

Add to `train_rf_4000kg.py` (do NOT run it — leave the existing rf_train service untouched):

```python
from feedback_db import export_training_labels

labels_feedback = export_training_labels(
    min_consensus=2,                # at least 2 users agree
    min_confidence='likely',
    include_report_missing=True,    # negative-sample mining
    include_admin_singletons=True,  # admin/trusted single labels also count
)
# returns rows: (kg_code, segment_id OR point, label, weight, source='feedback')

# In the existing label-merge logic, append these with weight 5.0 (cadastre=3.0, OSM=1.0)
```

Provide an `export_training_labels` function in `feedback_db.py` plus a CLI:
```bash
python3 feedback_db.py export-labels --out data/feedback_labels_v1.npz
```

For `report_missing` rows: emit a positive sample at the point (corrected_type) AND a negative sample for whatever the segment at that point currently predicts. The trainer's existing feature-extraction (`extract_object_features`) handles the rest.

Version the export: each NPZ records its hash, and a future RF model file can be tagged `model_v3_2026-XX-XX_fbHASH.joblib` so we can A/B old vs new.

### 9. Tests

- `tests/test_quality_flags.py` — feed synthetic GPKG with known outliers (102 m tree, 0 m² building, etc.), assert correct flags fire.
- `tests/test_feedback_resolve.py` — known KG, click coordinates, assert correct segment resolved; click outside any segment, assert `no_segment`.
- `tests/test_feedback_consensus.py` — 1 user reject ≠ consensus; 2 users agree → consensus; 1 admin → consensus.
- `tests/test_query_overrides.py` — submit feedback that changes a segment's type, assert `/api/v1/query?type=mast` now returns it and `?type=tree` does not.
- `tests/test_aggregate_overrides.py` — KG profile aggregates change after consensus correction.
- `tests/test_bulk_import.py` — CSV with mixed segment_ref + point rows.

Run: `python3 -m pytest tests/ -k feedback or flags`.

### 10. Operational

- `data/feedback.sqlite` should be backed up (add to whatever backup we have, or document that it must be).
- Add a `feedback_db.py status` CLI: total feedback, resolved %, top contributors, recent activity.
- `/api/v1/feedback/stats` returns the same as JSON for dashboard use.
- Document everything new in `AGENTS.md` under a new `## Ground-Truth Feedback System` section before "Developing".

## What to do first

1. Read `AGENTS.md` cover to cover.
2. Read `object_segmentation.py` `OBJECT_TYPES` + `extract_object_features` + `classify_object`.
3. Read `learned_classifier.py` `FEATURE_KEYS` + `CADASTRE_TO_TYPE`.
4. Read `search_index.py` `query_compound` + the kg/segment schema.
5. Inspect a real light GPKG to confirm what columns/properties segments expose: `ogrinfo -al -so data/austria_processor/gpkg/49006-south_light.gpkg` (or pull from Zenodo if not local).
6. Build deliverable 1 (`quality_flags.py`) and run it on `49006-south` to confirm the 102 m tree is flagged. **Show me the output before proceeding.**
7. Then deliverable 2 + 3 (storage + API). Show me a working `POST /api/v1/feedback` with a coord-only payload resolving correctly.
8. Then deliverable 4 (transparent overrides). Demonstrate that a feedback correction changes a `/api/v1/query?type=...` result.
9. Then 5/6/7/8/9/10 in any sensible order; commit between each.

## Style

- Match the existing code style (Flask blueprints inline in app.py, snake_case, type hints where helpful but not enforced).
- Don't introduce new heavy deps. Use stdlib + what's already in `requirements.txt` (rasterio, shapely, numpy, sqlite3, requests, flask, fiona).
- Each commit: descriptive message, runs `python3 -c 'import app'` cleanly, passes added tests.
- After each deliverable: restart srv, smoke-test the endpoint, append a one-line note to `AGENTS.md` if a new endpoint or DB lands.

## Open questions to resolve up front

Before writing code, send me your answers to:

1. Does `search_index.db` already store enough per-KG geometry to resolve a point to a KG quickly? (If yes, reuse; if not, propose adding it.)
2. What's the actual column layout of the segments layer in light GPKGs? Confirm `segment_id`, `type`, `ndsm_max`, `rf_confidence`, etc. exist.
3. For coord-based resolution at scale (1000s of feedback/day), is the per-KG R-tree cache approach OK, or should we precompute a single flat segments R-tree across all KGs? Estimate sizes.
4. Should `effective_type` overrides also propagate into the Zenodo-published JSONs the next time a KG is re-processed? (My instinct: no — Zenodo stays frozen, feedback always layers on top. Confirm.)
5. Throttle/abuse: a malicious student could spam corrections. The consensus threshold + role weights handle most of it; do you want a stricter "admin must approve trusted role assignments" workflow?

Ask me if anything is unclear. Do not start coding before I answer the open questions.
