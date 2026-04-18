# Refactor: extract shared module + split austria_processor.py

Two tasks. Do task 1 first, restart srv, verify it works, then do task 2.

## Task 1: Extract shared constants into `segment_types.py`

`SEGMENT_COLORS` is copy-pasted in `app.py` (line ~2028) and `austria_processor.py` (line ~1096). They're identical now but `_height_class()` has already diverged between the two files. Three other functions are also duplicated: `_viridis_rgb()`, `_write_gpkg_categorized_style()`. The frontend (`static/index.html` line 815-818) has its own JS copy that must stay in sync.

Create `segment_types.py` containing:
- `SEGMENT_COLORS` — the canonical type→RGBA dict (move from austria_processor.py)
- `_height_class()` — use the app.py version (it has the finer 30-35m/35-40m split)
- `_viridis_rgb()` — identical in both files

Then in both `app.py` and `austria_processor.py`:
- `from segment_types import SEGMENT_COLORS, _height_class, _viridis_rgb`
- Delete the local definitions

Do NOT touch `_write_gpkg_categorized_style()` — it's used differently in each file (app.py writes to temp GPKG exports, austria_processor.py writes to per-KG GPKGs with different style logic). Leave as duplicated for now.

After this change:
```bash
sudo systemctl restart srv
python3 -c "from segment_types import SEGMENT_COLORS; print(len(SEGMENT_COLORS), 'types')"
curl -s http://localhost:8000/api/v1/info | head -5  # verify app starts
```

## Task 2: Split austria_processor.py into 3 files

The processor MUST be stopped first:
```bash
sudo systemctl kill -s SIGKILL austria_processor
sleep 2
```

Split the ~5100 lines into:

### `austria_processor.py` (~2000 lines) — orchestration only
Keep:
- Config section
- Disk cache management
- Logging + directories
- ProgressTracker class
- Circuit breaker
- `get_all_kgs()`
- `process_one_kg()` — the core per-KG pipeline (lines ~2888-3900)
- Output validation section
- Zenodo upload helpers
- JSON dir cleanup
- `main()` — the whole main loop

Import from the new modules.

### `kg_builders.py` (~1800 lines) — GPKG + JSON output builders
Move:
- `SEGMENT_COLORS` import (from segment_types.py now)
- All GPKG style/vector writers: `_write_segment_vectors()`, `_write_segment_points()`, `_write_gpkg_point_style()`, `_write_gpkg_all_styles()`, `_write_gpkg_categorized_style()`
- `build_full_gpkg_tiled()`
- `build_light_gpkg_tiled()`
- `compute_data_quality()`
- `build_json_summary_tiled()`
- Helper functions they use: `_find_tile_for_point()`, `_read_dtm_for_tile()`, `_height_class` import, `_viridis_rgb` import, `_to_multi()`

These functions are called from `process_one_kg()`, so the import is: `from kg_builders import build_full_gpkg_tiled, build_light_gpkg_tiled, build_json_summary_tiled, compute_data_quality`

### `kg_enrichment.py` (~800 lines) — cadastre enrichment + vectorisation
Move:
- Geometry helpers: `transform_to_3035()`, `transform_to_wgs()`
- `fetch_cadastre_data()`
- `enrich_parcels_with_heights()`
- `enrich_buildings_with_heights()`
- `_segment_touches_edge()`
- `vectorise_unmatched_buildings()`
- `vectorise_infrastructure()`
- `resolve_edge_clipped_features()`

These are called from `process_one_kg()`: `from kg_enrichment import fetch_cadastre_data, enrich_parcels_with_heights, enrich_buildings_with_heights, vectorise_unmatched_buildings, vectorise_infrastructure, resolve_edge_clipped_features, transform_to_3035, transform_to_wgs`

### Important details

- Each new file gets `# === SECTION:` markers matching the ones currently in austria_processor.py
- `process_one_kg()` does lazy imports inside the function body (because it runs in a subprocess). The new modules should also be imported lazily inside `process_one_kg()` — NOT at module top level.
- The `DATA_DIR`, `GPKG_DIR`, `JSON_DIR` constants are needed by builders. Either pass them as args or import from austria_processor. Passing as args is cleaner.
- `log = logging.getLogger(__name__)` in each new file.
- Keep `_compute_tile_grid()` in austria_processor.py (it's part of the pipeline logic in process_one_kg).

### After the split

```bash
# Verify all files compile
python3 -m py_compile austria_processor.py
python3 -m py_compile kg_builders.py
python3 -m py_compile kg_enrichment.py
python3 -m py_compile segment_types.py

# Restart processor
sudo systemctl start austria_processor
sleep 10
systemctl status austria_processor  # should be active (running)
tail -20 data/austria_processor/logs/processor.log  # should show normal startup

# Restart app server
sudo systemctl restart srv
curl -s http://localhost:8000/api/v1/info | head -5
```

### Update AGENTS.md

- Update File Layout table (new files + updated line counts)
- Update austria_processor Code Map (remove moved sections, add imports)
- Add brief entries for kg_builders.py and kg_enrichment.py
- Update Cross-Cutting Concerns if needed
- `segment_types.py` is now the canonical source for SEGMENT_COLORS
