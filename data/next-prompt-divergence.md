# Add per-parcel and per-building RF divergence to KG JSON summary

## Context

We just added a KG-level `classification` section to the JSON summary in
`austria_processor.py:build_json_summary_tiled()` (around line 3721). It contains
aggregated RF stats: `rf_classified_pct`, `mean_confidence`, `diverged_pct`,
`top_divergences`, and `per_type_confidence`.

However, the **per-parcel details** and **per-building details** in the same JSON
don't include any RF classification info. They already loop over segments within
each parcel/building to build `area_summary` and `segment_types`, but they only
track `obj.obj_type` — they ignore `obj.rf_type`, `obj.rf_confidence`, and
`obj.classifier_source`. We need divergence and confidence at parcel/building
granularity so users can find *which specific parcels* have the most disagreement
between RF and final classification.

## What to do

### 1. Per-parcel detail enrichment (~line 3982 in austria_processor.py)

In `build_json_summary_tiled()`, the per-parcel loop iterates segments inside each
parcel. Currently it builds `area_summary` (type→pixel counts) and `height_distribution`.

Add to each parcel detail dict (`pd`):

```python
pd["classification"] = {
    "rf_classified": <count of segments with classifier_source=="rf">,
    "total_segments": <count of segments in parcel>,
    "mean_confidence": <mean obj.confidence across segments>,
    "rf_mean_confidence": <mean obj.rf_confidence for RF segments>,
    "diverged_count": <count where obj.rf_type != '' and obj.rf_type != obj.obj_type>,
    "diverged_pct": <percentage>,
    "divergences": [
        {"rf_type": "grass", "final_type": "crop", "count": N}, ...
    ]  # only include if non-empty, top 5
}
```

The segment loop already exists at ~line 3988. Currently:
```python
for lbl in np.unique(pl):
    obj = obj_map.get(int(lbl))
    if obj:
        npx = int((pl==lbl).sum())
        tc_[obj.obj_type] += npx
        th_[obj.obj_type].append(obj.height_max)
```

Extend this loop to also collect RF info per segment:
```python
_parcel_objs = []  # collect segment objects for this parcel
for lbl in np.unique(pl):
    obj = obj_map.get(int(lbl))
    if obj:
        npx = int((pl==lbl).sum())
        tc_[obj.obj_type] += npx
        th_[obj.obj_type].append(obj.height_max)
        _parcel_objs.append(obj)
```

Then after the existing area_summary/height_distribution/vegetated_fraction code,
add the classification stats from `_parcel_objs`.

**IMPORTANT:** This segment analysis currently only runs in the `else` branch
(when `tr and tdata` is False, i.e. parcel centroid has no matching tile). The main
`if tr and tdata:` branch only reads elevation. For the happy path (tile found),
you need to add the same segment analysis there too. The tile data is available:
`tr["labels"]` and `tdata["ndsm"]` exist. Rasterize the parcel geometry into the
tile, extract labels, look up objects, and compute the classification dict.

Be careful: don't duplicate all the code. Extract a helper function like:
```python
def _parcel_segment_stats(geom_3035, labels, ndsm, transform, shape, obj_map):
    """Compute area_summary, height_dist, classification for a parcel."""
    from rasterio.features import rasterize as rio_rasterize
    pm = rio_rasterize([(geom_3035,1)], out_shape=shape,
                       transform=transform, fill=0, dtype=np.uint8,
                       all_touched=True).astype(bool)
    pl = labels[pm]; pn = ndsm[pm]
    # ... type counts, height dist, classification ...
    return {"area_summary": ..., "height_distribution": ..., 
            "vegetated_fraction": ..., "is_vegetated": ...,
            "ndsm_max_m": ..., "ndsm_mean_m": ...,
            "classification": ...}
```
Call it from both branches.

### 2. Per-building detail enrichment (~line 4072)

Similar pattern. The building loop already has:
```python
if tr.get("labels") is not None and objects:
    bl = tr["labels"][bm]; tc_ = Counter()
    for lbl in np.unique(bl):
        obj = obj_map.get(int(lbl))
        if obj: tc_[obj.obj_type] += int((bl==lbl).sum())
    if tc_: bd["segment_types"] = {t:px for t,px in tc_.most_common()}
```

Extend to also add:
```python
bd["classification"] = {
    "rf_classified": ...,
    "total_segments": ...,
    "mean_confidence": ...,
    "rf_mean_confidence": ...,
    "diverged_count": ...,
    "divergences": [...]  # if any
}
```

### 3. Search index: per-parcel divergence queries

The search index doesn't need per-parcel data (too granular). But update the
llm.txt to document that per-parcel and per-building `classification` dicts
now exist in KG JSON details.

### 4. After all changes, restart the austria processor:

```bash
sudo systemctl kill -s SIGKILL austria_processor && sleep 2 && sudo systemctl start austria_processor
```

Verify it starts cleanly:
```bash
sleep 3 && systemctl status austria_processor | head -10
tail -5 data/austria_processor/logs/processor.log
```

## Files to modify

| File | What |
|------|------|
| `austria_processor.py` | Per-parcel + per-building classification dicts in `build_json_summary_tiled()` |
| `llm.txt` | Document new parcel/building classification fields |

## Verification

```bash
# Syntax check
python3 -c "import py_compile; py_compile.compile('austria_processor.py', doraise=True)"

# Restart processor
sudo systemctl kill -s SIGKILL austria_processor && sleep 2 && sudo systemctl start austria_processor

# Check it's running
sleep 3 && systemctl status austria_processor | head -10
tail -5 data/austria_processor/logs/processor.log
```

When a KG completes, check the JSON:
```bash
python3 -c "
import json; d = json.loads(open('data/austria_processor/json/91109.json').read())
for p in d['parcels']['details'][:3]:
    print(p.get('parcel_id'), p.get('classification'))
for b in d['building_footprints']['details'][:3]:
    print(b.get('building_id'), b.get('classification'))
"
```
