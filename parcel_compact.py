"""Compact per-parcel object representation.

Shared between austria_processor (writer) and the backfill script,
plus any consumer that wants to decode the compact form.

Why compact?
  Per-parcel ``top_5_objects`` and ``top_5_trees`` add up across 8000+ KGs
  with 200-500 parcels each. The verbose dict-per-entry form costs ~220 bytes;
  the compact array form costs ~60 bytes — an ~80% saving.

Format (lists, positional):
  parcel['top_objs'][i] = [type_letter, height_max_m, height_mean_m,
                           area_sqm, lon, lat, conf, rf_conf, manmade_int]
  parcel['top_trees'][i] = [height_max_m, canopy_height_m, height_p90_m,
                            area_sqm, lon, lat, ndvi_mean, ndvi_fused,
                            height_change_m, phenology_class, conf, rf_conf]

Fraction Area Vector (frav):
  parcel['frav'] = {type_letter: area_sqm_int, ...}
  Always present when any segments fall in the parcel — even when there's
  no full ``area_summary``. Lets every parcel be queried by composition.
"""
from __future__ import annotations

# Single-character type abbreviations used in `frav` and `top_objs[0]`.
# Lowercase = natural / vegetated, uppercase = man-made / disturbed.
TYPE_LETTER = {
    'tree': 't', 'shrub': 's', 'grass': 'g', 'hedge': 'h',
    'water': 'w', 'roof': 'R', 'greenhouse': 'G', 'solar_panel': 'P',
    'fence': 'F', 'wall': 'W', 'mast': 'M', 'wind_turbine': 'T',
    'substation': 'X', 'road': 'r', 'path': 'p', 'parking': 'k',
    'bridge': 'b', 'crop': 'c', 'orchard': 'o', 'vineyard': 'v',
    'garden': 'a', 'bare_soil': 'B', 'rock': 'K', 'excavation': 'E',
    'fill': 'L', 'tree_loss': 'l', 'construction': 'C',
    'earthwork': 'e', 'unclassified': 'u',
}

LETTER_TYPE = {v: k for k, v in TYPE_LETTER.items()}

# Schema definitions — callers use these to decode arrays into dicts.
TOP_OBJS_KEYS = [
    'type_letter', 'height_max_m', 'height_mean_m', 'area_sqm',
    'lon', 'lat', 'confidence', 'rf_confidence', 'is_manmade',
]
TOP_TREES_KEYS = [
    'height_m', 'canopy_height_m', 'height_p90_m', 'area_sqm',
    'lon', 'lat', 'ndvi_mean', 'ndvi_fused', 'height_change_m',
    'phenology_class', 'confidence', 'rf_confidence',
]

TOP_N = 5  # per parcel (KG-level top_10 lists are unchanged)


def decode_top_obj(arr):
    """Decode a compact top_objs array entry to a dict, expanding type letter."""
    if not isinstance(arr, (list, tuple)):
        return arr  # already a dict (legacy)
    d = dict(zip(TOP_OBJS_KEYS, arr))
    d['type'] = LETTER_TYPE.get(d.pop('type_letter', ''), 'unknown')
    return d


def decode_top_tree(arr):
    if not isinstance(arr, (list, tuple)):
        return arr
    return dict(zip(TOP_TREES_KEYS, arr))


def decode_frav(frav):
    """{letter: area_sqm} -> {type_name: area_sqm}."""
    return {LETTER_TYPE.get(k, k): v for k, v in (frav or {}).items()}


def encode_obj_letter(otype: str) -> str:
    return TYPE_LETTER.get(otype, '?')


# ---------------------------------------------------------------------------
# Parcel auto-classification
# ---------------------------------------------------------------------------
# Empirical labels derived from the frav (fraction-area vector), terrain
# attributes, and the per-parcel top_trees summary. Designed to capture
# what the parcel mostly *is* in plain language, complementing the raw
# `dominant_type` (which only names the largest single class).
#
# Output: a dict
#   {
#     'class': 'forest' | 'meadow' | 'cropland' | ... ,
#     'subclass': optional refinement (e.g. 'tall_forest', 'orchard_meadow'),
#     'confidence': 0..1 ,
#     'features': {tree_frac, grass_frac, crop_frac, built_frac, ...}
#   }
#
# Class taxonomy (15 labels, mutually exclusive, ordered by check priority):
#
#   built_up        — ≥35% built (roof+greenhouse+solar) OR ≥5 buildings
#   farmstead       — built 10-35%, embedded in agriculture / yard
#   infrastructure  — ≥35% road/path/parking/bridge
#   water_body      — ≥40% water
#   forest          — ≥60% trees, mean tree height ≥8m
#   young_forest    — ≥60% trees, mean tree height <8m
#   wooded          — 30-60% trees (open woodland / scattered)
#   orchard         — ≥5% orchard OR (8-25% trees in regular grid pattern, agriculture context)
#   vineyard        — ≥5% vineyard
#   cropland        — ≥50% crop/bare_soil
#   meadow          — ≥60% grass, terrain non-alpine
#   alpine_meadow   — ≥60% grass, slope>20° OR elev>1500m
#   shrubland       — ≥40% shrub, <30% tree
#   bare            — ≥50% bare_soil/rock/excavation/fill
#   disturbance     — ≥25% tree_loss/excavation/fill/construction (recent change)
#   mixed           — fallback when no class dominates
#
# Confidence reflects how decisive the class is: 1.0 means unambiguous,
# 0.4 means narrowly past threshold, 0.0 means we fell through to mixed.

NATURAL_VEG = {'tree', 'shrub', 'grass', 'hedge'}
BUILT_UP    = {'roof', 'greenhouse', 'solar_panel', 'construction'}
INFRA       = {'road', 'path', 'parking', 'bridge'}
AGRI        = {'crop', 'orchard', 'vineyard', 'garden', 'bare_soil'}
WATER       = {'water'}
DISTURB     = {'tree_loss', 'excavation', 'fill', 'construction', 'earthwork'}
BARE        = {'bare_soil', 'rock'}


def classify_parcel(parcel: dict) -> dict:
    """Classify a parcel into one of ~15 landscape classes.

    Reads (defensively) parcel['frav'] (preferred), parcel['area_summary'],
    parcel terrain fields, top_trees, building_count, area_sqm.
    Robust to missing fields. Returns a dict with class/subclass/confidence/features.
    """
    # --- 1) Build the type → fraction map --------------------------------
    frav = parcel.get('frav') or {}
    if frav:
        # frav values are area-in-sqm; convert to fractions
        type_areas = {LETTER_TYPE.get(k, k): float(v) for k, v in frav.items()}
    else:
        # fall back to area_summary
        s = parcel.get('area_summary') or {}
        type_areas = {t: float((info or {}).get('area_sqm', 0))
                      for t, info in s.items()}

    total = sum(type_areas.values()) or float(parcel.get('area_sqm') or 0) or 1.0
    f = {t: a / total for t, a in type_areas.items()}

    def frac(types):
        return sum(f.get(t, 0.0) for t in types)

    tree_frac    = f.get('tree', 0.0)
    shrub_frac   = f.get('shrub', 0.0)
    grass_frac   = f.get('grass', 0.0)
    hedge_frac   = f.get('hedge', 0.0)
    crop_frac    = f.get('crop', 0.0)
    orchard_frac = f.get('orchard', 0.0)
    vine_frac    = f.get('vineyard', 0.0)
    water_frac   = f.get('water', 0.0)
    built_frac   = frac(BUILT_UP)
    infra_frac   = frac(INFRA)
    bare_frac    = frac(BARE)
    disturb_frac = frac(DISTURB)
    nat_veg_frac = frac(NATURAL_VEG)
    farmland_frac = crop_frac + orchard_frac + vine_frac + bare_frac

    # --- 2) Terrain context ---------------------------------------------
    slope = parcel.get('slope_mean_deg') or 0.0
    elev  = parcel.get('elevation_m') or 0.0
    terrain_class = parcel.get('terrain_class') or ''

    # --- 3) Tree height context (from top_trees if available) ----------
    mean_tree_h = None
    n_trees = 0
    for raw in parcel.get('top_trees') or []:
        d = decode_top_tree(raw)
        h = d.get('canopy_height_m') or d.get('height_m')
        if h is not None:
            mean_tree_h = (mean_tree_h or 0.0) + h
            n_trees += 1
    if n_trees:
        mean_tree_h = mean_tree_h / n_trees

    building_count = int(parcel.get('building_count') or 0)
    building_max_h  = float(parcel.get('building_max_height_m') or 0)
    building_max_st = int(parcel.get('building_max_stories') or 0)
    building_foot   = float(parcel.get('building_total_footprint_sqm') or 0)
    area_sqm        = float(parcel.get('area_sqm') or 0)

    # Hansen: recent-5yr loss as fraction of parcel area (1 px ≈ 900 m² at 30m)
    hansen = parcel.get('hansen_loss') or {}
    hansen_recent_px = int(hansen.get('recent_5yr_pixels') or 0)
    hansen_total_px  = int(hansen.get('total_pixels') or 0)
    hansen_recent_frac = (hansen_recent_px * 900.0) / area_sqm if area_sqm > 0 else 0.0

    # Building footprint share (fall back to frav-built if buildings dict missing)
    if building_foot > 0 and area_sqm > 0:
        built_footprint_frac = min(1.0, building_foot / area_sqm)
    else:
        built_footprint_frac = built_frac

    # TRI (terrain ruggedness index) — already in m, useful as 'rugged' flag
    tri = float(parcel.get('tri_mean') or 0)
    elev_range = float(parcel.get('elevation_range_m') or 0)
    rugged = (tri >= 1.0 or elev_range >= 5.0 or slope >= 25)

    feats = {
        'tree_frac': round(tree_frac, 3),
        'grass_frac': round(grass_frac, 3),
        'crop_frac': round(crop_frac, 3),
        'built_frac': round(built_frac, 3),
        'built_footprint_frac': round(built_footprint_frac, 3),
        'infra_frac': round(infra_frac, 3),
        'water_frac': round(water_frac, 3),
        'shrub_frac': round(shrub_frac, 3),
        'farmland_frac': round(farmland_frac, 3),
        'natural_veg_frac': round(nat_veg_frac, 3),
        'disturbance_frac': round(disturb_frac, 3),
        'mean_tree_height_m': round(mean_tree_h, 1) if mean_tree_h else None,
        'slope_mean_deg': round(slope, 1),
        'elevation_m': round(elev, 0),
        'tri_mean': round(tri, 2),
        'elev_range_m': round(elev_range, 1),
        'rugged': rugged,
        'building_count': building_count,
        'building_max_height_m': round(building_max_h, 1) if building_max_h else None,
        'building_max_stories': building_max_st or None,
        'hansen_recent_5yr_frac': round(hansen_recent_frac, 3),
        'hansen_recent_5yr_px': hansen_recent_px,
        'hansen_total_px': hansen_total_px,
    }

    def out(label, conf, sub=None):
        return {'class': label, 'subclass': sub,
                'confidence': round(min(max(conf, 0.0), 1.0), 2),
                'features': feats}

    # --- 4) Decision ladder (priority-ordered) --------------------------
    if water_frac >= 0.40:
        return out('water_body', min(1.0, water_frac))

    if disturb_frac >= 0.25 or hansen_recent_frac >= 0.20:
        if hansen_recent_frac >= 0.20 and tree_frac < 0.40:
            return out('disturbance', min(1.0, 0.5 + hansen_recent_frac), 'recent_clearfell')
        sub = 'tree_loss' if f.get('tree_loss', 0) > 0.10 else \
              ('construction' if f.get('construction', 0) > 0.05 else 'earthwork')
        return out('disturbance', min(1.0, 0.5 + disturb_frac), sub)

    if built_footprint_frac >= 0.35 or building_count >= 5:
        if building_max_st >= 5:
            sub = 'apartments'
        elif building_max_st >= 3:
            sub = 'multi_storey'
        elif building_max_st >= 2:
            sub = 'house'
        elif built_footprint_frac >= 0.6:
            sub = 'dense'
        else:
            sub = None
        return out('built_up', min(1.0, 0.5 + built_footprint_frac), sub)

    if infra_frac >= 0.35:
        sub = 'road' if f.get('road', 0) > infra_frac * 0.5 else 'mixed'
        return out('infrastructure', min(1.0, 0.5 + infra_frac), sub)

    if 0.05 <= built_footprint_frac < 0.35 and building_count >= 1 \
            and (farmland_frac + grass_frac) >= 0.30:
        sub = 'with_house' if building_max_st >= 1 else None
        return out('farmstead', 0.5 + min(built_footprint_frac, 0.3), sub)

    if vine_frac >= 0.05:
        return out('vineyard', min(1.0, 0.5 + vine_frac * 5))

    if orchard_frac >= 0.05:
        return out('orchard', min(1.0, 0.5 + orchard_frac * 5))

    if tree_frac >= 0.60:
        if hansen_recent_frac >= 0.05:
            return out('forest', min(1.0, tree_frac), 'recently_thinned')
        if mean_tree_h is not None and mean_tree_h < 8:
            return out('young_forest', min(1.0, tree_frac), 'regenerating')
        if (mean_tree_h or 0) >= 25:
            return out('forest', min(1.0, tree_frac), 'tall_forest')
        return out('forest', min(1.0, tree_frac))

    if 0.30 <= tree_frac < 0.60:
        return out('wooded', 0.4 + tree_frac, 'open_woodland')

    if shrub_frac >= 0.40 and tree_frac < 0.30:
        return out('shrubland', min(1.0, 0.4 + shrub_frac))

    if (crop_frac + bare_frac) >= 0.50:
        sub = 'fallow' if bare_frac > crop_frac else None
        return out('cropland', min(1.0, 0.5 + crop_frac + bare_frac * 0.5), sub)

    if grass_frac >= 0.60:
        if slope >= 20 or elev >= 1500 or terrain_class in ('alpine', 'steep'):
            sub = 'pasture' if elev < 2000 else 'high_alpine'
            return out('alpine_meadow', min(1.0, grass_frac), sub)
        if rugged:
            return out('meadow', min(1.0, grass_frac), 'rugged')
        sub = 'orchard_meadow' if (tree_frac >= 0.05 or hedge_frac >= 0.03) else None
        return out('meadow', min(1.0, grass_frac), sub)

    if bare_frac >= 0.50:
        return out('bare', min(1.0, 0.5 + bare_frac))

    # No class dominates — pick the strongest signal as a hint
    hint = max(
        [('forest', tree_frac), ('meadow', grass_frac),
         ('cropland', crop_frac), ('built_up', built_frac),
         ('shrubland', shrub_frac), ('wooded', tree_frac + hedge_frac)],
        key=lambda x: x[1])
    return out('mixed', max(0.1, hint[1]), hint[0])
