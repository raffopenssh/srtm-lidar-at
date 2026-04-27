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
