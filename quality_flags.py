"""Quality-flag rules for KG JSON outputs.

Each rule is a small function (kg_data, obj) -> Optional[Flag].
Flags are produced for individual extracted objects (trees, buildings,
top_by_type segments, new buildings, infrastructure, parcels).

Thresholds are tuned against the empirical distribution of all currently
processed KGs (see scripts/inspect_distributions.py output baked into
DIST below). The rule_version embedded in each flag lets us re-tune
without losing history.

Designed to run as a post-step after each new KG JSON is written.
Takes ~50 ms per KG (no I/O beyond the JSON itself).

Public API:
    scan_json(json_path) -> dict   # writes flags + objects to feedback.sqlite
    scan_kg_data(data, kg_code) -> dict
    iter_objects(data) -> Iterator[obj_dict]
    apply_rules(obj) -> List[flag_dict]

CLI:
    python3 quality_flags.py scan-kg <kg_code>
    python3 quality_flags.py scan-all
    python3 quality_flags.py stats
    python3 quality_flags.py distributions  # re-print empirical pcts
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

log = logging.getLogger(__name__)

RULE_VERSION = 'v1.2026-04b'
JSON_DIR = Path('data/austria_processor/json')

# ----------------------------------------------------------------------
# Empirical thresholds (tuned against ~18 sample KGs spanning Austria)
# Trees: real Austria max is ~57m (Waldviertel Douglas firs). Above 60m
#   is implausible; above 80m is essentially always a mast/cliff/error.
# Buildings: garden houses can be 1.5-3m tall over ~15-25m². There are
#   no real "buildings" below ~1.6m. Tallest in Austria is DC Tower (220m).
# Hedges: shrub-row, real-world max ~5m. >8m is vegetation that should be
#   tree, not hedge. >15m is certainly mis-segmented forest.
# Shrubs: real-world max ~4m. >6m is a tree. >15m is forest segment.
# Ground types (grass/crop/road/path/parking/garden): height_max should
#   be near 0 (DTM not nDSM); we tolerate up to 3m for tractors/parked
#   cars/edge artefacts but flag >5m as definite mis-classification.
# Water: should be flat. >2m height_max means we picked up a riverbank
#   or bridge in the segment. >10m is wrong.
# Solar panels: real-world <3m above roof. Footprints typically <500m²;
#   >2000m² is a roof mis-classified as solar.
# Volumes (stem volume per single tree): real max ~60m³ (giant Douglas
#   firs). >150m³ is a forest patch grouped into one segment.
# ----------------------------------------------------------------------

THRESHOLDS = {
    # ---- Vegetation height ----
    # Distributions across 36k indexed parcels (top_objs) Apr-2026:
    #   tree   p95=37.7  p99=43.1  max=84.3   (real Austria max ~57m)
    #   shrub  p95=13.3  p99=18.5  max=51.7   → raised warn 4→8
    #   hedge  p95=17.0  p99=21.3  max=41.1
    #   solar  p99=3.0  max=3.5
    #   roof   p95=15.5  p99=22.7  max=50.1
    'tree_max_height_m':           {'warn': 50, 'high': 60, 'critical': 80},
    'shrub_max_height_m':          {'warn': 8,  'high': 12, 'critical': 20},
    'hedge_max_height_m':          {'warn': 8,  'high': 15, 'critical': 25},
    'orchard_max_height_m':        {'warn': 10, 'high': 13, 'critical': 20},
    'vineyard_max_height_m':       {'warn': 3,  'high': 5,  'critical': 10},
    # ---- Buildings ----
    'roof_min_height_m':           {'min': 1.6},   # garden-shed floor
    'roof_max_height_m':           {'warn': 50, 'high': 100, 'critical': 220},
    'roof_min_area_sqm':           {'min': 6.0},  # below this, classifier noise
    'roof_max_area_sqm':           {'warn': 50000, 'high': 100000},
    'tall_thin_roof_ratio':        {'warn': 2.5, 'high': 4.0},  # h/√A: apartment block ~2, mast >>
    'greenhouse_max_height_m':     {'warn': 12, 'high': 18},
    # ---- Ground/flat types: height_max should be near zero ----
    'flat_type_max_height_m':      {'warn': 3,  'high': 5, 'critical': 12},
    # ---- Water ----
    'water_max_height_m':          {'warn': 2,  'high': 5, 'critical': 10},
    # ---- Solar panels ----
    'solar_max_height_m':          {'warn': 3,  'high': 5},
    'solar_max_area_sqm':          {'warn': 1500, 'high': 5000},
    # ---- Mast/wind_turbine ----
    'mast_max_height_m':           {'high': 250},
    # ---- Confidence ----
    'low_confidence':              {'warn': 0.5, 'high': 0.35},
    # ---- Volumes ----
    'tree_stem_vol_m3':            {'warn': 80, 'high': 150, 'critical': 500},
    # ---- Tiny segment ----
    'min_segment_area_sqm':        {'min': 4.0},
    # ---- Mast-likely: tall + tiny + low NDVI but classified tree ----
    'mast_likely_height_m':        30,
    'mast_likely_max_area_sqm':    50,
}

# Types where height_max should be ~ground (DTM-relative, not nDSM).
# A non-trivial height here usually means a parked vehicle, edge artefact,
# or wrong segmentation (a tree segment merged in).
GROUND_TYPES = {
    'grass', 'crop', 'road', 'path', 'parking', 'garden',
    'bare_soil', 'fill', 'excavation', 'tree_loss',
}

VEG_TYPES = {'tree', 'shrub', 'hedge', 'orchard', 'vineyard', 'forest', 'woodland', 'hedgerow'}
BUILDING_TYPES = {'roof', 'greenhouse', 'building'}

# Soft severity ladder.
SEV_ORDER = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

# Numeric weight per severity (used to score concern + agreement).
# Multiple independent rules firing on the same object accumulate weight
# ("agreement" — different evidence pointing the same way).
SEV_WEIGHT = {'low': 1.0, 'medium': 2.0, 'high': 4.0, 'critical': 8.0}


# ----------------------------------------------------------------------
# Object inventory
# ----------------------------------------------------------------------

def iter_objects(data: dict, kg_code: Optional[str] = None) -> Iterator[dict]:
    """Yield a flat list of inspectable objects from a KG JSON.

    Each yielded dict carries:
        obj_ref     stable id  "<kg>:<kind>:<inner>"
        kg_code
        kind        top_tree | top_obj | top_by_type | building | new_building | infra | parcel
        obj_type    predicted segment type
        centroid_lon, centroid_lat
        area_sqm, height_max_m, height_mean_m
        rf_confidence, confidence
        attrs       full original dict (passed by reference for rules)
    """
    kg = kg_code or data.get('kg_code')
    if not kg:
        return

    # 1) top_10_trees
    for i, t in enumerate(data.get('top_10_trees') or []):
        coord = t.get('coordinate') or {}
        yield {
            'obj_ref': f'{kg}:top_tree:{i}',
            'kg_code': kg, 'kind': 'top_tree',
            'obj_type': t.get('rf_type') or 'tree',
            'centroid_lon': coord.get('lon'), 'centroid_lat': coord.get('lat'),
            'area_sqm': t.get('area_sqm'),
            'height_max_m': t.get('height_m'),
            'height_mean_m': t.get('canopy_height_m'),
            'rf_confidence': t.get('rf_confidence'),
            'confidence': t.get('confidence'),
            'attrs': t,
        }

    # 2) top_10_objects (general highlights)
    for i, o in enumerate(data.get('top_10_objects') or []):
        coord = o.get('coordinate') or {}
        yield {
            'obj_ref': f'{kg}:top_obj:{i}',
            'kg_code': kg, 'kind': 'top_obj',
            'obj_type': o.get('type'),
            'centroid_lon': coord.get('lon'), 'centroid_lat': coord.get('lat'),
            'area_sqm': o.get('area_sqm'),
            'height_max_m': o.get('height_max_m'),
            'height_mean_m': o.get('height_mean_m'),
            'rf_confidence': o.get('rf_confidence'),
            'confidence': o.get('confidence'),
            'attrs': o,
        }

    # 3) top_by_type[*] (per-type top representatives)
    for otype, entries in (data.get('top_by_type') or {}).items():
        for rank, e in enumerate(entries or []):
            coord = e.get('coordinate') or {}
            yield {
                'obj_ref': f'{kg}:top_by_type:{otype}:{rank}',
                'kg_code': kg, 'kind': 'top_by_type',
                'obj_type': otype,
                'centroid_lon': coord.get('lon'), 'centroid_lat': coord.get('lat'),
                'area_sqm': e.get('area_sqm'),
                'height_max_m': e.get('height_max_m'),
                'height_mean_m': e.get('height_mean_m'),
                'rf_confidence': e.get('rf_confidence'),
                'confidence': e.get('confidence'),
                'attrs': e,
            }

    # 4) building_footprints.details (matched cadastre buildings)
    for i, b in enumerate(data.get('building_footprints', {}).get('details') or []):
        bid = b.get('building_id') or f'#{i}'
        coord = b.get('centroid') or {}
        yield {
            'obj_ref': f'{kg}:building:{bid}',
            'kg_code': kg, 'kind': 'building',
            'obj_type': 'roof',
            'centroid_lon': coord.get('lon'), 'centroid_lat': coord.get('lat'),
            'area_sqm': b.get('footprint_area_sqm'),
            'height_max_m': b.get('max_height_m'),
            'height_mean_m': b.get('mean_height_m'),
            'rf_confidence': None,
            'confidence': None,
            'attrs': b,
        }

    # 5) new_buildings.features (vectorised unmatched roofs)
    for i, f in enumerate(data.get('new_buildings', {}).get('features') or []):
        yield {
            'obj_ref': f'{kg}:new_building:{i}',
            'kg_code': kg, 'kind': 'new_building',
            'obj_type': f.get('rf_type') or f.get('type') or 'roof',
            'centroid_lon': f.get('centroid_lon'), 'centroid_lat': f.get('centroid_lat'),
            'area_sqm': f.get('area_sqm'),
            'height_max_m': f.get('max_height_m'),
            'height_mean_m': f.get('mean_height_m'),
            'rf_confidence': f.get('rf_confidence'),
            'confidence': f.get('confidence'),
            'attrs': f,
        }

    # 6) infrastructure.by_type[*].features
    for otype, info in (data.get('infrastructure', {}).get('by_type') or {}).items():
        for i, f in enumerate(info.get('features') or []):
            yield {
                'obj_ref': f'{kg}:infra:{otype}:{i}',
                'kg_code': kg, 'kind': 'infra',
                'obj_type': f.get('rf_type') or otype,
                'centroid_lon': f.get('centroid_lon'), 'centroid_lat': f.get('centroid_lat'),
                'area_sqm': f.get('area_sqm'),
                'height_max_m': f.get('max_height_m') or f.get('height_max_m'),
                'height_mean_m': f.get('mean_height_m'),
                'rf_confidence': f.get('rf_confidence'),
                'confidence': f.get('confidence'),
                'attrs': f,
            }

    # 7) parcels.details (cadastre-classified parcels)
    for p in (data.get('parcels', {}).get('details') or []):
        pid = p.get('parcel_id') or ''
        coord = p.get('centroid') or {}
        cls = p.get('classification', {}) or {}
        yield {
            'obj_ref': f'{kg}:parcel:{pid}',
            'kg_code': kg, 'kind': 'parcel',
            'obj_type': p.get('dominant_type'),
            'centroid_lon': coord.get('lon'), 'centroid_lat': coord.get('lat'),
            'area_sqm': p.get('area_sqm'),
            'height_max_m': p.get('ndsm_max_m'),
            'height_mean_m': p.get('ndsm_mean_m'),
            'rf_confidence': cls.get('rf_mean_confidence'),
            'confidence': cls.get('mean_confidence'),
            'attrs': p,
        }
        # 7a) Per-parcel top objects/trees — flagged independently of KG-wide top_10.
        # Both compact (`top_objs` array form) and verbose (`top_10_objects`
        # dict form) layouts are supported for backward compatibility.
        try:
            from parcel_compact import decode_top_obj, decode_top_tree
        except ImportError:
            decode_top_obj = decode_top_tree = lambda x: x
        # Compact arrays (preferred, written by austria_processor v2026-04+)
        for i, raw in enumerate(p.get('top_objs') or []):
            o = decode_top_obj(raw)
            yield {
                'obj_ref': f'{kg}:parcel_top_obj:{pid}:{i}',
                'kg_code': kg, 'kind': 'parcel_top_obj',
                'obj_type': o.get('type'),
                'centroid_lon': o.get('lon'), 'centroid_lat': o.get('lat'),
                'area_sqm': o.get('area_sqm'),
                'height_max_m': o.get('height_max_m'),
                'height_mean_m': o.get('height_mean_m'),
                'rf_confidence': o.get('rf_confidence'),
                'confidence': o.get('confidence'),
                'attrs': o,
            }
        for i, raw in enumerate(p.get('top_trees') or []):
            t = decode_top_tree(raw)
            yield {
                'obj_ref': f'{kg}:parcel_top_tree:{pid}:{i}',
                'kg_code': kg, 'kind': 'parcel_top_tree',
                'obj_type': 'tree',
                'centroid_lon': t.get('lon'), 'centroid_lat': t.get('lat'),
                'area_sqm': t.get('area_sqm'),
                'height_max_m': t.get('height_m'),
                'height_mean_m': t.get('canopy_height_m'),
                'rf_confidence': t.get('rf_confidence'),
                'confidence': t.get('confidence'),
                'attrs': t,
            }
        # Verbose form (legacy)
        for i, o in enumerate(p.get('top_10_objects') or []):
            coord_o = o.get('coordinate') or {}
            yield {
                'obj_ref': f'{kg}:parcel_top_obj:{pid}:{i}',
                'kg_code': kg, 'kind': 'parcel_top_obj',
                'obj_type': o.get('rf_type') or o.get('type'),
                'centroid_lon': coord_o.get('lon'), 'centroid_lat': coord_o.get('lat'),
                'area_sqm': o.get('area_sqm'),
                'height_max_m': o.get('height_max_m'),
                'height_mean_m': o.get('height_mean_m'),
                'rf_confidence': o.get('rf_confidence'),
                'confidence': o.get('confidence'),
                'attrs': o,
            }
        for i, t in enumerate(p.get('top_10_trees') or []):
            coord_t = t.get('coordinate') or {}
            yield {
                'obj_ref': f'{kg}:parcel_top_tree:{pid}:{i}',
                'kg_code': kg, 'kind': 'parcel_top_tree',
                'obj_type': t.get('rf_type') or 'tree',
                'centroid_lon': coord_t.get('lon'), 'centroid_lat': coord_t.get('lat'),
                'area_sqm': t.get('area_sqm'),
                'height_max_m': t.get('height_m'),
                'height_mean_m': t.get('canopy_height_m'),
                'rf_confidence': t.get('rf_confidence'),
                'confidence': t.get('confidence'),
                'attrs': t,
            }


# ----------------------------------------------------------------------
# Rules
# ----------------------------------------------------------------------

def _flag(code: str, severity: str, msg: str, **extra) -> dict:
    return {'flag_code': code, 'severity': severity, 'message': msg,
            'weight': SEV_WEIGHT.get(severity, 1.0), 'attrs': extra}


def _h_severity(value, low_warn, high, critical=None):
    """Map a numeric value to a severity given threshold rungs."""
    if critical is not None and value >= critical: return 'critical'
    if value >= high: return 'high'
    if value >= low_warn: return 'medium'
    return None


def apply_rules(obj: dict) -> list[dict]:
    """Return zero or more flag dicts for one object."""
    out = []
    t = (obj.get('obj_type') or '').lower()
    h = obj.get('height_max_m')
    a = obj.get('area_sqm')
    rf = obj.get('rf_confidence')
    edge = obj.get('attrs', {}).get('edge_clipped')

    # ---- Trees / forest ----
    if t in ('tree', 'forest', 'woodland') and h is not None:
        T = THRESHOLDS['tree_max_height_m']
        sev = _h_severity(h, T['warn'], T['high'], T['critical'])
        if sev:
            out.append(_flag('tree_height_implausible', sev,
                f"{t} height_max={h:.1f}m exceeds expected for Austria "
                f"(real max ~57m, typical max ~45m)",
                value=h, threshold=T['high']))
        # mast-likely: tall, tiny footprint, low NDVI
        ndvi = obj.get('attrs', {}).get('ndvi_mean') or obj.get('attrs', {}).get('ndvi_fused')
        if h >= THRESHOLDS['mast_likely_height_m'] and a and a <= THRESHOLDS['mast_likely_max_area_sqm']:
            out.append(_flag('mast_likely_misclassified', 'high',
                f'tall ({h:.1f}m) but only {a:.0f}m² footprint — likely a mast/antenna mis-classified as tree',
                value=h, area=a, ndvi=ndvi))

    # ---- Shrubs ----
    if t == 'shrub' and h is not None:
        T = THRESHOLDS['shrub_max_height_m']
        sev = _h_severity(h, T['warn'], T['high'], T['critical'])
        if sev:
            out.append(_flag('shrub_height_implausible', sev,
                f'shrub height_max={h:.1f}m exceeds 4m (true shrubs are <4m — likely a tree)',
                value=h, threshold=T['high']))

    # ---- Hedges ----
    if t in ('hedge', 'hedgerow') and h is not None:
        T = THRESHOLDS['hedge_max_height_m']
        sev = _h_severity(h, T['warn'], T['high'], T['critical'])
        if sev:
            out.append(_flag('hedge_height_implausible', sev,
                f'hedge height_max={h:.1f}m exceeds 8m (likely a tree row or forest segment)',
                value=h, threshold=T['high']))
        # Hedge confidence in our model is always rule-fallback (~0.55).
        # When a 'hedge' is also clearly tree-tall (>10m), the rule is wrong.
        if h >= 10 and rf is not None and 0.45 <= rf <= 0.6:
            out.append(_flag('hedge_likely_tree_row', 'medium',
                f'hedge {h:.1f}m tall + RF conf={rf:.2f} (rule fallback) — likely tree row',
                value=h, rf=rf))

    # ---- Orchard / vineyard ----
    if t == 'orchard' and h is not None:
        T = THRESHOLDS['orchard_max_height_m']
        sev = _h_severity(h, T['warn'], T['high'], T['critical'])
        if sev:
            out.append(_flag('orchard_height_implausible', sev,
                f'orchard height_max={h:.1f}m above expected (10–13m)', value=h))
    if t == 'vineyard' and h is not None:
        T = THRESHOLDS['vineyard_max_height_m']
        sev = _h_severity(h, T['warn'], T['high'], T['critical'])
        if sev:
            out.append(_flag('vineyard_height_implausible', sev,
                f'vineyard height_max={h:.1f}m above expected (3–5m)', value=h))

    # ---- Buildings ----
    if t in BUILDING_TYPES and h is not None:
        # Too short to be a real building. Real garden sheds reach ~2m, so
        # we only flag when *also* tiny: that pattern is canopies, walls,
        # bus stops, mis-identified flat patches.
        if h < THRESHOLDS['roof_min_height_m']['min'] and (a is None or a < 30):
            out.append(_flag('building_height_too_low', 'medium',
                f'{t} height={h:.2f}m below 1.6m and area={a or 0:.0f}m² — likely a wall/canopy not a building',
                value=h, area=a))
        # Too tall
        T = THRESHOLDS['roof_max_height_m']
        sev = _h_severity(h, T['warn'], T['high'], T['critical'])
        if sev:
            out.append(_flag('building_height_implausible', sev,
                f'{t} height={h:.1f}m exceeds {T["high"]}m (DC Tower=220m is Austria\'s tallest)',
                value=h))
        # Tall + thin = mast-like
        if a and a > 0 and h > 12:
            ratio = h / math.sqrt(a)
            T2 = THRESHOLDS['tall_thin_roof_ratio']
            if ratio > T2['high']:
                out.append(_flag('building_aspect_extreme', 'high',
                    f'{t} h/√A = {ratio:.2f} (h={h:.1f}m, A={a:.0f}m²) — mast-like aspect, likely mis-classified',
                    height=h, area=a, ratio=round(ratio, 3)))
            elif ratio > T2['warn']:
                out.append(_flag('building_aspect_extreme', 'medium',
                    f'{t} h/√A = {ratio:.2f} (h={h:.1f}m, A={a:.0f}m²) — unusually slender for a roof',
                    height=h, area=a, ratio=round(ratio, 3)))

    if t == 'roof' and a is not None:
        if a < THRESHOLDS['roof_min_area_sqm']['min']:
            out.append(_flag('roof_area_tiny', 'low',
                f'roof area={a:.1f}m² below 6m² (sub-pixel artefact)', value=a))
        T = THRESHOLDS['roof_max_area_sqm']
        if a > T['high']:
            out.append(_flag('roof_area_huge', 'high',
                f'roof area={a:.0f}m² — verify against industrial site', value=a))
        elif a > T['warn']:
            out.append(_flag('roof_area_huge', 'medium',
                f'roof area={a:.0f}m² unusually large', value=a))

    # ---- Greenhouse ----
    if t == 'greenhouse' and h is not None:
        T = THRESHOLDS['greenhouse_max_height_m']
        sev = _h_severity(h, T['warn'], T['high'])
        if sev:
            out.append(_flag('greenhouse_height_implausible', sev,
                f'greenhouse height_max={h:.1f}m above expected (8–12m)', value=h))

    # ---- Ground/flat types should be near zero height ----
    if t in GROUND_TYPES and h is not None:
        T = THRESHOLDS['flat_type_max_height_m']
        sev = _h_severity(h, T['warn'], T['high'], T['critical'])
        if sev:
            out.append(_flag('flat_type_has_height', sev,
                f'{t} height_max={h:.1f}m — ground type should be near 0m, segment likely mixes vegetation',
                value=h))

    # ---- Water ----
    if t in ('water', 'waterbody') and h is not None:
        T = THRESHOLDS['water_max_height_m']
        sev = _h_severity(h, T['warn'], T['high'], T['critical'])
        if sev:
            out.append(_flag('water_has_height', sev,
                f'water height_max={h:.1f}m — segment likely captures riverbank/bridge',
                value=h))

    # ---- Solar panels ----
    if t == 'solar_panel':
        if h is not None and h > THRESHOLDS['solar_max_height_m']['warn']:
            sev = 'high' if h > THRESHOLDS['solar_max_height_m']['high'] else 'medium'
            out.append(_flag('solar_height_implausible', sev,
                f'solar_panel height={h:.1f}m above expected (≤3m)', value=h))
        if a is not None and a > THRESHOLDS['solar_max_area_sqm']['warn']:
            sev = 'high' if a > THRESHOLDS['solar_max_area_sqm']['high'] else 'medium'
            out.append(_flag('solar_area_huge', sev,
                f'solar_panel area={a:.0f}m² — verify (likely roof mis-classified)', value=a))

    # ---- Wind turbine / mast ----
    if t == 'mast' and h is not None and h > THRESHOLDS['mast_max_height_m']['high']:
        out.append(_flag('mast_height_implausible', 'high',
            f'mast height={h:.1f}m above 250m', value=h))

    # ---- Stem volume ----
    sv = obj.get('attrs', {}).get('est_stem_volume_m3')
    if sv is None: sv = obj.get('attrs', {}).get('volume_m3') if t in VEG_TYPES else None
    if sv is not None and t in ('tree',):
        T = THRESHOLDS['tree_stem_vol_m3']
        sev = _h_severity(sv, T['warn'], T['high'], T['critical'])
        if sev:
            out.append(_flag('tree_volume_implausible', sev,
                f'tree stem volume={sv:.0f}m³ — single segment likely groups multiple trees',
                value=sv))

    # ---- Tiny segment ----
    if a is not None and 0 < a < THRESHOLDS['min_segment_area_sqm']['min']:
        out.append(_flag('tiny_segment', 'low',
            f'{t} area={a:.1f}m² below 4m² (sub-pixel)', value=a))

    # ---- Low confidence ----
    if rf is not None:
        T = THRESHOLDS['low_confidence']
        if rf < T['high']:
            out.append(_flag('low_rf_confidence', 'medium',
                f'{t} RF confidence={rf:.2f} below 0.35', value=rf))
        elif rf < T['warn']:
            out.append(_flag('low_rf_confidence', 'low',
                f'{t} RF confidence={rf:.2f} below 0.5', value=rf))

    # ---- Edge artefact ----
    if edge and h is not None and h > 30 and t in VEG_TYPES:
        out.append(_flag('boundary_artifact', 'low',
            f'{t} touches tile edge with height={h:.1f}m — may be split across tiles', value=h))

    # ---- NDVI/class mismatch ----
    ndvi = obj.get('attrs', {}).get('ndvi_mean')
    if ndvi is not None:
        if t in ('tree', 'shrub', 'forest', 'woodland', 'hedge', 'hedgerow', 'grass') and ndvi < 0.15:
            out.append(_flag('ndvi_class_mismatch', 'medium',
                f'{t} but NDVI={ndvi:.2f} below 0.15 (vegetation should have positive NDVI)',
                value=ndvi, type=t))
        if t in ('water', 'waterbody') and ndvi > 0.4:
            out.append(_flag('ndvi_class_mismatch', 'medium',
                f'water but NDVI={ndvi:.2f} above 0.4 (water has NDVI<0)', value=ndvi))

    return out


# ----------------------------------------------------------------------
# Scan helpers
# ----------------------------------------------------------------------

def scan_kg_data(data: dict, kg_code: Optional[str] = None) -> dict:
    """Return {'kg_code', 'objects': [...], 'flags': [...]} without side-effects."""
    kg = kg_code or data.get('kg_code')
    objs = list(iter_objects(data, kg))
    flags = []
    for o in objs:
        for fl in apply_rules(o):
            flags.append({
                'obj_ref': o['obj_ref'],
                'kg_code': o['kg_code'],
                'centroid_lon': o.get('centroid_lon'),
                'centroid_lat': o.get('centroid_lat'),
                **fl,
            })
    return {'kg_code': kg, 'objects': objs, 'flags': flags}


def scan_json(json_path) -> dict:
    """Scan a KG JSON file and persist objects+flags to feedback.sqlite."""
    p = Path(json_path)
    data = json.loads(p.read_text())
    kg = p.stem  # may be a block code like '49006-south'
    res = scan_kg_data(data, kg)
    # Lazy-import to avoid hard cycle if feedback_db isn't ready yet.
    import feedback_db
    feedback_db.write_objects_and_flags(res['objects'], res['flags'], RULE_VERSION)
    log.info('quality_flags: %s → %d objects, %d flags',
             kg, len(res['objects']), len(res['flags']))
    return res


def scan_all(json_dir: Path = JSON_DIR) -> dict:
    n_obj = 0; n_flag = 0
    for jp in sorted(json_dir.glob('*.json')):
        try:
            r = scan_json(jp)
            n_obj += len(r['objects']); n_flag += len(r['flags'])
        except Exception as e:
            log.warning('quality_flags scan failed for %s: %s', jp.name, e)
    return {'objects': n_obj, 'flags': n_flag}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _cli():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    cmd = sys.argv[1]
    if cmd == 'scan-kg':
        kg = sys.argv[2]
        jp = JSON_DIR / f'{kg}.json'
        r = scan_json(jp)
        print(json.dumps({
            'kg_code': r['kg_code'],
            'n_objects': len(r['objects']),
            'n_flags': len(r['flags']),
            'flags_by_severity': dict(sorted(
                ((s, sum(1 for f in r['flags'] if f['severity']==s))
                 for s in {'low','medium','high','critical'}),
                key=lambda x: -SEV_ORDER[x[0]])),
            'flags_by_code': dict(sorted(
                ((c, sum(1 for f in r['flags'] if f['flag_code']==c))
                 for c in set(f['flag_code'] for f in r['flags'])),
                key=lambda x: -x[1]))[:20],
            'sample_flags': r['flags'][:5],
        }, indent=2))
    elif cmd == 'scan-all':
        t0 = time.time()
        r = scan_all()
        print(json.dumps({**r, 'elapsed_s': round(time.time()-t0, 2)}, indent=2))
    elif cmd == 'stats':
        import feedback_db
        print(json.dumps(feedback_db.flag_stats(), indent=2))
    elif cmd == 'distributions':
        # Re-emit empirical percentiles from local JSONs.
        from collections import defaultdict
        H = defaultdict(list); A = defaultdict(list)
        for jp in JSON_DIR.glob('*.json'):
            d = json.loads(jp.read_text())
            for o in iter_objects(d, jp.stem):
                if o['height_max_m'] is not None: H[o['obj_type']].append(o['height_max_m'])
                if o['area_sqm'] is not None: A[o['obj_type']].append(o['area_sqm'])
        def pct(L, ps=(50,90,95,99,99.9)):
            if not L: return None
            L = sorted(L); n=len(L)
            return {p: round(L[min(int(p/100*n), n-1)],2) for p in ps}
        out = {t: {'n': len(H[t]), 'h': pct(H[t]), 'a': pct(A[t])} for t in sorted(H)}
        print(json.dumps(out, indent=2))
    else:
        print('unknown command'); sys.exit(2)

if __name__ == '__main__':
    _cli()
