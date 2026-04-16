#!/usr/bin/env python3
"""Train Random Forest on 4000 random KGs using cadastre + OSM ground truth.

Improvements over train_rf_100kg.py:
- 4000 KGs (was 100)
- OSM roads/paths/landcover as supplementary ground truth
- Model checkpoint saved every 10 successful KGs
- n_kgs passed to clf.train() for metadata

Run: python3 train_rf_4000kg.py 2>&1 | tee /tmp/rf_train_4000kg.log
"""
import json
import logging
import os
import multiprocessing
import random
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import requests
from shapely.geometry import box, shape as shapely_shape, Polygon, MultiPolygon
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    stream=sys.stdout,
)
log = logging.getLogger('rf_train')

# Suppress noisy loggers
for name in ['rasterio', 'urllib3', 'botocore', 'PIL', 'fiona']:
    logging.getLogger(name).setLevel(logging.WARNING)

N_KGS = 300
MODEL_CHECKPOINT_INTERVAL = 10  # train & save model every N successful KGs
KG_TIMEOUT_SECONDS = 20 * 60  # 20 min max per KG (prevents stuck segmentation)
MAX_KG_PIXELS = 10_000_000  # skip KGs with > 10M valid pixels (OOM risk)

CADASTRE_BASE = "https://cadastre-process-api.exe.xyz/api/v1"
RESULTS_DIR = Path("/tmp/rf_train_4000kg")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Permanent storage — survives /tmp cleanup
PERMANENT_DIR = Path("/home/exedev/srtm-lidar/rf_training_data")
PERMANENT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = PERMANENT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Track KGs that crash (e.g. OOM) to avoid infinite retry loops
IN_PROGRESS_FILE = PERMANENT_DIR / "in_progress_kg.txt"
FAILED_KGS_FILE = PERMANENT_DIR / "failed_kgs.txt"

# Circuit breaker for openEO / Copernicus — file-based so it persists across
# forked subprocesses. When openEO is returning 503s, skip Copernicus for a
# cooldown period instead of burning 3×180s timeouts per KG.
_CIRCUIT_BREAKER_FILE = PERMANENT_DIR / "openeo_circuit.json"

def _read_circuit_breaker() -> dict:
    """Read circuit breaker state from file."""
    try:
        if _CIRCUIT_BREAKER_FILE.exists():
            import json
            return json.loads(_CIRCUIT_BREAKER_FILE.read_text())
    except Exception:
        pass
    return {"consecutive_failures": 0, "last_failure": 0.0, "cooldown": 120}

def _write_circuit_breaker(state: dict):
    """Write circuit breaker state to file."""
    import json
    try:
        _CIRCUIT_BREAKER_FILE.write_text(json.dumps(state))
    except Exception:
        pass

# WGS84 <-> EPSG:3035 transformers
_tx_to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
_tx_to_wgs = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)


def transform_to_3035(geom):
    return shapely_transform(_tx_to_3035.transform, geom)


def transform_to_wgs(geom):
    return shapely_transform(_tx_to_wgs.transform, geom)


def get_random_kgs(n: int = N_KGS) -> list[dict]:
    """Fetch KGs that overlap LIDAR coverage and pick n random ones."""
    states = [
        "Steiermark", "Kärnten", "Niederösterreich", "Oberösterreich",
        "Salzburg", "Tirol", "Vorarlberg", "Burgenland", "Wien",
    ]
    all_kgs = []
    for state in states:
        try:
            resp = requests.get(
                f"{CADASTRE_BASE}/search/district",
                params={"state": state},
                timeout=30,
            )
            resp.raise_for_status()
            districts = resp.json().get("data", [])
            for d in districts:
                code = d.get("district_code")
                if not code or not d.get("district_name", "").strip():
                    continue
                try:
                    resp2 = requests.get(
                        f"{CADASTRE_BASE}/search/kg",
                        params={"district": d["district_name"], "limit": 500},
                        timeout=30,
                    )
                    resp2.raise_for_status()
                    kgs = resp2.json().get("data", [])
                    for kg in kgs:
                        if kg.get("kg_code") and kg.get("parcel_count", 0) > 10:
                            all_kgs.append(kg)
                except Exception as e:
                    log.warning("Failed to fetch KGs for district %s: %s", code, e)
        except Exception as e:
            log.warning("Failed to fetch districts for %s: %s", state, e)

    log.info("Found %d KGs total across Austria", len(all_kgs))

    with_buildings = [kg for kg in all_kgs if kg.get("building_count", 0) > 5]
    log.info("%d KGs have >5 buildings", len(with_buildings))

    random.seed(42)
    if len(with_buildings) >= n:
        selected = random.sample(with_buildings, n)
    else:
        selected = list(with_buildings)
        if len(selected) < n and all_kgs:
            remaining = [k for k in all_kgs if k not in selected]
            extra = random.sample(remaining, min(n - len(selected), len(remaining)))
            selected.extend(extra)

    log.info("Selected %d KGs for training", len(selected))
    return selected


def get_infrastructure_kgs(
    n_solar: int = 20,
    n_wind: int = 20,
    n_substation: int = 10,
) -> list[dict]:
    """Find KGs containing power infrastructure (solar, wind, substation).

    Queries the austria-power API for infrastructure locations, samples
    a subset, then reverse-geocodes each to a KG via the cadastre
    spatial lookup.  Returns KG dicts with {kg_code, kg_name, infra_target}.
    """
    try:
        r = requests.get(
            "https://austria-power.exe.xyz:8000/api/infrastructure",
            params={"bbox": "9.5,46.3,17.2,49.0"},
            timeout=60,
        )
        r.raise_for_status()
        features = r.json().get("features", [])
    except Exception as e:
        log.warning("Infrastructure KG fetch failed (power API): %s", e)
        return []

    TYPE_MAP = {
        "solar": "solar_panel",
        "Windmill farm": "wind_turbine",
        "Windpower plant": "wind_turbine",
        "windpark": "wind_turbine",
        "substation": "substation",
        "substation_380kv": "substation",
        "transformer_station": "substation",
    }

    by_label: dict[str, list[tuple[float, float]]] = {}
    for f in features:
        t = f["properties"].get("type", "")
        label = TYPE_MAP.get(t)
        if not label:
            continue
        geom = f["geometry"]
        try:
            if geom["type"] == "Point":
                lon, lat = geom["coordinates"]
            elif geom["type"] in ("LineString", "MultiPoint"):
                coords = geom["coordinates"]
                lon = sum(c[0] for c in coords) / len(coords)
                lat = sum(c[1] for c in coords) / len(coords)
            elif geom["type"] == "Polygon":
                coords = geom["coordinates"][0]
                lon = sum(c[0] for c in coords) / len(coords)
                lat = sum(c[1] for c in coords) / len(coords)
            else:
                continue
        except (KeyError, IndexError, ZeroDivisionError):
            continue
        by_label.setdefault(label, []).append((lon, lat))

    # Sample locations
    rng = random.Random(42)
    sampled: list[tuple[str, float, float]] = []
    for label, n in [("solar_panel", n_solar), ("wind_turbine", n_wind),
                     ("substation", n_substation)]:
        pts = by_label.get(label, [])
        chosen = rng.sample(pts, min(n, len(pts)))
        sampled.extend((label, lon, lat) for lon, lat in chosen)

    # Reverse-geocode to KGs
    HALF = 0.005  # ~500m bbox around each point
    kgs: dict[str, dict] = {}  # kg_code -> dict
    for label, lon, lat in sampled:
        try:
            r2 = requests.get(
                f"{CADASTRE_BASE}/spatial/kgs",
                params={"west": lon - HALF, "south": lat - HALF,
                        "east": lon + HALF, "north": lat + HALF},
                timeout=15,
            )
            r2.raise_for_status()
            kg_codes = r2.json().get("data", {}).get("kg_codes", [])
            if kg_codes:
                code = kg_codes[0]
                if code not in kgs:
                    kgs[code] = {
                        "kg_code": code,
                        "kg_name": f"infra-{label}",
                        "infra_target": label,
                        "infra_center": (lon, lat),
                    }
        except Exception:
            pass

    result = list(kgs.values())
    log.info("Infrastructure KGs: %d (solar=%d, wind=%d, sub=%d)",
             len(result),
             sum(1 for k in result if k["infra_target"] == "solar_panel"),
             sum(1 for k in result if k["infra_target"] == "wind_turbine"),
             sum(1 for k in result if k["infra_target"] == "substation"))
    return result


def fetch_cadastre_layers(kg_code: str) -> dict:
    """Fetch all three cadastre layers for a KG."""
    result = {"parcels": [], "building_footprints": [], "landuse": []}

    try:
        resp = requests.get(
            f"{CADASTRE_BASE}/export/geojson",
            params={
                "kg": kg_code,
                "layers": "parcels,building_footprints,landuse_polygons",
                "include_geometry": "true",
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning("KG %s: cadastre fetch failed: %s", kg_code, e)
        return result

    parcels_fc = data.get("parcels", {}).get("features", [])
    for f in parcels_fc:
        try:
            geom = shapely_shape(f["geometry"])
            if geom.is_empty or not isinstance(geom, (Polygon, MultiPolygon)):
                continue
            geom_3035 = transform_to_3035(geom)
            props = f.get("properties", {})
            lu_summary = props.get("landuse_summary", {})
            dominant_code = _extract_dominant_code(lu_summary)
            result["parcels"].append({
                "geometry": geom_3035,
                "landuse_code": dominant_code,
                "landuse_summary": lu_summary,
                "area_sqm": props.get("area_sqm", 0),
                "parcel_id": props.get("parcel_id", ""),
            })
        except Exception:
            continue

    bfp_fc = data.get("building_footprints", {}).get("features", [])
    for f in bfp_fc:
        try:
            geom = shapely_shape(f["geometry"])
            if geom.is_empty or not isinstance(geom, (Polygon, MultiPolygon)):
                continue
            geom_3035 = transform_to_3035(geom)
            result["building_footprints"].append(geom_3035)
        except Exception:
            continue

    lu_fc = data.get("landuse_polygons", {}).get("features", [])
    for f in lu_fc:
        try:
            geom = shapely_shape(f["geometry"])
            if geom.is_empty or not isinstance(geom, (Polygon, MultiPolygon)):
                continue
            props = f.get("properties", {})
            code_str = props.get("landuse_code", "")
            abbr = props.get("landuse_abbr", "")
            try:
                code = int(code_str)
            except (ValueError, TypeError):
                code = None
            geom_3035 = transform_to_3035(geom)
            result["landuse"].append({
                "geometry": geom_3035,
                "code": code,
                "abbr": abbr,
                "area_sqm": props.get("area_sqm", 0),
            })
        except Exception:
            continue

    log.info("KG %s: %d parcels, %d building_footprints, %d landuse features",
             kg_code, len(result["parcels"]), len(result["building_footprints"]),
             len(result["landuse"]))
    return result


_LANDUSE_DESC_TO_CODE = {
    "B": 42, "B(bf)": 41, "B(bg)": 40,
    "V": 48, "W": 56, "LN": 52, "LN(W)": 52, "LN(Hu)": 53,
    "A": 51, "WG": 63, "GA": 64, "OG": 65, "Alpe": 54,
    "GW": 70, "Öd": 80, "Fe": 81,
    "Wald": 56, "Acker": 51, "Wiese": 52, "Weide": 55,
    "Obstgarten": 65, "Weingarten": 63, "Gewässer": 70,
}


def _extract_dominant_code(lu_summary: dict) -> int | None:
    if not lu_summary:
        return None
    best_key = max(lu_summary, key=lu_summary.get)
    if " - " in best_key:
        abbr = best_key.split(" - ")[-1]
        if abbr in _LANDUSE_DESC_TO_CODE:
            return _LANDUSE_DESC_TO_CODE[abbr]
    for word, code in _LANDUSE_DESC_TO_CODE.items():
        if word in best_key:
            return code
    return None


# ---------------------------------------------------------------------------
# Rasterized cadastre labelling (pixel-accurate, height-aware)
# ---------------------------------------------------------------------------

# Cadastre codes that represent ground-level surface types.
# Pixels with these codes that have elevated nDSM (tree canopy, etc.)
# should NOT be used as training labels for their ground type.
_GROUND_SURFACE_CODES = {
    # Roads/paths/parking — must be at ground level
    48, 73,       # Straße (road)
    74,           # Weg (path)
    41,           # Baufläche (paved) → parking
    # Water — must be at ground level
    70, 71,       # Gewässer, stehende Gewässer
    96,           # Feuchtgebiet
    60,           # Sumpf/Moor
    72,           # Quelle/Brunnen
    # Agricultural / grass — ground level
    51, 62,       # Acker (crop)
    52, 53, 54, 55, 58, 61,  # Wiese/Weide/Alpe/Grünland (grass)
    63,           # Weingarten (vineyard)
    64,           # Hausgarten (garden)
    65,           # Obstgarten (orchard) — allow some height for fruit trees
}

# Maximum nDSM height (m) per ground-level code category.
# Pixels above this threshold get cleared to 0 (unlabelled).
_GROUND_MAX_HEIGHT = {
    # Transport — very flat
    48: 1.5, 73: 1.5, 74: 1.5, 41: 1.5,
    # Water — flat
    70: 1.0, 71: 1.0, 96: 1.5, 60: 1.5, 72: 1.0,
    # Crops/grass — low vegetation
    51: 2.0, 62: 2.0,
    52: 2.0, 53: 2.0, 54: 2.0, 55: 2.0, 58: 2.0, 61: 2.0,
    # Gardens/vineyards — some structure allowed
    63: 3.0, 64: 3.0,
    # Orchards — fruit trees can be 4-5m
    65: 6.0,
}

# Codes that are elevated by nature — skip ground-level masking.
# Bridge (75) is deliberately excluded from _GROUND_SURFACE_CODES.
_ELEVATED_CODES = {42, 43, 44, 45, 46, 47, 75}  # buildings + bridge


def rasterize_cadastre_labels(
    cadastre_data: dict,
    transform,
    shape_hw: tuple[int, int],
    ndsm: np.ndarray | None = None,
    min_building_height: float = 2.0,
) -> np.ndarray:
    """Rasterize all cadastre features into a single label raster.

    Returns an int16 array where each pixel is a cadastre code (0 = unlabelled).
    Priority (highest first): building footprints (code 42), landuse polygons,
    parcel polygons.

    Height-aware masking:
    - Building footprint pixels require nDSM >= min_building_height.
    - Ground-level types (roads, paths, parking, water, grass, crop) require
      nDSM below a type-specific threshold.  Pixels with tree canopy or other
      elevated surfaces above these ground features are cleared to 0.
    - Bridge (code 75) is exempt from ground masking.
    """
    from rasterio.features import rasterize as rio_rasterize

    h, w = shape_hw
    label_raster = np.zeros((h, w), dtype=np.int16)

    # Layer 1 (lowest priority): parcel polygons
    parcel_pairs = []
    for p in cadastre_data["parcels"]:
        code = p.get("landuse_code")
        geom = p.get("geometry")
        if code is not None and geom is not None and not geom.is_empty:
            parcel_pairs.append((geom, int(code)))
    if parcel_pairs:
        try:
            parcel_raster = rio_rasterize(
                parcel_pairs, out_shape=(h, w), transform=transform,
                fill=0, dtype=np.int16,
            )
            mask = parcel_raster != 0
            label_raster[mask] = parcel_raster[mask]
        except Exception as e:
            log.warning("rasterize_cadastre: parcels failed: %s", e)

    # Layer 2 (medium priority): landuse polygons — overwrite parcels
    lu_pairs = []
    for lu in cadastre_data["landuse"]:
        code = lu.get("code")
        geom = lu.get("geometry")
        if code is not None and geom is not None and not geom.is_empty:
            lu_pairs.append((geom, int(code)))
    if lu_pairs:
        try:
            lu_raster = rio_rasterize(
                lu_pairs, out_shape=(h, w), transform=transform,
                fill=0, dtype=np.int16,
            )
            mask = lu_raster != 0
            label_raster[mask] = lu_raster[mask]
        except Exception as e:
            log.warning("rasterize_cadastre: landuse failed: %s", e)

    # Layer 3 (highest priority): building footprints — code 42, height-masked
    bfp_pairs = [(g, 42) for g in cadastre_data["building_footprints"]
                 if not g.is_empty]
    if bfp_pairs:
        try:
            bfp_raster = rio_rasterize(
                bfp_pairs, out_shape=(h, w), transform=transform,
                fill=0, dtype=np.int16, all_touched=True,
            )
            bfp_mask = bfp_raster != 0
            if ndsm is not None:
                # Only mark as roof where the surface is actually elevated
                elevated = ndsm >= min_building_height
                label_raster[bfp_mask & elevated] = 42
                # Pixels inside footprint but NOT elevated keep their
                # landuse/parcel label (or stay 0) — this is the key fix:
                # ground-level segments that happen to overlap a building
                # footprint will NOT be labelled as roof.
                n_bfp = int(bfp_mask.sum())
                n_elev = int((bfp_mask & elevated).sum())
                log.debug("rasterize_cadastre: %d building px, %d elevated (%.0f%%)",
                          n_bfp, n_elev, 100 * n_elev / max(n_bfp, 1))
            else:
                # No height data — fall back to all footprint pixels
                label_raster[bfp_mask] = 42
        except Exception as e:
            log.warning("rasterize_cadastre: buildings failed: %s", e)

    # Layer 4: Ground-level height masking
    # Clear pixels where a ground-level cadastre type has elevated nDSM
    # (e.g. tree canopy over a road, or forest on cadastre grass).
    if ndsm is not None:
        n_cleared = 0
        for code, max_h in _GROUND_MAX_HEIGHT.items():
            code_mask = label_raster == code
            if not code_mask.any():
                continue
            elevated_mask = code_mask & (ndsm > max_h)
            n_elev = int(elevated_mask.sum())
            if n_elev > 0:
                label_raster[elevated_mask] = 0
                n_cleared += n_elev
        if n_cleared > 0:
            log.info("rasterize_cadastre: cleared %d ground-level px with elevated nDSM",
                     n_cleared)

    return label_raster


# Maximum NDVI for cadastre ground-surface types.
# Segments above this threshold are vegetated → cadastre label unreliable.
# Real roads/parking have NDVI < 0.15; at 0.25 it's clearly vegetation.
_CADASTRE_SURFACE_MAX_NDVI = {
    "road": 0.25, "path": 0.25, "parking": 0.25,
    "bare_soil": 0.3,
}

# Maximum segment mean height (h_mean) for OSM ground-level type labels.
# Segments above this are likely tree canopy/structures over the ground feature.
_OSM_GROUND_TYPE_MAX_HEIGHT = {
    "road": 1.5, "path": 1.5, "parking": 1.5,
    "water": 1.0,
    "grass": 2.0, "crop": 2.0,
    "vineyard": 3.0, "garden": 3.0,
    "bare_soil": 1.5,
}


def match_segments_via_raster(
    features_list: list[dict],
    labels: np.ndarray,
    cadastre_raster: np.ndarray,
    osm_labels: np.ndarray | None,
    infra_labels: dict | None = None,
    min_overlap_frac: float = 0.15,
) -> tuple[list[dict], list[str], dict]:
    """Match segments to ground truth via pixel-level majority vote.

    For each segment (identified by labels == feat['label']), count how many
    pixels have each cadastre code.  If the dominant code covers at least
    min_overlap_frac of the segment, use it.  Otherwise fall back to OSM.

    Height-aware: cadastre raster is already height-masked (ground-level px
    with elevated nDSM cleared to 0).  For OSM fallback, we additionally
    reject ground-level type labels when the segment's mean height (h_mean)
    exceeds a type-specific threshold.

    Returns (train_features, train_labels, source_counts).
    """
    import learned_classifier as lc
    train_features = []
    train_labels = []
    source_counts = {
        "cadastre_raster": 0, "cadastre_ndvi_rejected": 0,
        "osm": 0, "osm_height_rejected": 0, "unmatched": 0,
    }

    for _seg_idx, feat in enumerate(features_list):
        seg_id = feat.get("label")
        if seg_id is None:
            source_counts["unmatched"] += 1
            continue

        seg_mask = labels == seg_id
        seg_px = int(seg_mask.sum())
        if seg_px < 2:
            source_counts["unmatched"] += 1
            continue

        # Extract cadastre codes within this segment
        codes_in_seg = cadastre_raster[seg_mask]
        nonzero = codes_in_seg[codes_in_seg != 0]

        if len(nonzero) > 0:
            # Majority vote
            unique, counts = np.unique(nonzero, return_counts=True)
            best_idx = counts.argmax()
            best_code = int(unique[best_idx])
            best_frac = counts[best_idx] / seg_px

            if best_frac >= min_overlap_frac and best_code in lc.CADASTRE_TO_TYPE:
                ctype = lc.CADASTRE_TO_TYPE[best_code]
                # NDVI sanity: reject hard-surface labels on vegetated segments.
                # Use fused NDVI (season-corrected) when available — BEV NDVI
                # can be very low in winter orthos even on green pastures.
                max_ndvi = _CADASTRE_SURFACE_MAX_NDVI.get(ctype)
                if max_ndvi is not None:
                    seg_ndvi = feat.get("fused_ndvi_mean", 0.0) or \
                               feat.get("ndvi_mean", 0.0)
                    if seg_ndvi > max_ndvi:
                        source_counts["cadastre_ndvi_rejected"] += 1
                        # Don't use — fall through to OSM or unmatched
                    else:
                        train_features.append(feat)
                        train_labels.append(ctype)
                        source_counts["cadastre_raster"] += 1
                        continue
                else:
                    train_features.append(feat)
                    train_labels.append(ctype)
                    source_counts["cadastre_raster"] += 1
                    continue

        # Fall back to OSM
        if osm_labels is not None:
            osm_in_seg = osm_labels[seg_mask]
            # osm_labels is a string/object array or int-coded; handle both
            osm_nonzero = osm_in_seg[osm_in_seg != ""]
            if hasattr(osm_nonzero, 'dtype') and osm_nonzero.dtype.kind in ('i', 'u', 'f'):
                osm_nonzero = osm_nonzero[osm_nonzero != 0]
            if len(osm_nonzero) > 0:
                unique_o, counts_o = np.unique(osm_nonzero, return_counts=True)
                best_osm = str(unique_o[counts_o.argmax()])
                if best_osm in lc.TYPE_CLASSES:
                    # Height check: reject ground-level OSM labels on
                    # elevated segments (tree canopy over road, etc.)
                    h_mean = feat.get("h_mean", 0.0)
                    max_h = _OSM_GROUND_TYPE_MAX_HEIGHT.get(best_osm)
                    if max_h is not None and h_mean > max_h:
                        source_counts["osm_height_rejected"] += 1
                        source_counts["unmatched"] += 1
                        continue
                    train_features.append(feat)
                    train_labels.append(best_osm)
                    source_counts["osm"] += 1
                    continue

        # Infrastructure labels (power API + OSM power)
        if infra_labels and _seg_idx in infra_labels:
            infra_lbl = infra_labels[_seg_idx]
            train_features.append(feat)
            train_labels.append(infra_lbl)
            source_counts["infrastructure"] = source_counts.get("infrastructure", 0) + 1
            continue

        source_counts["unmatched"] += 1

    # Post-pass: relabel tree_loss from Hansen evidence
    # Cadastre/OSM never label tree_loss, but Hansen + LIDAR height can.
    # Criteria: strong recent Hansen loss + evidence of cleared canopy.
    n_relabelled = 0
    for i, (feat, lbl) in enumerate(zip(train_features, train_labels)):
        if lbl not in ("tree", "shrub", "grass", "bare_soil", "crop"):
            continue
        hrlf = feat.get("hansen_recent_loss_frac", 0.0)
        tc2000 = feat.get("hansen_treecover2000", 0.0)
        h_mean = feat.get("h_mean", 0.0)
        h_change = feat.get("h_change", 0.0)
        # Must have been forest (tc2000 > 30) with recent Hansen loss
        if hrlf < 0.15 or tc2000 < 20:
            continue
        # Evidence of cleared canopy: low current height OR significant drop
        if h_mean < 5.0 or h_change < -2.0:
            train_labels[i] = "tree_loss"
            n_relabelled += 1
    if n_relabelled:
        source_counts["hansen_tree_loss"] = n_relabelled

    return train_features, train_labels, source_counts


def match_segment_to_cadastre(
    feat: dict,
    cadastre_data: dict,
    transform,
) -> tuple[int | None, str]:
    """Match a segment to cadastre ground truth.

    Priority:
    1. Building footprint overlap → code 42 (roof)
    2. Landuse polygon overlap → use landuse code
    3. Parcel polygon overlap → use dominant parcel landuse code
    """
    ce = feat.get("centroid_e", 0)
    cn = feat.get("centroid_n", 0)
    if ce == 0 and cn == 0:
        return None, ""

    from shapely.geometry import Point
    pt = Point(ce, cn)
    seg_area = feat.get("area", 0)

    seg_bbox = feat.get("bbox")
    if seg_bbox and len(seg_bbox) == 4:
        seg_poly = box(seg_bbox[0], seg_bbox[1], seg_bbox[2], seg_bbox[3])
    else:
        r = max(1.0, (seg_area / 3.14159) ** 0.5) if seg_area > 0 else 5.0
        seg_poly = pt.buffer(r)

    # 1. Building footprints
    for bfp in cadastre_data["building_footprints"]:
        try:
            if bfp.intersects(seg_poly):
                overlap = bfp.intersection(seg_poly).area
                if overlap > 0.3 * seg_poly.area or overlap > 0.3 * bfp.area:
                    return 42, "building_footprint"
        except Exception:
            continue

    # 2. Landuse polygons
    best_lu_code = None
    best_lu_overlap = 0
    for lu in cadastre_data["landuse"]:
        code = lu.get("code")
        if code is None:
            continue
        geom = lu["geometry"]
        try:
            if geom.intersects(seg_poly):
                overlap = geom.intersection(seg_poly).area
                if overlap > best_lu_overlap:
                    best_lu_overlap = overlap
                    best_lu_code = code
        except Exception:
            continue

    if best_lu_code is not None and best_lu_overlap > 0.2 * seg_poly.area:
        return best_lu_code, "landuse_polygon"

    # 3. Parcel polygon
    best_parcel_code = None
    best_parcel_overlap = 0
    for p in cadastre_data["parcels"]:
        code = p.get("landuse_code")
        if code is None:
            continue
        try:
            geom = p["geometry"]
            if geom.intersects(seg_poly):
                overlap = geom.intersection(seg_poly).area
                if overlap > best_parcel_overlap:
                    best_parcel_overlap = overlap
                    best_parcel_code = code
        except Exception:
            continue

    if best_parcel_code is not None:
        return best_parcel_code, "parcel"

    if best_lu_code is not None:
        return best_lu_code, "landuse_polygon_weak"

    return None, ""


def match_segment_to_osm(
    feat: dict,
    osm_labels: np.ndarray,
    transform,
) -> tuple[str | None, str]:
    """Match a segment to OSM ground truth via its centroid pixel.

    Returns (label_string_or_None, source_string).
    """
    ce = feat.get("centroid_e", 0)
    cn = feat.get("centroid_n", 0)
    if ce == 0 and cn == 0:
        return None, ""

    # Convert 3035 coordinate to pixel row/col
    import rasterio
    try:
        col, row = ~transform * (ce, cn)
        row, col = int(round(row)), int(round(col))
        if 0 <= row < osm_labels.shape[0] and 0 <= col < osm_labels.shape[1]:
            lbl = str(osm_labels[row, col])
            if lbl and lbl != "":
                return lbl, "osm"
    except Exception:
        pass
    return None, ""


def process_one_kg(
    kg: dict,
    include_copernicus: bool = True,
    include_osm: bool = True,
    max_km: float = 1.5,
) -> tuple[list[dict], list[str], dict]:
    """Process one KG: segment + match to cadastre + OSM.

    Args:
        max_km: crop KG bbox to center NxN km (default 1.5).

    Returns (features, labels, stats).
    """
    import raster_io
    import tile_index as ti
    import object_segmentation as oc
    import learned_classifier as lc

    kg_code = kg["kg_code"]
    stats = {"kg_code": kg_code, "kg_name": kg.get("kg_name", "")}

    # Get KG bbox
    kg_bbox = kg.get("bbox", {})
    if isinstance(kg_bbox, dict) and all(k in kg_bbox for k in ["min_lon", "min_lat", "max_lon", "max_lat"]):
        west, south = kg_bbox["min_lon"], kg_bbox["min_lat"]
        east, north = kg_bbox["max_lon"], kg_bbox["max_lat"]
    else:
        try:
            resp = requests.get(
                f"{CADASTRE_BASE}/search/kg",
                params={"code": kg_code},
                timeout=30,
            )
            resp.raise_for_status()
            kgs = resp.json().get("data", [])
            if not kgs:
                stats["error"] = "KG not found"
                return [], [], stats
            k = kgs[0]
            bb = k.get("bbox", {})
            west, south = bb["min_lon"], bb["min_lat"]
            east, north = bb["max_lon"], bb["max_lat"]
        except Exception as e:
            stats["error"] = f"bbox fetch: {e}"
            return [], [], stats

    # Limit KG size to center max_km x max_km
    # If an infrastructure target point is provided, center on it instead
    # of the KG centroid so the feature falls within the analysis window.
    dx_km = (east - west) * 111 * np.cos(np.radians((south + north) / 2))
    dy_km = (north - south) * 111
    if dx_km > max_km or dy_km > max_km:
        infra_center = kg.get("infra_center")  # (lon, lat) or None
        if infra_center:
            cx, cy = infra_center
            # Clamp to KG bbox so we don't drift outside
            cx = max(west, min(east, cx))
            cy = max(south, min(north, cy))
        else:
            cx, cy = (west + east) / 2, (south + north) / 2
        half = (max_km / 2) / 111  # approx degrees
        west, south, east, north = cx - half, cy - half, cx + half, cy + half
        log.info("KG %s: large (%.1f x %.1f km), cropping to %.1fkm around %s",
                 kg_code, dx_km, dy_km, max_km,
                 "infra target" if infra_center else "center")

    stats["bbox"] = [west, south, east, north]
    geom_wgs = box(west, south, east, north)
    geom_3035 = transform_to_3035(geom_wgs)

    # 1. Fetch cadastre layers
    t0 = time.time()
    cadastre_data = fetch_cadastre_layers(kg_code)
    stats["cadastre_time"] = round(time.time() - t0, 1)
    stats["n_parcels"] = len(cadastre_data["parcels"])
    stats["n_footprints"] = len(cadastre_data["building_footprints"])
    stats["n_landuse"] = len(cadastre_data["landuse"])

    if stats["n_parcels"] == 0 and stats["n_footprints"] == 0:
        stats["error"] = "no cadastre data"
        return [], [], stats

    # Observation year — drives Copernicus/SAR/Hansen year scoping
    obs_year = ti.dataset_to_year(ti.DEFAULT_DATASET)

    # 2. Read LIDAR
    t0 = time.time()
    try:
        data = raster_io.read_dtm_dsm(geom_3035, ti.DEFAULT_DATASET)
    except Exception as e:
        stats["error"] = f"LIDAR: {e}"
        return [], [], stats
    stats["lidar_time"] = round(time.time() - t0, 1)
    stats["shape"] = list(data["shape"])

    # Early bail-out for oversized KGs to avoid OOM in segmentation
    valid_px = int(data["mask"].sum()) if data.get("mask") is not None else (data["shape"][0] * data["shape"][1])
    if valid_px > MAX_KG_PIXELS:
        stats["error"] = f"too large: {valid_px} valid px > {MAX_KG_PIXELS}"
        log.warning("KG %s: skipping (%d valid px exceeds %d limit)",
                    kg_code, valid_px, MAX_KG_PIXELS)
        return [], [], stats

    # 2b. Multi-date DTM/DSM
    dtm_dates = None
    dsm_dates = None
    try:
        other_dates = sorted(d for d in ti.DATASETS if d != ti.DEFAULT_DATASET)
        if other_dates:
            dtm_dates = {}
            dsm_dates = {}
            ref_h, ref_w = data["shape"]
            for date_key in other_dates:
                try:
                    d2 = raster_io.read_dtm_dsm(geom_3035, date_key)
                    mh = min(ref_h, d2["shape"][0])
                    mw = min(ref_w, d2["shape"][1])
                    dtm_dates[date_key] = d2["dtm"][:mh, :mw]
                    dsm_dates[date_key] = d2["dsm"][:mh, :mw]
                except Exception as e:
                    log.warning("KG %s: multi-date %s failed: %s", kg_code, date_key, e)
            if dtm_dates:
                mh = min(ref_h, *(a.shape[0] for a in dtm_dates.values()))
                mw = min(ref_w, *(a.shape[1] for a in dtm_dates.values()))
                dtm_dates[ti.DEFAULT_DATASET] = data["dtm"][:mh, :mw]
                dsm_dates[ti.DEFAULT_DATASET] = data["dsm"][:mh, :mw]
                for dk in list(dtm_dates):
                    dtm_dates[dk] = dtm_dates[dk][:mh, :mw]
                    dsm_dates[dk] = dsm_dates[dk][:mh, :mw]
                log.info("KG %s: loaded %d temporal dates: %s",
                         kg_code, len(dtm_dates), sorted(dtm_dates))
            else:
                dtm_dates = None
                dsm_dates = None
    except Exception as e:
        log.warning("KG %s: multi-date read failed: %s", kg_code, e)
        dtm_dates = None
        dsm_dates = None
    stats["has_temporal"] = dtm_dates is not None and len(dtm_dates) >= 2

    # 3. Read ortho (with timeout — BEV server can be very slow)
    ORTHO_TIMEOUT = 180  # 3 min max
    t0 = time.time()
    spectral = None
    try:
        import ortho_io
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
            fut = exe.submit(ortho_io.read_ortho_for_als, data)
            try:
                rgb, nir = fut.result(timeout=ORTHO_TIMEOUT)
                spectral = ortho_io.compute_spectral_indices(rgb, nir=nir)
                if rgb is not None:
                    spectral["red"] = rgb[0].astype(np.float32)
                    spectral["green"] = rgb[1].astype(np.float32)
                    spectral["blue"] = rgb[2].astype(np.float32)
                if nir is not None:
                    spectral["nir"] = nir.astype(np.float32)
            except concurrent.futures.TimeoutError:
                log.warning("KG %s: ortho timed out after %ds, skipping",
                            kg_code, ORTHO_TIMEOUT)
    except Exception as e:
        log.warning("KG %s: ortho failed: %s", kg_code, e)
    stats["ortho_time"] = round(time.time() - t0, 1)
    stats["has_ortho"] = spectral is not None

    # 4. Copernicus data
    copernicus_data = None
    if include_copernicus:
        t0 = time.time()
        try:
            import copernicus
            import concurrent.futures
            bbox_dict = {"west": west, "south": south, "east": east, "north": north}
            cop = {}

            # Each credential gets its own openEO connection — openEO
            # limits 1 concurrent sync download per client_id.  With N
            # credentials we can run N downloads in parallel.
            n_creds = len(copernicus._CREDENTIALS)

            def _fetch_ndvi(cred_idx):
                conn = copernicus._get_connection_for_cred(cred_idx)
                d = copernicus.get_ndvi_composite(bbox_dict, year=obs_year, _conn=conn)
                return {"ndvi": d["ndvi"], "transform": d["transform"], "crs": d["crs"]}

            def _fetch_landcover(cred_idx):
                conn = copernicus._get_connection_for_cred(cred_idx)
                return copernicus.get_land_cover(bbox_dict, _conn=conn)

            def _fetch_sar(cred_idx):
                conn = copernicus._get_connection_for_cred(cred_idx)
                sar_start = f"{obs_year}-06-01"
                sar_end   = f"{obs_year}-09-30"
                d = copernicus.get_sar_backscatter(bbox_dict, sar_start, sar_end, _conn=conn)
                return {"vv": d["vv"], "vh": d["vh"], "sar_transform": d["transform"], "sar_crs": d["crs"]}

            def _fetch_harmonics():
                import ndvi_harmonics
                return ndvi_harmonics.get_harmonic_features(bbox_dict, year=obs_year)

            COP_TIMEOUT = 180
            HARM_TIMEOUT = 900

            # Assign each fetch to a credential: round-robin across available creds.
            # With 2 creds we run 2 in parallel, then the 3rd uses whichever finishes first.
            fetch_tasks = [
                ("ndvi",      lambda ci=0 % n_creds: _fetch_ndvi(ci)),
                ("landcover", lambda ci=1 % n_creds: _fetch_landcover(ci)),
                ("sar",       lambda ci=0 % n_creds: _fetch_sar(ci)),
            ]

            # --- Circuit breaker: skip Copernicus entirely if openEO is down ---
            # File-based so it persists across forked subprocesses.
            import time as _time
            _cb = _read_circuit_breaker()
            if _cb["consecutive_failures"] >= 3 and (_time.time() - _cb["last_failure"]) < _cb["cooldown"]:
                log.info("KG %s: Copernicus circuit breaker OPEN (%d consecutive failures, %.0fs remaining) — skipping all",
                         kg_code, _cb["consecutive_failures"],
                         _cb["cooldown"] - (_time.time() - _cb["last_failure"]))
                copernicus_data = None
            else:
                # Run NDVI + landcover + SAR in parallel.
                # max_workers = n_creds so each worker uses a different client_id,
                # avoiding "max connections reached: 1" from openEO.
                cop_success = 0
                cop_fail = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_creds) as exe:
                    fast_futs = {}
                    for name, func in fetch_tasks:
                        fast_futs[name] = exe.submit(func)
                    for name, fut in fast_futs.items():
                        try:
                            result = fut.result(timeout=COP_TIMEOUT)
                            if result is not None:
                                if name == "ndvi":
                                    cop.update(result)
                                elif name == "landcover":
                                    cop["landcover"] = result
                                elif name == "sar":
                                    cop.update(result)
                                log.info("KG %s: %s OK", kg_code, name)
                                cop_success += 1
                        except concurrent.futures.TimeoutError:
                            log.warning("KG %s: %s timed out after %ds, skipping",
                                        kg_code, name, COP_TIMEOUT)
                            cop_fail += 1
                        except copernicus.CreditsExhaustedError:
                            raise
                        except Exception as e:
                            log.debug("KG %s: %s failed: %s", kg_code, name, e)
                            cop_fail += 1

                # Only attempt harmonics if at least one fast layer succeeded
                if cop_success > 0:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as exe:
                        try:
                            fut = exe.submit(_fetch_harmonics)
                            result = fut.result(timeout=HARM_TIMEOUT)
                            if result is not None:
                                cop["harmonics"] = result
                                log.info("KG %s: harmonics OK", kg_code)
                        except concurrent.futures.TimeoutError:
                            log.warning("KG %s: harmonics timed out after %ds, skipping",
                                        kg_code, HARM_TIMEOUT)
                        except copernicus.CreditsExhaustedError:
                            raise
                        except Exception as e:
                            log.debug("KG %s: harmonics failed: %s", kg_code, e)

                # Update circuit breaker state
                if cop_fail >= 3 and cop_success == 0:
                    _cb["consecutive_failures"] += 1
                    _cb["last_failure"] = _time.time()
                    _cb["cooldown"] = min(600, 60 * (2 ** min(_cb["consecutive_failures"], 4)))  # 120s → 600s
                    log.warning("KG %s: all Copernicus layers failed — circuit breaker count=%d, cooldown=%.0fs",
                                kg_code, _cb["consecutive_failures"], _cb["cooldown"])
                    # Rotate credentials for next attempt
                    try:
                        copernicus.rotate_credentials()
                    except Exception:
                        pass
                else:
                    _cb["consecutive_failures"] = 0  # reset on any success
                _write_circuit_breaker(_cb)

                copernicus_data = cop if cop else None
        except copernicus.CreditsExhaustedError:
            raise  # propagate to main loop for pause
        except Exception as e:
            log.warning("KG %s: Copernicus: %s", kg_code, e)
        stats["copernicus_time"] = round(time.time() - t0, 1)
        stats["has_copernicus"] = copernicus_data is not None
        if copernicus_data:
            stats["cop_layers"] = list(copernicus_data.keys())

    # 5. Hansen Global Forest Change
    hansen_data = None
    t0 = time.time()
    try:
        import hansen
        hansen_data = hansen.get_forest_prior(
            (west, south, east, north),
            data["transform"], data["shape"],
        )
        stats["has_hansen"] = True
        log.info("KG %s: Hansen OK (forest=%d px, loss=%d px)",
                 kg_code,
                 int(hansen_data["current_forest"].sum()),
                 int((hansen_data["loss_year"] > 0).sum()))
    except Exception as e:
        log.warning("KG %s: Hansen failed: %s", kg_code, e)
        stats["has_hansen"] = False
    stats["hansen_time"] = round(time.time() - t0, 1)

    # 5c. Power infrastructure points + polygons (matched after segmentation)
    _infra_points = []
    _osm_power_polys = []
    t0 = time.time()
    try:
        import power_infrastructure as pi
        _infra_points = pi.fetch_power_infrastructure((west, south, east, north))
        _osm_power_polys = pi.fetch_osm_power_polygons((west, south, east, north))
        stats["has_infrastructure"] = bool(_infra_points or _osm_power_polys)
        stats["n_infra_points"] = len(_infra_points)
        stats["n_osm_power_polys"] = len(_osm_power_polys)
        log.info("KG %s: infrastructure data: %d points, %d polygons",
                 kg_code, len(_infra_points), len(_osm_power_polys))
    except Exception as e:
        log.warning("KG %s: infrastructure data failed: %s", kg_code, e)
        stats["has_infrastructure"] = False
    stats["infrastructure_time"] = round(time.time() - t0, 1)

    # 5b. OSM ground truth (roads, paths, landcover)
    osm_labels = None
    if include_osm:
        t0 = time.time()
        try:
            import osm_features
            osm_result = osm_features.fetch_osm_ground_truth(
                (west, south, east, north),
                data["transform"],
                data["shape"],
            )
            osm_labels = osm_result["labels"]
            stats["osm_road_px"] = osm_result["n_road_px"]
            stats["osm_landcover_px"] = osm_result["n_landcover_px"]
            stats["has_osm"] = osm_result["n_road_px"] + osm_result["n_landcover_px"] > 0
            log.info("KG %s: OSM OK (road=%d px, lc=%d px)",
                     kg_code, osm_result["n_road_px"], osm_result["n_landcover_px"])
        except Exception as e:
            log.warning("KG %s: OSM failed: %s", kg_code, e)
            stats["has_osm"] = False
        stats["osm_time"] = round(time.time() - t0, 1)

    # 5c. Rasterize building footprints into a bool mask for segment calibration
    building_fp_mask = None
    if cadastre_data["building_footprints"]:
        try:
            from rasterio.features import rasterize as rio_rasterize
            h, w = data["shape"]
            pairs = [(g, 1) for g in cadastre_data["building_footprints"] if not g.is_empty]
            if pairs:
                building_fp_mask = rio_rasterize(
                    pairs, out_shape=(h, w), transform=data["transform"],
                    fill=0, dtype=np.uint8, all_touched=True,
                ).astype(bool)
                log.info("KG %s: rasterized %d building footprints (%d px)",
                         kg_code, len(pairs), int(building_fp_mask.sum()))
        except Exception as e:
            log.warning("KG %s: building footprint rasterize failed: %s", kg_code, e)

    # 6. Segment
    t0 = time.time()
    try:
        result = oc.segment_and_classify(
            data["dtm"], data["dsm"], data["mask"], data["transform"],
            dtm_dates=dtm_dates, dsm_dates=dsm_dates,
            spectral=spectral, copernicus=copernicus_data,
            building_footprints=building_fp_mask,
            hansen=hansen_data,
            ortho_year=obs_year,
            observation_year=obs_year,
            features_only=True,  # skip RF classification — saves ~1.5GB RAM
        )
    except Exception as e:
        stats["error"] = f"segmentation: {e}"
        return [], [], stats
    stats["segment_time"] = round(time.time() - t0, 1)

    features_list = result.get("features", [obj.features for obj in result["objects"]])
    stats["n_segments"] = len(features_list)

    if not features_list:
        stats["error"] = "no segments"
        return [], [], stats

    # 6b. Match infrastructure points/polygons to segments
    infra_seg_labels = {}
    if _infra_points or _osm_power_polys:
        try:
            import power_infrastructure as pi
            seg_labels_arr = result.get("labels")
            infra_seg_labels = pi.match_infrastructure_to_segments(
                _infra_points, _osm_power_polys,
                features_list, seg_labels_arr,
                data["transform"], ndsm=data.get("ndsm"),
            )
            if infra_seg_labels:
                log.info("KG %s: %d segments matched to infrastructure labels",
                         kg_code, len(infra_seg_labels))
        except Exception as e:
            log.warning("KG %s: infrastructure matching failed: %s", kg_code, e)

    # 7. Match segments to ground truth via rasterized cadastre + OSM
    #    Rasterize all cadastre features to 1m grid, mask building pixels
    #    by nDSM height, then per-segment majority vote.
    t0 = time.time()
    seg_labels = result.get("labels")  # segment label array (h, w)

    cadastre_raster = rasterize_cadastre_labels(
        cadastre_data, data["transform"], data["shape"],
        ndsm=data.get("ndsm"), min_building_height=2.0,
    )
    n_bld_px = int((cadastre_raster == 42).sum())
    n_lu_px = int(((cadastre_raster != 0) & (cadastre_raster != 42)).sum())
    log.info("KG %s: rasterized cadastre — %d building px, %d landuse px",
             kg_code, n_bld_px, n_lu_px)

    train_features, train_labels, source_counts = match_segments_via_raster(
        features_list, seg_labels, cadastre_raster, osm_labels,
        infra_labels=infra_seg_labels,
    )

    stats["match_time"] = round(time.time() - t0, 1)
    stats["n_labelled"] = len(train_features)
    stats["source_counts"] = source_counts
    stats["label_distribution"] = {}
    for lbl in train_labels:
        stats["label_distribution"][lbl] = stats["label_distribution"].get(lbl, 0) + 1

    log.info("KG %s (%s): %d segments, %d labelled (%s)",
             kg_code, kg.get("kg_name", ""),
             len(features_list), len(train_features),
             ", ".join(f"{k}={v}" for k, v in source_counts.items() if v > 0))

    return train_features, train_labels, stats


def _clear_downloaded_caches():
    """Delete only cached error entries from /tmp caches to reclaim memory.

    Preserves valid cached data (ortho, lidar, Copernicus, Hansen, OSM)
    so restarts don't re-download everything.  Only removes:
    - Files smaller than 200 bytes (likely error/empty responses)
    - Files with 'error' in the name
    """
    cleared = 0
    for cache_dir in [
        Path("/tmp/hansen_cache"),
        Path("/tmp/osm_cache"),
        Path("/tmp/power_infra_cache"),
        Path("/tmp/cadastre_cache"),
    ]:
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                try:
                    if f.is_file():
                        # Only remove tiny files (likely cached errors)
                        if f.stat().st_size < 200 or 'error' in f.name:
                            f.unlink()
                            cleared += 1
                except Exception:
                    pass
    if cleared:
        log.info("Cleared %d cached raster entries to free memory", cleared)


def _load_all_checkpoints():
    """Load all checkpoint features/labels from disk (not kept in RAM)."""
    all_features = []
    all_labels = []
    for ckpt_file in sorted(CHECKPOINT_DIR.glob("kg_*.npz")):
        try:
            ckpt = np.load(ckpt_file, allow_pickle=True)
            all_features.extend(ckpt["features"].tolist())
            all_labels.extend(ckpt["labels"].tolist())
        except Exception as e:
            log.warning("Failed to load checkpoint %s: %s", ckpt_file, e)
    return all_features, all_labels


def train_and_save_model(n_kgs, tag="checkpoint"):
    """Load checkpoints, train RF model, and save to disk. Returns training stats or None."""
    import learned_classifier as lc

    log.info("Loading checkpoints for model training [%s]...", tag)
    all_features, all_labels = _load_all_checkpoints()

    if len(all_features) < 20:
        log.warning("Not enough samples (%d) for model %s", len(all_features), tag)
        return None

    log.info("Training RF model [%s] on %d samples from %d KGs...",
             tag, len(all_features), n_kgs)
    clf = lc.LearnedClassifier()
    try:
        train_stats = clf.train(all_features, all_labels, n_kgs=n_kgs)
        log.info("Model [%s]: OOB=%.4f, %d classes, %d samples",
                 tag, train_stats["oob_score"], train_stats["n_classes"],
                 train_stats["n_train"])
        return train_stats
    except Exception as e:
        log.error("Model training [%s] failed: %s", tag, traceback.format_exc())
        return None
    finally:
        del all_features, all_labels, clf
        # Clear the cached classifier singleton so the 1.5GB model
        # doesn't stay resident in the parent process
        try:
            lc._cached_classifier = None
        except Exception:
            pass
        import gc; gc.collect()


def main():
    # Use 'spawn' to avoid fork-safety issues with loaded C libraries
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass  # already set

    t_start = time.time()
    log.info("=" * 70)
    log.info("RF Training: %d random KGs with cadastre + OSM ground truth", N_KGS)
    log.info("=" * 70)

    # Get random KGs
    log.info("Fetching KG list from cadastre API...")
    kgs = get_random_kgs(N_KGS)
    log.info("Selected %d random KGs", len(kgs))

    # Prepend infrastructure-targeted KGs (solar, wind, substation)
    # These are placed first so they get processed early, boosting
    # representation of rare classes.  Duplicates with random KGs are
    # harmless — the checkpoint-resume logic skips already-processed KGs.
    infra_kgs = get_infrastructure_kgs()
    if infra_kgs:
        existing_codes = {k["kg_code"] for k in kgs}
        new_infra = [k for k in infra_kgs if k["kg_code"] not in existing_codes]
        kgs = new_infra + kgs
        log.info("Prepended %d infrastructure KGs (%d new)",
                 len(infra_kgs), len(new_infra))

    # Save KG list
    kg_list_data = [{"kg_code": k["kg_code"], "kg_name": k.get("kg_name", "")} for k in kgs]
    for dest in [RESULTS_DIR / "kg_list.json", PERMANENT_DIR / "kg_list.json"]:
        with open(dest, "w") as f:
            json.dump(kg_list_data, f, indent=2)

    # --- Resume from checkpoint (count only, don't load features into RAM) ---
    total_samples = 0
    all_stats = []
    n_success = 0
    n_fail = 0
    completed_kgs = set()

    for ckpt_file in sorted(CHECKPOINT_DIR.glob("kg_*.npz")):
        try:
            ckpt = np.load(ckpt_file, allow_pickle=True)
            kg_code = str(ckpt["kg_code"])
            total_samples += len(ckpt["features"])
            completed_kgs.add(kg_code)
            n_success += 1
        except Exception as e:
            log.warning("Failed to load checkpoint %s: %s", ckpt_file, e)

    stats_ckpt = PERMANENT_DIR / "kg_stats_partial.json"
    if stats_ckpt.exists():
        try:
            all_stats = json.loads(stats_ckpt.read_text())
        except Exception:
            pass

    if completed_kgs:
        log.info("RESUMING: loaded %d KGs (%d samples) from checkpoints",
                 len(completed_kgs), total_samples)

    # --- Load failed KGs (ones that crashed, e.g. OOM) ---
    # On restart, give previously-failed KGs one fresh attempt with the
    # full retry ladder (1.5km → 0.5km → 200m).  KGs that already got a
    # retry pass and failed again stay permanently skipped.
    failed_kgs = set()
    RETRIED_KGS_FILE = PERMANENT_DIR / "retried_kgs.txt"
    prev_retried = set()
    if RETRIED_KGS_FILE.exists():
        prev_retried = set(RETRIED_KGS_FILE.read_text().strip().splitlines()) - {""}
    if FAILED_KGS_FILE.exists():
        prev_failed = set(FAILED_KGS_FILE.read_text().strip().splitlines()) - {""}
        # KGs that already got a retry pass stay permanently failed
        permanently_failed = prev_failed & prev_retried
        # KGs that haven't been retried yet get a fresh chance
        to_retry = prev_failed - prev_retried
        if to_retry:
            log.info("Clearing %d previously-failed KGs for retry (200m window available): %s",
                     len(to_retry), sorted(to_retry))
            prev_retried |= to_retry
            RETRIED_KGS_FILE.write_text("\n".join(sorted(prev_retried)) + "\n")
        failed_kgs = permanently_failed
        if failed_kgs:
            FAILED_KGS_FILE.write_text("\n".join(sorted(failed_kgs)) + "\n")
        else:
            FAILED_KGS_FILE.unlink(missing_ok=True)
    # If we find an in-progress marker, the previous run was interrupted on that KG.
    # Don't add to failed list — it may have been a clean service restart.
    # Just clear the marker and let it be retried naturally.
    if IN_PROGRESS_FILE.exists():
        interrupted_kg = IN_PROGRESS_FILE.read_text().strip()
        if interrupted_kg:
            log.info("Previous run interrupted during KG %s — will retry (not marking as failed)", interrupted_kg)
        IN_PROGRESS_FILE.unlink()
    if failed_kgs:
        log.info("Skipping %d previously-failed KGs: %s", len(failed_kgs), sorted(failed_kgs))

    # Track when we last trained a model checkpoint
    last_model_at_n_success = (n_success // MODEL_CHECKPOINT_INTERVAL) * MODEL_CHECKPOINT_INTERVAL

    # Process each KG
    for i, kg in enumerate(kgs):
        kg_code = kg["kg_code"]

        # --- Check should_stop flag from monitor cronjob ---
        _monitor_state_file = Path("data/monitor_state.json")
        if _monitor_state_file.exists():
            try:
                _mstate = json.loads(_monitor_state_file.read_text())
                if _mstate.get("should_stop"):
                    log.info("="*60)
                    log.info("STOPPING: monitor cronjob set should_stop=true")
                    log.info("  Reason: status=%s, mature_peak_oob=%.4f at %d KGs",
                             _mstate.get("last_status", "?"),
                             _mstate.get("mature_peak_oob", 0),
                             _mstate.get("mature_peak_n_kgs", 0))
                    log.info("="*60)
                    break
            except Exception:
                pass

        if kg_code in completed_kgs:
            log.info("[%d/%d] KG %s — already checkpointed, skipping",
                     i + 1, len(kgs), kg_code)
            continue

        if kg_code in failed_kgs:
            log.info("[%d/%d] KG %s — previously failed (OOM/crash), skipping",
                     i + 1, len(kgs), kg_code)
            continue

        log.info("-" * 50)
        log.info("[%d/%d] Processing KG %s (%s)",
                 i + 1, len(kgs), kg_code, kg.get("kg_name", ""))

        # Memory check — force GC before starting a new KG
        import gc; gc.collect()
        try:
            rss_mb = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1024
            log.info("  Parent RSS: %.0f MB", rss_mb)
        except Exception:
            pass

        # Mark this KG as in-progress (crash detection)
        IN_PROGRESS_FILE.write_text(kg_code + "\n")

        try:
            # Run with timeout to prevent stuck KGs from blocking forever.
            # Retry ladder: 1.5km → 0.5km → 200m → 100m crop window.
            # 3km almost always timed out (segmentation too slow at that scale)
            # so we start at 1.5km which completes in ~10min typically.
            features, labels, stats = None, None, None
            for attempt_km in [1.5, 0.5, 0.2, 0.1]:
                import gc; gc.collect()
                pool = multiprocessing.Pool(processes=1)
                try:
                    # Skip slow Copernicus API on tiny windows — 10m
                    # resolution data isn't useful at 200m/100m and the
                    # API timeouts eat the entire 20-min budget.
                    use_cop = attempt_km >= 0.5
                    async_result = pool.apply_async(
                        process_one_kg,
                        args=(kg,),
                        kwds={"include_copernicus": use_cop,
                              "include_osm": True,
                              "max_km": attempt_km})
                    try:
                        features, labels, stats = async_result.get(
                            timeout=KG_TIMEOUT_SECONDS)
                        break  # success
                    except multiprocessing.TimeoutError:
                        log.warning(
                            "  → TIMEOUT after %d min for KG %s (%.1fkm window)",
                            KG_TIMEOUT_SECONDS // 60, kg_code, attempt_km)
                        pool.terminate()
                        pool.join()
                        if attempt_km <= 0.1:
                            # Already retried at 100m — give up
                            log.error(
                                "  → TIMEOUT on 100m retry too — skipping KG %s",
                                kg_code)
                            failed_kgs.add(kg_code)
                            FAILED_KGS_FILE.write_text(
                                "\n".join(sorted(failed_kgs)) + "\n")
                            if IN_PROGRESS_FILE.exists():
                                IN_PROGRESS_FILE.unlink()
                            n_fail += 1
                            features = None
                            break
                        else:
                            next_km = {1.5: 0.5, 0.5: 0.2, 0.2: 0.1}[attempt_km]
                            log.info("  → Retrying KG %s with %.0fm window",
                                     kg_code, next_km * 1000)
                            continue
                finally:
                    pool.close()
                    pool.join()
            if features is None and stats is None:
                # Timed out on all attempts
                continue
            stats["index"] = i
            all_stats.append(stats)

            if features:
                n_new = len(features)
                total_samples += n_new
                n_success += 1
                log.info("  → +%d samples (total: %d from %d KGs)",
                         n_new, total_samples, n_success)

                # Per-KG checkpoint
                ckpt_path = CHECKPOINT_DIR / f"kg_{kg_code}.npz"
                np.savez_compressed(
                    ckpt_path,
                    kg_code=kg_code,
                    features=np.array(features, dtype=object),
                    labels=np.array(labels, dtype=object),
                )
                log.info("  Checkpoint saved: %s (%d samples)", ckpt_path.name, len(features))

                # KG completed successfully — clear in-progress marker
                if IN_PROGRESS_FILE.exists():
                    IN_PROGRESS_FILE.unlink()

                # Model checkpoint every N successful KGs
                if n_success >= last_model_at_n_success + MODEL_CHECKPOINT_INTERVAL:
                    # Free per-KG memory before loading all checkpoints for training
                    del features, labels
                    import gc; gc.collect()
                    train_stats = train_and_save_model(
                        n_success,
                        tag=f"checkpoint_{n_success}kg")
                    last_model_at_n_success = n_success
                    if train_stats:
                        log.info("  ✓ Model checkpoint saved at %d KGs (OOB=%.4f)",
                                 n_success, train_stats["oob_score"])
                    # Clear downloaded cache data to prevent memory buildup
                    _clear_downloaded_caches()
            else:
                n_fail += 1
                log.warning("  → FAILED: %s", stats.get("error", "no labelled segments"))
                # Clear in-progress marker (KG completed, just had no data)
                if IN_PROGRESS_FILE.exists():
                    IN_PROGRESS_FILE.unlink()
        except Exception as e:
            # Detect Copernicus credits exhaustion from subprocess
            if 'CreditsExhaustedError' in type(e).__name__ or \
               ('402' in str(e) and 'PaymentRequired' in str(e)):
                log.warning("\n" + "=" * 60)
                log.warning("COPERNICUS CREDITS EXHAUSTED — PAUSING TRAINING")
                log.warning("Update credentials in copernicus.py and restart rf_train")
                log.warning("=" * 60)
                # Write a marker file so the status endpoint can report it
                pause_file = PERMANENT_DIR / "credits_paused.txt"
                pause_file.write_text(
                    f"{__import__('datetime').datetime.utcnow().isoformat()}\n"
                    f"Copernicus credits exhausted. Waiting for new credits.\n"
                )
                # Don't mark KG as failed — we want to retry it
                if IN_PROGRESS_FILE.exists():
                    IN_PROGRESS_FILE.unlink()
                # Sleep and retry periodically (15 min)
                import copernicus as _cop
                while True:
                    log.info("Credits paused — sleeping 15 min before retry...")
                    time.sleep(900)
                    # Try a tiny request to see if credits are back
                    try:
                        _cop._connection = None  # force reconnect
                        _cop.credits_exhausted = False
                        conn = _cop._get_connection()
                        cube = conn.load_collection(
                            'SENTINEL2_L2A',
                            spatial_extent={'west': 15, 'south': 47,
                                            'east': 15.01, 'north': 47.01},
                            temporal_extent=['2024-06-01', '2024-06-15'],
                            bands=['B04'],
                        )
                        cube.max_time().download()
                        log.info("Credits restored! Resuming training.")
                        if pause_file.exists():
                            pause_file.unlink()
                        break
                    except Exception:
                        log.info("Still no credits — will retry in 15 min")
                # Retry this KG (don't increment i, don't mark failed)
                continue
            n_fail += 1
            log.error("  → EXCEPTION: %s", traceback.format_exc())
            all_stats.append({"kg_code": kg_code, "error": str(e), "index": i})
            # Track this KG as failed so we skip it on restart
            failed_kgs.add(kg_code)
            FAILED_KGS_FILE.write_text("\n".join(sorted(failed_kgs)) + "\n")
            if IN_PROGRESS_FILE.exists():
                IN_PROGRESS_FILE.unlink()

        # Save progress
        progress = {
            "completed": n_success + n_fail,
            "total": len(kgs),
            "success": n_success,
            "fail": n_fail,
            "total_samples": total_samples,
            "elapsed_min": round((time.time() - t_start) / 60, 1),
            "completed_kgs": sorted(completed_kgs | {kg_code}),
        }
        for dest in [RESULTS_DIR / "progress.json", PERMANENT_DIR / "progress.json"]:
            with open(dest, "w") as f:
                json.dump(progress, f, indent=2)

        with open(stats_ckpt, "w") as f:
            json.dump(all_stats, f, indent=2, default=str)

        completed_kgs.add(kg_code)

        import gc
        gc.collect()

        if (i + 1) % 5 == 0 or len(completed_kgs) % 5 == 0:
            log.info("Progress: %d/%d done (%d success), %d samples, %.1f min elapsed",
                     len(completed_kgs), len(kgs), n_success, total_samples,
                     (time.time() - t_start) / 60)

    # Final stats
    for dest in [RESULTS_DIR / "kg_stats.json", PERMANENT_DIR / "kg_stats.json"]:
        with open(dest, "w") as f:
            json.dump(all_stats, f, indent=2, default=str)

    log.info("=" * 70)
    log.info("Collection complete: %d KGs succeeded, %d failed", n_success, n_fail)
    log.info("Total training samples: %d", total_samples)

    source_dist = {}
    for s in all_stats:
        for src, cnt in s.get("source_counts", {}).items():
            source_dist[src] = source_dist.get(src, 0) + cnt
    log.info("Ground truth sources:")
    for src, cnt in sorted(source_dist.items(), key=lambda x: -x[1]):
        log.info("  %-20s %6d", src, cnt)

    if total_samples < 20:
        log.error("Not enough samples to train (%d < 20)", total_samples)
        return

    # Final model training (loads all checkpoints from disk)
    log.info("=" * 70)
    log.info("Training FINAL Random Forest on %d samples from %d KGs...",
             total_samples, n_success)
    train_stats = train_and_save_model(n_success, tag="final")

    if train_stats:
        # Load labels just for the report
        _, all_labels_report = _load_all_checkpoints()
        label_dist = {}
        for lbl in all_labels_report:
            label_dist[lbl] = label_dist.get(lbl, 0) + 1
        del all_labels_report
        log.info("Label distribution:")
        for lbl, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
            log.info("  %-15s %6d (%.1f%%)", lbl, cnt, 100 * cnt / max(total_samples, 1))

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_kgs_processed": n_success,
            "n_kgs_failed": n_fail,
            "n_total_samples": total_samples,
            "training_stats": train_stats,
            "label_distribution": label_dist,
            "source_distribution": source_dist,
            "elapsed_minutes": round((time.time() - t_start) / 60, 1),
        }
        for dest in [RESULTS_DIR / "training_report.json", PERMANENT_DIR / "training_report.json"]:
            with open(dest, "w") as f:
                json.dump(report, f, indent=2, default=str)

        log.info("Report saved to %s", PERMANENT_DIR / "training_report.json")

    elapsed = time.time() - t_start
    log.info("Total time: %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
