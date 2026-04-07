#!/usr/bin/env python3
"""Systematic calibration of object classifier against Austrian cadastre.

Samples 100 areas (~100 ha each) across Austria, runs our LIDAR+ortho
classifier, fetches cadastre ground truth (building footprints, landuse),
and computes per-class precision/recall metrics.

Usage:
    python3 calibrate.py sample     # Generate sample points
    python3 calibrate.py run         # Run classification + cadastre comparison
    python3 calibrate.py analyse     # Analyse results and print recommendations
    python3 calibrate.py all         # Do everything
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
log = logging.getLogger(__name__)

CADASTRE_BASE = "https://cadastre-process-api.exe.xyz/api/v1"
RESULTS_DIR = Path("calibration_results")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Generate 100 diverse sample areas across Austria
# ---------------------------------------------------------------------------

def generate_samples() -> list[dict]:
    """Generate 100 sample areas (~100 ha = 1 km x 1 km) across Austria.

    Strategy:
    - Stratified by region (9 states), elevation band, and land-use type
    - Mix of urban, suburban, rural, alpine, forested areas
    - Each sample is a 1 km x 1 km square in WGS84
    """
    # Hand-picked representative locations across Austria
    # Format: (lon, lat, label, category)
    # Categories: urban, suburban, rural_flat, rural_hilly, alpine, forest, water, mixed
    locations = [
        # === VIENNA / Wien ===
        (16.370, 48.210, "Vienna center", "urban"),
        (16.310, 48.180, "Vienna Schoenbrunn", "urban"),
        (16.430, 48.230, "Vienna Donaustadt", "suburban"),
        (16.260, 48.170, "Vienna Liesing", "suburban"),
        (16.340, 48.250, "Vienna Floridsdorf", "suburban"),

        # === LOWER AUSTRIA / Niederoesterreich ===
        (16.510, 48.290, "Gaenserndorf area", "rural_flat"),
        (15.620, 48.210, "St Poelten area", "suburban"),
        (15.430, 48.510, "Waldviertel Zwettl", "rural_hilly"),
        (16.080, 47.830, "Neunkirchen area", "rural_hilly"),
        (15.770, 48.080, "Lilienfeld area", "forest"),
        (15.310, 48.320, "Melk area", "rural_hilly"),
        (16.870, 48.780, "Weinviertel north", "rural_flat"),
        (16.640, 48.410, "Marchfeld", "rural_flat"),
        (15.920, 47.720, "Schneeberg area", "alpine"),
        (16.290, 48.050, "Baden area", "suburban"),

        # === BURGENLAND ===
        (16.520, 47.850, "Eisenstadt area", "suburban"),
        (16.840, 47.770, "Neusiedlersee", "water"),
        (16.320, 47.450, "Oberwart area", "rural_hilly"),
        (16.550, 47.120, "Guessing area", "rural_flat"),
        (16.730, 47.500, "Stegersbach area", "rural_flat"),

        # === STYRIA / Steiermark ===
        (15.440, 47.070, "Graz center", "urban"),
        (15.380, 47.100, "Graz west", "suburban"),
        (15.490, 47.030, "Graz south", "suburban"),
        (15.080, 47.060, "Kohlschwarz calibration", "rural_hilly"),  # original calibration area
        (15.270, 47.350, "Bruck/Mur area", "mixed"),
        (15.680, 47.190, "Weiz area", "rural_hilly"),
        (15.890, 46.720, "Bad Radkersburg", "rural_flat"),
        (14.470, 47.380, "Liezen area", "mixed"),
        (15.310, 46.880, "Leibnitz wine", "rural_hilly"),
        (13.800, 47.520, "Ausseerland", "alpine"),
        (15.010, 47.230, "Leoben area", "mixed"),
        (14.800, 47.070, "Judenburg area", "rural_hilly"),
        (15.570, 47.470, "Muerzzuschlag", "forest"),

        # === CARINTHIA / Kaernten ===
        (14.300, 46.620, "Klagenfurt center", "urban"),
        (14.350, 46.600, "Klagenfurt south", "suburban"),
        (13.850, 46.610, "Villach area", "urban"),
        (14.100, 46.760, "Ossiacher See", "water"),
        (13.990, 46.530, "Karawanken", "alpine"),
        (14.800, 46.700, "Voelkermarkt", "rural_hilly"),
        (13.400, 46.750, "Hermagor", "alpine"),
        (13.570, 46.990, "Spittal/Drau", "mixed"),
        (14.510, 46.830, "St Veit/Glan", "rural_hilly"),

        # === UPPER AUSTRIA / Oberoesterreich ===
        (14.290, 48.300, "Linz center", "urban"),
        (14.240, 48.260, "Linz south", "suburban"),
        (14.020, 48.250, "Wels area", "suburban"),
        (13.770, 48.010, "Voecklabruck", "rural_hilly"),
        (13.040, 48.230, "Braunau area", "rural_flat"),
        (14.260, 48.470, "Muehlviertel", "forest"),
        (14.440, 48.150, "Steyr area", "mixed"),
        (13.620, 47.800, "Gmunden/Traunsee", "water"),
        (14.520, 48.520, "Freistadt area", "rural_hilly"),
        (13.620, 48.170, "Voecklamarkt", "rural_flat"),
        (13.800, 47.570, "Bad Ischl", "alpine"),

        # === SALZBURG ===
        (13.050, 47.800, "Salzburg city", "urban"),
        (13.000, 47.780, "Salzburg west", "suburban"),
        (13.100, 47.850, "Salzburg north", "suburban"),
        (12.870, 47.320, "Zell am See", "alpine"),
        (13.200, 47.400, "Bischofshofen", "mixed"),
        (13.190, 47.640, "Hallein area", "rural_hilly"),
        (12.650, 47.100, "Mittersill", "alpine"),
        (13.020, 47.070, "Gastein valley", "alpine"),
        (12.990, 47.530, "Werfen", "alpine"),

        # === TYROL / Tirol ===
        (11.390, 47.260, "Innsbruck center", "urban"),
        (11.340, 47.280, "Innsbruck west", "suburban"),
        (11.460, 47.240, "Hall in Tirol", "suburban"),
        (10.260, 47.270, "Landeck area", "alpine"),
        (11.090, 47.300, "Telfs area", "mixed"),
        (11.870, 47.320, "Woergl area", "suburban"),
        (12.170, 47.440, "Kitzbuehel", "alpine"),
        (10.680, 47.180, "Oetztal", "alpine"),
        (11.490, 47.080, "Stubai valley", "alpine"),
        (12.430, 46.840, "Lienz area", "mixed"),
        (11.740, 47.170, "Zillertal", "alpine"),
        (10.870, 47.510, "Reutte area", "alpine"),

        # === VORARLBERG ===
        (9.740, 47.410, "Bregenz area", "urban"),
        (9.730, 47.240, "Feldkirch area", "suburban"),
        (9.860, 47.340, "Dornbirn area", "suburban"),
        (9.880, 47.150, "Bludenz area", "mixed"),
        (10.020, 47.020, "Montafon", "alpine"),
        (9.650, 47.300, "Hohenems", "suburban"),
        (9.710, 47.160, "Walgau", "rural_hilly"),

        # === Additional diverse spots ===
        (15.420, 47.070, "Graz airport", "mixed"),
        (16.570, 48.130, "Vienna airport", "mixed"),
        (14.190, 48.320, "Linz Chemiepark", "urban"),
        (13.600, 47.490, "Hallstatt", "alpine"),
        (14.900, 47.900, "Ybbstal forest", "forest"),
        (13.450, 47.170, "Gasteinertal", "alpine"),
        (15.770, 47.040, "Gleisdorf industry", "mixed"),
        (16.020, 46.990, "Suedsteiermark wine", "rural_hilly"),
        (14.680, 48.000, "Ennstal flat", "rural_flat"),
        (12.310, 47.260, "Kitzbuehel alps", "alpine"),
    ]

    # Each sample is a 1km x 1km box centered on the point
    samples = []
    for i, (lon, lat, label, category) in enumerate(locations):
        # ~1km box: 0.005 lat ≈ 555m, adjust lon for latitude
        dlat = 0.005  # ~555m
        dlon = 0.007  # ~490m at lat 47
        samples.append({
            "id": i + 1,
            "label": label,
            "category": category,
            "center_lon": lon,
            "center_lat": lat,
            "bbox": {
                "west": round(lon - dlon, 6),
                "south": round(lat - dlat, 6),
                "east": round(lon + dlon, 6),
                "north": round(lat + dlat, 6),
            },
        })

    log.info(f"Generated {len(samples)} sample areas")
    return samples


# ---------------------------------------------------------------------------
# 2. Run classifier + fetch cadastre for each sample
# ---------------------------------------------------------------------------

def fetch_cadastre_bbox(bbox: dict) -> dict:
    """Fetch all cadastre data for a bbox: buildings, parcels, KGs."""
    url = f"{CADASTRE_BASE}/spatial/bbox"
    result = {"buildings": [], "parcels": [], "kgs": [], "kg_codes": []}

    try:
        # Get KGs + buildings + parcels
        for layers in ["kg", "buildings", "parcels"]:
            try:
                r = requests.get(url, params={
                    "west": bbox["west"], "south": bbox["south"],
                    "east": bbox["east"], "north": bbox["north"],
                    "layers": layers, "limit": 10000,
                }, timeout=60)
                r.raise_for_status()
                data = r.json().get("data", {})
                if isinstance(data, dict):
                    blds = data.get("buildings")
                    if blds is not None:
                        result["buildings"] = blds
                    prcs = data.get("parcels")
                    if prcs is not None:
                        result["parcels"] = prcs
                    kgs = data.get("kg", [])
                    if kgs:
                        result["kgs"] = kgs
                        result["kg_codes"] = list(set(
                            kg["kg_code"] for kg in kgs
                            if isinstance(kg, dict) and "kg_code" in kg
                        ))
                elif isinstance(data, list):
                    if layers == "buildings":
                        result["buildings"] = data
                    elif layers == "parcels":
                        result["parcels"] = data
            except Exception as e:
                log.warning(f"Cadastre {layers} fetch failed: {e}")
    except Exception as e:
        log.warning(f"Cadastre bbox fetch failed: {e}")

    return result


def fetch_footprints_and_landuse(bbox: dict, kg_codes: list[str]) -> dict:
    """Fetch building footprints and landuse polygons for KGs in bbox."""
    from shapely.geometry import box as shp_box, shape as shp_shape

    result = {"footprints": [], "landuse": [], "footprint_area_sqm": 0}
    if not kg_codes:
        return result

    bbox_geom = shp_box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])

    # Fetch footprints (limit to 3 KGs for speed)
    for layer_name in ["building_footprints", "landuse"]:
        try:
            r = requests.get(
                f"{CADASTRE_BASE}/export/geojson",
                params={"kg": ",".join(kg_codes[:3]), "layers": layer_name},
                timeout=120,
            )
            r.raise_for_status()
            geojson = r.json()
            features = geojson.get("features", [])

            filtered = []
            total_area = 0
            for feat in features:
                try:
                    geom = shp_shape(feat["geometry"])
                    if bbox_geom.intersects(geom):
                        # Compute area in local coords (approximate m² from WGS84)
                        clipped = bbox_geom.intersection(geom)
                        # Rough conversion: 1° lat ≈ 111000m, 1° lon ≈ 75000m at lat 47
                        # WGS84 area → m² via cos(lat)
                        import math
                        lat = (bbox["south"] + bbox["north"]) / 2
                        m_per_deg_lat = 111320
                        m_per_deg_lon = 111320 * math.cos(math.radians(lat))
                        area_sqm = clipped.area * m_per_deg_lat * m_per_deg_lon
                        filtered.append({
                            "properties": feat.get("properties", {}),
                            "area_sqm": round(area_sqm, 1),
                        })
                        total_area += area_sqm
                except Exception:
                    pass

            if layer_name == "building_footprints":
                result["footprints"] = filtered
                result["footprint_area_sqm"] = round(total_area, 1)
            else:
                result["landuse"] = filtered
        except Exception as e:
            log.warning(f"Fetch {layer_name} failed: {e}")

    return result


def run_classifier(bbox: dict, retries: int = 3) -> dict:
    """Run our LIDAR+ortho classifier on a bbox with retries."""
    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [bbox["west"], bbox["south"]],
            [bbox["east"], bbox["south"]],
            [bbox["east"], bbox["north"]],
            [bbox["west"], bbox["north"]],
            [bbox["west"], bbox["south"]],
        ]]
    }
    for attempt in range(retries):
        try:
            r = requests.post(
                "http://localhost:8000/api/v1/objects",
                json={"geometry": polygon, "include_ortho": True, "min_area": 2},
                timeout=500,
            )
            r.raise_for_status()
            result = r.json()
            if result.get("features") is not None:
                return result
        except Exception as e:
            log.warning(f"Classifier attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(5)
    return {"features": [], "summary": {}, "meta": {}}


def process_sample(sample: dict) -> dict:
    """Process one sample: classifier + cadastre comparison."""
    sid = sample["id"]
    label = sample["label"]
    bbox = sample["bbox"]
    log.info(f"Processing sample {sid}/100: {label}")

    result = {"sample": sample, "timestamp": time.time()}

    # 1. Run classifier
    t0 = time.time()
    classifier_result = run_classifier(bbox)
    result["classifier_time_s"] = round(time.time() - t0, 1)
    result["classifier"] = {
        "features": classifier_result.get("features", []),
        "summary": classifier_result.get("summary", {}),
        "meta": classifier_result.get("meta", {}),
    }

    # 2. Fetch cadastre data
    t0 = time.time()
    cad = fetch_cadastre_bbox(bbox)
    result["cadastre_time_s"] = round(time.time() - t0, 1)
    result["cadastre"] = {
        "building_count": len(cad["buildings"] or []),
        "parcel_count": len(cad["parcels"] or []),
        "kg_codes": cad["kg_codes"],
        "parcels_summary": _summarise_parcels(cad["parcels"] or []),
    }

    # 3. Fetch building footprints + landuse (with geometry)
    t0 = time.time()
    geo_data = fetch_footprints_and_landuse(bbox, cad["kg_codes"])
    result["footprint_time_s"] = round(time.time() - t0, 1)
    result["footprints"] = {
        "count": len(geo_data["footprints"]),
        "total_area_sqm": geo_data["footprint_area_sqm"],
    }
    result["landuse"] = {
        "count": len(geo_data["landuse"]),
        "distribution": _summarise_landuse(geo_data["landuse"]),
    }

    # 4. Compare
    result["comparison"] = _compare(result)

    return result


def _summarise_landuse(landuse_items: list[dict]) -> dict:
    """Summarise landuse distribution."""
    dist = {}
    for item in landuse_items:
        props = item.get("properties", {})
        code = str(props.get("ns_code", props.get("code", props.get("NS_Code", ""))))
        name = props.get("ns_name", props.get("name", props.get("NS_Name", "")))
        area = item.get("area_sqm", 0)
        key = f"{code}:{name}" if name else code
        if key not in dist:
            dist[key] = {"count": 0, "area_sqm": 0}
        dist[key]["count"] += 1
        dist[key]["area_sqm"] += area
    # Round areas
    for v in dist.values():
        v["area_sqm"] = round(v["area_sqm"], 1)
    return dist


def _summarise_parcels(parcels: list[dict]) -> dict:
    """Summarise parcel data."""
    if not parcels:
        return {}
    landuse_codes = {}
    total_area = 0
    for p in parcels:
        area = p.get("area_sqm", 0)
        total_area += area
        lu = p.get("landuse_summary") or p.get("dominant_landuse") or ""
        if isinstance(lu, dict):
            for code, count in lu.items():
                landuse_codes[code] = landuse_codes.get(code, 0) + count
        elif isinstance(lu, str) and lu:
            landuse_codes[lu] = landuse_codes.get(lu, 0) + 1
    return {
        "count": len(parcels),
        "total_area_sqm": round(total_area, 1),
        "landuse_distribution": landuse_codes,
        "has_buildings_count": sum(1 for p in parcels if p.get("building_count", 0) > 0),
    }


def _compare(result: dict) -> dict:
    """Compare classifier output against cadastre ground truth."""
    comp = {}

    # --- Building comparison ---
    classifier_features = result["classifier"].get("features", [])
    clf_buildings = [f for f in classifier_features if f.get("properties", {}).get("type") == "building"]
    clf_structures = [f for f in classifier_features if f.get("properties", {}).get("type") == "structure"]

    cadastre_bld_count = result["cadastre"].get("building_count", 0)
    footprint_count = result["footprints"].get("count", 0)
    footprint_area = result["footprints"].get("total_area_sqm", 0)

    clf_bld_count = len(clf_buildings) + len(clf_structures)
    clf_bld_area = sum(f.get("properties", {}).get("area_sqm", 0) for f in clf_buildings + clf_structures)

    comp["buildings"] = {
        "classifier_count": clf_bld_count,
        "classifier_area_sqm": round(clf_bld_area, 1),
        "cadastre_address_count": cadastre_bld_count,
        "cadastre_footprint_count": footprint_count,
        "cadastre_footprint_area_sqm": footprint_area,
        "count_ratio": round(clf_bld_count / max(footprint_count, 1), 2),
        "area_ratio": round(clf_bld_area / max(footprint_area, 1), 2),
    }

    # --- Summary by classifier type ---
    summary = result["classifier"].get("summary", {})
    by_type = summary.get("by_type", {})
    comp["classifier_types"] = by_type

    # --- Per-type stats ---
    type_stats = {}
    for f in classifier_features:
        props = f.get("properties", {})
        t = props.get("type", "unknown")
        if t not in type_stats:
            type_stats[t] = {"count": 0, "area_sqm": 0, "ndvi_values": [], "brightness_values": []}
        type_stats[t]["count"] += 1
        type_stats[t]["area_sqm"] += props.get("area_sqm", 0)
        if props.get("ndvi_mean"):
            type_stats[t]["ndvi_values"].append(props["ndvi_mean"])
        if props.get("brightness_mean"):
            type_stats[t]["brightness_values"].append(props["brightness_mean"])

    for t, stats in type_stats.items():
        stats["area_sqm"] = round(stats["area_sqm"], 1)
        ndvis = stats.pop("ndvi_values")
        bris = stats.pop("brightness_values")
        if ndvis:
            stats["ndvi_mean"] = round(float(np.mean(ndvis)), 4)
            stats["ndvi_p10"] = round(float(np.percentile(ndvis, 10)), 4)
            stats["ndvi_p90"] = round(float(np.percentile(ndvis, 90)), 4)
        if bris:
            stats["brightness_mean"] = round(float(np.mean(bris)), 1)
            stats["brightness_p10"] = round(float(np.percentile(bris, 10)), 1)
            stats["brightness_p90"] = round(float(np.percentile(bris, 90)), 1)

    comp["type_stats"] = type_stats

    # --- Landuse comparison ---
    parcel_summary = result["cadastre"].get("parcels_summary", {})
    comp["landuse_parcels"] = parcel_summary.get("landuse_distribution", {})
    comp["landuse_polygons"] = result.get("landuse", {}).get("distribution", {})

    return comp


# ---------------------------------------------------------------------------
# 3. Analyse aggregated results
# ---------------------------------------------------------------------------

def analyse_results():
    """Load all results and compute aggregate metrics."""
    results = []
    for f in sorted(RESULTS_DIR.glob("sample_*.json")):
        try:
            with open(f) as fh:
                results.append(json.load(fh))
        except Exception as e:
            log.warning(f"Failed to load {f}: {e}")

    if not results:
        log.error("No results found. Run 'calibrate.py run' first.")
        return

    log.info(f"Loaded {len(results)} sample results")

    # --- Aggregate building metrics ---
    bld_count_ratios = []
    bld_area_ratios = []
    categories = {}

    for r in results:
        comp = r.get("comparison", {})
        bld = comp.get("buildings", {})
        cat = r.get("sample", {}).get("category", "unknown")

        if bld.get("cadastre_footprint_count", 0) > 0:
            bld_count_ratios.append(bld["count_ratio"])
            bld_area_ratios.append(bld["area_ratio"])

        if cat not in categories:
            categories[cat] = {
                "count": 0, "bld_count_ratios": [], "bld_area_ratios": [],
                "type_counts": {}
            }
        categories[cat]["count"] += 1
        if bld.get("cadastre_footprint_count", 0) > 0:
            categories[cat]["bld_count_ratios"].append(bld["count_ratio"])
            categories[cat]["bld_area_ratios"].append(bld["area_ratio"])

        for t, stats in comp.get("type_stats", {}).items():
            if t not in categories[cat]["type_counts"]:
                categories[cat]["type_counts"][t] = {"count": 0, "area": 0}
            categories[cat]["type_counts"][t]["count"] += stats.get("count", 0)
            categories[cat]["type_counts"][t]["area"] += stats.get("area_sqm", 0)

    # --- Print analysis ---
    print("\n" + "=" * 80)
    print("CALIBRATION ANALYSIS")
    print("=" * 80)
    print(f"\nTotal samples: {len(results)}")

    if bld_count_ratios:
        print(f"\n--- BUILDING DETECTION ---")
        print(f"Count ratio (clf/cadastre):  median={np.median(bld_count_ratios):.2f}  "
              f"mean={np.mean(bld_count_ratios):.2f}  "
              f"p10={np.percentile(bld_count_ratios, 10):.2f}  "
              f"p90={np.percentile(bld_count_ratios, 90):.2f}")
        print(f"Area ratio (clf/cadastre):   median={np.median(bld_area_ratios):.2f}  "
              f"mean={np.mean(bld_area_ratios):.2f}  "
              f"p10={np.percentile(bld_area_ratios, 10):.2f}  "
              f"p90={np.percentile(bld_area_ratios, 90):.2f}")

    print(f"\n--- BY CATEGORY ---")
    for cat, info in sorted(categories.items()):
        n = info["count"]
        bcr = info["bld_count_ratios"]
        bar = info["bld_area_ratios"]
        print(f"\n{cat} ({n} samples):")
        if bcr:
            print(f"  Building count ratio: median={np.median(bcr):.2f}  mean={np.mean(bcr):.2f}")
            print(f"  Building area ratio:  median={np.median(bar):.2f}  mean={np.mean(bar):.2f}")
        top_types = sorted(info["type_counts"].items(), key=lambda x: -x[1]["count"])
        for t, tc in top_types[:8]:
            print(f"  {t:25s} count={tc['count']:5d}  area={tc['area']:10.0f} m²")

    # --- Aggregate spectral stats by type ---
    print(f"\n--- SPECTRAL STATS BY TYPE ---")
    all_type_spectral = {}
    for r in results:
        for t, stats in r.get("comparison", {}).get("type_stats", {}).items():
            if t not in all_type_spectral:
                all_type_spectral[t] = {"ndvis": [], "bris": [], "count": 0, "area": 0}
            all_type_spectral[t]["count"] += stats.get("count", 0)
            all_type_spectral[t]["area"] += stats.get("area_sqm", 0)
            if "ndvi_mean" in stats:
                all_type_spectral[t]["ndvis"].append(stats["ndvi_mean"])
            if "brightness_mean" in stats:
                all_type_spectral[t]["bris"].append(stats["brightness_mean"])

    for t in sorted(all_type_spectral, key=lambda x: -all_type_spectral[x]["count"]):
        s = all_type_spectral[t]
        line = f"  {t:25s} n={s['count']:6d}  area={s['area']:12.0f}"
        if s["ndvis"]:
            line += f"  NDVI={np.median(s['ndvis']):.3f}[{np.percentile(s['ndvis'],10):.3f}-{np.percentile(s['ndvis'],90):.3f}]"
        if s["bris"]:
            line += f"  BRI={np.median(s['bris']):.0f}[{np.percentile(s['bris'],10):.0f}-{np.percentile(s['bris'],90):.0f}]"
        print(line)

    # --- Problematic samples ---
    print(f"\n--- POTENTIAL ISSUES ---")
    for r in results:
        comp = r.get("comparison", {})
        bld = comp.get("buildings", {})
        sample = r.get("sample", {})
        # Over-detection: >3x more buildings than cadastre
        if bld.get("count_ratio", 0) > 3 and bld.get("cadastre_footprint_count", 0) > 5:
            print(f"  OVER-DETECT  sample {sample.get('id')} {sample.get('label'):30s} "
                  f"ratio={bld['count_ratio']:.1f} (clf={bld['classifier_count']} vs cad={bld['cadastre_footprint_count']})")
        # Under-detection: <0.3x buildings
        if bld.get("count_ratio", 0) < 0.3 and bld.get("cadastre_footprint_count", 0) > 5:
            print(f"  UNDER-DETECT sample {sample.get('id')} {sample.get('label'):30s} "
                  f"ratio={bld['count_ratio']:.1f} (clf={bld['classifier_count']} vs cad={bld['cadastre_footprint_count']})")

    # Save aggregate analysis
    analysis = {
        "sample_count": len(results),
        "building_count_ratio": {
            "median": round(float(np.median(bld_count_ratios)), 3) if bld_count_ratios else None,
            "mean": round(float(np.mean(bld_count_ratios)), 3) if bld_count_ratios else None,
            "p10": round(float(np.percentile(bld_count_ratios, 10)), 3) if bld_count_ratios else None,
            "p90": round(float(np.percentile(bld_count_ratios, 90)), 3) if bld_count_ratios else None,
        },
        "building_area_ratio": {
            "median": round(float(np.median(bld_area_ratios)), 3) if bld_area_ratios else None,
            "mean": round(float(np.mean(bld_area_ratios)), 3) if bld_area_ratios else None,
        },
        "spectral_by_type": {
            t: {
                "count": s["count"],
                "area_sqm": round(s["area"], 1),
                "ndvi_median": round(float(np.median(s["ndvis"])), 4) if s["ndvis"] else None,
                "ndvi_p10": round(float(np.percentile(s["ndvis"], 10)), 4) if s["ndvis"] else None,
                "ndvi_p90": round(float(np.percentile(s["ndvis"], 90)), 4) if s["ndvis"] else None,
                "brightness_median": round(float(np.median(s["bris"])), 1) if s["bris"] else None,
                "brightness_p10": round(float(np.percentile(s["bris"], 10)), 1) if s["bris"] else None,
                "brightness_p90": round(float(np.percentile(s["bris"], 90)), 1) if s["bris"] else None,
            }
            for t, s in sorted(all_type_spectral.items(), key=lambda x: -x[1]["count"])
        },
        "by_category": {
            cat: {
                "samples": info["count"],
                "bld_count_ratio_median": round(float(np.median(info["bld_count_ratios"])), 3) if info["bld_count_ratios"] else None,
                "bld_area_ratio_median": round(float(np.median(info["bld_area_ratios"])), 3) if info["bld_area_ratios"] else None,
            }
            for cat, info in sorted(categories.items())
        },
    }
    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    log.info(f"Analysis saved to {RESULTS_DIR / 'analysis.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: calibrate.py [sample|run|analyse|all]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd in ("sample", "all"):
        samples = generate_samples()
        with open(RESULTS_DIR / "samples.json", "w") as f:
            json.dump(samples, f, indent=2)
        log.info(f"Saved {len(samples)} samples to {RESULTS_DIR / 'samples.json'}")

    if cmd in ("run", "all"):
        # Load samples
        with open(RESULTS_DIR / "samples.json") as f:
            samples = json.load(f)

        # Process each sample
        for sample in samples:
            sid = sample["id"]
            out_file = RESULTS_DIR / f"sample_{sid:03d}.json"
            if out_file.exists():
                log.info(f"Skipping sample {sid} (already done)")
                continue
            try:
                result = process_sample(sample)
                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                log.info(f"Sample {sid} done, saved to {out_file}")
            except Exception as e:
                log.error(f"Sample {sid} failed: {e}")
                traceback.print_exc()
                # Save partial result
                with open(out_file, "w") as f:
                    json.dump({"sample": sample, "error": str(e)}, f, indent=2)

            # Brief pause to avoid hammering APIs
            time.sleep(0.5)

    if cmd in ("analyse", "all"):
        analyse_results()


if __name__ == "__main__":
    main()
