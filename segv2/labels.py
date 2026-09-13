"""v2 ground-truth label raster for one AOI (EPSG:3035, 1 m).

Sources, in *decreasing* priority (later sources never overwrite earlier ones):

  1. cadastre building footprints (B(Geb) 41)            → roof   (needs nDSM ≥ 2 m)
  2. OSM road centrelines buffered by fclass              → road / path
  3. OSM rail centrelines buffered                        → rail
  4. OSM water areas + waterway lines buffered            → water
  5. INVEKOS Schläge 2024-1 (AMA) crop-type polygons      → crop / grass / vineyard /
                                                            orchard / hedge / shrub /
                                                            water / garden …
  6. cadastre NFL landuse polygons (BEV NS codes, corrected Aug-2026 table)
     – cadastre 95 Straße *minus* OSM road buffer         → grass  (verge)
     – cadastre 59/60/64 Gewässer *minus* OSM water       → grass/shrub (riparian)
     – others per NS_CODE_TO_TYPE

Every pixel also gets a *source id* (LABEL_SOURCE) so the training harness can
weight / ablate sources, and the per-source height + NDVI vetoes from v1
(`train_rf_4000kg`) are kept: e.g. a 'grass' pixel with nDSM > 2 m is unlabelled
(tree canopy over a legal meadow), a 'road' pixel with NDVI > 0.3 is unlabelled.

Everything is fetched from cadastre-process-api (cadastre + OSM, gzip) and the
local INVEKOS GPKG (`data/invekos/INSPIRE_SCHLAEGE_2024-1_POLYGON.gpkg`,
EPSG:31287). No BEV / openEO traffic.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import requests
from pyproj import Transformer
from rasterio import features as rfeatures
from shapely.geometry import shape, box
from shapely.ops import transform as shp_transform

log = logging.getLogger("segv2.labels")

ROOT = Path(__file__).resolve().parent.parent
CADASTRE_BASE = "https://cadastre-process-api.exe.xyz/api/v1"
INVEKOS_GPKG = ROOT / "data/invekos/INSPIRE_SCHLAEGE_2024-1_POLYGON.gpkg"

_T_4326_3035 = Transformer.from_crs(4326, 3035, always_xy=True)
_T_31287_3035 = Transformer.from_crs(31287, 3035, always_xy=True)
_T_3035_4326 = Transformer.from_crs(3035, 4326, always_xy=True)

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
# v1 types kept, plus new ones the data clearly shows (rail, wetland, glacier).
TYPES = [
    "tree", "shrub", "grass", "hedge", "water", "wetland",
    "roof", "greenhouse",
    "road", "path", "parking", "rail",
    "crop", "orchard", "vineyard", "garden",
    "bare_soil", "rock", "glacier", "earthwork",
]
TYPE_ID = {t: i + 1 for i, t in enumerate(TYPES)}
ID_TYPE = {v: k for k, v in TYPE_ID.items()}

SOURCES = ["footprint", "osm_road", "osm_rail", "osm_water", "invekos", "cadastre", "cadastre_verge"]
SOURCE_ID = {s: i + 1 for i, s in enumerate(SOURCES)}

# BEV Katastralmappe SHP V2.9, Tabelle 8 (Nutzungssymbole) — CORRECT table.
NS_CODE_TO_TYPE = {
    40: "orchard",     # Dauerkulturanlagen / Erwerbsgärten (LN(Dk))
    41: "roof",        # Gebäude
    42: "parking",     # Parkplätze
    48: "grass",       # Äcker, Wiesen oder Weiden — ambiguous crop/grass; INVEKOS decides,
                       # cadastre-only fallback labelled 'grass' with low weight (see WEIGHT)
    52: "garden",      # Gärten
    53: "vineyard",    # Weingärten
    54: "grass",       # Alpen
    55: "shrub",       # Krummholzflächen
    56: "tree",        # Wälder
    57: "shrub",       # Verbuschte Flächen
    58: "path",        # Forststraßen
    59: "water",       # Fließende Gewässer (legal parcel incl. banks → OSM water refines)
    60: "water",       # Stehende Gewässer
    61: "wetland",     # Feuchtgebiete
    62: "bare_soil",   # Vegetationsarme Flächen
    63: "parking",     # Betriebsflächen (industrial yards — mostly sealed)
    64: "grass",       # Gewässerrandflächen (riparian strip)
    65: "grass",       # Verkehrsrandflächen (verge)
    72: "grass",       # Friedhöfe (mixed; low weight)
    83: "parking",     # Gebäudenebenflächen (yards/driveways)
    84: "earthwork",   # Abbauflächen, Halden, Deponien
    87: "rock",        # Fels- und Geröllflächen
    88: "glacier",     # Gletscher
    92: "rail",        # Schienenverkehrsanlagen
    95: "road",        # Straßenverkehrsanlagen (legal incl. shoulder → OSM refines)
    96: "grass",       # Freizeitflächen (sports grounds, parks)
}
# codes whose *legal* polygon is much wider than the physical surface — the
# OSM-buffer remainder becomes verge/riparian instead of the nominal type.
_CADASTRE_WIDE = {95: "grass", 59: "grass", 60: "water", 64: "grass", 92: "grass"}
# codes labelled with reduced sample weight (ambiguous / mixed)
LOW_WEIGHT_CODES = {48, 72, 96, 63, 83}

# Height vetoes (m nDSM) — above this the cadastre/OSM/INVEKOS label is probably
# under a canopy or structure and gets unlabelled.
MAX_H = {
    "road": 1.5, "path": 1.5, "parking": 1.5, "rail": 1.5, "water": 1.0, "wetland": 1.5,
    "grass": 2.0, "crop": 2.5, "vineyard": 3.5, "garden": 3.0, "bare_soil": 1.5,
    "glacier": 2.0, "earthwork": 3.0, "hedge": 8.0, "orchard": 8.0, "shrub": 6.0,
}
MIN_H = {"roof": 2.0, "tree": 3.0}
# NDVI vetoes — sealed surfaces must not be green (BEV ortho NDVI, 1 m)
MAX_NDVI = {"road": 0.30, "path": 0.35, "parking": 0.30, "rail": 0.35, "bare_soil": 0.30,
            "roof": 0.35, "water": 0.20, "glacier": 0.15, "rock": 0.30}
# Vegetation must be green. Alpine INVEKOS "Almfutterfläche" / cadastre 52
# (Alpen) polygons legally cover scree, snow beds and bare ridges; without this
# veto ~5 % of "grass" rows sit on bare rock (BEV NDVI < 0.15, elev > 1500 m) and
# the model learns rock→grass.  Thresholds are the ~5th percentile of the BEV
# ortho NDVI per class in the 143-KG build (leaf-off spring flights are real, so
# keep them low).  Same rule is applied segment-wise in train.py for parquets
# built before this veto existed (MIN_NDVI_SEG).
MIN_NDVI = {"grass": 0.15, "tree": 0.20, "shrub": 0.15, "hedge": 0.20, "orchard": 0.10, "wetland": 0.10}

# OSM highway fclass → (type, half-width m). Widths are *pavement* half widths;
# the cadastre legal parcel covers the rest.
OSM_ROAD = {
    "motorway": ("road", 12.0), "motorway_link": ("road", 5.0),
    "trunk": ("road", 8.0), "trunk_link": ("road", 4.5),
    "primary": ("road", 4.5), "primary_link": ("road", 3.5),
    "secondary": ("road", 3.8), "secondary_link": ("road", 3.2),
    "tertiary": ("road", 3.2), "tertiary_link": ("road", 3.0),
    "unclassified": ("road", 2.8), "residential": ("road", 2.8),
    "living_street": ("road", 2.5), "service": ("road", 2.0),
    "pedestrian": ("path", 2.0), "track": ("path", 1.5), "track_grade1": ("path", 1.5),
    "track_grade2": ("path", 1.4), "track_grade3": ("path", 1.2), "track_grade4": ("path", 1.0),
    "track_grade5": ("path", 0.8), "path": ("path", 0.7), "footway": ("path", 0.9),
    "cycleway": ("path", 1.2), "bridleway": ("path", 0.9), "steps": ("path", 0.8),
}
OSM_RAIL = {"rail": 2.0, "light_rail": 1.8, "tram": 1.6, "narrow_gauge": 1.5,
            "subway": 1.8, "monorail": 1.5, "funicular": 1.5, "miniature": 0.8}
OSM_WATER_LINE = {"river": 8.0, "canal": 4.0, "stream": 1.2, "drain": 0.8, "ditch": 0.6}

# INVEKOS SNAR_BEZEICHNUNG → type. Regex on the (upper-case) crop name; first
# match wins, ordered specific → generic.
INVEKOS_RULES: list[tuple[str, str]] = [
    (r"GLÖZ HECKE", "hedge"),
    (r"GLÖZ FELDGEHÖLZ|GLÖZ BAUM|GEBÜSCH", "shrub"),
    (r"GLÖZ TEICH|TÜMPEL", "water"),
    (r"GLÖZ GRABEN|UFERRANDSTREIFEN", "grass"),
    (r"GLÖZ RAIN|BÖSCHUNG|TROCKENSTEINMAUER|STEINRIEGEL|STEINHAGE", "grass"),
    (r"NATURDENKMAL", "tree"),
    (r"^WEIN|REBSCHUL|WEINGARTEN", "vineyard"),
    (r"TAFELÄPFEL|TAFELBIRNEN|MARILLEN|ZWETSCHKEN|KIRSCHEN|PFIRSICHE|WALNÜSSE|EDELKASTANIEN|"
     r"HASELNÜSSE|HOLUNDER|ANDERES OBST|STREUOBST|OBSTANLAGE|QUITTEN|WEICHSELN|BAUMSCHULE", "orchard"),
    (r"STRAUCHBEEREN|ERDBEEREN|HOPFEN", "crop"),
    (r"GEWÄCHSHAUS|FOLIENTUNNEL", "greenhouse"),
    (r"ENERGIEHOLZ|KURZUMTRIEB|CHRISTBAUM|FORST", "tree"),
    (r"ALMWEIDE|ALM|BERGMÄHDER", "grass"),
    (r"WIESE|WEIDE|GRÜNBRACHE|GRÜNLANDBRACHE|GRÜNLAND|STREUWIESE|HUTWEIDE|KLEEGRAS|"
     r"FUTTERGRÄSER|WECHSELWIESE|DAUERWEIDE|ACKERWEIDE|MÄHWIESE|RASEN", "grass"),
    (r"BRACHE|STILLLEGUNG", "grass"),
    (r"HAUS- UND NUTZGARTEN|HAUSGARTEN|NUTZGARTEN", "garden"),
    (r"BLUMEN|ZIERPFLANZEN", "crop"),
    (r".", "crop"),  # everything else in INVEKOS is an arable crop
]
_INVEKOS_RE = [(re.compile(p), t) for p, t in INVEKOS_RULES]


def invekos_type(name: str | None) -> str | None:
    if not name:
        return None
    u = name.upper()
    for rx, t in _INVEKOS_RE:
        if rx.search(u):
            return t
    return None


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------
_SESSION = requests.Session()
_SESSION.headers["Accept-Encoding"] = "gzip"


def _get_json(url, params, timeout=180, retries=3):
    for a in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            log.warning("GET %s failed (%d): %s", url, a, e)
    return None


def _to_3035(geom):
    return shp_transform(_T_4326_3035.transform, geom)


def fetch_cadastre(kg_code: str) -> dict:
    """{'footprints': [geom3035], 'landuse': [(geom3035, code)], 'parcels': [geom3035]}"""
    out = {"footprints": [], "landuse": [], "parcels": []}
    d = _get_json(f"{CADASTRE_BASE}/export/geojson",
                  {"kg": kg_code, "layers": "building_footprints,landuse_polygons,parcels",
                   "include_geometry": "true"})
    if not d:
        return out
    for f in d.get("building_footprints", {}).get("features", []):
        try:
            g = shape(f["geometry"])
            if not g.is_empty:
                out["footprints"].append(_to_3035(g))
        except Exception:  # noqa: BLE001
            pass
    for f in d.get("landuse_polygons", {}).get("features", []):
        try:
            code = int(f["properties"].get("landuse_code"))
            g = shape(f["geometry"])
            if not g.is_empty:
                out["landuse"].append((_to_3035(g), code))
        except Exception:  # noqa: BLE001
            pass
    for f in d.get("parcels", {}).get("features", []):
        try:
            g = shape(f["geometry"])
            if not g.is_empty:
                out["parcels"].append(_to_3035(g))
        except Exception:  # noqa: BLE001
            pass
    log.info("KG %s cadastre: %d footprints, %d landuse polys, %d parcels", kg_code,
             len(out["footprints"]), len(out["landuse"]), len(out["parcels"]))
    return out


def fetch_osm(bbox_wgs: tuple[float, float, float, float]) -> dict:
    """{'road': [(geom3035,fclass)], 'rail': [...], 'water_line': [...], 'water_area': [...]}"""
    w, s, e, n = bbox_wgs
    out = {"road": [], "rail": [], "water_line": [], "water_area": []}
    d = _get_json(f"{CADASTRE_BASE}/osm/geometry",
                  {"bbox": f"{w},{s},{e},{n}", "cat": "road,rail,water,water_area",
                   "limit": 100000})
    if not d:
        return out
    for f in d.get("features", []):
        p = f.get("properties", {})
        try:
            g = _to_3035(shape(f["geometry"]))
        except Exception:  # noqa: BLE001
            continue
        cat, fc = p.get("cat"), p.get("fclass", "")
        if cat == "road":
            out["road"].append((g, fc))
        elif cat == "rail":
            out["rail"].append((g, fc))
        elif cat == "water_area":
            out["water_area"].append((g, fc))
        elif cat == "water":
            if g.geom_type.startswith("Line"):
                out["water_line"].append((g, fc))
    return out


def fetch_invekos(bbox_3035: tuple[float, float, float, float]) -> list[tuple]:
    """[(geom3035, type, snar_name)] from the local AMA Schläge GPKG."""
    if not INVEKOS_GPKG.exists():
        return []
    import pyogrio
    t = Transformer.from_crs(3035, 31287, always_xy=True)
    x0, y0, x1, y1 = bbox_3035
    xs, ys = zip(*[t.transform(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])
    df = pyogrio.read_dataframe(str(INVEKOS_GPKG), bbox=(min(xs), min(ys), max(xs), max(ys)),
                                columns=["SNAR_BEZEICHNUNG"])
    out = []
    for geom, name in zip(df.geometry, df["SNAR_BEZEICHNUNG"]):
        if geom is None or geom.is_empty:
            continue
        ty = invekos_type(name)
        if ty:
            out.append((shp_transform(_T_31287_3035.transform, geom), ty, name))
    return out


# ---------------------------------------------------------------------------
# Rasteriser
# ---------------------------------------------------------------------------
def _burn(shapes, transform, shape_hw, dtype=np.uint8):
    if not shapes:
        return np.zeros(shape_hw, dtype)
    return rfeatures.rasterize(shapes, out_shape=shape_hw, transform=transform,
                               fill=0, dtype=dtype, all_touched=False)


def _paint(label, source, weight, mask, ty, src, w=1.0):
    free = mask & (label == 0)
    label[free] = TYPE_ID[ty]
    source[free] = SOURCE_ID[src]
    weight[free] = w


class LabelContext:
    """Fetched vector inputs for one KG; can rasterise any tile window."""

    def __init__(self, kg_code: str, bbox_3035: tuple[float, float, float, float]):
        self.kg = kg_code
        self.bbox_3035 = bbox_3035
        x0, y0, x1, y1 = bbox_3035
        pts = [_T_3035_4326.transform(x, y) for x, y in ((x0, y0), (x1, y1), (x0, y1), (x1, y0))]
        lons, lats = zip(*pts)
        pad = 0.002
        self.bbox_wgs = (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)
        self.cad = fetch_cadastre(kg_code)
        self.osm = fetch_osm(self.bbox_wgs)
        self.inv = fetch_invekos(bbox_3035)
        log.info("KG %s osm: %d road, %d rail, %d water lines, %d water areas; invekos: %d",
                 kg_code, len(self.osm["road"]), len(self.osm["rail"]),
                 len(self.osm["water_line"]), len(self.osm["water_area"]), len(self.inv))
        # pre-buffered OSM surfaces (3035 metres)
        self.road_polys, self.path_polys = [], []
        for g, fc in self.osm["road"]:
            ty, hw = OSM_ROAD.get(fc, ("road", 2.5))
            (self.road_polys if ty == "road" else self.path_polys).append(g.buffer(hw, cap_style=2))
        self.rail_polys = [g.buffer(OSM_RAIL.get(fc, 1.8), cap_style=2) for g, fc in self.osm["rail"]]
        self.water_polys = [g for g, _ in self.osm["water_area"]]
        self.water_polys += [g.buffer(OSM_WATER_LINE.get(fc, 1.5), cap_style=2)
                             for g, fc in self.osm["water_line"]]
        # OSM road+rail+water union buffered extra 1.5 m → zone where cadastre wide
        # codes should NOT paint their nominal type
        self._osm_zone = self.road_polys + self.path_polys + self.rail_polys + self.water_polys

    def rasterize(self, transform, shape_hw, ndsm: np.ndarray | None, ndvi: np.ndarray | None):
        """Returns (label u8, source u8, weight f32) arrays for the window."""
        h, w = shape_hw
        label = np.zeros((h, w), np.uint8)
        source = np.zeros((h, w), np.uint8)
        weight = np.zeros((h, w), np.float32)
        win = box(transform.c, transform.f + transform.e * h,
                  transform.c + transform.a * w, transform.f)

        def _clip(geoms):
            return [g for g in geoms if g.intersects(win)]

        # 1. footprints → roof
        fp = _burn([(g, 1) for g in _clip(self.cad["footprints"])], transform, shape_hw).astype(bool)
        _paint(label, source, weight, fp, "roof", "footprint")
        # 2. OSM roads / paths
        _paint(label, source, weight, _burn([(g, 1) for g in _clip(self.road_polys)], transform, shape_hw).astype(bool), "road", "osm_road")
        _paint(label, source, weight, _burn([(g, 1) for g in _clip(self.path_polys)], transform, shape_hw).astype(bool), "path", "osm_road")
        # 3. rail
        _paint(label, source, weight, _burn([(g, 1) for g in _clip(self.rail_polys)], transform, shape_hw).astype(bool), "rail", "osm_rail")
        # 4. water
        _paint(label, source, weight, _burn([(g, 1) for g in _clip(self.water_polys)], transform, shape_hw).astype(bool), "water", "osm_water")
        # 5. INVEKOS (shrink by 1 m to avoid boundary bleed)
        inv_by_type: dict[str, list] = {}
        for g, ty, _ in self.inv:
            if g.intersects(win):
                inv_by_type.setdefault(ty, []).append(g.buffer(-1.0))
        for ty, geoms in inv_by_type.items():
            m = _burn([(g, 1) for g in geoms if not g.is_empty], transform, shape_hw).astype(bool)
            _paint(label, source, weight, m, ty, "invekos")
        # 6. cadastre NFL landuse (shrunk 1 m); wide codes → verge remainder
        osm_zone = _burn([(g.buffer(1.5), 1) for g in _clip(self._osm_zone)], transform, shape_hw).astype(bool)
        by_code: dict[int, list] = {}
        for g, code in self.cad["landuse"]:
            if code in NS_CODE_TO_TYPE and g.intersects(win):
                by_code.setdefault(code, []).append(g.buffer(-1.0))
        for code, geoms in by_code.items():
            m = _burn([(g, 1) for g in geoms if not g.is_empty], transform, shape_hw).astype(bool)
            ty = NS_CODE_TO_TYPE[code]
            wgt = 0.5 if code in LOW_WEIGHT_CODES else 1.0
            if code in _CADASTRE_WIDE:
                # inside OSM zone the OSM label already won; remainder = verge/riparian
                _paint(label, source, weight, m & ~osm_zone, _CADASTRE_WIDE[code], "cadastre_verge", 0.7)
            else:
                _paint(label, source, weight, m, ty, "cadastre", wgt)

        # --- physical vetoes ---
        if ndsm is not None:
            for ty, mx in MAX_H.items():
                bad = (label == TYPE_ID[ty]) & (ndsm > mx)
                label[bad] = 0
            for ty, mn in MIN_H.items():
                bad = (label == TYPE_ID[ty]) & (ndsm < mn)
                label[bad] = 0
        if ndvi is not None:
            for ty, mx in MAX_NDVI.items():
                bad = (label == TYPE_ID[ty]) & (ndvi > mx)
                label[bad] = 0
            for ty, mn in MIN_NDVI.items():
                bad = (label == TYPE_ID[ty]) & (ndvi < mn)
                label[bad] = 0
        source[label == 0] = 0
        weight[label == 0] = 0
        return label, source, weight


def summarize(label: np.ndarray, source: np.ndarray) -> dict:
    out = {}
    tot = int((label > 0).sum())
    out["labelled_frac"] = round(tot / label.size, 3)
    ids, cnt = np.unique(label[label > 0], return_counts=True)
    out["by_type"] = {ID_TYPE[int(i)]: int(c) for i, c in zip(ids, cnt)}
    ids, cnt = np.unique(source[source > 0], return_counts=True)
    out["by_source"] = {SOURCES[int(i) - 1]: int(c) for i, c in zip(ids, cnt)}
    return out
