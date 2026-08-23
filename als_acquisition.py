"""Real ALS acquisition dates per flight block (Q1, forestry feedback round 2).

BEV's ALS folders (20220915 / 20230915 / 20240915) are *national mosaic
snapshots*, not flight dates: each yearly mosaic is stitched from rolling
block-by-block acquisitions, so a tile in the "20220915" mosaic may have
been flown 2010-2021.  BEV publishes per-mosaic "Aktualitaet DGM - ALS"
shapefiles (flight-block footprints with a ``Flugjahr`` attribute) on
www.bev.gv.at -> Digitales Gelaendehoehenmodell -> ALS-Hoehenraster.

We vendor those as simplified GeoJSON (EPSG:3035) in
``data/als_acquisition/flight_blocks_<dataset>.geojson`` and expose
:func:`lookup` which, given an AOI geometry, returns the intersecting
flight blocks, the area-dominant flight year (range), and an *effective
acquisition date* estimate for growth normalisation.

Caveats baked into the returned dict:
* ``Flugjahr`` is a year (sometimes a range like "2019-2020" or a
  DTM/DSM split like "DTM:2019, DSM:2010"); no month/day is published, so
  the effective date assumes mid-year (Jul 1) of the (DSM) flight year.
* Blocks are coarse (state-level campaign polygons); an AOI near a block
  boundary may mix flight years -> we report per-block coverage fractions.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data" / "als_acquisition"

#: Mosaic datasets with a vendored flight-block overlay.
AVAILABLE = ("20220915", "20230915", "20240915", "20250915")

_lock = threading.Lock()
_cache: dict[str, tuple[list, object]] = {}  # dataset -> (records, STRtree)

_YEAR_RE = re.compile(r"(19|20)\d{2}")


def _parse_flugjahr(raw: str | None) -> tuple[int | None, int | None]:
    """Parse BEV Flugjahr strings -> (year_from, year_to).

    Handles '2010', '2022-23', '2019-2020', '2008-12',
    'DTM:2019, DSM:2010' (uses the DSM year: canopy heights come from the
    DSM). Returns (None, None) when unparseable.
    """
    if not raw:
        return None, None
    s = str(raw).strip()
    if "DSM" in s.upper():
        m = re.search(r"DSM\s*:?\s*((19|20)\d{2})", s, re.I)
        if m:
            y = int(m.group(1))
            return y, y
    years = [int(m.group(0)) for m in _YEAR_RE.finditer(s)]
    if not years:
        return None, None
    y0, y1 = min(years), max(years)
    # short-suffix ranges: '2022-23', '2008-12'
    m = re.match(r"^\s*((19|20)\d{2})\s*-\s*(\d{2})\s*$", s)
    if m:
        y0 = int(m.group(1))
        y1 = int(str(y0)[:2] + m.group(3))
        if y1 < y0:
            y1 = y0
    return y0, y1


def _load(dataset: str):
    """Load + spatially index the flight blocks for a mosaic dataset."""
    with _lock:
        if dataset in _cache:
            return _cache[dataset]
        path = DATA_DIR / f"flight_blocks_{dataset}.geojson"
        if not path.exists():
            _cache[dataset] = ([], None)
            return _cache[dataset]
        from shapely.geometry import shape
        from shapely.strtree import STRtree
        try:
            fc = json.loads(path.read_text())
            recs = []
            for f in fc.get("features", []):
                g = shape(f["geometry"])
                if not g.is_valid:
                    g = g.buffer(0)
                p = f.get("properties", {})
                y0, y1 = _parse_flugjahr(p.get("flugjahr"))
                recs.append({
                    "geom": g,
                    "flugjahr_raw": p.get("flugjahr"),
                    "year_from": y0, "year_to": y1,
                    "gebiet": p.get("gebiet"),
                    "bemerkung": p.get("bemerkung"),
                })
            tree = STRtree([r["geom"] for r in recs]) if recs else None
            _cache[dataset] = (recs, tree)
        except Exception as e:
            log.warning("als_acquisition: failed to load %s: %s", path, e)
            _cache[dataset] = ([], None)
        return _cache[dataset]


def _effective_date(y0: int, y1: int) -> str:
    """Mid-year of the (midpoint) flight year — best available estimate."""
    return date((y0 + y1) // 2, 7, 1).isoformat()


def lookup(geom_3035, dataset: str) -> dict:
    """Acquisition metadata for an AOI geometry (EPSG:3035) + mosaic dataset.

    Returns::

        {nominal: '20240915', known: bool, blocks: [...],
         flown_from: '2022-01-01', flown_to: '2022-12-31',
         effective_date: '2022-07-01', source: '...', note: '...'}

    ``known`` is False when no vendored overlay exists for the dataset or
    no block intersects the AOI; callers must then fall back to the
    nominal mosaic date and say so.
    """
    out = {
        "nominal": dataset,
        "known": False,
        "blocks": [],
        "source": ("BEV 'Aktualitaet DGM - ALS' flight-block overlay "
                   "(www.bev.gv.at, ALS-Hoehenraster); Flugjahr has no "
                   "month/day, effective_date assumes mid-year"),
    }
    recs, tree = _load(dataset)
    if not recs or tree is None:
        out["note"] = ("acquisition dates unknown - no flight-block overlay "
                       f"for mosaic {dataset}; nominal mosaic date used")
        return out
    aoi_area = max(geom_3035.area, 1.0)
    hits = []
    try:
        idxs = tree.query(geom_3035)
    except Exception:
        idxs = range(len(recs))
    for i in idxs:
        r = recs[int(i)]
        try:
            inter = r["geom"].intersection(geom_3035).area
        except Exception:
            continue
        if inter <= 0:
            continue
        hits.append((inter, r))
    if not hits:
        out["note"] = ("acquisition dates unknown - AOI outside all flight "
                       "blocks; nominal mosaic date used")
        return out
    hits.sort(key=lambda t: -t[0])
    for inter, r in hits[:5]:
        out["blocks"].append({
            "flugjahr": r["flugjahr_raw"],
            "year_from": r["year_from"], "year_to": r["year_to"],
            "gebiet": r["gebiet"],
            "coverage_frac": round(inter / aoi_area, 3),
            **({"bemerkung": r["bemerkung"]} if r.get("bemerkung") else {}),
        })
    dom = hits[0][1]
    if dom["year_from"] is None:
        out["note"] = (f"dominant flight block has unparseable Flugjahr "
                       f"{dom['flugjahr_raw']!r}; nominal mosaic date used")
        return out
    out["known"] = True
    out["flown_from"] = f"{dom['year_from']}-01-01"
    out["flown_to"] = f"{dom['year_to']}-12-31"
    out["effective_date"] = _effective_date(dom["year_from"], dom["year_to"])
    dom_frac = hits[0][0] / aoi_area
    if dom_frac < 0.9 and len(hits) > 1:
        out["note"] = (f"AOI spans multiple flight blocks (dominant covers "
                       f"{dom_frac:.0%}); effective_date from dominant block")
    return out


def effective_days(acq_a: dict, acq_b: dict) -> int | None:
    """Days between two lookup() results' effective dates, if both known."""
    if not (acq_a.get("known") and acq_b.get("known")):
        return None
    da = date.fromisoformat(acq_a["effective_date"])
    db = date.fromisoformat(acq_b["effective_date"])
    return (db - da).days
