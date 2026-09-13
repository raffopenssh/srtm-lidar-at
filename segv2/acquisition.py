"""Real acquisition dates for the v2 features (segv2).

The full GPKG names layers by *mosaic* (DTM_2022 / DSM_2024 / Ortho_2020) —
none of those are flight dates:

* ALS mosaics 2022/2023/2024 are stitched from rolling flight blocks
  (2006-2024). 64 %% of KGs have the *same flight* in all three mosaics, so
  DSM_2024 - DSM_2022 is resampling noise there; elsewhere the true span is
  1-15 yr. ~100 KGs have DTM and DSM from different flights ("DTM:2019,
  DSM:2009"). Source: BEV "Aktualitaet DGM - ALS" flight blocks vendored in
  ``data/als_acquisition`` (see ``als_acquisition.py``); all 8440 KGs resolve.
* Ortho "2020" is the 20221027 RGBI series = flights 2018-2021; the operate
  ID (``2019370``) carries the flight year (``ortho_io.RGBI_OPERATES``).

:func:`als_year_rasters` paints per-pixel DSM / DTM flight years for a
window (blocks are polygons, so a tile can straddle two flights);
:func:`ortho_flight_year` resolves the dominant operate year for a bbox.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
from rasterio import features as rfeatures
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import als_acquisition  # noqa: E402
import ortho_io  # noqa: E402

log = logging.getLogger("segv2.acquisition")

#: mosaic label (as in GPKG layer names) → BEV dataset folder
MOSAIC_DATASET = {2022: "20220915", 2023: "20230915", 2024: "20240915", 2025: "20250915"}


def _block_years(rec: dict) -> tuple[int | None, int | None]:
    """(dtm_year, dsm_year) for one flight-block record."""
    raw = str(rec.get("flugjahr_raw") or "")
    if "DSM" in raw.upper():
        import re
        m_dtm = re.search(r"DTM\s*:?\s*((?:19|20)\d{2})", raw, re.I)
        m_dsm = re.search(r"DSM\s*:?\s*((?:19|20)\d{2})", raw, re.I)
        dtm = int(m_dtm.group(1)) if m_dtm else rec["year_to"]
        dsm = int(m_dsm.group(1)) if m_dsm else rec["year_to"]
        return dtm, dsm
    y = rec["year_to"]  # for ranges ('2010-11') take the later year: mosaic uses the newest
    return y, y


def als_year_rasters(transform, shape_hw, mosaic_year: int) -> dict:
    """Per-pixel flight years for one ALS mosaic over a window.

    Returns {'dtm_year': int16 raster, 'dsm_year': int16 raster,
             'known_frac': float, 'blocks': [...]} (0 = unknown pixel)."""
    ds = MOSAIC_DATASET.get(mosaic_year)
    h, w = shape_hw
    out = {"dtm_year": np.zeros((h, w), np.int16), "dsm_year": np.zeros((h, w), np.int16),
           "known_frac": 0.0, "blocks": []}
    if ds is None:
        return out
    recs, tree = als_acquisition._load(ds)
    if not recs or tree is None:
        return out
    win = box(transform.c, transform.f + h * transform.e, transform.c + w * transform.a, transform.f)
    hits = []
    for i in tree.query(win):
        r = recs[int(i)]
        inter = r["geom"].intersection(win)
        if inter.is_empty or r["year_to"] is None:
            continue
        hits.append((inter.area, r, inter))
    if not hits:
        return out
    hits.sort(key=lambda t: t[0])  # paint smallest first, largest last (largest wins overlaps)
    for area, r, inter in hits:
        dtm_y, dsm_y = _block_years(r)
        m = rfeatures.rasterize([(inter, 1)], out_shape=(h, w), transform=transform,
                                fill=0, dtype="uint8", all_touched=True).astype(bool)
        out["dtm_year"][m] = dtm_y
        out["dsm_year"][m] = dsm_y
        out["blocks"].append({"flugjahr": r["flugjahr_raw"], "dtm_year": dtm_y, "dsm_year": dsm_y,
                              "frac": round(area / win.area, 3)})
    out["known_frac"] = float((out["dsm_year"] > 0).mean())
    return out


def ortho_flight_year(bounds_3035, mosaic_year: int) -> int | None:
    """Flight year of the RGBI operate the processor would have used for this
    bbox + ortho slot (newest overlapping operate of the slot's series)."""
    from pyproj import Transformer
    t = Transformer.from_crs(3035, 4326, always_xy=True)
    x0, y0, x1, y1 = bounds_3035
    xs, ys = zip(*[t.transform(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])
    ops = ortho_io.find_rgbi_operates(min(ys), min(xs), max(ys), max(xs), year=mosaic_year)
    if not ops:
        return None
    return int(ops[0][:4])


def audit_kg(bounds_3035) -> dict:
    """Metadata completeness for one KG bbox — used by build_dataset as a gate."""
    from rasterio.transform import from_bounds
    x0, y0, x1, y1 = bounds_3035
    tf = from_bounds(x0, y0, x1, y1, 200, 200)
    rep = {}
    for my in (2022, 2023, 2024):
        r = als_year_rasters(tf, (200, 200), my)
        rep[f"als_{my}"] = {"known_frac": round(r["known_frac"], 3), "blocks": r["blocks"][:4]}
    for oy in (2024, 2023, 2020):
        rep[f"ortho_{oy}"] = ortho_flight_year(bounds_3035, oy)
    rep["ok"] = all(rep[f"als_{my}"]["known_frac"] >= 0.95 for my in (2022, 2023, 2024))
    return rep
