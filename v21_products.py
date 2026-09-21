"""v21_products — product_version 2.1 additions (docs/v2.1-product-spec.md).

Everything here is pure numpy / rasterio / fiona on arrays the processor already
holds; NO BEV / openEO / network I/O.  Two halves:

* **Peer side** (called from ``austria_processor`` tile loop + light GPKG build):
  - :func:`tile_coarse` — per-tile 5 m block means of DTM/DSM/slope/aspect on an
    absolute 25 m-snapped EPSG:3035 grid (stitchable without resampling), plus
    the per-tree evidence rasters (``dh_cm`` multi-date nDSM change, ``ndvi_i8``,
    ``nir_u8``) and the per-tile ``acq`` (flight-year) record.
  - :func:`stitch_coarse` — full-KG 5 m arrays from the per-tile results.
  - :func:`terrain_grid25` / :func:`landcover_grid25` — the two fixed-size JSON
    indexes (≤ 25 KB per KG together, see the disk rule).
  - :func:`build_tree_apices` — ``tree_inventory.build_inventory`` on the
    stitched 1 m nDSM in memory-bounded chunks, joined with the v2 segment
    classes (stand_context), cadastre footprints and the evidence rasters.
  - :func:`write_terrain_coarse` / :func:`write_tree_apices` — light GPKG layers.

* **Primary side** (``app.py`` fast paths, ``kg_v2_store``):
  - :func:`decode_terrain_grid25` / :func:`decode_landcover_grid25` — arrays back
    from the JSON sections.
  - :func:`split_grid25` / :func:`attach_grid25` — move the two sections between
    the doc and the store's ``grid25`` column (disk-neutral).

Disk rule (hard): per-tree / per-pixel / per-5 m data goes to the light GPKG
only.  JSON gets the 25 m indexes and a < 1 KB acquisition block.  Nothing
here may add per-tree rows to ``search_index.db``.
"""
from __future__ import annotations

import base64
import gzip
import json
import logging
import math

import numpy as np

log = logging.getLogger("austria_processor.v21")

PRODUCT_VERSION = "2.3"
#: manifest ``Entry.version`` written for ``_json_v2`` / ``_light_gpkg_v2`` uploads
MANIFEST_VERSION = "v2.3"
#: product line readable by trees_v3 / the primary (older = fewer fields)
READABLE_MANIFEST_VERSIONS = ("v2.1", "v2.2", "v2.3")


def _ent_get(e, k, default=None):
    if e is None:
        return default
    if isinstance(e, dict):
        return e.get(k, default)
    return getattr(e, k, default)


def v2_products_complete(entries, code: str) -> bool:
    """True iff *code* has the COMPLETE current-product v2 pair on Zenodo:
    a ``_json_v2`` at ``MANIFEST_VERSION`` **and** a committed (size>0)
    ``_light_gpkg_v2``.  Every "is this KG already upgraded?" decision
    (processor eligibility, director candidates/priority, primary
    v2_keep, regen settle) must go through here.  Sep 2026: the parent
    upload loop pushed ``json_v2`` after a failed ``light_gpkg_v2`` PUT
    and every gate keyed on ``_json_v2`` alone, so 60 KGs sat with a
    JSON and no light GPKG and were never re-dispatched.  *entries* may
    be the raw ``{key: dict}`` manifest map or a ``zenodo_client.Manifest``
    (anything with ``.get``)."""
    j2 = entries.get(f"{code}_json_v2")
    if j2 is None or str(_ent_get(j2, "version") or "") != MANIFEST_VERSION:
        return False
    if "error" in str(_ent_get(j2, "status", "") or ""):
        return False
    g2 = entries.get(f"{code}_light_gpkg_v2")
    if g2 is None or int(_ent_get(g2, "size", 0) or 0) <= 0:
        return False
    return "error" not in str(_ent_get(g2, "status", "") or "")
TREE_CROWNS_LAYER = "tree_crowns"
#: crown outline simplification tolerance (m) — half a pixel keeps the
#: watershed footprint honest while dropping ~60 % of the vertices
CROWN_SIMPLIFY_M = 0.5
COARSE_M = 5
GRID_M = 25
NODATA_I16 = -32768
TERRAIN_COARSE_BANDS = ("dtm", "dsm", "ndsm", "slope", "aspect")
TERRAIN_COARSE_LAYER = "terrain_coarse"      # tables: terrain_coarse_<band>
TREE_APICES_LAYER = "tree_apices"
TERRAIN_COARSE_NODATA = -9999.0
#: chunk geometry for the in-memory apex inventory (px on the 1 m grid)
APEX_CHUNK_PX = 2048
APEX_HALO_PX = 48
#: v2 segment types that count as "woody stand" context for an apex
STAND_CONTEXT_TYPES = ("tree", "orchard", "vineyard", "hedge", "garden", "shrub")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _b64gz(raw: bytes) -> str:
    return base64.b64encode(gzip.compress(raw, 6)).decode("ascii")


def _ungz64(s: str) -> bytes:
    return gzip.decompress(base64.b64decode(s))


def snap_down(v: float, step: float = GRID_M) -> float:
    return math.floor(v / step) * step


def snap_up(v: float, step: float = GRID_M) -> float:
    return math.ceil(v / step) * step


def block_mean(arr: np.ndarray, f: int, min_valid_frac: float = 0.3) -> np.ndarray:
    """NaN-aware ``f×f`` block mean.  ``arr`` must have dims divisible by *f*.
    Blocks with fewer than ``min_valid_frac`` finite px become NaN."""
    h, w = arr.shape
    a = arr.reshape(h // f, f, w // f, f).astype(np.float32)
    ok = np.isfinite(a)
    s = np.where(ok, a, 0.0).sum(axis=(1, 3))
    n = ok.sum(axis=(1, 3))
    out = np.full(s.shape, np.nan, np.float32)
    good = n >= max(1, int(min_valid_frac * f * f))
    out[good] = s[good] / n[good]
    return out


def _pad_to_grid(arr: np.ndarray, left: float, top: float, res: float,
                 step: float) -> tuple[np.ndarray, float, float]:
    """Embed the 1 m tile array in a NaN canvas whose origin is snapped to
    *step* metres and whose dims are multiples of ``step/res`` px."""
    h, w = arr.shape
    f = int(round(step / res))
    left_s = snap_down(left, step)
    top_s = snap_up(top, step)
    c_off = int(round((left - left_s) / res))
    r_off = int(round((top_s - top) / res))
    W = int(math.ceil((c_off + w) / f)) * f
    H = int(math.ceil((r_off + h) / f)) * f
    canvas = np.full((H, W), np.nan, np.float32)
    canvas[r_off:r_off + h, c_off:c_off + w] = arr
    return canvas, left_s, top_s


# --------------------------------------------------------------------------
# peer side — per tile
# --------------------------------------------------------------------------

def tile_coarse(tdata: dict, *, dsm_dates: dict | None = None,
                dtm_dates: dict | None = None, dataset: str | None = None,
                spectral: dict | None = None, geom_3035=None,
                ortho_year: int | None = None) -> dict:
    """Per-tile v2.1 evidence, computed from the arrays the segmenter used.

    Returns a dict (pickled with the tile checkpoint):
      ``coarse5``: {left, top, res, dtm, dsm, ndsm, slope, aspect}  (float32,
      NaN nodata, 5 m, origin snapped to 25 m)
      ``dh_cm``: int16 1 m raster of nDSM(newest) − nDSM(oldest) in cm
      (``NODATA_I16``), or None when only one epoch / gap unknown
      ``dh_years``: float flight-year gap (None if unknown)
      ``als_year``: int flight year of the default dataset (None if unknown)
      ``ndvi_i8``: int8 ×100 (−128 nodata) or None;  ``nir_u8``: uint8 or None
      ``red_u8`` / ``green_u8`` / ``blue_u8``: uint8 ortho bands (2.2) or absent
      ``acq``: acquisition record for the JSON ``acquisition.tiles`` table
    """
    import tile_index as ti
    from terrain_analysis import compute_slope, compute_aspect
    tf = tdata["transform"]
    res = abs(tf.a)
    dtm = tdata["dtm"].astype(np.float32)
    dsm = tdata.get("dsm")
    mask = tdata.get("mask")
    if mask is not None:
        dtm = np.where(mask, dtm, np.nan)
    h, w = dtm.shape
    left, top = tf.c, tf.f
    f = int(round(COARSE_M / res))

    slope = compute_slope(dtm, res)
    aspect = compute_aspect(dtm, res)
    asp_ok = (aspect >= 0) & np.isfinite(aspect)
    rad = np.radians(np.where(asp_ok, aspect, 0.0))
    asin = np.where(asp_ok, np.sin(rad), np.nan).astype(np.float32)
    acos = np.where(asp_ok, np.cos(rad), np.nan).astype(np.float32)

    def _c(a):
        canvas, ls, ts = _pad_to_grid(a, left, top, res, GRID_M)
        return block_mean(canvas, f), ls, ts

    dtm5, ls, ts = _c(dtm)
    slope5 = _c(slope)[0]
    s5, c5 = _c(asin)[0], _c(acos)[0]
    asp5 = np.degrees(np.arctan2(s5, c5)) % 360.0
    asp5 = np.where(np.isfinite(s5) & np.isfinite(c5), asp5, np.nan).astype(np.float32)
    out = {"coarse5": {"left": ls, "top": ts, "res": float(COARSE_M),
                       "dtm": dtm5, "slope": slope5, "aspect": asp5}}
    if dsm is not None:
        dsm_m = np.where(mask, dsm.astype(np.float32), np.nan) if mask is not None else dsm.astype(np.float32)
        out["coarse5"]["dsm"] = _c(dsm_m)[0]
        out["coarse5"]["ndsm"] = _c(np.clip(dsm_m - dtm, -5.0, 200.0))[0]

    # --- flight years / acquisition -----------------------------------
    dataset = dataset or ti.DEFAULT_DATASET
    acq_default = acq_old = None
    als_year = None
    try:
        import als_acquisition
        if geom_3035 is None:
            from shapely.geometry import box
            geom_3035 = box(left, top - h * res, left + w * res, top)
        acq_default = als_acquisition.lookup(geom_3035, dataset)
        if acq_default.get("known"):
            als_year = int(acq_default["effective_date"][:4])
    except Exception as e:  # noqa: BLE001
        log.debug("acquisition lookup failed: %s", e)
    out["als_year"] = als_year

    # --- multi-date nDSM change (per-tree growth evidence) --------------
    dh_cm = None
    dh_years = None
    if dsm_dates and dtm_dates and dsm is not None and len(dsm_dates) > 1:
        try:
            import als_acquisition
            oldest = sorted(k for k in dsm_dates if k != dataset)[0]
            acq_old = als_acquisition.lookup(geom_3035, oldest)
            if acq_default and acq_default.get("known") and acq_old.get("known"):
                days = als_acquisition.effective_days(acq_old, acq_default)
                if days and days >= 180:
                    dh_years = round(days / 365.25, 2)
                    d0 = dsm_dates[oldest].astype(np.float32) - dtm_dates[oldest].astype(np.float32)
                    d1 = dsm_dates[dataset].astype(np.float32) - dtm_dates[dataset].astype(np.float32)
                    mh, mw = min(d0.shape[0], d1.shape[0], h), min(d0.shape[1], d1.shape[1], w)
                    dh = np.full((h, w), np.nan, np.float32)
                    dh[:mh, :mw] = d1[:mh, :mw] - d0[:mh, :mw]
                    dh_cm = np.where(np.isfinite(dh), np.clip(np.round(dh * 100), -32000, 32000),
                                     NODATA_I16).astype(np.int16)
        except Exception as e:  # noqa: BLE001
            log.debug("dh computation failed: %s", e)
    out["dh_cm"] = dh_cm
    out["dh_years"] = dh_years

    # --- spectral evidence at 1 m ---------------------------------------
    ndvi_i8 = nir_u8 = None
    if spectral:
        nd = spectral.get("ndvi")
        if nd is not None and nd.shape == (h, w):
            ndvi_i8 = np.where(np.isfinite(nd), np.clip(np.round(nd * 100), -127, 127), -128).astype(np.int8)
        nir = spectral.get("nir")
        if nir is not None and nir.shape == (h, w):
            nir_u8 = np.clip(np.nan_to_num(nir, nan=0.0), 0, 255).astype(np.uint8)
        # 2.2: keep the RGB bands too (uint8, ~2 MB/tile each) so the
        # stitched apex inventory gets per-crown brightness / green_ratio /
        # rg_index → vitality + species_hint exactly as the live v2 engine.
        for band in ("red", "green", "blue"):
            arr = spectral.get(band)
            if arr is not None and arr.shape == (h, w):
                out[f"{band}_u8"] = np.clip(np.nan_to_num(arr, nan=0.0), 0, 255).astype(np.uint8)
    out["ndvi_i8"] = ndvi_i8
    out["nir_u8"] = nir_u8

    def _yr(acq):
        if not acq or not acq.get("known"):
            return None
        return int(acq["effective_date"][:4])
    out["acq"] = {
        "dtm_year": _yr(acq_default), "dsm_year": _yr(acq_default),
        "ortho_year": int(ortho_year) if ortho_year else None,
        "eff_date": (acq_default or {}).get("effective_date"),
        "years_gap": dh_years,
        "oldest_eff_date": (acq_old or {}).get("effective_date") if acq_old else None,
        "known": bool((acq_default or {}).get("known")),
    }
    return out


# --------------------------------------------------------------------------
# peer side — stitched
# --------------------------------------------------------------------------

def stitch_coarse(tile_seg_results: list) -> dict | None:
    """Full-KG 5 m arrays from ``tr['v21']['coarse5']`` of every tile.
    Returns {left, top, res, cols, rows, dtm, dsm, ndsm, slope, aspect} or
    None when no tile carries coarse data."""
    parts = [tr["v21"]["coarse5"] for tr in tile_seg_results
             if isinstance(tr.get("v21"), dict) and tr["v21"].get("coarse5")]
    if not parts:
        return None
    res = float(COARSE_M)
    left = min(p["left"] for p in parts)
    top = max(p["top"] for p in parts)
    right = max(p["left"] + p["dtm"].shape[1] * res for p in parts)
    bottom = min(p["top"] - p["dtm"].shape[0] * res for p in parts)
    # keep the full grid a multiple of GRID_M so grid25 is an exact 5×5 reduce
    right = snap_up(right, GRID_M)
    bottom = snap_down(bottom, GRID_M)
    cols = int(round((right - left) / res))
    rows = int(round((top - bottom) / res))
    if cols * rows > 60_000_000:
        log.warning("stitch_coarse: %dx%d too large — skipping", cols, rows)
        return None
    out = {"left": left, "top": top, "res": res, "cols": cols, "rows": rows}
    for band in TERRAIN_COARSE_BANDS:
        if not any(band in p for p in parts):
            continue
        full = np.full((rows, cols), np.nan, np.float32)
        for p in parts:
            a = p.get(band)
            if a is None:
                continue
            r0 = int(round((top - p["top"]) / res))
            c0 = int(round((p["left"] - left) / res))
            h, w = a.shape
            r1, c1 = min(r0 + h, rows), min(c0 + w, cols)
            if r1 <= r0 or c1 <= c0:
                continue
            sub = a[:r1 - r0, :c1 - c0]
            dst = full[r0:r1, c0:c1]
            fill = np.isnan(dst) & np.isfinite(sub)
            dst[fill] = sub[fill]
        out[band] = full
    return out


def terrain_grid25(coarse: dict) -> dict | None:
    """JSON ``terrain.grid25`` from the stitched 5 m arrays (5×5 block mean)."""
    if not coarse or "dtm" not in coarse:
        return None
    f = int(round(GRID_M / coarse["res"]))
    elev = block_mean(coarse["dtm"], f, 0.2)
    slope = block_mean(coarse["slope"], f, 0.2) if "slope" in coarse else None
    rows, cols = elev.shape
    valid = np.isfinite(elev)
    if not valid.any():
        return None
    base_dm = int(math.floor(np.nanmin(elev) * 10))
    e_dm = np.where(valid, np.round(elev * 10) - base_dm, 0).astype(np.int64).ravel()
    # carry the last valid value across nodata so deltas stay small
    if not valid.all():
        v = valid.ravel()
        idx = np.where(v, np.arange(v.size), 0)
        np.maximum.accumulate(idx, out=idx)
        e_dm = e_dm[idx]
    delta = np.diff(np.concatenate([[0], e_dm]))
    if np.abs(delta).max() > 32767:
        coding, payload = "i32", e_dm.astype("<i4").tobytes()
    else:
        coding, payload = "delta_i16", delta.astype("<i2").tobytes()
    sec = {
        "cell_m": GRID_M, "cols": int(cols), "rows": int(rows), "origin": "nw",
        "crs": "EPSG:3035", "x0": float(coarse["left"]), "y0": float(coarse["top"]),
        "nodata": NODATA_I16, "unit": "dm", "base_dm": base_dm, "coding": coding,
        "elev": _b64gz(payload),
        "n_valid": int(valid.sum()),
    }
    if not valid.all():
        sec["valid"] = _b64gz(np.packbits(valid.ravel()).tobytes())
    if slope is not None:
        sec["slope"] = _b64gz(np.where(np.isfinite(slope), np.clip(np.round(slope), 0, 90), 255)
                              .astype(np.uint8).tobytes())
        sec["slope_nodata"] = 255
    return sec


def landcover_grid25(seg_type_full: np.ndarray, full_left: float, full_top: float,
                     res: float, coarse: dict) -> dict | None:
    """JSON ``landcover.grid25`` — dominant v2 segment class per 25 m cell on
    exactly the ``terrain.grid25`` grid (same x0/y0/cols/rows)."""
    if seg_type_full is None or not coarse:
        return None
    from object_segmentation import ALL_TYPE_NAMES
    f = int(round(GRID_M / res))
    fc = int(round(GRID_M / coarse["res"]))
    rows, cols = coarse["rows"] // fc, coarse["cols"] // fc
    # embed seg_type on the 25 m-snapped canvas that coarse uses
    H, W = rows * f, cols * f
    canvas = np.zeros((H, W), np.uint8)
    r_off = int(round((coarse["top"] - full_top) / res))
    c_off = int(round((full_left - coarse["left"]) / res))
    h, w = seg_type_full.shape
    r1, c1 = min(r_off + h, H), min(c_off + w, W)
    if r1 <= r_off or c1 <= c_off:
        return None
    canvas[r_off:r1, c_off:c1] = seg_type_full[:r1 - r_off, :c1 - c_off]
    codes = [int(c) for c in np.unique(canvas) if c != 0]
    if not codes:
        return None
    best_cnt = np.zeros((rows, cols), np.int32)
    best_cls = np.zeros((rows, cols), np.uint8)
    blk = canvas.reshape(rows, f, cols, f)
    for c in codes:
        cnt = (blk == c).sum(axis=(1, 3), dtype=np.int32)
        win = cnt > best_cnt
        best_cnt[win] = cnt[win]
        best_cls[win] = c
    frac = np.clip(np.round(best_cnt * 255.0 / (f * f)), 0, 255).astype(np.uint8)
    legend = {str(c): ALL_TYPE_NAMES.get(c, f"code_{c}") for c in codes}
    # 2.2: class-fraction grids independent of dominance, so an AOI canopy
    # denominator can be AREA-WEIGHTED (cell ∩ AOI × frac) instead of the
    # centre-in-AOI + dominant-class rule that lost a 25 m rim per stand
    # (FEEDBACK-5 §6).  ~1 KB gz each.
    name_to_code = {v: k for k, v in ALL_TYPE_NAMES.items()}
    woody_codes = [name_to_code[t] for t in STAND_CONTEXT_TYPES if t in name_to_code]
    woody = np.isin(blk, woody_codes).sum(axis=(1, 3), dtype=np.int32)
    tree_code = name_to_code.get("tree")
    treec = (blk == tree_code).sum(axis=(1, 3), dtype=np.int32) if tree_code else np.zeros_like(woody)
    to_u8 = lambda cnt: np.clip(np.round(cnt * 255.0 / (f * f)), 0, 255).astype(np.uint8)  # noqa: E731
    return {
        "cell_m": GRID_M, "cols": int(cols), "rows": int(rows), "origin": "nw",
        "crs": "EPSG:3035", "x0": float(coarse["left"]), "y0": float(coarse["top"]),
        "coding": "u8", "cls": _b64gz(best_cls.tobytes()),
        "cover_frac": _b64gz(frac.tobytes()), "legend": legend,
        "woody_frac": _b64gz(to_u8(woody).tobytes()),
        "tree_frac": _b64gz(to_u8(treec).tobytes()),
        "n_classified": int((best_cls > 0).sum()),
    }


def attach_canopy_frac(lc: dict | None, canopy_counts: np.ndarray | None) -> None:
    """Add ``canopy_frac`` (u8, fraction of 1 m px in the apex-inventory canopy
    mask: nDSM ≥ min_tree_height inside a v2 segment, roofs/crop rejected) to
    a ``landcover.grid25`` section, in place.  The v2 pixel canopy, area-
    weighted onto the 25 m grid."""
    if not lc or canopy_counts is None:
        return
    if canopy_counts.shape != (int(lc["rows"]), int(lc["cols"])):
        return
    f2 = float(GRID_M * GRID_M)
    lc["canopy_frac"] = _b64gz(np.clip(np.round(canopy_counts * 255.0 / f2), 0, 255)
                               .astype(np.uint8).tobytes())


# --------------------------------------------------------------------------
# decode (primary side)
# --------------------------------------------------------------------------

def decode_terrain_grid25(sec: dict) -> dict:
    """→ {elev: float32 m (NaN nodata), slope: float32 deg or None, x0, y0,
    cell_m, cols, rows}"""
    rows, cols = int(sec["rows"]), int(sec["cols"])
    raw = _ungz64(sec["elev"])
    if sec.get("coding") == "i32":
        e_dm = np.frombuffer(raw, "<i4").astype(np.int64)
    else:
        e_dm = np.cumsum(np.frombuffer(raw, "<i2").astype(np.int64))
    elev = (e_dm + int(sec.get("base_dm", 0))) / 10.0
    elev = elev.astype(np.float32).reshape(rows, cols)
    if sec.get("valid"):
        valid = np.unpackbits(np.frombuffer(_ungz64(sec["valid"]), np.uint8))[:rows * cols]
        elev[~valid.reshape(rows, cols).astype(bool)] = np.nan
    slope = None
    if sec.get("slope"):
        s = np.frombuffer(_ungz64(sec["slope"]), np.uint8).reshape(rows, cols).astype(np.float32)
        slope = np.where(s == int(sec.get("slope_nodata", 255)), np.nan, s)
    return {"elev": elev, "slope": slope, "x0": float(sec["x0"]), "y0": float(sec["y0"]),
            "cell_m": float(sec.get("cell_m", GRID_M)), "cols": cols, "rows": rows}


def decode_landcover_grid25(sec: dict) -> dict:
    rows, cols = int(sec["rows"]), int(sec["cols"])
    cls = np.frombuffer(_ungz64(sec["cls"]), np.uint8).reshape(rows, cols)
    frac = np.frombuffer(_ungz64(sec["cover_frac"]), np.uint8).reshape(rows, cols) / 255.0
    out = {"cls": cls, "cover_frac": frac.astype(np.float32), "legend": dict(sec.get("legend") or {}),
           "x0": float(sec["x0"]), "y0": float(sec["y0"]),
           "cell_m": float(sec.get("cell_m", GRID_M)), "cols": cols, "rows": rows}
    for k in ("woody_frac", "tree_frac", "canopy_frac"):        # 2.2, optional
        if sec.get(k):
            out[k] = (np.frombuffer(_ungz64(sec[k]), np.uint8).reshape(rows, cols) / 255.0).astype(np.float32)
        else:
            out[k] = None
    return out


def split_grid25(doc: dict) -> tuple[dict, bytes | None]:
    """Remove ``terrain.grid25`` + ``landcover.grid25`` from *doc* (shallow-
    copied) and return (doc_without, gz_json_bytes_or_None)."""
    t = (doc.get("terrain") or {}).get("grid25")
    l = (doc.get("landcover") or {}).get("grid25")
    if t is None and l is None:
        return doc, None
    d = dict(doc)
    if t is not None:
        d["terrain"] = {k: v for k, v in doc["terrain"].items() if k != "grid25"}
    if l is not None:
        lc = {k: v for k, v in doc["landcover"].items() if k != "grid25"}
        if lc:
            d["landcover"] = lc
        else:
            d.pop("landcover", None)
    payload = gzip.compress(json.dumps({"terrain": t, "landcover": l},
                                       separators=(",", ":")).encode(), 6)
    return d, payload


def attach_grid25(doc: dict, payload: bytes | None) -> dict:
    if not payload:
        return doc
    g = json.loads(gzip.decompress(payload))
    d = dict(doc)
    if g.get("terrain") is not None:
        d["terrain"] = dict(doc.get("terrain") or {}, grid25=g["terrain"])
    if g.get("landcover") is not None:
        d["landcover"] = dict(doc.get("landcover") or {}, grid25=g["landcover"])
    return d


def grid25_payload(payload: bytes) -> dict:
    """Decode a store ``grid25`` column → {terrain: sec|None, landcover: sec|None}."""
    return json.loads(gzip.decompress(payload))


# --------------------------------------------------------------------------
# tree apices
# --------------------------------------------------------------------------

def _sample_tiles(tile_seg_results: list, key: str, es: np.ndarray, ns: np.ndarray,
                  nodata, scale: float = 1.0, radius: int = 0) -> np.ndarray:
    """Point-sample a per-tile 1 m raster ``tr['v21'][key]`` at EPSG:3035 coords.
    ``radius`` > 0 takes the mean of the (2r+1)² window (nodata-aware)."""
    out = np.full(es.shape, np.nan, np.float32)
    todo = np.ones(es.shape, bool)
    for tr in tile_seg_results:
        v = tr.get("v21") or {}
        arr = v.get(key)
        if arr is None or not todo.any():
            continue
        left, bottom, right, top = tr["bounds_3035"]
        sel = todo & (es >= left) & (es < right) & (ns > bottom) & (ns <= top)
        if not sel.any():
            continue
        h, w = arr.shape
        cols = np.clip(((es[sel] - left)).astype(int), 0, w - 1)
        rows = np.clip(((top - ns[sel])).astype(int), 0, h - 1)
        if radius <= 0:
            vals = arr[rows, cols].astype(np.float32)
            vals[arr[rows, cols] == nodata] = np.nan
        else:
            vals = np.full(rows.shape, np.nan, np.float32)
            for i, (r, c) in enumerate(zip(rows, cols)):
                win = arr[max(0, r - radius):r + radius + 1, max(0, c - radius):c + radius + 1]
                ok = win != nodata
                if ok.any():
                    vals[i] = float(win[ok].mean())
        out[sel] = vals * scale
        todo[sel] = False
    return out


def _compose_tiles(tile_seg_results: list, key: str, tf, shape, nodata=None,
                   scale: float = 1.0) -> np.ndarray | None:
    """Paint the per-tile 1 m rasters ``tr['v21'][key]`` onto a float32 canvas
    on grid *tf*/*shape* (NaN where no tile / nodata).  Returns None when no
    tile carries *key* — chunk-sized only, so the stitched apex inventory
    never holds a full-KG ortho band in memory."""
    H, W = shape
    out = None
    res = abs(tf.a)
    for tr in tile_seg_results:
        v = tr.get("v21") or {}
        arr = v.get(key)
        if arr is None:
            continue
        left, bottom, right, top = tr["bounds_3035"]
        c_off = int(round((left - tf.c) / res))
        r_off = int(round((tf.f - top) / res))
        h, w = arr.shape
        dr0, dc0 = max(0, r_off), max(0, c_off)
        dr1, dc1 = min(H, r_off + h), min(W, c_off + w)
        if dr1 <= dr0 or dc1 <= dc0:
            continue
        if out is None:
            out = np.full((H, W), np.nan, np.float32)
        src = arr[dr0 - r_off:dr1 - r_off, dc0 - c_off:dc1 - c_off].astype(np.float32)
        if nodata is not None:
            src = np.where(arr[dr0 - r_off:dr1 - r_off, dc0 - c_off:dc1 - c_off] == nodata, np.nan, src)
        dst = out[dr0:dr1, dc0:dc1]
        fill = np.isnan(dst) & np.isfinite(src)
        dst[fill] = src[fill] * scale
    return out


def _chunk_spectral(tile_seg_results: list, tf, shape) -> tuple[dict | None, np.ndarray | None]:
    """(spectral dict for tree_inventory.build_inventory, nir) from the tile
    ortho bands: ndvi, brightness, green_ratio, rg_index — same definitions
    as ``ortho_io.compute_spectral_indices`` so the product's vitality /
    species_hint equal a live v2 call."""
    ndvi = _compose_tiles(tile_seg_results, "ndvi_i8", tf, shape, nodata=-128, scale=0.01)
    nir = _compose_tiles(tile_seg_results, "nir_u8", tf, shape)
    if ndvi is None or nir is None:
        return None, None
    spec = {"ndvi": ndvi}
    r = _compose_tiles(tile_seg_results, "red_u8", tf, shape)
    g = _compose_tiles(tile_seg_results, "green_u8", tf, shape)
    bl = _compose_tiles(tile_seg_results, "blue_u8", tf, shape)
    if r is not None and g is not None and bl is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            tot = r + g + bl
            spec["brightness"] = (tot / 3.0).astype(np.float32)
            spec["green_ratio"] = np.where(tot > 0, g / tot, np.nan).astype(np.float32)
            spec["rg_index"] = np.where(r + g > 0, (r - g) / (r + g), np.nan).astype(np.float32)
    return spec, nir


def _crown_polygons(labels: np.ndarray, tf, wanted: dict) -> dict:
    """{label: shapely Polygon (EPSG:3035, simplified)} for the labels in
    *wanted* (label → tree).  Largest part when the watershed region is
    split by the halo cut."""
    from rasterio.features import shapes
    from shapely.geometry import shape as _shape
    keep = np.isin(labels, list(wanted.keys()))
    if not keep.any():
        return {}
    out: dict = {}
    for geom, val in shapes(labels.astype(np.int32), mask=keep, transform=tf, connectivity=8):
        lab = int(val)
        poly = _shape(geom)
        prev = out.get(lab)
        if prev is None or poly.area > prev.area:
            out[lab] = poly
    for lab, poly in list(out.items()):
        sp = poly.simplify(CROWN_SIMPLIFY_M, preserve_topology=True)
        if sp.is_empty or not sp.is_valid:
            sp = poly
        out[lab] = sp
    return out


def build_tree_apices(ndsm_full: np.ndarray, full_tf, seg_type_full: np.ndarray | None,
                      labels_full: np.ndarray | None, all_objects: list,
                      tile_seg_results: list, building_geoms: list | None,
                      *, det_info: dict | None = None, grid25: dict | None = None,
                      crowns_out: list | None = None,
                      canopy_counts_out: list | None = None) -> list[dict]:
    """Apex inventory on the stitched 1 m nDSM, memory-bounded by chunking.

    Trees whose apex falls inside a chunk's core are kept; the ``APEX_HALO_PX``
    halo makes crown geometry at chunk seams equal to an unchunked run.

    2.2 additions (FEEDBACK-5):
      * per-crown ortho RGBI spectra (``_chunk_spectral``) → ``vitality``,
        ``species_hint``, ``ndvi_mean/p10``, ``nir_mean``, ``brightness_mean``,
        ``green_ratio_mean``; vitality/species are re-ranked KG-wide after all
        chunks (``tree_inventory.assign_*``) so chunk seams don't bias them;
      * ``crowns_out`` (list) receives ``{tree_id, wkb}`` crown outlines for
        :func:`write_tree_crowns`;
      * ``canopy_counts_out`` (list) receives the int32 (rows, cols) count of
        canopy px per 25 m cell on the *grid25* grid (``{x0,y0,rows,cols}``)
        for :func:`attach_canopy_frac`.
    Returns rows ready for :func:`write_tree_apices`.
    """
    import tree_inventory as tv
    from object_segmentation import ALL_TYPE_NAMES
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    H, W = ndsm_full.shape
    # Grid anchor guard: the apex = pixel centre of the *integer-metre* BEV
    # grid; a fractional full_tf origin would silently shift every tree_id
    # (FEEDBACK-5 §1). Re-anchor + shout rather than bake an off-grid product.
    try:
        import raster_io as _rio
        _tf2, _chg = _rio.reanchor_transform(full_tf)
        if _chg:
            log.warning("build_tree_apices: full_tf origin (%.3f, %.3f) off the integer-metre "
                        "grid — re-anchored to (%.0f, %.0f)", full_tf.c, full_tf.f, _tf2.c, _tf2.f)
            full_tf = _tf2
    except Exception:  # noqa: BLE001
        pass
    obj_conf = {}
    if labels_full is not None:
        for o in all_objects:
            obj_conf[int(o.obj_id)] = float(getattr(o, "confidence", 0.0) or 0.0)
    trees_all: list = []            # (Tree, gr, gc)
    n_chunks = 0
    rej_total = 0
    spectral_chunks = 0
    canopy_counts = None
    if grid25:
        canopy_counts = np.zeros((int(grid25["rows"]), int(grid25["cols"])), np.int32)
    for r0 in range(0, H, APEX_CHUNK_PX):
        for c0 in range(0, W, APEX_CHUNK_PX):
            r1, c1 = min(r0 + APEX_CHUNK_PX, H), min(c0 + APEX_CHUNK_PX, W)
            hr0, hc0 = max(0, r0 - APEX_HALO_PX), max(0, c0 - APEX_HALO_PX)
            hr1, hc1 = min(H, r1 + APEX_HALO_PX), min(W, c1 + APEX_HALO_PX)
            nd = ndsm_full[hr0:hr1, hc0:hc1]
            mask = np.isfinite(nd)
            if not mask.any() or float(np.nanmax(nd)) < tv.DEFAULT_MIN_TREE_HEIGHT:
                continue
            tf = Affine(full_tf.a, full_tf.b, full_tf.c + hc0 * full_tf.a,
                        full_tf.d, full_tf.e, full_tf.f + hr0 * full_tf.e)
            bmask = None
            if building_geoms:
                try:
                    bmask = rasterize([(g, 1) for g in building_geoms], out_shape=nd.shape,
                                      transform=tf, fill=0, dtype=np.uint8, all_touched=True).astype(bool)
                    if not bmask.any():
                        bmask = None
                except Exception:  # noqa: BLE001
                    bmask = None
            spec, nir = _chunk_spectral(tile_seg_results, tf, nd.shape)
            if spec is not None:
                spectral_chunks += 1
            n_chunks += 1
            di: dict = {}
            try:
                trees, labels, canopy = tv.build_inventory(
                    np.nan_to_num(nd, nan=0.0), mask, tf, building_mask=bmask,
                    spectral=spec, nir=nir, det_info=di)
            except Exception as e:  # noqa: BLE001
                log.warning("tree_apices chunk (%d,%d) failed: %s", r0, c0, e)
                continue
            del spec, nir
            rej_total += int(di.get("rejected_by_surface", 0) or 0)
            core = {}
            for t in trees:
                gr, gc = t.apex_row + hr0, t.apex_col + hc0
                if not (r0 <= gr < r1 and c0 <= gc < c1):
                    continue     # halo — owned by the neighbouring chunk
                trees_all.append((t, gr, gc))
                core[t.label] = t
            if crowns_out is not None and core:
                try:
                    for lab, poly in _crown_polygons(labels, tf, core).items():
                        crowns_out.append({"tree_id": core[lab].tree_id, "geom": poly})
                except Exception as e:  # noqa: BLE001
                    log.warning("tree_crowns chunk (%d,%d) failed: %s", r0, c0, e)
            if canopy_counts is not None:
                # core part of the canopy mask → 25 m cell counts (area-weighted
                # canopy denominator for trees_v3; never double counts halos)
                cr = canopy[r0 - hr0:r1 - hr0, c0 - hc0:c1 - hc0]
                rr, cc = np.nonzero(cr)
                if rr.size:
                    es = full_tf.c + (cc + c0 + 0.5) * full_tf.a
                    ns = full_tf.f + (rr + r0 + 0.5) * full_tf.e
                    ci = np.floor((es - grid25["x0"]) / GRID_M).astype(np.int64)
                    ri = np.floor((grid25["y0"] - ns) / GRID_M).astype(np.int64)
                    ok = (ci >= 0) & (ri >= 0) & (ci < canopy_counts.shape[1]) & (ri < canopy_counts.shape[0])
                    np.add.at(canopy_counts, (ri[ok], ci[ok]), 1)
            del labels, canopy
    if canopy_counts_out is not None:
        canopy_counts_out.append(canopy_counts)
    has_spec = spectral_chunks > 0
    if det_info is not None:
        det_info.update(n_chunks=n_chunks, rejected_by_surface=rej_total,
                        building_mask_used=bool(building_geoms),
                        spectral_chunks=spectral_chunks, spectral="ortho_rgbi_1m" if has_spec else None)
    if not trees_all:
        return []
    # KG-wide re-ranking: vitality (stressed = ≤ p10 NDVI within leaf class)
    # and species hint percentiles are population-relative — rank over the
    # whole KG, not per 2 km chunk.
    tlist = [t for t, _, _ in trees_all]
    tv.assign_vitality(tlist, has_spec)
    tv.assign_species_hint(tlist, has_spec)

    rows_out: list[dict] = []
    for t, gr, gc in trees_all:
        st_code = int(seg_type_full[gr, gc]) if seg_type_full is not None else 0
        st_name = ALL_TYPE_NAMES.get(st_code, "none") if st_code else "none"
        conf = None
        if labels_full is not None:
            lab = int(labels_full[gr, gc])
            if lab:
                conf = obj_conf.get(lab)
        sp = t.spectral or {}
        _f = lambda k: (round(float(sp[k]), 3) if k in sp and np.isfinite(sp[k]) else None)  # noqa: E731
        rows_out.append({
            "tree_id": t.tree_id, "e": t.apex_e, "n": t.apex_n,
            "h_m": t.height_m, "crown_r_m": t.crown_radius_mean_m,
            "crown_area_m2": t.crown_area_sqm,
            "detection_source": t.detection_source, "detection_conf": t.detection_conf,
            "surface_class": t.surface_class, "tree_likelihood": t.tree_likelihood,
            "stand_context": st_name if st_name in STAND_CONTEXT_TYPES else
                             ("other" if st_code else "none"),
            "segment_type": st_name, "segment_type_conf": conf,
            "leaf_type_hint": t.leaf_type, "leaf_type_conf": t.leaf_type_conf,
            "vitality": t.vitality if has_spec else "unknown",
            "vitality_conf": t.vitality_conf if has_spec else None,
            "species_hint": t.species_hint if has_spec else "unknown",
            "species_conf": t.species_conf if has_spec else None,
            "ndvi_mean": _f("ndvi_mean"), "ndvi_p10": _f("ndvi_p10"),
            "nir_mean": (round(float(sp["nir_mean"]), 1) if "nir_mean" in sp else None),
            "brightness_mean": (round(float(sp["brightness_mean"]), 1) if "brightness_mean" in sp else None),
            "green_ratio_mean": _f("green_ratio_mean"),
            "dbh_est_cm": t.dbh_est_cm, "volume_m3_est": t.volume_m3_est,
        })
    # per-tree evidence from the tile rasters (point samples, no full arrays)
    es = np.array([r["e"] for r in rows_out]); ns = np.array([r["n"] for r in rows_out])
    dh = _sample_tiles(tile_seg_results, "dh_cm", es, ns, NODATA_I16, 0.01, radius=1)
    yrs = {}
    for tr in tile_seg_results:
        v = tr.get("v21") or {}
        yrs[id(tr)] = (v.get("dh_years"), v.get("als_year"))
    # tile lookup per tree for dh_years / als_year (first containing tile)
    dh_years = np.full(es.shape, np.nan); als_year = np.full(es.shape, np.nan)
    todo = np.ones(es.shape, bool)
    for tr in tile_seg_results:
        left, bottom, right, top = tr["bounds_3035"]
        sel = todo & (es >= left) & (es < right) & (ns > bottom) & (ns <= top)
        if sel.any():
            gy, ay = yrs[id(tr)]
            dh_years[sel] = gy if gy else np.nan
            als_year[sel] = ay if ay else np.nan
            todo[sel] = False
    for i, r in enumerate(rows_out):
        g = dh[i]; y = dh_years[i]
        r["dh_per_year_m"] = (round(float(g) / float(y), 3)
                              if np.isfinite(g) and np.isfinite(y) and y > 0 else None)
        r["als_year"] = None if not np.isfinite(als_year[i]) else int(als_year[i])
    return rows_out


TREE_APICES_SCHEMA = {"geometry": "Point", "properties": [
    ("tree_id", "str"), ("h_m", "float"), ("crown_r_m", "float"), ("crown_area_m2", "float"),
    ("detection_source", "str"), ("detection_conf", "float"), ("surface_class", "str"),
    ("tree_likelihood", "float"), ("stand_context", "str"), ("segment_type", "str"),
    ("segment_type_conf", "float"), ("dh_per_year_m", "float"), ("als_year", "int"),
    ("leaf_type_hint", "str"), ("leaf_type_conf", "float"),
    # 2.2 — per-crown BEV ortho RGBI spectra + derived (FEEDBACK-5 §3)
    ("vitality", "str"), ("vitality_conf", "float"), ("species_hint", "str"), ("species_conf", "float"),
    ("ndvi_mean", "float"), ("ndvi_p10", "float"), ("nir_mean", "float"),
    ("brightness_mean", "float"), ("green_ratio_mean", "float"),
    ("dbh_est_cm", "float"), ("volume_m3_est", "float")]}

TREE_CROWNS_SCHEMA = {"geometry": "Polygon", "properties": [("tree_id", "str")]}


def write_tree_crowns(gpkg_path: str, crowns: list[dict]) -> int:
    """``tree_crowns`` polygon layer (EPSG:3035, no R-tree): one simplified
    watershed crown outline per apex, joined to ``tree_apices`` by
    ``tree_id``.  Only ``tree_id`` is stored — every attribute lives on the
    apex row (file-size rule)."""
    import fiona
    from fiona.crs import from_epsg
    from shapely.geometry import mapping
    n = 0
    # no R-tree: crowns are joined by tree_id after the (indexed) apex query;
    # skipping the index saves ~25 % of the layer (~70 B/crown)
    with fiona.open(gpkg_path, "w", driver="GPKG", layer=TREE_CROWNS_LAYER,
                    schema=TREE_CROWNS_SCHEMA, crs=from_epsg(3035), SPATIAL_INDEX="NO") as dst:
        batch = []
        for c in crowns:
            g = c["geom"]
            if g.is_empty:
                continue
            if g.geom_type == "MultiPolygon":
                g = max(g.geoms, key=lambda p: p.area)
            batch.append({"geometry": mapping(g), "properties": {"tree_id": c["tree_id"]}})
            if len(batch) >= 5000:
                dst.writerecords(batch); n += len(batch); batch = []
        if batch:
            dst.writerecords(batch); n += len(batch)
    return n


def write_tree_apices(gpkg_path: str, rows: list[dict]) -> int:
    """Write the ``tree_apices`` point layer (EPSG:3035).  GDAL's GPKG driver
    creates the ``gpkg_rtree_index`` extension by default (SPATIAL_INDEX=YES)."""
    import fiona
    from fiona.crs import from_epsg
    keys = [k for k, _ in TREE_APICES_SCHEMA["properties"]]
    n = 0
    with fiona.open(gpkg_path, "w", driver="GPKG", layer=TREE_APICES_LAYER,
                    schema=TREE_APICES_SCHEMA, crs=from_epsg(3035)) as dst:
        batch = []
        for r in rows:
            batch.append({"geometry": {"type": "Point", "coordinates": (r["e"], r["n"])},
                          "properties": {k: r.get(k) for k in keys}})
            if len(batch) >= 5000:
                dst.writerecords(batch); n += len(batch); batch = []
        if batch:
            dst.writerecords(batch); n += len(batch)
    return n


def write_terrain_coarse(gpkg_path: str, coarse: dict) -> list[str]:
    """``terrain_coarse_<band>`` float32 gridded-coverage tables at 5 m
    (dtm/dsm/ndsm in m, slope/aspect in deg; nodata -9999).  float32 rather
    than int16: GDAL stores Int16 GPKG coverages as offset UInt16 PNG tiles
    and reads nodata back as +32768, which every consumer would have to
    special-case."""
    import rasterio
    from rasterio.transform import from_origin
    tf = from_origin(coarse["left"], coarse["top"], coarse["res"], coarse["res"])
    written = []
    units = {"dtm": "m", "dsm": "m", "ndsm": "m", "slope": "deg", "aspect": "deg"}
    for band in TERRAIN_COARSE_BANDS:
        a = coarse.get(band)
        if a is None:
            continue
        name = f"{TERRAIN_COARSE_LAYER}_{band}"
        arr = np.where(np.isfinite(a), a, TERRAIN_COARSE_NODATA).astype(np.float32)
        with rasterio.open(gpkg_path, "w", driver="GPKG", width=arr.shape[1], height=arr.shape[0],
                           count=1, dtype="float32", crs="EPSG:3035", transform=tf,
                           nodata=TERRAIN_COARSE_NODATA, RASTER_TABLE=name, RASTER_IDENTIFIER=name,
                           APPEND_SUBDATASET="YES") as dst:
            dst.write(arr, 1)
            dst.set_band_description(1, f"{band} ({units[band]}), 5 m block mean of BEV ALS 1 m")
        written.append(name)
    return written


def acquisition_section(tile_seg_results: list) -> dict | None:
    tiles = []
    for i, tr in enumerate(tile_seg_results):
        v = tr.get("v21") or {}
        a = v.get("acq")
        if not a:
            continue
        tiles.append({"tile": i, **{k: a.get(k) for k in
                                    ("dtm_year", "dsm_year", "ortho_year", "eff_date", "years_gap")}})
    if not tiles:
        return None
    yrs = [t["dtm_year"] for t in tiles if t.get("dtm_year")]
    gaps = [t["years_gap"] for t in tiles if t.get("years_gap")]
    return {"tiles": tiles, "summary": {
        "n_tiles": len(tiles), "n_known": len(yrs),
        "als_year_min": min(yrs) if yrs else None, "als_year_max": max(yrs) if yrs else None,
        "years_gap_median": round(float(np.median(gaps)), 2) if gaps else None,
        "source": "BEV Aktualitaet DGM-ALS flight-block overlay (als_acquisition)"}}
