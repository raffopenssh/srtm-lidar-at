"""v2_source — serve the processor's per-tile raster reads from a v1 *full*
GPKG (our own Zenodo product) instead of BEV / openEO / UMD, **with per-layer
health checks and fallback to the real upstream source**.

Used by ``austria_processor.process_one_kg(..., source_gpkg=...)`` in v2
upgrade mode (re-classifying an already-processed KG with the WILHELM v2
model).  The raster layers of a v2 KG are identical to v1, so the full GPKG
is the cheapest complete input we have: one Zenodo download vs. ~5 GB/h of
BEV range reads + Copernicus credits.

Why health checks: some v1 GPKGs have holes — DTM tiles lost in
``gpkg_full`` when a BEV read failed and no raster sidecar existed
(be7d155), ortho operates that returned black, Copernicus quadrants that
never stitched.  A v2 product built on a holed input would bake the hole
into the new segmentation.  So every window read is scored against the
KG's cadastre union (the area we actually care about) and, if the invalid
fraction inside the union exceeds a per-layer threshold, we fall back to
the original reader for that tile (BEV for LiDAR/ortho, the local+Zenodo
tile cache for Copernicus/Hansen — never openEO, upgrade peers hold no
credential).  What happened is recorded in :data:`HEALTH` and surfaced in
the JSON ``data_quality.v2_source`` block + INFO log lines (``v2src:``).

Install (inside the subprocess, from ``process_one_kg``)::

    import v2_source
    v2_source.install(gpkg_path, union_3035, proc_globals=globals())

It monkeypatches ``raster_io.read_dtm_dsm``, ``ortho_io.read_ortho_for_als``,
``ortho_io.pick_rgbi_year_for_als`` and, in the processor module globals,
``_fetch_copernicus_for_tile`` / ``_get_hansen_cache``.  ``uninstall()``
restores the originals (the subprocess dies after one KG anyway).
"""
from __future__ import annotations

import logging
import math
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("austria_processor.v2src")

NODATA = -9999.0

# Per-layer invalid-fraction thresholds (inside the cadastre union) above
# which we distrust the GPKG window and fall back to the real source.
THRESH = {
    "lidar": 0.005,      # DTM/DSM NaN inside union (BEV is ~100 %% inside AT)
    "lidar_date": 0.02,  # multi-date mosaics have real gaps; be lenient
    "ortho": 0.02,       # pure-black RGB pixels inside union
    "ndvi": 0.05,        # Copernicus NDVI NaN (10 m source, clouds masked)
    "worldcover": 0.02,  # WorldCover code 0
    "sar": 0.05,
    "hansen": 0.50,      # Hansen genuinely sparse at AT borders
}
MIN_UNION_PX = 2000        # below this the tile barely touches the KG — accept anything
EDGE_SLACK_PX = 3          # window may overhang GPKG grid by this much

HEALTH: list[dict] = []    # one entry per (tile, layer) decision
_STATE: dict = {}


# --------------------------------------------------------------------------
# GPKG access
# --------------------------------------------------------------------------

class _Gpkg:
    def __init__(self, path: str | Path):
        import rasterio
        self.path = str(path)
        c = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            self.layers = {n: t for n, t in c.execute(
                "SELECT table_name, data_type FROM gpkg_contents")}
        finally:
            c.close()
        if "DTM" not in self.layers:
            raise ValueError("GPKG has no DTM layer")
        with rasterio.open(f"GPKG:{self.path}:DTM") as ds:
            self.transform = ds.transform
            self.shape = (ds.height, ds.width)
            self.bounds = ds.bounds
            self.crs = ds.crs
        self.res = abs(self.transform.a)
        # v1 full GPKGs were stitched from fractional-origin windows: the
        # pixel DATA are true BEV pixels (rasterio floored the window
        # offset) but the transform claims the fractional AOI origin.
        # Re-anchor to the integer-metre BEV grid so upgrade products get
        # the same apex/tree_id convention as fresh ones (FEEDBACK-5 §1).
        # Residual: tiles inside the mosaic may be misregistered by ≤1 px.
        t = self.transform
        if abs(t.a - 1.0) < 0.01 and (abs(t.c - round(t.c)) > 1e-6 or abs(t.f - round(t.f)) > 1e-6):
            from rasterio import Affine
            self.transform = Affine(1.0, 0.0, math.floor(t.c), 0.0, -1.0, math.ceil(t.f))
            self.grid_reanchored = True
        else:
            self.grid_reanchored = False

    def years(self, prefix: str) -> list[int]:
        rx = re.compile(rf"^{prefix}_(\d{{4}})$")
        return sorted(int(m.group(1)) for n in self.layers for m in [rx.match(n)] if m)

    def window_for_bounds(self, w, s, e, n):
        """Integer pixel window on the GPKG grid covering bounds; returns
        (row0, row1, col0, col1, overhang_px).  Rows/cols are clipped to the
        grid; *overhang_px* is how far the request exceeded it."""
        inv = ~self.transform
        c0f, r0f = inv * (w, n)
        c1f, r1f = inv * (e, s)
        c0, r0 = int(math.floor(c0f)), int(math.floor(r0f))
        c1, r1 = int(math.ceil(c1f)), int(math.ceil(r1f))
        H, W = self.shape
        over = max(0, -r0, -c0, r1 - H, c1 - W)
        return max(0, r0), min(H, r1), max(0, c0), min(W, c1), over

    def read(self, layer: str, r0, r1, c0, c1):
        """Read a window; float layers → float32 with NaN nodata; uint8 PNG
        tables → uint8 (single band unless Ortho/CIR)."""
        if layer not in self.layers:
            return None
        import rasterio
        from rasterio.windows import Window
        win = Window(c0, r0, c1 - c0, r1 - r0)
        with rasterio.open(f"GPKG:{self.path}:{layer}") as ds:
            arr = ds.read(window=win)
            if ds.dtypes[0].startswith("float"):
                arr = arr.astype(np.float32)
                arr[arr <= NODATA + 1] = np.nan
            elif arr.shape[0] == 4 and not layer.startswith(("Ortho", "CIR")):
                arr = arr[:1]
        return arr[0] if arr.shape[0] == 1 else arr

    def window_transform(self, r0, c0):
        from rasterio import Affine
        t = self.transform
        return Affine(t.a, t.b, t.c + c0 * t.a, t.d, t.e, t.f + r0 * t.e)


def _union_mask(shape, transform):
    """Boolean mask of the cadastre union on the window grid (True inside)."""
    u = _STATE.get("union")
    if u is None or u.is_empty:
        return np.ones(shape, bool)
    from rasterio.features import geometry_mask
    try:
        return ~geometry_mask([u], out_shape=shape, transform=transform,
                              invert=False, all_touched=True)
    except Exception:
        return np.ones(shape, bool)


def _record(tile_key, layer, action, frac, note=""):
    ent = {"tile": tile_key, "layer": layer, "action": action,
           "invalid_frac": None if frac is None else round(float(frac), 4)}
    if note:
        ent["note"] = note
    HEALTH.append(ent)
    if action != "gpkg":
        log.info("v2src: %s %s → %s (invalid %.2f%% in union%s)", tile_key, layer,
                 action, 100.0 * (frac or 0.0), f", {note}" if note else "")


def _tile_key(bounds):
    return f"{bounds[0]:.0f},{bounds[1]:.0f}"


def _invalid_frac(invalid: np.ndarray, union: np.ndarray):
    n = int(union.sum())
    if n < MIN_UNION_PX:
        return 0.0, n
    return float((invalid & union).sum()) / n, n


# --------------------------------------------------------------------------
# LiDAR
# --------------------------------------------------------------------------

def _gpkg_read_dtm_dsm(geom_3035, dataset=None, pad=5.0):
    import tile_index as ti
    g: _Gpkg = _STATE["gpkg"]
    orig = _STATE["orig_read_dtm_dsm"]
    dataset = dataset or ti.DEFAULT_DATASET
    is_default = dataset == ti.DEFAULT_DATASET
    year = ti.dataset_to_year(dataset)
    lay_dtm, lay_dsm = ("DTM", "DSM") if is_default else (f"DTM_{year}", f"DSM_{year}")
    kind = "lidar" if is_default else "lidar_date"
    b = geom_3035.bounds
    tk = _tile_key(b)
    def _bev():
        d = orig(geom_3035, dataset, pad)
        if isinstance(d, dict):
            d["source"] = "bev"
        return d
    if lay_dtm not in g.layers or lay_dsm not in g.layers:
        _record(tk, lay_dtm, "fallback", None, "layer missing in GPKG")
        return _bev()
    r0, r1, c0, c1, over = g.window_for_bounds(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
    if over > EDGE_SLACK_PX + int(pad / g.res) or r1 - r0 < 10 or c1 - c0 < 10:
        _record(tk, lay_dtm, "fallback", None, f"window overhangs GPKG grid by {over}px")
        return _bev()
    dtm = g.read(lay_dtm, r0, r1, c0, c1)
    dsm = g.read(lay_dsm, r0, r1, c0, c1)
    tf = g.window_transform(r0, c0)
    shape = dtm.shape
    from rasterio.features import geometry_mask
    gmask = ~geometry_mask([geom_3035], out_shape=shape, transform=tf, invert=False)
    union = _union_mask(shape, tf) & gmask
    invalid = ~(np.isfinite(dtm) & np.isfinite(dsm))
    frac, n_union = _invalid_frac(invalid, union)
    if frac > THRESH[kind]:
        _record(tk, lay_dtm, "fallback", frac, f"{n_union} union px")
        try:
            return _bev()
        except Exception as e:  # noqa: BLE001
            # Real source failed too — for the default date this must
            # propagate (LiDAR is essential; the tile loop aborts/defers).
            # For extra dates, degrade to the GPKG copy.
            if is_default:
                raise
            log.warning("v2src: %s %s BEV fallback failed (%s) — using GPKG copy", tk, lay_dtm, e)
    else:
        _record(tk, lay_dtm, "gpkg", frac)
    valid = gmask & np.isfinite(dtm) & np.isfinite(dsm)
    ndsm = np.where(valid, dsm - dtm, np.nan).astype(np.float32)
    ndsm = np.where((ndsm < 0) & valid, 0.0, ndsm).astype(np.float32)
    return {"dtm": dtm, "dsm": dsm, "ndsm": ndsm, "mask": valid,
            "transform": tf, "crs": g.crs, "shape": shape, "dataset": dataset,
            "source": "gpkg"}


# --------------------------------------------------------------------------
# Ortho
# --------------------------------------------------------------------------

def _pick_ortho_year(g: _Gpkg, r0, r1, c0, c1, union):
    """Newest year whose Ortho has real coverage in the window; prefer
    years with a CIR (real NIR) layer.  Returns (year, has_nir, rgb, nir)."""
    years = g.years("Ortho")
    cands = sorted(years, key=lambda y: (f"CIR_{y}" in g.layers, y), reverse=True)
    for y in cands:
        arr = g.read(f"Ortho_{y}", r0, r1, c0, c1)
        if arr is None:
            continue
        if arr.ndim == 2:
            arr = arr[None]
        rgb = arr[:3]
        black = rgb.astype(np.uint16).sum(0) == 0
        frac, _ = _invalid_frac(black, union)
        if frac > THRESH["ortho"]:
            continue
        nir = None
        if f"CIR_{y}" in g.layers:
            if arr.shape[0] >= 4:
                nir = arr[3]
            else:
                cir = g.read(f"CIR_{y}", r0, r1, c0, c1)
                if cir is not None:
                    nir = cir[0] if cir.ndim == 3 else cir
        return y, nir is not None, rgb, nir, frac
    return None, False, None, None, 1.0


def _gpkg_read_ortho_for_als(als_result, dataset=None, year=None):
    g: _Gpkg = _STATE["gpkg"]
    orig = _STATE["orig_read_ortho"]
    tf = als_result["transform"]
    h, w = als_result["shape"]
    if als_result.get("source") != "gpkg":
        # LiDAR for this tile came from BEV (fallback) — its grid may not
        # align with the GPKG; take ortho from BEV as well.
        _record(_tile_key((tf.c, tf.f - h, 0, 0)), "Ortho", "fallback", None, "lidar came from BEV")
        return orig(als_result) if dataset is None else orig(als_result, dataset, year)
    inv = ~g.transform
    c0, r0 = inv * (tf.c, tf.f)
    r0, c0 = int(round(r0)), int(round(c0))
    r1, c1 = r0 + h, c0 + w
    tk = _tile_key((tf.c, tf.f - h * g.res))
    H, W = g.shape
    if r0 < 0 or c0 < 0 or r1 > H or c1 > W:
        _record(tk, "Ortho", "fallback", None, "window outside GPKG grid")
        return orig(als_result)
    union = _union_mask((h, w), tf) & als_result["mask"]
    y, has_nir, rgb, nir, frac = _pick_ortho_year(g, r0, r1, c0, c1, union)
    if y is None:
        _record(tk, "Ortho", "fallback", frac, "no healthy ortho year in GPKG")
        return orig(als_result)
    _STATE["last_ortho_year"] = y
    _record(tk, f"Ortho_{y}{'+NIR' if has_nir else ''}", "gpkg", frac)
    return np.ascontiguousarray(rgb), (np.ascontiguousarray(nir) if nir is not None else None)


def _gpkg_pick_rgbi_year(als_result):
    y = _STATE.get("last_ortho_year")
    if y is not None:
        return int(y)
    return _STATE["orig_pick_year"](als_result)


# --------------------------------------------------------------------------
# Copernicus (NDVI / WorldCover / SAR) + harmonics
# --------------------------------------------------------------------------

def _bbox_wgs_to_3035(bbox):
    from shapely.geometry import box
    from pyproj import Transformer
    t = Transformer.from_crs(4326, 3035, always_xy=True)
    xs, ys = zip(*[t.transform(x, y) for x, y in (
        (bbox["west"], bbox["south"]), (bbox["east"], bbox["north"]),
        (bbox["west"], bbox["north"]), (bbox["east"], bbox["south"]))])
    return box(min(xs), min(ys), max(xs), max(ys))


def _cache_only_fetch(bbox_dict, obs_year, cop_cache, layers):
    """Fetch selected layers from the local + Zenodo tile cache only (never
    openEO).  Returns partial dict; missing layers are simply absent."""
    out = {}
    try:
        import tile_cache
        prev = getattr(tile_cache, "FORBID_REMOTE", False)
        tile_cache.set_forbid_remote(True)
    except Exception:
        prev = None
    try:
        if "ndvi" in layers:
            try:
                nd = cop_cache.get_ndvi(bbox_dict, year=obs_year)
                if nd and nd.get("ndvi") is not None:
                    out["ndvi"] = nd["ndvi"]; out["transform"] = nd.get("transform"); out["crs"] = nd.get("crs")
            except Exception as e:  # noqa: BLE001
                out.setdefault("_miss", []).append(f"ndvi: {type(e).__name__}")
        if "landcover" in layers:
            try:
                lc = cop_cache.get_landcover(bbox_dict)
                if lc is not None:
                    out["landcover"] = lc
            except Exception as e:  # noqa: BLE001
                out.setdefault("_miss", []).append(f"landcover: {type(e).__name__}")
        if "sar" in layers:
            try:
                sar = cop_cache.get_sar(bbox_dict, year=obs_year)
                if sar:
                    out.update({k: sar[k] for k in ("vv", "vh") if k in sar})
                    if "transform" in sar:
                        out["sar_transform"] = sar["transform"]
                    if "crs" in sar:
                        out["sar_crs"] = sar["crs"]
            except Exception as e:  # noqa: BLE001
                out.setdefault("_miss", []).append(f"sar: {type(e).__name__}")
        if "harmonics" in layers:
            try:
                harm = cop_cache.get_harmonics(bbox_dict, year=obs_year)
                if harm and "h_mean" in harm:
                    out["harmonics"] = harm
            except Exception as e:  # noqa: BLE001
                out.setdefault("_miss", []).append(f"harmonics: {type(e).__name__}")
    finally:
        if prev is not None:
            try:
                import tile_cache
                tile_cache.set_forbid_remote(prev)
            except Exception:
                pass
    return out


def _gpkg_fetch_copernicus(bbox_dict, obs_year, cop_cache, report_fn=None, tile_label=""):
    g: _Gpkg = _STATE["gpkg"]
    geom = _bbox_wgs_to_3035(bbox_dict)
    b = geom.bounds
    tk = _tile_key(b)
    r0, r1, c0, c1, over = g.window_for_bounds(*b)
    if r1 - r0 < 10 or c1 - c0 < 10:
        _record(tk, "Copernicus", "fallback", None, "window outside GPKG grid")
        return _cache_only_fetch(bbox_dict, obs_year, cop_cache, {"ndvi", "landcover", "sar", "harmonics"}) or None
    tf = g.window_transform(r0, c0)
    shape = (r1 - r0, c1 - c0)
    union = _union_mask(shape, tf)
    cop = {"transform": tf, "crs": g.crs, "sar_transform": tf, "sar_crs": g.crs}
    need = set()

    nd = g.read("NDVI", r0, r1, c0, c1)
    if nd is None:
        need.add("ndvi"); _record(tk, "NDVI", "fallback", None, "layer missing")
    else:
        frac, _ = _invalid_frac(~np.isfinite(nd), union)
        if frac > THRESH["ndvi"]:
            need.add("ndvi"); _record(tk, "NDVI", "fallback", frac)
        else:
            cop["ndvi"] = nd; _record(tk, "NDVI", "gpkg", frac)

    wc = g.read("WorldCover", r0, r1, c0, c1)
    if wc is None:
        need.add("landcover"); _record(tk, "WorldCover", "fallback", None, "layer missing")
    else:
        wc = wc.astype(np.uint8)
        frac, _ = _invalid_frac(wc == 0, union)
        if frac > THRESH["worldcover"]:
            need.add("landcover"); _record(tk, "WorldCover", "fallback", frac)
        else:
            cop["landcover"] = {"map": wc, "transform": tf, "crs": g.crs}
            _record(tk, "WorldCover", "gpkg", frac)

    vv = g.read("SAR_VV", r0, r1, c0, c1)
    vh = g.read("SAR_VH", r0, r1, c0, c1)
    if vv is None and vh is None:
        need.add("sar"); _record(tk, "SAR", "fallback", None, "layer missing")
    else:
        inv = np.zeros(shape, bool)
        if vv is not None:
            inv |= ~np.isfinite(vv)
        if vh is not None:
            inv |= ~np.isfinite(vh)
        frac, _ = _invalid_frac(inv, union)
        if frac > THRESH["sar"]:
            need.add("sar"); _record(tk, "SAR", "fallback", frac)
        else:
            if vv is not None:
                cop["vv"] = vv
            if vh is not None:
                cop["vh"] = vh
            _record(tk, "SAR", "gpkg", frac)

    # Harmonics never live in the GPKG — cache-only attempt (v2 model does
    # not use harm_*, this only feeds the JSON ndvi_harmonics summary).
    need.add("harmonics")

    if need and cop_cache is not None:
        fb = _cache_only_fetch(bbox_dict, obs_year, cop_cache, need)
        miss = fb.pop("_miss", [])
        for k in ("ndvi", "landcover", "vv", "vh", "harmonics"):
            if k in fb and k not in cop:
                cop[k] = fb[k]
        # Fallback layers come on their own (10 m WGS84) grids.
        if "ndvi" in fb and fb.get("transform") is not None:
            cop["ndvi_transform"] = fb["transform"]; cop["ndvi_crs"] = fb.get("crs")
        if ("vv" in fb or "vh" in fb) and fb.get("sar_transform") is not None:
            cop["sar_transform"] = fb["sar_transform"]; cop["sar_crs"] = fb.get("sar_crs")
        if miss:
            _record(tk, "Copernicus", "cache_miss", None, "; ".join(miss)[:200])
        # object_segmentation resamples ``ndvi`` with cop["transform"] — when
        # NDVI came from the cache but SAR/WC from the GPKG we must hand it
        # the NDVI grid, and re-key the GPKG-grid layers with explicit
        # transforms (landcover already carries its own).
        if "ndvi_transform" in cop:
            cop["transform"] = cop.pop("ndvi_transform"); cop["crs"] = cop.pop("ndvi_crs", g.crs)
    if not any(k in cop for k in ("ndvi", "landcover", "vv", "vh")):
        return None
    return cop


# --------------------------------------------------------------------------
# Hansen
# --------------------------------------------------------------------------

class _HansenFromGpkg:
    def __init__(self, real_cache):
        self._real = real_cache

    def __getattr__(self, name):
        return getattr(self._real, name)

    def get_forest_prior(self, bbox_wgs, target_transform, target_shape):
        g: _Gpkg = _STATE["gpkg"]
        tk = _tile_key((target_transform.c, target_transform.f))
        inv = ~g.transform
        c0, r0 = inv * (target_transform.c, target_transform.f)
        r0, c0 = int(round(r0)), int(round(c0))
        h, w = target_shape
        r1, c1 = r0 + h, c0 + w
        H, W = g.shape
        aligned = (abs(target_transform.a - g.transform.a) < 1e-6
                   and 0 <= r0 and 0 <= c0 and r1 <= H and c1 <= W)
        if aligned and "Hansen_treecover" in g.layers:
            tc = g.read("Hansen_treecover", r0, r1, c0, c1)
            ly = g.read("Hansen_lossyear", r0, r1, c0, c1)
            if tc is not None:
                tc = tc.astype(np.uint8)
                ly = ly.astype(np.uint8) if ly is not None else np.zeros_like(tc)
                union = _union_mask((h, w), target_transform)
                # Hansen nodata is written as 0 treecover; only distrust the
                # window when *everything* inside the union is 0 (a dropped
                # tile) — 0 %% treecover2000 is a legitimate value.
                frac, n = _invalid_frac(tc == 0, union)
                if n < MIN_UNION_PX or frac < 0.999:
                    was = tc >= 25
                    _record(tk, "Hansen", "gpkg", frac)
                    return {"loss_year": ly, "current_forest": was & ~(ly > 0),
                            "treecover2000": tc, "gain": np.zeros_like(was)}
                _record(tk, "Hansen", "fallback", frac, "all-zero window")
        else:
            _record(tk, "Hansen", "fallback", None,
                    "layer missing" if "Hansen_treecover" not in g.layers else "grid misaligned")
        try:
            return self._real.get_forest_prior(bbox_wgs, target_transform, target_shape)
        except Exception as e:  # noqa: BLE001
            log.info("v2src: %s Hansen fallback unavailable (%s)", tk, e)
            return None


# --------------------------------------------------------------------------
# install / summary
# --------------------------------------------------------------------------

def install(gpkg_path: str | Path, union_3035=None, proc_globals: dict | None = None) -> dict:
    """Activate the GPKG source.  Returns the GPKG inventory summary."""
    import raster_io
    import ortho_io
    g = _Gpkg(gpkg_path)
    HEALTH.clear()
    _STATE.clear()
    _STATE["gpkg"] = g
    _STATE["union"] = union_3035
    _STATE["t0"] = time.time()
    _STATE["orig_read_dtm_dsm"] = raster_io.read_dtm_dsm
    _STATE["orig_read_ortho"] = ortho_io.read_ortho_for_als
    _STATE["orig_pick_year"] = ortho_io.pick_rgbi_year_for_als
    raster_io.read_dtm_dsm = _gpkg_read_dtm_dsm
    ortho_io.read_ortho_for_als = _gpkg_read_ortho_for_als
    ortho_io.pick_rgbi_year_for_als = _gpkg_pick_rgbi_year
    if proc_globals is not None:
        _STATE["proc_globals"] = proc_globals
        _STATE["orig_fetch_cop"] = proc_globals.get("_fetch_copernicus_for_tile")
        _STATE["orig_get_hansen"] = proc_globals.get("_get_hansen_cache")
        proc_globals["_fetch_copernicus_for_tile"] = _gpkg_fetch_copernicus
        _orig_gh = _STATE["orig_get_hansen"]

        def _get_hansen_cache_shim():
            real = _orig_gh() if _orig_gh else None
            return _HansenFromGpkg(real)
        proc_globals["_get_hansen_cache"] = _get_hansen_cache_shim
    inv = {"path": g.path, "layers": sorted(g.layers), "shape": list(g.shape),
           "ortho_years": g.years("Ortho"), "cir_years": g.years("CIR"),
           "dtm_years": g.years("DTM"), "size_bytes": Path(g.path).stat().st_size}
    log.info("v2src: installed GPKG source %s (%d layers, %dx%d px, ortho %s, CIR %s)",
             Path(g.path).name, len(g.layers), g.shape[1], g.shape[0],
             inv["ortho_years"], inv["cir_years"])
    return inv


def uninstall() -> None:
    if not _STATE:
        return
    try:
        import raster_io
        import ortho_io
        raster_io.read_dtm_dsm = _STATE["orig_read_dtm_dsm"]
        ortho_io.read_ortho_for_als = _STATE["orig_read_ortho"]
        ortho_io.pick_rgbi_year_for_als = _STATE["orig_pick_year"]
        pg = _STATE.get("proc_globals")
        if pg is not None:
            if _STATE.get("orig_fetch_cop") is not None:
                pg["_fetch_copernicus_for_tile"] = _STATE["orig_fetch_cop"]
            if _STATE.get("orig_get_hansen") is not None:
                pg["_get_hansen_cache"] = _STATE["orig_get_hansen"]
    finally:
        _STATE.clear()


def summary() -> dict:
    """Compact per-layer tally for the JSON ``data_quality.v2_source`` block."""
    by_layer: dict[str, dict] = {}
    for h in HEALTH:
        lay = re.sub(r"_\d{4}(\+NIR)?$", "", h["layer"])
        d = by_layer.setdefault(lay, {"gpkg": 0, "fallback": 0, "cache_miss": 0, "max_invalid_frac": 0.0})
        d[h["action"]] = d.get(h["action"], 0) + 1
        if h.get("invalid_frac") is not None:
            d["max_invalid_frac"] = max(d["max_invalid_frac"], h["invalid_frac"])
    n_fb = sum(1 for h in HEALTH if h["action"] == "fallback")
    return {"source": "gpkg_v1", "decisions": len(HEALTH), "fallbacks": n_fb,
            "cache_misses": sum(1 for h in HEALTH if h["action"] == "cache_miss"),
            "by_layer": by_layer,
            "fallback_tiles": sorted({h["tile"] for h in HEALTH if h["action"] == "fallback"})[:50]}
