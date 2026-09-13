"""Per-segment feature extraction for segv2 — vectorised, masked to the KG.

Inputs come from a Zenodo *full* GPKG (`gpkg_raster.FullGpkg`) plus the
Zenodo tile cache for NDVI harmonics + Hansen gain (both in the shared cache
deposit, so cache-only, no upstream traffic).

Produces, per tile window:
  * ``labels``      int32 segment raster (v1 segmentation, so the v1 model can
                    be scored on identical objects), or the v2 segmentation
                    when ``osm_edges`` is given.
  * ``feats``       DataFrame with one row per segment: every v1
                    ``learned_classifier.FEATURE_KEYS`` column (so the deployed
                    RF is a fair baseline) + v2 extras (`V2_EXTRA_KEYS`).

Speed: all per-segment statistics are computed with ``np.bincount`` /
lexsort-based grouped percentiles — O(pixels + segments), not
O(segments × pixels) like v1's ``labels == reg.label`` loop.  Only shape
metrics go through ``regionprops`` (slice-bounded).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import object_segmentation as oseg  # noqa: E402
from learned_classifier import FEATURE_KEYS  # noqa: E402
from terrain_analysis import compute_tri, compute_tpi, compute_curvature  # noqa: E402

log = logging.getLogger("segv2.features")

V2_EXTRA_KEYS = [
    # distances (m) to OSM / cadastre context — labels use *geometry* not distance,
    # so no circularity
    "dist_road", "dist_path", "dist_rail", "dist_water", "dist_building", "dist_parcel_edge",
    # neighbourhood (adjacent segments, area weighted)
    "nb_h_mean", "nb_ndvi_mean", "nb_h_diff", "nb_ndvi_diff", "nb_n",
    # height distribution shape
    "h_p50", "h_iqr", "h_cv", "ndsm_frac_gt2", "ndsm_frac_gt5", "ndsm_frac_lt03",
    # spectral extras
    "ndvi_p10", "ndvi_p90", "brightness_std", "nir_std", "ndwi_mean", "savi_mean",
    # terrain extras
    "tri_mean", "tpi_mean", "curvature_mean", "elevation_mean", "aspect_sin", "aspect_cos",
    "dtm_range",
    # acquisition (real flight dates via segv2.acquisition, NOT mosaic labels)
    "dsm_year", "dtm_year", "ortho_year", "ortho_lidar_gap", "years_span",
    "h_change_per_year", "dtm_change_per_year",
    "dsm_age", "dtm_dsm_split", "nir_year", "nir_lidar_gap",
    # multi-year BEV ortho spectral (real NIR years only)
    "ndvi_y_first", "ndvi_y_last", "ndvi_trend_per_year", "ndvi_tstd", "ndvi_years_span",
    "ndvi_ndsm_coherence",
    # shape extras
    "bbox_fill", "width_est", "length_est",
    # worldcover extras
    "esa_shrub_frac", "esa_bare_frac", "esa_snow_frac", "esa_wetland_frac",
    # hansen extra
    "hansen_lossyear_max",
]
ALL_KEYS = FEATURE_KEYS + V2_EXTRA_KEYS

_UF = ndimage.uniform_filter


# ---------------------------------------------------------------------------
# grouped statistics
# ---------------------------------------------------------------------------
class Grouped:
    """Grouped stats over a label raster (0 = ignore)."""

    def __init__(self, labels: np.ndarray, valid: np.ndarray):
        sel = valid & (labels > 0)
        self.idx = np.flatnonzero(sel)
        self.lab = labels.ravel()[self.idx]
        self.ids, self.inv = np.unique(self.lab, return_inverse=True)
        self.n = len(self.ids)
        self.count = np.bincount(self.inv, minlength=self.n).astype(np.float64)
        self._order = None

    def _vals(self, arr):
        return arr.ravel()[self.idx].astype(np.float64)

    def mean(self, arr, fill=0.0):
        v = self._vals(arr)
        ok = np.isfinite(v)
        s = np.bincount(self.inv[ok], v[ok], self.n)
        c = np.bincount(self.inv[ok], minlength=self.n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = s / c
        out[c == 0] = fill
        return out

    def sum(self, arr):
        v = np.nan_to_num(self._vals(arr))
        return np.bincount(self.inv, v, self.n)

    def std(self, arr):
        v = self._vals(arr)
        ok = np.isfinite(v)
        c = np.bincount(self.inv[ok], minlength=self.n)
        s = np.bincount(self.inv[ok], v[ok], self.n)
        s2 = np.bincount(self.inv[ok], v[ok] ** 2, self.n)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = s2 / c - (s / c) ** 2
        out = np.sqrt(np.clip(var, 0, None))
        out[c == 0] = 0
        return out

    def max(self, arr, fill=0.0):
        v = np.nan_to_num(self._vals(arr), nan=-np.inf)
        out = np.full(self.n, -np.inf)
        np.maximum.at(out, self.inv, v)
        out[~np.isfinite(out)] = fill
        return out

    def min(self, arr, fill=0.0):
        v = np.nan_to_num(self._vals(arr), nan=np.inf)
        out = np.full(self.n, np.inf)
        np.minimum.at(out, self.inv, v)
        out[~np.isfinite(out)] = fill
        return out

    def frac(self, boolarr):
        v = self._vals(boolarr.astype(np.float32))
        return np.bincount(self.inv, v, self.n) / self.count

    def percentiles(self, arr, qs):
        """dict q -> array; NaNs excluded (nearest-rank)."""
        v = self._vals(arr)
        ok = np.isfinite(v)
        inv = self.inv[ok]
        v = v[ok]
        order = np.lexsort((v, inv))
        inv_s, v_s = inv[order], v[order]
        c = np.bincount(inv_s, minlength=self.n)
        start = np.concatenate([[0], np.cumsum(c)[:-1]])
        out = {}
        for q in qs:
            pos = start + np.clip(np.round((c - 1) * q / 100.0), 0, None).astype(np.int64)
            pos = np.minimum(pos, max(len(v_s) - 1, 0))
            res = v_s[pos] if len(v_s) else np.zeros(self.n)
            res = np.where(c > 0, res, 0.0)
            out[q] = res
        return out

    def mode_frac(self, arr, minlength=256):
        """(mode value, mode fraction, labelled fraction) of a uint8 raster,
        ignoring value 0."""
        v = self._vals(arr).astype(np.int64)
        ok = v > 0
        key = self.inv[ok] * minlength + v[ok]
        cnt = np.bincount(key, minlength=self.n * minlength).reshape(self.n, minlength)
        mode = cnt.argmax(axis=1)
        mode_n = cnt.max(axis=1)
        lab_n = cnt.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            purity = np.where(lab_n > 0, mode_n / lab_n, 0.0)
        return mode, purity, lab_n / self.count


# ---------------------------------------------------------------------------
# pixel layers
# ---------------------------------------------------------------------------
def _local_std(a, size=3):
    a = np.nan_to_num(a, nan=0.0).astype(np.float64)
    m = _UF(a, size)
    m2 = _UF(a * a, size)
    return np.sqrt(np.clip(m2 - m * m, 0, None)).astype(np.float32)


def pixel_layers(g, window, *, ortho_year=None, cop_cache=None, hansen_cache=None,
                 obs_year=2024) -> dict | None:
    """All 1 m layers for a window. Returns None if DTM missing."""
    dtm = g.read("DTM", window)
    dsm = g.read("DSM", window)
    if dtm is None or dsm is None:
        return None
    mask = np.isfinite(dtm) & np.isfinite(dsm)
    if mask.sum() < 100:
        return None
    L = {"dtm": dtm, "dsm": dsm, "mask": mask}
    # DSM-DTM everywhere both are valid (the GPKG nDSM layer has ~7 % NaN where
    # the writer dropped negative/edge pixels; a NaN hole would bias frac_* stats)
    L["ndsm"] = np.where(mask, np.clip(dsm - dtm, 0, None), np.nan).astype(np.float32)
    L["transform"] = g.window_transform(window)
    dtm0 = np.where(mask, dtm, np.nanmean(dtm[mask]))
    L["slope"] = oseg._slope(dtm0)
    L["aspect"] = oseg._aspect(dtm0)
    L["tri"] = compute_tri(dtm0)
    L["tpi"] = compute_tpi(dtm0, radius=5)
    with np.errstate(invalid="ignore", divide="ignore"):
        # curvature is undefined (0/0) on perfectly flat pixels — NaN is correct
        L["curv"] = compute_curvature(dtm0)["profile_curvature"]
    L["dsm_rough"] = _local_std(dsm, 3)
    L["dtm_rough"] = _local_std(dtm, 3)
    d0 = np.nan_to_num(dsm)
    L["dsm_edge"] = np.hypot(ndimage.sobel(d0, 1), ndimage.sobel(d0, 0)).astype(np.float32)

    # --- ortho (all years; NIR only from years that carry a CIR layer) ---
    stack = g.read_ortho_stack(window)
    L["ortho_stack_years"] = sorted(stack)
    nir_years = [y for y in sorted(stack) if stack[y]["nir"] is not None]
    L["nir_years"] = nir_years
    # RGB from the newest year; NIR from the newest *real* NIR year.  Where the
    # two differ (2024 RGB-only + 2023 RGBI) NDVI is computed from the NIR year's
    # own red band so it is a true single-date index.
    rgb_year = L["ortho_stack_years"][-1] if stack else None
    nir_year = nir_years[-1] if nir_years else None
    L["ortho_year"] = rgb_year
    L["nir_year"] = nir_year
    spectral = None
    if rgb_year is not None:
        rgb = stack[rgb_year]["rgb"]
        r, gg, b = (rgb[i].astype(np.float32) for i in range(3))
        s = r + gg + b
        spectral = {"red": r, "green": gg, "blue": b, "brightness": s / 3.0}
        with np.errstate(divide="ignore", invalid="ignore"):
            spectral["green_ratio"] = np.where(s > 0, gg / s, np.nan).astype(np.float32)
            spectral["rg_index"] = np.where(r + gg > 0, (r - gg) / (r + gg), np.nan).astype(np.float32)
        if nir_year is not None:
            n = stack[nir_year]["nir"].astype(np.float32)
            rn = stack[nir_year]["rgb"][0].astype(np.float32)
            gn = stack[nir_year]["rgb"][1].astype(np.float32)
            spectral["nir"] = n
            with np.errstate(divide="ignore", invalid="ignore"):
                spectral["ndvi"] = np.where(n + rn > 0, (n - rn) / (n + rn), np.nan).astype(np.float32)
                spectral["ndwi"] = np.where(n + gn > 0, (gn - n) / (gn + n), np.nan).astype(np.float32)
                spectral["savi"] = (1.5 * (n - rn) / (n + rn + 0.5 * 255)).astype(np.float32)
            black_n = (rn + gn + stack[nir_year]["rgb"][2]) == 0
            for k in ("nir", "ndvi", "ndwi", "savi"):
                spectral[k] = np.where(black_n, np.nan, spectral[k]).astype(np.float32)
        # no-data pixels in ortho are pure black
        black = s == 0
        for k in ("red", "green", "blue", "brightness", "green_ratio", "rg_index"):
            spectral[k] = np.where(black, np.nan, spectral[k]).astype(np.float32)
    L["spectral"] = spectral
    # per-year NDVI from every real-NIR year (temporal vegetation signal)
    L["ndvi_years"] = {}
    for y in nir_years:
        rr = stack[y]["rgb"][0].astype(np.float32); nn = stack[y]["nir"].astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(nn + rr > 0, (nn - rr) / (nn + rr), np.nan).astype(np.float32)
        L["ndvi_years"][y] = np.where(stack[y]["rgb"].sum(0) == 0, np.nan, v).astype(np.float32)
    # real flight year of each ortho slot (operate id), for gap features
    L["ortho_flight_years"] = {}
    try:
        import acquisition as _acq
        wb = g.window_bounds(window)
        for y in L["ortho_stack_years"]:
            L["ortho_flight_years"][y] = _acq.ortho_flight_year(wb, y)
    except Exception as e:  # noqa: BLE001
        log.debug("ortho flight year lookup failed: %s", e)
    del stack

    # --- copernicus (from GPKG) ---
    L["cop_ndvi"] = g.read("NDVI", window)
    wc = g.read("WorldCover", window)
    L["worldcover"] = wc.astype(np.uint8) if wc is not None else None
    L["sar_vv"] = g.read("SAR_VV", window)
    L["sar_vh"] = g.read("SAR_VH", window)
    bev_ndvi = spectral.get("ndvi") if spectral else None
    L["fused_ndvi"] = oseg._fuse_ndvi(bev_ndvi, L["cop_ndvi"], mask)

    # --- multi-date ---
    dtm_years = g.years("DTM")
    L["dtm_dates"] = {y: g.read(f"DTM_{y}", window) for y in dtm_years}
    L["dsm_dates"] = {y: g.read(f"DSM_{y}", window) for y in dtm_years}
    L["dtm_dates"] = {y: a for y, a in L["dtm_dates"].items() if a is not None}
    L["dsm_dates"] = {y: a for y, a in L["dsm_dates"].items() if a is not None}
    # the default mosaic (DTM/DSM) is the newest (obs_year); add it as a date too
    L["dtm_dates"].setdefault(obs_year, dtm)
    L["dsm_dates"].setdefault(obs_year, dsm)
    # real per-pixel flight years for every mosaic we have (segv2.acquisition)
    L["als_years"] = {}
    try:
        import acquisition as _acq
        for my in sorted(set(L["dtm_dates"]) | set(L["dsm_dates"])):
            L["als_years"][my] = _acq.als_year_rasters(L["transform"], dtm.shape, my)
    except Exception as e:  # noqa: BLE001
        log.warning("ALS flight-year lookup failed: %s", e)

    # --- hansen (GPKG) ---
    tc = g.read("Hansen_treecover", window)
    ly = g.read("Hansen_lossyear", window)
    L["hansen"] = None
    if tc is not None:
        tc = tc.astype(np.uint8)
        ly = ly.astype(np.uint8) if ly is not None else np.zeros_like(tc)
        was = tc >= 25
        L["hansen"] = {"treecover2000": tc, "loss_year": ly, "gain": np.zeros_like(was),
                       "current_forest": was & ~(ly > 0)}

    # --- harmonics (Zenodo tile cache, cache-only) ---
    L["harmonics"] = None
    if cop_cache is not None:
        try:
            import tile_cache
            from pyproj import Transformer
            from rasterio.warp import reproject, Resampling
            from rasterio.crs import CRS
            tile_cache.set_forbid_remote(True)
            b = g.window_bounds(window)
            t = Transformer.from_crs(3035, 4326, always_xy=True)
            xs, ys = zip(*[t.transform(x, y) for x, y in ((b[0], b[1]), (b[2], b[3]), (b[0], b[3]), (b[2], b[1]))])
            bbox = {"west": min(xs), "south": min(ys), "east": max(xs), "north": max(ys)}
            harm = cop_cache.get_harmonics(bbox, year=obs_year)
            if harm and "h_mean" in harm:
                out = {}
                for k in ("h_mean", "h_amplitude", "h_phase", "h_rmse"):
                    if k not in harm:
                        continue
                    dst = np.full(dtm.shape, np.nan, np.float32)
                    reproject(np.asarray(harm[k], np.float32), dst,
                              src_transform=harm["transform"], src_crs=harm.get("crs") or CRS.from_epsg(4326),
                              dst_transform=L["transform"], dst_crs=CRS.from_epsg(3035),
                              resampling=Resampling.bilinear)
                    out[k] = dst
                L["harmonics"] = out
        except Exception as e:  # noqa: BLE001
            log.debug("harmonics unavailable: %s", e)
    return L


# ---------------------------------------------------------------------------
# segmentation
# ---------------------------------------------------------------------------
def segment(L: dict, kg_mask: np.ndarray, *, osm_edges: np.ndarray | None = None,
            felz_scale=150.0, rag_threshold=0.12, min_size=30) -> np.ndarray:
    """v1 segmentation (identical params) restricted to the KG mask. If
    ``osm_edges`` (bool raster of OSM road/water/rail centrelines) is given,
    they are injected as hard boundaries into the fused gradient (v2)."""
    mask = L["mask"] & kg_mask
    grad = oseg.compute_fused_gradient(L["dtm"], L["dsm"], L["ndsm"], mask, spectral=L["spectral"])
    if osm_edges is not None:
        grad = np.where(osm_edges & mask, 1.0, grad).astype(np.float32)
    return oseg.segment_landscape(grad, L["ndsm"], mask, felz_scale=felz_scale,
                                  felz_min_size=min_size, rag_threshold=rag_threshold)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def _adjacency(labels: np.ndarray):
    """Pairs of adjacent labels (4-connectivity), as (a, b) int arrays."""
    a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    sel = (a != b) & (a > 0) & (b > 0)
    a, b = a[sel], b[sel]
    pairs = np.unique(np.stack([np.concatenate([a, b]), np.concatenate([b, a])], 1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def _edt(mask_true: np.ndarray, cap=500.0) -> np.ndarray:
    if mask_true is None or not mask_true.any():
        return np.full(mask_true.shape if mask_true is not None else (1, 1), cap, np.float32)
    return np.minimum(ndimage.distance_transform_edt(~mask_true), cap).astype(np.float32)


def extract(labels: np.ndarray, L: dict, *, context: dict | None = None,
            obs_year: int = 2024, texture: bool = True) -> pd.DataFrame:
    """Feature table for every segment in ``labels``.

    ``context`` (optional): dict of bool rasters {'road','path','rail','water',
    'building','parcel_edge'} for distance features.
    """
    mask = L["mask"]
    G = Grouped(labels, mask)
    if G.n == 0:
        return pd.DataFrame(columns=["label"] + ALL_KEYS)
    F: dict[str, np.ndarray] = {"label": G.ids, "area": G.count}
    ndsm, dtm, dsm = L["ndsm"], L["dtm"], L["dsm"]
    sp = L["spectral"] or {}

    # height
    P = G.percentiles(ndsm, (10, 25, 50, 75, 90))
    F["h_mean"] = G.mean(ndsm); F["h_max"] = G.max(ndsm); F["h_std"] = G.std(ndsm)
    F["h_p90"] = P[90]; F["h_p10"] = P[10]; F["h_p50"] = P[50]; F["h_iqr"] = P[75] - P[25]
    F["h_cv"] = F["h_std"] / np.maximum(F["h_mean"], 0.05)
    F["ndsm_frac_gt2"] = G.frac(ndsm > 2); F["ndsm_frac_gt5"] = G.frac(ndsm > 5)
    F["ndsm_frac_lt03"] = G.frac(ndsm < 0.3)
    # terrain
    F["slope_mean"] = G.mean(L["slope"]); F["slope_std"] = G.std(L["slope"]); F["slope_max"] = G.max(L["slope"])
    F["dsm_roughness"] = G.mean(L["dsm_rough"]); F["dtm_roughness"] = G.mean(L["dtm_rough"])
    F["dsm_edge_strength"] = G.mean(L["dsm_edge"])
    F["tri_mean"] = G.mean(L["tri"]); F["tpi_mean"] = G.mean(L["tpi"]); F["curvature_mean"] = G.mean(L["curv"])
    F["elevation_mean"] = G.mean(dtm); F["dtm_range"] = G.max(dtm) - G.min(dtm)
    asp = np.radians(np.where(L["aspect"] >= 0, L["aspect"], np.nan))
    F["aspect_sin"] = G.mean(np.sin(asp)); F["aspect_cos"] = G.mean(np.cos(asp))

    # shape (regionprops is slice-bounded → fine)
    props = {p.label: p for p in measure.regionprops(labels)}
    per = np.zeros(G.n); elong = np.zeros(G.n); solid = np.zeros(G.n); ext = np.zeros(G.n)
    bfill = np.zeros(G.n); wid = np.zeros(G.n); ln = np.zeros(G.n)
    for i, lid in enumerate(G.ids):
        p = props.get(int(lid))
        if p is None:
            continue
        per[i] = p.perimeter
        minor = max(p.axis_minor_length, 1.0)
        elong[i] = p.axis_major_length / minor
        solid[i] = p.solidity; ext[i] = p.extent
        bfill[i] = p.extent
        wid[i] = p.axis_minor_length; ln[i] = p.axis_major_length
    F["perimeter"] = per; F["elongation"] = elong; F["solidity"] = solid; F["extent"] = ext
    F["compactness"] = 4 * np.pi * F["area"] / (per ** 2 + 1e-6)
    F["bbox_fill"] = bfill; F["width_est"] = wid; F["length_est"] = ln

    # spectral
    def _m(k): return G.mean(sp[k]) if k in sp else np.zeros(G.n)
    F["ndvi_mean"] = _m("ndvi"); F["ndvi_std"] = G.std(sp["ndvi"]) if "ndvi" in sp else np.zeros(G.n)
    F["ndvi_max"] = G.max(sp["ndvi"]) if "ndvi" in sp else np.zeros(G.n)
    if "ndvi" in sp:
        Pn = G.percentiles(sp["ndvi"], (10, 90)); F["ndvi_p10"] = Pn[10]; F["ndvi_p90"] = Pn[90]
    else:
        F["ndvi_p10"] = F["ndvi_p90"] = np.zeros(G.n)
    F["brightness_mean"] = _m("brightness"); F["brightness_std"] = G.std(sp["brightness"]) if "brightness" in sp else np.zeros(G.n)
    F["nir_mean"] = _m("nir"); F["nir_std"] = G.std(sp["nir"]) if "nir" in sp else np.zeros(G.n)
    F["red_mean"] = _m("red"); F["green_mean"] = _m("green"); F["blue_mean"] = _m("blue")
    F["green_ratio"] = _m("green_ratio"); F["rg_index"] = _m("rg_index")
    F["nir_brightness_ratio"] = np.where(F["brightness_mean"] > 0, F["nir_mean"] / np.maximum(F["brightness_mean"], 1), 0)
    F["nir_red_ratio"] = np.where(F["nir_mean"] > 0, F["nir_mean"] / np.maximum(F["red_mean"], 1), 0)
    F["ndwi_mean"] = _m("ndwi"); F["savi_mean"] = _m("savi")
    F["cop_ndvi_mean"] = G.mean(L["cop_ndvi"]) if L["cop_ndvi"] is not None else np.zeros(G.n)
    if L["fused_ndvi"] is not None:
        F["fused_ndvi_mean"] = G.mean(L["fused_ndvi"]); F["fused_ndvi_std"] = G.std(L["fused_ndvi"])
    else:
        F["fused_ndvi_mean"] = F["fused_ndvi_std"] = np.zeros(G.n)
    # SAR
    F["sar_vv"] = G.mean(L["sar_vv"]) if L["sar_vv"] is not None else np.zeros(G.n)
    F["sar_vh"] = G.mean(L["sar_vh"]) if L["sar_vh"] is not None else np.zeros(G.n)
    F["sar_ratio"] = np.where(F["sar_vv"] > 0, F["sar_vv"] / np.maximum(F["sar_vh"], 1e-6), 0)
    # harmonics
    H = L["harmonics"] or {}
    for k, hk in (("harm_mean", "h_mean"), ("harm_amplitude", "h_amplitude"),
                  ("harm_phase", "h_phase"), ("harm_rmse", "h_rmse")):
        F[k] = G.mean(H[hk]) if hk in H else np.zeros(G.n)
    # worldcover
    wc = L["worldcover"]
    if wc is not None:
        mode, _, _ = G.mode_frac(wc, 256)
        F["esa_dominant_lc"] = np.where((mode == 0) | (mode == 255), np.nan, mode)  # nodata → missing
        for k, code in (("esa_built_frac", 50), ("esa_tree_frac", 10), ("esa_crop_frac", 40),
                        ("esa_grass_frac", 30), ("esa_water_frac", 80), ("esa_shrub_frac", 20),
                        ("esa_bare_frac", 60), ("esa_snow_frac", 70), ("esa_wetland_frac", 90)):
            F[k] = G.frac(wc == code)
    else:
        for k in ("esa_dominant_lc", "esa_built_frac", "esa_tree_frac", "esa_crop_frac", "esa_grass_frac",
                  "esa_water_frac", "esa_shrub_frac", "esa_bare_frac", "esa_snow_frac", "esa_wetland_frac"):
            F[k] = np.zeros(G.n)

    # temporal — normalised by REAL flight dates (segv2.acquisition), not mosaic labels
    years = sorted(set(L["dtm_dates"]) & set(L["dsm_dates"]))
    als = L.get("als_years") or {}
    def _yr(my, key):
        r = als.get(my)
        if r is None:
            return np.full(G.n, np.nan)
        a = r[key].astype(np.float32); a[a <= 0] = np.nan
        return G.mean(a, fill=np.nan)
    newest = years[-1] if years else obs_year
    F["dsm_year"] = _yr(newest, "dsm_year"); F["dtm_year"] = _yr(newest, "dtm_year")
    F["dsm_age"] = float(obs_year) - F["dsm_year"]
    F["dtm_dsm_split"] = F["dsm_year"] - F["dtm_year"]
    if len(years) >= 2:
        y0, y1 = years[0], years[-1]
        # true span between the flights behind the oldest and newest mosaic, per segment
        span = _yr(y1, "dsm_year") - _yr(y0, "dsm_year")
        F["years_span"] = span
        stack = np.stack([np.clip(L["dsm_dates"][y] - L["dtm_dates"][y], 0, None) for y in years])
        with np.errstate(all="ignore"):
            tstd = np.nanstd(stack, 0)
            hch = stack[-1] - stack[0]
        dch = L["dtm_dates"][y1] - L["dtm_dates"][y0]
        F["temporal_h_std"] = G.mean(tstd); F["h_change"] = G.mean(hch)
        F["dtm_change"] = G.mean(dch); F["dtm_change_abs"] = G.mean(np.abs(dch))
        F["volume_change_m3"] = G.sum(dch); F["volume_change_abs_m3"] = G.sum(np.abs(dch))
        F["dtm_change_max"] = G.max(np.abs(dch)); F["dtm_change_frac_03m"] = G.frac(np.nan_to_num(dch) > 0.3)
        # same flight in both mosaics → "change" is resampling noise → missing, not 0
        same = ~(span >= 1)
        for k in ("temporal_h_std", "h_change", "dtm_change", "dtm_change_abs", "volume_change_m3",
                  "volume_change_abs_m3", "dtm_change_max", "dtm_change_frac_03m"):
            F[k] = np.where(same, np.nan, F[k])
        with np.errstate(invalid="ignore", divide="ignore"):
            F["h_change_per_year"] = F["h_change"] / span; F["dtm_change_per_year"] = F["dtm_change"] / span
    else:
        F["years_span"] = np.full(G.n, np.nan)
        for k in ("temporal_h_std", "h_change", "dtm_change", "dtm_change_abs", "volume_change_m3",
                  "volume_change_abs_m3", "dtm_change_max", "dtm_change_frac_03m", "h_change_per_year",
                  "dtm_change_per_year"):
            F[k] = np.full(G.n, np.nan)
    F["stability"] = 1.0 / (1.0 + np.nan_to_num(F["temporal_h_std"]) + np.nan_to_num(F["dtm_change_abs"]))
    ofy = L.get("ortho_flight_years") or {}
    oy = L.get("ortho_year"); ny = L.get("nir_year")
    F["ortho_year"] = np.full(G.n, float(ofy.get(oy) or oy or np.nan))
    F["nir_year"] = np.full(G.n, float(ofy.get(ny) or ny or np.nan))
    F["ortho_lidar_gap"] = F["ortho_year"] - F["dsm_year"]
    F["nir_lidar_gap"] = F["nir_year"] - F["dsm_year"]

    # multi-year BEV NDVI (only real-NIR years)
    ny_all = sorted((L.get("ndvi_years") or {}))
    if ny_all:
        v0 = G.mean(L["ndvi_years"][ny_all[0]], fill=np.nan); v1 = G.mean(L["ndvi_years"][ny_all[-1]], fill=np.nan)
        F["ndvi_y_first"] = v0; F["ndvi_y_last"] = v1
        fy = [ofy.get(y) or y for y in ny_all]
        sp_y = float(fy[-1] - fy[0])
        F["ndvi_years_span"] = np.full(G.n, sp_y)
        F["ndvi_trend_per_year"] = (v1 - v0) / sp_y if sp_y >= 1 else np.full(G.n, np.nan)
        if len(ny_all) >= 2:
            st = np.stack([L["ndvi_years"][y] for y in ny_all])
            with np.errstate(all="ignore"):
                F["ndvi_tstd"] = G.mean(np.nanstd(st, 0), fill=np.nan)
        else:
            F["ndvi_tstd"] = np.full(G.n, np.nan)
    else:
        for k in ("ndvi_y_first", "ndvi_y_last", "ndvi_years_span", "ndvi_trend_per_year", "ndvi_tstd"):
            F[k] = np.full(G.n, np.nan)
    # tall & green vs tall & not green (tree vs roof/dead) — per-pixel product, segment mean
    if "ndvi" in sp:
        F["ndvi_ndsm_coherence"] = G.mean(np.clip(sp["ndvi"], 0, 1) * np.clip(ndsm, 0, 30), fill=np.nan)
    else:
        F["ndvi_ndsm_coherence"] = np.full(G.n, np.nan)

    # hansen — "recent" anchored on the DSM flight year (loss since the LiDAR was flown),
    # falling back to obs_year where the flight year is unknown
    hz = L["hansen"]
    if hz is not None:
        ly = hz["loss_year"].astype(np.int16)
        re_ = min(obs_year - 2000, 24)
        F["hansen_treecover2000"] = G.mean(hz["treecover2000"].astype(np.float32))
        F["hansen_loss_frac"] = G.frac(ly > 0)
        F["hansen_recent_loss_frac"] = G.frac((ly >= max(obs_year - 2005, 1)) & (ly <= re_))
        F["hansen_gain_frac"] = G.frac(hz["gain"])
        F["hansen_current_forest_frac"] = G.frac(hz["current_forest"])
        F["hansen_lossyear_max"] = G.max(ly.astype(np.float32))
        # per-pixel anchor: DSM flight year raster (0 = unknown → obs_year)
        r = als.get(newest)
        anchor = (r["dsm_year"].astype(np.int16) if r is not None else np.zeros(ly.shape, np.int16))
        anchor = np.where(anchor > 0, anchor, obs_year) - 2000
        F["hansen_loss_3yr_frac"] = G.frac((ly > 0) & (ly >= anchor - 3) & (ly <= re_))
    else:
        for k in ("hansen_treecover2000", "hansen_loss_frac", "hansen_recent_loss_frac", "hansen_loss_3yr_frac",
                  "hansen_gain_frac", "hansen_current_forest_frac", "hansen_lossyear_max"):
            F[k] = np.zeros(G.n)

    # texture (v1 GLCM, per-segment bbox crops)
    for k in ("glcm_contrast", "glcm_homogeneity", "glcm_entropy", "glcm_dissimilarity", "glcm_energy",
              "texture_complexity"):
        F[k] = np.zeros(G.n)
    if texture and sp and "red" in sp:
        try:
            from texture_features import compute_texture_per_segment
            grey = (0.299 * np.nan_to_num(sp["red"]) + 0.587 * np.nan_to_num(sp["green"])
                    + 0.114 * np.nan_to_num(sp["blue"])).astype(np.float32)
            tex = compute_texture_per_segment(labels, {"transform": L["transform"], "shape": labels.shape},
                                              grey_image=grey)
            if tex:
                for i, lid in enumerate(G.ids):
                    t = tex.get(int(lid))
                    if t:
                        for k in ("glcm_contrast", "glcm_homogeneity", "glcm_entropy", "glcm_dissimilarity",
                                  "glcm_energy", "texture_complexity"):
                            F[k][i] = t.get(k, 0.0)
        except Exception as e:  # noqa: BLE001
            log.debug("texture failed: %s", e)

    # context distances
    ctx = context or {}
    for k in ("road", "path", "rail", "water", "building", "parcel_edge"):
        m = ctx.get(k)
        F[f"dist_{k}"] = G.mean(_edt(m)) if m is not None and m.shape == labels.shape else np.full(G.n, 500.0)

    # neighbourhood
    a, b = _adjacency(labels)
    pos = {int(l): i for i, l in enumerate(G.ids)}
    ai = np.array([pos.get(int(x), -1) for x in a]); bi = np.array([pos.get(int(x), -1) for x in b])
    ok = (ai >= 0) & (bi >= 0); ai, bi = ai[ok], bi[ok]
    wgt = F["area"][bi]
    nb_w = np.bincount(ai, wgt, G.n)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["nb_h_mean"] = np.nan_to_num(np.bincount(ai, wgt * F["h_mean"][bi], G.n) / nb_w)
        F["nb_ndvi_mean"] = np.nan_to_num(np.bincount(ai, wgt * F["fused_ndvi_mean"][bi], G.n) / nb_w)
    F["nb_n"] = np.bincount(ai, minlength=G.n).astype(float)
    F["nb_h_diff"] = F["h_mean"] - F["nb_h_mean"]; F["nb_ndvi_diff"] = F["fused_ndvi_mean"] - F["nb_ndvi_mean"]

    # centroid (3035)
    rows, cols = np.divmod(G.idx, labels.shape[1])
    cy = np.bincount(G.inv, rows.astype(float), G.n) / G.count
    cx = np.bincount(G.inv, cols.astype(float), G.n) / G.count
    tf = L["transform"]
    F["centroid_e"] = tf.c + (cx + 0.5) * tf.a
    F["centroid_n"] = tf.f + (cy + 0.5) * tf.e

    df = pd.DataFrame(F)
    for k in ALL_KEYS:
        if k not in df:
            df[k] = 0.0
    return df.replace([np.inf, -np.inf], np.nan)
