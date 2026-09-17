"""segv2 live-path adapter — run the v2 model on the arrays the v1 pipeline
already holds in memory (processor tile / ``/api/v1/segment`` request), without
a full GPKG.

Three pieces, mirroring ``build_dataset.build_kg`` step for step so the
feature distribution at inference matches training:

* :func:`layers_from_arrays` — the ``L`` dict ``features.extract`` expects,
  built from ``dtm/dsm/mask/transform`` + the processor's ``spectral``,
  resampled Copernicus, ``dtm_dates/dsm_dates`` (keys ``'20220915'`` → 2022),
  Hansen prior.  Real flight years come from ``acquisition`` (never mosaic
  labels).  ``harmonics=None`` always (dropped from the v2 contract).
* :class:`Context` — OSM (road/path/rail/water), cadastre footprints + parcel
  edges, INVEKOS Schläge for one bbox; ``.rasters()`` gives the ``dist_*``
  context rasters + an INVEKOS polygon-index raster + OSM hard edges.  The
  processor passes its already-fetched parcels/footprints; the app path
  fetches them via the cadastre fast paths (``/spatial/parcels|footprints``).
* :func:`classify` — ``ModelV2.predict`` + the INVEKOS post-hoc override
  (INVEKOS *was* the label source, so where a segment sits ≥ 70 %% inside one
  Schlag of an agri/veg type and the physical height veto holds, the Schlag
  type wins with conf ≥ 0.9, ``classifier_source='v2+invekos'``).

:func:`run` chains all three for one tile and returns per-label results in
the same ``{label: (type, conf, source, extras)}`` shape
``object_segmentation`` consumes.

Fleet-safety: nothing here is imported by the v1 path; the v2 branch in
``object_segmentation.segment_and_classify`` is opt-in (``model='v2'`` /
``SEG_MODEL_VERSION=v2``).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio import features as rfeatures
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import object_segmentation as oseg  # noqa: E402
from terrain_analysis import compute_tri, compute_tpi, compute_curvature  # noqa: E402
import acquisition as ACQ  # noqa: E402
import features as F2  # noqa: E402
import labels as L2  # noqa: E402

log = logging.getLogger("segv2.inference")

_T_3035_4326 = Transformer.from_crs(3035, 4326, always_xy=True)
_T_4326_3035 = Transformer.from_crs(4326, 3035, always_xy=True)

# INVEKOS override: types we trust from the Schlag polygon when the model already
# said "some vegetation / agri class" and the height veto holds.
INVEKOS_OVERRIDE_TYPES = {"crop", "grass", "vineyard", "orchard", "garden"}
INVEKOS_MIN_FRAC = 0.70
INVEKOS_CONF = 0.90
# model classes that an INVEKOS type may replace (never roof/road/water/rock …)
_VEG_LIKE = {"crop", "grass", "vineyard", "orchard", "garden", "shrub", "tree", "hedge"}


def _year_of(k) -> int:
    """'20220915' | 2022 | '2022' → 2022."""
    return int(str(k)[:4])


def _bounds(transform, shape_hw):
    h, w = shape_hw
    return (transform.c, transform.f + h * transform.e, transform.c + w * transform.a, transform.f)


def _bbox_wgs(bounds_3035, pad=0.002):
    x0, y0, x1, y1 = bounds_3035
    pts = [_T_3035_4326.transform(x, y) for x, y in ((x0, y0), (x1, y1), (x0, y1), (x1, y0))]
    lons, lats = zip(*pts)
    return (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)


# ---------------------------------------------------------------------------
# 1. pixel layers from in-memory arrays
# ---------------------------------------------------------------------------
def layers_from_arrays(dtm, dsm, mask, transform, *, spectral=None, cop=None,
                       dtm_dates=None, dsm_dates=None, hansen=None,
                       obs_year: int = 2024, ortho_year: int | None = None) -> dict | None:
    """Build the ``L`` dict for :func:`features.extract` from live arrays.

    ``spectral``: processor dict (``red/green/blue/brightness/green_ratio/
    rg_index`` + optional ``nir/ndvi``).  NIR is trusted iff present (the live
    path reads RGBI operates, never an alpha plane).
    ``cop``: *resampled* 1 m Copernicus dict (``ndvi/landcover/vv/vh``) as
    produced by ``object_segmentation._resample_copernicus``; a raw
    ``copernicus`` dict (with ``transform``) is resampled here.
    """
    dtm = np.asarray(dtm, np.float32); dsm = np.asarray(dsm, np.float32)
    mask = np.asarray(mask, bool) & np.isfinite(dtm) & np.isfinite(dsm)
    if mask.sum() < 100:
        return None
    h, w = dtm.shape
    L: dict = {"dtm": dtm, "dsm": dsm, "mask": mask, "transform": transform}
    L["ndsm"] = np.where(mask, np.clip(dsm - dtm, 0, None), np.nan).astype(np.float32)
    dtm0 = np.where(mask, dtm, np.nanmean(dtm[mask]))
    L["slope"] = oseg._slope(dtm0)
    L["aspect"] = oseg._aspect(dtm0)
    L["tri"] = compute_tri(dtm0)
    L["tpi"] = compute_tpi(dtm0, radius=5)
    with np.errstate(invalid="ignore", divide="ignore"):
        L["curv"] = compute_curvature(dtm0)["profile_curvature"]
    L["dsm_rough"] = F2._local_std(dsm, 3)
    L["dtm_rough"] = F2._local_std(dtm, 3)
    d0 = np.nan_to_num(dsm)
    L["dsm_edge"] = np.hypot(ndimage.sobel(d0, 1), ndimage.sobel(d0, 0)).astype(np.float32)

    # --- ortho: recompute every index from the bands (black → NaN, like training) ---
    sp_out = None
    bounds = _bounds(transform, (h, w))
    if ortho_year is None:
        try:
            import ortho_io
            bw = _bbox_wgs(bounds, pad=0)
            ops = ortho_io.find_rgbi_operates(bw[1], bw[0], bw[3], bw[2])
            ortho_year = int(ops[0][:4]) if ops else None
        except Exception as e:  # noqa: BLE001
            log.debug("ortho operate lookup failed: %s", e)
    L["ortho_year"] = ortho_year
    L["ortho_stack_years"] = [ortho_year] if ortho_year else []
    nir_year = None
    if spectral and spectral.get("red") is not None and spectral.get("green") is not None \
            and spectral.get("blue") is not None:
        r = np.asarray(spectral["red"], np.float32); gg = np.asarray(spectral["green"], np.float32)
        b = np.asarray(spectral["blue"], np.float32)
        s = r + gg + b
        black = s == 0
        sp_out = {"red": r, "green": gg, "blue": b, "brightness": s / 3.0}
        with np.errstate(divide="ignore", invalid="ignore"):
            sp_out["green_ratio"] = np.where(s > 0, gg / s, np.nan).astype(np.float32)
            sp_out["rg_index"] = np.where(r + gg > 0, (r - gg) / (r + gg), np.nan).astype(np.float32)
        for k in ("red", "green", "blue", "brightness", "green_ratio", "rg_index"):
            sp_out[k] = np.where(black, np.nan, sp_out[k]).astype(np.float32)
        nir = spectral.get("nir")
        # >1 % non-black coverage, same gate as features.pixel_layers
        if nir is not None and float(np.mean(~black)) > 0.01:
            n = np.asarray(nir, np.float32)
            with np.errstate(divide="ignore", invalid="ignore"):
                sp_out["nir"] = n
                sp_out["ndvi"] = np.where(n + r > 0, (n - r) / (n + r), np.nan).astype(np.float32)
                sp_out["ndwi"] = np.where(n + gg > 0, (gg - n) / (gg + n), np.nan).astype(np.float32)
                sp_out["savi"] = (1.5 * (n - r) / (n + r + 0.5 * 255)).astype(np.float32)
            for k in ("nir", "ndvi", "ndwi", "savi"):
                sp_out[k] = np.where(black, np.nan, sp_out[k]).astype(np.float32)
            nir_year = ortho_year
    L["spectral"] = sp_out
    L["nir_year"] = nir_year
    L["nir_years"] = [nir_year] if nir_year else []
    L["ndvi_years"] = {nir_year: sp_out["ndvi"]} if nir_year and sp_out and "ndvi" in sp_out else {}
    L["ortho_flight_years"] = {}
    if ortho_year:
        try:
            L["ortho_flight_years"][ortho_year] = ACQ.ortho_flight_year(bounds, ortho_year)
        except Exception as e:  # noqa: BLE001
            log.debug("ortho flight year lookup failed: %s", e)

    # --- copernicus ---
    if cop is not None and "transform" in cop and not isinstance(cop.get("ndvi"), type(None)) \
            and np.asarray(cop["ndvi"]).shape != (h, w):
        cop = oseg._resample_copernicus(cop, transform, (h, w))
    cop = cop or {}
    L["cop_ndvi"] = None if cop.get("ndvi") is None else np.asarray(cop["ndvi"], np.float32)
    lc = cop.get("landcover")
    if isinstance(lc, dict):
        lc = lc.get("map")
    L["worldcover"] = None if lc is None else np.asarray(lc).astype(np.uint8)
    L["sar_vv"] = None if cop.get("vv") is None else np.asarray(cop["vv"], np.float32)
    L["sar_vh"] = None if cop.get("vh") is None else np.asarray(cop["vh"], np.float32)
    bev_ndvi = sp_out.get("ndvi") if sp_out else None
    L["fused_ndvi"] = oseg._fuse_ndvi(bev_ndvi, L["cop_ndvi"], mask)

    # --- multi-date (mosaic year keys) ---
    L["dtm_dates"] = {}; L["dsm_dates"] = {}
    for k, a in (dtm_dates or {}).items():
        if a is not None and np.asarray(a).shape == (h, w):
            L["dtm_dates"][_year_of(k)] = np.asarray(a, np.float32)
    for k, a in (dsm_dates or {}).items():
        if a is not None and np.asarray(a).shape == (h, w):
            L["dsm_dates"][_year_of(k)] = np.asarray(a, np.float32)
    L["dtm_dates"].setdefault(obs_year, dtm)
    L["dsm_dates"].setdefault(obs_year, dsm)
    L["als_years"] = {}
    try:
        for my in sorted(set(L["dtm_dates"]) | set(L["dsm_dates"])):
            L["als_years"][my] = ACQ.als_year_rasters(transform, (h, w), my)
    except Exception as e:  # noqa: BLE001
        log.warning("ALS flight-year lookup failed: %s", e)

    # --- hansen (processor prior dict already has the right keys) ---
    L["hansen"] = None
    if hansen and hansen.get("treecover2000") is not None:
        tc = np.asarray(hansen["treecover2000"])
        if tc.shape == (h, w):
            tc = np.nan_to_num(tc).astype(np.uint8)
            ly = hansen.get("loss_year")
            ly = np.nan_to_num(np.asarray(ly)).astype(np.uint8) if ly is not None else np.zeros_like(tc)
            gain = hansen.get("gain")
            gain = np.asarray(gain).astype(bool) if gain is not None else np.zeros(tc.shape, bool)
            cur = hansen.get("current_forest")
            cur = np.asarray(cur).astype(bool) if cur is not None else ((tc >= 25) & ~(ly > 0))
            L["hansen"] = {"treecover2000": tc, "loss_year": ly, "gain": gain, "current_forest": cur}
    L["harmonics"] = None
    return L


# ---------------------------------------------------------------------------
# 2. vector context for one bbox
# ---------------------------------------------------------------------------
def _fetch_fast(path: str, bbox_wgs, key: str, limit=20000) -> list | None:
    """Cadastre viewport fast path → [geom3035]; None on failure/truncation."""
    w, s, e, n = bbox_wgs
    for attempt in range(3):
        d = L2._get_json(f"{L2.CADASTRE_BASE}/spatial/{path}",
                         {"west": w, "south": s, "east": e, "north": n, "limit": limit},
                         timeout=60, retries=1)
        if not d or d.get("truncated"):
            return None
        if d.get("ready", True):
            out = []
            for f in d.get(key, []):
                g = f.get("geometry")
                if g:
                    try:
                        gg = shape(g)
                        if not gg.is_empty:
                            out.append(shp_transform(_T_4326_3035.transform, gg))
                    except Exception:  # noqa: BLE001
                        pass
            return out
        time.sleep(1.5 * (attempt + 1))
    return None


class Context:
    """OSM + cadastre + INVEKOS vectors for one AOI (EPSG:3035).

    ``parcels`` / ``footprints``: shapely EPSG:3035 geometries when the caller
    already has them (processor ``cadastre_data``); otherwise fetched via the
    cadastre ``/spatial/*`` fast paths.  ``flight_year`` picks the INVEKOS
    release nearest the LiDAR flight (``acquisition.dsm_flight_year``).
    """

    def __init__(self, bbox_3035, *, flight_year: int | None = None, parcels=None,
                 footprints=None, invekos: bool = True, osm: bool = True):
        t0 = time.time()
        self.bbox_3035 = tuple(bbox_3035)
        self.bbox_wgs = _bbox_wgs(self.bbox_3035)
        if flight_year is None:
            try:
                flight_year = ACQ.dsm_flight_year(self.bbox_3035)
            except Exception:  # noqa: BLE001
                flight_year = None
        self.flight_year = flight_year
        self.parcels = list(parcels) if parcels is not None else (_fetch_fast("parcels", self.bbox_wgs, "parcels") or [])
        self.footprints = list(footprints) if footprints is not None else (_fetch_fast("footprints", self.bbox_wgs, "footprints") or [])
        self.osm = L2.fetch_osm(self.bbox_wgs) if osm else {"road": [], "rail": [], "water_line": [], "water_area": []}
        self.inv: list[tuple] = []
        self.inv_year = None
        if invekos:
            r = L2.fetch_invekos_api(self.bbox_3035, nearest_to=flight_year)
            if r is not None:
                self.inv, self.inv_year = r
            elif L2.INVEKOS_GPKG.exists():
                self.inv = L2.fetch_invekos(self.bbox_3035, nearest_to=flight_year)
        self.water_polys = [g for g, _ in self.osm["water_area"]] + \
            [g.buffer(L2.OSM_WATER_LINE.get(fc, 1.5), cap_style=2) for g, fc in self.osm["water_line"]]
        log.info("context: %d parcels, %d footprints, osm %d road/%d rail/%d water, invekos %d (yr %s, flight %s) in %.1fs",
                 len(self.parcels), len(self.footprints), len(self.osm["road"]), len(self.osm["rail"]),
                 len(self.water_polys), len(self.inv), self.inv_year, flight_year, time.time() - t0)

    # --- rasters -----------------------------------------------------------------
    @staticmethod
    def _burn(geoms, transform, shape_hw, all_touched=False):
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros(shape_hw, bool)
        return rfeatures.rasterize([(g, 1) for g in geoms], out_shape=shape_hw, transform=transform,
                                   fill=0, dtype=np.uint8, all_touched=all_touched).astype(bool)

    def rasters(self, transform, shape_hw) -> dict:
        """→ {'context': {road,path,rail,water,building,parcel_edge}, 'invekos_idx': int32,
        'invekos_types': [(type, snar)], 'osm_edges': bool}"""
        roads = [g for g, fc in self.osm["road"] if L2.OSM_ROAD.get(fc, ("road",))[0] == "road"]
        paths = [g for g, fc in self.osm["road"] if L2.OSM_ROAD.get(fc, ("road",))[0] == "path"]
        rails = [g for g, _ in self.osm["rail"]]
        ctx = {
            "road": self._burn(roads, transform, shape_hw),
            "path": self._burn(paths, transform, shape_hw),
            "rail": self._burn(rails, transform, shape_hw),
            "water": self._burn(self.water_polys, transform, shape_hw),
            "building": self._burn(self.footprints, transform, shape_hw),
            "parcel_edge": self._burn([g.boundary for g in self.parcels], transform, shape_hw),
        }
        # INVEKOS polygon index (1-based; 0 = none). Shrunk 1 m like the label pass.
        inv_idx = np.zeros(shape_hw, np.int32)
        inv_types: list[tuple[str, str]] = []
        shapes = []
        for g, ty, snar in self.inv:
            gg = g.buffer(-1.0)
            if gg.is_empty:
                continue
            inv_types.append((ty, snar or ""))
            shapes.append((gg, len(inv_types)))
        if shapes:
            inv_idx = rfeatures.rasterize(shapes, out_shape=shape_hw, transform=transform,
                                          fill=0, dtype=np.int32)
        lines = [g for g, _ in self.osm["road"]] + rails + [g for g, _ in self.osm["water_line"]] + \
                [g.boundary for g in self.water_polys]
        edges = self._burn(lines, transform, shape_hw, all_touched=True)
        return {"context": ctx, "invekos_idx": inv_idx, "invekos_types": inv_types, "osm_edges": edges}


# ---------------------------------------------------------------------------
# 3. classification (+ INVEKOS post-hoc)
# ---------------------------------------------------------------------------
def _sparse_mode(G: F2.Grouped, idx_raster: np.ndarray):
    """Per segment: (mode index, mode fraction of segment pixels) of an int raster,
    ignoring 0.  Sparse (np.unique on pairs) — polygon counts can be thousands."""
    v = idx_raster.ravel()[G.idx].astype(np.int64)
    ok = v > 0
    mode = np.zeros(G.n, np.int64); frac = np.zeros(G.n, np.float64)
    if not ok.any():
        return mode, frac
    pairs, cnt = np.unique(np.stack([G.inv[ok], v[ok]], 1), axis=0, return_counts=True)
    order = np.lexsort((-cnt, pairs[:, 0]))          # by segment, biggest count first
    seg = pairs[order, 0]
    first = np.concatenate([[True], seg[1:] != seg[:-1]])
    mode[seg[first]] = pairs[order, 1][first]
    frac[seg[first]] = cnt[order][first] / G.count[seg[first]]
    return mode, frac


def add_invekos_columns(df: pd.DataFrame, labels: np.ndarray, mask: np.ndarray,
                        invekos_idx: np.ndarray, invekos_types: list) -> pd.DataFrame:
    """Adds ``invekos_type`` / ``invekos_snar`` / ``invekos_frac`` per segment row."""
    df = df.copy()
    df["invekos_type"] = ""; df["invekos_snar"] = ""; df["invekos_frac"] = 0.0
    if invekos_idx is None or not invekos_types or len(df) == 0:
        return df
    G = F2.Grouped(labels, mask)
    mode, frac = _sparse_mode(G, invekos_idx)
    pos = pd.Series(np.arange(G.n), index=G.ids)
    rows = pos.reindex(df["label"].to_numpy()).to_numpy()
    okr = np.isfinite(rows)
    ri = rows[okr].astype(int)
    m = mode[ri]; f = frac[ri]
    ty = np.array([invekos_types[i - 1][0] if i > 0 else "" for i in m], dtype=object)
    sn = np.array([invekos_types[i - 1][1] if i > 0 else "" for i in m], dtype=object)
    df.loc[okr, "invekos_type"] = ty
    df.loc[okr, "invekos_snar"] = sn
    df.loc[okr, "invekos_frac"] = f
    return df


def classify(df: pd.DataFrame, *, model=None, invekos_override: bool = True) -> pd.DataFrame:
    """→ df + ``v2_type, v2_conf, v2_source, v2_type2, v2_conf2``.

    ``v2_source`` ∈ {'v2', 'v2+invekos'}.  INVEKOS override rule: segment ≥
    ``INVEKOS_MIN_FRAC`` inside one Schlag whose type ∈ INVEKOS_OVERRIDE_TYPES,
    the model's answer is vegetation-like, and ``h_p90`` ≤ ``labels.MAX_H``
    for the Schlag type.
    """
    from model_v2 import get_model
    m = model or get_model()
    out = df.copy()
    if len(out) == 0:
        for k, v in (("v2_type", ""), ("v2_conf", 0.0), ("v2_source", ""), ("v2_type2", ""), ("v2_conf2", 0.0)):
            out[k] = v
        return out
    types, conf, P = m.predict(out)
    t2, c2 = m.top2(P)
    types = types.astype(object); conf = conf.astype(np.float32)
    src = np.full(len(out), "v2", dtype=object)
    if invekos_override and "invekos_type" in out:
        ity = out["invekos_type"].to_numpy(dtype=object)
        ifr = out["invekos_frac"].to_numpy(dtype=np.float32)
        hp90 = out["h_p90"].to_numpy(dtype=np.float32) if "h_p90" in out else np.zeros(len(out), np.float32)
        n_ov = 0
        for i in range(len(out)):
            t = ity[i]
            if not t or t not in INVEKOS_OVERRIDE_TYPES or ifr[i] < INVEKOS_MIN_FRAC:
                continue
            if types[i] not in _VEG_LIKE or types[i] == t:
                continue
            if hp90[i] > L2.MAX_H.get(t, 99.0):
                continue
            types[i] = t; conf[i] = max(conf[i], INVEKOS_CONF); src[i] = "v2+invekos"
            n_ov += 1
        if n_ov:
            log.info("invekos override: %d/%d segments", n_ov, len(out))
    out["v2_type"] = types; out["v2_conf"] = conf; out["v2_source"] = src
    out["v2_type2"] = t2.astype(object); out["v2_conf2"] = c2.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 4. one-call tile runner
# ---------------------------------------------------------------------------
def run(labels: np.ndarray, dtm, dsm, mask, transform, *, spectral=None, cop=None,
        dtm_dates=None, dsm_dates=None, hansen=None, obs_year: int = 2024,
        ortho_year: int | None = None, context: Context | None = None,
        parcels=None, footprints=None, model=None, texture: bool = True) -> pd.DataFrame | None:
    """Features + classification for an existing segmentation ``labels``.

    Returns the feature DataFrame (one row per segment, ``label`` column) with
    ``v2_*`` and ``invekos_*`` columns, or None when the tile has no usable
    pixels.  ``context`` may be shared across tiles of one KG (pass the KG
    bbox); otherwise it is built for this tile's bounds.
    """
    t0 = time.time()
    L = layers_from_arrays(dtm, dsm, mask, transform, spectral=spectral, cop=cop, dtm_dates=dtm_dates,
                           dsm_dates=dsm_dates, hansen=hansen, obs_year=obs_year, ortho_year=ortho_year)
    if L is None:
        return None
    shp = L["mask"].shape
    if context is None:
        context = Context(_bounds(transform, shp), parcels=parcels, footprints=footprints)
    R = context.rasters(transform, shp)
    df = F2.extract(labels, L, context=R["context"], obs_year=obs_year, texture=texture)
    if df.empty:
        return df
    df = add_invekos_columns(df, labels, L["mask"], R["invekos_idx"], R["invekos_types"])
    df = classify(df, model=model)
    log.info("segv2 inference: %d segments in %.1fs (nir_year=%s dsm_year_med=%s)", len(df),
             time.time() - t0, L.get("nir_year"), float(np.nanmedian(df["dsm_year"])) if "dsm_year" in df else None)
    return df
