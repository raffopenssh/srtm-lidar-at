"""Apex-based single-tree inventory (v2 tree service).

Implements the forestry-grade tree detection contract requested by the
forestry-manager VM (docs/FEEDBACK-srtm-lidar-at-trees-v2.md in their repo):

* Variable-window local-maximum apex seeding on a lightly smoothed nDSM,
  window radius r(h) = a + b*h (defaults tuned for spruce/fir).
* Marker-controlled watershed constrained by those seeds.
* Per-height crown radius cap (r(h) * cap_factor) + max crown area cap.
* Stable, location-derived tree IDs: ``t_<E_dm>_<N_dm>`` (EPSG:3035 apex
  rounded to 1 dm) — deterministic across re-runs and epochs.
* Apex-based cross-epoch matching with a raster veto: a tree may only be
  labelled ``felled`` when the raster evidence (date-b nDSM around the
  apex) actually shows the drop; otherwise it is ``unmatched``.
* Segmentation-independent felling-patch layer from the raw nDSM drop.

Pure numpy/scipy/skimage — no BEV/network I/O here; callers pass rasters.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)

#: Bump when the algorithm changes in a way that invalidates caches.
TREE_ALGO_VERSION = "2.2.0"

# Defaults (spruce-oriented; all exposed as request params)
DEFAULT_CROWN_RADIUS_A = 1.2    # m — base local-max window radius
DEFAULT_CROWN_RADIUS_B = 0.08   # m per m tree height
DEFAULT_SMOOTH_SIGMA = 0.5      # m (px on the 1 m grid)
DEFAULT_MIN_TREE_HEIGHT = 3.0   # m
DEFAULT_CROWN_CAP_FACTOR = 1.5  # crown radius cap = r(h) * factor
DEFAULT_MAX_CROWN_AREA = 250.0  # m² hard cap per crown
DEFAULT_MATCH_RADIUS_M = 3.0    # apex matching radius floor
DEFAULT_FELLING_MIN_DROP_M = 5.0
DEFAULT_GROWTH_EPS_M = 0.3      # |dh| below this → stable
# Q3: crown-overlap second matching pass — minimum overlap as a fraction of
# the SMALLER crown. Within one epoch watershed crowns are disjoint, so a
# cross-epoch overlap this large is very unlikely for two different trees.
CROWN_OVERLAP_MIN_FRAC = 0.3
DEFAULT_MIN_APEX_PROMINENCE_M = 0.0  # m; 0 = off (Q3/Q10 request)

# --- v2.2 ortho-fused detection (FEEDBACK-3 recall round) ---------------
# Native-resolution apex seeding on the RGBI ortho: crown caps are bright
# blobs at the 1-3 m scale; a DoG band-pass + local maxima recovers apices
# the 1 m first-return gridding merged into one nDSM blob.
ORTHO_SEED_SIGMA_SMALL_M = 0.6   # DoG inner scale (crown-cap highlight)
ORTHO_SEED_SIGMA_LARGE_M = 2.4   # DoG outer scale (background canopy)
ORTHO_SEED_MIN_SEP_M = 1.6       # min spacing between accepted seeds
ORTHO_SEED_PCTL = 55.0           # response percentile threshold (positive
                                 # band values within canopy)
ORTHO_SEED_MAX_PER_CANOPY_HA = 1200  # sanity cap on added stems
ORTHO_SEED_HEIGHT_GATE_R_PX = 2  # nDSM max within this radius must clear
                                 # min_tree_height (kills ground FPs)
ORTHO_SEED_MIN_CROWN_SQM = 4.0   # two-pass prune: an ortho-added seed
                                 # whose FINAL watershed crown is smaller
                                 # than this was intra-crown branch
                                 # texture, not a merged sub-dominant top
                                 # — drop it and re-run the watershed
ORTHO_SEED_CLOSURE_WIN_M = 15    # local canopy-closure window (1 m grid)
ORTHO_SEED_CLOSURE_MIN = 0.7     # only seed where canopy closure >= this
                                 # (open stands are already nDSM-resolved;
                                 # restricts ortho FPs to where recall is
                                 # actually broken — closed fine canopy)

#: Allometry provenance (Q5) — surfaced via volume_method in summaries.
FORM_FACTOR = 0.42
VOLUME_METHOD = ("heuristic_h_crown: dbh_cm = max(5, 1.2*h + 3.0*max(0, "
                 "crown_diam-4)); vol = form_factor 0.42 * basal_area * h. "
                 "Generic spruce heuristic, NOT Pollanschütz. Over-estimates "
                 "DBH ~25-35% in dense stands where crowns under-split; "
                 "verify against local measurements.")


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


# ---------------------------------------------------------------------------
# Apex detection + crown segmentation
# ---------------------------------------------------------------------------

def detect_apices(
    ndsm: np.ndarray,
    mask: np.ndarray,
    min_height: float = DEFAULT_MIN_TREE_HEIGHT,
    a: float = DEFAULT_CROWN_RADIUS_A,
    b: float = DEFAULT_CROWN_RADIUS_B,
    smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
    min_prominence: float = DEFAULT_MIN_APEX_PROMINENCE_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variable-window local maxima on the smoothed nDSM.

    Returns (apex_bool, smooth, canopy_bool). Window radius per pixel is
    r(h) = a + b*h, quantised to integer pixel radii (1 m grid).

    ``min_prominence`` (m): if > 0, apices must stand at least that far
    above their surrounding saddle (h-maxima transform) — suppresses
    branch-tip apices in dense canopy (Q3/Q10 stabiliser).
    """
    z = np.where(mask & np.isfinite(ndsm), ndsm, 0.0).astype(np.float32)
    smooth = ndimage.gaussian_filter(z, sigma=smooth_sigma) if smooth_sigma > 0 else z
    canopy = (smooth >= min_height) & mask

    r = a + b * np.clip(smooth, 0.0, 60.0)
    r_int = np.clip(np.round(r).astype(np.int32), 1, 15)

    apex = np.zeros(smooth.shape, dtype=bool)
    for rv in np.unique(r_int[canopy]) if canopy.any() else []:
        fp = _disk(int(rv))
        mx = ndimage.maximum_filter(smooth, footprint=fp, mode="nearest")
        apex |= canopy & (r_int == rv) & (smooth >= mx - 1e-4)

    if min_prominence > 0 and apex.any():
        from skimage.morphology import h_maxima
        hm = h_maxima(smooth, float(min_prominence))
        # keep apices that coincide with (or touch) a prominent maximum
        apex &= ndimage.binary_dilation(hm.astype(bool), iterations=1)

    # Collapse plateaus (adjacent apex pixels) to a single representative px.
    lab, n = ndimage.label(apex)
    if n:
        pts = ndimage.maximum_position(smooth, lab, np.arange(1, n + 1))
        apex = np.zeros_like(apex)
        rr = np.array([p[0] for p in pts]); cc = np.array([p[1] for p in pts])
        apex[rr, cc] = True
    return apex, smooth, canopy


def ortho_apex_candidates(
    intensity: np.ndarray,
    res_m: float,
    canopy: np.ndarray,
    smooth: np.ndarray,
    min_height: float,
    min_sep_m: float = ORTHO_SEED_MIN_SEP_M,
    pctl: float = ORTHO_SEED_PCTL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native-resolution apex seeding on ortho intensity (v2.2).

    ``intensity`` is a 2-D float/uint8 array (NIR preferred, luminance
    fallback) on a grid aligned with the 1 m detection grid: shape must be
    ``(h*f, w*f)`` with ``f = round(1/res_m)``.  ``canopy``/``smooth`` are
    the 1 m rasters from :func:`detect_apices`.

    Returns ``(rows_1m, cols_1m, strength01)`` sorted strongest-first,
    height-gated (nDSM max within ~2 px must clear ``min_height``) and
    spaced at least ``min_sep_m`` apart.  Pure numpy/scipy — caller does
    all I/O.
    """
    z = intensity.astype(np.float32)
    hh, ww = z.shape
    h1, w1 = canopy.shape
    f = max(1, int(round(1.0 / max(res_m, 1e-6))))

    band = (ndimage.gaussian_filter(z, ORTHO_SEED_SIGMA_SMALL_M / res_m)
            - ndimage.gaussian_filter(z, ORTHO_SEED_SIGMA_LARGE_M / res_m))

    # canopy mask on the hi-res grid (nearest-neighbour zoom via indexing)
    ri = np.minimum(np.arange(hh) // f, h1 - 1)
    ci = np.minimum(np.arange(ww) // f, w1 - 1)
    canopy_hi = canopy[np.ix_(ri, ci)]
    if not canopy_hi.any():
        e = np.array([], dtype=np.int32)
        return e, e.copy(), np.array([], dtype=np.float32)

    pos = band[canopy_hi]
    pos = pos[pos > 0]
    if pos.size < 16:
        e = np.array([], dtype=np.int32)
        return e, e.copy(), np.array([], dtype=np.float32)
    thr = float(np.percentile(pos, pctl))

    # local maxima at ~half the min separation
    rad_px = max(2, int(round(min_sep_m * 0.5 / res_m)))
    mx = ndimage.maximum_filter(band, size=2 * rad_px + 1, mode="nearest")
    peaks = canopy_hi & (band >= mx - 1e-6) & (band > thr)
    pr, pc = np.nonzero(peaks)
    if pr.size == 0:
        e = np.array([], dtype=np.int32)
        return e, e.copy(), np.array([], dtype=np.float32)
    strength = band[pr, pc]

    # height gate on the 1 m grid: smooth nDSM max within ~2 px
    smax = ndimage.maximum_filter(smooth, size=2 * ORTHO_SEED_HEIGHT_GATE_R_PX + 1,
                                  mode="nearest")
    # closure gate: only seed inside locally CLOSED canopy — in open stands
    # the 1 m nDSM already resolves every crown, and ortho highlights there
    # are ground vegetation / branch glints, not missed trees.
    closure = ndimage.uniform_filter(canopy.astype(np.float32),
                                     size=int(ORTHO_SEED_CLOSURE_WIN_M),
                                     mode="nearest")
    r1 = np.minimum(pr // f, h1 - 1)
    c1 = np.minimum(pc // f, w1 - 1)
    ok = (smax[r1, c1] >= min_height) & \
         (closure[r1, c1] >= ORTHO_SEED_CLOSURE_MIN)
    r1, c1, strength = r1[ok], c1[ok], strength[ok]
    if r1.size == 0:
        e = np.array([], dtype=np.int32)
        return e, e.copy(), np.array([], dtype=np.float32)

    # greedy spacing (strongest-first) on a metre-bucket grid
    order = np.argsort(strength)[::-1]
    cell = max(min_sep_m, 1.0)
    taken: dict[tuple[int, int], list[tuple[float, float]]] = {}
    keep = []
    sep2 = min_sep_m * min_sep_m
    cap = int(ORTHO_SEED_MAX_PER_CANOPY_HA * max(canopy.sum() / 1e4, 0.05))
    for i in order:
        y, x = float(r1[i]), float(c1[i])  # 1 m px == metres
        ky, kx = int(y // cell), int(x // cell)
        clash = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for (oy, ox) in taken.get((ky + dy, kx + dx), ()):
                    if (y - oy) ** 2 + (x - ox) ** 2 < sep2:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if clash:
            continue
        taken.setdefault((ky, kx), []).append((y, x))
        keep.append(i)
        if len(keep) >= cap:
            break
    keep = np.array(keep, dtype=np.int64)
    s = strength[keep]
    # rank-normalised strength in (0, 1]
    rk = np.argsort(np.argsort(s))
    s01 = ((rk + 1.0) / max(len(rk), 1)).astype(np.float32)
    return r1[keep].astype(np.int32), c1[keep].astype(np.int32), s01


def fuse_apices(
    apex: np.ndarray,
    smooth: np.ndarray,
    cand_r: np.ndarray,
    cand_c: np.ndarray,
    cand_s: np.ndarray,
    a: float = DEFAULT_CROWN_RADIUS_A,
    b: float = DEFAULT_CROWN_RADIUS_B,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge ortho seed candidates into the nDSM apex set (v2.2).

    A candidate is added when it is farther than ``max(1.5, 0.6*r(h))``
    metres from every existing apex (nDSM apices win ties; they carry
    height evidence) AND sits near the top of its local canopy
    (``smooth >= 0.75 * local 9 m max`` — a merged sub-dominant top is
    nearly as tall as its dominant neighbour; a crown-flank highlight is
    not).  Returns ``(apex_fused, source, ortho_strength)``
    where ``source`` is an int8 raster (1 = ndsm, 2 = ortho-added) and
    ``ortho_strength`` a float raster holding seed strength at added px.
    """
    src = np.zeros(apex.shape, np.int8)
    src[apex] = 1
    strength = np.zeros(apex.shape, np.float32)
    if cand_r.size == 0:
        return apex, src, strength
    ar, ac = np.nonzero(apex)
    from scipy.spatial import cKDTree
    kd = cKDTree(np.column_stack([ar, ac])) if ar.size else None
    local_max = ndimage.maximum_filter(smooth, size=9, mode="nearest")

    fused = apex.copy()
    cell = 3.0
    taken: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for i in range(cand_r.size):
        y, x = float(cand_r[i]), float(cand_c[i])
        h = float(smooth[int(y), int(x)])
        if h < 0.75 * float(local_max[int(y), int(x)]):
            continue  # crown flank / gap-edge highlight, not a merged top
        sep = max(1.5, 0.6 * (a + b * min(h, 60.0)))
        if kd is not None:
            d, _ = kd.query([y, x], k=1)
            if d < sep:
                continue
        # also keep spacing among added seeds
        ky, kx = int(y // cell), int(x // cell)
        clash = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for (oy, ox) in taken.get((ky + dy, kx + dx), ()):
                    if (y - oy) ** 2 + (x - ox) ** 2 < sep * sep:
                        clash = True
                        break
                if clash:
                    break
            if clash:
                break
        if clash:
            continue
        taken.setdefault((ky, kx), []).append((y, x))
        fused[int(y), int(x)] = True
        src[int(y), int(x)] = 2
        strength[int(y), int(x)] = float(cand_s[i])
    return fused, src, strength


def segment_crowns(
    smooth: np.ndarray,
    canopy: np.ndarray,
    apex: np.ndarray,
    ndsm: np.ndarray,
    a: float = DEFAULT_CROWN_RADIUS_A,
    b: float = DEFAULT_CROWN_RADIUS_B,
    cap_factor: float = DEFAULT_CROWN_CAP_FACTOR,
    max_crown_area: float = DEFAULT_MAX_CROWN_AREA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Marker-controlled watershed seeded by apices, with radius/area caps.

    Returns (labels, apex_rows, apex_cols, apex_heights) where labels is an
    int32 raster (0 = background) and the arrays are indexed by label-1.
    """
    from skimage import segmentation as sk_seg

    markers, n = ndimage.label(apex)
    if n == 0:
        return np.zeros(smooth.shape, np.int32), np.array([]), np.array([]), np.array([])

    labels = sk_seg.watershed(-smooth, markers=markers, mask=canopy)

    # Apex coordinates per label (markers are single pixels).
    ar = np.zeros(n + 1, np.int32); ac = np.zeros(n + 1, np.int32)
    rr, cc = np.nonzero(apex)
    ar[markers[rr, cc]] = rr; ac[markers[rr, cc]] = cc
    ah = ndsm[ar[1:], ac[1:]].astype(np.float32)  # raw nDSM at apex

    # Per-label allowed radius: min(r(h)*cap, sqrt(max_area/pi))
    cap_r = (a + b * np.clip(ah, 0, 60)) * cap_factor
    if max_crown_area:
        cap_r = np.minimum(cap_r, np.sqrt(max_crown_area / np.pi))
    cap_r_full = np.concatenate([[0.0], cap_r]).astype(np.float32)

    gr, gc = np.indices(smooth.shape)
    lab_flat = labels
    d2 = (gr - ar[lab_flat]) ** 2 + (gc - ac[lab_flat]) ** 2
    too_far = (lab_flat > 0) & (d2 > cap_r_full[lab_flat] ** 2)
    labels = np.where(too_far, 0, labels).astype(np.int32)
    return labels, ar[1:], ac[1:], ah


# ---------------------------------------------------------------------------
# Per-tree record extraction
# ---------------------------------------------------------------------------

def _stable_tree_id(e: float, n: float) -> str:
    """Deterministic location-derived ID: apex EPSG:3035 rounded to 1 dm."""
    return f"t_{int(round(e * 10))}_{int(round(n * 10))}"


def _dbh_volume_est(h: float, crown_diam: float) -> tuple[float, float]:
    """Crude generic spruce allometry. Returns (dbh_cm, volume_m3)."""
    dbh_cm = max(5.0, 1.2 * h + 3.0 * max(0.0, crown_diam - 4.0))
    ba = np.pi / 4.0 * (dbh_cm / 100.0) ** 2
    vol = 0.42 * ba * h  # form factor 0.42
    return round(dbh_cm, 1), round(vol, 3)


@dataclass
class Tree:
    tree_id: str
    seq: int
    label: int
    apex_e: float
    apex_n: float
    apex_row: int
    apex_col: int
    height_m: float
    crown_area_sqm: float
    crown_radius_mean_m: float
    crown_radius_max_m: float
    is_edge: bool
    dbh_est_cm: float = 0.0
    volume_m3_est: float = 0.0
    leaf_type: str = "unknown"
    leaf_type_conf: float = 0.0
    leaf_type_prob_conifer: float = 0.5
    vitality: str = "unknown"
    vitality_conf: float = 0.0
    detection_source: str = "ndsm"
    detection_conf: float = 0.75
    species_hint: str = "unknown"
    species_conf: float = 0.0
    spectral: dict = field(default_factory=dict)


def build_inventory(
    ndsm: np.ndarray,
    mask: np.ndarray,
    transform,
    min_height: float = DEFAULT_MIN_TREE_HEIGHT,
    a: float = DEFAULT_CROWN_RADIUS_A,
    b: float = DEFAULT_CROWN_RADIUS_B,
    smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
    cap_factor: float = DEFAULT_CROWN_CAP_FACTOR,
    max_crown_area: float = DEFAULT_MAX_CROWN_AREA,
    min_apex_prominence: float = DEFAULT_MIN_APEX_PROMINENCE_M,
    leaf_type_min_conf: float = 0.5,
    spectral: dict | None = None,
    nir: np.ndarray | None = None,
    ortho_intensity: np.ndarray | None = None,
    ortho_res_m: float | None = None,
    det_info: dict | None = None,
) -> tuple[list[Tree], np.ndarray, np.ndarray]:
    """Full apex-based inventory. Returns (trees, labels, canopy_mask).

    v2.2: if ``ortho_intensity`` (hi-res NIR/luminance aligned to the 1 m
    grid at ``ortho_res_m``) is given, native-resolution ortho seeds are
    fused into the nDSM apex set before the watershed (recall recovery in
    closed canopy).  ``det_info`` (optional dict) is filled with seeding
    telemetry for the response meta.
    """
    apex, smooth, canopy = detect_apices(ndsm, mask, min_height, a, b,
                                         smooth_sigma, min_apex_prominence)
    n_ndsm = int(apex.sum())
    src_raster = None
    ortho_strength = None
    cand_mask = None
    if ortho_intensity is not None and ortho_res_m:
        cr, cc, cs = ortho_apex_candidates(
            ortho_intensity, ortho_res_m, canopy, smooth, min_height)
        apex, src_raster, ortho_strength = fuse_apices(
            apex, smooth, cr, cc, cs, a, b)
        cand_mask = np.zeros(canopy.shape, bool)
        if cr.size:
            cand_mask[cr, cc] = True
            cand_mask = ndimage.binary_dilation(cand_mask, iterations=2)
        if det_info is not None:
            det_info.update(
                n_seeds_ndsm=n_ndsm,
                n_seeds_ortho_candidates=int(cr.size),
                n_seeds_ortho_added=int((src_raster == 2).sum()),
                ortho_seed_res_m=round(float(ortho_res_m), 2))
    elif det_info is not None:
        det_info.update(n_seeds_ndsm=n_ndsm)
    labels, ar, ac, ah = segment_crowns(
        smooth, canopy, apex, ndsm, a, b, cap_factor, max_crown_area)

    # v2.2 two-pass prune: an ortho-added seed that ends up with a
    # sub-ORTHO_SEED_MIN_CROWN_SQM watershed crown was branch texture
    # inside an existing crown, not a merged sub-dominant top. Drop those
    # seeds and re-run the watershed so their pixels return to the real
    # crown (keeps crown geometry honest — no 1–2 px confetti crowns).
    if src_raster is not None and len(ar):
        px_area0 = abs(transform.a * transform.e)
        areas0 = np.bincount(labels.ravel(), minlength=len(ar) + 1)[1:] * px_area0
        added = src_raster[ar, ac] == 2
        bad = added & (areas0 < ORTHO_SEED_MIN_CROWN_SQM)
        if bad.any():
            apex2 = np.zeros_like(apex)
            keep = ~bad
            apex2[ar[keep], ac[keep]] = True
            apex = apex2
            src_raster[ar[bad], ac[bad]] = 0
            if det_info is not None:
                det_info['n_seeds_ortho_pruned'] = int(bad.sum())
                det_info['n_seeds_ortho_added'] = int((src_raster == 2).sum())
            labels, ar, ac, ah = segment_crowns(
                smooth, canopy, apex, ndsm, a, b, cap_factor, max_crown_area)

    n = len(ar)
    if n == 0:
        return [], labels, canopy

    px_area = abs(transform.a * transform.e)
    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:] * px_area

    # crown_radius_max per label
    gr, gc = np.indices(labels.shape)
    ar_full = np.concatenate([[0], ar]); ac_full = np.concatenate([[0], ac])
    d = np.sqrt((gr - ar_full[labels]) ** 2 + (gc - ac_full[labels]) ** 2)
    rmax = ndimage.maximum(np.where(labels > 0, d, 0), labels, np.arange(1, n + 1))
    rmax = np.asarray(rmax, dtype=np.float32) * abs(transform.a)

    # edge detection: crown touches raster border or no-data mask
    edge_zone = ~mask
    edge_zone[0, :] = True; edge_zone[-1, :] = True
    edge_zone[:, 0] = True; edge_zone[:, -1] = True
    edge_near = ndimage.binary_dilation(edge_zone, iterations=1)
    edge_labels = set(np.unique(labels[edge_near & (labels > 0)]).tolist())

    # spectral aggregates per crown
    idx = np.arange(1, n + 1)
    spec_stats: dict[str, np.ndarray] = {}
    if spectral:
        for key in ("ndvi", "brightness", "rg_index", "green_ratio"):
            arr = spectral.get(key)
            if arr is not None and arr.shape == labels.shape:
                with np.errstate(invalid="ignore"):
                    spec_stats[key + "_mean"] = np.asarray(
                        ndimage.mean(np.nan_to_num(arr, nan=0.0), labels, idx))
        ndvi = spectral.get("ndvi")
        if ndvi is not None and ndvi.shape == labels.shape:
            nd = np.nan_to_num(ndvi, nan=0.0)
            spec_stats["ndvi_p10"] = np.asarray(ndimage.labeled_comprehension(
                nd, labels, idx, lambda v: float(np.percentile(v, 10)) if v.size else 0.0,
                float, 0.0))
    if nir is not None and nir.shape == labels.shape:
        spec_stats["nir_mean"] = np.asarray(ndimage.mean(nir.astype(np.float32), labels, idx))

    # crown steepness for leaf-type hint: (apex height - crown mean height) / radius
    hmean = np.asarray(ndimage.mean(np.where(mask, ndsm, 0.0), labels, idx))

    has_real_ndvi = nir is not None and "ndvi_mean" in spec_stats

    trees: list[Tree] = []
    for i in range(n):
        e, np_ = transform * (float(ac[i]) + 0.5, float(ar[i]) + 0.5)
        h = float(ah[i])
        area = float(areas[i])
        r_mean = float(np.sqrt(area / np.pi))
        crown_diam = 2.0 * r_mean
        dbh, vol = _dbh_volume_est(h, crown_diam)

        spec = {}
        for k, v in spec_stats.items():
            spec[k] = round(float(v[i]), 3)

        # leaf type: crown steepness + red/green index → soft probability
        steep = (h - float(hmean[i])) / max(r_mean, 0.5)
        rg = spec.get("rg_index_mean", 0.0)
        # logistic blend: steeper crowns + redder-than-green → conifer
        p_con = 1.0 / (1.0 + np.exp(-(1.6 * (steep - 1.6) - 8.0 * (rg + 0.02))))
        p_con = float(np.clip(0.25 + 0.5 * p_con + 0.15, 0.05, 0.95))  # Austrian conifer prior
        if p_con >= 0.5:
            leaf, conf = "coniferous", p_con
        else:
            leaf, conf = "broadleaf", 1.0 - p_con
        if conf < leaf_type_min_conf:
            leaf = "unknown"  # Q8: don't report coin flips as classifications

        # v2.2 detection provenance
        rr_i, cc_i = int(ar[i]), int(ac[i])
        if src_raster is not None and src_raster[rr_i, cc_i] == 2:
            det_src = "ortho"
            det_conf = float(np.clip(0.35 + 0.4 * float(ortho_strength[rr_i, cc_i]), 0.3, 0.75))
        elif cand_mask is not None and cand_mask[rr_i, cc_i]:
            det_src = "fused"   # nDSM apex independently confirmed by ortho
            det_conf = 0.9
        else:
            det_src = "ndsm"
            det_conf = 0.75

        trees.append(Tree(
            tree_id=_stable_tree_id(e, np_), seq=i + 1, label=i + 1,
            apex_e=round(e, 2), apex_n=round(np_, 2),
            apex_row=int(ar[i]), apex_col=int(ac[i]),
            height_m=round(h, 2), crown_area_sqm=round(area, 1),
            crown_radius_mean_m=round(r_mean, 2),
            crown_radius_max_m=round(float(rmax[i]), 2),
            is_edge=(i + 1) in edge_labels,
            dbh_est_cm=dbh, volume_m3_est=vol,
            leaf_type=leaf, leaf_type_conf=round(conf, 2),
            leaf_type_prob_conifer=round(p_con, 3),
            detection_source=det_src, detection_conf=round(det_conf, 2),
            spectral=spec,
        ))

    # --- Vitality (Q7): relative NDVI anomaly, not a fixed cut ------------
    # 'dead' keeps the absolute threshold (it validated well); 'stressed'
    # is now an ANOMALY within the AOI + same leaf class: NDVI <= p10 of
    # its leaf-type population. Raw percentile is shipped per tree so
    # clients can re-threshold (ndvi_percentile_in_aoi).
    if has_real_ndvi and trees:
        nd_all = np.array([t.spectral.get("ndvi_mean", np.nan) for t in trees],
                          dtype=np.float32)
        # dead first (absolute)
        for t, nd in zip(trees, nd_all):
            nd_p10 = t.spectral.get("ndvi_p10", nd)
            if np.isfinite(nd) and nd < 0.15 and nd_p10 < 0.10 and t.height_m >= 5.0:
                t.vitality = "dead"
                t.vitality_conf = round(min(0.95, 0.6 + (0.15 - float(nd)) * 2), 2)
                t.leaf_type = "dead"
                t.leaf_type_conf = t.vitality_conf
        # percentile within leaf-type population (excluding dead)
        for cls in ("coniferous", "broadleaf", "unknown"):
            sel = [i for i, t in enumerate(trees)
                   if t.vitality != "dead"
                   and (t.leaf_type == cls or (cls == "unknown" and t.leaf_type
                                               not in ("coniferous", "broadleaf", "dead")))
                   and np.isfinite(nd_all[i])]
            if not sel:
                continue
            vals = nd_all[sel]
            order = np.argsort(np.argsort(vals))
            pct = (order + 0.5) / len(vals) * 100.0
            p10 = float(np.percentile(vals, 10))
            for j, i in enumerate(sel):
                t = trees[i]
                t.spectral["ndvi_percentile_in_aoi"] = round(float(pct[j]), 1)
                if vals[j] <= p10:
                    t.vitality = "stressed"
                    t.vitality_conf = round(min(0.8, 0.4 + (p10 - float(vals[j])) * 3), 2)
                else:
                    t.vitality = "vital"
                    t.vitality_conf = round(min(0.9, float(pct[j]) / 100.0 + 0.3), 2)

    # --- Species hint (v2.2, FEEDBACK-3 side request) --------------------
    # Coarse conifer species hint from single-epoch RGB+NIR ortho ONLY
    # (deliberately no Sentinel/openEO dependency). Physically separable
    # in a summer RGBI image:
    #   * Norway spruce (Picea abies): dark crowns, low brightness & low
    #     green_ratio within the conifer population, steep narrow crowns.
    #   * European larch (Larix decidua): deciduous conifer — fresh light
    #     green in leaf-on imagery → clearly higher NDVI/green_ratio +
    #     brightness than spruce in the SAME scene.
    #   * Pine (Pinus): intermediate brightness, flatter/rounder crowns.
    # We use within-AOI percentiles (illumination-invariant), never
    # absolute cuts. Confidence is capped at 0.6 — this is a HINT; real
    # species ID needs multitemporal or hyperspectral data.
    if has_real_ndvi and trees:
        con = [i for i, t in enumerate(trees) if t.leaf_type == "coniferous"]
        if len(con) >= 10:
            br = np.array([trees[i].spectral.get("brightness_mean", np.nan)
                           for i in con], dtype=np.float32)
            gr = np.array([trees[i].spectral.get("green_ratio_mean", np.nan)
                           for i in con], dtype=np.float32)
            nd = np.array([trees[i].spectral.get("ndvi_mean", np.nan)
                           for i in con], dtype=np.float32)

            def _pctile(v):
                ok = np.isfinite(v)
                p = np.full(v.shape, 50.0, np.float32)
                if ok.sum() > 1:
                    r = np.argsort(np.argsort(v[ok]))
                    p[ok] = (r + 0.5) / ok.sum() * 100.0
                return p

            br_p, gr_p, nd_p = _pctile(br), _pctile(gr), _pctile(nd)
            for j, i in enumerate(con):
                t = trees[i]
                if t.vitality == "dead":
                    continue
                bright_and_green = float(min(br_p[j], gr_p[j], nd_p[j]))
                dark = float(max(br_p[j], gr_p[j]))
                if bright_and_green >= 75.0:
                    t.species_hint = "larch"
                    t.species_conf = round(min(0.6, 0.3 +
                                               (bright_and_green - 75.0) / 100.0), 2)
                elif dark <= 45.0:
                    t.species_hint = "spruce"
                    t.species_conf = round(min(0.6, 0.3 + (45.0 - dark) / 150.0), 2)
                elif br_p[j] > 45.0 and gr_p[j] <= 60.0 and nd_p[j] <= 60.0:
                    t.species_hint = "pine"
                    t.species_conf = 0.3
                else:
                    t.species_hint = "conifer_unspecified"
                    t.species_conf = 0.2
        for t in trees:
            if t.leaf_type == "broadleaf" and t.species_hint == "unknown":
                t.species_hint = "broadleaf_unspecified"
                t.species_conf = round(min(0.4, t.leaf_type_conf * 0.5), 2)

    return trees, labels, canopy


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(trees: list[Tree], canopy: np.ndarray, mask: np.ndarray,
              transform, aoi_area_sqm: float,
              ndsm: np.ndarray | None = None,
              h_dom_basis: str = "canopy") -> dict:
    """Explicit, self-consistent summary block (denominators included).

    Q4: h_dom is computed over the CANOPY area by default (Oberhöhe is
    defined over stocked area); pass h_dom_basis='total' for the old
    behaviour. Also ships h_p99_m and h_top100_m (moving-hectare grid).
    Q6: understory detectability caveat + canopy_gap_fraction +
    layer_profile (nDSM PIXEL heights in 2 m bins, if ndsm given).
    """
    px_area = abs(transform.a * transform.e)
    area_ha_total = aoi_area_sqm / 1e4
    area_ha_canopy = float(canopy.sum()) * px_area / 1e4
    non_edge = [t for t in trees if not t.is_edge]
    hs = np.array([t.height_m for t in trees], dtype=np.float32)
    crown_area_total = float(sum(t.crown_area_sqm for t in trees))

    # h_dom: mean of the 100 tallest stems per ha over the chosen basis
    basis_ha = area_ha_canopy if h_dom_basis == "canopy" else area_ha_total
    n_dom = max(1, int(round(100 * max(basis_ha, 0.01))))
    hs_sorted = np.sort(hs)[::-1]
    h_dom = float(np.mean(hs_sorted[:min(n_dom, len(hs_sorted))])) if len(hs) else 0.0

    # h_top100: mean of the tallest 100/ha on a moving 100 m grid (robust
    # Oberhöhe — immune to the small-stand collapse noted in Q4)
    h_top100 = 0.0
    if trees:
        cell = {}
        for t in trees:
            key = (int(t.apex_e // 100), int(t.apex_n // 100))
            cell.setdefault(key, []).append(t.height_m)
        tops = [max(v) for v in cell.values()]
        h_top100 = float(np.mean(tops)) if tops else 0.0

    hist = {}
    if len(hs):
        top = int(np.ceil(hs.max() / 2.0) * 2)
        edges = np.arange(0, top + 2, 2)
        counts, _ = np.histogram(hs, bins=edges)
        hist = {f"{int(edges[i])}-{int(edges[i+1])}": int(c)
                for i, c in enumerate(counts) if c}

    def _pct(p):
        return round(float(np.percentile(hs, p)), 2) if len(hs) else 0.0

    by_leaf: dict[str, int] = {}
    by_vitality: dict[str, int] = {}
    by_species: dict[str, int] = {}
    by_detection: dict[str, int] = {}
    for t in trees:
        by_leaf[t.leaf_type] = by_leaf.get(t.leaf_type, 0) + 1
        by_vitality[t.vitality] = by_vitality.get(t.vitality, 0) + 1
        by_species[t.species_hint] = by_species.get(t.species_hint, 0) + 1
        by_detection[t.detection_source] = by_detection.get(t.detection_source, 0) + 1

    # Q6: canopy gap fraction — share of AOI where an nDSM 2-10 m pixel is
    # NOT overtopped, i.e. where low stems are actually visible to a
    # first-return 1 m DSM. Plus a pixel-height layer profile.
    gap_frac = None
    layer_profile = None
    if ndsm is not None and ndsm.shape == canopy.shape:
        z = np.where(mask & np.isfinite(ndsm), ndsm, np.nan)
        valid = np.isfinite(z)
        nvalid = int(valid.sum())
        if nvalid:
            low_visible = valid & (z >= 2.0) & (z < 10.0)
            gap_frac = round(float(low_visible.sum()) / nvalid, 4)
            zz = z[valid & (z >= 0)]
            if zz.size:
                top = int(np.ceil(min(float(np.nanmax(zz)), 60.0) / 2.0) * 2)
                edges2 = np.arange(0, top + 2, 2)
                cnt, _ = np.histogram(np.clip(zz, 0, 60), bins=edges2)
                layer_profile = {f"{int(edges2[i])}-{int(edges2[i+1])}": int(c)
                                 for i, c in enumerate(cnt) if c}

    out = {
        "n_trees": len(trees),
        "n_trees_edge": len(trees) - len(non_edge),
        "area_ha_total": round(area_ha_total, 3),
        "area_ha_canopy": round(area_ha_canopy, 3),
        "stems_per_ha_canopy": round(len(non_edge) / area_ha_canopy, 1)
        if area_ha_canopy > 0 else 0,
        "stems_per_ha_total": round(len(non_edge) / area_ha_total, 1)
        if area_ha_total > 0 else 0,
        "stems_per_ha_note": "non-edge trees only; canopy = nDSM >= min_tree_height",
        "h_mean_m": round(float(np.mean(hs)), 2) if len(hs) else 0.0,
        "h_p50_m": _pct(50), "h_p90_m": _pct(90), "h_p95_m": _pct(95),
        "h_p99_m": _pct(99),
        "h_max_m": round(float(hs.max()), 2) if len(hs) else 0.0,
        "h_dom_m": round(h_dom, 2),
        "h_dom_basis": h_dom_basis,
        "h_dom_note": f"mean of the {min(n_dom, len(hs))} tallest stems "
                      f"(100/ha x {round(basis_ha, 2)} ha {h_dom_basis} area)",
        "h_top100_m": round(h_top100, 2),
        "h_top100_note": "mean of the tallest stem per 100 m grid cell "
                         "(moving-hectare Oberhöhe; robust for small stands)",
        "crown_area_total_sqm": round(crown_area_total, 1),
        "crown_cover_pct_canopy": round(100.0 * crown_area_total /
                                        (area_ha_canopy * 1e4), 1)
        if area_ha_canopy > 0 else 0,
        "crown_cover_pct_total": round(100.0 * crown_area_total / aoi_area_sqm, 1)
        if aoi_area_sqm > 0 else 0,
        "height_histogram_2m": hist,
        "by_leaf_type": by_leaf,
        "by_vitality": by_vitality,
        "by_species_hint": by_species,
        "species_hint_note": (
            "single-epoch RGB+NIR ortho hint only (relative within-AOI "
            "spectral position of conifers): spruce=dark, larch=bright "
            "fresh-green deciduous conifer, pine=intermediate. Confidence "
            "capped at 0.6 — treat as assumption-grade, not species ID."),
        "by_detection_source": by_detection,
        # FEEDBACK-3 §5.3: residual under-detection, self-reported
        "recall_model": {
            "canopy_area_ha": round(float(area_ha_canopy), 3),
            "crown_area_ha": round(float(crown_area_total) / 1e4, 3),
            "unassigned_canopy_frac": round(float(
                max(0.0, 1.0 - crown_area_total / (area_ha_canopy * 1e4))), 4)
            if area_ha_canopy > 0 else None,
        },
        "volume_m3_est_total": round(sum(t.volume_m3_est for t in trees), 1),
        "dbh_method": "heuristic_h_crown (generic spruce; verify locally)",
        "volume_method": VOLUME_METHOD,
        # Q6: machine-readable detectability caveat
        "understory_detectable": False,
        "detection_floor_note": (
            "suppressed/sub-canopy stems below the dominant layer are not "
            "detectable from 1 m first-return DSM; the histogram below "
            "~10 m is valid only in gaps and open stands"),
    }
    if gap_frac is not None:
        out["canopy_gap_fraction"] = gap_frac
        out["canopy_gap_fraction_note"] = (
            "share of valid AOI pixels whose nDSM is 2-10 m (low vegetation "
            "actually visible to first-return DSM)")
    if layer_profile is not None:
        out["layer_profile_2m"] = layer_profile
        out["layer_profile_note"] = (
            "nDSM PIXEL heights in 2 m bins (not stems) — first-return "
            "surface distribution; use for multi-layer/plenter structure "
            "tests instead of the stem histogram")
    return out


# ---------------------------------------------------------------------------
# Cross-epoch matching with raster veto
# ---------------------------------------------------------------------------

def _matched_rec(ta: Tree, tb: Tree, d: float, method: str,
                 growth_eps_m: float, match_radius_m: float,
                 max_a3: np.ndarray, max_b3: np.ndarray,
                 crown_overlap: float | None = None) -> dict:
    dh = tb.height_m - ta.height_m
    if dh >= growth_eps_m:
        status = "grown"
    elif dh <= -growth_eps_m:
        status = "shrunk"
    else:
        status = "stable"
    rec = {
        "tree_id": ta.tree_id, "tree_id_b": tb.tree_id,
        "status": status, "match_method": method,
        "height_a_m": ta.height_m, "height_b_m": tb.height_m,
        "height_change_m": round(dh, 2),
        "crown_area_a_sqm": ta.crown_area_sqm,
        "crown_area_b_sqm": tb.crown_area_sqm,
        "ndsm_max_a_m": round(float(max_a3[ta.apex_row, ta.apex_col]), 2),
        "ndsm_max_b_m": round(float(max_b3[tb.apex_row, tb.apex_col]), 2),
        "match_distance_m": round(d, 2),
        "match_confidence": round(max(0.1, 1.0 - d / max(match_radius_m, 1e-6)
                                      * 0.5 - abs(dh) / 30.0), 2),
        "apex_a": (ta.apex_e, ta.apex_n), "apex_b": (tb.apex_e, tb.apex_n),
        "_ta": ta, "_tb": tb,
    }
    if crown_overlap is not None:
        rec["crown_overlap_frac"] = round(float(crown_overlap), 2)
    return rec


def match_trees(
    trees_a: list[Tree],
    trees_b: list[Tree],
    ndsm_a: np.ndarray,
    ndsm_b: np.ndarray,
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
    felling_min_drop_m: float = DEFAULT_FELLING_MIN_DROP_M,
    growth_eps_m: float = DEFAULT_GROWTH_EPS_M,
    a: float = DEFAULT_CROWN_RADIUS_A,
    b: float = DEFAULT_CROWN_RADIUS_B,
    labels_a: np.ndarray | None = None,
    labels_b: np.ndarray | None = None,
) -> list[dict]:
    """Two-pass matching (apex, then crown overlap) + raster veto.

    Rasters must share the same grid. Returns per-tree change records.

    Pass 1: greedy nearest-neighbour apex matching (match_method='apex').
    Pass 2 (Q3): apex-unmatched trees are matched at CROWN level — if the
    epoch-a and epoch-b crown label masks overlap >= CROWN_OVERLAP_MIN_FRAC
    (30%) of the smaller
    crown, they are the same tree whose local maximum jumped to another
    branch (match_method='crown'). Requires labels_a/labels_b.
    Unmatched leftovers are sub-classified by raster evidence:
    felled / unmatched_a_partial_drop (crown break) /
    unmatched_a_canopy_intact (matching failure, NOT a loss) /
    new / unmatched_b_canopy_preexisting.
    """
    from scipy.spatial import cKDTree

    # 3 m-radius max nDSM (raster evidence, sampled at apices)
    fp = _disk(3)
    max_a3 = ndimage.maximum_filter(np.nan_to_num(ndsm_a, nan=0.0), footprint=fp)
    max_b3 = ndimage.maximum_filter(np.nan_to_num(ndsm_b, nan=0.0), footprint=fp)

    recs: list[dict] = []
    matched_a: set[int] = set()
    matched_b: set[int] = set()

    if trees_a and trees_b:
        pa = np.array([[t.apex_e, t.apex_n] for t in trees_a])
        pb = np.array([[t.apex_e, t.apex_n] for t in trees_b])
        tree_b_kd = cKDTree(pb)
        # candidate pairs within the largest plausible radius
        max_r = max(match_radius_m,
                    0.75 * (a + b * max((t.height_m for t in trees_a), default=0)))
        pairs = []
        for ia, t in enumerate(trees_a):
            r_allow = max(match_radius_m, 0.75 * (a + b * t.height_m))
            for ib in tree_b_kd.query_ball_point(pa[ia], min(r_allow, max_r)):
                d = float(np.hypot(*(pa[ia] - pb[ib])))
                pairs.append((d, ia, ib))
        pairs.sort()
        for d, ia, ib in pairs:
            if ia in matched_a or ib in matched_b:
                continue
            matched_a.add(ia); matched_b.add(ib)
            recs.append(_matched_rec(trees_a[ia], trees_b[ib], d, "apex",
                                     growth_eps_m, match_radius_m,
                                     max_a3, max_b3))

    # --- Pass 2 (Q3): crown-overlap matching for apex-unmatched trees ---
    if labels_a is not None and labels_b is not None \
            and labels_a.shape == labels_b.shape:
        rem_a = [ia for ia in range(len(trees_a)) if ia not in matched_a]
        rem_b_by_label = {trees_b[ib].label: ib for ib in range(len(trees_b))
                          if ib not in matched_b}
        if rem_a and rem_b_by_label:
            # overlap histogram: for every px, (label_a, label_b) joint counts
            la = labels_a.ravel(); lb = labels_b.ravel()
            both = (la > 0) & (lb > 0)
            if both.any():
                key = la[both].astype(np.int64) * (int(lb.max()) + 1) + lb[both]
                uniq, cnt = np.unique(key, return_counts=True)
                ov_a = uniq // (int(lb.max()) + 1)
                ov_b = uniq % (int(lb.max()) + 1)
                area_a = np.bincount(la, minlength=int(la.max()) + 1)
                area_b = np.bincount(lb, minlength=int(lb.max()) + 1)
                cand = []
                a_by_label = {trees_a[ia].label: ia for ia in rem_a}
                for j in range(len(uniq)):
                    ia = a_by_label.get(int(ov_a[j]))
                    ib = rem_b_by_label.get(int(ov_b[j]))
                    if ia is None or ib is None:
                        continue
                    smaller = min(area_a[int(ov_a[j])], area_b[int(ov_b[j])])
                    frac = cnt[j] / max(int(smaller), 1)
                    if frac >= CROWN_OVERLAP_MIN_FRAC:
                        cand.append((-frac, ia, ib))
                cand.sort()
                for negf, ia, ib in cand:
                    if ia in matched_a or ib in matched_b:
                        continue
                    matched_a.add(ia); matched_b.add(ib)
                    ta, tb = trees_a[ia], trees_b[ib]
                    d = float(np.hypot(ta.apex_e - tb.apex_e,
                                       ta.apex_n - tb.apex_n))
                    recs.append(_matched_rec(ta, tb, d, "crown",
                                             growth_eps_m, match_radius_m,
                                             max_a3, max_b3,
                                             crown_overlap=-negf))

    # Unmatched in a → felled ONLY with raster veto passing, else
    # sub-classified (Q3): canopy_intact / partial_drop / ambiguous
    for ia, ta in enumerate(trees_a):
        if ia in matched_a:
            continue
        evid_b = float(max_b3[ta.apex_row, ta.apex_col])
        drop = ta.height_m - evid_b
        if drop >= felling_min_drop_m and evid_b < 0.4 * ta.height_m:
            status = "felled"
        elif evid_b >= 0.7 * ta.height_m:
            # canopy still there (and often taller) → matching failure,
            # NOT a loss
            status = "unmatched_a_canopy_intact"
        elif evid_b >= 0.4 * ta.height_m or drop < felling_min_drop_m:
            # 30%..felling-threshold drop → crown break / snow / wind damage
            status = "unmatched_a_partial_drop"
        else:
            status = "unmatched_a_ambiguous"
        recs.append({
            "tree_id": ta.tree_id, "status": status,
            "match_method": "none",
            "height_a_m": ta.height_m, "height_b_m": None,
            "height_change_m": round(-drop, 2) if status == "felled" else None,
            "crown_area_a_sqm": ta.crown_area_sqm, "crown_area_b_sqm": None,
            "ndsm_max_a_m": round(float(max_a3[ta.apex_row, ta.apex_col]), 2),
            "ndsm_max_b_m": round(evid_b, 2),
            "match_distance_m": None, "match_confidence": None,
            "apex_a": (ta.apex_e, ta.apex_n), "apex_b": None,
            **({"volume_m3_est": ta.volume_m3_est} if status == "felled" else {}),
            "_ta": ta, "_tb": None,
        })

    # Unmatched in b → new ONLY if date-a raster shows no tall canopy there
    for ib, tb in enumerate(trees_b):
        if ib in matched_b:
            continue
        evid_a = float(max_a3[tb.apex_row, tb.apex_col])
        if evid_a < max(3.0, 0.3 * tb.height_m):
            status = "new"
        else:
            # canopy of comparable height already existed at date a →
            # apex churn, not a genuinely new tree
            status = "unmatched_b_canopy_preexisting"
        recs.append({
            "tree_id": tb.tree_id, "status": status,
            "match_method": "none",
            "height_a_m": None, "height_b_m": tb.height_m,
            "height_change_m": None,
            "crown_area_a_sqm": None, "crown_area_b_sqm": tb.crown_area_sqm,
            "ndsm_max_a_m": round(evid_a, 2),
            "ndsm_max_b_m": round(float(max_b3[tb.apex_row, tb.apex_col]), 2),
            "match_distance_m": None, "match_confidence": None,
            "apex_a": None, "apex_b": (tb.apex_e, tb.apex_n),
            "_ta": None, "_tb": tb,
        })
    return recs


def felling_patches(
    ndsm_a: np.ndarray,
    ndsm_b: np.ndarray,
    mask: np.ndarray,
    transform,
    min_drop_m: float = DEFAULT_FELLING_MIN_DROP_M,
    min_tree_height: float = DEFAULT_MIN_TREE_HEIGHT,
    min_patch_sqm: float = 25.0,
) -> list[dict]:
    """Segmentation-independent nDSM-drop patches (the honest product).

    Returns [{geometry_3035, area_sqm, drop_mean_m, drop_max_m, height_a_mean_m}].
    """
    from rasterio import features as rio_features
    from shapely.geometry import shape as shp_shape
    from shapely.ops import unary_union

    za = np.nan_to_num(ndsm_a, nan=0.0)
    zb = np.nan_to_num(ndsm_b, nan=0.0)
    drop = za - zb
    patch = mask & (drop >= min_drop_m) & (za >= min_tree_height)
    # despeckle 1-px noise
    patch = ndimage.binary_opening(patch, structure=np.ones((2, 2)))
    lab, n = ndimage.label(patch)
    if n == 0:
        return []
    px_area = abs(transform.a * transform.e)
    idx = np.arange(1, n + 1)
    areas = np.bincount(lab.ravel(), minlength=n + 1)[1:] * px_area
    keep = np.nonzero(areas >= min_patch_sqm)[0] + 1
    if not len(keep):
        return []
    drop_mean = ndimage.mean(drop, lab, keep)
    drop_max = ndimage.maximum(drop, lab, keep)
    ha_mean = ndimage.mean(za, lab, keep)

    geoms: dict[int, list] = {}
    keepmask = np.isin(lab, keep)
    for geom, val in rio_features.shapes(lab.astype(np.int32), mask=keepmask,
                                         transform=transform):
        geoms.setdefault(int(val), []).append(shp_shape(geom))
    out = []
    for j, labv in enumerate(keep):
        parts = geoms.get(int(labv), [])
        if not parts:
            continue
        g = parts[0] if len(parts) == 1 else unary_union(parts)
        g = g.simplify(0.5, preserve_topology=True)
        out.append({
            "geometry_3035": g,
            "area_sqm": round(float(areas[labv - 1]), 1),
            "drop_mean_m": round(float(drop_mean[j]), 2),
            "drop_max_m": round(float(drop_max[j]), 2),
            "height_a_mean_m": round(float(ha_mean[j]), 2),
        })
    out.sort(key=lambda p: -p["area_sqm"])
    return out


def params_hash(params: dict) -> str:
    """Deterministic hash of effective parameters for client-side caching."""
    canon = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(canon.encode()).hexdigest()[:12]
