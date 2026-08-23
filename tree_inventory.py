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
TREE_ALGO_VERSION = "2.0.0"

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variable-window local maxima on the smoothed nDSM.

    Returns (apex_bool, smooth, canopy_bool). Window radius per pixel is
    r(h) = a + b*h, quantised to integer pixel radii (1 m grid).
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

    # Collapse plateaus (adjacent apex pixels) to a single representative px.
    lab, n = ndimage.label(apex)
    if n:
        pts = ndimage.maximum_position(smooth, lab, np.arange(1, n + 1))
        apex = np.zeros_like(apex)
        rr = np.array([p[0] for p in pts]); cc = np.array([p[1] for p in pts])
        apex[rr, cc] = True
    return apex, smooth, canopy


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
    vitality: str = "unknown"
    vitality_conf: float = 0.0
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
    spectral: dict | None = None,
    nir: np.ndarray | None = None,
) -> tuple[list[Tree], np.ndarray, np.ndarray]:
    """Full apex-based inventory. Returns (trees, labels, canopy_mask)."""
    apex, smooth, canopy = detect_apices(ndsm, mask, min_height, a, b, smooth_sigma)
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

        # leaf type: crown steepness + red/green index
        steep = (h - float(hmean[i])) / max(r_mean, 0.5)
        rg = spec.get("rg_index_mean", 0.0)
        if steep > 2.2 or (steep > 1.5 and rg < -0.02):
            leaf, conf = "coniferous", min(0.9, 0.5 + 0.1 * steep)
        elif steep < 1.2 and area > 20:
            leaf, conf = "broadleaf", 0.5
        else:
            leaf, conf = "coniferous", 0.4  # Austrian prior

        # vitality (needs real NDVI from NIR to be trustworthy)
        vit, vconf = "unknown", 0.0
        if has_real_ndvi:
            nd_mean = spec.get("ndvi_mean", 0.0)
            nd_p10 = spec.get("ndvi_p10", nd_mean)
            if nd_mean < 0.15 and nd_p10 < 0.10 and h >= 5.0:
                vit, vconf = "dead", min(0.95, 0.6 + (0.15 - nd_mean) * 2)
                leaf, conf = "dead", vconf
            elif nd_mean < 0.35:
                vit, vconf = "stressed", 0.5
            else:
                vit, vconf = "vital", min(0.9, nd_mean)

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
            vitality=vit, vitality_conf=round(vconf, 2),
            spectral=spec,
        ))
    return trees, labels, canopy


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(trees: list[Tree], canopy: np.ndarray, mask: np.ndarray,
              transform, aoi_area_sqm: float) -> dict:
    """Explicit, self-consistent summary block (denominators included)."""
    px_area = abs(transform.a * transform.e)
    area_ha_total = aoi_area_sqm / 1e4
    area_ha_canopy = float(canopy.sum()) * px_area / 1e4
    non_edge = [t for t in trees if not t.is_edge]
    hs = np.array([t.height_m for t in trees], dtype=np.float32)
    crown_area_total = float(sum(t.crown_area_sqm for t in trees))

    # h_dom: mean of the 100 tallest stems per ha, computed over TOTAL area
    n_dom = max(1, int(round(100 * area_ha_total)))
    hs_sorted = np.sort(hs)[::-1]
    h_dom = float(np.mean(hs_sorted[:min(n_dom, len(hs_sorted))])) if len(hs) else 0.0

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
    for t in trees:
        by_leaf[t.leaf_type] = by_leaf.get(t.leaf_type, 0) + 1
        by_vitality[t.vitality] = by_vitality.get(t.vitality, 0) + 1

    return {
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
        "h_max_m": round(float(hs.max()), 2) if len(hs) else 0.0,
        "h_dom_m": round(h_dom, 2),
        "h_dom_note": f"mean of the {min(n_dom, len(hs))} tallest stems "
                      f"(100/ha x {round(area_ha_total, 2)} ha total area)",
        "crown_area_total_sqm": round(crown_area_total, 1),
        "crown_cover_pct_canopy": round(100.0 * crown_area_total /
                                        (area_ha_canopy * 1e4), 1)
        if area_ha_canopy > 0 else 0,
        "crown_cover_pct_total": round(100.0 * crown_area_total / aoi_area_sqm, 1)
        if aoi_area_sqm > 0 else 0,
        "height_histogram_2m": hist,
        "by_leaf_type": by_leaf,
        "by_vitality": by_vitality,
        "volume_m3_est_total": round(sum(t.volume_m3_est for t in trees), 1),
        "dbh_method": "heuristic_h_crown (generic spruce; verify locally)",
    }


# ---------------------------------------------------------------------------
# Cross-epoch matching with raster veto
# ---------------------------------------------------------------------------

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
) -> list[dict]:
    """Greedy nearest-neighbour apex matching + raster veto for felled/new.

    Rasters must share the same grid. Returns per-tree change records.
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
            ta, tb = trees_a[ia], trees_b[ib]
            dh = tb.height_m - ta.height_m
            if dh >= growth_eps_m:
                status = "grown"
            elif dh <= -growth_eps_m:
                status = "shrunk"
            else:
                status = "stable"
            recs.append({
                "tree_id": ta.tree_id, "tree_id_b": tb.tree_id,
                "status": status,
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
            })

    # Unmatched in a → felled ONLY with raster veto passing, else unmatched_a
    for ia, ta in enumerate(trees_a):
        if ia in matched_a:
            continue
        evid_b = float(max_b3[ta.apex_row, ta.apex_col])
        drop = ta.height_m - evid_b
        if drop >= felling_min_drop_m and evid_b < 0.4 * ta.height_m:
            status = "felled"
        else:
            status = "unmatched_a"
        recs.append({
            "tree_id": ta.tree_id, "status": status,
            "height_a_m": ta.height_m, "height_b_m": None,
            "height_change_m": round(-drop, 2) if status == "felled" else None,
            "crown_area_a_sqm": ta.crown_area_sqm, "crown_area_b_sqm": None,
            "ndsm_max_a_m": round(float(max_a3[ta.apex_row, ta.apex_col]), 2),
            "ndsm_max_b_m": round(evid_b, 2),
            "match_distance_m": None, "match_confidence": None,
            "apex_a": (ta.apex_e, ta.apex_n), "apex_b": None,
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
            status = "unmatched_b"
        recs.append({
            "tree_id": tb.tree_id, "status": status,
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
