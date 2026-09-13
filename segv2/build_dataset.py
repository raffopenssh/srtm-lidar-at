"""Build the segv2 training dataset, one parquet per KG.

For each KG code (plain or split block):
  1. fetch + audit the Zenodo *full* GPKG (parallel range download)
  2. fetch vector truth (cadastre footprints/landuse/parcels, OSM, INVEKOS)
  3. KG mask = union of cadastre parcels (rasterised) — every step is masked
     to it and tiles with < 2 % KG coverage are skipped
  4. per 1500 px tile: pixel layers → v1-parameter segmentation → vectorised
     features → label raster → per-segment label (mode) + purity + source
  5. write ``data/segv2/dataset/<code>.parquet`` (+ ``.meta.json``), delete GPKG

Columns: label_id (segment), kg, tile, all feature keys, ``y`` (type name or
''), ``y_purity``, ``y_src``, ``y_weight``, ``v1_type`` (mode of the v1
``segment_type`` raster within the segment — the deployed pipeline's answer
on the same pixels), ``centroid_e/n``.
"""
from __future__ import annotations

import argparse
import pathlib
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio import features as rfeatures
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gpkg_fetch  # noqa: E402
import labels as L2  # noqa: E402
import features as F2  # noqa: E402
from gpkg_raster import FullGpkg  # noqa: E402

log = logging.getLogger("segv2.build")
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data/segv2/dataset"
TILE_PX = 1500
MIN_COVER = 0.02
MIN_PURITY = 0.6

_V1_ID_TYPE = None


def _v1_id_type():
    global _V1_ID_TYPE
    if _V1_ID_TYPE is None:
        from object_segmentation import OBJECT_TYPES
        _V1_ID_TYPE = {v: k for k, v in OBJECT_TYPES.items()}
    return _V1_ID_TYPE


def _context_rasters(ctx: L2.LabelContext, transform, shape_hw) -> dict:
    def burn(geoms):
        geoms = [g for g in geoms if not g.is_empty]
        if not geoms:
            return np.zeros(shape_hw, bool)
        return rfeatures.rasterize([(g, 1) for g in geoms], out_shape=shape_hw, transform=transform,
                                   fill=0, dtype=np.uint8).astype(bool)
    edges = [g.boundary for g in ctx.cad["parcels"]]
    return {
        "road": burn([g for g, _ in ctx.osm["road"] if L2.OSM_ROAD.get(_, ("road",))[0] == "road"]),
        "path": burn([g for g, _ in ctx.osm["road"] if L2.OSM_ROAD.get(_, ("road",))[0] == "path"]),
        "rail": burn([g for g, _ in ctx.osm["rail"]]),
        "water": burn(ctx.water_polys),
        "building": burn(ctx.cad["footprints"]),
        "parcel_edge": burn(edges),
    }


def _osm_edges(ctx: L2.LabelContext, transform, shape_hw) -> np.ndarray:
    lines = [g for g, _ in ctx.osm["road"]] + [g for g, _ in ctx.osm["rail"]] + \
            [g for g, _ in ctx.osm["water_line"]] + [g.boundary for g in ctx.water_polys]
    lines = [g for g in lines if not g.is_empty]
    if not lines:
        return np.zeros(shape_hw, bool)
    return rfeatures.rasterize([(g, 1) for g in lines], out_shape=shape_hw, transform=transform,
                               fill=0, dtype=np.uint8, all_touched=True).astype(bool)


def build_kg(code: str, *, keep_gpkg=False, v2_edges=False, cop_cache=None, obs_year=2024) -> dict:
    t0 = time.time()
    out_path = OUT_DIR / f"{code}.parquet"
    meta: dict = {"code": code, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = gpkg_fetch.fetch(code, "full_gpkg")
    if path is None:
        meta["error"] = "no_full_gpkg"
        return meta
    meta["fetch_s"] = round(time.time() - t0, 1)
    try:
        g = FullGpkg(path)
        parent = code.split("-")[0]
        b = g.bounds
        ctx = L2.LabelContext(parent, (b.left, b.bottom, b.right, b.top))
        if not ctx.cad["parcels"]:
            meta["error"] = "no_parcels"
            return meta
        kg_mask_full = rfeatures.rasterize([(p, 1) for p in ctx.cad["parcels"]], out_shape=g.shape,
                                           transform=g.transform, fill=0, dtype=np.uint8).astype(bool)
        meta["kg_cover_frac"] = round(float(kg_mask_full.mean()), 3)
        # --- metadata gate: real NIR + resolvable flight blocks, else skip loudly ---
        import acquisition as ACQ
        acq = ACQ.audit_kg((b.left, b.bottom, b.right, b.top))
        meta["acquisition"] = acq
        meta["nir_years"] = g.nir_years()
        meta["ortho_years"] = g.ortho_years()
        if not g.nir_years():
            meta["error"] = "no_real_nir"
            return meta
        if not acq["ok"]:
            meta["error"] = "als_flight_blocks_unknown"
            return meta
        v1_full = g.read_segment_type()
        frames = []
        tiles_done = tiles_skipped = 0
        lab_summary = {"by_type": {}, "by_source": {}}
        for ti, win in enumerate(g.windows(TILE_PX)):
            r0, c0 = int(win.row_off), int(win.col_off)
            kmask = kg_mask_full[r0:r0 + int(win.height), c0:c0 + int(win.width)]
            if kmask.mean() < MIN_COVER:
                tiles_skipped += 1
                continue
            L = F2.pixel_layers(g, win, cop_cache=cop_cache, obs_year=obs_year)
            if L is None:
                tiles_skipped += 1
                continue
            tf, shp = L["transform"], L["mask"].shape
            edges = _osm_edges(ctx, tf, shp) if v2_edges else None
            seg = F2.segment(L, kmask, osm_edges=edges)
            if seg.max() == 0:
                tiles_skipped += 1
                continue
            ctx_r = _context_rasters(ctx, tf, shp)
            df = F2.extract(seg, L, context=ctx_r, obs_year=obs_year)
            if df.empty:
                continue
            ndvi = L["spectral"].get("ndvi") if L["spectral"] else None
            assert ndvi is None or not np.all(np.nan_to_num(L["spectral"]["nir"], nan=255) >= 254), \
                "NIR is a constant alpha plane — writer contract violated"
            dsm_age = None
            newest = max(L["als_years"]) if L.get("als_years") else None
            if newest is not None:
                yr = L["als_years"][newest]["dsm_year"].astype(np.float32)
                dsm_age = np.where(yr > 0, obs_year - yr, np.nan).astype(np.float32)
            lab, src, wgt = ctx.rasterize(tf, shp, L["ndsm"], ndvi, dsm_age=dsm_age)
            s = L2.summarize(lab, src)
            for k in ("by_type", "by_source"):
                for kk, v in s[k].items():
                    lab_summary[k][kk] = lab_summary[k].get(kk, 0) + v
            G = F2.Grouped(seg, L["mask"] & kmask)
            mode, purity, labfrac = G.mode_frac(lab, 64)
            smode, _, _ = G.mode_frac(src, 16)
            wmean = G.mean(wgt)
            v1m, v1p, _ = G.mode_frac(v1_full[r0:r0 + shp[0], c0:c0 + shp[1]], 256)
            m = pd.DataFrame({"label": G.ids,
                              "y": [L2.ID_TYPE.get(int(x), "") for x in mode],
                              "y_purity": purity * labfrac, "y_labfrac": labfrac,
                              "y_src": [L2.SOURCES[int(x) - 1] if x > 0 else "" for x in smode],
                              "y_weight": wmean,
                              "v1_type": [_v1_id_type().get(int(x), "") for x in v1m],
                              "v1_purity": v1p})
            df = df.merge(m, on="label", how="left")
            df["kg"] = code
            df["tile"] = ti
            df["kg_frac"] = G.frac(kmask)
            frames.append(df)
            tiles_done += 1
            log.info("KG %s tile %d: %d segments, labelled %.0f%%", code, ti, len(df),
                     100 * (df["y"] != "").mean())
        if not frames:
            meta["error"] = "no_tiles"
            return meta
        out = pd.concat(frames, ignore_index=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
        meta.update({"n_segments": int(len(out)), "n_labelled": int((out["y"] != "").sum()),
                     "n_good": int(((out["y"] != "") & (out["y_purity"] >= MIN_PURITY)).sum()),
                     "tiles_done": tiles_done, "tiles_skipped": tiles_skipped,
                     "labels": lab_summary, "ortho_year": int(out["ortho_year"].iloc[0]),
                     "nir_year": int(out["nir_year"].iloc[0]),
                     "dsm_year_median": float(out["dsm_year"].median()),
                     "years_span_median": float(out["years_span"].median()),
                     "dtm_years": g.years("DTM"), "layers": sorted(g.raster_layers),
                     "seconds": round(time.time() - t0, 1)})
        (OUT_DIR / f"{code}.meta.json").write_text(json.dumps(meta))
        log.info("KG %s done: %d segs, %d labelled (%d good) in %.0fs", code, meta["n_segments"],
                 meta["n_labelled"], meta["n_good"], meta["seconds"])
    except Exception as e:  # noqa: BLE001
        log.exception("KG %s failed", code)
        meta["error"] = str(e)
    finally:
        if not keep_gpkg:
            gpkg_fetch.release(code, "full_gpkg")
    return meta


def pick_codes(n: int, seed: int = 0, max_mb: int = 900, min_mb: int = 150) -> list[str]:
    """Stratified random sample of processed codes: by manifest size band and
    lat/lon cell (from kg_list.json) so all of Austria is represented."""
    m = gpkg_fetch._manifest()
    kgl = {k["kg_code"]: k for k in json.load(open(ROOT / "data/austria_processor/kg_list.json"))} \
        if (ROOT / "data/austria_processor/kg_list.json").exists() else {}
    cands = []
    for k, e in m.items():
        if not k.endswith("_full_gpkg"):
            continue
        code = k[: -len("_full_gpkg")]
        mb = e["size"] / 1e6
        if not (min_mb <= mb <= max_mb):
            continue
        if f"{code}_json" not in m:
            continue
        kg = kgl.get(code.split("-")[0], {})
        lat, lon = kg.get("lat") or kg.get("centroid_lat"), kg.get("lon") or kg.get("centroid_lon")
        if lat is None and kg.get("bbox"):
            bb = kg["bbox"]
            lat, lon = (bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2
        cell = (round(lat or 0, 0), round((lon or 0) / 2) * 2)
        cands.append((cell, code))
    rnd = random.Random(seed)
    rnd.shuffle(cands)
    by_cell: dict = {}
    for cell, code in cands:
        by_cell.setdefault(cell, []).append(code)
    out = []
    while len(out) < n and any(by_cell.values()):
        for cell in sorted(by_cell):
            if by_cell[cell] and len(out) < n:
                out.append(by_cell[cell].pop())
    return out


def pick_alpine(n: int, seed: int = 0, max_mb: int = 3000, min_mb: int = 150) -> list[str]:
    """Pick N codes from sparsely built, large KGs in the alpine states (proxy for
    high elevation: kg_list has no terrain). One block per parent KG, parents
    already in the dataset excluded. Used to lift rock / bare_soil / earthwork /
    alpine-shrub support, which the size×cell stratification under-samples."""
    m = gpkg_fetch._manifest()
    kgl = {k["kg_code"]: k for k in json.load(open(ROOT / "data/austria_processor/kg_list.json"))}
    done = {f.name.split(".")[0].split("-")[0] for f in OUT_DIR.glob("*.parquet")}
    alpine = {"Tirol", "Salzburg", "Vorarlberg", "Kärnten"}
    by_parent: dict[str, list] = {}
    density: dict[str, float] = {}
    for k, e in m.items():
        if not k.endswith("_full_gpkg"):
            continue
        code = k[: -len("_full_gpkg")]
        parent = code.split("-")[0]
        if parent in done or f"{code}_json" not in m or not (min_mb <= e["size"] / 1e6 <= max_mb):
            continue
        kg = kgl.get(parent, {})
        area_km2 = (kg.get("total_area_sqm") or 0) / 1e6
        if area_km2 < 8 or kg.get("state_name") not in alpine:
            continue
        dens = (kg.get("building_count") or 0) / area_km2
        if dens > 4:
            continue
        by_parent.setdefault(parent, []).append(code)
        density[parent] = dens
    rnd = random.Random(seed)
    # lowest building density first (≈ highest / wildest terrain); random block per parent
    parents = sorted(by_parent, key=lambda p: (density[p], p))
    return [rnd.choice(by_parent[p]) for p in parents[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*")
    ap.add_argument("--n", type=int, default=0, help="pick N stratified codes")
    ap.add_argument("--alpine", type=int, default=0, help="pick N sparse alpine-state codes (rock/bare_soil support)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v2-edges", action="store_true")
    ap.add_argument("--keep-gpkg", action="store_true")
    ap.add_argument("--skip-done", action="store_true", default=True)
    ap.add_argument("--out", default="", help="output dir (default data/segv2/dataset)")
    args = ap.parse_args()
    global OUT_DIR
    if args.out:
        OUT_DIR = pathlib.Path(args.out)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    codes = list(args.codes) or (pick_alpine(args.alpine, args.seed) if args.alpine else pick_codes(args.n, args.seed))
    import tile_cache
    tile_cache.set_forbid_remote(True)
    cc = tile_cache.CopernicusTileCache()
    log.info("building %d KGs", len(codes))
    for code in codes:
        if args.skip_done and (OUT_DIR / f"{code}.parquet").exists():
            continue
        # unattended-run guards: stop before we starve gunicorn/director on the primary
        import shutil, gc
        free_gb = shutil.disk_usage("/tmp").free / 1e9
        if free_gb < 6:
            log.error("only %.1f GB free on /tmp — stopping", free_gb)
            break
        meta = build_kg(code, keep_gpkg=args.keep_gpkg, v2_edges=args.v2_edges, cop_cache=cc)
        gc.collect()
        with open(OUT_DIR.parent / "build_log.jsonl", "a") as f:
            f.write(json.dumps(meta) + "\n")


if __name__ == "__main__":
    main()
