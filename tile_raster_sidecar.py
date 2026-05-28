"""Tile raster sidecars — per-tile DTM/DSM/nDSM/Ortho stashed locally.

Written during the per-tile loop of ``process_one_kg`` so that
``build_full_gpkg_tiled`` does not have to re-read BEV for every tile.
BEV throttling is the dominant wall-time cost of ``gpkg_full`` today;
this turns 6+ BEV passes per tile (default-date DTM/DSM × 2, multi-date
DTM/DSM × 2 dates × 2 layers, ortho × up to 3 years) into a single
local mmap each.

Layout (one dir per tile, under each KG's existing tile_checkpoints dir):

    data/austria_processor/tile_checkpoints/<kg>/tile_<N>/raster/
        meta.json            {"transform": [...], "shape": [h,w],
                              "layers": {...}}
        dtm.npy              float32, default dataset
        dsm.npy              float32, default dataset
        ndsm.npy             float32, default dataset
        dtm_<dataset>.npy    float32, additional dataset (multi-date)
        dsm_<dataset>.npy    float32, additional dataset
        ortho_<year>_rgb.npy uint8 (3,H,W)
        ortho_<year>_nir.npy uint8 (H,W)   — only if NIR was available

All writes are atomic (``*.tmp`` + ``rename``). Loaders mmap_mode='r'
to avoid copying. ``persist_dtm_dsm`` / ``persist_ortho`` are no-ops
when free disk falls below ``SIDECAR_MIN_FREE_GB`` — the GPKG step
then falls back to the existing BEV reads (today's behavior).

Deletion is per-tile (``release_tile``) — called by ``build_full_gpkg_tiled``
as soon as each tile's rasters have been written into the GPKG — so the
on-disk peak stays bounded by ~3 tiles × ~250 MB ≈ 750 MB even on a
27-tile KG. The whole ``raster/`` dir is also torn down by
``release_kg`` once gpkg_full returns.

The metadata-tier pickles (``tile_<N>.pkl``) are unaffected: they stay
where they already are and continue to be managed by
``_save_tile_checkpoint`` / ``_clear_tile_checkpoints``.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_log = logging.getLogger("tile_raster_sidecar")

# Skip persisting sidecars when free disk is below this threshold (GB).
# Falls back to today's behavior (re-read BEV during gpkg_full).
#
# Must be **below** ``austria_processor.DISK_MIN_FREE_GB + 2`` (the
# post-cleanup target, currently 7 GB) so that disk_cleanup's target
# headroom leaves enough room for sidecars to persist. Per-tile peak
# is bounded by ~3 tiles × ~250 MB ≈ 750 MB (see ``release_tile``
# semantics above), so 4 GB leaves a comfortable margin and still keeps
# 2 GB above the system emergency floor.
SIDECAR_MIN_FREE_GB = float(os.environ.get("SIDECAR_MIN_FREE_GB", "4"))


def _enough_disk() -> bool:
    try:
        usage = shutil.disk_usage("/")
        return (usage.free / 1024 ** 3) >= SIDECAR_MIN_FREE_GB
    except Exception:
        return True  # fail-open — don't lose data because we couldn't stat


def _tile_raster_dir(ckpt_root: Path, kg_code: str, tile_idx: int) -> Path:
    return Path(ckpt_root) / kg_code / f"tile_{tile_idx}" / "raster"


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    # NB: np.save() auto-appends '.npy' if missing — use a file handle
    # so the exact tmp path is honored, then rename.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    tmp.rename(path)


def _write_meta(dir_path: Path, **fields) -> None:
    """Merge fields into meta.json (atomic)."""
    meta_path = dir_path / "meta.json"
    cur = {}
    if meta_path.exists():
        try:
            cur = json.loads(meta_path.read_text())
        except Exception:
            cur = {}
    for k, v in fields.items():
        if k == "layers":
            cur.setdefault("layers", {}).update(v)
        else:
            cur[k] = v
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur))
    tmp.rename(meta_path)


# ----------------------------------------------------------------------
# Persisters — called from the per-tile loop, idempotent, fail-soft
# ----------------------------------------------------------------------

def persist_dtm_dsm(ckpt_root: Path, kg_code: str, tile_idx: int,
                    tdata: dict, dataset: str = "default") -> None:
    """Persist DTM+DSM (+ndsm for default) for one tile.

    *dataset* is either ``"default"`` (writes dtm.npy/dsm.npy/ndsm.npy)
    or a date key like ``"20220128"`` (writes dtm_<key>.npy/dsm_<key>.npy).
    """
    if not _enough_disk():
        return
    try:
        d = _tile_raster_dir(ckpt_root, kg_code, tile_idx)
        d.mkdir(parents=True, exist_ok=True)
        if dataset == "default":
            _atomic_save_npy(d / "dtm.npy", np.asarray(tdata["dtm"], dtype=np.float32))
            _atomic_save_npy(d / "dsm.npy", np.asarray(tdata["dsm"], dtype=np.float32))
            _atomic_save_npy(d / "ndsm.npy", np.asarray(tdata["ndsm"], dtype=np.float32))
            tf = tdata["transform"]
            shape = tdata["shape"]
            _write_meta(d,
                        transform=[tf.a, tf.b, tf.c, tf.d, tf.e, tf.f],
                        shape=list(shape),
                        layers={"default": True})
        else:
            _atomic_save_npy(d / f"dtm_{dataset}.npy",
                             np.asarray(tdata["dtm"], dtype=np.float32))
            _atomic_save_npy(d / f"dsm_{dataset}.npy",
                             np.asarray(tdata["dsm"], dtype=np.float32))
            _write_meta(d, layers={f"dtm_dsm_{dataset}": True})
    except Exception as e:
        _log.warning("persist_dtm_dsm(%s,%s,%s) failed: %s",
                     kg_code, tile_idx, dataset, e)


def persist_ortho(ckpt_root: Path, kg_code: str, tile_idx: int,
                  rgb: Optional[np.ndarray], nir: Optional[np.ndarray],
                  year: int) -> None:
    """Persist ortho RGB(+NIR) for a year for one tile."""
    if not _enough_disk():
        return
    if rgb is None:
        return
    try:
        d = _tile_raster_dir(ckpt_root, kg_code, tile_idx)
        d.mkdir(parents=True, exist_ok=True)
        _atomic_save_npy(d / f"ortho_{year}_rgb.npy",
                         np.ascontiguousarray(rgb, dtype=np.uint8))
        if nir is not None:
            _atomic_save_npy(d / f"ortho_{year}_nir.npy",
                             np.ascontiguousarray(nir, dtype=np.uint8))
        _write_meta(d, layers={f"ortho_{year}": ("rgb+nir" if nir is not None else "rgb")})
    except Exception as e:
        _log.warning("persist_ortho(%s,%s,%s) failed: %s",
                     kg_code, tile_idx, year, e)


# ----------------------------------------------------------------------
# Loaders — sidecar-first, return None on miss (caller falls back to BEV)
# ----------------------------------------------------------------------

def load_dtm_dsm(ckpt_root: Path, kg_code: str, tile_idx: int,
                 dataset: str = "default") -> Optional[dict]:
    """Return dict shaped like ``raster_io.read_dtm_dsm`` output, or None.

    For ``dataset == "default"`` returns full dict with ndsm + mask.
    For other datasets, only ``dtm`` and ``dsm`` are filled (multi-date
    GPKG layers do not need ndsm/mask). Arrays are mmap'd read-only.
    """
    d = _tile_raster_dir(ckpt_root, kg_code, tile_idx)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if dataset == "default":
            if not meta.get("layers", {}).get("default"):
                return None
            dtm = np.load(str(d / "dtm.npy"), mmap_mode="r")
            dsm = np.load(str(d / "dsm.npy"), mmap_mode="r")
            ndsm = np.load(str(d / "ndsm.npy"), mmap_mode="r")
            from rasterio.transform import Affine
            tf_vals = meta.get("transform")
            tf = Affine(*tf_vals) if tf_vals else None
            return {
                "dtm": dtm, "dsm": dsm, "ndsm": ndsm,
                "mask": np.ones(dtm.shape, dtype=bool),  # mask was already applied at write time
                "transform": tf, "crs": None,
                "shape": tuple(meta.get("shape", dtm.shape)),
            }
        if not meta.get("layers", {}).get(f"dtm_dsm_{dataset}"):
            return None
        dtm = np.load(str(d / f"dtm_{dataset}.npy"), mmap_mode="r")
        dsm = np.load(str(d / f"dsm_{dataset}.npy"), mmap_mode="r")
        return {"dtm": dtm, "dsm": dsm, "shape": dtm.shape}
    except Exception as e:
        _log.warning("load_dtm_dsm(%s,%s,%s) failed: %s",
                     kg_code, tile_idx, dataset, e)
        return None


def load_ortho(ckpt_root: Path, kg_code: str, tile_idx: int,
               year: int) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Return (rgb, nir|None), or None if no sidecar for this (tile,year)."""
    d = _tile_raster_dir(ckpt_root, kg_code, tile_idx)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        kind = meta.get("layers", {}).get(f"ortho_{year}")
        if not kind:
            return None
        rgb = np.load(str(d / f"ortho_{year}_rgb.npy"), mmap_mode="r")
        nir = None
        if kind == "rgb+nir":
            nir = np.load(str(d / f"ortho_{year}_nir.npy"), mmap_mode="r")
        return rgb, nir
    except Exception as e:
        _log.warning("load_ortho(%s,%s,%s) failed: %s",
                     kg_code, tile_idx, year, e)
        return None


# ----------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------

def release_tile(ckpt_root: Path, kg_code: str, tile_idx: int) -> None:
    """Delete a single tile's raster sidecars."""
    d = _tile_raster_dir(ckpt_root, kg_code, tile_idx)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def release_kg(ckpt_root: Path, kg_code: str) -> None:
    """Delete all raster sidecars for a KG (leaves metadata pickles alone)."""
    base = Path(ckpt_root) / kg_code
    if not base.exists():
        return
    for sub in base.iterdir():
        if sub.is_dir() and sub.name.startswith("tile_"):
            rdir = sub / "raster"
            if rdir.exists():
                shutil.rmtree(rdir, ignore_errors=True)
            # if tile_<N>/ is now empty (no pickle either) drop it
            try:
                if not any(sub.iterdir()):
                    sub.rmdir()
            except OSError:
                pass


def total_bytes(ckpt_root: Path) -> int:
    """Best-effort total bytes of all raster sidecars under ckpt_root."""
    total = 0
    p = Path(ckpt_root)
    if not p.exists():
        return 0
    for r, _, files in os.walk(p):
        if Path(r).name != "raster":
            continue
        for f in files:
            try:
                total += (Path(r) / f).stat().st_size
            except OSError:
                pass
    return total
