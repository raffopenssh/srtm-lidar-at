"""Cross-peer registry of tile-checkpoint metadata pickles.

When a peer aborts a KG mid-flight (BEV throttle, Copernicus rotation,
bandwidth wall, role eviction, …) the partially-completed tile pickles
(``data/austria_processor/tile_checkpoints/<kg>/tile_*.pkl``) carry
the full per-tile result (segmentation, classified objects, vector
cadastre data, NDVI/SAR/Hansen extracts — *everything except the raw
DTM/DSM/ortho rasters*). They are small (KB–MB) and reusable by any
other peer picking up the same KG. Today they live only on the
aborting peer's local disk and are lost when it moves on.

This module wraps them as one gzipped tarball per KG, parks the tarball
in the shared Zenodo cache deposit, and exposes minimal upload /
download / delete operations. The same deposit already serves
``zenodo_cache`` tile bundles — we just live under a different prefix.

Raster sidecars (Phase 1) are intentionally NOT included — they would
blow the upload budget and the GPKG build step has its own re-read
fallback. This is the "metadata tier" only.

Local manifest at ``data/austria_processor/checkpoint_registry.json``
tracks ``{kg_code: {ts, n_tiles, bytes, name}}``. The manifest is added
to ``director_ha.SNAPSHOT_FILES`` so the shadow director sees the same
view after takeover.

Lifecycle hooks (see austria_processor.py):

* Upload — ``main()`` after a KG is deferred / re-queued.
* Download — ``process_one_kg()`` right after the checkpoint dir is
  created and BEFORE the tile loop runs.
* Delete — right after ``_clear_tile_checkpoints()`` on successful
  completion.

Bounded eviction: at most ``MAX_REGISTRY_KGS`` entries are kept (LRU
by ``ts``). Older entries are deleted from Zenodo when a new one is
added. The fleet rarely has > ~50 KGs in-flight, so the 200-entry
default is generous headroom.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import os
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

_log = logging.getLogger("chkpt_registry")

DATA_DIR = Path("data/austria_processor")
REGISTRY_MANIFEST = DATA_DIR / "checkpoint_registry.json"
CKPT_ROOT = DATA_DIR / "tile_checkpoints"

# Zenodo-side prefix (lives in the same deposit as the tile cache).
ZENODO_PREFIX = "chkpt"

# Hard cap on number of KGs in registry (LRU on ts). Plenty for our fleet.
MAX_REGISTRY_KGS = 200

# Cap per-KG bundle size. Metadata pickles should never exceed this;
# if they do, the KG has anomalous tile counts and we skip uploading.
MAX_BUNDLE_BYTES = 200 * 1024 * 1024  # 200 MB


# ----------------------------------------------------------------------
# Local manifest helpers
# ----------------------------------------------------------------------

def _load_manifest() -> Dict:
    if not REGISTRY_MANIFEST.exists():
        return {}
    try:
        return json.loads(REGISTRY_MANIFEST.read_text() or "{}")
    except Exception:
        return {}


def _save_manifest(m: Dict) -> None:
    REGISTRY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, sort_keys=True))
    tmp.rename(REGISTRY_MANIFEST)


def _bundle_name(kg_code: str) -> str:
    safe = "".join(c for c in kg_code if c.isalnum() or c in "-_")
    return f"{ZENODO_PREFIX}_{safe}.tar.gz"


# ----------------------------------------------------------------------
# Bundling
# ----------------------------------------------------------------------

def _bundle_tile_pickles(kg_code: str) -> Optional[bytes]:
    """Tar+gzip all tile_*.pkl files for kg_code into an in-memory blob.

    Returns None if there are no pickles to bundle or the bundle would
    exceed ``MAX_BUNDLE_BYTES``.
    """
    kg_dir = CKPT_ROOT / kg_code
    if not kg_dir.exists():
        return None
    pkls = sorted(p for p in kg_dir.iterdir()
                  if p.is_file() and p.name.startswith("tile_")
                  and p.name.endswith(".pkl"))
    if not pkls:
        return None
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            for p in pkls:
                tar.add(str(p), arcname=p.name)
    data = buf.getvalue()
    if len(data) > MAX_BUNDLE_BYTES:
        _log.warning("chkpt bundle for %s exceeds %d MB (%d MB) — skipping upload",
                     kg_code, MAX_BUNDLE_BYTES // (1024*1024), len(data) // (1024*1024))
        return None
    return data


def _unpack_bundle(blob: bytes, kg_code: str) -> int:
    """Extract tile_*.pkl entries into tile_checkpoints/<kg>/.

    Skips any tile pickle already present locally (don't clobber a
    freshly-written checkpoint). Returns count of files installed.
    """
    kg_dir = CKPT_ROOT / kg_code
    kg_dir.mkdir(parents=True, exist_ok=True)
    installed = 0
    try:
        buf = io.BytesIO(blob)
        with gzip.GzipFile(fileobj=buf, mode="rb") as gz:
            with tarfile.open(fileobj=gz, mode="r") as tar:
                for member in tar.getmembers():
                    name = os.path.basename(member.name)
                    if not (name.startswith("tile_") and name.endswith(".pkl")):
                        continue
                    dst = kg_dir / name
                    if dst.exists():
                        continue
                    fobj = tar.extractfile(member)
                    if fobj is None:
                        continue
                    tmp = kg_dir / (name + ".tmp")
                    with open(tmp, "wb") as f:
                        f.write(fobj.read())
                    tmp.rename(dst)
                    installed += 1
    except Exception as e:
        _log.warning("unpack chkpt bundle for %s failed: %s", kg_code, e)
    return installed


# ----------------------------------------------------------------------
# Zenodo upload/download (uses the cache deposit)
# ----------------------------------------------------------------------

def _get_cache() -> "object | None":
    try:
        from zenodo_cache import ZenodoCache
        return ZenodoCache()
    except Exception as e:
        _log.warning("chkpt_registry: cannot init ZenodoCache: %s", e)
        return None


def _upload_blob_to_deposit(cache, name: str, blob: bytes) -> bool:
    """Write blob to a temp file and upload via the cache deposit's bucket."""
    try:
        depo_id = cache._ensure_deposit()
    except Exception as e:
        _log.warning("chkpt_registry: ensure_deposit failed: %s", e)
        return False
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
        tf.write(blob)
        tmp_path = Path(tf.name)
    try:
        cache._upload_file(depo_id, tmp_path, name)
        return True
    except Exception as e:
        _log.warning("chkpt_registry: upload %s failed: %s", name, e)
        return False
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _delete_from_deposit(cache, name: str) -> bool:
    try:
        depo_id = cache._ensure_deposit()
        cache._delete_file(depo_id, name)
        return True
    except Exception as e:
        # 404 is common (already evicted) — quiet at debug.
        _log.info("chkpt_registry: delete %s skipped: %s", name, e)
        return False


def _download_from_deposit(cache, name: str) -> Optional[bytes]:
    """Stream a file out of the cache deposit. Returns bytes or None."""
    try:
        depo_id = cache._ensure_deposit()
        # Use deposition files API — returns the bucket download URL.
        r = cache._api("GET", f"/api/deposit/depositions/{depo_id}/files")
        for f in r.json():
            if f.get("filename") == name:
                links = f.get("links") or {}
                url = links.get("download") or links.get("self")
                if not url:
                    return None
                # Re-use cache's _session for keep-alive
                sess = cache._session
                rr = sess.get(url, params={"access_token": cache.token},
                              timeout=120)
                rr.raise_for_status()
                return rr.content
        return None
    except Exception as e:
        _log.info("chkpt_registry: download %s failed: %s", name, e)
        return None


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def upload_kg(kg_code: str) -> bool:
    """Bundle this KG's tile pickles and upload to Zenodo registry.

    Idempotent: if the manifest already records the KG and the local
    pickles haven't changed in count, skip the re-upload. Acquires the
    fleet zenodo upload lock for the duration of the PUT only.
    """
    blob = _bundle_tile_pickles(kg_code)
    if blob is None:
        return False
    n_tiles = sum(1 for p in (CKPT_ROOT / kg_code).iterdir()
                  if p.is_file() and p.name.startswith("tile_")
                  and p.name.endswith(".pkl"))
    m = _load_manifest()
    existing = m.get(kg_code)
    if existing and existing.get("n_tiles") == n_tiles \
            and existing.get("bytes") == len(blob):
        return True  # nothing to do
    cache = _get_cache()
    if cache is None:
        return False
    name = _bundle_name(kg_code)
    # Per-bundle fleet lock — keep critical section minimal.
    try:
        from zenodo_lock import zenodo_upload_lock
        with zenodo_upload_lock(purpose=f"chkpt_upload:{kg_code}"):
            ok = _upload_blob_to_deposit(cache, name, blob)
    except Exception as e:
        _log.warning("chkpt_registry: lock acquire for %s failed: %s", kg_code, e)
        return False
    if not ok:
        return False
    m[kg_code] = {
        "ts": time.time(),
        "n_tiles": n_tiles,
        "bytes": len(blob),
        "name": name,
    }
    # LRU eviction: keep at most MAX_REGISTRY_KGS entries.
    if len(m) > MAX_REGISTRY_KGS:
        ordered = sorted(m.items(), key=lambda kv: kv[1].get("ts", 0))
        n_drop = len(m) - MAX_REGISTRY_KGS
        for kg, _info in ordered[:n_drop]:
            try:
                _delete_from_deposit(cache, _bundle_name(kg))
            except Exception:
                pass
            m.pop(kg, None)
    _save_manifest(m)
    _log.info("chkpt_registry: uploaded %s (%d tiles, %.1f MB)",
              kg_code, n_tiles, len(blob) / 1e6)
    return True


def download_kg(kg_code: str) -> int:
    """If the registry has this KG, download + extract pickles locally.

    Returns the number of tile pickles installed (0 if not in registry
    or already present locally).
    """
    m = _load_manifest()
    if kg_code not in m:
        return 0
    cache = _get_cache()
    if cache is None:
        return 0
    blob = _download_from_deposit(cache, _bundle_name(kg_code))
    if not blob:
        return 0
    n = _unpack_bundle(blob, kg_code)
    if n:
        _log.info("chkpt_registry: restored %d tile pickles for %s from Zenodo",
                  n, kg_code)
    return n


def delete_kg(kg_code: str) -> bool:
    """Drop a KG from the registry (call on successful completion)."""
    m = _load_manifest()
    entry = m.pop(kg_code, None)
    if entry is None:
        return False
    _save_manifest(m)
    cache = _get_cache()
    if cache is None:
        return False
    try:
        from zenodo_lock import zenodo_upload_lock
        with zenodo_upload_lock(purpose=f"chkpt_delete:{kg_code}"):
            return _delete_from_deposit(cache, entry.get("name") or _bundle_name(kg_code))
    except Exception as e:
        _log.info("chkpt_registry: delete lock failed for %s: %s", kg_code, e)
        return False


def stats() -> Dict:
    """Return small dict for /process.txt: kgs, bytes_total, oldest_age_s."""
    m = _load_manifest()
    if not m:
        return {"kgs": 0, "bytes": 0, "oldest_age_s": 0}
    now = time.time()
    return {
        "kgs": len(m),
        "bytes": sum(int(e.get("bytes", 0)) for e in m.values()),
        "oldest_age_s": now - min(float(e.get("ts", now)) for e in m.values()),
    }
