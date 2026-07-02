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

# Cross-fleet visibility: mirror chkpt entries into cache_manifest.json
# (the same deposit hosts both tile-cache ZIPs and chkpt tarballs, and
# cache_manifest.json is already synced primary<->all peers every 5 min
# by app._sync_peer_data). Entries are tagged with the CHKPT_PREFIX
# filename so the primary's /process.txt + dashboard can derive a
# fleet-wide chkpt registry view without any extra network round-trips.
CACHE_MANIFEST = DATA_DIR / "cache_manifest.json"

# Zenodo-side prefix (lives in the same deposit as the tile cache).
ZENODO_PREFIX = "chkpt"

# Hard cap on number of KGs in registry (LRU on ts). The Zenodo deposit
# has a HARD 100-file limit shared with the tile-cache ZIPs (~45 files
# and growing as the cell layout fills in). Letting chkpt bundles crowd
# the deposit caused the Jul 2026 incident where a tile-cache ZIP was
# deleted-then-400'd on re-upload (deposit full) and its tiles were
# lost. Keep chkpt well under half the deposit.
MAX_REGISTRY_KGS = 30

# Never upload a NEW chkpt bundle when the deposit is already this full
# (Zenodo hard limit = 100 files). Leaves headroom for tile-cache ZIPs,
# whose delete-then-PUT re-upload cycle is NOT safe against a full
# deposit (the delete succeeds, the PUT 400s, the data is gone).
DEPOSIT_FILE_HEADROOM = 90

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


def _mirror_to_cache_manifest(kg_code: str, name: str, size: int,
                              n_tiles: int, deleted: bool = False) -> None:
    """Mirror a chkpt entry into cache_manifest.json.

    Hooks into the existing primary<->peers cache_manifest sync (5-min
    cadence in app._sync_peer_data). Deletes are propagated as
    size=0/tile_count=0 tombstones — the sync's last-writer-wins merge
    on updated_at handles eviction naturally. The dashboard /process.txt
    filters tombstones (size==0).
    """
    try:
        from zenodo_cache import CacheManifest
        cm = CacheManifest()
        now_iso = (
            __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat())
        cm.set_file(
            zip_name=name,
            url="",  # chkpt entries are not consumed via URL on peers
            size=0 if deleted else int(size),
            checksum="",
            tile_count=0 if deleted else int(n_tiles),
            updated_at=now_iso,
        )
        cm.save()
    except Exception as e:  # never block upload/download on mirror failure
        _log.debug("chkpt: mirror to cache_manifest skipped for %s: %s",
                   kg_code, e)


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

def upload_kg(kg_code: str) -> Optional[Dict]:
    """Bundle this KG's tile pickles and upload to Zenodo registry.

    Idempotent: if the manifest already records the KG and the local
    pickles haven't changed in count, skip the re-upload. Acquires the
    fleet zenodo upload lock for the duration of the PUT only.

    Returns ``{ok, n_tiles, bytes, name, skipped}`` on success/skip,
    or ``None`` if nothing was bundled / upload failed.
    """
    blob = _bundle_tile_pickles(kg_code)
    if blob is None:
        return None
    n_tiles = sum(1 for p in (CKPT_ROOT / kg_code).iterdir()
                  if p.is_file() and p.name.startswith("tile_")
                  and p.name.endswith(".pkl"))
    m = _load_manifest()
    existing = m.get(kg_code)
    name = _bundle_name(kg_code)
    if existing and existing.get("n_tiles") == n_tiles \
            and existing.get("bytes") == len(blob):
        # Already on Zenodo; refresh cache_manifest mirror so the
        # primary sees it even if its sync missed our last upload tick.
        _mirror_to_cache_manifest(kg_code, name, len(blob), n_tiles)
        return {"ok": True, "n_tiles": n_tiles, "bytes": len(blob),
                "name": name, "skipped": True}
    cache = _get_cache()
    if cache is None:
        return None
    # Deposit-fullness guard: the shared Zenodo deposit has a hard
    # 100-file cap. If the (fleet-synced) cache manifest shows we're
    # near it, refuse NEW bundles — a full deposit breaks the tile
    # cache's delete-then-PUT re-upload cycle (data loss, Jul 2026).
    # Re-uploads of an existing bundle are fine (net file count 0).
    if not existing:
        try:
            from zenodo_cache import CacheManifest
            cm = CacheManifest()
            live = sum(1 for _n, _e in (cm.all_files() or {}).items()
                       if (_e or {}).get("size", 0) > 0)
            if live >= DEPOSIT_FILE_HEADROOM:
                _log.warning(
                    "chkpt_registry: deposit near file cap (%d >= %d); "
                    "skipping upload of %s", live, DEPOSIT_FILE_HEADROOM,
                    name)
                return None
        except Exception:
            pass
    # Per-bundle fleet lock — keep critical section minimal.
    try:
        from zenodo_lock import zenodo_upload_lock
        with zenodo_upload_lock(purpose=f"chkpt_upload:{kg_code}"):
            ok = _upload_blob_to_deposit(cache, name, blob)
    except Exception as e:
        _log.warning("chkpt_registry: lock acquire for %s failed: %s", kg_code, e)
        return None
    if not ok:
        return None
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
    _mirror_to_cache_manifest(kg_code, name, len(blob), n_tiles)
    _log.info("chkpt_registry: uploaded %s (%d tiles, %.1f MB)",
              kg_code, n_tiles, len(blob) / 1e6)
    return {"ok": True, "n_tiles": n_tiles, "bytes": len(blob),
            "name": name, "skipped": False}


def download_kg(kg_code: str) -> Dict:
    """If the registry has this KG, download + extract pickles locally.

    Returns ``{n, bytes}`` (n=tile pickles installed, bytes=bundle size).
    Falls back to the fleet-wide cache_manifest mirror when the local
    chkpt manifest is empty — a fresh peer never authored an upload
    so its local registry is empty, but the cache_manifest sync gives
    it the full fleet view of available bundles.
    """
    m = _load_manifest()
    name = _bundle_name(kg_code)
    have_locally = kg_code in m
    have_in_mirror = False
    if not have_locally:
        try:
            from zenodo_cache import CacheManifest
            mirror = CacheManifest().get_file(name)
            have_in_mirror = bool(mirror and mirror.get("size"))
        except Exception:
            have_in_mirror = False
    if not (have_locally or have_in_mirror):
        return {"n": 0, "bytes": 0}
    cache = _get_cache()
    if cache is None:
        return {"n": 0, "bytes": 0}
    blob = _download_from_deposit(cache, name)
    if not blob:
        return {"n": 0, "bytes": 0}
    n = _unpack_bundle(blob, kg_code)
    if n:
        _log.info("chkpt_registry: restored %d tile pickles for %s from Zenodo",
                  n, kg_code)
    return {"n": n, "bytes": len(blob)}


def delete_kg(kg_code: str) -> bool:
    """Drop a KG from the registry (call on successful completion).

    Drops both the local registry entry and the Zenodo bundle, and
    writes a size=0 tombstone into cache_manifest.json so other peers
    drop their mirror view on the next sync tick.
    """
    m = _load_manifest()
    entry = m.pop(kg_code, None)
    # Always tombstone the cache_manifest mirror — a fresh peer that
    # consumed the bundle via the mirror (no local registry entry)
    # still owes the fleet a deletion signal.
    name = (entry.get("name") if entry else None) or _bundle_name(kg_code)
    _mirror_to_cache_manifest(kg_code, name, 0, 0, deleted=True)
    if entry is not None:
        _save_manifest(m)
    cache = _get_cache()
    if cache is None:
        return False
    try:
        from zenodo_lock import zenodo_upload_lock
        with zenodo_upload_lock(purpose=f"chkpt_delete:{kg_code}"):
            ok = _delete_from_deposit(cache, name)
        if ok:
            _log.info("chkpt_registry: deleted Zenodo bundle %s", name)
        return ok
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
