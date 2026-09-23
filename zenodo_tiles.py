"""zenodo_tiles — immutable per-tile objects in sharded Zenodo deposits.

Second-generation layout of the shared Copernicus tile cache
(docs/zenodo-cache.md → *Tile store v2*).  The first generation bundled
every 0.1° tile of a 1°×2° cell into one ZIP that had to be **rebuilt**
(download remote-only tiles + re-upload the whole archive) on every
flush.  For 12 MB harmonics tiles that meant re-uploading up to ~1 GB per
cell per flush, serialised behind the fleet Zenodo lock, and any
transient read error during the merge silently dropped a remote tile from
the rebuilt ZIP — which is why the cache held ~40 harmonics tiles for all
of Austria while NDVI had 160.

v2 rules:

* **One tile = one object.**  ``<product>_<S>_<W>_<N>_<E>[_<year>].npz``
  is PUT exactly once (same name → idempotent in-place PUT).  Nothing is
  ever rebuilt or merged; a peer only uploads what it has locally and the
  manifest does not know yet.
* **Sharded deposits.**  Zenodo caps a deposit at 100 files, so tiles are
  routed to a draft deposit per ``(product, 0.5° lat, 1.0° lon)`` shard
  (≤ 50 tiles each, ≤ 48 shards per product over Austria).  The shard
  registry lives in ``cache_manifest.json → shards`` and rides the
  existing primary↔peer sync; each tile entry also carries its own
  ``depo_id`` so readers never depend on the registry (or on the ``url``
  field, which older peers rewrite to the main deposit).
* **Parallel, lock-free PUTs.**  Distinct objects in distinct deposits
  cannot conflict, so peers upload with a small thread pool and take the
  fleet Zenodo lease only around shard *creation*.  The shared token's
  rate limit is handled by ``ZenodoCache._upload_file``'s 429 backoff.
* **Legacy read-through.**  Readers look up the per-tile entry first and
  fall back to the v1 ZIP indices (``ZenodoCache.fetch_copernicus``), so
  nothing already on Zenodo is lost and no migration is needed.  v1 ZIPs
  are frozen: ``upload_all`` no longer rebuilds Copernicus ZIPs.

Manifest entry (``files[<tile name>]``)::

    {"kind": "tile", "product": "harmonics", "depo_id": 123, "size": N,
     "checksum": "<md5>", "tile_count": 1, "updated_at": iso, "url": …}

Tombstones follow the v1 convention (``size == 0``).
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("zenodo_tiles")

SHARD_LAT = 0.5
SHARD_LON = 1.0
#: products stored per-tile (Hansen stays on the v1 ZIP path: 0.5° tiles, a
#: handful for all of Austria, never a size problem)
TILE_PRODUCTS = ("ndvi", "sar", "worldcover", "harmonics")
#: concurrent PUTs per peer flush
UPLOAD_WORKERS = int(os.environ.get("ZENODO_TILE_UPLOAD_WORKERS", "3"))
#: 100-file Zenodo cap; a shard holds ≤ 50 tiles by construction, this is
#: only a guard against a mis-keyed shard
SHARD_FILE_CAP = 95

_CREATE_LOCK = threading.Lock()


# === SECTION: naming ===

def shard_bounds(product: str, s: float, w: float) -> Tuple[float, float, float, float]:
    """(south, north, west, east) of the shard containing a tile's SW corner."""
    ss = math.floor(s / SHARD_LAT + 1e-9) * SHARD_LAT
    ww = math.floor(w / SHARD_LON + 1e-9) * SHARD_LON
    return (round(ss, 4), round(ss + SHARD_LAT, 4), round(ww, 4), round(ww + SHARD_LON, 4))


def shard_key(product: str, s: float, w: float) -> str:
    ss, sn, sw, se = shard_bounds(product, s, w)
    return f"{product}_{ss:.1f}_{sw:.1f}"


def shard_title(key: str) -> str:
    product, ss, sw = key.rsplit("_", 2)
    ss = float(ss); sw = float(sw)
    return (f"SRTM-LiDAR Austria tile cache: {product} "
            f"{ss:.1f}–{ss + SHARD_LAT:.1f}N {sw:.1f}–{sw + SHARD_LON:.1f}E")


def tile_name(product: str, w: float, s: float, e: float, n: float, **extra) -> str:
    from zenodo_cache import _npz_entry_name
    return _npz_entry_name(product, w, s, e, n, **extra)


def tile_url(depo_id: int, name: str, base_url: str = "https://zenodo.org") -> str:
    return f"{base_url}/api/records/{int(depo_id)}/draft/files/{name}/content"


def is_tile_entry(name: str, entry: dict | None) -> bool:
    return bool(entry) and (entry.get("kind") == "tile" or
                            (name.endswith(".npz") and entry.get("depo_id")))


# === SECTION: store ===

class TileStore:
    """Per-tile object store on top of a ``ZenodoCache`` (its session,
    ``_api`` retry ladder, ``_upload_file`` PUT-verify and manifest)."""

    def __init__(self, zc):
        self.zc = zc
        self.manifest = zc.manifest

    # --- shard registry -------------------------------------------------

    def shard(self, key: str, create: bool = True, lock_factory=None) -> Optional[dict]:
        """``{depo_id, bucket_url}`` for *key*; created on demand."""
        self.manifest.reload_if_changed()
        sh = self.manifest.get_shard(key)
        if sh and sh.get("depo_id"):
            return sh
        if not create:
            return None
        with _CREATE_LOCK:
            sh = self.manifest.get_shard(key)
            if sh and sh.get("depo_id"):
                return sh
            ctx = lock_factory(f"shard:{key}") if lock_factory else _nullctx()
            with ctx:
                self.manifest.reload_if_changed()
                sh = self.manifest.get_shard(key)
                if sh and sh.get("depo_id"):
                    return sh
                sh = self._find_shard_deposit(key) or self._create_shard_deposit(key)
                if sh:
                    self.manifest.set_shard(key, sh)
                    self.manifest.save()
                    try:
                        self._push_shard_to_director(key, sh)
                    except Exception:
                        pass
                return sh

    def _find_shard_deposit(self, key: str) -> Optional[dict]:
        """Adopt an existing draft with our exact title (another peer
        created it inside the ≤5-min manifest-sync lag)."""
        title = shard_title(key)
        try:
            r = self.zc._api("GET", "/api/deposit/depositions",
                             params={"q": f'title:"{title}"', "status": "draft", "size": 5})
            for d in r.json() or []:
                if (d.get("metadata") or {}).get("title") == title or d.get("title") == title:
                    sh = {"depo_id": int(d["id"]), "bucket_url": d["links"]["bucket"],
                          "created_at": d.get("created") or _now(), "adopted": True}
                    log.info("tile store: adopted existing shard deposit %s for %s", d["id"], key)
                    return sh
        except Exception as e:
            log.debug("tile store: shard search %s: %s", key, e)
        return None

    def _create_shard_deposit(self, key: str) -> Optional[dict]:
        import attributions as _attr
        product = key.rsplit("_", 2)[0]
        meta = {"metadata": {
            "title": shard_title(key),
            "upload_type": "dataset",
            "description": (
                f"Grid-aligned {product} cache tiles (0.1° NPZ, one file per tile) for "
                "Austrian landscape analysis. Shard of the SRTM-LiDAR Austria tile cache."
                + _attr.zenodo_description_footer()),
            "creators": [{"name": "SRTM-LiDAR Austria"}],
            "access_right": "open",
            "license": _attr.OUTPUT_LICENSE_ZENODO_ID,
            "notes": _attr.attribution_text(
                ["copernicus_s2", "copernicus_s1", "esa_worldcover"]),
            "keywords": _attr.zenodo_keywords(),
        }}
        r = self.zc._api("POST", "/api/deposit/depositions", json=meta)
        d = r.json()
        sh = {"depo_id": int(d["id"]), "bucket_url": d["links"]["bucket"], "created_at": _now()}
        log.info("tile store: created shard deposit %s for %s", d["id"], key)
        return sh

    def _push_shard_to_director(self, key: str, sh: dict) -> None:
        """Fast-path the new shard to the director so sibling peers see it
        before the 5-min sync (best effort)."""
        try:
            self_p = Path("data/austria_processor/self.json")
            url = (json.loads(self_p.read_text()).get("director_url") or "").rstrip("/")
        except Exception:
            url = ""
        if not url:
            return
        headers = {"Content-Type": "application/json"}
        try:
            tok = Path("data/admin_token").read_text().strip()
            if tok:
                headers["X-Admin-Token"] = tok
        except Exception:
            pass
        requests.put(url + "/api/v1/processing/cache_manifest",
                     data=json.dumps({"shards": {key: sh}, "files": {}}),
                     headers=headers, timeout=8)

    # --- read -----------------------------------------------------------

    def entry(self, name: str) -> Optional[dict]:
        e = self.manifest.get_file(name)
        return e if is_tile_entry(name, e) else None

    def has_tile(self, product: str, w, s, e, n, **extra) -> bool:
        return self.entry(tile_name(product, w, s, e, n, **extra)) is not None

    def fetch(self, product: str, w, s, e, n, dest_dir: Path, **extra) -> Optional[Path]:
        """Download one tile into *dest_dir* (local tile_key name).
        None when absent / 404 (tombstoned after deposit listing confirms)."""
        name = tile_name(product, w, s, e, n, **extra)
        ent = self.entry(name)
        if not ent:
            return None
        url = tile_url(ent["depo_id"], name, self.zc.base_url)
        try:
            r = self.zc._authed_session().get(url, timeout=120)
            if r.status_code == 404:
                self._confirm_404(name, ent)
                return None
            r.raise_for_status()
            data = r.content
        except requests.HTTPError as he:
            log.debug("tile store: fetch %s: %s", name, he)
            return None
        except Exception as ex:
            log.debug("tile store: fetch %s: %s", name, ex)
            return None
        from zenodo_cache import validate_tile_npz, _log_pollution_event
        ok, reason = validate_tile_npz(data, product)
        if not ok:
            log.warning("tile store: rejected polluted %s (%s)", name, reason)
            _log_pollution_event(product, name, reason, "zenodo_tile_download")
            return None
        from tile_cache import tile_key as _tk
        local = dest_dir / f"{product}_{_tk(product, w, s, e, n, **extra)}.npz"
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(".tmp.npz")
        try:
            tmp.write_bytes(data)
            tmp.rename(local)
        except Exception as ex:
            tmp.unlink(missing_ok=True)
            log.warning("tile store: write %s: %s", local, ex)
            return None
        try:
            from zenodo_cache import write_tile_meta
            write_tile_meta(local, product, w, s, e, n, **extra)
        except Exception:
            pass
        log.info("Restored %s from Zenodo tile store (%d bytes)", local.name, len(data))
        return local

    def _confirm_404(self, name: str, ent: dict) -> None:
        from zenodo_cache import zenodo_degraded
        if zenodo_degraded():
            return
        listing = self.zc._list_deposit_files(int(ent["depo_id"]))
        if listing is None or name in listing:
            return
        if self.manifest.tombstone(name, "404 on read, confirmed absent by shard listing"):
            self.manifest.save()
            log.warning("tile store: %s gone from deposit %s → tombstoned", name, ent["depo_id"])

    # --- write ----------------------------------------------------------

    def put(self, product: str, local_path: Path, w, s, e, n, extra: dict,
            lock_factory=None) -> Optional[dict]:
        """Upload one local tile (idempotent). Returns the manifest entry."""
        name = tile_name(product, w, s, e, n, **extra)
        if self.entry(name):
            return self.entry(name)
        from zenodo_cache import validate_tile_npz, _log_pollution_event, _UploadFailed
        ok, reason = validate_tile_npz(local_path, product)
        if not ok:
            _log_pollution_event(product, name, reason, "local_upload")
            log.warning("tile store: skipping polluted %s: %s", name, reason)
            return None
        sh = self.shard(shard_key(product, s, w), lock_factory=lock_factory)
        if not sh:
            return None
        depo_id = int(sh["depo_id"])
        try:
            res = self.zc._upload_file(depo_id, local_path, name, bucket_url=sh.get("bucket_url"))
        except _UploadFailed as uf:
            if uf.remote_state == "present_old" and uf.remote_entry:
                res = {"checksum": uf.remote_entry.get("checksum", ""),
                       "size": int(uf.remote_entry.get("size") or 0)}
            else:
                log.warning("tile store: PUT %s → %s (%s)", name, uf.remote_state, uf.status)
                return None
        ent = {"kind": "tile", "product": product, "depo_id": depo_id,
               "url": tile_url(depo_id, name, self.zc.base_url),
               "size": int(res.get("size") or local_path.stat().st_size),
               "checksum": str(res.get("checksum") or ""), "tile_count": 1,
               "updated_at": _now()}
        self.manifest.set_entry(name, ent)
        return ent

    def upload_local(self, dry_run: bool = False, lock_factory=None,
                     workers: int = UPLOAD_WORKERS) -> Dict[str, Any]:
        """PUT every local Copernicus tile the manifest does not have yet."""
        from zenodo_cache import _build_reverse_index, _scan_local_copernicus
        stats = {"tiles_uploaded": 0, "tiles_skipped": 0, "tiles_failed": 0,
                 "bytes_total": 0, "shards_touched": 0}
        rev = _build_reverse_index()
        todo = []
        for product, paths in _scan_local_copernicus().items():
            if product not in TILE_PRODUCTS:
                continue
            for p in paths:
                info = rev.get(p.name)
                if info is None:
                    continue
                _, w, s, e, n, extra = info
                name = tile_name(product, w, s, e, n, **extra)
                raw = self.manifest.get_raw(name)
                if raw and raw.get("size"):
                    stats["tiles_skipped"] += 1
                    continue
                if raw and not raw.get("size") and _recent(raw.get("updated_at"), 3600):
                    stats["tiles_skipped"] += 1  # fresh tombstone: let the reconciler settle it
                    continue
                todo.append((product, p, w, s, e, n, extra))
        if not todo:
            return stats
        if dry_run:
            stats["tiles_pending"] = len(todo)
            return stats
        shards = {shard_key(t[0], t[3], t[2]) for t in todo}
        stats["shards_touched"] = len(shards)
        # Shard deposits first (serialised, lock-protected), then PUTs in parallel.
        for k in sorted(shards):
            try:
                self.shard(k, lock_factory=lock_factory)
            except Exception as ex:
                log.warning("tile store: shard %s unavailable: %s", k, ex)
        done_t = time.time()

        def _one(t):
            product, p, w, s, e, n, extra = t
            try:
                return self.put(product, p, w, s, e, n, extra, lock_factory=lock_factory)
            except Exception as ex:
                log.warning("tile store: %s: %s", p.name, ex)
                return None

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(_one, t): t for t in todo}
            n_since_save = 0
            for f in as_completed(futs):
                ent = f.result()
                if ent:
                    stats["tiles_uploaded"] += 1
                    stats["bytes_total"] += int(ent.get("size") or 0)
                    n_since_save += 1
                    if n_since_save >= 5:
                        self.manifest.save(); n_since_save = 0
                else:
                    stats["tiles_failed"] += 1
        self.manifest.save()
        log.info("tile store: uploaded %d tile(s) (%.1f MB) to %d shard(s), %d skipped, %d failed in %.0fs",
                 stats["tiles_uploaded"], stats["bytes_total"] / 1e6, len(shards),
                 stats["tiles_skipped"], stats["tiles_failed"], time.time() - done_t)
        return stats

    # --- reconcile --------------------------------------------------------

    def reconcile(self, dry_run: bool = True, max_shards: int = 20) -> Dict[str, Any]:
        """List up to *max_shards* shard deposits (round-robin by last
        check) and settle tile entries: live-but-absent → tombstone,
        tombstoned-but-present → restore, unknown deposit file → adopt."""
        self.manifest.reload_if_changed()
        summary = {"shards": 0, "tombstoned": 0, "restored": 0, "adopted": 0, "details": []}
        shards = self.manifest.all_shards()
        by_depo: Dict[int, List[str]] = {}
        for name, e in self.manifest.all_files().items():
            if is_tile_entry(name, e) or (name.endswith(".npz") and (e or {}).get("depo_id")):
                by_depo.setdefault(int(e["depo_id"]), []).append(name)
        order = sorted(shards.items(), key=lambda kv: kv[1].get("checked_at") or "")
        changed = 0
        for key, sh in order[:max_shards]:
            did = int(sh.get("depo_id") or 0)
            if not did:
                continue
            listing = self.zc._list_deposit_files(did)
            if listing is None:
                continue
            summary["shards"] += 1
            sh["checked_at"] = _now(); sh["files"] = len(listing)
            product = key.rsplit("_", 2)[0]
            for name in by_depo.get(did, []):
                e = self.manifest.get_raw(name) or {}
                live = bool(e.get("size"))
                dep = listing.get(name)
                if live and dep is None:
                    summary["tombstoned"] += 1
                    summary["details"].append({"file": name, "action": "tombstoned"})
                    if not dry_run and self.manifest.tombstone(name, "reconcile: missing from shard"):
                        changed += 1
                elif not live and dep is not None:
                    summary["restored"] += 1
                    summary["details"].append({"file": name, "action": "restored"})
                    if not dry_run:
                        self.manifest.set_entry(name, {
                            "kind": "tile", "product": product, "depo_id": did,
                            "url": tile_url(did, name, self.zc.base_url),
                            "size": int(dep.get("size") or 0),
                            "checksum": str(dep.get("checksum") or ""), "tile_count": 1,
                            "updated_at": _now(), "restored_from_tombstone": True})
                        changed += 1
            known = set(by_depo.get(did, []))
            for name, dep in listing.items():
                if name in known or not name.endswith(".npz"):
                    continue
                summary["adopted"] += 1
                summary["details"].append({"file": name, "action": "adopted"})
                if not dry_run:
                    self.manifest.set_entry(name, {
                        "kind": "tile", "product": product, "depo_id": did,
                        "url": tile_url(did, name, self.zc.base_url),
                        "size": int(dep.get("size") or 0),
                        "checksum": str(dep.get("checksum") or ""), "tile_count": 1,
                        "updated_at": _now(), "adopted_from_listing": True})
                    changed += 1
            if not dry_run:
                self.manifest.set_shard(key, sh)
        if not dry_run and (changed or summary["shards"]):
            self.manifest.save()
        summary["changed"] = changed
        return summary

    def status(self) -> Dict[str, Any]:
        files = self.manifest.all_files()
        per: Dict[str, int] = {}
        nbytes = 0
        tomb = 0
        for name, e in files.items():
            if not is_tile_entry(name, e):
                continue
            if not e.get("size"):
                tomb += 1
                continue
            per[e.get("product", "?")] = per.get(e.get("product", "?"), 0) + 1
            nbytes += int(e.get("size") or 0)
        return {"tiles": sum(per.values()), "by_product": per, "bytes": nbytes,
                "tombstoned": tomb, "shards": len(self.manifest.all_shards())}


# === SECTION: helpers ===

class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _recent(ts: str | None, within_s: float) -> bool:
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds() < within_s
    except Exception:
        return False
