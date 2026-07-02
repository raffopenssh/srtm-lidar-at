"""Zenodo-backed persistent cache for Copernicus + Hansen tile data.

Local tile caches (Copernicus 0.1° grid, Hansen 0.5° grid) get evicted when
disk approaches 5 GB.  This module uploads completed cache tiles to a single
Zenodo deposit as ZIP archives (one per product × latitude strip), and can
restore them on demand via HTTP range reads — avoiding expensive re-fetches
from Copernicus/Hansen APIs.

Architecture
------------

**Upload** (periodic, independent of austria_processor):
  Scans local tile cache dirs, bundles tiles into ZIP files grouped by
  product type and 0.5° latitude strip, uploads to Zenodo.
  Uses its own manifest (`data/austria_processor/cache_manifest.json`)
  — never touches austria_processor's `zenodo_manifest.json`.

**Download** (on cache miss, before falling back to API):
  When ``tile_cache.py`` has a local miss, calls this module's
  ``fetch_from_zenodo()`` to retrieve the tile from the Zenodo ZIP.
  Uses 2-3 HTTP range requests: HEAD → central directory → entry data.
  Writes the tile into the local cache dir so subsequent reads are local.

Zenodo Record Layout
--------------------

One Zenodo record containing ~30 ZIP files::

    copernicus_ndvi_strip_46.0_46.5.zip
    copernicus_ndvi_strip_46.5_47.0.zip
    ...  (6 strips × 4 products = 24 Copernicus ZIPs)
    hansen_strip_46.0_46.5.zip
    ...  (6 strips = 6 Hansen ZIPs)

Each ZIP contains NPZ files named by their grid key::

    ndvi_46.3000_15.0000_46.4000_15.1000_2024.npz
    ndvi_46.3000_15.1000_46.4000_15.2000_2024.npz
    ...

Total: ~540 MB for all of Austria. Well within Zenodo's 50 GB / 100 file limits.

Usage
-----
::

    from zenodo_cache import ZenodoCache

    cache = ZenodoCache()

    # Upload local tiles to Zenodo (run periodically)
    cache.upload_all()

    # Fetch a tile from Zenodo (called by tile_cache on miss)
    data = cache.fetch_copernicus("ndvi", 15.0, 46.3, 15.1, 46.4, year=2024)
    data = cache.fetch_hansen(15.0, 46.0, 15.5, 46.5)
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import struct
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

log = logging.getLogger(__name__)

# === SECTION: Config ===

DATA_DIR = Path("data/austria_processor")
CACHE_MANIFEST_PATH = DATA_DIR / "cache_manifest.json"

# Austria bounding box (generous)
AT_WEST, AT_SOUTH, AT_EAST, AT_NORTH = 9.0, 46.0, 17.5, 49.5

# Grid steps (must match tile_cache.py)
COP_STEP = 0.1   # Copernicus grid
HANSEN_STEP = 0.5  # Hansen grid

# Bundle cell dimensions for ZIP grouping. Old layout used
# latitude-only strips (STRIP_HEIGHT = 0.5°). New layout adds a
# longitude axis (STRIP_HEIGHT × STRIP_WIDTH cells) so the Peer
# Director can give each frontier peer a disjoint cell -- which lifts
# the parallel-frontier ceiling above the cred count.
#
# Readers transparently support BOTH layouts so existing cached tiles
# bundled as strip ZIPs remain accessible without re-uploading
# anything (avoiding 700 MB of redundant Zenodo traffic).
STRIP_HEIGHT = 1.0
STRIP_WIDTH = 2.0

# Copernicus product types
COP_PRODUCTS = ("ndvi", "sar", "worldcover", "harmonics")

# Zenodo API
ZENODO_BASE_URL = "https://zenodo.org"
ZENODO_TOKEN = "2dnLSA2YYTc8jt3a1X0qDZUBb1hyOIpGJ44UoJr8N69wdePODgq4cjbJ0DJa"

# Local cache dirs (must match tile_cache.py)
COP_CACHE_DIR = DATA_DIR / "copernicus_tiles"
HANSEN_CACHE_DIR = DATA_DIR / "hansen_tiles"

# Download cache for Zenodo ZIP central directories
_ZIP_INDEX_CACHE_DIR = DATA_DIR / "zenodo_zip_index"

# Persistent pollution event log
_POLLUTION_LOG_PATH = DATA_DIR / "pollution_log.jsonl"


# === SECTION: Bundle cells (lat × lon) ===

def _lat_lon_cells() -> List[Tuple[float, float, float, float]]:
    """Return (south, north, west, east) bounds for each bundle cell
    covering Austria. Cells partition the plane on the
    STRIP_HEIGHT × STRIP_WIDTH grid.
    """
    cells = []
    s = math.floor(AT_SOUTH / STRIP_HEIGHT) * STRIP_HEIGHT
    while s < AT_NORTH:
        n = round(s + STRIP_HEIGHT, 4)
        w = math.floor(AT_WEST / STRIP_WIDTH) * STRIP_WIDTH
        while w < AT_EAST:
            e = round(w + STRIP_WIDTH, 4)
            cells.append((round(s, 4), n, round(w, 4), e))
            w = e
        s = n
    return cells


def _cell_for_bbox(lat: float, lon: float) -> Tuple[float, float, float, float]:
    """Return the (south, north, west, east) cell containing the point."""
    s = math.floor(lat / STRIP_HEIGHT) * STRIP_HEIGHT
    w = math.floor(lon / STRIP_WIDTH) * STRIP_WIDTH
    return (round(s, 4), round(s + STRIP_HEIGHT, 4),
            round(w, 4), round(w + STRIP_WIDTH, 4))


# --- Legacy strip helpers (kept for read-side compat with old ZIPs) ---

def _lat_strips() -> List[Tuple[float, float]]:
    """DEPRECATED: returns 1° lat-only ranges. Kept so callers that still
    plan by lat-strip don't break. New code should use _lat_lon_cells().
    """
    out = []
    s = math.floor(AT_SOUTH / STRIP_HEIGHT) * STRIP_HEIGHT
    while s < AT_NORTH:
        n = round(s + STRIP_HEIGHT, 4)
        out.append((round(s, 4), n))
        s = n
    return out


def _strip_for_lat(lat: float) -> Tuple[float, float]:
    """DEPRECATED: returns lat-only range for *lat*. Use _cell_for_bbox.
    Used by readers to look up legacy strip ZIPs as a fallback.
    """
    s = math.floor(lat / STRIP_HEIGHT) * STRIP_HEIGHT
    return (round(s, 4), round(s + STRIP_HEIGHT, 4))


def _zip_filename(product: str, *cell: float) -> str:
    """Canonical ZIP filename.

    Accepts either:
      * ``(south, north, west, east)`` -- new cell bundle name
        ``copernicus_<prod>_cell_<S>_<N>_<W>_<E>.zip``
      * ``(south, north)`` -- legacy strip name
        ``copernicus_<prod>_strip_<S>_<N>.zip`` (for read-side compat
        with deposits uploaded before the cell layout).
    """
    if len(cell) == 4:
        s, n, w, e = cell
        suffix = f"cell_{s:.1f}_{n:.1f}_{w:.1f}_{e:.1f}"
    elif len(cell) == 2:
        s, n = cell
        suffix = f"strip_{s:.1f}_{n:.1f}"
    else:
        raise ValueError(f"_zip_filename: expected 2 or 4 coords, got {cell!r}")
    if product == "hansen":
        return f"hansen_{suffix}.zip"
    return f"copernicus_{product}_{suffix}.zip"


def _legacy_strip_zip_for(product: str, lat_south: float) -> str:
    """Filename of the legacy 0.5° lat-strip ZIP that may contain a tile.

    Old deposits used STRIP_HEIGHT=0.5; this snaps *lat_south* to that
    grid and returns the single 0.5° strip ZIP it falls into.

    WARNING: callers that pass a 1° *cell* south (an integer degree) only
    ever get the LOWER half-strip back, so tiles physically stored in the
    upper half-strip (e.g. ``strip_47.5_48.0``) are invisible. Use
    ``_legacy_strip_zips_for_cell`` on the read path to probe BOTH halves.
    """
    # Pre-migration step was 0.5° -- snap the input lat to that grid.
    base = math.floor(lat_south / 0.5) * 0.5
    return _zip_filename(product, round(base, 4), round(base + 0.5, 4))


def _legacy_strip_zips_for_cell(product: str, cell_south: float) -> List[str]:
    """Both legacy 0.5° strip ZIPs spanning a 1° bundle cell.

    A 1° cell ``[cell_south, cell_south+1)`` overlaps two pre-migration
    0.5° strips: ``[cell_south, cell_south+0.5)`` and
    ``[cell_south+0.5, cell_south+1)``. The read path must probe BOTH,
    because before the cell layout tiles were grouped into true 0.5°
    strips (so an upper tile lives in the upper strip), AND the
    cell-era writer dumps a whole 1° cell's tiles into whichever single
    legacy strip already exists. Either way the actual NPZ entry name
    encodes the tile's real latitude, so probing both halves never
    yields a false positive.
    """
    base = math.floor(cell_south / 0.5) * 0.5
    out = [
        _zip_filename(product, round(base, 4), round(base + 0.5, 4)),
        _zip_filename(product, round(base + 0.5, 4), round(base + 1.0, 4)),
    ]
    seen: set = set()
    return [z for z in out if not (z in seen or seen.add(z))]


def _npz_entry_name(product: str, w: float, s: float, e: float, n: float,
                    **extra) -> str:
    """NPZ filename within a ZIP (encodes grid coordinates + params)."""
    name = f"{product}_{s:.4f}_{w:.4f}_{n:.4f}_{e:.4f}"
    if "year" in extra:
        name += f"_{extra['year']}"
    return name + ".npz"


# === SECTION: CacheManifest (separate from austria_processor manifest) ===

class CacheManifest:
    """Thread-safe manifest for Zenodo cache record.

    Stored at ``data/austria_processor/cache_manifest.json``.
    Tracks the Zenodo deposit ID and per-ZIP file URLs/sizes.
    Uses file locking (not just threading lock) to be safe against
    concurrent access from other processes.
    """

    def __init__(self, path: Path = CACHE_MANIFEST_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {"depo_id": None, "record_id": None,
                                       "files": {}}
        self._last_mtime: float = 0.0
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
                self._last_mtime = self._path.stat().st_mtime
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Failed to load cache manifest: %s", e)

    def reload_if_changed(self):
        """Re-read from disk if the file has been modified externally."""
        try:
            if self._path.exists():
                mtime = self._path.stat().st_mtime
                if mtime > self._last_mtime:
                    with self._lock:
                        self._data = json.loads(self._path.read_text())
                        self._last_mtime = mtime
                    log.info("Cache manifest reloaded (depo_id=%s, %d files)",
                             self._data.get('depo_id'), len(self._data.get('files', {})))
        except Exception as e:
            log.debug("Cache manifest reload check failed: %s", e)

    def save(self):
        """Persist to disk atomically."""
        with self._lock:
            data = dict(self._data)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self._path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, self._path)

    @property
    def depo_id(self) -> Optional[int]:
        return self._data.get("depo_id")

    @depo_id.setter
    def depo_id(self, val: int):
        with self._lock:
            self._data["depo_id"] = val

    @property
    def record_id(self) -> Optional[int]:
        return self._data.get("record_id")

    @record_id.setter
    def record_id(self, val: int):
        with self._lock:
            self._data["record_id"] = val

    def set_file(self, zip_name: str, url: str, size: int, checksum: str,
                 tile_count: int, updated_at: str):
        """Record a ZIP file's Zenodo metadata."""
        with self._lock:
            self._data.setdefault("files", {})[zip_name] = {
                "url": url,
                "size": size,
                "checksum": checksum,
                "tile_count": tile_count,
                "updated_at": updated_at,
            }

    def get_file(self, zip_name: str) -> Optional[Dict]:
        with self._lock:
            return self._data.get("files", {}).get(zip_name)

    def all_files(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._data.get("files", {}))

    def tile_count(self) -> int:
        """Total number of tiles across all ZIPs."""
        return sum(f.get("tile_count", 0) for f in self._data.get("files", {}).values())


# === SECTION: HTTP range-read helpers ===

class HTTPRangeFile:
    """File-like object backed by HTTP range requests.

    Supports seeking and reading, which is all zipfile needs.
    Caches the central directory region for efficiency.
    """

    def __init__(self, url: str, session: Optional[requests.Session] = None):
        self.url = url
        self._session = session or requests.Session()
        self._pos = 0
        self._size = None
        self._cache: Dict[Tuple[int, int], bytes] = {}  # (start, end) → data
        self._fetch_size()

    def _fetch_size(self):
        r = self._session.head(self.url, timeout=15, allow_redirects=True)
        r.raise_for_status()
        self._size = int(r.headers["Content-Length"])

    def __len__(self):
        return self._size

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset: int, whence: int = 0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        return self._pos

    def tell(self):
        return self._pos

    def read(self, n: int = -1) -> bytes:
        if n == -1:
            n = self._size - self._pos
        if n <= 0:
            return b""

        start = self._pos
        end = min(start + n - 1, self._size - 1)

        # Check cache
        for (cs, ce), data in self._cache.items():
            if cs <= start and ce >= end:
                offset = start - cs
                result = data[offset:offset + n]
                self._pos += len(result)
                return result

        # Fetch with range request (read a bit extra to reduce round-trips)
        fetch_end = min(end + 65536, self._size - 1)  # prefetch 64KB
        headers = {"Range": f"bytes={start}-{fetch_end}"}
        r = self._session.get(self.url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.content
        self._cache[(start, start + len(data) - 1)] = data

        result = data[:n]
        self._pos += len(result)
        return result


class _StaleOffsetError(Exception):
    """Raised when a ZIP entry's cached header_offset points at the
    wrong byte in the remote ZIP — typically because another peer
    rewrote the ZIP after this peer cached the central directory."""
    pass


# === SECTION: ZIP index (central directory cache) ===

class ZipIndex:
    """Cached index of entries in a remote ZIP file.

    On first access, reads the ZIP central directory (last ~64KB) and caches
    entry offsets locally.  Subsequent lookups are instant.
    """

    def __init__(self, url: str, cache_dir: Path = _ZIP_INDEX_CACHE_DIR,
                 session: Optional[requests.Session] = None):
        self.url = url
        self._session = session
        self._cache_dir = cache_dir
        self._entries: Optional[Dict[str, zipfile.ZipInfo]] = None
        self._cache_key = hashlib.md5(url.encode()).hexdigest()[:12]
        self._index_path = cache_dir / f"{self._cache_key}.json"

    def _load_or_fetch(self) -> Dict[str, zipfile.ZipInfo]:
        """Return dict of entry_name → ZipInfo."""
        # If we have an in-memory index but the on-disk file has been
        # deleted by an external invalidator (e.g. cache_manifest sync
        # detecting a strip rewrite by another peer), drop the cached
        # entries and re-fetch the central directory. Without this,
        # long-running subprocesses keep using stale offsets and we
        # only recover via the bad-local-header retry path.
        if self._entries is not None:
            try:
                if not self._index_path.exists():
                    self._entries = None
            except OSError:
                pass
        if self._entries is not None:
            return self._entries

        # Try local index cache first
        if self._index_path.exists():
            try:
                raw = json.loads(self._index_path.read_text())
                # Reconstruct minimal ZipInfo objects
                entries = {}
                for name, info in raw.items():
                    zi = zipfile.ZipInfo(name)
                    zi.compress_size = info["compress_size"]
                    zi.file_size = info["file_size"]
                    zi.header_offset = info["header_offset"]
                    zi.compress_type = info.get("compress_type", zipfile.ZIP_DEFLATED)
                    entries[name] = zi
                self._entries = entries
                return entries
            except Exception:
                self._index_path.unlink(missing_ok=True)

        # Fetch central directory from remote ZIP
        log.info("Fetching ZIP index from %s", self.url)
        hrf = HTTPRangeFile(self.url, session=self._session)
        with zipfile.ZipFile(hrf) as zf:
            entries = {zi.filename: zi for zi in zf.infolist()}

        # Cache locally
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_data = {
            name: {
                "compress_size": zi.compress_size,
                "file_size": zi.file_size,
                "header_offset": zi.header_offset,
                "compress_type": zi.compress_type,
            }
            for name, zi in entries.items()
        }
        try:
            tmp = str(self._index_path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cache_data, f)
            os.replace(tmp, self._index_path)
        except OSError:
            pass

        self._entries = entries
        return entries

    def has_entry(self, name: str) -> bool:
        return name in self._load_or_fetch()

    def list_entries(self) -> List[str]:
        return list(self._load_or_fetch().keys())

    def entry_info(self, name: str) -> Optional[zipfile.ZipInfo]:
        return self._load_or_fetch().get(name)

    def invalidate(self):
        """Drop in-memory + on-disk cached central directory.

        Forces the next call to _load_or_fetch() to re-read the ZIP
        central directory from Zenodo.  Used when a stale offset is
        detected (the remote ZIP was rewritten by another peer with
        more tiles merged in).
        """
        self._entries = None
        try:
            self._index_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _try_read_with_entries(self, name: str,
                                entries: Dict[str, zipfile.ZipInfo]) -> Optional[bytes]:
        """Attempt to read a single entry using the supplied entry table.

        Returns the decompressed bytes on success, None if the entry is
        missing, or raises _StaleOffsetError when the local file header
        signature is wrong (indicating the central directory is stale).
        """
        zi = entries.get(name)
        if zi is None:
            return None

        hrf = HTTPRangeFile(self.url, session=self._session)
        hrf.seek(zi.header_offset)

        local_header = hrf.read(30)
        if len(local_header) < 30:
            return None

        sig = struct.unpack("<I", local_header[:4])[0]
        if sig != 0x04034b50:  # PK\x03\x04
            raise _StaleOffsetError(
                f"Bad local header signature at offset {zi.header_offset}"
            )

        fname_len = struct.unpack("<H", local_header[26:28])[0]
        extra_len = struct.unpack("<H", local_header[28:30])[0]
        hrf.read(fname_len + extra_len)
        compressed = hrf.read(zi.compress_size)

        if zi.compress_type == zipfile.ZIP_STORED:
            return compressed
        elif zi.compress_type == zipfile.ZIP_DEFLATED:
            import zlib
            return zlib.decompress(compressed, -15)
        else:
            hrf2 = HTTPRangeFile(self.url, session=self._session)
            with zipfile.ZipFile(hrf2) as zf:
                return zf.read(name)

    def read_entry(self, name: str) -> Optional[bytes]:
        """Read a single entry from the remote ZIP via HTTP range request.

        On a bad local-header signature (stale cached central directory
        because another peer rewrote this ZIP with more tiles merged in),
        invalidate the cached index, re-fetch it from Zenodo, and retry
        the read once.  This recovers transparently from the
        "Bad local header signature at offset N" warning that used to
        cause cache-only peers to abort the KG.
        """
        entries = self._load_or_fetch()
        try:
            return self._try_read_with_entries(name, entries)
        except _StaleOffsetError as e:
            # Self-healing path: another peer rewrote the strip ZIP, so
            # our cached offsets are off. We invalidate + refetch + retry
            # transparently. INFO not WARNING because it's expected when
            # multiple peers share a Zenodo deposit.
            log.info(
                "%s for %s in %s — invalidating index and retrying",
                e, name, self.url,
            )
            self.invalidate()
            try:
                fresh = self._load_or_fetch()
            except Exception as exc:
                log.warning("ZIP index re-fetch failed: %s", exc)
                return None
            try:
                return self._try_read_with_entries(name, fresh)
            except _StaleOffsetError as e2:
                # Still bad after a fresh fetch — the ZIP itself is
                # corrupt at this offset, not our cache.  Give up.
                log.warning(
                    "Stale offset persists after re-fetch: %s. Giving up.", e2,
                )
                return None


# === SECTION: Inventory (what's in local cache) ===


def cleanup_orphan_tiles() -> int:
    """Delete local cache tiles that have no .meta.json sidecar.

    These are legacy tiles created before the per-cell refactor.  They
    can't be mapped to grid coordinates and are invisible to the Zenodo
    uploader.  They also can't serve cache hits because the hash-based
    filename no longer matches any current key.

    Returns the number of files deleted.
    """
    deleted = 0
    freed = 0
    for cache_dir in (COP_CACHE_DIR, HANSEN_CACHE_DIR):
        if not cache_dir.exists():
            continue
        for f in cache_dir.glob("*.npz"):
            meta = f.with_name(f.stem + ".meta.json")
            if not meta.exists():
                sz = f.stat().st_size
                try:
                    f.unlink()
                    deleted += 1
                    freed += sz
                except OSError:
                    pass
    if deleted:
        log.info("Cleaned up %d orphan tile cache files (%.1f MB)",
                 deleted, freed / 1e6)
    return deleted


def _scan_local_copernicus() -> Dict[str, List[Path]]:
    """Scan local Copernicus tile cache, group by product type.

    Returns {product: [path, ...]} where product is one of COP_PRODUCTS.
    """
    result = {p: [] for p in COP_PRODUCTS}
    if not COP_CACHE_DIR.exists():
        return result
    for f in COP_CACHE_DIR.glob("*.npz"):
        name = f.name
        for p in COP_PRODUCTS:
            if name.startswith(p + "_"):
                result[p].append(f)
                break
    return result


def _scan_local_hansen() -> List[Path]:
    """Return list of Hansen NPZ files in local cache."""
    if not HANSEN_CACHE_DIR.exists():
        return []
    return list(HANSEN_CACHE_DIR.glob("hansen_*.npz"))


def _build_reverse_index() -> Dict[str, Tuple[str, float, float, float, float, Dict]]:
    """Build filename → (product, w, s, e, n, extra) from tile_bbox_index + sidecars.

    Returns dict mapping NPZ filename (without dir) to grid coordinates + extra params.

    Strategy:
    1. Read .meta.json sidecars (written by write_tile_meta()) for exact params
    2. Fall back to tile_bbox_index.json and reconstruct filenames for product/year combos

    The sidecar approach is authoritative; the tile_bbox_index is a fallback.
    """
    from tile_cache import tile_key, _TILE_INDEX_PATH

    reverse: Dict[str, Tuple[str, float, float, float, float, Dict]] = {}

    # --- Pass 1: Read sidecars (.meta.json files alongside .npz) ---
    for cache_dir in (COP_CACHE_DIR, HANSEN_CACHE_DIR):
        if not cache_dir.exists():
            continue
        for meta_path in cache_dir.glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
                product = meta["product"]
                w, s, e, n = meta["w"], meta["s"], meta["e"], meta["n"]
                extra = meta.get("extra", {})
                npz_name = meta_path.name.replace(".meta.json", ".npz")
                reverse[npz_name] = (product, w, s, e, n, extra)
            except Exception:
                continue

    # --- Pass 2: Reconstruct from tile_bbox_index.json (fallback) ---
    index_path = _TILE_INDEX_PATH
    if index_path.exists():
        try:
            raw = json.loads(index_path.read_text())
        except Exception:
            raw = {}

        for key_str, entry in raw.items():
            source = entry.get("source", "")
            w = entry["w"]
            s = entry["s"]
            e = entry["e"]
            n = entry["n"]

            if source == "copernicus":
                for product in COP_PRODUCTS:
                    # Without year (worldcover)
                    tk = tile_key(product, w, s, e, n)
                    fname = f"{product}_{tk}.npz"
                    if fname not in reverse:
                        reverse[fname] = (product, w, s, e, n, {})

                    # With year (ndvi, sar, harmonics)
                    if product in ("ndvi", "sar", "harmonics"):
                        for year in (2023, 2024):
                            tk_y = tile_key(product, w, s, e, n, year=year)
                            fname_y = f"{product}_{tk_y}.npz"
                            if fname_y not in reverse:
                                reverse[fname_y] = (product, w, s, e, n,
                                                     {"year": year})

            elif source == "hansen":
                tk = tile_key("hansen", w, s, e, n)
                fname = f"hansen_{tk}.npz"
                if fname not in reverse:
                    reverse[fname] = ("hansen", w, s, e, n, {})

    return reverse


def write_tile_meta(npz_path: Path, product: str,
                    w: float, s: float, e: float, n: float,
                    **extra):
    """Write a sidecar .meta.json alongside an NPZ tile file.

    Called by tile_cache after writing a tile so zenodo_cache can later
    identify the tile's grid coordinates without reverse-engineering the hash.
    """
    meta = {"product": product, "w": w, "s": s, "e": e, "n": n}
    if extra:
        meta["extra"] = extra
    meta_path = npz_path.with_name(npz_path.stem + ".meta.json")
    try:
        tmp = str(meta_path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, meta_path)
    except OSError:
        pass


# === SECTION: Tile validation ===

# Minimum pixels per dimension for a tile to be considered valid.
# A 0.1° tile at 10m resolution is ~750×1110 px; anything below 100
# is clearly degenerate (partial download, empty response, etc.).
_MIN_TILE_DIM = 100

# Maximum fraction of NaN / nodata pixels before a tile is "polluted".
_MAX_NAN_FRACTION = 0.95


def validate_tile_npz(path_or_bytes, product: str) -> Tuple[bool, str]:
    """Validate a cached tile NPZ before upload or after download.

    Returns (ok, reason).  If ok is False, the tile should not be uploaded
    to Zenodo / should be discarded after download.

    Checks per product:
      ndvi       — has 'ndvi' array, shape ≥ 100×100, <95% NaN, not constant
      sar        — has 'vv'+'vh', matching shapes ≥ 100×100, <95% NaN
      worldcover — unpickles to dict with 'map' ≥ 100×100, not all-zero
      harmonics  — has h_mean/h_amplitude/h_phase ≥ 100×100, <95% NaN
      hansen     — has treecover2000+lossyear ≥ 100×100, datamask not all-zero
    """
    try:
        if isinstance(path_or_bytes, (str, Path)):
            d = np.load(str(path_or_bytes), allow_pickle=True)
        else:
            d = np.load(io.BytesIO(path_or_bytes), allow_pickle=True)
    except Exception as e:
        return False, f"cannot load NPZ: {e}"

    if product == "ndvi":
        if "ndvi" not in d:
            return False, "missing 'ndvi' key"
        arr = d["ndvi"]
        if arr.ndim < 2 or arr.shape[0] < _MIN_TILE_DIM or arr.shape[1] < _MIN_TILE_DIM:
            return False, f"ndvi shape too small: {arr.shape}"
        nan_frac = np.isnan(arr).mean()
        if nan_frac >= _MAX_NAN_FRACTION:
            return False, f"ndvi {nan_frac:.1%} NaN"
        valid = arr[~np.isnan(arr)]
        if len(valid) > 0 and valid.min() == valid.max():
            return False, f"ndvi constant value {valid.min()}"

    elif product == "sar":
        for key in ("vv", "vh"):
            if key not in d:
                return False, f"missing '{key}' key"
        vv, vh = d["vv"], d["vh"]
        if vv.shape != vh.shape:
            return False, f"vv/vh shape mismatch: {vv.shape} vs {vh.shape}"
        if vv.ndim < 2 or vv.shape[0] < _MIN_TILE_DIM or vv.shape[1] < _MIN_TILE_DIM:
            return False, f"sar shape too small: {vv.shape}"
        nan_frac = np.isnan(vv).mean()
        if nan_frac >= _MAX_NAN_FRACTION:
            return False, f"sar vv {nan_frac:.1%} NaN"

    elif product == "worldcover":
        if "data" not in d:
            return False, "missing 'data' key"
        try:
            obj = d["data"].item()
        except Exception as e:
            return False, f"cannot unpickle worldcover data: {e}"
        if not isinstance(obj, dict) or "map" not in obj:
            return False, "worldcover data is not a dict with 'map'"
        m = obj["map"]
        if m.ndim < 2 or m.shape[0] < _MIN_TILE_DIM or m.shape[1] < _MIN_TILE_DIM:
            return False, f"worldcover map shape too small: {m.shape}"
        if np.all(m == 0):
            return False, "worldcover map all zeros"

    elif product == "harmonics":
        for key in ("h_mean", "h_amplitude", "h_phase"):
            if key not in d:
                return False, f"missing '{key}' key"
        hm = d["h_mean"]
        if hm.ndim < 2 or hm.shape[0] < _MIN_TILE_DIM or hm.shape[1] < _MIN_TILE_DIM:
            return False, f"harmonics shape too small: {hm.shape}"
        nan_frac = np.isnan(hm).mean()
        if nan_frac >= _MAX_NAN_FRACTION:
            return False, f"harmonics h_mean {nan_frac:.1%} NaN"

    elif product == "hansen":
        for key in ("treecover2000", "lossyear"):
            if key not in d:
                return False, f"missing '{key}' key"
        tc = d["treecover2000"]
        if tc.ndim < 2 or tc.shape[0] < _MIN_TILE_DIM or tc.shape[1] < _MIN_TILE_DIM:
            return False, f"hansen shape too small: {tc.shape}"
        if "datamask" in d and np.all(d["datamask"] == 0):
            return False, "hansen datamask all zeros (no data coverage)"

    else:
        return False, f"unknown product '{product}'"

    return True, "ok"


def _log_pollution_event(product: str, entry: str, reason: str,
                         source: str, action: str = "rejected"):
    """Append a pollution event to the persistent JSONL log.

    Parameters
    ----------
    product : str    e.g. "ndvi", "hansen"
    entry : str      tile identifier (entry name or local filename)
    reason : str     validation failure reason
    source : str     "local_upload", "remote_merge", "zenodo_download"
    action : str     "rejected" (skipped) or "deleted" (removed)
    """
    from datetime import datetime, timezone
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "product": product,
        "entry": entry,
        "reason": reason,
        "source": source,
        "action": action,
    }
    try:
        _POLLUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_POLLUTION_LOG_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError:
        pass


def get_pollution_events(limit: int = 50) -> List[Dict]:
    """Read recent pollution events from the persistent log."""
    if not _POLLUTION_LOG_PATH.exists():
        return []
    try:
        lines = _POLLUTION_LOG_PATH.read_text().strip().splitlines()
        events = []
        for line in lines[-limit:]:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events
    except OSError:
        return []


def get_pollution_summary() -> Dict[str, Any]:
    """Return a summary of all pollution events."""
    events = get_pollution_events(limit=10000)
    if not events:
        return {"total": 0, "by_product": {}, "by_source": {}, "recent": []}
    by_product: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    for ev in events:
        by_product[ev.get("product", "?")] = by_product.get(ev.get("product", "?"), 0) + 1
        by_source[ev.get("source", "?")] = by_source.get(ev.get("source", "?"), 0) + 1
    return {
        "total": len(events),
        "by_product": by_product,
        "by_source": by_source,
        "recent": events[-5:],
    }


# === SECTION: ZIP builder ===

def _build_zip_for_strip(
    product: str,
    strip_south: float,
    strip_north: float,
    files_with_coords: List[Tuple[Path, float, float, float, float, Dict]],
) -> Optional[Path]:
    """Build a ZIP archive containing NPZ tiles for one product + strip.

    Returns path to the temporary ZIP file, or None if no tiles.
    """
    if not files_with_coords:
        return None

    zip_name = _zip_filename(product, strip_south, strip_north)
    zip_path = Path(tempfile.mkdtemp()) / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for local_path, w, s, e, n, extra in files_with_coords:
            entry_name = _npz_entry_name(product, w, s, e, n, **extra)
            zf.write(local_path, entry_name)

    n_tiles = len(files_with_coords)
    size_mb = zip_path.stat().st_size / 1024 / 1024
    log.info("Built %s: %d tiles, %.1f MB", zip_name, n_tiles, size_mb)
    return zip_path


# === SECTION: ZenodoCache main class ===

class _NULL_CTX:
    """No-op context manager. Used by upload_all() when no
    per-ZIP lock factory is supplied."""
    def __enter__(self): return self
    def __exit__(self, *a): return False


class ZenodoCache:
    """Zenodo-backed persistent tile cache.

    Parameters
    ----------
    token : str
        Zenodo API access token.
    base_url : str
        Zenodo API base URL.
    """

    def __init__(self, token: str = ZENODO_TOKEN,
                 base_url: str = ZENODO_BASE_URL):
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.manifest = CacheManifest()
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "srtm-lidar-zenodo-cache/1.0"
        self._zip_indices: Dict[str, ZipIndex] = {}  # zip_name → ZipIndex
        self._missing_zips: set = set()  # negative cache of ZIPs not in manifest
        self._missing_zips_mtime: float = 0.0  # manifest mtime when cache was built

    # --- Zenodo API helpers ---

    def _api(self, method: str, path: str, **kwargs) -> requests.Response:
        """Zenodo API call with retry on transient 5xx / network errors.

        With ~50 peers all flushing tile caches concurrently, Zenodo's
        gateway occasionally responds 502/503/504 (gateway timeout) on
        otherwise-fine deposits. These are transient — the next attempt
        almost always succeeds. We retry up to 4 times with exponential
        backoff (2/4/8/16 s) before letting the exception propagate.
        Non-retryable status codes (4xx other than 429) raise immediately.
        """
        url = f"{self.base_url}{path}"
        kwargs.setdefault("params", {})["access_token"] = self.token
        kwargs.setdefault("timeout", 60)
        last_exc: Optional[Exception] = None
        max_attempts = 7
        for attempt in range(max_attempts):
            try:
                r = self._session.request(method, url, **kwargs)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_exc = requests.HTTPError(
                        f"{r.status_code} {r.reason} for url: {url}",
                        response=r)
                    if attempt < max_attempts - 1:
                        delay = min(2.0 * (2 ** attempt), 300.0)
                        log.info(
                            "Zenodo %s %s → %d, retry %d/%d in %.0fs",
                            method, path, r.status_code,
                            attempt + 1, max_attempts, delay)
                        time.sleep(delay)
                        continue
                r.raise_for_status()
                return r
            except (requests.ConnectionError, requests.Timeout,
                    requests.exceptions.SSLError) as e:
                last_exc = e
                # Drop the wedged session so the retry uses a fresh
                # TCP+TLS handshake. Zenodo's edge sometimes half-closes
                # long-lived TLS streams — reusing the keep-alive socket
                # produces SSLEOFError on every retry.
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = requests.Session()
                self._session.headers["User-Agent"] = "srtm-lidar-zenodo-cache/1.0"
                if attempt < max_attempts - 1:
                    delay = min(2.0 * (2 ** attempt), 300.0)
                    log.info(
                        "Zenodo %s %s network error (%s), retry %d/%d in %.0fs",
                        method, path, str(e)[:80],
                        attempt + 1, max_attempts, delay)
                    time.sleep(delay)
                    continue
        # Exhausted retries
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Zenodo {method} {path}: retries exhausted")

    def _ensure_deposit(self) -> int:
        """Get or create the cache Zenodo deposit.  Returns depo_id."""
        if self.manifest.depo_id:
            return self.manifest.depo_id

        # Create a new deposit
        meta = {
            "metadata": {
                "title": "SRTM-LiDAR Austria: Copernicus + Hansen Tile Cache",
                "upload_type": "dataset",
                "description": (
                    "Grid-aligned cache tiles for Austrian landscape analysis. "
                    "Contains Copernicus (NDVI, SAR, WorldCover, NDVI harmonics) "
                    "at 0.1° resolution and Hansen Global Forest Change at 0.5° "
                    "resolution. ZIP archives with NPZ entries, one per grid cell."
                ),
                "creators": [{"name": "SRTM-LiDAR Austria"}],
                "access_right": "open",
                "license": "cc-by-4.0",
            }
        }
        r = self._api("POST", "/api/deposit/depositions", json=meta)
        depo = r.json()
        depo_id = depo["id"]
        log.info("Created cache deposit %d", depo_id)

        self.manifest.depo_id = depo_id
        self.manifest.save()
        return depo_id

    def _delete_file(self, depo_id: int, filename: str) -> None:
        """Delete a file from a Zenodo deposit."""
        r_files = self._api("GET", f"/api/deposit/depositions/{depo_id}/files")
        for f in r_files.json():
            if f["filename"] == filename:
                self._api("DELETE",
                          f"/api/deposit/depositions/{depo_id}/files/{f['id']}")
                log.info("Deleted %s from deposit %d", filename, depo_id)
                return
        log.warning("File %s not found in deposit %d", filename, depo_id)

    def _upload_file(self, depo_id: int, local_path: Path, filename: str
                     ) -> Dict:
        """Upload a file to a Zenodo deposit bucket."""
        # Get bucket URL
        r = self._api("GET", f"/api/deposit/depositions/{depo_id}")
        bucket_url = r.json()["links"]["bucket"]

        # Delete existing file with same name (if re-uploading)
        try:
            r_files = self._api("GET", f"/api/deposit/depositions/{depo_id}/files")
            for f in r_files.json():
                if f["filename"] == filename:
                    self._api("DELETE",
                              f"/api/deposit/depositions/{depo_id}/files/{f['id']}")
                    log.info("Deleted existing %s from deposit %d", filename, depo_id)
                    break
        except Exception:
            pass

        # Upload (retry on transient SSL/connection errors).
        # SSLError is a subclass of ConnectionError so it's already covered,
        # but we list it explicitly for clarity. After any network failure
        # we drop the keep-alive session and rebuild it — Zenodo's edge
        # sometimes half-closes TLS streams (manifests as SSLEOFError) and
        # the only reliable recovery is a fresh TCP+TLS handshake.
        last_exc = None
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                with open(local_path, "rb") as fh:
                    r = self._session.put(
                        f"{bucket_url}/{filename}",
                        data=fh,
                        params={"access_token": self.token},
                        timeout=600,
                    )
                    r.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout,
                    requests.exceptions.SSLError,
                    requests.exceptions.ChunkedEncodingError) as e:
                last_exc = e
                # Drop the wedged session before backing off.
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = requests.Session()
                self._session.headers["User-Agent"] = "srtm-lidar-zenodo-cache/1.0"
                wait = min(2 ** attempt * 5, 300)
                log.warning("Upload of %s failed (%s); retrying in %ds (attempt %d/%d)",
                            filename, type(e).__name__, wait,
                            attempt + 1, max_attempts)
                time.sleep(wait)
        else:
            raise last_exc if last_exc else RuntimeError("upload failed")

        result = r.json()
        log.info("Uploaded %s to deposit %d (%.1f MB)",
                 filename, depo_id, local_path.stat().st_size / 1e6)
        return result

    def _file_download_url(self, zip_name: str) -> Optional[str]:
        """Get the download URL for a ZIP in the cache deposit.

        For published records uses /records/{id}/files/{name}.
        For draft deposits uses /api/records/{id}/draft/files/{name}/content
        (requires access_token, which is appended by _authed_session()).
        """
        rid = self.manifest.record_id
        did = self.manifest.depo_id

        if rid:
            # Published record — public URL
            return f"{self.base_url}/records/{rid}/files/{zip_name}"

        if did:
            # Draft deposit — authenticated API URL
            return (f"{self.base_url}/api/records/{did}/draft/files/"
                    f"{zip_name}/content")

        return None

    def _authed_session(self) -> requests.Session:
        """Return a session with the access_token as a default query param."""
        s = requests.Session()
        s.headers["User-Agent"] = "srtm-lidar-zenodo-cache/1.0"
        s.params = {"access_token": self.token}  # type: ignore[assignment]
        return s

    # --- Upload ---

    def upload_all(self, dry_run: bool = False,
                   per_zip_lock=None) -> Dict[str, Any]:
        """Upload all local cache tiles to Zenodo.

        Scans local tile cache dirs, bundles into ZIP archives per
        product + latitude strip, uploads to a single Zenodo deposit.

        Parameters
        ----------
        dry_run : bool
            If True, build ZIPs but don't upload.
        per_zip_lock : callable | None
            Optional context-manager factory invoked once per ZIP
            *immediately around the upload+manifest write*. Lets the
            caller hold the fleet Zenodo upload lease for only the
            short critical section per ZIP (typically 5–30 s)
            instead of the entire batch (which may include many ZIPs
            and SSL-retry storms, blocking other peers for >10 min).
            When None, no lock is taken — caller is assumed to hold
            the lease for the whole batch (legacy behaviour).

        Returns
        -------
        dict with upload statistics.
        """
        reverse_idx = _build_reverse_index()
        if not reverse_idx:
            log.warning("No tile bbox index found — cannot determine tile coordinates")
            return {"error": "no tile bbox index"}

        stats = {"zips_built": 0, "zips_uploaded": 0, "tiles_total": 0,
                 "bytes_total": 0}

        # Group local files by (product, cell). Cell key is the
        # 4-tuple (south, north, west, east) bounding the bundle.
        groups: Dict[Tuple[str, float, float, float, float], List] = {}

        # Copernicus tiles
        cop_files = _scan_local_copernicus()
        for product, paths in cop_files.items():
            for p in paths:
                info = reverse_idx.get(p.name)
                if info is None:
                    continue
                prod, w, s, e, n, extra = info
                cs, cn, cw, ce = _cell_for_bbox(s, w)
                key = (product, cs, cn, cw, ce)
                groups.setdefault(key, []).append((p, w, s, e, n, extra))

        # Hansen tiles
        hansen_files = _scan_local_hansen()
        for p in hansen_files:
            info = reverse_idx.get(p.name)
            if info is None:
                continue
            _, w, s, e, n, extra = info
            cs, cn, cw, ce = _cell_for_bbox(s, w)
            key = ("hansen", cs, cn, cw, ce)
            groups.setdefault(key, []).append((p, w, s, e, n, extra))

        if not groups:
            log.info("No tiles to upload")
            return stats

        depo_id = None if dry_run else self._ensure_deposit()

        # Deposit-fullness guard: Zenodo enforces a hard 100-file cap per
        # deposit. Our _upload_file() cycle is delete-then-PUT, so
        # RE-uploading an existing ZIP is safe even at the cap (net file
        # count 0) — but creating a NEW ZIP when the deposit is full 400s
        # AFTER any local state was primed, and (worse) chkpt bundles had
        # been crowding the deposit so tile ZIPs couldn't grow (Jul 2026:
        # harmonics strips stuck at 1 tile / lost). Count live remote
        # files once per flush and skip creation of new names near cap.
        _live_files = 0
        try:
            _live_files = sum(
                1 for _n, _e in (self.manifest.all_files() or {}).items()
                if (_e or {}).get("size", 0) > 0)
        except Exception:
            pass

        for (product, cs, cn, cw, ce), files in sorted(groups.items()):
            zip_name = _zip_filename(product, cs, cn, cw, ce)
            # Read-side compat: if a legacy strip ZIP already covers
            # these tiles on Zenodo, keep extending it instead of
            # creating a parallel cell ZIP. This avoids re-uploading
            # ~700 MB worth of cached tiles already on Zenodo.
            legacy_name = _legacy_strip_zip_for(product, cs)
            if (self.manifest.get_file(legacy_name)
                    and not self.manifest.get_file(zip_name)):
                zip_name = legacy_name

            # Skip creating brand-new ZIP names when the deposit is near
            # the 100-file cap (see guard comment above the loop).
            if (not dry_run and _live_files >= 90
                    and not self.manifest.get_file(zip_name)):
                log.warning(
                    "Zenodo cache: deposit near file cap (%d live files); "
                    "deferring creation of new ZIP %s", _live_files,
                    zip_name)
                stats["zips_deferred_cap"] = (
                    stats.get("zips_deferred_cap", 0) + 1)
                continue

            # Build set of entry names we have locally, validating each
            local_entries = {}
            for local_path, w, s, e, n, extra in files:
                entry_name = _npz_entry_name(product, w, s, e, n, **extra)
                ok, reason = validate_tile_npz(local_path, product)
                if not ok:
                    stats["tiles_rejected"] = stats.get("tiles_rejected", 0) + 1
                    log.warning("Skipping polluted tile %s (%s): %s",
                                entry_name, local_path.name, reason)
                    _log_pollution_event(product, entry_name, reason,
                                         "local_upload")
                    continue
                local_entries[entry_name] = local_path

            # Check what Zenodo already has for this strip.
            # _get_zip_index returns lazily — list_entries() may HEAD/GET
            # the URL and discover that the manifest is stale (404 because
            # the file was never actually uploaded, or was deleted). We
            # treat that as "remote has nothing" rather than a warning, and
            # drop the stale manifest entry so future flushes don't keep
            # re-probing it.
            existing = self.manifest.get_file(zip_name)
            remote_idx = self._get_zip_index(zip_name) if existing else None
            remote_entry_names: set = set()
            if remote_idx is not None:
                try:
                    remote_entry_names = set(remote_idx.list_entries())
                except requests.HTTPError as he:
                    code = getattr(he.response, "status_code", None)
                    if code == 404:
                        log.info(
                            "Zenodo cache: stale manifest entry for %s (404), "
                            "dropping and uploading fresh", zip_name)
                        try:
                            with self.manifest._lock:
                                self.manifest._data.get("files", {}).pop(
                                    zip_name, None)
                            self.manifest.save()
                        except Exception:
                            pass
                        remote_idx = None
                    else:
                        raise
                except Exception as exc:
                    log.debug("Zenodo cache: cannot list %s: %s", zip_name, exc)
                    remote_idx = None

            # If remote already contains every local entry, skip
            if remote_entry_names and set(local_entries.keys()).issubset(remote_entry_names):
                stats["zips_skipped"] = stats.get("zips_skipped", 0) + 1
                continue

            # Merge: start with local files, then pull remote-only entries
            # so we never lose tiles that were uploaded before eviction.
            merged_count = len(local_entries)
            remote_only = remote_entry_names - set(local_entries.keys())
            remote_data: Dict[str, bytes] = {}  # entry_name → raw NPZ bytes
            if remote_only and remote_idx:
                for rname in remote_only:
                    try:
                        data = remote_idx.read_entry(rname)
                        if data:
                            ok, reason = validate_tile_npz(data, product)
                            if ok:
                                remote_data[rname] = data
                                merged_count += 1
                            else:
                                stats["tiles_rejected"] = stats.get("tiles_rejected", 0) + 1
                                log.warning("Dropping polluted remote tile %s: %s",
                                            rname, reason)
                                _log_pollution_event(product, rname, reason,
                                                     "remote_merge")
                    except Exception as exc:
                        log.debug("Could not fetch remote entry %s: %s", rname, exc)

            # Build merged ZIP
            tmp_dir = Path(tempfile.mkdtemp())
            zip_path = tmp_dir / zip_name
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Local files
                for entry_name, local_path in local_entries.items():
                    zf.write(local_path, entry_name)
                # Remote-only entries
                for entry_name, data in remote_data.items():
                    zf.writestr(entry_name, data)

            size_mb = zip_path.stat().st_size / 1024 / 1024
            log.info("Built %s: %d tiles (%d local + %d remote), %.1f MB",
                     zip_name, merged_count, len(local_entries),
                     len(remote_data), size_mb)

            stats["zips_built"] += 1
            stats["tiles_total"] += merged_count
            stats["bytes_total"] += zip_path.stat().st_size

            if not dry_run:
                # Take the fleet upload lease only for the actual
                # upload+manifest-write critical section (per ZIP).
                # Building the ZIP and merging remote-only tiles is
                # done above without holding the lock, so other peers
                # can proceed between ZIPs.
                _zip_lock_ctx = (per_zip_lock(zip_name)
                                 if per_zip_lock is not None
                                 else _NULL_CTX())
                try:
                    with _zip_lock_ctx:
                        # RACE GUARD (parallel-frontier lost-update).
                        # The remote read + merge above ran OUTSIDE the
                        # global upload lease, so another peer may have
                        # uploaded tiles into this same ZIP while we were
                        # building ours. Overwriting now would clobber
                        # their tiles. Re-read the remote central
                        # directory *under the lease* and append any
                        # entries we don't already carry before the
                        # upload. Harmless with a single active frontier;
                        # load-bearing with N parallel ones (the cache
                        # was flatlining because uploads stomped each
                        # other). The heavy local-zip build stays outside
                        # the lock — only this cheap reconcile (a few
                        # newly-arrived tiles at most) is serialised.
                        try:
                            self.manifest.reload_if_changed()
                            if self.manifest.get_file(zip_name):
                                self._zip_indices.pop(zip_name, None)
                                self._missing_zips.discard(zip_name)
                                fresh_idx = self._get_zip_index(zip_name)
                                if fresh_idx is not None:
                                    # Force re-fetch of the central dir —
                                    # the on-disk idx cache is keyed by
                                    # URL (stable per filename) and is now
                                    # stale if a peer just rewrote the ZIP.
                                    fresh_idx.invalidate()
                                    try:
                                        fresh_names = set(
                                            fresh_idx.list_entries())
                                    except Exception:
                                        fresh_names = set()
                                    have = (set(local_entries)
                                            | set(remote_data))
                                    newly = fresh_names - have
                                    added = 0
                                    if newly:
                                        with zipfile.ZipFile(
                                                zip_path, "a",
                                                zipfile.ZIP_DEFLATED) as zf:
                                            for rname in newly:
                                                try:
                                                    data = fresh_idx.read_entry(
                                                        rname)
                                                except Exception:
                                                    continue
                                                if not data:
                                                    continue
                                                ok, reason = validate_tile_npz(
                                                    data, product)
                                                if not ok:
                                                    stats["tiles_rejected"] = (
                                                        stats.get(
                                                            "tiles_rejected", 0)
                                                        + 1)
                                                    _log_pollution_event(
                                                        product, rname, reason,
                                                        "remote_merge_relock")
                                                    continue
                                                zf.writestr(rname, data)
                                                added += 1
                                    if added:
                                        merged_count += added
                                        stats["tiles_total"] += added
                                        log.info(
                                            "Race guard: merged %d tile(s) "
                                            "uploaded by another peer into %s "
                                            "before overwrite", added, zip_name)
                        except Exception as exc:
                            log.warning(
                                "Race-guard reconcile for %s failed "
                                "(uploading as-is): %s", zip_name, exc)

                        result = self._upload_file(depo_id, zip_path, zip_name)
                        checksum = result.get("checksum", "")
                        from datetime import datetime, timezone
                        self.manifest.set_file(
                            zip_name,
                            url=self._file_download_url(zip_name),
                            size=zip_path.stat().st_size,
                            checksum=checksum,
                            tile_count=merged_count,
                            updated_at=datetime.now(timezone.utc).isoformat(),
                        )
                        self.manifest.save()
                    # Invalidate cached ZipIndex — the file just changed
                    self._zip_indices.pop(zip_name, None)
                    idx_cache = _ZIP_INDEX_CACHE_DIR / f"{hashlib.md5(self.manifest.get_file(zip_name)['url'].encode()).hexdigest()[:12]}.json"
                    idx_cache.unlink(missing_ok=True)
                    stats["zips_uploaded"] += 1
                except Exception as e:
                    log.error("Failed to upload %s: %s", zip_name, e)

            # Clean up temp ZIP
            try:
                zip_path.unlink()
                tmp_dir.rmdir()
            except OSError:
                pass

        log.info("Upload complete: %d ZIPs, %d tiles, %.1f MB",
                 stats["zips_uploaded"], stats["tiles_total"],
                 stats["bytes_total"] / 1e6)
        return stats

    # --- Download (single tile from Zenodo) ---

    def _get_zip_index(self, zip_name: str) -> Optional[ZipIndex]:
        """Get or create a ZipIndex for a remote ZIP.

        Negative-caches ZIPs not listed in the local cache_manifest --
        without this, every cache-only peer would re-probe Zenodo for
        every missing cell ZIP on every KG, triggering hundreds of
        404s/min during the cell rollout.
        """
        if zip_name in self._zip_indices:
            return self._zip_indices[zip_name]
        # Drop the negative cache when the manifest changes (another
        # peer may have uploaded the bundle).
        try:
            mtime = self.manifest._last_mtime
        except AttributeError:
            mtime = 0.0
        if mtime != self._missing_zips_mtime:
            self._missing_zips.clear()
            self._missing_zips_mtime = mtime
        if zip_name in self._missing_zips:
            return None

        # Negative cache: if the manifest doesn't list this ZIP, don't
        # bother probing Zenodo. Manifest sync will populate the entry
        # once another peer uploads the bundle.
        if not self.manifest.get_file(zip_name):
            self._missing_zips.add(zip_name)
            return None

        url = self._file_download_url(zip_name)
        if not url:
            self._missing_zips.add(zip_name)
            return None

        try:
            session = self._authed_session() if not self.manifest.record_id else None
            idx = ZipIndex(url, session=session)
            self._zip_indices[zip_name] = idx
            return idx
        except Exception as e:
            log.debug("Cannot access ZIP %s on Zenodo: %s", zip_name, e)
            self._missing_zips.add(zip_name)
            return None

    def fetch_copernicus(
        self,
        product: str,
        w: float, s: float, e: float, n: float,
        year: int = 2024,
        dest_dir: Path = COP_CACHE_DIR,
    ) -> Optional[Path]:
        """Fetch a single Copernicus tile NPZ from Zenodo.

        Downloads the entry from the appropriate ZIP archive and writes it
        to the local cache directory.  Returns the local path, or None if
        not available on Zenodo.

        Parameters
        ----------
        product : str
            One of: ndvi, sar, worldcover, harmonics
        w, s, e, n : float
            Grid cell bounds (0.1° snapped)
        year : int
            Observation year (used in NPZ entry name)
        dest_dir : Path
            Local cache directory to write to

        Returns
        -------
        Path to the local NPZ file, or None if not found on Zenodo.
        """
        # Pick up manifest changes pushed by the peer-sync thread
        old_depo = self.manifest.depo_id
        self.manifest.reload_if_changed()
        if self.manifest.depo_id != old_depo:
            self._zip_indices.clear()  # URLs changed, invalidate cached indices
            self._missing_zips.clear()

        cs, cn, cw, ce = _cell_for_bbox(s, w)
        candidates = [
            _zip_filename(product, cs, cn, cw, ce),
            *_legacy_strip_zips_for_cell(product, cs),
        ]
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        extra = {"year": year} if product in ("ndvi", "sar", "harmonics") else {}
        entry_name = _npz_entry_name(product, w, s, e, n, **extra)

        idx = None
        for zip_name in candidates:
            idx = self._get_zip_index(zip_name)
            if idx is None:
                continue
            if idx.has_entry(entry_name):
                break
            if product == "worldcover":
                alt = _npz_entry_name(product, w, s, e, n)
                if idx.has_entry(alt):
                    entry_name = alt
                    break
            idx = None
        if idx is None:
            return None

        data = idx.read_entry(entry_name)
        if data is None:
            return None

        # Validate before writing to local cache
        ok, reason = validate_tile_npz(data, product)
        if not ok:
            log.warning("Rejected polluted %s tile from Zenodo (%s): %s",
                        product, entry_name, reason)
            _log_pollution_event(product, entry_name, reason,
                                 "zenodo_download")
            return None

        # Write to local cache with the correct filename
        from tile_cache import tile_key as _tile_key
        if product in ("ndvi", "sar", "harmonics"):
            tk = _tile_key(product, w, s, e, n, year=year)
        else:
            tk = _tile_key(product, w, s, e, n)
        local_name = f"{product}_{tk}.npz"
        local_path = dest_dir / local_name

        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_suffix(".tmp.npz")
        try:
            tmp.write_bytes(data)
            tmp.rename(local_path)
            log.info("Restored %s from Zenodo (%d bytes)",
                     local_name, len(data))
            return local_path
        except Exception as e:
            log.warning("Failed to write %s: %s", local_path, e)
            tmp.unlink(missing_ok=True)
            return None

    def fetch_hansen(
        self,
        w: float, s: float, e: float, n: float,
        dest_dir: Path = HANSEN_CACHE_DIR,
    ) -> Optional[Path]:
        """Fetch a single Hansen tile NPZ from Zenodo.

        Same pattern as fetch_copernicus but for Hansen 0.5° grid.
        """
        # Pick up manifest changes pushed by the peer-sync thread
        old_depo = self.manifest.depo_id
        self.manifest.reload_if_changed()
        if self.manifest.depo_id != old_depo:
            self._zip_indices.clear()
            self._missing_zips.clear()

        cs, cn, cw, ce = _cell_for_bbox(s, w)
        candidates = [
            _zip_filename("hansen", cs, cn, cw, ce),
            *_legacy_strip_zips_for_cell("hansen", cs),
        ]
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]
        entry_name = _npz_entry_name("hansen", w, s, e, n)

        idx = None
        for zip_name in candidates:
            idx = self._get_zip_index(zip_name)
            if idx is not None and idx.has_entry(entry_name):
                break
            idx = None
        if idx is None:
            return None

        data = idx.read_entry(entry_name)
        if data is None:
            return None

        # Validate before writing to local cache
        ok, reason = validate_tile_npz(data, "hansen")
        if not ok:
            log.warning("Rejected polluted hansen tile from Zenodo (%s): %s",
                        entry_name, reason)
            _log_pollution_event("hansen", entry_name, reason,
                                 "zenodo_download")
            return None

        from tile_cache import tile_key as _tile_key
        tk = _tile_key("hansen", w, s, e, n)
        local_name = f"hansen_{tk}.npz"
        local_path = dest_dir / local_name

        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_suffix(".tmp.npz")
        try:
            tmp.write_bytes(data)
            tmp.rename(local_path)
            log.info("Restored %s from Zenodo (%d bytes)",
                     local_name, len(data))
            return local_path
        except Exception as e:
            log.warning("Failed to write %s: %s", local_path, e)
            tmp.unlink(missing_ok=True)
            return None

    # --- Strip (remove faulty tiles from Zenodo ZIPs) ---

    def strip_tiles(self, entries_to_remove: List[str],
                    dry_run: bool = False) -> Dict[str, Any]:
        """Remove specific NPZ entries from Zenodo ZIP archives.

        Rebuilds affected ZIPs without the listed entries and re-uploads.
        Also deletes corresponding local .npz and .tif cache files.

        Parameters
        ----------
        entries_to_remove : list of str
            Entry names like ``ndvi_48.3000_15.5000_48.4000_15.6000_2024.npz``
            OR grid specs like ``ndvi:15.5,48.3,15.6,48.4`` (product:w,s,e,n).
        dry_run : bool
            If True, list what would be removed without changing anything.

        Returns
        -------
        dict with strip statistics.
        """
        # Normalise entries: convert grid specs to entry names
        resolved: List[str] = []
        for spec in entries_to_remove:
            if ":" in spec and "," in spec:
                parts = spec.split(":", 1)
                product = parts[0]
                coords = parts[1].split(",")
                if len(coords) == 4:
                    w, s, e, n = (float(c) for c in coords)
                    extra = {"year": 2024} if product in ("ndvi", "sar", "harmonics") else {}
                    entry = _npz_entry_name(product, w, s, e, n, **extra)
                    resolved.append(entry)
                else:
                    resolved.append(spec)
            else:
                resolved.append(spec)

        to_remove_set = set(resolved)
        log.info("Strip request: %d entries to remove%s",
                 len(to_remove_set), " (dry run)" if dry_run else "")
        for e in sorted(to_remove_set):
            log.info("  → %s", e)

        stats = {"entries_found": 0, "zips_rebuilt": 0,
                 "local_deleted": 0, "dry_run": dry_run}

        # Group entries by ZIP. Probe both the new cell ZIP and the
        # legacy strip ZIP -- old uploads still live under the strip
        # name and we must rebuild the right container.
        zip_entries: Dict[str, List[str]] = {}  # zip_name → [entry_names]
        for entry in to_remove_set:
            base = entry.rsplit(".", 1)[0]
            parts = base.split("_")
            product = parts[0]
            try:
                s_val = float(parts[1])
                w_val = float(parts[2])
            except (IndexError, ValueError):
                log.warning("Cannot parse entry name: %s", entry)
                continue
            cs, cn, cw, ce = _cell_for_bbox(s_val, w_val)
            cell_name = _zip_filename(product, cs, cn, cw, ce)
            legacy_name = _legacy_strip_zip_for(product, cs)
            target = (legacy_name if self.manifest.get_file(legacy_name)
                                       and not self.manifest.get_file(cell_name)
                      else cell_name)
            zip_entries.setdefault(target, []).append(entry)

        depo_id = None if dry_run else self._ensure_deposit()

        for zip_name, remove_list in sorted(zip_entries.items()):
            idx = self._get_zip_index(zip_name)
            if not idx:
                log.warning("ZIP %s not found on Zenodo — skipping", zip_name)
                continue

            all_entries = set(idx.list_entries())
            found = set(remove_list) & all_entries
            if not found:
                log.info("ZIP %s: none of the target entries found", zip_name)
                continue

            keep = all_entries - found
            stats["entries_found"] += len(found)
            log.info("ZIP %s: removing %d/%d entries, keeping %d",
                     zip_name, len(found), len(all_entries), len(keep))
            for e in sorted(found):
                log.info("  removing: %s", e)

            if dry_run:
                continue

            if not keep:
                # ZIP would be empty — delete it
                try:
                    self._delete_file(depo_id, zip_name)
                    log.info("Deleted now-empty %s from deposit", zip_name)
                except Exception as exc:
                    log.error("Failed to delete %s: %s", zip_name, exc)
                stats["zips_rebuilt"] += 1
                continue

            # Rebuild ZIP with only kept entries
            tmp_dir = Path(tempfile.mkdtemp())
            zip_path = tmp_dir / zip_name
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for entry_name in sorted(keep):
                    try:
                        data = idx.read_entry(entry_name)
                        if data:
                            zf.writestr(entry_name, data)
                    except Exception as exc:
                        log.warning("Failed to read %s: %s", entry_name, exc)

            size_mb = zip_path.stat().st_size / 1024 / 1024
            log.info("Rebuilt %s: %d entries, %.1f MB", zip_name, len(keep), size_mb)

            try:
                result = self._upload_file(depo_id, zip_path, zip_name)
                checksum = result.get("checksum", "")
                from datetime import datetime, timezone
                self.manifest.set_file(
                    zip_name,
                    url=self._file_download_url(zip_name),
                    size=zip_path.stat().st_size,
                    checksum=checksum,
                    tile_count=len(keep),
                    updated_at=datetime.now(timezone.utc).isoformat(),
                )
                self.manifest.save()
                self._zip_indices.pop(zip_name, None)
                stats["zips_rebuilt"] += 1
            except Exception as exc:
                log.error("Failed to upload rebuilt %s: %s", zip_name, exc)

            try:
                zip_path.unlink()
                tmp_dir.rmdir()
            except OSError:
                pass

        # Delete matching local cache files
        for entry in to_remove_set:
            # Local npz in tile_cache dir
            base = entry.rsplit(".", 1)[0]
            parts = base.split("_")
            product = parts[0]

            # Find and delete local npz files matching this entry
            cache_dir = COP_CACHE_DIR if product != "hansen" else HANSEN_CACHE_DIR
            for f in cache_dir.glob(f"{product}_*.npz"):
                if f.name == entry or _npz_matches_entry(f, entry, product):
                    if dry_run:
                        log.info("Would delete local: %s", f)
                    else:
                        f.unlink(missing_ok=True)
                        log.info("Deleted local: %s", f)
                    stats["local_deleted"] += 1

        return stats

    def strip_local_tif(self, pattern: str, dry_run: bool = False) -> int:
        """Delete local .tif copernicus cache files matching a pattern.

        Parameters
        ----------
        pattern : str
            Glob pattern, e.g. ``ndvi_ts_v2_7579042d*.tif``
        dry_run : bool
            If True, just list files without deleting.

        Returns count of deleted files.
        """
        import copernicus
        cache_dir = Path(copernicus.CACHE_DIR)
        deleted = 0
        for f in cache_dir.glob(pattern):
            if dry_run:
                log.info("Would delete: %s", f)
            else:
                f.unlink(missing_ok=True)
                log.info("Deleted: %s", f)
            deleted += 1
        # Also check _batch dirs
        for d in cache_dir.glob(pattern.replace(".tif", "_batch")):
            if d.is_dir():
                import shutil
                if dry_run:
                    log.info("Would delete dir: %s", d)
                else:
                    shutil.rmtree(d, ignore_errors=True)
                    log.info("Deleted dir: %s", d)
                deleted += 1
        return deleted

    # --- Status ---

    def status(self) -> Dict[str, Any]:
        """Return cache status for API consumption."""
        local_cop = _scan_local_copernicus()
        local_hansen = _scan_local_hansen()
        return {
            "depo_id": self.manifest.depo_id,
            "record_id": self.manifest.record_id,
            "zenodo_files": len(self.manifest.all_files()),
            "zenodo_tiles": self.manifest.tile_count(),
            "zenodo_size_bytes": sum(f.get("size", 0) for f in self.manifest.all_files().values()),
            "local_copernicus": {k: len(v) for k, v in local_cop.items()},
            "local_hansen": len(local_hansen),
            "pollution": get_pollution_summary(),
        }


def _npz_matches_entry(local_path: Path, entry_name: str, product: str) -> bool:
    """Check if a local NPZ file corresponds to a Zenodo entry name.

    Local files use tile_key hashes, entry names use coordinates.
    Match via .meta.json sidecar if available.
    """
    meta_path = local_path.with_suffix(".npz.meta.json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            w, s_, e, n = meta["west"], meta["south"], meta["east"], meta["north"]
            extra = {k: v for k, v in meta.items()
                     if k not in ("product", "west", "south", "east", "north")}
            reconstructed = _npz_entry_name(product, w, s_, e, n, **extra)
            return reconstructed == entry_name
        except Exception:
            pass
    return False


# === SECTION: Convenience / CLI ===

def _default_cache() -> ZenodoCache:
    return ZenodoCache()


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")

    parser = argparse.ArgumentParser(description="Zenodo tile cache manager")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Show cache status")
    sub.add_parser("upload", help="Upload local tiles to Zenodo")
    sub.add_parser("dry-run", help="Build ZIPs without uploading")

    p_strip = sub.add_parser("strip", help="Remove faulty tiles from Zenodo ZIPs + local cache")
    p_strip.add_argument("entries", nargs="+",
                         help="Entry names (e.g. ndvi_48.3000_15.5000_48.4000_15.6000_2024.npz) "
                              "or grid specs (e.g. ndvi:15.5,48.3,15.6,48.4 = product:w,s,e,n)")
    p_strip.add_argument("--dry-run", action="store_true", help="Show what would be removed")
    p_strip.add_argument("--local-tif", action="append", default=[],
                         help="Also delete local .tif cache files matching this glob "
                              "(e.g. ndvi_ts_v2_7579042d*.tif)")

    p_validate = sub.add_parser("validate", help="Validate local tile cache files")
    p_validate.add_argument("--product", choices=COP_PRODUCTS + ("hansen",),
                            help="Only check this product (default: all)")
    p_validate.add_argument("--delete", action="store_true",
                            help="Delete polluted tiles")

    p_list = sub.add_parser("list", help="List entries in a Zenodo ZIP")
    p_list.add_argument("zip_name", help="ZIP filename (e.g. copernicus_ndvi_strip_48.0_48.5.zip)")

    args = parser.parse_args()
    cache = _default_cache()

    if args.cmd == "status":
        import pprint
        pprint.pprint(cache.status())
    elif args.cmd == "upload":
        result = cache.upload_all(dry_run=False)
        print(json.dumps(result, indent=2))
    elif args.cmd == "dry-run":
        result = cache.upload_all(dry_run=True)
        print(json.dumps(result, indent=2))
    elif args.cmd == "strip":
        result = cache.strip_tiles(args.entries, dry_run=args.dry_run)
        # Also handle --local-tif patterns
        for pat in args.local_tif:
            n = cache.strip_local_tif(pat, dry_run=args.dry_run)
            result[f"local_tif_{pat}"] = n
        print(json.dumps(result, indent=2))
    elif args.cmd == "validate":
        products = [args.product] if args.product else list(COP_PRODUCTS) + ["hansen"]
        bad, good = 0, 0
        for product in products:
            if product == "hansen":
                files = list(HANSEN_CACHE_DIR.glob("hansen_*.npz")) if HANSEN_CACHE_DIR.exists() else []
            else:
                files = list(COP_CACHE_DIR.glob(f"{product}_*.npz")) if COP_CACHE_DIR.exists() else []
            for f in sorted(files):
                ok, reason = validate_tile_npz(f, product)
                if ok:
                    good += 1
                else:
                    bad += 1
                    print(f"  POLLUTED  {f.name}  ({reason})")
                    if args.delete:
                        f.unlink()
                        meta = f.with_name(f.stem + ".meta.json")
                        meta.unlink(missing_ok=True)
                        _log_pollution_event(product, f.name, reason,
                                             "local_validate", action="deleted")
                        print(f"    → deleted")
        print(f"\nTotal: {good} valid, {bad} polluted")
    elif args.cmd == "list":
        idx = cache._get_zip_index(args.zip_name)
        if idx:
            for e in sorted(idx.list_entries()):
                info = idx.entry_info(e)
                size_kb = info.file_size / 1024 if info else 0
                print(f"  {e}  ({size_kb:.0f} KB)")
        else:
            print(f"ZIP {args.zip_name} not found on Zenodo")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
