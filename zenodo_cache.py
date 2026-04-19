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

# Latitude strip height for ZIP bundling
STRIP_HEIGHT = 0.5

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


# === SECTION: Latitude strips ===

def _lat_strips() -> List[Tuple[float, float]]:
    """Return (south, north) bounds for each 0.5° latitude strip covering Austria."""
    strips = []
    s = math.floor(AT_SOUTH / STRIP_HEIGHT) * STRIP_HEIGHT
    while s < AT_NORTH:
        n = s + STRIP_HEIGHT
        strips.append((round(s, 4), round(n, 4)))
        s = n
    return strips


def _strip_for_lat(lat: float) -> Tuple[float, float]:
    """Return the (south, north) strip containing the given latitude."""
    s = math.floor(lat / STRIP_HEIGHT) * STRIP_HEIGHT
    return (round(s, 4), round(s + STRIP_HEIGHT, 4))


def _zip_filename(product: str, strip_south: float, strip_north: float) -> str:
    """Canonical ZIP filename for a product + strip."""
    if product == "hansen":
        return f"hansen_strip_{strip_south:.1f}_{strip_north:.1f}.zip"
    return f"copernicus_{product}_strip_{strip_south:.1f}_{strip_north:.1f}.zip"


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
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Failed to load cache manifest: %s", e)

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


# === SECTION: ZIP index (central directory cache) ===

class ZipIndex:
    """Cached index of entries in a remote ZIP file.

    On first access, reads the ZIP central directory (last ~64KB) and caches
    entry offsets locally.  Subsequent lookups are instant.
    """

    def __init__(self, url: str, cache_dir: Path = _ZIP_INDEX_CACHE_DIR):
        self.url = url
        self._cache_dir = cache_dir
        self._entries: Optional[Dict[str, zipfile.ZipInfo]] = None
        self._cache_key = hashlib.md5(url.encode()).hexdigest()[:12]
        self._index_path = cache_dir / f"{self._cache_key}.json"

    def _load_or_fetch(self) -> Dict[str, zipfile.ZipInfo]:
        """Return dict of entry_name → ZipInfo."""
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
        hrf = HTTPRangeFile(self.url)
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

    def read_entry(self, name: str) -> Optional[bytes]:
        """Read a single entry from the remote ZIP via HTTP range request."""
        entries = self._load_or_fetch()
        zi = entries.get(name)
        if zi is None:
            return None

        # Read the local file header + compressed data
        # Local header is 30 bytes + filename_len + extra_len,
        # followed by compressed data of compress_size bytes.
        hrf = HTTPRangeFile(self.url)
        hrf.seek(zi.header_offset)

        # Read local file header (30 bytes minimum)
        local_header = hrf.read(30)
        if len(local_header) < 30:
            return None

        # Parse local header to get filename_len and extra_len
        sig = struct.unpack("<I", local_header[:4])[0]
        if sig != 0x04034b50:  # PK\x03\x04
            log.warning("Bad local header signature at offset %d", zi.header_offset)
            return None

        fname_len = struct.unpack("<H", local_header[26:28])[0]
        extra_len = struct.unpack("<H", local_header[28:30])[0]

        # Skip filename + extra fields
        hrf.read(fname_len + extra_len)

        # Read compressed data
        compressed = hrf.read(zi.compress_size)

        if zi.compress_type == zipfile.ZIP_STORED:
            return compressed
        elif zi.compress_type == zipfile.ZIP_DEFLATED:
            import zlib
            return zlib.decompress(compressed, -15)
        else:
            # Fallback: download via zipfile (slower, full central dir parse)
            hrf2 = HTTPRangeFile(self.url)
            with zipfile.ZipFile(hrf2) as zf:
                return zf.read(name)


# === SECTION: Inventory (what's in local cache) ===

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

    # --- Zenodo API helpers ---

    def _api(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"
        kwargs.setdefault("params", {})["access_token"] = self.token
        kwargs.setdefault("timeout", 60)
        r = self._session.request(method, url, **kwargs)
        r.raise_for_status()
        return r

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

        # Upload
        with open(local_path, "rb") as fh:
            r = self._session.put(
                f"{bucket_url}/{filename}",
                data=fh,
                params={"access_token": self.token},
                timeout=600,
            )
            r.raise_for_status()

        result = r.json()
        log.info("Uploaded %s to deposit %d (%.1f MB)",
                 filename, depo_id, local_path.stat().st_size / 1e6)
        return result

    def _file_download_url(self, zip_name: str) -> Optional[str]:
        """Get the download URL for a ZIP in the cache deposit."""
        finfo = self.manifest.get_file(zip_name)
        if finfo and finfo.get("url"):
            return finfo["url"]

        # Try record_id-based URL
        rid = self.manifest.record_id or self.manifest.depo_id
        if rid:
            return f"{self.base_url}/records/{rid}/files/{zip_name}"
        return None

    # --- Upload ---

    def upload_all(self, dry_run: bool = False) -> Dict[str, Any]:
        """Upload all local cache tiles to Zenodo.

        Scans local tile cache dirs, bundles into ZIP archives per
        product + latitude strip, uploads to a single Zenodo deposit.

        Parameters
        ----------
        dry_run : bool
            If True, build ZIPs but don't upload.

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

        # Group local files by (product, strip)
        groups: Dict[Tuple[str, float, float], List] = {}

        # Copernicus tiles
        cop_files = _scan_local_copernicus()
        for product, paths in cop_files.items():
            for p in paths:
                info = reverse_idx.get(p.name)
                if info is None:
                    continue
                prod, w, s, e, n, extra = info
                strip_s, strip_n = _strip_for_lat(s)
                key = (product, strip_s, strip_n)
                groups.setdefault(key, []).append((p, w, s, e, n, extra))

        # Hansen tiles
        hansen_files = _scan_local_hansen()
        for p in hansen_files:
            info = reverse_idx.get(p.name)
            if info is None:
                continue
            _, w, s, e, n, extra = info
            strip_s, strip_n = _strip_for_lat(s)
            key = ("hansen", strip_s, strip_n)
            groups.setdefault(key, []).append((p, w, s, e, n, extra))

        if not groups:
            log.info("No tiles to upload")
            return stats

        depo_id = None if dry_run else self._ensure_deposit()

        for (product, strip_s, strip_n), files in sorted(groups.items()):
            zip_name = _zip_filename(product, strip_s, strip_n)

            # Skip upload if Zenodo already has >= as many tiles for this
            # strip.  This prevents overwriting a full ZIP with a smaller
            # one after local cache eviction.
            existing = self.manifest.get_file(zip_name)
            if existing and existing.get("tile_count", 0) >= len(files):
                stats["zips_skipped"] = stats.get("zips_skipped", 0) + 1
                continue

            zip_path = _build_zip_for_strip(product, strip_s, strip_n, files)
            if zip_path is None:
                continue

            stats["zips_built"] += 1
            stats["tiles_total"] += len(files)
            stats["bytes_total"] += zip_path.stat().st_size

            if not dry_run:
                try:
                    result = self._upload_file(depo_id, zip_path, zip_name)
                    checksum = result.get("checksum", "")
                    from datetime import datetime, timezone
                    self.manifest.set_file(
                        zip_name,
                        url=f"{self.base_url}/records/{depo_id}/files/{zip_name}",
                        size=zip_path.stat().st_size,
                        checksum=checksum,
                        tile_count=len(files),
                        updated_at=datetime.now(timezone.utc).isoformat(),
                    )
                    self.manifest.save()
                    stats["zips_uploaded"] += 1
                except Exception as e:
                    log.error("Failed to upload %s: %s", zip_name, e)

            # Clean up temp ZIP
            try:
                zip_path.unlink()
                zip_path.parent.rmdir()
            except OSError:
                pass

        log.info("Upload complete: %d ZIPs, %d tiles, %.1f MB",
                 stats["zips_uploaded"], stats["tiles_total"],
                 stats["bytes_total"] / 1e6)
        return stats

    # --- Download (single tile from Zenodo) ---

    def _get_zip_index(self, zip_name: str) -> Optional[ZipIndex]:
        """Get or create a ZipIndex for a remote ZIP."""
        if zip_name in self._zip_indices:
            return self._zip_indices[zip_name]

        url = self._file_download_url(zip_name)
        if not url:
            return None

        try:
            idx = ZipIndex(url)
            self._zip_indices[zip_name] = idx
            return idx
        except Exception as e:
            log.debug("Cannot access ZIP %s on Zenodo: %s", zip_name, e)
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
        strip_s, strip_n = _strip_for_lat(s)
        zip_name = _zip_filename(product, strip_s, strip_n)

        extra = {"year": year} if product in ("ndvi", "sar", "harmonics") else {}
        entry_name = _npz_entry_name(product, w, s, e, n, **extra)

        idx = self._get_zip_index(zip_name)
        if idx is None:
            return None

        if not idx.has_entry(entry_name):
            # Try without year for worldcover
            if product == "worldcover":
                entry_name = _npz_entry_name(product, w, s, e, n)
                if not idx.has_entry(entry_name):
                    return None
            else:
                return None

        data = idx.read_entry(entry_name)
        if data is None:
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
        strip_s, strip_n = _strip_for_lat(s)
        zip_name = _zip_filename("hansen", strip_s, strip_n)
        entry_name = _npz_entry_name("hansen", w, s, e, n)

        idx = self._get_zip_index(zip_name)
        if idx is None:
            return None

        if not idx.has_entry(entry_name):
            return None

        data = idx.read_entry(entry_name)
        if data is None:
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
            "local_copernicus": {k: len(v) for k, v in local_cop.items()},
            "local_hansen": len(local_hansen),
        }


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
    p_dry = sub.add_parser("dry-run", help="Build ZIPs without uploading")

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
    else:
        parser.print_help()
        sys.exit(1)
