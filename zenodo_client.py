"""Zenodo client — Python port of github.com/raffopenssh/zenodo-mirror-go-pkg.

Thread-safe manifest, retry with exponential backoff, atomic file writes,
MD5 verification on download.

Usage::

    from zenodo_client import Client, Manifest, default_metadata

    manifest = Manifest("zenodo_manifest.json")
    client = Client(token="...")
    client.upload(
        key="my-dataset",
        local_path="/tmp/data.tif",
        version="1.0.0",
        meta_func=default_metadata,
        manifest=manifest,
    )
    local = client.download("my-dataset", "/tmp/downloads", manifest)
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

__all__ = [
    "Entry",
    "Manifest",
    "Client",
    "default_metadata",
    "landscape_metadata",
    "DEFAULT_BASE_URL",
    "DEFAULT_TOKEN",
]

log = logging.getLogger(__name__)

# === SECTION: Constants ===

DEFAULT_BASE_URL = "https://zenodo.org"
DEFAULT_TOKEN = "2dnLSA2YYTc8jt3a1X0qDZUBb1hyOIpGJ44UoJr8N69wdePODgq4cjbJ0DJa"

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


# === SECTION: Entry dataclass ===


@dataclass
class Entry:
    """A single record in the manifest – tracks one Zenodo deposition file."""

    key: str = ""
    depo_id: int = 0
    bucket_url: str = ""
    filename: str = ""
    size: int = 0
    checksum: str = ""  # "md5:<hex>"
    uploaded_at: str = ""  # ISO-8601
    version: str = ""

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Entry":
        return cls(
            key=d.get("key", ""),
            depo_id=int(d.get("depo_id", 0)),
            bucket_url=d.get("bucket_url", ""),
            filename=d.get("filename", ""),
            size=int(d.get("size", 0)),
            checksum=d.get("checksum", ""),
            uploaded_at=d.get("uploaded_at", ""),
            version=d.get("version", ""),
        )


# === SECTION: Manifest (thread-safe, file-backed JSON registry) ===


class Manifest:
    """Thread-safe, file-backed JSON registry of :class:`Entry` objects.

    JSON on disk::

        {
            "entries": {
                "some-key": { ... entry fields ... },
                ...
            }
        }
    """

    def __init__(self, file_path: str | Path) -> None:
        self._path = Path(file_path)
        self._lock = threading.Lock()
        self._entries: Dict[str, Entry] = {}
        self._load()

    # -- private helpers -----------------------------------------------------

    def _load(self) -> None:
        """Load manifest from disk.  If the file doesn't exist, start empty."""
        if not self._path.exists():
            log.debug("Manifest file %s does not exist — starting empty", self._path)
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            entries_raw = raw.get("entries", {})
            for k, v in entries_raw.items():
                v.setdefault("key", k)
                self._entries[k] = Entry.from_dict(v)
            log.info("Loaded manifest with %d entries from %s", len(self._entries), self._path)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load manifest from %s: %s — starting empty", self._path, exc)
            self._entries = {}

    # -- public API ----------------------------------------------------------

    def get(self, key: str) -> Optional[Entry]:
        """Return a *copy* of the entry for *key*, or ``None``."""
        with self._lock:
            entry = self._entries.get(key)
            return copy.deepcopy(entry) if entry is not None else None

    def set(self, key: str, entry: Entry) -> None:  # noqa: A003 — mirrors Go API name
        """Insert or replace an entry."""
        with self._lock:
            self._entries[key] = copy.deepcopy(entry)

    def delete(self, key: str) -> None:
        """Remove an entry (no-op if absent)."""
        with self._lock:
            self._entries.pop(key, None)

    def save(self) -> None:
        """Persist to disk via atomic write (write .tmp then rename)."""
        with self._lock:
            data = {
                "entries": {k: v.to_dict() for k, v in self._entries.items()},
            }
        # Write to a temp file in the same directory so rename is atomic on
        # the same filesystem.
        parent = self._path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp", prefix=".manifest_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_path, self._path)
            log.debug("Manifest saved to %s", self._path)
        except BaseException:
            # Clean up the temp file on any failure.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def keys(self) -> List[str]:
        """Return sorted list of manifest keys."""
        with self._lock:
            return sorted(self._entries.keys())

    def entries(self) -> Dict[str, Entry]:
        """Return a shallow copy of the entries dict (values are deep-copied)."""
        with self._lock:
            return {k: copy.deepcopy(v) for k, v in self._entries.items()}

    def for_each(self, fn: Callable[[str, Entry], None]) -> None:
        """Call *fn(key, entry_copy)* for every entry."""
        snapshot = self.entries()
        for k, e in snapshot.items():
            fn(k, e)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries

    def __repr__(self) -> str:
        return f"Manifest(path={self._path!r}, entries={len(self)})"


# === SECTION: Metadata helpers (Zenodo deposit metadata) ===

#: Type alias for the callable that produces Zenodo metadata JSON.
MetadataFunc = Callable[[str, str, str], Dict[str, Any]]


def default_metadata(key: str, filename: str, version: str) -> Dict[str, Any]:
    """Return a minimal Zenodo metadata dict suitable for most uploads.

    Parameters
    ----------
    key:
        The manifest key (used in the title).
    filename:
        Name of the file being uploaded.
    version:
        Semantic version string.
    """
    return {
        "metadata": {
            "title": f"{key} ({filename})",
            "upload_type": "dataset",
            "description": f"Auto-mirrored file <code>{filename}</code> for key <code>{key}</code>, version {version}.",
            "creators": [{"name": "Automated Mirror"}],
            "access_right": "open",
            "license": "cc-by-4.0",
            "version": version,
        }
    }


def landscape_metadata(
    kg_code: str,
    kg_name: str,
    version: str,
    file_type: str = "GeoTIFF",
    quality_score: float = 0.0,
    quality_grade: str = "",
    available_layers: list = None,
    missing_layers: list = None,
) -> Dict[str, Any]:
    """Zenodo metadata for an Austrian landscape / cadastral unit upload.

    Parameters
    ----------
    kg_code:
        Cadastral municipality code (e.g. ``"01201"``).
    kg_name:
        Human-readable name (e.g. ``"Innere Stadt"``).
    version:
        Version tag, typically a date like ``"2024-09-15"``.
    file_type:
        Description of file format.
    quality_score:
        Data quality score 0.0–1.0.
    quality_grade:
        Quality grade letter (A/B/C/D/F).
    available_layers:
        List of data layers available for all tiles.
    missing_layers:
        List of data layers missing from all tiles.
    """
    avail = available_layers or []
    miss = missing_layers or []

    quality_line = ""
    if quality_grade:
        quality_line = (
            f"\n<b>Data Quality:</b> {quality_score:.0%} (Grade {quality_grade})."
        )
        if avail:
            quality_line += f"\nAvailable layers: {', '.join(avail)}."
        if miss:
            quality_line += f"\nMissing layers: {', '.join(miss)}."

    return {
        "metadata": {
            "title": f"Austria Landscape \u2014 KG {kg_code} {kg_name} ({file_type})",
            "upload_type": "dataset",
            "description": (
                f"Landscape classification for Austrian cadastral unit "
                f"<b>{kg_code} {kg_name}</b>.\n"
                f"Contains: segmentation raster/vectors, DTM/DSM/nDSM, cadastre parcels "
                f"with elevation, buildings with heights, new building detections, "
                f"infrastructure analysis, terrain stats, NDVI, SAR, phenology.\n"
                f"Format: {file_type}, version {version}."
                f"{quality_line}\n"
                f"Derived from BEV ALS LIDAR (DTM/DSM), Basemap.at orthophoto, "
                f"Sentinel-2 NDVI, Sentinel-1 SAR, ESA WorldCover, Hansen GFC."
            ),
            "creators": [{"name": "SRTM-LIDAR Pipeline"}],
            "access_right": "restricted",
            "access_conditions": "Contact the dataset creator for access.",
            "version": version,
            "keywords": [
                "austria",
                "landscape",
                "lidar",
                "dtm",
                "dsm",
                "segmentation",
                "cadastre",
                kg_code,
                kg_name,
            ],
        }
    }


# === SECTION: Client (Zenodo REST API) ===


class ZenodoError(Exception):
    """Raised on non-retryable Zenodo API errors.

    ``retry_after`` (seconds) is set when Zenodo returns a Retry-After
    header on the failing response.  Callers (peer director, austria
    processor) honour it for cooldown sizing, falling back to their own
    defaults when it's 0/missing.
    """

    def __init__(self, message: str, status_code: int = 0, body: str = "",
                 retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after


class ReadOnlyError(ZenodoError):
    """Raised when a mutating method is called on a read-only client."""


# 4 MB read chunks — http.client.HTTPConnection.send() reads file-like
# objects in blocksize (default 8 KB) chunks.  Each chunk goes through
# TLS encrypt + sendall, so 8 KB × 183K calls for a 1.5 GB file is
# extremely slow and can cause Zenodo to close the connection mid-stream
# (SSL EOF).  By overriding __len__ we let requests set Content-Length,
# and by buffering reads into 4 MB blocks we reduce TLS overhead ~500×.
_UPLOAD_CHUNK = 4 * 1024 * 1024  # 4 MB


def _md5_of_file(local_path: "Path") -> str:
    """Stream-MD5 a local file in 4 MB chunks."""
    h = hashlib.md5()
    with open(local_path, "rb") as fh:
        while True:
            chunk = fh.read(_UPLOAD_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class ProgressFileWrapper:
    """Wrap a file object to track read progress during uploads.

    Also acts as a ``__len__``-able object so ``requests`` sets
    ``Content-Length`` without relying on ``fileno()``/``fstat``.
    """

    def __init__(self, fh, total_size: int, callback=None, callback_interval: float = 0.5):
        self._fh = fh
        self._total = total_size
        self._sent = 0
        self._callback = callback
        self._interval = callback_interval
        self._last_cb = 0.0

    # --- size introspection for requests/urllib3 ---
    def __len__(self) -> int:
        return self._total

    def read(self, size=-1):
        # Force large reads even when http.client asks for 8 KB.
        if 0 < size < _UPLOAD_CHUNK:
            size = _UPLOAD_CHUNK
        data = self._fh.read(size)
        if data:
            self._sent += len(data)
            now = time.time()
            if self._callback and (now - self._last_cb >= self._interval):
                self._callback(self._sent, self._total)
                self._last_cb = now
        return data

    def seek(self, *args, **kwargs):
        self._sent = 0
        return self._fh.seek(*args, **kwargs)

    def tell(self):
        return self._fh.tell()

    def __getattr__(self, name):
        return getattr(self._fh, name)


class Client:
    """Zenodo API client with retry, backoff, and manifest integration.

    Parameters
    ----------
    token:
        Zenodo personal access token.
    base_url:
        Zenodo instance URL (default: ``https://zenodo.org``).
    max_retries:
        Maximum number of retries on transient failures.
    retry_base_wait:
        Base wait in seconds for exponential backoff (doubled each retry).
    read_only:
        If ``True``, mutating operations (:meth:`upload`, :meth:`delete_deposition`,
        etc.) raise :class:`ReadOnlyError`.
    """

    def __init__(
        self,
        token: str = DEFAULT_TOKEN,
        base_url: str = DEFAULT_BASE_URL,
        max_retries: int = 5,
        retry_base_wait: float = 4.0,
        read_only: bool = False,
    ) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_base_wait = retry_base_wait
        self.read_only = read_only

        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.token}"})

    def _reset_session(self) -> None:
        """Close the underlying HTTP session and start a fresh one.

        Called after a network/SSL failure so the next retry uses a brand-new
        TCP+TLS connection instead of reusing a wedged keep-alive socket.
        Zenodo's edge sometimes half-closes long-running TLS streams, which
        manifests as SSLEOFError; the only reliable recovery is a new
        handshake.
        """
        try:
            self._session.close()
        except Exception:
            pass
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.token}"})

    # -- internal helpers ----------------------------------------------------

    def _assert_writable(self) -> None:
        if self.read_only:
            raise ReadOnlyError("Client is in read-only mode")

    def _url(self, path: str) -> str:
        """Build absolute URL from a relative API path."""
        return f"{self.base_url}{path}"

    def _do_request(
        self,
        method: str,
        url: str,
        *,
        data: Any = None,
        json_body: Any = None,
        content_type: Optional[str] = None,
        stream: bool = False,
        max_retries: Optional[int] = None,
    ) -> requests.Response:
        """Execute an HTTP request with retry + exponential backoff on 5xx/429.

        Parameters
        ----------
        method:
            HTTP method (``GET``, ``POST``, ``PUT``, ``DELETE``, ``HEAD``).
        url:
            Fully qualified URL.
        data:
            Raw body bytes or file-like object.
        json_body:
            JSON-serialisable body (mutually exclusive with *data*).
        content_type:
            Explicit Content-Type header override.
        stream:
            If ``True``, the response body is not immediately downloaded.

        Returns
        -------
        requests.Response
            The successful response.

        Raises
        ------
        ZenodoError
            On non-retryable HTTP errors (4xx other than 429).
        requests.exceptions.RequestException
            On network-level failures after all retries are exhausted.
        """
        headers: Dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = content_type

        last_exc: Optional[BaseException] = None
        attempt = 0
        _max_retries = self.max_retries if max_retries is None else max_retries

        while True:
            attempt += 1
            try:
                log.debug("%s %s (attempt %d/%d)", method, url, attempt, _max_retries + 1)
                resp = self._session.request(
                    method,
                    url,
                    data=data,
                    json=json_body,
                    headers=headers,
                    stream=stream,
                    # 30 s connect, 10 min read/upload. The previous 2 h
                    # read timeout meant a half-closed TLS stream wedged
                    # us for 2 h before retrying — catastrophic for
                    # large GPKG PUTs that already burned bandwidth
                    # uploading the body. 10 min is generous: Zenodo
                    # acks within ~30 s once the body is on disk, and
                    # the post-upload bucket verification below recovers
                    # the common case where the body landed but the
                    # response was dropped (SSLEOFError / write timeout).
                    timeout=(30, 600),
                )

                # Success
                if resp.status_code < 400:
                    return resp

                # Retryable
                if resp.status_code in _RETRYABLE_STATUS:
                    body_preview = resp.text[:500] if not stream else "<stream>"
                    # First attempt logs INFO so a single transient flake
                    # on a small PUT doesn't feed the fleet throttle.
                    # From attempt 2 onward we escalate to WARNING so
                    # sustained Zenodo pushback (especially during big
                    # uploads) actually registers in austria_processor's
                    # zenodo warning-rate bucket and dampens fleet
                    # parallelism. Without this, 59 peers keep hammering
                    # Zenodo through a partial outage and multi-GB PUTs
                    # die with write-timeouts. Final exhaustion is also
                    # WARNING (handled at the `raise last_exc` site).
                    is_repeated = attempt >= 2
                    (log.warning if is_repeated else log.info)(
                        "%s %s returned %d (attempt %d/%d): %s",
                        method, url, resp.status_code, attempt,
                        _max_retries + 1, body_preview,
                    )
                    # Parse Retry-After once and stash it on the exception
                    # so callers can size their own cooldowns from it.
                    retry_after_secs = 0.0
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            retry_after_secs = float(retry_after)
                        except ValueError:
                            pass
                    last_exc = ZenodoError(
                        f"HTTP {resp.status_code}",
                        status_code=resp.status_code,
                        body=body_preview,
                        retry_after=retry_after_secs,
                    )
                    if attempt <= _max_retries:
                        wait = self.retry_base_wait * (2 ** (attempt - 1))
                        if retry_after_secs > 0:
                            wait = max(wait, retry_after_secs)
                        log.info("Retrying in %.1fs …", wait)
                        time.sleep(wait)
                        # Reset seekable data for retry.
                        if hasattr(data, "seek"):
                            data.seek(0)
                        continue
                    # Exhausted retries.
                    raise last_exc

                # Non-retryable client error (4xx except 429)
                body_text = resp.text[:1000] if not stream else "<stream>"
                raise ZenodoError(
                    f"HTTP {resp.status_code}: {body_text}",
                    status_code=resp.status_code,
                    body=body_text,
                )

            except requests.exceptions.RequestException as exc:
                # Same WARNING-from-attempt-2 split as the 5xx/429 path:
                # one blip is noise, repeated blips on the same request
                # are real upstream stress and should feed the throttle.
                is_repeated = attempt >= 2
                (log.warning if is_repeated else log.info)(
                    "%s %s network error (attempt %d/%d): %s",
                    method, url, attempt, _max_retries + 1, exc,
                )
                last_exc = exc
                # SSL/connection-reset failures often leave the keep-alive
                # socket in a half-closed state — the next retry on the
                # same Session will fail immediately with the same error.
                # Drop the session so the retry uses a fresh TCP+TLS
                # handshake. Costs ~100ms; saves the upload.
                # Reset session on any failure that may have left the
                # underlying socket in a half-closed state. Notably
                # includes ReadTimeout / Timeout: a multi-GB PUT that
                # times out leaves a dead TLS stream; reusing it gives
                # an instant 'write operation timed out' on the next
                # attempt. Cheap insurance (~100ms handshake).
                if isinstance(exc, (requests.exceptions.SSLError,
                                    requests.exceptions.ConnectionError,
                                    requests.exceptions.ChunkedEncodingError,
                                    requests.exceptions.ReadTimeout,
                                    requests.exceptions.Timeout)):
                    self._reset_session()
                if attempt <= _max_retries:
                    wait = self.retry_base_wait * (2 ** (attempt - 1))
                    # Cap at 5 minutes — long enough for Zenodo to recover
                    # from a transient outage, short enough to keep the
                    # processor responsive.
                    wait = min(wait, 300.0)
                    log.info("Retrying in %.1fs …", wait)
                    time.sleep(wait)
                    if hasattr(data, "seek"):
                        data.seek(0)
                    continue
                raise

    # -- internal: bucket file PUT (single or multipart) ---------------------

    # InvenioRDM/Zenodo multipart bucket API uses query params on the
    # bucket file URL:
    #   POST   {bucket}/{key}?uploads=&size=N&part_size=P  -> {upload_id}
    #   PUT    {bucket}/{key}?uploadId=U&partNumber=N      (one per part)
    #   POST   {bucket}/{key}?uploadId=U                   (complete)
    #   DELETE {bucket}/{key}?uploadId=U                   (abort)
    # Sandbox/old Zenodo may not support multipart — we fall back to
    # a single PUT on any 4xx other than 404.
    _MULTIPART_PART_SIZE = 256 * 1024 * 1024  # 256 MB

    def _verify_via_deposit(
        self,
        depo_id: int,
        filename: str,
        expected_size: int,
        expected_md5: Optional[str],
    ) -> tuple[bool, str]:
        """Confirm a bucket file via the deposition record's ``files`` array.

        Fallback for :meth:`_verify_bucket_file` when a bare HEAD on the
        bucket URL is rejected. Once a deposition leaves the draft-bucket
        state (and on the production zenodo.org edge generally), HEAD/GET
        on the public ``{bucket}/{filename}`` URL returns 403 even with a
        valid Bearer token — which silently defeated the post-failure
        verify recovery (every write-timeout fell through to a full
        multi-GB re-upload). ``GET /api/deposit/depositions/{id}`` IS
        authorised and embeds ``files: [{filename, filesize, checksum}]``
        where ``checksum`` is a plain 32-hex md5 — exactly what we need.

        Returns ``(ok, reason)``; never raises (verify must stay
        side-effect-free so the caller's retry policy owns control flow).
        """
        try:
            resp = self._session.get(
                self._url(f"/api/deposit/depositions/{depo_id}"),
                timeout=(10, 30),
            )
        except requests.exceptions.RequestException as exc:
            return False, f"depo_error:{type(exc).__name__}"
        if resp.status_code >= 400:
            return False, f"depo_status:{resp.status_code}"
        try:
            files = (resp.json() or {}).get("files") or []
        except Exception:
            return False, "depo_unparseable"
        match = next((f for f in files
                      if f.get("filename") == filename), None)
        if match is None:
            return False, "depo_absent"
        try:
            got_size = int(match.get("filesize", -1))
        except (TypeError, ValueError):
            got_size = -1
        if got_size != expected_size:
            return False, f"depo_size_mismatch:{got_size}!={expected_size}"
        got_md5 = str(match.get("checksum") or "").strip().lower()
        if got_md5.startswith("md5:"):
            got_md5 = got_md5.split(":", 1)[-1]
        if expected_md5 and got_md5:
            if got_md5 == expected_md5.lower():
                return True, f"depo size+md5 ok ({got_size}B)"
            return False, f"depo_md5_mismatch:{got_md5}!={expected_md5}"
        return True, f"depo size ok ({got_size}B, no md5)"

    def _verify_bucket_file(
        self,
        put_url: str,
        expected_size: int,
        expected_md5: Optional[str],
        depo_id: Optional[int] = None,
    ) -> tuple[bool, str]:
        """Check whether the file at ``put_url`` matches our expectations.

        Returns ``(ok, reason)``. Reason is a short human-readable
        token explaining the verdict (used in log lines).

        Zenodo's bucket API responds to HEAD on a file URL with:
          * ``Content-Length`` — always present.
          * ``ETag`` (or ``X-Checksum``) — MD5 of the stored object
            on most InvenioRDM versions. Format is either a quoted
            32-hex hash or ``md5:<hex>``.
        We accept the upload as good when size matches AND we can
        confirm md5 (or, if Zenodo doesn't return a checksum header,
        when size matches and we never had a local md5 to compare).

        This helper is the cornerstone of the post-failure recovery
        path: it lets us treat "SSL EOF after the last byte was
        written" as a success and skip a multi-GB re-upload.
        """
        # Bucket-URL HEAD is the cheap happy path, but on production
        # zenodo.org it returns 403 even with a valid token once the
        # deposit is past draft. When we know the deposition id, fall
        # back to the authorised deposition-record API, which carries
        # the size + md5 we need. ``filename`` is the last path segment
        # of put_url (bucket URLs have no query string).
        filename = put_url.rsplit("/", 1)[-1]
        try:
            resp = self._session.head(
                put_url,
                timeout=(10, 30),
                allow_redirects=True,
            )
        except requests.exceptions.RequestException as exc:
            if depo_id is not None:
                return self._verify_via_deposit(
                    depo_id, filename, expected_size, expected_md5)
            return False, f"head_error:{type(exc).__name__}"
        if resp.status_code == 404:
            # A 404 on the bucket URL is ambiguous on the public edge;
            # only trust "absent" when we can't cross-check via the
            # deposition record.
            if depo_id is not None:
                return self._verify_via_deposit(
                    depo_id, filename, expected_size, expected_md5)
            return False, "absent"
        if resp.status_code >= 400:
            if depo_id is not None:
                return self._verify_via_deposit(
                    depo_id, filename, expected_size, expected_md5)
            return False, f"head_status:{resp.status_code}"
        try:
            got_size = int(resp.headers.get("Content-Length", "-1"))
        except ValueError:
            got_size = -1
        if got_size != expected_size:
            return False, f"size_mismatch:{got_size}!={expected_size}"
        # Try every header that might carry the checksum.
        got_md5 = ""
        for hk in ("ETag", "X-Checksum", "Content-MD5", "Digest"):
            v = resp.headers.get(hk)
            if not v:
                continue
            v = v.strip().strip('"')
            # InvenioRDM emits 'md5:<hex>'; S3-style emits a raw 32-hex.
            if v.lower().startswith("md5:") or v.lower().startswith("md5="):
                v = v.split(":", 1)[-1].split("=", 1)[-1].strip()
            if len(v) == 32 and all(c in "0123456789abcdefABCDEF" for c in v):
                got_md5 = v.lower()
                break
        if expected_md5 and got_md5:
            if got_md5 == expected_md5.lower():
                return True, f"size+md5 ok ({got_size}B)"
            return False, f"md5_mismatch:{got_md5}!={expected_md5}"
        # Size-only confirmation. This is what we get on Zenodo
        # deployments that don't echo a checksum header on HEAD.
        # Still good enough: a partial PUT shows up as size_mismatch.
        return True, (f"size ok ({got_size}B, no md5 hdr)"
                      if not expected_md5
                      else f"size ok ({got_size}B, hdr lacked md5)")

    def _put_file_to_bucket(
        self,
        bucket_url: str,
        filename: str,
        local_path: "Path",
        file_size: int,
        progress_callback,
        *,
        use_multipart: bool,
        depo_id: Optional[int] = None,
    ) -> None:
        """Upload one local file to a Zenodo bucket URL.

        ``depo_id`` (when known) lets the post-failure verify fall back
        to the authorised deposition-record API when a bare bucket-URL
        HEAD is rejected with 403 on the public edge.

        Tries multipart for large files (cleaner recovery from edge
        timeouts on multi-GB streams). Falls back to a single PUT on
        any setup error so older Zenodo deployments keep working.
        """
        put_url = f"{bucket_url}/{filename}"

        if use_multipart:
            try:
                self._multipart_upload(
                    bucket_url, filename, local_path, file_size,
                    progress_callback)
                if progress_callback:
                    progress_callback(file_size, file_size)
                return
            except Exception as exc:
                # Fall through to single-PUT. Worth a WARNING because
                # multipart was supposed to be the safer path; if it
                # consistently fails we want to know.
                log.warning(
                    "Multipart upload setup failed for %s (%s) — "
                    "falling back to single PUT", filename, exc,
                )

        # Single-PUT fallback / small-file path.
        #
        # Reliability: on multi-GB uploads, Zenodo's edge frequently
        # finishes writing the body but never returns the JSON ack
        # (manifests as SSLEOFError or 'write operation timed out').
        # The session-level retry then re-uploads the *whole* file,
        # which (a) wastes bandwidth, (b) racks up duplicate writes
        # on the bucket, and (c) regularly fails again after 6
        # attempts because each attempt is itself ~10 min long. The
        # cure is to verify the bucket state after every network
        # failure: if size + checksum match, the upload actually
        # succeeded and we can return immediately. If size is wrong,
        # delete the partial and let the outer retry re-PUT cleanly.
        # See zenodo_client.py:_verify_bucket_file for the HEAD/GET
        # logic. This is robust even without multipart, because the
        # vast majority of "failed" >1 GB uploads in our logs were
        # actually delivered — only the response was lost.
        try:
            md5_hex = _md5_of_file(local_path)
        except Exception:
            md5_hex = None
        max_attempts = 6
        for attempt in range(1, max_attempts + 1):
            try:
                with open(local_path, "rb") as fh:
                    data = (ProgressFileWrapper(
                        fh, file_size, progress_callback)
                        if progress_callback else fh)
                    # max_retries=0 — the outer loop in this method
                    # owns retry policy. Without this we'd amplify
                    # every transient hiccup into 6 × 6 = 36 re-PUTs
                    # of the same multi-GB body.
                    self._do_request(
                        "PUT", put_url,
                        data=data,
                        content_type="application/octet-stream",
                        max_retries=0,
                    )
                if progress_callback:
                    progress_callback(file_size, file_size)
                # Defensive verification: Zenodo occasionally accepts
                # a body and then garbles the JSON ack; we want every
                # successful return to mean "file is in the bucket".
                ok, why = self._verify_bucket_file(
                    put_url, file_size, md5_hex, depo_id=depo_id)
                if ok:
                    return
                log.warning(
                    "PUT %s returned 2xx but bucket verify failed (%s) "
                    "— will retry", filename, why,
                )
                last_exc = ZenodoError(f"bucket verify failed: {why}")
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                # The 'write timeout' / SSLEOFError class of failures
                # almost always means the upload completed on Zenodo's
                # side but the response got dropped. Verify before
                # re-uploading; on a match we save a full file's worth
                # of bandwidth + ~10 min of wall time.
                ok, why = self._verify_bucket_file(
                    put_url, file_size, md5_hex, depo_id=depo_id)
                if ok:
                    log.info(
                        "PUT %s raised %s on attempt %d/%d but bucket "
                        "already holds the correct file (%s) — "
                        "treating as success",
                        filename, type(exc).__name__, attempt,
                        max_attempts, why,
                    )
                    if progress_callback:
                        progress_callback(file_size, file_size)
                    return
                log.warning(
                    "PUT %s failed (attempt %d/%d): %s; bucket verify: %s",
                    filename, attempt, max_attempts, exc, why,
                )
                # Drop any partial upload before retrying so Zenodo
                # doesn't keep a wrong-size file under the same name.
                if why and why.startswith("size_mismatch"):
                    try:
                        self._do_request("DELETE", put_url)
                    except Exception as _de:
                        log.info("  partial-cleanup DELETE failed: %s", _de)
            except ZenodoError as exc:
                # Non-retryable Zenodo error — propagate immediately.
                if exc.status_code and 400 <= exc.status_code < 500 \
                        and exc.status_code not in (408, 409, 429):
                    raise
                last_exc = exc
                log.warning(
                    "PUT %s zenodo error (attempt %d/%d): %s",
                    filename, attempt, max_attempts, exc,
                )
            # Refresh the TCP+TLS connection before retrying — the
            # session-level reset in _do_request already does this on
            # network failure, but a verify-fail success path skips
            # that branch. Belt + braces.
            self._reset_session()
            if attempt < max_attempts:
                wait = min(self.retry_base_wait * (2 ** (attempt - 1)), 300.0)
                log.info("  retrying PUT %s in %.1fs …", filename, wait)
                time.sleep(wait)
        # Exhausted attempts. One final verify before giving up — the
        # last PUT may have succeeded silently while we were sleeping.
        ok, why = self._verify_bucket_file(
            put_url, file_size, md5_hex, depo_id=depo_id)
        if ok:
            log.info(
                "PUT %s: bucket verify succeeded after %d attempts (%s)",
                filename, max_attempts, why,
            )
            if progress_callback:
                progress_callback(file_size, file_size)
            return
        raise last_exc

    def _multipart_upload(
        self,
        bucket_url: str,
        filename: str,
        local_path: "Path",
        file_size: int,
        progress_callback,
    ) -> None:
        """InvenioRDM multipart bucket upload.

        Each part is at most ``_MULTIPART_PART_SIZE`` (256 MB). On any
        per-part failure we abort the multipart upload (best-effort)
        and re-raise so the caller falls back to single-PUT or the
        outer retry can take over.
        """
        part_size = self._MULTIPART_PART_SIZE
        num_parts = (file_size + part_size - 1) // part_size
        put_url = f"{bucket_url}/{filename}"

        # 1) Initiate.
        init_url = (f"{put_url}?uploads=&size={file_size}"
                    f"&part_size={part_size}")
        resp = self._do_request("POST", init_url, json_body={},
                                 content_type="application/json")
        try:
            init_info = resp.json() or {}
        except Exception:
            init_info = {}
        # Multipart upload id field name varies across InvenioRDM
        # versions: 'upload_id' or just embedded in the response links.
        upload_id = (init_info.get("upload_id")
                     or init_info.get("id"))
        if not upload_id:
            raise ZenodoError(
                f"multipart init returned no upload_id: "
                f"{str(init_info)[:300]}")

        log.info(
            "multipart upload started: %s  size=%.1f MB  parts=%d  "
            "part_size=%d MB  upload_id=%s",
            filename, file_size / 1e6, num_parts,
            part_size // (1024 * 1024), upload_id,
        )

        sent_total = 0
        try:
            with open(local_path, "rb") as fh:
                for part_no in range(1, num_parts + 1):
                    part_bytes = fh.read(part_size)
                    if not part_bytes:
                        break
                    part_url = (f"{put_url}?uploadId={upload_id}"
                                f"&partNumber={part_no}")
                    self._do_request(
                        "PUT", part_url,
                        data=part_bytes,
                        content_type="application/octet-stream",
                    )
                    sent_total += len(part_bytes)
                    if progress_callback:
                        progress_callback(sent_total, file_size)
                    log.info(
                        "  part %d/%d ok  (%.1f / %.1f MB)",
                        part_no, num_parts,
                        sent_total / 1e6, file_size / 1e6,
                    )
        except Exception:
            # Best-effort abort so we don't leave dangling parts.
            try:
                self._do_request(
                    "DELETE",
                    f"{put_url}?uploadId={upload_id}",
                )
            except Exception as _ae:
                log.warning("multipart abort failed: %s", _ae)
            raise

        # 3) Complete.
        complete_url = f"{put_url}?uploadId={upload_id}"
        self._do_request(
            "POST", complete_url, json_body={},
            content_type="application/json",
        )

    # -- public API: upload --------------------------------------------------

    def upload(
        self,
        key: str,
        local_path: str | Path,
        version: str,
        meta_func: MetadataFunc,
        manifest: Manifest,
    ) -> Entry:
        """Upload a local file to Zenodo and record it in the manifest.

        If *key* already exists in the manifest the existing deposition's old
        file is replaced; otherwise a brand-new draft deposition is created.

        Parameters
        ----------
        key:
            Manifest key for this file.
        local_path:
            Path to the local file to upload.
        version:
            Version string stored in metadata and manifest.
        meta_func:
            Callable ``(key, filename, version) → dict`` producing the Zenodo
            metadata payload.
        manifest:
            Manifest to update on success.

        Returns
        -------
        Entry
            The newly created / updated manifest entry.
        """
        self._assert_writable()
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        filename = local_path.name
        file_size = local_path.stat().st_size
        file_bytes = local_path.read_bytes()
        md5_hex = hashlib.md5(file_bytes).hexdigest()

        existing = manifest.get(key)

        if existing is not None and existing.depo_id:
            # ------ update existing deposition ------------------------------
            log.info(
                "Replacing file in existing deposition %d for key=%s",
                existing.depo_id, key,
            )
            bucket_url = existing.bucket_url

            # Delete old file (ignore 404 — may already be gone).
            if existing.filename:
                del_url = f"{bucket_url}/{existing.filename}"
                try:
                    self._do_request("DELETE", del_url)
                    log.debug("Deleted old file %s", existing.filename)
                except ZenodoError as exc:
                    if exc.status_code != 404:
                        raise
                    log.debug("Old file already absent (404).")

            # Upload new file to the bucket.
            put_url = f"{bucket_url}/{filename}"
            resp = self._do_request(
                "PUT", put_url,
                data=file_bytes,
                content_type="application/octet-stream",
            )
            upload_info = resp.json()
            log.debug("Upload response: %s", upload_info)

            # Update metadata.
            meta_payload = meta_func(key, filename, version)
            meta_url = self._url(f"/api/deposit/depositions/{existing.depo_id}")
            self._do_request("PUT", meta_url, json_body=meta_payload, content_type="application/json")

            entry = Entry(
                key=key,
                depo_id=existing.depo_id,
                bucket_url=bucket_url,
                filename=filename,
                size=file_size,
                checksum=f"md5:{md5_hex}",
                uploaded_at=datetime.now(timezone.utc).isoformat(),
                version=version,
            )

        else:
            # ------ create new deposition -----------------------------------
            log.info("Creating new deposition for key=%s", key)

            # 1. Create empty draft.
            create_url = self._url("/api/deposit/depositions")
            resp = self._do_request("POST", create_url, json_body={}, content_type="application/json")
            depo = resp.json()
            depo_id: int = depo["id"]
            bucket_url: str = depo["links"]["bucket"]
            log.info("Created draft deposition %d, bucket=%s", depo_id, bucket_url)

            # 2. Upload file to bucket.
            put_url = f"{bucket_url}/{filename}"
            resp = self._do_request(
                "PUT", put_url,
                data=file_bytes,
                content_type="application/octet-stream",
            )
            upload_info = resp.json()
            log.debug("Upload response: %s", upload_info)

            # 3. Set metadata.
            meta_payload = meta_func(key, filename, version)
            meta_url = self._url(f"/api/deposit/depositions/{depo_id}")
            self._do_request("PUT", meta_url, json_body=meta_payload, content_type="application/json")

            entry = Entry(
                key=key,
                depo_id=depo_id,
                bucket_url=bucket_url,
                filename=filename,
                size=file_size,
                checksum=f"md5:{md5_hex}",
                uploaded_at=datetime.now(timezone.utc).isoformat(),
                version=version,
            )

        # Persist to manifest.
        manifest.set(key, entry)
        manifest.save()
        log.info("Manifest updated for key=%s (depo_id=%d)", key, entry.depo_id)
        return entry

    # -- public API: streaming upload ----------------------------------------

    def upload_stream(
        self,
        key: str,
        local_path: str | Path,
        version: str,
        meta_func: MetadataFunc,
        manifest: Manifest,
        *,
        delete_after: bool = False,
        progress_callback=None,
    ) -> Entry:
        """Upload a local file to Zenodo by streaming from disk.

        Unlike :meth:`upload`, this method never loads the entire file into
        memory.  It streams the file contents directly from disk to the
        Zenodo bucket API, keeping memory usage constant regardless of
        file size.

        The MD5 checksum is computed in a streaming pass before upload.

        Parameters
        ----------
        key:
            Manifest key for this file.
        local_path:
            Path to the local file to upload.
        version:
            Version string stored in metadata and manifest.
        meta_func:
            Callable ``(key, filename, version) → dict`` producing the
            Zenodo metadata payload.
        manifest:
            Manifest to update on success.
        delete_after:
            If ``True``, delete the local file after successful upload
            and verification.

        Returns
        -------
        Entry
            The newly created / updated manifest entry.
        """
        self._assert_writable()
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        filename = local_path.name
        file_size = local_path.stat().st_size

        # Stream-compute MD5 without loading entire file.
        md5 = hashlib.md5()
        with open(local_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                md5.update(chunk)
        md5_hex = md5.hexdigest()
        # Distinct log tag for ``_full.gpkg`` re-upload attempts:
        #  * intra-process retry on the same Client instance, OR
        #  * cross-process resume — the manifest already has an entry
        #    for this key, meaning a previous run uploaded (or tried
        #    to upload) the same file. After a subprocess crash these
        #    retries can be doomed (poisoned remote draft, dangling
        #    multipart parts) and spin for hours. The tag makes them
        #    trivially greppable in the processor log.
        is_full_gpkg = filename.endswith("_full.gpkg")
        attempt_tag = ""
        existing_for_log = manifest.get(key)
        if is_full_gpkg:
            if not hasattr(self, "_seen_full_gpkg"):
                self._seen_full_gpkg = set()
            seen = filename in self._seen_full_gpkg
            cross_process = (existing_for_log is not None
                             and existing_for_log.depo_id
                             and existing_for_log.filename == filename)
            if seen:
                attempt_tag = " [FULL-GPKG-REUPLOAD same-process]"
            elif cross_process:
                attempt_tag = (" [FULL-GPKG-REUPLOAD cross-process "
                               f"depo={existing_for_log.depo_id}]")
            self._seen_full_gpkg.add(filename)
        log.info("upload_stream:%s %s  size=%.1f MB  md5=%s",
                 attempt_tag, filename, file_size / 1e6, md5_hex)
        # NOTE 2026-05-09: multipart bucket uploads disabled.
        # zenodo.org (production) returns HTTP 405 on the InvenioRDM
        # multipart init endpoint (`POST {bucket}/{key}?uploads=`).
        # The fallback path then re-uploads the entire multi-GB file
        # via single PUT — but the multipart attempt has already
        # consumed connect/handshake budget and the 405 retry chain
        # left _full.gpkg files dangling on peer disks, eventually
        # triggering ``Disk critically low`` pauses. Until we have
        # a Zenodo deployment that actually supports multipart we
        # stick with single-PUT streaming (which worked for months).
        # The session-reset on ReadTimeout (above) and the
        # WARNING-from-attempt-2 throttle escalation are kept —
        # those addressed the *symptom* (timeout retry storm),
        # multipart was a separate, broken mitigation.
        use_multipart = False

        existing = manifest.get(key)

        # Guard against a deleted/vanished deposition. An operator (or the
        # orphan-tombstone cleanup) may have deleted the draft this manifest
        # entry points at. Reusing it then 404-wedges: every PUT to the dead
        # bucket fails (`depo_status:404`) and the 6-attempt retry chain just
        # spins for ~10 min before giving up, leaving the KG un-uploaded and
        # the peer stuck re-trying forever on subsequent runs. Probe the
        # deposition first; if it's gone, drop the stale manifest pointer and
        # fall through to create a fresh deposit.
        if existing is not None and existing.depo_id:
            try:
                self._do_request(
                    "GET",
                    self._url(f"/api/deposit/depositions/{existing.depo_id}"),
                    max_retries=0,
                )
            except ZenodoError as exc:
                if exc.status_code == 404:
                    log.warning(
                        "Existing deposition %d for key=%s is gone (404) — "
                        "creating a fresh deposit instead of re-uploading to a "
                        "dead bucket", existing.depo_id, key,
                    )
                    existing = None
                # Non-404 (transient/auth) errors: keep the existing pointer
                # and let the normal replace path + its retries handle it.

        if existing is not None and existing.depo_id:
            # ------ update existing deposition ----------------------------
            log.info(
                "Replacing file in existing deposition %d for key=%s",
                existing.depo_id, key,
            )
            bucket_url = existing.bucket_url

            # Delete old file (ignore 404).
            if existing.filename:
                del_url = f"{bucket_url}/{existing.filename}"
                try:
                    self._do_request("DELETE", del_url)
                except ZenodoError as exc:
                    if exc.status_code != 404:
                        raise

            # Stream-upload new file (multipart for >1 GB).
            self._put_file_to_bucket(
                bucket_url, filename, local_path, file_size,
                progress_callback, use_multipart=use_multipart,
                depo_id=existing.depo_id)

            # Update metadata.
            meta_payload = meta_func(key, filename, version)
            meta_url = self._url(f"/api/deposit/depositions/{existing.depo_id}")
            self._do_request("PUT", meta_url, json_body=meta_payload,
                             content_type="application/json")

            depo_id = existing.depo_id

        else:
            # ------ create new deposition ---------------------------------
            log.info("Creating new deposition for key=%s", key)

            create_url = self._url("/api/deposit/depositions")
            resp = self._do_request("POST", create_url, json_body={},
                                     content_type="application/json")
            depo = resp.json()
            depo_id = depo["id"]
            bucket_url = depo["links"]["bucket"]
            log.info("Created draft deposition %d, bucket=%s", depo_id, bucket_url)

            # Stream-upload file (multipart for >1 GB).
            self._put_file_to_bucket(
                bucket_url, filename, local_path, file_size,
                progress_callback, use_multipart=use_multipart,
                depo_id=depo_id)

            # Set metadata.
            meta_payload = meta_func(key, filename, version)
            meta_url = self._url(f"/api/deposit/depositions/{depo_id}")
            self._do_request("PUT", meta_url, json_body=meta_payload,
                             content_type="application/json")

        entry = Entry(
            key=key,
            depo_id=depo_id,
            bucket_url=bucket_url,
            filename=filename,
            size=file_size,
            checksum=f"md5:{md5_hex}",
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            version=version,
        )

        # Persist to manifest.
        manifest.set(key, entry)
        manifest.save()
        log.info("Manifest updated for key=%s (depo_id=%d, streamed)", key, entry.depo_id)

        # Delete local file after verified upload.
        if delete_after:
            try:
                local_path.unlink()
                log.info("Deleted local %s after stream upload", local_path)
            except OSError as exc:
                log.warning("Failed to delete %s: %s", local_path, exc)

        return entry

    # -- public API: download ------------------------------------------------

    def download(
        self,
        key: str,
        dest_dir: str | Path,
        manifest: Manifest,
    ) -> Path:
        """Download a file from Zenodo and verify its MD5 checksum.

        Parameters
        ----------
        key:
            Manifest key.
        dest_dir:
            Directory to write the file into.
        manifest:
            Manifest that contains the entry for *key*.

        Returns
        -------
        pathlib.Path
            Path to the downloaded file.

        Raises
        ------
        KeyError
            If *key* is not present in the manifest.
        ValueError
            If the downloaded file's MD5 doesn't match.
        """
        entry = manifest.get(key)
        if entry is None:
            raise KeyError(f"Key {key!r} not found in manifest")

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / entry.filename

        # Build download URL.
        download_url = f"{entry.bucket_url}/{entry.filename}"
        log.info("Downloading key=%s from %s", key, download_url)

        resp = self._do_request("GET", download_url, stream=True)

        # Atomic download: write to .tmp then rename.
        fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix=".tmp", prefix=f".{entry.filename}_")
        md5 = hashlib.md5()
        try:
            with os.fdopen(fd, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        md5.update(chunk)
            # Verify checksum.
            expected = entry.checksum
            actual_hex = md5.hexdigest()
            actual = f"md5:{actual_hex}"
            if expected and actual != expected:
                raise ValueError(
                    f"Checksum mismatch for {entry.filename}: "
                    f"expected {expected}, got {actual}"
                )
            os.replace(tmp_path, dest_path)
            log.info("Downloaded %s → %s (md5 OK)", key, dest_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return dest_path

    # -- public API: convenience wrappers ------------------------------------

    def upload_and_delete_local(
        self,
        key: str,
        local_path: str | Path,
        version: str,
        meta_func: MetadataFunc,
        manifest: Manifest,
    ) -> Entry:
        """Upload a file then remove the local copy.

        Parameters are identical to :meth:`upload`.
        """
        entry = self.upload(key, local_path, version, meta_func, manifest)
        local_path = Path(local_path)
        try:
            local_path.unlink()
            log.info("Deleted local file %s after upload", local_path)
        except OSError as exc:
            log.warning("Failed to delete local file %s: %s", local_path, exc)
        return entry

    def ensure_local(
        self,
        key: str,
        dest_dir: str | Path,
        manifest: Manifest,
    ) -> Tuple[Path, bool]:
        """Ensure a file exists locally — download from Zenodo if needed.

        Returns
        -------
        tuple[Path, bool]
            ``(local_path, was_already_local)`` — the second element is
            ``True`` if the file was already present with a matching checksum.
        """
        entry = manifest.get(key)
        if entry is None:
            raise KeyError(f"Key {key!r} not found in manifest")

        dest_dir = Path(dest_dir)
        dest_path = dest_dir / entry.filename

        if dest_path.is_file():
            # Verify checksum to avoid stale files.
            if entry.checksum:
                md5_hex = hashlib.md5(dest_path.read_bytes()).hexdigest()
                if f"md5:{md5_hex}" == entry.checksum:
                    log.debug("Local file %s exists with matching checksum", dest_path)
                    return dest_path, True
                log.info(
                    "Local file %s exists but checksum mismatch — re-downloading",
                    dest_path,
                )
            else:
                # No checksum to verify — trust the local file.
                log.debug("Local file %s exists (no checksum to verify)", dest_path)
                return dest_path, True

        downloaded = self.download(key, dest_dir, manifest)
        return downloaded, False

    def head_file(
        self,
        key: str,
        manifest: Manifest,
    ) -> int:
        """Send a HEAD request for the file and return the HTTP status code.

        Useful to check whether the remote file still exists and is accessible.

        Returns
        -------
        int
            HTTP status code (e.g. 200, 404).
        """
        entry = manifest.get(key)
        if entry is None:
            raise KeyError(f"Key {key!r} not found in manifest")

        url = f"{entry.bucket_url}/{entry.filename}"
        log.debug("HEAD %s", url)
        try:
            resp = self._do_request("HEAD", url)
            return resp.status_code
        except ZenodoError as exc:
            return exc.status_code

    def delete_deposition(
        self,
        key: str,
        manifest: Manifest,
    ) -> None:
        """Delete a *draft* deposition from Zenodo and remove it from the manifest.

        Only unpublished (draft) depositions can be deleted via the API.
        """
        self._assert_writable()
        entry = manifest.get(key)
        if entry is None:
            raise KeyError(f"Key {key!r} not found in manifest")

        url = self._url(f"/api/deposit/depositions/{entry.depo_id}")
        log.info("Deleting deposition %d for key=%s", entry.depo_id, key)
        self._do_request("DELETE", url)

        manifest.delete(key)
        manifest.save()
        log.info("Deposition %d deleted, manifest updated", entry.depo_id)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying requests session."""
        self._session.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Client(base_url={self.base_url!r}, "
            f"read_only={self.read_only}, "
            f"max_retries={self.max_retries})"
        )
