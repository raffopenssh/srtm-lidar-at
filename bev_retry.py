"""Graceful retry wrapper for remote BEV / Hansen GeoTIFF reads.

Wraps rasterio.open() calls with exponential backoff + proxy rotation
so transient HTTP errors (connection reset, 503, timeouts) don't kill
an entire analysis pipeline.  Failed proxies go on cooldown so they
have time to heal before being reused.

Usage:
    from bev_retry import open_with_retry

    with open_with_retry(url) as ds:
        data = ds.read(1, window=window)
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

import rasterio

import bev_proxy

log = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 4          # up to 4 retries (5 total attempts)
BASE_DELAY = 2.0         # seconds — doubles each retry: 2, 4, 8, 16
MAX_DELAY = 20.0         # cap per-retry wait

# Error substrings that indicate a transient / retryable failure.
# GDAL/rasterio surface HTTP errors as CPLE_* messages or generic IOError.
_RETRYABLE_PATTERNS = (
    "CPLE_HttpResponse",
    "CPLE_AppDefined",
    "curl error",
    "Connection reset",
    "Connection refused",
    "Connection timed out",
    "timed out",
    "Timeout",
    "503",
    "502",
    "429",
    "SSL",
    "Broken pipe",
    "server disconnected",
    "partial file",
    "transfer closed",
    "Empty reply",
    "Recv failure",
    "IReadBlock failed",
    "TIFFReadEncodedTile",
    "TIFFFillTile",
    "Read error at row",
    "Read failed",
    "response_code=0",
)


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception looks like a transient network error."""
    msg = str(exc)
    return any(p.lower() in msg.lower() for p in _RETRYABLE_PATTERNS)


def _apply_proxy() -> str | None:
    """Rotate to next healthy proxy for the upcoming GDAL open.

    Returns the proxy URL (or None for direct) so the caller can
    report success/failure back to bev_proxy.
    """
    proxy_url = bev_proxy.next_proxy()
    if proxy_url:
        os.environ["GDAL_HTTP_PROXY"] = proxy_url
    else:
        os.environ.pop("GDAL_HTTP_PROXY", None)
    os.environ.pop("GDAL_HTTP_PROXYUSERPWD", None)
    return proxy_url


@contextmanager
def open_with_retry(
    url: str,
    *,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    caller: str = "",
):
    """Context manager: rasterio.open() with retry + proxy rotation + healing.

    On each retry the proxy is rotated (via bev_proxy.next_proxy()) and
    an exponential backoff delay is applied.  Failed proxies are put on
    cooldown so they can heal.  Successful reads clear the cooldown.
    Non-retryable errors (e.g. "file not found", ValueError) are raised
    immediately.

    Parameters
    ----------
    url : str
        The /vsicurl/ or local path to open.
    max_retries : int
        Maximum number of retries (default 4 → 5 total attempts).
    base_delay : float
        Initial delay in seconds; doubles each retry.
    caller : str
        Optional label for log messages (e.g. "DTM N2850E4550").
    """
    label = caller or url.rsplit("/", 1)[-1]
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        used_proxy = _apply_proxy()
        try:
            ds = rasterio.open(url)
            try:
                yield ds
            finally:
                ds.close()
            # Success — clear any cooldown on this proxy
            bev_proxy.report_success(used_proxy)
            return
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                # Not transient — don't waste time retrying
                raise
            # Transient failure — put this proxy on cooldown
            bev_proxy.report_failure(used_proxy, error_msg=str(exc))
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), MAX_DELAY)
                log.warning(
                    "%s: attempt %d/%d failed (%s), retrying in %.0fs...",
                    label, attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
            else:
                log.error(
                    "%s: all %d attempts exhausted, last error: %s",
                    label, max_retries + 1, exc,
                )

    # All retries exhausted — re-raise the last exception
    raise last_exc  # type: ignore[misc]
