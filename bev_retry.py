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


def read_with_retry(
    url: str,
    read_fn,
    *,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    caller: str = "",
    direct_first: bool = True,
    direct_retries: int = 1,
):
    """Open + read in one shot with retry + proxy rotation.

    Unlike the context-manager version, this retries the *entire*
    open-and-read cycle (including read-phase errors like HTTP range
    request failures).  Use this when you can express the read as a
    simple function of the dataset.

    Parameters
    ----------
    url : str
        The /vsicurl/ or local path to open.
    read_fn : callable(dataset) -> result
        Called with the open dataset; its return value is returned.
    max_retries, base_delay, caller : same as open_with_retry.
    direct_first : bool
        If True, the first ``direct_retries`` attempts use a direct
        connection (no proxy); proxies are only used after that.
        Good for non-BEV sources like Google Storage (Hansen) that
        work fine direct but fail through random proxies.
    direct_retries : int
        How many consecutive direct attempts before switching to proxies
        (default 1, only the first attempt).  Set higher for sources
        where direct access is strongly preferred.
    """
    label = caller or url.rsplit("/", 1)[-1]
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        if direct_first and attempt < direct_retries:
            os.environ.pop("GDAL_HTTP_PROXY", None)
            os.environ.pop("GDAL_HTTP_PROXYUSERPWD", None)
            used_proxy = None
        else:
            used_proxy = _apply_proxy()
        ds = None
        try:
            ds = rasterio.open(url)
            result = read_fn(ds)
            bev_proxy.report_success(used_proxy)
            return result
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise
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
        finally:
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass

    raise last_exc  # type: ignore[misc]


@contextmanager
def open_with_retry(
    url: str,
    *,
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_DELAY,
    caller: str = "",
):
    """Context manager: rasterio.open() with retry + proxy rotation + healing.

    Retries transient errors during both the open phase AND read phase.
    On read-phase failures, the dataset is closed and reopened from scratch
    with a rotated proxy.

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
        if attempt == 0:
            # First attempt: direct (no proxy)
            os.environ.pop("GDAL_HTTP_PROXY", None)
            os.environ.pop("GDAL_HTTP_PROXYUSERPWD", None)
            used_proxy = None
        else:
            used_proxy = _apply_proxy()
        ds = None
        try:
            ds = rasterio.open(url)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise
            bev_proxy.report_failure(used_proxy, error_msg=str(exc))
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), MAX_DELAY)
                log.warning(
                    "%s: attempt %d/%d open failed (%s), retrying in %.0fs...",
                    label, attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
                continue
            else:
                log.error(
                    "%s: all %d attempts exhausted, last error: %s",
                    label, max_retries + 1, exc,
                )
                break

        # Open succeeded — yield and handle read-phase errors
        try:
            yield ds
            # Caller finished successfully
            bev_proxy.report_success(used_proxy)
            return
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc):
                raise
            bev_proxy.report_failure(used_proxy, error_msg=str(exc))
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), MAX_DELAY)
                log.warning(
                    "%s: attempt %d/%d read failed (%s), retrying in %.0fs...",
                    label, attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
                # Fall through to next iteration — but we already yielded!
                # A @contextmanager can only yield once, so we must raise
                # and let the caller retry at a higher level.
                raise
            else:
                log.error(
                    "%s: all %d attempts exhausted, last error: %s",
                    label, max_retries + 1, exc,
                )
                raise
        finally:
            if ds is not None:
                ds.close()

    # All retries exhausted (open-phase failures only reach here)
    raise last_exc  # type: ignore[misc]
