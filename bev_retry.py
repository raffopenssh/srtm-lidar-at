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

try:
    from osgeo import gdal as _gdal
except Exception:  # pragma: no cover
    _gdal = None


def _clear_vsicurl_cache(url: str) -> None:
    """Drop GDAL's /vsicurl cached metadata for *url*.

    GDAL caches the result of HEAD + initial range read per URL. When a
    transient HTTP 0 / partial-body response poisons that cache, every
    subsequent ``rasterio.open(url)`` returns 'not recognized as being
    in a supported file format' even though the server is healthy. We
    purge the per-URL cache between retries so each attempt re-probes
    the server fresh.
    """
    if _gdal is None:
        return
    try:
        # Strip /vsicurl/ prefix — VSICurlPartialClearCache wants the
        # raw URL, not the GDAL prefix.
        target = url
        if target.startswith('/vsicurl/'):
            target = target[len('/vsicurl/'):]
        if hasattr(_gdal, 'VSICurlPartialClearCache'):
            _gdal.VSICurlPartialClearCache(target)
    except Exception:
        # Last resort: nuke the whole vsicurl cache.
        try:
            if hasattr(_gdal, 'VSICurlClearCache'):
                _gdal.VSICurlClearCache()
        except Exception:
            pass

log = logging.getLogger(__name__)

# Retry configuration
#
# 2026-05-28: post-mortem on 3h slice showed BEV failures cluster by
# /24 egress pool (per-NAT-IP throttling) + a small set of contended
# server-side tiles, not random tcp flakiness. Burning 5 attempts in
# 30s mostly hit the same throttled IP repeatedly. Widened backoff
# (5/30/120s) gives the per-IP token bucket time to refill; cut to 3
# retries (4 attempts total) so net wall-time on a true-fail tile is
# bounded (~155s) and we surrender to the partial-KG path quickly.
MAX_RETRIES = 3          # up to 3 retries (4 total attempts)
# Explicit per-retry waits (index = attempt that just failed).
# Length must be >= MAX_RETRIES; values past MAX_RETRIES are unused.
_RETRY_SCHEDULE_S = (5.0, 30.0, 120.0)
BASE_DELAY = _RETRY_SCHEDULE_S[0]  # back-compat alias (1st retry wait)
MAX_DELAY = _RETRY_SCHEDULE_S[-1]  # back-compat alias


def _retry_delay(attempt: int, base_delay: float = BASE_DELAY) -> float:
    """Return wait seconds after *attempt* (0-indexed) has just failed.

    When ``base_delay`` is the default we use the fixed schedule above.
    Callers that override ``base_delay`` (e.g. tests) fall back to the
    classic exponential curve so behaviour stays predictable.
    """
    if abs(base_delay - BASE_DELAY) < 1e-6 and attempt < len(_RETRY_SCHEDULE_S):
        return _RETRY_SCHEDULE_S[attempt]
    return min(base_delay * (2 ** attempt), MAX_DELAY)

# Direct-first policy.  Our 24h post-mortem (2026-05-28) showed >87%
# of "BEV warnings" were transport failures via the free-proxy lane,
# while the BEV server itself returns 206 in <1.5s under 30x concurrency
# from our peer IPs.  We now prefer direct for the first TWO attempts
# on every read; only flaky-IP peers fall back to the proxy pool.
# Set BEV_DISABLE_PROXY=1 in the env to skip proxies entirely.
DIRECT_RETRIES_DEFAULT = 2
_PROXY_DISABLED = os.environ.get("BEV_DISABLE_PROXY", "").strip().lower() in ("1", "true", "yes")

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
    # GDAL surfaces non-TIFF responses (HTML 5xx pages, truncated bodies)
    # as 'not recognized as being in a supported file format'.  Treat as
    # transient: a retry / proxy rotation usually fetches the real TIFF.
    "not recognized as being in a supported file format",
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
    if _PROXY_DISABLED:
        os.environ.pop("GDAL_HTTP_PROXY", None)
        os.environ.pop("GDAL_HTTP_PROXYUSERPWD", None)
        return None
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
    direct_retries: int = DIRECT_RETRIES_DEFAULT,
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
        if attempt > 0:
            # Drop any poisoned per-URL cache from the previous attempt.
            _clear_vsicurl_cache(url)
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
                delay = _retry_delay(attempt, base_delay)
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
        if attempt > 0:
            # Drop any poisoned per-URL cache from the previous attempt.
            _clear_vsicurl_cache(url)
        if attempt < DIRECT_RETRIES_DEFAULT or _PROXY_DISABLED:
            # First N attempts: direct (no proxy).  See post-mortem 2026-05-28.
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
                delay = _retry_delay(attempt, base_delay)
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
                delay = _retry_delay(attempt, base_delay)
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
