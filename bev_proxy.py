"""Rotating HTTP proxy pool for BEV GeoTIFF reads with healing.

Alternates between direct access and Webshare proxies to avoid
BEV rate-limiting on sustained /vsicurl/ range requests.

Healing: when a proxy (or direct) hits a transient error, it goes on
cooldown for COOLDOWN_SECS.  next_proxy() skips cooled-down entries so
they have time to recover before being used again.  If ALL entries are
on cooldown, the least-recently-failed one is returned anyway (best
effort).

Usage:
    proxy_url = bev_proxy.next_proxy()   # returns URL or None (direct)
    bev_proxy.report_failure(proxy_url)  # put on cooldown
    bev_proxy.report_success(proxy_url)  # clear cooldown early
"""
import logging
import time
import threading

log = logging.getLogger(__name__)

# Webshare proxy credentials
_PROXY_USER = "mcygpktm"
_PROXY_PASS = "y8yamkwx20qg"

# Proxy list: (ip, port).  None = direct (no proxy).
_PROXY_LIST = [
    None,                           # direct
    ("31.58.9.4", 6077),            # DE
    None,                           # direct
    ("31.59.20.176", 6754),         # GB
    None,                           # direct
    ("198.23.239.134", 6540),       # US
    None,                           # direct
    ("45.38.107.97", 6014),         # GB
    None,                           # direct
    ("107.172.163.27", 6543),       # US
    None,                           # direct
    ("198.105.121.200", 6462),      # GB
    None,                           # direct
    ("216.10.27.159", 6837),        # US
    None,                           # direct
    ("142.111.67.146", 5611),       # JP
    None,                           # direct
    ("191.96.254.138", 6185),       # US
    None,                           # direct
    ("23.26.71.145", 5628),         # US
]

# ---------------------------------------------------------------------------
# Healing configuration
# ---------------------------------------------------------------------------
COOLDOWN_SECS = 60          # how long a failed proxy sits out
MAX_CONSECUTIVE_FAILS = 3   # consecutive fails → longer cooldown
LONG_COOLDOWN_SECS = 1800   # 30 min cooldown after repeated failures

_lock = threading.Lock()
_index = 0  # round-robin pointer

# Cooldown state: key → {"until": timestamp, "consecutive": int}
# key is the proxy URL string or "__direct__" for None entries
_cooldowns: dict[str, dict] = {}


def _proxy_key(entry) -> str:
    """Stable key for a proxy list entry."""
    if entry is None:
        return "__direct__"
    ip, port = entry
    return f"{ip}:{port}"


def _entry_to_url(entry) -> str | None:
    """Convert a proxy list entry to a URL or None."""
    if entry is None:
        return None
    ip, port = entry
    return f"http://{_PROXY_USER}:{_PROXY_PASS}@{ip}:{port}"


def _url_to_key(proxy_url: str | None) -> str:
    """Convert a proxy URL (or None) back to a cooldown key."""
    if proxy_url is None:
        return "__direct__"
    # Extract ip:port from http://user:pass@ip:port
    at = proxy_url.rfind("@")
    if at >= 0:
        return proxy_url[at + 1:]
    return proxy_url


def next_proxy() -> str | None:
    """Return next healthy proxy URL or None for direct access.

    Skips entries that are on cooldown.  If all are on cooldown,
    returns the one whose cooldown expires soonest.
    """
    global _index
    now = time.monotonic()
    n = len(_PROXY_LIST)

    with _lock:
        # Try up to a full rotation to find a healthy entry
        for _ in range(n):
            entry = _PROXY_LIST[_index % n]
            _index = (_index + 1) % n
            key = _proxy_key(entry)
            cd = _cooldowns.get(key)
            if cd is None or now >= cd["until"]:
                # Healthy — clear stale cooldown
                _cooldowns.pop(key, None)
                return _entry_to_url(entry)

        # All on cooldown — pick the one expiring soonest (best effort)
        best_entry = _PROXY_LIST[0]
        best_until = float("inf")
        for entry in _PROXY_LIST:
            key = _proxy_key(entry)
            cd = _cooldowns.get(key)
            if cd and cd["until"] < best_until:
                best_until = cd["until"]
                best_entry = entry
        log.warning(
            "All proxies on cooldown — using least-recently-failed (expires in %.0fs)",
            max(0, best_until - now),
        )
        return _entry_to_url(best_entry)


def report_failure(proxy_url: str | None) -> None:
    """Put a proxy on cooldown after a transient failure."""
    key = _url_to_key(proxy_url)
    now = time.monotonic()

    with _lock:
        cd = _cooldowns.get(key, {"until": 0, "consecutive": 0})
        cd["consecutive"] += 1
        if cd["consecutive"] >= MAX_CONSECUTIVE_FAILS:
            duration = LONG_COOLDOWN_SECS
        else:
            duration = COOLDOWN_SECS
        cd["until"] = now + duration
        _cooldowns[key] = cd

    label = "direct" if proxy_url is None else key
    log.info(
        "Proxy cooldown: %s → %ds (consecutive=%d)",
        label, duration, cd["consecutive"],
    )


def report_success(proxy_url: str | None) -> None:
    """Clear cooldown for a proxy after a successful request."""
    key = _url_to_key(proxy_url)
    with _lock:
        if key in _cooldowns:
            _cooldowns.pop(key)
            label = "direct" if proxy_url is None else key
            log.debug("Proxy healed: %s", label)


def status() -> dict:
    """Return current pool status for diagnostics."""
    now = time.monotonic()
    with _lock:
        healthy = 0
        cooling = 0
        for entry in _PROXY_LIST:
            key = _proxy_key(entry)
            cd = _cooldowns.get(key)
            if cd and now < cd["until"]:
                cooling += 1
            else:
                healthy += 1
        return {
            "total": len(_PROXY_LIST),
            "healthy": healthy,
            "cooling_down": cooling,
            "cooldowns": {
                k: {"remaining": max(0, round(v["until"] - now)),
                    "consecutive_fails": v["consecutive"]}
                for k, v in _cooldowns.items()
                if now < v["until"]
            },
        }
