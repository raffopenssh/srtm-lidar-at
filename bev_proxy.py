"""Rotating HTTP proxy pool for BEV GeoTIFF reads.

Alternates between direct access and Webshare proxies to avoid
BEV rate-limiting on sustained /vsicurl/ range requests.

Usage:
    proxy_url = bev_proxy.next_proxy()  # returns 'http://user:pass@ip:port' or None
"""
import itertools
import logging
import os
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

_cycle = itertools.cycle(_PROXY_LIST)
_lock = threading.Lock()


def next_proxy() -> str | None:
    """Return next proxy URL or None for direct access."""
    with _lock:
        entry = next(_cycle)
    if entry is None:
        return None
    ip, port = entry
    return f"http://{_PROXY_USER}:{_PROXY_PASS}@{ip}:{port}"
