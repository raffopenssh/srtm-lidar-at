"""Rotating HTTP proxy pool for BEV GeoTIFF reads with adaptive healing.

Fetches free proxy lists from GitHub aggregators, validates them against
BEV, and maintains a pool of working proxies.  Proxies that fail get
exponentially longer cooldowns (60s up to 3 days).  Successful proxies
get their failure scores decayed.

Pool is refreshed every REFRESH_INTERVAL (default 30 min) in a background
thread.  Proxy history is persisted to disk so learned bad proxies stay
cooled down across restarts.

Usage:
    proxy_url = bev_proxy.next_proxy()   # returns URL or None (direct)
    bev_proxy.report_failure(proxy_url)  # put on cooldown
    bev_proxy.report_success(proxy_url)  # decay failure score
"""
import concurrent.futures
import json
import logging
import os
import random
import subprocess
import time
import threading
import urllib.request

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Free proxy list sources (GitHub-hosted, updated periodically)
# ---------------------------------------------------------------------------
_PROXY_SOURCES = [
    # Original five (2026-05-XX)
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/https/data.txt",
    "https://raw.githubusercontent.com/ErcinDedeoglu/proxies/main/proxies/https.txt",
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    "https://raw.githubusercontent.com/ALIILAPRO/Proxy/main/http.txt",
    "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/protocols/http/data.txt",
    # Added 2026-05-28 to widen the funnel: BEV validation yields ~0
    # passes from the original five, so the pool sits at 0 proxies and
    # peers thrash on "All proxies on cooldown" + direct-only retries.
    # The phase-1 HTTPS probe and phase-2 two-tile BigTIFF-magic check
    # still gate everything, so a broader funnel only helps.
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/https.txt",
    "https://raw.githubusercontent.com/MuRongPIG/Proxy-Master/main/http.txt",
    "https://raw.githubusercontent.com/zloi-user/hideip.me/master/http.txt",
    "https://raw.githubusercontent.com/zloi-user/hideip.me/master/https.txt",
    "https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt",
    "https://raw.githubusercontent.com/r00tee/Proxy-List/main/Https.txt",
    "https://raw.githubusercontent.com/mmpx12/proxy-list/master/http.txt",
    "https://raw.githubusercontent.com/mmpx12/proxy-list/master/https.txt",
    "https://raw.githubusercontent.com/Anonym0usWork1221/Free-Proxies/main/proxy_files/http_proxies.txt",
    "https://raw.githubusercontent.com/Anonym0usWork1221/Free-Proxies/main/proxy_files/https_proxies.txt",
    "https://api.openproxylist.xyz/http.txt",
    "https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&protocol=http&timeout=10000&country=all",
    "https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&protocol=https&timeout=10000&country=all",
]

# URLs used to validate that a proxy can actually fetch BEV TIFF bytes.
# We probe TWO distinct tiles + check the TIFF magic byte on each so
# proxies that return cached error pages or partial bodies are caught
# before they enter the pool.  See 2026-05-28 post-mortem.
_BEV_TEST_URL = "https://data.bev.gv.at/download/ALS/DTM/20220915/ALS_DTM_CRS3035RES50000mN2800000E4750000.tif"
_BEV_TEST_URL_2 = "https://data.bev.gv.at/download/ALS/DTM/20240915/ALS_DTM_CRS3035RES50000mN2750000E4500000.tif"
# BigTIFF magic: 'II' (little-endian) + 0x002B (BigTIFF version).
# Classic TIFF would be 'II' + 0x002A.  BEV uses BigTIFF for these tiles
# but accept either to be safe.
_TIFF_MAGIC_LE = (b"II*\x00", b"II+\x00")

# ---------------------------------------------------------------------------
# Pool configuration
# ---------------------------------------------------------------------------
MIN_POOL_SIZE = 5               # minimum proxies to keep in active pool
MAX_POOL_SIZE = 30              # cap active pool
VALIDATION_WORKERS = 60         # parallel validation threads
VALIDATION_TIMEOUT = 10         # seconds per proxy test
REFRESH_INTERVAL = 1800         # re-fetch & validate every 30 min
DIRECT_WEIGHT = 3               # how many "direct" slots in rotation

# ---------------------------------------------------------------------------
# Healing / cooldown configuration
# ---------------------------------------------------------------------------
# 2026-05-28: bumped 60s→300s.  Most free-proxy failures we see are
# slow CONNECT/timeout patterns where the proxy was *probably* healthy
# in the validation phase 30 min ago but flaked under real load.  A
# 60s cooldown lets such proxies rejoin the pool immediately and keep
# producing transport failures; 300s = 5 min gives time for the next
# refresh cycle to re-test them before they get another shot.
BASE_COOLDOWN = 300             # initial cooldown (seconds)
MAX_COOLDOWN = 3 * 86400        # cap at 3 days
COOLDOWN_EXPONENT = 2.0         # cooldown = BASE * EXPONENT^(fail_score - 1)
SUCCESS_DECAY = 0.5             # on success, fail_score *= this
HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "proxy_history.json"
)

# ---------------------------------------------------------------------------
# Runtime state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_index = 0                      # round-robin pointer

# Active pool: list of proxy strings ("ip:port") + None entries for direct
_pool: list[str | None] = [None] * DIRECT_WEIGHT

# Per-proxy state: key -> {"fail_score": float, "cooldown_until": float (monotonic),
#                          "wall_cooldown_until": float (epoch), "last_fail": float (epoch)}
_state: dict[str, dict] = {}
_history_loaded = False
_refresh_thread: threading.Thread | None = None
_refresh_started = False


# ---------------------------------------------------------------------------
# Proxy key helpers
# ---------------------------------------------------------------------------
def _proxy_key(entry: str | None) -> str:
    if entry is None:
        return "__direct__"
    return entry


def _url_to_key(proxy_url: str | None) -> str:
    if proxy_url is None:
        return "__direct__"
    # Extract ip:port from http://ip:port
    url = proxy_url
    if "://" in url:
        url = url.split("://", 1)[1]
    url = url.rstrip("/")
    return url


# ---------------------------------------------------------------------------
# Cooldown math
# ---------------------------------------------------------------------------
def _compute_cooldown(fail_score: float) -> float:
    if fail_score <= 0:
        return 0
    duration = BASE_COOLDOWN * (COOLDOWN_EXPONENT ** (fail_score - 1))
    return min(duration, MAX_COOLDOWN)


def _fmt_duration(secs: float) -> str:
    if secs < 120:
        return f"{secs:.0f}s"
    if secs < 7200:
        return f"{secs / 60:.0f}m"
    if secs < 172800:
        return f"{secs / 3600:.1f}h"
    return f"{secs / 86400:.1f}d"


# ---------------------------------------------------------------------------
# Persistent history
# ---------------------------------------------------------------------------
def _ensure_dir():
    d = os.path.dirname(HISTORY_FILE)
    if d:
        os.makedirs(d, exist_ok=True)


def _load_history():
    global _history_loaded
    if _history_loaded:
        return
    _history_loaded = True
    try:
        with open(HISTORY_FILE) as f:
            saved = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return

    now_mono = time.monotonic()
    now_wall = time.time()

    for key, info in saved.items():
        fail_score = info.get("fail_score", 0)
        wall_until = info.get("wall_cooldown_until", 0)
        remaining = wall_until - now_wall
        if remaining > 0:
            _state[key] = {
                "fail_score": fail_score,
                "cooldown_until": now_mono + remaining,
                "wall_cooldown_until": wall_until,
                "last_fail": info.get("last_fail", 0),
            }
            log.info(
                "Proxy %s: restored — fail_score=%.1f, cooldown remaining=%s",
                key, fail_score, _fmt_duration(remaining),
            )
        elif fail_score > 0:
            hours_since = max(0, now_wall - info.get("last_fail", now_wall)) / 3600
            decayed = fail_score * (0.9 ** hours_since)
            if decayed > 0.5:
                _state[key] = {
                    "fail_score": decayed,
                    "cooldown_until": 0,
                    "wall_cooldown_until": 0,
                    "last_fail": info.get("last_fail", 0),
                }


def _save_history():
    try:
        _ensure_dir()
        to_save = {}
        for key, info in _state.items():
            if key == "__direct__":
                continue
            if info.get("fail_score", 0) > 0.5:
                to_save[key] = {
                    "fail_score": round(info["fail_score"], 2),
                    "wall_cooldown_until": round(info.get("wall_cooldown_until", 0), 1),
                    "last_fail": round(info.get("last_fail", 0), 1),
                }
        tmp = HISTORY_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(to_save, f, indent=2)
        os.replace(tmp, HISTORY_FILE)
    except OSError as e:
        log.debug("Failed to save proxy history: %s", e)


# ---------------------------------------------------------------------------
# Proxy fetching & validation
# ---------------------------------------------------------------------------
def _fetch_candidates() -> set[str]:
    """Fetch proxy candidates from all free-proxy-list sources."""
    candidates: set[str] = set()
    for url in _PROXY_SOURCES:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="ignore")
            for line in data.strip().split("\n"):
                line = line.strip()
                if line.startswith("http://"):
                    line = line[7:]
                elif line.startswith("https://"):
                    line = line[8:]
                # Must start with ip:port. Some sources append a
                # third ':country' / ':Anonymous' field (e.g.
                # hideip.me "ip:port:Country") which we tolerate by
                # taking just the first two colon-separated fields.
                if ":" in line and "/" not in line and " " not in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        try:
                            int(parts[1])
                            candidates.add(f"{parts[0]}:{parts[1]}")
                        except ValueError:
                            pass
        except Exception as e:
            log.debug("Failed to fetch %s: %s", url, e)
    log.info("Fetched %d unique proxy candidates from %d sources",
             len(candidates), len(_PROXY_SOURCES))
    return candidates


def _validate_proxy(proxy: str) -> tuple[str, bool, float]:
    """Phase-2 validation: must fetch real BigTIFF bytes from two BEV tiles.

    Returns ``(proxy, ok, latency)``.  ``ok=True`` only if BOTH tiles
    returned 206 with valid TIFF magic bytes — catches proxies that
    return error pages or partial bodies (the dominant failure mode we
    saw in the 24h post-mortem 2026-05-28).
    """
    t0 = time.time()
    import tempfile
    def _probe(url: str) -> bool:
        try:
            with tempfile.NamedTemporaryFile(delete=True) as tf:
                r = subprocess.run(
                    ["curl", "-s", "-o", tf.name, "-w", "%{http_code}",
                     "--proxy", f"http://{proxy}",
                     "--range", "0-1023",
                     "--connect-timeout", "5",
                     "--max-time", str(VALIDATION_TIMEOUT),
                     url],
                    capture_output=True, text=True,
                    timeout=VALIDATION_TIMEOUT + 5,
                )
                code = r.stdout.strip()
                if code not in ("200", "206"):
                    return False
                head = open(tf.name, "rb").read(4)
                return any(head.startswith(m) for m in _TIFF_MAGIC_LE)
        except Exception:
            return False
    ok = _probe(_BEV_TEST_URL) and _probe(_BEV_TEST_URL_2)
    return proxy, ok, time.time() - t0


def _refresh_pool():
    """Fetch, validate, and update the active proxy pool."""
    global _pool

    candidates = _fetch_candidates()
    if not candidates:
        log.warning("No proxy candidates fetched — keeping current pool")
        return

    # Skip proxies that are on long cooldown (known bad)
    now_wall = time.time()
    with _lock:
        _load_history()
        skip = set()
        for key, st in _state.items():
            if key == "__direct__":
                continue
            # Skip if cooldown remaining > 10 min
            remaining = st.get("wall_cooldown_until", 0) - now_wall
            if remaining > 600:
                skip.add(key)

    to_test = [p for p in candidates if p not in skip]
    # Shuffle and cap to avoid testing thousands
    random.shuffle(to_test)
    # Cap validation budget. 2026-05-28: bumped 2000→5000 alongside the
    # ~20x wider source funnel (~100k unique candidates) so the random
    # subsample stays representative — at 60 parallel workers + 8s phase-1
    # budget this is ~12 min wall time, comfortably under the 30 min
    # REFRESH_INTERVAL.
    to_test = to_test[:5000]

    log.info("Validating %d proxy candidates (%d skipped on cooldown)...",
             len(to_test), len(skip))

    # Phase 1: quick HTTPS CONNECT check against httpbin (faster than BEV)
    def quick_test(proxy: str) -> tuple[str, bool]:
        try:
            r = subprocess.run(
                ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                 "--proxy", f"http://{proxy}",
                 "--connect-timeout", "3",
                 "--max-time", "5",
                 "https://httpbin.org/ip"],
                capture_output=True, text=True, timeout=8,
            )
            return proxy, r.stdout.strip() == "200"
        except Exception:
            return proxy, False

    https_capable = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=VALIDATION_WORKERS) as ex:
        for proxy, ok in ex.map(quick_test, to_test):
            if ok:
                https_capable.append(proxy)

    log.info("Phase 1: %d/%d support HTTPS CONNECT", len(https_capable), len(to_test))

    if not https_capable:
        log.warning("No HTTPS-capable proxies found — keeping current pool")
        return

    # Phase 2: validate against BEV specifically
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(30, len(https_capable))) as ex:
        bev_results = list(ex.map(_validate_proxy, https_capable))

    # Sort by latency, take the best
    working = [(p, lat) for p, ok, lat in bev_results if ok]
    working.sort(key=lambda x: x[1])

    log.info("Phase 2: %d/%d work with BEV", len(working), len(https_capable))

    if not working:
        log.warning("No proxies passed BEV validation — keeping current pool")
        return

    # Build new pool: direct slots + best proxies
    new_proxies = [p for p, _ in working[:MAX_POOL_SIZE]]
    new_pool: list[str | None] = []
    # Interleave direct and proxy: direct, proxy, proxy, direct, proxy, proxy, ...
    pi = 0
    for i in range(len(new_proxies) + DIRECT_WEIGHT):
        if i % (1 + len(new_proxies) // max(1, DIRECT_WEIGHT)) == 0 and new_pool.count(None) < DIRECT_WEIGHT:
            new_pool.append(None)
        elif pi < len(new_proxies):
            new_pool.append(new_proxies[pi])
            pi += 1
    # Add any remaining proxies
    while pi < len(new_proxies):
        new_pool.append(new_proxies[pi])
        pi += 1
    # Ensure at least one direct
    if None not in new_pool:
        new_pool.insert(0, None)

    with _lock:
        _pool = new_pool

    proxy_count = sum(1 for p in new_pool if p is not None)
    direct_count = sum(1 for p in new_pool if p is None)
    log.info(
        "Pool refreshed: %d proxies + %d direct slots (best latency: %.1fs)",
        proxy_count, direct_count, working[0][1] if working else 0,
    )
    if working:
        for p, lat in working[:5]:
            log.info("  %s (%.1fs)", p, lat)


def _refresh_loop():
    """Background thread: refresh pool periodically."""
    while True:
        try:
            _refresh_pool()
        except Exception as e:
            log.error("Proxy pool refresh failed: %s", e)
        time.sleep(REFRESH_INTERVAL)


def _ensure_refresh_started():
    """Start the background refresh thread if not already running."""
    global _refresh_thread, _refresh_started
    if _refresh_started:
        return
    _refresh_started = True
    _refresh_thread = threading.Thread(target=_refresh_loop, daemon=True, name="proxy-refresh")
    _refresh_thread.start()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def next_proxy() -> str | None:
    """Return next healthy proxy URL or None for direct access."""
    global _index
    now = time.monotonic()

    with _lock:
        _load_history()
        _ensure_refresh_started()

        pool = _pool
        n = len(pool)
        if n == 0:
            return None

        # Try up to a full rotation to find a healthy entry
        for _ in range(n):
            entry = pool[_index % n]
            _index = (_index + 1) % n
            key = _proxy_key(entry)
            st = _state.get(key)
            if st is None or now >= st.get("cooldown_until", 0):
                if entry is None:
                    return None
                return f"http://{entry}"

        # All on cooldown — pick the one expiring soonest
        best_entry = pool[0]
        best_until = float("inf")
        for entry in pool:
            key = _proxy_key(entry)
            st = _state.get(key)
            if st and st.get("cooldown_until", 0) < best_until:
                best_until = st["cooldown_until"]
                best_entry = entry
        remaining = max(0, best_until - now)
        log.warning(
            "All proxies on cooldown — using least-recently-failed (expires in %s)",
            _fmt_duration(remaining),
        )
        if best_entry is None:
            return None
        return f"http://{best_entry}"


def report_failure(proxy_url: str | None, error_msg: str = "") -> None:
    """Record a failure and put the proxy on adaptive cooldown.

    CONNECT tunnel / 402 errors bump the score by 2 (known bad proxy),
    other transient errors by 1.
    """
    key = _url_to_key(proxy_url)
    now_mono = time.monotonic()
    now_wall = time.time()

    is_hard_fail = any(p in error_msg.lower() for p in (
        "connect tunnel", "response 402", "407", "proxy auth",
        "bandwidthlimit", "payment required",
    ))
    bump = 2.0 if is_hard_fail else 1.0

    with _lock:
        _load_history()
        st = _state.get(key, {
            "fail_score": 0, "cooldown_until": 0,
            "wall_cooldown_until": 0, "last_fail": 0,
        })
        st["fail_score"] = st["fail_score"] + bump
        st["last_fail"] = now_wall
        duration = _compute_cooldown(st["fail_score"])
        st["cooldown_until"] = now_mono + duration
        st["wall_cooldown_until"] = now_wall + duration
        _state[key] = st

        label = "direct" if proxy_url is None else key
        log.info(
            "Proxy cooldown: %s → %s (fail_score=%.1f%s)",
            label, _fmt_duration(duration), st["fail_score"],
            " [CONNECT/402]" if is_hard_fail else "",
        )

        _save_history()


def report_success(proxy_url: str | None) -> None:
    """Decay fail_score on success and clear cooldown."""
    key = _url_to_key(proxy_url)
    with _lock:
        st = _state.get(key)
        if st is None:
            return
        old_score = st["fail_score"]
        st["fail_score"] = old_score * SUCCESS_DECAY
        st["cooldown_until"] = 0
        st["wall_cooldown_until"] = 0
        if old_score > 0.5:
            label = "direct" if proxy_url is None else key
            log.info(
                "Proxy healed: %s (fail_score %.1f → %.1f)",
                label, old_score, st["fail_score"],
            )
        if st["fail_score"] < 0.1:
            _state.pop(key, None)
        _save_history()


def status() -> dict:
    """Return current pool status for diagnostics."""
    now = time.monotonic()
    with _lock:
        _load_history()
        healthy = 0
        cooling = 0
        entries = []
        for entry in _pool:
            key = _proxy_key(entry)
            st = _state.get(key)
            if st and now < st.get("cooldown_until", 0):
                cooling += 1
                remaining = st["cooldown_until"] - now
                entries.append({
                    "proxy": key,
                    "status": "cooldown",
                    "remaining": _fmt_duration(remaining),
                    "remaining_secs": round(remaining),
                    "fail_score": round(st["fail_score"], 1),
                })
            else:
                healthy += 1
                score = st["fail_score"] if st else 0
                entries.append({
                    "proxy": key,
                    "status": "healthy",
                    "fail_score": round(score, 1),
                })
        return {
            "total": len(_pool),
            "healthy": healthy,
            "cooling_down": cooling,
            "entries": entries,
        }
