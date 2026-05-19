"""Peer Director — Orchestrates processing across multiple exe.dev VMs.

Each VM has 100 GB/month bandwidth (resets on the 17th). The director:
1. Tracks bandwidth usage per peer via vnstat
2. Chooses which peer should be active based on remaining budget
3. Starts/stops processors on peers via their REST API
4. Proxies status/logs from the active peer to the dashboard
5. Operates sequentially — only one peer processes at a time
   (Copernicus credentials are shared, so parallelism would cause 402s)

Peer config: data/austria_processor/peers.json
State: data/austria_processor/director_state.json
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)


def _emit_director_event(msg: str, *, peer: str = '', kg: str = '',
                         level: str = 'info') -> None:
    """Lazy bridge into ``app.director_event`` so orchestration events
    (credential / cache-cell plan changes, peer updates) show up in the
    dashboard's 24h merged log alongside peer-sourced lines. Safe to
    call from any thread; never raises.
    """
    try:
        from app import director_event as _de
        _de(msg, peer=peer, kg=kg, level=level)
    except Exception:
        pass

# Git commit hash (read once at import)
try:
    _LOCAL_GIT_COMMIT = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        cwd=str(Path(__file__).parent), stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    _LOCAL_GIT_COMMIT = 'unknown'

# Cache of {peer_commit: bool} — True iff peer_commit is a descendant of
# (or equal to) _LOCAL_GIT_COMMIT, i.e. the peer is at or *ahead* of the
# director's frozen-at-import target. We must not flag such peers as
# "stale / needs_manual_update": pulling on them would *downgrade* them.
# Happens routinely when the director was started before a fresh git
# push to origin/main and peers already auto-pulled past us.
_PEER_COMMIT_AHEAD_CACHE: dict[str, bool] = {}

def _peer_commit_is_ahead_or_equal(peer_commit: str) -> bool:
    """True iff *peer_commit* contains _LOCAL_GIT_COMMIT in its history.

    Cached; resolved via ``git merge-base --is-ancestor LOCAL PEER``.
    Returns False on any git error (peer commit unknown locally, etc.).
    """
    if not peer_commit or _LOCAL_GIT_COMMIT in ('', 'unknown'):
        return False
    if peer_commit == _LOCAL_GIT_COMMIT:
        return True
    cached = _PEER_COMMIT_AHEAD_CACHE.get(peer_commit)
    if cached is not None:
        return cached
    try:
        rc = subprocess.call(
            ['git', 'merge-base', '--is-ancestor',
             _LOCAL_GIT_COMMIT, peer_commit],
            cwd=str(Path(__file__).parent),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        result = (rc == 0)
    except Exception:
        result = False
    _PEER_COMMIT_AHEAD_CACHE[peer_commit] = result
    return result

DATA_DIR = Path('data/austria_processor')
PEERS_CONFIG = DATA_DIR / 'peers.json'
DIRECTOR_STATE = DATA_DIR / 'director_state.json'

# Cluster-wide admin token (shared with app.py via data/admin_token).
# Must be the same on every peer or the director's outbound calls 401.
_ADMIN_TOKEN_PATH = Path('data/admin_token')


def _admin_headers() -> dict:
    """Return headers for outbound peer admin calls.

    Re-reads the token on every call so that token rotations propagate
    without a director restart.
    """
    try:
        tok = _ADMIN_TOKEN_PATH.read_text().strip()
        if tok:
            return {'X-Admin-Token': tok}
    except Exception:
        pass
    return {}

# Bandwidth budget per peer per billing cycle (bytes)
BANDWIDTH_BUDGET_GB = 95  # conservative — leave 5 GB headroom out of 100
BANDWIDTH_BUDGET_BYTES = BANDWIDTH_BUDGET_GB * (1024 ** 3)
BANDWIDTH_RENEW_DAY = 17  # day of month when exe.dev bandwidth resets


def _peer_budget_bytes(peer: dict, cfg: dict) -> int:
    """Per-peer bandwidth budget in bytes.

    Resolution order: ``peer.budget_gb`` (override) >
    ``cfg.budget_gb`` (global) > ``BANDWIDTH_BUDGET_GB`` (default).
    Used to let a few canary peers run past the conservative global
    cap so we can probe what exe.dev's real billing-cycle limits are
    without disturbing the rest of the fleet.
    """
    try:
        per = peer.get('budget_gb') if isinstance(peer, dict) else None
        if per is not None:
            per = float(per)
            if per > 0:
                return int(per * (1024 ** 3))
    except (TypeError, ValueError):
        pass
    g = cfg.get('budget_gb', BANDWIDTH_BUDGET_GB) if isinstance(cfg, dict) else BANDWIDTH_BUDGET_GB
    return int(g * (1024 ** 3))

# Number of enabled peers to keep idle as reserve (never started by
# the director).  Operational headroom for ad-hoc work, RF training,
# and bandwidth/credential burst capacity.
#
# Now that spinning up new peers is cheap and unlimited, the default
# reserve is 0 — use every enabled peer.  The capacity factor + Zenodo
# upload mutex + per-peer cooldowns are sufficient to keep us tender
# towards upstream servers.  Set ``min_reserve_peers`` in peers.json if
# you really want some peers idle (e.g. during ad-hoc development).
MIN_RESERVE_PEERS = 0
# How many cache-only peers may run concurrently with the frontier peer.
# Each cache-only peer only does BEV reads + CPU + Zenodo upload — no
# Copernicus credentials — so we can run as many as the whitelist and
# the reserve policy permit. The Zenodo upload mutex serialises writes
# so peers wait their turn rather than fighting. 64 is a soft ceiling
# well above the current fleet size; it just prevents the orchestrator
# from spinning up an absurd number of peers if min_reserve is misset.
MAX_CACHE_ONLY_PEERS = 64

# Per-peer circuit breaker. Each peer has a (failure_count, open_until)
# tuple. After CB_OPEN_THRESHOLD consecutive failures across *any*
# director-outbound HTTP call, the peer is skipped for CB_OPEN_SECONDS
# (exponential up to CB_OPEN_MAX_SECONDS) on every subsequent
# `_peer_request`. A single success resets the counter. Distinct from
# the bandwidth-specific backoff above (which only suppresses bandwidth
# polls), this stops the director from spending its whole tick waiting
# for a wedged peer's gunicorn to time out.
CB_OPEN_THRESHOLD = 3
CB_OPEN_BASE_SECONDS = 60        # 1 min on first trip
CB_OPEN_MAX_SECONDS = 1800       # cap at 30 min
_PEER_CB: dict[str, dict] = {}   # peer_id -> {fails, open_until, trips}
_PEER_CB_LOCK = threading.Lock()
# Reverse map: peer_url -> peer_id. Updated by the director loop each
# tick from peers.json so module-level helpers (get_peer_bandwidth,
# get_peer_status) can attribute failures to the right peer without
# threading peer_id through every call site.
_PEER_URL_TO_ID: dict[str, str] = {}
# Process-local throttle for revalidate_all_credentials. Each gunicorn
# worker has its own copy; the primary check is still the persisted
# state['_creds_revalidated_at'], but this avoids hitting OIDC twice
# in the same worker within a tick.
#
# Single-flight semantics: only one thread per process may run the
# OIDC probe at a time. Other threads see the throttle window as
# closed and use cached ``last_status`` instead. This is what stops
# the dashboard-induced OIDC storm: with 4 threads × 2 workers each
# racing past a stale timestamp, we'd otherwise issue 8 OIDC requests
# × 8 credentials = 64 outbound requests per tick.
_LAST_CREDS_REVALIDATED_AT: float = 0.0
_REVALIDATE_LOCK = threading.Lock()
# Hard floor on revalidation cadence. Even the director loop won't
# probe more often than this. The dashboard hot path uses cached
# ``last_status`` from ``copernicus.list_credentials()``; that cache
# is updated by ``_save_credentials_to_disk()`` after each probe.
_REVALIDATE_INTERVAL_S = 600
# Hard kill switch: only the director loop is permitted to call
# ``revalidate_all_credentials()``. The dashboard path
# (``get_status`` → ``_valid_credentials``) reads cached statuses
# only. This is the architectural fix for the OIDC storm: the hot
# request path can never trigger network probes, regardless of
# any cache miss or worker swap.
_REVALIDATE_OWNER_THREAD: int | None = None


def _peer_id_for_url(peer_url: str) -> str:
    if not peer_url:
        return ''
    return _PEER_URL_TO_ID.get(peer_url.rstrip('/'), '')


class _CircuitOpenError(Exception):
    """Raised when a peer's breaker is open and we skip the call."""


def _cb_should_skip(peer_id: str) -> bool:
    if not peer_id:
        return False
    with _PEER_CB_LOCK:
        ent = _PEER_CB.get(peer_id)
        if not ent:
            return False
        return time.time() < ent.get('open_until', 0.0)


def _cb_record_success(peer_id: str) -> None:
    if not peer_id:
        return
    with _PEER_CB_LOCK:
        if peer_id in _PEER_CB:
            _PEER_CB[peer_id]['fails'] = 0
            _PEER_CB[peer_id]['open_until'] = 0.0


def _cb_record_failure(peer_id: str, exc: Exception | None = None) -> None:
    if not peer_id:
        return
    with _PEER_CB_LOCK:
        ent = _PEER_CB.setdefault(
            peer_id, {'fails': 0, 'open_until': 0.0, 'trips': 0})
        ent['fails'] = int(ent.get('fails', 0)) + 1
        if ent['fails'] >= CB_OPEN_THRESHOLD:
            trips = int(ent.get('trips', 0)) + 1
            ent['trips'] = trips
            cooldown = min(
                CB_OPEN_MAX_SECONDS,
                CB_OPEN_BASE_SECONDS * (2 ** min(trips - 1, 8)))
            ent['open_until'] = time.time() + cooldown
            ent['fails'] = 0
            log.warning(
                'Peer %s circuit breaker OPEN for %ds (trip #%d, last err=%s)',
                peer_id, int(cooldown), trips, str(exc)[:120] if exc else 'n/a')


def _cb_state_for(peer_id: str) -> dict:
    """Return a serialisable snapshot of one peer's breaker state."""
    if not peer_id:
        return {}
    with _PEER_CB_LOCK:
        ent = _PEER_CB.get(peer_id)
        if not ent:
            return {'open': False, 'fails': 0, 'trips': 0}
        now = time.time()
        open_until = float(ent.get('open_until', 0.0))
        return {
            'open': now < open_until,
            'fails': int(ent.get('fails', 0)),
            'trips': int(ent.get('trips', 0)),
            'cooldown_remaining_s': max(0, int(open_until - now)),
        }


def _peer_request(method: str, url: str, *,
                  peer_id: str = '',
                  timeout=None,
                  raise_circuit: bool = False,
                  **kwargs):
    """requests.request wrapper with per-peer circuit breaker.

    * Refuses (raises ``_CircuitOpenError`` / returns None) when the
      peer's breaker is open.
    * On any transport exception or 5xx response, records a failure.
    * On 2xx/3xx/4xx (i.e. peer responded), records a success.

    Callers that already have peer-id context should pass ``peer_id``.
    Returns the ``Response`` on success or None if the call was
    short-circuited (and ``raise_circuit`` is False).
    """
    if peer_id and _cb_should_skip(peer_id):
        if raise_circuit:
            raise _CircuitOpenError(peer_id)
        return None
    if timeout is None:
        timeout = PEER_TIMEOUT_PROBE  # resolved at call time
    kwargs.setdefault('headers', {})
    if 'X-Admin-Token' not in kwargs['headers']:
        kwargs['headers'].update(_admin_headers())
    try:
        r = requests.request(method, url, timeout=timeout, **kwargs)
    except Exception as e:
        _cb_record_failure(peer_id, e)
        raise
    if r.status_code >= 500:
        _cb_record_failure(peer_id, RuntimeError(f'HTTP {r.status_code}'))
    else:
        _cb_record_success(peer_id)
    return r


# Per-peer dedup for graceful cache-only stop signals. The processor's
# signal handler sets _shutdown_requested=True but only exits *after*
# the current KG finishes — sometimes 1–2h later. Without dedup, every
# director tick re-sends SIGTERM (we observed 63 in 36h on one peer).
# Map peer_id → unix ts of last graceful stop emitted by this director.
_LAST_GRACEFUL_STOP_TS: dict[str, float] = {}

# How often the director checks state (seconds)
DIRECTOR_POLL_INTERVAL = 30
# Grace period after stopping a peer before starting another (seconds)
SWITCH_COOLDOWN = 10
# HTTP timeout for peer API calls.  Tuple = (connect_timeout, read_timeout).
# A 3 s connect catches DNS / TLS-handshake hangs without waiting forever.
# Read timeout is short for cheap probes (bandwidth/status) and longer for
# operations that legitimately take time (start/stop, cache_manifest sync).
PEER_TIMEOUT = (3, 8)
PEER_TIMEOUT_PROBE = (3, 8)         # bandwidth, status, log
PEER_TIMEOUT_CONTROL = (5, 25)      # start/stop/cache-manifest/queue PUT
# Bandwidth poll concurrency — keep small to avoid ephemeral-port exhaustion
# but enough that the loop finishes well within DIRECTOR_POLL_INTERVAL even
# when most peers are wedged.
BANDWIDTH_POLL_CONCURRENCY = 10
# After this many consecutive bandwidth-poll failures, back off polling
# the peer for BANDWIDTH_BACKOFF_SECONDS so a single dead peer can't drag
# out the loop on every tick.
BANDWIDTH_BACKOFF_THRESHOLD = 3
BANDWIDTH_BACKOFF_SECONDS = 900     # 15 min — be tender with flaky peers
# Number of consecutive unreachable polls before failover (avoids killing
# peers during heavy GPKG builds that briefly starve gunicorn).
UNREACHABLE_FAILOVER_THRESHOLD = 3
# How long to keep a peer out of rotation after a local Zenodo network
# failure (the same peer hitting the same network problem on retry would
# loop forever).  Cleared automatically when not_before passes.
# 60 min base — Zenodo's targeted rate-limits are slow to clear and we'd
# rather a peer sit out longer than thrash in/out.
ZENODO_NETWORK_COOLDOWN_MIN = 60
# Escalating cooldown for peers with a 'hold tendency' (repeat offenders).
# Each cooldown applied within HOLD_TENDENCY_WINDOW_HOURS is counted; the next
# cooldown is multiplied by 2**(count-1), capped at HOLD_TENDENCY_MAX_MIN.
# Short window so we react quickly to targeted rate-limits: a peer that
# trips twice within ~3 h is almost certainly being throttled by name/IP
# and should sit out aggressively rather than thrash in/out every 30 min.
# Aggressive escalation factor: each repeat multiplies the cooldown by
# HOLD_TENDENCY_FACTOR (cubic-ish growth instead of doubling).
# Wider window so we count repeats over a longer baseline; with cheap
# peers we can afford to keep an offender out for the rest of the day.
HOLD_TENDENCY_WINDOW_HOURS = 6
HOLD_TENDENCY_FACTOR = 3
HOLD_TENDENCY_MAX_MIN = 24 * 60   # 24 h ceiling

# --- Server-friendliness throttle ----------------------------------------
#
# When BEV / Zenodo / Copernicus servers start emitting warnings (HTTP 0
# range-read drops, 429s, 503s, openEO 402s) we voluntarily reduce the
# number of concurrent peers so we don't hammer them. The capacity factor
# is recomputed on every director tick from the fleet-wide warning rate.
#
# We never go below ``THROTTLE_MIN_FACTOR`` (so a single noisy peer can't
# drop the fleet to zero) and we never run more than 100% of the
# configured max. We also overlay a slow sinusoidal drift (period
# ~2 hours) so the activity pattern looks organic rather than
# bang‑on‑max all the time. Phase per peer comes from a stable hash so
# different VMs take their breaks at different times.
THROTTLE_MIN_FACTOR = 0.20
THROTTLE_MAX_FACTOR = 1.00
# warnings/min (per kind) at which the factor reaches its minimum.
# A few stray retries are normal; sustained > ~6/min is real pressure.
# Tightened down 2026-04-30: with a 20+ peer fleet a small per-peer
# warning rate aggregates into real fleet-wide pressure on upstream
# servers. Lower saturation rates ⇒ throttle bites earlier, peers back
# off sooner. We'd rather lose ~30 % of cache-only slots for an hour
# than get the whole token banned.
# Re-tuned 2026-04-23 for the ~50 peer fleet. Saturation is computed
# against the fleet-wide *max* per-peer rate, but ambient noise (one
# log line every few minutes per peer) aggregates faster as the fleet
# grows. Values below assume ~50 peers; scale up roughly proportional
# to len(peers)/50 if the fleet keeps growing.
THROTTLE_SATURATION_RATE = {
    'bev': 5.0,
    'zenodo': 1.5,        # Zenodo rate-limits aggressively
    'copernicus': 0.4,    # 402s should be near-zero in steady state
}
# Dead-zone: fleet-max rates below this fraction of saturation are
# treated as zero. Keeps ambient noise (a stale-manifest 404 every
# couple of minutes from one peer) from pulling capacity off 100%.
THROTTLE_DEAD_ZONE_FRAC = 0.10
# EMA smoothing factor for the capacity decision (per tick, ~30 s).
# Smaller = slower to react / recover. 0.25 gives a half‑life of ~3 ticks.
# Smaller alpha → slower reaction up *and* down. Recovery is slow on
# purpose: once we've upset Zenodo/BEV, easing back gradually is far
# safer than snapping back to full throttle the moment warnings stop.
THROTTLE_EMA_ALPHA = 0.30
# Slow per-peer noise EMA. Updated every director tick from the peer's
# current 5-min warning rate. Used by choose_active_peer / orchestrators
# so a peer that has been noisy in the last few hours stays penalised
# even after the 5/10-min sliding windows have rolled to zero. With a
# 30 s tick cadence, alpha=0.006 gives ~58 min half-life — a peer that
# upset Zenodo at 06:00 still scores noisy at 09:00.
PEER_NOISE_LONG_EMA_ALPHA = 0.006
# Sinusoidal drift: period and amplitude (fraction of total range).
THROTTLE_DRIFT_PERIOD_S = 2 * 3600    # 2 hours
# Sinusoidal drift amplitude. Reduced 2026-04-23 from 0.10 to 0.04 —
# the previous ±10 % was loud enough to flip the dashboard color
# between green/yellow/red on its own when the EMA sat near a
# threshold, even with no upstream pressure. Cosmetic, not operational.
THROTTLE_DRIFT_AMPLITUDE = 0.04        # ±4% wobble around the EMA value

# Ramp limiter: how many *new* peers we start per director tick. Without
# this, after a restart we can fire up 19 peers in ~2 seconds, each of
# which immediately POSTs cache-manifest sync + priority-queue PUT to
# the primary AND starts hammering BEV/Zenodo. A gentler ramp (1-3 per
# tick depending on capacity factor) lets each new peer warm its caches
# and the upstreams notice gradually. Empirically this is the difference
# between Zenodo accepting all uploads and Zenodo emitting blanket 429s
# on the whole cluster for 30 minutes.
RAMP_MIN_STARTS_PER_TICK = 1
RAMP_MAX_STARTS_PER_TICK = 3

# Minimum interval between processor restarts on the same active frontier
# peer when the cred/strip plan + exclude set haven't changed. The
# director sees ``state in ('idle','stopped')`` between KGs (the
# subprocess exits cleanly per-KG) and would otherwise fire a fresh
# start every 30s tick. Empirically we see ~8 restarts/hour without
# this guard. Plan changes always restart immediately regardless.
FRONTIER_RESTART_COOLDOWN_S = 180

# Canary bandwidth history & auto-park.
#
# As of 2026-05 every enabled peer is treated as a canary: exe.dev's
# real per-account billing-cycle limits are unknown (each VM was created
# in a different account at a different date) so we don't trust the
# global 95 GB / 17th-of-month assumption. Instead we sample every
# peer's cumulative ``used_bytes`` on every bandwidth poll and, if the
# observed throughput collapses (exe.dev started shaping), park that
# peer with ``not_before = now + cooldown`` and persist
# ``observed_cap_gb = used_gb-at-park`` in peers.json so we learn each
# peer's wall organically. Slowdown park applies to ALL peers; the
# stricter noise-based park (warning-rate spike) stays canary-only —
# generic noisy peers are already handled by load-shifting in the
# scheduler and we don't want to remove them from rotation just for
# being upstream-grumpy.
#
# A peer with a per-peer ``budget_gb`` override is still called a canary
# because its budget is intentionally raised above the conservative
# global cap to probe further; the dashboard surfaces this distinctly.
#
# Sampling cadence is whatever ``BANDWIDTH_POLL_INTERVAL`` is doing
# (every ~minute on the active poll loop, less for backed-off peers).
# Ring is bounded so the state file can't blow up.
CANARY_HISTORY_MAX = 240          # ~4h at 1 sample/min
CANARY_BASELINE_MIN_SAMPLES = 6   # need a baseline before we can compare
CANARY_BASELINE_WINDOW_S = 1800   # 30-min trailing baseline
CANARY_RECENT_WINDOW_S = 600      # 10-min recent throughput
CANARY_SLOWDOWN_RATIO = 0.30      # park if recent < 30% of baseline
CANARY_NOISE_PARK_THRESHOLD = 1.5 # park if noise_score >= this
CANARY_PARK_COOLDOWN_S = 6 * 3600 # 6 h not_before (quality-grade parks)
# Soft parks (defensive, not network-grade or not persistent) are
# almost always upstream blips (Zenodo upload stall, openEO 502,
# transient internet hiccup) rather than exe.dev shaping. A 6 h
# cooldown is wasteful here — the blip resolves in minutes and we
# burn hours of peer capacity. Use a shorter cooldown so soft-parked
# peers cycle back into rotation; if the underlying issue is real
# they’ll soft-park again on the next tick (and eventually graduate
# to quality-grade once persistence + network-grade both fire).
CANARY_PARK_COOLDOWN_SOFT_S = 60 * 60   # 1 h
# Don't trip the slowdown check until the canary has actually moved
# enough bytes for the average to be meaningful (otherwise a peer that
# happens to be idle between KGs reads as 'shaped').
CANARY_MIN_BYTES_FOR_PARK = 500 * 1024 * 1024  # 500 MB in baseline window
# Tightened thresholds to avoid false positives from peers that are
# Zenodo-upload-bound (their cumulative byte counter advances slowly
# even when the network is healthy).
#
# Park only when the peer was actually moving NETWORK-GRADE bytes
# before the collapse — baseline >= CANARY_BASELINE_NETWORK_MBPS —
# AND it has truly stalled — recent <= CANARY_RECENT_PARKED_MBPS.
# This keeps the watchdog active for real shaping events while not
# tripping when a peer happens to be uploading a 4 GB GPKG to Zenodo.
#
# Even tighter: only persist ``observed_cap_gb`` when the trigger
# meets these thresholds (so we don't pollute the fleet wall estimate
# with park events caused by upload-bound idleness).
CANARY_BASELINE_NETWORK_MBPS = 5.0   # was implicit, now explicit (BEV reads sustain >>5 MB/s)
CANARY_RECENT_PARKED_MBPS = 0.5      # really stalled, not just 'a bit slow'
# Fleet wall estimate is only published once we've gathered this many
# *quality* observations across distinct peers — so the dashboard
# doesn't claim a confident wall from 3 noisy datapoints.
FLEET_WALL_MIN_QUALITY_OBS = 5
# Time persistence: a peer's slowdown must be continuously observed for
# this many seconds before we count the park as a quality observation.
# 15 min eats roughly through any general internet hiccup or BEV/Zenodo
# blip without delaying real shaping detection meaningfully.
CANARY_QUALITY_PERSIST_S = 15 * 60
# A peer's slowdown streak resets when its ratio recovers ABOVE this
# value (sticky hysteresis around the park threshold of 0.30).
CANARY_SLOWDOWN_RECOVERY_RATIO = 0.60
# Cross-peer correlation: if a fraction ≥ this of peers with usable
# canary samples are currently in slowdown, treat as a fleet-wide
# upstream event — NO peer parked during this window earns a quality
# observation. Soft-park (defensive removal) still fires.
FLEET_CONCURRENT_SLOWDOWN_FRAC = 0.30

# Warmup hold for fresh peers. A brand-new peer has no tile cache and
# zero history; throwing it straight at frontier work means it starts
# by hammering Copernicus/BEV. We let it sit eligible-but-unused for a
# few minutes so the first cache-only or cache-manifest pre-sync can
# warm its local caches before it's asked to do real work.
#
# 'first_seen' is recorded in peers.json the first time the director
# observes a peer. Cache-only is allowed (it's the gentlest workload),
# but frontier promotion is held off for this many seconds.
WARMUP_HOLD_SECONDS = 5 * 60  # 5 minutes

# Cache-manifest sync backoff. After this many consecutive PUT failures
# we skip the peer's manifest sync for SYNC_BACKOFF_SECONDS, mirroring
# bandwidth backoff. Without this, a peer with a flaky network triggers
# a full sync on every start, every tick, blowing latency budgets.
SYNC_BACKOFF_THRESHOLD = 3
SYNC_BACKOFF_SECONDS = 600        # 10 min


def _default_peers_config() -> dict:
    """Return default peers.json structure."""
    return {
        'budget_gb': BANDWIDTH_BUDGET_GB,
        'renew_day': BANDWIDTH_RENEW_DAY,
        'peers': [
            {
                'id': 'primary',
                'url': None,  # null = this instance (local)
                'enabled': True,
            },
            {
                'id': 'at2',
                'url': 'https://srtm-lidar-at2.exe.xyz:8000',
                'enabled': True,
            },
        ],
    }


def load_peers_config() -> dict:
    """Load or create peers config.

    Always passes through ``director_ha.sanitise_peers_json`` first to
    repair URL corruption before any caller (director loop, dashboard,
    snapshot builder) sees it.
    """
    if PEERS_CONFIG.exists():
        try:
            import director_ha as _dha  # local import to avoid cycle
            try:
                rep = _dha.sanitise_peers_json()
                if rep.get('changes'):
                    log.warning('load_peers_config: sanitised peers.json: %s',
                                rep['changes'])
            except Exception as e:
                log.warning('load_peers_config: sanitise failed: %s', e)
            return json.loads(PEERS_CONFIG.read_text())
        except Exception:
            pass
    cfg = _default_peers_config()
    PEERS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    PEERS_CONFIG.write_text(json.dumps(cfg, indent=2))
    return cfg


# Sidecar fcntl lockfile for peers.json. We hold an exclusive flock
# here for the duration of every save_peers_config() call, so that
# concurrent writers (e.g. a gunicorn worker handling /peers/add and
# the director loop adding ``first_seen`` or a canary park) cannot
# race and silently clobber each other's roster changes. See
# ``save_peers_config`` for the merge-with-disk semantics that build
# on top of this lock.
PEERS_CONFIG_LOCK = PEERS_CONFIG.with_suffix(PEERS_CONFIG.suffix + '.lock')


def _merge_peers_with_disk(mem_cfg: dict,
                           disk_cfg: dict | None,
                           removed_ids: set | None = None) -> dict:
    """Reconcile ``mem_cfg`` with the freshly-read ``disk_cfg``.

    Goal: prevent two concurrent writers from silently dropping each
    other's peer-roster changes. Specifically the historical bug where
    ``/api/v1/director/peers/add`` (gunicorn worker) appended ``atNN``
    to its load-time snapshot, while the director loop (different
    worker) was simultaneously stamping ``first_seen`` on its own
    snapshot — whichever finished last clobbered the other.

    Rules:
    * Top-level fields (budget_gb, renew_day, frontier_plan, …) come
      from ``mem_cfg`` — the caller is the latest thinker.
    * Peer list is the union by ``id``. For an id present in both,
      ``mem_cfg`` wins (the caller is mutating that peer's fields).
      For a disk-only id (added by another worker since we loaded),
      the disk row is preserved. For a mem-only id (we just added
      it), it's appended.
    * ``removed_ids`` is the explicit channel for deletions — those
      ids are filtered out of the merged roster regardless of which
      side they appear on. ``remove_peer`` uses this so a removal
      can't be "undone" by a disk-only entry.
    * Order: disk order first (stable for operators reading the file),
      mem-only peers appended at the end.
    """
    if not isinstance(disk_cfg, dict):
        disk_cfg = {}
    removed = set(removed_ids or ())
    out = dict(mem_cfg)
    mem_peers = list(mem_cfg.get('peers') or [])
    disk_peers = list(disk_cfg.get('peers') or [])
    mem_by_id = {p.get('id'): p for p in mem_peers if isinstance(p, dict) and p.get('id')}
    disk_by_id = {p.get('id'): p for p in disk_peers if isinstance(p, dict) and p.get('id')}
    seen: set = set()
    merged: list = []
    for pid, dp in disk_by_id.items():
        if pid in removed:
            seen.add(pid)
            continue
        if pid in mem_by_id:
            merged.append(mem_by_id[pid])
        else:
            merged.append(dp)
        seen.add(pid)
    for mp in mem_peers:
        pid = mp.get('id') if isinstance(mp, dict) else None
        if not pid or pid in seen or pid in removed:
            continue
        merged.append(mp)
        seen.add(pid)
    out['peers'] = merged
    return out


def save_peers_config(cfg: dict, removed_ids: set | None = None):
    """Atomically persist ``cfg`` to ``peers.json`` under fcntl lock.

    Concurrent writers (other gunicorn workers, director loop ticks,
    HA shadow snapshots) all funnel through the same lockfile, so a
    given save is serialised against every other save. Within the
    locked region we re-read the on-disk cfg and ``_merge_peers_with_disk``
    it with the caller's ``cfg`` so disk-only peers (added by some
    other worker since this caller loaded) are preserved.

    Deletions: pass ``removed_ids={'atNN', ...}`` — the merge will
    filter those ids out of the union regardless of which side they
    were on. A plain ``save_peers_config(cfg)`` will NOT delete a peer
    that's also on disk; this prevents stale snapshots from quietly
    dropping peers.
    """
    PEERS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    import fcntl as _fcntl
    import tempfile as _tempfile
    # Open lockfile (create if missing). Keep open for the whole
    # critical section; flock is per-fd and released on close.
    with open(PEERS_CONFIG_LOCK, 'a+') as _lf:
        try:
            _fcntl.flock(_lf.fileno(), _fcntl.LOCK_EX)
        except OSError:
            # Filesystem doesn't support flock — fall back to a
            # best-effort unlocked write. Better than crashing.
            disk = None
            try:
                if PEERS_CONFIG.exists():
                    disk = json.loads(PEERS_CONFIG.read_text())
            except Exception:
                disk = None
            merged = _merge_peers_with_disk(cfg, disk, removed_ids)
            PEERS_CONFIG.write_text(json.dumps(merged, indent=2, default=str))
            return
        try:
            disk = None
            try:
                if PEERS_CONFIG.exists():
                    disk = json.loads(PEERS_CONFIG.read_text())
            except Exception:
                disk = None
            merged = _merge_peers_with_disk(cfg, disk, removed_ids)
            fd, tmp = _tempfile.mkstemp(
                dir=str(PEERS_CONFIG.parent), suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(merged, f, indent=2, default=str)
                os.replace(tmp, PEERS_CONFIG)
            except BaseException:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
                raise
        finally:
            try:
                _fcntl.flock(_lf.fileno(), _fcntl.LOCK_UN)
            except Exception:
                pass


def load_director_state() -> dict:
    """Load persisted director state."""
    if DIRECTOR_STATE.exists():
        try:
            return json.loads(DIRECTOR_STATE.read_text())
        except Exception:
            pass
    return {
        'active_peer': None,
        'last_switch': None,
        'peer_bandwidth': {},  # peer_id -> {used_bytes, cycle_start, last_check}
        'mode': 'auto',  # auto | manual | paused
    }


def save_director_state(state: dict):
    DIRECTOR_STATE.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=DIRECTOR_STATE.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, DIRECTOR_STATE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _peer_renew_day(peer: dict, cfg: dict) -> int:
    """Effective renew day for a peer.

    Override > day-of-month of first_seen > global cfg renew_day.
    Day is clamped to 1..28 so it always resolves on every month.
    """
    rd = peer.get('renew_day')
    if not rd:
        fs = peer.get('first_seen')
        if isinstance(fs, str) and len(fs) >= 10:
            try:
                rd = int(fs[8:10])
            except (ValueError, TypeError):
                rd = None
    if not rd:
        rd = cfg.get('renew_day', BANDWIDTH_RENEW_DAY)
    try:
        rd = max(1, min(28, int(rd)))
    except (TypeError, ValueError):
        rd = BANDWIDTH_RENEW_DAY
    return rd


def _peer_next_renew(peer: dict, cfg: dict,
                    *, now: datetime | None = None) -> datetime:
    """Return the next billing-cycle renewal datetime for a peer (UTC).

    Uses the peer's effective renew_day. If today's renew day hasn't
    passed yet this month, returns this month's anchor at 00:00 UTC;
    otherwise next month's. Day is clamped to 1..28.
    """
    rd = _peer_renew_day(peer, cfg)
    n = now or datetime.now(timezone.utc)
    candidate = n.replace(day=rd, hour=0, minute=0, second=0, microsecond=0)
    if candidate <= n:
        m = n.month + 1
        y = n.year + (1 if m > 12 else 0)
        m = ((m - 1) % 12) + 1
        candidate = candidate.replace(year=y, month=m)
    return candidate


# Bandwidth thresholds for the active frontier peer.
#
# When the active peer is below LOW_WATER_GRACEFUL_GB remaining and it
# is currently mid-KG, we send a *graceful* stop (finish-then-exit) so
# the in-flight KG gets uploaded instead of being killed and retried.
# When it drops below HARD_DEPLETED_GB we treat it as depleted and
# rotate even mid-KG (credential safety > 1 KG of work). Park the peer
# until its next renewal so the director's normal scheduler skips it
# without us having to track per-peer budget state separately.
BANDWIDTH_LOW_WATER_GB = 4.0
BANDWIDTH_HARD_DEPLETED_GB = 1.0


def get_billing_cycle_start() -> datetime:
    """Return the start of the current billing cycle (17th of this/previous month)."""
    now = datetime.now(timezone.utc)
    day = BANDWIDTH_RENEW_DAY
    if now.day >= day:
        return now.replace(day=day, hour=0, minute=0, second=0, microsecond=0)
    # Before the 17th — cycle started last month
    if now.month == 1:
        return now.replace(year=now.year - 1, month=12, day=day,
                           hour=0, minute=0, second=0, microsecond=0)
    return now.replace(month=now.month - 1, day=day,
                       hour=0, minute=0, second=0, microsecond=0)


def get_local_bandwidth() -> dict:
    """Get bandwidth usage from local vnstat."""
    try:
        result = subprocess.run(
            ['vnstat', '--json', 'm'],
            capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout)
        iface = data['interfaces'][0]
        now = datetime.now(timezone.utc)
        cycle_start = get_billing_cycle_start()

        # Sum traffic from cycle_start to now
        total_rx = 0
        total_tx = 0
        for m in iface.get('traffic', {}).get('month', []):
            dt = m['date']
            month_date = datetime(dt['year'], dt['month'], 1, tzinfo=timezone.utc)
            # Include months that overlap with the billing cycle
            if month_date.year == cycle_start.year and month_date.month == cycle_start.month:
                total_rx += m['rx']
                total_tx += m['tx']
            elif month_date.year == now.year and month_date.month == now.month:
                if now.month != cycle_start.month:  # avoid double-counting
                    total_rx += m['rx']
                    total_tx += m['tx']

        total = total_rx + total_tx
        return {
            'used_bytes': total,
            'used_gb': round(total / (1024 ** 3), 2),
            'rx_bytes': total_rx,
            'tx_bytes': total_tx,
            'budget_gb': BANDWIDTH_BUDGET_GB,
            'remaining_gb': round(max(0, BANDWIDTH_BUDGET_BYTES - total) / (1024 ** 3), 2),
            'pct_used': round(100 * total / BANDWIDTH_BUDGET_BYTES, 1),
            'cycle_start': cycle_start.isoformat(),
            'checked_at': now.isoformat(),
        }
    except Exception as e:
        log.warning('Failed to read local bandwidth: %s', e)
        return {'error': str(e), 'used_bytes': 0, 'used_gb': 0,
                'remaining_gb': BANDWIDTH_BUDGET_GB, 'pct_used': 0}


def get_peer_bandwidth(peer_url: str, peer_id: str = '') -> dict:
    """Get bandwidth from a remote peer.  Short timeout — must not block the loop."""
    pid = peer_id or _peer_id_for_url(peer_url)
    # Prefer pushed bandwidth if fresh — saves an HTTP round-trip.
    pushed = get_pushed_bandwidth(pid)
    if pushed is not None:
        out = dict(pushed)
        out['_pushed'] = True
        return out
    try:
        r = _peer_request(
            'GET',
            peer_url.rstrip('/') + '/api/v1/bandwidth',
            peer_id=pid,
            timeout=PEER_TIMEOUT_PROBE,
        )
        if r is None:
            return {'error': 'circuit_open', 'used_bytes': 0, 'used_gb': 0,
                    'remaining_gb': BANDWIDTH_BUDGET_GB, 'pct_used': 0,
                    'estimated': True, '_circuit_open': True}
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # Don't log here — the caller (_update_bandwidth) decides whether to
        # warn based on the consecutive-failure count.  Avoids log spam when
        # a peer is wedged.
        return {'error': str(e), 'used_bytes': 0, 'used_gb': 0,
                'remaining_gb': BANDWIDTH_BUDGET_GB, 'pct_used': 0,
                'estimated': True}


# Last-known peer status cache. Busy peers (e.g. running heavy GPKG
# builds) sometimes take >8s to respond on /processing/status, which
# blows past PEER_TIMEOUT_PROBE and makes the dashboard report them as
# 'unreachable, completed=0' even though they're actively working. We
# stash the last successful response per peer_url and surface it (with
# state='busy') when a fresh poll times out, as long as it's recent.
_PEER_STATUS_CACHE: dict[str, tuple[float, dict]] = {}
_PEER_STATUS_CACHE_TTL = 300.0   # 5 min — covers a slow GPKG step.
# Status-push cache: peers POST /api/v1/director/peer_status every
# PEER_PUSH_INTERVAL seconds. The director loop prefers this over
# pulling /processing/status; polls only when push is stale or
# unavailable. Slashes outbound director traffic ~50x at fleet scale.
PEER_PUSH_INTERVAL = 30          # seconds (peers run a ticker)
PEER_PUSH_FRESH_S = 75           # consider push fresh for this long
_PEER_PUSH: dict[str, dict] = {}  # peer_id -> {ts, status, bandwidth}
_PEER_PUSH_LOCK = threading.Lock()


# Status-fields that are bandwidth-shipped only once per process
# lifetime (constants until the peer restarts). The director merges
# them from the previous push so heartbeats don't lose them.
_STICKY_SYSTEM_FIELDS = ('host',)


def record_peer_push(peer_id: str, status: dict,
                     bandwidth: dict | None = None) -> None:
    """Record a peer-pushed status payload. Called from the HTTP handler.

    Merges 'sticky' fields (currently ``system.host``) from the
    previous push so idle heartbeats — which intentionally omit
    constants to save bandwidth — don't erase them. Without this the
    fleet-profile view would lose cpu_model / sys_vendor as soon as a
    peer went idle.
    """
    if not peer_id:
        return
    status = status or {}
    with _PEER_PUSH_LOCK:
        prev = _PEER_PUSH.get(peer_id) or {}
        prev_status = prev.get('status') or {}
        prev_sys = prev_status.get('system') or {}
        new_sys = status.get('system') or {}
        # Pull sticky fields forward when the new payload lacks them.
        merged_sys = dict(new_sys)
        for k in _STICKY_SYSTEM_FIELDS:
            if k not in merged_sys and k in prev_sys:
                merged_sys[k] = prev_sys[k]
        if merged_sys:
            status = dict(status)
            status['system'] = merged_sys
        _PEER_PUSH[peer_id] = {
            'ts': time.time(),
            'status': status,
            'bandwidth': bandwidth,
        }


def get_pushed_status(peer_id: str) -> dict | None:
    """Return fresh pushed status for a peer, or None if stale/missing."""
    if not peer_id:
        return None
    with _PEER_PUSH_LOCK:
        ent = _PEER_PUSH.get(peer_id)
        if not ent:
            return None
        if (time.time() - ent['ts']) > PEER_PUSH_FRESH_S:
            return None
        return ent


def get_pushed_bandwidth(peer_id: str) -> dict | None:
    ent = get_pushed_status(peer_id)
    if ent is None:
        return None
    return ent.get('bandwidth')


def get_peer_status(peer_url: str | None, peer_id: str = '') -> dict:
    """Get processing status from a peer. None = local."""
    if peer_url is None:
        # Local — read progress.json directly
        pf = DATA_DIR / 'progress.json'
        if pf.exists():
            try:
                d = json.loads(pf.read_text())
                # Check if processor is actually alive
                try:
                    subprocess.check_output(
                        ['pgrep', '-f', 'austria_processor.py'],
                        text=True, timeout=2
                    )
                except Exception:
                    if d.get('state') in ('running', 'processing'):
                        d['state'] = 'stopped'
                # Always reflect the live srv's commit, not whatever
                # value an old processor left in progress.json.
                d['git_commit'] = _LOCAL_GIT_COMMIT
                return d
            except Exception:
                pass
        return {'state': 'idle', 'git_commit': _LOCAL_GIT_COMMIT}
    pid = peer_id or _peer_id_for_url(peer_url)
    # Prefer pushed status if fresh — avoids polling the peer entirely.
    pushed = get_pushed_status(pid)
    if pushed is not None:
        age = time.time() - pushed['ts']
        # Short-circuit: a push <60 s old is the fastest possible
        # answer and there's no value in re-deriving it. Skip even
        # the pull-cache write so concurrent callers from get_status()
        # / capacity-tick fanouts don't all serialise on the dict
        # under GIL contention. Anything older than 60 s falls
        # through to the slower path that also refreshes the pull
        # cache as a fallback for circuit-open scenarios.
        if age < 60.0:
            d = dict(pushed.get('status') or {})
            d['_pushed'] = True
            d['_push_age_s'] = round(age, 1)
            d['_push_fresh'] = True
            return d
        d = dict(pushed.get('status') or {})
        d['_pushed'] = True
        d['_push_age_s'] = round(age, 1)
        # Still cache for fallback / proxy reads.
        _PEER_STATUS_CACHE[peer_url] = (time.time(), d)
        return d
    try:
        r = _peer_request(
            'GET',
            peer_url.rstrip('/') + '/api/v1/processing/status',
            peer_id=pid,
            timeout=PEER_TIMEOUT_PROBE,
        )
        if r is None:
            # Circuit open — use cached value if available.
            cached = _PEER_STATUS_CACHE.get(peer_url)
            if cached and (time.time() - cached[0]) < _PEER_STATUS_CACHE_TTL:
                d = dict(cached[1])
                d['_stale'] = True
                d['_circuit_open'] = True
                d['_stale_age_s'] = round(time.time() - cached[0], 1)
                return d
            return {'state': 'unreachable', 'error': 'circuit_open',
                    '_circuit_open': True}
        r.raise_for_status()
        d = r.json()
        _PEER_STATUS_CACHE[peer_url] = (time.time(), d)
        return d
    except Exception as e:
        # Fall back to last-known good status if recent enough — a slow
        # peer that's actually doing work shouldn't show as unreachable.
        cached = _PEER_STATUS_CACHE.get(peer_url)
        if cached and (time.time() - cached[0]) < _PEER_STATUS_CACHE_TTL:
            d = dict(cached[1])
            d['_stale'] = True
            d['_stale_age_s'] = round(time.time() - cached[0], 1)
            d['_stale_error'] = str(e)[:120]
            # Don't override the original 'state' — the peer is most
            # likely still doing whatever it was doing. The dashboard
            # surfaces _stale so it's clear data is cached.
            return d
        return {'state': 'unreachable', 'error': str(e)}


def get_peer_log(peer_url: str | None, lines: int = 50) -> list[str]:
    """Get recent log lines from a peer."""
    if peer_url is None:
        log_file = DATA_DIR / 'logs' / 'processor.log'
        if log_file.exists():
            try:
                result = subprocess.run(
                    ['tail', f'-{lines}', str(log_file)],
                    capture_output=True, text=True, timeout=5
                )
                return result.stdout.splitlines()
            except Exception:
                pass
        return []
    try:
        r = requests.get(
            peer_url.rstrip('/') + '/api/v1/processing/log',
            params={'lines': lines},
            timeout=PEER_TIMEOUT_PROBE,
        headers=_admin_headers())
        r.raise_for_status()
        d = r.json()
        return d.get('lines', d.get('log', []))
    except Exception:
        return []


# Per-peer cache-manifest sync backoff state. Module-global so we
# don't have to thread the director instance through every call site.
# Maps peer_url -> {fails: int, suppress_until: float epoch}.
_SYNC_BACKOFF: dict[str, dict] = {}


def _sync_kg_strikes_to_peer(peer_url: str) -> None:
    """Push local kg_strikes.json (adaptive-split counters) to a peer.

    Best-effort, max(local, remote) per KG. Without it, peers would only
    see strikes they observed themselves, missing strikes accumulated on
    other peers (e.g. the same KG bounced between three frontiers).
    """
    p = DATA_DIR / 'kg_strikes.json'
    if not p.exists():
        return
    try:
        local = json.loads(p.read_text())
        if not local:
            return
        r = requests.put(
            peer_url.rstrip('/') + '/api/v1/processing/kg_strikes',
            json=local, timeout=PEER_TIMEOUT_CONTROL,
            headers=_admin_headers())
        if r.ok:
            log.info('KG strikes sync to %s: %d updated', peer_url,
                     r.json().get('updated', 0))
        else:
            log.debug('KG strikes sync to %s: HTTP %d', peer_url, r.status_code)
    except Exception as e:
        log.debug('KG strikes sync to %s: %s', peer_url, e)


def _sync_cache_manifest_to_peer(peer_url: str) -> None:
    """Push local Zenodo tile-cache manifest to a remote peer.

    Ensures the peer shares the same Zenodo cache deposit, so it can
    read cached Copernicus/Hansen tiles instead of re-fetching them.

    Backs off after ``SYNC_BACKOFF_THRESHOLD`` consecutive failures so a
    peer with a flaky network doesn't burn the budget on every tick.
    Honours ``Retry-After`` if the peer responds with one.
    """
    manifest_path = DATA_DIR / 'cache_manifest.json'
    if not manifest_path.exists():
        return
    bo = _SYNC_BACKOFF.get(peer_url) or {}
    if bo.get('suppress_until', 0) > time.time():
        log.debug('Cache manifest sync to %s suppressed (%.0fs left)',
                  peer_url, bo['suppress_until'] - time.time())
        return
    try:
        local_cm = json.loads(manifest_path.read_text())
        if not local_cm.get('depo_id'):
            return
        r = requests.put(
            peer_url.rstrip('/') + '/api/v1/processing/cache_manifest',
            json=local_cm, timeout=PEER_TIMEOUT_CONTROL,
        headers=_admin_headers())
        if r.ok:
            result = r.json()
            log.info('Cache manifest sync to %s: %d entries updated',
                     peer_url, result.get('updated', 0))
            _SYNC_BACKOFF[peer_url] = {'fails': 0, 'suppress_until': 0}
        else:
            # Honour Retry-After if present (Zenodo proxy etc).
            ra = r.headers.get('Retry-After')
            ra_secs = 0.0
            if ra:
                try:
                    ra_secs = float(ra)
                except ValueError:
                    pass
            fails = int(bo.get('fails', 0)) + 1
            entry = {'fails': fails, 'suppress_until': 0.0}
            if fails >= SYNC_BACKOFF_THRESHOLD or ra_secs > 0:
                entry['suppress_until'] = (
                    time.time() + max(SYNC_BACKOFF_SECONDS, ra_secs))
                log.warning(
                    'Cache manifest sync to %s: HTTP %d (fail %d) — '
                    'suppressing for %.0fs%s',
                    peer_url, r.status_code, fails,
                    entry['suppress_until'] - time.time(),
                    f' (Retry-After {ra_secs:.0f}s)' if ra_secs else '',
                )
            else:
                log.warning('Cache manifest sync to %s: HTTP %d (fail %d)',
                            peer_url, r.status_code, fails)
            _SYNC_BACKOFF[peer_url] = entry
    except Exception as e:
        fails = int(bo.get('fails', 0)) + 1
        entry = {'fails': fails, 'suppress_until': 0.0}
        if fails >= SYNC_BACKOFF_THRESHOLD:
            entry['suppress_until'] = time.time() + SYNC_BACKOFF_SECONDS
            log.warning(
                'Cache manifest sync to %s failed (%s, fail %d) — '
                'suppressing for %ds',
                peer_url, e, fails, SYNC_BACKOFF_SECONDS)
        else:
            log.warning('Cache manifest sync to %s failed: %s', peer_url, e)
        _SYNC_BACKOFF[peer_url] = entry


def sync_queue_to_peer(peer_url: str, exclude: set | None = None) -> dict:
    """Push the local priority queue to a remote peer.

    Reads the local retry_queue.json and POSTs it to the peer's
    /api/v1/processing/queue endpoint (position=0 = front of queue).
    Skips if queue is empty.

    `exclude` is a set of KG codes that must NOT be sent to this peer
    (typically KGs reserved for cooled-down peers so they can resume
    them from tile checkpoints when their cooldown lifts).
    """
    queue_path = DATA_DIR / 'retry_queue.json'
    try:
        queue = json.loads(queue_path.read_text()) if queue_path.exists() else []
    except Exception:
        queue = []
    if exclude:
        queue = [c for c in queue if c not in exclude]
    if not queue:
        return {'status': 'empty_queue', 'synced': 0}

    # Identify tombstoned codes (force-requeued KGs that are locally
    # "completed" but the operator wants re-processed). The peer's
    # /processing/queue endpoint filters out already-completed KGs when
    # skip_processed=True — so tombstoned codes would be silently dropped.
    # Push them in a second request with skip_processed=False so the peer
    # accepts them at the front of its queue.
    tombstoned = set()
    try:
        import re as _re
        tpath = DATA_DIR / 'manifest_tombstones.json'
        if tpath.exists():
            tdata = json.loads(tpath.read_text())
            if isinstance(tdata, dict):
                for key in tdata:
                    m = _re.match(r'^(\d+(?:-[a-z][-a-z0-9]*)?)_', key)
                    if m:
                        tombstoned.add(m.group(1))
    except Exception:
        tombstoned = set()

    tombstoned_in_queue = [c for c in queue if c in tombstoned]
    normal_queue = [c for c in queue if c not in tombstoned]

    try:
        # Push normal codes first (they go to position 0).
        result = {'status': 'empty_queue'}
        if normal_queue:
            r = requests.post(
                peer_url.rstrip('/') + '/api/v1/processing/queue',
                json={'kgs': normal_queue, 'position': 0, 'skip_processed': True},
                timeout=PEER_TIMEOUT_CONTROL,
            headers=_admin_headers())
            result = r.json()
        # Then push tombstoned codes at position 0 — ends up in front,
        # ahead of the normal codes, preserving the original ordering.
        if tombstoned_in_queue:
            r2 = requests.post(
                peer_url.rstrip('/') + '/api/v1/processing/queue',
                json={'kgs': tombstoned_in_queue, 'position': 0,
                      'skip_processed': False},
                timeout=PEER_TIMEOUT_CONTROL,
            headers=_admin_headers())
            result = r2.json()
        log.info('Queue sync to %s: pushed %d KGs (%d tombstoned) — %s%s',
                 peer_url, len(queue), len(tombstoned_in_queue),
                 result.get('status', '?'),
                 (' (excluded ' + ','.join(sorted(exclude)) + ')') if exclude else '')
        return result
    except Exception as e:
        log.warning('Queue sync to %s failed: %s', peer_url, e)
        return {'error': str(e)}


def _push_queue_to_peer(peer_url: str, codes: list) -> dict:
    """Push an explicit list of KG codes as the peer's priority queue.

    Used when sending a *cache-only* peer a whitelist of fully-cached KGs.
    Replaces the peer's priority queue with the given codes.
    """
    if not codes:
        return {'status': 'empty_whitelist', 'synced': 0}
    try:
        # Replace the queue (PUT) so previous frontier work doesn't leak in.
        r = requests.put(
            peer_url.rstrip('/') + '/api/v1/processing/queue',
            json={'queue': codes},
            timeout=PEER_TIMEOUT_CONTROL,
        headers=_admin_headers())
        if r.ok:
            log.info('Whitelist queue PUT to %s: %d KGs', peer_url, len(codes))
            return r.json()
        # Fallback: POST at front
        r2 = requests.post(
            peer_url.rstrip('/') + '/api/v1/processing/queue',
            json={'kgs': codes, 'position': 0, 'skip_processed': True},
            timeout=PEER_TIMEOUT_CONTROL,
        headers=_admin_headers())
        log.info('Whitelist queue POST to %s: %d KGs (PUT was %d)',
                 peer_url, len(codes), r.status_code)
        return r2.json() if r2.ok else {'error': f'http {r2.status_code}'}
    except Exception as e:
        log.warning('Whitelist queue push to %s failed: %s', peer_url, e)
        return {'error': str(e)}


def _in_progress_kgs(cfg: dict, exclude_peer_id: str | None = None) -> set:
    """Return the set of KG codes currently being processed by *other* peers.

    Probes each peer's `/api/v1/processing/status` and pulls `current_kg.code`.
    Used to exclude an active KG from the priority queue we push to a peer
    we're about to (re)start, so two peers don't race the same KG.

    Without this, the retry-queue sync happily re-issues a KG to a fresh
    frontier even though another peer is mid-tile on it. Tile checkpoints
    are per-peer, so the duplicate work is wasted.
    """
    import re as _re
    from concurrent.futures import ThreadPoolExecutor, as_completed
    targets = [p for p in cfg.get('peers', [])
               if p.get('url') and (not exclude_peer_id
                                    or p.get('id') != exclude_peer_id)]
    out: set = set()
    if not targets:
        return out
    with ThreadPoolExecutor(max_workers=min(16, len(targets))) as pool:
        futs = {pool.submit(get_peer_status, p['url']): p for p in targets}
        # PEER_TIMEOUT_PROBE is a (connect, read) tuple; cap the wall-clock
        # wait at read*2 + a small buffer so a single hung peer doesn't
        # stall the director loop.
        _read_to = (PEER_TIMEOUT_PROBE[1]
                    if isinstance(PEER_TIMEOUT_PROBE, tuple)
                    else PEER_TIMEOUT_PROBE)
        for f in as_completed(futs, timeout=_read_to * 2 + 2):
            try:
                ps = f.result()
            except Exception:
                continue
            if ps.get('state') not in ('running', 'processing'):
                continue
            ck = ps.get('current_kg') or {}
            code = ck.get('code') if isinstance(ck, dict) else None
            if code:
                out.add(str(code))
                # Also block the parent KG code so a sibling block of an
                # already-being-split KG isn't issued to another peer.
                m = _re.match(r'^(\d+)(?:-[a-z][-a-z0-9]*)?$', str(code))
                if m:
                    out.add(m.group(1))
    return out


def _excluded_kgs(cfg: dict, exclude_peer_id: str | None = None) -> set:
    """Union of reservations and in-progress KGs across other peers.

    The single source of truth for "don't issue these KGs to *exclude_peer_id*".
    Combines:
      * `_reserved_kgs` — KGs held for cooled-down peers (resume from
        tile checkpoints once their not_before expires).
      * `_in_progress_kgs` — KGs another peer is currently processing.
    """
    return (_reserved_kgs(cfg, exclude_peer_id=exclude_peer_id)
            | _in_progress_kgs(cfg, exclude_peer_id=exclude_peer_id))


def _reserved_kgs(cfg: dict, exclude_peer_id: str | None = None) -> set:
    """Collect KGs reserved by other peers (not `exclude_peer_id`).

    Reservations persist past `not_before` so that the holding peer can
    actually pick the KG back up once cooldown lifts. Stale reservations
    are pruned separately by `_clear_completed_reservations` once the KG
    appears in the local `_get_completed_kgs()` set.

    Split-KG awareness: a reservation on a block code (``60336-northwest``)
    also blocks the parent code (``60336``), and a reservation on a
    parent blocks the parent itself (block siblings are still OK to run
    in parallel — adding parent → all-blocks would over-block legitimate
    parallel block work). The asymmetric expansion is correct because
    a peer that holds a *block* still leaves other blocks free for
    parallel processing.
    """
    import re as _re
    out = set()
    for p in cfg.get('peers', []):
        if exclude_peer_id and p.get('id') == exclude_peer_id:
            continue
        kg = p.get('reserved_kg')
        if not kg:
            continue
        s = str(kg)
        out.add(s)
        # Block parent code too: an old (pre-splitter) peer would
        # otherwise treat ``60336`` as free while another peer holds
        # ``60336-northwest`` mid-tile.
        m = _re.match(r'^(\d+)-[a-z][-a-z0-9]*$', s)
        if m:
            out.add(m.group(1))
    return out


def start_peer_processor(peer_url: str | None, exclude_kgs: set | None = None,
                         *, cache_only: bool = False,
                         queue_whitelist: list | None = None,
                         cred_indices: list | None = None,
                         lat_strips: list | None = None) -> dict:
    """Start the processor on a peer.

    For remote peers, syncs the local priority queue first so the peer
    processes the same KGs in priority order. `exclude_kgs` lets callers
    suppress KGs reserved for other (cooled-down) peers.

    If *cache_only* is True, the peer is started with the ``--cache-only``
    flag so it refuses any Copernicus/Hansen API call.  Use *queue_whitelist*
    to send only KGs known to be fully cached.
    """
    payload = {}
    if cache_only:
        payload['cache_only'] = True
    if cred_indices is not None:
        payload['cred_indices'] = list(cred_indices)
    if lat_strips is not None:
        payload['lat_strips'] = list(lat_strips)
    if peer_url is None:
        # Local start — use the local API so _processor_process is tracked
        try:
            r = requests.post('http://127.0.0.1:8000/api/v1/processing/start',
                              json=payload, timeout=PEER_TIMEOUT_CONTROL, headers=_admin_headers())
            return r.json() if r.ok else {'error': f'local start: {r.status_code}'}
        except Exception as e:
            return {'error': str(e)}
    # Remote peer — sync cache manifest + KG strikes + priority queue before starting
    _sync_cache_manifest_to_peer(peer_url)
    _sync_kg_strikes_to_peer(peer_url)
    if queue_whitelist is not None:
        queue_result = _push_queue_to_peer(peer_url, list(queue_whitelist))
    else:
        queue_result = sync_queue_to_peer(peer_url, exclude=exclude_kgs)
    # Try API start; if it transiently 500s (often due to a race in
    # processor state — e.g. an externally-detected processor that
    # didn't fully exit), retry once after a short sleep.
    last_err: str | None = None
    last_status: int | None = None
    for attempt in (1, 2):
        try:
            r = requests.post(
                peer_url.rstrip('/') + '/api/v1/processing/start',
                json=payload,
                timeout=PEER_TIMEOUT_CONTROL,
            headers=_admin_headers())
            if r.ok:
                result = r.json()
                result['queue_sync'] = queue_result
                if attempt > 1:
                    result['retry'] = attempt
                return result
            if r.status_code == 409:
                # Already running. If body says "external", the previous
                # subprocess hasn't fully exited yet — surface as a
                # transient error so the caller retries on the next tick
                # (don't drop the error key, callers gate on it).
                try:
                    body = r.json()
                except Exception:
                    body = {'error': r.text[:200]}
                err = (body.get('error') or '').lower()
                if 'external' in err:
                    last_err = body.get('error') or 'external processor'
                    last_status = 409
                    if attempt < 2:
                        time.sleep(3.0)
                        continue
                    return {'error': f'api_start_failed: {last_err}',
                            'queue_sync': queue_result,
                            'method': 'no_fallback_constrained'}
                # Tracked-state 409 (we already started it) — fine.
                body['queue_sync'] = queue_result
                body['already_running'] = True
                body.pop('error', None)
                return body
            last_status = r.status_code
            last_err = (r.text or '')[:200]
        except Exception as e:
            last_err = str(e)
        if attempt < 2:
            time.sleep(1.5)
    # The systemd fallback restarts ``austria_processor.service`` which
    # (a) is disabled on peers and (b) has no awareness of cache_only,
    # cred_indices, lat_strips, or queue_whitelist. Falling back here
    # silently bypasses the director's per-peer assignments — e.g. a
    # cache-only start turns into a full frontier start, then the
    # director kills it as 'non-active running frontier' and retries
    # forever. Only fall back when the call had no per-peer contract.
    is_constrained = bool(cache_only or cred_indices or lat_strips
                            or queue_whitelist is not None)
    if is_constrained:
        log.warning(
            'API start on %s failed (status=%s err=%s); skipping systemd '
            'fallback because contract requires cache_only=%s '
            'cred_indices=%s lat_strips=%s queue_whitelist=%s',
            peer_url, last_status, last_err,
            cache_only, cred_indices, lat_strips,
            None if queue_whitelist is None else len(queue_whitelist))
        return {'error': f'api_start_failed: {last_status or last_err}',
                'queue_sync': queue_result,
                'method': 'no_fallback_constrained'}
    log.warning('API start on %s returned %s, trying systemd fallback',
                peer_url, last_status if last_status else last_err)
    # Fallback: ask the peer to restart the processor via systemd
    try:
        r2 = requests.post(
            peer_url.rstrip('/') + '/api/v1/admin/restart_processor',
            json={},
            timeout=30,
        headers=_admin_headers())
        result = r2.json() if r2.ok else {'error': f'systemd fallback: {r2.status_code}'}
        result['queue_sync'] = queue_result
        result['method'] = 'systemd_fallback'
        return result
    except Exception as e2:
        return {'error': str(e2), 'queue_sync': queue_result, 'method': 'both_failed'}


def stop_peer_processor(peer_url: str | None, graceful: bool = False) -> dict:
    """Stop the processor on a peer.

    If ``graceful`` is True, asks the peer to exit cleanly after the
    current KG instead of SIGTERM'ing mid-KG. Used for cache-only peers
    where mid-KG kills waste pure CPU work without any credential-safety
    benefit.
    """
    url = peer_url if peer_url else 'http://127.0.0.1:8000'
    try:
        params = {'graceful': '1'} if graceful else None
        r = requests.post(
            url.rstrip('/') + '/api/v1/processing/stop',
            params=params,
            timeout=30,  # stop can take a moment
            headers=_admin_headers(),
        )
        if r.ok:
            return r.json()
        # If 404 (not running), that's fine
        if r.status_code == 404:
            return {'status': 'already_stopped'}
        return {'error': f'stop returned {r.status_code}'}
    except Exception as e:
        return {'error': str(e)}


def safely_stop_peer(peer_url: str | None, peer_id: str = '?',
                     max_wait: int = 60) -> dict:
    """Stop a peer's processor and verify it actually stopped.

    CRITICAL for credential safety: returns success only after we've
    confirmed via /processing/status that the peer is no longer running.
    On failure, the caller MUST NOT start another peer (would cause
    parallel processing and 402 rate-limit errors on shared credentials).

    Returns: {'status': 'stopped'}, {'status': 'already_stopped'},
             or {'error': '...', 'last_state': '...'}.
    """
    # 1. Send stop request
    stop_result = stop_peer_processor(peer_url)
    if 'error' in stop_result and stop_result.get('status') != 'already_stopped':
        log.warning('Peer %s stop API failed: %s', peer_id, stop_result['error'])
        # Don't give up yet — status check below may show it's actually stopped

    # 2. Poll status until processor is stopped (or timeout)
    deadline = time.time() + max_wait
    last_state = 'unknown'
    while time.time() < deadline:
        status = get_peer_status(peer_url)
        last_state = status.get('state', 'unknown')
        if last_state in ('idle', 'stopped'):
            return {'status': 'stopped', 'verified': True}
        if last_state == 'unreachable':
            # Peer's gunicorn is down/overloaded; we can't verify. Wait a bit.
            time.sleep(3)
            continue
        # Still running — give it time, retry stop after a few seconds
        time.sleep(2)
        if time.time() < deadline - 5:
            stop_peer_processor(peer_url)  # retry stop in case first one was lost
            time.sleep(3)

    # Timeout — could not verify peer is stopped
    log.error('Peer %s did NOT stop within %ds (last state: %s) — '
              'refusing to activate another peer (credential safety)',
              peer_id, max_wait, last_state)
    return {'error': f'stop_not_verified after {max_wait}s',
            'last_state': last_state}


def install_token_on_peer(peer_url: str, new_token: str) -> dict:
    """Push the cluster admin token to a peer.

    Used by /api/v1/director/update_peers to keep the cluster secret in
    sync without manual scp. The peer's /api/v1/admin/install_token
    endpoint accepts the new token if (a) the peer has none yet
    (bootstrap), or (b) the request presents the peer's current token
    via X-Admin-Token. We send our local token in the header, which
    matches case (b) when peers are already in sync and is harmless
    (ignored) under case (a).

    Peers running pre-auth code return 404 — reported as 'not_installed';
    after they pull the new code their next /admin/update will pick up
    the seeded token from data/admin_token.
    """
    if not new_token:
        return {'status': 'skipped_empty_token'}
    try:
        r = requests.post(
            peer_url.rstrip('/') + '/api/v1/admin/install_token',
            json={'new_token': new_token, 'current_token': new_token},
            headers=_admin_headers(),
            timeout=PEER_TIMEOUT_CONTROL,
        )
        if r.status_code == 404:
            return {'status': 'endpoint_missing',
                    'note': 'peer running pre-auth code; will be ok after update'}
        if r.ok:
            return {'status': 'installed'}
        return {'status': f'http_{r.status_code}', 'body': r.text[:200]}
    except Exception as e:
        return {'error': str(e)}


def trigger_peer_update(peer_url: str, graceful: bool = False) -> dict:
    """Tell a remote peer to git pull and restart its web server.
    The peer kills itself on restart so the connection always drops — treat
    any ConnectionError/ReadTimeout after the request was sent as success.

    If ``graceful`` is True, the peer will defer git-pull + restart until
    the current KG finishes (no mid-KG kills).
    """
    try:
        r = requests.post(
            peer_url.rstrip('/') + '/api/v1/admin/update',
            params={'graceful': '1'} if graceful else None,
            timeout=15,
        headers=_admin_headers())
        return r.json()
    except requests.exceptions.ConnectionError:
        # Expected: peer restarted and dropped the connection
        return {'status': 'restarting'}
    except requests.exceptions.ReadTimeout:
        # Expected: restart took longer than read timeout
        return {'status': 'restarting'}
    except Exception as e:
        return {'error': str(e)}


def _peer_is_scheduled(peer: dict) -> bool:
    """Check if a peer has a not_before date that hasn't passed yet."""
    not_before = peer.get('not_before')
    if not not_before:
        return False
    try:
        nb = datetime.fromisoformat(not_before)
        if nb.tzinfo is None:
            nb = nb.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) < nb
    except (ValueError, TypeError):
        return False


def _clear_completed_reservations(cfg: dict, state: dict | None = None) -> bool:
    """Drop reserved_kg for peers whose held KG is already completed,
    or whose holder is no longer eligible to claim the reservation.

    A reservation otherwise persists past `not_before` so the holding
    peer can pick its held KG back up after the cooldown lifts. But if
    the holder's `not_before` has elapsed AND the holder remains
    bandwidth-exhausted (cannot run the KG), the reservation deadlocks
    the KG — ``_reserved_kgs`` excludes other peers. Release it so a
    parallel frontier picks it up.
    Returns True if the config was modified.
    """
    completed = set()
    try:
        # Imported lazily — app.py imports peer_director at startup.
        from app import _get_completed_kgs
        completed = _get_completed_kgs()
    except Exception:
        pass
    bw_map = (state or {}).get('peer_bandwidth', {}) if state else {}
    changed = False
    for p in cfg.get('peers', []):
        kg = p.get('reserved_kg')
        if not kg:
            continue
        if str(kg) in completed:
            p.pop('reserved_kg', None)
            changed = True
            continue
        # Holder still in cooldown? keep reservation.
        if _peer_is_scheduled(p):
            continue
        # not_before elapsed (or absent). If holder is bandwidth-
        # exhausted, it can never claim — release so others can.
        if state is not None:
            bw = bw_map.get(p['id'], {})
            used = bw.get('used_bytes', 0)
            if (_peer_budget_bytes(p, cfg) - used) < 2 * (1024 ** 3):
                log.warning('Releasing held KG %s from %s '
                            '(cooldown elapsed but bandwidth exhausted)',
                            kg, p['id'])
                p.pop('reserved_kg', None)
                changed = True
    return changed


def _ready_reservation_holder(cfg: dict, state: dict | None = None) -> str | None:
    """Return peer_id of an enabled, non-scheduled peer that holds a
    reserved KG ready to be resumed. Returns None if no such peer.

    A holder with insufficient bandwidth (<2 GB remaining) cannot
    actually run the held KG, so it is ignored. Otherwise the
    director would pre-empt the active peer in a loop without ever
    being able to activate the holder (choose_active_peer also
    bandwidth-gates), starving parallel frontiers.
    """
    bw_map = (state or {}).get('peer_bandwidth', {}) if state else {}
    for p in cfg.get('peers', []):
        if not p.get('enabled', True):
            continue
        if not p.get('reserved_kg'):
            continue
        if _peer_is_scheduled(p):
            continue
        if state is not None:
            bw = bw_map.get(p['id'], {})
            used = bw.get('used_bytes', 0)
            if (_peer_budget_bytes(p, cfg) - used) < 2 * (1024 ** 3):
                continue
        return p['id']
    return None


def _peer_age_seconds(peer: dict) -> float:
    """Seconds since the peer was first seen by the director.

    Falls back to a large value if first_seen is missing (legacy entries)
    so the warmup hold doesn't block long-lived peers retroactively.
    """
    fs = peer.get('first_seen')
    if not fs:
        return float('inf')
    try:
        from datetime import datetime as _dt
        t = _dt.fromisoformat(fs)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds()
    except Exception:
        return float('inf')


def _peer_in_warmup(peer: dict) -> bool:
    """True if peer is too fresh to be promoted to frontier work.

    During warmup we still allow cache-only assignments (which are the
    gentlest workload) so the peer's tile cache fills naturally.
    """
    return _peer_age_seconds(peer) < WARMUP_HOLD_SECONDS


# --- KG complexity weights (used by complexity-weighted LPT partition
# in _orchestrate_cache_only). Cached at module level so we don't
# re-read kg_list.json + re-derive tile counts on every director tick.
_KG_WEIGHT_CACHE: dict[str, float] = {}
_KG_WEIGHT_CACHE_BUILT_AT: float = 0.0
_KG_WEIGHT_CACHE_TTL = 3600.0   # 1 h — KG geometry is static, this is
                                # just defensive against config edits.


def _kg_weights(codes: list) -> dict:
    """Return {kg_code: weight} for the requested codes.

    Weight ≈ expected processing cost relative to a tiny KG. Built
    from tile count (the dominant cost driver — each tile runs
    Felzenszwalb + RAG + RF inference) plus a small term for parcel
    count (vectorise + per-parcel features). Block codes inherit their
    parent's geometry; ``maybe_split_kg`` is invoked lazily so this
    works correctly for both parent and block codes.

    Unknown codes get weight=1.0 (one tile's worth) so the partition
    falls back gracefully if kg_list.json is missing entries.
    """
    global _KG_WEIGHT_CACHE_BUILT_AT
    now = time.time()
    if (now - _KG_WEIGHT_CACHE_BUILT_AT) > _KG_WEIGHT_CACHE_TTL:
        _KG_WEIGHT_CACHE.clear()
        _KG_WEIGHT_CACHE_BUILT_AT = now
    out: dict[str, float] = {}
    missing: list[str] = []
    for c in codes:
        w = _KG_WEIGHT_CACHE.get(c)
        if w is not None:
            out[c] = w
        else:
            missing.append(c)
    if not missing:
        return out
    # Lazy import — avoid pulling kg_splitter on module load.
    try:
        from kg_splitter import (maybe_split_kg, is_block_code,
                                   parent_kg_code, _compute_n_tiles)
    except Exception:
        # No splitter available — fall back to unit weights.
        for c in missing:
            _KG_WEIGHT_CACHE[c] = 1.0
            out[c] = 1.0
        return out
    kg_by_code = {}
    try:
        kg_list_path = DATA_DIR / 'kg_list.json'
        if kg_list_path.exists():
            for kg in json.loads(kg_list_path.read_text()):
                cc = kg.get('kg_code')
                if cc:
                    kg_by_code[cc] = kg
    except Exception:
        kg_by_code = {}
    for c in missing:
        try:
            parent = parent_kg_code(c) if is_block_code(c) else c
            kg = kg_by_code.get(parent)
            if not kg:
                _KG_WEIGHT_CACHE[c] = 1.0
                out[c] = 1.0
                continue
            if is_block_code(c):
                blocks = maybe_split_kg(kg)
                # Find this block by code; if mismatch, average the
                # parent uniformly.
                tgt = next((b for b in blocks
                            if b.get('kg_code') == c), None)
                if tgt and 'bbox' in tgt:
                    bb = tgt['bbox']
                    n_tiles = _compute_n_tiles(
                        bb['min_lon'], bb['min_lat'],
                        bb['max_lon'], bb['max_lat'])
                    # Per-parcel cost — parent's count divided
                    # across blocks (cheap & good enough).
                    parcels = kg.get('parcel_count', 0) / max(
                        len(blocks), 1)
                else:
                    bb = kg.get('bbox') or {}
                    n_tiles = _compute_n_tiles(
                        bb.get('min_lon', 0), bb.get('min_lat', 0),
                        bb.get('max_lon', 0), bb.get('max_lat', 0)
                    ) / max(len(blocks) if blocks else 1, 1)
                    parcels = kg.get('parcel_count', 0) / max(
                        len(blocks) if blocks else 1, 1)
            else:
                bb = kg.get('bbox') or {}
                n_tiles = _compute_n_tiles(
                    bb.get('min_lon', 0), bb.get('min_lat', 0),
                    bb.get('max_lon', 0), bb.get('max_lat', 0))
                parcels = kg.get('parcel_count', 0)
            # Weight: tile count dominates; tiny parcel-count nudge so
            # a 1-tile KG with 5000 parcels still ranks above a
            # 1-tile KG with 50.
            w = max(1.0, float(n_tiles) + 0.0005 * float(parcels))
        except Exception:
            w = 1.0
        _KG_WEIGHT_CACHE[c] = w
        out[c] = w
    return out


def _peer_cpu_steal(peer_id: str) -> float | None:
    """Return EWMA CPU steal %% for *peer_id*, or None if unknown.

    Reads the freshest value we have: peer-pushed status
    (``system.perf.cpu_steal_ewma`` > ``system.cpu_steal``). Returns
    None when the peer hasn't pushed telemetry yet (e.g. old build,
    first 30 s after start) so callers can treat it as "no signal"
    rather than "0 %% steal".
    """
    pushed = get_pushed_status(peer_id)
    if not pushed:
        return None
    sysd = pushed.get('status', {}).get('system') or {}
    perf = sysd.get('perf') or {}
    v = perf.get('cpu_steal_ewma')
    if v is None:
        v = sysd.get('cpu_steal')
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _peer_cpu_capacity(peer_id: str, default: float = 1.0) -> float:
    """Effective CPU capacity for *peer_id* in [0.10, 1.0].

    capacity = max(0.10, 1 - steal_ewma / 100). Floor at 0.10 so a
    pathological 95 %% steal peer still gets *some* work assigned (LPT
    becomes degenerate at exactly 0). Returns ``default`` when the
    peer hasn't pushed telemetry yet.
    """
    s = _peer_cpu_steal(peer_id)
    if s is None:
        return default
    return max(0.10, 1.0 - s / 100.0)


def _fleet_steal_median(peer_ids: list) -> float | None:
    """Median CPU steal across *peer_ids*; None when nobody reported."""
    vals = [v for v in (_peer_cpu_steal(pid) for pid in peer_ids)
            if v is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def _peer_noise_score(peer_id: str, state: dict) -> float:
    """Return a [0, +inf) noise score for a peer.

    Sums (rate_5m / saturation) across BEV / Zenodo / Copernicus from
    the peer's last reported ``warning_rates`` (cached on the director
    state under ``peer_warning_rates``). Higher = noisier upstreams
    *seen by this peer*. Used to bias scheduling toward quiet peers so
    fresh peers without warnings carry more load while peers that just
    upset Zenodo cool off naturally.

    Score 0.0 means "silent" — either truly quiet or no data yet (a
    fresh peer has no warning history, which is what we want: it's
    preferred for the next slot).
    """
    wr = ((state.get('peer_warning_rates') or {}).get(peer_id) or {})
    long_ema = ((state.get('peer_noise_long_ema') or {}).get(peer_id) or {})
    if not wr and not long_ema:
        return 0.0
    score = 0.0
    for kind, sat in THROTTLE_SATURATION_RATE.items():
        if sat <= 0:
            continue
        r_now = float(((wr.get(kind) or {}).get('5m')) or 0.0)
        r_long = float(long_ema.get(kind) or 0.0)
        # Take the max of "current pressure" and "recent-history pressure"
        # so a peer that just upset Zenodo five minutes ago still scores
        # noisy even though its 5-min rate already rolled back to zero.
        score += max(r_now, r_long) / sat
    return score


def choose_active_peer(cfg: dict, state: dict) -> str | None:
    """Pick the best peer to run the processor on.

    Reservation holders win first — if an enabled, non-scheduled peer
    still holds a reserved KG, give it priority so it can resume the
    held KG from its tile checkpoints. Otherwise pick the peer with the
    most remaining bandwidth.
    Returns None if all candidates have <2 GB remaining.
    """
    holder = _ready_reservation_holder(cfg, state)
    if holder:
        return holder

    # Score-based pick: prefer quiet peers (low warning fingerprint),
    # break ties by remaining bandwidth. Fresh peers have score 0 and
    # a full budget, so they naturally win until they earn warnings.
    candidates: list[tuple[float, int, str]] = []
    for peer in cfg.get('peers', []):
        if not peer.get('enabled', True):
            continue
        if _peer_is_scheduled(peer):
            continue
        pinned = (peer.get('pinned_role') or '').strip().lower()
        if pinned in ('idle', 'off', 'pause', 'paused', 'parked'):
            continue  # user-pinned idle
        if pinned in ('cache_only', 'cache-only', 'cacheonly'):
            continue  # user-pinned cache_only — not eligible as primary frontier
        if _peer_in_warmup(peer):
            continue  # fresh peer — hold off frontier promotion
        pid = peer['id']
        bw = state.get('peer_bandwidth', {}).get(pid, {})
        used = bw.get('used_bytes', 0)
        remaining = _peer_budget_bytes(peer, cfg) - used
        if remaining < 2 * (1024 ** 3):
            continue  # not enough headroom
        noise = _peer_noise_score(pid, state)
        # CPU-steal penalty: prefer peers with real CPU. A frontier
        # peer burns scarce Copernicus credentials, so running it on a
        # hypervisor-starved box (60-80%% steal is common on exe.dev's
        # over-subscribed pools) wastes credit per unit of progress.
        # 1.0 / capacity ∈ [1.0, 10.0]; a 50 %% steal peer is twice as
        # costly as a 0 %% peer, a 90 %% peer ten times. Cubed so the
        # penalty dominates a small noise difference but not a 5-x bw
        # gap.
        cap = _peer_cpu_capacity(pid)
        steal_pen = (1.0 / cap) - 1.0    # 0.0 at 100%% capacity
        # Sort key: low noise+steal first, then high remaining bandwidth.
        # Negate remaining so larger sorts earlier under ascending sort.
        candidates.append((noise + steal_pen, -remaining, pid))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def get_peer_by_id(cfg: dict, peer_id: str) -> dict | None:
    for p in cfg.get('peers', []):
        if p['id'] == peer_id:
            return p
    return None


class PeerDirector:
    """Background thread that orchestrates peer processing."""

    def __init__(self):
        self.cfg = load_peers_config()
        self.state = load_director_state()
        # Seed URL→id reverse map for circuit-breaker attribution.
        global _PEER_URL_TO_ID
        _PEER_URL_TO_ID = {
            (p.get('url') or '').rstrip('/'): p.get('id') or ''
            for p in (self.cfg.get('peers') or [])
            if p.get('url') and p.get('id')
        }
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._lock_fd = None
        # Dedicated oidc-reval daemon thread, owned by THIS Director
        # instance. Started in start(), stopped in stop() so it cleanly
        # joins on step_down / HA handover (singleton swap). Lifecycle
        # is intentionally tied to the instance, NOT a module global,
        # so a demoted director never keeps probing OIDC in the
        # background after relinquishing the role.
        self._oidc_thread: threading.Thread | None = None
        self._oidc_stop = threading.Event()
        # EMA capacity factor (0..1). Persisted to director_state.json so
        # restart of the director (or a gunicorn worker swap) doesn't
        # erase recent fleet-wide warning history. We also restore a
        # per-peer slow noise EMA below so a peer that misbehaved is
        # remembered for hours, not just for the 10-min sliding window.
        self._capacity_ema: float = float(
            self.state.get('capacity_ema_persisted',
                           THROTTLE_MAX_FACTOR) or THROTTLE_MAX_FACTOR)
        # Per-kind smoothed sub-factor EMAs (bev / zenodo / copernicus).
        # Used by ``_effective_creds_per_frontier`` so the per=1↔per=2
        # decision doesn't flap on every transient cop blip — only the
        # smoothed signal crosses the hysteresis thresholds.
        _persisted_sub_ema = self.state.get('sub_factor_ema') or {}
        self._sub_factor_ema: dict[str, float] = {
            k: float(_persisted_sub_ema.get(k, THROTTLE_MAX_FACTOR)
                      or THROTTLE_MAX_FACTOR)
            for k in ('bev', 'zenodo', 'copernicus')
        }
        self._capacity_components: dict = {}
        # Sliding history of (ts, factor, bev, zenodo, copernicus) tuples.
        # Sized for ~2h of 30s ticks (240 entries). Survives ticks but not
        # restarts; that's fine for a UI sparkline.
        from collections import deque as _dq
        self._capacity_history = _dq(maxlen=240)
        # Sliding history of (ts, frontiers, cache_only, cache_ready)
        # tuples for the Progress card sparkline. Same retention
        # window (~2h) and persistence as capacity_history so the
        # chart survives restarts and is visible to all gunicorn
        # workers.
        self._peer_history = _dq(maxlen=240)
        # Restore persisted history on startup (and let non-director
        # gunicorn workers see the same data via load_director_state).
        try:
            for entry in (self.state.get('capacity_history') or []):
                # stl/cpu use None for legacy entries (pre-2026-05-19)
                # so /process.txt min/med/max stats don't get
                # contaminated by 0.0 placeholders. Live ticks always
                # write real values.
                _stl = entry.get('stl')
                _cpu = entry.get('cpu')
                self._capacity_history.append((
                    int(entry.get('t') or 0),
                    float(entry.get('f') or 0.0),
                    float(entry.get('bev') or 0.0),
                    float(entry.get('zen') or 0.0),
                    float(entry.get('cop') or 0.0),
                    float(_stl) if isinstance(_stl, (int, float)) else None,
                    float(_cpu) if isinstance(_cpu, (int, float)) else None,
                ))
        except Exception:
            pass
        try:
            for entry in (self.state.get('peer_history') or []):
                self._peer_history.append((
                    int(entry.get('t') or 0),
                    int(entry.get('fr') or 0),
                    int(entry.get('co') or 0),
                    int(entry.get('cr') or 0),
                ))
        except Exception:
            pass

    # --- Server-friendliness throttle ------------------------------------
    # When the fleet has at least this many reporting peers, switch from
    # ``max`` aggregation to the high-percentile aggregation below.
    # Below this count a single hot peer must still drive the decision;
    # above it, one peer in a stale-cred retry storm shouldn't drag the
    # whole fleet down (we have ~59 peers in production).
    _PERCENTILE_MIN_PEERS = 10
    # 90th percentile across reporting peers — i.e. ignore the worst
    # ~10% of outliers. With 59 peers that means up to 5 hot peers can
    # be misbehaving without throttling the rest.
    _PERCENTILE = 0.90

    def _fleet_warning_rates(self, statuses: dict) -> dict:
        """Aggregate per-peer ``warning_rates`` into fleet-wide signals.

        Small fleets (< _PERCENTILE_MIN_PEERS reporting): use ``max`` per
        kind — a single peer seeing 6 BEV warnings/min is already a
        strong signal that the upstream is pushing back.

        Large fleets: use the 90th percentile per kind. With 59 peers a
        couple of misconfigured / stale-cred peers shouldn't drag the
        capacity factor for the other 57. Persistent saturation across
        many peers still wins because the 90th percentile follows the
        body of the distribution.

        ``auth`` is reported but doesn't feed throttle (see comment on
        ``THROTTLE_SATURATION_RATE`` — 401 / invalid_client means a
        peer's local cred file is stale, not that the upstream is
        unhealthy).
        """
        # Per-kind collected rates from live peers (one entry per peer).
        per_kind: dict[str, list[float]] = {
            'bev': [], 'zenodo': [], 'copernicus': [], 'auth': [],
        }
        # Per-kind: which peer pushed the highest rate (for tooltips).
        top_peer: dict[str, tuple[float, str]] = {
            k: (0.0, '') for k in per_kind
        }
        peer_count = 0
        # Track per-peer 'last seen running' so we can give unreachable
        # peers a grace period before excluding them. A peer doing a
        # heavy GPKG upload may stop responding to /processing/status
        # for several minutes — we don't want to drop its warning
        # signal in that case. Only after UNREACHABLE_GRACE seconds do
        # we treat the peer's stale warning window as no-longer-current.
        UNREACHABLE_GRACE = 30 * 60   # 30 minutes
        now = time.time()
        last_live = self.state.setdefault('peer_last_live_ts', {})
        for pid, ps in (statuses or {}).items():
            wr = (ps or {}).get('warning_rates') or {}
            state = (ps or {}).get('state')
            is_live = state in ('running', 'processing', 'idle')
            if is_live:
                last_live[pid] = int(now)
            if not wr:
                continue
            # Stopped is a clean exit (SIGTERM during update / manual
            # stop) — we know the peer isn't doing work, so its stale
            # warning window is safely droppable immediately.
            #
            # 'complete' is the post-run terminal state when a peer
            # finishes its assigned KGs. Same logic applies: the
            # processor isn't running, no new warnings are landing,
            # whatever's in warning_rates is a frozen snapshot of the
            # last KG. Including it pinned the fleet capacity_factor at
            # ~0.2 even when nothing live was emitting warnings.
            if state in ('stopped', 'complete'):
                continue
            # Unreachable peers may be busy uploading or briefly
            # network-flaky. Honour their last warning rates until
            # they've been silent for UNREACHABLE_GRACE.
            if state == 'unreachable':
                last = last_live.get(pid, 0)
                if last and (now - last) > UNREACHABLE_GRACE:
                    continue
                # else: include as if live (peer may be uploading)
            peer_count += 1
            for kind in per_kind:
                # 5-min window is the sweet spot: long enough to ignore
                # one-off retries, short enough to react within ~5 min.
                rate = float(((wr.get(kind) or {}).get('5m')) or 0.0)
                # Always record the rate (even 0) so percentile
                # computation reflects the true fleet shape.
                per_kind[kind].append(rate)
                if rate > top_peer[kind][0]:
                    top_peer[kind] = (rate, str(pid))
        # Aggregate per kind. ``auth`` always uses max because it's
        # diagnostic only (not fed into the throttle).
        agg: dict[str, float] = {}
        use_pctl = peer_count >= self._PERCENTILE_MIN_PEERS
        for kind, rates in per_kind.items():
            if not rates:
                agg[kind] = 0.0
                continue
            if kind == 'auth' or not use_pctl:
                agg[kind] = max(rates)
            else:
                # 90th percentile via nearest-rank.
                xs = sorted(rates)
                idx = max(0, min(len(xs) - 1,
                                  int(round(self._PERCENTILE *
                                            (len(xs) - 1)))))
                agg[kind] = xs[idx]
        agg['_peers_reporting'] = peer_count
        agg['_aggregation'] = 'p90' if use_pctl else 'max'
        # Surface the loudest peer per kind so operators can see which
        # peer is dragging the signal up (especially useful when p90
        # filtered it out and the throttle didn't engage).
        agg['_top_peer'] = {
            k: {'rate': round(v[0], 3), 'pid': v[1]}
            for k, v in top_peer.items() if v[0] > 0
        }
        # Track which peers contributed and which didn't so we can
        # surface gaps in /api/v1/director/status. Useful when we
        # expect 50 peers and only 45 report — the missing 5 are
        # silent because they're scheduled, unreachable past grace,
        # or stopped.
        all_pids = sorted((statuses or {}).keys())
        reporting_pids = sorted(
            pid for pid, ps in (statuses or {}).items()
            if (ps or {}).get('warning_rates')
            and (ps or {}).get('state') not in ('stopped', 'complete'))
        agg['_peers_reporting_ids'] = reporting_pids
        agg['_peers_silent_ids'] = [
            pid for pid in all_pids if pid not in set(reporting_pids)]
        return agg

    def _capacity_factor(self, statuses: dict) -> float:
        """Combined capacity factor in [THROTTLE_MIN_FACTOR, 1.0].

        Per-kind sub-factors decay linearly from 1.0 (no warnings) down to
        THROTTLE_MIN_FACTOR at the saturation rate. The minimum sub-factor
        wins (i.e. whichever upstream is angriest dominates). An EMA
        smooths it across ticks; a slow sinusoidal drift adds a natural
        wobble so we don't sit pinned at the cap.
        """
        rates = self._fleet_warning_rates(statuses)
        # Fleet CPU-steal median across *running* peers — same population
        # the cache-only ramp brake uses (see _max_cache_only_peers).
        # Snapshot it here so /api/v1/director/status carries it on
        # every tick (cheap; we already have ``statuses``) and the
        # sparkline can plot the rolling history.
        try:
            _steal_vals = []
            for _pid, _ps in (statuses or {}).items():
                _st = (_ps or {}).get('state')
                if _st not in ('running', 'processing'):
                    continue
                _sys = (_ps or {}).get('system') or {}
                _perf = _sys.get('perf') or {}
                _v = _perf.get('cpu_steal_ewma')
                if _v is None:
                    _v = _sys.get('cpu_steal')
                if isinstance(_v, (int, float)):
                    _steal_vals.append(float(_v))
            _steal_vals.sort()
            if _steal_vals:
                _n = len(_steal_vals)
                _steal_med = (_steal_vals[_n // 2] if _n % 2
                               else (_steal_vals[_n // 2 - 1]
                                     + _steal_vals[_n // 2]) / 2)
            else:
                _steal_med = 0.0
        except Exception:
            _steal_vals = []
            _steal_med = 0.0
        # cpu_factor curve mirrors the damping in _max_cache_only_peers:
        # gentle ramp-brake that floors at 0.55. Surfaces *why* the
        # director may be running fewer cache-only peers than the
        # warning-rate ceiling alone would allow.
        if _steal_med >= 40.0:
            _cpu_factor = max(0.55, 1.0 - (_steal_med - 30.0) / 200.0)
        else:
            _cpu_factor = 1.0
        sub = {}
        for kind, sat in THROTTLE_SATURATION_RATE.items():
            r = float(rates.get(kind, 0.0))
            if sat <= 0:
                sub[kind] = 1.0
                continue
            # Dead-zone: tiny ambient rates read as zero. Without this,
            # a single peer logging one warning every ~3 min sits at
            # ~0.3/min fleet-max and drags capacity off 100% forever.
            dead = THROTTLE_DEAD_ZONE_FRAC * sat
            if r <= dead:
                sub[kind] = THROTTLE_MAX_FACTOR
                continue
            # 1.0 at r=dead, MIN_FACTOR at r>=sat, linear in between.
            frac = max(0.0, min(1.0, (r - dead) / (sat - dead)))
            sub[kind] = THROTTLE_MAX_FACTOR - frac * (
                THROTTLE_MAX_FACTOR - THROTTLE_MIN_FACTOR)
        raw = min(sub.values()) if sub else THROTTLE_MAX_FACTOR
        # EMA smoothing.
        a = THROTTLE_EMA_ALPHA
        self._capacity_ema = a * raw + (1.0 - a) * self._capacity_ema
        # Per-kind EMAs (same alpha) — feed adaptive decisions that
        # depend only on one upstream (e.g. cred-per-frontier).
        for _k in ('bev', 'zenodo', 'copernicus'):
            _v = float(sub.get(_k, THROTTLE_MAX_FACTOR))
            self._sub_factor_ema[_k] = (
                a * _v + (1.0 - a) * self._sub_factor_ema.get(
                    _k, THROTTLE_MAX_FACTOR))
        # Slow sinusoidal drift to mimic an organic activity pattern.
        # Phase derived from local hostname so different primaries (and
        # the future-self of the same primary post‑restart) stay roughly
        # in sync but not identical.
        import math, socket
        phase = (abs(hash(socket.gethostname())) % 1000) / 1000.0
        t = time.time() / THROTTLE_DRIFT_PERIOD_S + phase
        drift = THROTTLE_DRIFT_AMPLITUDE * math.sin(2 * math.pi * t)
        final = max(THROTTLE_MIN_FACTOR,
                    min(THROTTLE_MAX_FACTOR, self._capacity_ema + drift))
        self._capacity_components = {
            'rates': {k: rates.get(k, 0.0)
                       for k in ('bev', 'zenodo', 'copernicus', 'auth')},
            'steal_median': round(float(_steal_med), 1),
            'steal_n': len(_steal_vals),
            'cpu_factor': round(float(_cpu_factor), 3),
            'sub_factors': {k: round(v, 3) for k, v in sub.items()},
            'sub_factor_emas': {k: round(float(v), 3)
                                 for k, v in self._sub_factor_ema.items()},
            'raw': round(raw, 3),
            'ema': round(self._capacity_ema, 3),
            'drift': round(drift, 3),
            'factor': round(final, 3),
            'peers_reporting': rates.get('_peers_reporting', 0),
            'peers_reporting_ids': rates.get('_peers_reporting_ids') or [],
            'peers_silent_ids': rates.get('_peers_silent_ids') or [],
            # Aggregation strategy (max for small fleets, p90 for >=10
            # reporting peers) and per-kind loudest peer. Both surface
            # in /api/v1/director/status so the dashboard can show
            # "Cop signal driven by peer at43 (1.8/min); fleet p90 0.0".
            'aggregation': rates.get('_aggregation', 'max'),
            'top_peer': rates.get('_top_peer') or {},
        }
        # Append to ring buffer. Compact tuple (no dict) to keep the
        # JSON payload small even when serialised in get_status().
        self._capacity_history.append((
            int(time.time()),
            round(final, 3),
            round(float(rates.get('bev', 0.0)), 3),
            round(float(rates.get('zenodo', 0.0)), 3),
            round(float(rates.get('copernicus', 0.0)), 3),
            round(float(_steal_med), 1),
            round(float(_cpu_factor), 3),
        ))
        return final

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        # Use a file lock to ensure only one director loop runs across
        # all gunicorn workers. The lock is non-blocking; if another worker
        # already holds it, this worker skips the director loop.
        import fcntl
        lock_path = DATA_DIR / 'director.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock_fd = open(lock_path, 'w')
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, IOError):
            log.info('PeerDirector lock held by another worker — skipping')
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # Freshen director_state.json mtime immediately so heartbeats
        # return 200 right after a srv restart. Without this, the file
        # carries its pre-restart mtime (often 5+ minutes old), the
        # heartbeat liveness check fails, peer watchdogs trip a takeover,
        # and the cluster cascades. The loop will refresh the contents
        # within ~30s; this is just a marker that says 'a director loop
        # is here'.
        try:
            DIRECTOR_STATE.parent.mkdir(parents=True, exist_ok=True)
            DIRECTOR_STATE.touch(exist_ok=True)
        except Exception:
            pass
        # Spin up the dedicated oidc-reval daemon. Must come after the
        # file-lock acquisition so we never have a non-director thread
        # probing OIDC. Stopped in self.stop() on step_down/handover.
        self._oidc_stop.clear()
        self._oidc_thread = threading.Thread(
            target=self._oidc_reval_loop, name='oidc-reval', daemon=True)
        self._oidc_thread.start()
        log.info('PeerDirector started (lock acquired)')
        # Smoke test: a director with zero Copernicus credentials cannot
        # run frontiers. The 2026-05-08 incident silently disarmed the
        # whole fleet because builtin creds were removed and the
        # cred-fanout helper had a NameError. Surfacing this loud at
        # startup makes the failure mode obvious in journalctl.
        try:
            import copernicus as _cop_smoke
            n_creds = len(_cop_smoke.list_credentials() or [])
            if n_creds == 0:
                log.critical(
                    'CRITICAL: director starting with 0 Copernicus '
                    'credentials. Frontier KGs will fail. Add via '
                    'POST /api/v1/credentials or set '
                    'COPERNICUS_BOOTSTRAP_CREDS env on srv.service.')
            else:
                log.info('Director credential pool: %d entries', n_creds)
        except Exception as _e:
            log.warning('credential smoke test failed: %s', _e)

    def stop(self, *, join_timeout: float = 8.0):
        """Stop the director loop, join the thread, and release the file lock.

        Critical for in-process handover: ``_do_takeover`` calls
        ``old.stop()`` and immediately tries ``new.start()``. If we don't
        join + release here, the new instance's ``fcntl.LOCK_NB`` fails
        (same-process), the takeover silently aborts, and we end up with
        ``IS_DIRECTOR_FLAG`` set on disk but no loop running. The
        heartbeat then returns 200 forever while the cluster is
        actually unmanaged.
        """
        self._running = False
        # Stop the oidc-reval daemon first so it can't fire a long
        # OIDC sweep right after we release the director lock. join
        # under the same budget as the loop thread.
        try:
            self._oidc_stop.set()
        except Exception:
            pass
        oidc_thr = self._oidc_thread
        if (oidc_thr and oidc_thr.is_alive()
                and oidc_thr is not threading.current_thread()):
            try:
                oidc_thr.join(timeout=max(1.0, join_timeout))
            except Exception as e:
                log.warning('PeerDirector.stop: oidc-reval join failed: %s', e)
        self._oidc_thread = None
        thr = self._thread
        if thr and thr.is_alive() and thr is not threading.current_thread():
            try:
                thr.join(timeout=join_timeout)
            except Exception as e:
                log.warning('PeerDirector.stop: thread join failed: %s', e)
        self._thread = None
        # Explicitly release the file lock so the next instance can
        # acquire it in this same process.
        fd = getattr(self, '_lock_fd', None)
        if fd is not None:
            try:
                import fcntl as _fcntl
                _fcntl.flock(fd, _fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                fd.close()
            except Exception:
                pass
            self._lock_fd = None

    # Short-lived cache for get_status() output. Dashboard polls
    # /director/status (15s) AND /director/proxy/status (5s, also calls
    # get_status). Each fanout hits all 60 peers (status + credentials)
    # with PEER_TIMEOUT_PROBE up to (3,8) — easily 5–15s wall-clock.
    # Without coalescing, every gunicorn thread (8 total) is permanently
    # blocked inside as_completed(), wedging the entire web app.
    # 4s TTL means at most one fanout per dashboard cycle; single-flight
    # lock means concurrent callers share the same in-flight result
    # instead of each launching their own 60-peer fanout.
    # Fresh window: callers within this many seconds of the last
    # successful compute get the cached value with zero work. The
    # 60-peer fanout costs ~3–6s on a warm system; setting TTL below
    # that guarantees most callers pay full price (every dashboard
    # poll, every process.txt curl from an agent). 30s is short
    # enough that stale director state is never user-visible (the
    # dashboard refreshes faster than a peer can change role) but
    # long enough that all five dashboard pollers (process.html
    # poll/5s, fetchPrioQueue/15s, fetchDirectorStatus/15s,
    # fetchAllPeerStatuses/8s, fetchLog24h/30s) hit cache during a
    # single page-load wave.
    # Cross-worker disk cache: gunicorn runs 2 workers. Without a
    # shared cache each worker has its own in-memory copy, and a
    # client doing one request every TTL/2 seconds still pays cold
    # cost on every alternating worker. The disk file is a tiny
    # JSON payload written atomically; readers tolerate parse
    # errors silently.
    _STATUS_CACHE_FILE = Path('/tmp/srtm_director_status_cache.json')
    _STATUS_CACHE_TTL = 30.0
    # Path of the director-role flag. Mirrors director_ha.IS_DIRECTOR_FLAG
    # but kept as a module constant to avoid the import cycle on the hot
    # path. Both module references must agree (asserted in tests).
    _IS_DIRECTOR_FLAG = DATA_DIR / 'is_director'
    # Stale window: if a recompute is in flight, callers that arrive
    # with a cached value at most this old are served the stale copy
    # immediately instead of blocking. Keeps /process.txt and
    # /process.html responsive even during fanout. We trigger a
    # background refresh when the cache crosses TTL but is still
    # within STALE, so the next caller sees fresh data without any
    # request having had to wait.
    _STATUS_CACHE_STALE = 180.0

    def _is_director_signature(self) -> tuple[bool, int]:
        """Cheap fingerprint of the ``is_director`` flag file.

        Returned tuple is (exists, mtime_ns). Used by ``get_status`` to
        detect a director flip that happened in *another* gunicorn
        worker (or via direct disk write — handover, HA promotion,
        manual recovery) and invalidate this worker's status cache.

        Without this guard, a 2-worker gunicorn that flipped role in
        only one worker would have the other worker keep returning a
        cached status reflecting the old role for up to
        ``_STATUS_CACHE_STALE`` (3 min). On a primary that just got
        demoted (e.g. bandwidth-walled, manual park, takeover by a
        peer) that stale worker would also keep running director
        side-effects driven from cached state — including the 60-peer
        fanout that burns external bandwidth. Cheap stat() per call is
        well worth it.
        """
        try:
            st = self._IS_DIRECTOR_FLAG.stat()
            return (True, int(st.st_mtime_ns))
        except FileNotFoundError:
            return (False, 0)
        except Exception:
            # Be conservative: on stat error treat as "changed" so we
            # recompute rather than serve a possibly-stale cache.
            return (False, -1)

    def get_status(self) -> dict:
        """Full director status for the dashboard.

        TTL-cached + single-flight + stale-while-revalidate:
        - within ``_STATUS_CACHE_TTL``: serve cache, no work
        - between TTL and STALE: serve cache, kick a background
          refresh (if none in flight)
        - older than STALE or no cache: block on a recompute
          (single-flight: concurrent callers share the same in-flight
          result instead of each launching their own 60-peer fanout)
        """
        import time as _t
        # Lazy-init cache attrs (instance is constructed before this method
        # may be called from multiple threads).
        if not hasattr(self, '_status_cache_lock'):
            self._status_cache_lock = threading.Lock()
            self._status_cache_inflight = threading.Lock()
            self._status_cache_value: dict | None = None
            self._status_cache_ts: float = 0.0
        now = _t.time()
        # Snapshot the director-flag fingerprint *before* we read the
        # cache. If it differs from the fingerprint stamped on the
        # cache, drop both the in-memory and on-disk copies and force
        # a recompute. This catches the cross-worker drift case where
        # one gunicorn worker flipped role (writing/removing
        # ``is_director``) but the other worker's cache still reflects
        # the old role. See ``_is_director_signature`` for rationale.
        flag_sig = self._is_director_signature()
        with self._status_cache_lock:
            cached = self._status_cache_value
            cached_sig = getattr(self, '_status_cache_flag_sig', None)
            if cached is not None and cached_sig != flag_sig:
                log.info(
                    'director-flag flip detected (cache_sig=%s '
                    'disk_sig=%s) — invalidating get_status cache',
                    cached_sig, flag_sig,
                )
                self._status_cache_value = None
                self._status_cache_ts = 0.0
                cached = None
                # Also nuke the cross-worker disk cache so the *other*
                # worker doesn't immediately re-promote a stale value
                # we just rejected.
                try:
                    self._STATUS_CACHE_FILE.unlink()
                except FileNotFoundError:
                    pass
                except Exception as _e:  # noqa: BLE001
                    log.debug('status cache disk unlink failed: %s', _e)
            age = now - self._status_cache_ts if cached is not None else None
        # Cross-worker promotion: if our in-memory copy is stale (or
        # missing) but the disk file is fresher, adopt it. This is
        # what lets the second gunicorn worker piggy-back on a
        # recompute the first worker just paid for.
        if cached is None or (age is not None and age >= self._STATUS_CACHE_TTL):
            disk = self._status_cache_load_disk()
            if disk is not None:
                disk_ts = float(disk.get('_cached_at', 0) or 0)
                disk_sig = (
                    bool(disk.get('_flag_exists', False)),
                    int(disk.get('_flag_mtime_ns', 0) or 0),
                )
                disk_age = now - disk_ts
                # Reject the disk cache too if it was written by a
                # worker that saw a different director-flag state.
                # Prevents a stale cross-worker promotion from
                # resurrecting the cache we just invalidated.
                if disk_sig != flag_sig:
                    log.debug(
                        'disk status cache flag mismatch '
                        '(disk_sig=%s live_sig=%s) — ignoring',
                        disk_sig, flag_sig,
                    )
                elif disk_age >= 0 and (age is None or disk_age < age):
                    with self._status_cache_lock:
                        self._status_cache_value = disk
                        self._status_cache_ts = disk_ts
                        self._status_cache_flag_sig = disk_sig
                    cached = disk
                    age = disk_age
        if cached is not None and age is not None and age < self._STATUS_CACHE_TTL:
            return cached
        # Stale-while-revalidate: serve aged cache + spawn a refresh.
        if cached is not None and age is not None and age < self._STATUS_CACHE_STALE:
            if self._status_cache_inflight.acquire(blocking=False):
                def _bg_refresh():
                    try:
                        # Re-snapshot the flag fingerprint at the
                        # moment of compute, not at the time the
                        # request came in.
                        sig = self._is_director_signature()
                        result = self._compute_status()
                        with self._status_cache_lock:
                            self._status_cache_value = result
                            self._status_cache_ts = _t.time()
                            self._status_cache_flag_sig = sig
                        self._status_cache_save_disk(result, sig)
                    except Exception as _e:  # noqa: BLE001
                        log.warning('get_status background refresh: %s', _e)
                    finally:
                        try:
                            self._status_cache_inflight.release()
                        except RuntimeError:
                            pass
                threading.Thread(
                    target=_bg_refresh,
                    name='dir-status-bg',
                    daemon=True,
                ).start()
            # Else: another thread is already refreshing — fine.
            return cached
        # No usable cache: block on compute under single-flight.
        acquired = self._status_cache_inflight.acquire(timeout=20)
        if not acquired:
            with self._status_cache_lock:
                if self._status_cache_value is not None:
                    return self._status_cache_value
            # No prior cache and we couldn't get the lock — fall through
            # and compute (rare, only on first request).
            self._status_cache_inflight.acquire()
        try:
            # Re-check after acquiring lock — another thread may have
            # just populated the cache.
            with self._status_cache_lock:
                if (self._status_cache_value is not None and
                        (_t.time() - self._status_cache_ts) < self._STATUS_CACHE_TTL):
                    return self._status_cache_value
            sig = self._is_director_signature()
            result = self._compute_status()
            with self._status_cache_lock:
                self._status_cache_value = result
                self._status_cache_ts = _t.time()
                self._status_cache_flag_sig = sig
            self._status_cache_save_disk(result, sig)
            return result
        finally:
            try:
                self._status_cache_inflight.release()
            except RuntimeError:
                pass

    def _status_cache_load_disk(self) -> dict | None:
        try:
            if self._STATUS_CACHE_FILE.exists():
                with self._STATUS_CACHE_FILE.open() as f:
                    return json.load(f)
        except Exception:
            return None
        return None

    def _status_cache_save_disk(
        self, payload: dict, flag_sig: tuple[bool, int] | None = None,
    ) -> None:
        try:
            d = dict(payload)
            d['_cached_at'] = time.time()
            if flag_sig is None:
                flag_sig = self._is_director_signature()
            d['_flag_exists'] = bool(flag_sig[0])
            d['_flag_mtime_ns'] = int(flag_sig[1])
            tmp = self._STATUS_CACHE_FILE.with_suffix('.tmp')
            tmp.write_text(json.dumps(d, default=str))
            tmp.replace(self._STATUS_CACHE_FILE)
        except Exception as _e:  # noqa: BLE001
            log.debug('status cache disk save failed: %s', _e)

    def _compute_status(self) -> dict:
        """Full director status for the dashboard."""
        # Always re-read config from disk so all gunicorn workers see new peers
        try:
            disk_cfg = load_peers_config()
            with self._lock:
                self.cfg = disk_cfg
        except Exception:
            pass
        # Also pull director_state from disk so workers that don't run
        # the director loop (e.g. the worker handling this request) see
        # fields written by the worker that does — peer_update_state,
        # capacity_factor, etc.
        try:
            disk_state = load_director_state()
        except Exception:
            disk_state = {}
        with self._lock:
            cfg = self.cfg.copy()
            state = self.state.copy()
        # Merge disk values for fields the director writes; keep our
        # in-memory values otherwise so per-worker scratch state isn't
        # clobbered.
        for _k in ('peer_update_state', 'capacity_factor',
                   'capacity_components', 'capacity_history',
                   'peer_history', 'canary_history',
                   'canary_slowdown_streaks', 'canary_fleet_slowdown',
                   'capacity_ema_persisted', 'sub_factor_ema',
                   '_target_frontier_count',
                   'peer_warning_rates', 'peer_noise_long_ema',
                   'peer_last_live_ts', 'peer_meta',
                   'parallel_frontiers_active', 'frontier_cred_plan',
                   'frontier_strip_plan', 'cache_only_active',
                   'parallel_unreachable_count',
                   '_creds_revalidated_at',
                   '_cache_ready_cache',
                   'active_peer', 'mode', 'last_switch'):
            if _k in disk_state:
                state[_k] = disk_state[_k]
        # Sync the per-process EMA caches from disk too. Workers that
        # don't run the director loop never update these in memory, so
        # _max_parallel_frontiers / _target_frontier_count would compute
        # against the THROTTLE_MAX_FACTOR defaults — producing a higher
        # frontier target than the director-running worker. Two workers
        # serving alternating /api/v1/director/status requests then make
        # the dashboard flicker between e.g. 4 and 8 max frontiers.
        try:
            _disk_ema = disk_state.get('capacity_ema_persisted')
            if _disk_ema is not None:
                self._capacity_ema = float(_disk_ema)
            _disk_sub = disk_state.get('sub_factor_ema') or {}
            for _kk in ('bev', 'zenodo', 'copernicus'):
                if _kk in _disk_sub:
                    self._sub_factor_ema[_kk] = float(_disk_sub[_kk])
        except Exception:
            pass

        # Poll all peer statuses in parallel — a single wedged peer
        # must never wedge the dashboard request.
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
        statuses: dict[str, dict] = {}
        peers_list = list(cfg.get('peers', []))

        def _fallback_from_cache(reason: str) -> None:
            """Populate ``statuses`` from the per-URL cache (or mark
            as unreachable). Used when the thread pool is unusable.
            """
            import time as _t
            for p in peers_list:
                cached = _PEER_STATUS_CACHE.get(p.get('url') or '')
                if cached and (_t.time() - cached[0]) < _PEER_STATUS_CACHE_TTL:
                    d = dict(cached[1])
                    d['_stale'] = True
                    d['_stale_age_s'] = round(_t.time() - cached[0], 1)
                    d['_stale_error'] = reason
                    statuses[p['id']] = d
                else:
                    statuses[p['id']] = {'state': 'unreachable',
                                         'error': reason}

        # Guard against ``RuntimeError: cannot schedule new futures
        # after interpreter shutdown`` — happens when a gunicorn worker
        # has hit EMFILE (Too many open files) and is mid-shutdown but
        # Flask is still trying to dispatch /director/status. The
        # exception used to fire on every poll for the dying request,
        # spamming the log with thousands of identical tracebacks
        # (which previously masqueraded as a recursion error in the
        # dashboard log).
        try:
            with ThreadPoolExecutor(
                max_workers=BANDWIDTH_POLL_CONCURRENCY,
                thread_name_prefix='dir-st',
            ) as ex:
                futs = {ex.submit(get_peer_status, p.get('url')): p
                        for p in peers_list}
                import time as _t
                deadline = _t.time() + 15
                for fut, p in list(futs.items()):
                    try:
                        statuses[p['id']] = fut.result(
                            timeout=max(0.1, deadline - _t.time())
                        )
                    except (FTimeout, Exception) as e:
                        # Worker-pool budget blew up before the per-call
                        # timeout fired — try the last-known cached
                        # status before declaring this peer unreachable.
                        cached = _PEER_STATUS_CACHE.get(p.get('url') or '')
                        if cached and (_t.time() - cached[0]) < _PEER_STATUS_CACHE_TTL:
                            d = dict(cached[1])
                            d['_stale'] = True
                            d['_stale_age_s'] = round(_t.time() - cached[0], 1)
                            d['_stale_error'] = str(e)[:120]
                            statuses[p['id']] = d
                        else:
                            statuses[p['id']] = {'state': 'unreachable',
                                                 'error': str(e)}
        except RuntimeError as _rte:
            log.warning('get_status: thread pool unavailable (%s) — '
                        'returning cached peer statuses', _rte)
            _fallback_from_cache('interpreter_shutdown')

        # Persistent per-peer metadata: remembers last-known git_commit,
        # region, last_kg_code/name/uploaded_at across peer reboots and
        # status timeouts. Mutated below when a peer reports fresh values.
        meta_cache = state.setdefault('peer_meta', {})
        peers_status = []
        for peer in peers_list:
            pid = peer['id']
            url = peer.get('url')
            ps = statuses.get(pid, {'state': 'unreachable'})
            # Refresh meta_cache from the live status (only when the peer
            # actually reported a value — never overwrite a known value
            # with an empty string from a stale/proxied response).
            try:
                _meta = meta_cache.setdefault(pid, {})
                for _k in ('git_commit', 'region', 'last_kg_code',
                           'last_kg_name', 'last_kg_uploaded_at',
                           'last_kg_seconds'):
                    _v = ps.get(_k)
                    if _v not in (None, '', 0):
                        _meta[_k] = _v
            except Exception:
                pass
            bw = dict(state.get('peer_bandwidth', {}).get(pid, {}))
            proc_status = ps.get('state', 'unknown')
            # Surface effective per-peer budget so dashboard / API
            # consumers see the canary override, not just the global.
            _eff_budget_gb = round(_peer_budget_bytes(peer, cfg) / (1024 ** 3), 2)
            bw['effective_budget_gb'] = _eff_budget_gb
            if peer.get('budget_gb') is not None:
                # Recompute remaining_gb against the per-peer budget so
                # the dashboard doesn't show "0.0 GB remaining" for a
                # canary that actually has headroom under its override.
                _used_bytes = bw.get('used_bytes', 0) or 0
                bw['remaining_gb'] = round(
                    max(0, int(_eff_budget_gb * (1024 ** 3)) - _used_bytes)
                    / (1024 ** 3), 2)
                bw['budget_gb'] = _eff_budget_gb

            # Per-peer renew_day: explicit override > day-of-month of
            # first_seen > global cfg renew_day. This is informational
            # only — vnstat cycle math still uses the global anchor (no
            # peer rollout required to change behaviour) — but it lets
            # the dashboard show realistic per-peer reset estimates.
            _renew_day = peer.get('renew_day')
            if not _renew_day:
                _fs = peer.get('first_seen')
                if isinstance(_fs, str) and len(_fs) >= 10:
                    try:
                        _renew_day = int(_fs[8:10])
                    except (ValueError, TypeError):
                        _renew_day = None
            if not _renew_day:
                _renew_day = cfg.get('renew_day', BANDWIDTH_RENEW_DAY)

            peers_status.append({
                'id': pid,
                'url': url,
                'enabled': peer.get('enabled', True),
                'not_before': peer.get('not_before'),
                'scheduled': _peer_is_scheduled(peer),
                'reserved_kg': peer.get('reserved_kg'),
                'renew_day': _renew_day,
                'observed_cap_gb': peer.get('observed_cap_gb'),
                'observed_cap_at': peer.get('observed_cap_at'),
                'first_seen': peer.get('first_seen'),
                'stale_status': bool(ps.get('_stale')),
                'stale_age_s': ps.get('_stale_age_s'),
                'zenodo_cooldown_history': peer.get('zenodo_cooldown_history') or [],
                # Warning fingerprint for load-shifting visibility.
                'warning_rates': ps.get('warning_rates') or {},
                'noise_score': round(_peer_noise_score(pid, state), 3),
                'circuit_breaker': _cb_state_for(pid),
                'canary': self._canary_summary(peer, state),
                'role': self._peer_role(peer),
                'cache_only_run': bool(ps.get('cache_only')),
                'is_active': pid == state.get('active_peer'),
                'processor_state': proc_status,
                'current_kg': (ps.get('current_kg') or {}).get('code'),
                'current_kg_name': (ps.get('current_kg') or {}).get('name'),
                'current_kg_started_at': (ps.get('current_kg') or {}).get('started_at'),
                'current_kg_step': (ps.get('current_kg') or {}).get('step'),
                'current_kg_tile': (ps.get('current_kg') or {}).get('current_tile'),
                'current_kg_n_tiles': (ps.get('current_kg') or {}).get('n_tiles'),
                'completed': ps.get('completed', 0),
                'bandwidth': bw,
                'online': proc_status != 'unreachable',
                'git_commit': (ps.get('git_commit')
                               or (meta_cache.get(pid) or {}).get('git_commit', '')),
                'region': (ps.get('region')
                           or (meta_cache.get(pid) or {}).get('region', '')),
                # Last completed KG (sticky across status timeouts).
                'last_kg_code': (ps.get('last_kg_code')
                                 or (meta_cache.get(pid) or {}).get('last_kg_code')),
                'last_kg_name': (ps.get('last_kg_name')
                                 or (meta_cache.get(pid) or {}).get('last_kg_name')),
                'last_kg_uploaded_at': (ps.get('last_kg_uploaded_at')
                                        or (meta_cache.get(pid) or {}).get('last_kg_uploaded_at')),
                'last_kg_seconds': (ps.get('last_kg_seconds')
                                    or (meta_cache.get(pid) or {}).get('last_kg_seconds')),
                # Authoritative cred_indices/lat_strips from the peer's own
                # progress.json — what the running processor subprocess
                # actually has in env. Used as fallback when the director's
                # frontier_cred_plan / frontier_strip_plan is stale/empty.
                '_reported_cred_indices': ps.get('cred_indices'),
                '_reported_lat_strips': ps.get('lat_strip_filter'),
                '_reported_cells': ps.get('cell_filter'),
                # Stale-update tracking (auto-retry status)
                'update_state': (state.get('peer_update_state') or {}).get(pid),
                # Host telemetry (CPU steal / iowait / busy ewma + perf
                # summary + static host_profile). Peers push these via
                # /api/v1/director/peer_status; without forwarding them
                # here the fleet_cpu line in /process.txt and the
                # steal=/iow= bw_extras chips silently render empty.
                # Trimmed to the fields actually consumed downstream so
                # /director/status doesn't bloat by ~3 KB/peer.
                'system': {
                    k: v for k, v in (ps.get('system') or {}).items()
                    if k in ('cpu_steal', 'cpu_iowait', 'cpu_total',
                             'cpu_user', 'cpu_system',
                             'perf', 'host',
                             'load_1m', 'cpu_pct', 'ram_pct',
                             'disk_free_gb')
                    and v is not None
                },
            })

        cache_ready = state.get('_cache_ready_cache') or {}
        # Count only peers whose processor is actively running in cache-only
        # mode — ``cache_only_run`` reflects last-started mode and stays True
        # for stopped/complete peers too.
        cache_only_running = sum(
            1 for p in peers_status
            if p['cache_only_run']
            and p.get('processor_state') in ('running', 'processing')
        )
        # Theoretical max cache-only peers: enabled, online, not
        # scheduled-out (BW park / not_before), not the active frontier,
        # not authorised for parallel-frontier work, not reserved, and
        # with >2 GB headroom on their budget. Matches the gates in
        # ``_orchestrate_cache_only`` so the dashboard chip can show a
        # realistic denominator instead of the static hard cap.
        _parallel_authorised = set(
            state.get('parallel_frontiers_active') or [])
        _parallel_authorised |= set(
            (state.get('frontier_cred_plan') or {}).keys())
        _active_id = state.get('active_peer')
        _bw_map = state.get('peer_bandwidth') or {}
        cache_only_eligible = 0
        for _p in cfg.get('peers', []):
            _pid = _p.get('id')
            if not _pid or not _p.get('enabled', True):
                continue
            if _peer_is_scheduled(_p):
                continue
            if _pid == _active_id:
                continue
            if _pid in _parallel_authorised:
                continue
            if _p.get('reserved_kg'):
                continue
            _used = (_bw_map.get(_pid) or {}).get('used_bytes', 0)
            if (_peer_budget_bytes(_p, cfg) - _used) < 2 * (1024 ** 3):
                continue
            # Online check via peers_status (which already polled).
            _row = next((r for r in peers_status if r['id'] == _pid), None)
            if _row is None or not _row.get('online', False):
                continue
            cache_only_eligible += 1

        # --- Credential pool & assignment plan ----------------------
        try:
            cred_pool = self._credential_pool()
        except Exception:
            cred_pool = []
        try:
            valid_creds = self._valid_credentials()
        except Exception:
            valid_creds = []
        per = self._effective_creds_per_frontier(cfg)
        active_id_now = state.get('active_peer')
        max_par = self._max_parallel_frontiers(cfg)
        cred_plan = state.get('frontier_cred_plan') or {}
        strip_plan = state.get('frontier_strip_plan') or {}
        cached_strips = self._cached_lat_ranges() if hasattr(self, '_cached_lat_ranges') else []
        austria_strips = self._austria_lat_strips()
        austria_cells = self._austria_cells()
        # Annotate peer rows with their assignments — but only when
        # the peer is actually doing frontier work. A stale frontier_cred_plan
        # for a peer currently running cache-only would otherwise mislead
        # the dashboard into thinking those creds are locked.
        for row in peers_status:
            pid = row['id']
            proc_st = row.get('processor_state')
            cache_only_run = bool(row.get('cache_only_run'))
            is_running = proc_st in ('running', 'processing')
            holds_creds = is_running and not cache_only_run
            if holds_creds:
                # Prefer the peer's own reported cred_indices (ground truth
                # from its env) over the director's plan, which can be
                # stale across worker restarts or director state resets.
                reported = row.pop('_reported_cred_indices', None)
                if reported:
                    row['cred_indices'] = list(reported)
                elif pid in cred_plan:
                    row['cred_indices'] = cred_plan[pid]
                reported_strips = row.pop('_reported_lat_strips', None)
                reported_cells = row.pop('_reported_cells', None)
                if reported_cells:
                    row['cells'] = list(reported_cells)
                elif reported_strips:
                    row['lat_strips'] = list(reported_strips)
                elif pid in strip_plan:
                    # strip_plan now stores cell 4-tuples; report under
                    # whichever key matches the data shape so the
                    # dashboard can render it.
                    plan = strip_plan[pid]
                    if plan and len(plan[0]) == 4:
                        row['cells'] = plan
                    else:
                        row['lat_strips'] = plan
            else:
                row.pop('_reported_cred_indices', None)
                row.pop('_reported_lat_strips', None)
                row.pop('_reported_cells', None)
            row['pinned_role'] = (
                next((p.get('pinned_role') for p in cfg.get('peers', [])
                      if p['id'] == pid), None)
            )

        # Refresh capability cache for all peers (cheap when cached).
        for peer in peers_list:
            try:
                self._refresh_peer_caps(peer)
            except Exception:
                pass

        # --- Credential holders: which peer currently holds each cred ---
        # A peer 'holds' a credential if it's running frontier work using it.
        # All peers expose cred_subset_env now — they only hold the cred
        # indices in cred_plan. Cache-only peers hold no credentials.
        cred_holders: dict[int, list[str]] = {i: [] for i in range(len(cred_pool))}
        for row in peers_status:
            pid = row['id']
            proc_st = row.get('processor_state')
            # Only actively-running peers hold credentials. Paused peers
            # release their creds so other peers can use them.
            if proc_st not in ('running', 'processing'):
                continue
            if row.get('cache_only_run'):
                continue
            # Use the row's annotated cred_indices, which already prefers
            # the peer-reported ground truth over a possibly-stale plan.
            indices = row.get('cred_indices') or cred_plan.get(pid) or []
            for idx in indices:
                if 0 <= idx < len(cred_pool):
                    cred_holders[idx].append(pid)
        # Annotate cred_pool entries with holders
        for i, c in enumerate(cred_pool):
            c['holders'] = cred_holders.get(i, [])
        # Aggregate per-credential usage telemetry across all peers — the
        # processor runs on peers, so the director's local stats are 0.
        try:
            self._aggregate_credential_usage(cred_pool, peers_list)
        except Exception as e:
            log.debug('aggregate_credential_usage: %s', e)
        # Append a fleet-shape sample using the SAME numbers we return
        # to the dashboard so the sparkline's last point always matches
        # the card text. Throttled to avoid pile-on from concurrent
        # dashboard polls; the director tick still records its own
        # samples on the 30s cadence as a fallback.
        try:
            _now = int(time.time())
            _last_t = (
                self._peer_history[-1][0]
                if self._peer_history else 0
            )
            if _now - _last_t >= 5:
                _fr = len(state.get('parallel_frontiers_active') or [])
                _co = int(cache_only_running)
                _cr = len(cache_ready.get('codes') or [])
                self._peer_history.append((_now, _fr, _co, _cr))
                with self._lock:
                    self.state['peer_history'] = [
                        {'t': t, 'fr': fr, 'co': co, 'cr': cr}
                        for (t, fr, co, cr) in list(self._peer_history)
                    ]
        except Exception:
            log.debug('peer_history status snapshot failed', exc_info=True)
        return {
            'mode': state.get('mode', 'auto'),
            'active_peer': state.get('active_peer'),
            'last_switch': state.get('last_switch'),
            'budget_gb': cfg.get('budget_gb', BANDWIDTH_BUDGET_GB),
            'renew_day': cfg.get('renew_day', BANDWIDTH_RENEW_DAY),
            'min_reserve_peers': cfg.get('min_reserve_peers', MIN_RESERVE_PEERS),
            'max_cache_only_peers': cfg.get('max_cache_only_peers', MAX_CACHE_ONLY_PEERS),
            'cache_only_eligible': cache_only_eligible,
            'cache_only_running': cache_only_running,
            'cache_ready_kgs': len(cache_ready.get('codes') or []),
            'cache_ready_at': cache_ready.get('at'),
            'credentials': cred_pool,
            'valid_credentials': valid_creds,
            'min_creds_per_frontier': per,
            'min_creds_per_frontier_floor': max(1, int(cfg.get(
                'min_creds_per_frontier',
                self.MIN_CREDS_PER_FRONTIER_DEFAULT))),
            'adaptive_creds_per_frontier': bool(cfg.get(
                'adaptive_creds_per_frontier', True)),
            'max_parallel_frontiers': max_par,
            'cached_lat_strips': [[s, n] for s, n in cached_strips],
            'cached_cells': [list(c) for c in self._cached_cells()],
            'austria_lat_strips': [[s, n] for s, n in austria_strips],
            'austria_cells': [[s, n, w, e] for s, n, w, e in austria_cells],
            'parallel_frontiers_active': state.get('parallel_frontiers_active', []),
            'cache_miss_count': len(self._load_cache_misses()),
            'cycle_start': get_billing_cycle_start().isoformat(),
            'fleet_bw': self._fleet_bw_summary(peers_status),
            'canary_fleet_slowdown': state.get('canary_fleet_slowdown'),
            'capacity_factor': state.get(
                'capacity_factor', self._capacity_ema),
            'capacity_ema': state.get(
                'capacity_ema_persisted', self._capacity_ema),
            'capacity_components': state.get(
                'capacity_components', self._capacity_components),
            # Use in-memory history if this worker runs the loop;
            # otherwise fall back to the persisted snapshot so workers
            # that never tick still serve a populated chart.
            'capacity_history': (
                [({'t': t, 'f': f, 'bev': b, 'zen': z, 'cop': c}
                  | ({'stl': s} if s is not None else {})
                  | ({'cpu': cf} if cf is not None else {}))
                 for (t, f, b, z, c, s, cf) in list(self._capacity_history)]
                if self._capacity_history
                else (state.get('capacity_history') or [])
            ),
            'peer_history': (
                [{'t': t, 'fr': fr, 'co': co, 'cr': cr}
                 for (t, fr, co, cr) in list(self._peer_history)]
                if self._peer_history
                else (state.get('peer_history') or [])
            ),
            # Per-peer warning rates from the last director tick. Used
            # by the dashboard noise pill and by load-shifting logic.
            'peer_warning_rates': state.get('peer_warning_rates') or {},
            'peer_noise_long_ema': state.get('peer_noise_long_ema') or {},
            'peer_last_live_ts': state.get('peer_last_live_ts') or {},
            'peer_meta': state.get('peer_meta') or {},
            'shadow_peer': state.get('shadow_peer'),
            'shadow_url': state.get('shadow_url'),
            'shadow_last_push_ts': state.get('shadow_last_push_ts'),
            'shadow_last_push_ok': state.get('shadow_last_push_ok'),
            'is_director_local': (DATA_DIR / 'is_director').exists(),
            'self_id': self._self_id_safe(),
            # Pointer to the current director (peer view): when this
            # VM is *not* the director, the dashboard renders a
            # "⇗ director: <id>" link to the director's process.html
            # so the user always knows where to find the live
            # orchestration view.
            'self_url': self._self_url_safe(),
            'director_url': self._director_url_safe(),
            'frontier_cred_plan': state.get('frontier_cred_plan') or {},
            'frontier_strip_plan': state.get('frontier_strip_plan') or {},
            'frontier_cell_plan': state.get('frontier_strip_plan') or {},
            'peers': peers_status,
        }

    def _self_id_safe(self) -> str:
        try:
            import director_ha as _dha
            return _dha.self_id()
        except Exception:
            return 'primary'

    def _self_url_safe(self) -> str | None:
        try:
            import director_ha as _dha
            return _dha.self_url()
        except Exception:
            return None

    def _director_url_safe(self) -> str | None:
        """URL of the currently-authoritative director.

        On the director itself this returns its own URL (self_url).
        On a peer it returns the pointer in ``self.json:director_url``
        (mirrored from ``zenodo_lock_url.txt``). Used by the dashboard
        to render a link to the director's process.html so users
        opening a peer dashboard can navigate to the live
        orchestration view in one click.
        """
        try:
            if (DATA_DIR / 'is_director').exists():
                return self._self_url_safe()
            import director_ha as _dha
            return _dha.director_url()
        except Exception:
            return None

    def set_mode(self, mode: str):
        with self._lock:
            self.state['mode'] = mode
            save_director_state(self.state)

    def set_active_peer(self, peer_id: str | None):
        """Manually set the active peer (for manual mode)."""
        with self._lock:
            self.state['active_peer'] = peer_id
            self.state['last_switch'] = datetime.now(timezone.utc).isoformat()
            save_director_state(self.state)

    def reload_config(self):
        with self._lock:
            self.cfg = load_peers_config()
        # Refresh the URL→id reverse map so module-level helpers can
        # attribute circuit-breaker events without threaded peer_id args.
        global _PEER_URL_TO_ID
        _PEER_URL_TO_ID = {
            (p.get('url') or '').rstrip('/'): p.get('id') or ''
            for p in (self.cfg.get('peers') or [])
            if p.get('url') and p.get('id')
        }

    def propagate_throttle(self, enabled: bool) -> dict:
        """Propagate throttle state to all remote peers."""
        results = {}
        with self._lock:
            cfg = self.cfg.copy()
        for peer in cfg.get('peers', []):
            url = peer.get('url')
            if url is None:
                continue
            try:
                # First check current state
                r = requests.get(url.rstrip('/') + '/api/v1/processing/throttle',
                                 timeout=PEER_TIMEOUT_PROBE,
                                 headers=_admin_headers())
                current = r.json().get('throttle', False) if r.ok else None
                if current == enabled:
                    results[peer['id']] = 'already_' + ('on' if enabled else 'off')
                    continue
                # Toggle to match desired state
                r2 = requests.post(url.rstrip('/') + '/api/v1/processing/throttle',
                                   timeout=PEER_TIMEOUT_CONTROL,
                                   headers=_admin_headers())
                if r2.ok:
                    results[peer['id']] = 'set_' + ('on' if enabled else 'off')
                else:
                    results[peer['id']] = f'error_{r2.status_code}'
            except Exception as e:
                results[peer['id']] = f'error: {e}'
        return results

    def remove_peer(self, peer_id: str) -> dict:
        """Remove a peer from config. Stops its processor first."""
        with self._lock:
            cfg = self.cfg
        peer = get_peer_by_id(cfg, peer_id)
        if not peer:
            return {'error': f'Peer {peer_id} not found'}
        if peer.get('url') is None:
            return {'error': 'Cannot remove the primary peer'}
        # Stop processor on the peer
        url = peer['url']
        try:
            stop_peer_processor(url)
        except Exception as e:
            log.warning('Failed to stop peer %s before removal: %s', peer_id, e)
        # Remove from config
        with self._lock:
            cfg['peers'] = [p for p in cfg.get('peers', []) if p['id'] != peer_id]
            # Pass removed_ids so the merge-with-disk in save_peers_config
            # can't resurrect this peer from a stale on-disk snapshot
            # written by a different worker between our load and save.
            save_peers_config(cfg, removed_ids={peer_id})
            # Clear from state
            self.state.get('peer_bandwidth', {}).pop(peer_id, None)
            if self.state.get('active_peer') == peer_id:
                self.state['active_peer'] = None
            save_director_state(self.state)
            self.cfg = cfg
        # Remove from peer_urls.txt
        peer_urls_path = DATA_DIR / 'peer_urls.txt'
        if peer_urls_path.exists():
            try:
                urls = {u.strip() for u in peer_urls_path.read_text().splitlines() if u.strip()}
                urls.discard(url)
                peer_urls_path.write_text('\n'.join(sorted(urls)) + '\n' if urls else '')
            except Exception:
                pass
        log.info('Removed peer %s (%s)', peer_id, url)
        return {'status': 'removed', 'peer_id': peer_id}

    def _update_bandwidth(self):
        """Refresh bandwidth data for all peers in parallel.

        A single wedged peer must never delay the loop.  Each remote peer
        is polled in a worker thread with short tuple-timeouts (connect=3,
        read=8).  After BANDWIDTH_BACKOFF_THRESHOLD consecutive failures
        we skip polling that peer for BANDWIDTH_BACKOFF_SECONDS so the
        loop stays fast even when half the fleet is dead.
        """
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
        import time as _t

        peers = list(self.cfg.get('peers', []))
        # Local first — cheap, in-process.
        for peer in peers:
            if peer.get('url') is None:
                bw = get_local_bandwidth()
                with self._lock:
                    self.state.setdefault('peer_bandwidth', {})[peer['id']] = bw

        remote = [p for p in peers if p.get('url')]
        if not remote:
            return

        with self._lock:
            backoff = self.state.setdefault('_bandwidth_backoff', {})
            misses = self.state.setdefault('_bandwidth_misses', {})
        now = _t.time()

        # Filter peers currently in backoff so the loop stays fast.
        to_poll = []
        for p in remote:
            pid = p['id']
            until = backoff.get(pid, 0)
            if until > now:
                continue
            to_poll.append(p)

        results: dict[str, dict] = {}
        if to_poll:
            poll_started = _t.time()
            try:
                with ThreadPoolExecutor(
                    max_workers=BANDWIDTH_POLL_CONCURRENCY,
                    thread_name_prefix='dir-bw'
                ) as ex:
                    futs = {ex.submit(get_peer_bandwidth, p['url']): p for p in to_poll}
                    # Hard wall-clock budget: even if a worker hangs we
                    # don't wait more than this for the whole batch.
                    deadline = poll_started + 15
                    for fut, p in list(futs.items()):
                        try:
                            results[p['id']] = fut.result(
                                timeout=max(0.1, deadline - _t.time())
                            )
                        except FTimeout:
                            results[p['id']] = {
                                'error': 'bandwidth poll exceeded 15 s budget',
                                'used_bytes': 0, 'used_gb': 0,
                                'remaining_gb': BANDWIDTH_BUDGET_GB,
                                'pct_used': 0, 'estimated': True,
                            }
                        except Exception as e:
                            results[p['id']] = {
                                'error': str(e),
                                'used_bytes': 0, 'used_gb': 0,
                                'remaining_gb': BANDWIDTH_BUDGET_GB,
                                'pct_used': 0, 'estimated': True,
                            }
            except Exception:
                log.exception('bandwidth pool error')
            log.debug('bandwidth poll: %d peers in %.1fs',
                     len(to_poll), _t.time() - poll_started)

        # Apply results + maintain backoff bookkeeping.
        with self._lock:
            self.state.setdefault('peer_bandwidth', {})
            for p in remote:
                pid = p['id']
                if pid in results:
                    bw = results[pid]
                    self.state['peer_bandwidth'][pid] = bw
                    if bw.get('error'):
                        misses[pid] = misses.get(pid, 0) + 1
                        if misses[pid] >= BANDWIDTH_BACKOFF_THRESHOLD:
                            backoff[pid] = now + BANDWIDTH_BACKOFF_SECONDS
                            log.warning(
                                'Backing off bandwidth poll for %s for %ds '
                                'after %d failures (last err: %s)',
                                pid, BANDWIDTH_BACKOFF_SECONDS,
                                misses[pid], str(bw.get('error'))[:80]
                            )
                    else:
                        misses.pop(pid, None)
                        backoff.pop(pid, None)
                # else: still in backoff, keep last known value
            self.state['_bandwidth_backoff'] = backoff
            self.state['_bandwidth_misses'] = misses
        # Sample canary throughput AFTER the bandwidth state has
        # settled. Cheap (one append per peer) and guarded so the
        # main loop is unaffected if it raises.
        try:
            self._sample_canary_history()
        except Exception:
            log.debug('canary sample failed', exc_info=True)

    # ------------------------------------------------------------------
    # Canary bandwidth history
    # ------------------------------------------------------------------
    def _sample_canary_history(self) -> None:
        """Append a (ts, used_bytes, mode) sample for every enabled peer.

        ``mode`` is 1 when the peer is in frontier mode at sample time
        (active frontier or one of the parallel frontiers), 0 otherwise.
        Tagging samples lets the quality-gate restrict its baseline to
        ticks when the peer was network-bound (BEV reads sustain >>5
        MB/s); pure cache-only / Zenodo-upload baselines are <0.5 MB/s
        and can never satisfy the network-grade gate, so without
        tagging the quality observation never fires.

        Legacy 2-element samples are upgraded in-place on read.
        """
        with self._lock:
            cfg = self.cfg
            bw_map = self.state.get('peer_bandwidth') or {}
            hist = self.state.setdefault('canary_history', {})
            now = int(time.time())
            keep_ids = {p['id'] for p in cfg.get('peers', [])
                        if p.get('enabled', True)}
            for stale_pid in list(hist.keys()):
                if stale_pid not in keep_ids:
                    hist.pop(stale_pid, None)
            # Frontier-mode set: active + parallel frontiers as last
            # planned by the director loop. Cheap O(N).
            frontier_ids = set()
            ap = self.state.get('active_peer')
            if ap:
                frontier_ids.add(ap)
            frontier_ids |= set(
                self.state.get('parallel_frontiers_active') or [])
            frontier_ids |= set(
                (self.state.get('frontier_cred_plan') or {}).keys())
            for p in cfg.get('peers', []):
                if not p.get('enabled', True):
                    hist.pop(p['id'], None)
                    continue
                pid = p['id']
                bw = bw_map.get(pid) or {}
                used = bw.get('used_bytes')
                if used is None or bw.get('error'):
                    continue
                series = hist.setdefault(pid, [])
                mode = 1 if pid in frontier_ids else 0
                # Skip duplicate samples (peer cached, no fresh data).
                if series and series[-1][0] == now:
                    continue
                series.append([now, int(used), mode])
                # Trim to ring size.
                if len(series) > CANARY_HISTORY_MAX:
                    del series[: len(series) - CANARY_HISTORY_MAX]

    def _canary_throughput(self, pid: str, window_s: int,
                           *, frontier_only: bool = False) -> dict | None:
        """Return mean throughput (bytes/s) over the trailing window.

        Returns dict with ``samples``, ``span_s``, ``delta_bytes``,
        ``rate_bps``, or None if not enough data.

        ``frontier_only=True`` restricts the trailing window to samples
        captured while the peer was in frontier mode. Used by the
        quality gate so the fleet wall isn't polluted by Zenodo-bound
        cache-only baselines. Returns None if fewer than 2 such samples
        in the window.
        """
        series = (self.state.get('canary_history') or {}).get(pid) or []
        if len(series) < 2:
            return None
        cutoff = series[-1][0] - window_s
        # Find first sample >= cutoff.
        i = 0
        for i in range(len(series)):
            if series[i][0] >= cutoff:
                break
        window = series[i:]
        if frontier_only:
            # Sample is [ts, used, mode]; legacy 2-tuples treated as
            # cache-mode (mode=0) so legacy data can never falsely
            # satisfy the quality gate.
            window = [s for s in window
                      if len(s) >= 3 and s[2] == 1]
        if len(window) < 2:
            return None
        span = window[-1][0] - window[0][0]
        if span <= 0:
            return None
        delta = max(0, window[-1][1] - window[0][1])
        return {
            'samples': len(window),
            'span_s': span,
            'delta_bytes': delta,
            'rate_bps': delta / span,
        }

    def _canary_summary(self, peer: dict, state: dict) -> dict | None:
        """Compact summary for the status payload.

        Every enabled peer is sampled (canary-by-default). ``budget_gb``
        on the peer flags the *override* canaries probing past the
        global cap, surfaced to the dashboard as ``override:true``.
        Returns None only for peers with no samples yet so the
        dashboard can keep using ``peer.canary`` as both a presence
        check and a data source for the badge.
        """
        pid = peer['id']
        if not (state.get('canary_history') or {}).get(pid):
            return None
        series = (state.get('canary_history') or {}).get(pid) or []
        base = self._canary_throughput(pid, CANARY_BASELINE_WINDOW_S)
        recent = self._canary_throughput(pid, CANARY_RECENT_WINDOW_S)
        # Last ~30 sample points for sparkline rendering.
        spark = [
            {'t': int(s[0]), 'used_bytes': int(s[1])}
            for s in series[-60:]
        ]
        return {
            'budget_gb': peer.get('budget_gb'),
            'override': peer.get('budget_gb') is not None,
            'observed_cap_gb': peer.get('observed_cap_gb'),
            'samples': len(series),
            'baseline_mbps': round(base['rate_bps'] / 1e6, 3) if base else None,
            'recent_mbps': round(recent['rate_bps'] / 1e6, 3) if recent else None,
            'ratio': (round(recent['rate_bps'] / base['rate_bps'], 3)
                       if base and recent and base['rate_bps'] > 0 else None),
            'noise_score': round(_peer_noise_score(pid, state), 3),
            'park_thresholds': {
                'slowdown_ratio': CANARY_SLOWDOWN_RATIO,
                'noise_score': CANARY_NOISE_PARK_THRESHOLD,
                'cooldown_s': CANARY_PARK_COOLDOWN_S,
            },
            'history': spark,
            'notes': (peer.get('canary_notes') or [])[-5:],
        }

    @staticmethod
    def _fleet_bw_summary(peers_status: list[dict]) -> dict:
        """Aggregate bandwidth view that doesn't lie about the wall.

        Replaces the old "GB left of N\u00a0GB" aggregate (which assumed
        all peers share one cycle anchor and one budget). Reports:

          * ``used_gb``: actual bytes consumed across enabled peers
            this cycle (the only number we measure, not guess).
          * ``budget_gb``: nominal sum of effective per-peer budgets,
            for context only — NOT a reliable wall.
          * ``parked``: peers currently in not_before cooldown.
          * ``observed_cap_gb_min/median``: distilled from
            ``observed_cap_gb`` across peers that have hit the wall.
          * ``next_renew_in_days``: minimum days-to-renew across peers
            (per-peer renew_day) so the dashboard shows the soonest
            upcoming reset rather than averaging.
        """
        from datetime import datetime as _dt, timezone as _tz
        now = _dt.now(_tz.utc)
        used = 0.0
        budget = 0.0
        n_total = 0
        n_parked = 0
        n_with_cap = 0
        caps: list[float] = []
        soonest_days = None
        for p in peers_status:
            if not p.get('enabled'):
                continue
            n_total += 1
            bw = p.get('bandwidth') or {}
            if isinstance(bw.get('used_gb'), (int, float)):
                used += float(bw['used_gb'])
            eb = bw.get('effective_budget_gb') or bw.get('budget_gb')
            if isinstance(eb, (int, float)):
                budget += float(eb)
            if p.get('scheduled'):
                n_parked += 1
            cap = p.get('observed_cap_gb')
            cap_quality = bool(p.get('observed_cap_quality'))
            if isinstance(cap, (int, float)) and cap_quality:
                n_with_cap += 1
                caps.append(float(cap))
            rd = p.get('renew_day')
            if isinstance(rd, int) and 1 <= rd <= 28:
                # next occurrence of rd this/next month
                try:
                    nxt = now.replace(day=rd, hour=0, minute=0,
                                      second=0, microsecond=0)
                    if nxt <= now:
                        m = now.month + 1
                        y = now.year + (1 if m > 12 else 0)
                        m = ((m - 1) % 12) + 1
                        nxt = nxt.replace(year=y, month=m)
                    days = max(0, int((nxt - now).total_seconds() // 86400))
                    if soonest_days is None or days < soonest_days:
                        soonest_days = days
                except ValueError:
                    pass
        # Only publish a fleet wall once we have enough QUALITY
        # datapoints to mean something. Until then the dashboard sees
        # ``observed_cap_gb_count`` so it can show 'wall=? (gathering)'.
        cap_min = None
        cap_med = None
        if caps and len(caps) >= FLEET_WALL_MIN_QUALITY_OBS:
            cap_min = min(caps)
            scaps = sorted(caps)
            cap_med = scaps[len(scaps) // 2]
        return {
            'peers_enabled': n_total,
            'peers_parked': n_parked,
            'used_gb': round(used, 2),
            'budget_gb_nominal': round(budget, 2),
            'observed_cap_gb_count': n_with_cap,
            'observed_cap_gb_min': round(cap_min, 2) if cap_min is not None else None,
            'observed_cap_gb_median': round(cap_med, 2) if cap_med is not None else None,
            'next_renew_in_days': soonest_days,
        }

    def _enforce_primary_park(self) -> None:
        """Belt-and-braces: keep the primary peer parked.

        The primary VM is the public-facing dashboard host (DNS, search
        index, share storage, Zenodo lock broker). It must NEVER carry
        frontier or cache-only processing load — even briefly during a
        rotation — because that would steal CPU/IO from request
        handling. Operationally we keep it pinned ``idle`` and
        ``not_before`` far in the future.

        This enforcer runs every director tick and:
          * sets ``pinned_role='idle'`` if missing
          * extends ``not_before`` to >= 30 days from now if shorter or
            absent
          * if the primary somehow ended up the active_peer (race),
            demotes it.

        Cheap: O(1), no I/O unless cfg actually changed.
        """
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        with self._lock:
            cfg = self.cfg
            primary = None
            for p in cfg.get('peers', []):
                if p.get('id') == 'primary':
                    primary = p
                    break
            if primary is None:
                return
            changed = False
            cur_role = (primary.get('pinned_role') or '').strip().lower()
            if cur_role not in ('idle', 'off', 'pause', 'paused', 'parked'):
                primary['pinned_role'] = 'idle'
                changed = True
            # Primary is intentionally pinned far into the future:
            # the public dashboard host must never carry processing
            # load. Floor at 2027-01-01 (well past any conceivable
            # exe.dev billing cycle) and extend if we ever drift
            # below.
            target = _dt(2027, 1, 1, tzinfo=_tz.utc)
            nb = primary.get('not_before')
            extend = False
            if not nb:
                extend = True
            else:
                try:
                    cur = _dt.fromisoformat(nb)
                    if cur.tzinfo is None:
                        cur = cur.replace(tzinfo=_tz.utc)
                    if cur < target:
                        extend = True
                except (TypeError, ValueError):
                    extend = True
            if extend:
                primary['not_before'] = target.isoformat()
                changed = True
            if self.state.get('active_peer') == 'primary':
                log.warning('Primary somehow set as active_peer — demoting')
                self.state['active_peer'] = None
        if changed:
            try:
                save_peers_config(self.cfg)
                log.info('Primary park enforced: pinned_role=%s not_before=%s',
                         primary.get('pinned_role'), primary.get('not_before'))
            except Exception:
                log.exception('Primary park enforce: save_peers_config failed')

    def _release_unverified_bw_parks(self) -> None:
        """One-shot cleanup: clear not_before from peers we parked on
        a budget guess (no observed_cap_gb) so they rejoin the rotation.

        Runs once per process under a state flag so we don't churn.
        Only releases parks tagged by ``_park_peer_until_renewal``
        (event=='park_until_renewal' in canary_notes) where the peer
        never earned an ``observed_cap_gb``. Hand-set ``not_before``
        values (e.g. primary 2027-01-01, manual cooldowns) are
        preserved by looking for the matching note.
        """
        from datetime import datetime as _dt, timezone as _tz
        with self._lock:
            if self.state.get('_unverified_bw_parks_released'):
                return
            cfg = self.cfg
            changed = False
            released: list[str] = []
            for p in cfg.get('peers', []):
                pid = p.get('id')
                if pid in ('primary',):
                    continue  # primary is enforced separately
                if not p.get('enabled', True):
                    continue
                if p.get('observed_cap_gb') is not None:
                    continue  # genuine cap evidence; keep park
                nb = p.get('not_before')
                if not nb:
                    continue
                # Did *we* set this via park-until-renewal recently?
                notes = p.get('canary_notes') or []
                ours = False
                for n in reversed(notes[-8:]):
                    if not isinstance(n, dict):
                        continue
                    if n.get('event') == 'park_until_renewal' and \
                            n.get('cooldown_until') == nb:
                        ours = True
                        break
                if not ours:
                    continue
                p.pop('not_before', None)
                notes.append({
                    'at': _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'event': 'park_released',
                    'reason': 'no observed_cap_gb evidence; '
                              'canary-by-default policy',
                })
                if len(notes) > 32:
                    del notes[: len(notes) - 32]
                changed = True
                released.append(pid)
            if changed:
                try:
                    save_peers_config(self.cfg)
                except Exception:
                    log.exception('release_unverified_bw_parks: save failed')
            self.state['_unverified_bw_parks_released'] = True
        if released:
            log.warning('Released unverified BW parks on %d peers '
                        '(no observed_cap_gb evidence): %s',
                        len(released), ','.join(released))
            try:
                _emit_director_event(
                    f'released {len(released)} unverified BW parks '
                    f'(canary-by-default): {",".join(released)}',
                    peer='director', level='warning')
            except Exception:
                pass

    # Short rolling cooldown for role-based parks (director, shadow).
    # Refreshed every director tick while the role is held, so the
    # peer naturally rejoins rotation within this window after a
    # handover. Long enough to survive HA flapping and the next tick.
    _ROLE_PARK_REFRESH_S = 2 * 3600

    def _enforce_director_self_park(self) -> None:
        """Keep current director + shadow out of the processing pool.

        Why park them:
          * Director pays a recurring fleet-fanout BW cost (status
            polls, log merge, credential probes). Stacking frontier
            or cache-only work on top is exactly how a director
            burns through bandwidth faster than peers.
          * Shadow must be ready to take over instantly; an in-flight
            KG would have to be aborted on promotion.

        Implementation: a short ROLLING cooldown (2 h, refreshed every
        tick) instead of a long +30 d stamp. As soon as a peer stops
        being director / shadow, its ``not_before`` naturally expires
        within 2 h with NO explicit release — fixes the bug where
        ex-directors (at55, at52, at66) remained parked for 30 d
        because the self-park ran only when the peer was still
        director.

        Primary is exempt (handled by ``_enforce_primary_park``
        with the 2027 floor).
        """
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        try:
            import director_ha as _dha  # local import to avoid cycle
            if not _dha.IS_DIRECTOR_FLAG.exists():
                return
            me = _dha.self_id()
        except Exception:
            return
        if not me:
            return
        shadow = (self.state.get('shadow_peer') or '').strip()
        target = _dt.now(_tz.utc) + _td(seconds=self._ROLE_PARK_REFRESH_S)
        targets = {}
        if me != 'primary':
            targets[me] = 'director'
        if shadow and shadow != 'primary' and shadow != me:
            targets[shadow] = 'shadow'
        # Note: empty targets is fine and falls through to the cleanup
        # loop below — we still want to release ex-directors / ex-shadows
        # whose pinned_role was set by a previous tick but never cleared.
        with self._lock:
            cfg = self.cfg
            changed = False
            # First pass: clear stale role-parks on peers that are no
            # longer director/shadow. Bug fix: previously this method
            # only stamped roles; pinned_role='idle' accumulated forever
            # on every ex-shadow, leaving them permanently excluded
            # from rotation even after not_before rolled off.
            for p in cfg.get('peers', []):
                pid = p.get('id')
                if pid in targets or pid == 'primary':
                    continue
                notes = p.get('canary_notes') or []
                last_rp = next((n for n in reversed(notes[-12:])
                                if isinstance(n, dict)
                                and n.get('event') == 'role_park'), None)
                if not last_rp:
                    continue  # role wasn't set by us — leave it alone
                # Was this peer parked by us, but is no longer in the
                # active director/shadow set? Clear our stamp.
                cur_role = (p.get('pinned_role') or '').strip().lower()
                if cur_role != 'idle':
                    continue
                p['pinned_role'] = None
                # Only drop not_before if it matches a role_park stamp
                # (don't shrink canary/manual cooldowns).
                nb = p.get('not_before')
                if nb and last_rp.get('cooldown_until') == nb:
                    p.pop('not_before', None)
                notes.append({
                    'at': _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'event': 'park_released',
                    'reason': 'role_park auto-clear: peer no longer '
                              f'{last_rp.get("role", "director/shadow")}',
                })
                if len(notes) > 32:
                    del notes[: len(notes) - 32]
                p['canary_notes'] = notes
                changed = True
            if not targets:
                # No-op fast path — the outer save block below handles
                # persistence if the cleanup pass made any changes.
                pass
            for p in cfg.get('peers', []):
                pid = p.get('id')
                if pid not in targets:
                    continue
                role_tag = targets[pid]
                cur_role = (p.get('pinned_role') or '').strip().lower()
                if cur_role not in ('idle', 'off', 'pause', 'paused', 'parked'):
                    p['pinned_role'] = 'idle'
                    changed = True
                nb = p.get('not_before')
                extend = False
                if not nb:
                    extend = True
                else:
                    try:
                        cur = _dt.fromisoformat(nb)
                        if cur.tzinfo is None:
                            cur = cur.replace(tzinfo=_tz.utc)
                        # Don't shrink a longer (canary / manual) park.
                        # Only refresh if our rolling target is later.
                        if cur < target:
                            extend = True
                    except (TypeError, ValueError):
                        extend = True
                if extend:
                    p['not_before'] = target.isoformat()
                    notes = p.setdefault('canary_notes', [])
                    notes.append({
                        'at': _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'event': 'role_park',
                        'role': role_tag,
                        'cooldown_until': p['not_before'],
                    })
                    if len(notes) > 32:
                        del notes[: len(notes) - 32]
                    changed = True
            if self.state.get('active_peer') in targets:
                log.warning(
                    'Role-parked peer %s set as active_peer — demoting',
                    self.state.get('active_peer'),
                )
                self.state['active_peer'] = None
        if changed:
            try:
                save_peers_config(self.cfg)
            except Exception:
                log.exception('role-park: save_peers_config failed')
        # Stop any local processor running on the director itself.
        if me in targets:
            try:
                r = requests.get(
                    'http://127.0.0.1:8000/api/v1/processing/status',
                    timeout=5,
                )
                running = bool(r.ok and (r.json() or {}).get('running'))
            except Exception:
                running = False
            if running:
                log.warning(
                    'Director %s has local processor running — issuing '
                    'graceful stop', me,
                )
                try:
                    stop_peer_processor(None, graceful=True)
                    _emit_director_event(
                        f'director self-park: {me} stopping local '
                        'processor (director must not carry load)',
                        peer=me, level='warning',
                    )
                except Exception:
                    log.exception('director self-park: local stop failed')

    def _park_peer_until_renewal(self, peer: dict, next_renew,
                                 *, remaining_gb: float,
                                 mid_kg: bool, hard: bool) -> None:
        """Park a peer until its next billing-cycle renewal.

        ``hard``: depleted, rotate now — verified hard stop.
        ``mid_kg`` (and not hard): low-water graceful — ask the peer
        to finish the current KG then exit; park record is written
        immediately so the director's idle/stopped branch demotes
        cleanly when the KG completes.
        ``mid_kg=False`` and not hard: idle near low-water — park now
        (no work to finish).
        """
        from datetime import datetime as _dt, timezone as _tz
        pid = peer['id']
        peer_url = peer.get('url')
        # 1) write not_before + audit note under the cfg lock.
        with self._lock:
            live = get_peer_by_id(self.cfg, pid)
            if not live:
                return
            nb = next_renew.isoformat()
            existing = live.get('not_before')
            # Don't shrink an existing later cooldown.
            try:
                if existing:
                    cur = _dt.fromisoformat(existing)
                    if cur.tzinfo is None:
                        cur = cur.replace(tzinfo=_tz.utc)
                    if cur >= next_renew:
                        nb = existing
            except (TypeError, ValueError):
                pass
            live['not_before'] = nb
            notes = live.setdefault('canary_notes', [])
            notes.append({
                'at': _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                'event': 'park_until_renewal',
                'remaining_gb': round(float(remaining_gb), 2),
                'mid_kg': bool(mid_kg),
                'hard': bool(hard),
                'cooldown_until': nb,
            })
            if len(notes) > 32:
                del notes[: len(notes) - 32]
        try:
            save_peers_config(self.cfg)
        except Exception:
            log.exception('park_peer_until_renewal: save_peers_config failed')

        tag = 'depleted' if hard else ('low-water graceful' if mid_kg else 'low-water park')
        log.warning('Peer %s %s (%.2f GB remaining) — not_before=%s%s',
                    pid, tag, remaining_gb, next_renew.isoformat(),
                    ' (mid-KG; sending graceful stop)' if (mid_kg and not hard) else '')
        try:
            _emit_director_event(
                f'park-until-renewal {pid}: {tag}, '
                f'{remaining_gb:.2f} GB remaining, resumes {next_renew.isoformat()}',
                peer=pid, level='warning')
        except Exception:
            pass

        # 2) issue stop. Hard stop = verified; graceful = fire-and-keep-running.
        if peer_url is None:
            return
        try:
            if hard:
                res = safely_stop_peer(peer_url, pid)
                if 'error' in res:
                    log.error('park_peer_until_renewal %s: hard stop failed: %s',
                              pid, res['error'])
            else:
                stop_peer_processor(peer_url, graceful=True)
        except Exception:
            log.exception('park_peer_until_renewal %s: stop failed', pid)

    def _enforce_peer_bandwidth_walls(self) -> None:
        """Park-until-renewal for ANY peer that has hit *its own* wall.

        Canary-by-default philosophy: we do NOT know exe.dev's real
        per-account limits, billing-cycle anchors, or whether 200 GB
        is even the right number. Every peer probes its individual
        ceiling by running until throughput actually collapses. The
        canary slowdown detector then stamps ``observed_cap_gb`` on
        the peer — that's the *only* evidence-based wall.

        This sweep extends ``_park_peer_until_renewal`` to non-active
        peers, but ONLY when we have measured evidence:

          * ``observed_cap_gb`` set AND peer's used_gb ≥ cap → the
            peer has revisited its known wall → park.
          * remaining < HARD_DEPLETED_GB AND cap evidence → hard
            stop (depleted by its own observation).

        Peers without an ``observed_cap_gb`` are explicitly NOT
        parked, no matter how far past the 95 GB nominal budget they
        run. The 95 GB number is a guess; the canary slowdown gate
        is the real authority. This avoids stranding the fleet on
        an incorrect billing-cycle assumption.
        """
        with self._lock:
            cfg = self.cfg.copy()
            state_copy = self.state.copy()
        active_id = state_copy.get('active_peer')
        bw_map = state_copy.get('peer_bandwidth') or {}
        for p in list(cfg.get('peers', [])):
            if not p.get('enabled', True):
                continue
            pid = p['id']
            if pid == active_id:
                continue  # handled by _check_and_switch
            if _peer_is_scheduled(p):
                continue  # already parked
            bw = bw_map.get(pid) or {}
            used = bw.get('used_bytes')
            if used is None:
                continue
            used_gb = used / (1024 ** 3)
            # Evidence requirement: peer must have a quality
            # observed_cap_gb (from a network-grade slowdown) AND
            # its current used must be at/past that cap.
            cap = p.get('observed_cap_gb')
            if not isinstance(cap, (int, float)):
                continue
            if used_gb < float(cap):
                continue
            # Compute remaining vs *cap* (the observed wall), not the
            # nominal 95 GB budget.
            remaining = float(cap) - used_gb  # ≤ 0 by gate above
            # Determine mid-KG via cached peer status (no extra HTTP).
            proc_state_now = 'unknown'
            url = p.get('url')
            if url is None:
                # Local peer; read progress directly via get_peer_status.
                try:
                    proc_state_now = get_peer_status(None, pid).get(
                        'state', 'unknown')
                except Exception:
                    proc_state_now = 'unknown'
            else:
                cached = _PEER_STATUS_CACHE.get(url)
                if cached and (time.time() - cached[0]) < _PEER_STATUS_CACHE_TTL:
                    proc_state_now = cached[1].get('state', 'unknown')
                else:
                    pushed = get_pushed_status(pid)
                    if pushed is not None:
                        proc_state_now = (pushed.get('status') or {}
                                          ).get('state', 'unknown')
            mid_kg = proc_state_now in ('running', 'processing')
            hard = remaining < BANDWIDTH_HARD_DEPLETED_GB
            try:
                next_renew = _peer_next_renew(p, cfg)
            except Exception:
                log.exception('peer-bw-wall %s: next_renew failed', pid)
                continue
            try:
                self._park_peer_until_renewal(
                    p, next_renew, remaining_gb=remaining,
                    mid_kg=mid_kg, hard=hard,
                )
            except Exception:
                log.exception('peer-bw-wall %s: park failed', pid)

    def _check_canary_health(self) -> None:
        """Park peers whose throughput collapsed (whole fleet) or whose
        warning noise spiked (canary-override peers only).

        Slowdown park: applies to every enabled peer — the canonical
        signal for exe.dev shaping. We persist ``observed_cap_gb`` on
        the peer the moment we trip, so we learn each peer's wall.

        Noise park: applies only to peers with an explicit
        ``budget_gb`` override (intentional canaries). Generic peers
        get a noisy score for transient upstream issues that the
        load-shifter already handles — yanking them out of rotation
        for 6 h would be too aggressive.

        Sets ``not_before`` so the director's normal scheduling logic
        stops sending the peer work; existing graceful-stop /
        not_before paths in ``_check_and_switch`` handle the cleanup.
        """
        with self._lock:
            cfg = self.cfg
            state = self.state
            peers = list(cfg.get('peers', []))
            # Streaks: peer_id -> {since: epoch, last_ratio: float}
            streaks = state.setdefault('canary_slowdown_streaks', {})
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        cfg_changed = False
        now_ts = time.time()

        # First pass: classify each peer's current state without acting,
        # update streaks, and tally fleet-wide concurrent slowdown so
        # the second pass can suppress quality flags during fleet-wide
        # upstream events (Zenodo / BEV outages).
        per_peer: list[dict] = []
        n_with_canary = 0
        n_in_slowdown = 0
        for p in peers:
            if not p.get('enabled', True):
                continue
            pid = p['id']
            base = self._canary_throughput(pid, CANARY_BASELINE_WINDOW_S)
            recent = self._canary_throughput(pid, CANARY_RECENT_WINDOW_S)
            usable = (base and recent
                      and base['samples'] >= CANARY_BASELINE_MIN_SAMPLES
                      and base['delta_bytes'] >= CANARY_MIN_BYTES_FOR_PARK
                      and base['rate_bps'] > 0)
            ratio = base_mbps = recent_mbps = None
            if usable:
                ratio = recent['rate_bps'] / base['rate_bps']
                base_mbps = base['rate_bps'] / 1e6
                recent_mbps = recent['rate_bps'] / 1e6
                n_with_canary += 1
                if ratio < CANARY_SLOWDOWN_RATIO:
                    n_in_slowdown += 1
                    # Start / continue streak.
                    s = streaks.setdefault(pid, {'since': now_ts})
                    s['last_ratio'] = ratio
                    s['last_ts'] = now_ts
                elif ratio >= CANARY_SLOWDOWN_RECOVERY_RATIO:
                    streaks.pop(pid, None)
            per_peer.append({
                'peer': p,
                'pid': pid,
                'usable': usable,
                'ratio': ratio,
                'base_mbps': base_mbps,
                'recent_mbps': recent_mbps,
            })
        # Trim streaks for peers no longer enabled.
        live_ids = {pp['pid'] for pp in per_peer}
        for stale_id in list(streaks.keys()):
            if stale_id not in live_ids:
                streaks.pop(stale_id, None)

        # Cross-peer correlation: are we in a fleet-wide upstream event?
        # If so, no peer parked this tick can earn a quality observation.
        fleet_wide = False
        if n_with_canary >= 4:
            frac = n_in_slowdown / n_with_canary
            if frac >= FLEET_CONCURRENT_SLOWDOWN_FRAC:
                fleet_wide = True
                log.info('Canary: fleet-wide slowdown detected '
                         '(%d/%d peers ≥ %.0f%%) — suppressing quality '
                         'observations this tick',
                         n_in_slowdown, n_with_canary,
                         FLEET_CONCURRENT_SLOWDOWN_FRAC * 100)
        # Stash for status payload.
        state['canary_fleet_slowdown'] = {
            'with_canary': n_with_canary,
            'in_slowdown': n_in_slowdown,
            'fleet_wide': fleet_wide,
            'ts': now_ts,
        }

        # Second pass: act.
        for entry in per_peer:
            p = entry['peer']
            pid = entry['pid']
            # Already parked? skip.
            if _peer_is_scheduled(p):
                continue
            reasons: list[str] = []
            slowdown_tripped = False
            quality_obs = False
            if entry['usable']:
                ratio = entry['ratio']
                base_mbps = entry['base_mbps']
                recent_mbps = entry['recent_mbps']
                if ratio < CANARY_SLOWDOWN_RATIO:
                    slowdown_tripped = True
                    streak = streaks.get(pid) or {}
                    streak_age = max(0.0, now_ts - float(streak.get('since') or now_ts))
                    # Network-grade gate must look at FRONTIER-mode
                    # samples only — cache-only baselines are
                    # Zenodo-bound (<0.5 MB/s) and can never satisfy
                    # the ≥5 MB/s threshold. Without this, the quality
                    # path is dead for the steady-state fleet workload
                    # mix and the fleet wall never converges.
                    fb = self._canary_throughput(
                        pid, CANARY_BASELINE_WINDOW_S, frontier_only=True)
                    fr = self._canary_throughput(
                        pid, CANARY_RECENT_WINDOW_S, frontier_only=True)
                    fbase_mbps = (fb['rate_bps'] / 1e6) if fb else None
                    frecent_mbps = (fr['rate_bps'] / 1e6) if fr else None
                    network_grade = (
                        fbase_mbps is not None
                        and frecent_mbps is not None
                        and fbase_mbps >= CANARY_BASELINE_NETWORK_MBPS
                        and frecent_mbps <= CANARY_RECENT_PARKED_MBPS
                    )
                    persistent = streak_age >= CANARY_QUALITY_PERSIST_S
                    quality_obs = (
                        network_grade and persistent and not fleet_wide
                    )
                    qtag = ' [QUALITY]' if quality_obs else ' [soft'
                    if not quality_obs:
                        why = []
                        if not network_grade:
                            why.append('not network-grade')
                        if not persistent:
                            why.append(f'streak {int(streak_age)}s<{CANARY_QUALITY_PERSIST_S}s')
                        if fleet_wide:
                            why.append('fleet-wide event')
                        qtag = ' [soft: ' + ', '.join(why) + ']'
                    reasons.append(
                        f'throughput collapsed: '
                        f'recent={recent_mbps:.2f} MB/s '
                        f'vs baseline {base_mbps:.2f} MB/s '
                        f'(ratio={ratio:.2f})' + qtag
                    )
            # Noise-park: stricter, canary-override peers only.
            is_override_canary = p.get('budget_gb') is not None
            if is_override_canary:
                noise = _peer_noise_score(pid, state)
                if noise >= CANARY_NOISE_PARK_THRESHOLD:
                    reasons.append(f'noise_score={noise:.2f}')
            if not reasons:
                continue
            # Park: write not_before, persist, send graceful stop.
            # Quality-grade parks (real shaping observation) earn the
            # full 6 h cooldown; soft parks get the shorter window so
            # peers don't sit idle for hours after a transient blip.
            cooldown_s = (CANARY_PARK_COOLDOWN_S
                          if quality_obs
                          else CANARY_PARK_COOLDOWN_SOFT_S)
            cooldown = _dt.now(_tz.utc) + _td(seconds=cooldown_s)
            # Used-GB at park time — the observed wall for this peer.
            bw_now = (state.get('peer_bandwidth') or {}).get(pid) or {}
            used_gb_at_park = bw_now.get('used_gb')
            with self._lock:
                # Re-resolve under lock in case cfg was reloaded.
                live = get_peer_by_id(self.cfg, pid)
                if not live or not live.get('enabled', True):
                    continue
                if _peer_is_scheduled(live):
                    continue
                live['not_before'] = cooldown.isoformat()
                # Persist the observed shaping wall ONLY when the
                # trigger met the stricter network-grade gate. Soft
                # parks (low baseline, plausibly Zenodo-bound peers)
                # don't pollute the fleet wall estimate.
                if (quality_obs and slowdown_tripped
                        and isinstance(used_gb_at_park, (int, float))):
                    prev = live.get('observed_cap_gb')
                    if prev is None or used_gb_at_park < float(prev):
                        live['observed_cap_gb'] = round(float(used_gb_at_park), 2)
                    live['observed_cap_at'] = _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                    live['observed_cap_quality'] = True
                notes = live.setdefault('canary_notes', [])
                note: dict = {
                    'at': _dt.now(_tz.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'event': 'auto_park',
                    'reasons': reasons,
                    'cooldown_until': cooldown.isoformat(),
                }
                if slowdown_tripped and isinstance(used_gb_at_park, (int, float)):
                    note['used_gb_at_park'] = round(float(used_gb_at_park), 2)
                    note['quality'] = bool(quality_obs)
                notes.append(note)
                # Cap notes ring so peers.json doesn't grow unbounded.
                if len(notes) > 32:
                    del notes[: len(notes) - 32]
                cfg_changed = True
                peer_url = live.get('url')
            tag = 'canary park' if is_override_canary else 'auto-park'
            log.warning('%s %s: %s — not_before=%s',
                        tag, pid, '; '.join(reasons), cooldown.isoformat())
            try:
                _emit_director_event(
                    f'{tag} {pid}: {"; ".join(reasons)} (cooldown until {cooldown.isoformat()})',
                    peer=pid, level='warning')
            except Exception:
                pass
            try:
                if peer_url:
                    stop_peer_processor(peer_url, graceful=True)
            except Exception as e:
                log.warning('%s %s: graceful stop failed: %s', tag, pid, e)
        if cfg_changed:
            try:
                save_peers_config(self.cfg)
            except Exception:
                log.exception('canary park: save_peers_config failed')

    def _check_and_switch(self):
        """Check if we need to switch the active peer."""
        with self._lock:
            mode = self.state.get('mode', 'auto')
            if mode == 'paused':
                return
            if mode == 'manual':
                # In manual mode, just ensure the designated peer is running
                active_id = self.state.get('active_peer')
                if active_id:
                    peer = get_peer_by_id(self.cfg, active_id)
                    if peer:
                        status = get_peer_status(peer.get('url'))
                        if status.get('state') in ('idle', 'stopped', 'unknown'):
                            excl = _excluded_kgs(self.cfg, exclude_peer_id=active_id)
                            log.info('Manual mode: starting processor on %s%s', active_id,
                                     (' — excluding ' + ','.join(sorted(excl))) if excl else '')
                            start_peer_processor(peer.get('url'), exclude_kgs=excl)
                return

            # Auto mode — check bandwidth and switch if needed
            active_id = self.state.get('active_peer')
            cfg = self.cfg.copy()
            state_copy = self.state.copy()

        budget_bytes = cfg.get('budget_gb', BANDWIDTH_BUDGET_GB) * (1024 ** 3)

        # Pre-empt the active peer when a different peer's reservation
        # becomes ready (cooldown lifted, KG still pending). The held KG
        # has tile checkpoints on the holder, so it finishes much faster
        # there than starting fresh on the substitute. We only pre-empt
        # between KGs (never mid-KG) by waiting for idle/stopped state.
        if active_id:
            holder = _ready_reservation_holder(cfg, state_copy)
            if holder and holder != active_id:
                active_peer_cfg = get_peer_by_id(cfg, active_id)
                ps = get_peer_status(active_peer_cfg.get('url')) if active_peer_cfg else {}
                proc_state = ps.get('state', 'unknown')
                if proc_state in ('idle', 'stopped'):
                    log.info('Reservation ready on %s — pre-empting %s (between KGs)',
                             holder, active_id)
                    with self._lock:
                        # Clear graceful-stop flag before nilling active_id
                        self.state.get('graceful_stop_sent', {}).pop(active_id, None)
                        self.state['active_peer'] = None
                        active_id = None
                else:
                    # Mid-KG: send graceful stop once. The flag prevents
                    # us from re-sending it on every tick.
                    sent = self.state.get('graceful_stop_sent', {})
                    if sent.get(active_id) != holder:
                        log.info('Reservation ready on %s; sending graceful stop to %s (will exit after current KG)',
                                 holder, active_id)
                        try:
                            stop_peer_processor(active_peer_cfg.get('url'), graceful=True)
                            with self._lock:
                                sent[active_id] = holder
                                self.state['graceful_stop_sent'] = sent
                        except Exception as e:
                            log.warning('Graceful stop on %s failed: %s', active_id, e)
                    else:
                        log.info('Reservation ready on %s; %s still finishing current KG (graceful stop pending)',
                                 holder, active_id)

        # Check if active peer is scheduled (not_before in the future)
        if active_id:
            active_peer_cfg = get_peer_by_id(cfg, active_id)
            if active_peer_cfg and _peer_is_scheduled(active_peer_cfg):
                nb = active_peer_cfg.get('not_before', '?')
                log.info('Peer %s is scheduled not before %s, switching', active_id, nb)
                peer = get_peer_by_id(cfg, active_id)
                if peer:
                    res = safely_stop_peer(peer.get('url'), active_id)
                    if 'error' in res:
                        log.error('Cannot deactivate %s: %s — holding off switch',
                                  active_id, res['error'])
                        return  # leave active_peer as is; retry next tick
                with self._lock:
                    self.state['active_peer'] = None
                    active_id = None

        # Check if active peer is near / over its budget.
        #
        # Three regimes (mirror the canary-park flow so peers waiting
        # for renewal land back in rotation automatically):
        #
        #   remaining < HARD_DEPLETED_GB — rotate even mid-KG; park
        #     until next renewal so the scheduler skips the peer.
        #   remaining < LOW_WATER_GB & mid-KG — send graceful stop so
        #     the in-flight KG finishes and uploads, then the peer
        #     auto-parks on idle (handled in the idle/stopped branch
        #     further down via _peer_is_scheduled).
        #   remaining < LOW_WATER_GB & idle — park (no work to finish).
        if active_id:
            bw = state_copy.get('peer_bandwidth', {}).get(active_id, {})
            used = bw.get('used_bytes', 0)
            active_peer_cfg = get_peer_by_id(cfg, active_id) or {}
            active_budget_bytes = _peer_budget_bytes(active_peer_cfg, cfg)
            remaining_gb = (active_budget_bytes - used) / (1024 ** 3)

            if remaining_gb < BANDWIDTH_LOW_WATER_GB:
                peer = get_peer_by_id(cfg, active_id)
                if peer:
                    next_renew = _peer_next_renew(peer, cfg)
                    proc_state_now = (state_copy.get('peer_status_cache') or {}).get(active_id, {}).get('state')
                    if not proc_state_now:
                        try:
                            proc_state_now = get_peer_status(peer.get('url'), active_id).get('state', 'unknown')
                        except Exception:
                            proc_state_now = 'unknown'
                    mid_kg = proc_state_now == 'running'
                    hard = remaining_gb < BANDWIDTH_HARD_DEPLETED_GB
                    self._park_peer_until_renewal(
                        peer, next_renew, remaining_gb=remaining_gb,
                        mid_kg=mid_kg, hard=hard,
                    )
                    if hard or not mid_kg:
                        # Rotate now: stop verified, park written.
                        with self._lock:
                            if self.state.get('active_peer') == active_id:
                                self.state['active_peer'] = None
                            active_id = None
                    else:
                        # Mid-KG with low-water: graceful stop has
                        # been sent; let the peer keep ticking until
                        # it finishes the KG and reports idle. The
                        # not_before park is already written, so when
                        # the idle/stopped branch fires it'll see
                        # _peer_is_scheduled and demote cleanly.
                        return

        # Check if active peer's processor has stopped unexpectedly
        if active_id:
            peer = get_peer_by_id(cfg, active_id)
            if peer:
                status = get_peer_status(peer.get('url'))
                proc_state = status.get('state', 'unknown')

                # --- Zenodo pause detection ----------------------------
                # paused_zenodo with global=true (rate_limit/auth on the
                # token) blocks every peer using the same token, so we
                # leave the peer paused (probing itself) and don't switch.
                # Non-global (network) is a peer-local issue; cool that
                # peer down for a bit and switch to another peer.
                zinfo = status.get('zenodo_pause') or {}
                # Only act when the parent loop has truly entered the
                # paused_zenodo state (writes that state to progress.json).
                # Mid-KG the subprocess may have written the pause file
                # but the parent is still building light GPKG / JSON; we
                # don't want to abort that work.
                if proc_state == 'paused_zenodo':
                    is_global = bool(zinfo.get('global'))
                    reason = zinfo.get('reason', 'network')
                    if is_global:
                        log.info('Active peer %s: Zenodo paused globally (reason=%s) \u2014 holding all peers',
                                 active_id, reason)
                        # Don't switch \u2014 every peer would hit the same
                        # token-wide issue.  The peer probes Zenodo itself.
                        return
                    # Hold-tendency escalation: peers that repeatedly trip
                    # the Zenodo network cooldown are likely on a flaky
                    # route. Track recent cooldown timestamps and double
                    # the duration on each repeat within the window so we
                    # don't burn the whole day cycling them in/out.
                    now = datetime.now(timezone.utc)
                    cutoff = now - timedelta(hours=HOLD_TENDENCY_WINDOW_HOURS)
                    history = []
                    for p in self.cfg.get('peers', []):
                        if p['id'] == active_id:
                            raw = p.get('zenodo_cooldown_history') or []
                            for ts in raw:
                                try:
                                    t = datetime.fromisoformat(ts)
                                    if t.tzinfo is None:
                                        t = t.replace(tzinfo=timezone.utc)
                                except Exception:
                                    continue
                                if t >= cutoff:
                                    history.append(t)
                            break
                    repeat_count = len(history) + 1  # this incident
                    cd_min = min(
                        ZENODO_NETWORK_COOLDOWN_MIN * (HOLD_TENDENCY_FACTOR ** (repeat_count - 1)),
                        HOLD_TENDENCY_MAX_MIN,
                    )
                    # Honour Retry-After if Zenodo provided one. We use
                    # the larger of our escalated cooldown vs the server's
                    # ask, capped at the hold-tendency ceiling. This is
                    # the cooperative thing to do: the server told us
                    # exactly when to come back, so don't come back
                    # earlier just because our own escalation said so.
                    try:
                        ra_secs = float(zinfo.get('retry_after', 0) or 0)
                    except Exception:
                        ra_secs = 0.0
                    if ra_secs > 0:
                        cd_min = min(
                            max(cd_min, ra_secs / 60.0),
                            HOLD_TENDENCY_MAX_MIN,
                        )
                    if repeat_count > 1:
                        log.warning('Active peer %s: Zenodo network failure '
                                    '(repeat #%d in %dh, retry_after=%.0fs) \u2014 cooling down '
                                    '%.1f min (escalated) and switching',
                                    active_id, repeat_count,
                                    HOLD_TENDENCY_WINDOW_HOURS, ra_secs, cd_min)
                    else:
                        log.warning('Active peer %s: Zenodo network failure (retry_after=%.0fs) \u2014 cooling down %.1f min and switching',
                                    active_id, ra_secs, cd_min)
                    # Apply not_before cooldown so choose_active_peer skips this peer.
                    # Also reserve the in-progress KG so substitute peers skip
                    # it — the cooled peer keeps its tile checkpoints and
                    # will finish quickly when the cooldown lifts.
                    cd_until = now + timedelta(minutes=cd_min)
                    cur_kg = (status.get('current_kg') or {}).get('code')
                    # On escalation (repeat #2+) release the held KG so
                    # another peer can pick it up — the offender clearly
                    # can't reach Zenodo and shouldn't block progress.
                    # First trip: keep the reservation so the peer can
                    # resume from its own tile checkpoints when cooldown
                    # lifts.
                    release_hold = repeat_count > 1
                    for p in self.cfg.get('peers', []):
                        if p['id'] == active_id:
                            p['not_before'] = cd_until.isoformat()
                            if release_hold:
                                released = p.pop('reserved_kg', None)
                                if released or cur_kg:
                                    log.warning('Releasing held KG %s from %s (escalated cooldown)',
                                                released or cur_kg, active_id)
                            elif cur_kg:
                                p['reserved_kg'] = str(cur_kg)
                            # Record this incident; prune older than window.
                            history.append(now)
                            p['zenodo_cooldown_history'] = [
                                t.isoformat() for t in history
                            ]
                            break
                    save_peers_config(self.cfg)
                    # Stop processor on this peer (still uploading retries)
                    res = safely_stop_peer(peer.get('url'), active_id)
                    if 'error' in res:
                        log.error('Cannot stop %s during Zenodo failover: %s',
                                  active_id, res['error'])
                        return
                    with self._lock:
                        self.state['active_peer'] = None
                        active_id = None
                # ------------------------------------------------------
                if active_id and proc_state in ('idle', 'stopped'):
                    # Processor stopped (finished a KG or was stopped externally)
                    # Check if it should continue
                    bw = state_copy.get('peer_bandwidth', {}).get(active_id, {})
                    used = bw.get('used_bytes', 0)
                    remaining_gb = (_peer_budget_bytes(peer, cfg) - used) / (1024 ** 3)
                    # Honour not_before cooldown — if scheduled, demote
                    # the active peer so a different one can be picked.
                    if _peer_is_scheduled(peer):
                        log.info('Active peer %s is scheduled (not_before=%s) \u2014 letting director pick another',
                                 active_id, peer.get('not_before'))
                        with self._lock:
                            self.state['active_peer'] = None
                            active_id = None
                    elif remaining_gb >= BANDWIDTH_HARD_DEPLETED_GB:
                        excl = _excluded_kgs(cfg, exclude_peer_id=active_id)
                        # Re-issue with the current cred/strip plan so
                        # the active peer stays pinned to its slice.
                        plan_cred = (self.state.get('frontier_cred_plan') or {}).get(active_id)
                        plan_strip = (self.state.get('frontier_strip_plan') or {}).get(active_id)
                        if not plan_cred:
                            self._refresh_peer_caps(peer)
                            plan_cred = (self._assign_cred_indices([active_id], cfg)
                                          .get(active_id))
                        if not plan_strip:
                            strips = self._austria_cells()
                            # No prior plan: give all strips to this
                            # lone frontier; orchestrator will trim later.
                            plan_strip = [list(s) for s in strips] if strips else None
                        # Throttle: don't fire restart more than once per
                        # FRONTIER_RESTART_COOLDOWN_S unless the cred/strip
                        # plan or the exclude set actually changed. Without
                        # this we issue ~8 restarts/hour just because the
                        # processor reports idle between KGs (each KG ends
                        # in a clean subprocess exit).
                        plan_fp = [
                            sorted(int(i) for i in (plan_cred or [])),
                            [list(s) for s in (plan_strip or [])],
                            sorted(excl),
                        ]
                        last_restart = self.state.setdefault(
                            '_frontier_restart_log', {})
                        last_entry = last_restart.get(active_id) or {}
                        last_ts = float(last_entry.get('ts') or 0)
                        last_fp = last_entry.get('fp')
                        now_ts = time.time()
                        cooldown = float(
                            cfg.get('frontier_restart_cooldown_s',
                                     FRONTIER_RESTART_COOLDOWN_S))
                        plan_unchanged = (last_fp == plan_fp)
                        if plan_unchanged and (now_ts - last_ts) < cooldown:
                            log.debug(
                                'Skip restart on %s: plan unchanged and '
                                'within cooldown (%.0fs/%.0fs)',
                                active_id, now_ts - last_ts, cooldown)
                        elif self._has_pending_graceful_update(active_id):
                            log.info(
                                'Skip restart on %s: graceful update '
                                'pending (waiting for git pull + srv '
                                'restart on peer)',
                                active_id)
                        else:
                            log.info(
                                'Restarting processor on %s (%.1f GB remaining)%s creds=%s strip=%s',
                                active_id, remaining_gb,
                                (' — excluding ' + ','.join(sorted(excl))) if excl else '',
                                plan_cred, plan_strip,
                            )
                            start_peer_processor(
                                peer.get('url'), exclude_kgs=excl,
                                cred_indices=plan_cred,
                                lat_strips=plan_strip,
                            )
                            last_restart[active_id] = {
                                'ts': now_ts, 'fp': plan_fp,
                            }
                    else:
                        log.info('Peer %s depleted, finding next peer', active_id)
                        with self._lock:
                            self.state['active_peer'] = None
                            active_id = None
                elif proc_state == 'unreachable':
                    # Don't fail over on a single timeout — heavy GPKG builds
                    # can briefly starve gunicorn. Require N consecutive misses.
                    with self._lock:
                        misses = self.state.get('unreachable_count', {})
                        misses[active_id] = misses.get(active_id, 0) + 1
                        self.state['unreachable_count'] = misses
                        n = misses[active_id]
                    if n >= UNREACHABLE_FAILOVER_THRESHOLD:
                        log.warning('Active peer %s unreachable %d times — failing over',
                                    active_id, n)
                        with self._lock:
                            self.state['unreachable_count'].pop(active_id, None)
                            self.state['active_peer'] = None
                            active_id = None
                    else:
                        log.info('Active peer %s unreachable (%d/%d) — waiting',
                                 active_id, n, UNREACHABLE_FAILOVER_THRESHOLD)
                else:
                    # Reachable — clear miss counter
                    with self._lock:
                        self.state.get('unreachable_count', {}).pop(active_id, None)

        # Enforce single-active for the FRONTIER role: stop any non-active
        # peers that are running a non-cache-only processor UNLESS the
        # director has authorised them as a parallel frontier (in
        # ``parallel_frontiers_active`` or as a key of
        # ``frontier_cred_plan``) AND the cred pool supports it.
        # Cache-only peers (no Copernicus credentials) may run in parallel —
        # they're managed by ``_orchestrate_cache_only``.
        #
        # Why both sets? ``parallel_frontiers_active`` is a current-tick
        # observation rebuilt by ``_orchestrate_parallel_frontiers``,
        # which only runs *after* this guard. ``frontier_cred_plan`` is
        # the director's persisted authorisation map. After a srv
        # restart the new worker has not yet ticked the parallel orch,
        # so trusting only ``parallel_frontiers_active`` would hard-stop
        # any inherited parallel frontier (the 2026-05-06 cascade).
        # Mirrors the same union used by ``_orchestrate_cache_only``.
        cred_plan = self.state.get('frontier_cred_plan') or {}
        strip_plan = self.state.get('frontier_strip_plan') or {}
        parallel_ok = set(self.state.get('parallel_frontiers_active') or [])
        parallel_ok |= set(cred_plan.keys())
        # Active peer is implicitly authorised — belt and braces in
        # case the plan map drifted out from under us.
        if active_id:
            parallel_ok.add(active_id)
        if active_id:
            for p in cfg.get('peers', []):
                if p['id'] != active_id and p.get('url') is not None:
                    ps = get_peer_status(p.get('url'))
                    if ps.get('state') in ('running', 'processing'):
                        if ps.get('cache_only'):
                            continue  # benign — doesn't touch credentials
                        if p['id'] in parallel_ok:
                            continue  # authorised parallel frontier
                        log.warning(
                            'Non-active peer %s is running frontier work — '
                            'stopping it (active=%s, '
                            'parallel_frontiers_active=%s, '
                            'cred_plan_keys=%s, strip_plan_keys=%s)',
                            p['id'], active_id,
                            sorted(self.state.get(
                                'parallel_frontiers_active') or []),
                            sorted(cred_plan.keys()),
                            sorted(strip_plan.keys()),
                        )
                        safely_stop_peer(p.get('url'), p['id'])

        # If no active peer, choose one
        if not active_id:
            with self._lock:
                new_peer = choose_active_peer(self.cfg, self.state)
            if new_peer:
                peer = get_peer_by_id(cfg, new_peer)
                if peer:
                    # CRITICAL: verify no other peer is running FRONTIER
                    # work before activating the new one.  Cache-only
                    # peers don't touch Copernicus credentials and may
                    # remain running in parallel.
                    blocked = False
                    # Same union as the single-active guard above:
                    # trust the persisted cred_plan keys, not just the
                    # current-tick observation set.
                    parallel_ok = set(
                        self.state.get('parallel_frontiers_active') or [])
                    parallel_ok |= set(
                        (self.state.get('frontier_cred_plan') or {}).keys())
                    for p in cfg.get('peers', []):
                        if p['id'] == new_peer:
                            continue
                        ps = get_peer_status(p.get('url'))
                        st = ps.get('state', 'unknown')
                        if st in ('running', 'processing') and not ps.get('cache_only'):
                            if p['id'] in parallel_ok:
                                continue
                            log.info('Stopping frontier peer %s before activating %s',
                                     p['id'], new_peer)
                            res = safely_stop_peer(p.get('url'), p['id'])
                            if 'error' in res:
                                log.error('Could not verify %s is stopped: %s — '
                                          'aborting activation of %s '
                                          '(credential safety)',
                                          p['id'], res.get('error'), new_peer)
                                blocked = True
                                break
                    if blocked:
                        return  # try again next tick

                    # Exclude KGs reserved for other cooled-down peers,
                    # so a substitute does useful work on a different KG.
                    excl = _excluded_kgs(cfg, exclude_peer_id=new_peer)
                    # Build credential + lat-strip plan. The active peer
                    # always gets the FIRST slice of creds and the FIRST
                    # Austria-wide lat strip; parallel-frontier peers
                    # (orchestrated separately) get the remaining slices.
                    self._refresh_peer_caps(peer)
                    plan = self._assign_cred_indices([new_peer], cfg)
                    creds_for_peer = plan.get(new_peer)
                    strips = self._austria_cells()
                    # Lone frontier on activation: own all strips.
                    # The parallel orchestrator will redistribute later
                    # when more frontiers come online.
                    strip_for_peer = [list(s) for s in strips] if strips else None
                    log.info(
                        'Activating peer %s%s creds=%s strip=%s',
                        new_peer,
                        (' (excluding reserved KGs: ' + ','.join(sorted(excl)) + ')') if excl else '',
                        creds_for_peer, strip_for_peer,
                    )
                    result = start_peer_processor(
                        peer.get('url'), exclude_kgs=excl,
                        cred_indices=creds_for_peer,
                        lat_strips=strip_for_peer,
                    )
                    log.info('Start result for %s: %s', new_peer, result)
                    with self._lock:
                        self.state['active_peer'] = new_peer
                        self.state['last_switch'] = datetime.now(timezone.utc).isoformat()
                        # Seed the strip plan so the parallel-frontier
                        # orchestrator knows the active peer's strip.
                        sp = dict(self.state.get('frontier_strip_plan') or {})
                        if strip_for_peer:
                            sp[new_peer] = strip_for_peer
                            self.state['frontier_strip_plan'] = sp
                        cp = dict(self.state.get('frontier_cred_plan') or {})
                        if creds_for_peer:
                            cp[new_peer] = list(creds_for_peer)
                            self.state['frontier_cred_plan'] = cp
                        save_director_state(self.state)
            else:
                log.info('No peers with sufficient bandwidth available')

    # ---- cache-only peer orchestration -----------------------------

    def _peer_role(self, peer: dict) -> str:
        """Return the peer's effective role.

        ``pinned_role`` (set by user via UI) overrides ``role``.
        Values: ``frontier`` (default), ``cache_only``, ``idle``
        (never started by the director, regardless of bandwidth/whitelist).
        """
        pinned = (peer.get('pinned_role') or '').strip().lower()
        if pinned in ('idle', 'off', 'pause', 'paused', 'parked'):
            return 'idle'
        if pinned in ('cache_only', 'cache-only', 'cacheonly'):
            return 'cache_only'
        if pinned in ('frontier', 'active'):
            return 'frontier'
        role = (peer.get('role') or '').strip().lower()
        if role in ('cache_only', 'cache-only', 'cacheonly'):
            return 'cache_only'
        return 'frontier'

    # --- Credential pool & capability tracking -----------------------

    # Minimum credentials required to start a new frontier KG. Set by the
    # user as 'min_creds_per_frontier' in peers.json (default 2).
    MIN_CREDS_PER_FRONTIER_DEFAULT = 2

    def _credential_pool(self) -> list[dict]:
        """Return the local credential list (no secrets)."""
        try:
            import copernicus as _cop
            return _cop.list_credentials()
        except Exception as e:
            log.warning('credential_pool: %s', e)
            return []

    def _aggregate_credential_usage(self, cred_pool: list[dict],
                                     peers_list: list[dict]) -> None:
        """Merge per-peer /api/v1/credentials usage into cred_pool entries.

        The processor only runs on peers, so credential traffic is recorded
        there. The director (primary) shows zero unless we sum across the
        fleet. Keyed by client_id (stable across peers; index is identical
        when the builtin list matches but client_id is the safer key).

        Cheap: each peer's /api/v1/credentials returns ~ a few KB. Run in
        parallel and tolerate failures (legacy peers without `usage` are
        simply skipped).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _fetch(peer_url: str | None) -> list[dict]:
            if peer_url is None:
                return []
            try:
                r = requests.get(peer_url.rstrip('/') + '/api/v1/credentials',
                                 timeout=PEER_TIMEOUT_PROBE,
                                 headers=_admin_headers())
                if not r.ok:
                    return []
                return (r.json() or {}).get('credentials') or []
            except Exception:
                return []

        # Initialise aggregate buckets per client_id
        agg: dict[str, dict] = {}
        for c in cred_pool:
            cid = c.get('client_id')
            if not cid:
                continue
            agg[cid] = {'success_7d': 0, 'error_7d': 0, 'rotated_7d': 0,
                        'success': 0, 'error': 0, 'rotated': 0,
                        'last_use': 0, 'last_success': 0, 'last_error': 0,
                        'buckets_by_hour': {},  # h -> {s,e,r}
                        'by_product': {},
                        'sources': [],  # peer ids contributing
                        'window_hours': 168}

        # Include primary (url=None) by reading its local copernicus.list_credentials
        # so its own traffic counts when it ever runs frontier work.
        try:
            import copernicus as _cop_local
            local_creds = _cop_local.list_credentials()
        except Exception:
            local_creds = []
        for c in local_creds:
            cid = c.get('client_id')
            u = c.get('usage')
            if not cid or not u or cid not in agg:
                continue
            a = agg[cid]
            contributed = False
            for k in ('success_7d', 'error_7d', 'rotated_7d',
                      'success', 'error', 'rotated'):
                v = int(u.get(k) or 0)
                a[k] += v
                if v:
                    contributed = True
            for k in ('last_use', 'last_success', 'last_error'):
                v = int(u.get(k) or 0)
                if v > a[k]:
                    a[k] = v
            for b in (u.get('buckets') or []):
                h = b.get('h')
                if h is None:
                    continue
                cell = a['buckets_by_hour'].setdefault(
                    int(h), {'s': 0, 'e': 0, 'r': 0})
                cell['s'] += int(b.get('s') or 0)
                cell['e'] += int(b.get('e') or 0)
                cell['r'] += int(b.get('r') or 0)
            for prod, pc in (u.get('by_product') or {}).items():
                dst = a['by_product'].setdefault(
                    prod, {'success': 0, 'error': 0, 'rotated': 0})
                if isinstance(pc, dict):
                    dst['success'] += int(pc.get('success') or 0)
                    dst['error'] += int(pc.get('error') or 0)
                    dst['rotated'] += int(pc.get('rotated') or 0)
            if contributed and 'primary' not in a['sources']:
                a['sources'].append('primary')

        targets = [p for p in peers_list if p.get('url')]
        if not targets:
            # Still materialise local-only contributions below.
            pass
        if targets:
          with ThreadPoolExecutor(max_workers=min(16, len(targets))) as ex:
            futs = {ex.submit(_fetch, p.get('url')): p for p in targets}
            try:
                done_iter = as_completed(futs, timeout=20)
            except Exception:
                done_iter = []
            for fut in done_iter:
                peer = futs[fut]
                pid = peer.get('id')
                try:
                    creds = fut.result()
                except Exception:
                    continue
                for c in creds:
                    cid = c.get('client_id')
                    u = c.get('usage')
                    if not cid or not u or cid not in agg:
                        continue
                    a = agg[cid]
                    contributed = False
                    for k in ('success_7d', 'error_7d', 'rotated_7d',
                              'success', 'error', 'rotated'):
                        v = int(u.get(k) or 0)
                        a[k] += v
                        if v:
                            contributed = True
                    for k in ('last_use', 'last_success', 'last_error'):
                        v = int(u.get(k) or 0)
                        if v > a[k]:
                            a[k] = v
                    # Merge per-hour buckets
                    for b in (u.get('buckets') or []):
                        h = b.get('h')
                        if h is None:
                            continue
                        cell = a['buckets_by_hour'].setdefault(
                            int(h), {'s': 0, 'e': 0, 'r': 0})
                        cell['s'] += int(b.get('s') or 0)
                        cell['e'] += int(b.get('e') or 0)
                        cell['r'] += int(b.get('r') or 0)
                    # Merge per-product counts (best-effort)
                    for prod, pc in (u.get('by_product') or {}).items():
                        dst = a['by_product'].setdefault(
                            prod, {'success': 0, 'error': 0, 'rotated': 0})
                        if isinstance(pc, dict):
                            dst['success'] += int(pc.get('success') or 0)
                            dst['error'] += int(pc.get('error') or 0)
                            dst['rotated'] += int(pc.get('rotated') or 0)
                    if contributed and pid and pid not in a['sources']:
                        a['sources'].append(pid)

        # Materialise back onto cred_pool: convert buckets_by_hour to a
        # sorted list spanning the standard 168-hour window.
        now_h = int(time.time()) // 3600
        for c in cred_pool:
            cid = c.get('client_id')
            if not cid or cid not in agg:
                continue
            a = agg[cid]
            window = a['window_hours']
            cutoff = now_h - window
            buckets_out = []
            bb = a['buckets_by_hour']
            for h in range(cutoff + 1, now_h + 1):
                cell = bb.get(h, {'s': 0, 'e': 0, 'r': 0})
                buckets_out.append({'h': h, 's': cell['s'],
                                     'e': cell['e'], 'r': cell['r']})
            c['usage'] = {
                'success_7d': a['success_7d'],
                'error_7d': a['error_7d'],
                'rotated_7d': a['rotated_7d'],
                'success': a['success'],
                'error': a['error'],
                'rotated': a['rotated'],
                'last_use': a['last_use'],
                'last_success': a['last_success'],
                'last_error': a['last_error'],
                'buckets': buckets_out,
                'window_hours': window,
                'by_product': a['by_product'],
                'aggregated_from': a['sources'],
            }
            # Recompute the health score against the *aggregated* usage
            # so the director's view (process.html) matches the cred
            # ordering used by ``_assign_cred_indices``. Each peer's
            # local /api/v1/credentials still returns its own per-peer
            # score using its local usage.
            try:
                import copernicus as _cop_health
                c['health'] = _cop_health.score_credential_health(c)
            except Exception:
                pass

    def _valid_credentials(self) -> list[int]:
        """Return indices of credentials that are not exhausted.

        **Pure cache read.** Never triggers an OIDC probe. The
        director loop is the *only* call site permitted to refresh
        the credential health (via ``_refresh_credentials_if_due``);
        every other reader — including ``get_status`` running in any
        gunicorn worker — sees the cached ``last_status`` from
        ``copernicus.list_credentials()``.

        Why this is strict: in 2026-05-06 we saw the primary's worker
        pool wedge with ~50 OIDC requests/s and the listen backlog
        overflow (process.html became unreachable). Root cause: any
        cache miss in any worker thread fanned out into 8 OIDC token
        requests. Multiple threads racing past a stale timestamp
        amplified that to 64+ probes per tick. The architectural fix
        is that the request path *cannot* probe — only the director
        loop can.
        """
        creds = self._credential_pool()
        valid = []
        for c in creds:
            if c.get('exhausted'):
                continue
            st = (c.get('last_status') or '').lower()
            if st in ('exhausted', 'invalid'):
                continue
            valid.append(int(c.get('index')))
        return valid

    def _oidc_reval_loop(self) -> None:
        """Dedicated daemon: probes OIDC credentials on its own cadence
        so the director loop never blocks on a 6-worker parallel sweep
        (which still takes a few seconds wall-clock when CDSE is slow).

        Lifecycle is bound to the owning Director instance: started in
        ``start()``, stopped in ``stop()`` via ``self._oidc_stop``.
        Critically NOT a module-global thread — on step_down or HA
        handover, the demoted instance is destroyed and its event is
        set, so a stale director can't keep poking OIDC after losing
        the role. The new Director will spawn its own loop.
        """
        # Stagger the first probe slightly so a srv restart doesn't fire
        # an OIDC sweep simultaneously with the director's first tick.
        if self._oidc_stop.wait(timeout=5.0):
            return
        log.info('oidc-reval daemon started (interval=%ds)',
                 _REVALIDATE_INTERVAL_S)
        try:
            while self._running and not self._oidc_stop.is_set():
                try:
                    self._refresh_credentials_if_due()
                except Exception:
                    log.debug('oidc-reval iteration failed', exc_info=True)
                # Re-check cadence on a tight wake schedule so stop()
                # joins quickly; the interval check inside
                # _refresh_credentials_if_due() is the real throttle.
                if self._oidc_stop.wait(timeout=30.0):
                    break
        finally:
            log.info('oidc-reval daemon stopped')

    def _refresh_credentials_if_due(self) -> None:
        """Director-loop hook: re-run OIDC probes at most once every
        ``_REVALIDATE_INTERVAL_S`` seconds. Single-flight via a process-
        wide lock + module-level timestamp; if another thread is
        already probing (extremely rare given the loop is single-
        threaded but defensive against future call-sites), this returns
        immediately. Cached ``last_status`` updates land on disk via
        ``copernicus._save_credentials_to_disk()`` so all gunicorn
        workers see fresh data without each having to probe.
        """
        global _LAST_CREDS_REVALIDATED_AT, _REVALIDATE_OWNER_THREAD
        now = time.time()
        last_mem = float(_LAST_CREDS_REVALIDATED_AT or 0)
        last_disk = float(self.state.get('_creds_revalidated_at') or 0)
        last = max(last_mem, last_disk)
        if now - last <= _REVALIDATE_INTERVAL_S:
            return
        # Single-flight — if another thread already holds the lock,
        # skip this tick. The next tick will retry once the holder is
        # done; meanwhile cached statuses remain valid.
        if not _REVALIDATE_LOCK.acquire(blocking=False):
            return
        try:
            # Re-check inside the lock (someone else may have just
            # finished a probe).
            if (time.time() - max(
                    float(_LAST_CREDS_REVALIDATED_AT or 0),
                    float(self.state.get('_creds_revalidated_at') or 0),
                )) <= _REVALIDATE_INTERVAL_S:
                return
            _REVALIDATE_OWNER_THREAD = threading.get_ident()
            try:
                import copernicus as _cop
                _before = list(_cop.list_credentials() or [])
                _cop.revalidate_all_credentials()
                _after = list(_cop.list_credentials() or [])
                # Surface a one-line summary: valid/total + any
                # transitions (ok→bad, bad→ok). Keeps the merged log
                # quiet on no-op revalidations — only logs deltas.
                def _key(c):
                    return (c.get('client_id') or '')[:12]
                def _ok(c):
                    return (c.get('last_status') or '').lower() in ('ok', 'valid')
                _b = {_key(c): _ok(c) for c in _before}
                _a = {_key(c): _ok(c) for c in _after}
                _flips = []
                for k, ok_after in _a.items():
                    if k in _b and _b[k] != ok_after:
                        _flips.append(k + (':→ok' if ok_after else ':→bad'))
                _valid = sum(1 for v in _a.values() if v)
                _total = len(_a)
                if _flips:
                    _emit_director_event(
                        'creds revalidated: ' + str(_valid) + '/' + str(_total)
                        + ' valid (' + ', '.join(_flips) + ')')
            except Exception as e:
                log.debug('revalidate_all_credentials failed: %s', e)
                return
            _LAST_CREDS_REVALIDATED_AT = time.time()
            with self._lock:
                self.state['_creds_revalidated_at'] = (
                    _LAST_CREDS_REVALIDATED_AT
                )
                try:
                    save_director_state(self.state)
                except Exception:
                    log.debug('save_director_state during revalidate failed',
                              exc_info=True)
        finally:
            _REVALIDATE_OWNER_THREAD = None
            _REVALIDATE_LOCK.release()

    def _peer_capabilities(self, ps: dict) -> set[str]:
        """Read capability flags from a peer's /api/v1/info response.

        ps is a dict that may have come from get_peer_status() (which
        already contains 'git_commit') OR from a cached /info call.
        We refresh capabilities lazily and cache them in director state.
        """
        caps = set(ps.get('capabilities') or [])
        return caps

    def _refresh_peer_caps(self, peer: dict) -> set[str]:
        """Fetch /api/v1/info from a peer and cache its capabilities.

        Also opportunistically probes /api/v1/credentials to learn how
        many creds the peer's local pool has (used to scope holder
        annotations for legacy peers that use the full local pool).
        """
        url = peer.get('url')
        pid = peer['id']
        cache = self.state.setdefault('_peer_caps', {})
        entry = cache.get(pid) or {}
        if entry and (time.time() - entry.get('at', 0)) < 300:
            return set(entry.get('caps') or [])
        try:
            if url is None:
                # Local peer: import is fastest
                import copernicus as _cop  # noqa: F401
                caps = {'cred_subset_env', 'lat_strip_filter',
                        'cred_api_v1', 'parallel_frontiers'}
                cred_count = len(_cop.list_credentials())
            else:
                r = requests.get(url.rstrip('/') + '/api/v1/info',
                                 timeout=PEER_TIMEOUT_PROBE,
                                 headers=_admin_headers())
                if not r.ok:
                    return set(entry.get('caps') or [])
                d = r.json()
                caps = set(d.get('capabilities') or [])
                cred_count = entry.get('cred_count') or 4
                # Probe /api/v1/credentials (works on both upgraded and
                # legacy peers that expose it). 4 is the legacy default.
                try:
                    cr = requests.get(url.rstrip('/') + '/api/v1/credentials',
                                      timeout=PEER_TIMEOUT_PROBE,
                                      headers=_admin_headers())
                    if cr.ok:
                        cred_count = len((cr.json() or {}).get('credentials') or [])
                except Exception:
                    pass
                # Self-heal: a peer with 0 creds can't run frontier work
                # and burns CPU on 401 invalid_client storms. This happens
                # when add_peer() ran while the peer was still booting
                # (online=False at the time, so the credential bootstrap
                # in app.py:director_add_peer was skipped). Push from the
                # director's own store now that it's responding.
                if cred_count == 0:
                    try:
                        self._bootstrap_peer_credentials(url, pid)
                        # Re-probe so the cache reflects the new count.
                        cr = requests.get(url.rstrip('/') + '/api/v1/credentials',
                                          timeout=PEER_TIMEOUT_PROBE,
                                          headers=_admin_headers())
                        if cr.ok:
                            cred_count = len((cr.json() or {}).get('credentials') or [])
                    except Exception as _be:
                        log.debug('bootstrap creds to %s failed: %s', pid, _be)
        except Exception as e:
            log.debug('cap probe %s failed: %s', pid, e)
            return set(entry.get('caps') or [])
        cache[pid] = {'caps': sorted(caps), 'at': time.time(),
                      'cred_count': cred_count}
        return caps

    def _bootstrap_peer_credentials(self, url: str, pid: str) -> int:
        """Push every credential from the director's local pool to *peer*.

        Idempotent at the peer side (re-adding an existing client_id is a
        meta-update). Throttled per-peer to once every 5 min via
        ``state['_cred_bootstrap_at']`` so a transiently-flapping peer
        doesn't spin us. Marked X-Cred-Fanout=1 so the receiving peer
        does NOT re-broadcast (we're addressing it directly).
        """
        if not url:
            return 0
        now = time.time()
        with self._lock:
            book = self.state.setdefault('_cred_bootstrap_at', {})
            last = float(book.get(pid) or 0)
            if now - last < 300:
                return 0
            book[pid] = now
        try:
            import copernicus as _cop
            store = _cop.list_credentials_with_secrets() or []
        except Exception as e:
            log.debug('list_credentials_with_secrets failed: %s', e)
            return 0
        if not store:
            return 0
        hdrs = dict(_admin_headers())
        hdrs['X-Cred-Fanout'] = '1'
        ok = 0
        for c in store:
            cid = (c.get('client_id') or '').strip()
            sec = (c.get('client_secret') or '').strip()
            if not cid or not sec:
                continue
            try:
                rr = requests.post(
                    url.rstrip('/') + '/api/v1/credentials',
                    json={'client_id': cid, 'client_secret': sec,
                          'label': c.get('label', ''),
                          'notes': c.get('notes', ''),
                          'validate': False},
                    headers=hdrs, timeout=10,
                )
                if rr.ok:
                    ok += 1
            except Exception:
                pass
        if ok:
            try:
                _emit_director_event(
                    'creds bootstrapped to ' + pid + ': ' + str(ok)
                    + '/' + str(len(store)) + ' (peer had empty store)',
                    peer=pid)
            except Exception:
                pass
        return ok

    def _effective_creds_per_frontier(self, cfg: dict) -> int:
        """Legacy hook — returns floor (per-peer cred count) for the
        static path or the value derived from adaptive frontier-count
        planning. Kept because several callers (status payload,
        cred-slice assignment, cache-only orchestrator) read it.

        Static path: ``min_creds_per_frontier`` from peers.json (default 2).

        Adaptive path (default ON): scales the *target frontier count*
        from the smoothed Copernicus sub-factor (see
        :py:meth:`_target_frontier_count`) and back-derives a per-peer
        cred count = floor(len(valid) / target). With 8 valid creds:
        target=8 → per=1, target=5 → per=1 (3 peers get 1, 5 get 1 —
        see _assign_cred_indices for the uneven split), target=2 →
        per=4. Per is always ≥ 1 and never exceeds the static floor.
        """
        floor = max(1, int(cfg.get('min_creds_per_frontier',
                                    self.MIN_CREDS_PER_FRONTIER_DEFAULT)))
        if not bool(cfg.get('adaptive_creds_per_frontier', True)):
            return floor
        valid = self._valid_credentials()
        if not valid:
            return floor
        target = self._target_frontier_count(cfg, len(valid))
        # Per-peer cred count: at least 1, capped by floor. Frontiers
        # past the first ``len(valid) % target`` get the floor (= per),
        # the rest get one extra cred — but the worker only sees its
        # own slice, so reporting ``per`` here is fine for status.
        per = max(1, min(floor, len(valid) // max(1, target)))
        return per

    def _target_frontier_count(self, cfg: dict, n_valid: int) -> int:
        """Smoothly scaled target frontier count in [1, n_valid].

        Driven by the smoothed Copernicus sub-factor EMA:
          * EMA = 1.0 (clean) → target = n_valid (e.g. 8)
          * EMA = THROTTLE_MIN_FACTOR (saturated) → target = ceil(n_valid / floor)
            (e.g. 4 with floor=2)

        Linear interp between those endpoints, with sticky rounding
        (±0.4 hysteresis around the previous decision) so we don't
        flap one frontier up/down on every tick. Decision is persisted
        across ticks in ``state['_target_frontier_count']``.
        """
        floor = max(1, int(cfg.get('min_creds_per_frontier',
                                    self.MIN_CREDS_PER_FRONTIER_DEFAULT)))
        ceiling = max(1, n_valid)
        baseline = max(1, n_valid // floor)
        if ceiling <= baseline:
            return ceiling
        sub_ema = float(self._sub_factor_ema.get(
            'copernicus', THROTTLE_MAX_FACTOR))
        # Map [MIN_FACTOR, MAX_FACTOR] → [baseline, ceiling].
        span = max(1e-6, THROTTLE_MAX_FACTOR - THROTTLE_MIN_FACTOR)
        frac = (sub_ema - THROTTLE_MIN_FACTOR) / span
        frac = max(0.0, min(1.0, frac))
        raw = baseline + frac * (ceiling - baseline)
        prev = float(self.state.get('_target_frontier_count', raw))
        # Sticky rounding: only move when raw is >0.4 away from prev,
        # otherwise keep prev rounded. Prevents a single 402 from
        # bumping us 5↔6.
        if abs(raw - prev) < 0.4:
            decision = int(round(prev))
        else:
            decision = int(round(raw))
        decision = max(baseline, min(ceiling, decision))
        self.state['_target_frontier_count'] = float(decision)
        return decision

    def _max_parallel_frontiers(self, cfg: dict) -> int:
        """How many frontier peers we may run concurrently.

        Each frontier holds ``min_creds_per_frontier`` credentials — one
        active + one (or more) hot-standby spares for in-peer rotation
        on 402. With per=2 and 8 valid creds, we get 4 frontiers, each
        with its own pair (1 working + 1 spare).

        With adaptive mode and a healthy Copernicus sub-factor we drop
        per=1 — 8 valid creds give 8 frontier slots.

        Also bounded by the number of 0.5° lat strips covering Austria
        (so each frontier can take a disjoint strip). With 7 strips this
        rarely binds.
        """
        valid = self._valid_credentials()
        if not valid:
            return 0
        if not bool(cfg.get('adaptive_creds_per_frontier', True)):
            per = self._effective_creds_per_frontier(cfg)
            cap_creds = max(1, len(valid) // per)
        else:
            cap_creds = self._target_frontier_count(cfg, len(valid))
        # Cap on how many disjoint cells we can hand out. With ~14
        # non-empty cells across Austria this should never bind
        # — cred capacity is the real ceiling.
        cap_cells = max(1, len(self._austria_cells()))
        return max(1, min(cap_creds, cap_cells))

    def _assign_cred_indices(self, frontier_ids: list[str], cfg: dict,
                              prior: dict | None = None) -> dict:
        """Distribute valid credential indices across frontier peers.

        Each peer gets a disjoint slice of length ``min_creds_per_frontier``
        (1 active + (per-1) hot-standby spares for in-peer rotation).
        Returns {peer_id: [cred_idx, ...]}.

        ``prior`` is the previous plan ({peer_id: [idx,...]}). When
        provided, peers in ``prior`` keep their slice if it is still
        valid (all indices still in the valid pool, length == per, no
        overlap with another peer's slice). New peers get fresh slices
        from whatever credentials remain. This avoids restarts caused
        by membership churn re-sorting the assignment.
        """
        valid_set_all = set(self._valid_credentials())
        # Order valid creds by *health* score (highest = best candidate)
        # so fresh, low-error, under-used credentials are picked first
        # and recently-erroring or hot creds are deprioritised. This
        # rotates the warm/hot set across peers instead of always
        # landing on the same indices. See ``copernicus.score_credential_health``.
        try:
            cred_pool = self.state.get('credentials') or self._credential_pool()
        except Exception:
            cred_pool = []
        health_by_idx: dict[int, dict] = {}
        for c in cred_pool:
            try:
                idx = int(c.get('index'))
            except Exception:
                continue
            h = c.get('health')
            if not h:
                try:
                    import copernicus as _cop
                    h = _cop.score_credential_health(c)
                except Exception:
                    h = None
            health_by_idx[idx] = h or {'score': 0.5}
        # Sort key: best health first (descending score), then index
        # ascending for stable tie-break. Negate score so ``sorted``
        # ascending puts the best candidate first.
        def _rank(i: int) -> tuple:
            h = health_by_idx.get(i) or {'score': 0.5}
            return (-float(h.get('score') or 0.0), i)
        valid = sorted(valid_set_all, key=_rank)
        per = self._effective_creds_per_frontier(cfg)
        out: dict[str, list[int]] = {}
        if not valid or not frontier_ids:
            return out
        n_peers = len(frontier_ids)
        # When the planner asked for more frontiers than per*N can
        # cover (uneven split: e.g. 8 creds, 5 frontiers with per=1),
        # spread the leftover creds one extra each across the first K
        # peers. Each peer thus gets per or per+1 creds. If
        # n_peers*per > len(valid) we fall back to per=1 for everyone
        # and drop the last few peers if there still aren't enough.
        if per * n_peers > len(valid):
            per_eff = max(1, len(valid) // max(1, n_peers))
        else:
            per_eff = per
        extra = max(0, len(valid) - per_eff * n_peers)  # leftovers
        valid_set = set(valid)
        used: set[int] = set()
        prior = prior or {}
        # Keep existing assignments where possible. We trim down
        # over-allocated slices (e.g. peer was activated solo with
        # per=2, target now wants per_eff=1) so cred capacity isn't
        # hoarded by early peers. Lengths above per_eff+1 are also
        # trimmed to per_eff. This was the bug that capped parallel
        # frontiers at 5 (with 8 valid creds and target=8) because
        # the active peer's prior creds=[0,1] survived as a 2-cred
        # slice instead of being squeezed to a single index.
        # Health threshold below which we evict a held credential and
        # let the leftover pass refill from a healthier candidate.
        # 0.35 ~ "degraded" — rotation churn or recent errors.
        UNHEALTHY = 0.35
        for pid in frontier_ids:
            slice_ = prior.get(pid)
            if not slice_:
                continue
            slice_ = [int(i) for i in slice_ if int(i) in valid_set]
            if not slice_:
                continue
            # Evict creds whose health has dropped — only when there's
            # a strictly healthier leftover candidate available, so a
            # globally-degraded pool doesn't cause assignment churn.
            healthy_pool = [
                i for i in valid
                if i not in used
                and i not in slice_
                and float((health_by_idx.get(i) or {}).get('score') or 0)
                    > UNHEALTHY + 0.15
            ]
            if healthy_pool:
                slice_ = [
                    i for i in slice_
                    if float((health_by_idx.get(i) or {}).get('score') or 0)
                       >= UNHEALTHY
                ]
                if not slice_:
                    continue
            # Keep deterministic order, then sort by health for the
            # in-slice rotation order (worker reads index 0 first).
            slice_ = sorted(slice_,
                            key=lambda i: -float(
                                (health_by_idx.get(i) or {}).get('score') or 0))
            if set(slice_) & used:
                slice_ = [i for i in slice_ if i not in used]
                if not slice_:
                    continue
            if len(slice_) > per_eff:
                slice_ = slice_[:per_eff]
            if len(slice_) < per_eff:
                # Defer to the leftover-distribution pass; it can top
                # up shorts from the leftover pool.
                continue
            out[pid] = slice_
            used.update(slice_)
        # Recompute extras based on what's already locked in.
        leftovers = [i for i in valid if i not in used]
        remaining_peers = sorted(p for p in frontier_ids if p not in out)
        # Distribute: each peer gets per_eff; first ``extra`` peers get +1.
        # ``extra`` is the number of leftover creds beyond per_eff*n_peers,
        # but we've already given some peers their share via ``out``
        # — so recompute as (leftovers - per_eff*remaining).
        bonus = max(0, len(leftovers) - per_eff * len(remaining_peers))
        for i, pid in enumerate(remaining_peers):
            take = per_eff + (1 if i < bonus else 0)
            slice_ = leftovers[:take]
            if len(slice_) < per_eff:
                break
            out[pid] = slice_
            leftovers = leftovers[take:]
        return out

    def _austria_lat_strips(self) -> list[tuple[float, float]]:
        """DEPRECATED. Returns 1° lat-only ranges covering Austria.

        Kept only for backward compat with old state on disk; new
        planning uses :py:meth:`_austria_cells` (1° lat × 2° lon cells).
        """
        try:
            from zenodo_cache import _lat_strips
            return [tuple(s) for s in _lat_strips()]
        except Exception:
            strips = []
            s = 46.0
            while s < 49.5:
                strips.append((round(s, 4), round(s + 1.0, 4)))
                s += 1.0
            return strips

    def _austria_cells(self) -> list[tuple[float, float, float, float]]:
        """Return the canonical (south, north, west, east) cells that
        cover Austria. These match the Zenodo cache bundle layout, so
        frontier peers pinned to disjoint cells never collide on a
        Zenodo ZIP write.

        With STRIP_HEIGHT=1° × STRIP_WIDTH=2° the raw grid spans
        46–50°N × 8–18°E (20 cells), but Austria only occupies
        ~13 of them — the 49–50°N row is entirely empty, as are
        (48–49, 8–12). Cells with zero KGs are pruned so frontier
        peers never get pinned to dead cells (which causes the
        processor to start, find nothing, exit, repeat — and the
        peer's assigned credential is never used).

        Result is cached on the instance for the life of the director.
        """
        cached = getattr(self, '_austria_cells_cache', None)
        if cached is not None:
            return cached
        try:
            from zenodo_cache import _lat_lon_cells
            raw = [tuple(c) for c in _lat_lon_cells()]
        except Exception:
            raw = []
            s = 46.0
            while s < 49.5:
                w = 8.0
                while w < 18.0:
                    raw.append((round(s, 4), round(s + 1.0, 4),
                                round(w, 4), round(w + 2.0, 4)))
                    w += 2.0
                s += 1.0
        # Prune cells with no KGs. kg_list.json is the canonical KG
        # list (~8440 entries), each carrying lat/lon centroids.
        try:
            kg_path = DATA_DIR / 'kg_list.json'
            if kg_path.exists():
                kgs = json.loads(kg_path.read_text())
                non_empty: set[tuple[float, float, float, float]] = set()
                for k in kgs:
                    bb = k.get('bbox') or {}
                    try:
                        lat = (float(bb['min_lat']) + float(bb['max_lat'])) / 2
                        lon = (float(bb['min_lon']) + float(bb['max_lon'])) / 2
                    except Exception:
                        lat = k.get('lat'); lon = k.get('lon')
                        if lat is None or lon is None:
                            continue
                    for c in raw:
                        s, n, w, e = c
                        if s <= lat < n and w <= lon < e:
                            non_empty.add(c)
                            break
                pruned = [c for c in raw if c in non_empty]
                if pruned:
                    dropped = len(raw) - len(pruned)
                    if dropped:
                        log.info('austria_cells: pruned %d empty cells '
                                 '(%d → %d)', dropped, len(raw), len(pruned))
                    raw = pruned
        except Exception as e:
            log.warning('austria_cells: empty-cell prune failed: %s', e)
        self._austria_cells_cache = raw
        return raw

    def _strip_fingerprint(self, lat_south: float, lat_north: float) -> str:
        """Return a cheap fingerprint of the cache manifest for one lat strip.

        Concatenates the ``updated_at`` of every relevant ZIP for that
        strip across all required products. When ANY ZIP is rewritten
        (a new tile uploaded), the fingerprint changes and any
        cache-miss entries pinned to the old fingerprint expire.
        """
        manifest_path = DATA_DIR / 'cache_manifest.json'
        if not manifest_path.exists():
            return ''
        try:
            d = json.loads(manifest_path.read_text())
        except Exception:
            return ''
        files = d.get('files') or {}
        # Match both old strip ZIPs and new cell ZIPs that fall inside
        # this 1° lat band. Cells are 1°-tall so any cell whose south
        # is in [lat_south, lat_north) overlaps. Strips are 0.5°-tall;
        # treat any strip whose midpoint falls in this band as matching.
        parts = []
        for product in ('ndvi', 'sar', 'harmonics', 'worldcover', 'hansen'):
            prefix = ('hansen_' if product == 'hansen'
                      else f'copernicus_{product}_')
            for name, ent in sorted(files.items()):
                if not name.startswith(prefix):
                    continue
                base_n = name.replace('.zip', '')
                try:
                    if '_cell_' in base_n:
                        coords = base_n.split('_cell_', 1)[1].split('_')
                        s_val = float(coords[0])
                    elif '_strip_' in base_n:
                        coords = base_n.split('_strip_', 1)[1].split('_')
                        s_val = float(coords[0])
                    else:
                        continue
                except Exception:
                    continue
                if not (lat_south - 1e-9 <= s_val < lat_north - 1e-9):
                    continue
                parts.append(name + '@' + str(ent.get('updated_at') or '-'))
        import hashlib
        return hashlib.md5('|'.join(parts).encode()).hexdigest()[:12]

    def _strip_for_bbox(self, bb: dict) -> tuple[float, float] | None:
        s = bb.get('min_lat') or bb.get('south')
        n = bb.get('max_lat') or bb.get('north')
        if s is None or n is None:
            return None
        for ls, ln in self._cached_lat_ranges():
            if s >= ls - 1e-9 and n <= ln + 1e-9:
                return (ls, ln)
        return None

    def _load_cache_misses(self) -> dict:
        """Return {kg_code: {fingerprint, recorded_at, peer_id, strip}}."""
        p = DATA_DIR / 'cache_miss_kgs.json'
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text()) or {}
        except Exception:
            return {}

    def _save_cache_misses(self, misses: dict):
        p = DATA_DIR / 'cache_miss_kgs.json'
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix('.tmp')
            tmp.write_text(json.dumps(misses, indent=2))
            os.replace(tmp, p)
        except Exception as e:
            log.warning('save cache_misses failed: %s', e)

    def record_cache_miss(self, kg_code: str, peer_id: str = '?',
                          bbox: dict | None = None,
                          tile_info: str = '') -> dict:
        """Mark a KG as cache-miss.

        Records the *cell* (1°×2°) the KG falls in and a fingerprint
        of that cell's ZIPs. Future cache-only orchestration skips the
        KG until either:
          * the cell's fingerprint changes (its tiles were rewritten
            — i.e. the frontier finished filling it), or
          * 24h elapse (anti-snowball escape valve, in case the cell
            never gets rewritten but the miss was transient).
        Strip is kept for back-compat / dashboard but no longer drives
        the clearing logic — the old strip-keyed approach kept whole
        groups of KGs locked out while the frontier finished a single
        neighbouring cell.
        """
        kg_code = str(kg_code)
        misses = self._load_cache_misses()
        strip = None
        cell = None
        if bbox:
            strip = self._strip_for_bbox(bbox)
            try:
                cell = self._cell_for_bbox(bbox)
            except Exception:
                cell = None
        cell_fp = (self._cell_fingerprint(*cell) if cell else '')
        entry = {
            'fingerprint': cell_fp,  # cell-level (preferred)
            'cell': list(cell) if cell else None,
            'strip': list(strip) if strip else None,
            'strip_fingerprint': (
                self._strip_fingerprint(*strip) if strip else ''),
            'recorded_at': datetime.now(timezone.utc).isoformat(),
            'peer_id': peer_id,
            'tile_info': tile_info,
        }
        misses[kg_code] = entry
        self._save_cache_misses(misses)
        log.info('cache_miss recorded: KG %s (cell=%s fp=%s peer=%s)',
                 kg_code, cell, cell_fp, peer_id)
        # Invalidate the cache-ready KG cache so next tick excludes this KG
        with self._lock:
            self.state.pop('_cache_ready_cache', None)
        return entry

    def _cache_miss_excluded(self) -> set:
        """Return the set of KG codes currently excluded from the
        cache-only whitelist due to unresolved cache misses.

        Clearing rules (any one triggers):
          1. The cell's fingerprint changed since the miss (its tiles
             were rewritten).
          2. Legacy entries with only a strip fingerprint: clear if
             that strip fingerprint changed (back-compat path).
          3. Anti-snowball: any entry older than 24h is dropped
             unconditionally so transient misses can't pin a KG
             forever when the cell happens not to be rewritten.
        """
        misses = self._load_cache_misses()
        if not misses:
            return set()
        cell_fp_cache: dict[tuple, str] = {}
        strip_fp_cache: dict[tuple, str] = {}
        excluded = set()
        changed = False
        for kg, ent in list(misses.items()):
            # Anti-snowball: hard 24h ceiling.
            try:
                age = (datetime.now(timezone.utc) -
                       datetime.fromisoformat(ent['recorded_at'])
                       ).total_seconds()
            except Exception:
                age = 0
            if age > 86400:
                misses.pop(kg, None)
                changed = True
                log.info('cache_miss cleared: KG %s (age %.0fh > 24h)',
                         kg, age / 3600)
                continue

            cleared = False
            cell = ent.get('cell')
            old_cell_fp = ent.get('fingerprint') or ''
            if cell:
                key = (round(cell[0], 4), round(cell[1], 4),
                       round(cell[2], 4), round(cell[3], 4))
                cur = cell_fp_cache.get(key)
                if cur is None:
                    cur = self._cell_fingerprint(*key)
                    cell_fp_cache[key] = cur
                if cur and cur != old_cell_fp:
                    misses.pop(kg, None)
                    changed = True
                    cleared = True
                    log.info('cache_miss cleared: KG %s (cell %s fp %s→%s)',
                             kg, key, old_cell_fp, cur)
            else:
                # Legacy strip-only entry. Clear on strip churn so old
                # records eventually drain.
                strip = ent.get('strip')
                if strip:
                    sk = (round(strip[0], 1), round(strip[1], 1))
                    cur = strip_fp_cache.get(sk)
                    if cur is None:
                        cur = self._strip_fingerprint(strip[0], strip[1])
                        strip_fp_cache[sk] = cur
                    old_strip_fp = (ent.get('strip_fingerprint')
                                    or old_cell_fp)
                    if cur and cur != old_strip_fp:
                        misses.pop(kg, None)
                        changed = True
                        cleared = True
                        log.info('cache_miss cleared: KG %s (legacy strip %s fp %s→%s)',
                                 kg, sk, old_strip_fp, cur)
            if not cleared:
                excluded.add(kg)
        if changed:
            self._save_cache_misses(misses)
        return excluded

    def _cached_cells(self) -> list[tuple[float, float, float, float]]:
        """Cells (s,n,w,e) fully cached for ALL required products.

        Stricter than :py:meth:`_cached_lat_ranges` — that one only
        checks lat coverage and lets a KG sneak past the pre-filter
        when its specific cell isn't yet uploaded. Used by the
        cache-ready whitelist so cache-only peers never get a KG
        whose cell is mid-fill on the frontier (which would abort
        with a cache miss and snowball into ``cache_miss_kgs.json``).
        """
        manifest_path = DATA_DIR / 'cache_manifest.json'
        if not manifest_path.exists():
            return []
        try:
            d = json.loads(manifest_path.read_text())
        except Exception:
            return []
        files = d.get('files') or {}
        per_product_cells: dict[str, set[tuple[float, float, float, float]]] = {}
        # Collect legacy strip lat-ranges per product so we can stitch
        # 0.5° strips into 1° lat bands for the austria_cells expansion.
        per_product_strip_ranges: dict[str, list[tuple[float, float]]] = {}
        for name in files:
            base_n = name.replace('.zip', '')
            try:
                if '_cell_' in base_n:
                    head, coords = base_n.split('_cell_', 1)
                    s, n, w, e = (float(x) for x in coords.split('_'))
                    product = head
                    if product.startswith('copernicus_'):
                        product = product[len('copernicus_'):]
                    per_product_cells.setdefault(product, set()).add(
                        (round(s, 4), round(n, 4),
                         round(w, 4), round(e, 4)))
                elif '_strip_' in base_n:
                    head, coords = base_n.split('_strip_', 1)
                    s, n = (float(x) for x in coords.split('_'))
                    product = head
                    if product.startswith('copernicus_'):
                        product = product[len('copernicus_'):]
                    per_product_strip_ranges.setdefault(
                        product, []).append((s, n))
            except Exception:
                continue
        # Expand legacy strips into cell coverage: a cell (cs,cn,cw,ce)
        # is covered by legacy strips for product P iff every sub-band
        # within [cs,cn] is covered by the union of P's strips. Cheap
        # check via interval merging.
        def _merge(ranges: list[tuple[float, float]]
                   ) -> list[tuple[float, float]]:
            if not ranges:
                return []
            r = sorted(ranges)
            out = [list(r[0])]
            for s, n in r[1:]:
                if s <= out[-1][1] + 1e-9:
                    out[-1][1] = max(out[-1][1], n)
                else:
                    out.append([s, n])
            return [(a, b) for a, b in out]
        try:
            all_cells = self._austria_cells()
        except Exception:
            all_cells = []
        for product, ranges in per_product_strip_ranges.items():
            merged = _merge(ranges)
            bucket = per_product_cells.setdefault(product, set())
            for cs, cn, cw, ce in all_cells:
                for ms, mn in merged:
                    if cs >= ms - 1e-9 and cn <= mn + 1e-9:
                        bucket.add((round(cs, 4), round(cn, 4),
                                    round(cw, 4), round(ce, 4)))
                        break
        required = ['ndvi', 'sar', 'harmonics', 'worldcover', 'hansen']
        if not all(p in per_product_cells for p in required):
            return []
        common = set.intersection(
            *[per_product_cells[p] for p in required])
        return sorted(common)

    def _cell_for_bbox(self, bb: dict) -> tuple[float, float, float, float] | None:
        """Return the cached cell that fully contains *bb*, or None."""
        s = bb.get('min_lat') or bb.get('south')
        n = bb.get('max_lat') or bb.get('north')
        w = bb.get('min_lon') or bb.get('west')
        e = bb.get('max_lon') or bb.get('east')
        if None in (s, n, w, e):
            return None
        for cs, cn, cw, ce in self._cached_cells():
            if (s >= cs - 1e-9 and n <= cn + 1e-9
                    and w >= cw - 1e-9 and e <= ce + 1e-9):
                return (cs, cn, cw, ce)
        return None

    def _cell_fingerprint(self, cs: float, cn: float,
                          cw: float, ce: float) -> str:
        """Fingerprint of all required-product ZIPs for one cell.

        Tighter than :py:meth:`_strip_fingerprint`: a miss recorded
        against cell (s,n,w,e) clears as soon as THAT cell's ZIPs are
        rewritten, instead of waiting for any tile anywhere in the
        strip.
        """
        manifest_path = DATA_DIR / 'cache_manifest.json'
        if not manifest_path.exists():
            return ''
        try:
            d = json.loads(manifest_path.read_text())
        except Exception:
            return ''
        files = d.get('files') or {}
        parts = []
        for product in ('ndvi', 'sar', 'harmonics', 'worldcover', 'hansen'):
            prefix = ('hansen_' if product == 'hansen'
                      else f'copernicus_{product}_')
            for name, ent in sorted(files.items()):
                if not name.startswith(prefix):
                    continue
                base_n = name.replace('.zip', '')
                try:
                    if '_cell_' in base_n:
                        coords = base_n.split('_cell_', 1)[1].split('_')
                        s_val = float(coords[0])
                        w_val = float(coords[2])
                        if (abs(s_val - cs) > 1e-6
                                or abs(w_val - cw) > 1e-6):
                            continue
                    elif '_strip_' in base_n:
                        coords = base_n.split('_strip_', 1)[1].split('_')
                        s_val = float(coords[0])
                        if not (cs - 1e-9 <= s_val < cn - 1e-9):
                            continue
                    else:
                        continue
                except Exception:
                    continue
                parts.append(name + '@' + str(ent.get('updated_at') or '-'))
        import hashlib
        return hashlib.md5('|'.join(parts).encode()).hexdigest()[:12]

    def _cached_lat_ranges(self) -> list[tuple[float, float]]:
        """Lat strips covered by ALL required products in the Zenodo cache.

        Reads ``cache_manifest.json`` and intersects strip availability
        across ndvi, sar, harmonics, worldcover, hansen.  Returns
        list of (south, north) lat pairs.  Cheap; lets the predicate
        skip the >90 %% of KGs that fall outside any covered strip.
        """
        manifest_path = DATA_DIR / 'cache_manifest.json'
        if not manifest_path.exists():
            return []
        try:
            d = json.loads(manifest_path.read_text())
        except Exception:
            return []
        files = d.get('files') or {}
        # Parse both legacy strip names and new cell names. We only
        # care about latitude coverage here -- frontier cell
        # disjointness is enforced separately via _austria_cells().
        # Normalise to the new 1°-tall lat band so legacy 0.5°
        # strips and new 1° cells line up in the intersection.
        import math as _m
        def _norm_lat_band(s: float) -> tuple[float, float]:
            base = _m.floor(s / 1.0) * 1.0
            return (round(base, 4), round(base + 1.0, 4))
        per_product: dict[str, set[tuple[float, float]]] = {}
        for name in files:
            try:
                base_n = name.replace('.zip', '')
                if '_cell_' in base_n:
                    head, coords = base_n.split('_cell_', 1)
                    s, n, _w, _e = coords.split('_')
                elif '_strip_' in base_n:
                    head, coords = base_n.split('_strip_', 1)
                    s, n = coords.split('_')
                else:
                    continue
                product = head
                if product.startswith('copernicus_'):
                    product = product[len('copernicus_'):]
                per_product.setdefault(product, set()).add(
                    _norm_lat_band(float(s)))
            except Exception:
                continue
        required = ['ndvi', 'sar', 'harmonics', 'worldcover', 'hansen']
        if not all(p in per_product for p in required):
            return []
        common = set.intersection(*[per_product[p] for p in required])
        return sorted(common)

    def _compute_cache_ready_kgs(self, max_kgs: int = 200) -> list[str]:
        """Return KG codes that are fully present in the local+Zenodo cache.

        Two-stage: cheap lat-strip filter (intersects covered strips
        across all products) followed by per-cell check via
        ``tile_cache.is_kg_fully_cached(bbox)``.  Result is cached for
        5 minutes — the cache extends as the frontier peer fetches new
        tiles.
        """
        now = time.time()
        cached = self.state.get('_cache_ready_cache') or {}
        if cached.get('codes') is not None and (now - cached.get('at', 0)) < 300:
            return cached['codes']

        codes: list[str] = []
        try:
            from tile_cache import (CopernicusTileCache, HansenTileCache,
                                     is_kg_fully_cached)
            cop_cache = CopernicusTileCache()
            hansen_cache = HansenTileCache()

            # Use cell-strict filter so we never whitelist a KG
            # whose specific cell is mid-fill on the frontier —
            # the strip-level pre-filter let those through and they
            # would abort with a cache miss, snowballing into
            # cache_miss_kgs.json.
            cached_cells = self._cached_cells()
            if not cached_cells:
                log.info('Cache-ready scan: no fully-cached cell yet')
                with self._lock:
                    self.state['_cache_ready_cache'] = {'codes': [], 'at': now}
                return []

            kg_list_path = DATA_DIR / 'kg_list.json'
            kgs = []
            if kg_list_path.exists():
                kgs = json.loads(kg_list_path.read_text())

            try:
                from app import _get_completed_kgs
                completed = _get_completed_kgs()
            except Exception:
                completed = set()
            failed = set()
            failed_path = DATA_DIR / 'failed_kgs.json'
            if failed_path.exists():
                try:
                    failed = set(json.loads(failed_path.read_text()))
                except Exception:
                    pass

            try:
                import tile_index as ti
                year = ti.dataset_to_year(ti.DEFAULT_DATASET)
            except Exception:
                year = 2024

            def _within_cells(s: float, n: float,
                              w: float, e: float) -> bool:
                for cs, cn, cw, ce in cached_cells:
                    if (s >= cs - 1e-9 and n <= cn + 1e-9
                            and w >= cw - 1e-9 and e <= ce + 1e-9):
                        return True
                return False

            cache_miss_excluded = self._cache_miss_excluded()
            if cache_miss_excluded:
                log.info('Cache-ready scan: excluding %d KGs with unresolved cache misses',
                         len(cache_miss_excluded))
            scanned = 0
            prefiltered = 0
            for kg in kgs:
                code = kg.get('kg_code')
                if not code or code in completed or code in failed:
                    continue
                if code in cache_miss_excluded:
                    continue
                bb = kg.get('bbox') or {}
                w, s = bb.get('min_lon'), bb.get('min_lat')
                e, n = bb.get('max_lon'), bb.get('max_lat')
                if None in (w, s, e, n):
                    continue
                if not _within_cells(s, n, w, e):
                    continue
                prefiltered += 1
                bbox = {'west': w, 'south': s, 'east': e, 'north': n}
                scanned += 1
                try:
                    if is_kg_fully_cached(bbox, year=year,
                                          cop_cache=cop_cache,
                                          hansen_cache=hansen_cache,
                                          local_ok=False):
                        codes.append(code)
                        if len(codes) >= max_kgs:
                            break
                except Exception:
                    continue
            log.info('Cache-ready scan: %d/%d KGs in covered cells (%d), '
                     '%d fully cached (max %d)',
                     prefiltered, len(kgs), len(cached_cells),
                     len(codes), max_kgs)
        except Exception as e:
            log.warning('Cache-ready scan failed: %s', e)

        with self._lock:
            self.state['_cache_ready_cache'] = {'codes': codes, 'at': now}
        return codes

    def _orchestrate_parallel_frontiers(self):
        """Start additional frontier peers when credential capacity permits.

        Each new frontier gets:
          * a disjoint slice of valid Copernicus credentials
          * a disjoint slice of fully-cached lat strips
        Both come from the same cache_manifest used by cache-only peers.

        Skipped silently when:
          * primary active_peer is not running yet
          * fewer than 2*min_creds_per_frontier valid creds (can't fit a
            second peer)
          * peer lacks the ``cred_subset_env`` capability (graceful upgrade)
        """
        with self._lock:
            mode = self.state.get('mode', 'auto')
            cfg = self.cfg.copy()
            state_copy = self.state.copy()
        if mode == 'paused':
            return

        active_id = state_copy.get('active_peer')
        if not active_id:
            return

        valid = self._valid_credentials()
        per = self._effective_creds_per_frontier(cfg)

        # Cap on total concurrent frontiers (incl. active). With 8 valid
        # creds and per=2 this is min(3, 7) = 3 — we always reserve `per`
        # creds as a fleet-wide spare so a 402 can rotate without
        # colliding with another peer's slice. With adaptive mode and a
        # healthy Copernicus sub-factor we drop to per=1 — 8 frontier
        # slots, one cred per slot.

        max_par = self._max_parallel_frontiers(cfg)
        # Server-friendliness throttle: scale down when BEV / Zenodo /
        # Copernicus are getting hammered. Round, but never below 1 if
        # max_par was already ≥ 1 — we always allow the active frontier.
        _factor = float(state_copy.get('capacity_factor',
                                        self._capacity_ema))
        if max_par > 0:
            max_par_throttled = max(1, int(round(max_par * _factor)))
            if max_par_throttled < max_par:
                log.info(
                    'capacity factor %.2f → max_parallel_frontiers %d → %d',
                    _factor, max_par, max_par_throttled,
                )
            max_par = max_par_throttled

        # Frontier peers each pin to a disjoint Austria-wide cell
        # (1° lat × 2° lon by default) so they never collide on a
        # Zenodo cache ZIP write — even when opening regions that aren't
        # yet cached. Cached vs uncached cells are equivalent here:
        # tile uploads are scoped to one cell per ZIP.
        strips = self._austria_cells()
        if not strips:
            # Empty cells is transient (config glitch / startup race).
            # Do NOT wipe parallel_frontiers_active here — the
            # single-active guard in ``_check_and_switch`` consults
            # this set to decide whether to hard-stop running
            # frontiers. Wiping it caused the 2026-05-06 cascade
            # where every authorised parallel frontier was killed on
            # the next tick. Just bail out and try again next tick.
            log.debug('parallel orch: no Austria cells available, skipping')
            return

        budget_bytes = cfg.get('budget_gb', BANDWIDTH_BUDGET_GB) * (1024 ** 3)
        min_reserve = int(cfg.get('min_reserve_peers', MIN_RESERVE_PEERS))
        peers = list(cfg.get('peers', []))
        total_enabled = sum(1 for p in peers
                            if p.get('enabled', True)
                            and not _peer_is_scheduled(p))

        # Already running parallel frontiers (excluding active_peer).
        # ``prev_par`` is the set of peers we authorised as parallel
        # frontiers on the previous tick. When a peer in this set
        # shows up as ``unreachable`` for a single tick (common during
        # a director takeover, a peer's gunicorn worker swap, or a
        # heavy GPKG-build that briefly starves the status endpoint)
        # we keep it in ``running`` for up to
        # ``UNREACHABLE_FAILOVER_THRESHOLD`` consecutive misses — same
        # grace the active-peer path already uses (line ~2599). Without
        # this, a single transient blip drops the peer from
        # ``running`` → it's omitted from the new
        # ``parallel_frontiers_active`` set → the next tick's
        # single-active guard hard-stops it as a non-authorised
        # frontier. That was the slow drain observed during the
        # 2026-05-06 at40-as-interim-director window: ~3 peers lost
        # over ~30 min, never auto-recovered until primary took back
        # over.
        prev_par = set(state_copy.get('parallel_frontiers_active') or [])
        retained_unreachable: list[str] = []
        running: list[str] = []
        candidates: list[dict] = []
        for p in peers:
            pid = p['id']
            if pid == active_id:
                continue
            if not p.get('enabled', True):
                continue
            if _peer_is_scheduled(p):
                continue
            role = self._peer_role(p)
            if role in ('idle', 'cache_only'):
                continue
            if _peer_in_warmup(p):
                continue  # fresh peer — not yet eligible for frontier work
            # Reservation holders are eligible as parallel frontiers —
            # they just process their reserved KG first (priority queue
            # ensures it is at the head, _reserved_kgs excludes other
            # peers from racing them on it). Without this, a reservation
            # holder whose pre-emption is pending (active peer mid‑KG)
            # would sit idle indefinitely instead of running parallel
            # frontier work in the meantime.
            bw = state_copy.get('peer_bandwidth', {}).get(pid, {})
            if (_peer_budget_bytes(p, cfg) - bw.get('used_bytes', 0)) < 2 * (1024 ** 3):
                continue
            ps = get_peer_status(p.get('url'))
            if ps.get('state') == 'unreachable':
                if pid in prev_par:
                    # Tolerate transient unreachable on an authorised
                    # parallel frontier. Bump per-peer miss counter;
                    # only drop after the same threshold the active
                    # peer uses.
                    with self._lock:
                        misses = self.state.setdefault(
                            'parallel_unreachable_count', {})
                        misses[pid] = int(misses.get(pid, 0)) + 1
                        n = int(misses[pid])
                    if n < UNREACHABLE_FAILOVER_THRESHOLD:
                        log.info('Parallel frontier %s unreachable '
                                 '(%d/%d) — preserving slot',
                                 pid, n, UNREACHABLE_FAILOVER_THRESHOLD)
                        retained_unreachable.append(pid)
                        continue
                    log.warning('Parallel frontier %s unreachable %d '
                                'times — releasing slot', pid, n)
                    with self._lock:
                        self.state.get(
                            'parallel_unreachable_count', {}).pop(pid, None)
                continue
            # Reachable — clear miss counter if any.
            if pid in (self.state.get('parallel_unreachable_count') or {}):
                with self._lock:
                    self.state.get(
                        'parallel_unreachable_count', {}).pop(pid, None)
            # Refresh caps opportunistically; all current peers expose
            # cred_subset_env so we no longer gate on it.
            self._refresh_peer_caps(p)
            is_running = ps.get('state') in ('running', 'processing')
            is_cache_only_run = bool(ps.get('cache_only'))
            if is_running and not is_cache_only_run:
                running.append(pid)
            else:
                candidates.append({
                    'peer': p,
                    'state': ps.get('state', 'unknown'),
                    'needs_stop_cache_only': is_running and is_cache_only_run,
                })

        # Slot budget = max_par - 1 (subtract active). Also keep reserve.
        slack = max(0, total_enabled - (1 if active_id else 0)
                    - len(running) - min_reserve)
        max_add = min(slack, max_par - 1 - len(running))
        # Ramp limiter (see RAMP_*_STARTS_PER_TICK). Scale gentleness
        # to capacity factor: never burst more than a few new frontier
        # peers per tick.
        _ramp_span = RAMP_MAX_STARTS_PER_TICK - RAMP_MIN_STARTS_PER_TICK
        _ramp_cap = RAMP_MIN_STARTS_PER_TICK + int(round(_ramp_span * _factor))
        _ramp_cap = max(RAMP_MIN_STARTS_PER_TICK, _ramp_cap)
        max_add = min(max_add, _ramp_cap)

        # Stable assignment: active + sorted(running+candidates) -> ids in
        # order define which cred slice and lat strip they get.
        # Prefer idle/stopped candidates over cache-only-running ones so
        # we don't kill in-progress cache-only work when there's a free
        # peer available.
        # Order: don't preempt running cache-only peers; among the rest,
        # quiet peers first (so fresh peers absorb new frontier slots).
        # Steal-aware ordering: a fresh frontier slot is far more
        # valuable on a low-steal peer (cred efficiency). Add
        # (1/capacity - 1) so a 50%% steal peer ranks +1.0 worse than
        # a 0%% peer, ahead of the typical noise spread (<0.5).
        candidates.sort(key=lambda c: (
            1 if c.get('needs_stop_cache_only') else 0,
            _peer_noise_score(c['peer']['id'], state_copy)
            + ((1.0 / _peer_cpu_capacity(c['peer']['id'])) - 1.0),
            c['peer']['id'],
        ))
        # Include retained_unreachable peers up front so their cred /
        # strip plan is reissued (preserving their prior slice) instead
        # of being treated as a fresh candidate and triggering a
        # plan-drift restart when they reappear.
        ordered = [active_id] + sorted(set(running) | set(retained_unreachable))
        if max_add > 0:
            ordered += sorted(c['peer']['id'] for c in candidates[:max_add])
        # De-dup while preserving order (active_id may also be in running).
        _seen: set[str] = set()
        ordered = [pid for pid in ordered
                   if pid and not (pid in _seen or _seen.add(pid))]
        ordered = ordered[:max_par]  # cap to credential capacity
        # Preserve existing peer→creds mapping where valid; only newly
        # promoted peers get fresh slices. Without this, a peer entering
        # or leaving the running set would shuffle every other peer's
        # cred slice, triggering plan-drift restart loops.
        prior_cred_plan = state_copy.get('frontier_cred_plan') or {}
        cred_plan = self._assign_cred_indices(
            ordered, cfg, prior=prior_cred_plan)

        # Lat strips: distribute ALL Austria strips contiguously across
        # the frontier peers. With 7 strips and 3 frontiers each peer
        # gets ~2-3 contiguous strips. This matters for straddler KGs
        # (the ~6% whose bbox crosses a 0.5° strip boundary): the owning
        # frontier (by centroid) reads/writes neighbour-strip cache tiles
        # for the overlap. If the neighbour strip is owned by the same
        # peer (contiguous range), there's no contention. Cross-peer
        # contention only happens at the inter-peer boundary — at most
        # N-1 such boundaries instead of one boundary per strip.
        # When fewer strips than frontiers (very rare with 7 strips),
        # tail peers get nothing — we then trim ``ordered``.
        strip_plan: dict[str, list] = {}
        n_peers = len(ordered)
        n_strips = len(strips)
        prior_strip_plan = state_copy.get('frontier_strip_plan') or {}
        if n_peers > 0 and n_strips > 0:
            # Preserve existing peer→strip assignments where the prior
            # slice still fits within the canonical strip set; only fill
            # gaps for new peers from whatever strips remain. Without
            # this, churn in `ordered` re-shuffles every peer's strip
            # range each tick, triggering plan-drift restarts.
            #
            # However: cap each preserved slice at the peer's fair
            # share for the current ``n_peers``. Otherwise an existing
            # peer that was activated alone (and given ALL strips) would
            # hog them forever, leaving newly-promoted parallel
            # frontiers with empty plans → trimmed → never started.
            # Fair share: base = n_strips // n_peers, plus 1 extra for
            # the first ``n_strips % n_peers`` peers in ``ordered``.
            base_fair = n_strips // n_peers
            extra_fair = n_strips % n_peers
            fair_share = {
                pid: base_fair + (1 if i < extra_fair else 0)
                for i, pid in enumerate(ordered)
            }
            strip_set = {tuple(s) for s in strips}
            used: set[tuple[float, float]] = set()
            for pid in ordered:
                old = prior_strip_plan.get(pid) or []
                old_t = [tuple(s) for s in old]
                if not old:
                    continue
                # Drop entries no longer in the canonical strip set or
                # already taken by another peer (preserves order).
                kept = [s for s in old_t
                        if s in strip_set and s not in used]
                # Trim to fair share so leftover strips remain for new
                # frontiers.
                kept = kept[:fair_share.get(pid, 0)]
                if kept:
                    strip_plan[pid] = [list(s) for s in kept]
                    used.update(kept)
            leftover = [s for s in strips if tuple(s) not in used]
            unassigned = [pid for pid in ordered if pid not in strip_plan]
            if unassigned and leftover:
                base = len(leftover) // len(unassigned)
                extra = len(leftover) % len(unassigned)
                # Tail-extra: existing peers may already hold extra
                # strips, so give the bonus to new peers in id order.
                offset = 0
                for i, pid in enumerate(unassigned):
                    size = base + (1 if i < extra else 0)
                    if size <= 0:
                        strip_plan[pid] = []
                        continue
                    slice_ = leftover[offset:offset + size]
                    strip_plan[pid] = [list(c) for c in slice_]
                    offset += size
            for pid in ordered:
                strip_plan.setdefault(pid, [])
        else:
            for pid in ordered:
                strip_plan[pid] = []
        # Trim peers that ended up without a strip (no cap collision).
        ordered = [pid for pid in ordered if strip_plan.get(pid)]
        cred_plan = {pid: cred_plan[pid] for pid in ordered if pid in cred_plan}

        # Detect peers running with a stale plan (creds OR strips don't
        # match). Hard-stop them so the next tick re-issues with the
        # correct env. This is what catches the case where the active
        # peer was started without a lat_strips env (e.g. via the
        # legacy choose_active_peer path).
        old_cred_plan = state_copy.get('frontier_cred_plan') or {}
        old_strip_plan = state_copy.get('frontier_strip_plan') or {}
        peers_to_restart: list[str] = []
        for pid in [active_id] + list(running):
            want_creds = sorted(cred_plan.get(pid) or [])
            want_strips = sorted(
                tuple(s) for s in (strip_plan.get(pid) or [])
            )
            have_creds = sorted(old_cred_plan.get(pid) or [])
            have_strips = sorted(
                tuple(s) for s in (old_strip_plan.get(pid) or [])
            )
            if want_creds and want_strips and (
                want_creds != have_creds or want_strips != have_strips
            ):
                log.warning(
                    'Frontier %s plan drift: have creds=%s strips=%s, '
                    'want creds=%s strips=%s — restarting',
                    pid, have_creds, have_strips, want_creds, want_strips,
                )
                peers_to_restart.append(pid)

        for pid in peers_to_restart:
            peer = get_peer_by_id(cfg, pid)
            if not peer:
                continue
            try:
                stop_peer_processor(peer.get('url'), graceful=False)
            except Exception as e:
                log.warning('Restart-stop %s failed: %s', pid, e)
        # If we hard-stopped any peers, return early; the next tick will
        # see them as candidates with the correct plan.
        if peers_to_restart:
            with self._lock:
                self.state['frontier_cred_plan'] = cred_plan
                self.state['frontier_strip_plan'] = strip_plan
                self.state['parallel_frontiers_active'] = [
                    pid for pid in running if pid not in peers_to_restart
                ]
                save_director_state(self.state)
            return

        # Start new candidates with their plan.
        to_start_ids = [pid for pid in ordered
                        if pid != active_id and pid not in running]
        cand_by_id = {c['peer']['id']: c for c in candidates}
        started = []
        for pid in to_start_ids:
            peer = get_peer_by_id(cfg, pid)
            if not peer:
                continue
            creds = cred_plan.get(pid)
            strips_for = strip_plan.get(pid)
            if not creds or not strips_for:
                continue
            cand = cand_by_id.get(pid) or {}
            # If the peer is currently running cache-only, stop it first so
            # we can re-start it as a frontier with the assigned creds /
            # lat-strip env. Otherwise the start API returns 409 and the
            # peer keeps doing cache-only work, leaving creds idle.
            #
            # IMPORTANT: must be a HARD stop (graceful=False). A graceful
            # stop only flips the after-KG shutdown flag (~hours away).
            # The subsequent start API would 409 and we'd ping-pong every
            # tick, with the peer never actually entering frontier mode.
            # Tile checkpoints survive the kill so the partial KG resumes
            # cheaply on the next run.
            if cand.get('needs_stop_cache_only'):
                log.info('Parallel frontier: hard-stopping cache-only run on '
                         '%s before promoting to frontier', pid)
                try:
                    stop_peer_processor(peer.get('url'), graceful=False)
                except Exception as e:
                    log.warning('Stop cache-only on %s failed: %s', pid, e)
                    continue
                # Poll until processor_state reflects stopped — the
                # subprocess can take several seconds to exit (and the
                # `Processor already running (external)` 409 will fire
                # otherwise, blocking the promotion for many ticks).
                import time as _t
                settled = False
                for _ in range(15):
                    _t.sleep(1.0)
                    ps2 = get_peer_status(peer.get('url'))
                    st = ps2.get('state', '')
                    if st in ('idle', 'stopped', 'complete', 'unreachable'):
                        settled = True
                        break
                if not settled:
                    log.warning('Parallel frontier: %s did not settle after '
                                'cache-only stop; deferring', pid)
                    continue
            # Exclude KGs reserved for OTHER peers from this peer's
            # priority queue, so we don't double-process a reservation
            # holder's KG on a parallel frontier that happens to share
            # the same lat strip.
            par_excl = _excluded_kgs(cfg, exclude_peer_id=pid)
            if self._has_pending_graceful_update(pid):
                log.info('Parallel frontier: skipping %s — graceful '
                         'update pending (waiting for git pull + srv '
                         'restart on peer)', pid)
                continue
            log.info('Parallel frontier: starting %s with creds=%s strips=%s%s',
                     pid, creds, strips_for,
                     (' (excluding ' + ','.join(sorted(par_excl)) + ')')
                     if par_excl else '')
            try:
                res = start_peer_processor(peer.get('url'),
                                            exclude_kgs=par_excl,
                                            cred_indices=creds,
                                            lat_strips=strips_for)
            except Exception as e:
                log.warning('Start parallel frontier on %s failed: %s', pid, e)
                continue
            if not isinstance(res, dict) or res.get('error'):
                # API start failed (e.g. 500 on the peer, no systemd
                # fallback for constrained starts). Don't mark this
                # peer as an active parallel frontier — otherwise its
                # cred slice is held hostage forever and the cache-only
                # orchestrator also leaves it alone.
                log.warning(
                    'Parallel frontier start on %s did not succeed: %s',
                    pid, (res or {}).get('error') or res)
                continue
            started.append(pid)
            # Surface the start — with its cred/strip assignment — in the
            # 24h merged log so operators see when a peer was promoted
            # and what slice it owns. (log.info above is logs/srv only.)
            try:
                _strips_txt = ','.join(
                    '%.1f-%.1f' % (s[0], s[1]) for s in strips_for) \
                    if strips_for else '-'
            except Exception:
                _strips_txt = '-'
            _emit_director_event(
                'frontier start → creds=' + ','.join(str(c) for c in creds)
                + ' strips=' + _strips_txt,
                peer=pid)

        # Diff against prior plans — emit per-peer events for any
        # cred / strip assignment that actually changed. Quiet when the
        # plan is a no-op tick (the common case).
        try:
            _all = sorted(set(cred_plan) | set(old_cred_plan)
                          | set(strip_plan) | set(old_strip_plan))
            for _pid in _all:
                _bc = sorted(old_cred_plan.get(_pid) or [])
                _ac = sorted(cred_plan.get(_pid) or [])
                _bs = [tuple(s) for s in (old_strip_plan.get(_pid) or [])]
                _as_ = [tuple(s) for s in (strip_plan.get(_pid) or [])]
                _cred_changed = _bc != _ac
                _strip_changed = sorted(_bs) != sorted(_as_)
                if not (_cred_changed or _strip_changed):
                    continue
                _parts = []
                if _cred_changed:
                    _parts.append('creds ' + (','.join(str(c) for c in _bc) or '-')
                                  + ' → ' + (','.join(str(c) for c in _ac) or '-'))
                if _strip_changed:
                    def _stxt(ss):
                        return ','.join('%.1f-%.1f' % (a, b) for a, b in ss) or '-'
                    _parts.append('strips ' + _stxt(_bs) + ' → ' + _stxt(_as_))
                _emit_director_event('plan: ' + '; '.join(_parts), peer=_pid)
        except Exception:
            pass

        with self._lock:
            # Union running + started + retained_unreachable. The
            # last term keeps temporarily-unreachable peers in the
            # authorised set so the single-active guard does not
            # hard-stop them when they reappear next tick.
            self.state['parallel_frontiers_active'] = list(
                set(running) | set(started) | set(retained_unreachable))
            self.state['frontier_cred_plan'] = cred_plan
            self.state['frontier_strip_plan'] = strip_plan
            save_director_state(self.state)

    # --- Stale-peer auto-retry update -----------------------------
    # If a graceful update was triggered but the peer didn't pull the
    # new commit (e.g. dropped connection, restart skipped), re-trigger
    # an immediate update once the peer has been idle for >10 min. After
    # a second failed retry, surface the peer as needing a manual update
    # in the dashboard.
    STALE_UPDATE_GRACE_S = 600          # 10 min idle before first auto-retry
    STALE_UPDATE_RETRY_GAP_S = 600      # 10 min between retries
    STALE_UPDATE_MAX_ATTEMPTS = 2       # then surface manual command
    # Hard ceiling on how long a "manual update needed" verdict
    # sticks. After this many seconds since the last failed attempt,
    # we wipe the attempt counters and let the auto-retry path try
    # again from scratch — covers the case where the peer was
    # transiently wedged (long upload, network hiccup) and is now
    # reachable + idle but still flagged as needing a human. Tracked
    # commit must of course still be stale; if the peer has caught
    # up the rec is removed in the per-peer loop above.
    STALE_UPDATE_RESET_AFTER_S = 12 * 3600  # 12 h
    # Graceful nudges to mid-KG peers also need a hard ceiling. Without
    # this we re-fire SIGTERMs every 30 min indefinitely on a peer that
    # never reaches a KG boundary in our gap window (long-tail upload,
    # or a recurring failure mode — see 2026-05-08 incident where 12 h
    # of graceful kicks SIGTERMed the same KGs over and over while the
    # cred fan-out was broken). After this many attempts we give up
    # nudging and let the next idle window be picked up by the hard
    # path; this also stops needs_manual_update from being suppressed
    # forever just because the peer is always busy.
    STALE_GRACEFUL_MAX_ATTEMPTS = 3
    # Wave-based update rollout: cap how many peers we trigger per tick
    # so the cluster restarts in waves rather than a single thundering
    # herd. With 50 peers all triggered at once (the 2026-05-06
    # incident) gunicorn workers exhausted file descriptors, circuit
    # breakers tripped fleet-wide, and orchestrators couldn't see
    # candidates for ~2 minutes. Stepping by ~3-5 peers/tick spreads
    # restarts over ~5 minutes — dashboard / cache-only / frontier
    # never lose more than a fraction of the fleet at once.
    STALE_UPDATE_HARD_PER_TICK = 3      # hard restarts (idle peers)
    STALE_UPDATE_GRACEFUL_PER_TICK = 5  # graceful nudges (mid-KG peers)

    def _has_pending_graceful_update(self, pid: str) -> bool:
        """True iff a graceful update was issued to *pid* recently.

        ``/admin/update?graceful=1`` SIGTERMs the processor and spawns
        a deferred-update thread that polls ``pgrep austria_processor.py``
        and only runs ``git pull && systemctl restart srv`` once the
        processor exits. If we respawn the processor before that pgrep
        empties (cache-only orchestrator, parallel-frontier orchestrator,
        active-frontier restart), the deferred update never fires and
        the peer stays on the stale commit while we re-kick it mid-KG
        every retry window. This predicate gates all respawn paths.
        """
        if not pid:
            return False
        rec = (self.state.get('peer_update_state') or {}).get(pid)
        if not rec:
            return False
        ts = float(rec.get('last_graceful_attempt') or 0)
        if ts <= 0:
            return False
        # Holdoff covers the longest realistic KG tail so a peer
        # mid-upload of a multi-GB _full.gpkg still finishes before
        # we consider respawning it. Matches the 1800s gate used in
        # ``_orchestrate_stale_peer_updates`` for re-firing graceful
        # updates.
        return (time.time() - ts) < max(self.STALE_UPDATE_RETRY_GAP_S, 1800.0)

    def _orchestrate_stale_peer_updates(self, statuses: dict):
        """Re-trigger update on peers stuck on an old commit while idle.

        Only acts on peers that are reachable, on a stale commit, and
        whose processor is not actively running a KG (stopped / idle /
        complete). Never interrupts running work.

        Before retrying, ensures the director's local commit is pushed
        to origin/main — peers update via ``git fetch origin && reset
        --hard origin/main``, so an unpushed local commit causes peers
        to silently reset to the *previous* tip and report success.
        Whenever the director commit advances past the value seen on
        the previous tick, all per-peer attempt counters are reset so
        peers get a fresh round of retries (instead of being stuck on
        ``needs_manual_update`` from a previous, now-superseded commit).
        """
        if not _LOCAL_GIT_COMMIT or _LOCAL_GIT_COMMIT == 'unknown':
            return
        with self._lock:
            cfg = self.cfg.copy()
            tracked = dict(self.state.get('peer_update_state') or {})
            last_director_commit = self.state.get('peer_update_director_commit')
        # If the director's commit advanced since last tick, reset *attempt*
        # counters so peers get a fresh round (a new push effectively
        # invalidates the previous "manual required" verdict). But preserve
        # the graceful-update timing fields (``last_graceful_attempt``,
        # ``graceful_attempts``, ``first_seen_stale``) — wiping these caused
        # the 2026-05-09 incident where the active frontier was graceful-
        # kicked every 8 min mid-upload, with each rollout commit truncating
        # its own KG's Zenodo upload before the previous one finished. Hard
        # rollouts only act on idle peers, so resetting *their* attempt
        # counters is fine; graceful kicks already have their own per-peer
        # ``STALE_GRACEFUL_MAX_ATTEMPTS`` ceiling that should not reset on
        # every push.
        if last_director_commit and last_director_commit != _LOCAL_GIT_COMMIT:
            log.info('Director commit advanced %s -> %s; resetting peer update '
                     'attempt counters (preserving graceful timing)',
                     last_director_commit[:8], _LOCAL_GIT_COMMIT[:8])
            for _pid, _rec in list(tracked.items()):
                _rec['attempts'] = 0
                _rec.pop('last_attempt', None)
                _rec.pop('last_result', None)
                _rec.pop('needs_manual_update', None)
                tracked[_pid] = _rec
        # Make sure origin/main matches our local HEAD before we ask peers
        # to fetch+reset. Without this, peers reset to a stale origin and
        # silently stay behind.
        try:
            self._ensure_origin_synced()
        except Exception:
            log.exception('Origin push during stale-peer orchestration failed')
        now = time.time()
        live_ids = set()
        # Pre-classify so we can apply per-tick budgets across the
        # fleet rather than fire-and-forget on every stale peer.
        graceful_candidates: list[tuple[dict, dict, str]] = []  # (peer, rec, commit)
        hard_candidates: list[tuple[dict, dict, str, int]] = []  # (peer, rec, commit, attempts)
        for peer in cfg.get('peers', []):
            pid = peer.get('id')
            url = peer.get('url')
            if not pid or not url:
                continue
            ps = statuses.get(pid) or {}
            commit = (ps.get('git_commit') or '').strip()
            proc_state = ps.get('state', 'unknown')
            online = proc_state != 'unreachable' and not ps.get('error')
            # Idle = no active KG processing. 'running'/'processing'
            # means the peer is mid-work; we never interrupt that.
            idle = proc_state in ('stopped', 'idle', 'complete', 'paused')
            # "Ahead or equal" peers are NOT stale: their commit
            # contains _LOCAL_GIT_COMMIT in its history (e.g. operator
            # pushed a new commit to origin/main and the peer pulled
            # before this director restarted). Forcing them back would
            # be a downgrade.
            ahead = bool(commit) and _peer_commit_is_ahead_or_equal(commit)
            stale = bool(commit) and commit != _LOCAL_GIT_COMMIT and not ahead
            # Only drop the tracked record when we've *confirmed* the
            # peer landed on the local commit. If the peer is briefly
            # offline (e.g. during the graceful-restart window) or its
            # commit is unknown, keep the rec so debounce timers like
            # ``last_graceful_attempt`` survive the flicker. Without
            # this, a SIGTERM/respawn cycle wipes the 30-min debounce
            # and we re-fire graceful updates every tick — kicking the
            # peer mid-KG repeatedly without ever pulling the new
            # commit (the in-flight `_deferred_update` thread can't
            # see ``pgrep austria_processor.py`` go empty because the
            # cache-only orchestrator respawns the processor first).
            if commit and (commit == _LOCAL_GIT_COMMIT or ahead):
                # Ahead-or-equal: drop tracking + clear any leftover
                # ``needs_manual_update`` flag from before the director
                # noticed the peer was actually ahead.
                tracked.pop(pid, None)
                continue
            if not (online and stale):
                # Offline or unknown-commit: preserve rec for next tick.
                if pid in tracked:
                    live_ids.add(pid)
                continue
            live_ids.add(pid)
            rec = dict(tracked.get(pid) or {})
            prev_commit = rec.get('commit')
            # Peer's commit advanced since last tick (manual update or
            # an out-of-band pull): wipe the auto-retry counters and
            # clear the manual-update flag. Stays stale (still != local)
            # so we'll let the auto path fire a fresh round of retries
            # if needed; but the operator no longer sees the scary
            # "manual update needed" tag for a peer that has clearly
            # been touched.
            if prev_commit and commit and prev_commit != commit:
                log.info('Peer %s commit advanced %s -> %s; resetting '
                         'auto-retry state', pid, prev_commit[:8], commit[:8])
                rec['attempts'] = 0
                rec['graceful_attempts'] = 0
                rec['first_seen_stale'] = now
                rec.pop('needs_manual_update', None)
                rec.pop('last_attempt', None)
                rec.pop('last_result', None)
            rec.setdefault('first_seen_stale', now)
            rec.setdefault('attempts', 0)
            rec['commit'] = commit
            rec['last_state'] = proc_state
            if not idle:
                rec['waiting_for_idle'] = True
                last_graceful = float(rec.get('last_graceful_attempt') or 0)
                gattempts = int(rec.get('graceful_attempts') or 0)
                if gattempts >= self.STALE_GRACEFUL_MAX_ATTEMPTS:
                    rec['needs_manual_update'] = True
                    tracked[pid] = rec
                    continue
                # Don't graceful-kick a peer that is mid-finalization for
                # the current KG — GPKG build / validation / Zenodo
                # uploads can each take many minutes for large KGs and a
                # SIGTERM here forces the next run to redo all of them
                # (see 2026-05-09 incident: at43 90107-west truncated
                # 5 times in 8 h, manifest stalled). The processor checks
                # ``_shutdown_requested`` only between KGs, so once it
                # has finished the heavy raster steps it will exit
                # naturally at the next KG boundary anyway. Letting the
                # current KG complete its full upload chain (full_gpkg,
                # light_gpkg, json) is always cheaper than re-running.
                step = ((ps.get('current_kg') or {}).get('step') or '').lower()
                FINALIZE_STEPS = (
                    'gpkg_full', 'validate_full_gpkg', 'upload_full_gpkg',
                    'gpkg_light', 'validate_light_gpkg', 'upload_light_gpkg',
                    'json', 'upload_json', 'upload', 'disk_cleanup',
                )
                if any(step.startswith(s) for s in FINALIZE_STEPS):
                    log.info('Skip graceful kick on %s: mid-finalization '
                             '(step=%s) — letting current KG upload chain '
                             'finish before restart', pid, step)
                    tracked[pid] = rec
                    continue
                if (now - last_graceful) >= 1800:
                    graceful_candidates.append((peer, rec, commit))
                else:
                    tracked[pid] = rec
                continue
            rec.pop('waiting_for_idle', None)
            attempts = int(rec.get('attempts') or 0)
            last_attempt = float(rec.get('last_attempt') or 0)
            # 12-hour amnesty: if the peer has been flagged as
            # "manual update needed" for too long (still stale, still
            # idle, but reachable), reset counters and try again. A
            # successful auto-update meanwhile clears the rec entirely
            # in the per-peer loop above, so this branch only fires
            # when the peer remained genuinely stuck.
            if (attempts >= self.STALE_UPDATE_MAX_ATTEMPTS and
                    last_attempt > 0 and
                    (now - last_attempt) >= self.STALE_UPDATE_RESET_AFTER_S):
                log.info('Resetting stale-update counters for %s after %.1fh '
                         '(was needs_manual_update=%s, attempts=%d)',
                         pid, (now - last_attempt) / 3600.0,
                         rec.get('needs_manual_update'), attempts)
                rec['attempts'] = 0
                rec.pop('needs_manual_update', None)
                rec.pop('last_attempt', None)
                rec.pop('last_result', None)
                rec['graceful_attempts'] = 0
                rec['first_seen_stale'] = now
                attempts = 0
                last_attempt = 0.0
            if attempts >= self.STALE_UPDATE_MAX_ATTEMPTS:
                if (now - last_attempt) >= self.STALE_UPDATE_RETRY_GAP_S:
                    rec['needs_manual_update'] = True
                tracked[pid] = rec
                continue
            if attempts == 0:
                ready = (now - float(rec['first_seen_stale'])) >= self.STALE_UPDATE_GRACE_S
            else:
                ready = (now - last_attempt) >= self.STALE_UPDATE_RETRY_GAP_S
            if not ready:
                tracked[pid] = rec
                continue
            hard_candidates.append((peer, rec, commit, attempts))

        # Stable ordering: oldest "first_seen_stale" first, so peers
        # waiting longest get serviced first. Tie-break by id.
        graceful_candidates.sort(
            key=lambda t: (float(t[1].get('first_seen_stale') or 0),
                           t[0]['id']))
        hard_candidates.sort(
            key=lambda t: (float(t[1].get('first_seen_stale') or 0),
                           t[0]['id']))

        graceful_budget = self.STALE_UPDATE_GRACEFUL_PER_TICK
        hard_budget = self.STALE_UPDATE_HARD_PER_TICK
        graceful_done: list[str] = []
        hard_done: list[str] = []
        for peer, rec, commit in graceful_candidates:
            pid = peer['id']
            if graceful_budget <= 0:
                tracked[pid] = rec  # try again next tick
                continue
            gattempts = int(rec.get('graceful_attempts') or 0)
            _msg = (
                'graceful update → ' + str(_LOCAL_GIT_COMMIT)
                + ' (peer on ' + str(commit)
                + '; attempt ' + str(gattempts + 1) + '/'
                + str(self.STALE_GRACEFUL_MAX_ATTEMPTS) + ')'
            )
            log.info('Stale peer %s mid-KG (%s on %s); sending '
                     'graceful update to schedule restart at KG boundary '
                     '(attempt %d/%d)',
                     pid, rec.get('last_state'), commit,
                     gattempts + 1, self.STALE_GRACEFUL_MAX_ATTEMPTS)
            _emit_director_event(_msg, peer=pid)
            try:
                gres = trigger_peer_update(peer['url'], graceful=True)
            except Exception as e:
                gres = {'error': str(e)}
            rec['last_graceful_attempt'] = now
            rec['last_graceful_result'] = gres
            rec['graceful_attempts'] = gattempts + 1
            tracked[pid] = rec
            graceful_budget -= 1
            graceful_done.append(pid)
        for peer, rec, commit, attempts in hard_candidates:
            pid = peer['id']
            if hard_budget <= 0:
                tracked[pid] = rec  # try again next tick
                continue
            log.info('Auto-retry update on stale peer %s '
                     '(commit=%s, attempt=%d)',
                     pid, commit, attempts + 1)
            _emit_director_event(
                'hard update → ' + str(_LOCAL_GIT_COMMIT)
                + ' (peer on ' + str(commit)
                + '; attempt ' + str(attempts + 1) + ')',
                peer=pid)
            try:
                res = trigger_peer_update(peer['url'], graceful=False)
            except Exception as e:
                res = {'error': str(e)}
            rec['attempts'] = attempts + 1
            rec['last_attempt'] = now
            rec['last_result'] = res
            tracked[pid] = rec
            hard_budget -= 1
            hard_done.append(pid)
        if graceful_done or hard_done:
            log.info('Stale-peer rollout this tick: %d graceful (%s), '
                     '%d hard (%s); deferred=%d/%d',
                     len(graceful_done), ','.join(graceful_done) or '-',
                     len(hard_done), ','.join(hard_done) or '-',
                     max(0, len(graceful_candidates) - len(graceful_done)),
                     max(0, len(hard_candidates) - len(hard_done)))
        # Drop only records for peers that have either confirmed-updated
        # or been gone long enough that any pending graceful-update is
        # no longer relevant. ``live_ids`` includes peers we kept across
        # an offline blip, so they survive here. Peers that disappeared
        # entirely from the config still get cleaned up.
        cfg_ids = {p.get('id') for p in cfg.get('peers', []) if p.get('id')}
        for pid in list(tracked.keys()):
            if pid not in cfg_ids:
                tracked.pop(pid, None)
        with self._lock:
            self.state['peer_update_state'] = tracked
            self.state['peer_update_director_commit'] = _LOCAL_GIT_COMMIT

    def _ensure_origin_synced(self) -> None:
        """Push local main to origin if it's ahead.

        Peers update via ``git fetch origin && reset --hard origin/main``,
        so the director must keep origin in sync with its own HEAD.
        Cheap to call every tick: a no-op when origin is already current
        (single ``git rev-parse`` + ``git rev-list --count``).
        """
        import subprocess as sp
        repo = str(Path(__file__).parent)
        try:
            local = sp.run(['git', 'rev-parse', 'main'],
                           capture_output=True, text=True, timeout=5,
                           cwd=repo).stdout.strip()
            if not local:
                return
            # NOTE: do NOT gate on local == _LOCAL_GIT_COMMIT.
            # _LOCAL_GIT_COMMIT is frozen at srv-import time; commits
            # landed via shelley/git after srv started will make local
            # main strictly ahead, and we still want them on origin so
            # peers can pick them up. The previous gate caused a 12 h
            # outage where 8 unpushed commits left peers stuck on the
            # pre-multipart zenodo_client and frontier uploads failing.
            ahead = sp.run(['git', 'rev-list', '--count',
                            'origin/main..main'],
                           capture_output=True, text=True, timeout=5,
                           cwd=repo).stdout.strip()
            if ahead and int(ahead) > 0:
                log.info('Origin behind by %s commit(s); pushing main', ahead)
                push = sp.run(['git', 'push', 'origin', 'main'],
                              capture_output=True, text=True, timeout=60,
                              cwd=repo)
                if push.returncode != 0:
                    log.warning('git push failed (rc=%d): %s',
                                push.returncode,
                                (push.stdout + push.stderr)[-300:])
        except Exception as e:
            log.warning('_ensure_origin_synced error: %s', e)

    def _orchestrate_cache_only(self):
        """Start/stop cache-only peers around the frontier peer.

        Rules:
          * keep at least ``min_reserve`` enabled peers idle (never started).
          * only enabled, non-scheduled peers with role=='cache_only' are
            considered.  If no peers are explicitly tagged cache_only,
            pick from idle frontier peers (their primary role is still
            frontier; we just borrow them for cache-only work).
          * never touch the active frontier peer.
          * cap concurrent cache-only peers at MAX_CACHE_ONLY_PEERS.
          * each cache-only peer gets a whitelist of fully-cached KGs.
          * cache-only peers without enough whitelist work are stopped
            so they free their slot.
        """
        with self._lock:
            mode = self.state.get('mode', 'auto')
            cfg = self.cfg.copy()
            state_copy = self.state.copy()
        if mode == 'paused':
            return

        budget_bytes = cfg.get('budget_gb', BANDWIDTH_BUDGET_GB) * (1024 ** 3)
        min_reserve = int(cfg.get('min_reserve_peers', MIN_RESERVE_PEERS))
        max_cache_only = int(cfg.get('max_cache_only_peers', MAX_CACHE_ONLY_PEERS))
        # Server-friendliness throttle (see _capacity_factor docstring).
        # Cache-only peers hit BEV + Zenodo, so this is exactly the knob
        # to turn down when those servers complain. We *do* allow zero
        # here — when servers are really angry, idling cache-only peers
        # is the right move (the active frontier still makes progress).
        _factor = float(state_copy.get('capacity_factor',
                                        self._capacity_ema))
        _max_cache_only_full = max_cache_only
        max_cache_only = max(0, int(round(max_cache_only * _factor)))
        if max_cache_only < _max_cache_only_full:
            log.info(
                'capacity factor %.2f → max_cache_only_peers %d → %d',
                _factor, _max_cache_only_full, max_cache_only,
            )


        active_frontier = state_copy.get('active_peer')
        # Peers the parallel-frontier orchestrator has authorised /
        # plans to run as additional frontiers. Cache-only must NOT
        # touch them, otherwise we ping-pong: parallel orch starts a
        # peer with creds+strips, 18s later cache-only orch hard-stops
        # and re-starts it as cache-only, then parallel re-promotes
        # the next tick. Net: only the active peer ever runs frontier.
        parallel_authorised: set[str] = set(
            state_copy.get('parallel_frontiers_active') or [])
        parallel_authorised |= set(
            (state_copy.get('frontier_cred_plan') or {}).keys())
        peers = list(cfg.get('peers', []))

        # Currently running cache-only peers (by id).
        running_cache_only: list[str] = []
        # Eligible candidate peers (enabled, not scheduled, not the
        # active frontier, has bandwidth, online).
        candidates: list[dict] = []
        unreachable = 0
        for p in peers:
            pid = p['id']
            if not p.get('enabled', True):
                continue
            if _peer_is_scheduled(p):
                continue
            if pid == active_frontier:
                continue
            if pid in parallel_authorised:
                # Owned by the parallel-frontier orchestrator — leave
                # alone, otherwise we ping-pong frontier↔cache-only.
                continue
            if p.get('reserved_kg'):
                # Holds a frontier-only reservation — leave alone.
                continue
            bw = state_copy.get('peer_bandwidth', {}).get(pid, {})
            used = bw.get('used_bytes', 0)
            if (_peer_budget_bytes(p, cfg) - used) < 2 * (1024 ** 3):
                continue
            ps = get_peer_status(p.get('url'))
            st = ps.get('state', 'unknown')
            if st == 'unreachable':
                unreachable += 1
                continue
            role = self._peer_role(p)
            running = st in ('running', 'processing')
            is_cache_only_run = bool(ps.get('cache_only'))
            # Stale cached status — don't make stop/start decisions on
            # potentially out-of-date information. Treat as 'leave alone'.
            stale = bool(ps.get('_stale'))
            if running and is_cache_only_run and not stale:
                running_cache_only.append(pid)
            candidates.append({'peer': p, 'role': role, 'state': st,
                               'is_cache_only_run': is_cache_only_run,
                               'stale': stale})

        # If we're over the cap (e.g. capacity factor dropped, or peers
        # were started before the cap shrank), gracefully stop the
        # excess. Pick the noisiest peers first — they're the ones
        # we want off the upstreams anyway. The graceful stop lets them
        # finish the current KG; we keep the count unchanged so we
        # don't start a fresh peer in the same tick to replace one
        # we just told to drain.
        # CPU-steal damping. exe.dev steal is almost entirely from
        # other tenants on the shared hypervisor pool, not from us —
        # so reducing peer count rarely recovers cycles 1:1. We still
        # damp the ramp *slightly* at very-high steal so:
        #   (a) we don't pile new starts on top of a wedged pool, and
        #   (b) the LPT partition has a chance to drain the heaviest
        #       KGs from the worst peers before more land.
        # The damping curve is intentionally gentle: 0.85 at steal=40%,
        # 0.70 at steal=60%, 0.55 floor. The LPT partition itself does
        # the *real* load-balancing per-peer; this is just a fleet-wide
        # ramp brake.
        try:
            _med_steal = _fleet_steal_median(list(running_cache_only))
            if _med_steal is not None and _med_steal >= 40.0:
                _cpu_cap_factor = max(0.55,
                                       1.0 - (_med_steal - 30.0) / 200.0)
                _prev_cap = max_cache_only
                max_cache_only = max(0, int(round(
                    max_cache_only * _cpu_cap_factor)))
                if max_cache_only < _prev_cap:
                    log.info(
                        'fleet steal median %.0f%% (n=%d) → cpu_factor '
                        '%.2f → max_cache_only_peers %d → %d',
                        _med_steal, len(running_cache_only),
                        _cpu_cap_factor, _prev_cap, max_cache_only,
                    )
        except Exception:
            pass

        excess = len(running_cache_only) - max_cache_only
        if excess > 0:
            # Rank running peers by noise score (highest first), then by
            # id for stability. Bias toward stopping recently-started
            # peers if scores tie (fresh starts are cheap to forfeit).
            noisy_first = sorted(
                running_cache_only,
                key=lambda pid: (-_peer_noise_score(pid, state_copy), pid),
            )
            to_stop = noisy_first[:excess]
            log.info(
                'cache-only over cap: %d running > %d max — gracefully '
                'stopping %d (noisiest first): %s',
                len(running_cache_only), max_cache_only, excess, to_stop,
            )
            # Don't re-send graceful stop within 60s — the peer's signal
            # handler sets _shutdown_requested=True but only exits after
            # finishing its current KG. Re-sending every director tick
            # spams the peer's log with SIGTERM warnings (we saw 63 in 36h
            # on at32 with only 1 actual restart). The first stop is
            # enough; let the peer drain in peace.
            now_ts = time.time()
            stop_skipped = []
            for pid in to_stop:
                last = _LAST_GRACEFUL_STOP_TS.get(pid, 0)
                if now_ts - last < 60:
                    stop_skipped.append(pid)
                    continue
                p = get_peer_by_id(cfg, pid)
                if not p:
                    continue
                try:
                    stop_peer_processor(p.get('url'), graceful=True)
                    _LAST_GRACEFUL_STOP_TS[pid] = now_ts
                except Exception as e:
                    log.warning('Graceful stop excess cache-only %s failed: %s',
                                pid, e)
            if stop_skipped:
                log.debug('Graceful stop already in flight for %d peers (<60s ago): %s',
                          len(stop_skipped), stop_skipped)
            # Deliberately *don't* shrink running_cache_only — a graceful
            # stop only takes effect when the peer finishes its current
            # KG. Until then it's still consuming an upstream slot, so
            # max_add must stay clamped to zero or negative.
            # Also drop them from `to_start` candidate pools later.
            _draining_cache_only = set(to_stop)
        else:
            _draining_cache_only = set()

        # Compute reserve target.  Reserve peers must be enabled+online
        # but idle.  Count idle-eligible peers in `candidates`.
        # 'complete' is a terminal post-run state set when a peer's
        # processor exited cleanly after finishing its last KG; it's
        # just as idle as 'stopped' from our perspective. Without
        # including it here, peers that finished a cache-only KG
        # would never be re-eligible for the next one, and the fleet
        # would drain to ~1 running cache-only peer (matches the
        # observed '1/24 cache' state).
        idle_eligible = [c for c in candidates
                         if c['state'] in ('idle', 'stopped', 'complete',
                                            'unknown')]

        # Total enabled peers excluding scheduled-out (e.g. not_before in
        # the future, like a primary that's been parked until the next
        # bandwidth cycle). Without this, the reserve math wastes slots
        # on peers we can't ever activate.
        total_enabled = sum(1 for p in peers
                            if p.get('enabled', True)
                            and not _peer_is_scheduled(p))
        # We must keep `min_reserve` peers idle.  Subtract running peers
        # (frontier active + already-running cache-only) from total.
        running_count = (1 if active_frontier else 0) + len(running_cache_only)
        # Maximum we may add such that idle remaining ≥ min_reserve.
        # idle_after_add = total_enabled - running_count - add
        # We want idle_after_add ≥ min_reserve.
        slack = max(0, total_enabled - running_count - min_reserve)
        max_add = min(slack, max_cache_only - len(running_cache_only))

        # Compute cache-ready whitelist (cheap due to caching).
        whitelist = self._compute_cache_ready_kgs()
        if not whitelist:
            # Nothing to do for cache-only peers.  Stop any that are
            # running (they'd otherwise idle-loop a fresh subprocess).
            # Use a graceful stop so the peer finishes its current KG
            # before exiting — cache-only work is pure CPU + Zenodo
            # upload, no Copernicus credentials at risk, so there's no
            # reason to kill mid-KG and waste the work.  If the
            # whitelist refills before the peer drains, the next tick
            # will see it still running and let it carry on.
            if running_cache_only:
                log.info('No cache-ready KGs — gracefully stopping %d running '
                         'cache-only peers (will exit after current KG)',
                         len(running_cache_only))
                for pid in running_cache_only:
                    p = get_peer_by_id(cfg, pid)
                    if p:
                        try:
                            stop_peer_processor(p.get('url'), graceful=True)
                        except Exception as e:
                            log.warning('Graceful stop cache-only %s failed: %s', pid, e)
            return

        if max_add <= 0 and not running_cache_only:
            # No room to add and none running — nothing to do.
            return

        # Choose new peers to start.  Prefer peers explicitly tagged
        # role=='cache_only'; fall back to idle frontier peers.
        # Within each group prefer quiet peers (low warning fingerprint)
        # so fresh / clean peers absorb load while noisy ones cool off.
        def _noise_key(c):
            return (_peer_noise_score(c['peer']['id'], state_copy),
                    c['peer']['id'])
        idle_cache_only = sorted(
            [c for c in idle_eligible if c['role'] == 'cache_only'],
            key=_noise_key)
        idle_frontier = sorted(
            [c for c in idle_eligible if c['role'] != 'cache_only'],
            key=_noise_key)
        # Ramp limiter: scale starts/tick by capacity_factor so we add
        # peers gradually when upstreams are unhappy. At factor=1.0 we
        # add up to RAMP_MAX_STARTS_PER_TICK; at THROTTLE_MIN_FACTOR
        # only RAMP_MIN_STARTS_PER_TICK. The cap shrinks max_add but the
        # next tick (~30s later) will fill any remaining slots.
        #
        # Deficit override: if the running fleet is well below cap
        # (e.g. after a fleet-wide graceful-update wave drained
        # cache-only peers at KG boundary), the 1-3 starts/tick cap
        # can't keep up with the drain rate (5 graceful/tick from
        # ``_orchestrate_stale_peer_updates``). Without an override
        # the chart drops to 0/24 cache and stays there for ~10 min
        # after every srv restart that advances the director commit.
        # When deficit > ramp_cap we lift the cap so refill matches
        # drain. Throttle still applies via ``max_cache_only`` itself
        # (already scaled by capacity_factor above).
        ramp_span = RAMP_MAX_STARTS_PER_TICK - RAMP_MIN_STARTS_PER_TICK
        ramp_cap = RAMP_MIN_STARTS_PER_TICK + int(round(ramp_span * _factor))
        ramp_cap = max(RAMP_MIN_STARTS_PER_TICK, ramp_cap)
        deficit = max_cache_only - len(running_cache_only)
        if deficit > ramp_cap and max_add > ramp_cap:
            log.info(
                'cache-only deficit %d > ramp_cap %d (running=%d/%d) '
                '— lifting ramp cap for this tick',
                deficit, ramp_cap, len(running_cache_only),
                max_cache_only,
            )
            # Cap at the deficit itself so we don't overshoot when
            # drain catches up. ``max_add`` was already clamped by
            # ``slack`` (reserve) and (max_cache_only - running),
            # so this can't push us past the cap.
            ramp_cap = min(deficit, max_add)
        max_add = min(max_add, ramp_cap)
        # Skip peers with a pending graceful update (see
        # ``_has_pending_graceful_update`` docstring). Their
        # ``/admin/update?graceful=1`` deferred-update thread waits for
        # ``pgrep austria_processor.py`` to be empty before pulling +
        # restarting srv. If we respawn the processor here, that pgrep
        # never empties and the peer is stuck on the stale commit
        # forever — while we kick it mid-KG every retry window from
        # the stale-peer orchestrator.
        to_start = []
        _skipped_pending: list[str] = []
        for c in idle_cache_only + idle_frontier:
            if len(to_start) >= max_add:
                break
            pid = c['peer'].get('id')
            if pid and self._has_pending_graceful_update(pid):
                _skipped_pending.append(pid)
                continue
            to_start.append(c['peer'])
        if _skipped_pending:
            log.info('cache-only: skipping %d peer(s) with pending graceful '
                     'update (waiting for git pull + srv restart): %s',
                     len(_skipped_pending), ','.join(_skipped_pending))

        # --- Build claimed-KG set so no two cache-only peers (and not
        # the frontier) target the same KG.  We pull each peer's current
        # KG from the status snapshot we already gathered above, plus
        # the active frontier's current_kg.
        #
        # Split-KG awareness: a peer mid-work on block ``60336-northwest``
        # also blocks the parent ``60336`` (cache-only whitelist contains
        # parent KG codes from kg_list.json, never block codes), so we
        # always add the parent for any block we see.
        import re as _re
        in_progress: set[str] = set()
        def _mark(code):
            if not code:
                return
            s = str(code)
            in_progress.add(s)
            m = _re.match(r'^(\d+)-[a-z][-a-z0-9]*$', s)
            if m:
                in_progress.add(m.group(1))
        for c in candidates:
            ps_pid = c['peer']['id']
            try:
                ps2 = get_peer_status(c['peer'].get('url'))
            except Exception:
                ps2 = {}
            # Only mark KGs from peers whose processor is actually
            # running/paused right now. ``current_kg`` lingers in
            # progress.json after SIGTERM with state='stopped' (e.g.
            # after a fleet-wide update_peers wave) — using those stale
            # codes drains the whitelist (``30 in-progress KGs > 15
            # ready`` -> empty after filter -> 0/60 cache-only running).
            st2 = ps2.get('state')
            if st2 not in ('running', 'processing', 'paused',
                            'paused_zenodo'):
                continue
            ckg = (ps2.get('current_kg') or {})
            _mark(ckg.get('code') if isinstance(ckg, dict) else None)
            _mark(ps2.get('in_progress'))
            del ps_pid  # silence linter
        # Frontier's KG — both the primary active frontier AND every
        # parallel-authorised frontier. Earlier we excluded those peers
        # from ``candidates`` (they're owned by the parallel orch), so
        # their current_kg never reached _mark — meaning the cache-only
        # whitelist filter couldn't see them, and a cache-only peer
        # could be assigned the same KG a parallel frontier was already
        # processing (observed: at60 on 72321 Mitteregg for 20h42m, then
        # at17 handed 72321 as its cache-only slice). Iterate both sets,
        # de-duplicated, to close that hole.
        frontier_ids = set()
        if active_frontier:
            frontier_ids.add(active_frontier)
        frontier_ids |= parallel_authorised
        for fid in frontier_ids:
            fp = get_peer_by_id(cfg, fid)
            if not fp:
                continue
            try:
                fps = get_peer_status(fp.get('url'))
                fst = fps.get('state')
                if fst in ('running', 'processing', 'paused',
                            'paused_zenodo'):
                    fkg = (fps.get('current_kg') or {})
                    _mark(fkg.get('code') if isinstance(fkg, dict) else None)
                    _mark(fps.get('in_progress'))
            except Exception:
                pass
        # Filter the whitelist before partitioning.
        whitelist = [k for k in whitelist if k not in in_progress]
        if not whitelist:
            log.info('Cache-only orchestrate: whitelist drained after excluding '
                     '%d in-progress KGs', len(in_progress))
            return

        # --- Expand parent KG codes into block codes for split KGs.
        # Each peer independently runs maybe_split_kg() at startup, but
        # they all start at block 0 — so two peers given the same parent
        # code race the first block (e.g. both grabbing 92117-west).
        # Pre-expanding here means the stride partition below hands
        # different blocks of e.g. Ramsau to different peers.
        # Honors strikes (adaptive split for repeat-failed KGs that are
        # otherwise quick from cache).
        try:
            from kg_splitter import maybe_split_kg, is_block_code
            kg_list_path = DATA_DIR / 'kg_list.json'
            kg_by_code = {}
            if kg_list_path.exists():
                for kg in json.loads(kg_list_path.read_text()):
                    c = kg.get('kg_code')
                    if c:
                        kg_by_code[c] = kg
            expanded: list[str] = []
            n_split = 0
            for code in whitelist:
                if is_block_code(code):
                    # Already a block code (e.g. queued explicitly) —
                    # honor as-is, skip in_progress siblings.
                    if code not in in_progress:
                        expanded.append(code)
                    continue
                kg = kg_by_code.get(code)
                if not kg:
                    expanded.append(code)
                    continue
                blocks = maybe_split_kg(kg)
                if len(blocks) > 1:
                    n_split += 1
                    for blk in blocks:
                        bc = blk['kg_code']
                        # Drop sibling blocks already claimed by some peer.
                        if bc in in_progress:
                            continue
                        expanded.append(bc)
                else:
                    expanded.append(code)
            if n_split:
                log.info('Cache-only whitelist: expanded %d parent KGs into '
                         'blocks (%d → %d codes) for cross-peer distribution',
                         n_split, len(whitelist), len(expanded))
            whitelist = expanded
            if not whitelist:
                log.info('Cache-only orchestrate: whitelist empty after block expansion')
                return
        except Exception as e:
            log.warning('Block expansion failed (continuing with parent codes): %s', e)

        # --- Complexity-weighted, capacity-aware LPT partition.
        #
        # Old behaviour (kept here for context): stride by sorted peer
        # id (``whitelist[i::n]``). Disjoint and stable, but blind to
        # KG size *and* peer strength — so a 28-tile KG and a 1-tile KG
        # were equally likely to land on a 80 %-steal peer as a 0 %-
        # steal one. With exe.dev pools regularly delivering 30+ peers
        # at ≥15 %% steal, that asymmetry costs hours of wall time per
        # day.
        #
        # New behaviour: Longest-Processing-Time (LPT) bin-packing.
        # Sort the whitelist by weight (≈ tile count) descending, then
        # for each KG pick the worker minimising ``load[p] / capacity[p]``
        # where capacity = max(0.10, 1 - steal_ewma/100). Stable across
        # ticks because weights + capacities change slowly; sticky-
        # ownership pass below still wins for in-flight KGs.
        all_workers = sorted((set(running_cache_only) |
                              {p['id'] for p in to_start})
                             - _draining_cache_only)
        if not all_workers:
            return
        weights = _kg_weights(whitelist)
        capacities = {pid: _peer_cpu_capacity(pid) for pid in all_workers}
        load: dict[str, float] = {pid: 0.0 for pid in all_workers}
        slices: dict[str, list] = {pid: [] for pid in all_workers}
        # Sort by weight desc, then code for determinism. Codes with
        # equal weight then land round-robin which preserves the
        # diversification property.
        ordered_codes = sorted(whitelist,
                                key=lambda c: (-weights.get(c, 1.0), c))
        for code in ordered_codes:
            w = weights.get(code, 1.0)
            # Pick the worker with smallest projected normalised load.
            # Tiebreak on raw load, then peer id, for determinism.
            chosen = min(
                all_workers,
                key=lambda pid: (
                    (load[pid] + w) / max(capacities[pid], 0.01),
                    load[pid],
                    pid,
                ),
            )
            slices[chosen].append(code)
            load[chosen] += w
        # Log a summary so we can verify diversification in journalctl.
        try:
            nonzero = [v for v in load.values() if v > 0]
            mx = max(load.values()) if load else 0.0
            mn_nz = min(nonzero) if nonzero else 0.0
            n_used = len(nonzero)
            log.info(
                'cache-only LPT partition: workers=%d (used=%d) kgs=%d '
                'weight_total=%.0f per_used min=%.0f max=%.0f '
                'imbalance=%.2f',
                len(all_workers), n_used, len(whitelist),
                sum(load.values()), mn_nz, mx,
                (mx / max(mn_nz, 0.01)) if mn_nz else 1.0,
            )
        except Exception:
            pass

        # --- Sticky ownership: honor prior-tick assignments so a KG
        # that's already in flight on peer P doesn't get reassigned to
        # peer Q just because the stride math shifted (peer joined/left
        # all_workers, whitelist length changed, etc.). Without this, two
        # peers can be told to run the same KG within a few ticks of
        # each other; we observed at25 and at26 both processing 72321
        # Mitteregg.
        #
        # The in_progress filter above only catches duplicates *after*
        # the second peer's status reflects current_kg — there's a small
        # race window between PUT-queue and current_kg being visible.
        # Sticky ownership closes that window deterministically.
        sticky_ttl = 600.0  # seconds; covers a long-running KG comfortably
        sticky_now = time.time()
        prior = self.state.get('cache_only_assigned') or {}
        owner: dict[str, str] = {}
        worker_set = set(all_workers)
        for pid_p, info in prior.items():
            if pid_p not in worker_set:
                continue  # peer no longer eligible — release its claims
            ts = info.get('ts', 0)
            if (sticky_now - ts) > sticky_ttl:
                continue  # stale — release
            for kg in info.get('slice', []) or []:
                # First-write wins; deterministic because prior is a dict
                # keyed by pid, but iteration order is insertion order so
                # we sort to be safe.
                owner.setdefault(kg, pid_p)
        if owner:
            fixed = {pid: [kg for kg in chunk
                           if owner.get(kg, pid) == pid]
                     for pid, chunk in slices.items()}
            stolen = sum(len(slices[p]) - len(fixed[p]) for p in slices)
            if stolen:
                log.info('Cache-only sticky ownership: kept %d KGs with '
                         'their prior owners (stride wanted to reshuffle)',
                         stolen)
            slices = fixed

        # Start new peers with their slice.
        for p in to_start:
            chunk = slices.get(p['id'], [])
            if not chunk:
                continue
            log.info('Starting cache-only peer %s with %d KGs (LPT %d/%d, '
                     '%d total ready)', p['id'], len(chunk),
                     all_workers.index(p['id']) + 1, len(all_workers),
                     len(whitelist))
            try:
                start_peer_processor(p.get('url'), cache_only=True,
                                     queue_whitelist=chunk)
            except Exception as e:
                log.warning('Start cache-only on %s failed: %s', p['id'], e)

        # Re-sync slices to already-running peers (whitelist may have
        # grown/shrunk since they started).  Cheap PUT.  Skip peers we
        # asked to drain — we don't want to feed them more work.
        for pid in running_cache_only:
            if pid in _draining_cache_only:
                continue
            p = get_peer_by_id(cfg, pid)
            if not p:
                continue
            chunk = slices.get(pid, [])
            if not chunk:
                continue
            try:
                _push_queue_to_peer(p['url'], chunk)
            except Exception as e:
                log.debug('Resync queue to %s failed: %s', pid, e)

        # Persist sticky ownership for next tick. Only record peers that
        # actually got a non-empty slice this tick — empty-slice peers
        # have no claims to defend.
        new_assigned = {pid: {'slice': chunk, 'ts': sticky_now}
                        for pid, chunk in slices.items() if chunk}

        # Status accounting
        with self._lock:
            self.state['cache_only_active'] = list(set(
                running_cache_only + [p['id'] for p in to_start]))
            self.state['cache_only_assigned'] = new_assigned
            save_director_state(self.state)

    def _prune_priority_queue(self) -> int:
        """Drop stale codes from the local retry_queue.json.

        The primary VM is disabled (never runs the processor), so the
        in-processor remover (`_remove_from_retry_queue` in
        ``austria_processor.py``) never fires there — leaving the
        file append-only.  Manifest entries / local JSONs from peers
        do get synced back to the primary, but the queue keeps the
        stale codes around forever and re-pushes them to peers next
        tick.  Even with ``skip_processed=True`` on the peer side that
        creates rare races where a peer slightly behind on manifest
        sync re-processes a finished KG.

        We prune codes that are either:
          * present in `_get_completed_kgs()` (locally finished),
          * a parent KG whose blocks all have ``_json`` in the manifest
            (block expansion is implicit at processor startup).

        Tombstoned codes (force-requeue) are exempt — they survive in
        the queue by design.
        """
        queue_path = DATA_DIR / 'retry_queue.json'
        try:
            queue = json.loads(queue_path.read_text()) if queue_path.exists() else []
        except Exception:
            return 0
        if not queue:
            return 0
        try:
            from app import _get_completed_kgs, _tombstone_path  # type: ignore
            completed = _get_completed_kgs()
        except Exception:
            completed = set()
            _tombstone_path = None  # type: ignore
        tombstoned: set = set()
        try:
            if _tombstone_path and _tombstone_path.exists():  # type: ignore
                import re as _re
                tdata = json.loads(_tombstone_path.read_text())  # type: ignore
                if isinstance(tdata, dict):
                    for tk in tdata.keys():
                        m = _re.match(r'^(\d+(?:-[a-z][-a-z0-9]*)?)_', tk)
                        if m:
                            tombstoned.add(m.group(1))
        except Exception:
            pass
        # Parent→done check via kg_splitter, mirroring app.py's GET
        # handler so on-disk pruning is consistent with what the
        # dashboard already does at view time.
        _parent_done = None
        try:
            import search_index as _si
            from kg_splitter import (maybe_split_kg, all_block_codes_for_parent,
                                     is_block_code)
            _conn = _si.get_index()._conn()

            def _parent_done(code: str) -> bool:
                if is_block_code(code):
                    return False
                row = _conn.execute(
                    'SELECT min_lon, min_lat, max_lon, max_lat, kg_name '
                    'FROM kg WHERE kg_code=?', (code,)).fetchone()
                if not row or row['min_lon'] is None:
                    return False
                fake_kg = {'kg_code': code, 'kg_name': row['kg_name'],
                           'bbox': {'min_lon': row['min_lon'],
                                    'min_lat': row['min_lat'],
                                    'max_lon': row['max_lon'],
                                    'max_lat': row['max_lat']}}
                blocks = maybe_split_kg(fake_kg)
                if len(blocks) <= 1:
                    return False
                done = all_block_codes_for_parent(code, completed)
                return len(done) >= len(blocks)
        except Exception:
            _parent_done = None

        def _is_stale(c: str) -> bool:
            if c in tombstoned:
                return False
            if c in completed:
                return True
            if _parent_done and _parent_done(c):
                return True
            return False
        kept = [c for c in queue if not _is_stale(c)]
        dropped = len(queue) - len(kept)
        if dropped <= 0:
            return 0
        try:
            tmp = queue_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(kept))
            tmp.replace(queue_path)
            log.info('prune retry_queue: dropped %d stale code(s); %d remain',
                     dropped, len(kept))
        except Exception as _e:
            log.warning('prune retry_queue write failed: %s', _e)
            return 0
        return dropped

    def _sync_queue_to_active(self):
        """Push the director's local priority queue to the active remote peer.

        Only syncs if the queue has changed since the last sync (compares hash).
        """
        with self._lock:
            active_id = self.state.get('active_peer')
        if not active_id:
            return
        peer = get_peer_by_id(self.cfg, active_id)
        if not peer or peer.get('url') is None:
            return  # local peer doesn't need sync

        # Read current queue
        queue_path = DATA_DIR / 'retry_queue.json'
        try:
            queue = json.loads(queue_path.read_text()) if queue_path.exists() else []
        except Exception:
            queue = []

        # Hash check — skip if unchanged
        import hashlib
        q_hash = hashlib.md5(json.dumps(queue).encode()).hexdigest()
        with self._lock:
            if self.state.get('_last_queue_hash') == q_hash:
                return

        # Push to peer (excluding KGs reserved for cooled-down peers)
        with self._lock:
            cfg = self.cfg.copy()
        excl = _excluded_kgs(cfg, exclude_peer_id=active_id)
        result = sync_queue_to_peer(peer['url'], exclude=excl)
        if 'error' not in result:
            with self._lock:
                self.state['_last_queue_hash'] = q_hash

    def _loop(self):
        """Main director loop."""
        time.sleep(5)  # startup delay
        sync_counter = 0
        while self._running:
            # Cross-worker stop guard. ``stop()`` on the singleton only
            # affects the worker that called it; another worker that
            # had `_director` constructed (e.g. via ``get_director()``
            # for status reads) keeps its own loop running. Disk flags
            # are the source of truth: if the operator / takeover /
            # handover removed ``is_director`` (or wrote
            # ``stepped_down``) we must self-terminate. Without this,
            # a stepped-down primary kept hammering peers with
            # whitelist PUTs / identity broadcasts while the new
            # director (at63) was already running, producing the very
            # split-brain the file flag is supposed to prevent
            # (2026-05-10 incident).
            try:
                is_dir = (DATA_DIR / 'is_director').exists()
                stepped = (DATA_DIR / 'stepped_down').exists()
            except Exception:
                is_dir, stepped = True, False  # don't kill loop on stat error
            if not is_dir or stepped:
                log.warning('director loop self-stopping: is_director=%s '
                            'stepped_down=%s (cross-worker step_down or '
                            'operator demotion)', is_dir, stepped)
                self._running = False
                # Release fcntl lock so a future takeover in this same
                # process can acquire it cleanly.
                fd = getattr(self, '_lock_fd', None)
                if fd is not None:
                    try:
                        import fcntl as _fcntl
                        _fcntl.flock(fd, _fcntl.LOCK_UN)
                        fd.close()
                    except Exception:
                        pass
                    self._lock_fd = None
                return
            try:
                # Re-read config + state from disk each tick so cross-worker
                # changes (e.g. /director/activate from another gunicorn worker)
                # are picked up by the worker actually running the loop.
                try:
                    disk_cfg = load_peers_config()
                    disk_state = load_director_state()
                    with self._lock:
                        self.cfg = disk_cfg
                        # Merge: prefer disk values for active_peer/mode/last_switch
                        # but keep our in-memory bandwidth + unreachable_count.
                        for key in ('active_peer', 'mode', 'last_switch'):
                            if key in disk_state:
                                self.state[key] = disk_state[key]
                    # Stamp first_seen on legacy peers that lack it so
                    # the warmup hold doesn't retroactively block them.
                    # New peers added via the API already get a stamp.
                    _now_iso = datetime.now(timezone.utc).isoformat()
                    _dirty = False
                    for _p in self.cfg.get('peers', []):
                        if not _p.get('first_seen'):
                            _p['first_seen'] = _now_iso
                            _dirty = True
                    if _dirty:
                        save_peers_config(self.cfg)
                except Exception:
                    pass
                self._update_bandwidth()
                # Credential revalidation moved off the director loop
                # into a dedicated daemon thread (see _oidc_reval_loop)
                # so a 30 s parallel sweep can't stall this tick.
                # Capacity factor: poll each peer's processing status once
                # per tick (cheap; we already do it implicitly inside the
                # orchestrators) and use the warning_rates field to derive
                # how aggressively we should run. We stash both the factor
                # and the components on self.state so the dashboard can
                # show the user *why* we slowed down.
                try:
                    from concurrent.futures import (
                        ThreadPoolExecutor as _Tpe,
                        TimeoutError as _Fto,
                    )
                    _peers = list(self.cfg.get('peers', []))
                    _statuses: dict[str, dict] = {}
                    with _Tpe(max_workers=BANDWIDTH_POLL_CONCURRENCY,
                                thread_name_prefix='dir-cap') as _ex:
                        _futs = {_ex.submit(get_peer_status, p.get('url')): p
                                 for p in _peers}
                        _deadline = time.time() + 12
                        for _f, _p in list(_futs.items()):
                            try:
                                _statuses[_p['id']] = _f.result(
                                    timeout=max(0.1, _deadline - time.time())
                                )
                            except (_Fto, Exception):
                                _statuses[_p['id']] = {'state': 'unreachable'}
                    factor = self._capacity_factor(_statuses)
                    # Per-peer warning fingerprint for load-shifting:
                    # let cleaner peers carry more work. Stored in state
                    # so choose_active_peer + orchestrators can use it.
                    _peer_wr: dict[str, dict] = {}
                    for _pid, _ps in (_statuses or {}).items():
                        _peer_wr[_pid] = (_ps or {}).get('warning_rates') or {}
                    # Slow per-peer noise EMA across ALL warning kinds.
                    # Persists across director restarts via director_state.json,
                    # so a peer that misbehaved an hour ago stays penalised
                    # even though its 5/10-min sliding windows have decayed.
                    with self._lock:
                        _long = dict(self.state.get('peer_noise_long_ema') or {})
                        _alpha = PEER_NOISE_LONG_EMA_ALPHA
                        for _pid, _wr in _peer_wr.items():
                            _prev = dict(_long.get(_pid) or {})
                            for _kind in ('bev', 'zenodo', 'copernicus'):
                                _r = float(((_wr.get(_kind) or {}).get('5m')) or 0.0)
                                _p = float(_prev.get(_kind) or 0.0)
                                # Asymmetric: ramp up fast on new pressure,
                                # decay slow when quiet. We take max of the
                                # standard EMA and the current rate so a
                                # spike registers immediately.
                                _new = max(_r, _alpha * _r + (1.0 - _alpha) * _p)
                                _prev[_kind] = round(_new, 4)
                            _long[_pid] = _prev
                        # Drop entries for peers no longer in the fleet.
                        _live = {p['id'] for p in self.cfg.get('peers', [])}
                        for _pid in list(_long.keys()):
                            if _pid not in _live:
                                _long.pop(_pid, None)
                        self.state['peer_noise_long_ema'] = _long
                        self.state['capacity_factor'] = round(factor, 3)
                        self.state['capacity_components'] = (
                            self._capacity_components)
                        self.state['capacity_ema_persisted'] = round(
                            self._capacity_ema, 4)
                        self.state['sub_factor_ema'] = {
                            k: round(float(v), 4)
                            for k, v in self._sub_factor_ema.items()
                        }
                        # Persist the rolling sparkline history so other
                        # gunicorn workers + a fresh director after restart
                        # all see the same chart immediately.
                        self.state['capacity_history'] = [
                            ({'t': t, 'f': f, 'bev': b, 'zen': z, 'cop': c}
                              | ({'stl': s} if s is not None else {})
                              | ({'cpu': cf} if cf is not None else {}))
                            for (t, f, b, z, c, s, cf) in list(self._capacity_history)
                        ]
                        self.state['peer_warning_rates'] = _peer_wr
                except Exception:
                    log.exception('capacity factor computation failed')
                with self._lock:
                    if _clear_completed_reservations(self.cfg, self.state):
                        save_peers_config(self.cfg)
                # Belt-and-braces: ensure primary stays parked every
                # tick, before any peer scheduling decisions.
                try:
                    self._enforce_primary_park()
                except Exception:
                    log.exception('primary park enforce failed')
                # One-shot: release park-until-renewal records we wrote
                # on a *guessed* 95 GB budget without canary-shaping
                # evidence. exe.dev's real per-account limits and
                # billing anchors are unknown; canary-by-default means
                # we only park on observed throughput collapse.
                try:
                    self._release_unverified_bw_parks()
                except Exception:
                    log.exception('release_unverified_bw_parks failed')
                # Belt-and-braces: the *active director* must also not
                # carry processing load. Director duty already costs
                # 60-peer fanout bandwidth; layering frontier work on
                # top is what burned the primary on 2026-05.
                try:
                    self._enforce_director_self_park()
                except Exception:
                    log.exception('director self-park enforce failed')
                self._check_and_switch()
                # Canary health: park peers running past the global
                # budget if exe.dev visibly throttles them or they
                # start polluting the warning stream.
                try:
                    self._check_canary_health()
                except Exception:
                    log.exception('Canary health check error')
                # Per-peer bandwidth-wall enforcement (low-water /
                # hard-depleted park-until-renewal) for non-active
                # peers. _check_and_switch handles the active one.
                try:
                    self._enforce_peer_bandwidth_walls()
                except Exception:
                    log.exception('Peer bandwidth wall enforce error')
                # Auto-retry stale peer updates: re-trigger update on
                # peers that are idle on an old commit (graceful update
                # didn't take). After 2 failed attempts surfaces a
                # manual-update prompt in the dashboard.
                try:
                    self._orchestrate_stale_peer_updates(
                        locals().get('_statuses') or {})
                except Exception:
                    log.exception('Stale-peer update orchestration error')
                # Parallel-frontier orchestration: start additional
                # frontier peers when credential capacity permits.
                try:
                    self._orchestrate_parallel_frontiers()
                except Exception:
                    log.exception('Parallel-frontier orchestration error')
                # Cache-only orchestration runs alongside the frontier
                # peer.  It only ever starts/stops peers that are NOT the
                # active frontier and that have no reservation.
                try:
                    self._orchestrate_cache_only()
                except Exception:
                    log.exception('Cache-only orchestration error')
                # Record fleet-shape sample for the Progress card
                # sparkline. Counts as observed *after* orchestration
                # this tick — gives the truest 30s-resolution view of
                # what's actually running.
                try:
                    _statuses_for_hist = locals().get('_statuses') or {}
                    _fr = len(self.state.get(
                        'parallel_frontiers_active') or [])
                    # Match status()'s definition exactly: cache_only_run
                    # (peer self-report) AND running/processing. Using
                    # ps['cache_only'] alone misses peers whose status
                    # poll didn't include the flag this tick.
                    _co = 0
                    for _pid, _ps in _statuses_for_hist.items():
                        if not isinstance(_ps, dict):
                            continue
                        if (_ps.get('cache_only') and
                                _ps.get('state') in (
                                    'running', 'processing')):
                            _co += 1
                    _cr_cache = self.state.get(
                        '_cache_ready_cache') or {}
                    _cr = len(_cr_cache.get('codes') or [])
                    self._peer_history.append((
                        int(time.time()), int(_fr), int(_co), int(_cr),
                    ))
                    self.state['peer_history'] = [
                        {'t': t, 'fr': fr, 'co': co, 'cr': cr}
                        for (t, fr, co, cr) in list(self._peer_history)
                    ]
                except Exception:
                    log.exception('peer-history sample failed')
                # Sync queue every 5 iterations (~2.5 min at 30s interval)
                sync_counter += 1
                if sync_counter >= 5:
                    try:
                        self._prune_priority_queue()
                    except Exception:
                        log.exception('priority queue prune failed')
                    self._sync_queue_to_active()
                    sync_counter = 0
                # Elect / refresh shadow & push snapshot every tick.
                try:
                    self._maintain_shadow(_statuses)
                except Exception as _e:
                    log.debug('shadow maintenance failed: %s', _e)
                # Push identity to one peer per tick that doesn't yet
                # know who the director is.
                try:
                    self._push_identity_to_unaware_peers()
                except Exception as _e:
                    log.debug('identity push failed: %s', _e)
                # Auto-handback: if we're not the primary, and the
                # primary is healthy enough, hand the director role
                # back to it. Primary is the canonical home for the
                # director: it has the search index, dashboard URL,
                # public DNS, and operators expect to find it there.
                try:
                    self._maybe_handback_to_primary()
                except Exception as _e:
                    log.debug('auto-handback check failed: %s', _e)
                with self._lock:
                    save_director_state(self.state)
            except Exception:
                log.exception('Director loop error')
            time.sleep(DIRECTOR_POLL_INTERVAL)

    # --- Shadow election & snapshot push ---------------------------------
    def _broadcast_identity_to_all_peers(self) -> None:
        """One-shot: push (peer_id, peer_url, director_url) to every peer
        in parallel. Used at director startup so every peer learns who
        the director is without waiting len(peers) ticks.
        """
        try:
            import director_ha as dha
        except Exception:
            return
        my_url = dha.self_url()
        if not my_url:
            return
        peers = [p for p in self.cfg.get('peers') or [] if p.get('url')]
        if not peers:
            return
        from concurrent.futures import ThreadPoolExecutor

        def _push(peer):
            try:
                requests.post(
                    peer['url'].rstrip('/') + '/api/v1/director/identity',
                    json={'id': peer['id'], 'url': peer['url'],
                          'director_url': my_url},
                    headers=_admin_headers(),
                    timeout=PEER_TIMEOUT_CONTROL,
                )
                return True
            except Exception:
                return False

        ok = 0
        with ThreadPoolExecutor(max_workers=10,
                                thread_name_prefix='dir-id-bcast') as ex:
            for r in ex.map(_push, peers):
                if r:
                    ok += 1
        log.info('director identity broadcast: %d/%d peers reachable',
                 ok, len(peers))

    def _push_identity_to_unaware_peers(self) -> None:
        """Push director_url + peer_id to peers that don't yet know us.

        Cheap GET to ``/api/v1/director/identity``; if the peer reports
        ``director_url=null`` we POST it. Throttled to one peer per tick.
        """
        try:
            import director_ha as dha
        except Exception:
            return
        my_url = dha.self_url()
        if not my_url:
            return
        # Cycle through peers, one per tick.
        peers = [p for p in self.cfg.get('peers') or [] if p.get('url')]
        if not peers:
            return
        idx = int(self.state.get('_identity_push_idx', 0)) % len(peers)
        peer = peers[idx]
        self.state['_identity_push_idx'] = idx + 1
        try:
            r = requests.get(
                peer['url'].rstrip('/') + '/api/v1/director/identity',
                headers=_admin_headers(), timeout=PEER_TIMEOUT_PROBE,
            )
            if not r.ok:
                return
            d = r.json() or {}
            needs = (not d.get('director_url')) or (d.get('director_url') != my_url) \
                or (d.get('id') != peer['id'])
            if not needs:
                return
            requests.post(
                peer['url'].rstrip('/') + '/api/v1/director/identity',
                json={'id': peer['id'], 'url': peer['url'],
                      'director_url': my_url},
                headers=_admin_headers(), timeout=PEER_TIMEOUT_CONTROL,
            )
            log.info('director identity pushed to %s', peer['id'])
        except Exception:
            pass

    def _maybe_handback_to_primary(self) -> None:
        """If we're not the primary, hand the director role back to
        primary as soon as it is reachable, on the same git commit, and
        not in a stepped_down/scheduled state.

        Throttled. Every entry past the basic gates advances the
        throttle so an early-return path doesn't fire every 30 s tick
        (the bug that made at40 hammer primary's clear_stepped_down
        endpoint for an hour on 2026-05-07). The throttle uses two
        durations:

          * ``HANDBACK_RETRY_S`` (60 s) on transient/early returns —
            unreachable primary, stepped_down clear in flight, etc.
            Short enough that handback fires within one or two minutes
            of conditions becoming favourable; long enough that we
            don't spam.
          * ``HANDBACK_BACKOFF_S`` (300 s) on "hard" failures — commit
            ancestry blocks or operator must reconcile. We retry less
            often because the situation needs human intervention.

        All logged at INFO/WARNING so journals always show the reason.
        """
        # Short retry on transient/expected paths (primary still
        # restarting, stepped_down flag clear just landed) so we hand
        # the role back within one director tick of primary becoming
        # ready. Was 60s — too conservative; meant operators saw
        # ~90-120s of at-X-acting-director after every update_peers.
        HANDBACK_RETRY_S = 15
        HANDBACK_BACKOFF_S = 300
        try:
            import director_ha as dha
        except Exception:
            return
        me = dha.self_id()
        if me == 'primary':
            return
        # Throttle: read once, advance later based on outcome.
        last = float(self.state.get('_handback_last_attempt') or 0.0)
        if (time.time() - last) < HANDBACK_RETRY_S:
            return
        # Locate primary entry.
        cfg = self.cfg
        primary = None
        for p in cfg.get('peers') or []:
            if p.get('id') == 'primary' and p.get('url'):
                primary = p
                break
        if not primary or not primary.get('enabled', True):
            # No primary configured / primary disabled — throttle longer
            # so we don't recheck cfg every tick.
            self.state['_handback_last_attempt'] = time.time()
            self.state['_handback_last_reason'] = (
                'no_primary' if not primary else 'primary_disabled')
            return
        # Advance throttle now — every code path below will set it again
        # with its own reason, but this guarantees no early return
        # silently re-fires next tick.
        self.state['_handback_last_attempt'] = time.time()
        # NOTE: we deliberately do NOT gate on _peer_is_scheduled() here.
        # `not_before` parks the primary out of frontier / cache-only work
        # (those gates honour it in choose_active_peer / _orchestrate_*),
        # but the director role itself is just HTTP orchestration + the
        # Zenodo lock broker — ~zero bandwidth. The primary is the
        # canonical home for the director (search index, public DNS,
        # dashboard URL), so hand the role back even when scheduled.
        if _peer_is_scheduled(primary):
            log.info('Auto-handback: primary scheduled (not_before=%s) — '
                     'frontier-disabled but still eligible for director role',
                     primary.get('not_before'))
        url = primary['url']
        # Probe primary: must be reachable, on our git commit, and not
        # currently flagged stepped_down (which would mean operator
        # demoted it on purpose).
        try:
            r = requests.get(url.rstrip('/') + '/api/v1/director/identity',
                             headers=_admin_headers(),
                             timeout=PEER_TIMEOUT_PROBE)
            if not r.ok:
                log.info('Auto-handback: primary identity HTTP %d — retry in %ds',
                         r.status_code, HANDBACK_RETRY_S)
                self.state['_handback_last_reason'] = f'http_{r.status_code}'
                return
            d = r.json() or {}
        except Exception as e:
            log.info('Auto-handback: primary unreachable (%s) — retry in %ds',
                     str(e)[:80], HANDBACK_RETRY_S)
            self.state['_handback_last_reason'] = 'unreachable'
            return
        if d.get('stepped_down'):
            # Primary still has the stepped_down flag set. Clear it
            # remotely so it's eligible again — the flag exists to
            # prevent cold restarts from auto-promoting; once we hand
            # back voluntarily it must be cleared.
            cleared_ok = False
            try:
                cr = requests.post(url.rstrip('/')
                              + '/api/v1/admin/clear_stepped_down',
                              headers=_admin_headers(),
                              timeout=PEER_TIMEOUT_CONTROL)
                if cr.ok:
                    cleared_ok = True
                    log.info('Auto-handback: cleared stepped_down on primary '
                             '(was %s) — proceeding with handover this tick',
                             cr.json().get('cleared') if cr.headers.get(
                                 'content-type', '').startswith('application/json')
                             else 'unknown')
                else:
                    log.warning('Auto-handback: clear_stepped_down on primary '
                                'returned HTTP %d — retry in %ds',
                                cr.status_code, HANDBACK_RETRY_S)
            except Exception as e:
                log.warning('Auto-handback: clear_stepped_down failed (%s) — '
                            'retry in %ds', str(e)[:80], HANDBACK_RETRY_S)
            self.state['_handback_last_reason'] = (
                'cleared_stepped_down' if cleared_ok else 'clear_failed')
            if not cleared_ok:
                return
            # Fall through and attempt handover in the same tick —
            # do_handover targets primary via URL and primary will
            # accept now that its stepped_down flag is gone. Saves a
            # full HANDBACK_RETRY_S window.
            d['stepped_down'] = False
        # Same git commit?
        try:
            info = requests.get(url.rstrip('/') + '/api/v1/info',
                                timeout=PEER_TIMEOUT_PROBE).json()
            primary_commit = (info.get('git_commit') or '')[:7]
        except Exception:
            primary_commit = ''
        my_commit = ''
        try:
            import subprocess as _sp
            my_commit = _sp.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=str(Path(__file__).parent), stderr=_sp.DEVNULL,
                timeout=2).decode().strip()
        except Exception:
            pass
        # We used to require an exact commit match; that made handback
        # fail any time the operator pushed an update on the primary
        # while a non-primary peer was acting as director. Now we use
        # ancestry checks: primary must be at-or-ahead (my_commit is an
        # ancestor of primary_commit, OR commits are equal). We refuse
        # if primary is strictly behind us (could lose schema fields)
        # OR if commits are unrelated (we can't reason about state
        # compatibility — operator must intervene).
        if my_commit and primary_commit and primary_commit != my_commit:
            import subprocess as _sp
            repo = str(Path(__file__).parent)
            def _is_ancestor(a: str, b: str) -> bool | None:
                """True if a is ancestor of b. None if either commit is
                not resolvable locally (try `git fetch` first)."""
                try:
                    _sp.check_output(
                        ['git', 'cat-file', '-e', a + '^{commit}'],
                        cwd=repo, stderr=_sp.DEVNULL, timeout=3,
                    )
                    _sp.check_output(
                        ['git', 'cat-file', '-e', b + '^{commit}'],
                        cwd=repo, stderr=_sp.DEVNULL, timeout=3,
                    )
                except Exception:
                    return None
                try:
                    _sp.check_output(
                        ['git', 'merge-base', '--is-ancestor', a, b],
                        cwd=repo, stderr=_sp.DEVNULL, timeout=3,
                    )
                    return True
                except Exception:
                    return False
            primary_is_behind = _is_ancestor(primary_commit, my_commit)
            primary_is_ahead = _is_ancestor(my_commit, primary_commit)
            if primary_is_behind is None or primary_is_ahead is None:
                # Try a cheap fetch to resolve unknown commits.
                try:
                    _sp.run(['git', 'fetch', '--quiet', 'origin'],
                            cwd=repo, capture_output=True, timeout=20)
                    primary_is_behind = _is_ancestor(primary_commit, my_commit)
                    primary_is_ahead = _is_ancestor(my_commit, primary_commit)
                except Exception:
                    pass
            if primary_is_behind is True:
                log.warning('handback skipped: primary on %s is behind '
                            'us %s — primary needs to git pull first '
                            '(retry in %ds)',
                            primary_commit, my_commit, HANDBACK_BACKOFF_S)
                # Long backoff — needs operator action (or stale-peer
                # rollout) to update primary.
                self.state['_handback_last_attempt'] = (
                    time.time() - HANDBACK_RETRY_S + HANDBACK_BACKOFF_S)
                self.state['_handback_last_reason'] = 'primary_behind'
                return
            if primary_is_ahead is True:
                log.info('handback proceeding: primary on %s is ahead '
                         'of us %s', primary_commit, my_commit)
            else:
                # Commits unrelated (different branches?) or unresolvable.
                # Refuse — operator must reconcile.
                log.warning('handback skipped: primary commit %s and '
                            'our commit %s are unrelated/unresolvable; '
                            'operator must reconcile (retry in %ds)',
                            primary_commit, my_commit, HANDBACK_BACKOFF_S)
                self.state['_handback_last_attempt'] = (
                    time.time() - HANDBACK_RETRY_S + HANDBACK_BACKOFF_S)
                self.state['_handback_last_reason'] = 'commits_unrelated'
                return
        # Healthy enough — hand over.
        log.warning('Auto-handback: primary is healthy, handing director '
                    'role back from %s to primary', me)
        self.state['_handback_last_attempt'] = time.time()
        self.state['_handback_last_reason'] = 'attempting'
        try:
            res = dha.do_handover('primary', url)
            log.warning('Auto-handback result: %s', res)
            self.state['_handback_last_reason'] = 'handed_over'
        except Exception as e:
            log.warning('Auto-handback failed: %s', e)
            self.state['_handback_last_reason'] = f'handover_failed:{str(e)[:80]}'

    def _maintain_shadow(self, statuses: dict) -> None:
        """Elect a shadow each tick and push the current state snapshot.

        Shadow == the most-trustworthy peer that is not us. Picks the peer
        with: enabled=True, reachable, lowest noise score (most reliable),
        not already running heavy frontier work. Re-evaluated every tick;
        sticky for ~5 min unless the current shadow turns unreachable.
        """
        try:
            import director_ha as dha
        except Exception:
            return
        cfg = self.cfg
        peers = cfg.get('peers') or []
        my_commit = _LOCAL_GIT_COMMIT
        # Candidate filter: enabled, reachable, running OUR git commit
        # (so the snapshot endpoint exists), enough disk (>5 GB free) and
        # enough remaining bandwidth (>10 GB) to act as director if needed.
        SHADOW_MIN_DISK_GB = 5.0
        SHADOW_MIN_BANDWIDTH_GB = 10.0
        candidates: list[tuple[float, dict]] = []
        rejected: list[tuple[str, str]] = []
        for p in peers:
            if not p.get('enabled') or not p.get('url'):
                continue
            ps = (statuses or {}).get(p['id']) or {}
            if ps.get('state') == 'unreachable':
                rejected.append((p['id'], 'unreachable'))
                continue
            commit = (ps.get('git_commit') or '').strip()
            if my_commit != 'unknown' and commit and commit != my_commit:
                rejected.append((p['id'], f'commit_mismatch({commit})'))
                continue
            sysd = ps.get('system') or {}
            disk_free = sysd.get('disk_free_gb')
            if disk_free is not None and disk_free < SHADOW_MIN_DISK_GB:
                rejected.append((p['id'], f'low_disk({disk_free}GB)'))
                continue
            bw = (self.state.get('peer_bandwidth') or {}).get(p['id']) or {}
            rem_gb = bw.get('remaining_gb')
            if rem_gb is not None and rem_gb < SHADOW_MIN_BANDWIDTH_GB:
                rejected.append((p['id'], f'low_bw({rem_gb}GB)'))
                continue
            # Shadow must be able to *act* as director if it gets
            # promoted (run frontiers, build GPKGs). A peer on a 75%%
            # steal pool is a poor failover target; weight it down.
            score = (_peer_noise_score(p['id'], self.state)
                     + ((1.0 / _peer_cpu_capacity(p['id'])) - 1.0))
            candidates.append((score, p))
        if not candidates:
            # Log only when set changes to avoid spam.
            tag = ','.join(f'{i}:{r}' for i, r in rejected[:6])
            if self.state.get('_shadow_reject_tag') != tag:
                log.warning('director shadow: no eligible candidates (%s)', tag)
                self.state['_shadow_reject_tag'] = tag
                self.state['shadow_peer'] = None
                self.state['shadow_url'] = None
            return
        else:
            self.state.pop('_shadow_reject_tag', None)
        candidates.sort(key=lambda x: x[0])
        prev_shadow = self.state.get('shadow_peer')
        # Stickiness: keep the previous shadow unless it's gone or we have
        # a much-better candidate (>0.3 score gap).
        chosen = None
        if prev_shadow:
            for s, p in candidates:
                if p['id'] == prev_shadow:
                    chosen = (s, p)
                    break
        if chosen is None:
            chosen = candidates[0]
        else:
            best_s, best_p = candidates[0]
            if best_p['id'] != prev_shadow and (chosen[0] - best_s) > 0.3:
                chosen = (best_s, best_p)
        s_score, s_peer = chosen
        # Only push snapshot every SHADOW_SYNC_INTERVAL seconds.
        last_push = float(self.state.get('shadow_last_push_ts') or 0.0)
        now = time.time()
        push_due = (now - last_push) >= dha.SHADOW_SYNC_INTERVAL
        # Always push immediately when the shadow changes.
        shadow_changed = prev_shadow != s_peer['id']
        self.state['shadow_peer'] = s_peer['id']
        self.state['shadow_url'] = s_peer.get('url')
        self.state['shadow_score'] = round(float(s_score), 4)
        if not (push_due or shadow_changed):
            return
        try:
            snap = dha.build_snapshot()
            snap['_meta']['shadow_id'] = s_peer['id']
            snap['_meta']['director_id'] = (load_director_state()
                                            .get('active_peer') or 'director')
            res = dha.push_snapshot_to_shadow(s_peer['url'], snap)
            ok = isinstance(res, dict) and res.get('status') == 'staged'
            self.state['shadow_last_push_ts'] = now
            self.state['shadow_last_push_ok'] = ok
            self.state['shadow_last_push_result'] = res if not ok else 'staged'
            if shadow_changed:
                log.info('director shadow set to %s (score=%.3f, push=%s)',
                         s_peer['id'], s_score, 'ok' if ok else res)
            # Replicate the long-term log archive too. Separate throttle
            # (hourly today-only steady state, full sweep on shadow
            # change) so the archive doesn't inflate the 30s snapshot
            # traffic. Best-effort: failures are logged but do NOT
            # affect the small-state snapshot path above.
            try:
                if not hasattr(self, '_log_arch_sha'):
                    self._log_arch_sha = {}
                if not hasattr(self, '_log_arch_last_push'):
                    self._log_arch_last_push = 0.0
                arch_due = (now - self._log_arch_last_push
                            >= dha.LOG_ARCHIVE_PUSH_INTERVAL)
                if shadow_changed:
                    # New shadow — wipe per-shadow sha cache and do a
                    # full sweep so the new shadow gets every day.
                    self._log_arch_sha = {}
                if shadow_changed or arch_due:
                    res_a = dha.push_log_archive_to_shadow(
                        s_peer['url'], full=shadow_changed,
                        cache=self._log_arch_sha)
                    self._log_arch_last_push = now
                    self.state['shadow_log_archive_last'] = res_a
            except Exception as e:
                log.warning('shadow log_archive push to %s failed: %s',
                            s_peer.get('id'), e)
        except Exception as e:
            log.warning('shadow snapshot push to %s failed: %s',
                        s_peer.get('id'), e)
            self.state['shadow_last_push_ok'] = False
            self.state['shadow_last_push_result'] = str(e)[:200]


# Singleton
_director: PeerDirector | None = None


def get_director() -> PeerDirector:
    global _director
    if _director is None:
        _director = PeerDirector()
    return _director
