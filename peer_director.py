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

# Git commit hash (read once at import)
try:
    _LOCAL_GIT_COMMIT = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        cwd=str(Path(__file__).parent), stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    _LOCAL_GIT_COMMIT = 'unknown'

DATA_DIR = Path('data/austria_processor')
PEERS_CONFIG = DATA_DIR / 'peers.json'
DIRECTOR_STATE = DATA_DIR / 'director_state.json'

# Bandwidth budget per peer per billing cycle (bytes)
BANDWIDTH_BUDGET_GB = 95  # conservative — leave 5 GB headroom out of 100
BANDWIDTH_BUDGET_BYTES = BANDWIDTH_BUDGET_GB * (1024 ** 3)
BANDWIDTH_RENEW_DAY = 17  # day of month when exe.dev bandwidth resets

# How often the director checks state (seconds)
DIRECTOR_POLL_INTERVAL = 30
# Grace period after stopping a peer before starting another (seconds)
SWITCH_COOLDOWN = 10
# HTTP timeout for peer API calls
PEER_TIMEOUT = 30
# Number of consecutive unreachable polls before failover (avoids killing
# peers during heavy GPKG builds that briefly starve gunicorn).
UNREACHABLE_FAILOVER_THRESHOLD = 3
# How long to keep a peer out of rotation after a local Zenodo network
# failure (the same peer hitting the same network problem on retry would
# loop forever).  Cleared automatically when not_before passes.
ZENODO_NETWORK_COOLDOWN_MIN = 30


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
    """Load or create peers config."""
    if PEERS_CONFIG.exists():
        try:
            return json.loads(PEERS_CONFIG.read_text())
        except Exception:
            pass
    cfg = _default_peers_config()
    PEERS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    PEERS_CONFIG.write_text(json.dumps(cfg, indent=2))
    return cfg


def save_peers_config(cfg: dict):
    PEERS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    PEERS_CONFIG.write_text(json.dumps(cfg, indent=2))


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


def get_peer_bandwidth(peer_url: str) -> dict:
    """Get bandwidth from a remote peer."""
    try:
        r = requests.get(
            peer_url.rstrip('/') + '/api/v1/bandwidth',
            timeout=PEER_TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning('Failed to get bandwidth from %s: %s', peer_url, e)
        # On error, assume peer has budget remaining (don't penalize for missing endpoint)
        return {'error': str(e), 'used_bytes': 0, 'used_gb': 0,
                'remaining_gb': BANDWIDTH_BUDGET_GB, 'pct_used': 0,
                'estimated': True}


def get_peer_status(peer_url: str | None) -> dict:
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
                d.setdefault('git_commit', _LOCAL_GIT_COMMIT)
                return d
            except Exception:
                pass
        return {'state': 'idle', 'git_commit': _LOCAL_GIT_COMMIT}
    try:
        r = requests.get(
            peer_url.rstrip('/') + '/api/v1/processing/status',
            timeout=PEER_TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
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
            timeout=PEER_TIMEOUT
        )
        r.raise_for_status()
        d = r.json()
        return d.get('lines', d.get('log', []))
    except Exception:
        return []


def _sync_cache_manifest_to_peer(peer_url: str) -> None:
    """Push local Zenodo tile-cache manifest to a remote peer.

    Ensures the peer shares the same Zenodo cache deposit, so it can
    read cached Copernicus/Hansen tiles instead of re-fetching them.
    """
    manifest_path = DATA_DIR / 'cache_manifest.json'
    if not manifest_path.exists():
        return
    try:
        local_cm = json.loads(manifest_path.read_text())
        if not local_cm.get('depo_id'):
            return
        r = requests.put(
            peer_url.rstrip('/') + '/api/v1/processing/cache_manifest',
            json=local_cm, timeout=PEER_TIMEOUT,
        )
        if r.ok:
            result = r.json()
            log.info('Cache manifest sync to %s: %d entries updated',
                     peer_url, result.get('updated', 0))
        else:
            log.warning('Cache manifest sync to %s: HTTP %d', peer_url, r.status_code)
    except Exception as e:
        log.warning('Cache manifest sync to %s failed: %s', peer_url, e)


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
    try:
        r = requests.post(
            peer_url.rstrip('/') + '/api/v1/processing/queue',
            json={'kgs': queue, 'position': 0, 'skip_processed': True},
            timeout=PEER_TIMEOUT,
        )
        result = r.json()
        log.info('Queue sync to %s: pushed %d KGs — %s%s', peer_url, len(queue),
                 result.get('status', '?'),
                 (' (excluded ' + ','.join(sorted(exclude)) + ')') if exclude else '')
        return result
    except Exception as e:
        log.warning('Queue sync to %s failed: %s', peer_url, e)
        return {'error': str(e)}


def _reserved_kgs(cfg: dict, exclude_peer_id: str | None = None) -> set:
    """Collect KGs reserved by other peers (not `exclude_peer_id`).

    Reservations persist past `not_before` so that the holding peer can
    actually pick the KG back up once cooldown lifts. Stale reservations
    are pruned separately by `_clear_completed_reservations` once the KG
    appears in the local `_get_completed_kgs()` set.
    """
    out = set()
    for p in cfg.get('peers', []):
        if exclude_peer_id and p.get('id') == exclude_peer_id:
            continue
        kg = p.get('reserved_kg')
        if kg:
            out.add(str(kg))
    return out


def start_peer_processor(peer_url: str | None, exclude_kgs: set | None = None) -> dict:
    """Start the processor on a peer.

    For remote peers, syncs the local priority queue first so the peer
    processes the same KGs in priority order. `exclude_kgs` lets callers
    suppress KGs reserved for other (cooled-down) peers.
    """
    if peer_url is None:
        # Local start — use the local API so _processor_process is tracked
        try:
            r = requests.post('http://127.0.0.1:8000/api/v1/processing/start',
                              json={}, timeout=PEER_TIMEOUT)
            return r.json() if r.ok else {'error': f'local start: {r.status_code}'}
        except Exception as e:
            return {'error': str(e)}
    # Remote peer — sync cache manifest + priority queue before starting
    _sync_cache_manifest_to_peer(peer_url)
    queue_result = sync_queue_to_peer(peer_url, exclude=exclude_kgs)
    # Try API start first, fall back to systemd restart via admin endpoint
    try:
        r = requests.post(
            peer_url.rstrip('/') + '/api/v1/processing/start',
            json={},
            timeout=PEER_TIMEOUT,
        )
        if r.ok:
            result = r.json()
            result['queue_sync'] = queue_result
            return result
        if r.status_code == 409:
            # Already running — that's fine
            result = r.json()
            result['queue_sync'] = queue_result
            result['already_running'] = True
            return result
        # API start failed (500 etc) — try systemd restart as fallback
        log.warning('API start on %s returned %d, trying systemd fallback', peer_url, r.status_code)
    except Exception as e:
        log.warning('API start on %s failed: %s, trying systemd fallback', peer_url, e)
    # Fallback: ask the peer to restart the processor via systemd
    try:
        r2 = requests.post(
            peer_url.rstrip('/') + '/api/v1/admin/restart_processor',
            json={},
            timeout=30,
        )
        result = r2.json() if r2.ok else {'error': f'systemd fallback: {r2.status_code}'}
        result['queue_sync'] = queue_result
        result['method'] = 'systemd_fallback'
        return result
    except Exception as e2:
        return {'error': str(e2), 'queue_sync': queue_result, 'method': 'both_failed'}


def stop_peer_processor(peer_url: str | None) -> dict:
    """Stop the processor on a peer."""
    url = peer_url if peer_url else 'http://127.0.0.1:8000'
    try:
        r = requests.post(
            url.rstrip('/') + '/api/v1/processing/stop',
            timeout=30  # stop can take a moment
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


def trigger_peer_update(peer_url: str) -> dict:
    """Tell a remote peer to git pull and restart its web server.
    The peer kills itself on restart so the connection always drops — treat
    any ConnectionError/ReadTimeout after the request was sent as success.
    """
    try:
        r = requests.post(
            peer_url.rstrip('/') + '/api/v1/admin/update',
            timeout=15
        )
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


def _clear_completed_reservations(cfg: dict) -> bool:
    """Drop reserved_kg for peers whose held KG is already completed.

    A reservation otherwise persists past `not_before` so the holding
    peer can pick its held KG back up after the cooldown lifts.
    Returns True if the config was modified.
    """
    completed = set()
    try:
        # Imported lazily — app.py imports peer_director at startup.
        from app import _get_completed_kgs
        completed = _get_completed_kgs()
    except Exception:
        pass
    changed = False
    for p in cfg.get('peers', []):
        kg = p.get('reserved_kg')
        if kg and str(kg) in completed:
            p.pop('reserved_kg', None)
            changed = True
    return changed


def _ready_reservation_holder(cfg: dict) -> str | None:
    """Return peer_id of an enabled, non-scheduled peer that holds a
    reserved KG ready to be resumed. Returns None if no such peer.
    """
    for p in cfg.get('peers', []):
        if not p.get('enabled', True):
            continue
        if not p.get('reserved_kg'):
            continue
        if _peer_is_scheduled(p):
            continue
        return p['id']
    return None


def choose_active_peer(cfg: dict, state: dict) -> str | None:
    """Pick the best peer to run the processor on.

    Reservation holders win first — if an enabled, non-scheduled peer
    still holds a reserved KG, give it priority so it can resume the
    held KG from its tile checkpoints. Otherwise pick the peer with the
    most remaining bandwidth.
    Returns None if all candidates have <2 GB remaining.
    """
    holder = _ready_reservation_holder(cfg)
    if holder:
        bw = state.get('peer_bandwidth', {}).get(holder, {})
        budget_bytes = cfg.get('budget_gb', BANDWIDTH_BUDGET_GB) * (1024 ** 3)
        used = bw.get('used_bytes', 0)
        if (budget_bytes - used) >= 2 * (1024 ** 3):
            return holder
    budget_gb = cfg.get('budget_gb', BANDWIDTH_BUDGET_GB)
    budget_bytes = budget_gb * (1024 ** 3)
    best_id = None
    best_remaining = -1

    for peer in cfg.get('peers', []):
        if not peer.get('enabled', True):
            continue
        if _peer_is_scheduled(peer):
            continue
        pid = peer['id']
        bw = state.get('peer_bandwidth', {}).get(pid, {})
        used = bw.get('used_bytes', 0)
        remaining = budget_bytes - used
        if remaining > best_remaining:
            best_remaining = remaining
            best_id = pid

    # Don't use a peer that has < 2 GB remaining
    if best_remaining < 2 * (1024 ** 3):
        return None

    return best_id


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
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

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
        log.info('PeerDirector started (lock acquired)')

    def stop(self):
        self._running = False

    def get_status(self) -> dict:
        """Full director status for the dashboard."""
        # Always re-read config from disk so all gunicorn workers see new peers
        try:
            disk_cfg = load_peers_config()
            with self._lock:
                self.cfg = disk_cfg
        except Exception:
            pass
        with self._lock:
            cfg = self.cfg.copy()
            state = self.state.copy()

        peers_status = []
        for peer in cfg.get('peers', []):
            pid = peer['id']
            url = peer.get('url')
            ps = get_peer_status(url)
            bw = state.get('peer_bandwidth', {}).get(pid, {})
            proc_status = ps.get('state', 'unknown')

            peers_status.append({
                'id': pid,
                'url': url,
                'enabled': peer.get('enabled', True),
                'not_before': peer.get('not_before'),
                'scheduled': _peer_is_scheduled(peer),
                'reserved_kg': peer.get('reserved_kg'),
                'is_active': pid == state.get('active_peer'),
                'processor_state': proc_status,
                'current_kg': (ps.get('current_kg') or {}).get('code'),
                'current_kg_name': (ps.get('current_kg') or {}).get('name'),
                'completed': ps.get('completed', 0),
                'bandwidth': bw,
                'online': proc_status != 'unreachable',
                'git_commit': ps.get('git_commit', ''),
                'region': ps.get('region', ''),
            })

        return {
            'mode': state.get('mode', 'auto'),
            'active_peer': state.get('active_peer'),
            'last_switch': state.get('last_switch'),
            'budget_gb': cfg.get('budget_gb', BANDWIDTH_BUDGET_GB),
            'renew_day': cfg.get('renew_day', BANDWIDTH_RENEW_DAY),
            'cycle_start': get_billing_cycle_start().isoformat(),
            'peers': peers_status,
        }

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
                                 timeout=PEER_TIMEOUT)
                current = r.json().get('throttle', False) if r.ok else None
                if current == enabled:
                    results[peer['id']] = 'already_' + ('on' if enabled else 'off')
                    continue
                # Toggle to match desired state
                r2 = requests.post(url.rstrip('/') + '/api/v1/processing/throttle',
                                   timeout=PEER_TIMEOUT)
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
            save_peers_config(cfg)
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
        """Refresh bandwidth data for all peers."""
        for peer in self.cfg.get('peers', []):
            pid = peer['id']
            url = peer.get('url')
            if url is None:
                bw = get_local_bandwidth()
            else:
                bw = get_peer_bandwidth(url)

            with self._lock:
                if 'peer_bandwidth' not in self.state:
                    self.state['peer_bandwidth'] = {}
                self.state['peer_bandwidth'][pid] = bw

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
                            excl = _reserved_kgs(self.cfg, exclude_peer_id=active_id)
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
            holder = _ready_reservation_holder(cfg)
            if holder and holder != active_id:
                active_peer_cfg = get_peer_by_id(cfg, active_id)
                ps = get_peer_status(active_peer_cfg.get('url')) if active_peer_cfg else {}
                proc_state = ps.get('state', 'unknown')
                if proc_state in ('idle', 'stopped'):
                    log.info('Reservation ready on %s — pre-empting %s (between KGs)',
                             holder, active_id)
                    with self._lock:
                        self.state['active_peer'] = None
                        active_id = None
                else:
                    log.info('Reservation ready on %s but %s is mid-KG (%s) — waiting for KG boundary',
                             holder, active_id, proc_state)

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

        # Check if active peer is over budget
        if active_id:
            bw = state_copy.get('peer_bandwidth', {}).get(active_id, {})
            used = bw.get('used_bytes', 0)
            remaining_gb = (budget_bytes - used) / (1024 ** 3)

            if remaining_gb < 2:  # less than 2 GB remaining
                log.info('Peer %s near bandwidth limit (%.1f GB remaining), switching',
                         active_id, remaining_gb)
                peer = get_peer_by_id(cfg, active_id)
                if peer:
                    res = safely_stop_peer(peer.get('url'), active_id)
                    if 'error' in res:
                        log.error('Cannot deactivate %s (over-budget): %s — '
                                  'holding off switch (credential safety)',
                                  active_id, res['error'])
                        return  # don't activate next peer until this one is verified stopped
                with self._lock:
                    self.state['active_peer'] = None
                    active_id = None

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
                    log.warning('Active peer %s: Zenodo network failure \u2014 cooling down %d min and switching',
                                active_id, ZENODO_NETWORK_COOLDOWN_MIN)
                    # Apply not_before cooldown so choose_active_peer skips this peer.
                    # Also reserve the in-progress KG so substitute peers skip
                    # it — the cooled peer keeps its tile checkpoints and
                    # will finish quickly when the cooldown lifts.
                    cd_until = (datetime.now(timezone.utc)
                                + timedelta(minutes=ZENODO_NETWORK_COOLDOWN_MIN))
                    cur_kg = (status.get('current_kg') or {}).get('code')
                    for p in self.cfg.get('peers', []):
                        if p['id'] == active_id:
                            p['not_before'] = cd_until.isoformat()
                            if cur_kg:
                                p['reserved_kg'] = str(cur_kg)
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
                    remaining_gb = (budget_bytes - used) / (1024 ** 3)
                    # Honour not_before cooldown — if scheduled, demote
                    # the active peer so a different one can be picked.
                    if _peer_is_scheduled(peer):
                        log.info('Active peer %s is scheduled (not_before=%s) \u2014 letting director pick another',
                                 active_id, peer.get('not_before'))
                        with self._lock:
                            self.state['active_peer'] = None
                            active_id = None
                    elif remaining_gb >= 2:
                        excl = _reserved_kgs(cfg, exclude_peer_id=active_id)
                        log.info('Restarting processor on %s (%.1f GB remaining)%s',
                                 active_id, remaining_gb,
                                 (' — excluding ' + ','.join(sorted(excl))) if excl else '')
                        start_peer_processor(peer.get('url'), exclude_kgs=excl)
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

        # Enforce single-active: stop any non-active peers that are running
        # (they may have been started independently or before director took over)
        if active_id:
            for p in cfg.get('peers', []):
                if p['id'] != active_id and p.get('url') is not None:
                    ps = get_peer_status(p.get('url'))
                    if ps.get('state') in ('running', 'processing'):
                        log.warning('Non-active peer %s is running — stopping it '
                                    '(only %s should be active)', p['id'], active_id)
                        safely_stop_peer(p.get('url'), p['id'])

        # If no active peer, choose one
        if not active_id:
            with self._lock:
                new_peer = choose_active_peer(self.cfg, self.state)
            if new_peer:
                peer = get_peer_by_id(cfg, new_peer)
                if peer:
                    # CRITICAL: verify ALL other peers are fully stopped
                    # before activating the new one. If we can't verify a
                    # peer is stopped, abort — starting a new processor while
                    # another may be running causes Copernicus 402 errors.
                    blocked = False
                    for p in cfg.get('peers', []):
                        if p['id'] == new_peer:
                            continue
                        ps = get_peer_status(p.get('url'))
                        st = ps.get('state', 'unknown')
                        if st in ('running', 'processing'):
                            log.info('Stopping peer %s before activating %s',
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
                    excl = _reserved_kgs(cfg, exclude_peer_id=new_peer)
                    log.info('Activating peer %s%s', new_peer,
                             (' (excluding reserved KGs: ' + ','.join(sorted(excl)) + ')') if excl else '')
                    result = start_peer_processor(peer.get('url'), exclude_kgs=excl)
                    log.info('Start result for %s: %s', new_peer, result)
                    with self._lock:
                        self.state['active_peer'] = new_peer
                        self.state['last_switch'] = datetime.now(timezone.utc).isoformat()
                        save_director_state(self.state)
            else:
                log.info('No peers with sufficient bandwidth available')

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
        excl = _reserved_kgs(cfg, exclude_peer_id=active_id)
        result = sync_queue_to_peer(peer['url'], exclude=excl)
        if 'error' not in result:
            with self._lock:
                self.state['_last_queue_hash'] = q_hash

    def _loop(self):
        """Main director loop."""
        time.sleep(5)  # startup delay
        sync_counter = 0
        while self._running:
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
                except Exception:
                    pass
                self._update_bandwidth()
                with self._lock:
                    if _clear_completed_reservations(self.cfg):
                        save_peers_config(self.cfg)
                self._check_and_switch()
                # Sync queue every 5 iterations (~2.5 min at 30s interval)
                sync_counter += 1
                if sync_counter >= 5:
                    self._sync_queue_to_active()
                    sync_counter = 0
                with self._lock:
                    save_director_state(self.state)
            except Exception:
                log.exception('Director loop error')
            time.sleep(DIRECTOR_POLL_INTERVAL)


# Singleton
_director: PeerDirector | None = None


def get_director() -> PeerDirector:
    global _director
    if _director is None:
        _director = PeerDirector()
    return _director
