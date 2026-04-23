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
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)

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
PEER_TIMEOUT = 15


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
                return d
            except Exception:
                pass
        return {'state': 'idle'}
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


def sync_queue_to_peer(peer_url: str) -> dict:
    """Push the local priority queue to a remote peer.

    Reads the local retry_queue.json and POSTs it to the peer's
    /api/v1/processing/queue endpoint (position=0 = front of queue).
    Skips if queue is empty.
    """
    queue_path = DATA_DIR / 'retry_queue.json'
    try:
        queue = json.loads(queue_path.read_text()) if queue_path.exists() else []
    except Exception:
        queue = []
    if not queue:
        return {'status': 'empty_queue', 'synced': 0}
    try:
        r = requests.post(
            peer_url.rstrip('/') + '/api/v1/processing/queue',
            json={'kgs': queue, 'position': 0, 'skip_processed': True},
            timeout=PEER_TIMEOUT,
        )
        result = r.json()
        log.info('Queue sync to %s: pushed %d KGs — %s', peer_url, len(queue), result.get('status', '?'))
        return result
    except Exception as e:
        log.warning('Queue sync to %s failed: %s', peer_url, e)
        return {'error': str(e)}


def start_peer_processor(peer_url: str | None) -> dict:
    """Start the processor on a peer.

    For remote peers, syncs the local priority queue first so the peer
    processes the same KGs in priority order.
    """
    if peer_url is None:
        # Local start — use subprocess
        try:
            import sys
            log_file = DATA_DIR / 'logs' / 'processor.log'
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fd = open(log_file, 'a')
            proc = subprocess.Popen(
                [sys.executable, 'austria_processor.py', '--mark-uncertain'],
                stdout=log_fd, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            return {'status': 'started', 'pid': proc.pid}
        except Exception as e:
            return {'error': str(e)}
    # Remote peer — sync priority queue before starting
    queue_result = sync_queue_to_peer(peer_url)
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
    if peer_url is None:
        try:
            subprocess.run(
                ['pkill', '-f', 'austria_processor.py'],
                timeout=5
            )
            # Update progress
            pf = DATA_DIR / 'progress.json'
            if pf.exists():
                d = json.loads(pf.read_text())
                d['state'] = 'stopped'
                pf.write_text(json.dumps(d, indent=2, default=str))
            return {'status': 'stopped'}
        except Exception as e:
            return {'error': str(e)}
    try:
        r = requests.post(
            peer_url.rstrip('/') + '/api/v1/processing/stop',
            timeout=PEER_TIMEOUT
        )
        return r.json()
    except Exception as e:
        return {'error': str(e)}


def trigger_peer_update(peer_url: str) -> dict:
    """Tell a remote peer to git pull and restart its web server."""
    try:
        r = requests.post(
            peer_url.rstrip('/') + '/api/v1/admin/update',
            timeout=60
        )
        return r.json()
    except Exception as e:
        return {'error': str(e)}


def choose_active_peer(cfg: dict, state: dict) -> str | None:
    """Pick the best peer to run the processor on.

    Returns peer_id with most remaining bandwidth, or None if all exhausted.
    """
    budget_gb = cfg.get('budget_gb', BANDWIDTH_BUDGET_GB)
    budget_bytes = budget_gb * (1024 ** 3)
    best_id = None
    best_remaining = -1

    for peer in cfg.get('peers', []):
        if not peer.get('enabled', True):
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
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info('PeerDirector started')

    def stop(self):
        self._running = False

    def get_status(self) -> dict:
        """Full director status for the dashboard."""
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
                'is_active': pid == state.get('active_peer'),
                'processor_state': proc_status,
                'current_kg': (ps.get('current_kg') or {}).get('code'),
                'current_kg_name': (ps.get('current_kg') or {}).get('name'),
                'completed': ps.get('completed', 0),
                'bandwidth': bw,
                'online': proc_status != 'unreachable',
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
                            log.info('Manual mode: starting processor on %s', active_id)
                            start_peer_processor(peer.get('url'))
                return

            # Auto mode — check bandwidth and switch if needed
            active_id = self.state.get('active_peer')
            cfg = self.cfg.copy()
            state_copy = self.state.copy()

        budget_bytes = cfg.get('budget_gb', BANDWIDTH_BUDGET_GB) * (1024 ** 3)

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
                    stop_peer_processor(peer.get('url'))
                    time.sleep(SWITCH_COOLDOWN)
                with self._lock:
                    self.state['active_peer'] = None
                    active_id = None

        # Check if active peer's processor has stopped unexpectedly
        if active_id:
            peer = get_peer_by_id(cfg, active_id)
            if peer:
                status = get_peer_status(peer.get('url'))
                proc_state = status.get('state', 'unknown')
                if proc_state in ('idle', 'stopped'):
                    # Processor stopped (finished a KG or was stopped externally)
                    # Check if it should continue
                    bw = state_copy.get('peer_bandwidth', {}).get(active_id, {})
                    used = bw.get('used_bytes', 0)
                    remaining_gb = (budget_bytes - used) / (1024 ** 3)
                    if remaining_gb >= 2:
                        log.info('Restarting processor on %s (%.1f GB remaining)',
                                 active_id, remaining_gb)
                        start_peer_processor(peer.get('url'))
                    else:
                        log.info('Peer %s depleted, finding next peer', active_id)
                        with self._lock:
                            self.state['active_peer'] = None
                            active_id = None
                elif proc_state == 'unreachable':
                    log.warning('Active peer %s unreachable', active_id)
                    with self._lock:
                        self.state['active_peer'] = None
                        active_id = None

        # If no active peer, choose one
        if not active_id:
            with self._lock:
                new_peer = choose_active_peer(self.cfg, self.state)
            if new_peer:
                peer = get_peer_by_id(cfg, new_peer)
                if peer:
                    log.info('Activating peer %s', new_peer)
                    # Ensure any other peers are stopped
                    for p in cfg.get('peers', []):
                        if p['id'] != new_peer:
                            ps = get_peer_status(p.get('url'))
                            if ps.get('state') in ('running', 'processing'):
                                log.info('Stopping peer %s before switching', p['id'])
                                stop_peer_processor(p.get('url'))

                    result = start_peer_processor(peer.get('url'))
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

        # Push to peer
        result = sync_queue_to_peer(peer['url'])
        if 'error' not in result:
            with self._lock:
                self.state['_last_queue_hash'] = q_hash

    def _loop(self):
        """Main director loop."""
        time.sleep(5)  # startup delay
        sync_counter = 0
        while self._running:
            try:
                self._update_bandwidth()
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
