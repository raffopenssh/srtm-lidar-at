"""Director high-availability — shadow election, watchdog & handover.

Design
------
One peer is the *director* at any time, marked by
``data/austria_processor/is_director``.
The director nominates a *shadow* each tick (most-reliable enabled peer)
and pushes its state snapshot to it every ``SHADOW_SYNC_INTERVAL`` seconds.
The shadow stages the snapshot under ``data/austria_processor/shadow/`` and
holds it warm.

Every peer runs a watchdog thread that pings the current director's
``/api/v1/director/heartbeat`` every ``WATCHDOG_INTERVAL`` seconds. If the
shadow misses ``WATCHDOG_MISS_THRESHOLD`` consecutive heartbeats it promotes
itself: installs the staged snapshot, writes ``is_director``, restarts the
director loop in-process, and broadcasts ``/api/v1/director/announce`` to
all peers so they flip ``zenodo_lock_url.txt`` and their watchdog target.
The old director, if it ever returns, sees ``stepped_down`` (written by the
announce handler) and refuses to start its director loop — it lives on as a
plain peer until manually re-promoted.

Manual handover is the same dance, initiated by
``POST /api/v1/director/handover?to=<peer_id>`` on the current director.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

log = logging.getLogger(__name__)

DATA_DIR = Path('data/austria_processor')
IS_DIRECTOR_FLAG = DATA_DIR / 'is_director'
STEPPED_DOWN_FLAG = DATA_DIR / 'stepped_down'
SELF_FILE = DATA_DIR / 'self.json'
ZENODO_LOCK_URL_FILE = DATA_DIR / 'zenodo_lock_url.txt'
SHADOW_DIR = DATA_DIR / 'shadow'
SHADOW_META_FILE = SHADOW_DIR / 'meta.json'
ADMIN_TOKEN_PATH = Path('data/admin_token')

# Files that make up a director snapshot. All small JSON; total < 200 KB.
# (cache_manifest.json + manifest_tombstones.json already sync via the
# cluster sync thread; we include them in the snapshot so a fresh director
# has them immediately rather than waiting for the next sync tick.)
SNAPSHOT_FILES: tuple[str, ...] = (
    'director_state.json',
    'kg_strikes.json',
    'failure_counts.json',
    'cache_miss_kgs.json',
    'deferred_kgs.json',
    'retry_queue.json',
    'failed_kgs.json',
    'manifest_tombstones.json',
    'copernicus_credentials.json',
    'copernicus_credential_usage.json',
    'peers.json',
    'cache_manifest.json',
)
# Plain-text peer URL list — handled separately because it isn't JSON.
SNAPSHOT_TEXT_FILES: tuple[str, ...] = ('peer_urls.txt',)

WATCHDOG_INTERVAL = 30           # seconds between heartbeat probes
WATCHDOG_MISS_THRESHOLD = 3      # consecutive misses → shadow takes over
WATCHDOG_TIMEOUT = (3, 5)        # (connect, read)
SHADOW_SYNC_INTERVAL = 30        # seconds between snapshot pushes
HEARTBEAT_GRACE = 90             # seconds before director is considered dead


# ── Self identity ────────────────────────────────────────────────

def _derive_self_id_from_hostname() -> str:
    """Derive a peer id from hostname (e.g. ``srtm-lidar-at3`` → ``at3``,
    ``srtm-lidar-at`` → ``primary``).
    """
    h = socket.gethostname().split('.')[0]
    if h.startswith('srtm-lidar-'):
        suffix = h[len('srtm-lidar-'):]
        return 'primary' if suffix == 'at' else suffix
    return h


def load_self() -> dict:
    """Return ``{id, url, director_url}`` for this VM.

    Falls back to derived defaults when the file is missing so a fresh
    install on the primary still works without a self.json.
    """
    try:
        d = json.loads(SELF_FILE.read_text())
        if isinstance(d, dict) and d.get('id'):
            return d
    except Exception:
        pass
    # Best-effort defaults.
    sid = _derive_self_id_from_hostname()
    h = socket.gethostname().split('.')[0]
    derived_url = f'https://{h}.exe.xyz:8000' if h.startswith('srtm-lidar-') else None
    out = {'id': sid, 'url': derived_url, 'director_url': None}
    try:
        if ZENODO_LOCK_URL_FILE.exists():
            out['director_url'] = ZENODO_LOCK_URL_FILE.read_text().strip() or None
    except Exception:
        pass
    # Persist so later loads are stable.
    try:
        save_self(out)
    except Exception:
        pass
    return out


def save_self(d: dict) -> None:
    SELF_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=SELF_FILE.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, SELF_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def self_id() -> str:
    return load_self().get('id') or socket.gethostname()


def self_url() -> str | None:
    return load_self().get('url')


def director_url() -> str | None:
    """URL of the current cluster director, as known by *this* VM.

    Returns None when this VM **is** the director.
    """
    if IS_DIRECTOR_FLAG.exists():
        return None
    s = load_self()
    if s.get('director_url'):
        return s['director_url']
    try:
        if ZENODO_LOCK_URL_FILE.exists():
            return ZENODO_LOCK_URL_FILE.read_text().strip() or None
    except Exception:
        pass
    return None


def set_director_url(url: str | None) -> None:
    """Update both ``self.json`` and ``zenodo_lock_url.txt``.

    Setting ``url=None`` removes both files (this VM is the director).
    """
    s = load_self()
    s['director_url'] = url
    save_self(s)
    if url:
        ZENODO_LOCK_URL_FILE.parent.mkdir(parents=True, exist_ok=True)
        ZENODO_LOCK_URL_FILE.write_text(url + '\n')
    else:
        try:
            ZENODO_LOCK_URL_FILE.unlink()
        except FileNotFoundError:
            pass


# ── Snapshot helpers ─────────────────────────────────────────────

def _admin_headers() -> dict:
    try:
        tok = ADMIN_TOKEN_PATH.read_text().strip()
        if tok:
            return {'X-Admin-Token': tok}
    except Exception:
        pass
    return {}


def build_snapshot() -> dict:
    """Read every snapshot file from disk and return a JSON-safe dict."""
    snap: dict[str, dict | str | None] = {}
    for name in SNAPSHOT_FILES:
        p = DATA_DIR / name
        if not p.exists():
            snap[name] = None
            continue
        try:
            snap[name] = json.loads(p.read_text())
        except Exception as e:
            log.warning('snapshot read %s failed: %s', name, e)
            snap[name] = None
    text: dict[str, str | None] = {}
    for name in SNAPSHOT_TEXT_FILES:
        p = DATA_DIR / name
        try:
            text[name] = p.read_text() if p.exists() else None
        except Exception:
            text[name] = None
    snap['_text'] = text
    snap['_meta'] = {
        'origin': self_id(),
        'origin_url': self_url(),
        'ts': datetime.now(timezone.utc).isoformat(),
        'ts_epoch': time.time(),
    }
    return snap


def stage_snapshot(snap: dict) -> None:
    """Write a received snapshot into ``shadow/`` (warm standby)."""
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    for name in SNAPSHOT_FILES:
        v = snap.get(name)
        out = SHADOW_DIR / name
        if v is None:
            try:
                out.unlink()
            except FileNotFoundError:
                pass
            continue
        _atomic_write(out, json.dumps(v, indent=2))
    for name, v in (snap.get('_text') or {}).items():
        out = SHADOW_DIR / name
        if v is None:
            try:
                out.unlink()
            except FileNotFoundError:
                pass
            continue
        _atomic_write(out, v)
    meta = dict(snap.get('_meta') or {})
    meta['received_ts'] = time.time()
    _atomic_write(SHADOW_META_FILE, json.dumps(meta, indent=2))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def install_snapshot_from_shadow() -> dict:
    """Promote staged snapshot files into ``data/austria_processor/``.

    Called during takeover. Each file is moved atomically.
    Files missing from shadow are left as-is (e.g. retry_queue.json that
    we already have locally is preserved).
    Returns ``{installed: [...], skipped: [...], meta: ...}``.
    """
    if not SHADOW_DIR.exists():
        return {'installed': [], 'skipped': list(SNAPSHOT_FILES),
                'meta': None, 'error': 'no shadow staged'}
    installed: list[str] = []
    skipped: list[str] = []
    for name in list(SNAPSHOT_FILES) + list(SNAPSHOT_TEXT_FILES):
        src = SHADOW_DIR / name
        dst = DATA_DIR / name
        if not src.exists():
            skipped.append(name)
            continue
        try:
            content = src.read_bytes()
            _atomic_write_bytes(dst, content)
            installed.append(name)
        except Exception as e:
            log.warning('install_snapshot %s failed: %s', name, e)
            skipped.append(name)
    meta = None
    try:
        meta = json.loads(SHADOW_META_FILE.read_text())
    except Exception:
        pass
    return {'installed': installed, 'skipped': skipped, 'meta': meta}


def install_snapshot_inline(snap: dict) -> dict:
    """Install a snapshot dict directly (used by manual handover)."""
    stage_snapshot(snap)
    return install_snapshot_from_shadow()


def _normalise_local_identity_in_peers(prev_director_url: str | None) -> dict:
    """Fix up ``peers.json`` and ``peer_urls.txt`` after a snapshot install.

    Snapshots use ``url=None`` to mark the *origin's* local entry. We
    must rewire only that single entry — NEVER touch other peers'
    URLs (cascading handovers used to clobber every ``url=None`` entry
    with stale ``prev_director_url`` values, which is how the cluster
    ended up with multiple peers all pointing at the same URL).

    Rules:
    * Promote our own entry to ``url=None`` (canonical local marker).
    * If the *previous director's* entry is ``url=None`` after install
      (because the snapshot came from them), set its url to
      ``prev_director_url``. Identify that entry by ``id == origin`` in
      ``shadow/meta.json``, NOT by “any other url=None”.
    * Leave every other peer entry alone.
    """
    me = self_id()
    my_url = (self_url() or '').rstrip('/')
    prev_url = (prev_director_url or '').rstrip('/') or None
    # Read the snapshot meta to learn the previous director's id.
    prev_id = None
    try:
        meta = json.loads(SHADOW_META_FILE.read_text())
        prev_id = meta.get('origin')
    except Exception:
        pass
    info: dict = {'changes': [], 'prev_id': prev_id, 'prev_url': prev_url}
    pj = DATA_DIR / 'peers.json'
    try:
        cfg = json.loads(pj.read_text())
    except Exception as e:
        info['peers_json_error'] = str(e)[:200]
        cfg = None
    if isinstance(cfg, dict) and isinstance(cfg.get('peers'), list):
        peers = cfg['peers']
        # 1. Promote our own entry to url=None.
        promoted = False
        for p in peers:
            if p.get('id') == me:
                if p.get('url') is not None:
                    info['changes'].append(
                        f'set {me} url to None (local marker)')
                p['url'] = None
                p['enabled'] = True
                promoted = True
                break
        if not promoted:
            peers.insert(0, {'id': me, 'url': None, 'enabled': True})
            info['changes'].append(f'added missing local entry {me}')
        # 2. Heal the previous director's entry IFF identified by
        #    snapshot origin and currently url=None and not us.
        if prev_id and prev_id != me and prev_url:
            for p in peers:
                if p.get('id') == prev_id and p.get('url') is None:
                    p['url'] = prev_url
                    info['changes'].append(
                        f'set {prev_id} url to prev_director {prev_url}')
                    break
        try:
            _atomic_write(pj, json.dumps(cfg, indent=2))
        except Exception as e:
            info['peers_json_write_error'] = str(e)[:200]
    # peer_urls.txt: drop our URL, ensure prev director URL is present.
    pu = DATA_DIR / 'peer_urls.txt'
    try:
        if pu.exists():
            urls = [u.strip().rstrip('/') for u in pu.read_text().splitlines()
                    if u.strip()]
        else:
            urls = []
        out: list[str] = []
        for u in urls:
            if my_url and u == my_url:
                continue
            if u not in out:
                out.append(u)
        if prev_url and prev_url not in out:
            out.append(prev_url)
            info['changes'].append(f'added prev_director {prev_url} to peer_urls')
        if out != urls:
            _atomic_write(pu, '\n'.join(out) + ('\n' if out else ''))
    except Exception as e:
        info['peer_urls_error'] = str(e)[:200]
    return info


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Director ↔ shadow plumbing ───────────────────────────────────

def push_snapshot_to_shadow(shadow_url: str, snap: dict | None = None,
                            timeout: tuple = (5, 20)) -> dict:
    """PUT a snapshot to shadow. Returns the peer's response or error dict."""
    if snap is None:
        snap = build_snapshot()
    try:
        r = requests.put(
            shadow_url.rstrip('/') + '/api/v1/director/snapshot',
            json=snap, headers=_admin_headers(), timeout=timeout,
        )
        if r.ok:
            return r.json()
        return {'error': f'http_{r.status_code}', 'body': r.text[:200]}
    except Exception as e:
        return {'error': str(e)[:200]}


def ping_heartbeat(director_url_str: str,
                   timeout: tuple = WATCHDOG_TIMEOUT) -> dict | None:
    try:
        r = requests.get(
            director_url_str.rstrip('/') + '/api/v1/director/heartbeat',
            headers=_admin_headers(), timeout=timeout,
        )
        if r.ok:
            return r.json()
    except Exception:
        return None
    return None


def announce_to_peer(peer_url: str, payload: dict,
                     timeout: tuple = (5, 8)) -> dict:
    try:
        r = requests.post(
            peer_url.rstrip('/') + '/api/v1/director/announce',
            json=payload, headers=_admin_headers(), timeout=timeout,
        )
        if r.ok:
            return r.json()
        return {'error': f'http_{r.status_code}'}
    except Exception as e:
        return {'error': str(e)[:200]}


def broadcast_announce(peer_urls: Iterable[str], payload: dict,
                       max_workers: int = 10) -> dict[str, dict]:
    from concurrent.futures import ThreadPoolExecutor
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers,
                            thread_name_prefix='dir-announce') as ex:
        futs = {ex.submit(announce_to_peer, u, payload): u for u in peer_urls}
        for f, url in futs.items():
            try:
                results[url] = f.result(timeout=12)
            except Exception as e:
                results[url] = {'error': str(e)[:200]}
    return results


# ── Watchdog (runs on every VM) ──────────────────────────────────

class _WatchdogState:
    def __init__(self) -> None:
        self.misses = 0
        self.last_heartbeat: dict | None = None
        self.last_director_url: str | None = None
        self.takeover_in_flight = False
        self.lock = threading.Lock()


_WATCHDOG = _WatchdogState()


def watchdog_state() -> dict:
    with _WATCHDOG.lock:
        return {
            'misses': _WATCHDOG.misses,
            'last_heartbeat': _WATCHDOG.last_heartbeat,
            'last_director_url': _WATCHDOG.last_director_url,
            'takeover_in_flight': _WATCHDOG.takeover_in_flight,
            'is_director': IS_DIRECTOR_FLAG.exists(),
            'stepped_down': STEPPED_DOWN_FLAG.exists(),
            'self': load_self(),
            'shadow_meta': _read_json_safe(SHADOW_META_FILE),
        }


def _read_json_safe(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def start_watchdog() -> None:
    """Spawn a daemon thread that watches the director liveness."""
    t = threading.Thread(target=_watchdog_loop, daemon=True,
                         name='director-watchdog')
    t.start()


def _watchdog_loop() -> None:
    time.sleep(20)  # let services boot
    while True:
        try:
            _watchdog_tick()
        except Exception as e:
            log.warning('director watchdog tick failed: %s', e)
        time.sleep(WATCHDOG_INTERVAL)


def _watchdog_tick() -> None:
    if IS_DIRECTOR_FLAG.exists():
        # We *are* the director — nothing to watch.
        with _WATCHDOG.lock:
            _WATCHDOG.misses = 0
            _WATCHDOG.takeover_in_flight = False
        return
    durl = director_url()
    if not durl:
        return
    with _WATCHDOG.lock:
        _WATCHDOG.last_director_url = durl
    hb = ping_heartbeat(durl)
    if hb is not None:
        with _WATCHDOG.lock:
            _WATCHDOG.misses = 0
            _WATCHDOG.last_heartbeat = hb
        return
    with _WATCHDOG.lock:
        _WATCHDOG.misses += 1
        misses = _WATCHDOG.misses
        in_flight = _WATCHDOG.takeover_in_flight
    if misses < WATCHDOG_MISS_THRESHOLD or in_flight:
        return
    # Only the *shadow* takes over. Read shadow meta to confirm.
    meta = _read_json_safe(SHADOW_META_FILE) or {}
    designated_shadow = (meta.get('shadow_id')
                         or (meta.get('director_state') or {}).get('shadow_peer'))
    me = self_id()
    if designated_shadow != me:
        log.info('director watchdog: %d misses but I am not shadow '
                 '(shadow=%s, me=%s) — waiting', misses, designated_shadow, me)
        return
    # Freshness gate: stale shadow_meta is the #1 cause of split-brain.
    # A peer that *was* shadow once, then was de-elected, keeps a stale
    # meta.json forever. On any later heartbeat blip the watchdog reads
    # 'shadow_id == me' and re-promotes. We refuse to promote unless the
    # meta was received recently (3 × push interval = 90 s).
    received_ts = float(meta.get('received_ts') or 0.0)
    age = time.time() - received_ts
    SHADOW_META_FRESH_S = SHADOW_SYNC_INTERVAL * 3
    if received_ts <= 0 or age > SHADOW_META_FRESH_S:
        log.warning('director watchdog: shadow meta stale (age=%.0fs > %ds) — '
                    'refusing to promote, will wait for fresh push',
                    age, SHADOW_META_FRESH_S)
        return
    log.warning('director watchdog: director %s missed %d heartbeats — '
                'promoting self (%s) to director', durl, misses, me)
    with _WATCHDOG.lock:
        _WATCHDOG.takeover_in_flight = True
    try:
        _do_takeover(reason=f'watchdog_{misses}_misses',
                     prev_director_url=durl)
    finally:
        with _WATCHDOG.lock:
            _WATCHDOG.takeover_in_flight = False


# ── Takeover & step-down ─────────────────────────────────────────

def _do_takeover(reason: str, prev_director_url: str | None = None,
                snapshot_inline: dict | None = None) -> dict:
    """Promote this VM to director. Idempotent.

    Steps:
      1. Install snapshot (from inline payload or staged shadow/).
      2. Write ``is_director`` flag, remove ``stepped_down``.
      3. Remove our own ``zenodo_lock_url.txt`` / clear ``director_url``
         (we are now the broker).
      4. Start the director loop in-process.
      5. Broadcast announce to every peer (peers flip their pointer).
      6. Tell the previous director (if reachable) to step down.
    """
    if snapshot_inline is not None:
        install_result = install_snapshot_inline(snapshot_inline)
    else:
        install_result = install_snapshot_from_shadow()
    log.info('takeover: snapshot install: %s', install_result)
    # Snapshots use url=None as a 'this host' marker which now points
    # at the previous director, not us. Rewire peers.json/peer_urls.txt
    # before the director loop reads them.
    try:
        ident_fix = _normalise_local_identity_in_peers(prev_director_url)
        log.info('takeover: identity normalise: %s', ident_fix)
        install_result = dict(install_result or {})
        install_result['identity_fix'] = ident_fix
    except Exception as e:
        log.warning('takeover: identity normalise failed: %s', e)
    try:
        STEPPED_DOWN_FLAG.unlink()
    except FileNotFoundError:
        pass
    IS_DIRECTOR_FLAG.parent.mkdir(parents=True, exist_ok=True)
    IS_DIRECTOR_FLAG.write_text(f'{self_id()} (took over via {reason} '
                                f'at {datetime.now(timezone.utc).isoformat()})\n')
    # We are the director now → no zenodo_lock_url pointer.
    set_director_url(None)
    # Restart director loop in-process. Replace the singleton so
    # __init__ re-reads the freshly-installed director_state.json
    # (capacity_ema, capacity_history, peer_noise_long_ema, ...).
    try:
        import peer_director as pd  # noqa: WPS433
        try:
            old = pd.get_director()
            old.stop()
        except Exception:
            pass
        try:
            pd._director = None  # force fresh instance on next get_director()
        except Exception:
            pass
        d = pd.get_director()
        d.start()
    except Exception as e:
        log.warning('takeover: director restart failed: %s', e)
    # Announce to every peer.
    my_url = self_url() or ''
    payload = {
        'new_director_id': self_id(),
        'new_director_url': my_url,
        'prev_director_url': prev_director_url,
        'reason': reason,
        'ts': datetime.now(timezone.utc).isoformat(),
    }
    # _all_peer_urls() only returns peers with non-null URLs, which used
    # to skip the previous director (whose url=None marker we just
    # rewrote, but the old director's local entry is canonical url=None
    # in *its* peers.json). Make sure the previous director URL is in
    # the announce list explicitly.
    peer_urls = _all_peer_urls(exclude=[my_url])
    if prev_director_url:
        prev_norm = prev_director_url.rstrip('/')
        if prev_norm and prev_norm not in (my_url.rstrip('/'),) \
                and prev_norm not in peer_urls:
            peer_urls.append(prev_norm)
    results = broadcast_announce(peer_urls, payload)
    log.info('takeover: announced to %d peers', len(results))
    # Tell the old director to step down (best-effort).
    if prev_director_url and prev_director_url not in (my_url, ''):
        try:
            r = requests.post(
                prev_director_url.rstrip('/') + '/api/v1/director/step_down',
                json=payload, headers=_admin_headers(), timeout=(3, 5),
            )
            log.info('takeover: step_down on %s: %s',
                     prev_director_url, r.status_code)
        except Exception as e:
            log.info('takeover: prev director %s unreachable for step_down: %s',
                     prev_director_url, e)
    return {'status': 'promoted', 'install': install_result,
            'announced': len(results), 'reason': reason}


def _all_peer_urls(exclude: Iterable[str] = ()) -> list[str]:
    excl = {u.rstrip('/') for u in exclude if u}
    out: list[str] = []
    try:
        cfg = json.loads((DATA_DIR / 'peers.json').read_text())
        for p in cfg.get('peers', []):
            u = (p.get('url') or '').rstrip('/')
            if u and u not in excl:
                out.append(u)
    except Exception:
        pass
    return out


def step_down(announce_payload: dict) -> dict:
    """Voluntarily relinquish director role.

    Removes ``is_director``, writes ``stepped_down`` (so a cold restart
    won't auto-revive us as director), updates pointers, stops the
    director loop. Safe to call when we already aren't the director.
    """
    new_url = (announce_payload or {}).get('new_director_url') or None
    was_director = IS_DIRECTOR_FLAG.exists()
    try:
        IS_DIRECTOR_FLAG.unlink()
    except FileNotFoundError:
        pass
    # Invalidate any stale shadow_meta. We are no longer the shadow
    # of anyone (we just stepped down). Without this, a future heartbeat
    # blip would let our watchdog re-promote based on a meta file that
    # still says 'shadow_id == me'. This is the cascade trigger.
    try:
        SHADOW_META_FILE.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass
    STEPPED_DOWN_FLAG.write_text(
        json.dumps({
            'ts': datetime.now(timezone.utc).isoformat(),
            'new_director_id': (announce_payload or {}).get('new_director_id'),
            'new_director_url': new_url,
            'reason': (announce_payload or {}).get('reason', 'step_down'),
        }, indent=2))
    if new_url:
        set_director_url(new_url)
    if was_director:
        try:
            import peer_director as pd  # noqa: WPS433
            pd.get_director().stop()
            log.warning('step_down: director loop stopped (handed over to %s)',
                        new_url or '?')
        except Exception as e:
            log.warning('step_down: failed to stop director loop: %s', e)
    return {'status': 'stepped_down', 'was_director': was_director,
            'new_director_url': new_url}


def accept_announce(payload: dict) -> dict:
    """Handle an inbound ``/director/announce`` notification.

    If the announce names *us* as the new director, ignore (we already know).
    If we currently believe ourselves to be the director, step down.
    Otherwise just flip the pointer to the new director URL.
    """
    new_id = payload.get('new_director_id')
    new_url = payload.get('new_director_url')
    if not new_url:
        return {'status': 'rejected', 'reason': 'no_new_director_url'}
    if new_id == self_id():
        return {'status': 'self', 'note': 'I am the new director'}
    if IS_DIRECTOR_FLAG.exists():
        return step_down(payload)
    set_director_url(new_url)
    # Same invariant as in step_down: any peer that receives an announce
    # is now a follower of ``new_id``, so its prior shadow_meta (if any)
    # is stale by definition.
    try:
        SHADOW_META_FILE.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass
    # Clear watchdog miss counter so we don't immediately try to re-takeover.
    with _WATCHDOG.lock:
        _WATCHDOG.misses = 0
        _WATCHDOG.last_director_url = new_url
    return {'status': 'pointed_at', 'director_url': new_url}


# ── Manual handover (from current director) ──────────────────────

def do_handover(target_peer_id: str, target_peer_url: str) -> dict:
    """Initiated *by* the current director. Ships state to target and
    asks it to take over, then steps down locally.
    """
    if not IS_DIRECTOR_FLAG.exists():
        return {'error': 'not_director',
                'note': 'Only the current director may initiate handover.'}
    snap = build_snapshot()
    # Annotate snapshot with shadow_id so the target's takeover gate accepts it.
    snap['_meta'] = dict(snap.get('_meta') or {})
    snap['_meta']['shadow_id'] = target_peer_id
    payload = {
        'reason': 'manual_handover',
        'prev_director_url': self_url(),
        'snapshot': snap,
    }
    try:
        r = requests.post(
            target_peer_url.rstrip('/') + '/api/v1/director/takeover',
            json=payload, headers=_admin_headers(), timeout=(5, 30),
        )
        if not r.ok:
            return {'error': f'http_{r.status_code}', 'body': r.text[:200]}
        body = r.json()
    except Exception as e:
        return {'error': str(e)[:200]}
    # Target will broadcast announce (which will tell us to step down too,
    # but step down here proactively so the loop on this box stops fast).
    step_down({
        'new_director_id': target_peer_id,
        'new_director_url': target_peer_url,
        'reason': 'manual_handover',
    })
    return {'status': 'handed_over', 'target': target_peer_id,
            'target_response': body}
