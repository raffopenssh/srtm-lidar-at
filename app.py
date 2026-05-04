"""Austrian LIDAR & Orthophoto Analysis API.

Endpoints:
  1. POST /api/v1/elevation        — Enrich features with DSM/DTM elevation
  2. POST /api/v1/terrain          — Terrain characterisation (slope, ruggedness, …)
  3. POST /api/v1/changes          — Temporal change detection between ALS dates
  6. POST /api/v1/changes/trees    — Per-tree growth / felling analysis
  7. POST /api/v1/changes/summary  — Multi-epoch change summary
  8. GET  /api/v1/info             — Datasets, object types, event types
  9. GET  /api/v1/docs/llm.txt     — Machine-readable API reference
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import pickle
import sys
import re
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from flask import Flask, request, jsonify, send_file, send_from_directory, Response, redirect
from shapely.geometry import mapping, shape, Point, LineString as SLineString

import tile_index as ti
import raster_io
import terrain_analysis as ta
import object_segmentation as seg  # watershed-based segmentation
import hansen  # Hansen Global Forest Change calibration
import temporal_analysis as tca
import geo_parse
import search_index as si
import cadastre_bridge as cb
import feedback_db
import quality_flags

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')

# === Admin auth ===
# Shared cluster-wide secret stored in data/admin_token. Required header
# X-Admin-Token (or ?admin_token=) on every mutating admin/director/processing
# /zenodo endpoint. Loopback requests (127.0.0.1) are exempt so the in-process
# director and dashboard fetches via /admin/token bootstrap work without
# plumbing the secret everywhere. The token is auto-generated on first start
# and is replicated by deploy.sh + peer-sync to every peer.

ADMIN_TOKEN_PATH = Path('data/admin_token')


def _load_or_create_admin_token() -> str:
    """Load shared admin token, creating one if missing.

    Lives in data/admin_token (gitignored). deploy.sh copies the primary's
    token to peers; the data sync thread keeps it consistent.
    """
    try:
        if ADMIN_TOKEN_PATH.exists():
            tok = ADMIN_TOKEN_PATH.read_text().strip()
            if tok:
                return tok
    except Exception:
        pass
    import secrets
    tok = secrets.token_urlsafe(32)
    try:
        ADMIN_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        ADMIN_TOKEN_PATH.write_text(tok)
        ADMIN_TOKEN_PATH.chmod(0o600)
    except Exception as e:
        log.warning('admin_token: could not persist token: %s', e)
    return tok


ADMIN_TOKEN = _load_or_create_admin_token()


def _current_admin_token() -> str:
    """Read the admin token fresh each time.

    Lets `/api/v1/admin/install_token` rotate the cluster secret without
    a srv restart — the next request picks up the new value.
    """
    global ADMIN_TOKEN
    try:
        if ADMIN_TOKEN_PATH.exists():
            tok = ADMIN_TOKEN_PATH.read_text().strip()
            if tok:
                ADMIN_TOKEN = tok
                return tok
    except Exception:
        pass
    return ADMIN_TOKEN

# URL prefixes that require the admin token. GET-only inspection endpoints
# (/director/status, /processing/queue, /processing/peers, /bandwidth, ...)
# stay open so the dashboard remains usable from anywhere.
_PROTECTED_PREFIXES = (
    '/api/v1/admin/',
    '/api/v1/director/mode',
    '/api/v1/director/activate',
    '/api/v1/director/stop',
    '/api/v1/director/peers',  # POST/PUT/DELETE; GET also requires token (contains URLs)
    '/api/v1/director/throttle',
    '/api/v1/director/update_peers',
    '/api/v1/director/restart_peer',
    '/api/v1/director/handover',
    '/api/v1/director/takeover',
    '/api/v1/director/step_down',
    '/api/v1/director/announce',
    '/api/v1/director/snapshot',
    '/api/v1/processing/start',
    '/api/v1/processing/stop',
    '/api/v1/processing/pause',
    '/api/v1/processing/resume',
    '/api/v1/processing/single',
    '/api/v1/processing/retry',
    '/api/v1/processing/throttle',
    '/api/v1/processing/queue',
    '/api/v1/processing/cache_manifest',
    '/api/v1/processing/kg_strikes',
    '/api/v1/zenodo/lock',
    '/api/v1/credentials',
    '/api/v1/processing/cache_misses',
)


# Endpoints that handle their own auth (before_request must let them through).
_AUTH_SELF_HANDLED = ('/api/v1/admin/install_token',)


# Paths where GET is read-only inspection (let through) but POST/PUT/DELETE
# still require the admin token. The dashboard polls these without a token.
_PROTECTED_GET_OPEN = (
    '/api/v1/processing/queue',
    '/api/v1/processing/cache_manifest',
    '/api/v1/processing/kg_strikes',
)


def _is_protected_path(path: str, method: str) -> bool:
    if path in _AUTH_SELF_HANDLED:
        return False
    if method == 'GET' and any(path.startswith(p) for p in _PROTECTED_GET_OPEN):
        return False
    return any(path.startswith(p) for p in _PROTECTED_PREFIXES)


def _request_is_loopback() -> bool:
    """True for requests originating on this host.

    gunicorn behind exe.dev's proxy: remote_addr is 127.0.0.1 for proxied
    traffic too, so we also require X-Forwarded-For to be absent for the
    loopback bypass to apply.
    """
    try:
        if request.remote_addr not in ('127.0.0.1', '::1'):
            return False
        if request.headers.get('X-Forwarded-For'):
            return False
        return True
    except Exception:
        return False


@app.before_request
def _enforce_admin_token():
    path = request.path or ''
    if not _is_protected_path(path, request.method):
        return None
    if _request_is_loopback():
        return None
    tok = (request.headers.get('X-Admin-Token')
           or request.args.get('admin_token')
           or request.cookies.get('admin_token'))
    if tok and tok == _current_admin_token():
        return None
    return jsonify({'error': 'admin token required',
                    'hint': 'set X-Admin-Token header'}), 401


@app.route('/api/v1/admin/token', methods=['GET'])
def admin_token_bootstrap():
    """Loopback-only endpoint: returns the admin token.

    The dashboard fetches this on load so the operator's browser can call
    protected endpoints. Public callers get 401 from the before_request hook
    (this route is under /api/v1/admin/ so it is protected).
    """
    return jsonify({'token': _current_admin_token()})


@app.route('/api/v1/admin/install_token', methods=['POST'])
def admin_install_token():
    """Install/rotate the cluster admin token.

    Authentication rules (the before_request hook lets this route through;
    we authenticate here ourselves):
      * If this peer currently has NO token (data/admin_token missing or
        empty), accept any non-empty new token — first-write-wins
        bootstrap. Used by the director to seed peers running pre-auth
        code that have just been updated.
      * Otherwise the request must present a valid X-Admin-Token (the
        peer's current token) OR pass current_token in the body.

    Body JSON: {"new_token": "...", "current_token": "<optional>"}
    """
    global ADMIN_TOKEN
    body = request.get_json(silent=True) or {}
    new_tok = (body.get('new_token') or '').strip()
    if not new_tok or len(new_tok) < 16:
        return jsonify({'error': 'new_token must be a non-empty string >=16 chars'}), 400

    # Loopback always allowed (on-box CLI / in-process callers).
    if not _request_is_loopback():
        existing = ''
        try:
            if ADMIN_TOKEN_PATH.exists():
                existing = ADMIN_TOKEN_PATH.read_text().strip()
        except Exception:
            existing = ''
        if existing:
            presented = (request.headers.get('X-Admin-Token')
                         or request.args.get('admin_token')
                         or body.get('current_token') or '')
            if presented != existing:
                return jsonify({
                    'error': 'token already installed; current X-Admin-Token required to rotate',
                }), 401
        # else: bootstrap, allow.

    try:
        ADMIN_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        ADMIN_TOKEN_PATH.write_text(new_tok)
        ADMIN_TOKEN_PATH.chmod(0o600)
    except Exception as e:
        return jsonify({'error': f'failed to write token: {e}'}), 500
    ADMIN_TOKEN = new_tok
    log.info('admin_install_token: token installed/rotated (len=%d)', len(new_tok))
    return jsonify({'status': 'installed'})


# === SECTION: Copernicus credentials API ===

@app.route('/api/v1/credentials', methods=['GET'])
def credentials_list():
    """List Copernicus credentials known to this peer (no secrets).

    GET is allowed without admin token to support read-only dashboards;
    the response never includes client_secret.
    """
    try:
        import copernicus as _cop
        return jsonify({'credentials': _cop.list_credentials()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/credentials', methods=['POST'])
def credentials_add():
    """Add a Copernicus credential pair, validate, and persist.

    Body JSON: {"client_id":"sh-...","client_secret":"...",
                "label":"optional","notes":"optional",
                "validate": true|false}
    """
    data = request.get_json(silent=True) or {}
    cid = (data.get('client_id') or '').strip()
    csec = (data.get('client_secret') or '').strip()
    if not cid or not csec:
        return jsonify({'error': 'client_id and client_secret required'}), 400
    import copernicus as _cop
    res = _cop.add_credential(
        cid, csec,
        label=(data.get('label') or '').strip(),
        notes=(data.get('notes') or '').strip(),
        validate=bool(data.get('validate', True)),
    )
    if not res.get('ok'):
        return jsonify(res), 400
    return jsonify(res)


@app.route('/api/v1/credentials/<client_id>', methods=['DELETE'])
def credentials_remove(client_id):
    import copernicus as _cop
    res = _cop.remove_credential(client_id)
    if not res.get('ok'):
        return jsonify(res), 404
    return jsonify(res)


@app.route('/api/v1/credentials/validate', methods=['POST'])
def credentials_validate():
    """Probe one credential pair without storing. Body: client_id, client_secret.
    Or POST with no body to revalidate all known credentials."""
    data = request.get_json(silent=True) or {}
    import copernicus as _cop
    if data.get('client_id') and data.get('client_secret'):
        return jsonify(_cop.validate_credential(
            data['client_id'], data['client_secret']))
    # No body — revalidate all
    return jsonify({'results': _cop.revalidate_all_credentials()})


# Initialize search index + watch for new KG JSON files
class _GitBusy(RuntimeError):
    """Raised when another git sync is already running on this VM."""
    pass


_GIT_SYNC_LOCK_PATH = '/tmp/srtm_git_sync.lock'


def _safe_git_sync(repo: str, sp):
    """Robust git pull for `/admin/update` and the deferred-update path.

    Failure modes seen in production rollouts (15+ peers, observed 2026-04-29):
      * Stale .git/index.lock from crashed/concurrent git ops -> every
        subsequent update fails until manual cleanup.
      * Concurrent /admin/update calls racing on the same repo.
      * `git pull` 30s timeout too tight on cold-cache / slow-network VMs.
      * `git checkout -- .` 10s timeout too tight when the working tree is
        large (data/* dir mtimes get stat-walked).

    This helper:
      1. Takes an exclusive non-blocking flock on /tmp/srtm_git_sync.lock
         so concurrent calls fail fast (409) instead of corrupting state.
      2. Removes any .git/index.lock older than 60s (left by crashed git).
      3. Bumps timeouts: checkout 30s, fetch 60s, pull 120s.
      4. Uses `git fetch` then `git reset --hard origin/<branch>` instead of
         `pull --ff-only` -- robust against local commits/divergence and
         deterministic across 150 peers.
    Returns the CompletedProcess of the final reset (with stdout/stderr that
    callers stuff into the JSON response).
    """
    import fcntl, os as _os, time as _t
    from pathlib import Path as _P
    lock_fd = _os.open(_GIT_SYNC_LOCK_PATH, _os.O_CREAT | _os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise _GitBusy('git sync already in progress on this VM')
        # Stale .git/index.lock cleanup (60s grace).
        idx_lock = _P(repo) / '.git' / 'index.lock'
        try:
            if idx_lock.exists() and (_t.time() - idx_lock.stat().st_mtime) > 60:
                idx_lock.unlink()
        except Exception:
            pass
        # Determine current branch (default main).
        branch = sp.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                        capture_output=True, text=True, timeout=10,
                        cwd=repo).stdout.strip() or 'main'
        # Reset any tracked file modifications.
        sp.run(['git', 'checkout', '--', '.'], capture_output=True, text=True,
               timeout=30, cwd=repo)
        # Fetch + hard-reset (robust against divergence; deterministic).
        fetch = sp.run(['git', 'fetch', 'origin', branch],
                       capture_output=True, text=True, timeout=60, cwd=repo)
        reset = sp.run(['git', 'reset', '--hard', f'origin/{branch}'],
                       capture_output=True, text=True, timeout=30, cwd=repo)
        # Synthesize a pull-like CompletedProcess for the response.
        reset.stdout = (fetch.stdout + reset.stdout).strip()
        reset.stderr = (fetch.stderr + reset.stderr).strip()
        return reset
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except Exception:
            pass
        _os.close(lock_fd)


_INDEX_WATCH_LOCK_PATH = '/tmp/srtm_index_watch.lock'


def _init_search_index():
    import fcntl as _fcntl, os as _os
    # Demoted peers (not primary/director/shadow) skip the index entirely.
    # The role-data eviction loop will purge any leftover database files
    # after the grace period. This avoids rebuilding ~5 GB of FTS+R-tree
    # indices on a peer that's about to delete them anyway.
    try:
        if not _is_keep_role_data():
            log.info('🔍 Search index disabled on demoted peer (not primary/director/shadow)')
            return
    except Exception:
        pass
    try:
        idx = si.get_index()
        feedback_db.ensure_schema()
        json_dir = Path('data/austria_processor/json')

        # Acquire single-worker lock for the watcher loop. The other gunicorn
        # worker exits this function early so we don't double-build.
        lock_fd = _os.open(_INDEX_WATCH_LOCK_PATH,
                           _os.O_CREAT | _os.O_RDWR, 0o644)
        try:
            _fcntl.flock(lock_fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except BlockingIOError:
            log.info('🔍 Search index watcher already running in another worker; skipping')
            try: _os.close(lock_fd)
            except Exception: pass
            return

        # Initial build (only the lock-holder does this)
        idx.build()

        # Initial quality_flags sweep
        try:
            for jp in json_dir.glob('*.json'):
                try: quality_flags.scan_json(jp)
                except Exception as e: log.warning('quality_flags initial scan %s: %s', jp.name, e)
        except Exception as e:
            log.warning('quality_flags initial sweep: %s', e)

        # Watch for new/updated JSON files every 60s. Use mtime tracking so
        # that re-uploaded JSONs (e.g. after backfill) get re-enriched.
        def _snapshot():
            if not json_dir.exists():
                return {}
            out = {}
            for f in json_dir.glob('*.json'):
                try: out[f.stem] = f.stat().st_mtime
                except OSError: pass
            return out

        known = _snapshot()
        while True:
            time.sleep(60)
            try:
                current = _snapshot()
                changed = [code for code, mt in current.items()
                           if known.get(code) != mt]
                if not changed:
                    continue
                preview = changed[:10] + (['...'] if len(changed) > 10 else [])
                log.info('🔍 %d new/updated KG JSON(s): %s',
                         len(changed), preview)
                # Incremental update -- no full rebuild
                manifest = {}
                mp = Path('data/austria_processor/zenodo_manifest.json')
                if mp.exists():
                    try:
                        md = json.loads(mp.read_text())
                        manifest = md.get('entries', md)
                    except Exception:
                        pass
                for code in changed:
                    jp = json_dir / f'{code}.json'
                    try:
                        idx.update_kg(code, json_path=str(jp), manifest=manifest)
                    except Exception as e:
                        log.warning('index update %s: %s', code, e)
                    try: quality_flags.scan_json(jp)
                    except Exception as e: log.warning('quality_flags scan %s: %s', code, e)
                known = current
            except Exception as e:
                log.warning('Search index watch: %s', e)
    except Exception as e:
        log.warning('Search index init failed: %s', e)
threading.Thread(target=_init_search_index, daemon=True).start()


def _get_peer_urls():
    """Read peer URLs from peer_urls.txt or from systemd service --peers flag."""
    pf = Path('data/austria_processor/peer_urls.txt')
    if pf.exists():
        return [u.strip() for u in pf.read_text().splitlines() if u.strip() and not u.startswith('#')]
    # Fallback: parse from systemd service / drop-in configs
    svc_paths = list(Path('/etc/systemd/system/austria_processor.service.d').glob('*.conf')) \
        if Path('/etc/systemd/system/austria_processor.service.d').exists() else []
    svc_paths.append(Path('/etc/systemd/system/austria_processor.service'))
    for svc in svc_paths:
        try:
            if not svc.exists():
                continue
            for line in svc.read_text().splitlines():
                if '--peers' in line:
                    parts = line.split('--peers')[1].strip().split()
                    urls = []
                    for p in parts:
                        if p.startswith('http'):
                            urls.append(p)
                        elif p.startswith('--'):
                            break
                    if urls:
                        return urls
        except PermissionError:
            continue
    return []


# Manifest keys deleted intentionally — don't re-merge from peers
# Dict of key -> ISO timestamp when tombstoned. Entries newer than tombstone are allowed.
_MANIFEST_TOMBSTONES: dict = {}
_tombstone_path = Path('data/austria_processor/manifest_tombstones.json')
if _tombstone_path.exists():
    try:
        raw = json.loads(_tombstone_path.read_text())
        if isinstance(raw, list):
            # Migrate old format (list of keys) → dict with current timestamp
            _MANIFEST_TOMBSTONES = {k: datetime.utcnow().isoformat() for k in raw}
            _tombstone_path.write_text(json.dumps(_MANIFEST_TOMBSTONES, indent=2))
        elif isinstance(raw, dict):
            _MANIFEST_TOMBSTONES = raw
    except Exception: pass

def _sync_peer_data():
    """Background thread: sync KG JSONs and manifest entries from peers."""
    import requests as req
    time.sleep(30)  # Wait for startup

    json_dir = Path('data/austria_processor/json')
    json_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path('data/austria_processor/zenodo_manifest.json')

    while True:
        try:
            # Skip JSON download + manifest merge entirely on demoted peers
            # (not primary, not director, not shadow). Those VMs are about to
            # purge their JSON corpus via the role-data eviction loop — there
            # is no point downloading new ones in the meantime.
            try:
                if not _is_keep_role_data():
                    time.sleep(300)
                    continue
            except Exception:
                pass
            peer_urls = _get_peer_urls()
            if not peer_urls:
                time.sleep(300)
                continue

            new_count = 0
            merged_manifest_entries = {}

            for peer_url in peer_urls:
                try:
                    r = req.get(peer_url.rstrip('/') + '/api/v1/processing/peers', timeout=15)
                    r.raise_for_status()
                    peer_data = r.json()
                except Exception as e:
                    log.debug('Peer sync: %s unreachable: %s', peer_url, e)
                    continue

                peer_manifest = peer_data.get('manifest', {})
                peer_tombstones = peer_data.get('tombstones', {}) or {}

                # Merge peer tombstones into ours (newest timestamp wins).
                # Propagates force-requeue requests across the fleet so
                # peers with stale local JSONs don't silently skip a
                # re-queued KG.
                if isinstance(peer_tombstones, dict) and peer_tombstones:
                    changed_tomb = False
                    for tk, tv in peer_tombstones.items():
                        if not isinstance(tv, str):
                            continue
                        cur = _MANIFEST_TOMBSTONES.get(tk, '')
                        if tv > cur:
                            _MANIFEST_TOMBSTONES[tk] = tv
                            changed_tomb = True
                    if changed_tomb:
                        try:
                            _tombstone_path.write_text(
                                json.dumps(_MANIFEST_TOMBSTONES, indent=2))
                            log.info('Peer sync: merged tombstones from peer (now %d entries)',
                                     len(_MANIFEST_TOMBSTONES))
                        except Exception:
                            pass

                # Download KG JSONs we don't have, OR re-download when the
                # peer's manifest entry is newer/larger than our local copy
                # (e.g. after a backfill rewrite on a peer).
                for key, entry in peer_manifest.items():
                    if not key.endswith('_json'):
                        continue
                    code = key.replace('_json', '')
                    local_path = json_dir / f'{code}.json'
                    needs_dl = True
                    if local_path.exists():
                        try:
                            local_size = local_path.stat().st_size
                            local_mtime = local_path.stat().st_mtime
                            remote_size = int(entry.get('size') or 0)
                            remote_ts = entry.get('uploaded_at') or ''
                            # Compare mtime vs uploaded_at if both available
                            from datetime import datetime as _dt2
                            remote_mtime = 0.0
                            if remote_ts:
                                try:
                                    remote_mtime = _dt2.fromisoformat(
                                        remote_ts.replace('Z', '+00:00')).timestamp()
                                except Exception:
                                    pass
                            # Skip only if local is up-to-date AND same size
                            if remote_mtime and local_mtime >= remote_mtime - 5:
                                needs_dl = False
                            elif remote_size and abs(remote_size - local_size) < 32:
                                needs_dl = False
                        except Exception:
                            pass
                    if not needs_dl:
                        continue

                    # Construct download URL – prefer explicit link, then draft API, then bucket
                    from zenodo_client import DEFAULT_TOKEN
                    link = entry.get('link', '')
                    if not link and entry.get('depo_id') and entry.get('filename'):
                        link = f"https://zenodo.org/api/records/{entry['depo_id']}/draft/files/{entry['filename']}/content?access_token={DEFAULT_TOKEN}"
                    if not link and entry.get('bucket_url') and entry.get('filename'):
                        link = f"{entry['bucket_url']}/{entry['filename']}"
                    if not link:
                        continue

                    try:
                        jr = req.get(link, timeout=120, stream=True)
                        jr.raise_for_status()
                        # Atomic write: temp file then rename
                        tmp_path = local_path.with_suffix('.tmp')
                        with open(tmp_path, 'wb') as f:
                            for chunk in jr.iter_content(chunk_size=65536):
                                f.write(chunk)
                        tmp_path.rename(local_path)
                        new_count += 1
                        action = 're-downloaded' if local_path.exists() else 'downloaded'
                        log.info('Peer sync: %s %s.json (%s bytes) from peer',
                                 action, code, local_path.stat().st_size)
                    except Exception as e:
                        # Clean up partial download
                        local_path.with_suffix('.tmp').unlink(missing_ok=True)
                        log.warning('Peer sync: failed to download %s from %s: %s', code, link, e)

                # Collect manifest entries to merge (skip tombstoned keys unless newer)
                for key, entry in peer_manifest.items():
                    if key in merged_manifest_entries:
                        continue
                    tombstone_ts = _MANIFEST_TOMBSTONES.get(key)
                    if tombstone_ts:
                        # Allow if peer entry was uploaded after tombstone was created
                        entry_ts = entry.get('uploaded_at', '')
                        if entry_ts <= tombstone_ts:
                            continue
                        # Newer entry — clear the tombstone
                        del _MANIFEST_TOMBSTONES[key]
                        _tombstone_path.write_text(json.dumps(_MANIFEST_TOMBSTONES, indent=2))
                        log.info('Peer sync: tombstone cleared for %s (peer entry %s > tombstone %s)',
                                 key, entry_ts, tombstone_ts)
                    if True:  # always collect (guard above skips stale)
                        merged_manifest_entries[key] = entry

            # Merge into local manifest (atomic read-modify-write)
            if merged_manifest_entries:
                try:
                    local_manifest = {}
                    if manifest_path.exists():
                        md = json.loads(manifest_path.read_text())
                        local_manifest = md.get('entries', md)

                    added = 0
                    for key, entry in merged_manifest_entries.items():
                        if key not in local_manifest:
                            local_manifest[key] = entry
                            added += 1

                    if added > 0:
                        # Atomic write: temp file then rename (same pattern as Manifest.save())
                        import tempfile as _tf
                        manifest_path.parent.mkdir(parents=True, exist_ok=True)
                        fd, tmp = _tf.mkstemp(dir=manifest_path.parent, suffix='.tmp', prefix='.manifest_')
                        try:
                            with os.fdopen(fd, 'w') as f:
                                json.dump({'entries': local_manifest}, f, indent=2, sort_keys=True)
                            os.replace(tmp, manifest_path)
                        except BaseException:
                            try: os.unlink(tmp)
                            except OSError: pass
                            raise
                        log.info('Peer sync: merged %d manifest entries from peers', added)
                except Exception as e:
                    log.warning('Peer sync: manifest merge failed: %s', e)

            if new_count > 0:
                log.info('Peer sync: %d new KG JSONs downloaded; the index watcher will pick them up incrementally', new_count)
                # No build()/update_kg() here -- the _init_search_index
                # watcher (single-worker, fcntl-locked) detects mtime
                # changes within ~60s and incrementally updates each
                # changed KG row. Calling build() on every peer sync
                # caused thrashing (14-30s rebuild every minute).
                # Bump last_kg_code in our local progress.json to the most
                # recently arrived JSON so the dashboard's "Last Completed
                # KG" card updates as cache-only peers complete work.
                try:
                    pf = Path('data/austria_processor/progress.json')
                    if pf.exists():
                        pdata = json.loads(pf.read_text())
                        latest = max(json_dir.glob('*.json'),
                                     key=lambda p: p.stat().st_mtime, default=None)
                        if latest is not None:
                            new_code = latest.stem
                            if pdata.get('last_kg_code') != new_code:
                                pdata['last_kg_code'] = new_code
                                tmp = pf.with_suffix('.tmp')
                                tmp.write_text(json.dumps(pdata, indent=2, default=str))
                                tmp.replace(pf)
                                log.info('Peer sync: bumped last_kg_code to %s', new_code)
                except Exception as e:
                    log.debug('Peer sync: last_kg_code bump failed: %s', e)

            # --- Sync Zenodo tile-cache manifest across peers ---
            # Read our local cache_manifest, push to each peer, pull theirs.
            # This ensures all peers share the same Zenodo cache deposit.
            try:
                cache_manifest_path = Path('data/austria_processor/cache_manifest.json')
                local_cm = {}
                if cache_manifest_path.exists():
                    local_cm = json.loads(cache_manifest_path.read_text())

                # Collect all peer manifests + merge incoming
                incoming_merged = {}
                for peer_url in peer_urls:
                    try:
                        r = req.get(peer_url.rstrip('/') + '/api/v1/processing/cache_manifest', timeout=15)
                        if r.status_code == 200:
                            peer_cm = r.json()
                            # Only merge file entries from peers — never adopt their depo_id.
                            # The primary's depo_id is authoritative; peers get it via PUT.
                            for zn, entry in peer_cm.get('files', {}).items():
                                existing = incoming_merged.get(zn)
                                if existing is None or entry.get('updated_at', '') > existing.get('updated_at', ''):
                                    incoming_merged[zn] = entry
                    except Exception as e:
                        log.debug('Peer sync: cache manifest fetch from %s failed: %s', peer_url, e)

                # Merge into local
                import re as _re
                local_files = local_cm.get('files', {})
                local_depo = local_cm.get('depo_id')
                cm_updated = 0
                for zn, entry in incoming_merged.items():
                    loc = local_files.get(zn)
                    if loc is None or entry.get('updated_at', '') > loc.get('updated_at', ''):
                        # Rewrite URL to point at our deposit
                        if local_depo and entry.get('url'):
                            entry = dict(entry)
                            entry['url'] = _re.sub(
                                r'/api/records/\d+/draft/files/',
                                f'/api/records/{local_depo}/draft/files/',
                                entry['url'])
                        local_files[zn] = entry
                        cm_updated += 1
                if cm_updated > 0:
                    local_cm['files'] = local_files
                    import tempfile as _tf2
                    cache_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    fd2, tmp2 = _tf2.mkstemp(dir=cache_manifest_path.parent, suffix='.tmp', prefix='.cache_manifest_')
                    try:
                        with os.fdopen(fd2, 'w') as f:
                            json.dump(local_cm, f, indent=2, sort_keys=True)
                        os.replace(tmp2, cache_manifest_path)
                    except BaseException:
                        try: os.unlink(tmp2)
                        except OSError: pass
                        raise
                    # Invalidate zip index cache
                    zip_idx_dir = Path('data/austria_processor/zenodo_zip_index')
                    if zip_idx_dir.exists():
                        for zf in zip_idx_dir.iterdir():
                            zf.unlink(missing_ok=True)
                    log.info('Peer sync: merged %d cache manifest entries from peers', cm_updated)

                # Push our (now merged) manifest to all peers
                if local_cm.get('files'):
                    for peer_url in peer_urls:
                        try:
                            req.put(
                                peer_url.rstrip('/') + '/api/v1/processing/cache_manifest',
                                json=local_cm, timeout=15
                            )
                        except Exception as e:
                            log.debug('Peer sync: cache manifest push to %s failed: %s', peer_url, e)

            except Exception as e:
                log.warning('Peer sync: cache manifest sync failed: %s', e)

        except Exception as e:
            log.warning('Peer sync error: %s', e)

        time.sleep(300)  # Every 5 minutes

threading.Thread(target=_sync_peer_data, daemon=True, name='peer-sync').start()


# === SECTION: Status push to director ===
#
# Every peer pushes its /processing/status payload to the director
# every PEER_PUSH_INTERVAL_S seconds. The director consults this cache
# instead of polling 50 peers on every tick. Cuts director outbound
# traffic by ~50x at fleet scale and isolates the director from peer
# slowdowns (a wedged peer no longer blocks the director loop).
#
# Skipped on the director itself — it reads its own progress.json
# directly. Skipped when director_url is null (single-instance mode).

PEER_PUSH_INTERVAL_S = 30
PEER_PUSH_TIMEOUT_S = 5


def _peer_status_push_loop():
    import requests as _req
    import director_ha as _dha
    import peer_director as _pd
    while True:
        try:
            time.sleep(PEER_PUSH_INTERVAL_S)
            # Skip on the director itself.
            try:
                if _dha.IS_DIRECTOR_FLAG.exists():
                    continue
            except Exception:
                pass
            # Resolve target director URL.
            try:
                self_info = _dha.load_self() or {}
                director_url = (self_info.get('director_url') or '').strip()
            except Exception:
                director_url = ''
            if not director_url:
                # Fallback: read zenodo_lock_url.txt (always points at
                # the current director on a healthy peer).
                try:
                    p = Path('data/austria_processor/zenodo_lock_url.txt')
                    if p.exists():
                        director_url = p.read_text().strip()
                except Exception:
                    pass
            if not director_url:
                continue
            # Build payload from local progress.json + bandwidth.
            try:
                pf = Path('data/austria_processor/progress.json')
                status = json.loads(pf.read_text()) if pf.exists() else {}
            except Exception:
                status = {}
            try:
                bw = _pd.get_local_bandwidth()
            except Exception:
                bw = None
            try:
                peer_id = (self_info or {}).get('id') or ''
            except Exception:
                peer_id = ''
            if not peer_id:
                continue
            try:
                tok = Path('data/admin_token').read_text().strip()
                hdrs = {'X-Admin-Token': tok} if tok else {}
            except Exception:
                hdrs = {}
            try:
                _req.post(
                    director_url.rstrip('/') + '/api/v1/director/peer_status',
                    json={'peer_id': peer_id, 'status': status,
                          'bandwidth': bw},
                    timeout=PEER_PUSH_TIMEOUT_S,
                    headers=hdrs,
                )
            except Exception as e:
                log.debug('peer status push failed: %s', e)
        except Exception as e:
            log.debug('peer status push loop: %s', e)
            time.sleep(PEER_PUSH_INTERVAL_S)


threading.Thread(target=_peer_status_push_loop, daemon=True,
                 name='peer-status-push').start()


# === SECTION: Role-data eviction (free disk on demoted peers) ===
#
# The per-KG JSON corpus and search_index.db are only needed by:
#   - the **primary** (canonical home for search/dashboard)
#   - the **current director** (peer_director consults the index for
#     cache-ready KGs, KG splits, etc.)
#   - the **current shadow** (must be ready to take over)
#
# A peer that *was* director (e.g. at17 after handback) accumulates the
# full JSON corpus + search_index.db (~5–10 GB on a fully-built fleet).
# When such a peer is demoted, that data becomes dead weight and triggers
# disk-pressure eviction of expensive Copernicus tile caches.
#
# Policy:
#   1. Every tick, classify our role (primary / director / shadow / other).
#   2. If "other", record the demotion timestamp in role_demoted_at.
#   3. After ROLE_EVICT_GRACE_SECONDS (1h) of continuous demotion, delete
#      data/austria_processor/json/*.json and data/search_index.db*.
#   4. Skip the JSON peer-sync download and the index rebuild on demoted
#      peers (so we don't immediately re-fetch what we just freed).
#   5. Promotion clears the timestamp; data starts replenishing via
#      _sync_peer_data on the next tick.
#
# The primary (id=='primary') is *always* kept-role, even when it is not
# currently the director. This guarantees the search index has a stable
# home that survives any failover.

ROLE_EVICT_GRACE_SECONDS = 3600        # 1h grace before purging
ROLE_EVICT_TICK_SECONDS = 600          # check every 10 min
_ROLE_DEMOTED_AT_FILE = Path('data/austria_processor/role_demoted_at')


def _is_keep_role_data() -> bool:
    """Return True iff this VM should retain the JSON corpus + index.

    Keep when:
      - we are the primary (canonical home, regardless of director state)
      - we are the current director (running peer_director loop)
      - we are the designated shadow (per shadow/meta.json)
    """
    try:
        import director_ha as _dha
        if _dha.IS_DIRECTOR_FLAG.exists():
            return True
        sid = _dha.self_id()
        if sid == 'primary':
            return True
        meta = {}
        try:
            if _dha.SHADOW_META_FILE.exists():
                meta = json.loads(_dha.SHADOW_META_FILE.read_text())
        except Exception:
            meta = {}
        designated = (meta.get('shadow_id')
                      or (meta.get('director_state') or {}).get('shadow_peer'))
        if designated and designated == sid:
            return True
    except Exception as e:
        log.debug('keep-role check failed (assuming keep): %s', e)
        return True   # fail-safe: never delete on uncertainty
    return False


def _role_data_eviction_tick() -> dict:
    """Single tick of the role-data eviction policy. Returns a status dict."""
    keep = _is_keep_role_data()
    now = time.time()
    json_dir = Path('data/austria_processor/json')
    idx_path = Path('data/search_index.db')
    status = {
        'keep_role': keep,
        'demoted_at': None,
        'grace_remaining_s': None,
        'purged': False,
        'json_count': 0,
        'index_size_mb': 0.0,
    }
    try:
        if json_dir.exists():
            status['json_count'] = sum(1 for _ in json_dir.glob('*.json'))
        if idx_path.exists():
            status['index_size_mb'] = round(idx_path.stat().st_size / (1024 ** 2), 1)
    except Exception:
        pass
    if keep:
        # Clear any stale demotion marker.
        try:
            if _ROLE_DEMOTED_AT_FILE.exists():
                _ROLE_DEMOTED_AT_FILE.unlink()
                log.info('Role-data eviction: cleared demotion marker (we are keep-role again)')
        except Exception:
            pass
        return status
    # Demoted. Stamp timestamp on first observation.
    try:
        if not _ROLE_DEMOTED_AT_FILE.exists():
            _ROLE_DEMOTED_AT_FILE.parent.mkdir(parents=True, exist_ok=True)
            _ROLE_DEMOTED_AT_FILE.write_text(str(int(now)))
            log.info('Role-data eviction: demoted; will purge JSON+index after %ds grace',
                     ROLE_EVICT_GRACE_SECONDS)
        demoted_at = int(_ROLE_DEMOTED_AT_FILE.read_text().strip() or now)
    except Exception:
        demoted_at = int(now)
    status['demoted_at'] = demoted_at
    status['grace_remaining_s'] = max(0, ROLE_EVICT_GRACE_SECONDS - int(now - demoted_at))
    if now - demoted_at < ROLE_EVICT_GRACE_SECONDS:
        return status
    # Grace expired — purge.
    n_purged = 0
    purged_bytes = 0
    try:
        if json_dir.exists():
            for f in list(json_dir.glob('*.json')):
                try:
                    purged_bytes += f.stat().st_size
                    f.unlink()
                    n_purged += 1
                except Exception:
                    pass
    except Exception as e:
        log.warning('Role-data eviction: JSON purge failed: %s', e)
    for sfx in ('', '-wal', '-shm', '-journal'):
        p = Path(f'{idx_path}{sfx}')
        try:
            if p.exists():
                purged_bytes += p.stat().st_size
                p.unlink()
        except Exception as e:
            log.warning('Role-data eviction: %s purge failed: %s', p, e)
    log.warning('Role-data eviction: purged %d JSONs + index (%.1f MB freed)',
                n_purged, purged_bytes / (1024 ** 2))
    status['purged'] = True
    status['purged_count'] = n_purged
    status['purged_mb'] = round(purged_bytes / (1024 ** 2), 1)
    return status


def _role_data_eviction_loop():
    time.sleep(60)   # let role/identity settle after boot
    while True:
        try:
            _role_data_eviction_tick()
        except Exception as e:
            log.warning('Role-data eviction tick failed: %s', e)
        time.sleep(ROLE_EVICT_TICK_SECONDS)


threading.Thread(target=_role_data_eviction_loop, daemon=True,
                 name='role-data-evict').start()


MAX_AREA_SQM = 25_000_000  # 25 km²

# === SECTION: Processing queue (semaphore for concurrent heavy tasks) ===
_TASK_SEMAPHORE = threading.Semaphore(2)   # max 2 concurrent heavy tasks
_TASK_QUEUE_LOCK = threading.Lock()
_TASK_QUEUE_SIZE = 0  # current waiting count
MAX_QUEUE_SIZE = 4    # reject if more than 4 waiting


def _height_class(h):
    """Classify height (m) into a forestry-relevant height class string."""
    if h is None or h < 0.5:
        return 'ground'
    if h < 2:
        return 'low (<2m)'
    if h < 5:
        return 'shrub (2-5m)'
    if h < 10:
        return 'young (5-10m)'
    if h < 15:
        return 'pole (10-15m)'
    if h < 20:
        return 'mid (15-20m)'
    if h < 25:
        return 'mature (20-25m)'
    if h < 30:
        return 'tall (25-30m)'
    if h < 35:
        return 'very tall (30-35m)'
    if h < 40:
        return 'old growth (35-40m)'
    return 'old growth (40m+)'

def _parse_height_filter(params: dict):
    """Parse height filter params. Returns a filter function or None.

    Supports:
      height_min=X          → height >= X  (shorthand: height_op=gt)
      height_max=X          → height <= X  (shorthand: height_op=lt)
      height_min=X&height_max=Y → X <= height <= Y  (shorthand: height_op=between)
      height_op=gt&height_min=X   → height > X  (alias for consistency)
      height_op=lt&height_max=X   → height < X
      height_op=between&height_min=X&height_max=Y
    """
    h_min = params.get('height_min')
    h_max = params.get('height_max')
    h_op = (params.get('height_op') or '').lower().strip()

    if h_min is not None:
        try:
            h_min = float(h_min)
        except (ValueError, TypeError):
            h_min = None
    if h_max is not None:
        try:
            h_max = float(h_max)
        except (ValueError, TypeError):
            h_max = None

    if h_min is None and h_max is None:
        return None

    # Infer operator from which params are present
    if not h_op:
        if h_min is not None and h_max is not None:
            h_op = 'between'
        elif h_min is not None:
            h_op = 'gt'
        elif h_max is not None:
            h_op = 'lt'

    if h_op == 'gt' and h_min is not None:
        return lambda h: (h or 0) >= h_min
    elif h_op == 'lt' and h_max is not None:
        return lambda h: (h or 0) <= h_max
    elif h_op == 'between' and h_min is not None and h_max is not None:
        return lambda h: h_min <= (h or 0) <= h_max
    return None


def _apply_height_filter_features(features: list, params: dict) -> list:
    """Apply height filter to a list of GeoJSON feature dicts."""
    hf = _parse_height_filter(params)
    if not hf:
        return features
    return [f for f in features
            if hf(f.get('properties', {}).get('height_max_m')
                  or f.get('properties', {}).get('height_after_m') or 0)]


def _apply_height_filter_objects(objects: list, params: dict) -> list:
    """Apply height filter to a list of segment objects (with .height_max)."""
    hf = _parse_height_filter(params)
    if not hf:
        return objects
    return [o for o in objects if hf(o.height_max)]


# === SECTION: Async task system (file-backed progress + result storage) ===
_PROGRESS_DIR = Path('/tmp/segment_progress')
_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
_RESULTS_DIR = Path('/tmp/segment_results')
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _progress_set(task_id: str, step: str, detail: str = ""):
    """Update progress for a running segment task."""
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    try:
        t0 = 0.0
        if p.exists():
            try:
                t0 = json.loads(p.read_text()).get('t0', 0.0)
            except Exception:
                pass
        p.write_text(json.dumps(dict(step=step, detail=detail, t0=t0,
                                     updated=time.time())))
    except Exception:
        pass

def _progress_start(task_id: str):
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    p.write_text(json.dumps(dict(step='starting', detail='',
                                 t0=time.time(), updated=time.time())))

def _progress_done(task_id: str, auto_share_id: str = None):
    """Mark task as completed (keeps file so polling sees 'done')."""
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    try:
        t0 = 0.0
        if p.exists():
            try:
                t0 = json.loads(p.read_text()).get('t0', 0.0)
            except Exception:
                pass
        d = dict(step='done', detail='', t0=t0, updated=time.time())
        if auto_share_id:
            d['auto_share_id'] = auto_share_id
        p.write_text(json.dumps(d))
    except Exception:
        pass

def _progress_error(task_id: str, error: str):
    """Mark task as failed."""
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    try:
        t0 = 0.0
        if p.exists():
            try:
                t0 = json.loads(p.read_text()).get('t0', 0.0)
            except Exception:
                pass
        p.write_text(json.dumps(dict(step='error', detail=error,
                                     t0=t0, updated=time.time())))
    except Exception:
        pass

def _progress_end(task_id: str):
    """Clean up progress file (legacy, used for non-async)."""
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

def _store_result(task_id: str, result: dict):
    """Store JSON result for async retrieval."""
    p = _RESULTS_DIR / f"{task_id}.json.gz"
    data = json.dumps(result).encode()
    with gzip.open(str(p), 'wb') as f:
        f.write(data)

def _get_result(task_id: str) -> dict | None:
    """Retrieve stored result, or None."""
    p = _RESULTS_DIR / f"{task_id}.json.gz"
    if not p.exists():
        return None
    with gzip.open(str(p), 'rb') as f:
        return json.loads(f.read())

def _cleanup_old_results(max_age_s: int = 14400):
    """Remove results older than max_age_s."""
    try:
        cutoff = time.time() - max_age_s
        for f in _RESULTS_DIR.glob('*.json.gz'):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
        for f in _RESULTS_DIR.glob('*.gpkg'):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
        for f in _PROGRESS_DIR.glob('*.json'):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
        for f in _QUERY_RESULTS_DIR.glob('*.json.gz'):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except Exception:
        pass

@app.route('/api/v1/segment/progress')
def segment_progress():
    """Poll progress of a running segment task.

    Returns {active, step, detail, elapsed, done, error}.
    When step=='done', the result is available at /api/v1/segment/result?task_id=...
    When step=='error', detail contains the error message.
    """
    task_id = request.args.get('task_id', '')
    p = _PROGRESS_DIR / f"{task_id}.json"
    if not task_id or not p.exists():
        # Check if result already exists (progress file cleaned up)
        if task_id and (_RESULTS_DIR / f"{task_id}.json.gz").exists():
            return jsonify(dict(active=False, step='done', detail='', elapsed=0, done=True, error=None))
        return jsonify(dict(active=False, step='', detail='', elapsed=0, done=False, error=None))
    try:
        info = json.loads(p.read_text())
        step = info.get('step', '')
        elapsed = round(time.time() - info.get('t0', time.time()), 1)
        is_done = step == 'done'
        is_error = step == 'error'
        return jsonify(dict(
            active=not is_done and not is_error,
            step=step,
            detail=info.get('detail', ''),
            elapsed=elapsed,
            done=is_done,
            error=info.get('detail', '') if is_error else None,
            auto_share_id=info.get('auto_share_id'),
        ))
    except Exception:
        return jsonify(dict(active=False, step='', detail='', elapsed=0, done=False, error=None))


@app.route('/api/v1/segment/abort', methods=['POST'])
def segment_abort():
    """Abort a running segment task. Marks it as cancelled."""
    task_id = request.args.get('task_id', '') or (request.get_json(silent=True) or {}).get('task_id', '')
    if not task_id:
        return _error('task_id required')
    p = _PROGRESS_DIR / f"{task_id}.json"
    if p.exists():
        try:
            info = json.loads(p.read_text())
            if info.get('step') not in ('done', 'error'):
                _progress_error(task_id, 'Cancelled by user')
        except Exception:
            _progress_error(task_id, 'Cancelled by user')
    return jsonify({"ok": True, "task_id": task_id})


@app.route('/api/v1/segment/result')
def segment_result():
    """Retrieve the result of an async segment task."""
    task_id = request.args.get('task_id', '')
    if not task_id:
        return _error('task_id required')
    result = _get_result(task_id)
    if result is None:
        return _error('Result not found or not ready', 404)
    # Don't delete after retrieval — let _cleanup_old_results() handle it
    return jsonify(result)


@app.route('/api/v1/training/status')
def training_status():
    """Return RF training job status: running, current KG, model info, resource usage.

    Uses cgroup/proc filesystem reads instead of subprocess to avoid
    fork failures when the srv cgroup is under memory pressure.
    """
    import re, pathlib

    result = dict(running=False, current_kg=None, progress=None,
                  model=None, pid=None, ram_mb=None,
                  service_state=None)

    # Check rf_train service state via cgroup filesystem (no subprocess needed)
    cgroup_base = pathlib.Path('/sys/fs/cgroup/system.slice/rf_train.service')
    try:
        events_text = (cgroup_base / 'cgroup.events').read_text()
        populated = 'populated 1' in events_text
        frozen = 'frozen 1' in events_text
        if populated and not frozen:
            result['running'] = True
            result['service_state'] = 'active'
        elif populated and frozen:
            result['running'] = True
            result['service_state'] = 'activating'  # possibly restarting
        else:
            result['service_state'] = 'inactive'
    except FileNotFoundError:
        result['service_state'] = 'not-found'
    except Exception:
        pass

    # Get memory usage and PID from cgroup/proc (no subprocess needed)
    try:
        mem_bytes = int((cgroup_base / 'memory.current').read_text().strip())
        result['ram_mb'] = round(mem_bytes / (1024 * 1024))
    except Exception:
        pass
    try:
        pids = (cgroup_base / 'cgroup.procs').read_text().strip().splitlines()
        for pid_str in pids:
            pid = int(pid_str)
            # Read null-separated args; first arg is the executable
            args = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode('utf-8', errors='replace').split('\x00')
            if len(args) >= 2 and 'python' in args[0] and 'train_rf_4000kg' in args[1]:
                result['pid'] = pid
                break
    except Exception:
        pass

    # Check oom_kill events for this cgroup
    try:
        mem_events = (cgroup_base / 'memory.events').read_text()
        for line in mem_events.splitlines():
            if line.startswith('oom_kill '):
                oom_kills = int(line.split()[1])
                if oom_kills > 0:
                    result['oom_kills'] = oom_kills
    except Exception:
        pass

    # Parse last log lines for current KG, progress, and last activity
    log_path = pathlib.Path('/tmp/rf_train_4000kg.log')
    if log_path.exists():
        try:
            # Read last 32KB of log (enough to find current KG line)
            with open(log_path, 'rb') as f:
                f.seek(max(0, f.seek(0, 2) - 32768))
                tail = f.read().decode('utf-8', errors='replace')
            lines = tail.strip().splitlines()
            # Find last "Processing KG" line
            for line in reversed(lines):
                m = re.search(r'\[(\d+)/(\d+)\]\s+Processing KG (\d+)\s+\(([^)]+)\)', line)
                if m:
                    result['current_kg'] = dict(
                        index=int(m.group(1)), total=int(m.group(2)),
                        kg_code=m.group(3), kg_name=m.group(4))
                    result['progress'] = f"{m.group(1)}/{m.group(2)}"
                    break
            # Last log line with timestamp → current step
            for line in reversed(lines):
                tm = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if tm:
                    result['last_log_time'] = tm.group(1)
                    # Extract step info from the line
                    step = re.search(r'(Step \d+\S*:.*?)$', line)
                    if step:
                        result['current_step'] = step.group(1).strip()[:80]
                    else:
                        # Just the message part after the log prefix
                        msg = re.sub(r'^.*?(INFO|WARNING|ERROR)\s+\S+:\s*', '', line)
                        if msg:
                            result['current_step'] = msg.strip()[:80]
                    break
        except Exception:
            pass

    # Checkpoint count + failed KGs + next model checkpoint
    MODEL_CP_INTERVAL = 10
    ckpt_dir = pathlib.Path('/home/exedev/srtm-lidar/rf_training_data/checkpoints')
    n_success = 0
    if ckpt_dir.exists():
        n_success = len(list(ckpt_dir.glob('kg_*.npz')))
        result['n_checkpoints'] = n_success
    failed_file = pathlib.Path('/home/exedev/srtm-lidar/rf_training_data/failed_kgs.txt')
    n_fail = 0
    if failed_file.exists():
        failed = [l.strip() for l in failed_file.read_text().strip().splitlines() if l.strip()]
        n_fail = len(failed)
        result['n_failed_kgs'] = n_fail
        result['failed_kgs'] = failed
    result['n_completed'] = n_success + n_fail
    next_cp = ((n_success // MODEL_CP_INTERVAL) + 1) * MODEL_CP_INTERVAL
    result['next_model_checkpoint'] = next_cp
    result['to_next_model'] = max(0, next_cp - n_success)

    # Credits paused?
    pause_file = pathlib.Path('/home/exedev/srtm-lidar/rf_training_data/credits_paused.txt')
    if pause_file.exists():
        result['credits_paused'] = True
        result['credits_paused_since'] = pause_file.read_text().strip().split('\n')[0]
    else:
        result['credits_paused'] = False

    # Model info (prefer best_model from curve eval)
    from learned_classifier import BEST_META_PATH
    active_meta_path = BEST_META_PATH if BEST_META_PATH.exists() else pathlib.Path('/tmp/learned_classifier/rf_meta.json')
    if active_meta_path.exists():
        try:
            meta = json.loads(active_meta_path.read_text())
            result['model'] = dict(
                oob_score=round(meta.get('oob_score', 0), 4),
                composite_score=round(meta.get('composite_score', 0), 4),
                n_train=meta.get('n_train', 0),
                n_kgs=meta.get('n_kgs', 0),
                best_seed=meta.get('best_seed'),
                n_classes=len(meta.get('classes', [])),
                trained_at=meta.get('trained_at', ''),
                source='best_model' if active_meta_path == BEST_META_PATH else 'live')
        except Exception:
            pass

    # OOB learning curve (from checkpoint evaluation, multi-seed)
    curve_path = pathlib.Path('data/oob_curve.csv')
    if curve_path.exists():
        try:
            import csv as _csv
            from collections import defaultdict
            import statistics
            by_kgs = defaultdict(list)
            with open(curve_path) as _f:
                reader = _csv.DictReader(_f)
                for r in reader:
                    n_kgs = int(r['n_kgs'])
                    by_kgs[n_kgs].append({
                        'oob': float(r['oob']),
                        'composite': float(r['composite']) if 'composite' in r and r['composite'] else float(r['oob']),
                        'mean_class_oob': float(r['mean_class_oob']) if r.get('mean_class_oob') else None,
                        'min_class_oob': float(r['min_class_oob']) if r.get('min_class_oob') else None,
                        'n_samples': int(r['n_samples']),
                        'n_classes': int(r['n_classes']),
                    })
            curve = []
            for n_kgs in sorted(by_kgs):
                runs = by_kgs[n_kgs]
                oobs = [r['oob'] for r in runs]
                comps = [r['composite'] for r in runs]
                mean_cls = [r['mean_class_oob'] for r in runs if r['mean_class_oob'] is not None]
                min_cls = [r['min_class_oob'] for r in runs if r['min_class_oob'] is not None]
                entry = {
                    'n_kgs': n_kgs,
                    'n_samples': runs[0]['n_samples'],
                    'n_classes': runs[0]['n_classes'],
                    'n_seeds': len(oobs),
                    'oob_median': round(statistics.median(oobs), 6),
                    'oob_min': round(min(oobs), 6),
                    'oob_max': round(max(oobs), 6),
                    'oob_all': sorted(round(o, 6) for o in oobs),
                    'composite_median': round(statistics.median(comps), 6),
                    'composite_min': round(min(comps), 6),
                    'composite_max': round(max(comps), 6),
                    'composite_all': sorted(round(c, 6) for c in comps),
                }
                if mean_cls:
                    entry['mean_class_median'] = round(statistics.median(mean_cls), 6)
                if min_cls:
                    entry['min_class_median'] = round(statistics.median(min_cls), 6)
                curve.append(entry)
            result['oob_curve'] = curve
        except Exception:
            pass

    # Live history (monitor cron)
    hist_path = pathlib.Path('data/oob_history.csv')
    if hist_path.exists():
        try:
            import csv as _csv
            with open(hist_path) as _f:
                reader = _csv.DictReader(_f)
                result['oob_history'] = [dict(r) for r in reader]
        except Exception:
            pass

    # Monitor state (convergence tracking)
    state_path = pathlib.Path('data/monitor_state.json')
    if state_path.exists():
        try:
            result['monitor'] = json.loads(state_path.read_text())
        except Exception:
            pass

    # Curve detail (per-checkpoint expanded info from JSONL)
    detail_path = pathlib.Path('data/oob_curve_detail.jsonl')
    if detail_path.exists():
        try:
            import statistics
            from collections import defaultdict
            # Parse JSONL: group by n_kgs
            by_kgs = defaultdict(list)
            with open(detail_path) as _f:
                for line in _f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        by_kgs[int(obj['n_kgs'])].append(obj)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            curve_detail = {}
            for n_kgs in sorted(by_kgs):
                entries = by_kgs[n_kgs]
                n_seeds = len(entries)
                # Hyperparams (same for all seeds, take first)
                entry0 = entries[0]
                # Median feature importances across seeds
                all_feat_keys = set()
                for e in entries:
                    all_feat_keys.update(e.get('all_importances', {}).keys())
                feat_medians = {}
                for fk in all_feat_keys:
                    vals = [e['all_importances'][fk] for e in entries
                            if fk in e.get('all_importances', {})]
                    if vals:
                        feat_medians[fk] = round(statistics.median(vals), 6)
                # Median per-class OOB across seeds (only where class present)
                all_classes = set()
                for e in entries:
                    all_classes.update(e.get('per_class_oob', {}).keys())
                class_medians = {}
                classes_present = sorted(all_classes)
                for cls in all_classes:
                    vals = [e['per_class_oob'][cls] for e in entries
                            if cls in e.get('per_class_oob', {})]
                    if vals:
                        class_medians[cls] = round(statistics.median(vals), 4)
                # Composite stats across seeds
                comp_vals = [e.get('composite', e.get('oob', 0)) for e in entries]
                mean_cls_vals = [e.get('mean_class_oob') for e in entries if e.get('mean_class_oob') is not None]
                min_cls_vals = [e.get('min_class_oob') for e in entries if e.get('min_class_oob') is not None]
                curve_detail[str(n_kgs)] = {
                    'n_estimators': entry0.get('n_estimators', 200),
                    'max_depth': entry0.get('max_depth', 20),
                    'min_samples_leaf': entry0.get('min_samples_leaf', 5),
                    'feature_importances': feat_medians,
                    'per_class_oob': class_medians,
                    'classes_present': classes_present,
                    'n_seeds': n_seeds,
                    'composite': round(statistics.median(comp_vals), 6) if comp_vals else None,
                    'mean_class_oob': round(statistics.median(mean_cls_vals), 6) if mean_cls_vals else None,
                    'min_class_oob': round(statistics.median(min_cls_vals), 6) if min_cls_vals else None,
                }
            result['curve_detail'] = curve_detail

            # Expose the exact deployed model's detail (specific seed)
            if result.get('model') and result['model'].get('best_seed') is not None:
                dep_kgs = result['model']['n_kgs']
                dep_seed = result['model']['best_seed']
                for e in by_kgs.get(dep_kgs, []):
                    if e.get('seed') == dep_seed:
                        result['deployed_detail'] = {
                            'n_kgs': dep_kgs,
                            'seed': dep_seed,
                            'n_samples': e.get('n_samples'),
                            'n_classes': e.get('n_classes'),
                            'oob': e.get('oob'),
                            'composite': e.get('composite'),
                            'mean_class_oob': e.get('mean_class_oob'),
                            'min_class_oob': e.get('min_class_oob'),
                            'n_estimators': e.get('n_estimators', 200),
                            'max_depth': e.get('max_depth', 20),
                            'min_samples_leaf': e.get('min_samples_leaf', 5),
                            'per_class_oob': e.get('per_class_oob', {}),
                            'feature_importances': e.get('all_importances', {}),
                            'classes_present': sorted(e.get('per_class_oob', {}).keys()),
                        }
                        break
        except Exception:
            pass

    return jsonify(result)


def _is_curve_eval_running():
    """Check lockfile to see if any curve eval (cron or triggered) is running."""
    import fcntl
    from pathlib import Path
    lockfile = Path('/tmp/rf_curve_eval.lock')
    if not lockfile.exists():
        return False
    try:
        fd = os.open(str(lockfile), os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except (OSError, IOError):
            return True
        finally:
            os.close(fd)
    except FileNotFoundError:
        return False


@app.route('/api/v1/training/evaluate', methods=['POST'])
def trigger_curve_eval():
    """Trigger OOB curve re-evaluation in background."""
    if _is_curve_eval_running():
        return jsonify(running=True, msg='Already running (cron or previous trigger)'), 409
    import subprocess
    # Detach from gunicorn process group so it survives service restarts
    subprocess.Popen(
        ['python3', 'evaluate_checkpoints.py', 'curve', '--step', '5'],
        cwd='/home/exedev/srtm-lidar',
        start_new_session=True,
        stdout=open('/tmp/rf_curve_eval.log', 'a'),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    return jsonify(running=True, msg='Curve evaluation started')


@app.route('/api/v1/training/evaluate', methods=['GET'])
def curve_eval_status():
    """Check if curve evaluation is running, with progress detail."""
    running = _is_curve_eval_running()
    info = {'running': running}
    if running:
        # Read CSV to figure out progress (adaptive seed count)
        try:
            import csv
            from pathlib import Path
            from evaluate_checkpoints import seeds_for_n_kgs
            csv_path = Path('data/oob_curve.csv')
            n_ckpt = len(list(Path('rf_training_data/checkpoints').glob('kg_*.npz')))
            steps = list(range(5, n_ckpt + 1, 5))
            all_combos = [(n, s) for n in steps for s in seeds_for_n_kgs(n)]
            existing = set()
            if csv_path.exists():
                with open(csv_path) as f:
                    for r in csv.DictReader(f):
                        existing.add((int(r['n_kgs']), int(r.get('seed', 0))))
            work = [(n, s) for n, s in all_combos if (n, s) not in existing]
            done_combos = len(all_combos) - len(work)
            if work:
                info['current_kgs'] = work[0][0]
                info['current_seed'] = work[0][1]
            info['done'] = done_combos
            info['total'] = len(all_combos)
        except Exception:
            pass
    return jsonify(info)


@app.route('/api/v1/training/evaluate/stop', methods=['POST'])
def stop_curve_eval():
    """Kill running curve evaluation subprocess."""
    import signal
    if not _is_curve_eval_running():
        return jsonify(running=False, msg='Not running')
    # Find the evaluate_checkpoints process and kill its process group
    try:
        import subprocess
        result = subprocess.run(
            ['pgrep', '-f', 'evaluate_checkpoints.py curve'],
            capture_output=True, text=True
        )
        for pid in result.stdout.strip().split('\n'):
            if pid:
                os.kill(int(pid), signal.SIGTERM)
        return jsonify(running=False, msg='Stopped')
    except Exception as e:
        return jsonify(running=_is_curve_eval_running(), msg=str(e)), 500


# === SECTION: Austria Processor endpoints (proxy to processor state) ===

_processor_process = None  # subprocess.Popen for the processor

def _is_processor_running() -> bool:
    """Check if Austria processor is actively running (systemd or subprocess)."""
    try:
        p = Path('data/austria_processor/progress.json')
        if p.exists():
            d = json.loads(p.read_text())
            if d.get('state') == 'running':
                return True
    except Exception:
        pass
    return False

# Git commit hash (read once at startup)
try:
    import subprocess as _sp
    _GIT_COMMIT = _sp.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        cwd=str(Path(__file__).parent), stderr=_sp.DEVNULL
    ).decode().strip()
except Exception:
    _GIT_COMMIT = 'unknown'

# Region lookup (once at startup via ipinfo.io)
def _detect_region() -> str:
    """Detect exe.dev region by geolocating the first public NAT gateway hop."""
    _TZ_TO_REGION = {
        'America/Los_Angeles': 'PDX',
        'America/Vancouver':   'PDX',
        'America/New_York':    'NYC',
        'America/Toronto':     'NYC',
        'Europe/Berlin':       'FRA',
        'Europe/Frankfurt':    'FRA',
        'Europe/London':       'LON',
        'Asia/Tokyo':          'TYO',
        'Australia/Sydney':    'SYD',
        'Asia/Singapore':      'SGP',
    }
    try:
        import subprocess as _sp, urllib.request as _ur
        # Geolocate the first public (non-RFC1918) NAT hop instead of the VM's
        # own outbound IP, which may be NAT'd through a US gateway regardless of
        # actual datacenter location.
        public_ip = None
        import re as _re, ipaddress as _ipa
        # Try traceroute to find first globally-routable NAT hop (more accurate than own IP)
        try:
            tr = _sp.run(
                ['traceroute', '-m', '5', '-q', '1', '-w', '1', '8.8.8.8'],
                capture_output=True, text=True, timeout=12)
            for line in tr.stdout.splitlines():
                if not _re.match(r'^\s*\d+\s', line):  # skip header line
                    continue
                for part in line.split():
                    segs = part.split('.')
                    if len(segs) != 4 or not all(s.isdigit() for s in segs):
                        continue
                    try:
                        if _ipa.ip_address(part).is_global:
                            public_ip = part
                            break
                    except ValueError:
                        continue
                if public_ip:
                    break
        except Exception:
            pass  # fall back to own IP below
        url = f'https://ipinfo.io/{public_ip}/json' if public_ip else 'https://ipinfo.io/json'
        d = json.loads(_ur.urlopen(url, timeout=5).read())
        tz = d.get('timezone', '')
        if tz in _TZ_TO_REGION:
            return _TZ_TO_REGION[tz]
        city = (d.get('city') or '').lower()
        if 'frankfurt' in city or 'berlin' in city: return 'FRA'
        if 'london' in city: return 'LON'
        if 'tokyo' in city: return 'TYO'
        if 'sydney' in city: return 'SYD'
        if 'singapore' in city: return 'SGP'
        if 'new york' in city: return 'NYC'
        return tz or '???'
    except Exception:
        return '???'
try:
    _REGION = _detect_region()
except Exception:
    _REGION = '???'

@app.route('/api/v1/processing/status')
def processing_status():
    """Return Austria processor progress (read from progress.json)."""
    progress_file = Path('data/austria_processor/progress.json')
    if not progress_file.exists():
        return jsonify({
            'state': 'idle', 'total_kgs': 0, 'completed': 0,
            'success': 0, 'failed': 0, 'uploaded': 0,
            'upload_size_bytes': 0, 'current_kg': None,
            'rate_kgs_per_hour': 0, 'avg_seconds_per_kg': 0,
            'elapsed_seconds': 0, 'eta_seconds': 0,
            'recent_log': [], 'failed_kgs': [],
            'parcels_total': 0, 'buildings_total': 0, 'started_at': None,
            'git_commit': _GIT_COMMIT,
            'region': _REGION,
            'instance': os.environ.get('INSTANCE_ID', 'primary'),
        })
    try:
        data = json.loads(progress_file.read_text())
        # Check if processor is actually running
        global _processor_process
        if _processor_process is not None and _processor_process.poll() is not None:
            data['state'] = 'stopped'
            _processor_process = None
        # Always add fresh system metrics (even if processor writes them too)
        if 'system' not in data:
            data['system'] = {}
        try:
            import pathlib as _pl
            # Live RAM
            mi = open('/proc/meminfo').read()
            mt = ma = 0
            for line in mi.splitlines():
                if line.startswith('MemTotal:'):
                    mt = int(line.split()[1]) // 1024
                elif line.startswith('MemAvailable:'):
                    ma = int(line.split()[1]) // 1024
            if mt:
                data['system']['ram_total_mb'] = mt
                data['system']['ram_used_mb'] = mt - ma
                data['system']['ram_pct'] = round(100 * (mt - ma) / mt, 1)
            # Disk
            st = os.statvfs('/')
            fg = (st.f_bavail * st.f_frsize) / (1024**3)
            tg = (st.f_blocks * st.f_frsize) / (1024**3)
            data['system']['disk_free_gb'] = round(fg, 1)
            data['system']['disk_used_pct'] = round(100*(1 - fg/tg), 1)
            # CPU
            data['system']['cpu_pct'] = round(100 * os.getloadavg()[0] / max(os.cpu_count() or 1, 1), 1)
            # Processor PID check — try managed process first, then find via systemd/pidfile
            _proc_pid = None
            if _processor_process and _processor_process.poll() is None:
                _proc_pid = _processor_process.pid
            else:
                # Find processor started via systemd
                try:
                    import subprocess as _sp
                    _pids = _sp.check_output(
                        ['pgrep', '-f', 'austria_processor.py'],
                        text=True, timeout=2
                    ).strip().split('\n')
                    if _pids and _pids[0]:
                        _proc_pid = int(_pids[0])
                except Exception:
                    pass
            if _proc_pid:
                data['system']['proc_pid'] = _proc_pid
                try:
                    rss_kb = int(open(f'/proc/{_proc_pid}/status').read().split('VmRSS:')[1].split()[0])
                    data['system']['proc_ram_mb'] = rss_kb // 1024
                except Exception:
                    pass
            elif data.get('state') in ('running', 'processing'):
                # No processor PID found but progress.json says running → stale
                data['state'] = 'stopped'
            # Tile caches
            try:
                from tile_cache import cache_summary
                data['system']['tile_caches'] = cache_summary()
            except Exception:
                pass
            # Zenodo persistent cache
            try:
                from zenodo_cache import ZenodoCache
                data['system']['zenodo_cache'] = ZenodoCache().status()
            except Exception:
                pass
            # Zenodo / Copernicus pause flags
            for _flag, _key in (('zenodo_paused', 'zenodo_pause'),
                                ('copernicus_paused', 'copernicus_pause')):
                _fp = _pl.Path(f'data/austria_processor/{_flag}')
                if _fp.exists():
                    try:
                        data[_key] = json.loads(_fp.read_text())
                    except Exception:
                        try:
                            data[_key] = {'raw': _fp.read_text()[:500]}
                        except Exception:
                            data[_key] = {'present': True}
            # Manifest summary
            mf = _pl.Path('data/austria_processor/zenodo_manifest.json')
            if mf.exists():
                md = json.loads(mf.read_text())
                ents = md.get('entries', {})
                data['manifest'] = {
                    'count': len(ents),
                    'total_size_bytes': sum(e.get('size', 0) for e in ents.values()),
                }
                # Completion rate / ETA based on actual Zenodo upload timestamps.
                # One KG "completed" = max(uploaded_at) across its entries
                # (full_gpkg / light_gpkg / json) — i.e. when the last
                # product landed on Zenodo.
                try:
                    from datetime import datetime as _dt
                    kg_done_at = {}
                    for key, e in ents.items():
                        ts = e.get('uploaded_at')
                        if not ts:
                            continue
                        kg = key.split('_', 1)[0]
                        try:
                            t = _dt.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                        except Exception:
                            continue
                        if kg not in kg_done_at or t > kg_done_at[kg]:
                            kg_done_at[kg] = t
                    if kg_done_at:
                        now_ts = time.time()
                        times = sorted(kg_done_at.values())
                        # Prefer last 24h; fall back to the last 50 KGs if
                        # the window is sparse (e.g. peers idle overnight).
                        WIN = 24 * 3600
                        recent = [t for t in times if now_ts - t <= WIN]
                        if len(recent) >= 5:
                            window_s = max(now_ts - recent[0], 1.0)
                            n_recent = len(recent)
                        elif len(times) >= 5:
                            tail = times[-50:]
                            window_s = max(now_ts - tail[0], 1.0)
                            n_recent = len(tail)
                        else:
                            window_s = max(now_ts - times[0], 1.0)
                            n_recent = len(times)
                        rate_per_h = n_recent / (window_s / 3600.0)
                        avg_s = window_s / n_recent
                        data['manifest_rate_kgs_per_hour'] = round(rate_per_h, 2)
                        data['manifest_avg_seconds_per_kg'] = round(avg_s, 1)
                        data['manifest_completion_count'] = len(kg_done_at)
                        data['manifest_last_completion_ts'] = max(times)
                        data['manifest_window_kgs'] = n_recent
                        data['manifest_window_seconds'] = int(window_s)
                except Exception as _ex:
                    log.debug('manifest rate calc failed: %s', _ex)
        except Exception:
            pass
        data['git_commit'] = _GIT_COMMIT
        # Surface assignment env so the director/dashboard see what this
        # processor is dedicated to.
        try:
            _ci = os.environ.get('COPERNICUS_CRED_INDICES', '').strip()
            if not _ci:
                # Read from running processor's environ if possible.
                if data.get('system', {}).get('proc_pid'):
                    try:
                        with open(f"/proc/{data['system']['proc_pid']}/environ", 'rb') as _f:
                            envs = _f.read().split(b'\0')
                        for kv in envs:
                            if kv.startswith(b'COPERNICUS_CRED_INDICES='):
                                _ci = kv.split(b'=', 1)[1].decode('utf-8', 'ignore')
                            elif kv.startswith(b'KG_LAT_STRIP_FILTER='):
                                try:
                                    data['lat_strip_filter'] = json.loads(
                                        kv.split(b'=', 1)[1].decode('utf-8', 'ignore'))
                                except Exception:
                                    pass
                            elif kv.startswith(b'KG_CELL_FILTER='):
                                try:
                                    data['cell_filter'] = json.loads(
                                        kv.split(b'=', 1)[1].decode('utf-8', 'ignore'))
                                except Exception:
                                    pass
                    except Exception:
                        pass
            if _ci:
                data['cred_indices'] = [int(x) for x in _ci.split(',') if x.strip()]
        except Exception:
            pass
        # DB-sourced processed count (authoritative: counts all peers via Zenodo manifest sync)
        try:
            _row = si.get_index()._conn().execute(
                'SELECT COUNT(*), COALESCE(SUM(parcel_count),0), COALESCE(SUM(total_area_sqm),0)/1e6, '
                'COALESCE(SUM(building_count),0), COALESCE(SUM(new_building_count),0), '
                'COALESCE(SUM(tree_count),0), COALESCE(SUM(infrastructure_count),0) '
                'FROM kg WHERE processed=1'
            ).fetchone()
            data['db_processed'] = _row[0]
            data['db_parcels_total'] = int(_row[1] or 0)
            data['db_area_km2'] = round(float(_row[2] or 0), 2)
            data['db_buildings_total'] = int(_row[3] or 0)
            data['db_new_buildings_total'] = int(_row[4] or 0)
            data['db_trees_total'] = int(_row[5] or 0)
            data['db_infrastructure_total'] = int(_row[6] or 0)
            # Total Austria area for area-based progress (denominator).
            _row2 = si.get_index()._conn().execute(
                'SELECT COALESCE(SUM(total_area_sqm),0)/1e6 FROM kg'
            ).fetchone()
            data['db_area_km2_total'] = round(float(_row2[0] or 0), 2)
        except Exception:
            pass
        # Sparkline series: KGs completed per day (last 30 days, by Zenodo upload).
        try:
            mf2 = _pl.Path('data/austria_processor/zenodo_manifest.json')
            if mf2.exists():
                from datetime import datetime as _dt2, timedelta as _td2, timezone as _tz2
                md2 = json.loads(mf2.read_text())
                ents2 = md2.get('entries', {})
                kg_done_at2 = {}
                for key, e in ents2.items():
                    ts = e.get('uploaded_at')
                    if not ts:
                        continue
                    kg = key.split('_', 1)[0]
                    try:
                        t = _dt2.fromisoformat(ts.replace('Z', '+00:00'))
                    except Exception:
                        continue
                    if kg not in kg_done_at2 or t > kg_done_at2[kg]:
                        kg_done_at2[kg] = t
                if kg_done_at2:
                    now_dt = _dt2.now(_tz2.utc)
                    today = now_dt.date()
                    DAYS = 30
                    buckets = [0] * DAYS
                    for t in kg_done_at2.values():
                        d_ago = (today - t.date()).days
                        if 0 <= d_ago < DAYS:
                            buckets[DAYS - 1 - d_ago] += 1
                    data['manifest_daily_completions'] = buckets
        except Exception:
            pass
        # Include persisted tile history for all completed/failed KGs
        try:
            th_path = Path('data/austria_processor/tile_history.json')
            if th_path.exists():
                data['tile_history'] = json.loads(th_path.read_text())
        except Exception:
            pass
        data['throttle'] = Path('data/austria_processor/upload_throttle').exists()
        data['region'] = _REGION
        data['instance'] = os.environ.get('INSTANCE_ID', 'primary')
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/processing/throttle', methods=['GET', 'POST'])
def processing_throttle():
    """Toggle bandwidth throttle mode (skip GPKG uploads to Zenodo)."""
    throttle_file = Path('data/austria_processor/upload_throttle')
    if request.method == 'GET':
        return jsonify({'throttle': throttle_file.exists()})
    # POST toggles
    if throttle_file.exists():
        throttle_file.unlink()
        return jsonify({'throttle': False, 'message': 'Throttle disabled \u2014 full uploads resume'})
    else:
        throttle_file.write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ'))
        return jsonify({'throttle': True, 'message': 'Throttle enabled \u2014 skipping GPKG uploads'})


@app.route('/api/v1/processing/peers', methods=['GET'])
def processing_peers_status():
    """Return compact state for peer coordination.

    Returns completed KG codes, current KG, priority queue, and failed KGs
    so a peer instance can avoid duplicate work.
    """
    data_dir = Path('data/austria_processor')
    result = {'instance': os.environ.get('INSTANCE_ID', 'primary')}

    # Completed KGs
    result['completed'] = sorted(_get_completed_kgs())

    # Current KG being processed
    current = None
    pf = data_dir / 'progress.json'
    if pf.exists():
        try:
            pd = json.loads(pf.read_text())
            ckg = pd.get('current_kg') or {}
            current = ckg.get('code')
        except Exception:
            pass
    result['current'] = current
    # Parent code for split-KG awareness. Older peers (pre-kg_splitter)
    # don't know that ``60336-northwest`` is a sub-block of ``60336``;
    # exposing the parent code lets them de-dupe correctly.
    try:
        from kg_splitter import is_block_code, parent_kg_code
        if current and is_block_code(current):
            result['current_parent'] = parent_kg_code(current)
    except Exception:
        pass

    # In-progress marker (crash recovery file)
    ipf = data_dir / 'in_progress_kg.txt'
    if ipf.exists():
        try:
            ip = ipf.read_text().strip()
            if ip and ip != current:
                result['in_progress'] = ip
        except Exception:
            pass

    # Priority queue
    retry_path = data_dir / 'retry_queue.json'
    if retry_path.exists():
        try:
            result['priority'] = json.loads(retry_path.read_text())
        except Exception:
            result['priority'] = []
    else:
        result['priority'] = []

    # Failed KGs (permanent)
    failed_path = data_dir / 'failed_kgs.json'
    if failed_path.exists():
        try:
            result['failed'] = json.loads(failed_path.read_text())
        except Exception:
            result['failed'] = []
    else:
        result['failed'] = []

    # Manifest entries (Zenodo URLs) — so peers can discover our uploads
    manifest_path = data_dir / 'zenodo_manifest.json'
    if manifest_path.exists():
        try:
            md = json.loads(manifest_path.read_text())
            result['manifest'] = md.get('entries', md)
        except Exception:
            result['manifest'] = {}
    else:
        result['manifest'] = {}

    # Tombstones: KG keys that have been force-requeued.  Peers must
    # honor these when computing completed_codes so they don't skip a
    # re-queued KG just because they have an old local JSON.
    tomb_path = data_dir / 'manifest_tombstones.json'
    if tomb_path.exists():
        try:
            result['tombstones'] = json.loads(tomb_path.read_text())
        except Exception:
            result['tombstones'] = {}
    else:
        result['tombstones'] = {}

    return jsonify(result)


@app.route('/api/v1/processing/kg_strikes', methods=['GET', 'PUT'])
def processing_kg_strikes():
    """GET/PUT the adaptive-split strike counter (kg_strikes.json).

    GET: returns {kg_code: strike_count}.
    PUT: merges (max(local, incoming) per KG) so we never undo a peer's
         observed strikes.
    """
    p = Path('data/austria_processor/kg_strikes.json')
    if request.method == 'GET':
        if not p.exists():
            return jsonify({})
        try:
            return jsonify(json.loads(p.read_text()))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    incoming = request.get_json(force=True) or {}
    if not isinstance(incoming, dict):
        return jsonify({'error': 'expected object'}), 400
    try:
        local = {}
        if p.exists():
            try:
                local = json.loads(p.read_text())
            except Exception:
                local = {}
        merged = dict(local)
        updated = 0
        for k, v in incoming.items():
            try:
                v_int = int(v)
            except Exception:
                continue
            if v_int > int(merged.get(k, 0)):
                merged[k] = v_int
                updated += 1
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.tmp')
        tmp.write_text(json.dumps(merged, indent=2))
        tmp.replace(p)
        return jsonify({'updated': updated, 'total': len(merged)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/processing/cache_manifest', methods=['GET', 'PUT'])
def processing_cache_manifest():
    """GET/PUT the Zenodo tile-cache manifest (cache_manifest.json).

    GET: returns the full manifest (depo_id, record_id, files).
    PUT: merges incoming manifest data — only updates files whose
         updated_at is newer than the local version.
    """
    manifest_path = Path('data/austria_processor/cache_manifest.json')

    if request.method == 'GET':
        if not manifest_path.exists():
            return jsonify({})
        try:
            return jsonify(json.loads(manifest_path.read_text()))
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # PUT — merge incoming manifest
    incoming = request.get_json(force=True)
    if not incoming:
        return jsonify({'error': 'empty body'}), 400

    try:
        local = {}
        if manifest_path.exists():
            local = json.loads(manifest_path.read_text())

        # Always adopt depo_id / record_id from the incoming manifest.
        # The primary is the authority — peers must use the shared deposit.
        if incoming.get('depo_id'):
            local['depo_id'] = incoming['depo_id']
        if incoming.get('record_id'):
            local['record_id'] = incoming['record_id']

        # Rewrite file URLs to point at the correct deposit
        target_depo = local['depo_id']

        def _rewrite_url(url, depo_id):
            """Rewrite a Zenodo draft file URL to use the given deposit ID."""
            if not url or not depo_id:
                return url
            import re
            return re.sub(r'/api/records/\d+/draft/files/', f'/api/records/{depo_id}/draft/files/', url)

        # Merge files — keep the entry with the newer updated_at
        local_files = local.get('files', {})
        incoming_files = incoming.get('files', {})
        updated = 0
        for zip_name, inc_entry in incoming_files.items():
            # Rewrite URL to target deposit
            if inc_entry.get('url') and target_depo:
                inc_entry = dict(inc_entry)
                inc_entry['url'] = _rewrite_url(inc_entry['url'], target_depo)
            loc_entry = local_files.get(zip_name)
            if loc_entry is None:
                local_files[zip_name] = inc_entry
                updated += 1
            else:
                # Compare updated_at timestamps
                loc_ts = loc_entry.get('updated_at', '')
                inc_ts = inc_entry.get('updated_at', '')
                if inc_ts > loc_ts:
                    local_files[zip_name] = inc_entry
                    updated += 1
        # Also rewrite existing file URLs in case depo_id changed
        if target_depo:
            for zn, entry in local_files.items():
                if entry.get('url'):
                    local_files[zn] = dict(entry)
                    local_files[zn]['url'] = _rewrite_url(entry['url'], target_depo)
        local['files'] = local_files

        # Atomic write
        import tempfile as _tf
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = _tf.mkstemp(dir=manifest_path.parent, suffix='.tmp', prefix='.cache_manifest_')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(local, f, indent=2, sort_keys=True)
            os.replace(tmp, manifest_path)
        except BaseException:
            try: os.unlink(tmp)
            except OSError: pass
            raise

        # Invalidate cached ZipIndex entries so next fetch picks up new URLs
        zip_idx_dir = Path('data/austria_processor/zenodo_zip_index')
        if updated > 0 and zip_idx_dir.exists():
            for f in zip_idx_dir.iterdir():
                f.unlink(missing_ok=True)
            log.info('Cache manifest: merged %d entries, cleared zip index cache', updated)

        return jsonify({'updated': updated, 'total_files': len(local_files)})
    except Exception as e:
        log.warning('Cache manifest PUT failed: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/processing/peers/status')
def processing_peers_combined():
    """Combined processing status across all peers + this instance."""
    import requests as req

    peer_urls = _get_peer_urls()

    instances = []

    # This instance
    local_completed = _get_completed_kgs()
    progress_file = Path('data/austria_processor/progress.json')
    local_state = 'idle'
    local_current = None
    local_rate = 0
    if progress_file.exists():
        try:
            pd = json.loads(progress_file.read_text())
            local_state = pd.get('state', 'idle')
            ckg = pd.get('current_kg') or {}
            local_current = ckg.get('code')
            local_rate = pd.get('rate_kgs_per_hour', 0)
        except Exception:
            pass
    # Detect stale running state (no processor PID found)
    if local_state in ('running', 'processing'):
        _alive = False
        try:
            import subprocess as _sp
            _sp.check_output(['pgrep', '-f', 'austria_processor.py'], text=True, timeout=2)
            _alive = True
        except Exception:
            pass
        if not _alive:
            local_state = 'stopped'
            local_current = None
            local_rate = 0

    instances.append({
        'instance': os.environ.get('INSTANCE_ID', 'primary'),
        'url': None,  # self
        'state': local_state,
        'current': local_current,
        'completed': len(local_completed),
        'completed_codes': sorted(local_completed),
        'rate': local_rate,
        'online': True,
    })

    all_completed = set(local_completed)

    # Peers
    for peer_url in peer_urls:
        entry = {
            'instance': '?',
            'url': peer_url,
            'state': 'unknown',
            'current': None,
            'completed': 0,
            'completed_codes': [],
            'rate': 0,
            'online': False,
        }
        try:
            r = req.get(peer_url.rstrip('/') + '/api/v1/processing/status', timeout=10)
            r.raise_for_status()
            pd = r.json()
            entry['state'] = pd.get('state', 'unknown')
            ckg = pd.get('current_kg') or {}
            entry['current'] = ckg.get('code')
            entry['completed'] = pd.get('completed', 0)
            entry['rate'] = pd.get('rate_kgs_per_hour', 0)
            entry['online'] = True

            # Also get peer identity from /peers
            try:
                r2 = req.get(peer_url.rstrip('/') + '/api/v1/processing/peers', timeout=5)
                r2.raise_for_status()
                pd2 = r2.json()
                entry['instance'] = pd2.get('instance', peer_url)
                entry['completed_codes'] = pd2.get('completed', [])
                all_completed.update(entry['completed_codes'])
            except Exception:
                pass
        except Exception as e:
            entry['error'] = str(e)

        instances.append(entry)

    return jsonify({
        'instances': instances,
        'total_kgs': 8440,
        'combined_completed': len(all_completed),
        'combined_completed_codes': sorted(all_completed),
        'combined_rate': sum(i.get('rate', 0) for i in instances if i.get('online')),
    })


@app.route('/api/v1/processing/start', methods=['POST'])
def processing_start():
    """Start the Austria processor as a background process."""
    global _processor_process
    if _processor_process is not None and _processor_process.poll() is None:
        return jsonify({'error': 'Processor already running', 'pid': _processor_process.pid}), 409
    # Also check if processor is running externally (e.g. via systemd or another start)
    import subprocess as _sp
    try:
        _sp.check_output(['pgrep', '-f', 'austria_processor.py'], text=True, timeout=2)
        return jsonify({'error': 'Processor already running (external)'}), 409
    except Exception:
        pass

    args = [sys.executable, 'austria_processor.py']
    state = request.args.get('state') or request.json.get('state', '') if request.is_json else ''
    kg = request.args.get('kg') or (request.json.get('kg', '') if request.is_json else '')
    no_cop = request.args.get('no_copernicus', 'false').lower() in ('true', '1')
    body = request.get_json(silent=True) or {}
    cache_only = (
        request.args.get('cache_only', '').lower() in ('1', 'true', 'yes')
        or bool(body.get('cache_only'))
    )
    # Per-peer assignments from the director:
    #   cred_indices: list[int] of credentials this process may use
    #   lat_strips:   list[[south,north]] of lat strips this peer is
    #                 dedicated to (cache-only orchestration)
    cred_indices = body.get('cred_indices')
    lat_strips = body.get('lat_strips')

    if kg:
        args.extend(['--kg', kg])
    elif state:
        args.extend(['--state', state])
    if no_cop:
        args.append('--no-copernicus')
    if cache_only:
        args.append('--cache-only')

    # Pass peer URLs so the subprocess can de-dup against peers' current
    # KGs (block-aware via parent_kg_code). Without this, two cache-only
    # peers can race the same KG when their whitelist slices overlap or
    # they fall through to nearest-neighbour scanning. Self-URL is
    # filtered out so we don't claim work against ourselves.
    try:
        peer_urls_for_proc = _get_peer_urls() or []
        # Drop our own URL — match by host
        import socket as _sock
        _self_host = _sock.gethostname()
        peer_urls_for_proc = [u for u in peer_urls_for_proc if _self_host not in u]
        if peer_urls_for_proc:
            args.append('--peers')
            args.extend(peer_urls_for_proc)
    except Exception as _pe:
        log.warning('Could not enumerate peer URLs for subprocess: %s', _pe)

    log_file = Path('data/austria_processor/logs/processor.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fd = open(log_file, 'a')

    # Inject Zenodo upload mutex broker URL.  Primary serves the broker
    # on its own gunicorn; peers point at the primary.  Reads from
    # data/austria_processor/zenodo_lock_url.txt if present, else
    # localhost when this instance is the director, else unset (no-op).
    proc_env = os.environ.copy()
    if cred_indices:
        try:
            cs = ','.join(str(int(i)) for i in cred_indices if int(i) >= 0)
            if cs:
                proc_env['COPERNICUS_CRED_INDICES'] = cs
        except Exception:
            pass
    if lat_strips:
        try:
            # Cells are 4-tuples [south, north, west, east]; legacy
            # strips are 2-tuples [south, north]. Route to the right
            # env var so the processor can apply the correct filter.
            tuples = [list(t) for t in lat_strips]
            if tuples and len(tuples[0]) == 4:
                proc_env['KG_CELL_FILTER'] = json.dumps(tuples)
            else:
                proc_env['KG_LAT_STRIP_FILTER'] = json.dumps(tuples)
        except Exception:
            pass
    if 'ZENODO_LOCK_URL' not in proc_env:
        lock_file = Path('data/austria_processor/zenodo_lock_url.txt')
        is_director = Path('data/austria_processor/is_director').exists()
        if lock_file.exists():
            proc_env['ZENODO_LOCK_URL'] = lock_file.read_text().strip()
        elif is_director:
            # On the director, prefer the standalone broker on port
            # 8001 when reachable. Insulates lock state from gunicorn
            # slow paths. Fallback to in-gunicorn route on port 8000.
            broker = 'http://127.0.0.1:8001'
            try:
                import requests as _req
                _r = _req.get(broker + '/api/v1/zenodo/lock', timeout=1.0)
                if _r.status_code == 200:
                    proc_env['ZENODO_LOCK_URL'] = broker
                else:
                    proc_env['ZENODO_LOCK_URL'] = 'http://127.0.0.1:8000'
            except Exception:
                proc_env['ZENODO_LOCK_URL'] = 'http://127.0.0.1:8000'

    import subprocess
    # Launch inside a transient systemd scope so the processor lives in its
    # own cgroup with proper memory limits (8G/7G), instead of inheriting
    # srv.service's 5G/4G limits.  Without this, heavy KGs trigger memory
    # throttling (mem_cgroup_handle_over_high — observed 3-hour Felzenszwalb
    # stalls on at2).  We use --scope (not --service) so:
    #   * lifecycle stays with the director (no Restart=, no autostart)
    #   * _processor_process Popen handle still works for stop/pause/resume
    #   * subprocess survives srv.service restarts (start_new_session)
    #   * single source of truth: only the director starts processors
    unit_name = f'austria-processor-{int(time.time())}'
    # Probe which -p properties this systemd accepts. systemd 255 (Ubuntu
    # 24.04) rejects OOMScoreAdjust as a transient property and aborts
    # systemd-run with rc=1 — which silently kills the processor at start.
    # Probe each property with a tiny /bin/true and only keep the ones
    # that are accepted.
    def _probe_prop(prop):
        try:
            r = subprocess.run(
                ['sudo', '-n', 'systemd-run', '--scope', '--quiet',
                 '--unit', f'srtm-probe-{int(time.time()*1000)}',
                 '-p', prop, '--', '/bin/true'],
                capture_output=True, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False
    cgroup_props = []
    for p in ('MemoryMax=8G', 'MemoryHigh=7G', 'OOMScoreAdjust=100'):
        if _probe_prop(p):
            cgroup_props += ['-p', p]
        else:
            log.warning('systemd-run does not accept %s on this host — dropping', p)
    # Drop privileges back to the exedev user so the subprocess can
    # import packages installed under ~/.local (e.g. pyproj). sudo +
    # systemd-run --scope otherwise spawns the processor as root with
    # HOME=/root, which fails to discover user-site packages and dies
    # with `ModuleNotFoundError: No module named 'pyproj'` at the very
    # first import. PYTHONUNBUFFERED keeps the log tail responsive.
    import pwd as _pwd
    try:
        _user = _pwd.getpwuid(os.getuid())
        _user_name = _user.pw_name
        _user_home = _user.pw_dir
    except Exception:
        _user_name, _user_home = 'exedev', '/home/exedev'
    # NOTE: `-p User=` is silently ignored by systemd-run --scope (it only
    # applies to --service units). The processor was running as root,
    # which made /proc/<pid>/environ unreadable from gunicorn (uid=exedev)
    # and required sudo escalation just to kill it. Fix: drop privileges
    # explicitly via `runuser -u exedev --preserve-environment` inside the
    # scope, so all -E env vars reach the python child but uid is exedev.
    user_props = [
        '-E', f'HOME={_user_home}',
        '-E', f'USER={_user_name}',
        '-E', 'PYTHONUNBUFFERED=1',
    ]
    # systemd-run --scope with -p User= drops the Popen env=. Only -E
    # values are forwarded into the new user session. Promote our
    # director-set assignment env vars (and a few critical infra ones)
    # into -E flags so they actually reach the subprocess.
    _forward_keys = (
        'COPERNICUS_CRED_INDICES',
        'KG_LAT_STRIP_FILTER',
        'KG_CELL_FILTER',
        'ZENODO_LOCK_URL',
        'COPERNICUS_FORBIDDEN',
        'PATH',
        'LANG',
        'LC_ALL',
    )
    for _k in _forward_keys:
        _v = proc_env.get(_k)
        if _v:
            user_props += ['-E', f'{_k}={_v}']
    scope_args = (
        ['sudo', '-n', 'systemd-run', '--scope', '--quiet',
         '--unit', unit_name]
        + cgroup_props
        + user_props
        + ['--', 'runuser', '--user', _user_name,
           '--preserve-environment', '--']
        + args
    )
    # Verify the scope can actually start by waiting briefly. If systemd-run
    # exits with non-zero (rejected property, sudo failure, ...), fall back
    # to plain Popen instead of letting the processor silently die.
    use_scope = True
    try:
        _processor_process = subprocess.Popen(
            scope_args, stdout=log_fd, stderr=subprocess.STDOUT,
            start_new_session=True, env=proc_env,
        )
        # Brief liveness check: systemd-run --scope keeps the parent alive
        # for the lifetime of the child. If it dies within 2s, the child
        # is gone.
        time.sleep(2)
        if _processor_process.poll() is not None:
            log.warning('systemd-run scope died with rc=%s — falling back to plain Popen',
                        _processor_process.returncode)
            raise RuntimeError(f'scope rc={_processor_process.returncode}')
        log.info('Austria processor started in scope %s: PID %d, args=%s',
                 unit_name, _processor_process.pid, args)
    except Exception as e:
        log.warning('systemd-run scope failed (%s); falling back to plain Popen', e)
        use_scope = False
        _processor_process = subprocess.Popen(
            args, stdout=log_fd, stderr=subprocess.STDOUT,
            start_new_session=True, env=proc_env,
        )
        log.info('Austria processor started (no cgroup): PID %d, args=%s',
                 _processor_process.pid, args)
    return jsonify({
        'status': 'started',
        'pid': _processor_process.pid,
        'method': 'systemd_scope' if use_scope else 'subprocess',
        'scope': unit_name if use_scope else None,
    })


@app.route('/api/v1/processing/pause', methods=['POST'])
def processing_pause():
    """Pause the processor (sends SIGSTOP to its process group)."""
    global _processor_process
    if _processor_process is None or _processor_process.poll() is not None:
        return jsonify({'error': 'Processor not running'}), 404
    import signal as _sig
    # Use killpg: when launched via `sudo systemd-run --scope`, the Popen
    # PID is sudo's, and sudo doesn't forward SIGSTOP to its child.  The
    # whole chain (sudo → systemd-run → python) shares one process group
    # because we set start_new_session=True, so killpg reaches python.
    try:
        os.killpg(os.getpgid(_processor_process.pid), _sig.SIGSTOP)
    except (ProcessLookupError, PermissionError, OSError):
        os.kill(_processor_process.pid, _sig.SIGSTOP)
    # Update progress file
    pf = Path('data/austria_processor/progress.json')
    if pf.exists():
        d = json.loads(pf.read_text())
        d['state'] = 'paused'
        pf.write_text(json.dumps(d, indent=2, default=str))
    return jsonify({'status': 'paused', 'pid': _processor_process.pid})


@app.route('/api/v1/processing/resume', methods=['POST'])
def processing_resume():
    """Resume the processor (sends SIGCONT to its process group)."""
    global _processor_process
    if _processor_process is None or _processor_process.poll() is not None:
        return jsonify({'error': 'Processor not running'}), 404
    import signal as _sig
    try:
        os.killpg(os.getpgid(_processor_process.pid), _sig.SIGCONT)
    except (ProcessLookupError, PermissionError, OSError):
        os.kill(_processor_process.pid, _sig.SIGCONT)
    pf = Path('data/austria_processor/progress.json')
    if pf.exists():
        d = json.loads(pf.read_text())
        d['state'] = 'running'
        pf.write_text(json.dumps(d, indent=2, default=str))
    return jsonify({'status': 'resumed', 'pid': _processor_process.pid})


@app.route('/api/v1/processing/stop', methods=['POST'])
def processing_stop():
    """Stop the processor and ALL its child subprocesses.

    The processor spawns children via multiprocessing.Pool(1) which can
    get stuck in long I/O (e.g. SSL uploads).  We kill the entire
    process group so nothing survives.

    Query/JSON params:
      graceful=1 → send SIGTERM only, don't escalate to SIGKILL. The
        processor sets _shutdown_requested and exits cleanly *after* the
        current KG finishes (can take an hour or more). Used by the
        director when pre-empting between KGs.
      after_kg=1 → alias for graceful, returns immediately.

    Strategy (default = hard stop):
    1. If we have the Popen handle → kill the whole process group
       (parent started with start_new_session=True so its PID == PGID).
    2. Else systemctl stop (sends SIGTERM to the cgroup).
    3. Else pkill SIGTERM all matching processes.
    4. Wait 3 s, then SIGKILL any survivors.
    """
    global _processor_process
    import subprocess as _sp
    import signal as _sig
    method = None

    # Parse graceful flag from query string OR JSON body
    body = request.get_json(silent=True) or {}
    graceful = (request.args.get('graceful') in ('1', 'true', 'yes')
                or request.args.get('after_kg') in ('1', 'true', 'yes')
                or body.get('graceful') is True
                or body.get('after_kg') is True)
    if graceful:
        # Send SIGTERM to the process group; the processor's signal
        # handler sets _shutdown_requested and the loop exits at the next
        # KG boundary. We do NOT escalate to SIGKILL.
        sent = False
        if _processor_process is not None and _processor_process.poll() is None:
            try:
                os.killpg(os.getpgid(_processor_process.pid), _sig.SIGTERM)
                sent = True
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if not sent:
            try:
                _sp.run(['pkill', '-TERM', '-f', 'austria_processor.py'],
                        capture_output=True, text=True, timeout=5)
                sent = True
            except Exception:
                pass
        return jsonify({
            'status': 'graceful_stop_requested' if sent else 'no_process',
            'method': 'sigterm_no_escalate',
            'note': 'processor will exit after current KG finishes',
        })

    def _proc_alive():
        try:
            _sp.check_output(['pgrep', '-f', 'austria_processor.py'], text=True, timeout=2)
            return True
        except Exception:
            return False

    def _kill_process_group(pid, sig):
        """Kill the entire process group rooted at *pid*."""
        try:
            os.killpg(os.getpgid(pid), sig)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    if _processor_process is not None and _processor_process.poll() is None:
        pid = _processor_process.pid
        # Kill entire process group (parent + multiprocessing children)
        _kill_process_group(pid, _sig.SIGTERM)
        _processor_process = None
        method = 'subprocess_pgkill'
    else:
        # Try systemctl stop — this sends SIGTERM to the whole cgroup
        try:
            r = _sp.run(['sudo', 'systemctl', 'stop', 'austria_processor'],
                        capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                method = 'systemctl'
        except Exception:
            pass
        # Fallback: pkill SIGTERM
        if not method:
            try:
                r = _sp.run(['pkill', '-f', 'austria_processor.py'],
                            capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    method = 'pkill'
            except Exception:
                pass

    # Wait briefly, then SIGKILL if still alive
    if method and _proc_alive():
        time.sleep(3)
        if _proc_alive():
            # SIGKILL via process-group first, then pkill as backstop
            killed_pg = False
            if _processor_process is not None:
                killed_pg = _kill_process_group(
                    _processor_process.pid, _sig.SIGKILL)
            if not killed_pg:
                try:
                    _sp.run(['pkill', '-9', '-f', 'austria_processor.py'],
                            capture_output=True, text=True, timeout=5)
                except Exception:
                    pass
            method = method + '+kill9'
    elif not method:
        # Nothing worked with SIGTERM — go straight to SIGKILL
        try:
            r = _sp.run(['pkill', '-9', '-f', 'austria_processor.py'],
                        capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                method = 'kill9'
        except Exception:
            pass
    # Final escalation: processor was launched via `sudo systemd-run
    # --scope`, which puts the python child in its own systemd cgroup
    # owned by root. Plain SIGKILL from the gunicorn user fails with
    # EPERM on a few processes (the scope wrapper) even though the
    # python child is exedev-owned. Killing the entire scope cgroup via
    # sudo systemctl kill is the only thing that reliably reaches
    # everything (parent + multiprocessing children + any sudo wrapper).
    if _proc_alive():
        try:
            scopes = _sp.check_output(
                ['systemctl', 'list-units', '--type=scope', '--no-legend',
                 '--plain', '--all'],
                text=True, timeout=5,
            ).splitlines()
            for ln in scopes:
                name = ln.strip().split(None, 1)[0]
                if name.startswith('austria-processor-') and name.endswith('.scope'):
                    _sp.run(['sudo', '-n', 'systemctl', 'kill',
                             '--signal=SIGKILL', '--kill-whom=all', name],
                            capture_output=True, text=True, timeout=10)
                    _sp.run(['sudo', '-n', 'systemctl', 'stop', name],
                            capture_output=True, text=True, timeout=10)
                    method = (method or '') + '+scope_kill'
        except Exception as _e:
            log.warning('scope-kill failed: %s', _e)
        # Last resort: sudo pkill -9 (matches root-owned wrapper too).
        try:
            _sp.run(['sudo', '-n', 'pkill', '-9', '-f', 'austria_processor.py'],
                    capture_output=True, text=True, timeout=5)
        except Exception:
            pass
        if _proc_alive():
            method = (method or '') + '+still_alive'
    if not method:
        return jsonify({'error': 'Processor not running'}), 404
    pf = Path('data/austria_processor/progress.json')
    if pf.exists():
        try:
            d = json.loads(pf.read_text())
            d['state'] = 'stopped'
            pf.write_text(json.dumps(d, indent=2, default=str))
        except Exception:
            pass
    return jsonify({'status': 'stopped', 'method': method})


@app.route('/api/v1/processing/postpone', methods=['POST'])
def processing_postpone():
    """Postpone current KG — kill subprocess, re-queue 5 KGs later, no fail count bump.

    Optional ``peer_id`` (query or JSON body) proxies the request to that
    peer so the dashboard can postpone whichever peer's KG card is on
    screen, not just the local primary.
    """
    peer_id = request.args.get('peer_id') or (
        (request.get_json(silent=True) or {}).get('peer_id'))
    if peer_id:
        cfg = pd.load_peers_config()
        peer_cfg = pd.get_peer_by_id(cfg, peer_id)
        if not peer_cfg:
            return jsonify({'error': f'Peer {peer_id} not found'}), 404
        url = peer_cfg.get('url')
        if url:
            try:
                import requests as _req
                r = _req.post(url.rstrip('/') + '/api/v1/processing/postpone',
                              timeout=(3, 8), headers=pd._admin_headers())
                return (r.text, r.status_code,
                        {'Content-Type': r.headers.get('Content-Type', 'application/json')})
            except Exception as e:
                return jsonify({'error': f'Proxy to {peer_id} failed: {e}'}), 502
        # else: peer is the local primary, fall through
    postpone_file = Path('data/austria_processor/postpone_signal.json')
    # Read current KG from progress
    pf = Path('data/austria_processor/progress.json')
    kg_code = None
    if pf.exists():
        try:
            d = json.loads(pf.read_text())
            ckg = d.get('current_kg', {})
            kg_code = ckg.get('code')
        except Exception:
            pass
    if not kg_code:
        return jsonify({'error': 'No current KG to postpone'}), 404
    # Write signal file — processor main loop picks this up
    import datetime as _dt
    postpone_file.write_text(json.dumps({
        'kg_code': kg_code,
        'ts': _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }))
    return jsonify({'status': 'postpone_requested', 'kg_code': kg_code})


@app.route('/api/v1/processing/single', methods=['POST'])
def processing_single():
    """Process a single KG (async in background)."""
    kg = request.args.get('kg') or (request.json.get('kg', '') if request.is_json else '')
    if not kg:
        return jsonify({'error': 'kg parameter required'}), 400
    return processing_start()  # reuse start with kg param


@app.route('/api/v1/processing/retry', methods=['POST'])
def processing_retry():
    """Retry a specific failed KG.

    Writes the KG code to ``retry_queue.json``.  The running processor
    picks this up each iteration and inserts the KG as the very next
    item in its processing queue.  Also cleans up the failed/progress
    state so the KG is no longer shown as failed.

    If the processor is NOT running, it is started automatically.
    """
    kg = request.args.get('kg') or (request.json.get('kg', '') if request.is_json else '')
    if not kg:
        return jsonify({'error': 'kg parameter required'}), 400

    data_dir = Path('data/austria_processor')
    actions = []

    # Write to retry queue (processor reads + clears each iteration)
    retry_path = data_dir / 'retry_queue.json'
    try:
        existing = []
        if retry_path.exists():
            existing = json.loads(retry_path.read_text())
        if kg not in existing:
            existing.append(kg)
        retry_path.write_text(json.dumps(existing))
        actions.append('added to retry_queue')
    except Exception as e:
        log.warning('retry: retry_queue.json: %s', e)

    # Remove from failed_kgs.json
    failed_path = data_dir / 'failed_kgs.json'
    if failed_path.exists():
        try:
            codes = set(json.loads(failed_path.read_text()))
            if kg in codes:
                codes.discard(kg)
                failed_path.write_text(json.dumps(sorted(codes), indent=2))
                actions.append('removed from failed_kgs')
        except Exception as e:
            log.warning('retry: failed_kgs.json: %s', e)

    # Remove from retried_kgs.json so it gets a fresh retry pass
    retried_path = data_dir / 'retried_kgs.json'
    if retried_path.exists():
        try:
            codes = set(json.loads(retried_path.read_text()))
            if kg in codes:
                codes.discard(kg)
                retried_path.write_text(json.dumps(sorted(codes), indent=2))
                actions.append('removed from retried_kgs')
        except Exception as e:
            log.warning('retry: retried_kgs.json: %s', e)

    # Reset failure count
    fc_path = data_dir / 'failure_counts.json'
    if fc_path.exists():
        try:
            fc = json.loads(fc_path.read_text())
            if kg in fc:
                del fc[kg]
                fc_path.write_text(json.dumps(fc, indent=2))
                actions.append('reset failure count')
        except Exception as e:
            log.warning('retry: failure_counts.json: %s', e)

    # Remove from progress tracker failed_kgs list
    progress_path = data_dir / 'progress.json'
    if progress_path.exists():
        try:
            d = json.loads(progress_path.read_text())
            flist = d.get('failed_kgs', [])
            d['failed_kgs'] = [f for f in flist if f.get('code') != kg]
            if len(d['failed_kgs']) != len(flist):
                d['failed'] = max(0, d.get('failed', 0) - 1)
                progress_path.write_text(json.dumps(d, indent=2, default=str))
                actions.append('removed from progress')
        except Exception as e:
            log.warning('retry: progress.json: %s', e)

    # Remove error entry from manifest
    manifest_path = data_dir / 'zenodo_manifest.json'
    if manifest_path.exists():
        try:
            from zenodo_client import Manifest
            m = Manifest(str(manifest_path))
            m.delete(f'{kg}_error')
            m.save()
            actions.append('removed from manifest')
        except Exception:
            pass

    # If processor is not running (neither systemd nor subprocess), start it
    processor_running = ((_processor_process is not None
                          and _processor_process.poll() is None)
                         or _is_processor_running())
    if not processor_running:
        return processing_start()

    # Processor is running — it will pick up the retry queue entry
    return jsonify({
        'status': 'queued_for_retry',
        'kg': kg,
        'actions': actions,
        'note': 'KG will be processed next by the running processor.',
    })


@app.route('/api/v1/processing/prioritize', methods=['POST'])
def processing_prioritize():
    """Prioritize KGs for processing — by bbox or explicit codes.

    Accepts JSON body with either:
      - bbox: {west, south, east, north} — resolves to KG codes via cadastre API
      - kgs: ["63349", "63350", ...] — explicit KG codes

    Filters out already-processed KGs and writes remaining to retry_queue.json.
    The running processor picks these up as next-in-queue.
    """
    data = request.get_json(silent=True) or {}
    # Also accept query params for bbox
    bbox = data.get('bbox')
    kgs = data.get('kgs', [])

    if not bbox and not kgs:
        # Try query params
        try:
            bbox = {
                'west': float(request.args['west']),
                'south': float(request.args['south']),
                'east': float(request.args['east']),
                'north': float(request.args['north']),
            }
        except (KeyError, ValueError):
            pass

    if not bbox and not kgs:
        return jsonify({'error': 'Provide bbox (west/south/east/north) or kgs array'}), 400

    # Resolve bbox to KG codes via cadastre API
    if bbox:
        try:
            import requests as req
            r = req.get(
                'https://cadastre-process-api.exe.xyz/api/v1/spatial/kgs',
                params={
                    'west': bbox['west'], 'south': bbox['south'],
                    'east': bbox['east'], 'north': bbox['north'],
                    'fields': 'kg_code,kg_name',
                },
                timeout=15,
            )
            r.raise_for_status()
            resp = r.json()
            kgs = resp.get('data', {}).get('kg_codes', [])
        except Exception as e:
            return jsonify({'error': f'Failed to resolve bbox to KGs: {e}'}), 502

    if not kgs:
        return jsonify({'error': 'No KGs found in the given area'}), 404

    # Filter out already-processed KGs.
    # `processed` covers direct completion + block codes (e.g. '49006-south').
    # If a parent KG was *split* and all blocks are done, the parent code
    # itself is NOT in `processed` (only its block codes are). Detect that
    # case so we don't requeue '49006' when '49006-south/center/north' are
    # already done. Mirrors the GET handler's `_parent_fully_done` logic.
    data_dir = Path('data/austria_processor')
    processed = _get_completed_kgs()
    try:
        from kg_splitter import maybe_split_kg, all_block_codes_for_parent, is_block_code
        idx = si.get_index()
        _conn = idx._conn()
        def _parent_fully_done(code: str) -> bool:
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
            done = all_block_codes_for_parent(code, processed)
            return len(done) >= len(blocks)
    except Exception:
        def _parent_fully_done(code: str) -> bool:
            return False
    unprocessed = [k for k in kgs
                   if k not in processed and not _parent_fully_done(k)]
    already_done = [k for k in kgs
                    if k in processed or _parent_fully_done(k)]

    # Write unprocessed to front of retry queue (priority = first)
    queued = []
    if unprocessed:
        retry_path = data_dir / 'retry_queue.json'
        try:
            existing = []
            if retry_path.exists():
                existing = json.loads(retry_path.read_text())
            existing_set = set(existing)
            for code in unprocessed:
                if code not in existing_set:
                    queued.append(code)
            # Prepend new KGs to front of queue
            existing = queued + existing
            retry_path.write_text(json.dumps(existing))
        except Exception as e:
            return jsonify({'error': f'Failed to write retry queue: {e}'}), 500

    return jsonify({
        'status': 'prioritized',
        'total_kgs': len(kgs),
        'queued': len(queued),
        'already_processed': len(already_done),
        'already_in_queue': len(unprocessed) - len(queued),
        'queued_codes': queued,
        'already_done_codes': already_done,
        'note': f'{len(queued)} KGs queued for priority processing.' if queued else 'All KGs already processed.',
    })


def _get_completed_kgs() -> set:
    """Return set of KG codes that have been successfully processed.

    Tombstoned KGs (force-requeued by an operator) are excluded so they
    aren't silently filtered out of the priority queue or advertised
    as completed to peers.
    """
    data_dir = Path('data/austria_processor')
    completed = set()
    manifest_path = data_dir / 'zenodo_manifest.json'
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            entries = m.get('entries', m)
            for key in entries:
                if key.endswith('_json') and 'error' not in entries[key].get('status', ''):
                    completed.add(key.replace('_json', ''))
        except Exception:
            pass
    json_dir = data_dir / 'json'
    if json_dir.exists():
        for jf in json_dir.glob('*.json'):
            completed.add(jf.stem)
    # Drop tombstoned KGs (force-requeued by operator).
    try:
        if _tombstone_path.exists():
            tdata = json.loads(_tombstone_path.read_text())
            if isinstance(tdata, dict):
                import re as _re_c
                for tk in tdata.keys():
                    _m = _re_c.match(r'^(\d+(?:-[a-z][-a-z0-9]*)?)_', tk)
                    if _m:
                        completed.discard(_m.group(1))
    except Exception:
        pass
    return completed


@app.route('/api/v1/processing/completed_recent')
def processing_completed_recent():
    """Return recently completed KGs in chronological order (most recent last).

    Used by the dashboard to swipe through completed KGs.
    """
    data_dir = Path('data/austria_processor')
    manifest_path = data_dir / 'zenodo_manifest.json'
    items = []
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            entries = m.get('entries', m)
            for key, val in entries.items():
                if not key.endswith('_json'):
                    continue
                if 'error' in (val.get('status', '') or ''):
                    continue
                code = key[:-5]
                ua = val.get('uploaded_at') or ''
                items.append((ua, code))
        except Exception:
            pass
    items.sort()
    codes = [c for _, c in items]
    try:
        limit = int(request.args.get('limit', '500'))
    except Exception:
        limit = 500
    if limit > 0 and len(codes) > limit:
        codes = codes[-limit:]
    return jsonify({'codes': codes, 'count': len(codes)})


@app.route('/api/v1/processing/queue')
def processing_queue_get():
    """Read the priority queue with KG names and failure counts."""
    data_dir = Path('data/austria_processor')
    retry_path = data_dir / 'retry_queue.json'
    codes = []
    if retry_path.exists():
        try:
            codes = json.loads(retry_path.read_text())
        except Exception:
            pass
    # Filter out already-completed KGs, BUT keep tombstoned ones (queued for reprocessing)
    completed = _get_completed_kgs()
    # Re-read tombstone file from disk — gunicorn workers don't share memory,
    # so a POST that adds tombstones in worker A must be visible to GET in worker B.
    if _tombstone_path.exists():
        try:
            _disk_tombstones = json.loads(_tombstone_path.read_text())
            if isinstance(_disk_tombstones, dict):
                _MANIFEST_TOMBSTONES.update(_disk_tombstones)
        except Exception:
            pass
    # Extract KG code (digits, or digits-label for blocks) from tombstone keys like '12362_json', '49006-north_json'
    import re as _re
    tombstoned_kgs = set()
    # Resolve tombstones that have already been satisfied: if the KG has a
    # _json manifest entry uploaded AFTER the tombstone timestamp, the peer
    # has reprocessed it — drop the stale tombstone.  This handles the case
    # where a peer ran the KG (cleared its own tombstone, but the primary's
    # _requeue tombstone is invisible to peers).
    _stale_tombstones = []
    try:
        _mf_path = Path('data/austria_processor/zenodo_manifest.json')
        _mf_entries = {}
        if _mf_path.exists():
            _mf_data = json.loads(_mf_path.read_text())
            _mf_entries = _mf_data.get('entries', _mf_data) or {}
    except Exception:
        _mf_entries = {}
    for key, ts_val in list(_MANIFEST_TOMBSTONES.items()):
        _m = _re.match(r'^(\d+(?:-[a-z][-a-z0-9]*)?)_', key)
        if not _m:
            continue
        _kg = _m.group(1)
        # Check if this KG has a fresh _json upload after the tombstone
        _json_entry = _mf_entries.get(_kg + '_json')
        if _json_entry:
            _uploaded_at = _json_entry.get('uploaded_at', '')
            if _uploaded_at and _uploaded_at > str(ts_val):
                _stale_tombstones.append(key)
                continue
        tombstoned_kgs.add(_kg)
    if _stale_tombstones:
        for k in _stale_tombstones:
            _MANIFEST_TOMBSTONES.pop(k, None)
        try:
            _tombstone_path.write_text(json.dumps(_MANIFEST_TOMBSTONES, indent=2))
            log.info('Cleared %d stale tombstone(s) (KG reprocessed by peer): %s',
                     len(_stale_tombstones), _stale_tombstones[:5])
        except Exception:
            pass
    # Build a set of "effectively completed" codes — includes parent codes whose
    # all blocks are done (e.g. '49006' when '49006-south/center/north' are done).
    from kg_splitter import is_block_code, maybe_split_kg, all_block_codes_for_parent
    try:
        idx = si.get_index()
        _conn = idx._conn()
        def _parent_fully_done(code: str) -> bool:
            if is_block_code(code):
                return False  # block codes checked directly
            row = _conn.execute(
                'SELECT min_lon, min_lat, max_lon, max_lat, kg_name FROM kg WHERE kg_code=?',
                (code,)
            ).fetchone()
            if not row or row['min_lon'] is None:
                return False
            fake_kg = {'kg_code': code, 'kg_name': row['kg_name'],
                       'bbox': {'min_lon': row['min_lon'], 'min_lat': row['min_lat'],
                                'max_lon': row['max_lon'], 'max_lat': row['max_lat']}}
            blocks = maybe_split_kg(fake_kg)
            if len(blocks) <= 1:
                return False  # not a split KG
            done = all_block_codes_for_parent(code, completed)
            return len(done) >= len(blocks)
    except Exception:
        def _parent_fully_done(code: str) -> bool:
            return False

    dirty = len(codes)
    codes = [c for c in codes if
             (c not in completed and not _parent_fully_done(c)) or c in tombstoned_kgs]
    if len(codes) < dirty:
        # Persist the cleaned list
        try:
            retry_path.write_text(json.dumps(codes))
        except Exception:
            pass
    # Also filter out the currently-processing KG for display
    # (it's already shown in the Current KG card)
    current_kg = None
    pf = data_dir / 'progress.json'
    if pf.exists():
        try:
            pd = json.loads(pf.read_text())
            current_kg = (pd.get('current_kg') or {}).get('code')
        except Exception:
            pass
    if current_kg:
        codes = [c for c in codes if c != current_kg]
    # Load deferred KG codes (KGs awaiting retry after transient failure)
    deferred_codes = set()
    deferred_path = data_dir / 'deferred_kgs.json'
    if deferred_path.exists():
        try:
            deferred_codes = set(json.loads(deferred_path.read_text()))
        except Exception:
            pass
    # Load failure counts
    failure_counts = {}
    fc_path = data_dir / 'failure_counts.json'
    if fc_path.exists():
        try:
            failure_counts = json.loads(fc_path.read_text())
        except Exception:
            pass
    # Load permanently failed KGs — also filter out completed
    perm_failed = []
    failed_path = data_dir / 'failed_kgs.json'
    if failed_path.exists():
        try:
            all_failed = json.loads(failed_path.read_text())
            perm_failed = [c for c in all_failed if c not in completed]
            if len(perm_failed) < len(all_failed):
                failed_path.write_text(json.dumps(sorted(perm_failed), indent=2))
        except Exception:
            pass
    # Resolve names from search index
    items = []
    perm_failed_items = []
    try:
        import math as _math
        idx = si.get_index()
        conn = idx._conn()
        def _est_tiles(min_lon, min_lat, max_lon, max_lat, tile_km=1.5, overlap_km=0.1):
            """Estimate number of processing tiles from KG bbox."""
            if min_lon is None or min_lat is None:
                return None
            cos_lat = _math.cos(_math.radians((min_lat + max_lat) / 2))
            step_x = (tile_km - overlap_km) / (111 * cos_lat)
            step_y = (tile_km - overlap_km) / 111
            nx = max(1, _math.ceil((max_lon - min_lon) / step_x))
            ny = max(1, _math.ceil((max_lat - min_lat) / step_y))
            return nx * ny
        def _resolve(code):
            from kg_splitter import (is_block_code, parent_kg_code as _parent_code,
                                     block_label as _block_label, maybe_split_kg,
                                     all_block_codes_for_parent)
            _lookup_code = _parent_code(code) if is_block_code(code) else code
            row = conn.execute(
                'SELECT kg_name, gemeinde_name, district_name, min_lon, min_lat, max_lon, max_lat FROM kg WHERE kg_code=?',
                (_lookup_code,)
            ).fetchone()
            is_tombstoned = code in tombstoned_kgs
            is_deferred = code in deferred_codes
            if row:
                _name = row['kg_name']
                _lbl = _block_label(code)
                if _lbl:
                    _name = f"{_name} ({_lbl})"
                item = {'code': code, 'name': _name,
                        'gemeinde': row['gemeinde_name'],
                        'district': row['district_name'],
                        'failures': failure_counts.get(code, 0),
                        'est_tiles': _est_tiles(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat']),
                        'tombstoned': is_tombstoned,
                        'deferred': is_deferred}
                # Add block split info for parent KG codes (not already a block)
                if not is_block_code(code) and row['min_lon'] is not None:
                    _fake_kg = {
                        'kg_code': code,
                        'kg_name': row['kg_name'],
                        'bbox': {'min_lon': row['min_lon'], 'min_lat': row['min_lat'],
                                 'max_lon': row['max_lon'], 'max_lat': row['max_lat']},
                    }
                    _blocks = maybe_split_kg(_fake_kg)
                    if len(_blocks) > 1:
                        _done_codes = all_block_codes_for_parent(code, completed)
                        _done_labels = [_block_label(c) for c in _done_codes]
                        _all_labels = [b['_block_label'] for b in _blocks]
                        item['n_blocks'] = len(_blocks)
                        item['blocks_done'] = _done_labels
                        item['blocks_pending'] = [l for l in _all_labels if l not in _done_labels]
                return item
            return {'code': code, 'name': code, 'failures': failure_counts.get(code, 0), 'tombstoned': is_tombstoned, 'deferred': is_deferred}
        items = [_resolve(c) for c in codes]
        perm_failed_items = [_resolve(c) for c in perm_failed]
    except Exception:
        items = [{'code': c, 'name': c, 'failures': failure_counts.get(c, 0), 'tombstoned': c in tombstoned_kgs} for c in codes]
        perm_failed_items = [{'code': c, 'name': c, 'failures': failure_counts.get(c, 0), 'tombstoned': c in tombstoned_kgs} for c in perm_failed]
    return jsonify({
        'queue': items, 'count': len(items),
        'permanently_failed': perm_failed_items,
        'permanently_failed_count': len(perm_failed_items),
    })


@app.route('/api/v1/processing/queue', methods=['POST'])
def processing_queue_add():
    """Add KG codes to the priority queue at a specific position.

    Body JSON:
      kgs: list of KG codes to add (required)
      position: 0-based insertion index (default 0 = front)
               Use -1 or omit to append at end.
      skip_processed: if true (default), silently drop already-processed KGs

    Duplicates already in the queue are moved to the new position.
    """
    data = request.get_json(silent=True) or {}
    new_codes = data.get('kgs') or data.get('kg_codes') or data.get('queue', [])
    if isinstance(new_codes, str):
        new_codes = [new_codes]
    if not isinstance(new_codes, list) or not new_codes:
        return jsonify({'error': 'kgs must be a non-empty array of KG codes'}), 400
    new_codes = [str(c).strip() for c in new_codes if str(c).strip()]
    position = data.get('position', -1)
    skip_processed = data.get('skip_processed', True)

    retry_path = Path('data/austria_processor/retry_queue.json')
    try:
        codes = json.loads(retry_path.read_text()) if retry_path.exists() else []
    except Exception:
        codes = []

    # Optionally filter out already-processed; otherwise tombstone their manifest entries
    skipped = []
    tombstoned = []
    completed = _get_completed_kgs()
    if skip_processed:
        kept = []
        for c in new_codes:
            if c in completed:
                skipped.append(c)
            else:
                kept.append(c)
        new_codes = kept
    else:
        # Force re-process: tombstone existing manifest entries so sync doesn't re-import them.
        # ALSO add a synthetic '_requeue' tombstone for any completed KG so the GET handler
        # (which filters out completed codes unless tombstoned) keeps it in the queue.
        import datetime as _dt
        ts = _dt.datetime.utcnow().isoformat()
        manifest_path = Path('data/austria_processor/zenodo_manifest.json')
        mentries = None
        mdata = None
        if manifest_path.exists():
            try:
                mdata = json.loads(manifest_path.read_text())
                mentries = mdata.get('entries', mdata)
            except Exception:
                mentries = None
        changed = False
        for c in new_codes:
            # Mark as force-requeued whenever the KG is either fully
            # completed OR has any partial manifest entries from a
            # failed prior run (e.g. _full_gpkg / _light_gpkg uploaded
            # but _json missing). Without this, partial entries linger
            # in the Zenodo manifest and get re-merged via peer-sync.
            # A re-queued parent code (e.g. '91119') expands into block
            # codes ('91119-west', '91119-east', ...) at processing time;
            # partial manifest entries can be keyed by either form.  Match
            # both: ``c_<suffix>`` AND ``c-<dir>_<suffix>``.
            partial_keys = []
            if mentries is not None:
                suffixes = ('_full_gpkg', '_light_gpkg', '_json')
                for key in mentries:
                    if not any(key.endswith(s) for s in suffixes):
                        continue
                    code = key.rsplit('_', 2)[0] if key.endswith('_full_gpkg') or key.endswith('_light_gpkg') else key.rsplit('_', 1)[0]
                    # Strip suffix more reliably: split off the trailing
                    # _full_gpkg / _light_gpkg / _json.
                    for s in suffixes:
                        if key.endswith(s):
                            code = key[:-len(s)]
                            break
                    if code == c or code.startswith(c + '-'):
                        partial_keys.append(key)
            if c not in completed and not partial_keys:
                continue
            _MANIFEST_TOMBSTONES[c + '_requeue'] = ts
            tombstoned.append(c + '_requeue')
            for key in partial_keys:
                del mentries[key]
                _MANIFEST_TOMBSTONES[key] = ts
                tombstoned.append(key)
                changed = True
        if changed and mentries is not None:
            try:
                import tempfile as _tf
                fd, tmp = _tf.mkstemp(dir=manifest_path.parent, suffix='.tmp', prefix='.manifest_')
                try:
                    with os.fdopen(fd, 'w') as f:
                        json.dump({'entries': mentries}, f, indent=2, sort_keys=True)
                    os.replace(tmp, manifest_path)
                except BaseException:
                    try: os.unlink(tmp)
                    except OSError: pass
            except Exception:
                pass
        if tombstoned:
            try:
                _tombstone_path.write_text(json.dumps(_MANIFEST_TOMBSTONES, indent=2))
            except Exception:
                pass

    if not new_codes:
        return jsonify({
            'status': 'nothing_to_add',
            'skipped_processed': skipped,
            'queue_length': len(codes),
        })

    # Remove duplicates from existing queue (they'll be re-inserted)
    moved = [c for c in new_codes if c in codes]
    codes = [c for c in codes if c not in set(new_codes)]

    # Insert at position
    if position < 0 or position >= len(codes):
        codes.extend(new_codes)
        actual_pos = len(codes) - len(new_codes)
    else:
        for i, c in enumerate(new_codes):
            codes.insert(position + i, c)
        actual_pos = position

    retry_path.write_text(json.dumps(codes))

    # Resolve names
    added_info = []
    try:
        idx = si.get_index()
        conn = idx._conn()
        for c in new_codes:
            row = conn.execute(
                'SELECT kg_name, gemeinde_name, district_name FROM kg WHERE kg_code=?',
                (c,)
            ).fetchone()
            added_info.append({
                'code': c, 'name': row['kg_name'] if row else c,
                'gemeinde': row['gemeinde_name'] if row else None,
                'position': codes.index(c),
            })
    except Exception:
        added_info = [{'code': c, 'position': codes.index(c)} for c in new_codes]

    return jsonify({
        'status': 'added',
        'added': added_info,
        'added_count': len(new_codes),
        'moved_from_existing': moved,
        'skipped_processed': skipped,
        'tombstoned': tombstoned,
        'queue_length': len(codes),
    })


@app.route('/api/v1/processing/queue', methods=['PUT'])
def processing_queue_put():
    """Replace the entire priority queue."""
    data = request.get_json(silent=True) or {}
    codes = data.get('queue', [])
    if not isinstance(codes, list):
        return jsonify({'error': 'queue must be an array of KG codes'}), 400
    retry_path = Path('data/austria_processor/retry_queue.json')
    retry_path.write_text(json.dumps(codes))
    return jsonify({'status': 'saved', 'count': len(codes)})


@app.route('/api/v1/processing/queue', methods=['DELETE'])
def processing_queue_delete():
    """Remove a KG from the priority queue."""
    code = request.args.get('kg') or (request.get_json(silent=True) or {}).get('kg', '')
    if not code:
        return jsonify({'error': 'kg parameter required'}), 400
    retry_path = Path('data/austria_processor/retry_queue.json')
    try:
        codes = json.loads(retry_path.read_text()) if retry_path.exists() else []
        if code in codes:
            codes.remove(code)
            retry_path.write_text(json.dumps(codes))
            return jsonify({'status': 'removed', 'kg': code, 'remaining': len(codes)})
        return jsonify({'status': 'not_found', 'kg': code}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/processing/log')
def processing_log():
    """Return recent processor log lines."""
    log_file = Path('data/austria_processor/logs/processor.log')
    n = int(request.args.get('lines', 200))
    if not log_file.exists():
        return jsonify({'lines': [], 'total': 0})
    try:
        lines = log_file.read_text().splitlines()[-n:]
        return jsonify({'lines': lines, 'total': len(lines)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


_zenodo_warm_lock = threading.Lock()
_zenodo_warm_started: set = set()


def _warm_zenodo_zip_indices_async():
    """Fetch ZIP central directories for every file in cache_manifest.json.

    Runs once per (process, zip_name). Each fetched index is written to
    ``data/austria_processor/zenodo_zip_index/<md5>.json`` by ``ZipIndex``,
    where the ``processing/tiles`` endpoint picks it up on subsequent calls.
    """
    try:
        manifest_path = Path('data/austria_processor/cache_manifest.json')
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return

    files = (manifest.get('files') or {})
    to_warm = []
    with _zenodo_warm_lock:
        for zip_name in files:
            if zip_name in _zenodo_warm_started:
                continue
            _zenodo_warm_started.add(zip_name)
            to_warm.append(zip_name)
    if not to_warm:
        return

    def _worker(names):
        try:
            from zenodo_cache import ZenodoCache
        except Exception:
            return
        try:
            cache = ZenodoCache()
        except Exception:
            return
        for name in names:
            try:
                idx = cache._get_zip_index(name)
                if idx is not None:
                    # Trigger fetch + on-disk caching
                    idx.list_entries()
            except Exception:
                pass

    t = threading.Thread(target=_worker, args=(to_warm,), daemon=True)
    t.start()


@app.route('/api/v1/processing/tiles')
def processing_tiles():
    """Return Zenodo cache tile bboxes for map overlay in process.html.

    Reads the locally-cached ZIP central-directory indices written by
    ``zenodo_cache.ZipIndex``.  Each entry name encodes the grid bbox as
    ``{product}_{s}_{w}_{n}_{e}[_{year}].npz``.
    """
    cop_seen: set = set()
    han_seen: set = set()
    cop_tiles: list = []
    han_tiles: list = []

    # Kick off a background warm-up of the Zenodo ZIP central-directory cache
    # so subsequent calls can include peer-uploaded tiles. Idempotent: each
    # ZIP is fetched at most once per process and cached on disk.
    _warm_zenodo_zip_indices_async()

    def _add(product: str, w: float, s: float, e: float, n: float):
        key = (round(w, 4), round(s, 4), round(e, 4), round(n, 4))
        if product == 'hansen':
            if key in han_seen:
                return
            han_seen.add(key)
            han_tiles.append({'w': w, 's': s, 'e': e, 'n': n})
        else:
            if key in cop_seen:
                return
            cop_seen.add(key)
            cop_tiles.append({'w': w, 's': s, 'e': e, 'n': n})

    try:
        # Source 1: cached central-directory indices of remote Zenodo ZIPs.
        idx_dir = Path('data/austria_processor/zenodo_zip_index')
        if idx_dir.exists():
            for fp in idx_dir.iterdir():
                if fp.suffix != '.json':
                    continue
                try:
                    raw = json.loads(fp.read_text())
                except Exception:
                    continue
                for name in raw:
                    base = name.replace('.npz', '')
                    parts = base.split('_')
                    product = parts[0]
                    floats = []
                    for p in parts[1:]:
                        try:
                            floats.append(float(p))
                        except ValueError:
                            break
                    if len(floats) < 4:
                        continue
                    s, w, n, e = floats[0], floats[1], floats[2], floats[3]
                    _add(product, w, s, e, n)

        # Source 2: locally-present tile .meta.json sidecars (tiles physically
        # on this disk right now — useful when ZIP indices are stale or absent).
        for sub, default_product in (
            ('copernicus_tiles', 'copernicus'),
            ('hansen_tiles', 'hansen'),
        ):
            d = Path('data/austria_processor') / sub
            if not d.exists():
                continue
            for fp in d.glob('*.meta.json'):
                npz = fp.with_suffix('').with_suffix('.npz')
                if not npz.exists():
                    continue
                try:
                    meta = json.loads(fp.read_text())
                    w = float(meta['w']); s = float(meta['s'])
                    e = float(meta['e']); n = float(meta['n'])
                except Exception:
                    continue
                product = meta.get('product') or default_product
                bucket = 'hansen' if (default_product == 'hansen' or product == 'hansen') else 'copernicus'
                _add(bucket, w, s, e, n)

        return jsonify({'copernicus': cop_tiles, 'hansen': han_tiles})
    except Exception as e:
        return jsonify({'error': str(e), 'copernicus': cop_tiles, 'hansen': han_tiles})


@app.route('/api/v1/processing/manifest')
def processing_manifest():
    """Return the Zenodo manifest for the processor."""
    manifest_path = Path('data/austria_processor/zenodo_manifest.json')
    if not manifest_path.exists():
        return jsonify({'entries': {}, 'count': 0})
    try:
        from zenodo_client import DEFAULT_TOKEN
        data = json.loads(manifest_path.read_text())
        entries = data.get('entries', {})
        # Add authenticated download URLs for the dashboard
        for e in entries.values():
            depo_id = e.get('depo_id')
            fn = e.get('filename')
            if depo_id and fn:
                e['download_url'] = f'https://zenodo.org/api/records/{depo_id}/draft/files/{fn}/content?access_token={DEFAULT_TOKEN}'
        return jsonify({
            'count': len(entries),
            'entries': entries,
            'total_size_bytes': sum(e.get('size', 0) for e in entries.values()),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/processing/manifest/<key>', methods=['DELETE'])
def processing_manifest_delete(key):
    """Delete a single entry from the Zenodo manifest."""
    manifest_path = Path('data/austria_processor/zenodo_manifest.json')
    if not manifest_path.exists():
        return jsonify({'error': 'no manifest'}), 404
    try:
        data = json.loads(manifest_path.read_text())
        entries = data.get('entries', data)
        if key not in entries:
            return jsonify({'error': f'{key} not found'}), 404
        del entries[key]
        import tempfile as _tf
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = _tf.mkstemp(dir=manifest_path.parent, suffix='.tmp', prefix='.manifest_')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump({'entries': entries}, f, indent=2, sort_keys=True)
            os.replace(tmp, manifest_path)
        except BaseException:
            try: os.unlink(tmp)
            except OSError: pass
            raise
        # Add tombstone so sync thread doesn't re-merge stale entries from peers
        import datetime as _dt
        _MANIFEST_TOMBSTONES[key] = _dt.datetime.utcnow().isoformat()
        _tombstone_path.write_text(json.dumps(_MANIFEST_TOMBSTONES, indent=2))
        return jsonify({'deleted': key, 'remaining': len(entries)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === SECTION: Bandwidth & Peer Director ===

import peer_director as pd

@app.route('/api/v1/bandwidth')
def api_bandwidth():
    """Return bandwidth usage for this VM (from vnstat)."""
    bw = pd.get_local_bandwidth()
    bw['budget_gb'] = pd.load_peers_config().get('budget_gb', pd.BANDWIDTH_BUDGET_GB)
    return jsonify(bw)


@app.route('/api/v1/director/status')
def director_status():
    """Full director status: mode, active peer, bandwidth per peer."""
    d = pd.get_director()
    return jsonify(d.get_status())


@app.route('/api/v1/director/mode', methods=['POST'])
def director_set_mode():
    """Set director mode: auto, manual, paused."""
    mode = request.args.get('mode') or (request.json or {}).get('mode', '')
    if mode not in ('auto', 'manual', 'paused'):
        return jsonify({'error': 'mode must be auto, manual, or paused'}), 400
    d = pd.get_director()
    d.set_mode(mode)
    return jsonify({'mode': mode})


@app.route('/api/v1/director/activate', methods=['POST'])
def director_activate_peer():
    """Manually activate a specific peer."""
    peer_id = request.args.get('peer') or (request.json or {}).get('peer', '')
    if not peer_id:
        return jsonify({'error': 'peer parameter required'}), 400
    d = pd.get_director()
    cfg = pd.load_peers_config()
    peer = pd.get_peer_by_id(cfg, peer_id)
    if not peer:
        return jsonify({'error': f'Unknown peer: {peer_id}'}), 404

    # Stop current active peer if different
    state = pd.load_director_state()
    old_active = state.get('active_peer')
    if old_active and old_active != peer_id:
        old_peer = pd.get_peer_by_id(cfg, old_active)
        if old_peer:
            pd.stop_peer_processor(old_peer.get('url'))

    # CRITICAL: persist new active_peer + mode to disk BEFORE starting the
    # processor. start_peer_processor can take 10-15s (systemd_fallback);
    # during that window the director loop in another gunicorn worker would
    # read stale state, see at3 as 'non-active running', and kill it.
    d.set_mode('manual')
    d.set_active_peer(peer_id)

    # Start new peer
    result = pd.start_peer_processor(peer.get('url'))
    return jsonify({'status': 'activated', 'peer': peer_id, 'start_result': result})


@app.route('/api/v1/director/stop', methods=['POST'])
def director_stop_all():
    """Stop all peers and pause the director."""
    d = pd.get_director()
    cfg = pd.load_peers_config()
    results = {}
    for peer in cfg.get('peers', []):
        ps = pd.get_peer_status(peer.get('url'))
        if ps.get('state') in ('running', 'processing'):
            results[peer['id']] = pd.stop_peer_processor(peer.get('url'))
    d.set_active_peer(None)
    d.set_mode('paused')
    return jsonify({'status': 'all_stopped', 'results': results})


@app.route('/api/v1/director/peers', methods=['GET', 'POST'])
def director_peers_config():
    """GET: return peers config. POST: update it."""
    if request.method == 'GET':
        return jsonify(pd.load_peers_config())
    # POST: update config
    new_cfg = request.json
    if not new_cfg or 'peers' not in new_cfg:
        return jsonify({'error': 'peers array required'}), 400
    pd.save_peers_config(new_cfg)
    d = pd.get_director()
    d.reload_config()
    return jsonify({'status': 'updated', 'config': new_cfg})


@app.route('/api/v1/director/peers/add', methods=['POST'])
def director_add_peer():
    """Add a new peer dynamically.

    Body JSON:
      id: peer identifier (e.g. 'at3')
      url: peer URL (e.g. 'https://srtm-lidar-at3.exe.xyz:8000')
      enabled: bool (default true)
    """
    data = request.get_json(silent=True) or {}
    peer_id = data.get('id', '').strip()
    peer_url = data.get('url', '').strip().rstrip('/')
    enabled = data.get('enabled', True)
    if not peer_id or not peer_url:
        return jsonify({'error': 'id and url are required'}), 400
    if not peer_url.startswith('https://'):
        return jsonify({'error': 'url must start with https://'}), 400

    cfg = pd.load_peers_config()
    # Check for duplicate
    existing_ids = {p['id'] for p in cfg.get('peers', [])}
    existing_urls = {p.get('url') for p in cfg.get('peers', [])}
    if peer_id in existing_ids:
        return jsonify({'error': f'Peer {peer_id} already exists'}), 409
    if peer_url in existing_urls:
        return jsonify({'error': f'URL {peer_url} already registered'}), 409

    # Test connectivity
    online = False
    try:
        r = requests.get(peer_url + '/api/v1/info', timeout=10)
        online = r.ok
    except Exception:
        pass

    # Push self-identity to the new peer so it knows its own id, URL and
    # who the current director is. This is the source-of-truth for
    # director_ha (watchdog comparisons, takeover gating, announce flips).
    try:
        import director_ha as dha
        my_url = dha.self_url() or request.host_url.rstrip('/')
        admin_tok = ''
        try:
            admin_tok = Path('data/admin_token').read_text().strip()
        except Exception:
            pass
        requests.post(
            peer_url + '/api/v1/director/identity',
            json={'id': peer_id, 'url': peer_url, 'director_url': my_url},
            headers=({'X-Admin-Token': admin_tok} if admin_tok else {}),
            timeout=10,
        )
    except Exception as _e:
        log.warning('Could not push identity to new peer %s: %s', peer_id, _e)

    from datetime import datetime as _dtnow, timezone as _tznow
    cfg['peers'].append({
        'id': peer_id, 'url': peer_url, 'enabled': enabled,
        # Warmup hold: record first-seen timestamp so the director
        # holds off on frontier promotion for ~5 min.
        'first_seen': _dtnow.now(_tznow.utc).isoformat(),
    })
    pd.save_peers_config(cfg)
    d = pd.get_director()
    d.reload_config()

    # Also add to peer_urls.txt for the sync thread
    peer_urls_path = Path('data/austria_processor/peer_urls.txt')
    current_urls = set()
    if peer_urls_path.exists():
        current_urls = {u.strip() for u in peer_urls_path.read_text().splitlines() if u.strip()}
    if peer_url not in current_urls:
        current_urls.add(peer_url)
        peer_urls_path.write_text('\n'.join(sorted(current_urls)) + '\n')

    return jsonify({
        'status': 'added',
        'peer': {'id': peer_id, 'url': peer_url, 'enabled': enabled, 'online': online},
        'total_peers': len(cfg['peers']),
    })


@app.route('/api/v1/processing/cache_misses', methods=['GET'])
def processing_cache_misses_get():
    """Return the recorded cache-miss KG list (director-side store)."""
    p = Path('data/austria_processor/cache_miss_kgs.json')
    if not p.exists():
        return jsonify({})
    try:
        return jsonify(json.loads(p.read_text()))
    except Exception:
        return jsonify({})


@app.route('/api/v1/processing/cache_misses', methods=['POST'])
def processing_cache_misses_post():
    """Record a cache-miss KG.

    Body JSON: {kg_code, peer_id?, bbox?, tile_info?}
    Stored in data/austria_processor/cache_miss_kgs.json keyed by KG code.
    Future cache-only orchestration skips it until the strip's manifest
    fingerprint changes (i.e. new tiles uploaded).
    """
    data = request.get_json(silent=True) or {}
    kg = (data.get('kg_code') or '').strip()
    if not kg:
        return jsonify({'error': 'kg_code required'}), 400
    d = pd.get_director()
    entry = d.record_cache_miss(
        kg, peer_id=data.get('peer_id', '?'),
        bbox=data.get('bbox'), tile_info=data.get('tile_info', ''),
    )
    return jsonify({'status': 'recorded', 'kg_code': kg, 'entry': entry})


@app.route('/api/v1/processing/cache_misses/<kg>', methods=['DELETE'])
def processing_cache_misses_delete(kg):
    """Clear a cache-miss entry (e.g. after manual cache push)."""
    p = Path('data/austria_processor/cache_miss_kgs.json')
    if not p.exists():
        return jsonify({'status': 'not_found'}), 404
    try:
        misses = json.loads(p.read_text())
    except Exception:
        return jsonify({'error': 'corrupt store'}), 500
    if kg in misses:
        del misses[kg]
        p.write_text(json.dumps(misses, indent=2))
        # Drop director cache so next tick refreshes
        d = pd.get_director()
        with d._lock:
            d.state.pop('_cache_ready_cache', None)
        return jsonify({'status': 'cleared', 'kg_code': kg})
    return jsonify({'status': 'not_found'}), 404


@app.route('/api/v1/director/peers/<peer_id>/cooldown', methods=['POST'])
def director_cooldown_peer(peer_id):
    """Set a manual cooldown on a peer.

    Body JSON: {"hours": 24} (default 24). Sets ``not_before`` to now+hours
    and stops the peer's processor. The director will skip it until the
    cooldown lifts. Pass ``hours=0`` to clear an existing cooldown.
    """
    body = request.get_json(silent=True) or {}
    try:
        hours = float(body.get('hours', 24))
    except (TypeError, ValueError):
        return jsonify({'error': 'hours must be a number'}), 400
    cfg = pd.load_peers_config()
    peer = pd.get_peer_by_id(cfg, peer_id)
    if not peer:
        return jsonify({'error': f'Peer {peer_id} not found'}), 404
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    if hours <= 0:
        peer.pop('not_before', None)
        nb_iso = None
    else:
        nb_iso = (_dt.now(_tz.utc) + _td(hours=hours)).isoformat()
        peer['not_before'] = nb_iso
    # Cooldown always releases any KG hold so the substitute can pick it up.
    released_kg = peer.pop('reserved_kg', None)
    pd.save_peers_config(cfg)
    # Stop the peer's processor so the cooldown takes immediate effect.
    stopped = False
    if hours > 0:
        try:
            pd.safely_stop_peer(peer.get('url'), peer_id)
            stopped = True
        except Exception as e:
            log.warning('cooldown: stop %s failed: %s', peer_id, e)
    d = pd.get_director()
    d.reload_config()
    return jsonify({'status': 'ok', 'peer_id': peer_id,
                    'not_before': nb_iso, 'stopped': stopped, 'hours': hours,
                    'released_kg': released_kg})


@app.route('/api/v1/director/peers/<peer_id>/update', methods=['POST'])
def director_update_peer(peer_id):
    """Trigger git pull + restart on a single peer.

    Body JSON: {"graceful": true|false}. ``graceful=true`` (default)
    defers the restart until the peer finishes the current KG.
    ``graceful=false`` restarts immediately, killing any mid-KG work.
    """
    body = request.get_json(silent=True) or {}
    graceful = body.get('graceful', True) is not False
    cfg = pd.load_peers_config()
    peer = pd.get_peer_by_id(cfg, peer_id)
    if not peer:
        return jsonify({'error': f'Peer {peer_id} not found'}), 404
    url = peer.get('url')
    if not url:
        # Primary updates locally via /api/v1/admin/update — proxy in-process.
        return admin_update()  # type: ignore[name-defined]
    res = pd.trigger_peer_update(url, graceful=graceful)
    return jsonify({'status': 'ok', 'peer_id': peer_id,
                    'graceful': graceful, 'result': res})


@app.route('/api/v1/director/peers/<peer_id>/release_hold', methods=['POST'])
def director_release_hold(peer_id):
    """Release a peer's reserved KG hold.

    Peers acquire a ``reserved_kg`` when they cool down mid-KG so the
    substitute does not steal it. This endpoint clears that reservation
    and (optionally) the associated ``not_before`` cooldown so the KG
    re-enters the normal queue.

    Body JSON: {"clear_cooldown": true}
    """
    body = request.get_json(silent=True) or {}
    clear_cd = bool(body.get('clear_cooldown', True))
    cfg = pd.load_peers_config()
    peer = pd.get_peer_by_id(cfg, peer_id)
    if not peer:
        return jsonify({'error': f'Peer {peer_id} not found'}), 404
    released_kg = peer.pop('reserved_kg', None)
    cleared = False
    if clear_cd and peer.get('not_before'):
        peer.pop('not_before', None)
        cleared = True
    pd.save_peers_config(cfg)
    pd.get_director().reload_config()
    return jsonify({'status': 'ok', 'peer_id': peer_id,
                    'released_kg': released_kg,
                    'cooldown_cleared': cleared})


@app.route('/api/v1/director/peers/<peer_id>/pin', methods=['POST'])
def director_pin_peer(peer_id):
    """Set or clear a peer's pinned_role.

    Body JSON: {"pinned_role": "frontier"|"cache_only"|"idle"|null}
    Pinning overrides automatic role selection. Setting null clears it.
    """
    data = request.get_json(silent=True) or {}
    role = data.get('pinned_role')
    if role is not None:
        role = str(role).strip().lower() or None
        if role and role not in ('frontier', 'cache_only', 'idle'):
            return jsonify({'error': 'pinned_role must be one of frontier, cache_only, idle, or null'}), 400
    cfg = pd.load_peers_config()
    found = None
    for p in cfg.get('peers', []):
        if p['id'] == peer_id:
            found = p
            break
    if not found:
        return jsonify({'error': f'Peer {peer_id} not found'}), 404
    if role is None:
        found.pop('pinned_role', None)
    else:
        found['pinned_role'] = role
    pd.save_peers_config(cfg)
    d = pd.get_director()
    d.reload_config()
    return jsonify({'status': 'updated', 'peer_id': peer_id,
                    'pinned_role': found.get('pinned_role')})


@app.route('/api/v1/director/peers/<peer_id>/cmd/<action>', methods=['POST'])
def director_peer_cmd(peer_id, action):
    """Proxy pause/resume/stop/start to a remote peer's processor.

    The dashboard talks to *this* primary (so X-Admin-Token + cookies
    work normally); we forward the request to the peer with the
    cluster admin token. Without this proxy, the dashboard's
    cross-origin POST to ``https://srtm-lidar-atN.exe.xyz:8000/api/v1/processing/stop``
    arrives without the X-Admin-Token (the JS fetch wrapper only
    injects it for same-origin URLs) and returns 401, which is what
    made "■ Stop" appear to do nothing.

    Action whitelist matches the peer's processing endpoints. Query
    string and JSON body are forwarded verbatim so flags like
    ``graceful=1`` work.
    """
    allowed = {'start', 'stop', 'pause', 'resume', 'single', 'retry'}
    if action not in allowed:
        return jsonify({'error': f'unknown action {action!r}'}), 400
    cfg = pd.load_peers_config()
    peer = pd.get_peer_by_id(cfg, peer_id)
    if not peer:
        return jsonify({'error': f'Peer {peer_id} not found'}), 404
    url = (peer.get('url') or '').rstrip('/')
    if not url:
        # Local primary: dispatch to the same-process route. The
        # dashboard already does this directly when meta.url is null,
        # but we accept it here too for symmetry.
        from flask import url_for
        try:
            from werkzeug.test import EnvironBuilder
            return app.full_dispatch_request()
        except Exception:
            return jsonify({'error': 'primary peer has no url; call /api/v1/processing/' + action + ' directly'}), 400
    target = url + '/api/v1/processing/' + action
    try:
        import requests as _req
        headers = {'X-Admin-Token': _current_admin_token() or ''}
        # Forward query string + JSON body verbatim.
        body = request.get_json(silent=True)
        r = _req.post(target,
                      params=request.args.to_dict(flat=True) or None,
                      json=body if body is not None else None,
                      headers=headers, timeout=20)
        try:
            return jsonify(r.json()), r.status_code
        except Exception:
            return (r.text, r.status_code, {'Content-Type': r.headers.get('Content-Type', 'text/plain')})
    except Exception as e:
        return jsonify({'error': f'proxy to {peer_id} failed: {e}'}), 502


@app.route('/api/v1/director/peers/<peer_id>', methods=['DELETE'])
def director_remove_peer(peer_id):
    """Remove a peer from the director.

    Stops the peer's processor and removes it from peers.json and peer_urls.txt.
    Cannot remove the primary peer (url=null).
    """
    d = pd.get_director()
    result = d.remove_peer(peer_id)
    if 'error' in result:
        return jsonify(result), 400 if 'Cannot remove' in result.get('error', '') else 404
    return jsonify(result)


@app.route('/api/v1/director/throttle', methods=['GET', 'POST'])
def director_throttle():
    """GET: current throttle state. POST: set and propagate throttle to all peers."""
    d = pd.get_director()
    if request.method == 'GET':
        local = Path('data/austria_processor/upload_throttle').exists()
        return jsonify({'throttle': local})
    data = request.get_json(silent=True) or {}
    enabled = data.get('throttle', True)
    results = d.propagate_throttle(enabled)
    return jsonify({'status': 'propagated', 'throttle': enabled, 'peers': results})


@app.route('/api/v1/director/peer_status', methods=['POST'])
def director_peer_status():
    """Peer pushes its own /processing/status payload here.

    Cached on the primary so the director loop reads from cache instead
    of polling 50 peers every tick. Optional fields:
      * peer_id (preferred) or peer (id fallback)
      * status: full status dict from /api/v1/processing/status
      * bandwidth: optional bandwidth dict to avoid a separate poll

    No body validation — we trust admin-token auth.
    """
    body = request.get_json(silent=True) or {}
    peer_id = (body.get('peer_id') or body.get('peer') or '').strip()
    if not peer_id:
        return jsonify({'error': 'peer_id required'}), 400
    status = body.get('status') or {}
    bw = body.get('bandwidth')
    pd.record_peer_push(peer_id, status, bandwidth=bw)
    return jsonify({'ok': True, 'next_push_in_s': pd.PEER_PUSH_INTERVAL})


@app.route('/api/v1/director/proxy/status')
def director_proxy_status():
    """Return the active peer's *live* processing state on top of the
    primary's locally-computed enrichment (manifest_* rate / ETA,
    tile_history, db_* totals, kgs_completed_per_day, total_kgs, ...).
    Without this overlay we lose all the cross-peer aggregates as soon
    as a remote peer becomes active."""
    d = pd.get_director()
    status = d.get_status()
    active_id = status.get('active_peer')
    # Start from the primary's enriched status so the dashboard keeps
    # its rate / ETA / DB totals / sparkline / tile history even when
    # a remote peer is the frontier.
    base = processing_status().get_json() or {}
    if not active_id:
        base['_director'] = status
        if not base.get('state'):
            base['state'] = 'no_active_peer'
        return jsonify(base)
    cfg = pd.load_peers_config()
    peer = pd.get_peer_by_id(cfg, active_id)
    if not peer:
        base['_director'] = status
        base['state'] = 'peer_not_found'
        return jsonify(base)
    ps = pd.get_peer_status(peer.get('url')) or {}
    # Overlay only the fields that describe the peer's live work.
    # Everything else (manifest_*, db_*, total_kgs, tile_history,
    # kgs_completed_per_day, system, manifest, ...) stays from the
    # primary's view.
    LIVE_KEYS = (
        'state', 'current_kg', 'current_step', 'step', 'step_detail',
        'step_started_at', 'step_detail_ts', 'step_times', 'step_issues',
        'started_at', 'last_kg_code', 'last_kg_seconds',
        'recent_log', 'failed_kgs', 'subtile',
        'instance', 'region', 'git_commit',
    )
    for k in LIVE_KEYS:
        if k in ps and ps[k] is not None:
            base[k] = ps[k]
    base['_director'] = status
    base['_active_peer_id'] = active_id
    base['_active_peer_url'] = peer.get('url')
    for p in status.get('peers', []):
        if p.get('id') == active_id:
            base['_bandwidth'] = p.get('bandwidth', {})
            break
    return jsonify(base)


@app.route('/api/v1/director/proxy/log')
def director_proxy_log():
    """Proxy a peer's processor log. Defaults to the active (frontier)
    peer; pass ?peer_id=<id> to fetch a specific peer (e.g. a cache-only
    worker)."""
    d = pd.get_director()
    state = d.state
    lines = int(request.args.get('lines', 80))
    target_id = request.args.get('peer_id') or state.get('active_peer')
    if not target_id:
        return jsonify({'lines': [], 'peer': None})
    cfg = pd.load_peers_config()
    peer = pd.get_peer_by_id(cfg, target_id)
    if not peer:
        return jsonify({'lines': [], 'peer': target_id})
    log_lines = pd.get_peer_log(peer.get('url'), lines)
    return jsonify({'lines': log_lines, 'peer': target_id})


_ALL_STATUS_CACHE = {'ts': 0.0, 'data': None, 'refreshing': False}
_ALL_STATUS_TTL = 8.0       # serve cached if newer
_ALL_STATUS_STALE = 60.0    # serve stale + bg refresh if newer
_ALL_STATUS_CACHE_FILE = Path('/tmp/srtm_all_status_cache.json')

def _all_status_load_disk():
    try:
        if _ALL_STATUS_CACHE_FILE.exists():
            d = json.loads(_ALL_STATUS_CACHE_FILE.read_text())
            _ALL_STATUS_CACHE['ts'] = float(d.get('cached_at', 0))
            _ALL_STATUS_CACHE['data'] = d
    except Exception:
        pass

def _all_status_save_disk(payload):
    try:
        tmp = _ALL_STATUS_CACHE_FILE.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload))
        tmp.replace(_ALL_STATUS_CACHE_FILE)
    except Exception:
        pass

def _all_status_compute():
    """Probe every running peer in parallel and return the payload.
    Extracted from the route so background refresh can call it."""
    from concurrent.futures import ThreadPoolExecutor
    import time as _time
    d = pd.get_director()
    status = d.get_status()
    cfg = pd.load_peers_config()
    targets = []
    for p in status.get('peers', []):
        pid = p.get('id')
        if not pid:
            continue
        is_active = p.get('is_active') or pid == status.get('active_peer')
        is_cache_only = bool(p.get('cache_only_run'))
        running = p.get('processor_state') in ('running', 'processing')
        # Include the active frontier, any other running frontier peers
        # (parallel-frontier mode), and all running cache-only peers.
        if not (is_active or running):
            continue
        # Skip peers that the director already knows are offline
        # (they hang the response while we wait for connect timeout).
        if not p.get('online', True):
            continue
        peer_cfg = pd.get_peer_by_id(cfg, pid)
        if not peer_cfg:
            continue
        targets.append((pid, peer_cfg.get('url'), bool(is_active), bool(is_cache_only)))
    out: list[dict] = []
    if targets:
        def _probe(t):
            pid, url, is_active, is_cache_only = t
            # Local primary: read directly, no HTTP round-trip.
            if not url:
                try:
                    ps = processing_status().get_json() or {}
                except Exception as e:
                    ps = {'state': 'unreachable', 'error': str(e)}
            else:
                try:
                    # Tight timeout: this endpoint feeds a fast carousel,
                    # slow/hung peers should drop out rather than block.
                    import requests as _req
                    r = _req.get(
                        url.rstrip('/') + '/api/v1/processing/status',
                        timeout=(2, 4),
                        headers=pd._admin_headers(),
                    )
                    r.raise_for_status()
                    ps = r.json() or {}
                except Exception as e:
                    ps = {'state': 'unreachable', 'error': str(e)}
            ps['_peer_id'] = pid
            ps['_peer_url'] = url
            ps['_is_active'] = is_active
            ps['_cache_only'] = is_cache_only
            return ps
        with ThreadPoolExecutor(max_workers=min(20, len(targets))) as pool:
            out = list(pool.map(_probe, targets))
    now = _time.time()
    payload = {'peers': out, 'active_peer': status.get('active_peer'),
               'cached_at': now}
    _ALL_STATUS_CACHE['ts'] = now
    _ALL_STATUS_CACHE['data'] = payload
    _all_status_save_disk(payload)
    return payload


@app.route('/api/v1/director/proxy/all_status')
def director_proxy_all_status():
    """Return processing status for *every* running peer (active frontier
    + all cache-only running peers). Used by process.html so the user
    can swipe through current-KG cards across the fleet. Peers are
    probed in parallel so the response stays under ~3s even with 20
    peers. Stale-while-refresh keeps it snappy."""
    import time as _time
    import threading as _th
    # In-memory cache may be cold (each gunicorn worker has its own);
    # fall back to a tiny on-disk cache so workers share results.
    if _ALL_STATUS_CACHE.get('data') is None:
        _all_status_load_disk()
    now = _time.time()
    cached = _ALL_STATUS_CACHE.get('data')
    age = now - _ALL_STATUS_CACHE.get('ts', 0)
    if cached is not None and age < _ALL_STATUS_TTL:
        return jsonify(cached)
    # Re-read from disk in case another worker just refreshed.
    _all_status_load_disk()
    cached = _ALL_STATUS_CACHE.get('data')
    age = now - _ALL_STATUS_CACHE.get('ts', 0)
    if cached is not None and age < _ALL_STATUS_TTL:
        return jsonify(cached)
    if cached is not None and age < _ALL_STATUS_STALE and \
       not _ALL_STATUS_CACHE.get('refreshing'):
        def _bg_refresh():
            try:
                _all_status_compute()
            except Exception:
                pass
            finally:
                _ALL_STATUS_CACHE['refreshing'] = False
        _ALL_STATUS_CACHE['refreshing'] = True
        _th.Thread(target=_bg_refresh, daemon=True).start()
        return jsonify(cached)
    return jsonify(_all_status_compute())


@app.route('/api/v1/director/proxy/combined_log')
def director_proxy_combined_log():
    """Return a merged ``recent_log`` from the active frontier peer plus
    all running cache-only peers, tagged with the source peer id. Used
    by process.html so the Live Log shows what every running peer is
    doing — not just the frontier."""
    d = pd.get_director()
    status = d.get_status()
    cfg = pd.load_peers_config()
    merged: list[dict] = []
    seen_ids: list[str] = []
    for p in status.get('peers', []):
        pid = p.get('id')
        if not pid:
            continue
        is_active = p.get('is_active') or pid == status.get('active_peer')
        is_cache_only = bool(p.get('cache_only_run'))
        running = p.get('processor_state') in ('running', 'processing')
        if not (is_active or (is_cache_only and running)):
            continue
        peer_cfg = pd.get_peer_by_id(cfg, pid)
        if not peer_cfg:
            continue
        ps = pd.get_peer_status(peer_cfg.get('url'))
        for entry in ps.get('recent_log', []) or []:
            e = dict(entry)
            e['peer'] = pid
            merged.append(e)
        seen_ids.append(pid)
    # Newest first
    merged.sort(key=lambda e: e.get('ts', ''), reverse=True)
    return jsonify({'log': merged[:300], 'peers': seen_ids})


@app.route('/api/v1/director/update_peers', methods=['POST'])
def director_update_peers():
    """Tell all remote peers to git pull and restart their web servers.
    Optional body: {"peer_id": "at3"} to update a single peer.
    """
    body = request.get_json(silent=True) or {}
    target_id = body.get('peer_id')
    skip_push = bool(body.get('skip_push'))
    # Default: graceful update (let peers finish current KG first).
    # Pass {"graceful": false} to force immediate restart.
    graceful = body.get('graceful', True) is not False
    cfg = pd.load_peers_config()

    # Ensure local commits are on origin before peers pull. Otherwise peers
    # report 'Already up to date.' against a stale remote and silently stay
    # behind the primary.
    push_info = {'attempted': False}
    if not skip_push:
        try:
            import subprocess as sp
            repo = str(Path(__file__).parent)
            branch = sp.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                            capture_output=True, text=True, timeout=5,
                            cwd=repo).stdout.strip() or 'main'
            local = sp.run(['git', 'rev-parse', branch],
                           capture_output=True, text=True, timeout=5,
                           cwd=repo).stdout.strip()
            sp.run(['git', 'fetch', 'origin', branch],
                   capture_output=True, text=True, timeout=20, cwd=repo)
            remote = sp.run(['git', 'rev-parse', f'origin/{branch}'],
                            capture_output=True, text=True, timeout=5,
                            cwd=repo).stdout.strip()
            push_info = {'attempted': True, 'branch': branch,
                         'local': local[:7], 'remote_before': remote[:7]}
            if local and remote and local != remote:
                # Check whether local is ahead (fast-forwardable on remote)
                ahead = sp.run(['git', 'rev-list', '--count',
                                f'origin/{branch}..{branch}'],
                               capture_output=True, text=True, timeout=5,
                               cwd=repo).stdout.strip()
                push_info['ahead'] = ahead
                push = sp.run(['git', 'push', 'origin', branch],
                              capture_output=True, text=True, timeout=60,
                              cwd=repo)
                push_info['push_rc'] = push.returncode
                push_info['push_out'] = (push.stdout + push.stderr).strip()[-500:]
                if push.returncode != 0:
                    return jsonify({
                        'error': 'git push failed; aborting peer update',
                        'push': push_info,
                        'hint': 'Resolve manually or pass {"skip_push": true} to force.',
                    }), 409
            else:
                push_info['push_rc'] = 0
                push_info['note'] = 'origin already up to date'
        except Exception as e:
            push_info['error'] = str(e)
            return jsonify({
                'error': f'pre-push check failed: {e}',
                'push': push_info,
                'hint': 'Pass {"skip_push": true} to update peers anyway.',
            }), 500

    # Push the cluster admin token to every peer first. Idempotent: if
    # the peer already has it, install_token is a no-op (current_token
    # check passes). If the peer has none (fresh deploy or pre-auth code
    # that just pulled), bootstrap-allowed first install seeds it. This
    # makes the dashboard "Update Peers" button self-bootstrapping for
    # the cluster auth rollout, even if the peer's web server has not
    # yet picked up the new auth code (the install_token endpoint is
    # part of the same release; peers running pre-release code simply
    # skip this step harmlessly with a 404).
    cluster_tok = _current_admin_token()
    token_results = {}
    for peer in cfg.get('peers', []):
        if not peer.get('url'):
            continue
        if target_id and peer['id'] != target_id:
            continue
        token_results[peer['id']] = pd.install_token_on_peer(
            peer['url'], cluster_tok)

    results = {}
    # Update peers in parallel with bounded concurrency. Sequential loops
    # don't scale to 150 peers; trigger_peer_update returns quickly
    # because the peer drops the connection on srv restart.
    targets = [p for p in cfg.get('peers', [])
               if p.get('url') and (not target_id or p['id'] == target_id)]
    if targets:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(20, len(targets))) as ex:
            futs = {ex.submit(pd.trigger_peer_update, p['url'], graceful): p['id']
                    for p in targets}
            for f in as_completed(futs):
                pid = futs[f]
                try:
                    results[pid] = f.result(timeout=60)
                except Exception as e:
                    results[pid] = {'error': str(e)}
    return jsonify({
        'results': results,
        'token_install': token_results,
        'push': push_info,
        'graceful': graceful,
    })


@app.route('/api/v1/admin/update', methods=['POST'])
def admin_update():
    """Git pull and restart the web server (called by director on peers).

    Optional body/query: graceful=1 → if a processor is running mid-KG,
    send a graceful stop (SIGTERM, no escalation) and defer the actual
    git-pull + srv restart until the processor has exited at the next KG
    boundary. Director will start it again on the next tick.
    """
    body = request.get_json(silent=True) or {}
    graceful = (request.args.get('graceful') in ('1', 'true', 'yes')
                or body.get('graceful') is True
                or body.get('after_kg') is True)
    try:
        import subprocess as sp
        repo = str(Path(__file__).parent)
        if graceful:
            # Are we mid-KG?  If so, send graceful stop and defer the
            # actual update to a background thread that polls until the
            # processor exits, then runs the same code path as below.
            try:
                proc_running = sp.run(
                    ['pgrep', '-f', 'austria_processor.py'],
                    capture_output=True, timeout=3,
                ).returncode == 0
            except Exception:
                proc_running = False
            if proc_running:
                # Ask the processor to stop after the current KG.
                try:
                    sp.run(['pkill', '-TERM', '-f', 'austria_processor.py'],
                           capture_output=True, text=True, timeout=5)
                except Exception:
                    pass
                import threading as _th, time as _t
                def _deferred_update():
                    deadline = _t.time() + 4 * 3600  # 4 h cap
                    while _t.time() < deadline:
                        try:
                            still = sp.run(
                                ['pgrep', '-f', 'austria_processor.py'],
                                capture_output=True, timeout=3,
                            ).returncode == 0
                        except Exception:
                            still = False
                        if not still:
                            break
                        _t.sleep(15)
                    # Now run the actual update + restart sequence.
                    try:
                        _safe_git_sync(repo, sp)
                    except Exception:
                        pass
                    sp.Popen(['sudo', 'systemctl', 'restart', 'srv'])
                _th.Thread(target=_deferred_update, daemon=True).start()
                return jsonify({
                    'status': 'graceful_update_scheduled',
                    'note': 'will git-pull + restart srv once processor exits at next KG boundary',
                })
            # No processor running → fall through to immediate path.
        # Reset any tracked files that have local modifications + pull, with
        # stale-lock cleanup, serialization, and bumped timeouts. See
        # _safe_git_sync() docstring.
        try:
            pull = _safe_git_sync(repo, sp)
        except _GitBusy as _gb:
            return jsonify({'error': 'another update in progress', 'detail': str(_gb)}), 409
        except Exception as _ge:
            return jsonify({'error': f'git sync failed: {_ge}'}), 500
        # Ensure traceroute is available (needed for region detection)
        if sp.run(['which', 'traceroute'], capture_output=True).returncode != 0:
            sp.run(['sudo', 'apt-get', 'install', '-y', '-q', 'traceroute'],
                   capture_output=True, timeout=60)
        # Fix ownership of data/ tree. Earlier processor runs (before the
        # uid=exedev fix) wrote files as root, which the now-uid=exedev
        # subprocess can't read or rewrite. Symptoms: "Permission denied:
        # 'data/austria_processor/zenodo_manifest.json'", "Corrupt BEV
        # cache <hash>.npz: Permission denied". Idempotent — only chowns
        # files actually owned by another uid.
        try:
            data_dir = str(Path(repo) / 'data')
            sp.run(['sudo', 'chown', '-R', 'exedev:exedev', data_dir],
                   capture_output=True, timeout=60)
        except Exception as _e:
            log.warning('data/ chown failed: %s', _e)
        # Install / refresh systemd watchdog units (idempotent).  These
        # restart srv if /api/v1/ping stops answering — protects against
        # the wedged-gunicorn failure mode.
        try:
            from pathlib import Path as _P
            ws = _P(repo) / 'srv-watchdog.service'
            wt = _P(repo) / 'srv-watchdog.timer'
            wsh = _P(repo) / 'srv_watchdog.sh'
            if ws.exists() and wt.exists() and wsh.exists():
                sp.run(['chmod', '+x', str(wsh)], capture_output=True, timeout=5)
                sp.run(['sudo', 'cp', str(ws), '/etc/systemd/system/srv-watchdog.service'],
                       capture_output=True, timeout=10)
                sp.run(['sudo', 'cp', str(wt), '/etc/systemd/system/srv-watchdog.timer'],
                       capture_output=True, timeout=10)
                sp.run(['sudo', 'systemctl', 'daemon-reload'],
                       capture_output=True, timeout=10)
                sp.run(['sudo', 'systemctl', 'enable', '--now', 'srv-watchdog.timer'],
                       capture_output=True, timeout=10)
        except Exception as _e:
            log.warning('srv-watchdog install failed: %s', _e)
        # Defer restart if processor is mid-upload (avoid broken manifest entries)
        current_step = ''
        try:
            step_data = json.loads((Path(__file__).parent / 'data/austria_processor/current_step.json').read_text())
            current_step = step_data.get('step', '')
        except Exception:
            pass
        upload_steps = {'upload_full_gpkg', 'upload'}
        if current_step in upload_steps:
            # Schedule restart after a short delay to let upload finish
            def _delayed_restart():
                import time as _t
                _t.sleep(90)
                sp.Popen(['sudo', 'systemctl', 'restart', 'srv'])
            import threading as _th
            _th.Thread(target=_delayed_restart, daemon=True).start()
            return jsonify({
                'status': 'updated',
                'git': pull.stdout.strip(),
                'git_err': pull.stderr.strip(),
                'restart': 'deferred_upload',
            })
        # Restart gunicorn in background so this response gets out first
        sp.Popen(['sudo', 'systemctl', 'restart', 'srv'])
        return jsonify({
            'status': 'updated',
            'git': pull.stdout.strip(),
            'git_err': pull.stderr.strip(),
            'restart': 'immediate',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/director/restart_peer', methods=['POST'])
def director_restart_peer():
    """Force-restart the processor on a specific peer (stop + start).

    Body: {"peer_id": "at3"}  (or omit for active peer)

    Calls processing/stop (kills parent + children) then processing/start.
    """
    body = request.get_json(silent=True) or {}
    peer_id = body.get('peer_id')
    cfg = pd.load_peers_config()
    peer_url = None

    if peer_id:
        for p in cfg.get('peers', []):
            if p['id'] == peer_id:
                peer_url = p.get('url')
                break
        else:
            return jsonify({'error': f'Unknown peer: {peer_id}'}), 404
    else:
        # Default to active peer
        d = pd.get_director()
        active_id = d.state.get('active_peer')
        if not active_id:
            return jsonify({'error': 'No active peer'}), 404
        peer_id = active_id
        for p in cfg.get('peers', []):
            if p['id'] == active_id:
                peer_url = p.get('url')
                break

    # Stop (kills entire process group including stuck subprocesses)
    stop_result = pd.stop_peer_processor(peer_url)

    # Wait for processes to die
    time.sleep(3)

    # Start
    start_result = pd.start_peer_processor(peer_url)

    return jsonify({
        'peer_id': peer_id,
        'stop': stop_result,
        'start': start_result,
    })


# === SECTION: Zenodo upload mutex ===
#
# Serialise Zenodo write operations across peers — the API token is
# shared, and concurrent PUTs to the same draft deposition fail.  All
# peers acquire a lease before uploading and renew it via heartbeat.
# Stale leases (no heartbeat for >120s) are auto-released.

_ZENODO_LOCK = {
    'holder': None,        # peer id (str) or None
    'token': None,         # uuid string returned to the holder
    'acquired_at': 0.0,    # epoch s
    'last_heartbeat': 0.0, # epoch s
    'purpose': None,       # 'kg_upload' | 'cache_flush' | ...
    'kg': None,            # optional KG code for diagnostics
}
_ZENODO_LOCK_LOCK = threading.Lock()
_ZENODO_LOCK_TTL = 120.0  # seconds without heartbeat → stale
# Persistence: surviving an srv restart matters because peers heartbeat
# every 30s and a fresh in-memory dict would 410 every active lease,
# aborting in-flight uploads. We dump on every mutation and restore at
# import time. Only the most recent (still-fresh) lease is restored.
_ZENODO_LOCK_FILE = Path('data/austria_processor/zenodo_lock_state.json')


def _zenodo_lock_persist() -> None:
    try:
        _ZENODO_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _ZENODO_LOCK_FILE.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(_ZENODO_LOCK))
        tmp.replace(_ZENODO_LOCK_FILE)
    except Exception as e:
        log.warning('Zenodo lock persist failed: %s', e)


def _zenodo_lock_restore() -> None:
    try:
        if not _ZENODO_LOCK_FILE.exists():
            return
        d = json.loads(_ZENODO_LOCK_FILE.read_text())
        # Skip stale leases on restore.
        last_hb = float(d.get('last_heartbeat') or 0.0)
        if last_hb and (time.time() - last_hb) <= _ZENODO_LOCK_TTL:
            _ZENODO_LOCK.update(d)
            log.info('Zenodo lock restored: holder=%s purpose=%s idle=%.1fs',
                     d.get('holder'), d.get('purpose'),
                     time.time() - last_hb)
    except Exception as e:
        log.warning('Zenodo lock restore failed: %s', e)


_zenodo_lock_restore()


def _zenodo_lock_is_stale(now: float) -> bool:
    if _ZENODO_LOCK['holder'] is None:
        return False
    return (now - _ZENODO_LOCK['last_heartbeat']) > _ZENODO_LOCK_TTL


# === SECTION: Zenodo lock broker proxy ===
#
# When the standalone broker on :8001 is up, gunicorn's lock routes
# transparently proxy to it. Peers can keep their existing
# ZENODO_LOCK_URL pointing at :8000 (the public exe.dev proxy) and
# still benefit from the broker's slow-path isolation — their lock
# traffic still goes via gunicorn, but gunicorn just forwards it,
# which is much cheaper than holding the lock state itself.
# Falls back to in-process state when the broker is unreachable.

_BROKER_BASE = 'http://127.0.0.1:8001'
_BROKER_HEALTHY_AT = 0.0
_BROKER_HEALTHY_TTL = 30.0   # re-probe at most every 30s
_BROKER_HEALTHY = False
_BROKER_LOCK = threading.Lock()


def _broker_alive() -> bool:
    """Cached liveness probe for the standalone broker."""
    global _BROKER_HEALTHY, _BROKER_HEALTHY_AT
    now = time.time()
    with _BROKER_LOCK:
        if (now - _BROKER_HEALTHY_AT) < _BROKER_HEALTHY_TTL:
            return _BROKER_HEALTHY
    try:
        import requests as _req
        r = _req.get(_BROKER_BASE + '/api/v1/zenodo/lock', timeout=0.5)
        ok = r.status_code == 200
    except Exception:
        ok = False
    with _BROKER_LOCK:
        _BROKER_HEALTHY = ok
        _BROKER_HEALTHY_AT = time.time()
    return ok


def _broker_proxy(method: str, path_suffix: str = '') -> tuple | None:
    """Forward the current request to the broker. Returns Flask response
    tuple ``(body, status)`` or None when the broker is unreachable
    (caller should fall back to local state).
    """
    if not _broker_alive():
        return None
    try:
        import requests as _req
        url = _BROKER_BASE + '/api/v1/zenodo/lock' + path_suffix
        # Forward auth header so the broker sees the same token check.
        hdrs = {}
        tok = request.headers.get('X-Admin-Token')
        if tok:
            hdrs['X-Admin-Token'] = tok
        body = request.get_data() or None
        if body is not None:
            hdrs['Content-Type'] = request.headers.get(
                'Content-Type', 'application/json')
        r = _req.request(method, url, data=body, headers=hdrs, timeout=3.0)
        # Mirror broker's status + JSON body.
        try:
            return jsonify(r.json()), r.status_code
        except Exception:
            return r.text, r.status_code
    except Exception as e:
        log.debug('Zenodo lock broker proxy failed: %s', e)
        with _BROKER_LOCK:
            global _BROKER_HEALTHY, _BROKER_HEALTHY_AT
            _BROKER_HEALTHY = False
            _BROKER_HEALTHY_AT = time.time()
        return None


@app.route('/api/v1/zenodo/lock', methods=['POST'])
def zenodo_lock_acquire():
    """Acquire the global Zenodo upload lease.

    Body: {peer: <id>, purpose: 'kg_upload'|'cache_flush'|..., kg?: <code>}
    Returns 200 + {token, ttl_s} on success, 423 + {holder, age_s} on conflict.
    The caller must POST /api/v1/zenodo/lock/heartbeat at least every TTL/2 s.
    """
    proxied = _broker_proxy('POST')
    if proxied is not None:
        return proxied
    import uuid as _uuid
    body = request.get_json(silent=True) or {}
    peer = str(body.get('peer') or 'anon')
    purpose = str(body.get('purpose') or 'unknown')
    kg = body.get('kg')
    now = time.time()
    with _ZENODO_LOCK_LOCK:
        if _ZENODO_LOCK['holder'] is not None and not _zenodo_lock_is_stale(now):
            # Same peer asking again — reuse the lease (idempotent acquire)
            if _ZENODO_LOCK['holder'] == peer:
                _ZENODO_LOCK['last_heartbeat'] = now
                return jsonify({
                    'token': _ZENODO_LOCK['token'],
                    'ttl_s': _ZENODO_LOCK_TTL,
                    'reacquired': True,
                })
            return jsonify({
                'error': 'locked',
                'holder': _ZENODO_LOCK['holder'],
                'purpose': _ZENODO_LOCK['purpose'],
                'kg': _ZENODO_LOCK['kg'],
                'age_s': round(now - _ZENODO_LOCK['acquired_at'], 1),
                'idle_s': round(now - _ZENODO_LOCK['last_heartbeat'], 1),
            }), 423
        # Free or stale — grant the lease
        if _ZENODO_LOCK['holder'] is not None:
            log.warning('Zenodo lock: stale holder=%s reclaimed by %s',
                        _ZENODO_LOCK['holder'], peer)
        token = _uuid.uuid4().hex
        _ZENODO_LOCK.update({
            'holder': peer, 'token': token,
            'acquired_at': now, 'last_heartbeat': now,
            'purpose': purpose, 'kg': kg,
        })
        _zenodo_lock_persist()
        return jsonify({'token': token, 'ttl_s': _ZENODO_LOCK_TTL})


@app.route('/api/v1/zenodo/lock/heartbeat', methods=['POST'])
def zenodo_lock_heartbeat():
    """Renew an active lease.  Body: {token: <uuid>}."""
    proxied = _broker_proxy('POST', '/heartbeat')
    if proxied is not None:
        return proxied
    body = request.get_json(silent=True) or {}
    token = body.get('token')
    now = time.time()
    with _ZENODO_LOCK_LOCK:
        if _ZENODO_LOCK['holder'] is None or _ZENODO_LOCK['token'] != token:
            return jsonify({'error': 'no_lease'}), 410
        _ZENODO_LOCK['last_heartbeat'] = now
        _zenodo_lock_persist()
        return jsonify({'ok': True, 'ttl_s': _ZENODO_LOCK_TTL,
                        'age_s': round(now - _ZENODO_LOCK['acquired_at'], 1)})


@app.route('/api/v1/zenodo/lock', methods=['DELETE'])
def zenodo_lock_release():
    """Release the lease.  Body: {token: <uuid>}."""
    proxied = _broker_proxy('DELETE')
    if proxied is not None:
        return proxied
    body = request.get_json(silent=True) or {}
    token = body.get('token')
    with _ZENODO_LOCK_LOCK:
        if _ZENODO_LOCK['holder'] is None or _ZENODO_LOCK['token'] != token:
            return jsonify({'error': 'no_lease'}), 410
        log.info('Zenodo lock: released by %s (purpose=%s, held %.1fs)',
                 _ZENODO_LOCK['holder'], _ZENODO_LOCK['purpose'],
                 time.time() - _ZENODO_LOCK['acquired_at'])
        _ZENODO_LOCK.update({
            'holder': None, 'token': None,
            'acquired_at': 0.0, 'last_heartbeat': 0.0,
            'purpose': None, 'kg': None,
        })
        _zenodo_lock_persist()
        return jsonify({'ok': True})


@app.route('/api/v1/zenodo/lock', methods=['GET'])
def zenodo_lock_status():
    """Inspect current lease (no auth)."""
    proxied = _broker_proxy('GET')
    if proxied is not None:
        return proxied
    now = time.time()
    with _ZENODO_LOCK_LOCK:
        if _ZENODO_LOCK['holder'] is None:
            return jsonify({'free': True})
        return jsonify({
            'free': False,
            'holder': _ZENODO_LOCK['holder'],
            'purpose': _ZENODO_LOCK['purpose'],
            'kg': _ZENODO_LOCK['kg'],
            'age_s': round(now - _ZENODO_LOCK['acquired_at'], 1),
            'idle_s': round(now - _ZENODO_LOCK['last_heartbeat'], 1),
            'stale': _zenodo_lock_is_stale(now),
            'ttl_s': _ZENODO_LOCK_TTL,
        })


@app.route('/api/v1/admin/flush_tiles', methods=['POST'])
def admin_flush_tiles():
    """Force-upload local Copernicus + Hansen tiles to Zenodo.

    Idempotent. Skips on cache-only peers (COPERNICUS_FORBIDDEN=1).
    Returns immediately; the actual upload runs in the background so
    the request doesn't tie up the gunicorn worker for minutes.
    """
    import threading as _th
    def _do():
        try:
            from austria_processor import flush_tile_cache_to_zenodo
            flush_tile_cache_to_zenodo(force=True)
        except Exception as _e:
            log.warning('flush_tiles background: %s', _e)
    _th.Thread(target=_do, daemon=True).start()
    return jsonify({'status': 'started'})


@app.route('/api/v1/admin/restart_processor', methods=['POST'])
def admin_restart_processor():
    """Restart the austria_processor via systemd (called by director as fallback)."""
    try:
        import subprocess as sp
        sp.run(['sudo', 'systemctl', 'restart', 'austria_processor'],
               capture_output=True, text=True, timeout=15)
        time.sleep(2)
        status = sp.run(['systemctl', 'is-active', 'austria_processor'],
                        capture_output=True, text=True, timeout=5)
        return jsonify({
            'status': 'restarted',
            'service_state': status.stdout.strip(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/admin/run_backfill', methods=['POST'])
def admin_run_backfill():
    """Run the per-parcel top10 backfill in a detached background process.

    POST body (JSON, optional):
        {"kg": "49006-south"}     # only this KG
        {"dry_run": true}            # validate without uploading
        {"no_upload": true}          # write JSON locally but skip Zenodo
        {"force": true}              # re-run even if parcels already enriched
        {"allow_warnings": true}     # upload despite validation issues

    Designed to be invoked from the primary against a peer (e.g. at2) so
    backfill bandwidth doesn't come out of the primary's monthly quota.
    """
    import subprocess as sp
    body = request.get_json(silent=True) or {}
    args = ['python3', '-u', 'backfill_parcel_top10.py']
    if body.get('kg'):
        args += ['--kg', str(body['kg'])]
    if body.get('dry_run'):
        args.append('--dry-run')
    if body.get('no_upload'):
        args.append('--no-upload')
    if body.get('force'):
        args.append('--force')
    if body.get('allow_warnings'):
        args.append('--allow-warnings')
    log_path = Path('/tmp/backfill.log')
    pid_path = Path('/tmp/backfill.pid')
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
            try:
                os.kill(old_pid, 0)
                return jsonify({'status': 'already_running', 'pid': old_pid,
                                'log': str(log_path)}), 409
            except OSError:
                pid_path.unlink()  # stale
        except Exception:
            pid_path.unlink(missing_ok=True)
    import datetime as _dt
    log_fp = open(log_path, 'a')
    log_fp.write(f'\n=== {_dt.datetime.now(_dt.timezone.utc).isoformat()} starting: {" ".join(args)} ===\n')
    log_fp.flush()
    # Launch via `sudo systemd-run --scope` so the backfill lives in its
    # own systemd scope and survives gunicorn (`srv.service`) restarts.
    # Without this, the subprocess inherits srv's cgroup and dies whenever
    # /api/v1/admin/update bounces srv.
    unit_name = f'backfill-parcel-top10-{int(time.time())}'
    full_args = ['sudo', '-n', 'systemd-run', '--scope', '--quiet',
                 '--unit', unit_name, '--'] + args
    try:
        p = sp.Popen(
            full_args, cwd=str(Path(__file__).parent),
            stdout=log_fp, stderr=sp.STDOUT,
            start_new_session=True,
        )
        log.info('Backfill launched via systemd-run scope unit=%s', unit_name)
    except Exception as e:
        log.warning('systemd-run scope failed (%s); falling back to plain Popen', e)
        p = sp.Popen(
            args, cwd=str(Path(__file__).parent),
            stdout=log_fp, stderr=sp.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(p.pid))
    return jsonify({'status': 'started', 'pid': p.pid, 'log': str(log_path),
                    'args': args})


@app.route('/api/v1/admin/backfill_kill', methods=['POST'])
def admin_backfill_kill():
    """Kill a running backfill subprocess."""
    pid_path = Path('/tmp/backfill.pid')
    if not pid_path.exists():
        return jsonify({'status': 'not_running'})
    try:
        pid = int(pid_path.read_text().strip())
        try:
            os.kill(pid, 15)  # SIGTERM
            time.sleep(1)
            try:
                os.kill(pid, 0)
                os.kill(pid, 9)  # SIGKILL
            except OSError:
                pass
        except OSError:
            pass
        pid_path.unlink(missing_ok=True)
        return jsonify({'status': 'killed', 'pid': pid})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/admin/backfill_status', methods=['GET'])
def admin_backfill_status():
    """Return PID + tail of /tmp/backfill.log for the running/last backfill."""
    pid_path = Path('/tmp/backfill.pid')
    log_path = Path('/tmp/backfill.log')
    running = False
    pid = None
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            try:
                os.kill(pid, 0)
                running = True
            except OSError:
                running = False
        except Exception:
            pass
    tail = ''
    if log_path.exists():
        try:
            with open(log_path, 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 32_000))
                tail = f.read().decode('utf-8', errors='replace')
        except Exception as e:
            tail = f'(log read failed: {e})'
    return jsonify({'running': running, 'pid': pid, 'log_tail': tail})


@app.route('/api/v1/admin/proc_env', methods=['GET'])
def admin_proc_env():
    """Diagnostic: dump environ + cmdline for the running processor.
    Used to debug whether director-set assignment env vars actually
    reach the subprocess. Returns a small filtered set of keys.
    """
    import subprocess as sp
    info = {}
    try:
        out = sp.check_output(['pgrep', '-af', 'austria_processor.py'],
                              text=True, timeout=3).strip().splitlines()
        info['pgrep'] = out
    except Exception as e:
        info['pgrep'] = f'err: {e}'
        out = []
    rows = []
    for line in out:
        try:
            pid_s, *_ = line.split(None, 1)
            pid = int(pid_s)
        except Exception:
            continue
        row = {'pid': pid}
        try:
            with open(f'/proc/{pid}/cmdline', 'rb') as f:
                row['cmdline'] = f.read().replace(b'\0', b' ').decode('utf-8', 'ignore').strip()
        except Exception as e:
            row['cmdline_err'] = str(e)
        try:
            st = open(f'/proc/{pid}/status').read()
            for ln in st.splitlines():
                if ln.startswith(('Uid:', 'Name:', 'PPid:', 'State:')):
                    row[ln.split(':')[0].lower()] = ln.split(':',1)[1].strip()
        except Exception as e:
            row['status_err'] = str(e)
        try:
            with open(f'/proc/{pid}/environ', 'rb') as f:
                envs = f.read().split(b'\0')
            interesting = {}
            for kv in envs:
                if not kv or b'=' not in kv:
                    continue
                k, v = kv.split(b'=', 1)
                k = k.decode('utf-8', 'ignore')
                if k in ('COPERNICUS_CRED_INDICES', 'KG_LAT_STRIP_FILTER',
                        'KG_CELL_FILTER',
                        'ZENODO_LOCK_URL', 'COPERNICUS_FORBIDDEN', 'HOME',
                        'PYTHONUNBUFFERED', 'USER'):
                    interesting[k] = v.decode('utf-8', 'ignore')
            row['env'] = interesting
        except Exception as e:
            row['env_err'] = str(e)
        rows.append(row)
    info['procs'] = rows
    return jsonify(info)


@app.route('/api/v1/admin/diskstat', methods=['GET'])
def admin_diskstat():
    """Per-VM disk + role status. Used to diagnose disk pressure on
    demoted peers and verify the role-data eviction policy."""
    import shutil
    out = {}
    try:
        u = shutil.disk_usage('/')
        out['disk_total_gb'] = round(u.total / (1024 ** 3), 1)
        out['disk_free_gb'] = round(u.free / (1024 ** 3), 1)
        out['disk_used_pct'] = round(100 * u.used / u.total, 1)
    except Exception as e:
        out['disk_err'] = str(e)
    paths = {
        'json': 'data/austria_processor/json',
        'search_index': 'data/search_index.db',
        'shares': 'data/shares',
        'tile_checkpoints': 'data/austria_processor/tile_checkpoints',
        'zenodo_zip_index': 'data/austria_processor/zenodo_zip_index',
        'logs': 'data/austria_processor/logs',
        'cop_cache': 'data/austria_processor/copernicus_tiles',
        'hansen_cache': 'data/austria_processor/hansen_tiles',
        'bev_cache': 'data/austria_processor/bev_tile_cache',
        'ortho_cache': 'data/austria_processor/ortho_tile_cache',
        'segment_results': '/tmp/segment_results',
    }
    sizes = {}
    for k, p in paths.items():
        pp = Path(p)
        try:
            if not pp.exists():
                sizes[k] = None; continue
            if pp.is_file():
                sizes[k] = round(pp.stat().st_size / (1024 ** 2), 1)
            else:
                tot = sum(f.stat().st_size for f in pp.rglob('*') if f.is_file())
                sizes[k] = round(tot / (1024 ** 2), 1)
        except Exception as e:
            sizes[k] = f'err:{e}'
    out['sizes_mb'] = sizes
    try:
        out['role'] = _role_data_eviction_tick()
    except Exception as e:
        out['role_err'] = str(e)
    return jsonify(out)


@app.route('/api/v1/admin/role_evict', methods=['POST'])
def admin_role_evict():
    """Force-run a role-data eviction tick. If keep_role and force=true,
    purge anyway (use only on standalone peers being decommissioned)."""
    body = request.get_json(silent=True) or {}
    force = bool(body.get('force', False))
    if force:
        # Bypass keep_role check by stamping a far-past demotion marker.
        try:
            _ROLE_DEMOTED_AT_FILE.parent.mkdir(parents=True, exist_ok=True)
            _ROLE_DEMOTED_AT_FILE.write_text(str(int(time.time()) - ROLE_EVICT_GRACE_SECONDS - 1))
        except Exception as e:
            return jsonify({'ok': False, 'err': str(e)}), 500
        # Override keep check for one tick by directly running the purge body.
        # Simpler: call tick. If force pretends keep=False, the existing tick
        # would still see keep=True. So bypass: do the purge inline.
        json_dir = Path('data/austria_processor/json')
        idx_path = Path('data/search_index.db')
        n = 0; bytes_freed = 0
        for f in list(json_dir.glob('*.json')) if json_dir.exists() else []:
            try:
                bytes_freed += f.stat().st_size
                f.unlink(); n += 1
            except Exception:
                pass
        for sfx in ('', '-wal', '-shm', '-journal'):
            p = Path(f'{idx_path}{sfx}')
            try:
                if p.exists():
                    bytes_freed += p.stat().st_size; p.unlink()
            except Exception:
                pass
        return jsonify({'ok': True, 'forced': True, 'purged_count': n,
                        'purged_mb': round(bytes_freed / (1024 ** 2), 1)})
    return jsonify({'ok': True, 'tick': _role_data_eviction_tick()})


@app.route('/api/v1/admin/disable_autostart', methods=['POST'])
def admin_disable_autostart():
    """Disable austria_processor systemd auto-start.

    Called by the director to ensure peers don't restart their processor
    independently via systemd. The director manages processor lifecycle.
    """
    try:
        import subprocess as sp
        # Disable auto-start on boot (don't stop — let current KG finish)
        sp.run(['sudo', 'systemctl', 'disable', 'austria_processor'],
               capture_output=True, text=True, timeout=5)
        # Mask prevents manual start too — use unmask to re-enable
        # sp.run(['sudo', 'systemctl', 'mask', 'austria_processor'], ...)
        return jsonify({'status': 'disabled'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/admin/clear_stepped_down', methods=['POST'])
def admin_clear_stepped_down():
    """Clear the ``stepped_down`` flag so this peer is eligible to be
    re-promoted to director (e.g. via auto-handback to primary).
    """
    try:
        from director_ha import STEPPED_DOWN_FLAG
        existed = STEPPED_DOWN_FLAG.exists()
        try:
            STEPPED_DOWN_FLAG.unlink()
        except FileNotFoundError:
            pass
        return jsonify({'cleared': existed})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === SECTION: Director high-availability (heartbeat / shadow / handover) ===

@app.route('/api/v1/director/heartbeat', methods=['GET'])
def director_heartbeat():
    """Liveness ping. Public (no admin token) so peers can probe cheaply.

    Returns 200 only when this VM holds ``is_director``. Otherwise 410 so
    callers know to look elsewhere (peer watchdog will then check whom it
    thinks the director is and re-evaluate).
    """
    import director_ha as dha
    if not dha.IS_DIRECTOR_FLAG.exists():
        return jsonify({'is_director': False, 'self': dha.load_self()}), 410
    # Liveness check: the loop must actually be running. A dead loop with
    # IS_DIRECTOR_FLAG still on disk means the cluster is unmanaged —
    # report 410 so peers fail over to a healthier shadow instead of
    # trusting our flag forever.
    #
    # Use ON-DISK signals, not the in-process singleton. With 2 gunicorn
    # workers only 1 holds the director loop (fcntl lock); the other
    # worker's singleton is initialised but never started, so a naive
    # `_running` check would return 410 on ~50% of heartbeats, which
    # makes peer watchdogs trigger spurious takeovers (this caused
    # ~hourly cascading failovers on the cluster). Instead, trust
    # director_state.json freshness — it's rewritten every ~30s by
    # whichever worker holds the loop, regardless of which worker is
    # serving this heartbeat.
    try:
        d = pd.get_director()
        thr = getattr(d, '_thread', None)
        in_proc_running = bool(getattr(d, '_running', False)) and bool(
            thr and thr.is_alive())
    except Exception:
        in_proc_running = False
    state_fresh = False
    state_age = None
    try:
        st_path = pd.DIRECTOR_STATE
        if st_path.exists():
            state_age = time.time() - st_path.stat().st_mtime
            # Loop saves state every ~30s; allow 3x slack for slow ticks.
            state_fresh = state_age < 120.0
    except Exception:
        pass
    running = in_proc_running or state_fresh
    if not running:
        return jsonify({
            'is_director': True,
            'running': False,
            'note': 'flag set but loop not running',
            'state_age_s': state_age,
            'self': dha.load_self(),
        }), 410
    state = {}
    try:
        state = pd.load_director_state()
    except Exception:
        pass
    return jsonify({
        'is_director': True,
        'running': running,
        'self': dha.load_self(),
        'shadow_peer': state.get('shadow_peer'),
        'active_peer': state.get('active_peer'),
        'mode': state.get('mode'),
        'ts': time.time(),
    })


@app.route('/api/v1/director/snapshot', methods=['GET', 'PUT'])
def director_snapshot():
    """GET: build & return a snapshot (director only).
    PUT: stage a snapshot from the director (shadow only).
    """
    import director_ha as dha
    if request.method == 'GET':
        if not dha.IS_DIRECTOR_FLAG.exists():
            return jsonify({'error': 'not_director'}), 410
        return jsonify(dha.build_snapshot())
    body = request.get_json(silent=True) or {}
    if not isinstance(body, dict) or '_meta' not in body:
        return jsonify({'error': 'invalid_snapshot'}), 400
    dha.stage_snapshot(body)
    return jsonify({'status': 'staged', 'self': dha.load_self()})


@app.route('/api/v1/director/announce', methods=['POST'])
def director_announce():
    """Inbound announce — a new director claims authority."""
    import director_ha as dha
    body = request.get_json(silent=True) or {}
    return jsonify(dha.accept_announce(body))


@app.route('/api/v1/director/step_down', methods=['POST'])
def director_step_down():
    """Old director is asked to step down (idempotent)."""
    import director_ha as dha
    body = request.get_json(silent=True) or {}
    return jsonify(dha.step_down(body))


@app.route('/api/v1/director/takeover', methods=['POST'])
def director_takeover():
    """Inbound takeover (manual handover or shadow promotion).

    Body: ``{snapshot, prev_director_url, reason}``. If snapshot provided,
    install it inline; otherwise promote the staged shadow snapshot.
    """
    import director_ha as dha
    body = request.get_json(silent=True) or {}
    snap = body.get('snapshot')
    return jsonify(dha._do_takeover(
        reason=body.get('reason', 'inbound_takeover'),
        prev_director_url=body.get('prev_director_url'),
        snapshot_inline=snap if isinstance(snap, dict) else None,
    ))


@app.route('/api/v1/director/handover', methods=['POST'])
def director_handover():
    """Manual handover initiated by the current director.

    Body: ``{to: <peer_id>}`` (or ?to=<peer_id>).
    """
    import director_ha as dha
    if not dha.IS_DIRECTOR_FLAG.exists():
        return jsonify({'error': 'not_director'}), 409
    target = (request.args.get('to') or
              (request.get_json(silent=True) or {}).get('to', '')).strip()
    if not target:
        return jsonify({'error': 'to=<peer_id> required'}), 400
    cfg = pd.load_peers_config()
    peer = pd.get_peer_by_id(cfg, target)
    if not peer or not peer.get('url'):
        return jsonify({'error': f'unknown peer: {target}'}), 404
    return jsonify(dha.do_handover(target, peer['url']))


@app.route('/api/v1/director/identity', methods=['GET', 'POST'])
def director_identity():
    """GET/POST self identity (id, url, director_url)."""
    import director_ha as dha
    if request.method == 'GET':
        return jsonify({**dha.load_self(),
                        'is_director': dha.IS_DIRECTOR_FLAG.exists(),
                        'stepped_down': dha.STEPPED_DOWN_FLAG.exists(),
                        'watchdog': dha.watchdog_state()})
    body = request.get_json(silent=True) or {}
    cur = dha.load_self()
    # SECURITY/SANITY: only the receiver itself can authoritatively know its
    # own id and url. We therefore NEVER override `id`/`url` from a remote
    # POST: the director may have stale peers.json mappings that point an
    # id at a different host, and accepting them would corrupt our self.json
    # (which has happened during cascaded director handovers). The receiver
    # derives id from hostname via dha.load_self() and trusts that.
    # Only `director_url` is accepted from a remote POST — that's the whole
    # point of the identity broadcast: tell the peer who its director is.
    incoming_id = body.get('id')
    incoming_url = body.get('url')
    rejected = {}
    if incoming_id and incoming_id != cur.get('id'):
        rejected['id'] = {'incoming': incoming_id, 'self': cur.get('id')}
    if incoming_url and incoming_url != cur.get('url'):
        rejected['url'] = {'incoming': incoming_url, 'self': cur.get('url')}
    if rejected:
        log.warning('director_identity: refusing to override self id/url '
                    'from remote POST: %s', rejected)
    # Local writes (loopback / same-process) are still permitted to set
    # id/url so the on-box bootstrap can fix things up.
    # NOTE: must use _request_is_loopback() — exe.dev's HTTPS proxy makes
    # remote_addr=127.0.0.1 for *every* request, so a naive check would let
    # any peer overwrite our id/url. _request_is_loopback() also requires
    # X-Forwarded-For to be absent, which the proxy always sets for remote
    # callers. Without this guard, cascading takeovers on at26 corrupted
    # the primary's self.json (id flipped primary→at8→at3).
    if _request_is_loopback():
        for k in ('id', 'url'):
            if k in body and body[k] is not None:
                cur[k] = body[k]
    if 'director_url' in body:
        cur['director_url'] = body['director_url']
    dha.save_self(cur)
    if 'director_url' in body:
        dha.set_director_url(body['director_url'])
    out = dict(cur)
    if rejected:
        out['_rejected'] = rejected
    return jsonify(out)


# Start the director background thread — only on the primary instance.
# The director loop actively starts/stops processors on peers, so only one
# instance should run it. Enabled by the flag file data/austria_processor/is_director.
def _start_director():
    time.sleep(3)  # let Flask boot first
    flag = Path('data/austria_processor/is_director')
    stepped_down = Path('data/austria_processor/stepped_down')
    if stepped_down.exists() and flag.exists():
        # Conflict: both flags set. stepped_down wins (safer). The peer
        # was demoted; don't re-promote it on cold restart.
        try:
            flag.unlink()
            log.warning('Both is_director and stepped_down present; honouring '
                        'stepped_down and removing is_director.')
        except Exception:
            pass
    if not flag.exists():
        log.info('Not a director instance (no %s) — director loop disabled', flag)
        return
    d = pd.get_director()
    d.start()
    # Best-effort bulk identity broadcast — tells every peer who the
    # current director is (and what their own peer_id is) so their
    # watchdog can start probing /director/heartbeat. Self-healed each
    # tick, but a one-shot at startup gets the cluster aware faster.
    try:
        d._broadcast_identity_to_all_peers()  # noqa: SLF001
    except Exception as _e:
        log.warning('startup identity broadcast failed: %s', _e)
threading.Thread(target=_start_director, daemon=True).start()

# Start the director high-availability watchdog on every VM. On the
# director it's mostly a no-op; on peers it pings the director every 30s
# and if the shadow misses 3 in a row, the shadow promotes itself.
try:
    import director_ha as dha
    import socket as _sock
    # ALWAYS heal self.json on startup from the local hostname — it is
    # the only authoritative source of identity for this VM. Past
    # cascading handovers corrupted self.json with the wrong id/url
    # (e.g. at37 ended up with id='at44'). The hostname does not lie.
    _self = dha.load_self()
    _hn = _sock.gethostname().split('.')[0]
    if _hn.startswith('srtm-lidar-'):
        _suffix = _hn[len('srtm-lidar-'):]
        _correct_id = 'primary' if _suffix == 'at' else _suffix
        _correct_url = f'https://{_hn}.exe.xyz:8000'
        _changed = False
        if _self.get('id') != _correct_id:
            log.warning('self.json id mismatch: was %r, healing to %r',
                        _self.get('id'), _correct_id)
            _self['id'] = _correct_id
            _changed = True
        if _self.get('url') != _correct_url:
            log.warning('self.json url mismatch: was %r, healing to %r',
                        _self.get('url'), _correct_url)
            _self['url'] = _correct_url
            _changed = True
        if dha.IS_DIRECTOR_FLAG.exists() and _self.get('director_url'):
            # The director should not have a director_url pointer.
            _self['director_url'] = None
            _changed = True
        if _changed:
            dha.save_self(_self)
    dha.start_watchdog()
except Exception as _e:
    log.warning('director_ha watchdog disabled: %s', _e)


# === SECTION: Search index API endpoints ===

@app.route('/api/v1/index/status')
def index_status():
    """Search index status and statistics."""
    try:
        idx = si.get_index()
        return jsonify(idx.stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/processing/kg_outlines')
def api_kg_outlines():
    """Return processed-KG bboxes for the dashboard map. Cheap (one
    SQL query against the search index).

    Query params:
      processed=1     only KGs marked processed (default)
      processed=0     all KGs (warning: large response)
    """
    only_processed = request.args.get('processed', '1') != '0'
    try:
        conn = si.get_index()._conn()
        sql = ('SELECT kg_code, kg_name, min_lon, min_lat, max_lon, max_lat '
               'FROM kg WHERE min_lon IS NOT NULL')
        if only_processed:
            sql += ' AND processed=1'
        rows = conn.execute(sql).fetchall()
        out = [{'code': r[0], 'name': r[1] or '',
                'bbox': [r[2], r[3], r[4], r[5]]} for r in rows]
        return jsonify({'kgs': out, 'count': len(out)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/index/rebuild', methods=['POST'])
def index_rebuild():
    """Rebuild the search index from scratch."""
    try:
        idx = si.get_index()
        idx.build()
        return jsonify(idx.stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/kg/<kg_code>')
def api_kg(kg_code):
    """Return KG info from index. If JSON exists locally, serves the full JSON.
    Otherwise returns index data with Zenodo download links."""
    # If full JSON exists locally, serve it directly (backwards-compatible)
    json_path = Path(f'data/austria_processor/json/{kg_code}.json')
    if json_path.exists() and not request.args.get('index_only'):
        return send_file(str(json_path), mimetype='application/json')
    # Fall back to search index
    try:
        idx = si.get_index()
        result = idx.query_kg(kg_code)
        if result:
            return jsonify(result)
    except Exception as e:
        log.warning('index query_kg %s: %s', kg_code, e)
    return jsonify({'error': f'KG {kg_code} not found'}), 404


@app.route('/api/v1/parcel/<path:parcel_id>')
def api_parcel(parcel_id):
    """Look up a parcel. Returns KG summary + parcel detail if available."""
    if '-' not in parcel_id:
        return jsonify({'error': 'Invalid parcel_id format, expected KGCODE-GNR'}), 400
    try:
        idx = si.get_index()
        result = idx.query_parcel(parcel_id)
        if result:
            return jsonify(result)
    except Exception as e:
        log.warning('index query_parcel %s: %s', parcel_id, e)
    return jsonify({'error': f'Parcel {parcel_id} not found'}), 404


# === SECTION: GPKG detail lazy-load endpoints ===

def _parse_bbox_param(s):
    """Parse 'w,s,e,n' bbox string to tuple, or None."""
    if not s:
        return None
    try:
        parts = [float(x) for x in s.split(',')]
        if len(parts) == 4:
            return tuple(parts)
    except (ValueError, TypeError):
        pass
    return None


@app.route('/api/v1/kg/<kg_code>/buildings')
def api_kg_buildings(kg_code):
    """Height-enriched building footprints from the light GPKG.
    Lazy-loads from Zenodo if not cached locally.

    Params: bbox=w,s,e,n  limit=N  offset=N
    """
    try:
        idx = si.get_index()
        bbox = _parse_bbox_param(request.args.get('bbox'))
        limit = min(int(request.args.get('limit', 500)), 5000)
        offset = int(request.args.get('offset', 0))
        result = idx.query_buildings(kg_code, bbox=bbox, limit=limit, offset=offset)
        if result is None:
            return jsonify({'error': f'No building data available for KG {kg_code}'}), 404
        return jsonify(result)
    except Exception as e:
        log.warning('api_kg_buildings %s: %s', kg_code, e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/kg/<kg_code>/new_buildings')
def api_kg_new_buildings(kg_code):
    """Detected new buildings from the light GPKG.

    Params: bbox=w,s,e,n  limit=N  offset=N
    """
    try:
        idx = si.get_index()
        bbox = _parse_bbox_param(request.args.get('bbox'))
        limit = min(int(request.args.get('limit', 500)), 5000)
        offset = int(request.args.get('offset', 0))
        result = idx.query_new_buildings_detail(kg_code, bbox=bbox, limit=limit, offset=offset)
        if result is None:
            return jsonify({'error': f'No new building data available for KG {kg_code}'}), 404
        return jsonify(result)
    except Exception as e:
        log.warning('api_kg_new_buildings %s: %s', kg_code, e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/kg/<kg_code>/infrastructure')
def api_kg_infrastructure(kg_code):
    """Detected infrastructure from the light GPKG.

    Params: bbox=w,s,e,n  limit=N  offset=N
    """
    try:
        idx = si.get_index()
        bbox = _parse_bbox_param(request.args.get('bbox'))
        limit = min(int(request.args.get('limit', 500)), 5000)
        offset = int(request.args.get('offset', 0))
        result = idx.query_infrastructure_detail(kg_code, bbox=bbox, limit=limit, offset=offset)
        if result is None:
            return jsonify({'error': f'No infrastructure data available for KG {kg_code}'}), 404
        return jsonify(result)
    except Exception as e:
        log.warning('api_kg_infrastructure %s: %s', kg_code, e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/buildings/search')
def api_buildings_search():
    """Search building records from the index (fast, no GPKG).

    Params: kg=CODE  min_height=N  max_height=N  min_stories=N  max_stories=N
            roof_type=flat|pitched  min_area=N  max_area=N  limit=N  offset=N
    """
    try:
        idx = si.get_index()
        result = idx.query_buildings_index(
            kg_code=request.args.get('kg'),
            min_height=_float_or_none(request.args.get('min_height')),
            max_height=_float_or_none(request.args.get('max_height')),
            min_stories=_int_or_none(request.args.get('min_stories')),
            max_stories=_int_or_none(request.args.get('max_stories')),
            roof_type=request.args.get('roof_type'),
            min_area=_float_or_none(request.args.get('min_area')),
            max_area=_float_or_none(request.args.get('max_area')),
            limit=min(int(request.args.get('limit', 500)), 5000),
            offset=int(request.args.get('offset', 0)),
        )
        return jsonify(result)
    except Exception as e:
        log.warning('api_buildings_search: %s', e)
        return jsonify({'error': str(e)}), 500


def _float_or_none(v):
    if v is None: return None
    try: return float(v)
    except (ValueError, TypeError): return None

def _int_or_none(v):
    if v is None: return None
    try: return int(v)
    except (ValueError, TypeError): return None


@app.route('/api/v1/kg/<kg_code>/segments')
def api_kg_segments(kg_code):
    """Segment polygons from the light GPKG.

    Params: bbox=w,s,e,n  type=t1,t2,...  limit=N  offset=N
    """
    try:
        idx = si.get_index()
        bbox = _parse_bbox_param(request.args.get('bbox'))
        type_str = request.args.get('type', '')
        type_filter = [t.strip() for t in type_str.split(',') if t.strip()] or None
        limit = min(int(request.args.get('limit', 500)), 5000)
        offset = int(request.args.get('offset', 0))
        result = idx.query_segments_detail(kg_code, bbox=bbox, type_filter=type_filter,
                                           limit=limit, offset=offset)
        if result is None:
            return jsonify({'error': f'No segment data available for KG {kg_code}'}), 404
        return jsonify(result)
    except Exception as e:
        log.warning('api_kg_segments %s: %s', kg_code, e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/kg/<kg_code>/layers')
def api_kg_layers(kg_code):
    """List available vector layers in a KG's GPKG.

    Params: variant=light|full (default: light)
    """
    try:
        idx = si.get_index()
        variant = request.args.get('variant', 'light')
        if variant not in ('light', 'full'):
            return jsonify({'error': 'variant must be light or full'}), 400
        result = idx.gpkg_layers(kg_code, variant=variant)
        if result is None:
            return jsonify({'error': f'No GPKG available for KG {kg_code}'}), 404
        return jsonify({'kg_code': kg_code, 'variant': variant, 'layers': result})
    except Exception as e:
        log.warning('api_kg_layers %s: %s', kg_code, e)
        return jsonify({'error': str(e)}), 500


# === SECTION: GET query param parsers for compound/parcel filters ===

def _parse_compound_from_args(args):
    """Parse compound filter dict from flat GET query params.

    Supports all query_compound() filter keys as query params.
    Special handling:
      - bbox=w,s,e,n → [float, float, float, float]
      - aspect=S,SW,W → ["S","SW","W"]
      - type_filter=tree:0.8:800 (repeatable) → type_filters list
      - landcover_filter=grass:1300:0.1 (repeatable) → landcover_filters list
      - All min_*/max_* numeric params parsed as float/int
    """
    filters = {}

    # bbox
    if 'bbox' in args:
        try:
            filters['bbox'] = [float(x) for x in args['bbox'].split(',')]
        except ValueError:
            pass

    # String params (exact match)
    for k in ('state', 'district', 'gemeinde', 'dominant_type', 'phenology',
              'terrain_class', 'quality_grade', 'sort', 'sort_dir'):
        if k in args:
            filters[k] = args[k]

    # aspect — comma-separated list
    if 'aspect' in args:
        filters['aspect'] = [x.strip() for x in args['aspect'].split(',') if x.strip()]

    # All numeric range params (min_*/max_*)
    _numeric_keys = [
        'min_slope', 'max_slope', 'min_roughness', 'min_elevation', 'max_elevation',
        'min_elevation_min', 'max_elevation_min', 'min_elevation_max', 'max_elevation_max',
        'min_elevation_range', 'min_steepness_max', 'min_tri', 'max_tri',
        'min_total_area', 'max_total_area', 'min_parcels', 'max_parcels',
        'min_segments', 'max_segments',
        'min_buildings', 'max_buildings', 'min_new_buildings', 'min_infrastructure',
        'min_building_height', 'max_building_height',
        'min_building_max_height', 'max_building_max_height',
        'min_building_stories', 'max_building_stories',
        'min_building_stories_max', 'max_building_stories_max',
        'min_building_pitched_pct', 'max_building_pitched_pct',
        'min_building_footprint', 'max_building_footprint',
        'min_new_building_footprint', 'min_new_building_height', 'min_new_building_stories',
        'min_building_height_coverage',
        'min_tree_count', 'min_tree_height', 'min_tree_canopy_sqm', 'min_tree_volume',
        'min_ndvi', 'max_ndvi', 'min_vegetated_fraction', 'max_vegetated_fraction',
        'min_shannon_diversity',
        'min_ndvi_amplitude', 'min_ndvi_harm_mean', 'max_ndvi_harm_mean',
        'min_ndvi_phase', 'max_ndvi_phase',
        'min_sar_vv', 'max_sar_vv', 'min_sar_vh', 'max_sar_vh',
        'min_dtm_change', 'max_dtm_change', 'min_volume_change', 'max_volume_change',
        'min_changed_segments', 'min_disturbed_volume', 'min_temporal_stability',
        'min_confidence', 'min_rf_confidence', 'max_diverged_pct',
        'max_rf_diverged_count', 'min_rf_classified_pct', 'min_quality_score',
    ]
    for k in _numeric_keys:
        if k in args:
            try:
                v = float(args[k])
                filters[k] = int(v) if v == int(v) and 'pct' not in k and 'fraction' not in k and 'confidence' not in k and 'slope' not in k and 'roughness' not in k and 'elevation' not in k and 'height' not in k and 'ndvi' not in k and 'sar' not in k and 'tri' not in k and 'stability' not in k and 'score' not in k and 'change' not in k and 'volume' not in k and 'canopy' not in k and 'diversity' not in k and 'amplitude' not in k and 'phase' not in k and 'steepness' not in k and 'area' not in k and 'footprint' not in k and 'stories' not in k else v
            except ValueError:
                pass

    # type_filter — repeatable: type_filter=tree:0.8:800&type_filter=grass:0.8:1300
    tf_raw = args.getlist('type_filter')
    if tf_raw:
        type_filters = []
        for raw in tf_raw:
            parts = raw.split(':')
            tf = {'type': parts[0]}
            if len(parts) > 1 and parts[1]:
                try: tf['min_confidence'] = float(parts[1])
                except ValueError: pass
            if len(parts) > 2 and parts[2]:
                try: tf['min_area_sqm'] = float(parts[2])
                except ValueError: pass
            type_filters.append(tf)
        filters['type_filters'] = type_filters

    # landcover_filter — repeatable: landcover_filter=grass:1300:0.1
    lf_raw = args.getlist('landcover_filter')
    if lf_raw:
        landcover_filters = []
        for raw in lf_raw:
            parts = raw.split(':')
            lf = {'type': parts[0]}
            if len(parts) > 1 and parts[1]:
                try: lf['min_area_sqm'] = float(parts[1])
                except ValueError: pass
            if len(parts) > 2 and parts[2]:
                try: lf['min_fraction'] = float(parts[2])
                except ValueError: pass
            if len(parts) > 3 and parts[3]:
                try: lf['min_height_mean'] = float(parts[3])
                except ValueError: pass
            if len(parts) > 4 and parts[4]:
                try: lf['max_height_mean'] = float(parts[4])
                except ValueError: pass
            landcover_filters.append(lf)
        filters['landcover_filters'] = landcover_filters

    return filters


def _parse_parcel_filters_from_args(args):
    """Parse parcel_filters dict from GET query params with pf_ prefix.

    All parcel filter params use a 'pf_' prefix to avoid collision with compound filters.
    Examples:
      pf_aspect=E,SE → aspect: ["E","SE"]
      pf_terrain_class=level → terrain_class: "level"
      pf_min_vegetated_fraction=0.5 → min_vegetated_fraction: 0.5
      pf_types=tree,grass → types: ["tree","grass"]
      pf_type_confidence=tree:0.7:500 (repeatable)
      pf_sort=conservation_score, pf_sort_dir=desc
    """
    pf = {}

    # String params
    for k in ('terrain_class', 'sort', 'sort_dir', 'cadastre_landuse', 'roof_type', 'dominant_type'):
        pk = f'pf_{k}'
        if pk in args:
            pf[k] = args[pk]

    # Bool params
    for k in ('is_vegetated', 'cadastre_has_buildings'):
        pk = f'pf_{k}'
        if pk in args:
            pf[k] = args[pk].lower() in ('true', '1', 'yes')

    # Comma-separated list params
    for k in ('aspect', 'types'):
        pk = f'pf_{k}'
        if pk in args:
            pf[k] = [x.strip() for x in args[pk].split(',') if x.strip()]

    # Numeric params
    _pf_numeric = [
        'min_vegetated_fraction', 'max_vegetated_fraction',
        'min_forested_fraction', 'max_forested_fraction',
        'min_elevation', 'max_elevation',
        'min_type_fraction', 'min_ndsm_max', 'max_ndsm_max',
        'min_parcel_area', 'max_parcel_area',
        'min_slope', 'max_slope', 'min_tri', 'max_tri',
        'min_confidence', 'min_rf_confidence',
        'cadastre_min_area', 'cadastre_max_area',
        'min_stories', 'max_stories',
        'min_hansen_recent_5yr', 'max_hansen_recent_5yr',
        'min_hansen_total', 'max_hansen_total',
    ]
    _pf_int = {'min_stories', 'max_stories', 'min_hansen_recent_5yr',
               'max_hansen_recent_5yr', 'min_hansen_total', 'max_hansen_total'}
    for k in _pf_numeric:
        pk = f'pf_{k}'
        if pk in args:
            try:
                pf[k] = int(args[pk]) if k in _pf_int else float(args[pk])
            except ValueError:
                pass

    # pf_type_confidence — repeatable: pf_type_confidence=tree:0.7:500
    tc_raw = args.getlist('pf_type_confidence')
    if tc_raw:
        type_confidence = []
        for raw in tc_raw:
            parts = raw.split(':')
            tc = {'type': parts[0]}
            if len(parts) > 1 and parts[1]:
                try: tc['min_confidence'] = float(parts[1])
                except ValueError: pass
            if len(parts) > 2 and parts[2]:
                try: tc['min_area_sqm'] = float(parts[2])
                except ValueError: pass
            if len(parts) > 3 and parts[3]:
                try: tc['min_rf_confidence'] = float(parts[3])
                except ValueError: pass
            type_confidence.append(tc)
        pf['type_confidence'] = type_confidence

    return pf


@app.route('/api/v1/query/parcels')
def api_query_parcels():
    """Query parcels from the search index (fast SQL, no GPKG/JSON load).

    Supports all per-parcel landscape attributes plus building join filters.
    This is the endpoint for complex cross-KG parcel queries like:
    "Show parcels with a 1-storey pitched-roof building, south-facing >30% slope,
     <5000 sqm, above 900m, with little deforestation in the last 5 years."

    Params:
      kg=<code>                   Filter by KG code
      state=<name|code>           Filter by Bundesland
      district=<name|code>        Filter by Bezirk
      gemeinde=<name|code>        Filter by Gemeinde
      bbox=<w,s,e,n>              Spatial bbox filter
      min_area / max_area         Parcel area m²
      min_elevation / max_elevation  Elevation m
      min_slope / max_slope       Slope degrees
      min_tri / max_tri           Terrain Roughness Index
      terrain_class=<str>         level / nearly_level / slightly_rugged / ...
      aspect=<N,NE,E,SE,S,SW,W,NW>  Comma-separated aspect filter
      dominant_type=<str>         Dominant segment type (tree/grass/roof/...)
      min_vegetated_fraction / max_vegetated_fraction   0-1
      min_forested_fraction / max_forested_fraction     0-1
      min_ndsm_max / max_ndsm_max                      nDSM max height m
      is_vegetated=true/false     Vegetation boolean
      min_confidence              Min mean classification confidence
      min_rf_confidence           Min RF classification confidence
      max_hansen_recent_5yr       Max recent 5-year forest loss pixels
      min_hansen_recent_5yr       Min recent 5-year forest loss pixels
      max_hansen_total            Max total forest loss pixels
      min_hansen_total            Min total forest loss pixels
      building_roof_type=pitched/flat  Parcel must contain matching building
      building_min_stories=<N>    Building stories filter (any building in parcel)
      building_max_stories=<N>    Building stories filter
      min_buildings / max_buildings   Number of buildings on parcel (PIP)
      has_buildings=true/false        Parcel contains ≥1 building
      min_building_height / max_building_height   Tallest building height (m)
      min_building_stories / max_building_stories Tallest building stories
      auto_class=<class>          Auto-class filter (comma-separated allowed):
                                  forest, young_forest, wooded, meadow,
                                  alpine_meadow, cropland, vineyard, orchard,
                                  shrubland, built_up, farmstead,
                                  infrastructure, water_body, disturbance,
                                  bare, mixed
      auto_subclass=<sub>         multi_storey, apartments, with_house,
                                  recently_thinned, regenerating, pasture,
                                  rugged, dense, recent_clearfell, ...
      min_auto_class_confidence   0-1 minimum classifier confidence
      sort=<col>                  Sort column (default: elevation_m)
      sort_dir=asc/desc           Sort direction (default: desc)
      limit=<N>                   Max results (default 100, max 1000)
      offset=<N>                  Pagination offset
    """
    try:
        idx = si.get_index()
        args = request.args
        limit = min(int(args.get('limit', 100)), 1000)
        offset = int(args.get('offset', 0))

        # Parse bbox
        bbox = None
        if args.get('bbox'):
            parts = [float(x) for x in args['bbox'].split(',')]
            if len(parts) == 4:
                bbox = tuple(parts)

        # Numeric params
        _num = {
            'min_area': float, 'max_area': float,
            'min_elevation': float, 'max_elevation': float,
            'min_slope': float, 'max_slope': float,
            'min_tri': float, 'max_tri': float,
            'min_vegetated_fraction': float, 'max_vegetated_fraction': float,
            'min_forested_fraction': float, 'max_forested_fraction': float,
            'min_ndsm_max': float, 'max_ndsm_max': float,
            'min_confidence': float, 'min_rf_confidence': float,
            'max_hansen_recent_5yr': int, 'min_hansen_recent_5yr': int,
            'max_hansen_total': int, 'min_hansen_total': int,
            'building_min_stories': int, 'building_max_stories': int,
            'min_buildings': int, 'max_buildings': int,
            'min_building_height': float, 'max_building_height': float,
            'min_building_stories': int, 'max_building_stories': int,
            'min_auto_class_confidence': float,
        }
        kwargs = {}
        for k, conv in _num.items():
            if k in args:
                try:
                    kwargs[k] = conv(args[k])
                except ValueError:
                    pass

        has_buildings = None
        if 'has_buildings' in args:
            has_buildings = args.get('has_buildings', '').lower() in ('true', '1', 'yes')

        result = idx.query_parcels_index(
            kg_code=args.get('kg'),
            terrain_class=args.get('terrain_class'),
            aspect=args.get('aspect'),
            dominant_type=args.get('dominant_type'),
            building_roof_type=args.get('building_roof_type'),
            auto_class=args.get('auto_class'),
            auto_subclass=args.get('auto_subclass'),
            has_buildings=has_buildings,
            is_vegetated=args.get('is_vegetated', '').lower() in ('true', '1') if 'is_vegetated' in args else None,
            state=args.get('state'),
            district=args.get('district'),
            gemeinde=args.get('gemeinde'),
            bbox=bbox,
            sort=args.get('sort', 'elevation_m'),
            sort_dir=args.get('sort_dir', 'desc'),
            limit=limit,
            offset=offset,
            **kwargs,
        )
        return jsonify(result)
    except Exception as e:
        log.exception('query_parcels error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/query/compound', methods=['GET', 'POST'])
def api_query_compound():
    """Compound query: filter KGs by any combination of attributes.

    Accepts both POST (JSON body) and GET (query params).

    GET query params map directly to filter keys. Special syntax:
      - bbox=w,s,e,n
      - aspect=S,SW,W (comma-separated)
      - type_filter=tree:0.8:800 (repeatable, type:min_confidence:min_area_sqm)
      - landcover_filter=grass:1300:0.1 (repeatable, type:min_area_sqm:min_fraction)
      - All min_*/max_* numeric params
      - limit, offset, async, task_id as query params

    POST JSON body with filter keys (all optional):
      bbox: [w, s, e, n]
      state, district, gemeinde: str
      aspect: ["S","SW","W"]           dominant aspect direction
      dominant_type, phenology, terrain_class, quality_grade: str (exact match)

      Numeric ranges (min_X / max_X):
        --- Terrain ---
        slope, roughness, elevation, elevation_min, elevation_max,
        elevation_range, steepness_max, tri
        --- Area / parcels / segments ---
        total_area, parcels, segments
        --- Buildings ---
        buildings, new_buildings, infrastructure,
        building_height, building_max_height,
        building_stories, building_stories_max,
        building_pitched_pct, building_footprint,
        new_building_footprint, new_building_height, new_building_stories,
        building_height_coverage
        --- Trees ---
        tree_count, tree_height, tree_canopy_sqm, tree_volume
        --- Vegetation ---
        ndvi, vegetated_fraction, shannon_diversity
        --- NDVI harmonics ---
        ndvi_amplitude, ndvi_harm_mean, ndvi_phase
        --- SAR ---
        sar_vv, sar_vh
        --- Temporal change ---
        dtm_change, volume_change, changed_segments, disturbed_volume,
        temporal_stability
        --- Classification quality ---
        confidence, rf_confidence, diverged_pct (max),
        rf_diverged_count (max), rf_classified_pct, quality_score

      type_filters: [{"type": "tree", "min_confidence": 0.8, "min_area_sqm": 800}, ...]
        Filter by RF classification confidence + area per object type.
      landcover_filters: [{"type": "grass", "min_area_sqm": 1300, "min_fraction": 0.1}, ...]
        Filter by landcover area/fraction/height per object type.
      sort: str (any kg column or type-derived column)
      sort_dir: "asc" | "desc"
      limit: int (default 50, max 1000), offset: int

    Example:
      POST /api/v1/query/compound
      {
        "type_filters": [
          {"type": "tree", "min_confidence": 0.8, "min_area_sqm": 800},
          {"type": "grass", "min_confidence": 0.8, "min_area_sqm": 1300}
        ],
        "max_buildings": 0,
        "aspect": ["S", "SW", "W"],
        "min_roughness": 2.0,
        "sort": "tree_area_sqm",
        "sort_dir": "desc"
      }
    """
    try:
        idx = si.get_index()
        # Accept both POST JSON body and GET query params
        if request.method == 'POST' and request.content_type and 'json' in request.content_type:
            body = request.get_json(force=True) or {}
        elif request.method == 'POST':
            # POST with no JSON — try to parse, fall back to query args
            try:
                body = request.get_json(force=True) or {}
            except Exception:
                body = _parse_compound_from_args(request.args)
        else:
            # GET — parse from query params
            body = _parse_compound_from_args(request.args)
        limit = min(int(body.pop('limit', None) or request.args.get('limit', 50)), 1000)
        offset = int(body.pop('offset', None) or request.args.get('offset', 0))
        _async_val = body.pop('async', None) or request.args.get('async', '')
        do_async = str(_async_val).lower() in ('true', '1', 'yes')
        # Poll existing task
        _task_id_val = body.pop('task_id', None) or request.args.get('task_id')
        if _task_id_val:
            task_id = _task_id_val
            p = _PROGRESS_DIR / f"{task_id}.json"
            if not p.exists():
                return jsonify({'error': 'Unknown task_id'}), 404
            try:
                info = json.loads(p.read_text())
            except Exception:
                return jsonify({'active': False})
            step = info.get('step', '')
            elapsed = round(time.time() - info.get('t0', time.time()), 1)
            if step == 'done':
                result = _get_query_result(task_id)
                if result is not None:
                    try: p.unlink(missing_ok=True)
                    except Exception: pass
                    return jsonify(result)
                return jsonify({'active': False, 'done': True, 'elapsed': elapsed})
            if step == 'error':
                try: p.unlink(missing_ok=True)
                except Exception: pass
                return jsonify({'error': info.get('detail', 'Query failed')}), 500
            return jsonify({'active': True, 'task_id': task_id, 'step': step,
                            'detail': info.get('detail', ''), 'elapsed': elapsed}), 202
        if do_async:
            task_id = str(uuid.uuid4())
            t = threading.Thread(target=_query_worker, daemon=True,
                args=(task_id, idx.query_compound, (body,),
                      dict(limit=limit, offset=offset)))
            t.start()
            return jsonify({'task_id': task_id, 'status': 'running',
                            'poll': f'/api/v1/query/compound'}), 202
        result = idx.query_compound(body, limit=limit, offset=offset)
        return jsonify(result)
    except Exception as e:
        log.warning('api_query_compound: %s', e)
        return jsonify({'error': str(e)}), 500


# === SECTION: Async query task support ===
_QUERY_RESULTS_DIR = Path('/tmp/query_results')
_QUERY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _query_worker(task_id, query_fn, args, kwargs):
    """Run a slow query in a background thread."""
    try:
        _progress_start(task_id)
        _progress_set(task_id, 'running', f'Scanning KG JSONs...')
        result = query_fn(*args, **kwargs)
        p = _QUERY_RESULTS_DIR / f"{task_id}.json.gz"
        data = json.dumps(result).encode()
        with gzip.open(str(p), 'wb') as f:
            f.write(data)
        _progress_done(task_id)
    except Exception as e:
        log.exception('async query %s failed', task_id)
        _progress_error(task_id, str(e))

def _get_query_result(task_id):
    """Retrieve stored query result."""
    p = _QUERY_RESULTS_DIR / f"{task_id}.json.gz"
    if not p.exists():
        return None
    with gzip.open(str(p), 'rb') as f:
        return json.loads(f.read())

@app.route('/api/v1/query/progress')
def query_progress():
    """Poll progress of an async query task."""
    task_id = request.args.get('task_id', '')
    if not task_id:
        return jsonify({'error': 'task_id required'}), 400
    p = _PROGRESS_DIR / f"{task_id}.json"
    if not p.exists():
        return jsonify({'error': 'Unknown task_id'}), 404
    try:
        info = json.loads(p.read_text())
    except Exception:
        return jsonify({'active': False})
    step = info.get('step', '')
    elapsed = round(time.time() - info.get('t0', time.time()), 1)
    if step == 'done':
        result = _get_query_result(task_id)
        if result is not None:
            # Clean up progress file
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
            return jsonify(result)
        return jsonify({'active': False, 'done': True, 'elapsed': elapsed})
    if step == 'error':
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({'error': info.get('detail', 'Query failed')}), 500
    return jsonify({'active': True, 'step': step, 'detail': info.get('detail', ''),
                    'elapsed': elapsed}), 202


@app.route('/api/v1/query')
def api_query():
    """Unified query endpoint. Supports multiple query modes via params.

    Params:
      q=<text>          Full-text search (KG/gemeinde/district/state names)
      kg=<code>         Exact KG code lookup
      parcel=<id>       Parcel lookup (KGCODE-GNR)
      bbox=<w,s,e,n>    Spatial bbox query
      point=<lon,lat>   Point proximity query
      radius=<km>       Radius for point query (default 5)
      state=<code|name>         Filter/aggregate by Bundesland
      district=<code>           Filter/aggregate by Bezirk
      gemeinde=<code>           Filter/aggregate by Gemeinde
      type=<object_type>        Rank KGs by object type
      metric=<area|fraction|count|height>  Ranking metric (with type=)
      hansen=true               Hansen forest loss query
      year_from=<YYYY>          Start year for hansen/temporal filter
      year_to=<YYYY>            End year for hansen/temporal filter
      new_buildings=true         Query KGs with new buildings
      min_count=<N>             Min count for new_buildings
      divergence=true           Query KGs ranked by RF→final type divergence
      min_divergence=<pct>      Min divergence % (0-100, with divergence=true)
      rf_type=<type>            Filter divergences FROM this RF type
      final_type=<type>         Filter divergences TO this final type
      low_confidence=true       Query KGs with lowest confidence
      max_confidence=<float>    Max confidence threshold (with low_confidence=true)
      confidence_rank=asc|desc  Rank KGs by classification confidence
      type_confidence=<type>    Rank KGs by RF confidence for specific type
      divergence_pairs=true     Get most common RF→final divergence pairs
      high_confidence_type=<type>  KGs where type has high RF confidence
      parcels_by_type=<type>    Per-parcel filter by type RF confidence+area
      top_features=<trees|objects|new_buildings|infrastructure>  Cross-KG top features
      min_confidence=<float>    Min RF confidence (with high_confidence_type/parcels_by_type/top_features)
      min_area_sqm=<float>      Min area m² (with high_confidence_type/parcels_by_type)
      processed_only=true       Only return processed KGs
      aggregate=true            Return aggregate stats instead of KG list
      limit=<N>                 Max results (default 100)
      offset=<N>                Offset for pagination
    """
    try:
        idx = si.get_index()
        args = request.args
        limit = min(int(args.get('limit', 100)), 1000)
        offset = int(args.get('offset', 0))
        processed_only = args.get('processed_only', '').lower() in ('true', '1', 'yes')
        do_aggregate = args.get('aggregate', '').lower() in ('true', '1', 'yes')
        do_async = args.get('async', '').lower() in ('true', '1', 'yes')

        # Poll existing async query task
        if args.get('task_id') and not args.get('q') and not args.get('kg'):
            task_id = args['task_id']
            p = _PROGRESS_DIR / f"{task_id}.json"
            if not p.exists():
                return jsonify({'error': 'Unknown task_id'}), 404
            try:
                info = json.loads(p.read_text())
            except Exception:
                return jsonify({'active': False})
            step = info.get('step', '')
            elapsed = round(time.time() - info.get('t0', time.time()), 1)
            if step == 'done':
                result = _get_query_result(task_id)
                if result is not None:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return jsonify(result)
                return jsonify({'active': False, 'done': True, 'elapsed': elapsed})
            if step == 'error':
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
                return jsonify({'error': info.get('detail', 'Query failed')}), 500
            return jsonify({'active': True, 'task_id': task_id, 'step': step,
                            'detail': info.get('detail', ''), 'elapsed': elapsed}), 202

        # Single KG lookup
        if args.get('kg'):
            result = idx.query_kg(args['kg'])
            return jsonify(result) if result else (jsonify({'error': 'KG not found'}), 404)

        # Parcel lookup
        if args.get('parcel'):
            result = idx.query_parcel(args['parcel'])
            return jsonify(result) if result else (jsonify({'error': 'Parcel not found'}), 404)

        # Aggregate queries
        # Aggregate queries (only when aggregate=true)
        if do_aggregate:
            if args.get('state'):
                return jsonify(idx.aggregate_state(args['state']))
            if args.get('district'):
                return jsonify(idx.aggregate_district(args['district']))
            if args.get('gemeinde'):
                return jsonify(idx.aggregate_gemeinde(args['gemeinde']))
            return jsonify(idx.aggregate_country())

        # Admin hierarchy — list KGs in a state/district/gemeinde
        if args.get('state'):
            sc = args['state']
            # Accept name or code
            if not sc.isdigit():
                for code, name in si.STATE_CODES.items():
                    if name.lower() == sc.lower():
                        sc = code
                        break
            return jsonify(idx.query_admin('state', sc, processed_only=processed_only, limit=limit, offset=offset))
        if args.get('district'):
            dc = args['district']
            # Accept name or code
            if not dc.isdigit():
                for code, name in si.DISTRICT_NAMES.items():
                    if name.lower() == dc.lower():
                        dc = code
                        break
            return jsonify(idx.query_admin('district', dc,
                                           processed_only=processed_only, limit=limit, offset=offset))
        if args.get('gemeinde'):
            gm = args['gemeinde']
            # Accept name or code — look up in DB if name given
            if not gm.isdigit():
                c = idx._conn()
                row = c.execute('SELECT gemeinde_code FROM kg WHERE gemeinde_name = ? LIMIT 1', (gm,)).fetchone()
                if row:
                    gm = row[0]
            return jsonify(idx.query_admin('gemeinde', gm,
                                           processed_only=processed_only, limit=limit, offset=offset))

        # Object type ranking
        if args.get('type'):
            metric = args.get('metric', 'area')
            return jsonify(idx.query_type_ranking(args['type'], metric=metric, limit=limit, offset=offset))

        # Hansen forest loss
        if args.get('hansen', '').lower() in ('true', '1', 'yes'):
            yf = int(args['year_from']) if args.get('year_from') else None
            yt = int(args['year_to']) if args.get('year_to') else None
            ml = int(args.get('min_loss', 0))
            return jsonify(idx.query_hansen_loss(year_from=yf, year_to=yt,
                                                 min_loss=ml, limit=limit, offset=offset))

        # New buildings
        if args.get('new_buildings', '').lower() in ('true', '1', 'yes'):
            mc = int(args.get('min_count', 1))
            return jsonify(idx.query_new_buildings(min_count=mc, limit=limit, offset=offset))

        # Classification divergence
        if args.get('divergence', '').lower() in ('true', '1', 'yes'):
            min_div = float(args.get('min_divergence', 0))
            rf_t = args.get('rf_type')
            fin_t = args.get('final_type')
            return jsonify(idx.query_divergence(
                min_pct=min_div, rf_type=rf_t, final_type=fin_t,
                limit=limit, offset=offset))

        # Divergence pairs (most common RF→final mismatches across all KGs)
        if args.get('divergence_pairs', '').lower() in ('true', '1', 'yes'):
            return jsonify(idx.query_divergence_pairs(limit=limit, offset=offset))

        # Low confidence KGs
        if args.get('low_confidence', '').lower() in ('true', '1', 'yes'):
            mc = float(args.get('max_confidence', 0.5))
            return jsonify(idx.query_low_confidence(
                max_confidence=mc, limit=limit, offset=offset))

        # Confidence ranking
        if args.get('confidence_rank'):
            order = args['confidence_rank']
            return jsonify(idx.query_confidence_ranking(
                order=order, limit=limit, offset=offset))

        # Per-type RF confidence ranking
        if args.get('type_confidence'):
            return jsonify(idx.query_type_confidence(
                args['type_confidence'], limit=limit, offset=offset))

        # High-confidence type filter (KG-level)
        # e.g. ?high_confidence_type=tree&min_confidence=0.8&min_area_sqm=1500
        if args.get('high_confidence_type'):
            mc = float(args.get('min_confidence', 0.7))
            ma = float(args.get('min_area_sqm', 0))
            return jsonify(idx.query_high_confidence_type(
                args['high_confidence_type'], min_confidence=mc,
                min_area_sqm=ma, limit=limit, offset=offset))

        # Per-parcel type+confidence filter (scans KG JSONs — supports async)
        # e.g. ?parcels_by_type=tree&min_confidence=0.8&min_area_sqm=1500
        if args.get('parcels_by_type'):
            mc = float(args.get('min_confidence', 0.7))
            ma = float(args.get('min_area_sqm', 0))
            if do_async:
                task_id = args.get('task_id') or str(uuid.uuid4())
                t = threading.Thread(target=_query_worker, daemon=True,
                    args=(task_id, idx.query_parcels_by_type_confidence,
                          (args['parcels_by_type'],),
                          dict(min_confidence=mc, min_area_sqm=ma,
                               limit=limit, offset=offset)))
                t.start()
                return jsonify({'task_id': task_id, 'status': 'running',
                                'poll': f'/api/v1/query?task_id={task_id}'}), 202
            return jsonify(idx.query_parcels_by_type_confidence(
                args['parcels_by_type'], min_confidence=mc,
                min_area_sqm=ma, limit=limit, offset=offset))

        # Cross-KG top features (trees/objects/new_buildings/infrastructure — supports async)
        # e.g. ?top_features=trees&min_confidence=0.9
        # e.g. ?top_features=new_buildings&min_confidence=0.75
        # e.g. ?top_features=infrastructure&type=mast&min_confidence=0.8
        if args.get('top_features'):
            mc = float(args.get('min_confidence', 0))
            otype = args.get('type')
            tf_bbox = None
            if args.get('bbox'):
                bp = [float(x) for x in args['bbox'].split(',')]
                if len(bp) == 4:
                    tf_bbox = tuple(bp)
            if do_async:
                task_id = args.get('task_id') or str(uuid.uuid4())
                t = threading.Thread(target=_query_worker, daemon=True,
                    args=(task_id, idx.query_top_features,
                          (args['top_features'],),
                          dict(object_type=otype, min_confidence=mc,
                               bbox=tf_bbox, limit=limit, offset=offset)))
                t.start()
                return jsonify({'task_id': task_id, 'status': 'running',
                                'poll': f'/api/v1/query?task_id={task_id}'}), 202
            res = idx.query_top_features(
                args['top_features'], object_type=otype,
                min_confidence=mc, bbox=tf_bbox,
                limit=limit, offset=offset)
            try:
                _enrich_top_features_with_flags(res, args['top_features'],
                    exclude_flagged=args.get('exclude_flagged','').lower() in ('true','1','yes'),
                    min_severity=args.get('min_flag_severity'))
            except Exception as e:
                log.warning('flag enrichment: %s', e)
            return jsonify(res)

        # Cross-KG segment-level power queries
        # e.g. ?segments=true&object_type=tree&min_rf_confidence=0.9&percentile=0.01
        # e.g. ?segments=true&object_type=excavation&sort=volume&min_rf_confidence=0.7
        # e.g. ?segments=true&object_type=tree_loss&min_rf_confidence=0.8&percentile=0.05
        if args.get('segments', '').lower() in ('true', '1', 'yes'):
            seg_bbox = None
            if args.get('bbox'):
                bp = [float(x) for x in args['bbox'].split(',')]
                if len(bp) == 4:
                    seg_bbox = tuple(bp)
            seg_kwargs = dict(
                object_type=args.get('object_type') or args.get('type'),
                min_rf_confidence=float(args['min_rf_confidence']) if args.get('min_rf_confidence') else None,
                max_rf_confidence=float(args['max_rf_confidence']) if args.get('max_rf_confidence') else None,
                min_confidence=float(args['min_confidence']) if args.get('min_confidence') else None,
                max_confidence=float(args['max_confidence']) if args.get('max_confidence') else None,
                min_area_sqm=float(args['min_area_sqm']) if args.get('min_area_sqm') else None,
                max_area_sqm=float(args['max_area_sqm']) if args.get('max_area_sqm') else None,
                min_height=float(args['min_height']) if args.get('min_height') else None,
                max_height=float(args['max_height']) if args.get('max_height') else None,
                min_volume=float(args['min_volume']) if args.get('min_volume') else None,
                max_volume=float(args['max_volume']) if args.get('max_volume') else None,
                bbox=seg_bbox,
                state=args.get('state'),
                district=args.get('district'),
                sort=args.get('sort', 'height_max_m'),
                sort_dir=args.get('sort_dir', 'desc'),
                percentile=float(args['percentile']) if args.get('percentile') else None,
                limit=limit, offset=offset,
            )
            return jsonify(idx.query_segments(**seg_kwargs))

        # Spatial bbox
        if args.get('bbox'):
            parts = [float(x) for x in args['bbox'].split(',')]
            if len(parts) == 4:
                return jsonify(idx.query_bbox(*parts, processed_only=processed_only, limit=limit, offset=offset))
            return jsonify({'error': 'bbox must be w,s,e,n'}), 400

        # Point proximity
        if args.get('point'):
            parts = [float(x) for x in args['point'].split(',')]
            if len(parts) == 2:
                radius = float(args.get('radius', 5))
                return jsonify(idx.query_point(parts[0], parts[1], radius_km=radius, limit=limit, offset=offset))
            return jsonify({'error': 'point must be lon,lat'}), 400

        # Full-text search
        if args.get('q'):
            return jsonify(idx.query_text(args['q'], limit=limit, offset=offset))

        # Default: list processed KGs
        return jsonify(idx.query_processed(limit=limit, offset=offset))

    except Exception as e:
        log.exception('query error')
        return jsonify({'error': str(e)}), 500


# === SECTION: Cross-API bridge (cadastre + landscape) ===

@app.route('/api/v1/lookup')
def api_lookup():
    """Proxy to cadastre lookup — fuzzy diacritics-insensitive search.

    Searches Austria's federal register (EDM): Gemeinden, KGs, Ortschaften, PLZ.
    Handles umlauts gracefully ("Kofla" → "Köflach").

    Params:
      q (required): Search text (name, PLZ, code)
      type: Filter by entity type (plz|gemeinde|kg|ortschaft)
      limit: Max results (default 20, max 200)
    """
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'error': 'q parameter required'}), 400
    try:
        result = cb.lookup(
            q=q,
            type=request.args.get('type'),
            limit=int(request.args.get('limit', 20)),
        )
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_lookup error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/parcels/batch', methods=['GET', 'POST'])
def api_parcels_batch():
    """Batch parcel landscape enrichment — 3 modes.

    Accepts both POST (JSON body) and GET (query params).

    GET maps to Mode 3 (compound → parcels). Compound filters use direct
    query params; parcel filters use pf_ prefix.

    GET examples:
      /api/v1/parcels/batch?state=Vorarlberg&pf_aspect=E&pf_terrain_class=level&limit=100
      /api/v1/parcels/batch?min_tree_count=50&pf_min_vegetated_fraction=0.5&pf_sort=conservation_score
      /api/v1/parcels/batch?type_filter=tree:0.8:800&pf_types=tree,grass&limit=50
      /api/v1/parcels/batch?parcel_ids=63349-505/3,75414-1314/1  (Mode 1 via GET)
      /api/v1/parcels/batch?min_elevation=900&aspect=SE,S,SW&min_slope=17&max_building_stories_max=1&min_building_pitched_pct=80&pf_aspect=SE,S,SW&pf_min_slope=17&pf_min_elevation=900&pf_max_parcel_area=5000&pf_roof_type=pitched&pf_max_stories=1&pf_cadastre_has_buildings=true&limit=100

    POST modes:

    Mode 1 — Explicit IDs:
      Body: {"parcel_ids": ["63349-505/3", "75414-1314/1", ...]}
      Max 200 parcel IDs per request.

    Mode 2 — Cadastre query:
      Body: {
        "query": {<cadastre /query params>},
        "landscape_filters": {<landscape post-filters>},
        "limit": 50, "offset": 0
      }

    Mode 3 — Landscape-first (compound → parcels):
      Start from our landscape index, find KGs matching compound filters,
      then expand to individual parcels from our KG JSONs.
      This is the power query for nature conservation screening.

      Body: {
        "compound": {                 // SearchIndex.query_compound() filters
          "type_filters": [
            {"type": "tree", "min_confidence": 0.8, "min_area_sqm": 800},
            {"type": "grass", "min_confidence": 0.8, "min_area_sqm": 1300}
          ],
          "max_buildings": 0,
          "aspect": ["S", "SW", "W"],
          "min_roughness": 2.0,
          // also: bbox, state, district, gemeinde, min_slope, min_elevation,
          // min_ndvi, min_vegetated_fraction, min_tree_count, phenology,
          // landcover_filters, sort, sort_dir, etc.
        },
        "parcel_filters": {           // Per-parcel post-filters on our JSON data
          "min_vegetated_fraction": 0.5,
          "min_elevation": 500,
          "max_elevation": 2000,
          "types": ["tree", "grass"],  // require these types in parcel
          "min_type_fraction": 0.1,   // min fraction per required type
          "min_ndsm_max": 5,          // min height above ground (m)
          "min_parcel_area": 1000,    // min parcel area (sqm)
          "is_vegetated": true,
          // Per-type classification confidence:
          "type_confidence": [
            {"type": "tree", "min_confidence": 0.7, "min_area_sqm": 500},
            {"type": "grass", "min_rf_confidence": 0.8},
            // Also: min_rf_count, min_rules_count, min_fraction, max_diverged_pct
          ],
          "min_confidence": 0.6,       // overall combined confidence
          "min_rf_confidence": 0.7,    // overall RF confidence
          // Building attribute filters (centroid → /spatial/points point-in-polygon):
          "roof_type": "pitched",      // "pitched" or "flat"
          "min_stories": 1,             // building stories_est >= N
          "max_stories": 1,             // building stories_est <= N
          // Cadastre-side filters (applied after enrichment):
          "cadastre_has_buildings": false,
          "cadastre_landuse": "W",
          "cadastre_min_area": 5000,
          "sort": "conservation_score", // |vegetated_fraction|elevation|ndsm_max
          "sort_dir": "desc"
        },
        "cadastre_enrich": true,      // fetch cadastre data (default true)
        "limit": 100, "offset": 0
      }

    Returns:
      {"results": [{"parcel_id": ..., "kg_code": ..., "landscape": {...},
                     "cadastre": {...}|null, "conservation_score": N}, ...],
       "total": N, "offset": N, "limit": N, "meta": {...}}
    """
    # --- Parse body: POST JSON or GET query params ---
    if request.method == 'POST':
        try:
            body = request.get_json(force=True) or {}
        except Exception:
            return jsonify({'error': 'Invalid JSON body'}), 400
    else:
        # GET — parse from query params
        args = request.args
        body = {}
        # Mode 1 via GET: parcel_ids=id1,id2,...
        if 'parcel_ids' in args:
            body['parcel_ids'] = [x.strip() for x in args['parcel_ids'].split(',') if x.strip()]
        else:
            # Mode 3 via GET: compound filters from query params, parcel filters from pf_ prefix
            compound = _parse_compound_from_args(args)
            if compound:
                body['compound'] = compound
                pf = _parse_parcel_filters_from_args(args)
                if pf:
                    body['parcel_filters'] = pf
                if 'cadastre_enrich' in args:
                    body['cadastre_enrich'] = args['cadastre_enrich'].lower() in ('true', '1', 'yes')
            # Check for Mode 2 via GET: query params with cq_ prefix
            # (not commonly used via GET, but supported)
        if 'limit' in args:
            body.setdefault('limit', int(args['limit']))
        if 'offset' in args:
            body.setdefault('offset', int(args['offset']))

    try:
        # Mode 1: Explicit parcel IDs
        if 'parcel_ids' in body:
            parcel_ids = body['parcel_ids']
            if isinstance(parcel_ids, str):
                parcel_ids = [x.strip() for x in parcel_ids.split(',') if x.strip()]
            if not isinstance(parcel_ids, list):
                return jsonify({'error': 'parcel_ids must be a list'}), 400
            if len(parcel_ids) > 200:
                return jsonify({'error': f'Max 200 parcels per request, got {len(parcel_ids)}'}), 400
            if len(parcel_ids) == 0:
                return jsonify({'results': [], 'total': 0})
            result = cb.batch_parcel_landscape(parcel_ids)
            return jsonify(result)

        # Mode 2: Query-based batch
        if 'query' in body:
            query_filters = body['query']
            if not isinstance(query_filters, dict) or not query_filters:
                return jsonify({'error': 'query must be a non-empty dict of cadastre filter params'}), 400
            landscape_filters = body.get('landscape_filters') or {}
            limit = min(int(body.get('limit', 50)), 500)
            offset = int(body.get('offset', 0))
            result = cb.batch_parcel_landscape_by_query(
                query_filters=query_filters,
                landscape_filters=landscape_filters,
                limit=limit,
                offset=offset,
            )
            return jsonify(result)

        # Mode 3: Landscape-first (compound query → parcels from our KG JSONs)
        if 'compound' in body:
            compound_filters = body['compound']
            if not isinstance(compound_filters, dict) or not compound_filters:
                return jsonify({'error': 'compound must be a non-empty dict of landscape filter params'}), 400
            parcel_filters = body.get('parcel_filters') or {}
            cadastre_enrich = body.get('cadastre_enrich', True)
            limit = min(int(body.get('limit', 100)), 1000)
            offset = int(body.get('offset', 0))
            result = cb.landscape_parcel_query(
                compound_filters=compound_filters,
                parcel_filters=parcel_filters,
                cadastre_enrich=cadastre_enrich,
                limit=limit,
                offset=offset,
            )
            return jsonify(result)

        return jsonify({'error': 'No filters provided. Use query params (state=, min_slope=, etc.) or POST JSON body with parcel_ids/query/compound.'}), 400

    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_parcels_batch error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/parcels/landscape')
def api_parcels_landscape():
    """Query parcels with landscape analysis context.

    Forwards cadastre query params to the cadastre /query endpoint, then enriches
    results with landscape data from our index/JSON files.

    Cadastre params (forwarded to cadastre /query):
      q, kg, gemeinde, district, state, plz, landuse, min_area, max_area,
      has_buildings, status, ez, has_legal_refs, legal_context,
      min_lon, min_lat, max_lon, max_lat, sort

    Landscape params (post-filter on our data):
      min_vegetated_fraction, max_vegetated_fraction: 0-1
      min_ndvi, max_ndvi: float
      min_tree_canopy_sqm: float
      min_elevation, max_elevation: float
      min_conservation_score: 0-100
      dominant_type: landscape object type
      landscape_sort: conservation_score|area|ndvi|tree_canopy|vegetated_fraction
      landscape_sort_dir: asc|desc

    Pagination: limit (default 50, max 500), offset
    """
    args = request.args

    # Separate cadastre params from landscape params
    cadastre_keys = {
        'q', 'kg', 'gemeinde', 'district', 'state', 'plz', 'landuse',
        'min_area', 'max_area', 'has_buildings', 'status', 'ez',
        'has_legal_refs', 'legal_context',
        'min_lon', 'min_lat', 'max_lon', 'max_lat', 'sort',
    }
    landscape_keys = {
        'min_vegetated_fraction', 'max_vegetated_fraction',
        'min_ndvi', 'max_ndvi', 'min_tree_canopy_sqm',
        'min_elevation', 'max_elevation', 'min_conservation_score',
        'dominant_type', 'landscape_sort', 'landscape_sort_dir',
    }

    query_filters = {k: args[k] for k in cadastre_keys if k in args}
    if not query_filters:
        return jsonify({'error': 'At least one cadastre filter param required'}), 400

    landscape_filters = {}
    for k in landscape_keys:
        if k in args:
            # Convert numeric landscape filters
            if k.startswith('min_') or k.startswith('max_'):
                try:
                    landscape_filters[k] = float(args[k])
                except ValueError:
                    return jsonify({'error': f'Invalid numeric value for {k}'}), 400
            elif k == 'landscape_sort':
                landscape_filters['sort'] = args[k]
            elif k == 'landscape_sort_dir':
                landscape_filters['sort_dir'] = args[k]
            else:
                landscape_filters[k] = args[k]

    limit = min(int(args.get('limit', 50)), 500)
    offset = int(args.get('offset', 0))

    try:
        result = cb.batch_parcel_landscape_by_query(
            query_filters=query_filters,
            landscape_filters=landscape_filters,
            limit=limit,
            offset=offset,
        )
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_parcels_landscape error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/query/nature')
def api_query_nature():
    """Nature conservation opportunity finder.

    Cross-references cadastre, protected areas, legal refs, and landscape
    analysis to find and rank parcels by conservation value.

    Params:
      bbox=w,s,e,n            Spatial filter (WGS84)
      state                   State name or code
      district                District code
      gemeinde                Gemeinde code or name
      protected_area=<name>   Near/in a WDPA protected area
      legal_context=<ctx>     Filter by legal context
                              (national_park, nature_protection, landscape_protection,
                               water_protection, species_protection, etc.)
      min_vegetated_fraction  Min vegetation fraction from landscape (0-1)
      min_ndvi                Min NDVI from landscape analysis
      min_tree_canopy_sqm     Min tree canopy area (sq metres)
      min_area_sqm            Min parcel area
      max_area_sqm            Max parcel area
      landuse                 Cadastre landuse code or abbreviation (W=Wald, LN=Landwirtschaft)
      has_buildings           Building presence filter (true|false)
      sort                    Sort key: conservation_score|area|ndvi|tree_canopy|vegetated_fraction
      limit (default 50), offset
    """
    args = request.args
    try:
        result = cb.nature_conservation_screen(
            bbox=args.get('bbox'),
            state=args.get('state'),
            district=args.get('district'),
            gemeinde=args.get('gemeinde'),
            protected_area=args.get('protected_area'),
            legal_context=args.get('legal_context'),
            min_vegetated_fraction=_float_or_none(args.get('min_vegetated_fraction')),
            min_ndvi=_float_or_none(args.get('min_ndvi')),
            min_tree_canopy_sqm=_float_or_none(args.get('min_tree_canopy_sqm')),
            min_area_sqm=_float_or_none(args.get('min_area_sqm')),
            max_area_sqm=_float_or_none(args.get('max_area_sqm')),
            landuse=args.get('landuse'),
            has_buildings=_bool_or_none(args.get('has_buildings')),
            sort=args.get('sort', 'conservation_score'),
            limit=min(int(args.get('limit', 50)), 500),
            offset=int(args.get('offset', 0)),
        )
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_query_nature error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/parcel/<path:parcel_id>/detail')
def api_parcel_detail(parcel_id):
    """Full combined detail for a single parcel (both APIs).

    Combines cadastre data (area, landuse, EZ, buildings, legal refs) with
    landscape analysis (elevation, NDVI, vegetation, classification, heights).
    Also checks protected area containment and computes conservation score.
    """
    try:
        result = cb.parcel_landscape_detail(parcel_id)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_parcel_detail error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/kg/<kg_code>/profile')
def api_kg_profile(kg_code):
    """Combined KG profile from both APIs.

    Merges cadastre data (parcels, buildings, landuse distribution, legal refs)
    with landscape analysis (landcover, elevation, NDVI, trees, new buildings).
    """
    try:
        result = cb.kg_combined_profile(kg_code)
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_kg_profile error')
        return jsonify({'error': str(e)}), 500


# --- Cadastre proxy endpoints ---

@app.route('/api/v1/cadastre/legal/search')
def api_cadastre_legal_search():
    """Proxy to cadastre legal search — find parcels referenced in Austrian law.

    Params: q (text), context (legal_context), type (listed|boundary_walk),
            bundesland, kg (KG code), limit, offset
    """
    try:
        params = {k: v for k, v in request.args.items()}
        result = cb.cadastre_proxy('/legal/search', params=params)
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/cadastre/protected_areas')
def api_cadastre_protected_areas():
    """Proxy to cadastre protected area search — 43 WDPA areas in Austria.

    Params: q (text), near_lon+near_lat (sort by proximity),
            contains_lon+contains_lat (point-in-polygon), limit, offset
    """
    try:
        params = {k: v for k, v in request.args.items()}
        result = cb.cadastre_proxy('/search/protected_area', params=params)
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/cadastre/landuse/distribution')
def api_cadastre_landuse_distribution():
    """Proxy to cadastre landuse distribution — aggregated by geography.

    Params: kg, gemeinde, district, state, code, abbr, group_by, limit, offset
    """
    try:
        params = {k: v for k, v in request.args.items()}
        result = cb.cadastre_proxy('/landuse/distribution', params=params)
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/cadastre/landuse/codes')
def api_cadastre_landuse_codes():
    """Proxy to cadastre landuse codes — reference table of all Austrian codes."""
    try:
        result = cb.cadastre_proxy('/landuse/codes')
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _float_or_none(val: str | None) -> float | None:
    """Parse a float from a query param, or return None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _bool_or_none(val: str | None) -> bool | None:
    """Parse a bool from a query param, or return None."""
    if val is None:
        return None
    return val.lower() in ('true', '1', 'yes')


# === SECTION: Geometry + parameter helpers ===

def _get_geometry():
    """Extract geometry from request. Supports JSON body, form data, file upload.

    Also validates each feature is within Austria and unions multiple features
    into a single convex-hull geometry.
    """
    if 'file' in request.files:
        f = request.files['file']
        content = f.read().decode('utf-8')
        features = geo_parse.parse_input(content)
    elif request.is_json:
        body = request.get_json()
        if 'geometry' in body:
            geom_input = body['geometry']
        elif 'type' in body:
            geom_input = body
        else:
            raise ValueError("JSON body must contain 'geometry' or be a valid GeoJSON")
        features = geo_parse.parse_input(geom_input)
    elif request.form.get('geometry'):
        features = geo_parse.parse_input(request.form['geometry'])
    else:
        data = request.get_data(as_text=True)
        if data:
            features = geo_parse.parse_input(data)
        else:
            raise ValueError("No geometry provided. Send GeoJSON, KML, or coordinates.")

    # Validate each feature is within Austria
    for feat in features:
        geo_parse.validate_austria_bounds(feat['geometry'])

    # Union multiple features into one
    if len(features) > 1:
        features = geo_parse.union_features(features)

    # Convert non-polygon geometries to polygon via buffer + union
    for feat in features:
        geom = feat['geometry']
        if geom.geom_type not in ('Polygon', 'MultiPolygon'):
            feat['geometry'] = _non_polygon_to_polygon(geom)

    return features


def _clean_polygon(geom, min_hole_area=500, min_part_area=100):
    """Remove small holes and tiny polygon slivers from a geometry (metric CRS)."""
    from shapely.geometry import Polygon as ShpPolygon, MultiPolygon as ShpMultiPolygon
    if geom.geom_type == 'Polygon':
        kept = [h for h in geom.interiors if ShpPolygon(h).area >= min_hole_area]
        return ShpPolygon(geom.exterior, kept)
    elif geom.geom_type == 'MultiPolygon':
        parts = [_clean_polygon(p, min_hole_area, min_part_area)
                 for p in geom.geoms if p.area >= min_part_area]
        if not parts:
            return geom
        return parts[0] if len(parts) == 1 else ShpMultiPolygon(parts)
    return geom


def _non_polygon_to_polygon(geom):
    """Convert non-polygon geometry to Polygon.

    For lines: polygonize the network, union, buffer to close gaps, simplify.
    For points: buffer 10m.
    Falls back to convex hull on error.
    """
    from shapely.ops import transform as shp_transform, linemerge, polygonize, unary_union
    import pyproj

    try:
        to_m = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True).transform
        to_ll = pyproj.Transformer.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True).transform

        if geom.geom_type in ('LineString', 'MultiLineString'):
            # Polygonize the line network first
            merged_lines = linemerge(geom)
            polys = list(polygonize(merged_lines))
            if polys:
                result = unary_union(polys)
                # Project to metric; gently close road-width gaps while
                # preserving boundary detail: expand 5m to bridge narrow
                # road gaps, remove small holes (< 2000 m² = road strips),
                # then shrink back and lightly simplify.
                result_m = shp_transform(to_m, result)
                result_m = result_m.buffer(5)
                result_m = _clean_polygon(result_m, min_hole_area=2000)
                result_m = result_m.buffer(-5)
                result_m = _clean_polygon(result_m, min_hole_area=500)
                result_m = result_m.simplify(0.5)
                result = shp_transform(to_ll, result_m)
                if result.geom_type in ('Polygon', 'MultiPolygon') and not result.is_empty:
                    n_verts = (len(result.exterior.coords) if result.geom_type == 'Polygon'
                               else sum(len(p.exterior.coords) for p in result.geoms))
                    log.info("_parse_geometry: polygonize+union %s → %s (%d polygonized, %d vertices)",
                             geom.geom_type, result.geom_type, len(polys), n_verts)
                    return result

            # Polygonize failed — buffer lines directly
            geom_m = shp_transform(to_m, geom)
            result_m = geom_m.buffer(10).simplify(1)
            result = shp_transform(to_ll, result_m)
            if result.geom_type in ('Polygon', 'MultiPolygon') and not result.is_empty:
                log.info("_parse_geometry: line buffer fallback %s → %s", geom.geom_type, result.geom_type)
                return result
        else:
            # Points/other: buffer 10m
            geom_m = shp_transform(to_m, geom)
            result_m = geom_m.buffer(10).simplify(1)
            result = shp_transform(to_ll, result_m)
            if result.geom_type in ('Polygon', 'MultiPolygon') and not result.is_empty:
                log.info("_parse_geometry: buffer %s → %s", geom.geom_type, result.geom_type)
                return result
    except Exception:
        log.warning("_parse_geometry: polygonize/buffer failed, falling back", exc_info=True)

    hull = geom.convex_hull
    if hull.geom_type == 'Polygon' and not hull.is_empty:
        return hull
    return geom.buffer(0.001)


def _get_params():
    params = {}
    if request.is_json:
        body = request.get_json()
        params = {k: v for k, v in body.items()
                  if k not in ('geometry', 'type', 'features', 'coordinates')}
    for key in ('dataset', 'date_a', 'date_b', 'dates',
                'min_height', 'max_height', 'min_area', 'min_change',
                'object_types', 'resolution', 'format',
                'include_ortho', 'include_temporal',
                'include_copernicus', 'include_cadastre',
                'include_hansen', 'include_infra', 'mark_uncertain', 'color_mode', 'types',
                'ortho_year', 'min_object_size',
                'felz_scale', 'rag_threshold', 'groups',
                'include_dtm', 'include_dsm', 'include_segments', 'include_segments_vector',
                'ortho_years', 'raster_layers',
                'top_n_classes', 'top_n_objects', 'min_height_m',
                'layer', 'min_zoom', 'max_zoom',
                'share_id',
                'height_min', 'height_max', 'height_op'):
        val = request.args.get(key)
        if val is not None:
            params[key] = val
    return params


def _validate_area(geom_3035):
    area = geom_3035.area
    if area > MAX_AREA_SQM:
        raise ValueError(
            f"The selected area is too large ({area/1e6:.1f} km²). "
            f"Please choose a smaller region — the maximum allowed is "
            f"{MAX_AREA_SQM/1e6:.0f} km²."
        )


def _error(msg, code=400):
    return jsonify({"error": str(msg)}), code


def _rf_model_meta() -> dict:
    """Return RF model version info for response metadata."""
    try:
        import learned_classifier as lc
        clf = lc.get_classifier()
        if clf.is_trained:
            return {
                "rf_trained_at": clf.trained_at,
                "rf_n_kgs": clf.n_kgs,
                "rf_oob": round(clf.oob_score, 4),
                "rf_n_train": clf.n_train,
            }
    except Exception:
        pass
    return {}


def _try_read_ortho(data: dict) -> tuple:
    """Attempt to read RGB+NIR ortho aligned to ALS data.

    Returns (rgb, spectral) or (None, None).  *spectral* will include
    an ``"ndvi"`` key when NIR is available from an RGBI operate.
    """
    try:
        import ortho_io
        rgb, nir = ortho_io.read_ortho_for_als(data)
        spectral = ortho_io.compute_spectral_indices(rgb, nir=nir)
        # Add raw bands for object_segmentation fused gradient + classification
        if rgb is not None:
            spectral["red"] = rgb[0].astype(np.float32)
            spectral["green"] = rgb[1].astype(np.float32)
            spectral["blue"] = rgb[2].astype(np.float32)
        if nir is not None:
            spectral["nir"] = nir.astype(np.float32)
        return rgb, spectral
    except Exception as e:
        log.warning("Ortho read failed (non-fatal): %s", e)
        return None, None


def _try_copernicus(geom_wgs84, *, ndvi=True, landcover=True, sar=False,
                    harmonics=False, year: int = 2023) -> dict | None:
    """Attempt to fetch Copernicus data for a geometry.

    Always tries the tile cache (local files + Zenodo) first.  Falls back
    to the live openEO API only when cache misses AND the Austria processor
    is not running (to avoid credential conflicts).

    Parameters
    ----------
    year : int
        Observation year.  NDVI composite and SAR backscatter are fetched
        for the growing season (Apr–Sep) of this year.
    """
    bbox = geom_wgs84.bounds  # (minx, miny, maxx, maxy)
    bbox_dict = {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]}

    # --- Fast path: serve from tile cache (local + Zenodo) ---
    cached = _try_copernicus_cached(bbox_dict, ndvi=ndvi, landcover=landcover,
                                    sar=sar, harmonics=harmonics, year=year)
    if cached is not None:
        return cached

    # Cache miss — need live API.  Block if processor is running.
    if _is_processor_running():
        log.info('Copernicus cache miss — skipping (processor running)')
        return None

    # --- Slow path: live openEO API ---
    try:
        import copernicus
        result = {}
        if ndvi:
            try:
                ndvi_data = copernicus.get_ndvi_composite(bbox_dict, year=year)
                result["ndvi"] = ndvi_data["ndvi"]
                result["transform"] = ndvi_data["transform"]
                result["crs"] = ndvi_data["crs"]
            except Exception as e:
                log.warning("Copernicus NDVI failed: %s", e)
        if landcover:
            try:
                lc = copernicus.get_land_cover(bbox_dict)
                result["landcover"] = lc
            except Exception as e:
                log.warning("Copernicus land cover failed: %s", e)
        if sar:
            try:
                sar_start = f"{year}-06-01"
                sar_end   = f"{year}-09-30"
                sar_data = copernicus.get_sar_backscatter(bbox_dict, sar_start, sar_end)
                result["vv"] = sar_data["vv"]
                result["vh"] = sar_data["vh"]
                result["sar_transform"] = sar_data["transform"]
                result["sar_crs"] = sar_data["crs"]
            except Exception as e:
                log.warning("Copernicus SAR failed: %s", e)
        if harmonics:
            try:
                import ndvi_harmonics
                harm = ndvi_harmonics.get_harmonic_features(bbox_dict, year=year)
                if harm:
                    result["harmonics"] = harm
                    log.info("NDVI harmonics: mean amp=%.3f",
                             float(np.nanmean(harm.get("h_amplitude", [0]))))
            except Exception as e:
                log.warning("NDVI harmonics failed: %s", e)
        return result if result else None
    except ImportError:
        log.info("Copernicus module not available")
        return None
    except Exception as e:
        log.warning("Copernicus data failed: %s", e)
        return None


def _try_copernicus_cached(bbox_dict: dict, *, ndvi=True, landcover=True,
                           sar=False, harmonics=False,
                           year: int = 2023) -> dict | None:
    """Serve Copernicus data from tile cache (local + Zenodo) only.

    Reads per-cell (0.1°) cached tiles and mosaics them if the request
    spans multiple cells.  Returns None on any miss so the caller can
    fall back to the live API.
    """
    try:
        from tile_cache import CopernicusTileCache
        cache = CopernicusTileCache()

        if not cache.has_cached(bbox_dict, ndvi=ndvi, landcover=landcover,
                                sar=sar, harmonics=harmonics, year=year):
            return None

        log.info("Copernicus serving from tile cache")
        result = {}
        if ndvi:
            d = cache.read_cached_product(bbox_dict, "ndvi", year=year)
            if d and "ndvi" in d:
                result["ndvi"] = d["ndvi"]
                result["transform"] = d["transform"]
                result["crs"] = d["crs"]
            else:
                return None
        if landcover:
            d = cache.read_cached_product(bbox_dict, "worldcover", year=year)
            if d:
                result["landcover"] = d
            else:
                return None
        if sar:
            d = cache.read_cached_product(bbox_dict, "sar", year=year)
            if d and "vv" in d:
                result["vv"] = d["vv"]
                result["vh"] = d["vh"]
                result["sar_transform"] = d["transform"]
                result["sar_crs"] = d["crs"]
            else:
                return None
        if harmonics:
            d = cache.read_cached_product(bbox_dict, "harmonics", year=year)
            if d:
                result["harmonics"] = d
            else:
                return None
        return result if result else None
    except Exception as e:
        log.debug("Copernicus cache read failed: %s", e)
        return None


def _try_cadastre(geom_wgs84, transform, shape) -> np.ndarray | None:
    """Attempt to fetch building footprints from cadastre."""
    try:
        import cadastre
        bbox = geom_wgs84.bounds
        return cadastre.get_building_mask(bbox, transform, shape)
    except ImportError:
        log.info("Cadastre module not available")
        return None
    except Exception as e:
        log.warning("Cadastre fetch failed: %s", e)
        return None


def _try_hansen(geom_wgs84, transform, shape) -> dict | None:
    """Attempt to load Hansen forest prior for segment_and_classify."""
    try:
        bbox = geom_wgs84.bounds
        return hansen.get_forest_prior(bbox, transform, shape)
    except Exception as e:
        log.warning("Hansen prior failed: %s", e)
        return None


def _clear_raster_caches():
    """Delete cached .npz / .tif / batch dirs to reclaim memory after training."""
    import pathlib, shutil
    cleared = 0
    for cache_dir in [
        pathlib.Path("/tmp/copernicus_cache"),
        pathlib.Path("/tmp/hansen_cache"),
    ]:
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                try:
                    if f.is_dir():
                        shutil.rmtree(f)
                    else:
                        f.unlink()
                    cleared += 1
                except Exception:
                    pass
    if cleared:
        log.info("Cleared %d cached raster entries after training", cleared)


# === SECTION: /api/v1/elevation endpoint ===

@app.route('/api/v1/elevation', methods=['POST'])
def elevation():
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)

        result_features = []
        for feat in features:
            geom = feat['geometry']
            geom_3035 = ti.geometry_to_3035(geom)

            if geom.geom_type == 'Point':
                e, n = geom_3035.coords[0][:2]
                bounds = (e - 5, n - 5, e + 5, n + 5)
                dsm_data, tf, _ = raster_io.read_window_bbox('DSM', *bounds, dataset, pad=0)
                dtm_data, _, _ = raster_io.read_window_bbox('DTM', *bounds, dataset, pad=0)
                row = max(0, min(int((tf.f - n) / abs(tf.e)), dsm_data.shape[0]-1))
                col = max(0, min(int((e - tf.c) / tf.a), dsm_data.shape[1]-1))
                dsm_val = round(float(dsm_data[row, col]), 2)
                dtm_val = round(float(dtm_data[row, col]), 2)
                props = dict(feat.get('properties', {}))
                props['dsm_elevation_m'] = dsm_val
                props['dtm_elevation_m'] = dtm_val
                props['object_height_m'] = round(float(dsm_val - dtm_val), 2)
                # Add DSM altitude as Z coordinate
                geom_z = Point(geom.x, geom.y, dsm_val)
                result_features.append({"type": "Feature", "properties": props, "geometry": mapping(geom_z)})

            elif geom.geom_type in ('LineString', 'MultiLineString'):
                _validate_area(geom_3035.buffer(10))
                coords_3035 = list(geom_3035.coords) if geom.geom_type == 'LineString' else \
                    [c for ls in geom_3035.geoms for c in ls.coords]
                bounds = geom_3035.bounds
                dsm_data, tf, _ = raster_io.read_window_bbox('DSM', *bounds, dataset)
                dtm_data, _, _ = raster_io.read_window_bbox('DTM', *bounds, dataset)
                enriched_coords = []
                coords_wgs_3d = []
                for e, n in coords_3035:
                    row = max(0, min(int((tf.f - n) / abs(tf.e)), dsm_data.shape[0]-1))
                    col = max(0, min(int((e - tf.c) / tf.a), dsm_data.shape[1]-1))
                    dsm_val = round(float(dsm_data[row, col]), 2)
                    dtm_val = round(float(dtm_data[row, col]), 2)
                    pt_wgs = ti.geometry_from_3035(Point(e, n))
                    enriched_coords.append({
                        "lon": round(pt_wgs.x, 8), "lat": round(pt_wgs.y, 8),
                        "dsm_elevation_m": dsm_val,
                        "dtm_elevation_m": dtm_val,
                        "object_height_m": round(float(dsm_val - dtm_val), 2),
                    })
                    coords_wgs_3d.append((round(pt_wgs.x, 8), round(pt_wgs.y, 8), dsm_val))
                props = dict(feat.get('properties', {}))
                props['elevation_profile'] = enriched_coords
                props['dsm_elevation_min'] = min(p['dsm_elevation_m'] for p in enriched_coords)
                props['dsm_elevation_max'] = max(p['dsm_elevation_m'] for p in enriched_coords)
                # Return geometry with DSM altitude as Z coordinate
                geom_z = SLineString(coords_wgs_3d)
                result_features.append({"type": "Feature", "properties": props, "geometry": mapping(geom_z)})

            else:
                _validate_area(geom_3035)
                data = raster_io.read_dtm_dsm(geom_3035, dataset)
                dtm_valid = data['dtm'][data['mask']]
                dsm_valid = data['dsm'][data['mask']]
                ndsm_valid = data['ndsm'][data['mask']]
                props = dict(feat.get('properties', {}))
                for name, arr in [('dsm_elevation', dsm_valid), ('dtm_elevation', dtm_valid), ('object_heights', ndsm_valid)]:
                    props[name] = {'min': round(float(np.nanmin(arr)), 2), 'max': round(float(np.nanmax(arr)), 2), 'mean': round(float(np.nanmean(arr)), 2)}
                props['area_sqm'] = int(np.sum(data['mask']))
                # Add mean DSM altitude as Z coordinate on polygon exterior
                dsm_mean = round(float(np.nanmean(dsm_valid)), 2)
                geom_dict = mapping(geom)
                if geom_dict.get('type') == 'Polygon' and geom_dict.get('coordinates'):
                    geom_dict['coordinates'] = tuple(
                        tuple((x, y, dsm_mean) for x, y, *_ in ring)
                        for ring in geom_dict['coordinates']
                    )
                result_features.append({"type": "Feature", "properties": props, "geometry": geom_dict})

        return jsonify({"type": "FeatureCollection", "features": result_features,
                        "meta": {"dataset": dataset, "processing_time_s": round(time.time()-t0, 2)}})
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# === SECTION: /api/v1/terrain endpoint ===

@app.route('/api/v1/terrain', methods=['POST'])
def terrain():
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        results = []
        for feat in features:
            geom = feat['geometry']
            geom_3035 = ti.geometry_to_3035(geom)
            _validate_area(geom_3035.buffer(10) if geom.geom_type == 'Point' else geom_3035)
            if geom.geom_type == 'Point':
                geom_3035 = geom_3035.buffer(50)
            dtm_data, mask, tf, crs = raster_io.read_masked('DTM', geom_3035, dataset)
            terrain_stats = ta.characterise_terrain(dtm_data, mask)
            props = dict(feat.get('properties', {}))
            props['terrain'] = terrain_stats
            results.append({"type": "Feature", "properties": props, "geometry": mapping(feat['geometry'])})
        return jsonify({"type": "FeatureCollection", "features": results,
                        "meta": {"dataset": dataset, "processing_time_s": round(time.time()-t0, 2)}})
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)





# === SECTION: /api/v1/segment endpoint (async segmentation pipeline) ===

@app.route('/api/v1/segment', methods=['POST'])
def segment_objects():
    """Watershed-based object segmentation and classification.

    Fused gradient → Felzenszwalb → RAG merge → per-object features → classify → group.
    Returns individual objects (tree, roof, road_surface, …) AND groups (forest, building, …).

    When async=true is passed, returns 202 immediately with {task_id, status: 'running'}.
    Poll /api/v1/segment/progress?task_id=... for status.
    Fetch /api/v1/segment/result?task_id=... when done.
    """
    task_id = request.args.get('task_id', '')
    run_async = str(request.args.get('async', 'false')).lower() in ('true', '1', 'yes')

    # Parse request data upfront (must happen in request context)
    try:
        features = _get_geometry()
        params = _get_params()
    except Exception as e:
        return _error(str(e))

    # Capture raw geometry text for auto-save share recovery
    geometry_text = ''
    try:
        from shapely.geometry import mapping
        if features:
            geom_dict = mapping(features[0]['geometry'])
            geometry_text = json.dumps(geom_dict)
    except Exception:
        pass

    if run_async:
        if not task_id:
            task_id = str(uuid.uuid4())
        # Queue check: reject early if too many waiting
        with _TASK_QUEUE_LOCK:
            if _TASK_QUEUE_SIZE >= MAX_QUEUE_SIZE:
                return _error('Server busy — too many tasks queued. Try again in a few minutes.', 503)
        _progress_start(task_id)
        _cleanup_old_results()
        thread = threading.Thread(
            target=_segment_worker,
            args=(task_id, features, params, geometry_text),
            daemon=True,
        )
        thread.start()
        return jsonify({"task_id": task_id, "status": "running", "queued": _TASK_QUEUE_SIZE}), 202

    return _segment_sync(task_id, features, params)


def _segment_worker(task_id: str, features: list, params: dict, geometry_text: str = ''):
    """Background worker for async segment processing (queue-aware)."""
    global _TASK_QUEUE_SIZE
    with _TASK_QUEUE_LOCK:
        _TASK_QUEUE_SIZE += 1
    _progress_set(task_id, 'queued', f'Waiting for slot ({_TASK_QUEUE_SIZE} in queue)…')
    try:
        acquired = _TASK_SEMAPHORE.acquire(timeout=300)  # wait up to 5 min for a slot
        with _TASK_QUEUE_LOCK:
            _TASK_QUEUE_SIZE = max(0, _TASK_QUEUE_SIZE - 1)
        if not acquired:
            _progress_error(task_id, 'Server busy — timed out waiting in queue. Try again later.')
            return
        try:
            resp = _segment_core(task_id, features, params)
            _store_result(task_id, resp)
            import gc; gc.collect()
            share_id = _auto_save_share(task_id, resp, geometry_text, params)
            _progress_done(task_id, auto_share_id=share_id)
            log.info("Async segment task %s completed (auto-share=%s)", task_id, share_id)
        except Exception as e:
            log.error("Async segment task %s failed: %s", task_id, traceback.format_exc())
            _progress_error(task_id, str(e))
        finally:
            _TASK_SEMAPHORE.release()
    except Exception as e:
        with _TASK_QUEUE_LOCK:
            _TASK_QUEUE_SIZE = max(0, _TASK_QUEUE_SIZE - 1)
        _progress_error(task_id, str(e))


def _unique_share_id(base_name: str) -> str:
    """Return base_name if unused, otherwise append date + counter (e.g. MyArea-0415-2)."""
    candidate = base_name
    if not (SHARE_DIR / f"{candidate}.json.gz").exists():
        return candidate
    # Collision — append date stamp
    date_suffix = time.strftime('%m%d')
    candidate = f"{base_name}-{date_suffix}"
    if not (SHARE_DIR / f"{candidate}.json.gz").exists():
        return candidate
    # Still collides — add counter
    for i in range(2, 100):
        candidate = f"{base_name}-{date_suffix}-{i}"
        if not (SHARE_DIR / f"{candidate}.json.gz").exists():
            return candidate
    # Fallback
    return f"{base_name}-{date_suffix}-{int(time.time()) % 10000}"


def _auto_save_share(task_id: str, result: dict, geometry_text: str, params: dict):
    """Auto-save completed analysis as a share for recovery."""
    try:
        # Check if onestop meta specifies a custom save name
        custom_name = None
        onestop_meta_path = _PROGRESS_DIR / f"{task_id}.onestop.json"
        if onestop_meta_path.exists():
            try:
                meta = json.loads(onestop_meta_path.read_text())
                custom_name = meta.get('params', {}).get('name', '')
                if custom_name:
                    custom_name = custom_name.strip().strip("'").strip('"')
            except Exception:
                pass
        if custom_name and _valid_share_id(custom_name):
            share_id = _unique_share_id(custom_name)
        else:
            share_id = f"auto-{task_id[:8]}"
        state = {
            'v': 1, 'center': [47.3, 15.3], 'zoom': 14,
            'endpoint': 'segment',
            'min_area': int(params.get('min_object_size', 30)),
            'ortho': str(params.get('include_ortho', 'true')).lower() in ('true', '1', 'yes'),
            'temporal': str(params.get('include_temporal', 'false')).lower() in ('true', '1', 'yes'),
            'copernicus': str(params.get('include_copernicus', 'false')).lower() in ('true', '1', 'yes'),
            'cadastre': str(params.get('include_cadastre', 'false')).lower() in ('true', '1', 'yes'),
            'hansen': str(params.get('include_hansen', 'false')).lower() in ('true', '1', 'yes'),
            'mark_uncertain': str(params.get('mark_uncertain', 'false')).lower() in ('true', '1', 'yes'),
            'geometry': geometry_text,
        }
        # Compute center from result bbox if available
        if result.get('features'):
            lats, lngs = [], []
            for f in result['features'][:100]:
                coords = f.get('geometry', {}).get('coordinates')
                if coords:
                    def _extract(c):
                        if isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (int, float)):
                            lngs.append(c[0]); lats.append(c[1])
                        elif isinstance(c, (list, tuple)):
                            for x in c: _extract(x)
                    _extract(coords)
            if lats and lngs:
                state['center'] = [round(sum(lats)/len(lats), 6), round(sum(lngs)/len(lngs), 6)]
                state['zoom'] = 15

        name = custom_name if custom_name and _valid_share_id(custom_name) else f"Auto-save {time.strftime('%Y-%m-%d %H:%M')}"
        payload = {'state': state, 'result': result, 'name': name}

        # Tag one-stop shares with their onestop params for direct-download UX
        onestop_meta_path = _PROGRESS_DIR / f"{task_id}.onestop.json"
        if onestop_meta_path.exists():
            try:
                payload['onestop'] = json.loads(onestop_meta_path.read_text())
            except Exception:
                pass
        data_json = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        data = gzip.compress(data_json.encode())

        SHARE_DIR.mkdir(parents=True, exist_ok=True)
        (SHARE_DIR / f'{share_id}.json.gz').write_bytes(data)
        _share_evict()
        log.info("auto-save: saved share %s (%d KB)", share_id, len(data) // 1024)
        return share_id
    except Exception as e:
        log.error("auto-save failed: %s", e)
        return None


def _segment_sync(task_id: str, features: list, params: dict):
    """Synchronous segment processing (original behavior)."""
    try:
        resp = _segment_core(task_id, features, params)
        _progress_end(task_id)
        return jsonify(resp)
    except ValueError as e:
        _progress_end(task_id)
        return _error(str(e))
    except Exception as e:
        _progress_end(task_id)
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


def _segment_core(task_id: str, features: list, params: dict) -> dict:
    """Core segment processing logic. Returns response dict."""
    def _prog(step, detail=''):
        if task_id:
            _progress_set(task_id, step, detail)

    t0 = time.time()
    if task_id:
        _progress_start(task_id)
    _prog('Parsing geometry')
    dataset = params.get('dataset', ti.DEFAULT_DATASET)
    min_object_size = int(params.get('min_object_size', 30))
    felz_scale = float(params.get('felz_scale', 150))
    rag_threshold = float(params.get('rag_threshold', 0.12))
    include_ortho = str(params.get('include_ortho', 'true')).lower() in ('true', '1', 'yes')
    include_temporal = str(params.get('include_temporal', 'false')).lower() in ('true', '1', 'yes')
    include_copernicus = str(params.get('include_copernicus', 'false')).lower() in ('true', '1', 'yes')
    include_cadastre = str(params.get('include_cadastre', 'false')).lower() in ('true', '1', 'yes')
    include_hansen = str(params.get('include_hansen', 'false')).lower() in ('true', '1', 'yes')
    include_infra = str(params.get('include_infra', 'true')).lower() in ('true', '1', 'yes')
    mark_uncertain = str(params.get('mark_uncertain', 'false')).lower() in ('true', '1', 'yes')
    type_filter = params.get('types', None)
    if isinstance(type_filter, str):
        type_filter = [t.strip() for t in type_filter.split(',')]
    group_filter = params.get('groups', None)
    if isinstance(group_filter, str):
        group_filter = [g.strip() for g in group_filter.split(',')]

    all_objects = []
    all_stats = None
    all_evaluation = None
    all_labels = None
    all_transform = None
    all_shape = None
    all_mask = None
    hansen_evaluation = None

    for feat in features:
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(100)
        _validate_area(geom_3035)

        # Load DTM/DSM
        _prog('Loading DTM/DSM', 'remote raster reads')
        dtm_dates, dsm_dates = None, None
        if include_temporal:
            _prog('Loading DTM/DSM', 'multi-temporal (3 dates)')
            try:
                multi = raster_io.read_multi_date_ndsm(geom_3035)
                data = {
                    'dtm': multi['dtm'], 'dsm': multi['dsm'],
                    'ndsm': multi['ndsm'], 'mask': multi['mask'],
                    'transform': multi['transform'], 'crs': multi['crs'],
                    'shape': multi['shape'],
                }
                dtm_dates, dsm_dates = {}, {}
                for d in multi['dates_loaded']:
                    try:
                        _prog('Loading DTM/DSM', f'date {d}')
                        dd = raster_io.read_dtm_dsm(geom_3035, dataset=d)
                        mh = min(dd['shape'][0], data['shape'][0])
                        mw = min(dd['shape'][1], data['shape'][1])
                        dtm_dates[d] = dd['dtm'][:mh, :mw]
                        dsm_dates[d] = dd['dsm'][:mh, :mw]
                    except Exception as e:
                        log.warning("Date %s load failed: %s", d, e)
            except Exception as e:
                log.warning("Multi-temporal failed, single date: %s", e)
                data = raster_io.read_dtm_dsm(geom_3035, dataset)
        else:
            data = raster_io.read_dtm_dsm(geom_3035, dataset)

        rgb, spectral = (None, None)
        if include_ortho:
            _prog('Loading orthophoto', 'RGBI 20cm')
            rgb, spectral = _try_read_ortho(data)

        copernicus_data = None
        if include_copernicus:
            _prog('Loading Copernicus', 'NDVI + landcover + SAR + harmonics')
            copernicus_data = _try_copernicus(geom, sar=True, harmonics=True, year=ti.dataset_to_year(dataset))

        building_footprints = None
        if include_cadastre:
            _prog('Loading cadastre', 'building footprints')
            building_footprints = _try_cadastre(
                geom, data['transform'], data['shape'],
            )

        hansen_data = None
        if include_hansen:
            _prog('Loading Hansen', 'forest change data')
            hansen_data = _try_hansen(geom, data['transform'], data['shape'])

        # Infrastructure lookup (for rule-based solar/wind/substation/mast)
        _infra_lookup = None
        if include_infra:
            try:
                from infrastructure_lookup import InfrastructureLookup
                from pyproj import Transformer as _Tx
                _tx = _Tx.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
                bounds_3035 = geom.bounds  # (minx, miny, maxx, maxy) in EPSG:3035
                w4, s4 = _tx.transform(bounds_3035[0], bounds_3035[1])
                e4, n4 = _tx.transform(bounds_3035[2], bounds_3035[3])
                _infra_lookup = InfrastructureLookup.for_bbox(w4, s4, e4, n4)
                log.info('Infrastructure lookup: %d features', len(_infra_lookup))
            except Exception as _ie:
                log.warning('Infrastructure lookup failed: %s', _ie)

        # Run segmentation pipeline
        _prog('Segmenting & classifying', 'watershed + classification')
        obs_year = ti.dataset_to_year(dataset)
        result = seg.segment_and_classify(
            data['dtm'], data['dsm'], data['mask'], data['transform'],
            dtm_dates=dtm_dates,
            dsm_dates=dsm_dates,
            spectral=spectral,
            copernicus=copernicus_data,
            building_footprints=building_footprints,
            hansen=hansen_data,
            min_object_size=min_object_size,
            felz_scale=felz_scale,
            rag_threshold=rag_threshold,
            observation_year=obs_year,
            infra_lookup=_infra_lookup,
            mark_uncertain=mark_uncertain,
        )

        objects = result['objects']
        labels = result['labels']

        # Free heavy intermediates that segment_and_classify already consumed
        del dtm_dates, dsm_dates, spectral, copernicus_data, building_footprints
        import gc; gc.collect()

        # Hansen forest loss calibration
        hansen_evaluation = None
        if include_hansen and hansen_data:
            try:
                objects = hansen.calibrate_tree_loss(objects, labels, hansen_data, observation_year=obs_year)
                hansen_evaluation = hansen.evaluate_forest_loss(objects, labels, hansen_data, observation_year=obs_year)
            except Exception as e:
                log.warning("Hansen calibration failed: %s", e)

        # Populate seg_cache so overlay/gpkg endpoints can reuse results
        seg_cache_key = f"{geom_3035.bounds}_{dataset}_{include_ortho}_{include_copernicus}_{include_cadastre}_{include_hansen}_{include_infra}_{mark_uncertain}_temporal"
        _seg_cache.update({
            "labels": labels, "objects": objects,
            "mask": data['mask'], "transform": data['transform'],
            "shape": data['shape'], "ndsm": data.get('ndsm'), "key": seg_cache_key,
        })
        # Populate raster data cache so overlay endpoints don't re-fetch
        raster_cache_key = f"{geom_3035.bounds}_{dataset}"
        _raster_cache.update({"key": raster_cache_key, "data": data})
        if rgb is not None:
            _raster_cache.update({"ortho": rgb, "ortho_key": raster_cache_key})
        log.info("segment: cached results for overlay reuse")
        # Persist to disk so other gunicorn workers can reuse
        _seg_cache_save(seg_cache_key, labels, objects, data['mask'],
                        data['transform'], data['shape'], data.get('ndsm'))

        # Filters
        if type_filter:
            objects = [o for o in objects if o.obj_type in type_filter]
        if group_filter:
            objects = [o for o in objects if o.group_type in group_filter]

        # top_n_classes: keep only top N most frequent types
        top_n_classes = params.get('top_n_classes')
        if top_n_classes:
            top_n_classes = int(top_n_classes)
            from collections import Counter
            type_counts = Counter(o.obj_type for o in objects)
            top_types = set(t for t, _ in type_counts.most_common(top_n_classes))
            objects = [o for o in objects if o.obj_type in top_types]

        # top_n_objects: keep only the N tallest objects
        top_n_objects = params.get('top_n_objects')
        if top_n_objects:
            top_n_objects = int(top_n_objects)
            objects = sorted(objects, key=lambda o: o.height_max, reverse=True)[:top_n_objects]

        # min_height_m: keep only objects taller than Y metres
        min_height_m = params.get('min_height_m')
        if min_height_m:
            min_height_m = float(min_height_m)
            objects = [o for o in objects if o.height_max >= min_height_m]

        all_objects.extend(objects)
        all_stats = result.get('stats')
        all_labels = labels
        all_transform = data['transform']
        all_shape = data['shape']
        all_mask = data['mask']
        if result.get('evaluation'):
            all_evaluation = result['evaluation']

    # Build GeoJSON response
    obj_features = []
    for obj in all_objects:
        centroid_wgs = ti.geometry_from_3035(Point(obj.centroid_e, obj.centroid_n))
        props = {
            "id": obj.obj_id,
            "type": obj.obj_type,
            "type_code": obj.type_code,
            "group_id": obj.group_id,
            "group_type": obj.group_type,
            "height_max_m": obj.height_max,
            "height_mean_m": obj.height_mean,
            "height_p90_m": obj.height_p90,
            "area_sqm": obj.area_sqm,
            "compactness": obj.compactness,
            "elongation": obj.elongation,
            "solidity": obj.solidity,
            "extent": obj.extent,
            "dsm_edge_strength": obj.dsm_edge_strength,
            "slope_mean": obj.slope_mean,
            "aspect_mean": obj.aspect_mean,
            "aspect_dominant": obj.aspect_dominant,
            "roughness": obj.roughness,
            "elevation_mean": obj.elevation_mean,
            "elevation_min": obj.elevation_min,
            "elevation_max": obj.elevation_max,
            "tri_mean": obj.tri_mean,
            "tpi_mean": obj.tpi_mean,
            "curvature_mean": obj.curvature_mean,
            "terrain_class": obj.terrain_class,
            "is_manmade": obj.is_manmade,
            "confidence": obj.confidence,
            "classifier_source": obj.classifier_source,
            "rf_type": obj.rf_type,
            "rf_confidence": obj.rf_confidence,
        }
        if include_ortho or include_copernicus:
            props["ndvi_mean"] = obj.ndvi_mean
            props["ndvi_fused"] = obj.ndvi_fused
            props["brightness_mean"] = obj.brightness_mean
            props["nir_mean"] = obj.nir_mean
        if include_temporal:
            props["height_change"] = obj.height_change
            props["dtm_change"] = obj.dtm_change
            props["temporal_stability"] = obj.temporal_stability
            props["volume_change_m3"] = obj.volume_change_m3
            props["volume_change_abs_m3"] = obj.volume_change_abs_m3
            props["dtm_change_max"] = obj.dtm_change_max
        # Texture features
        if obj.glcm_entropy > 0:
            props["glcm_entropy"] = obj.glcm_entropy
            props["glcm_homogeneity"] = obj.glcm_homogeneity
            props["texture_complexity"] = obj.texture_complexity
        # SAR features
        if obj.sar_vv > 0:
            props["sar_vv"] = obj.sar_vv
            props["sar_vh"] = obj.sar_vh
        # Phenology features
        if obj.harm_amplitude > 0:
            props["harm_amplitude"] = obj.harm_amplitude
            props["harm_phase"] = obj.harm_phase
            props["phenology_class"] = obj.phenology_class
        obj_features.append({
            "type": "Feature",
            "properties": props,
            "geometry": mapping(centroid_wgs),
        })

    resp = {
        "type": "FeatureCollection",
        "features": obj_features,
        "stats": all_stats,
        "meta": {
            "classifier": "watershed_v1",
            "pipeline": "Sobel→Felzenszwalb→RAG→classify→group",
            "dataset": dataset,
            "min_object_size": min_object_size,
            "felz_scale": felz_scale,
            "rag_threshold": rag_threshold,
            "include_ortho": include_ortho,
            "include_temporal": include_temporal,
            "include_copernicus": include_copernicus,
            "include_cadastre": include_cadastre,
            "include_hansen": include_hansen,
            "include_infra": include_infra,
            "mark_uncertain": mark_uncertain,
            "processing_time_s": round(time.time() - t0, 2),
            **_rf_model_meta(),
        },
    }
    if all_evaluation:
        resp["cadastre_evaluation"] = all_evaluation
    if hansen_evaluation:
        resp["hansen_evaluation"] = hansen_evaluation
    if include_copernicus and copernicus_data is None and _is_processor_running():
        resp.setdefault('warnings', []).append(
            'Copernicus data not in cache — served without Sentinel-2/SAR (processor running)')

    return resp


# === SECTION: /api/v1/segment/overlay + raster rendering ===

# Cache for last segmentation result so legend filter re-renders are instant
_seg_cache = {"labels": None, "objects": None, "mask": None,
              "transform": None, "shape": None, "ndsm": None, "key": None}

# File-backed seg cache so all gunicorn workers share segmentation results
_SEG_CACHE_DIR = Path('/tmp/seg_cache')
_SEG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _seg_cache_path(cache_key: str) -> Path:
    """Deterministic filename for a seg cache key."""
    h = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    return _SEG_CACHE_DIR / f"seg_{h}.pkl"

def _seg_cache_save(cache_key: str, labels, objects, mask, transform, shape_hw, ndsm):
    """Persist seg cache to disk for cross-worker reuse.

    Snapshots all data into a single dict up-front (in the calling thread),
    then pickles + writes to disk in a background thread so the API response
    is not blocked.
    """
    # Capture everything NOW — numpy arrays are refcounted, not copied
    payload = {
        'key': cache_key,
        'labels': labels,
        'objects': objects,
        'mask': mask,
        'transform': tuple(transform)[:6] if hasattr(transform, '__iter__') else transform,
        'shape': shape_hw,
        'ndsm': ndsm,
        'ts': time.time(),
    }

    def _do_save():
        try:
            p = _seg_cache_path(cache_key)
            tmp = p.with_suffix('.tmp')
            with open(tmp, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.rename(p)
            log.info("seg_cache: saved to disk (%s, %.1f MB)", p.name, p.stat().st_size / 1e6)
            # Evict old entries (keep at most 3)
            entries = sorted(_SEG_CACHE_DIR.glob('seg_*.pkl'), key=lambda f: f.stat().st_mtime)
            for old in entries[:-3]:
                old.unlink(missing_ok=True)
        except Exception as e:
            log.warning("seg_cache: disk save failed: %s", e)

    threading.Thread(target=_do_save, daemon=True).start()


def _seg_cache_load(cache_key: str):
    """Try to load seg cache from disk. Returns dict or None."""
    try:
        p = _seg_cache_path(cache_key)
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > 3600:
            p.unlink(missing_ok=True)
            return None
        with open(p, 'rb') as f:
            data = pickle.load(f)
        if data.get('key') != cache_key:
            return None
        # Restore transform as rasterio Affine
        from rasterio.transform import Affine
        t = data['transform']
        if isinstance(t, (list, tuple)):
            data['transform'] = Affine(*t[:6])
        log.info("seg_cache: loaded from disk (%s)", p.name)
        return data
    except Exception as e:
        log.warning("seg_cache: disk load failed: %s", e)
        return None


def _seg_cache_scan(cache_key_substring: str):
    """Scan all disk cache files for one whose key contains the substring.

    Used by GeoPackage/MBTiles which match on a partial key (bounds+dataset).
    Returns dict with labels/objects/mask/ndsm or None.
    """
    for p in sorted(_SEG_CACHE_DIR.glob('seg_*.pkl'),
                    key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            if time.time() - p.stat().st_mtime > 3600:
                continue
            with open(p, 'rb') as f:
                data = pickle.load(f)
            if data.get('key') and cache_key_substring in data['key']:
                from rasterio.transform import Affine
                t = data['transform']
                if isinstance(t, (list, tuple)):
                    data['transform'] = Affine(*t[:6])
                log.info("seg_cache: scan hit (%s)", p.name)
                return data
        except Exception:
            pass
    return None

# Cache for raster data (DTM/DSM/nDSM/ortho) keyed by (bounds, dataset)
# so overlay endpoints don't re-fetch from remote after segment has loaded them
_raster_cache = {"key": None, "data": None, "ortho": None, "ortho_key": None}

# Segment type → RGBA colour (matches frontend TYPE_SHAPES)
SEGMENT_COLORS = {
    "tree":         (0, 100, 0, 180),
    "shrub":        (34, 139, 34, 180),
    "grass":        (124, 252, 0, 150),
    "hedge":        (46, 139, 87, 170),
    "water":        (30, 144, 255, 180),
    "roof":         (220, 20, 60, 200),
    "greenhouse":   (255, 105, 180, 180),
    "solar_panel":  (65, 105, 225, 200),
    "fence":        (160, 82, 45, 170),
    "wall":         (139, 69, 19, 170),
    "mast":         (64, 64, 64, 200),
    "wind_turbine": (21, 101, 192, 200),
    "substation":   (255, 111, 0, 200),
    "road":         (128, 128, 128, 160),
    "path":         (169, 169, 169, 150),
    "parking":      (105, 105, 105, 160),
    "bridge":       (112, 128, 144, 170),
    "crop":         (218, 165, 32, 160),
    "orchard":      (107, 142, 35, 170),
    "vineyard":     (147, 112, 219, 170),
    "garden":       (60, 179, 113, 160),
    "bare_soil":    (210, 180, 140, 140),
    "rock":         (139, 134, 130, 160),
    "excavation":   (139, 0, 0, 200),
    "fill":         (255, 140, 0, 200),
    "tree_loss":    (255, 0, 255, 200),
    "construction": (255, 69, 0, 200),
    "unclassified": (190, 190, 190, 120),
}


def _viridis_rgb(t):
    """Return (R,G,B) for t in [0,1] on the viridis scale."""
    VIRIDIS = [
        (68,1,84),(72,35,116),(64,67,135),(52,94,141),(41,120,142),
        (32,144,140),(34,167,132),(68,190,112),(121,209,81),(189,222,38),(253,231,37)
    ]
    t = max(0.0, min(1.0, t))
    idx = t * (len(VIRIDIS) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(VIRIDIS) - 1)
    f = idx - lo
    return tuple(int(VIRIDIS[lo][c] + f * (VIRIDIS[hi][c] - VIRIDIS[lo][c])) for c in range(3))


def _diverging_rgb(t):
    """Blue-white-red diverging scale, t in [-1,1] mapped to [0,1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        # Blue to white
        f = t * 2
        return (int(33 + f * 222), int(102 + f * 153), int(172 + f * 83))
    else:
        # White to red
        f = (t - 0.5) * 2
        return (int(255 - f * 37), int(255 - f * 192), int(255 - f * 192))


def _segment_rgba(labels, objects, mask, type_filter=None, color_mode='type', ndsm=None, type_overrides=None, height_filter=None):
    """Render segmentation labels as RGBA image.

    color_mode: 'type' = categorical colors, 'height' = viridis by height
    type_overrides: optional dict {obj_id: type_name} to override object types
                    (used when rendering a share's stored result over cached labels)
    """
    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    obj_map = {o.obj_id: o for o in objects}

    def _effective_type(obj_id, obj):
        if type_overrides and obj_id in type_overrides:
            return type_overrides[obj_id]
        return obj.obj_type

    if color_mode == 'height' and ndsm is not None:
        # Per-pixel viridis coloring from actual nDSM values
        included = np.zeros((h, w), dtype=bool)
        for obj_id, obj in obj_map.items():
            etype = _effective_type(obj_id, obj)
            if type_filter and etype not in type_filter:
                continue
            if height_filter and not height_filter(obj.height_max):
                continue
            included |= (labels == obj_id)
        # Build viridis LUT (256 entries)
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            lut[i] = _viridis_rgb(i / 255.0)
        idx = np.clip((np.clip(np.sqrt(np.clip(ndsm, 0, 45) / 45.0), 0, 1) * 255).astype(np.uint8), 0, 255)
        for c in range(3):
            rgba[:, :, c] = lut[idx, c]
        rgba[:, :, 3] = np.where(included & mask, np.where(ndsm > 0.3, 180, 60).astype(np.uint8), 0)
    else:
        for obj_id, obj in obj_map.items():
            etype = _effective_type(obj_id, obj)
            if type_filter and etype not in type_filter:
                continue
            if height_filter and not height_filter(obj.height_max):
                continue
            seg_mask = labels == obj_id
            color = SEGMENT_COLORS.get(etype, (128, 128, 128, 120))
            for c in range(4):
                rgba[:, :, c][seg_mask] = color[c]

    # Transparent where no data
    rgba[:, :, 3][~mask] = 0
    return rgba


def _render_seg_overlay(labels, objects, mask, transform, shape_hw, type_filter=None, color_mode='type', ndsm=None, type_overrides=None):
    """Render segmentation as RGBA, reproject to WGS84, return overlay response."""
    from rasterio.warp import calculate_default_transform, reproject as rp, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds

    rgba_3035 = _segment_rgba(labels, objects, mask, type_filter, color_mode=color_mode, ndsm=ndsm, type_overrides=type_overrides)

    src_crs = CRS.from_epsg(3035)
    dst_crs = CRS.from_epsg(4326)
    h, w = shape_hw
    dst_tf, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, w, h, *array_bounds(h, w, transform),
    )
    rgba_wgs = np.zeros((4, dst_h, dst_w), dtype=np.uint8)
    for band in range(4):
        rp(
            source=rgba_3035[:, :, band],
            destination=rgba_wgs[band],
            src_transform=transform,
            src_crs=src_crs,
            dst_transform=dst_tf,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    rgba_out = np.transpose(rgba_wgs, (1, 2, 0))  # (H,W,4)
    bounds = array_bounds(dst_h, dst_w, dst_tf)
    bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])  # south,west,north,east
    return _send_rgba_overlay(rgba_out, bounds_wgs)


def _load_share_type_overrides(share_id: str) -> dict | None:
    """Load a share's features and return {obj_id: type_name} mapping.

    This lets the overlay endpoint render with the share's stored classification
    instead of whatever the current model or cache contains.
    """
    try:
        resolved_id, path = _resolve_share(share_id)
        if not path or not path.exists():
            log.warning("share type overrides: share %s not found", share_id)
            return None
        raw = gzip.decompress(path.read_bytes())
        share_data = json.loads(raw)
        result = share_data.get('result', {})
        features = result.get('features', [])
        if not features:
            return None
        overrides = {}
        for f in features:
            props = f.get('properties', {})
            obj_id = props.get('id')
            obj_type = props.get('type')
            if obj_id is not None and obj_type:
                overrides[int(obj_id)] = obj_type
        return overrides if overrides else None
    except Exception as e:
        log.warning("share type overrides: failed for %s: %s", share_id, e)
        return None


@app.route('/api/v1/segment/overlay', methods=['POST'])
def segment_overlay():
    """Return segment classification as a coloured PNG overlay (reprojected to WGS84).

    First call runs full segmentation and caches the result.
    Subsequent calls with only ?types= changed use the cache for instant re-renders.

    When share_id is provided, the overlay uses the share's stored type assignments
    instead of the current model's classification, so historical shares render correctly.
    """
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        include_ortho = str(params.get('include_ortho', 'true')).lower() in ('true', '1', 'yes')
        include_copernicus = str(params.get('include_copernicus', 'false')).lower() in ('true', '1', 'yes')
        include_cadastre = str(params.get('include_cadastre', 'false')).lower() in ('true', '1', 'yes')
        include_hansen = str(params.get('include_hansen', 'false')).lower() in ('true', '1', 'yes')
        include_infra = str(params.get('include_infra', 'true')).lower() in ('true', '1', 'yes')
        mark_uncertain = str(params.get('mark_uncertain', 'false')).lower() in ('true', '1', 'yes')
        type_filter_str = params.get('types', None)
        type_filter = None
        if type_filter_str:
            type_filter = set(t.strip() for t in type_filter_str.split(','))
        color_mode = params.get('color_mode', 'type')  # 'type' or 'height'
        top_n_classes = params.get('top_n_classes')
        share_id = params.get('share_id')

        # Build type overrides from share's stored result features
        type_overrides = None
        if share_id:
            type_overrides = _load_share_type_overrides(share_id)
            if type_overrides:
                log.info("segment overlay: using type overrides from share %s (%d mappings)",
                         share_id, len(type_overrides))

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(100)
        _validate_area(geom_3035)

        # Build a cache key from geometry bounds + dataset + analysis options
        cache_key = f"{geom_3035.bounds}_{dataset}_{include_ortho}_{include_copernicus}_{include_cadastre}_{include_hansen}_{include_infra}_{mark_uncertain}_temporal"

        # When share_id is provided, try to find ANY cached labels for this geometry
        # (regardless of analysis options) — we only need the pixel labels, not the types
        if type_overrides:
            bounds_prefix = str(geom_3035.bounds)
            cached = None
            # Check in-process cache first (any key with same bounds)
            if _seg_cache["labels"] is not None and _seg_cache.get("key", "").startswith(bounds_prefix):
                cached = _seg_cache
            if cached is None:
                # Try exact key on disk
                cached = _seg_cache_load(cache_key)
            if cached is None:
                # Scan disk caches for any entry with matching geometry bounds
                cached = _seg_cache_scan(bounds_prefix)
            if cached is not None:
                log.info("segment overlay: rendering with share type overrides (share=%s)", share_id)
                return _render_seg_overlay(
                    cached["labels"], cached["objects"],
                    cached["mask"], cached["transform"],
                    cached["shape"], type_filter, color_mode,
                    ndsm=cached.get("ndsm"), type_overrides=type_overrides,
                )
            # Fall through to full pipeline if no cached labels found
            log.info("segment overlay: no cached labels for share %s, running full pipeline", share_id)

        if _seg_cache["key"] == cache_key:
            # Re-render from cache — instant
            log.info("segment overlay: re-render from cache (filter=%s)", type_filter)
            if top_n_classes:
                from collections import Counter
                _tn = int(top_n_classes)
                _tc = Counter(o.obj_type for o in _seg_cache["objects"])
                _top = set(t for t, _ in _tc.most_common(_tn))
                type_filter = (type_filter & _top) if type_filter else _top
            return _render_seg_overlay(
                _seg_cache["labels"], _seg_cache["objects"],
                _seg_cache["mask"], _seg_cache["transform"],
                _seg_cache["shape"], type_filter, color_mode,
                ndsm=_seg_cache.get("ndsm"), type_overrides=type_overrides,
            )

        # Check file-backed cache (shared across gunicorn workers)
        _disk = _seg_cache_load(cache_key)
        if _disk is not None:
            log.info("segment overlay: loaded from disk cache (filter=%s)", type_filter)
            # Populate in-process cache for subsequent re-renders
            _seg_cache.update({
                "labels": _disk["labels"], "objects": _disk["objects"],
                "mask": _disk["mask"], "transform": _disk["transform"],
                "shape": _disk["shape"], "ndsm": _disk.get("ndsm"), "key": cache_key,
            })
            if top_n_classes:
                from collections import Counter
                _tn = int(top_n_classes)
                _tc = Counter(o.obj_type for o in _disk["objects"])
                _top = set(t for t, _ in _tc.most_common(_tn))
                type_filter = (type_filter & _top) if type_filter else _top
            return _render_seg_overlay(
                _disk["labels"], _disk["objects"],
                _disk["mask"], _disk["transform"],
                _disk["shape"], type_filter, color_mode,
                ndsm=_disk.get("ndsm"), type_overrides=type_overrides,
            )

        # Full segmentation pipeline
        # Always load temporal data for stability-based building detection
        dtm_dates, dsm_dates = None, None
        try:
            multi = raster_io.read_multi_date_ndsm(geom_3035)
            data = {
                'dtm': multi['dtm'], 'dsm': multi['dsm'],
                'ndsm': multi['ndsm'], 'mask': multi['mask'],
                'transform': multi['transform'], 'crs': multi['crs'],
                'shape': multi['shape'],
            }
            dtm_dates, dsm_dates = {}, {}
            for d in multi['dates_loaded']:
                try:
                    dd = raster_io.read_dtm_dsm(geom_3035, dataset=d)
                    mh = min(dd['shape'][0], data['shape'][0])
                    mw = min(dd['shape'][1], data['shape'][1])
                    dtm_dates[d] = dd['dtm'][:mh, :mw]
                    dsm_dates[d] = dd['dsm'][:mh, :mw]
                except Exception as e:
                    log.warning("overlay: date %s load failed: %s", d, e)
        except Exception as e:
            log.warning("overlay: multi-temporal failed, single date: %s", e)
            data = raster_io.read_dtm_dsm(geom_3035, dataset)

        rgb, spectral = (None, None)
        if include_ortho:
            rgb, spectral = _try_read_ortho(data)

        copernicus_data = None
        if include_copernicus:
            copernicus_data = _try_copernicus(geom, sar=True, harmonics=True, year=ti.dataset_to_year(dataset))

        building_footprints = None
        if include_cadastre:
            building_footprints = _try_cadastre(geom, data['transform'], data['shape'])

        hansen_data = None
        if include_hansen:
            hansen_data = _try_hansen(geom, data['transform'], data['shape'])

        _infra_lookup = None
        if include_infra:
            try:
                from infrastructure_lookup import InfrastructureLookup
                from pyproj import Transformer as _Tx
                _tx = _Tx.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
                b = geom.bounds
                w4, s4 = _tx.transform(b[0], b[1])
                e4, n4 = _tx.transform(b[2], b[3])
                _infra_lookup = InfrastructureLookup.for_bbox(w4, s4, e4, n4)
            except Exception:
                pass

        obs_year = ti.dataset_to_year(dataset)
        result = seg.segment_and_classify(
            data['dtm'], data['dsm'], data['mask'], data['transform'],
            dtm_dates=dtm_dates,
            dsm_dates=dsm_dates,
            spectral=spectral,
            copernicus=copernicus_data,
            building_footprints=building_footprints,
            hansen=hansen_data,
            observation_year=obs_year,
            infra_lookup=_infra_lookup,
            mark_uncertain=mark_uncertain,
        )

        objects = result['objects']
        labels = result['labels']

        # Hansen calibration
        if include_hansen and hansen_data:
            try:
                objects = hansen.calibrate_tree_loss(objects, labels, hansen_data, observation_year=obs_year)
            except Exception as e:
                log.warning("Hansen calibration failed in overlay: %s", e)

        # Store in cache for fast re-renders with different type filters
        _seg_cache.update({
            "labels": labels, "objects": objects,
            "mask": data['mask'], "transform": data['transform'],
            "shape": data['shape'], "ndsm": data['ndsm'], "key": cache_key,
        })
        log.info("segment overlay: full pipeline %.1fs, cached for re-renders", time.time() - t0)
        _seg_cache_save(cache_key, labels, objects, data['mask'],
                        data['transform'], data['shape'], data['ndsm'])

        if top_n_classes:
            from collections import Counter
            _tn = int(top_n_classes)
            _tc = Counter(o.obj_type for o in objects)
            _top = set(t for t, _ in _tc.most_common(_tn))
            type_filter = (type_filter & _top) if type_filter else _top

        return _render_seg_overlay(labels, objects, data['mask'],
                                   data['transform'], data['shape'], type_filter, color_mode,
                                   ndsm=data['ndsm'], type_overrides=type_overrides)
    except Exception as e:
        log.error("segment overlay: %s", traceback.format_exc())
        return _error(str(e))


# === SECTION: /api/v1/export/geopackage endpoint ===

@app.route('/api/v1/export/geopackage', methods=['POST'])
def export_geopackage():
    """Export selected raster layers as a GeoPackage.

    Query params:
      include_dtm=true       Include DTM/DSM/nDSM bands
      include_dsm=true       (same as above, kept for compat)
      include_segments=true   Include segment_type + segment_height raster bands
      include_segments_vector=true  Include vector polygon layer with segment outlines
      ortho_years=2024,2023   Ortho RGB(I) for listed years
      raster_layers=dtm-2024,dsm-2023,hansen,raster  RGBA overlay renders
      types=tree,road,...     Segment type filter
      height_min=X            Filter segments: height >= X metres
      height_max=X            Filter segments: height <= X metres
      height_op=gt|lt|between Explicit operator (inferred if omitted)
      color_mode=type|height  Segment raster color mode
      include_ortho=true      Legacy: same as ortho_years=default
      async=true              Return 202 with task_id; poll progress; download via GET
    """
    try:
        import fiona
        from fiona.crs import from_epsg
    except ImportError:
        pass

    try:
        features = _get_geometry()
        params = _get_params()

        run_async = str(request.args.get('async', params.get('async', 'false'))).lower() in ('true', '1', 'yes')
        task_id = request.args.get('task_id', params.get('task_id', ''))

        if run_async:
            if not task_id:
                task_id = str(uuid.uuid4())
            _progress_start(task_id)
            thread = threading.Thread(
                target=_gpkg_worker,
                args=(task_id, features, params),
                daemon=True,
            )
            thread.start()
            return jsonify({"task_id": task_id, "status": "running"}), 202

        # Synchronous path
        tmp_path, table_count, elapsed = _gpkg_core(features, params)
        if table_count == 0:
            return _error('No layers selected')
        log.info("GeoPackage export: %d tables, %.1fs", table_count, elapsed)
        return send_file(
            tmp_path, mimetype='application/geopackage+sqlite3',
            as_attachment=True, download_name='landscape_export.gpkg',
        )
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error("geopackage export: %s", traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


def _gpkg_worker(task_id: str, features: list, params: dict):
    """Background worker for async GeoPackage export."""
    try:
        _progress_set(task_id, 'gpkg_export', 'Building GeoPackage…')
        tmp_path, table_count, elapsed = _gpkg_core(features, params, task_id=task_id)
        if table_count == 0:
            _progress_error(task_id, 'No layers selected')
            return
        # Move result to well-known location
        dest = _RESULTS_DIR / f"{task_id}.gpkg"
        import shutil
        shutil.move(tmp_path, str(dest))
        _progress_done(task_id)
        log.info("Async GPKG task %s completed: %d tables, %.1fs", task_id, table_count, elapsed)
    except Exception as e:
        log.error("Async GPKG task %s failed: %s", task_id, traceback.format_exc())
        _progress_error(task_id, str(e))


@app.route('/api/v1/export/geopackage/download/<task_id>', methods=['GET'])
def download_gpkg(task_id):
    """Download a completed async GeoPackage export."""
    if not task_id or not re.match(r'^[a-f0-9\-]+$', task_id):
        return _error('Invalid task_id', 400)
    dest = _RESULTS_DIR / f"{task_id}.gpkg"
    if not dest.exists():
        return _error('GeoPackage not found or not ready yet', 404)
    try:
        resp = send_file(
            str(dest), mimetype='application/geopackage+sqlite3',
            as_attachment=True, download_name='landscape_export.gpkg',
        )
        # Clean up after download
        @resp.call_on_close
        def _cleanup():
            try:
                dest.unlink(missing_ok=True)
                (_PROGRESS_DIR / f"{task_id}.json").unlink(missing_ok=True)
            except Exception:
                pass
        return resp
    except Exception as e:
        log.error("GPKG download failed: %s", e)
        return _error(f"Download error: {e}", 500)


# === SECTION: /api/v1/export/kml endpoint ===

@app.route('/api/v1/export/kml', methods=['POST'])
def export_kml():
    """Export segment features as KML with type/height_class folders.

    Query params:
      types=tree,grass,...    Filter object types (default: all)
      group_by=type|height_class  Folder grouping (default: type)
      height_min=X            Filter: height >= X metres
      height_max=X            Filter: height <= X metres
      height_op=gt|lt|between Explicit operator (inferred from min/max if omitted)
    """
    try:
        features = _get_geometry()
        params = _get_params()
        result_json = params.get('result_json', '')

        type_filter_str = params.get('types', None)
        type_filter = set(t.strip() for t in type_filter_str.split(',')) if type_filter_str else None
        group_by = params.get('group_by', 'type')
        seg_geom = params.get('segment_geometry', 'point').lower().strip()

        # Get features from posted result or from cache
        if result_json:
            try:
                result_data = json.loads(result_json)
                obj_features = result_data.get('features', [])
            except Exception:
                obj_features = []
        else:
            obj_features = []

        if not obj_features:
            return _error('No features to export. Include result_json in the request body.')

        # Filter by type
        if type_filter:
            obj_features = [f for f in obj_features
                           if f.get('properties', {}).get('type') in type_filter]

        # Filter by height
        obj_features = _apply_height_filter_features(obj_features, params)

        # Polygon vectorisation if requested
        kml_features = obj_features
        if seg_geom == 'polygon' and features:
            try:
                geom_wgs = features[0]['geometry']
                dataset = params.get('dataset', ti.DEFAULT_DATASET)
                poly_features = _vectorise_segments_to_geojson(
                    geom_wgs, dataset, type_filter=type_filter,
                    height_params=params)
                if poly_features:
                    kml_features = poly_features
                else:
                    log.warning('kml export polygon: vectorisation returned 0 features, using points')
            except Exception as e:
                log.warning('kml export polygon fallback to points: %s', e)

        # Build KML
        style_mode = params.get('segment_geometry_style', 'type').lower().strip()
        kml = _build_kml(kml_features, group_by, style_mode=style_mode)

        tmp = tempfile.NamedTemporaryFile(suffix='.kml', delete=False, mode='w', encoding='utf-8')
        tmp.write(kml)
        tmp.close()

        return send_file(
            tmp.name, mimetype='application/vnd.google-earth.kml+xml',
            as_attachment=True, download_name='landscape_export.kml',
        )
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error("kml export: %s", traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


def _vectorise_segments_to_geojson(geom_wgs, dataset: str, type_filter=None,
                                    height_params: dict = None,
                                    style_mode: str = 'type') -> list:
    """Vectorise cached segment labels into GeoJSON polygon features (WGS84).

    Returns a list of GeoJSON Feature dicts with Polygon geometry and rich
    properties.  Falls back to empty list if segment cache is unavailable.
    """
    from shapely.ops import transform as shapely_transform
    import pyproj

    geom_3035 = ti.geometry_to_3035(geom_wgs)
    bounds_prefix = f"{geom_3035.bounds}_{dataset}"

    # Try to load from seg cache (in-memory → disk scan)
    cached = None
    if (_seg_cache.get('labels') is not None and
            _seg_cache.get('key') and bounds_prefix in _seg_cache['key']):
        cached = _seg_cache
    else:
        cached = _seg_cache_load(bounds_prefix) or _seg_cache_scan(bounds_prefix)
    if cached is None or cached.get('labels') is None:
        return []

    v_labels = cached['labels']
    v_objects = cached['objects']
    v_mask = cached['mask']
    v_tf = cached['transform']
    from rasterio.transform import Affine
    if isinstance(v_tf, (list, tuple)):
        v_tf = Affine(*v_tf[:6])

    obj_map = {o.obj_id: o for o in v_objects}
    # Apply type filter
    if type_filter:
        v_filtered = [o for o in v_objects if o.obj_type in type_filter]
    else:
        v_filtered = list(v_objects)
    # Apply height filter
    if height_params:
        v_filtered = _apply_height_filter_objects(v_filtered, height_params)
    filtered_ids = {o.obj_id for o in v_filtered}
    if not filtered_ids:
        return []

    from rasterio.features import shapes as rasterize_shapes
    label_int = v_labels.astype(np.int32)
    seg_mask = v_mask & np.isin(label_int, list(filtered_ids))

    # CRS transform: EPSG:3035 → EPSG:4326
    proj = pyproj.Transformer.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)

    out_features = []
    for geom_dict, val in rasterize_shapes(
        label_int, mask=seg_mask, transform=v_tf, connectivity=4,
    ):
        oid = int(val)
        obj = obj_map.get(oid)
        if obj is None:
            continue
        # Reproject polygon coords 3035→4326
        from shapely.geometry import shape, mapping
        poly_3035 = shape(geom_dict)
        poly_wgs = shapely_transform(proj.transform, poly_3035)

        tc = SEGMENT_COLORS.get(obj.obj_type, (128, 128, 128, 120))
        hex_type = '#{:02X}{:02X}{:02X}'.format(tc[0], tc[1], tc[2])
        hv = _viridis_rgb(min(1.0, (max(0, obj.height_max) / 45.0) ** 0.5))
        hex_height = '#{:02X}{:02X}{:02X}'.format(*hv)

        out_features.append({
            'type': 'Feature',
            'geometry': mapping(poly_wgs),
            'properties': {
                'id': oid,
                'type': obj.obj_type,
                'group_type': obj.group_type or '',
                'height_class': _height_class(obj.height_max),
                'height_max_m': round(obj.height_max, 2),
                'height_mean_m': round(obj.height_mean, 2),
                'area_sqm': round(obj.area_sqm, 1),
                'confidence': round(obj.confidence, 2),
                'is_manmade': int(obj.is_manmade) if obj.is_manmade else 0,
                'ndvi_mean': round(obj.ndvi_mean, 3) if obj.ndvi_mean else 0.0,
                'color': hex_type,
                'color_height': hex_height,
            },
        })
    log.info('vectorise_to_geojson: %d polygon features', len(out_features))
    return out_features


def _build_kml(features, group_by='type', style_mode='type'):
    """Build a KML string from GeoJSON features (points or polygons), organised into folders.

    style_mode: 'type' — colour by object type (default)
                'height' — colour by viridis height ramp
    """
    import xml.etree.ElementTree as ET

    # Derive KML colours (AABBGGRR) from the canonical SEGMENT_COLORS
    TYPE_KML_COLORS = {
        t: 'ff{:02x}{:02x}{:02x}'.format(rgba[2], rgba[1], rgba[0])
        for t, rgba in SEGMENT_COLORS.items()
    }

    # Group features
    groups = {}
    for f in features:
        p = f.get('properties', {})
        h_max = p.get('height_max_m') or p.get('height_after_m') or 0
        hc = _height_class(h_max)
        key = p.get('type', 'unknown') if group_by == 'type' else hc
        groups.setdefault(key, []).append((f, p, hc))

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2">',
             '<Document>',
             '<name>Landscape Analysis Export</name>']

    # Styles — include IconStyle (points), PolyStyle (fill), LineStyle (outline)
    for tname, color in TYPE_KML_COLORS.items():
        # Semi-transparent fill: replace 'ff' alpha with '80' (~50% opacity)
        fill_color = '80' + color[2:]
        lines.append(
            f'<Style id="style-{tname}">'
            f'<IconStyle><color>{color}</color><scale>0.8</scale>'
            '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>'
            '</IconStyle>'
            f'<LineStyle><color>{color}</color><width>1.5</width></LineStyle>'
            f'<PolyStyle><color>{fill_color}</color></PolyStyle>'
            '</Style>'
        )

    # Default style
    lines.append(
        '<Style id="style-default">'
        '<IconStyle><color>ff888888</color><scale>0.6</scale>'
        '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>'
        '</IconStyle>'
        '<LineStyle><color>ff888888</color><width>1</width></LineStyle>'
        '<PolyStyle><color>80888888</color></PolyStyle>'
        '</Style>'
    )

    # Height-based viridis styles (10 buckets: 0-5, 5-10, ..., 40-45, 45+)
    HEIGHT_BUCKETS = 10
    for i in range(HEIGHT_BUCKETS):
        t_val = (i + 0.5) / HEIGHT_BUCKETS  # midpoint fraction 0..1
        rgb = _viridis_rgb(t_val)
        color = 'ff{:02x}{:02x}{:02x}'.format(rgb[2], rgb[1], rgb[0])  # KML AABBGGRR
        fill_color = '80' + color[2:]
        lines.append(
            f'<Style id="style-h{i}">'
            f'<IconStyle><color>{color}</color><scale>0.8</scale>'
            '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>'
            '</IconStyle>'
            f'<LineStyle><color>{color}</color><width>1.5</width></LineStyle>'
            f'<PolyStyle><color>{fill_color}</color></PolyStyle>'
            '</Style>'
        )

    def _coords_to_kml_ring(ring_coords):
        """Convert a GeoJSON ring [[lon,lat], ...] to KML coordinate string."""
        parts = []
        for c in ring_coords:
            lon, lat = c[0], c[1]
            alt = c[2] if len(c) > 2 else 0
            parts.append(f'{lon},{lat},{alt}')
        return ' '.join(parts)

    for group_name in sorted(groups.keys()):
        items = groups[group_name]
        lines.append(f'<Folder><name>{_xml_escape(group_name)} ({len(items)})</name>')
        for feat, props, hc in items:
            geom = feat.get('geometry', {})
            geom_type = geom.get('type', '')
            coords = geom.get('coordinates', [])
            if not coords:
                continue
            tname = props.get('type', 'unknown')
            if style_mode == 'height':
                h_bucket = min(int((h_max / 45.0) ** 0.5 * HEIGHT_BUCKETS),
                               HEIGHT_BUCKETS - 1)
                if h_bucket < 0: h_bucket = 0
                style_id = f'style-h{h_bucket}'
            else:
                style_id = f'style-{tname}' if tname in TYPE_KML_COLORS else 'style-default'
            h_max = props.get('height_max_m', 0) or 0
            area = props.get('area_sqm', 0) or 0
            rgba = SEGMENT_COLORS.get(tname, (128, 128, 128, 120))
            hex_color = '#{:02X}{:02X}{:02X}'.format(rgba[0], rgba[1], rgba[2])
            desc_parts = [f'Type: {tname}', f'Color: {hex_color}',
                         f'Height class: {hc}',
                         f'Height max: {h_max:.1f}m', f'Area: {area:.0f} m\u00b2']
            if props.get('confidence'):
                desc_parts.append(f'Confidence: {props["confidence"]:.0%}')
            if props.get('group_type'):
                desc_parts.append(f'Group: {props["group_type"]}')
            desc = '<br/>'.join(desc_parts)
            name_str = f'{tname} ({h_max:.1f}m, {area:.0f}m\u00b2)'

            # Build geometry KML
            if geom_type == 'Polygon' and coords:
                rings_kml = ''
                for i, ring in enumerate(coords):
                    tag = 'outerBoundaryIs' if i == 0 else 'innerBoundaryIs'
                    rings_kml += (f'<{tag}><LinearRing>'
                                  f'<coordinates>{_coords_to_kml_ring(ring)}</coordinates>'
                                  f'</LinearRing></{tag}>')
                geom_kml = f'<Polygon><tessellate>1</tessellate>{rings_kml}</Polygon>'
            elif geom_type == 'MultiPolygon' and coords:
                polys_kml = ''
                for poly_coords in coords:
                    rings_kml = ''
                    for i, ring in enumerate(poly_coords):
                        tag = 'outerBoundaryIs' if i == 0 else 'innerBoundaryIs'
                        rings_kml += (f'<{tag}><LinearRing>'
                                      f'<coordinates>{_coords_to_kml_ring(ring)}</coordinates>'
                                      f'</LinearRing></{tag}>')
                    polys_kml += f'<Polygon><tessellate>1</tessellate>{rings_kml}</Polygon>'
                geom_kml = f'<MultiGeometry>{polys_kml}</MultiGeometry>'
            elif geom_type == 'Point' and len(coords) >= 2:
                lon, lat = coords[0], coords[1]
                alt = coords[2] if len(coords) > 2 else 0
                geom_kml = f'<Point><coordinates>{lon},{lat},{alt}</coordinates></Point>'
            else:
                continue

            lines.append(f'<Placemark><name>{_xml_escape(name_str)}</name>'
                         f'<description><![CDATA[{desc}]]></description>'
                         f'<styleUrl>#{style_id}</styleUrl>'
                         f'<ExtendedData>'
                         f'<Data name="color"><value>{hex_color}</value></Data>'
                         f'<Data name="type"><value>{_xml_escape(tname)}</value></Data>'
                         f'</ExtendedData>'
                         f'{geom_kml}'
                         '</Placemark>')
        lines.append('</Folder>')

    lines.append('</Document></kml>')
    return '\n'.join(lines)


def _xml_escape(s):
    """Escape XML special characters."""
    return (str(s).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;'))


def _fix_gpkg_raster_crs(gpkg_path: str):
    """Ensure GPKG raster layers are properly registered for QGIS.

    Layers written by rasterio fall into two categories:
    - float32 (DTM/DSM/etc): already '2d-gridded-coverage' with proper ancillary
    - uint8 (Ortho/CIR/segment_type/WorldCover/Hansen): registered as 'tiles'

    The 'tiles' data_type is the correct GPKG standard for PNG/JPEG tile pyramids.
    GDAL and QGIS both read CRS from 'tiles' layers via gpkg_tile_matrix_set.srs_id.
    Do NOT convert 'tiles' to '2d-gridded-coverage' — that extension is for
    single-band gridded elevation data (TIFF tiles) and breaks multi-band
    JPEG/PNG layers (Ortho, CIR) in QGIS, causing 'h_band null' errors and
    the raster to appear at 0,0 without CRS.

    This function:
    1. Ensures the gpkg_spatial_ref_sys entry is complete (WKT1 + WKT2)
    2. Repairs uint8 layers that were wrongly converted to '2d-gridded-coverage'
       by an older version of this code (reverts them to 'tiles' and cleans up
       bogus extension/ancillary entries)
    """
    import sqlite3 as _sq
    if not os.path.exists(gpkg_path) or os.path.getsize(gpkg_path) == 0:
        return
    conn = _sq.connect(gpkg_path)
    try:
        # --- 1. Ensure EPSG:3035 has both WKT1 and WKT2 definitions ---
        row = conn.execute(
            "SELECT definition, definition_12_063 FROM gpkg_spatial_ref_sys "
            "WHERE srs_id = 3035"
        ).fetchone()
        if row:
            wkt1, wkt2 = row
            needs_update = False
            if not wkt1 or wkt1 == 'undefined':
                from pyproj import CRS
                wkt1 = CRS.from_epsg(3035).to_wkt('WKT1_GDAL')
                needs_update = True
            if not wkt2 or wkt2 == 'undefined':
                from pyproj import CRS
                wkt2 = CRS.from_epsg(3035).to_wkt('WKT2_2019')
                needs_update = True
            if needs_update:
                conn.execute(
                    "UPDATE gpkg_spatial_ref_sys "
                    "SET definition = ?, definition_12_063 = ? "
                    "WHERE srs_id = 3035",
                    (wkt1, wkt2)
                )
                log.info("Updated EPSG:3035 WKT definitions in %s", gpkg_path)

        # --- 2. Repair uint8 layers wrongly registered as 2d-gridded-coverage ---
        # Legitimate float32 gridded layers have column_name='tile_data' in
        # gpkg_extensions.  Bogus uint8 layers (from old code) have
        # column_name IS NULL.  Revert those to 'tiles' data_type and clean
        # up the spurious extension + ancillary rows.
        try:
            bogus = conn.execute(
                "SELECT table_name FROM gpkg_extensions "
                "WHERE extension_name = 'gpkg_2d_gridded_coverage' "
                "AND column_name IS NULL "
                "AND table_name NOT IN "
                "  ('gpkg_2d_gridded_coverage_ancillary', "
                "   'gpkg_2d_gridded_tile_ancillary')"
            ).fetchall()
            if bogus:
                bogus_names = [r[0] for r in bogus]
                log.info("Repairing %d uint8 layers wrongly registered as "
                         "2d-gridded-coverage: %s", len(bogus_names), bogus_names)
                for tname in bogus_names:
                    conn.execute(
                        "UPDATE gpkg_contents SET data_type = 'tiles' "
                        "WHERE table_name = ? AND data_type = '2d-gridded-coverage'",
                        (tname,)
                    )
                    conn.execute(
                        "DELETE FROM gpkg_extensions "
                        "WHERE table_name = ? "
                        "AND extension_name = 'gpkg_2d_gridded_coverage' "
                        "AND column_name IS NULL",
                        (tname,)
                    )
                    try:
                        conn.execute(
                            "DELETE FROM gpkg_2d_gridded_coverage_ancillary "
                            "WHERE tile_matrix_set_name = ?",
                            (tname,)
                        )
                    except Exception:
                        pass
                    try:
                        conn.execute(
                            "DELETE FROM gpkg_2d_gridded_tile_ancillary "
                            "WHERE tpudt_name = ?",
                            (tname,)
                        )
                    except Exception:
                        pass
        except Exception as e:
            log.debug("GPKG repair check skipped (no extensions table?): %s", e)

        conn.commit()
    except Exception as e:
        log.warning("GPKG CRS fix failed for %s: %s", gpkg_path, e)
    finally:
        conn.close()


def _gpkg_core(features: list, params: dict, task_id: str = '') -> tuple:
    """Core GeoPackage building logic. Returns (tmp_path, table_count, elapsed_s)."""
    t0 = time.time()
    dataset = params.get('dataset', ti.DEFAULT_DATASET)

    # --- Parse layer selection params ---
    _bool = lambda k, d='false': str(params.get(k, d)).lower() in ('true', '1', 'yes')
    include_dtm = _bool('include_dtm', 'true') or _bool('include_dsm', 'true')
    include_segments = _bool('include_segments', 'false')
    include_segments_vector = _bool('include_segments_vector', 'false')
    include_ortho_legacy = _bool('include_ortho', 'false')

    ortho_years_str = params.get('ortho_years', '')
    ortho_years = [int(y.strip()) for y in ortho_years_str.split(',') if y.strip().isdigit()] if ortho_years_str else []
    if include_ortho_legacy and not ortho_years:
        ortho_years = [ti.dataset_to_year(dataset)]

    raster_layers_str = params.get('raster_layers', '')
    raster_layers = [x.strip() for x in raster_layers_str.split(',') if x.strip()] if raster_layers_str else []

    type_filter_str = params.get('types', None)
    type_filter = set(t.strip() for t in type_filter_str.split(',')) if type_filter_str else None
    color_mode = params.get('color_mode', 'type')

    feat = features[0]
    geom = feat['geometry']
    geom_3035 = ti.geometry_to_3035(geom)
    if geom.geom_type == 'Point':
        geom_3035 = geom_3035.buffer(100)
    _validate_area(geom_3035)

    _progress_set(task_id, 'gpkg_raster', 'Reading DTM/DSM…')
    data = raster_io.read_dtm_dsm(geom_3035, dataset)
    dtm = data['dtm']
    dsm = data['dsm']
    ndsm = data['ndsm']
    tf = data['transform']
    h, w = data['shape']
    mask = data['mask']

    tmp = tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False)
    tmp_path = tmp.name
    tmp.close()
    os.unlink(tmp_path)  # GPKG driver needs a fresh path

    table_count = 0

    def _write_gpkg_table(name, arrays_list, dtype='float32', descriptions=None):
        """Write one raster table to the GPKG. First call creates, rest append."""
        nonlocal table_count
        n = len(arrays_list)
        opts = dict(
            driver='GPKG', width=w, height=h, count=n,
            dtype=dtype, crs='EPSG:3035', transform=tf,
            RASTER_TABLE=name, RASTER_IDENTIFIER=name,
        )
        if dtype == 'float32':
            opts['nodata'] = float('nan')
        if table_count > 0:
            opts['APPEND_SUBDATASET'] = 'YES'
        with rasterio.open(tmp_path, 'w', **opts) as dst:
            for i, arr in enumerate(arrays_list, 1):
                out = arr[:h, :w] if arr.shape[0] >= h and arr.shape[1] >= w else arr
                dst.write(out, i)
                if descriptions and i <= len(descriptions):
                    dst.set_band_description(i, descriptions[i - 1])
        table_count += 1
        log.info("GPKG table %s: %d bands (%s)", name, n, dtype)

    # --- Core DTM/DSM/nDSM (1 band each = cleaner in QGIS) ---
    if include_dtm:
        _progress_set(task_id, 'gpkg_dtm', 'Writing DTM/DSM/nDSM…')
        _write_gpkg_table('DTM', [dtm.astype(np.float32)])
        _write_gpkg_table('DSM', [dsm.astype(np.float32)])
        _write_gpkg_table('nDSM', [ndsm.astype(np.float32)])

    # --- Orthophoto for each requested year (RGBA uint8) ---
    rgb = None
    for year in ortho_years:
        try:
            import ortho_io
            _progress_set(task_id, 'gpkg_ortho', f'Writing ortho {year}…')
            rgb_arr, nir = ortho_io.read_ortho_for_als(data, year=year)
            if rgb_arr is not None:
                bands_u8 = [rgb_arr[0], rgb_arr[1], rgb_arr[2]]
                descs = ['Red', 'Green', 'Blue']
                if nir is not None:
                    bands_u8.append(nir)
                    descs.append('NIR')
                _write_gpkg_table(f'Ortho_{year}', bands_u8,
                                  dtype='uint8', descriptions=descs)
                if rgb is None:
                    rgb = rgb_arr
        except Exception as e:
            log.warning("Ortho %d for gpkg failed: %s", year, e)

    # --- Segment type/height rasters ---
    if include_segments:
        try:
            _progress_set(task_id, 'gpkg_segments', 'Building segment rasters…')
            labels = None
            objects = None
            cache_key_check = f"{geom_3035.bounds}_{dataset}"
            if _seg_cache.get("labels") is not None and _seg_cache["key"] and cache_key_check in _seg_cache["key"]:
                log.info("GeoPackage: using cached segmentation (in-process)")
                labels = _seg_cache["labels"]
                objects = _seg_cache["objects"]
            else:
                # Try file-backed cache (cross-worker)
                _dc = _seg_cache_scan(cache_key_check)
                if _dc is not None:
                    log.info("GeoPackage: using cached segmentation (disk)")
                    labels = _dc['labels']
                    objects = _dc['objects']
            if labels is None:
                spectral = None
                if rgb is not None:
                    import ortho_io
                    _, nir_arr = ortho_io.read_ortho_for_als(data)
                    spectral = ortho_io.compute_spectral_indices(rgb, nir=nir_arr)
                    spectral["red"] = rgb[0].astype(np.float32)
                    spectral["green"] = rgb[1].astype(np.float32)
                    spectral["blue"] = rgb[2].astype(np.float32)
                    if nir_arr is not None:
                        spectral["nir"] = nir_arr.astype(np.float32)
                _il = None
                _incl_infra = str(params.get('include_infra', 'true')).lower() in ('true', '1', 'yes')
                if _incl_infra:
                    try:
                        from infrastructure_lookup import InfrastructureLookup
                        from pyproj import Transformer as _Tx2
                        _tx2 = _Tx2.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
                        _b = geom_3035.bounds
                        _w4, _s4 = _tx2.transform(_b[0], _b[1])
                        _e4, _n4 = _tx2.transform(_b[2], _b[3])
                        _il = InfrastructureLookup.for_bbox(_w4, _s4, _e4, _n4)
                    except Exception:
                        pass
                _mark_uncertain = str(params.get('mark_uncertain', 'false')).lower() in ('true', '1', 'yes')
                result = seg.segment_and_classify(
                    dtm, dsm, mask, tf, spectral=spectral,
                    observation_year=ti.dataset_to_year(dataset),
                    infra_lookup=_il,
                    mark_uncertain=_mark_uncertain,
                )
                objects = result['objects']
                labels = result['labels']

            if labels is not None and objects is not None:
                if type_filter:
                    seg_filtered = [o for o in objects if o.obj_type in type_filter]
                else:
                    seg_filtered = list(objects)
                seg_filtered = _apply_height_filter_objects(seg_filtered, params)
                filtered_ids = {o.obj_id for o in seg_filtered}

                type_raster = np.zeros((h, w), dtype=np.float32)
                obj_map = {o.obj_id: o for o in objects}
                for oid in filtered_ids:
                    if oid in obj_map:
                        type_raster[labels == oid] = float(obj_map[oid].type_code)
                height_raster = np.where(ndsm > 0, ndsm, 0).astype(np.float32)

                _write_gpkg_table('segment_type', [type_raster])
                _write_gpkg_table('segment_height', [height_raster])
            else:
                log.info("GeoPackage: skipping segment layers (no segment data available)")
        except Exception as e:
            log.warning("Segments for gpkg failed: %s", e)

    # --- Vector segment polygons ---
    if include_segments_vector:
        try:
            _progress_set(task_id, 'gpkg_vector', 'Vectorising segments…')
            # Get labels + objects (reuse from raster segments if already loaded)
            v_labels = None
            v_objects = None
            cache_key_check = f"{geom_3035.bounds}_{dataset}"
            if include_segments and labels is not None:
                v_labels, v_objects = labels, objects
            else:
                if _seg_cache.get('labels') is not None and _seg_cache.get('key') and cache_key_check in _seg_cache['key']:
                    v_labels = _seg_cache['labels']
                    v_objects = _seg_cache['objects']
                else:
                    _dc = _seg_cache_scan(cache_key_check)
                    if _dc is not None:
                        v_labels = _dc['labels']
                        v_objects = _dc['objects']
            if v_labels is not None and v_objects is not None:
                from rasterio.features import shapes as rasterize_shapes
                import fiona
                from fiona.crs import from_epsg

                obj_map = {o.obj_id: o for o in v_objects}
                # Apply type filter
                if type_filter:
                    v_filtered = [o for o in v_objects if o.obj_type in type_filter]
                else:
                    v_filtered = list(v_objects)
                # Apply height filter
                v_filtered = _apply_height_filter_objects(v_filtered, params)
                filtered_ids = {o.obj_id for o in v_filtered}

                # Vectorise the labels raster
                label_int = v_labels.astype(np.int32)
                seg_mask = mask & np.isin(label_int, list(filtered_ids))

                schema = {
                    'geometry': 'Polygon',
                    'properties': [
                        ('id', 'int'),
                        ('type', 'str'),
                        ('group_type', 'str'),
                        ('height_class', 'str'),
                        ('height_max_m', 'float'),
                        ('height_mean_m', 'float'),
                        ('area_sqm', 'float'),
                        ('slope_mean_deg', 'float'),
                        ('aspect_mean_deg', 'float'),
                        ('aspect_dominant', 'str'),
                        ('elevation_mean_m', 'float'),
                        ('tri_mean', 'float'),
                        ('terrain_class', 'str'),
                        ('confidence', 'float'),
                        ('is_manmade', 'int'),
                        ('ndvi_mean', 'float'),
                        ('color', 'str'),
                        ('color_height', 'str'),
                    ],
                }
                vec_path = tmp_path  # append to same GPKG
                with fiona.open(vec_path, 'w', driver='GPKG', layer='segments',
                                schema=schema, crs=from_epsg(3035)) as dst:
                    written = 0
                    for geom_dict, val in rasterize_shapes(
                        label_int, mask=seg_mask, transform=tf,
                        connectivity=4,
                    ):
                        oid = int(val)
                        obj = obj_map.get(oid)
                        if obj is None:
                            continue
                        # Type-based color
                        tc = SEGMENT_COLORS.get(obj.obj_type, (128, 128, 128, 120))
                        hex_type = '#{:02X}{:02X}{:02X}'.format(tc[0], tc[1], tc[2])
                        # Height-based viridis color (sqrt-scaled 0-45m)
                        hv = _viridis_rgb(min(1.0, (max(0, obj.height_max) / 45.0) ** 0.5))
                        hex_height = '#{:02X}{:02X}{:02X}'.format(*hv)
                        dst.write({
                            'geometry': geom_dict,
                            'properties': {
                                'id': oid,
                                'type': obj.obj_type,
                                'group_type': obj.group_type or '',
                                'height_class': _height_class(obj.height_max),
                                'height_max_m': round(obj.height_max, 2),
                                'height_mean_m': round(obj.height_mean, 2),
                                'area_sqm': round(obj.area_sqm, 1),
                                'slope_mean_deg': round(obj.slope_mean, 1),
                                'aspect_mean_deg': round(obj.aspect_mean, 1),
                                'aspect_dominant': obj.aspect_dominant or 'flat',
                                'elevation_mean_m': round(obj.elevation_mean, 2),
                                'tri_mean': round(obj.tri_mean, 3),
                                'terrain_class': obj.terrain_class or 'level',
                                'confidence': round(obj.confidence, 2),
                                'is_manmade': int(obj.is_manmade) if obj.is_manmade else 0,
                                'ndvi_mean': round(obj.ndvi_mean, 3) if obj.ndvi_mean else 0.0,
                                'color': hex_type,
                                'color_height': hex_height,
                            },
                        })
                        written += 1
                table_count += 1
                log.info("GPKG vector layer 'segments': %d polygons", written)
                # Write QGIS layer_styles table for auto-rendering
                try:
                    _write_gpkg_categorized_style(tmp_path, 'segments', color_mode)
                except Exception as e:
                    log.warning('GPKG style table failed: %s', e)
            else:
                log.info('GeoPackage: skipping vector segments (no segment data available)')
        except Exception as e:
            log.warning('Vector segments for gpkg failed: %s', e)

    # --- Rendered RGBA raster overlays ---
    geom_wgs = _extract_single_geom(features)
    for rlayer in raster_layers:
        try:
            _progress_set(task_id, 'gpkg_overlay', f'Rendering {rlayer}…')
            rgba = _render_overlay_for_gpkg(
                rlayer, data, geom_3035, geom_wgs, dataset,
                type_filter, color_mode, params,
            )
            if rgba is not None:
                bands_u8 = [rgba[:, :, i] for i in range(4)]
                _write_gpkg_table(rlayer, bands_u8, dtype='uint8',
                                  descriptions=['R', 'G', 'B', 'A'])
        except Exception as e:
            log.warning("Raster overlay %s for gpkg failed: %s", rlayer, e)

    # Fix CRS for uint8 raster layers (GDAL registers as 'tiles' not '2d-gridded-coverage')
    _fix_gpkg_raster_crs(tmp_path)

    elapsed = round(time.time() - t0, 2)
    return tmp_path, table_count, elapsed


def _write_gpkg_categorized_style(gpkg_path: str, layer_name: str, color_mode: str = 'type'):
    """Write a QGIS-compatible layer_styles table so the GPKG auto-renders.

    Uses the ``color`` (type) or ``color_height`` field as a data-defined
    fill colour, depending on *color_mode*.
    """
    import sqlite3
    color_field = 'color_height' if color_mode == 'height' else 'color'

    # Build a QML style XML that uses data-defined colour from the color field
    qml = (
        '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
        '<qgis version="3.34">'
        '<renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0">'
        '<symbols>'
        '<symbol type="fill" name="0" clip_to_extent="1" alpha="0.7">'
        '<layer class="SimpleFill" enabled="1" locked="0" pass="0">'
        '<Option type="Map">'
        '<Option type="QString" value="solid" name="style"/>'
        '<Option type="QString" value="0.35,0.35,0.35,255,rgb:0,0,0,1" name="outline_color"/>'
        '<Option type="QString" value="0.2" name="outline_width"/>'
        '</Option>'
        f'<data_defined_properties><Property><Option type="Map">'
        f'<Option type="Map" name="properties"><Option type="Map" name="fillColor">'
        f'<Option type="bool" value="true" name="active"/>'
        f'<Option type="QString" value="&quot;{color_field}&quot;" name="expression"/>'
        f'<Option type="int" value="3" name="type"/>'
        f'</Option></Option></Option></Property></data_defined_properties>'
        '</layer></symbol></symbols></renderer-v2></qgis>'
    )

    conn = sqlite3.connect(gpkg_path)
    try:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS layer_styles ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'f_table_catalog TEXT DEFAULT \'\','
            'f_table_schema TEXT DEFAULT \'\','
            'f_table_name TEXT,'
            'f_geometry_column TEXT,'
            'styleName TEXT,'
            'styleQML TEXT,'
            'styleSLD TEXT,'
            'useAsDefault BOOLEAN,'
            'description TEXT,'
            'owner TEXT,'
            'ui TEXT,'
            'update_time TIMESTAMP DEFAULT (strftime(\'%Y-%m-%dT%H:%M:%fZ\',\'now\'))'
            ')'
        )
        conn.execute(
            'INSERT INTO layer_styles '
            '(f_table_name, f_geometry_column, styleName, styleQML, useAsDefault, description) '
            'VALUES (?, ?, ?, ?, 1, ?)',
            (layer_name, 'geom', f'Segment {color_mode}', qml,
             f'Auto-generated colour-by-{color_mode} style'),
        )
        conn.commit()
    finally:
        conn.close()
    log.info('GPKG style written for %s (color_mode=%s)', layer_name, color_mode)


def _render_overlay_for_gpkg(layer_id, data, geom_3035, geom_wgs, dataset,
                              type_filter, color_mode, params):
    """Render a single overlay layer as RGBA array in EPSG:3035 space.

    Returns (h, w, 4) uint8 array or None.
    """
    h, w = data['shape']
    tf = data['transform']
    mask = data['mask']

    ds_map = {
        'dtm-2024': '20240915', 'dtm-2023': '20230915', 'dtm-2022': '20220915',
        'dsm-2024': '20240915', 'dsm-2023': '20230915', 'dsm-2022': '20220915',
    }

    if layer_id.startswith('dtm-'):
        ds = ds_map.get(layer_id, dataset)
        d = raster_io.read_dtm_dsm(geom_3035, ds) if ds != dataset else data
        return _dtm_rgba(d['dtm'], d['mask'])

    elif layer_id.startswith('dsm-'):
        ds = ds_map.get(layer_id, dataset)
        d = raster_io.read_dtm_dsm(geom_3035, ds) if ds != dataset else data
        return _ndsm_rgba(d['ndsm'], d['dsm'], d['mask'])

    elif layer_id.startswith('cir-'):
        # CIR false-color: NIR→R, Red→G, Green→B
        year_str = layer_id.split('-', 1)[1]
        year = int(year_str) if year_str.isdigit() else None
        import ortho_io
        rgb_arr, nir = ortho_io.read_ortho_for_als(data, year=year)
        if nir is None or rgb_arr is None:
            return None
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = nir[:h, :w]
        rgba[:, :, 1] = rgb_arr[0, :h, :w]
        rgba[:, :, 2] = rgb_arr[1, :h, :w]
        rgba[:, :, 3] = np.where(mask[:h, :w], 255, 0)
        return rgba

    elif layer_id == 'hansen':
        bbox_wgs = geom_wgs.bounds if hasattr(geom_wgs, 'bounds') else _extract_single_geom([{'geometry': geom_wgs}]).bounds
        prior = hansen.get_forest_prior(bbox_wgs, tf, (h, w))
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        forest = prior['current_forest']
        rgba[forest] = [20, 120, 20, 140]
        gain = prior['gain']
        rgba[gain] = [0, 200, 200, 180]
        ly = prior['loss_year']
        loss = ly > 0
        brightness = np.clip(80 + (ly.astype(np.float32) / 24.0) * 175, 80, 255).astype(np.uint8)
        rgba[:, :, 0][loss] = brightness[loss]
        rgba[:, :, 1][loss] = 0
        rgba[:, :, 2][loss] = brightness[loss]
        rgba[:, :, 3][loss] = 200
        rgba[:, :, 3][~mask] = 0
        return rgba

    elif layer_id == 'raster':
        # Segment classification raster — needs segmentation
        import object_segmentation as oseg
        labels = None
        objects = None
        seg_mask = None
        seg_ndsm = None
        cache_key_check = f"{geom_3035.bounds}_{dataset}"
        if _seg_cache["key"] and cache_key_check in _seg_cache["key"]:
            labels = _seg_cache["labels"]
            objects = _seg_cache["objects"]
            seg_mask = _seg_cache["mask"]
            seg_ndsm = _seg_cache.get("ndsm")
        else:
            # Try file-backed cache (cross-worker)
            _dc = _seg_cache_scan(cache_key_check)
            if _dc is not None:
                log.info("MBTiles raster: using cached segmentation (disk)")
                labels = _dc['labels']
                objects = _dc['objects']
                seg_mask = _dc['mask']
                seg_ndsm = _dc.get('ndsm')
        if not (labels is not None and objects is not None):
            log.info("gpkg raster overlay: running segmentation")
            spectral = None
            try:
                import ortho_io
                rgb_arr, nir = ortho_io.read_ortho_for_als(data)
                if rgb_arr is not None:
                    spectral = ortho_io.compute_spectral_indices(rgb_arr, nir=nir)
                    spectral["red"] = rgb_arr[0].astype(np.float32)
                    spectral["green"] = rgb_arr[1].astype(np.float32)
                    spectral["blue"] = rgb_arr[2].astype(np.float32)
                    if nir is not None:
                        spectral["nir"] = nir.astype(np.float32)
            except Exception:
                pass
            _il2 = None
            _incl_infra2 = str(params.get('include_infra', 'true')).lower() in ('true', '1', 'yes')
            if _incl_infra2:
                try:
                    from infrastructure_lookup import InfrastructureLookup
                    from pyproj import Transformer as _Tx3
                    _tx3 = _Tx3.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
                    _b2 = geom_3035.bounds
                    _w42, _s42 = _tx3.transform(_b2[0], _b2[1])
                    _e42, _n42 = _tx3.transform(_b2[2], _b2[3])
                    _il2 = InfrastructureLookup.for_bbox(_w42, _s42, _e42, _n42)
                except Exception:
                    pass
            _mark_uncertain2 = str(params.get('mark_uncertain', 'false')).lower() in ('true', '1', 'yes')
            result = seg.segment_and_classify(
                data['dtm'], data['dsm'], mask, tf, spectral=spectral,
                observation_year=ti.dataset_to_year(dataset),
                infra_lookup=_il2,
                mark_uncertain=_mark_uncertain2,
            )
            labels = result['labels']
            objects = result['objects']
            seg_mask = mask
            seg_ndsm = data.get('ndsm')

        hf = _parse_height_filter(params) if params else None
        rgba = _segment_rgba(
            labels, objects, seg_mask,
            type_filter, color_mode=color_mode, ndsm=seg_ndsm,
            height_filter=hf,
        )
        return rgba

    return None


# === SECTION: /api/v1/export/mbtiles endpoint ===

@app.route('/api/v1/export/mbtiles', methods=['POST'])
def export_mbtiles():
    """Export a single raster layer as MBTiles for offline mobile use.

    Query params:
      layer=dtm-2024|ortho-2024|cir-2024|... (required)
      min_zoom=10 (default 10)
      max_zoom=18 (default 18)
      async=true  Return 202 with task_id
    """
    try:
        features = _get_geometry()
        params = _get_params()
        layer = request.args.get('layer', params.get('layer', ''))
        if not layer:
            return _error('layer parameter required')

        run_async = str(request.args.get('async', 'false')).lower() in ('true', '1', 'yes')
        task_id = request.args.get('task_id', str(uuid.uuid4()))

        if run_async:
            _progress_start(task_id)
            thread = threading.Thread(
                target=_mbtiles_worker,
                args=(task_id, features, params, layer),
                daemon=True,
            )
            thread.start()
            return jsonify({"task_id": task_id, "status": "running"}), 202

        # Sync
        path, elapsed = _mbtiles_core(features, params, layer)
        return send_file(path, mimetype='application/x-sqlite3',
                        as_attachment=True, download_name=f'{layer}.mbtiles')
    except Exception as e:
        log.error("mbtiles export: %s", traceback.format_exc())
        return _error(str(e))


def _mbtiles_worker(task_id, features, params, layer):
    try:
        _progress_set(task_id, 'mbtiles_export', f'Building {layer} MBTiles\u2026')
        path, elapsed = _mbtiles_core(features, params, layer, task_id=task_id)
        dest = _RESULTS_DIR / f"{task_id}.mbtiles"
        import shutil
        shutil.move(path, str(dest))
        _progress_done(task_id)
        log.info("MBTiles %s completed: %.1fs", task_id, elapsed)
    except Exception as e:
        log.error("MBTiles %s failed: %s", task_id, traceback.format_exc())
        _progress_error(task_id, str(e))


@app.route('/api/v1/export/mbtiles/download/<task_id>', methods=['GET'])
def download_mbtiles(task_id):
    if not task_id or not re.match(r'^[a-f0-9\\-]+$', task_id):
        return _error('Invalid task_id', 400)
    dest = _RESULTS_DIR / f"{task_id}.mbtiles"
    if not dest.exists():
        return _error('MBTiles not found or not ready yet', 404)
    layer_name = request.args.get('layer', 'layer')
    resp = send_file(str(dest), mimetype='application/x-sqlite3',
                    as_attachment=True, download_name=f'{layer_name}.mbtiles')
    @resp.call_on_close
    def _cleanup():
        try:
            dest.unlink(missing_ok=True)
            (_PROGRESS_DIR / f"{task_id}.json").unlink(missing_ok=True)
        except Exception:
            pass
    return resp


def _mbtiles_core(features, params, layer, task_id=''):
    """Generate MBTiles for a single raster layer."""
    import sqlite3, math
    t0 = time.time()

    feat = features[0]
    geom = feat['geometry']
    geom_3035 = ti.geometry_to_3035(geom)
    if geom.geom_type == 'Point':
        geom_3035 = geom_3035.buffer(100)
    _validate_area(geom_3035)

    dataset = params.get('dataset', ti.DEFAULT_DATASET)
    min_zoom = int(params.get('min_zoom', '10'))
    max_zoom = int(params.get('max_zoom', '16'))

    # Generate the overlay image
    _progress_set(task_id, 'mbtiles_render', f'Rendering {layer}\u2026')

    # Get the overlay PNG + bounds (reuse existing overlay logic)
    geom_wgs = geom
    data = _get_cached_raster(geom_3035, dataset)
    if data is None:
        data = raster_io.read_dtm_dsm(geom_3035, dataset)

    rgba, bounds_wgs = _render_overlay_for_mbtiles(layer, data, geom_3035, geom_wgs, dataset, params, task_id)
    if rgba is None:
        raise ValueError(f'Could not render layer {layer}')

    # bounds_wgs = (south, west, north, east)
    s, w, n, e = bounds_wgs

    # Create MBTiles
    tmp = tempfile.NamedTemporaryFile(suffix='.mbtiles', delete=False)
    tmp_path = tmp.name
    tmp.close()

    conn = sqlite3.connect(tmp_path)
    c = conn.cursor()
    c.execute('CREATE TABLE metadata (name TEXT, value TEXT)')
    c.execute('CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)')
    c.execute('CREATE UNIQUE INDEX tile_index ON tiles (zoom_level, tile_column, tile_row)')

    c.execute("INSERT INTO metadata VALUES ('name', ?)", (layer,))
    c.execute("INSERT INTO metadata VALUES ('format', 'png')")
    c.execute("INSERT INTO metadata VALUES ('bounds', ?)", (f'{w},{s},{e},{n}',))
    c.execute("INSERT INTO metadata VALUES ('minzoom', ?)", (str(min_zoom),))
    c.execute("INSERT INTO metadata VALUES ('maxzoom', ?)", (str(max_zoom),))
    c.execute("INSERT INTO metadata VALUES ('type', 'overlay')")

    from PIL import Image
    # rgba is a numpy array (H, W, 4)
    src_img = Image.fromarray(rgba)
    src_w, src_h = src_img.size

    total_tiles = 0
    for zoom in range(min_zoom, max_zoom + 1):
        _progress_set(task_id, 'mbtiles_tiles', f'Zoom {zoom}/{max_zoom}')
        n_tiles = 2 ** zoom

        # Tile bounds in tile coordinates
        x_min = int((w + 180) / 360 * n_tiles)
        x_max = int((e + 180) / 360 * n_tiles)
        y_min_merc = int((1 - math.log(math.tan(math.radians(n)) + 1/math.cos(math.radians(n))) / math.pi) / 2 * n_tiles)
        y_max_merc = int((1 - math.log(math.tan(math.radians(s)) + 1/math.cos(math.radians(s))) / math.pi) / 2 * n_tiles)

        x_min = max(0, x_min)
        x_max = min(n_tiles - 1, x_max)
        y_min_merc = max(0, y_min_merc)
        y_max_merc = min(n_tiles - 1, y_max_merc)

        for tx in range(x_min, x_max + 1):
            for ty_merc in range(y_min_merc, y_max_merc + 1):
                # Tile bounds in WGS84
                tile_w = tx / n_tiles * 360 - 180
                tile_e = (tx + 1) / n_tiles * 360 - 180
                tile_n_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty_merc / n_tiles)))
                tile_s_rad = math.atan(math.sinh(math.pi * (1 - 2 * (ty_merc + 1) / n_tiles)))
                tile_n_deg = math.degrees(tile_n_rad)
                tile_s_deg = math.degrees(tile_s_rad)

                # Map source image pixels to this tile
                # Source image covers (s,w)->(n,e)
                px_left = (tile_w - w) / (e - w) * src_w
                px_right = (tile_e - w) / (e - w) * src_w
                px_top = (n - tile_n_deg) / (n - s) * src_h
                px_bottom = (n - tile_s_deg) / (n - s) * src_h

                # Crop and resize to 256x256
                crop_box = (int(px_left), int(px_top), int(px_right), int(px_bottom))
                # Skip fully out-of-bounds tiles
                if crop_box[2] <= 0 or crop_box[0] >= src_w or crop_box[3] <= 0 or crop_box[1] >= src_h:
                    continue

                # Clamp
                crop_box = (max(0, crop_box[0]), max(0, crop_box[1]),
                           min(src_w, crop_box[2]), min(src_h, crop_box[3]))
                if crop_box[2] - crop_box[0] < 1 or crop_box[3] - crop_box[1] < 1:
                    continue

                tile_img = src_img.crop(crop_box).resize((256, 256), Image.LANCZOS)

                import io as _io
                buf = _io.BytesIO()
                tile_img.save(buf, format='PNG', optimize=True)
                tile_data = buf.getvalue()

                # TMS y-flip
                tms_y = n_tiles - 1 - ty_merc
                c.execute('INSERT OR REPLACE INTO tiles VALUES (?,?,?,?)',
                         (zoom, tx, tms_y, tile_data))
                total_tiles += 1

    conn.commit()
    conn.close()
    elapsed = time.time() - t0
    log.info("MBTiles %s: %d tiles, %.1fs", layer, total_tiles, elapsed)
    return tmp_path, elapsed


def _render_overlay_for_mbtiles(layer, data, geom_3035, geom_wgs, dataset, params, task_id=''):
    """Render a layer overlay and return (rgba_array, bounds_wgs).
    bounds_wgs = (south, west, north, east)
    """
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import array_bounds

    tf = data['transform']
    h, w_px = data['shape']
    mask = data['mask']

    def _reproject_rgba_to_wgs84(rgba_3035):
        """Reproject an (H, W, 4) uint8 RGBA from EPSG:3035 → EPSG:4326.

        Returns (rgba_wgs, bounds_wgs) where bounds_wgs = (south, west, north, east).
        """
        src_crs = 'EPSG:3035'
        dst_crs = 'EPSG:4326'
        bounds_3035 = array_bounds(h, w_px, tf)
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, w_px, h, *bounds_3035)
        rgba_wgs = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
        for band in range(4):
            reproject(
                rgba_3035[:, :, band], rgba_wgs[:, :, band],
                src_transform=tf, src_crs=src_crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
        # (left, bottom, right, top) → (south, west, north, east)
        b = array_bounds(dst_h, dst_w, dst_transform)
        bounds_wgs = (b[1], b[0], b[3], b[2])
        return rgba_wgs, bounds_wgs

    try:
        # Extract params needed by _render_overlay_for_gpkg
        type_filter_str = params.get('types', None)
        type_filter = set(t.strip() for t in type_filter_str.split(',')) if type_filter_str else None
        color_mode = params.get('color_mode', 'type')

        # Delegate to _render_overlay_for_gpkg which handles all layer types
        # in EPSG:3035, then reproject the result to WGS84.
        rgba_3035 = _render_overlay_for_gpkg(
            layer, data, geom_3035, geom_wgs, dataset,
            type_filter, color_mode, params,
        )

        if rgba_3035 is None:
            # _render_overlay_for_gpkg doesn't handle ortho-* layers;
            # render them here directly.
            if layer.startswith('ortho-'):
                year_str = layer.split('-', 1)[1]
                year = int(year_str) if year_str.isdigit() else None
                import ortho_io
                rgb_arr, nir = ortho_io.read_ortho_for_als(data, year=year)
                if rgb_arr is None:
                    log.error("MBTiles render %s: no ortho data", layer)
                    return None, None
                rgba_3035 = np.zeros((h, w_px, 4), dtype=np.uint8)
                rgba_3035[:, :, 0] = rgb_arr[0, :h, :w_px]
                rgba_3035[:, :, 1] = rgb_arr[1, :h, :w_px]
                rgba_3035[:, :, 2] = rgb_arr[2, :h, :w_px]
                rgba_3035[:, :, 3] = np.where(mask[:h, :w_px], 255, 0)
            else:
                # Unknown layer – fall back to DTM hillshade
                log.warning("MBTiles render %s: unknown layer, falling back to DTM", layer)
                rgba_3035 = _dtm_rgba(data['dtm'], mask)

        if rgba_3035 is None:
            return None, None

        return _reproject_rgba_to_wgs84(rgba_3035)

    except Exception as e:
        log.error("MBTiles render %s failed: %s", layer, e)
        return None, None


# === SECTION: /api/v1/changes endpoint (temporal analysis) ===

@app.route('/api/v1/changes', methods=['POST'])
def changes():
    """Detect changes between two ALS dates. Returns classified change events."""
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        date_a = params.get('date_a', '20220915')
        date_b = params.get('date_b', '20240915')
        min_change = float(params.get('min_change', 1.0))

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(200)
        _validate_area(geom_3035)

        comparison = tca.compare_dates(geom_3035, date_a, date_b)
        events = tca.detect_changes(geom_3035, date_a, date_b,
                                    min_change=min_change, comparison=comparison)

        event_features = []
        for ev in events:
            centroid_wgs = ti.geometry_from_3035(Point(ev.centroid_e, ev.centroid_n))
            props = {
                "event_type": ev.event_type,
                "area_sqm": ev.area_sqm,
                "height_before_m": ev.height_before,
                "height_after_m": ev.height_after,
                "height_change_mean_m": ev.height_change_mean,
                "height_change_max_m": ev.height_change_max,
                "dtm_change_mean_m": ev.dtm_change_mean,
                "dtm_change_max_m": ev.dtm_change_max,
                "dsm_change_mean_m": ev.dsm_change_mean,
                "confidence": ev.confidence,
                "detail": ev.detail,
            }
            event_features.append({"type": "Feature", "properties": props,
                                   "geometry": mapping(centroid_wgs)})

        # Summarise by type
        by_type = {}
        for ev in events:
            t2 = ev.event_type
            by_type.setdefault(t2, {"count": 0, "total_area_sqm": 0})
            by_type[t2]["count"] += 1
            by_type[t2]["total_area_sqm"] += ev.area_sqm
        for v in by_type.values():
            v["total_area_sqm"] = round(v["total_area_sqm"], 1)

        return jsonify({
            "type": "FeatureCollection", "features": event_features,
            "summary": {"total_events": len(events), "by_type": by_type},
            "comparison_stats": comparison["stats"],
            "meta": {"date_a": date_a, "date_b": date_b, "min_change_m": min_change,
                     "processing_time_s": round(time.time()-t0, 2)},
        })
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# === SECTION: /api/v1/changes/trees endpoint ===

@app.route('/api/v1/changes/trees', methods=['POST'])
def changes_trees():
    """Per-tree growth / felling analysis between two dates."""
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        date_a = params.get('date_a', '20220915')
        date_b = params.get('date_b', '20240915')

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(200)
        _validate_area(geom_3035)

        tree_changes = tca.detect_tree_growth(geom_3035, date_a, date_b)

        tree_features = []
        for tc in tree_changes:
            centroid_wgs = ti.geometry_from_3035(Point(tc.centroid_e, tc.centroid_n))
            tree_features.append({"type": "Feature", "properties": {
                "tree_id": tc.tree_id, "status": tc.status,
                "height_before_m": tc.height_before, "height_after_m": tc.height_after,
                "height_change_m": tc.height_change,
                "crown_area_before_sqm": tc.crown_area_before,
                "crown_area_after_sqm": tc.crown_area_after,
            }, "geometry": mapping(centroid_wgs)})

        by_status = {}
        for tc in tree_changes:
            by_status.setdefault(tc.status, {"count": 0, "mean_dh": []})
            by_status[tc.status]["count"] += 1
            by_status[tc.status]["mean_dh"].append(tc.height_change)
        for v in by_status.values():
            dh_list = v.pop("mean_dh")
            v["height_change_mean_m"] = round(float(np.mean(dh_list)), 2) if dh_list else 0

        return jsonify({
            "type": "FeatureCollection", "features": tree_features,
            "summary": {"total_trees": len(tree_changes), "by_status": by_status},
            "meta": {"date_a": date_a, "date_b": date_b,
                     "processing_time_s": round(time.time()-t0, 2)},
        })
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# === SECTION: Multi-epoch summary ===

@app.route('/api/v1/changes/summary', methods=['POST'])
def changes_summary():
    """Multi-epoch change summary across all available dates."""
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dates = params.get('dates', None)
        if isinstance(dates, str):
            dates = [d.strip() for d in dates.split(',')]

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(200)
        _validate_area(geom_3035)

        result = tca.temporal_summary(geom_3035, dates=dates)
        result["meta"] = {"processing_time_s": round(time.time()-t0, 2)}
        return jsonify(result)
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# === SECTION: /api/v1/info endpoint ===

# === SECTION: LiDAR/ortho/DTM/CIR/Hansen overlay + GeoTIFF download ===

def _extract_single_geom(features_or_geom):
    """Extract a single shapely geometry from _get_geometry() output."""
    if isinstance(features_or_geom, list):
        if not features_or_geom:
            raise ValueError("No geometry provided.")
        feat = features_or_geom[0]
        if isinstance(feat, dict) and 'geometry' in feat:
            return feat['geometry']
        return shape(feat) if isinstance(feat, dict) else feat
    return features_or_geom


def _geometry_to_3035_bbox(features_or_geom):
    """Convert WGS84 geometry to EPSG:3035 and return (geom_3035, bbox_3035, bbox_wgs84)."""
    geom_wgs84 = _extract_single_geom(features_or_geom)
    geom_3035 = ti.geometry_to_3035(geom_wgs84)
    _validate_area(geom_3035)
    b = geom_3035.bounds  # (minx, miny, maxx, maxy)
    bw = geom_wgs84.bounds
    return geom_3035, b, bw


def _get_cached_raster(geom_3035, dataset):
    """Return DTM/DSM data from cache if available, else read from remote."""
    cache_key = f"{geom_3035.bounds}_{dataset}"
    if _raster_cache["key"] == cache_key and _raster_cache["data"] is not None:
        log.info("raster cache hit for %s", cache_key)
        return _raster_cache["data"]
    log.info("raster cache miss, reading from remote")
    data = raster_io.read_dtm_dsm(geom_3035, dataset)
    _raster_cache.update({"key": cache_key, "data": data})
    return data


def _hillshade(elevation, azimuth=315, altitude=45, z_factor=1.0):
    """Compute hillshade from an elevation array."""
    dy, dx = np.gradient(elevation, 1.0)
    dx *= z_factor
    dy *= z_factor
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    hs = (np.sin(alt_rad) * np.cos(slope) +
          np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return np.clip(hs, 0, 1).astype(np.float32)


def _dtm_rgba(dtm, mask):
    """Render DTM as hillshade relief with hypsometric tinting.

    Shows actual terrain: valleys, ridges, slopes — the relief you see on
    a good topographic map.
    """
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    h, w = dtm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # --- Multi-directional hillshade for rich relief ---
    hs1 = _hillshade(dtm, azimuth=315, altitude=40)
    hs2 = _hillshade(dtm, azimuth=90, altitude=55)
    hs_combined = np.clip(0.65 * hs1 + 0.35 * hs2, 0, 1)

    # --- Hypsometric tint based on DTM elevation ---
    vmin = float(np.nanpercentile(dtm[mask], 2)) if mask.any() else 0
    vmax = float(np.nanpercentile(dtm[mask], 98)) if mask.any() else 1000
    if vmax - vmin < 10:
        vmax = vmin + 10

    hypso_colors = [
        (0.0, '#2d6a2e'),   # valley: deep green
        (0.15, '#5a9e3c'),  # lower slopes: green
        (0.3, '#8ebb4a'),   # mid-low: yellow-green
        (0.45, '#c4b85c'),  # mid: olive/tan
        (0.6, '#b8956a'),   # upper mid: brown
        (0.75, '#a08070'),  # upper slopes: grey-brown
        (0.9, '#c8c0b8'),   # near summit: light grey
        (1.0, '#f0ece8'),   # summit: near-white
    ]
    cmap_hypso = LinearSegmentedColormap.from_list('hypso', hypso_colors)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    hypso = cmap_hypso(norm(np.clip(dtm, vmin, vmax)))[:, :, :3]  # (H,W,3) float

    # Modulate by hillshade
    for c in range(3):
        hypso[:, :, c] = hypso[:, :, c] * (0.25 + 0.75 * hs_combined)

    rgba[:, :, :3] = (np.clip(hypso, 0, 1) * 255).astype(np.uint8)
    rgba[:, :, 3] = np.where(mask, 255, 0)
    return rgba


def _ndsm_rgba(ndsm, dsm, mask, vmax=45):
    """Render nDSM as viridis height coloring with DSM hillshade for 3D effect.

    Uses sqrt scaling to better distinguish old-growth forest (25-45m).
    Ground (ndsm < 0.3) is transparent so it can layer over the DTM relief.
    """
    import matplotlib
    import matplotlib.cm as cm

    h, w = ndsm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    hs_dsm = _hillshade(dsm, azimuth=315, altitude=40)

    colormap = cm.get_cmap('viridis')
    ndsm_clamped = np.clip(ndsm, 0, vmax)
    # sqrt scaling: spreads upper range for better old-growth forest distinction
    ndsm_sqrt = np.sqrt(ndsm_clamped / vmax)
    rgb_float = colormap(ndsm_sqrt)[:, :, :3]  # (H,W,3)

    # Modulate by DSM hillshade
    for c in range(3):
        rgb_float[:, :, c] = rgb_float[:, :, c] * (0.3 + 0.7 * hs_dsm)

    elevated = mask & (ndsm > 0.3)
    rgba[:, :, :3] = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    rgba[:, :, 3] = np.where(elevated, 220, 0)
    return rgba


def _reproject_rasters_to_wgs84(arrays_3035, transform_3035, shape_3035, mask_3035=None):
    """Reproject raw float32 rasters from EPSG:3035 to EPSG:4326.

    Parameters
    ----------
    arrays_3035 : dict[str, np.ndarray]
        Named 2D float32 arrays in EPSG:3035 (e.g. dtm, dsm, ndsm).
    transform_3035 : Affine
        Source rasterio transform.
    shape_3035 : (int, int)
        Source (rows, cols).
    mask_3035 : np.ndarray or None
        Boolean mask. Reprojected as nearest-neighbour.

    Returns
    -------
    (arrays_wgs, mask_wgs, transform_wgs, bounds_wgs)
        arrays_wgs: dict with same keys, reprojected.
        mask_wgs: boolean mask in WGS84 grid.
        transform_wgs: rasterio Affine for the WGS84 grid.
        bounds_wgs: (south, west, north, east) for Leaflet.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds

    src_crs = CRS.from_epsg(3035)
    dst_crs = CRS.from_epsg(4326)
    h, w = shape_3035

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, w, h, *array_bounds(h, w, transform_3035),
    )

    arrays_wgs = {}
    for name, src in arrays_3035.items():
        dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=src.astype(np.float32),
            destination=dst,
            src_transform=transform_3035,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        arrays_wgs[name] = dst

    # Reproject mask
    if mask_3035 is not None:
        mask_src = mask_3035.astype(np.uint8)
        mask_dst = np.zeros((dst_height, dst_width), dtype=np.uint8)
        reproject(
            source=mask_src,
            destination=mask_dst,
            src_transform=transform_3035,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        mask_wgs = mask_dst > 0
    else:
        # Derive mask from any reprojected array (non-NaN)
        first = next(iter(arrays_wgs.values()))
        mask_wgs = ~np.isnan(first)

    bounds = array_bounds(dst_height, dst_width, dst_transform)
    # (left, bottom, right, top) = (west, south, east, north)
    bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])  # south, west, north, east

    return arrays_wgs, mask_wgs, dst_transform, bounds_wgs


def _reproject_rgb_to_wgs84(rgb_3035, transform_3035, shape_3035):
    """Reproject a (3,H,W) uint8 RGB array from EPSG:3035 to EPSG:4326.

    Returns (rgb_wgs, mask_wgs, bounds_wgs).
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds

    src_crs = CRS.from_epsg(3035)
    dst_crs = CRS.from_epsg(4326)
    h, w = shape_3035

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, w, h, *array_bounds(h, w, transform_3035),
    )

    rgb_wgs = np.zeros((3, dst_height, dst_width), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=rgb_3035[band],
            destination=rgb_wgs[band],
            src_transform=transform_3035,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    mask_wgs = (rgb_wgs[0] > 0) | (rgb_wgs[1] > 0) | (rgb_wgs[2] > 0)

    bounds = array_bounds(dst_height, dst_width, dst_transform)
    bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])

    return rgb_wgs, mask_wgs, bounds_wgs


def _send_rgba_overlay(rgba, bounds_wgs):
    """Encode RGBA array as PNG and return Flask response with bounds header."""
    from PIL import Image
    img = Image.fromarray(rgba, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, 'PNG', optimize=True)
    buf.seek(0)
    resp = send_file(buf, mimetype='image/png')
    south, west, north, east = bounds_wgs
    resp.headers['X-Bounds'] = f'{south},{west},{north},{east}'
    resp.headers['Access-Control-Expose-Headers'] = 'X-Bounds'
    return resp


@app.route('/api/v1/dtm/overlay', methods=['POST'])
def dtm_overlay():
    """Return DTM hillshade relief as a PNG overlay (reprojected to WGS84)."""
    try:
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        data = _get_cached_raster(geom_3035, dataset)

        # Reproject raw elevation to WGS84 *before* computing hillshade
        arrays_wgs, mask_wgs, tf_wgs, bounds_wgs = _reproject_rasters_to_wgs84(
            {'dtm': data['dtm']},
            data['transform'], data['shape'], data['mask'],
        )
        dtm_wgs = arrays_wgs['dtm']
        # Fill NaN for hillshade computation
        dtm_wgs = np.nan_to_num(dtm_wgs, nan=float(np.nanmedian(dtm_wgs[mask_wgs])) if mask_wgs.any() else 0)

        rgba = _dtm_rgba(dtm_wgs, mask_wgs)
        return _send_rgba_overlay(rgba, bounds_wgs)
    except Exception as e:
        log.error("dtm overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/lidar/overlay', methods=['POST'])
def lidar_overlay():
    """Return nDSM as a PNG overlay (viridis + hillshade, reprojected to WGS84)."""
    try:
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        data = _get_cached_raster(geom_3035, dataset)

        # Reproject raw elevation to WGS84 before rendering
        arrays_wgs, mask_wgs, tf_wgs, bounds_wgs = _reproject_rasters_to_wgs84(
            {'ndsm': data['ndsm'], 'dsm': data['dsm']},
            data['transform'], data['shape'], data['mask'],
        )
        ndsm_wgs = np.nan_to_num(arrays_wgs['ndsm'], nan=0)
        dsm_wgs = np.nan_to_num(arrays_wgs['dsm'], nan=0)

        rgba = _ndsm_rgba(ndsm_wgs, dsm_wgs, mask_wgs)
        return _send_rgba_overlay(rgba, bounds_wgs)
    except Exception as e:
        log.error("lidar overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/ortho/overlay', methods=['POST'])
def ortho_overlay():
    """Return orthophoto as a PNG overlay (reprojected to WGS84)."""
    try:
        import ortho_io
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        ortho_year = params.get('ortho_year')
        if ortho_year:
            ortho_year = int(ortho_year)
        data = _get_cached_raster(geom_3035, dataset)

        # Use cached ortho if available and matches (same geometry, default year)
        raster_cache_key = f"{geom_3035.bounds}_{dataset}"
        if (not ortho_year and _raster_cache.get("ortho") is not None
                and _raster_cache.get("ortho_key") == raster_cache_key):
            log.info("ortho overlay: using cached ortho")
            rgb = _raster_cache["ortho"]
            nir = None  # nir not needed for RGB overlay
        else:
            rgb, nir = ortho_io.read_ortho_for_als(data, year=ortho_year)

        # Reproject RGB to WGS84
        rgb_wgs, mask_wgs, bounds_wgs = _reproject_rgb_to_wgs84(
            rgb, data['transform'], data['shape'],
        )

        h, w = rgb_wgs.shape[1], rgb_wgs.shape[2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = rgb_wgs[0]
        rgba[:, :, 1] = rgb_wgs[1]
        rgba[:, :, 2] = rgb_wgs[2]
        rgba[:, :, 3] = np.where(mask_wgs, 255, 0)

        return _send_rgba_overlay(rgba, bounds_wgs)
    except Exception as e:
        log.error("ortho overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/cir/overlay', methods=['POST'])
def cir_overlay():
    """Return Color Infrared (CIR) false-color overlay.

    CIR maps NIR→Red, Red→Green, Green→Blue, highlighting vegetation
    health (bright red = vigorous vegetation, dark = bare/water).
    """
    try:
        import ortho_io
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        ortho_year = params.get('ortho_year')
        if ortho_year:
            ortho_year = int(ortho_year)
        data = _get_cached_raster(geom_3035, dataset)

        rgb, nir = ortho_io.read_ortho_for_als(data, year=ortho_year)
        if nir is None:
            return _error('NIR band not available for this area (no RGBI operate coverage)', 404)

        # CIR: NIR→R, Red→G, Green→B
        cir = np.stack([nir, rgb[0], rgb[1]], axis=0)  # (3,H,W) uint8

        cir_wgs, mask_wgs, bounds_wgs = _reproject_rgb_to_wgs84(
            cir, data['transform'], data['shape'],
        )

        h, w = cir_wgs.shape[1], cir_wgs.shape[2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = cir_wgs[0]
        rgba[:, :, 1] = cir_wgs[1]
        rgba[:, :, 2] = cir_wgs[2]
        rgba[:, :, 3] = np.where(mask_wgs, 255, 0)

        return _send_rgba_overlay(rgba, bounds_wgs)
    except Exception as e:
        log.error("cir overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/hansen/overlay', methods=['POST'])
def hansen_overlay():
    """Return Hansen forest loss as a coloured PNG overlay.

    Green = current forest, magenta = loss (brighter = more recent),
    cyan = forest gain.
    """
    try:
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        geom_wgs = _extract_single_geom(geom_wgs84)
        bbox_wgs = geom_wgs.bounds

        prior = hansen.get_forest_prior(
            bbox_wgs, data['transform'], data['shape'],
        )

        h, w = data['shape']
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Current forest: green
        forest = prior['current_forest']
        rgba[:, :, 0][forest] = 20
        rgba[:, :, 1][forest] = 120
        rgba[:, :, 2][forest] = 20
        rgba[:, :, 3][forest] = 140

        # Forest gain: cyan
        gain = prior['gain']
        rgba[:, :, 0][gain] = 0
        rgba[:, :, 1][gain] = 200
        rgba[:, :, 2][gain] = 200
        rgba[:, :, 3][gain] = 180

        # Loss: magenta, brightness by recency (year 1=2001 dark, 24=2024 bright)
        ly = prior['loss_year']
        loss = ly > 0
        brightness = np.clip(80 + (ly.astype(np.float32) / 24.0) * 175, 80, 255).astype(np.uint8)
        rgba[:, :, 0][loss] = brightness[loss]
        rgba[:, :, 1][loss] = 0
        rgba[:, :, 2][loss] = brightness[loss]
        rgba[:, :, 3][loss] = 200

        # Transparent where no data
        rgba[:, :, 3][~data['mask']] = 0

        from rasterio.warp import calculate_default_transform, reproject as rp, Resampling
        from rasterio.crs import CRS
        from rasterio.transform import array_bounds

        src_crs = CRS.from_epsg(3035)
        dst_crs = CRS.from_epsg(4326)
        dst_tf, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, w, h, *array_bounds(h, w, data['transform']),
        )
        rgba_wgs = np.zeros((4, dst_h, dst_w), dtype=np.uint8)
        for band in range(4):
            rp(
                source=rgba[:, :, band],
                destination=rgba_wgs[band],
                src_transform=data['transform'],
                src_crs=src_crs,
                dst_transform=dst_tf,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
        rgba_out = np.transpose(rgba_wgs, (1, 2, 0))
        bounds = array_bounds(dst_h, dst_w, dst_tf)
        bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])
        return _send_rgba_overlay(rgba_out, bounds_wgs)
    except Exception as e:
        log.error("hansen overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/lidar/geotiff', methods=['POST'])
def lidar_geotiff():
    """Download nDSM as a georeferenced GeoTIFF."""
    try:
        geom_wgs84 = _extract_single_geom(_get_geometry())
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035 = ti.geometry_to_3035(geom_wgs84)
        _validate_area(geom_3035)

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        ndsm = data['ndsm']
        dtm = data['dtm']
        dsm = data['dsm'] if 'dsm' in data else dtm + ndsm
        tf = data['transform']
        h, w = data['shape']

        tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        with rasterio.open(tmp.name, 'w', driver='GTiff', width=w, height=h,
                           count=3, dtype='float32', crs='EPSG:3035',
                           transform=tf, compress='deflate') as dst:
            dst.write(dtm.astype(np.float32), 1)
            dst.write(dsm.astype(np.float32), 2)
            dst.write(ndsm.astype(np.float32), 3)
            dst.set_band_description(1, 'DTM')
            dst.set_band_description(2, 'DSM')
            dst.set_band_description(3, 'nDSM')

        return send_file(tmp.name, mimetype='image/tiff', as_attachment=True,
                         download_name='lidar_dtm_dsm_ndsm.tif')
    except Exception as e:
        log.error("lidar geotiff: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/ortho/geotiff', methods=['POST'])
def ortho_geotiff():
    """Download orthophoto as a georeferenced GeoTIFF (RGB + NIR if available)."""
    try:
        import ortho_io
        geom_wgs84 = _extract_single_geom(_get_geometry())
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035 = ti.geometry_to_3035(geom_wgs84)
        _validate_area(geom_3035)

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        rgb, nir = ortho_io.read_ortho_for_als(data)
        tf = data['transform']
        h, w = data['shape']

        n_bands = 4 if nir is not None else 3
        tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        with rasterio.open(tmp.name, 'w', driver='GTiff', width=w, height=h,
                           count=n_bands, dtype='uint8', crs='EPSG:3035',
                           transform=tf, compress='deflate') as dst:
            dst.write(rgb[0], 1)
            dst.write(rgb[1], 2)
            dst.write(rgb[2], 3)
            dst.set_band_description(1, 'Red')
            dst.set_band_description(2, 'Green')
            dst.set_band_description(3, 'Blue')
            if nir is not None:
                dst.write(nir, 4)
                dst.set_band_description(4, 'NIR')

        return send_file(tmp.name, mimetype='image/tiff', as_attachment=True,
                         download_name='orthophoto_rgbi.tif')
    except Exception as e:
        log.error("ortho geotiff: %s", traceback.format_exc())
        return _error(str(e))


# === SECTION: RF classifier training endpoints ===

@app.route('/api/v1/classifier/train', methods=['POST'])
def train_classifier():
    """Train RF classifier from cadastre ground truth over a bbox.

    Params: geometry (bbox/geojson), dataset, include_ortho, include_copernicus,
            include_temporal, include_hansen.
    Fetches segment features + cadastre parcel codes, trains RF model.
    All data sources (ortho, copernicus, hansen, temporal) are included by
    default so the RF sees the same features it will use at inference time.
    """
    try:
        import learned_classifier as lc
        import object_segmentation as oc
        import cadastre

        params = _parse_params()
        geom, geom_3035 = _parse_geometry(params)
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        obs_year = ti.dataset_to_year(dataset)

        include_temporal = str(params.get('include_temporal', 'true')).lower() in ('true', '1', 'yes')
        include_hansen = str(params.get('include_hansen', 'true')).lower() in ('true', '1', 'yes')

        # Read LIDAR
        data = raster_io.read_dtm_dsm(geom_3035, dataset)

        # Multi-temporal DTM/DSM
        dtm_dates, dsm_dates = None, None
        if include_temporal:
            try:
                multi = raster_io.read_multi_date_ndsm(geom_3035)
                dtm_dates, dsm_dates = {}, {}
                for d in multi['dates_loaded']:
                    try:
                        dd = raster_io.read_dtm_dsm(geom_3035, dataset=d)
                        mh = min(dd['shape'][0], data['shape'][0])
                        mw = min(dd['shape'][1], data['shape'][1])
                        dtm_dates[d] = dd['dtm'][:mh, :mw]
                        dsm_dates[d] = dd['dsm'][:mh, :mw]
                    except Exception as e:
                        log.warning("Train: date %s load failed: %s", d, e)
            except Exception as e:
                log.warning("Train: multi-temporal failed: %s", e)

        # Read ortho
        rgb, spectral = _try_read_ortho(data)

        # Copernicus (NDVI, land cover, SAR, harmonics)
        copernicus_data = None
        if not _is_processor_running():
            copernicus_data = _try_copernicus(geom, sar=True, harmonics=True, year=ti.dataset_to_year(dataset))

        # Hansen forest prior
        hansen_data = None
        if include_hansen:
            hansen_data = _try_hansen(geom, data['transform'], data['shape'])

        # Building footprints from cadastre (for calibration features)
        building_footprints = _try_cadastre(geom, data['transform'], data['shape'])

        # Infrastructure lookup
        _il3 = None
        include_infra = str(params.get('include_infra', 'true')).lower() in ('true', '1', 'yes')
        if include_infra:
            try:
                from infrastructure_lookup import InfrastructureLookup
                from pyproj import Transformer as _Tx4
                _tx4 = _Tx4.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
                _b3 = geom.bounds
                _w43, _s43 = _tx4.transform(_b3[0], _b3[1])
                _e43, _n43 = _tx4.transform(_b3[2], _b3[3])
                _il3 = InfrastructureLookup.for_bbox(_w43, _s43, _e43, _n43)
            except Exception:
                pass

        # Segment (feature extraction) — pass ALL data sources
        result = oc.segment_and_classify(
            data['dtm'], data['dsm'], data['mask'], data['transform'],
            dtm_dates=dtm_dates,
            dsm_dates=dsm_dates,
            spectral=spectral,
            copernicus=copernicus_data,
            building_footprints=building_footprints,
            hansen=hansen_data,
            ortho_year=obs_year,
            observation_year=obs_year,
            infra_lookup=_il3,
        )
        objects = result['objects']
        labels_arr = result['labels']

        # Hansen tree-loss calibration
        if include_hansen and hansen_data:
            try:
                objects = hansen.calibrate_tree_loss(
                    objects, labels_arr, hansen_data,
                    observation_year=obs_year,
                )
            except Exception as e:
                log.warning("Train: Hansen calibration failed: %s", e)

        features = [obj.features for obj in objects]

        # Fetch cadastre parcel codes
        bbox_wgs = geom.bounds
        try:
            parcels = cadastre.fetch_parcel_land_use(
                (bbox_wgs[0], bbox_wgs[1], bbox_wgs[2], bbox_wgs[3]))
        except Exception:
            parcels = None

        if parcels is None:
            return _error("Could not fetch cadastre parcel codes")

        # Match segments to cadastre labels
        train_features = []
        train_labels = []
        for feat in features:
            code = _dominant_cadastre_code(feat, parcels)
            if code and code in lc.CADASTRE_TO_TYPE:
                train_features.append(feat)
                train_labels.append(lc.CADASTRE_TO_TYPE[code])

        if len(train_features) < 20:
            return _error(f"Only {len(train_features)} labelled segments, need >= 20")

        clf = lc.LearnedClassifier()
        stats = clf.train(train_features, train_labels)

        # Clear cached raster data to reclaim memory
        _clear_raster_caches()

        return jsonify({
            "status": "trained",
            "training_stats": stats,
            "n_segments_total": len(features),
            "n_segments_labelled": len(train_features),
            "dataset": dataset,
            "observation_year": obs_year,
            "data_sources": {
                "temporal": dtm_dates is not None and len(dtm_dates) >= 2,
                "ortho": spectral is not None,
                "copernicus": copernicus_data is not None,
                "hansen": hansen_data is not None,
                "building_footprints": building_footprints is not None,
            },
        })

    except Exception as e:
        log.error("classifier train: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/classifier/status', methods=['GET'])
def classifier_status():
    """Check if a trained RF model exists."""
    try:
        import learned_classifier as lc
        clf = lc.get_classifier()
        # Report which model source is active
        if lc.BEST_MODEL_PATH.exists() and lc.BEST_META_PATH.exists():
            model_source = "best_model"
        elif lc.MODEL_PATH.exists():
            model_source = "live"
        else:
            model_source = "none"
        return jsonify({
            "trained": clf.is_trained,
            "trained_at": clf.trained_at,
            "n_kgs": clf.n_kgs,
            "n_train": clf.n_train,
            "oob_score": clf.oob_score,
            "n_classes": len(clf.classes),
            "classes": clf.classes,
            "model_source": model_source,
            "top_features": dict(sorted(
                clf.feature_importances.items(),
                key=lambda x: -x[1]
            )[:15]) if clf.feature_importances else {},
        })
    except Exception as e:
        return _error(str(e))


def _dominant_cadastre_code(feat, parcels):
    """Find the most common cadastre code overlapping a segment."""
    # parcels is expected to be a list of {geometry, code} or similar
    # For now, find parcels whose centroid falls within segment bbox
    ce = feat.get("centroid_e", 0)
    cn = feat.get("centroid_n", 0)
    if not parcels:
        return None
    # Simple: find nearest parcel
    best = None
    best_dist = float('inf')
    for p in parcels:
        pc = p.get("centroid")
        code = p.get("code")
        if pc and code:
            d = ((pc[0] - ce)**2 + (pc[1] - cn)**2)**0.5
            if d < best_dist:
                best_dist = d
                best = code
    return best


# === SECTION: /api/v1/docs + share endpoints ===

@app.route('/api/v1/ping', methods=['GET'])
def ping():
    """Cheap liveness probe — NEVER blocks on disk/network.

    Used by the local watchdog (`srv_watchdog.sh`) and the director to
    distinguish 'gunicorn is alive' from 'gunicorn is wedged'.  Must stay
    fast (no FS I/O, no subprocess, no requests).
    """
    import time as _t
    return jsonify({'pong': True, 'ts': _t.time()})


@app.route('/api/v1/info', methods=['GET'])
def info():
    return jsonify({
        "name": "Austrian LIDAR & Orthophoto Analysis API",
        "version": "3.0.0",
        "description": "Landscape transformation analysis: LIDAR DTM/DSM time series + Sentinel-2 + cadastre",
        "classifier": "landscape_v2 — 10 types focused on human transformation of terrain",
        "source": "data.bev.gv.at",
        "resolution_lidar": "1m",
        "resolution_ortho": "0.2m (1m for analysis, 0.5m for GLCM texture)",
        "crs": "EPSG:3035",
        "datasets_als": {k: {"dtm": True, "dsm": True} for k in sorted(ti.DATASETS.keys())},
        "datasets_ortho": ["20220128 (RGB 50km tiles)"],
        "tiles": len(ti.TILE_COORDS),
        "landscape_types": seg.OBJECT_TYPES,
        "data_sources": {
            "bev_als_dtm_dsm": "1m resolution, 3 dates (2022/2023/2024)",
            "bev_dop_rgbi": "0.2m orthophoto, 47 RGBI operates",
            "copernicus_sentinel2": "10m NDVI growing-season composite (via openEO)",
            "copernicus_worldcover": "10m ESA land cover classification",
            "copernicus_sentinel1_sar": "SAR backscatter (VV+VH)",
            "cadastre_footprints": "mm-precision building polygons (ground truth)",
        },
        "change_event_types": tca.EVENT_TYPES,
        "endpoints": {
            "POST /api/v1/elevation": "Enrich features with DSM/DTM elevation",
            "POST /api/v1/terrain": "Terrain characterisation (slope, ruggedness, etc.)",
            "POST /api/v1/segment": "Watershed segmentation: 25 object types + 11 group types (Felzenszwalb+RAG)",
            "POST /api/v1/changes": "Temporal change detection (earthworks, trees, buildings, roads)",
            "POST /api/v1/changes/trees": "Per-tree growth / felling analysis",
            "POST /api/v1/changes/summary": "Multi-epoch change summary (2022→2023→2024)",
            "GET /api/v1/info": "This endpoint",
            "GET /api/v1/docs/llm.txt": "Machine-readable API reference",
        },
        "max_area_sqkm": MAX_AREA_SQM / 1e6,
        "proxy_pool": __import__('bev_proxy').status(),
        # Capability flags. The director uses these to decide whether to
        # send features the peer supports (graceful upgrade path).
        "capabilities": [
            'cred_subset_env',     # honors COPERNICUS_CRED_INDICES
            'lat_strip_filter',    # honors KG_LAT_STRIP_FILTER
            'cell_filter',         # honors KG_CELL_FILTER (1° lat × 2° lon)
            'cred_api_v1',         # /api/v1/credentials available
            'parallel_frontiers',  # safe to run alongside other frontiers
        ],
        "capability_version": 1,
    })


# === SECTION: /api/v1/layers endpoint (layer availability check) ===

@app.route('/api/v1/layers', methods=['GET'])
def layers_availability():
    """Return which data layers are available for a given bounding box.

    Query params:
      bbox=lon_min,lat_min,lon_max,lat_max   (WGS84)
    """
    try:
        import ortho_io
        bbox_str = request.args.get('bbox', '')
        if not bbox_str:
            return _error('bbox parameter required (lon_min,lat_min,lon_max,lat_max)')
        parts = [float(x.strip()) for x in bbox_str.split(',')]
        if len(parts) != 4:
            return _error('bbox must have 4 values: lon_min,lat_min,lon_max,lat_max')
        lon_min, lat_min, lon_max, lat_max = parts

        # Tile coverage check
        from shapely.geometry import box as shp_box
        bbox_wgs = shp_box(lon_min, lat_min, lon_max, lat_max)
        geom_3035 = ti.geometry_to_3035(bbox_wgs)
        b = geom_3035.bounds
        tiles = ti.find_tiles_for_bbox(b[0], b[1], b[2], b[3])
        has_tiles = len(tiles) > 0

        # RGBI operates per year (needed for CIR; ortho falls back to DOP)
        rgbi_by_year = {}
        for year in (2024, 2023):
            try:
                operates = ortho_io.find_rgbi_operates(
                    lat_min, lon_min, lat_max, lon_max, year=year,
                )
                rgbi_by_year[str(year)] = len(operates) > 0
            except Exception:
                rgbi_by_year[str(year)] = False
        # The "~2020" slot covers the 20221027 series (years 2018-2021)
        try:
            old_ops = ortho_io.find_rgbi_operates(
                lat_min, lon_min, lat_max, lon_max,
            )
            # Filter to operates from the 20221027 series
            rgbi_by_year["2020"] = any(
                ortho_io.RGBI_OPERATES[o]["series"] == "20221027"
                for o in old_ops
            )
        except Exception:
            rgbi_by_year["2020"] = False

        # Ortho: RGBI operates preferred, but DOP (2022) is the fallback
        # So ortho is available if tiles exist (DOP covers all Austria)
        # We mark the year that has RGBI as "best" but the ~2020 slot
        # also covers the DOP 2022 fallback.
        ortho_result = {
            "2024": rgbi_by_year.get("2024", False) or has_tiles,
            "2023": rgbi_by_year.get("2023", False),
            "2020": has_tiles,  # DOP ~2022 always available where tiles exist
        }

        # CIR requires NIR from RGBI operates — no DOP fallback
        cir_result = {
            "2024": rgbi_by_year.get("2024", False),
            "2023": rgbi_by_year.get("2023", False),
            "2020": rgbi_by_year.get("2020", False),
        }

        dtm_result = {y: has_tiles for y in ('2024', '2023', '2022')}
        dsm_result = {y: has_tiles for y in ('2024', '2023', '2022')}
        hansen_available = has_tiles

        resp = {
            "ortho": ortho_result,
            "cir": cir_result,
            "dtm": dtm_result,
            "dsm": dsm_result,
            "hansen": hansen_available,
        }
        if _is_processor_running():
            # Check if tile cache covers this bbox — if so, Copernicus is
            # still available from cache even while processor runs.
            try:
                from tile_cache import CopernicusTileCache
                _cop_cache = CopernicusTileCache()
                _cop_bbox = {"west": lon_min, "south": lat_min,
                             "east": lon_max, "north": lat_max}
                if not _cop_cache.has_cached(_cop_bbox, ndvi=True, landcover=True,
                                            sar=True, harmonics=True):
                    resp["copernicus_disabled"] = True
            except Exception:
                resp["copernicus_disabled"] = True
        return jsonify(resp)
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error("layers endpoint: %s", traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# === SECTION: /api/v1/onestop endpoint (single-URL segment + download) ===

@app.route('/api/v1/onestop', methods=['GET'])
def onestop():
    """One-stop URL: trigger segmentation and get result/download from a single GET.

    Designed for users on limited connections who want a bookmarkable URL that
    runs analysis and produces a downloadable result.

    Query params (all via URL):
      bbox=lon_min,lat_min,lon_max,lat_max   Bounding box (required)
      name=MySave                            Save name / share ID (slug, 1-80 chars, [A-Za-z0-9_-])
      min_object_size=10                     Min segment area in m² (default: 10)
      include_ortho=true                     Include orthophoto (default: true)
      include_temporal=false                 Include temporal analysis
      include_copernicus=false               Include Sentinel-2/SAR
      include_cadastre=false                 Include cadastre
      include_hansen=false                   Include Hansen forest change
      include_infra=true                     Include infrastructure spatial match
      types=tree,road                        Object type filter
      height_min=X                           Height filter: >= X metres
      height_max=X                           Height filter: <= X metres
      height_op=gt|lt|between                Height filter operator
      format=json|gpkg|kml                   Output format (default: json)
      layers=segments                        GPKG layers (default: segments)
      include_segments_vector=true            GPKG vector polygons (default: true)
      segment_geometry=point|polygon         Feature geometry in KML/GPKG (default: point for KML, polygon for GPKG)
      segment_geometry_style=type|height      Colour scheme: by object type or height ramp (default: type)
      group_by=type|height_class             KML folder grouping

    Returns:
      - If no task running: starts analysis, returns 202 JSON with task_id + poll URL
      - Poll with same URL + task_id=X or via /api/v1/segment/progress?task_id=X
      - When done: returns the file (gpkg/kml) or JSON, plus auto_share_id
      - Add &task_id=X to check/download a previously started task

    The result is auto-saved as a share for later access.
    Processing is queued (max 2 concurrent, 4 in queue) to prevent overload.

    Timing estimates (< 1 km², ortho=true):
      ~30-60s with ortho only, ~60-90s with ortho+temporal,
      ~90-120s with all sources. GPKG/KML adds ~5-10s.
    """
    try:
        bbox_str = request.args.get('bbox', '')
        task_id = request.args.get('task_id', '')
        fmt = request.args.get('format', 'json').lower().strip()

        # If task_id provided, check its status
        if task_id:
            return _onestop_check(task_id, fmt, dict(request.args))

        # Parse bbox
        if not bbox_str:
            return _error('bbox parameter required (lon_min,lat_min,lon_max,lat_max)', 400)
        parts = [x.strip() for x in bbox_str.split(',')]
        if len(parts) != 4:
            return _error('bbox must have 4 values: lon_min,lat_min,lon_max,lat_max', 400)
        try:
            lon_min, lat_min, lon_max, lat_max = [float(x) for x in parts]
        except ValueError:
            return _error('bbox values must be numbers', 400)

        # Build polygon from bbox
        from shapely.geometry import box as shapely_box
        geom = shapely_box(lon_min, lat_min, lon_max, lat_max)
        features = [{'geometry': geom}]
        geo_parse.validate_austria_bounds(geom)
        geom_3035 = ti.geometry_to_3035(geom)
        _validate_area(geom_3035)

        # Build params
        params = {}
        for key in ('dataset', 'min_object_size', 'include_ortho', 'include_temporal',
                    'include_copernicus', 'include_cadastre', 'include_hansen', 'include_infra',
                    'types', 'groups', 'felz_scale', 'rag_threshold',
                    'height_min', 'height_max', 'height_op',
                    'top_n_classes', 'top_n_objects', 'min_height_m'):
            val = request.args.get(key)
            if val is not None:
                params[key] = val
        # Sensible defaults for one-stop
        params.setdefault('min_object_size', '10')
        params.setdefault('include_ortho', 'true')

        geometry_text = json.dumps(mapping(geom))

        # Queue check
        with _TASK_QUEUE_LOCK:
            if _TASK_QUEUE_SIZE >= MAX_QUEUE_SIZE:
                return jsonify({
                    'error': 'Server busy — too many tasks queued. Try again in a few minutes.',
                    'queue_size': _TASK_QUEUE_SIZE,
                    'retry_after_seconds': 120,
                }), 503

        # Start async task
        task_id = str(uuid.uuid4())
        _progress_start(task_id)
        _cleanup_old_results()

        # Store onestop params so we can produce the right format on completion
        onestop_meta = {
            'format': fmt,
            'params': {k: request.args.get(k) for k in request.args if k != 'task_id'},
        }
        (_PROGRESS_DIR / f"{task_id}.onestop.json").write_text(
            json.dumps(onestop_meta))

        thread = threading.Thread(
            target=_segment_worker,
            args=(task_id, features, params, geometry_text),
            daemon=True,
        )
        thread.start()

        est_seconds = _estimate_time(geom_3035, params)

        # Browser: return auto-polling HTML page
        if _wants_html():
            return _onestop_poll_page(task_id, fmt, step='queued',
                                     detail='Starting analysis…', est=est_seconds)

        host = request.headers.get('X-Forwarded-Host', request.host)
        proto = request.headers.get('X-Forwarded-Proto', request.scheme)
        base = f"{proto}://{host}"
        poll_url = f"{base}/api/v1/onestop?task_id={task_id}&format={fmt}"
        progress_url = f"{base}/api/v1/segment/progress?task_id={task_id}"

        return jsonify({
            'task_id': task_id,
            'status': 'running',
            'queue_size': _TASK_QUEUE_SIZE,
            'poll_url': poll_url,
            'progress_url': progress_url,
            'estimated_seconds': est_seconds,
            'message': 'Analysis started. Poll poll_url until status=done, then the final response will contain the download.',
        }), 202

    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error("onestop: %s", traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


def _estimate_time(geom_3035, params: dict) -> int:
    """Rough estimate of processing time in seconds for a given area + options."""
    area_sqm = geom_3035.area
    area_ha = area_sqm / 10000
    base = 20 + area_ha * 0.5  # ~20s base + 0.5s per hectare
    if str(params.get('include_ortho', 'true')).lower() in ('true', '1', 'yes'):
        base += 10 + area_ha * 0.3
    if str(params.get('include_temporal', 'false')).lower() in ('true', '1', 'yes'):
        base += 15 + area_ha * 0.4
    if str(params.get('include_copernicus', 'false')).lower() in ('true', '1', 'yes'):
        base += 20
    if str(params.get('include_cadastre', 'false')).lower() in ('true', '1', 'yes'):
        base += 5
    if str(params.get('include_hansen', 'false')).lower() in ('true', '1', 'yes'):
        base += 10
    return int(base)


def _wants_html():
    """Return True if the client prefers HTML (i.e. a browser)."""
    accept = request.headers.get('Accept', '')
    # Browsers send text/html first; curl/programmatic clients send */* or application/json
    return 'text/html' in accept and 'application/json' not in accept


def _onestop_poll_page(task_id: str, fmt: str, step: str = '', detail: str = '',
                       elapsed: int = 0, est: int = 0):
    """Return an HTML page that auto-polls onestop and redirects to the download."""
    host = request.headers.get('X-Forwarded-Host', request.host)
    proto = request.headers.get('X-Forwarded-Proto', request.scheme)
    base = f"{proto}://{host}"
    progress_url = f"{base}/api/v1/segment/progress?task_id={task_id}"
    download_url = f"{base}/api/v1/onestop?task_id={task_id}&format={fmt}"
    pct = min(95, int(elapsed / max(est, 1) * 100)) if est else 0
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Processing…</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         display: flex; justify-content: center; align-items: center; min-height: 100vh;
         margin: 0; background: #f5f7fa; color: #333; }}
  .card {{ background: #fff; border-radius: 12px; padding: 40px 48px; text-align: center;
           box-shadow: 0 2px 12px rgba(0,0,0,.08); max-width: 480px; width: 90%; }}
  h2 {{ margin: 0 0 8px; font-size: 20px; }}
  .sub {{ color: #888; font-size: 14px; margin-bottom: 24px; }}
  .bar-bg {{ background: #e9ecef; border-radius: 8px; height: 8px; overflow: hidden; margin: 16px 0; }}
  .bar {{ background: linear-gradient(90deg, #4361ee, #3a86ff); height: 100%; border-radius: 8px;
          transition: width .5s ease; }}
  .status {{ font-size: 14px; color: #555; min-height: 20px; }}
  .elapsed {{ font-size: 12px; color: #aaa; margin-top: 8px; }}
  .err {{ color: #dc3545; font-weight: 600; }}
</style>
</head><body><div class="card">
  <h2>⚙️ Processing</h2>
  <p class="sub">Your analysis is running. This page will auto-download when ready.</p>
  <div class="bar-bg"><div class="bar" id="bar" style="width:{pct}%"></div></div>
  <div class="status" id="status">{_html_esc(step)}: {_html_esc(detail)}</div>
  <div class="elapsed" id="elapsed">{elapsed}s elapsed{f' / ~{est}s est.' if est else ''}</div>
</div>
<script>
const progressUrl = "{progress_url}";
const downloadUrl = "{download_url}";
const est = {est};
let t0 = Date.now() - {elapsed * 1000};

async function poll() {{
  try {{
    const r = await fetch(progressUrl);
    const d = await r.json();
    const el = Math.round((Date.now() - t0) / 1000);
    document.getElementById('elapsed').textContent = el + 's elapsed' + (est ? ' / ~' + est + 's est.' : '');
    if (d.error) {{
      document.getElementById('status').innerHTML = '<span class="err">❌ ' + d.error + '</span>';
      document.getElementById('bar').style.background = '#dc3545';
      return;
    }}
    if (d.done) {{
      document.getElementById('bar').style.width = '100%';
      document.getElementById('status').textContent = 'Downloading…';
      window.location.href = downloadUrl;
      setTimeout(function() {{
        document.querySelector('h2').textContent = '✅ Done';
        document.getElementById('status').textContent = 'Download started. You can close this page.';
        document.querySelector('.sub').textContent = el + 's total processing time.';
      }}, 2000);
      return;
    }}
    const pct = Math.min(95, est ? Math.round(el / est * 100) : 50);
    document.getElementById('bar').style.width = pct + '%';
    document.getElementById('status').textContent = (d.step || '') + (d.detail ? ': ' + d.detail : '');
  }} catch(e) {{ /* retry */ }}
  setTimeout(poll, 3000);
}}
setTimeout(poll, 2000);
</script></body></html>"""
    return html, 200, {'Content-Type': 'text/html; charset=utf-8'}


def _html_esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def _onestop_check(task_id: str, fmt: str, url_params: dict):
    """Check status of a one-stop task. Return result when done."""
    if not task_id or not re.match(r'^[a-f0-9\-]+$', task_id):
        return _error('Invalid task_id', 400)

    p = _PROGRESS_DIR / f"{task_id}.json"
    if not p.exists():
        return _error('Task not found', 404)

    progress = json.loads(p.read_text())
    t0 = progress.get('t0', 0)
    elapsed = round(time.time() - t0) if t0 else 0

    if progress.get('step') == 'error':
        return jsonify({
            'task_id': task_id,
            'status': 'error',
            'error': progress.get('detail', 'Unknown error'),
            'elapsed_seconds': elapsed,
        }), 500

    host = request.headers.get('X-Forwarded-Host', request.host)
    proto = request.headers.get('X-Forwarded-Proto', request.scheme)
    base = f"{proto}://{host}"

    if progress.get('step') != 'done':
        if _wants_html():
            return _onestop_poll_page(task_id, fmt,
                                     step=progress.get('step', ''),
                                     detail=progress.get('detail', ''),
                                     elapsed=elapsed)
        return jsonify({
            'task_id': task_id,
            'status': 'running',
            'step': progress.get('step', ''),
            'detail': progress.get('detail', ''),
            'elapsed_seconds': elapsed,
            'poll_url': f"{base}/api/v1/onestop?task_id={task_id}&format={fmt}",
        }), 202

    # Done! Return result in requested format
    auto_share_id = progress.get('auto_share_id')
    result = _get_result(task_id)
    if not result:
        if auto_share_id:
            # Load from auto-save share
            share_path = SHARE_DIR / f"{auto_share_id}.json.gz"
            if share_path.exists():
                share_data = json.loads(gzip.decompress(share_path.read_bytes()))
                result = share_data.get('result')
        if not result:
            return _error('Result expired. Re-run the analysis.', 410)

    share_url = f"{base}/?share={auto_share_id}" if auto_share_id else None

    if fmt == 'json':
        result['_onestop'] = {
            'task_id': task_id,
            'elapsed_seconds': elapsed,
            'auto_share_id': auto_share_id,
            'share_url': share_url,
        }
        return jsonify(result)

    # For gpkg/kml, we need to build the export from the result
    obj_features = result.get('features', [])

    # Apply type filter
    type_filter_str = url_params.get('types')
    if type_filter_str:
        type_set = set(t.strip() for t in type_filter_str.split(','))
        obj_features = [f for f in obj_features
                       if f.get('properties', {}).get('type') in type_set]

    # Apply height filter
    obj_features = _apply_height_filter_features(obj_features, url_params)

    if fmt == 'kml':
        group_by = url_params.get('group_by', 'type')
        seg_geom = url_params.get('segment_geometry', 'point').lower().strip()

        kml_features = obj_features
        if seg_geom == 'polygon':
            # Vectorise segment raster into polygon features
            try:
                onestop_meta_path = _PROGRESS_DIR / f"{task_id}.onestop.json"
                meta_params = {}
                if onestop_meta_path.exists():
                    meta_params = json.loads(onestop_meta_path.read_text()).get('params', {})
                bbox_str = meta_params.get('bbox', url_params.get('bbox', ''))
                parts = [float(x.strip()) for x in bbox_str.split(',')]
                lon_min, lat_min, lon_max, lat_max = parts
                from shapely.geometry import box as shapely_box
                geom_wgs = shapely_box(lon_min, lat_min, lon_max, lat_max)
                dataset = meta_params.get('dataset', ti.DEFAULT_DATASET)
                type_filter = set(t.strip() for t in url_params['types'].split(',')) if url_params.get('types') else None
                poly_features = _vectorise_segments_to_geojson(
                    geom_wgs, dataset, type_filter=type_filter,
                    height_params=url_params)
                if poly_features:
                    kml_features = poly_features
                else:
                    log.warning('onestop kml polygon: vectorisation returned 0 features, falling back to points')
            except Exception as e:
                log.warning('onestop kml polygon fallback to points: %s', e)

        style_mode = url_params.get('segment_geometry_style', 'type').lower().strip()
        kml = _build_kml(kml_features, group_by, style_mode=style_mode)
        tmp = tempfile.NamedTemporaryFile(suffix='.kml', delete=False, mode='w', encoding='utf-8')
        tmp.write(kml)
        tmp.close()
        resp = send_file(tmp.name, mimetype='application/vnd.google-earth.kml+xml',
                        as_attachment=True, download_name='onestop_export.kml')
        resp.headers['X-Auto-Share-Id'] = auto_share_id or ''
        resp.headers['X-Elapsed-Seconds'] = str(elapsed)
        return resp

    if fmt == 'gpkg':
        # Re-run GPKG export using the segment result's geometry
        try:
            onestop_meta_path = _PROGRESS_DIR / f"{task_id}.onestop.json"
            meta_params = {}
            if onestop_meta_path.exists():
                meta_params = json.loads(onestop_meta_path.read_text()).get('params', {})

            bbox_str = meta_params.get('bbox', url_params.get('bbox', ''))
            parts = [float(x.strip()) for x in bbox_str.split(',')]
            lon_min, lat_min, lon_max, lat_max = parts
            from shapely.geometry import box as shapely_box
            geom = shapely_box(lon_min, lat_min, lon_max, lat_max)
            features = [{'geometry': geom}]

            # Merge: meta_params as defaults, url_params as overrides
            def _mp(key, default=''):
                return url_params.get(key) or meta_params.get(key, default)

            # Map segment_geometry / segment_geometry_style to GPKG params
            seg_geom = _mp('segment_geometry', 'polygon').lower().strip()
            seg_style = _mp('segment_geometry_style', '') or _mp('color_mode', 'type')

            gpkg_params = {
                'include_segments': 'true',
                'include_segments_vector': _mp('include_segments_vector',
                    'false' if seg_geom == 'point' else 'true'),
                'include_dtm': _mp('include_dtm', 'false'),
                'types': _mp('types'),
                'height_min': _mp('height_min'),
                'height_max': _mp('height_max'),
                'height_op': _mp('height_op'),
                'color_mode': seg_style if seg_style in ('type', 'height') else 'type',
            }
            # Override layers param
            layers_str = _mp('layers', 'segments')
            if layers_str and layers_str != 'segments':
                layer_set = set(l.strip() for l in layers_str.split(',') if l.strip())
                _resolve_layer_set(layer_set, gpkg_params)
            else:
                gpkg_params['include_segments'] = 'true'
                if seg_geom != 'point':
                    gpkg_params['include_segments_vector'] = 'true'

            tmp_path, table_count, gpkg_elapsed = _gpkg_core(features, gpkg_params)
            if table_count == 0:
                return _error('No layers produced')

            resp = send_file(tmp_path, mimetype='application/geopackage+sqlite3',
                            as_attachment=True, download_name='onestop_export.gpkg')
            resp.headers['X-Auto-Share-Id'] = auto_share_id or ''
            resp.headers['X-Elapsed-Seconds'] = str(elapsed)
            return resp
        except Exception as e:
            log.error("onestop gpkg: %s", traceback.format_exc())
            return _error(f"GPKG export failed: {e}", 500)

    return _error(f"Unknown format: {fmt}. Use json, gpkg, or kml.", 400)


def _enrich_top_features_with_flags(result: dict, feature_type: str,
                                    exclude_flagged: bool = False,
                                    min_severity: str = None):
    """Attach flags + community overrides to top_features rows. Layered
    on top of frozen JSONs — no rewrite of the source data."""
    items = result.get('results') or []
    if not items: return
    # Recreate the canonical obj_ref to look up flags/overrides
    refs = []
    for i, it in enumerate(items):
        kg = it.get('kg_code')
        if not kg: refs.append(None); continue
        if feature_type == 'trees':
            ref = f'{kg}:top_tree:{i}'  # only valid if items list mirrors top_10_trees order
        elif feature_type == 'objects':
            ref = f'{kg}:top_obj:{i}'
        elif feature_type == 'new_buildings':
            ref = f'{kg}:new_building:{i}'
        elif feature_type == 'infrastructure':
            ref = None  # infra needs (otype, idx); fall through to coord-resolve below
        else:
            ref = None
        refs.append(ref)
    # For items that don't have an obvious ref (cross-KG sort), resolve by coord.
    coord_keys = []
    for i, it in enumerate(items):
        if refs[i]: continue
        coord = it.get('coordinate') or {}
        lon = coord.get('lon') or it.get('centroid_lon')
        lat = coord.get('lat') or it.get('centroid_lat')
        if lon is None or lat is None: continue
        try:
            r = feedback_db.resolve_point(lon, lat,
                hint={'predicted_type': it.get('type') or it.get('rf_type'),
                      'height_max_m': it.get('height_m') or it.get('height_max_m') or it.get('max_height_m'),
                      'area_sqm': it.get('area_sqm')},
                kg_code=it.get('kg_code'), radius_m=10)
            if r.get('obj_ref'): refs[i] = r['obj_ref']
        except Exception: pass
    # Bulk fetch flags + overrides
    valid_refs = [r for r in refs if r]
    overrides = feedback_db.effective_overrides(valid_refs) if valid_refs else {}
    flags_by_ref = {}
    if valid_refs:
        feedback_db.ensure_schema()
        import sqlite3 as _s
        c = _s.connect(feedback_db.DB_PATH); c.row_factory = _s.Row
        qm = ','.join(['?'] * len(valid_refs))
        for row in c.execute(f'SELECT obj_ref, flag_code, severity, message FROM flags WHERE obj_ref IN ({qm})',
                              valid_refs):
            flags_by_ref.setdefault(row['obj_ref'], []).append({
                'flag_code': row['flag_code'], 'severity': row['severity'],
                'message': row['message']})
        c.close()
    # Attach + filter
    sev_rank = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
    min_rank = sev_rank.get((min_severity or '').lower(), 0)
    out = []
    for it, ref in zip(items, refs):
        if ref:
            it['obj_ref'] = ref
            fl = flags_by_ref.get(ref) or []
            if fl: it['flags'] = fl
            ov = overrides.get(ref)
            if ov and (ov.get('effective_type') or ov.get('community_verified')):
                if ov.get('effective_type'):
                    it['predicted_type'] = it.get('type')
                    it['type'] = ov['effective_type']
                    it['effective_type'] = ov['effective_type']
                it['community_verified'] = bool(ov.get('community_verified'))
                it['n_confirms'] = ov.get('n_confirms')
                it['n_rejects'] = ov.get('n_rejects')
                it['n_corrections'] = ov.get('n_corrections')
        if exclude_flagged:
            cur_max = max((sev_rank.get(f['severity'], 0) for f in it.get('flags', [])), default=-1)
            threshold = sev_rank.get((min_severity or 'medium').lower(), 1)
            if cur_max >= threshold:
                continue
        out.append(it)
    if exclude_flagged:
        result['results'] = out
        result['filtered_count'] = len(out)


# === SECTION: Quality flags + feedback ============================

def _bbox_arg(s):
    if not s: return None
    try:
        parts = [float(x) for x in s.split(',')]
        return parts if len(parts) == 4 else None
    except Exception: return None


@app.route('/api/v1/flags', methods=['GET'])
def api_flags_list():
    """List quality flags. Filterable + paginated.

    Params: kg, severity (low|medium|high|critical), code, type, kind,
            bbox=w,s,e,n, min_value, order=severity|value|recent,
            limit (default 200, max 1000), offset.
    """
    a = request.args
    try:
        rows = feedback_db.list_flags(
            kg_code=a.get('kg'), severity=a.get('severity'),
            flag_code=a.get('code'), obj_type=a.get('type'),
            kind=a.get('kind'), obj_ref=a.get('obj_ref'),
            bbox=_bbox_arg(a.get('bbox')),
            min_value=float(a['min_value']) if a.get('min_value') else None,
            limit=min(int(a.get('limit', 200)), 1000),
            offset=int(a.get('offset', 0)),
            order=a.get('order', 'severity'))
        # Attach per-object aggregate (total_weight, n_flags, codes)
        refs = list({r['obj_ref'] for r in rows if r.get('obj_ref')})
        aggs = feedback_db.object_aggregates(refs) if refs else {}
        for r in rows:
            r['aggregate'] = aggs.get(r.get('obj_ref'))
        return jsonify({'count': len(rows), 'flags': rows})
    except Exception as e:
        log.exception('flags list')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/flags/stats', methods=['GET'])
def api_flags_stats():
    return jsonify(feedback_db.flag_stats())


@app.route('/api/v1/flags/match', methods=['GET', 'POST'])
def api_flags_match():
    """Match a free-text snippet (e.g. '102.2m tree') and/or coordinates
    to known objects so the user can flag/correct without an ID.

    Params (GET) or JSON (POST):
        text=<snippet>     e.g. '102.2m tree'
        kg=<kg_code>       optional KG scope
        lon=, lat=         optional coordinate (improves resolution)
        radius_m=<m>       default 200
    """
    if request.method == 'POST':
        body = request.get_json(silent=True) or {}
    else:
        body = {}
    a = request.args
    text = body.get('text') or a.get('text') or ''
    kg   = body.get('kg')   or a.get('kg')
    lon  = body.get('lon')  or a.get('lon')
    lat  = body.get('lat')  or a.get('lat')
    radius = body.get('radius_m') or a.get('radius_m') or 200
    try:
        lon = float(lon) if lon not in (None, '') else None
        lat = float(lat) if lat not in (None, '') else None
        radius = float(radius)
    except Exception:
        lon = lat = None
    try:
        result = feedback_db.match_text(text, kg_code=kg, lon=lon, lat=lat, radius_m=radius)
        # If a candidate exists, attach its known flags + nearby flags + agg
        cand_refs = [c.get('obj_ref') for c in (result.get('candidates') or []) if c.get('obj_ref')]
        # include any aliases collapsed during dedup
        for c in (result.get('candidates') or []):
            for al in (c.get('aliases') or []):
                if al: cand_refs.append(al)
        if result.get('obj_ref') and result['obj_ref'] not in cand_refs:
            cand_refs.append(result['obj_ref'])
        agg = feedback_db.object_aggregates(cand_refs) if cand_refs else {}
        # Aggregate per *unique* (alias-collapsed) object: dedupe flags by
        # flag_code, take the max weight, so summing across aliases
        # doesn't triple-count when top_tree/top_obj/top_by_type point to
        # the same physical segment.
        import sqlite3 as _s2
        feedback_db.ensure_schema()
        _conn2 = _s2.connect(feedback_db.DB_PATH); _conn2.row_factory = _s2.Row
        for c in (result.get('candidates') or []):
            ref_pool = [c.get('obj_ref')] + (c.get('aliases') or [])
            ref_pool = [r for r in ref_pool if r]
            if not ref_pool:
                c['agg'] = None; continue
            qm = ','.join(['?'] * len(ref_pool))
            rows = _conn2.execute(
                f'SELECT flag_code, MAX(weight) AS w, severity FROM flags '
                f'WHERE obj_ref IN ({qm}) GROUP BY flag_code', ref_pool).fetchall()
            wsum = sum(r['w'] for r in rows)
            codes = sorted({r['flag_code'] for r in rows})
            max_sev = None; rank = -1
            for r in rows:
                ms = r['severity']
                if ms and feedback_db.SEV_ORDER.get(ms, -1) > rank:
                    rank = feedback_db.SEV_ORDER[ms]; max_sev = ms
            c['agg'] = {'total_weight': round(wsum, 2),
                        'codes': codes,
                        'max_severity': max_sev}
        _conn2.close()
        if result.get('obj_ref'):
            obj_flags = feedback_db.list_flags(limit=200, kg_code=result.get('kg_code'))
            primary_pool = set([result['obj_ref']])
            for c in (result.get('candidates') or []):
                if c.get('obj_ref') == result['obj_ref']:
                    primary_pool.update(c.get('aliases') or [])
                    break
            result['flags'] = [f for f in obj_flags if f['obj_ref'] in primary_pool]
            # Prediction previews for each action button
            result['action_predictions'] = {
                k: feedback_db.predict_action_impact(result['obj_ref'], kind=k)
                for k in ('confirm', 'reject', 'correct_type')
            }
        return jsonify(result)
    except Exception as e:
        log.exception('flags match')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/flags/predict', methods=['GET'])
def api_flags_predict():
    """Forecast the effect of submitting `kind` on `obj_ref`. Used by the
    flag widget to show users what their action will entail before they
    commit. Read-only; nothing is persisted.
    """
    a = request.args
    ref = a.get('obj_ref')
    if not ref: return jsonify({'error': 'obj_ref required'}), 400
    return jsonify(feedback_db.predict_action_impact(
        ref, kind=a.get('kind', 'reject'),
        corrected_type=a.get('corrected_type'),
        user_role=a.get('role', 'student')))


@app.route('/api/v1/flags/events', methods=['GET'])
def api_flag_events():
    """Audit log of every rule-based flag created/changed/removed.

    Useful for resampling or for explaining model drift. Filter by kg,
    obj_ref, kind, since=epoch.
    """
    a = request.args
    rows = feedback_db.list_flag_events(
        kg_code=a.get('kg'), obj_ref=a.get('obj_ref'),
        since=a.get('since'), kind=a.get('kind'),
        limit=min(int(a.get('limit', 200)), 1000),
        offset=int(a.get('offset', 0)))
    return jsonify({'count': len(rows), 'events': rows})


@app.route('/api/v1/feedback/events', methods=['GET'])
def api_feedback_events():
    """Audit log of every user feedback submission/supersession."""
    a = request.args
    rows = feedback_db.list_feedback_events(
        kg_code=a.get('kg'), obj_ref=a.get('obj_ref'),
        since=a.get('since'), user_id=a.get('user'),
        limit=min(int(a.get('limit', 200)), 1000),
        offset=int(a.get('offset', 0)))
    return jsonify({'count': len(rows), 'events': rows})


@app.route('/api/v1/flags/object/<path:obj_ref>', methods=['GET'])
def api_flag_object(obj_ref):
    """Return the object record + all its flags + any feedback."""
    feedback_db.ensure_schema()
    import sqlite3
    c = sqlite3.connect(feedback_db.DB_PATH); c.row_factory = sqlite3.Row
    obj = c.execute('SELECT * FROM objects WHERE obj_ref=?', (obj_ref,)).fetchone()
    if not obj:
        c.close()
        return jsonify({'error': 'unknown obj_ref'}), 404
    obj = dict(obj)
    if obj.get('attrs_json'):
        try: obj['attrs'] = json.loads(obj.pop('attrs_json'))
        except Exception: obj.pop('attrs_json', None)
    flags = feedback_db.list_flags(kg_code=obj['kg_code'], limit=200)
    flags = [f for f in flags if f['obj_ref'] == obj_ref]
    fb = feedback_db.list_feedback(obj_ref=obj_ref, limit=50)
    overrides = feedback_db.effective_overrides([obj_ref])
    agg = feedback_db.object_aggregates([obj_ref]).get(obj_ref)
    flag_events = feedback_db.list_flag_events(obj_ref=obj_ref, limit=50)
    fb_events = feedback_db.list_feedback_events(obj_ref=obj_ref, limit=50)
    c.close()
    return jsonify({'object': obj, 'flags': flags, 'feedback': fb,
                    'override': overrides.get(obj_ref),
                    'aggregate': agg,
                    'flag_events': flag_events,
                    'feedback_events': fb_events,
                    'predictions': {
                        k: feedback_db.predict_action_impact(obj_ref, kind=k)
                        for k in ('confirm', 'reject', 'correct_type')
                    }})


@app.route('/api/v1/flags/rebuild', methods=['POST'])
def api_flags_rebuild():
    """Re-run quality_flags. ?kg=CODE for one KG, otherwise all local JSONs."""
    kg = request.args.get('kg')
    json_dir = Path('data/austria_processor/json')
    if kg:
        jp = json_dir / f'{kg}.json'
        if not jp.exists(): return jsonify({'error': 'no JSON for that KG'}), 404
        r = quality_flags.scan_json(jp)
        return jsonify({'kg_code': kg, 'objects': len(r['objects']), 'flags': len(r['flags'])})
    n_obj = n_flag = 0
    for jp in json_dir.glob('*.json'):
        try:
            r = quality_flags.scan_json(jp)
            n_obj += len(r['objects']); n_flag += len(r['flags'])
        except Exception as e:
            log.warning('rebuild %s: %s', jp.name, e)
    return jsonify({'objects': n_obj, 'flags': n_flag})


def _feedback_user(req):
    """Resolve user from header/body. Anonymous fallback."""
    tok = req.headers.get('X-Feedback-Token') or (req.get_json(silent=True) or {}).get('token')
    user_id = (req.get_json(silent=True) or {}).get('user') or req.args.get('user') or 'anon'
    role = 'student'
    # Trivial role mapping; tighten later via data/feedback_users.json if needed
    return user_id, role


@app.route('/api/v1/feedback', methods=['POST'])
def api_feedback_submit():
    """Record one feedback item.

    Accepts ANY of:
      {"obj_ref":"...", "kind":"confirm|reject|correct_type|report_missing", ...}
      {"point":{"lon":...,"lat":...}, "context_text":"102.2m tree", ...}
      {"selected_text":"...", "kg_code":"...", ...}    # text-only (for embedded JS widget)

    Returns the resolved obj_ref + status.
    """
    payload = request.get_json(silent=True) or {}
    user, role = _feedback_user(request)
    src = payload.get('source_app') or 'web'
    try:
        result = feedback_db.record_feedback(payload, user_id=user, user_role=role,
                                              source_app=src)
        return jsonify({'ok': True, **result})
    except Exception as e:
        log.exception('feedback submit')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/feedback', methods=['GET'])
def api_feedback_list():
    a = request.args
    rows = feedback_db.list_feedback(
        kg_code=a.get('kg'), user=a.get('user'),
        since=a.get('since'), obj_ref=a.get('obj_ref'),
        limit=min(int(a.get('limit', 200)), 1000),
        offset=int(a.get('offset', 0)))
    return jsonify({'count': len(rows), 'feedback': rows})


@app.route('/api/v1/feedback/resolve', methods=['GET', 'POST'])
def api_feedback_resolve():
    """Preview-only. Same matching logic as /flags/match — doesn't save."""
    return api_flags_match()


# === END SECTION ===================================================

@app.route('/docs')
def docs_page():
    return send_from_directory('static', 'docs.html')

@app.route('/api/v1/docs/llm.txt', methods=['GET'])
def llm_docs():
    p = Path(__file__).parent / 'llm.txt'
    if p.exists():
        return Response(p.read_text(), mimetype='text/plain')
    return Response("Documentation not yet generated.", mimetype='text/plain')


# === SECTION: /api/v1/parse-geometry endpoint ===

@app.route('/api/v1/parse-geometry', methods=['POST'])
def parse_geometry_file():
    """Parse an uploaded geometry file (Shapefile ZIP, GeoPackage, GeoJSON, KML, GPX, WKT, etc).
    Returns GeoJSON FeatureCollection with all features.
    """
    import fiona
    import fiona.io
    import zipfile
    from shapely.geometry import shape as shp_shape, mapping as shp_mapping
    from shapely.ops import unary_union
    try:
        if 'file' not in request.files:
            return _error('No file uploaded', 400)
        f = request.files['file']
        fname = (f.filename or '').lower()
        raw = f.read()
        if not raw:
            return _error('Empty file', 400)

        # Auto-decompress gzipped uploads (client may gzip text files)
        if fname.endswith('.gz') or (len(raw) >= 2 and raw[:2] == b'\x1f\x8b'):
            import gzip as _gzip
            try:
                raw = _gzip.decompress(raw)
                if fname.endswith('.gz'):
                    fname = fname[:-3]
            except Exception:
                pass  # not actually gzipped, use raw

        features = []

        # Content-sniff: detect KML/GeoJSON/WKT regardless of file extension
        text_raw = None
        try:
            text_raw = raw.decode('utf-8', errors='replace')
        except Exception:
            pass

        is_kml = fname.endswith(('.kml', '.xml'))
        is_json = fname.endswith(('.geojson', '.json'))
        is_wkt = fname.endswith('.wkt')

        # Sniff content for unknown extensions (e.g. .txt)
        if text_raw and not (is_kml or is_json or is_wkt):
            stripped = text_raw.strip()[:500]
            if '<?xml' in stripped or '<kml' in stripped.lower():
                is_kml = True
            elif stripped.startswith('{') and '"type"' in stripped:
                is_json = True
            elif re.match(r'^(POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION)\s*\(', stripped, re.IGNORECASE):
                is_wkt = True

        # Try text-based formats first
        if is_json:
            gj = json.loads(raw)
            return jsonify(gj if gj.get('type') == 'FeatureCollection' else {
                'type': 'FeatureCollection',
                'features': gj.get('features', [{'type': 'Feature', 'geometry': gj, 'properties': {}}])
            })

        if is_kml:
            parsed = geo_parse.parse_input(text_raw or raw.decode('utf-8', errors='replace'))
            # Convert non-polygon geometries (lines from road boundaries etc)
            for feat in parsed:
                geom = feat['geometry']
                if geom.geom_type not in ('Polygon', 'MultiPolygon'):
                    feat['geometry'] = _non_polygon_to_polygon(geom)
            return jsonify(geo_parse.features_to_geojson(parsed))

        if is_wkt:
            from shapely import wkt
            geom = wkt.loads(raw.decode('utf-8', errors='replace').strip())
            return jsonify({'type': 'FeatureCollection', 'features': [
                {'type': 'Feature', 'geometry': shp_mapping(geom), 'properties': {}}
            ]})

        # Binary formats via fiona
        tmp_dir = tempfile.mkdtemp(prefix='geo_upload_')
        try:
            # Shapefile in ZIP
            if fname.endswith('.zip'):
                zpath = os.path.join(tmp_dir, 'upload.zip')
                with open(zpath, 'wb') as wf:
                    wf.write(raw)
                # Find .shp inside zip
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(tmp_dir)
                # Find shp file
                shp_files = []
                for root, dirs, files in os.walk(tmp_dir):
                    for fn in files:
                        if fn.lower().endswith('.shp'):
                            shp_files.append(os.path.join(root, fn))
                if not shp_files:
                    # Try as GPKG or other format inside zip
                    for root, dirs, files in os.walk(tmp_dir):
                        for fn in files:
                            if fn.lower().endswith(('.gpkg', '.geojson', '.json', '.kml', '.gpx')):
                                shp_files.append(os.path.join(root, fn))
                if not shp_files:
                    return _error('No shapefile (.shp) or supported file found in ZIP', 400)
                src_path = shp_files[0]
            else:
                # Single file (gpkg, gpx, shp, etc)
                ext = os.path.splitext(fname)[1] or '.gpkg'
                src_path = os.path.join(tmp_dir, 'upload' + ext)
                with open(src_path, 'wb') as wf:
                    wf.write(raw)

            with fiona.open(src_path) as src:
                src_crs = src.crs
                # Reproject to WGS84 if needed
                need_reproject = False
                if src_crs and str(src_crs).upper() not in ('EPSG:4326', '{"INIT": "EPSG:4326"}'):
                    try:
                        from pyproj import CRS, Transformer
                        c = CRS(src_crs)
                        if c.to_epsg() != 4326:
                            need_reproject = True
                            transformer = Transformer.from_crs(c, CRS.from_epsg(4326), always_xy=True)
                    except Exception:
                        pass

                for feat in src:
                    geom = shp_shape(feat['geometry'])
                    if need_reproject:
                        from shapely.ops import transform
                        geom = transform(transformer.transform, geom)
                    props = dict(feat.get('properties', {}))
                    # Convert non-serializable values
                    for k, v in list(props.items()):
                        if v is not None and not isinstance(v, (str, int, float, bool)):
                            props[k] = str(v)
                    features.append({
                        'type': 'Feature',
                        'geometry': shp_mapping(geom),
                        'properties': props,
                    })
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if not features:
            return _error('No features found in file', 400)

        log.info("parse-geometry: %s → %d features", fname, len(features))
        return jsonify({'type': 'FeatureCollection', 'features': features})

    except Exception as e:
        log.error("parse-geometry: %s", traceback.format_exc())
        return _error(f'Failed to parse file: {e}')


# === SECTION: /api/v1/share endpoints (save/load/rename/list) ===

SHARE_DIR = Path('data/shares')
SHARE_DIR.mkdir(parents=True, exist_ok=True)
SHARE_MAX_BYTES = 2_000_000_000  # 2 GB

_SHARE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,80}$')  # hex hashes or named slugs
def _valid_share_id(s):
    return bool(s and _SHARE_ID_RE.match(s))

def _resolve_share(share_id):
    """Resolve a share ID, following redirect stubs from renames.
    Returns (resolved_id, path) or (None, None) if not found."""
    p = SHARE_DIR / f'{share_id}.json.gz'
    if not p.exists():
        return None, None
    # Check for redirect alias (small file left after rename)
    if p.stat().st_size < 200:
        try:
            stub = json.loads(gzip.decompress(p.read_bytes()))
            if 'redirect' in stub:
                new_id = stub['redirect']
                new_p = SHARE_DIR / f'{new_id}.json.gz'
                if new_p.exists():
                    return new_id, new_p
                return None, None
        except Exception:
            pass
    return share_id, p


def _share_eviction_tier(path: Path) -> int:
    """Classify a share into eviction tiers (lower = evicted first).

    Tier 0: unnamed auto-saves and hex-hash shares (evict first)
    Tier 1: redirect stubs (tiny, but keep them to preserve old links)
    Tier 2: shares with a user-set name inside the payload (e.g. name='Westbahnhof')
    Tier 3: shares with a user-renamed ID (e.g. 'Wienwest', 'Kohlschwarz90') — evict last
    """
    try:
        share_id = path.stem.replace('.json', '')
        sz = path.stat().st_size
        id_is_hex = bool(re.fullmatch(r'[0-9a-f]{12}', share_id))
        id_is_auto = share_id.startswith('auto-')

        # Redirect stubs: small files that alias old IDs → new named share.
        # Protect them (tier 1) so old links keep working.
        if sz < 200:
            try:
                stub = json.loads(gzip.decompress(path.read_bytes()))
                if 'redirect' in stub:
                    return 1
            except Exception:
                pass
            # Other tiny files (empty state etc.) are low-value
            return 0

        # User-renamed ID: the share filename is a human-readable slug.
        # These are the most valuable — user deliberately chose the name.
        if not id_is_hex and not id_is_auto:
            return 3

        # Check the name field inside the payload (only first ~2KB for speed)
        # Shares that have a name set but still have hex/auto IDs (user set name
        # but didn't rename the ID).
        try:
            raw = gzip.decompress(path.read_bytes())
            data = json.loads(raw)
            name = data.get('name', '')
            if name and name != share_id and not name.startswith('Auto-save '):
                return 2
        except Exception:
            pass

        return 0
    except Exception:
        return 0


MAX_AUTO_SAVES = 15  # keep at most this many tier-0 auto-saves

def _share_evict():
    """Remove shares until total size < SHARE_MAX_BYTES.

    Also caps the number of tier-0 (auto-save / hex-hash) shares to
    MAX_AUTO_SAVES so they don't crowd out named shares in the UI.

    Eviction tiers (lower evicted first):
      0: unnamed auto-saves and hex-hash shares
      1: redirect stubs (preserve old links)
      2: shares with user-set name field
      3: shares with user-renamed ID slug (most protected)
    Within each tier, oldest (by mtime) evicted first.
    """
    all_files = list(SHARE_DIR.glob('*.json.gz'))
    total = sum(f.stat().st_size for f in all_files)

    # --- Phase 1: cap tier-0 auto-saves to MAX_AUTO_SAVES ---
    tier0 = sorted(
        [f for f in all_files if _share_eviction_tier(f) == 0],
        key=lambda f: f.stat().st_mtime, reverse=True  # newest first
    )
    for victim in tier0[MAX_AUTO_SAVES:]:
        total -= victim.stat().st_size
        victim.unlink(missing_ok=True)
        all_files.remove(victim)
        log.info("share: evicted excess auto-save %s (total now %d MB)", victim.name, total // 1_000_000)

    # --- Phase 2: size-based eviction ---
    if total <= SHARE_MAX_BYTES:
        return
    # Sort by (tier, mtime) — lowest tier and oldest files evicted first
    evict_order = sorted(all_files, key=lambda f: (_share_eviction_tier(f), f.stat().st_mtime))
    while total > SHARE_MAX_BYTES and evict_order:
        victim = evict_order.pop(0)
        tier = _share_eviction_tier(victim)
        total -= victim.stat().st_size
        victim.unlink(missing_ok=True)
        log.info("share: evicted %s (tier %d, total now %d MB)", victim.name, tier, total // 1_000_000)


@app.route('/api/v1/shares', methods=['GET'])
def share_list():
    """List saved shares with metadata.

    Ordering: user-renamed shares first (tier 3), then named (tier 2),
    then auto-saves/hex (tier 0).  Within each group, most recent first.
    This ensures manually-named shares are always visible regardless of
    how many auto-saves accumulate.
    """
    try:
        all_files = list(SHARE_DIR.glob('*.json.gz'))
        # Sort by (-tier, -mtime) so high-tier (named) shares come first
        files = sorted(all_files, key=lambda f: (-_share_eviction_tier(f), -f.stat().st_mtime))
        limit = int(request.args.get('limit', 30))
        items = []
        for f in files[:limit]:
            share_id = f.stem.replace('.json', '')
            try:
                data = json.loads(gzip.decompress(f.read_bytes()))
                # Skip redirect stubs (left after rename for old-link compat)
                if 'redirect' in data and 'state' not in data:
                    continue
                state = data.get('state', {})
                stored_name = data.get('name', '')
                # Prefer share ID as display name if it's a human-readable slug
                # (i.e. user-renamed, not a hex hash or auto- prefix)
                id_is_hex = bool(re.fullmatch(r'[0-9a-f]{12}', share_id))
                id_is_auto = share_id.startswith('auto-')
                if not id_is_hex and not id_is_auto:
                    name = share_id  # user-renamed ID IS the name
                else:
                    name = stored_name or share_id
                endpoint = state.get('endpoint', '')
                has_result = 'result' in data and bool(data['result'])
                has_geometry = bool(state.get('geometry', ''))
                # Build a short description
                tags = []
                if endpoint:
                    tags.append(endpoint)
                if has_result:
                    # Try to get object count from result
                    result = data.get('result', {})
                    n_features = len(result.get('features', []))
                    if n_features:
                        tags.append(f"{n_features} objects")
                elif has_geometry:
                    tags.append('geometry only')
                is_onestop = 'onestop' in data
                items.append(dict(
                    id=share_id,
                    name=name,
                    description=', '.join(tags) if tags else '',
                    has_result=has_result,
                    has_geometry=has_geometry,
                    endpoint=endpoint,
                    updated=f.stat().st_mtime,
                    size_kb=round(f.stat().st_size / 1024, 1),
                    onestop=is_onestop,
                ))
            except Exception:
                items.append(dict(id=share_id, name=share_id, description='', has_result=False,
                                  has_geometry=False, endpoint='', updated=f.stat().st_mtime, size_kb=0))
        return jsonify(items)
    except Exception as e:
        return _error(str(e))


@app.route('/api/v1/share', methods=['POST'])
def share_save():
    """Save analysis result + UI state for sharing. Returns {id, url}.
    
    Content-hash dedup: if payload matches an existing share, reuse its ID.
    Client can also send {reuse_id: "abc123"} to update an existing share in-place.
    Overlays (base64 PNG images) are included in the stored payload for instant restore.
    """
    try:
        import hashlib
        payload = request.get_json(force=True)
        if not payload:
            return _error('Empty payload')
        
        host = request.headers.get('X-Forwarded-Host', request.host)
        proto = request.headers.get('X-Forwarded-Proto', 'https')
        
        # Extract reuse_id (not stored)
        reuse_id = payload.pop('reuse_id', None)
        
        # Hash based on state+result only (exclude overlays and name — they're metadata)
        hash_payload = {k: v for k, v in payload.items() if k not in ('overlays', 'name')}
        hash_json = json.dumps(hash_payload, separators=(',', ':'), sort_keys=True)
        content_hash = hashlib.sha256(hash_json.encode()).hexdigest()[:24]
        
        # Full payload including overlays for storage
        data_json = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        data = gzip.compress(data_json.encode())
        
        # Check if client wants to reuse an existing share ID (update in-place)
        if reuse_id and _valid_share_id(reuse_id):
            existing = SHARE_DIR / f'{reuse_id}.json.gz'
            if existing.exists():
                # Merge: keep existing result/overlays if not provided in new payload
                try:
                    existing_obj = json.loads(gzip.decompress(existing.read_bytes()).decode())
                except Exception:
                    existing_obj = {}
                if 'result' not in payload and 'result' in existing_obj:
                    payload['result'] = existing_obj['result']
                if 'overlays' not in payload and 'overlays' in existing_obj:
                    payload['overlays'] = existing_obj['overlays']
                if 'name' not in payload and 'name' in existing_obj:
                    payload['name'] = existing_obj['name']
                merged_json = json.dumps(payload, separators=(',', ':'), sort_keys=True)
                existing.write_bytes(gzip.compress(merged_json.encode()))
                existing.touch()
                url = f'{proto}://{host}/?share={reuse_id}'
                ovl_count = len(payload.get('overlays', {}))
                log.info("share: updated existing %s (%d KB, %d overlays)",
                         reuse_id, len(gzip.compress(merged_json.encode())) // 1024, ovl_count)
                return jsonify({'id': reuse_id, 'url': url, 'reused': True})
        
        # Content-hash dedup: check existing shares (state+result match)
        # If overlays differ but state+result same, update overlays in existing share
        for existing_file in SHARE_DIR.glob('*.json.gz'):
            try:
                existing_data = gzip.decompress(existing_file.read_bytes()).decode()
                existing_obj = json.loads(existing_data)
                existing_hash_payload = {k: v for k, v in existing_obj.items() if k not in ('overlays', 'name')}
                existing_hash = hashlib.sha256(
                    json.dumps(existing_hash_payload, separators=(',', ':'), sort_keys=True).encode()
                ).hexdigest()[:24]
                if existing_hash == content_hash:
                    share_id = existing_file.stem.split('.')[0]
                    # If new payload has overlays/name changes, update stored data
                    new_ovl = payload.get('overlays', {})
                    old_ovl = existing_obj.get('overlays', {})
                    new_name = payload.get('name')
                    old_name = existing_obj.get('name')
                    if len(new_ovl) > len(old_ovl) or new_name != old_name:
                        existing_file.write_bytes(data)
                        log.info("share: dedup hit %s, updated (overlays %d→%d, name=%s)",
                                 share_id, len(old_ovl), len(new_ovl), new_name)
                    else:
                        log.info("share: dedup hit %s", share_id)
                    existing_file.touch()
                    url = f'{proto}://{host}/?share={share_id}'
                    return jsonify({'id': share_id, 'url': url, 'reused': True, 'name': new_name})
            except Exception:
                continue
        
        # New share
        share_id = uuid.uuid4().hex[:12]
        (SHARE_DIR / f'{share_id}.json.gz').write_bytes(data)
        _share_evict()
        ovl_count = len(payload.get('overlays', {}))
        url = f'{proto}://{host}/?share={share_id}'
        log.info("share: saved %s (%d KB, %d overlays)", share_id, len(data) // 1024, ovl_count)
        return jsonify({'id': share_id, 'url': url, 'reused': False})
    except Exception as e:
        log.error("share save: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/share/<share_id>', methods=['GET'])
def share_load(share_id):
    """Retrieve a saved share. Serves gzip-compressed if client accepts it.
    
    Follows redirect aliases: if a share was renamed, the old ID contains
    a small {"redirect": "new_id"} stub that points to the new location.
    """
    try:
        if not _valid_share_id(share_id):
            return _error('Invalid share ID', 400)
        share_id, p = _resolve_share(share_id)
        if not p or not p.exists():
            return _error('Share not found', 404)
        raw = p.read_bytes()
        # Touch file to keep it alive (LRU)
        p.touch()
        # Header to let the client know the resolved ID (may differ if redirect followed)
        extra_headers = {'X-Share-Id': share_id}
        # Serve pre-compressed gzip if client accepts it (saves bandwidth, critical for mobile)
        accept_enc = request.headers.get('Accept-Encoding', '')
        if 'gzip' in accept_enc:
            return Response(raw, mimetype='application/json',
                            headers={'Content-Encoding': 'gzip',
                                     'Vary': 'Accept-Encoding',
                                     **extra_headers})
        data = gzip.decompress(raw)
        return Response(data, mimetype='application/json', headers=extra_headers)
    except Exception as e:
        log.error("share load: %s", traceback.format_exc())
        return _error(str(e))


def _resolve_layer_set(layer_set: set, params: dict, state: dict = None):
    """Translate UI layer IDs into _gpkg_core params.

    Layer IDs match the UI data-lyr attributes:
      dtm, segments, raster, hansen,
      ortho-YYYY, cir-YYYY, dtm-YYYY, dsm-YYYY
    """
    if 'dtm' in layer_set:
        params['include_dtm'] = 'true'
        layer_set.discard('dtm')
    if 'segments' in layer_set:
        params['include_segments'] = 'true'
        layer_set.discard('segments')
    layer_set.discard('base')  # basemap tile layer, not exportable

    # ortho-YYYY → ortho_years (raw RGBI raster)
    ortho_yrs = []
    for lid in list(layer_set):
        if lid.startswith('ortho-') and lid[6:].isdigit():
            ortho_yrs.append(lid[6:])
            layer_set.discard(lid)
    if ortho_yrs:
        params['ortho_years'] = ','.join(sorted(ortho_yrs))

    # Everything else (raster, hansen, dtm-YYYY, dsm-YYYY, cir-YYYY)
    # goes to raster_layers for RGBA overlay rendering
    if layer_set:
        params['raster_layers'] = ','.join(sorted(layer_set))


@app.route('/api/v1/share/<share_id>/download.gpkg', methods=['GET'])
def share_download_gpkg(share_id):
    """Direct GeoPackage download from a share — usable as QGIS data source URL.

    Reads the share's geometry, runs GPKG export.

    Query params:
      layers=all         Include DTM/DSM/nDSM + ortho + segments (default)
      layers=active      Only layers that were active in the share's UI state
      layers=dtm,ortho-2024,segments,raster,hansen  Comma-separated layer IDs:
                         dtm        → DTM + DSM + nDSM (raw 1m float32)
                         segments   → Segment type + height rasters
                         ortho-YYYY → Orthophoto RGBI for that year
                         cir-YYYY   → CIR false-colour for that year
                         raster     → Coloured segment overlay (RGBA)
                         hansen     → Hansen forest change overlay (RGBA)
                         dtm-YYYY   → DTM hillshade overlay (RGBA)
                         dsm-YYYY   → DSM hillshade overlay (RGBA)
      types=tree,road    Filter segment types
      color_mode=type    Segment colour mode (type or height)
    """
    try:
        if not _valid_share_id(share_id):
            return _error('Invalid share ID', 400)
        share_id, p = _resolve_share(share_id)
        if not p or not p.exists():
            return _error('Share not found', 404)
        share_data = json.loads(gzip.decompress(p.read_bytes()).decode())
        state = share_data.get('state', {})
        geom_str = state.get('geometry', '')
        if not geom_str:
            return _error('Share has no geometry', 400)

        # Parse geometry
        features = geo_parse.parse_input(geom_str)
        features = geo_parse.union_features(features)
        for feat in features:
            geom = feat['geometry']
            if geom.geom_type not in ('Polygon', 'MultiPolygon'):
                feat['geometry'] = _non_polygon_to_polygon(geom)

        # --- Resolve layers param ---
        layers_raw = request.args.get('layers', 'all').strip().lower()
        params = {}

        if layers_raw == 'all':
            params['include_dtm'] = 'true'
            params['include_segments'] = 'true'
            params['ortho_years'] = str(ti.dataset_to_year(ti.DEFAULT_DATASET))

        elif layers_raw == 'active':
            # Use only layers the user had enabled in the share's UI
            active = set(state.get('layers', []))
            _resolve_layer_set(active, params, state)

        else:
            # Explicit comma-separated list
            layer_set = set(l.strip() for l in layers_raw.split(',') if l.strip())
            _resolve_layer_set(layer_set, params, state)

        # Allow extra query-string overrides
        for key in ('types', 'color_mode', 'dataset', 'height_min', 'height_max', 'height_op'):
            val = request.args.get(key)
            if val is not None:
                params[key] = val

        tmp_path, table_count, elapsed = _gpkg_core(features, params)
        if table_count == 0:
            return _error('No layers produced')
        log.info("share GPKG download %s: %d tables, %.1fs, layers=%s",
                 share_id, table_count, elapsed, layers_raw)
        return send_file(
            tmp_path, mimetype='application/geopackage+sqlite3',
            as_attachment=True, download_name=f'{share_id}.gpkg',
        )
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error("share gpkg download: %s", traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


@app.route('/api/v1/share/<old_id>/rename', methods=['POST'])
def share_rename(old_id):
    """Rename a share: move file from old_id to new slug.
    Body: {"new_id": "MySlug"}
    Returns: {"id": "MySlug", "old_id": "abc123def456"}
    """
    try:
        if not _valid_share_id(old_id):
            return _error('Invalid share ID', 400)
        body = request.get_json(force=True)
        new_id = (body.get('new_id') or '').strip()
        if not new_id:
            return _error('new_id required', 400)
        # Sanitise: replace spaces/special chars with hyphens
        new_id = re.sub(r'[^A-Za-z0-9_-]+', '-', new_id).strip('-')[:80]
        if not _valid_share_id(new_id):
            return _error('Invalid new ID', 400)
        old_path = SHARE_DIR / f'{old_id}.json.gz'
        if not old_path.exists():
            return _error('Share not found', 404)
        new_path = SHARE_DIR / f'{new_id}.json.gz'
        if new_path.exists() and new_id != old_id:
            return _error('Name already taken', 409)
        # Update the name field inside the JSON to match the new ID
        try:
            payload = json.loads(gzip.decompress(old_path.read_bytes()))
            payload['name'] = new_id
            # Track rename history so old links keep working
            aliases = payload.get('aliases', [])
            if old_id not in aliases:
                aliases.append(old_id)
            payload['aliases'] = aliases
            new_data = gzip.compress(json.dumps(payload, separators=(',', ':'), sort_keys=True).encode())
            new_path.write_bytes(new_data)
            if old_path != new_path:
                old_path.unlink()
        except Exception:
            # Fallback: just move the file
            if not new_path.exists():
                old_path.rename(new_path)
        # Leave a small redirect file at old path so old links still resolve
        if old_id != new_id:
            try:
                alias_data = json.dumps({'redirect': new_id}, separators=(',', ':'))
                (SHARE_DIR / f'{old_id}.json.gz').write_bytes(
                    gzip.compress(alias_data.encode()))
            except Exception:
                pass  # best-effort
        new_path.touch()
        log.info("share: renamed %s → %s", old_id, new_id)
        return jsonify({'id': new_id, 'old_id': old_id})
    except Exception as e:
        log.error("share rename: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/process.html')
def process_html():
    """Serve the processor dashboard with the admin token pre-injected.

    The dashboard exercises every cluster admin endpoint (queue add/delete,
    tombstones, bbox add, reshuffle, peers add/update, throttle, ...).  We
    inject the current admin token as an inline script before the page's own
    fetch interceptor so all those calls succeed without prompting the
    operator. Anyone who can load /process.html therefore obtains the token;
    that's by design — the dashboard is the admin UI.
    """
    try:
        html = (Path('static') / 'process.html').read_text(encoding='utf-8')
    except Exception as e:
        return _error(f'process.html missing: {e}', 500)
    tok = _current_admin_token() or ''
    # JSON-encode so quotes/backslashes survive embedding.
    inject = (
        '<script>(function(){try{'
        f'var t={json.dumps(tok)};'
        'if(t){localStorage.setItem("srtm_admin_token",t);}'
        '}catch(e){}})();</script>\n'
    )
    # Insert right after <head> so it runs before the existing interceptor.
    lower = html.lower()
    idx = lower.find('<head>')
    if idx >= 0:
        cut = idx + len('<head>')
        html = html[:cut] + '\n' + inject + html[cut:]
    else:
        html = inject + html
    return Response(html, mimetype='text/html; charset=utf-8')


@app.route('/')
def index():
    # ?share=X&dl=gpkg → redirect to GPKG download
    share_id = request.args.get('share', '')
    dl = request.args.get('dl', '').lower()
    if share_id and dl in ('gpkg', 'geopackage'):
        qs = '&'.join(f'{k}={v}' for k, v in request.args.items()
                       if k not in ('share', 'dl'))
        url = f'/api/v1/share/{share_id}/download.gpkg'
        if qs:
            url += '?' + qs
        return redirect(url)
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
