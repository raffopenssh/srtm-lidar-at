#!/usr/bin/env python3
"""Standalone Zenodo upload-lock broker.

Why a separate process?
-----------------------
The primary's gunicorn pool serves dashboard, manifest API, peer-data
sync, search index, director loop, and HA snapshot push — every one of
which can briefly stall a worker for 1–2 s.  When it does, every
active peer's heartbeat to ``/api/v1/zenodo/lock/heartbeat`` 5xxs or
times out, and we get cascading ``Upload lease lost (410)`` storms
that abort in-flight uploads across the fleet.

Moving lock state to a tiny single-purpose HTTP server insulates lease
management from gunicorn slow paths entirely.  This file implements:

* ``POST   /api/v1/zenodo/lock``            — acquire or 423
* ``POST   /api/v1/zenodo/lock/heartbeat``  — renew, 410 if expired
* ``DELETE /api/v1/zenodo/lock``            — release
* ``GET    /api/v1/zenodo/lock``            — status (no auth)

Lease classes (since 2026-09-21)
--------------------------------
* **shared** — ``purpose`` starting with ``kg_upload`` (KG product
  uploads).  Every KG product lands in its *own* deposition, so these
  never 409 against each other; the only reason to bound them is to be
  a good citizen towards Zenodo.  Up to ``SHARED_SLOTS`` (env
  ``ZENODO_LOCK_KG_SLOTS``, default 8) are held concurrently.
  Previously they were fully serialised behind one mutex: with ~20
  peers each streaming a 100–900 MB GPKG to a degraded Zenodo
  (10–30 min per upload) every waiter hit the client's 30-min acquire
  timeout and "proceeded without lease" — i.e. the lock cost every KG
  30 idle minutes and provided no exclusivity at all.
* **exclusive** — everything else (``cache_flush_zip:*``,
  ``reconcile``, chkpt registry …): writers of the *shared* tile-cache
  draft deposition, where concurrent PUTs 409 and orphan files.  An
  exclusive lease requires zero live leases of any class.  While an
  exclusive request is being refused, new shared acquires are refused
  too (``draining``) so the exclusive writer cannot starve behind a
  rolling stream of uploads.

State is held in memory and persisted to disk on every mutation so a
restart doesn't 410 every active heartbeat.  TTL = 120 s per lease.

Auth: same admin token model as gunicorn (header ``X-Admin-Token``
matching ``data/admin_token``).  Loopback exempt.

Runs under systemd as ``zenodo_lock_broker.service`` on the primary,
listening on 127.0.0.1:8001.  Only port 8000 is reachable from outside,
so peers keep ``ZENODO_LOCK_URL`` pointed at the primary's :8000 and
gunicorn's ``/api/v1/zenodo/lock*`` routes proxy to this broker
(``app._broker_proxy``), falling back to in-process single-mutex state
if the broker is down.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / 'data' / 'austria_processor'
STATE_FILE = DATA_DIR / 'zenodo_lock_state.json'
ADMIN_TOKEN_FILE = _HERE / 'data' / 'admin_token'

TTL_S = 120.0
SHARED_SLOTS = max(1, int(os.environ.get('ZENODO_LOCK_KG_SLOTS', '8') or 8))
# How long a refused exclusive request keeps the shared class draining.
EXCLUSIVE_DRAIN_S = 300.0
LISTEN_HOST = os.environ.get('ZENODO_LOCK_BROKER_HOST', '127.0.0.1')
LISTEN_PORT = int(os.environ.get('ZENODO_LOCK_BROKER_PORT', '8001'))

_lock = threading.Lock()
# token -> {holder, purpose, kg, cls, acquired_at, last_heartbeat}
_leases: dict = {}
_exclusive_waiting: dict = {'since': 0.0, 'peer': None, 'purpose': None}


def _cls(purpose: str) -> str:
    return 'shared' if str(purpose or '').startswith('kg_upload') else 'exclusive'


def _persist() -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix('.json.tmp')
        tmp.write_text(json.dumps({'v': 2, 'leases': _leases}))
        tmp.replace(STATE_FILE)
    except Exception as e:
        print(f'[broker] persist error: {e}', file=sys.stderr)


def _restore() -> None:
    try:
        if not STATE_FILE.exists():
            return
        d = json.loads(STATE_FILE.read_text())
        now = time.time()
        if isinstance(d, dict) and d.get('v') == 2:
            for tok, l in (d.get('leases') or {}).items():
                if now - float(l.get('last_heartbeat') or 0.0) <= TTL_S:
                    _leases[tok] = l
        elif isinstance(d, dict) and d.get('holder') and d.get('token'):
            # Legacy single-holder shape (pre 2026-09-21).
            if now - float(d.get('last_heartbeat') or 0.0) <= TTL_S:
                _leases[d['token']] = {
                    'holder': d['holder'], 'purpose': d.get('purpose'),
                    'kg': d.get('kg'), 'cls': _cls(d.get('purpose')),
                    'acquired_at': float(d.get('acquired_at') or now),
                    'last_heartbeat': float(d.get('last_heartbeat') or now),
                }
        if _leases:
            print(f'[broker] restored {len(_leases)} lease(s)', file=sys.stderr)
    except Exception as e:
        print(f'[broker] restore error: {e}', file=sys.stderr)


def _expire(now: float) -> None:
    """Drop leases whose heartbeat is older than TTL (caller holds _lock)."""
    dead = [t for t, l in _leases.items() if now - l['last_heartbeat'] > TTL_S]
    for t in dead:
        l = _leases.pop(t)
        print(f'[broker] stale lease dropped holder={l["holder"]} '
              f'purpose={l["purpose"]} kg={l.get("kg")}', file=sys.stderr)
    if dead:
        _persist()


def _lease_view(l: dict, now: float) -> dict:
    return {
        'holder': l['holder'], 'purpose': l['purpose'], 'kg': l.get('kg'),
        'cls': l.get('cls'),
        'age_s': round(now - l['acquired_at'], 1),
        'idle_s': round(now - l['last_heartbeat'], 1),
    }


def _status(now: float) -> dict:
    """Status payload. Top-level fields mirror the legacy single-holder
    shape (oldest live lease) so existing consumers keep working."""
    shared = [l for l in _leases.values() if l.get('cls') == 'shared']
    excl = [l for l in _leases.values() if l.get('cls') != 'shared']
    draining = (now - _exclusive_waiting['since']) < EXCLUSIVE_DRAIN_S
    out = {
        'free': not _leases,
        'ttl_s': TTL_S,
        'shared_slots': SHARED_SLOTS,
        'shared_used': len(shared),
        'exclusive_held': bool(excl),
        'draining_for_exclusive': draining,
        'leases': sorted((_lease_view(l, now) for l in _leases.values()),
                         key=lambda x: -x['age_s']),
    }
    if _leases:
        oldest = min(_leases.values(), key=lambda l: l['acquired_at'])
        out.update(_lease_view(oldest, now))
        out['stale'] = False
    return out


def _admin_token() -> str:
    try:
        return ADMIN_TOKEN_FILE.read_text().strip()
    except Exception:
        return ''


class Handler(BaseHTTPRequestHandler):
    server_version = 'srtm-lidar-zenodo-lock/2.0'

    def log_message(self, fmt, *args):  # quiet by default
        if os.environ.get('ZENODO_LOCK_BROKER_VERBOSE'):
            super().log_message(fmt, *args)

    # --- helpers -----------------------------------------------------

    def _read_body(self) -> dict:
        try:
            n = int(self.headers.get('Content-Length', '0') or '0')
        except ValueError:
            n = 0
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode('utf-8') or '{}')
        except Exception:
            return {}

    def _is_loopback(self) -> bool:
        try:
            host = self.client_address[0]
        except Exception:
            return False
        return host in ('127.0.0.1', '::1', 'localhost')

    def _check_auth(self) -> bool:
        if self._is_loopback():
            return True
        tok = _admin_token()
        if not tok:
            return True
        sent = self.headers.get('X-Admin-Token', '')
        return sent == tok

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --- routes ------------------------------------------------------

    def do_GET(self):
        if self.path.split('?', 1)[0] != '/api/v1/zenodo/lock':
            return self._json(404, {'error': 'not_found'})
        now = time.time()
        with _lock:
            _expire(now)
            return self._json(200, _status(now))

    def do_POST(self):
        if not self._check_auth():
            return self._json(401, {'error': 'auth'})
        path = self.path.split('?', 1)[0]
        if path == '/api/v1/zenodo/lock':
            self._acquire()
        elif path == '/api/v1/zenodo/lock/heartbeat':
            self._heartbeat()
        else:
            self._json(404, {'error': 'not_found'})

    def do_DELETE(self):
        if not self._check_auth():
            return self._json(401, {'error': 'auth'})
        if self.path.split('?', 1)[0] != '/api/v1/zenodo/lock':
            return self._json(404, {'error': 'not_found'})
        self._release()

    # --- handlers ----------------------------------------------------

    def _acquire(self):
        body = self._read_body()
        peer = str(body.get('peer') or 'anon')
        purpose = str(body.get('purpose') or 'unknown')
        kg = body.get('kg')
        cls = _cls(purpose)
        now = time.time()
        with _lock:
            _expire(now)
            # Idempotent re-acquire: same peer + purpose + kg → same token.
            for tok, l in _leases.items():
                if l['holder'] == peer and l['purpose'] == purpose and l.get('kg') == kg:
                    l['last_heartbeat'] = now
                    _persist()
                    return self._json(200, {'token': tok, 'ttl_s': TTL_S,
                                            'reacquired': True})
            shared_live = sum(1 for l in _leases.values() if l.get('cls') == 'shared')
            excl_live = any(l.get('cls') != 'shared' for l in _leases.values())
            draining = (now - _exclusive_waiting['since']) < EXCLUSIVE_DRAIN_S
            if cls == 'exclusive':
                ok = not _leases
                if not ok:
                    _exclusive_waiting.update(since=now, peer=peer, purpose=purpose)
            else:
                ok = (not excl_live) and (not draining) and shared_live < SHARED_SLOTS
            if not ok:
                st = _status(now)
                st.update({'error': 'locked', 'requested_cls': cls})
                if cls == 'exclusive':
                    st['reason'] = 'leases_live'
                elif excl_live:
                    st['reason'] = 'exclusive_held'
                elif draining:
                    st['reason'] = 'draining_for_exclusive'
                else:
                    st['reason'] = 'shared_slots_full'
                return self._json(423, st)
            if cls == 'exclusive':
                _exclusive_waiting.update(since=0.0, peer=None, purpose=None)
            tok = uuid.uuid4().hex
            _leases[tok] = {
                'holder': peer, 'purpose': purpose, 'kg': kg, 'cls': cls,
                'acquired_at': now, 'last_heartbeat': now,
            }
            _persist()
            self._json(200, {'token': tok, 'ttl_s': TTL_S, 'cls': cls,
                             'shared_used': shared_live + (cls == 'shared'),
                             'shared_slots': SHARED_SLOTS})

    def _heartbeat(self):
        body = self._read_body()
        token = body.get('token')
        now = time.time()
        with _lock:
            _expire(now)
            l = _leases.get(token)
            if l is None:
                return self._json(410, {'error': 'no_lease'})
            l['last_heartbeat'] = now
            _persist()
            self._json(200, {'ok': True, 'ttl_s': TTL_S,
                             'age_s': round(now - l['acquired_at'], 1)})

    def _release(self):
        body = self._read_body()
        token = body.get('token')
        with _lock:
            l = _leases.pop(token, None)
            if l is None:
                return self._json(410, {'error': 'no_lease'})
            print(f'[broker] released by {l["holder"]} (purpose={l["purpose"]}, '
                  f'held {time.time() - l["acquired_at"]:.1f}s)', file=sys.stderr)
            _persist()
            self._json(200, {'ok': True})


def main():
    _restore()
    srv = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    print(f'[broker] listening on {LISTEN_HOST}:{LISTEN_PORT} '
          f'(ttl={TTL_S}s, kg_slots={SHARED_SLOTS}, state={STATE_FILE})',
          file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == '__main__':
    main()
