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

State is held in memory and persisted to disk on every mutation so a
restart doesn't 410 every active heartbeat.  TTL = 120 s.

Auth: same admin token model as gunicorn (header ``X-Admin-Token``
matching ``data/admin_token``).  Loopback exempt.

Runs under systemd as ``zenodo_lock_broker.service`` on the primary.
Peers point ``ZENODO_LOCK_URL`` at it (e.g.
``http://srtm-lidar-at.exe.xyz:8001``).  The exe.dev proxy forwards
port 8001 same as 8000.
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
LISTEN_HOST = os.environ.get('ZENODO_LOCK_BROKER_HOST', '0.0.0.0')
LISTEN_PORT = int(os.environ.get('ZENODO_LOCK_BROKER_PORT', '8001'))

_lock = threading.Lock()
_state: dict = {
    'holder': None,
    'token': None,
    'acquired_at': 0.0,
    'last_heartbeat': 0.0,
    'purpose': None,
    'kg': None,
}


def _persist() -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(_state))
        tmp.replace(STATE_FILE)
    except Exception as e:
        print(f'[broker] persist error: {e}', file=sys.stderr)


def _restore() -> None:
    try:
        if not STATE_FILE.exists():
            return
        d = json.loads(STATE_FILE.read_text())
        last_hb = float(d.get('last_heartbeat') or 0.0)
        if last_hb and (time.time() - last_hb) <= TTL_S:
            _state.update(d)
            print(
                f'[broker] restored lease holder={d.get("holder")} '
                f'idle={time.time() - last_hb:.1f}s',
                file=sys.stderr,
            )
    except Exception as e:
        print(f'[broker] restore error: {e}', file=sys.stderr)


def _is_stale(now: float) -> bool:
    if _state['holder'] is None:
        return False
    return (now - _state['last_heartbeat']) > TTL_S


def _admin_token() -> str:
    try:
        return ADMIN_TOKEN_FILE.read_text().strip()
    except Exception:
        return ''


class Handler(BaseHTTPRequestHandler):
    server_version = 'srtm-lidar-zenodo-lock/1.0'

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
            if _state['holder'] is None:
                return self._json(200, {'free': True})
            return self._json(200, {
                'free': False,
                'holder': _state['holder'],
                'purpose': _state['purpose'],
                'kg': _state['kg'],
                'age_s': round(now - _state['acquired_at'], 1),
                'idle_s': round(now - _state['last_heartbeat'], 1),
                'stale': _is_stale(now),
                'ttl_s': TTL_S,
            })

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
        now = time.time()
        with _lock:
            if _state['holder'] is not None and not _is_stale(now):
                if _state['holder'] == peer:
                    _state['last_heartbeat'] = now
                    _persist()
                    return self._json(200, {
                        'token': _state['token'],
                        'ttl_s': TTL_S,
                        'reacquired': True,
                    })
                return self._json(423, {
                    'error': 'locked',
                    'holder': _state['holder'],
                    'purpose': _state['purpose'],
                    'kg': _state['kg'],
                    'age_s': round(now - _state['acquired_at'], 1),
                    'idle_s': round(now - _state['last_heartbeat'], 1),
                })
            if _state['holder'] is not None:
                print(
                    f'[broker] stale holder={_state["holder"]} '
                    f'reclaimed by {peer}', file=sys.stderr,
                )
            tok = uuid.uuid4().hex
            _state.update({
                'holder': peer, 'token': tok,
                'acquired_at': now, 'last_heartbeat': now,
                'purpose': purpose, 'kg': kg,
            })
            _persist()
            self._json(200, {'token': tok, 'ttl_s': TTL_S})

    def _heartbeat(self):
        body = self._read_body()
        token = body.get('token')
        now = time.time()
        with _lock:
            if _state['holder'] is None or _state['token'] != token:
                return self._json(410, {'error': 'no_lease'})
            _state['last_heartbeat'] = now
            _persist()
            self._json(200, {
                'ok': True, 'ttl_s': TTL_S,
                'age_s': round(now - _state['acquired_at'], 1),
            })

    def _release(self):
        body = self._read_body()
        token = body.get('token')
        with _lock:
            if _state['holder'] is None or _state['token'] != token:
                return self._json(410, {'error': 'no_lease'})
            print(
                f'[broker] released by {_state["holder"]} '
                f'(purpose={_state["purpose"]}, '
                f'held {time.time() - _state["acquired_at"]:.1f}s)',
                file=sys.stderr,
            )
            _state.update({
                'holder': None, 'token': None,
                'acquired_at': 0.0, 'last_heartbeat': 0.0,
                'purpose': None, 'kg': None,
            })
            _persist()
            self._json(200, {'ok': True})


def main():
    _restore()
    srv = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    print(
        f'[broker] listening on {LISTEN_HOST}:{LISTEN_PORT} '
        f'(ttl={TTL_S}s, state={STATE_FILE})',
        file=sys.stderr, flush=True,
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == '__main__':
    main()
