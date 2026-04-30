"""Distributed Zenodo upload mutex client.

All Zenodo write operations across the peer fleet share one API token
and (sometimes) one draft deposition.  Concurrent PUTs would fail with
409s and orphan files.  This module provides a tiny client for the
broker on the primary instance (see ``app.py`` ``/api/v1/zenodo/lock``
endpoints).

Usage::

    from zenodo_lock import zenodo_upload_lock

    with zenodo_upload_lock(purpose='kg_upload', kg='91109'):
        client.upload_stream(...)

If no broker URL is configured (env ``ZENODO_LOCK_URL`` empty), the
lock is a no-op so single-instance development still works.
"""
from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Optional

import requests

_ADMIN_TOKEN_PATH = Path('data/admin_token')


def _admin_headers() -> dict:
    """Cluster auth header for the broker. Re-read each call so token
    rotations on disk propagate without restarts. Empty dict if missing
    (single-host dev or loopback)."""
    try:
        tok = _ADMIN_TOKEN_PATH.read_text().strip()
        if tok:
            return {'X-Admin-Token': tok}
    except Exception:
        pass
    return {}

log = logging.getLogger(__name__)

_DEFAULT_TTL = 120.0  # must match _ZENODO_LOCK_TTL on the server
_HEARTBEAT_INTERVAL = 30.0  # seconds; well under TTL/2
_ACQUIRE_RETRY_INTERVAL = 5.0  # seconds between retries when locked
_ACQUIRE_TIMEOUT = 1800.0  # max wait for the lease (30 min)
# HTTP timeout for the broker POST. Primary's gunicorn workers can stall
# briefly under load (heavy GPKG building, search-index rebuild, etc.) so
# 15 s was too tight — a slow tick produced spurious "proceeding without
# lease" warnings on every cache-only peer simultaneously.
_BROKER_HTTP_TIMEOUT = (5, 60)  # (connect, read)
# Number of consecutive transport failures we tolerate quietly before
# escalating to a warning. The broker is best‑effort — a single blip is
# normal, but sustained failures should be loud.


def _broker_url() -> Optional[str]:
    url = os.environ.get('ZENODO_LOCK_URL', '').strip()
    return url.rstrip('/') if url else None


def _peer_id() -> str:
    return (
        os.environ.get('ZENODO_LOCK_PEER')
        or os.environ.get('INSTANCE_ID')
        or socket.gethostname()
        or 'unknown'
    )


class ZenodoLease:
    """Active lease on the broker.  Renews itself in a daemon thread."""

    def __init__(self, broker: str, peer: str, token: str, ttl_s: float):
        self.broker = broker
        self.peer = peer
        self.token = token
        self.ttl_s = ttl_s
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._heartbeat_loop, name='zenodo-lease-hb', daemon=True)
        self._thread.start()

    def _heartbeat_loop(self):
        while not self._stop.wait(_HEARTBEAT_INTERVAL):
            try:
                r = requests.post(
                    f"{self.broker}/api/v1/zenodo/lock/heartbeat",
                    json={'token': self.token}, timeout=15,
                    headers=_admin_headers(),
                )
                if r.status_code == 410:
                    log.warning('Zenodo lease lost (410) — broker reclaimed it')
                    self._stop.set()
                    return
                r.raise_for_status()
            except Exception as e:
                log.warning('Zenodo lease heartbeat failed: %s', e)

    def release(self):
        self._stop.set()
        try:
            requests.delete(
                f"{self.broker}/api/v1/zenodo/lock",
                json={'token': self.token}, timeout=15,
                headers=_admin_headers(),
            )
        except Exception as e:
            log.warning('Zenodo lease release failed: %s', e)


@contextlib.contextmanager
def zenodo_upload_lock(purpose: str = 'unknown', kg: str | None = None,
                       max_wait_s: float = _ACQUIRE_TIMEOUT):
    """Block until a Zenodo upload lease is granted, then auto-release.

    No-op if ``ZENODO_LOCK_URL`` is not set (single-instance mode).
    """
    broker = _broker_url()
    if broker is None:
        yield None
        return
    peer = _peer_id()
    deadline = time.monotonic() + max_wait_s
    lease: Optional[ZenodoLease] = None
    backoff = _ACQUIRE_RETRY_INTERVAL
    transport_failures = 0
    while True:
        try:
            r = requests.post(
                f"{broker}/api/v1/zenodo/lock",
                json={'peer': peer, 'purpose': purpose, 'kg': kg},
                timeout=_BROKER_HTTP_TIMEOUT,
                headers=_admin_headers(),
            )
        except Exception as e:
            # Broker unreachable. We retry a couple of times on transport
            # failures before giving up — the primary's gunicorn workers
            # can briefly stall under load and we'd rather wait 5–10 s for
            # the lease than fan out into uncoordinated uploads.
            transport_failures += 1
            if transport_failures <= 2 and time.monotonic() < deadline:
                log.info('Zenodo lock broker transport error (%s) — retry %d',
                         str(e)[:120], transport_failures)
                time.sleep(min(backoff, max(1.0, deadline - time.monotonic())))
                backoff = min(backoff * 1.5, 30.0)
                continue
            # Sustained failure — fail open. We'd rather process than
            # deadlock the fleet on a network outage; the director's
            # single‑active rule already keeps Zenodo writes mostly serial.
            log.warning(
                'Zenodo lock broker unreachable after %d attempts (%s) — '
                'proceeding without lease', transport_failures, str(e)[:160])
            yield None
            return
        if r.status_code == 200:
            data = r.json()
            lease = ZenodoLease(broker, peer, data['token'],
                                float(data.get('ttl_s', _DEFAULT_TTL)))
            log.info('Zenodo lease acquired (purpose=%s, kg=%s)', purpose, kg)
            break
        if r.status_code == 423:
            data = r.json() if r.content else {}
            log.info('Zenodo lock busy (held by %s for %ss, idle %ss) — waiting',
                     data.get('holder'), data.get('age_s'), data.get('idle_s'))
        else:
            log.warning('Zenodo lock acquire returned HTTP %d: %s',
                        r.status_code, r.text[:200])
        if time.monotonic() >= deadline:
            log.error('Zenodo lock acquire timeout after %.0fs — proceeding without lease',
                      max_wait_s)
            yield None
            return
        time.sleep(min(backoff, max(1.0, deadline - time.monotonic())))
        backoff = min(backoff * 1.5, 30.0)
    try:
        yield lease
    finally:
        if lease is not None:
            lease.release()
