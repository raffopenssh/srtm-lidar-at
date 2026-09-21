"""Lightweight per-endpoint byte accounting for HTTP traffic.

Two probes, both cheap (a dict increment per request):

* ``install_requests_meter()`` wraps ``requests.Session.send`` so every
  *outbound* call this process makes (director fanout, shadow snapshot
  pushes, peer sync, Zenodo) is bucketed by ``host + path`` with request
  and response byte counts.
* ``install_flask_meter(app)`` adds ``before/after_request`` hooks so
  every *inbound* request is bucketed by ``method + path`` (numeric
  path segments collapsed) with body-in / body-out byte counts.

Counters are per-process; each gunicorn worker flushes its dict to
``/tmp/net_meter/<pid>.json`` every ``FLUSH_INTERVAL`` seconds and
``snapshot()`` merges every file so ``/api/v1/net_stats`` shows the
whole box regardless of which worker answers. Files are keyed by pid
so a restart starts fresh (stale pids are dropped on read).

Purpose: attribute the primary's egress/ingress (metered by exe.dev per
VM) to concrete endpoints so bandwidth work targets the real hogs
instead of guesses. See ``docs/peer-director.md → Bandwidth accounting``.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

METER_DIR = Path('/tmp/net_meter')
FLUSH_INTERVAL = 20.0
_lock = threading.Lock()
_out: dict[str, list] = {}   # key -> [n, req_bytes, resp_bytes]
_in: dict[str, list] = {}
_started = time.time()
_num_re = re.compile(r'/\d[\w.-]*')
_peer_host_re = re.compile(r'^srtm-lidar-at\d+\.exe\.xyz')
_installed = {'requests': False, 'flask': False, 'flusher': False}


def _norm_path(path: str) -> str:
    return _num_re.sub('/<n>', path or '/')


def _bump(table: dict, key: str, req_b: int, resp_b: int) -> None:
    with _lock:
        e = table.get(key)
        if e is None:
            table[key] = [1, int(req_b), int(resp_b)]
        else:
            e[0] += 1
            e[1] += int(req_b)
            e[2] += int(resp_b)


# ── outbound (requests) ────────────────────────────────────────────

def install_requests_meter() -> None:
    if _installed['requests']:
        return
    try:
        import requests
        from urllib.parse import urlsplit
    except Exception:
        return
    orig_send = requests.Session.send

    def send(self, prep, **kw):
        body = prep.body
        if body is None:
            rb = 0
        elif isinstance(body, (bytes, bytearray, str)):
            rb = len(body)
        else:
            rb = 0  # file-like / generator (streamed upload) — see below
        rb += sum(len(k) + len(v) + 4 for k, v in prep.headers.items())
        u = urlsplit(prep.url or '')
        host = _peer_host_re.sub('<peer>', u.netloc)
        key = f'{prep.method} {host}{_norm_path(u.path)}'
        try:
            r = orig_send(self, prep, **kw)
        except Exception:
            _bump(_out, key, rb, 0)
            raise
        resp_b = 0
        try:
            cl = r.headers.get('Content-Length')
            if prep.method == 'HEAD':
                cl = 0  # header only; no body crosses the wire
            if cl is not None:
                resp_b = int(cl)
            elif not kw.get('stream'):
                resp_b = len(r.content)
        except Exception:
            pass
        if rb == 0 and body is not None and hasattr(body, 'tell'):
            try:
                rb = body.tell()
            except Exception:
                pass
        _bump(_out, key, rb, resp_b)
        return r

    requests.Session.send = send
    _installed['requests'] = True
    _ensure_flusher()


# ── inbound (flask) ────────────────────────────────────────────────

def install_flask_meter(app) -> None:
    if _installed['flask']:
        return
    from flask import request

    @app.after_request
    def _net_meter_after(resp):
        try:
            rb = request.content_length or 0
            ob = 0
            cl = resp.headers.get('Content-Length')
            if cl is not None:
                ob = int(cl)
            elif not (resp.direct_passthrough or resp.is_streamed):
                ob = len(resp.get_data())
            # exe.dev proxy traffic arrives on 127.0.0.1 but carries
            # X-Forwarded-For; true loopback (director thread, local
            # curl) does not. Only 'ext' costs metered bandwidth.
            if request.headers.get('X-Forwarded-For') or not (
                    request.remote_addr or '').startswith('127.'):
                src = 'ext'
            else:
                src = 'lo'
            key = f'{request.method} {_norm_path(request.path)} [{src}]'
            _bump(_in, key, rb, ob)
        except Exception:
            pass
        return resp

    _installed['flask'] = True
    _ensure_flusher()


# ── persistence + snapshot ─────────────────────────────────────────

def _flush() -> None:
    try:
        METER_DIR.mkdir(parents=True, exist_ok=True)
        with _lock:
            payload = {'pid': os.getpid(), 'started': _started,
                       'ts': time.time(), 'out': _out, 'in': _in}
            data = json.dumps(payload)
        tmp = METER_DIR / f'.{os.getpid()}.tmp'
        tmp.write_text(data)
        os.replace(tmp, METER_DIR / f'{os.getpid()}.json')
    except Exception:
        pass


def _flusher():
    while True:
        time.sleep(FLUSH_INTERVAL)
        _flush()


def _ensure_flusher() -> None:
    if _installed['flusher']:
        return
    _installed['flusher'] = True
    threading.Thread(target=_flusher, daemon=True, name='net-meter').start()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def snapshot(top: int = 40) -> dict:
    """Merge all live workers' counters; rank by total bytes."""
    _flush()
    out: dict[str, list] = {}
    inn: dict[str, list] = {}
    started = time.time()
    procs = []
    for f in METER_DIR.glob('*.json'):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        pid = int(d.get('pid') or 0)
        if not _pid_alive(pid):
            try:
                f.unlink()
            except Exception:
                pass
            continue
        procs.append(pid)
        started = min(started, float(d.get('started') or started))
        for tbl, src in ((out, d.get('out') or {}), (inn, d.get('in') or {})):
            for k, v in src.items():
                e = tbl.setdefault(k, [0, 0, 0])
                e[0] += v[0]; e[1] += v[1]; e[2] += v[2]
    win = max(1.0, time.time() - started)

    def rows(tbl):
        r = sorted(tbl.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))[:top]
        return [{'key': k, 'n': v[0], 'req_bytes': v[1], 'resp_bytes': v[2],
                 'mb_per_h': round((v[1] + v[2]) / win * 3600 / 1e6, 2)}
                for k, v in r]

    tot_out = sum(v[1] + v[2] for v in out.values())
    tot_in = sum(v[1] + v[2] for v in inn.values())
    return {
        'window_s': round(win), 'pids': procs,
        'outbound_total_mb_per_h': round(tot_out / win * 3600 / 1e6, 2),
        'inbound_total_mb_per_h': round(tot_in / win * 3600 / 1e6, 2),
        'outbound': rows(out), 'inbound': rows(inn),
    }


def snapshot_text(top: int = 30) -> str:
    s = snapshot(top)
    L = [f"# net_meter window={s['window_s']}s pids={s['pids']} "
         f"out={s['outbound_total_mb_per_h']}MB/h in={s['inbound_total_mb_per_h']}MB/h",
         '', f"{'MB/h':>8} {'n':>6} {'req_MB':>8} {'resp_MB':>8}  OUTBOUND (this process → remote)"]
    for r in s['outbound']:
        L.append(f"{r['mb_per_h']:8.2f} {r['n']:6d} {r['req_bytes']/1e6:8.2f} "
                 f"{r['resp_bytes']/1e6:8.2f}  {r['key']}")
    L += ['', f"{'MB/h':>8} {'n':>6} {'in_MB':>8} {'out_MB':>8}  INBOUND (remote → this process)"]
    for r in s['inbound']:
        L.append(f"{r['mb_per_h']:8.2f} {r['n']:6d} {r['req_bytes']/1e6:8.2f} "
                 f"{r['resp_bytes']/1e6:8.2f}  {r['key']}")
    return '\n'.join(L) + '\n'
