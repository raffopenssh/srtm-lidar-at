"""Progress-aware, resumable, single-flight downloads from Zenodo.

Every request-path read of a Zenodo product (light GPKGs for the v3 tree
service and the ``/api/v1/kg/<code>/*`` layer endpoints) goes through
``download()`` so that

* **progress is observable** — bytes done / total, EMA throughput, ETA,
  attempt, status — both in-process (``hook`` callback → task progress
  file) and cross-process (sidecar JSON under ``<cache>/.fetch/`` that the
  sibling gunicorn worker reads for ``/api/v1/zenodo/fetches`` and the
  ``zenodo_fetch:`` line in ``/process.txt``);
* **one download per file** — in-process waiters share the same fetch,
  and an ``fcntl`` lock on ``<dest>.lock`` makes the other worker wait and
  *relay* the sidecar progress instead of downloading a second copy;
* **slow Zenodo does not wedge a request** — per-read stall timeout
  (``STALL_S``), up to ``MAX_ATTEMPTS`` attempts with HTTP ``Range`` resume
  from the partial ``.tmp``, 429/503 ``Retry-After`` honoured; a bounded
  number of concurrent downloads per process (``MAX_PARALLEL``), the rest
  wait in ``queued`` state;
* callers may bound how long they block (``wait=`` seconds) and get a
  ``FetchInProgress`` carrying the live state so a sync GET can answer
  ``202 Accepted`` + ``Retry-After`` while the download continues in the
  background.

``prefetch_many()`` runs several downloads in parallel and folds their
states into one aggregate (files done/total, bytes, rate, ETA) for the
hook — used by trees_v3 to pull every covering GPKG at once.

A short ring of completed fetches (``recent()``) feeds ``stats()``:
median throughput over the last hour + a ``slow`` verdict, so operators
and agents can tell "Zenodo is slow right now" from "this KG is big".
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

MAX_PARALLEL = int(os.environ.get('ZENODO_FETCH_PARALLEL', '4'))
MAX_ATTEMPTS = 4
STALL_S = 60.0            # no bytes for this long → abort attempt, resume
CONNECT_S = 20.0
CHUNK = 256 * 1024
SIDECAR_EVERY_S = 0.5
SLOW_MBPS = 1.0           # fleet-wide "Zenodo is slow" verdict threshold
RECENT_MAX = 50
SIDECAR_DIRNAME = '.fetch'

_LOCAL = threading.local()
_SEM = threading.BoundedSemaphore(MAX_PARALLEL)
_INFLIGHT: dict[str, 'FetchState'] = {}
_INFLIGHT_LOCK = threading.Lock()
_RECENT: list[dict] = []
_RECENT_LOCK = threading.Lock()
_TOKEN_RE = re.compile(r'(access_token=)[^&]+')


def redact(url: str) -> str:
    return _TOKEN_RE.sub(r'\1***', url or '')


class FetchInProgress(Exception):
    """Raised by ``download(wait=N)`` when the file is still downloading
    after *N* seconds.  ``.state`` is the live progress dict."""

    def __init__(self, state: dict):
        super().__init__(state.get('detail') or 'download in progress')
        self.state = state


class FetchState:
    """Mutable progress record for one file."""

    def __init__(self, key: str, dest: Path, url: str, label: str, expected: int):
        self.key = key
        self.dest = dest
        self.url = redact(url)
        self.label = label or dest.name
        self.file = dest.name
        self.bytes_total = int(expected or 0)
        self.bytes_done = 0
        self.status = 'queued'        # queued|connecting|downloading|verifying|done|error
        self.attempt = 0
        self.error: Optional[str] = None
        self.started = time.time()
        self.updated = self.started
        self.finished: Optional[float] = None
        self.rate_bps = 0.0            # EMA over ~5 s
        self._rate_t = self.started
        self._rate_b = 0
        self.relay = False             # progress relayed from another worker
        self.pid = os.getpid()
        self.done_evt = threading.Event()
        self.result: Optional[Path] = None
        self.exc: Optional[BaseException] = None
        self._last_sidecar = 0.0
        self._hooks: list[Callable[[dict], None]] = []
        self._hooks_lock = threading.Lock()

    # -- rate / eta ---------------------------------------------------------
    def _tick_rate(self, now: float) -> None:
        dt = now - self._rate_t
        if dt >= 1.0:
            inst = (self.bytes_done - self._rate_b) / dt
            self.rate_bps = inst if self.rate_bps <= 0 else 0.6 * self.rate_bps + 0.4 * inst
            self._rate_t, self._rate_b = now, self.bytes_done

    @property
    def eta_s(self) -> Optional[float]:
        if self.bytes_total and self.rate_bps > 1024 and self.bytes_done < self.bytes_total:
            return (self.bytes_total - self.bytes_done) / self.rate_bps
        return None

    @property
    def pct(self) -> Optional[float]:
        if self.bytes_total:
            return min(100.0, 100.0 * self.bytes_done / self.bytes_total)
        return None

    def detail(self) -> str:
        mb = self.bytes_done / 1e6
        if self.status == 'queued':
            return f'Queued for Zenodo download: {self.label}'
        if self.status == 'connecting':
            return f'Connecting to Zenodo for {self.label} (attempt {self.attempt})…'
        if self.status == 'verifying':
            return f'Verifying {self.label} ({mb:.1f} MB)…'
        if self.status == 'done':
            return f'Downloaded {self.label} ({mb:.1f} MB)'
        if self.status == 'error':
            return f'Download failed: {self.label}: {self.error}'
        tot = f'/{self.bytes_total / 1e6:.1f}' if self.bytes_total else ''
        rate = f' · {self.rate_bps / 1e6:.2f} MB/s' if self.rate_bps > 0 else ''
        eta = self.eta_s
        eta_s = f' · ETA {_fmt_s(eta)}' if eta is not None else ''
        att = f' · attempt {self.attempt}' if self.attempt > 1 else ''
        src = ' (relayed)' if self.relay else ''
        return f'Downloading {self.label} from Zenodo {mb:.1f}{tot} MB{rate}{eta_s}{att}{src}'

    def to_dict(self) -> dict:
        now = time.time()
        return {
            'key': self.key, 'file': self.file, 'label': self.label, 'url': self.url,
            'status': self.status, 'attempt': self.attempt, 'error': self.error,
            'bytes_done': self.bytes_done, 'bytes_total': self.bytes_total or None,
            'pct': round(self.pct, 1) if self.pct is not None else None,
            'rate_mbps': round(self.rate_bps / 1e6, 3),
            'eta_s': round(self.eta_s, 1) if self.eta_s is not None else None,
            'elapsed_s': round((self.finished or now) - self.started, 1),
            'started': self.started, 'updated': self.updated,
            'pid': self.pid, 'relay': self.relay, 'source': 'zenodo',
            'detail': self.detail(),
        }

    # -- notifications --------------------------------------------------------
    def add_hook(self, fn) -> None:
        if fn is None:
            return
        with self._hooks_lock:
            self._hooks.append(fn)

    def emit(self, force: bool = False) -> None:
        now = time.time()
        self.updated = now
        self._tick_rate(now)
        if force or now - self._last_sidecar >= SIDECAR_EVERY_S:
            self._last_sidecar = now
            d = self.to_dict()
            if not self.relay:
                _write_sidecar(self.dest, d)
            with self._hooks_lock:
                hooks = list(self._hooks)
            for h in hooks:
                try:
                    h(d)
                except Exception:  # noqa: BLE001
                    pass


def _fmt_s(s: float) -> str:
    s = int(round(s))
    if s < 60:
        return f'{s}s'
    if s < 3600:
        return f'{s // 60}m{s % 60:02d}s'
    return f'{s // 3600}h{(s % 3600) // 60:02d}m'


# ---------------------------------------------------------------------------
# sidecars (cross-worker visibility)
# ---------------------------------------------------------------------------

def _sidecar_path(dest: Path) -> Path:
    return dest.parent / SIDECAR_DIRNAME / f'{dest.name}.json'


def _write_sidecar(dest: Path, d: dict) -> None:
    p = _sidecar_path(dest)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(d))
        tmp.replace(p)
    except OSError:
        pass


def _read_sidecar(dest: Path) -> Optional[dict]:
    try:
        return json.loads(_sidecar_path(dest).read_text())
    except Exception:  # noqa: BLE001
        return None


def _remove_sidecar(dest: Path) -> None:
    try:
        _sidecar_path(dest).unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# per-thread hook (bound by the async task worker)
# ---------------------------------------------------------------------------

def set_hook(fn: Optional[Callable[[dict], None]]) -> None:
    """Bind a progress callback to the current thread.  Any ``download()``
    on this thread (or a ``prefetch_many`` started from it) reports through
    it — the async task worker binds ``_progress_set(task_id, …)``."""
    _LOCAL.hook = fn


def get_hook() -> Optional[Callable[[dict], None]]:
    return getattr(_LOCAL, 'hook', None)


# ---------------------------------------------------------------------------
# core download
# ---------------------------------------------------------------------------

def _auth_headers(url: str, headers: Optional[dict]) -> dict:
    hdr = {'User-Agent': 'srtm-lidar/1.0 zenodo_fetch'}
    if url.startswith('https://zenodo.org/api/files/') and 'access_token=' not in url:
        try:
            from zenodo_client import DEFAULT_TOKEN
            hdr['Authorization'] = f'Bearer {DEFAULT_TOKEN}'
        except Exception:  # noqa: BLE001
            pass
    if headers:
        hdr.update(headers)
    return hdr


def _stream_attempt(st: FetchState, url: str, hdr: dict, tmp: Path, expected: int) -> None:
    """One HTTP attempt, resuming from ``tmp`` if it has bytes.  Raises on
    any failure (caller retries)."""
    import requests
    have = tmp.stat().st_size if tmp.exists() else 0
    if expected and have >= expected:
        st.bytes_done = have
        return
    h = dict(hdr)
    if have > 0:
        h['Range'] = f'bytes={have}-'
    st.status = 'connecting'
    st.emit(force=True)
    try:
        from http_pool import session
        sess = session()
    except Exception:  # noqa: BLE001
        sess = requests
    r = sess.get(url, headers=h, stream=True, timeout=(CONNECT_S, STALL_S), allow_redirects=True)
    try:
        if r.status_code in (429, 503):
            ra = r.headers.get('Retry-After')
            try:
                delay = min(120.0, float(ra)) if ra else 15.0
            except ValueError:
                delay = 15.0
            raise _Backoff(f'HTTP {r.status_code}', delay)
        r.raise_for_status()
        mode = 'ab'
        if have > 0 and r.status_code != 206:
            # server ignored Range → start over
            have = 0
            mode = 'wb'
        if not expected:
            cl = r.headers.get('Content-Length')
            if cl:
                try:
                    st.bytes_total = int(cl) + (have if r.status_code == 206 else 0)
                except ValueError:
                    pass
        st.bytes_done = have
        st.status = 'downloading'
        st.emit(force=True)
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                st.bytes_done += len(chunk)
                st.emit()
    finally:
        r.close()


class _Backoff(Exception):
    def __init__(self, msg: str, delay: float):
        super().__init__(msg)
        self.delay = delay


def _verify(st: FetchState, tmp: Path, expected: int, md5: Optional[str]) -> None:
    n = tmp.stat().st_size
    if expected and n != expected:
        raise RuntimeError(f'size mismatch: got {n} want {expected}')
    if md5:
        st.status = 'verifying'
        st.emit(force=True)
        h = hashlib.md5()
        with open(tmp, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        if h.hexdigest() != md5:
            raise RuntimeError(f'md5 mismatch: {h.hexdigest()} != {md5}')


def _run(st: FetchState, url: str, headers: Optional[dict], expected: int,
         md5: Optional[str], lock_path: Path) -> None:
    """Worker body: single-flight lock → attempts → verify → rename."""
    dest, tmp = st.dest, st.dest.with_name(st.dest.name + '.tmp')
    hdr = _auth_headers(url, headers)
    lock_f = None
    acquired_sem = False
    try:
        lock_f = open(lock_path, 'a+')
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # the sibling worker is downloading this very file — relay
            st.relay = True
            st.status = 'downloading'
            deadline = time.time() + 3600
            while time.time() < deadline:
                if dest.exists() and dest.stat().st_size > 0 and (
                        not expected or dest.stat().st_size == expected):
                    st.bytes_done = dest.stat().st_size
                    st.bytes_total = st.bytes_total or st.bytes_done
                    break
                sc = _read_sidecar(dest)
                if sc:
                    st.bytes_done = int(sc.get('bytes_done') or 0)
                    st.bytes_total = int(sc.get('bytes_total') or st.bytes_total or 0)
                    st.rate_bps = float(sc.get('rate_mbps') or 0) * 1e6
                    st.attempt = int(sc.get('attempt') or 0)
                    st.status = sc.get('status') if sc.get('status') in ('connecting', 'downloading', 'verifying') else 'downloading'
                    if sc.get('status') == 'error' and time.time() - float(sc.get('updated') or 0) < 5:
                        raise RuntimeError(f"sibling worker failed: {sc.get('error')}")
                st.emit(force=True)
                # try to take over the lock (sibling finished / died)
                try:
                    fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    st.relay = False
                    break
                except BlockingIOError:
                    time.sleep(0.5)
            if st.relay:
                if dest.exists() and dest.stat().st_size > 0:
                    st.status = 'done'
                    st.finished = time.time()
                    st.result = dest
                    return
                raise RuntimeError('timed out waiting for sibling download')
        # -- we hold the lock ------------------------------------------------
        if dest.exists() and dest.stat().st_size > 0 and (not expected or dest.stat().st_size == expected):
            st.bytes_done = st.bytes_total = dest.stat().st_size
            st.status = 'done'
            st.finished = time.time()
            st.result = dest
            return
        _SEM.acquire()
        acquired_sem = True
        last_exc: Optional[BaseException] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            st.attempt = attempt
            try:
                _stream_attempt(st, url, hdr, tmp, expected)
                _verify(st, tmp, expected, md5)
                tmp.replace(dest)
                st.bytes_done = dest.stat().st_size
                st.bytes_total = st.bytes_total or st.bytes_done
                st.status = 'done'
                st.finished = time.time()
                st.result = dest
                log.info('zenodo_fetch: %s %.1f MB in %.1fs (%.2f MB/s, %d attempt%s)',
                         st.label, st.bytes_done / 1e6, st.finished - st.started,
                         st.bytes_done / 1e6 / max(0.001, st.finished - st.started),
                         attempt, '' if attempt == 1 else 's')
                return
            except _Backoff as e:
                last_exc = e
                log.warning('zenodo_fetch: %s attempt %d: %s — backing off %.0fs',
                            st.label, attempt, e, e.delay)
                st.error = f'{e} (retry in {e.delay:.0f}s)'
                st.emit(force=True)
                time.sleep(e.delay)
            except Exception as e:  # noqa: BLE001
                last_exc = e
                # size/md5 mismatch → partial file is poison, restart clean
                if 'mismatch' in str(e):
                    tmp.unlink(missing_ok=True)
                log.warning('zenodo_fetch: %s attempt %d failed at %.1f MB: %s',
                            st.label, attempt, st.bytes_done / 1e6, e)
                st.error = str(e)[:200]
                st.emit(force=True)
                if attempt < MAX_ATTEMPTS:
                    time.sleep(min(30.0, 2.0 ** attempt))
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'download failed after {MAX_ATTEMPTS} attempts: {last_exc}')
    except BaseException as e:  # noqa: BLE001
        st.status = 'error'
        st.error = str(e)[:200]
        st.exc = e
        st.finished = time.time()
    finally:
        if acquired_sem:
            _SEM.release()
        st.emit(force=True)
        if not st.relay:
            _remove_sidecar(dest)
        if lock_f is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
            except OSError:
                pass
            lock_f.close()
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        _record_recent(st)
        with _INFLIGHT_LOCK:
            if _INFLIGHT.get(st.key) is st:
                del _INFLIGHT[st.key]
        st.done_evt.set()


def _record_recent(st: FetchState) -> None:
    if st.relay:
        return
    d = st.to_dict()
    with _RECENT_LOCK:
        _RECENT.append(d)
        del _RECENT[:-RECENT_MAX]


def start(url: str, dest: Path, *, headers: Optional[dict] = None, expected_size: int = 0,
          md5: Optional[str] = None, label: str = '', hook=None) -> FetchState:
    """Begin (or join) the download of *url* → *dest*.  Returns the shared
    ``FetchState``; ``hook`` (if any) is attached for progress callbacks."""
    dest = Path(dest)
    key = str(dest.resolve())
    if md5 and md5.startswith('md5:'):
        md5 = md5[4:]
    with _INFLIGHT_LOCK:
        st = _INFLIGHT.get(key)
        if st is None:
            st = FetchState(key, dest, url, label, expected_size)
            _INFLIGHT[key] = st
            dest.parent.mkdir(parents=True, exist_ok=True)
            lock_path = dest.with_name(dest.name + '.lock')
            t = threading.Thread(target=_run, args=(st, url, headers, int(expected_size or 0), md5, lock_path),
                                 name=f'zenodo_fetch:{dest.name}', daemon=True)
            st.add_hook(hook)
            st.emit(force=True)
            t.start()
            return st
    st.add_hook(hook)
    return st


def download(url: str, dest: Path, *, headers: Optional[dict] = None, expected_size: int = 0,
             md5: Optional[str] = None, label: str = '', wait: Optional[float] = None,
             hook=None) -> Path:
    """Download *url* to *dest* (atomic), reporting progress.

    ``wait`` — seconds to block.  ``None`` = until finished (max 1 h);
    ``0`` = start and raise ``FetchInProgress`` immediately if not cached.
    Raises ``FetchInProgress`` when still running after ``wait`` seconds,
    ``RuntimeError`` on terminal failure.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and (
            not expected_size or dest.stat().st_size == int(expected_size)):
        return dest
    if hook is None:
        hook = get_hook()
    st = start(url, dest, headers=headers, expected_size=expected_size, md5=md5, label=label, hook=hook)
    timeout = 3600.0 if wait is None else max(0.0, float(wait))
    if not st.done_evt.wait(timeout):
        raise FetchInProgress(st.to_dict())
    if st.exc is not None or st.result is None:
        raise RuntimeError(st.error or 'download failed')
    return st.result


# ---------------------------------------------------------------------------
# aggregate: many files at once
# ---------------------------------------------------------------------------

class _Group:
    def __init__(self, n: int, hook, label: str):
        self.n = n
        self.hook = hook
        self.label = label
        self.states: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.t0 = time.time()
        self._last = 0.0

    def update(self, d: dict) -> None:
        with self.lock:
            self.states[d['key']] = d
            now = time.time()
            if now - self._last < SIDECAR_EVERY_S and d.get('status') not in ('done', 'error'):
                return
            self._last = now
            agg = self.aggregate()
        if self.hook:
            try:
                self.hook(agg)
            except Exception:  # noqa: BLE001
                pass

    def aggregate(self) -> dict:
        sts = list(self.states.values())
        done = [s for s in sts if s['status'] == 'done']
        err = [s for s in sts if s['status'] == 'error']
        active = [s for s in sts if s['status'] in ('connecting', 'downloading', 'verifying')]
        b_done = sum(int(s.get('bytes_done') or 0) for s in sts)
        b_tot = sum(int(s.get('bytes_total') or 0) for s in sts)
        known_tot = all(s.get('bytes_total') for s in sts) and len(sts) == self.n
        rate = sum(float(s.get('rate_mbps') or 0) for s in active)
        eta = (b_tot - b_done) / (rate * 1e6) if known_tot and rate > 0.001 and b_tot > b_done else None
        pct = 100.0 * b_done / b_tot if known_tot and b_tot else (100.0 * len(done) / self.n if self.n else None)
        files_s = f'{len(done)}/{self.n} files'
        tot_s = f'/{b_tot / 1e6:.1f}' if known_tot else ''
        rate_s = f' · {rate:.2f} MB/s' if rate > 0 else ''
        eta_s = f' · ETA {_fmt_s(eta)}' if eta is not None else ''
        cur = ', '.join(s['label'] for s in active[:3])
        cur_s = f' — {cur}' if cur else ''
        err_s = f' · {len(err)} failed' if err else ''
        detail = (f'Downloading {self.label or "products"} from Zenodo: {files_s}, '
                  f'{b_done / 1e6:.1f}{tot_s} MB{rate_s}{eta_s}{err_s}{cur_s}')
        return {
            'status': 'downloading' if active or len(done) + len(err) < self.n else 'done',
            'label': self.label, 'files_done': len(done), 'files_total': self.n,
            'files_failed': len(err), 'files_active': len(active),
            'bytes_done': b_done, 'bytes_total': b_tot if known_tot else None,
            'pct': round(pct, 1) if pct is not None else None,
            'rate_mbps': round(rate, 3), 'eta_s': round(eta, 1) if eta is not None else None,
            'elapsed_s': round(time.time() - self.t0, 1), 'source': 'zenodo',
            'active': [s['label'] for s in active], 'detail': detail,
        }


def prefetch_many(items: list[dict], *, hook=None, label: str = '',
                  max_parallel: int = MAX_PARALLEL, wait: Optional[float] = None) -> dict:
    """Download many files concurrently with one aggregate progress stream.

    ``items``: ``[{url, dest, headers?, expected_size?, md5?, label?}]``.
    Returns ``{dest_str: Path | Exception}``.  Hook receives the aggregate
    dict (``files_done/total``, bytes, rate, ETA, ``detail``).
    """
    if hook is None:
        hook = get_hook()
    items = [i for i in items if i.get('url')]
    if not items:
        return {}
    grp = _Group(len(items), hook, label)
    out: dict[str, object] = {}

    def one(it: dict):
        dest = Path(it['dest'])
        try:
            p = download(it['url'], dest, headers=it.get('headers'),
                         expected_size=int(it.get('expected_size') or 0), md5=it.get('md5'),
                         label=it.get('label') or dest.name, wait=wait, hook=grp.update)
            if dest.exists() and str(dest.resolve()) not in grp.states:
                grp.update({'key': str(dest.resolve()), 'label': it.get('label') or dest.name,
                            'status': 'done', 'bytes_done': dest.stat().st_size,
                            'bytes_total': dest.stat().st_size, 'rate_mbps': 0})
            return p
        except Exception as e:  # noqa: BLE001
            grp.update({'key': str(dest.resolve()), 'label': it.get('label') or dest.name,
                        'status': 'error', 'bytes_done': 0, 'bytes_total': it.get('expected_size') or None,
                        'rate_mbps': 0, 'error': str(e)})
            return e

    with ThreadPoolExecutor(max_workers=max(1, min(max_parallel, len(items))),
                            thread_name_prefix='zenodo_prefetch') as ex:
        for it, res in zip(items, ex.map(one, items)):
            out[str(it['dest'])] = res
    if hook:
        try:
            hook(grp.aggregate())
        except Exception:  # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------

def inflight(cache_dirs: Optional[list] = None) -> list[dict]:
    """All in-flight fetches visible to this host: this process' registry
    plus fresh sidecars from sibling workers (updated < 3 min ago)."""
    with _INFLIGHT_LOCK:
        mine = {k: v.to_dict() for k, v in _INFLIGHT.items()}
    out = dict(mine)
    dirs = set(cache_dirs or [])
    try:
        import search_index as si
        dirs.add(Path(si.GPKG_CACHE_DIR))
    except Exception:  # noqa: BLE001
        pass
    now = time.time()
    for d in dirs:
        sd = Path(d) / SIDECAR_DIRNAME
        if not sd.is_dir():
            continue
        for p in sd.glob('*.json'):
            try:
                sc = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if now - float(sc.get('updated') or 0) > 180:
                try:
                    p.unlink()
                except OSError:
                    pass
                continue
            k = sc.get('key')
            if k and (k not in out or out[k].get('relay')):
                out[k] = sc
    return sorted(out.values(), key=lambda s: s.get('started') or 0)


def recent() -> list[dict]:
    with _RECENT_LOCK:
        return list(reversed(_RECENT))


def stats() -> dict:
    """Aggregate: in-flight count/bytes/rate + recent-hour median throughput
    and a ``slow`` verdict (median < ``SLOW_MBPS`` over ≥ 2 fetches)."""
    inf = inflight()
    rec = [r for r in recent() if r.get('status') == 'done' and time.time() - (r.get('started') or 0) < 3600
           and (r.get('bytes_done') or 0) > 1e6]
    rates = sorted((r['bytes_done'] / 1e6) / max(0.001, r['elapsed_s']) for r in rec)
    med = rates[len(rates) // 2] if rates else None
    fails = [r for r in recent() if r.get('status') == 'error' and time.time() - (r.get('started') or 0) < 3600]
    live_rate = sum(float(s.get('rate_mbps') or 0) for s in inf)
    return {
        'inflight': len(inf),
        'inflight_bytes_done': sum(int(s.get('bytes_done') or 0) for s in inf),
        'inflight_bytes_total': sum(int(s.get('bytes_total') or 0) for s in inf),
        'inflight_rate_mbps': round(live_rate, 3),
        'recent_1h': len(rec), 'recent_1h_failed': len(fails),
        'recent_1h_median_mbps': round(med, 3) if med is not None else None,
        'recent_1h_min_mbps': round(rates[0], 3) if rates else None,
        'slow': bool(med is not None and len(rates) >= 2 and med < SLOW_MBPS),
        'slow_threshold_mbps': SLOW_MBPS,
        'max_parallel': MAX_PARALLEL,
    }


def text_line() -> str:
    """One-line summary for ``/process.txt``."""
    s = stats()
    parts = [f'zenodo_fetch: inflight={s["inflight"]}']
    if s['inflight']:
        tot = s['inflight_bytes_total']
        parts.append(f'{s["inflight_bytes_done"] / 1e6:.0f}/{tot / 1e6:.0f}MB' if tot
                     else f'{s["inflight_bytes_done"] / 1e6:.0f}MB')
        parts.append(f'{s["inflight_rate_mbps"]:.2f}MB/s')
    parts.append(f'recent_1h={s["recent_1h"]}')
    if s['recent_1h_failed']:
        parts.append(f'failed={s["recent_1h_failed"]}')
    if s['recent_1h_median_mbps'] is not None:
        parts.append(f'med={s["recent_1h_median_mbps"]:.2f}MB/s')
    if s['slow']:
        parts.append('ZENODO-SLOW')
    for st in inflight()[:5]:
        parts.append(f'[{st.get("label")} {st.get("status")} {(st.get("pct") or 0):.0f}% '
                     f'{(st.get("rate_mbps") or 0):.2f}MB/s a{st.get("attempt", 0)}]')
    return ' '.join(parts)
