"""Connection reuse for module-level ``requests.get/post/put`` calls.

``requests.get(...)`` builds a throw-away ``Session`` per call, so every
director fanout / peer push / heartbeat pays a fresh TCP + TLS handshake
(~6 KB on the wire through the exe.dev proxy). At ~10k requests/h on
the primary that is tens of MB/h of pure handshake overhead — a large
slice of the bandwidth budget once the payloads themselves are slim.

``install()`` swaps ``requests.api.request`` for a version that routes
through one process-wide pooled ``Session`` (urllib3 pools are
thread-safe; the Session carries no cookies/auth so sharing is safe).
Servers that close idle keep-alive connections are handled by urllib3
transparently (stale connection → reconnect). ``requests.Session``
used explicitly elsewhere is untouched.
"""
from __future__ import annotations

import threading

_installed = False
_lock = threading.Lock()
_session = None


def session():
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                import requests
                from requests.adapters import HTTPAdapter
                s = requests.Session()
                s.trust_env = True
                ad = HTTPAdapter(pool_connections=64, pool_maxsize=16,
                                 max_retries=0, pool_block=False)
                s.mount('https://', ad)
                s.mount('http://', ad)
                _session = s
    return _session


def install() -> None:
    global _installed
    if _installed:
        return
    try:
        import requests
        import requests.api as _api
    except Exception:
        return

    def request(method, url, **kwargs):
        return session().request(method=method, url=url, **kwargs)

    _api.request = request
    # requests.get/post/... call ``request`` by name inside requests.api,
    # so rebinding ``_api.request`` above already routes them. Re-export
    # for callers that imported the helpers before install() ran.
    def get(url, params=None, **kw):
        kw.setdefault('allow_redirects', True)
        return request('get', url, params=params, **kw)

    def head(url, **kw):
        kw.setdefault('allow_redirects', False)
        return request('head', url, **kw)

    def post(url, data=None, json=None, **kw):
        return request('post', url, data=data, json=json, **kw)

    def put(url, data=None, **kw):
        return request('put', url, data=data, **kw)

    def patch(url, data=None, **kw):
        return request('patch', url, data=data, **kw)

    def delete(url, **kw):
        return request('delete', url, **kw)

    for fn in (get, head, post, put, patch, delete):
        setattr(_api, fn.__name__, fn)
        setattr(requests, fn.__name__, fn)
    _installed = True
