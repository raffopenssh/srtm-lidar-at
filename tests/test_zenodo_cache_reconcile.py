"""Unit tests for zenodo_cache upload-failure semantics + reconcile.

Run: python3 -m pytest tests/test_zenodo_cache_reconcile.py -q
(or: python3 tests/test_zenodo_cache_reconcile.py)

All Zenodo I/O is mocked: ``_api`` (deposition GET → bucket url +
listing), ``_session.put`` (the bucket PUT), and — for reconcile —
``_probe_zip_usable``. No network.
"""
import io
import json
import os
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests  # noqa: E402

import zenodo_cache as zc  # noqa: E402

DEPO = 19650075
NAME = "copernicus_ndvi_cell_47.0_48.0_16.0_18.0.zip"
URL = f"https://zenodo.org/api/records/{DEPO}/draft/files/{NAME}/content"


class _Resp:
    def __init__(self, status, payload=None, reason="x"):
        self.status_code = status
        self.reason = reason
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} {self.reason}",
                                     response=self)


def _mk_cache(tmp: Path, files: dict):
    # Isolate from the real cache_manifest.json (its live circuit flag
    # would otherwise steer _handle_upload_failure into the degraded branch).
    zc.zenodo_degraded = lambda *a, **k: False
    mpath = tmp / "cache_manifest.json"
    mpath.write_text(json.dumps({"depo_id": DEPO, "record_id": None,
                                 "files": files}))
    cache = zc.ZenodoCache.__new__(zc.ZenodoCache)
    cache.token = "t"
    cache.base_url = "https://zenodo.org"
    cache.manifest = zc.CacheManifest(mpath)
    cache._session = mock.Mock()
    cache._reset_session = lambda: None  # keep the mocked session
    cache._zip_indices = {}
    cache._missing_zips = set()
    cache._missing_zips_mtime = 0.0
    return cache, mpath


def _local_zip(tmp: Path, nbytes=1000) -> Path:
    p = tmp / NAME
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("ndvi_47.3000_16.0000_47.4000_16.1000_2024.npz",
                    b"x" * nbytes)
    return p


def _api_factory(listing_files):
    """Mock ``_api``: every GET on the deposition returns bucket + files."""
    def _api(method, path, **kw):
        assert method == "GET"
        return _Resp(200, {"links": {"bucket": "https://zenodo.org/api/files/B"},
                           "files": [
                               {"filename": n, "filesize": s,
                                "checksum": f"md5:{n[:6]}", "id": "i",
                                "links": {"download": "d"}}
                               for n, s in listing_files.items()]})
    return _api


def _live_entry(size=5000, tiles=3):
    return {"url": URL, "size": size, "checksum": "old", "tile_count": tiles,
            "updated_at": "2026-09-20T10:00:00+00:00"}


def _run_upload(cache, tmp, put_side_effect, listing):
    zp = _local_zip(tmp)
    cache._api = _api_factory(listing(zp.stat().st_size))
    cache._session.put = mock.Mock(side_effect=put_side_effect)
    with mock.patch.object(zc.time, "sleep", lambda *_a, **_k: None):
        try:
            return cache._upload_file(DEPO, zp, NAME), None
        except zc._UploadFailed as e:
            return None, e


def test_504_after_landing_is_success():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cache, _ = _mk_cache(tmp, {NAME: _live_entry()})
        res, err = _run_upload(
            cache, tmp, lambda *a, **k: _Resp(504, reason="Gateway Time-out"),
            listing=lambda local: {NAME: local})
        assert err is None and res["verified_via"] == "listing"
        assert cache._session.put.call_count == 6  # 5xx retried, then listed


def test_5xx_old_still_present_keeps_entry():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cache, mpath = _mk_cache(tmp, {NAME: _live_entry(size=5000)})
        res, err = _run_upload(
            cache, tmp, lambda *a, **k: _Resp(502, reason="Bad Gateway"),
            listing=lambda local: {NAME: 5000})
        assert res is None and err.remote_state == "present_old"
        action = cache._handle_upload_failure(NAME, err)
        assert action == "kept"
        e = json.loads(mpath.read_text())["files"][NAME]
        assert e["size"] == 5000 and "unverified" not in e
        assert "tombstone_reason" not in e


def test_5xx_truly_gone_marks_unverified_not_tombstone():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cache, mpath = _mk_cache(tmp, {NAME: _live_entry(size=5000)})
        res, err = _run_upload(
            cache, tmp, requests.Timeout("write timed out"),
            listing=lambda local: {})
        assert res is None and err.remote_state == "absent" and not err.definitive
        action = cache._handle_upload_failure(NAME, err)
        assert action == "unverified"
        e = json.loads(mpath.read_text())["files"][NAME]
        assert e["size"] == 5000 and e["unverified"] is True
        assert e["unverified_since"] and "timed out" in e["last_error"]
        # readers still see it
        assert cache.manifest.get_file(NAME) is not None


def test_listing_unavailable_marks_unverified():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cache, mpath = _mk_cache(tmp, {NAME: _live_entry()})
        zp = _local_zip(tmp)
        calls = {"n": 0}

        def _api(method, path, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Resp(200, {"links": {"bucket": "https://b"}, "files": []})
            raise requests.ConnectionError("down")
        cache._api = _api
        cache._session.put = mock.Mock(return_value=_Resp(503))
        with mock.patch.object(zc.time, "sleep", lambda *a, **k: None):
            try:
                cache._upload_file(DEPO, zp, NAME)
                assert False
            except zc._UploadFailed as e:
                assert e.remote_state == "unknown"
                assert cache._handle_upload_failure(NAME, e) == "unverified"


def test_single_400_does_not_tombstone_second_after_10min_does():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cache, mpath = _mk_cache(tmp, {NAME: _live_entry(size=5000)})
        res, err = _run_upload(
            cache, tmp, lambda *a, **k: _Resp(400, reason="BAD REQUEST"),
            listing=lambda local: {})
        assert err.definitive and err.remote_state == "absent"
        assert cache._session.put.call_count == 1  # 4xx not retried
        assert cache._handle_upload_failure(NAME, err) == "unverified"
        e = json.loads(mpath.read_text())["files"][NAME]
        assert e["size"] == 5000 and len(e["upload_4xx_history"]) == 1
        # second 400 two minutes later → still not enough
        assert cache._handle_upload_failure(NAME, err) == "unverified"
        # backdate the first attempt by 11 min → tombstone with prev_*
        e = cache.manifest._data["files"][NAME]
        old = (datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat()
        e["upload_4xx_history"] = [old]
        assert cache._handle_upload_failure(NAME, err) == "tombstoned"
        e = json.loads(mpath.read_text())["files"][NAME]
        assert e["size"] == 0 and e["prev_size"] == 5000
        assert e["prev_tile_count"] == 3 and e["prev_checksum"] == "old"
        assert "4xx" in e["tombstone_reason"]
        assert cache.manifest.get_file(NAME) is None


def test_confirmed_404_tombstones_unconfirmed_does_not():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cache, mpath = _mk_cache(tmp, {NAME: _live_entry(size=5000)})
        # present in listing → not tombstoned
        cache._api = _api_factory({NAME: 5000})
        assert cache._confirm_404_then_tombstone(NAME, "404 on upload") == "present"
        assert cache.manifest.get_file(NAME) is not None
        # absent → tombstoned, prev_* preserved
        cache._api = _api_factory({})
        assert cache._confirm_404_then_tombstone(NAME, "404 on upload") == "tombstoned"
        e = json.loads(mpath.read_text())["files"][NAME]
        assert e["size"] == 0 and e["prev_size"] == 5000
        assert "confirmed absent" in e["tombstone_reason"]


def test_degraded_circuit_never_tombstones():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        cache, mpath = _mk_cache(tmp, {NAME: _live_entry(size=5000)})
        m = json.loads(mpath.read_text())
        m["zenodo_circuit"] = {"degraded": True, "updated_at": "x"}
        mpath.write_text(json.dumps(m))
        cache.manifest._load()
        with mock.patch.object(zc, "CACHE_MANIFEST_PATH", mpath), \
                mock.patch.object(zc, "zenodo_degraded",
                                  lambda *a, **k: True):
            cache._api = _api_factory({})
            assert cache._confirm_404_then_tombstone(NAME, "404") == "unverified"
            err = zc._UploadFailed(NAME, requests.HTTPError("400"),
                                   remote_state="absent", status=400)
            assert cache._handle_upload_failure(NAME, err) == "unverified"
        assert cache.manifest.get_file(NAME) is not None


def _tomb(reason="upload failed: 504 Server Error"):
    return {"url": URL, "size": 0, "checksum": "", "tile_count": 0,
            "updated_at": "2026-09-20T16:30:00+00:00",
            "tombstone_reason": reason, "prev_size": 2981866,
            "prev_checksum": "b6a9e7", "prev_tile_count": 1}


def test_reconcile_restores_usable_and_flags_corrupt_and_tombstones_missing():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        gone = "copernicus_harmonics_strip_48.0_48.5.zip"
        corrupt = "copernicus_sar_cell_47.0_48.0_16.0_18.0.zip"
        unv = "hansen_cell_48.0_49.0_12.0_14.0.zip"
        files = {
            NAME: _tomb(),
            corrupt: _tomb("rebuild upload failed: 400"),
            gone: _live_entry(size=228149381, tiles=19),
            unv: dict(_live_entry(size=777), unverified=True,
                      unverified_since="x", last_error="504"),
            "chkpt_01002.tar.gz": {"size": 0, "updated_at": "x"},
        }
        cache, mpath = _mk_cache(tmp, files)
        cache._api = _api_factory({NAME: 2981866, corrupt: 4096, unv: 777,
                                   "probetest.bin": 25,
                                   "chkpt_01002.tar.gz": 10})

        def _probe(name, url, size):
            if name == NAME:
                return True, "ok (1 entries, 1 tiles)", 1
            return False, "central directory unreadable: bad zip", 0
        cache._probe_zip_usable = _probe
        with mock.patch.object(zc, "_ZIP_INDEX_CACHE_DIR", tmp / "idx"), \
                mock.patch.object(zc, "DATA_DIR", tmp):
            dry = cache.reconcile_manifest(dry_run=True, take_lock=False)
            assert dry["changed"] == 0 and dry["restored"] == 1
            assert json.loads(mpath.read_text())["files"][NAME]["size"] == 0
            s = cache.reconcile_manifest(dry_run=False, take_lock=False)
        assert (s["tombstoned"], s["restored"], s["corrupt"], s["unknown"],
                s["unverified_cleared"]) == (1, 1, 1, 1, 1)
        m = json.loads(mpath.read_text())
        assert set(m) >= {"depo_id", "files"}
        e = m["files"][NAME]
        assert e["size"] == 2981866 and e["tile_count"] == 1
        assert e["checksum"] == NAME[:6] and "tombstone_reason" not in e
        assert e["restored_from_tombstone"]["reason"].startswith("upload failed: 504")
        assert e["updated_at"] > "2026-09-20T16:30:00+00:00"
        assert cache.manifest.get_file(NAME) is not None
        c = m["files"][corrupt]
        assert c["size"] == 0 and c["tombstone_reason"].startswith(
            "reconcile: present but corrupt (")
        assert c["prev_size"] == 2981866  # preserved
        g = m["files"][gone]
        assert g["size"] == 0 and g["prev_size"] == 228149381
        assert g["prev_tile_count"] == 19
        assert g["tombstone_reason"] == "reconcile: missing from deposit"
        u = m["files"][unv]
        assert "unverified" not in u and u["size"] == 777
        assert m["files"]["chkpt_01002.tar.gz"] == {"size": 0, "updated_at": "x"}
        # the merge on peers is newest-updated_at-wins → restore propagates
        assert e["updated_at"] > files[NAME]["updated_at"]


if __name__ == "__main__":
    import inspect
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            fn()
            n += 1
            print("ok", name)
    print(f"{n} tests passed")
