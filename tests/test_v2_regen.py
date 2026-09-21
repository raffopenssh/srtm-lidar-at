"""v2_regen + keep-old-product-until-replaced semantics.

Run: python3 -m pytest tests/test_v2_regen.py -q

* zenodo_client._replace_in_bucket — never DELETE before a successful PUT
* austria_processor._record_v2_upgrade_failed(transient=True) — deferrals
  don't strike until V2_UPGRADE_MAX_DEFERS
* peer_director v2_regen state helpers (candidate merge, summary)
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zenodo_client as zcl  # noqa: E402


def _client():
    c = zcl.Client.__new__(zcl.Client)
    return c


def test_replace_same_filename_is_single_put():
    c = _client()
    calls = []
    c._do_request = lambda m, u, **kw: calls.append((m, u))
    c._replace_in_bucket("https://b/x", "a.gpkg", "a.gpkg", lambda: calls.append(("PUT", "a.gpkg")))
    assert calls == [("PUT", "a.gpkg")]


def test_replace_different_filename_puts_then_deletes():
    c = _client()
    calls = []
    c._do_request = lambda m, u, **kw: calls.append((m, u))
    c._replace_in_bucket("https://b/x", "old.gpkg", "new.gpkg", lambda: calls.append(("PUT", "new.gpkg")))
    assert calls == [("PUT", "new.gpkg"), ("DELETE", "https://b/x/old.gpkg")]


def test_replace_failed_put_never_deletes():
    c = _client()
    calls = []
    c._do_request = lambda m, u, **kw: calls.append((m, u))

    def _put():
        raise zcl.ZenodoError("504", status_code=504) if _accepts_kw() else zcl.ZenodoError("504")
    try:
        c._replace_in_bucket("https://b/x", "old.gpkg", "new.gpkg", _put)
    except zcl.ZenodoError:
        pass
    assert all(m != "DELETE" for m, _ in calls)


def _accepts_kw():
    import inspect
    return "status_code" in inspect.signature(zcl.ZenodoError.__init__).parameters


def test_transient_defer_does_not_strike(tmp_path, monkeypatch):
    import austria_processor as ap
    monkeypatch.setattr(ap, "V2_UPGRADE_FAILED_FILE", tmp_path / "f.json")
    for _ in range(ap.V2_UPGRADE_MAX_DEFERS - 1):
        n = ap._record_v2_upgrade_failed("123", "504", transient=True)
        assert n == 0
    assert ap._record_v2_upgrade_failed("123", "504", transient=True) >= ap.V2_UPGRADE_MAX_STRIKES
    assert ap._record_v2_upgrade_failed("124", "verify FAIL") == 1
    assert ap._record_v2_upgrade_failed("125", "gone", fatal=True) == ap.V2_UPGRADE_MAX_STRIKES


def test_regen_candidates_and_summary(tmp_path, monkeypatch):
    import peer_director as pd
    monkeypatch.setattr(pd, "V2_REGEN_FILE", tmp_path / "v2_regen.json")
    pd.v2_regen_add_candidates("at1", {"18127": "t", "bad code": "t"})
    pd.v2_regen_add_candidates("at2", {"18127": "t", "80110-southeast-9": "t"})
    d = pd._v2_regen_load()
    assert set(d) == {"18127", "80110-southeast-9"}
    assert d["18127"]["peers"] == ["at1", "at2"] and d["18127"]["state"] == "candidate"
    s = pd.v2_regen_summary()
    assert s["counts"] == {"candidate": 2} and len(s["rows"]) == 2


if __name__ == "__main__":
    import inspect as _i
    class _MP:
        def setattr(self, obj, name, val):
            setattr(obj, name, val)
    mp = _MP()
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            kw = {}
            params = _i.signature(fn).parameters
            if "tmp_path" in params:
                kw["tmp_path"] = Path(tempfile.mkdtemp())
            if "monkeypatch" in params:
                kw["monkeypatch"] = mp
            try:
                fn(**kw); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
