"""v2_ingest — primary-side ingest of fleet ``_json_v2`` products.

Runs as a daemon thread inside ``app.py`` (primary only, every
``TICK_S``).  Per tick:

1. ``pending()`` — manifest ``<code>_json_v2`` entries (committed: size>0,
   uploaded_at) whose code is NOT in ``kg_v2_store`` and has < ``MAX_FAILS``
   failed ingest attempts (``meta.ingest_failed``).
2. For up to ``max_n`` codes: download blob → ``v2_verify.verify_for_ingest``
   (v1 doc from ``kg_docs`` = file or older store copy; flat index row as
   fallback) → ``kg_v2_store.put`` FIRST (the search-index selector reads
   the store) → ``search_index.update_kg`` → delete ``json/<code>.json`` +
   ``add_freed`` → ``quality_flags.scan_doc``.
3. ``kg_log_harvest.harvest_live()`` + ``harvest_archive_step(1)``.

Failures are recorded in ``meta`` key ``ingest_failed`` as
``{code: {"n": int, "reason": str, "ts": str}}``; a *newer* upload of the
same code (different ``uploaded_at``) resets the strike.

Test against a scratch store: ``kg_v2_store.STORE_PATH = Path('/tmp/x.db')``
and call ``ingest_one(code, entries, delete_v1=False, update_index=False)``.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import kg_docs
import kg_v2_store as store
import v2_verify

log = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/austria_processor/zenodo_manifest.json")
TICK_S = 300
MAX_PER_TICK = 20
MAX_FAILS = 3
DOWNLOAD_TIMEOUT = 120

_lock = threading.Lock()
_last_tick: dict = {}


def load_manifest_entries(path: Path = MANIFEST_PATH) -> dict:
    try:
        md = json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return {}
    return md.get("entries", md) or {}


def _committed(e) -> bool:
    return isinstance(e, dict) and int(e.get("size") or 0) > 0 and bool(e.get("uploaded_at"))


def failed() -> dict:
    try:
        return json.loads(store.meta_get("ingest_failed", "{}") or "{}")
    except Exception:  # noqa: BLE001
        return {}


def _record_failure(code: str, reason: str, uploaded_at: str) -> None:
    f = failed()
    cur = f.get(code) or {}
    n = int(cur.get("n", 0)) + 1 if cur.get("uploaded_at") == uploaded_at else 1
    f[code] = {"n": n, "reason": reason[:300], "uploaded_at": uploaded_at,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    store.meta_set("ingest_failed", json.dumps(f, separators=(",", ":")))


def _clear_failure(code: str) -> None:
    f = failed()
    if code in f:
        f.pop(code)
        store.meta_set("ingest_failed", json.dumps(f, separators=(",", ":")))


def pending(entries: dict | None = None) -> list[str]:
    """Codes with a committed ``_json_v2`` not yet in the store (oldest
    upload first), excluding codes with >= MAX_FAILS strikes on the same
    upload."""
    entries = entries if entries is not None else load_manifest_entries()
    have = store.all_codes()
    fl = failed()
    out = []
    for k, e in entries.items():
        if not k.endswith("_json_v2") or not _committed(e):
            continue
        code = k[:-len("_json_v2")]
        if code in have:
            # re-ingest only if the manifest upload is newer than the stored one
            m = store.get_meta(code) or {}
            if (e.get("uploaded_at") or "") <= (m.get("uploaded_at") or ""):
                continue
        st = fl.get(code)
        if st and st.get("uploaded_at") == e.get("uploaded_at") and int(st.get("n", 0)) >= MAX_FAILS:
            continue
        out.append((e.get("uploaded_at") or "", code))
    out.sort()
    return [c for _, c in out]


def _link(entry: dict) -> tuple[str, dict]:
    from zenodo_client import DEFAULT_TOKEN
    hdr = {}
    link = entry.get("link", "") or ""
    if not link and entry.get("bucket_url") and entry.get("filename"):
        link = f"{entry['bucket_url']}/{entry['filename']}"
        hdr = {"Authorization": f"Bearer {DEFAULT_TOKEN}"}
    if not link and entry.get("depo_id") and entry.get("filename"):
        link = (f"https://zenodo.org/api/records/{entry['depo_id']}/draft/files/"
                f"{entry['filename']}/content?access_token={DEFAULT_TOKEN}")
    return link, hdr


def download(entry: dict) -> bytes:
    import requests
    link, hdr = _link(entry)
    if not link:
        raise RuntimeError("manifest entry has no download link")
    r = requests.get(link, headers=hdr, timeout=DOWNLOAD_TIMEOUT)
    r.raise_for_status()
    blob = r.content
    want = int(entry.get("size") or 0)
    if want and len(blob) != want:
        raise RuntimeError(f"size mismatch: got {len(blob)} want {want}")
    ck = entry.get("checksum") or ""
    if ck.startswith("md5:"):
        import hashlib
        got = hashlib.md5(blob).hexdigest()
        if got != ck[4:]:
            raise RuntimeError(f"md5 mismatch: {got} != {ck[4:]}")
    return blob


def _v1_row(code: str) -> dict | None:
    try:
        import search_index as si
        from kg_splitter import parent_kg_code
        if parent_kg_code(code) != code:
            # The index row is the *parent* aggregate (sum over sibling
            # blocks).  Comparing a single block's parcel/segment count
            # against it is meaningless — 45306-south struck out on
            # `parcels_vs_v1_row: v2=1906 v1=975` (the row then held only
            # the -southwest sibling).  Blocks have no v1 baseline.
            return None
        r = si.get_index()._conn().execute(
            "SELECT parcel_count, n_segments FROM kg WHERE kg_code=?",
            (parent_kg_code(code),)).fetchone()
        return {"n_parcels": r[0], "n_segments": r[1]} if r else None
    except Exception:  # noqa: BLE001
        return None


def ingest_one(code: str, entries: dict, *, delete_v1: bool = True,
               update_index: bool = True, scan_flags: bool = True,
               json_dir: Path | None = None) -> dict:
    """Ingest one code.  Returns ``{code, ok, reason|size, freed}``."""
    e = entries.get(f"{code}_json_v2")
    if not _committed(e):
        return {"code": code, "ok": False, "reason": "no committed _json_v2 entry"}
    up_at = e.get("uploaded_at") or ""
    t0 = time.time()
    try:
        blob = download(e)
    except Exception as ex:  # noqa: BLE001
        _record_failure(code, f"download: {ex}", up_at)
        return {"code": code, "ok": False, "reason": f"download: {ex}"}
    # v1 baseline: the file (or an older store copy); flat index row fallback
    v1_doc = kg_docs.load(code, json_dir)
    rep, doc = v2_verify.verify_for_ingest(code, blob, v1_doc, entries,
                                           v1_row=None if v1_doc else _v1_row(code))
    if not rep["ok"] or doc is None:
        _record_failure(code, rep.summary(), up_at)
        log.warning("v2_ingest: %s verify FAILED: %s", code, rep.summary())
        return {"code": code, "ok": False, "reason": rep.summary()}
    # 1) store FIRST — the index selector reads the store
    info = store.put(code, blob, uploaded_at=up_at, doc=doc, source="zenodo")
    # 2) index row
    if update_index:
        try:
            import search_index as si
            si.get_index().update_kg(code, manifest=entries)
        except Exception as ex:  # noqa: BLE001
            log.warning("v2_ingest: %s update_kg: %s", code, ex)
    # 3) drop the v1 file, account the saving
    freed = 0
    if delete_v1:
        p = kg_docs.file_path(code, json_dir)
        try:
            if p.exists():
                freed = p.stat().st_size
                p.unlink()
                store.add_freed(code, freed)
        except Exception as ex:  # noqa: BLE001
            log.warning("v2_ingest: %s delete v1 file: %s", code, ex)
    # 4) quality flags from the in-memory doc
    if scan_flags:
        try:
            import quality_flags
            quality_flags.scan_doc(doc, code)
        except Exception as ex:  # noqa: BLE001
            log.warning("v2_ingest: %s scan_doc: %s", code, ex)
    _clear_failure(code)
    log.info("v2_ingest: %s ingested (%d B gz, v1 freed %d B, %s, %.1fs)",
             code, info["size"], freed, rep.summary().split(" |")[0], time.time() - t0)
    return {"code": code, "ok": True, "size": info["size"], "freed": freed,
            "warnings": rep["warnings"]}


def run_tick(max_n: int = MAX_PER_TICK, *, harvest: bool = True) -> dict:
    """One ingest pass.  Safe to call from any thread; single-flight."""
    if not _lock.acquire(blocking=False):
        return {"skipped": "busy"}
    t0 = time.time()
    out = {"ingested": 0, "failed": 0, "pending": 0, "codes": [], "freed": 0}
    try:
        entries = load_manifest_entries()
        todo = pending(entries)
        out["pending"] = len(todo)
        for code in todo[:max_n]:
            try:
                r = ingest_one(code, entries)
            except Exception as ex:  # noqa: BLE001
                log.warning("v2_ingest: %s crashed: %s", code, ex)
                _record_failure(code, f"crash: {ex}",
                                (entries.get(f"{code}_json_v2") or {}).get("uploaded_at") or "")
                r = {"ok": False}
            if r.get("ok"):
                out["ingested"] += 1
                out["freed"] += int(r.get("freed") or 0)
                out["codes"].append(code)
            else:
                out["failed"] += 1
        out["pending"] = max(0, len(todo) - out["ingested"] - out["failed"])
        if harvest:
            try:
                import kg_log_harvest
                out["log_live"] = kg_log_harvest.harvest_live()
                out["log_archive"] = kg_log_harvest.harvest_archive_step(max_days=1)
            except Exception as ex:  # noqa: BLE001
                log.warning("v2_ingest: log harvest: %s", ex)
    finally:
        out["dt"] = round(time.time() - t0, 1)
        out["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _last_tick.clear()
        _last_tick.update(out)
        _lock.release()
    if out["ingested"] or out["failed"]:
        log.info("v2_ingest tick: %s", {k: v for k, v in out.items() if k != "codes"})
    return out


def last_tick() -> dict:
    return dict(_last_tick)


def _json_dir_stats(json_dir: Path = kg_docs.JSON_DIR) -> tuple[int, int]:
    n = b = 0
    try:
        for p in Path(json_dir).glob("*.json"):
            try:
                b += p.stat().st_size
                n += 1
            except OSError:
                pass
    except Exception:  # noqa: BLE001
        pass
    return n, b


_stats_cache: dict = {"t": 0.0, "v": None}


def status(total_kgs: int = 8440, ttl: float = 60.0) -> dict:
    """Structured block for ``/api/v1/director/status.v2`` and the
    ``v2:`` line in ``/process.txt``.  TTL-cached (manifest + json dir
    scans are not free)."""
    now = time.time()
    if _stats_cache["v"] is not None and now - _stats_cache["t"] < ttl:
        return _stats_cache["v"]
    entries = load_manifest_entries()
    day_ago = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now - 86400))
    upgraded = set()
    up_24h = 0
    first_up = ""
    fresh_v2 = 0
    for k, e in entries.items():
        if not _committed(e):
            continue
        if k.endswith("_json_v2"):
            code = k[:-8]
            upgraded.add(code.split("-", 1)[0])
            ts = e.get("uploaded_at") or ""
            if ts > day_ago:
                up_24h += 1
            if ts and (not first_up or ts < first_up):
                first_up = ts
        elif k.endswith("_json") and (e.get("version") == "v2"):
            fresh_v2 += 1
    rate_h = up_24h / 24.0
    remaining = max(0, total_kgs - len(upgraded))
    eta_d = round(remaining / rate_h / 24.0, 1) if rate_h > 0 else None
    st = store.stats()
    ls = store.log_stats()
    fl = failed()
    n_files, b_files = _json_dir_stats()
    try:
        import kg_log_harvest
        done = set((store.meta_get("log_harvest_archive_done", "") or "").split(",")) - {""}
        arch_remaining = sum(1 for p in kg_log_harvest.ARCHIVE_DIR.glob("*.jsonl.gz")
                             if p.name[:10] not in done) if kg_log_harvest.ARCHIVE_DIR.is_dir() else 0
    except Exception:  # noqa: BLE001
        arch_remaining = None
    try:
        pend = len(pending(entries))
    except Exception:  # noqa: BLE001
        pend = None
    v = {
        "upgraded_parents": len(upgraded), "total": total_kgs,
        "upgraded_pct": round(100.0 * len(upgraded) / max(total_kgs, 1), 2),
        "upgraded_24h": up_24h, "rate_per_h": round(rate_h, 2), "eta_days": eta_d,
        "first_upload": first_up or None,
        "fresh_v2": fresh_v2,
        "verify_fail": len(fl), "verify_fail_codes": sorted(fl)[:20],
        "pending": pend,
        "store": st, "kg_log": ls, "archive_days_remaining": arch_remaining,
        "json_files": n_files, "json_bytes": b_files,
        "last_tick": last_tick(),
    }
    _stats_cache.update(t=now, v=v)
    return v


def _fmt_b(n) -> str:
    n = float(n or 0)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f}{u}" if u in ("B", "KB") else f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def text_line(v: dict | None = None) -> str:
    """One-line summary for ``/process.txt``."""
    v = v or status()
    st = v["store"]
    n = st.get("codes") or 0
    avg = (st.get("v1_bytes_freed") or 0) / n if n else 0
    eta = f"{v['eta_days']}d" if v.get("eta_days") is not None else "?"
    ls = v["kg_log"]
    parts = [
        f"v2: upgraded={v['upgraded_parents']}/{v['total']} ({v['upgraded_pct']}%) "
        f"+{v['upgraded_24h']}/24h @{v['rate_per_h']}/h eta={eta}",
        f"fresh_v2={v['fresh_v2']}",
        f"verify_fail={v['verify_fail']}" + ("" if v["pending"] is None else f" pending={v['pending']}"),
        f"store={n}codes/{_fmt_b(st.get('db_bytes'))} freed={_fmt_b(st.get('v1_bytes_freed'))} "
        f"avg/kg={_fmt_b(avg)}",
        f"json_files={v['json_files']} disk={_fmt_b(v['json_bytes'])}",
        f"kg_log={ls.get('codes', 0)}codes/{ls.get('rows', 0)}rows/{_fmt_b(ls.get('bytes'))}"
        + (f" archive_remaining={v['archive_days_remaining']}d"
           if v.get("archive_days_remaining") is not None else ""),
    ]
    return " · ".join(parts)


def loop(should_run, tick_s: int = TICK_S) -> None:
    """Daemon-thread body.  ``should_run()`` is polled every tick (primary
    check lives in app.py so this module stays import-safe)."""
    time.sleep(90)   # let srv settle after (re)start
    while True:
        try:
            if should_run():
                run_tick()
        except Exception as ex:  # noqa: BLE001
            log.warning("v2_ingest loop: %s", ex)
        time.sleep(tick_s)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(text_line())
        print(json.dumps(status(), indent=1, default=str))
    elif len(sys.argv) > 1:
        ents = load_manifest_entries()
        for c in sys.argv[1:]:
            print(ingest_one(c, ents, delete_v1=False, update_index=False, scan_flags=False))
    else:
        print(run_tick())
