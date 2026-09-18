"""kg_v2_store — primary-side store for v2 KG documents (gzip'd kgjson/2 blobs).

Why a separate SQLite file and not a table in ``search_index.db``: the
search index is *derived* and gets dropped + rebuilt (``SearchIndex.build``
DROPs every table).  The v2 blobs are *source data* on the primary — the
only local copy of the per-KG document once the v1 ``json/<code>.json``
file has been deleted.  They must survive index rebuilds, so they live in
``data/kg_v2_store.db`` (WAL, one row per product code incl. block codes).

Every blob is verified (``v2_verify.verify_for_ingest``) before ``put`` —
this module does not re-verify, it just persists + serves.

Schema
------
kg_v2(code TEXT PK, parent TEXT, blob BLOB, size INT, sha256 TEXT,
      version TEXT, model TEXT, uploaded_at TEXT, ingested_at TEXT,
      v1_bytes_freed INT, n_parcels INT, source TEXT)
meta(key TEXT PK, value TEXT)

``v1_bytes_freed`` records the on-disk size of the v1 JSON file(s) deleted
when this code was ingested — the per-KG disk saving surfaced on
``/process.txt`` (``v2:`` line).
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable

import kg_json_v2

log = logging.getLogger(__name__)

STORE_PATH = Path("data/kg_v2_store.db")

_lock = threading.RLock()
_conn_cache: dict[int, sqlite3.Connection] = {}

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS kg_v2 (
        code TEXT PRIMARY KEY,
        parent TEXT NOT NULL,
        blob BLOB NOT NULL,
        size INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        version TEXT,
        model TEXT,
        uploaded_at TEXT,
        ingested_at TEXT,
        v1_bytes_freed INTEGER DEFAULT 0,
        n_parcels INTEGER,
        source TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS kg_v2_parent ON kg_v2(parent)",
    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)",
    # Per-code processing history: a bounded ring of merged-log lines that
    # mention the code (starts, tile steps, warnings, uploads, upgrades,
    # verify verdicts) harvested from data/combined_log_24h.jsonl + the
    # per-day archive by kg_log_harvest.  Flattened rows [ts, peer, level,
    # msg] as one gzip'd JSON array per code (≈40 B/row), newest last.
    """CREATE TABLE IF NOT EXISTS kg_log (
        code TEXT PRIMARY KEY,
        n INTEGER NOT NULL,
        first_ts TEXT,
        last_ts TEXT,
        blob BLOB NOT NULL
    )""",
]
LOG_RING_MAX = 600          # rows kept per code (oldest dropped)
# Added columns (ALTER on open, idempotent).  ``bbox`` = "min_lon,min_lat,
# max_lon,max_lat" so the coverage oracle never has to decode a blob.
_MIGRATIONS = [
    ("bbox", "ALTER TABLE kg_v2 ADD COLUMN bbox TEXT"),
    ("generated_at", "ALTER TABLE kg_v2 ADD COLUMN generated_at TEXT"),
]
_DOC_CACHE_MAX = 24
_doc_cache: dict[str, tuple[str, dict]] = {}   # code -> (sha256, decoded doc)


def _conn(path: Path | None = None) -> sqlite3.Connection:
    p = Path(path or STORE_PATH)
    key = threading.get_ident()
    c = _conn_cache.get(key)
    if c is not None:
        return c
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(p), timeout=30, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    for s in _SCHEMA:
        c.execute(s)
    cols = {r[1] for r in c.execute("PRAGMA table_info(kg_v2)")}
    for col, stmt in _MIGRATIONS:
        if col not in cols:
            c.execute(stmt)
    c.commit()
    _conn_cache[key] = c
    return c


def parent_of(code: str) -> str:
    return code.split("-", 1)[0] if "-" in code else code


def has(code: str) -> bool:
    if not STORE_PATH.exists():
        return False
    r = _conn().execute("SELECT 1 FROM kg_v2 WHERE code=?", (code,)).fetchone()
    return r is not None


def codes_for_parent(parent: str) -> list[str]:
    """All product codes (plain and/or blocks) stored for a parent KG."""
    if not STORE_PATH.exists():
        return []
    return [r[0] for r in _conn().execute(
        "SELECT code FROM kg_v2 WHERE parent=? ORDER BY code", (parent,))]


def all_codes() -> set[str]:
    if not STORE_PATH.exists():
        return set()
    return {r[0] for r in _conn().execute("SELECT code FROM kg_v2")}


def all_parents() -> set[str]:
    if not STORE_PATH.exists():
        return set()
    return {r[0] for r in _conn().execute("SELECT DISTINCT parent FROM kg_v2")}


def get_blob(code: str) -> bytes | None:
    if not STORE_PATH.exists():
        return None
    r = _conn().execute("SELECT blob FROM kg_v2 WHERE code=?", (code,)).fetchone()
    return bytes(r[0]) if r else None


def get(code: str) -> dict | None:
    """Decoded v1-shape document for *code* (plain or block), or None.
    Small LRU of decoded docs (a decode is ~50 ms / few MB); entries are
    keyed by sha256 so a re-ingest invalidates automatically.  Callers
    MUST NOT mutate the returned dict (use ``copy.deepcopy`` if needed)."""
    if not STORE_PATH.exists():
        return None
    r = _conn().execute("SELECT sha256, blob FROM kg_v2 WHERE code=?", (code,)).fetchone()
    if not r:
        return None
    sha = r[0]
    with _lock:
        hit = _doc_cache.get(code)
        if hit and hit[0] == sha:
            _doc_cache[code] = _doc_cache.pop(code)   # LRU touch
            return hit[1]
    try:
        d = kg_json_v2.decode(bytes(r[1]))
    except Exception as e:  # noqa: BLE001
        log.error("kg_v2_store: blob for %s undecodable: %s", code, e)
        return None
    d.setdefault("kg_code", code)
    with _lock:
        _doc_cache[code] = (sha, d)
        while len(_doc_cache) > _DOC_CACHE_MAX:
            _doc_cache.pop(next(iter(_doc_cache)))
    return d


def get_bbox(code: str) -> tuple | None:
    """(min_lon, min_lat, max_lon, max_lat) from the indexed column."""
    if not STORE_PATH.exists():
        return None
    r = _conn().execute("SELECT bbox FROM kg_v2 WHERE code=?", (code,)).fetchone()
    if not r or not r[0]:
        return None
    try:
        v = [float(x) for x in r[0].split(",")]
        return tuple(v) if len(v) == 4 else None
    except Exception:
        return None


def timestamp(code: str) -> str:
    """Freshness stamp for the selector: uploaded_at, else generated_at."""
    if not STORE_PATH.exists():
        return ""
    r = _conn().execute("SELECT uploaded_at, generated_at, ingested_at FROM kg_v2 WHERE code=?",
                        (code,)).fetchone()
    if not r:
        return ""
    return r[0] or r[1] or r[2] or ""


def get_meta(code: str) -> dict | None:
    if not STORE_PATH.exists():
        return None
    c = _conn()
    c.row_factory = sqlite3.Row
    r = c.execute("SELECT code,parent,size,sha256,version,model,uploaded_at,"
                  "ingested_at,v1_bytes_freed,n_parcels,source,bbox,generated_at "
                  "FROM kg_v2 WHERE code=?",
                  (code,)).fetchone()
    c.row_factory = None
    return dict(r) if r else None


def put(code: str, blob: bytes, *, uploaded_at: str = "", model: str = "",
        version: str = "v2", n_parcels: int | None = None,
        v1_bytes_freed: int = 0, source: str = "zenodo", doc: dict | None = None) -> dict:
    """Persist a verified blob.  Idempotent (REPLACE); keeps the larger of
    the stored / supplied ``v1_bytes_freed`` so a re-ingest never zeroes
    the saving already accounted for."""
    if blob[:2] != b"\x1f\x8b":
        raise ValueError("blob is not gzip (kgjson/2 containers are gzip'd)")
    sha = hashlib.sha256(blob).hexdigest()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bbox_s, gen_at = None, None
    if doc is None:
        try:
            doc = kg_json_v2.decode(blob)
        except Exception:
            doc = None
    if isinstance(doc, dict):
        bb = doc.get("bbox") or {}
        try:
            bbox_s = ",".join(repr(float(bb[k])) for k in ("min_lon", "min_lat", "max_lon", "max_lat"))
        except Exception:
            bbox_s = None
        gen_at = doc.get("generated_at") or None
        if n_parcels is None:
            try:
                n_parcels = int((doc.get("parcels") or {}).get("count") or 0)
            except Exception:
                pass
        if not model:
            model = str(((doc.get("model") or {}).get("classifier")) or "")
    with _lock:
        c = _conn()
        prev = c.execute("SELECT v1_bytes_freed FROM kg_v2 WHERE code=?", (code,)).fetchone()
        freed = max(int(v1_bytes_freed or 0), int(prev[0]) if prev else 0)
        c.execute(
            "INSERT OR REPLACE INTO kg_v2(code,parent,blob,size,sha256,version,model,"
            "uploaded_at,ingested_at,v1_bytes_freed,n_parcels,source,bbox,generated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (code, parent_of(code), sqlite3.Binary(blob), len(blob), sha, version, model,
             uploaded_at, now, freed, n_parcels, source, bbox_s, gen_at))
        c.commit()
        _doc_cache.pop(code, None)
    return {"code": code, "size": len(blob), "sha256": sha, "v1_bytes_freed": freed}


def add_freed(code: str, nbytes: int) -> None:
    with _lock:
        c = _conn()
        c.execute("UPDATE kg_v2 SET v1_bytes_freed = v1_bytes_freed + ? WHERE code=?",
                  (int(nbytes), code))
        c.commit()


def delete(code: str) -> bool:
    with _lock:
        c = _conn()
        n = c.execute("DELETE FROM kg_v2 WHERE code=?", (code,)).rowcount
        c.commit()
        _doc_cache.pop(code, None)
    return n > 0


def stats() -> dict:
    """Aggregate for ``/process.txt`` and ``/api/v1/director/status``."""
    if not STORE_PATH.exists():
        return {"codes": 0, "parents": 0, "bytes": 0, "v1_bytes_freed": 0,
                "db_bytes": 0, "last_ingested_at": None, "ingested_24h": 0}
    c = _conn()
    r = c.execute("SELECT COUNT(*), COUNT(DISTINCT parent), COALESCE(SUM(size),0), "
                  "COALESCE(SUM(v1_bytes_freed),0), MAX(ingested_at) FROM kg_v2").fetchone()
    day_ago = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 86400))
    n24 = c.execute("SELECT COUNT(*) FROM kg_v2 WHERE ingested_at > ?", (day_ago,)).fetchone()[0]
    db_bytes = 0
    for suf in ("", "-wal"):
        p = Path(str(STORE_PATH) + suf)
        if p.exists():
            db_bytes += p.stat().st_size
    return {"codes": r[0], "parents": r[1], "bytes": r[2], "v1_bytes_freed": r[3],
            "db_bytes": db_bytes, "last_ingested_at": r[4], "ingested_24h": n24}


def recent(limit: int = 10) -> list[dict]:
    if not STORE_PATH.exists():
        return []
    return [{"code": a, "size": b, "v1_bytes_freed": d, "ingested_at": e, "model": f}
            for a, b, d, e, f in _conn().execute(
                "SELECT code,size,v1_bytes_freed,ingested_at,model FROM kg_v2 "
                "ORDER BY ingested_at DESC LIMIT ?", (int(limit),))]


def vacuum_if_needed(min_free_ratio: float = 0.25) -> bool:
    """VACUUM when freelist > 25 %% of pages (after bulk deletes)."""
    c = _conn()
    pages = c.execute("PRAGMA page_count").fetchone()[0]
    free = c.execute("PRAGMA freelist_count").fetchone()[0]
    if pages and free / pages > min_free_ratio:
        with _lock:
            c.execute("VACUUM")
        return True
    return False


# --------------------------------------------------------------------------
# per-KG log ring
# --------------------------------------------------------------------------

def _log_rows(code: str, c=None) -> list:
    c = c or _conn()
    r = c.execute("SELECT blob FROM kg_log WHERE code=?", (code,)).fetchone()
    if not r:
        return []
    import gzip
    try:
        return json.loads(gzip.decompress(bytes(r[0])).decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("kg_v2_store: kg_log %s undecodable: %s", code, e)
        return []


def append_log(code: str, rows: Iterable, *, cap: int = LOG_RING_MAX) -> int:
    """Merge ``rows`` (``[ts, peer, level, msg]`` or dicts with those keys)
    into the code's ring.  Dedups on (ts, peer, msg), sorts by ts, trims to
    ``cap`` newest.  Returns the number of rows now stored."""
    import gzip
    new = []
    for r in rows:
        if isinstance(r, dict):
            r = [r.get("ts", ""), r.get("peer", ""), r.get("level", ""), r.get("msg", "")]
        if not r or not r[0]:
            continue
        new.append([str(r[0])[:32], str(r[1] or ""), str(r[2] or "")[:1], str(r[3] or "")[:400]])
    if not new:
        return 0
    with _lock:
        c = _conn()
        cur = _log_rows(code, c)
        seen = {(x[0], x[1], x[3]) for x in cur}
        for r in new:
            k = (r[0], r[1], r[3])
            if k not in seen:
                seen.add(k)
                cur.append(r)
        cur.sort(key=lambda x: x[0])
        if len(cur) > cap:
            cur = cur[-cap:]
        blob = gzip.compress(json.dumps(cur, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
                             compresslevel=6, mtime=0)
        c.execute("INSERT OR REPLACE INTO kg_log(code,n,first_ts,last_ts,blob) VALUES(?,?,?,?,?)",
                  (code, len(cur), cur[0][0] if cur else None, cur[-1][0] if cur else None,
                   sqlite3.Binary(blob)))
        c.commit()
    return len(cur)


def get_log(code: str, *, include_parent: bool = True, limit: int | None = None) -> list[dict]:
    """History rows for ``code`` as dicts, newest last.  For a plain parent
    code the rings of its split blocks are merged in (each row tagged with
    its ``code``)."""
    if not STORE_PATH.exists():
        return []
    c = _conn()
    codes = [code]
    if include_parent and "-" not in code:
        codes += [r[0] for r in c.execute("SELECT code FROM kg_log WHERE code LIKE ? AND code != ?",
                                          (code + "-%", code))]
    out = []
    for cc in codes:
        for ts, peer, lvl, msg in _log_rows(cc, c):
            out.append({"ts": ts, "peer": peer, "level": {"i": "info", "w": "warning", "e": "error",
                                                          "s": "success"}.get(lvl, lvl),
                        "msg": msg, "code": cc})
    out.sort(key=lambda r: r["ts"])
    if limit:
        out = out[-int(limit):]
    return out


def log_stats() -> dict:
    if not STORE_PATH.exists():
        return {"codes": 0, "rows": 0, "bytes": 0}
    r = _conn().execute("SELECT COUNT(*), COALESCE(SUM(n),0), COALESCE(SUM(LENGTH(blob)),0), "
                        "MAX(last_ts) FROM kg_log").fetchone()
    return {"codes": r[0], "rows": r[1], "bytes": r[2], "last_ts": r[3]}


def meta_get(key: str, default=None):
    if not STORE_PATH.exists():
        return default
    r = _conn().execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return r[0] if r else default


def meta_set(key: str, value) -> None:
    with _lock:
        c = _conn()
        c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, str(value)))
        c.commit()
