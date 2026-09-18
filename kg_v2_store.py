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
]


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
    """Decoded v1-shape document for *code* (plain or block), or None."""
    b = get_blob(code)
    if b is None:
        return None
    try:
        d = kg_json_v2.decode(b)
    except Exception as e:  # noqa: BLE001
        log.error("kg_v2_store: blob for %s undecodable: %s", code, e)
        return None
    d.setdefault("kg_code", code)
    return d


def get_meta(code: str) -> dict | None:
    if not STORE_PATH.exists():
        return None
    c = _conn()
    c.row_factory = sqlite3.Row
    r = c.execute("SELECT code,parent,size,sha256,version,model,uploaded_at,"
                  "ingested_at,v1_bytes_freed,n_parcels,source FROM kg_v2 WHERE code=?",
                  (code,)).fetchone()
    c.row_factory = None
    return dict(r) if r else None


def put(code: str, blob: bytes, *, uploaded_at: str = "", model: str = "",
        version: str = "v2", n_parcels: int | None = None,
        v1_bytes_freed: int = 0, source: str = "zenodo") -> dict:
    """Persist a verified blob.  Idempotent (REPLACE); keeps the larger of
    the stored / supplied ``v1_bytes_freed`` so a re-ingest never zeroes
    the saving already accounted for."""
    if blob[:2] != b"\x1f\x8b":
        raise ValueError("blob is not gzip (kgjson/2 containers are gzip'd)")
    sha = hashlib.sha256(blob).hexdigest()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _lock:
        c = _conn()
        prev = c.execute("SELECT v1_bytes_freed FROM kg_v2 WHERE code=?", (code,)).fetchone()
        freed = max(int(v1_bytes_freed or 0), int(prev[0]) if prev else 0)
        c.execute(
            "INSERT OR REPLACE INTO kg_v2(code,parent,blob,size,sha256,version,model,"
            "uploaded_at,ingested_at,v1_bytes_freed,n_parcels,source) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (code, parent_of(code), sqlite3.Binary(blob), len(blob), sha, version, model,
             uploaded_at, now, freed, n_parcels, source))
        c.commit()
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
