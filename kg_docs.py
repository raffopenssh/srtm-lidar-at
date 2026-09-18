"""kg_docs — one place to load a KG document on the primary.

Since the v2 rollout a KG's per-code document lives in ONE of two places:

* ``data/kg_v2_store.db`` (``kg_v2_store``) — verified kgjson/2 blob, the
  authoritative copy once the KG has been upgraded/ingested; or
* ``data/austria_processor/json/<code>.json`` — the legacy v1 file, which
  the ingest deletes after the blob is stored.

Every reader (search index selector, ``/api/v1/kg``, coverage oracle,
share merge, quality flags, …) goes through here so "store first, then
file" is decided once.  Codes are product codes: plain ``"19570"`` or
split blocks ``"49006-north"``.

Returned dicts from the store are shared (LRU-cached) — do not mutate.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

JSON_DIR = Path("data/austria_processor/json")


def _store():
    try:
        import kg_v2_store
        return kg_v2_store
    except Exception:  # noqa: BLE001
        return None


def file_path(code: str, json_dir: Path | str | None = None) -> Path:
    return Path(json_dir or JSON_DIR) / f"{code}.json"


def source(code: str, json_dir=None) -> str | None:
    """'store' | 'file' | None."""
    st = _store()
    if st is not None and st.has(code):
        return "store"
    if file_path(code, json_dir).exists():
        return "file"
    return None


def exists(code: str, json_dir=None) -> bool:
    return source(code, json_dir) is not None


def load(code: str, json_dir=None) -> dict | None:
    """Decoded v1-shape document, store first."""
    st = _store()
    if st is not None:
        d = st.get(code)
        if d is not None:
            return d
    p = file_path(code, json_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            log.warning("kg_docs: %s unreadable: %s", p, e)
    return None


def bbox(code: str, json_dir=None) -> tuple | None:
    """(min_lon, min_lat, max_lon, max_lat) or None — cheap for the store."""
    st = _store()
    if st is not None:
        b = st.get_bbox(code)
        if b:
            return b
    d = load(code, json_dir) if source(code, json_dir) == "file" else None
    if d:
        bb = d.get("bbox") or {}
        try:
            return tuple(float(bb[k]) for k in ("min_lon", "min_lat", "max_lon", "max_lat"))
        except Exception:
            return None
    return None


def timestamp(code: str, json_dir=None) -> str:
    """ISO-ish freshness stamp: store uploaded_at/generated_at, else file
    generated_at, else mtime.  Lexicographically sortable."""
    st = _store()
    if st is not None:
        t = st.timestamp(code)
        if t:
            return t
    p = file_path(code, json_dir)
    if p.exists():
        try:
            head = p.open().read(4096)
            import re
            m = re.search(r'"generated_at"\s*:\s*"([^"]+)"', head)
            if m:
                return m.group(1)
        except Exception:
            pass
        try:
            import datetime as _dt
            return _dt.datetime.utcfromtimestamp(p.stat().st_mtime).isoformat()
        except Exception:
            pass
    return ""


def codes_for_parent(parent: str, json_dir=None) -> tuple[bool, list[str]]:
    """``(plain_present, sorted block codes)`` across store + files."""
    plain = False
    blocks: set[str] = set()
    st = _store()
    if st is not None:
        try:
            for c in st.codes_for_parent(parent):
                if c == parent:
                    plain = True
                elif c.startswith(parent + "-"):
                    blocks.add(c)
        except Exception:
            pass
    jd = Path(json_dir or JSON_DIR)
    if file_path(parent, jd).exists():
        plain = True
    try:
        for bp in jd.glob(f"{parent}-*.json"):
            if bp.stem.startswith(parent + "-"):
                blocks.add(bp.stem)
    except Exception:
        pass
    return plain, sorted(blocks)


def all_codes(json_dir=None) -> set[str]:
    """Every product code with a local document (store ∪ files)."""
    out: set[str] = set()
    st = _store()
    if st is not None:
        try:
            out |= st.all_codes()
        except Exception:
            pass
    jd = Path(json_dir or JSON_DIR)
    if jd.is_dir():
        out |= {p.stem for p in jd.glob("*.json")}
    return out


def iter_docs(kg_code: str, json_dir=None):
    """Yield ``(code, doc)`` for the parent + its blocks (deterministic)."""
    plain, blocks = codes_for_parent(kg_code, json_dir)
    if plain:
        d = load(kg_code, json_dir)
        if d is not None:
            yield kg_code, d
    for b in blocks:
        d = load(b, json_dir)
        if d is not None:
            yield b, d


def history(code: str, limit: int | None = None) -> list[dict]:
    """Per-KG processing history (merged-log lines mentioning the code),
    newest last — from the ``kg_v2_store.kg_log`` rings that
    ``kg_log_harvest`` fills at ring-prune time + archive backfill."""
    st = _store()
    if st is None:
        return []
    try:
        return st.get_log(code, limit=limit)
    except Exception as e:  # noqa: BLE001
        log.debug("kg_docs.history %s: %s", code, e)
        return []
