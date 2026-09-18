"""KG JSON v2 — compact, columnar, gzip'd container for the per-KG summary.

The v1 product (``<code>.json``) is a pretty-printed row-oriented document:
median 6.6 MB on disk, 21 MB for dense urban KGs, ~45 % of it repeated
object keys and ~40 % ``vertex_heights`` dicts.  v2 keeps the *same logical
document* (every reader in ``search_index`` / ``app`` / ``cadastre_bridge``
keeps working on the decoded dict) but stores it as:

* **columnar tables** for every record array (``parcels.details``,
  ``building_footprints.details``, ``new_buildings.features``,
  ``infrastructure.by_type.*.features``, ``top_by_type.*``, ...):
  ``{"_cols": {key: [v0, v1, ...]}, "_n": N}`` — keys written once, missing
  values ``null``; fixed-shape sub-dicts (``centroid``, ``classification``)
  are flattened to dotted columns.
* **packed vertex heights**: per row a flat int array — delta-coded
  lat/lon in 1e-7 deg + dtm (and obj_height) in cm.  v1 already rounds
  to 7 decimals / 2 decimals, so the round trip is exact.
* **gzip** (level 6) on the compact serialisation.  Files are
  ``<code>_v2.json.gz``; manifest key ``<code>_json_v2``.

Measured on 01002 (Alsergrund, 2442 parcels / 12 720 new buildings):
21.6 MB pretty v1 → 4.1 MB compact v2 → **0.99 MB** gzip'd (22×).  Decode
is a few hundred ms for the largest KGs.

API
---
``encode(doc) -> bytes``            v1-shape dict → gzip'd v2 bytes
``decode(blob) -> dict``            gzip'd v2 bytes (or a v2 dict) → v1-shape dict
``is_v2(obj)``                      container sniff
``load(path)`` / ``dump(doc, path)`` file helpers (``.json.gz``)
``roundtrip_ok(doc)``               codec self-test used by ``v2_verify``

The container carries ``"_codec": "kgjson/2"`` at top level; everything
else is the v1 document with tables swapped in-place.  Unknown record
arrays are handled generically, so new v2 sections (``acquisition``,
``parcels.details[].outline_z`` …) need no codec change.
"""
from __future__ import annotations

import gzip
import io
import json
import math
from pathlib import Path
from typing import Any

CODEC = "kgjson/2"
GZIP_LEVEL = 6

# Record arrays we always columnarise (path tuples; "*" = any key).
_TABLE_PATHS = [
    ("parcels", "details"),
    ("building_footprints", "details"),
    ("new_buildings", "features"),
    ("infrastructure", "by_type", "*", "features"),
    ("top_by_type", "*"),
    ("top_10_objects",),
    ("top_10_trees",),
    ("data_quality", "tiles"),
    ("acquisition", "tiles"),
]
# Fixed-shape sub-dicts flattened to dotted columns inside tables.
_FLATTEN_KEYS = {"centroid", "classification", "coordinate"}
# Columns holding a list of {lat, lon, dtm_m[, obj_height_m]} dicts.
_VERTEX_KEYS = {"vertex_heights"}
_MIN_ROWS_FOR_TABLE = 4


# --------------------------------------------------------------------------
# vertex packing
# --------------------------------------------------------------------------

def _q(v: Any, scale: int) -> int | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return int(round(f * scale))


def pack_vertices(vhs: list | None) -> list | None:
    """[{lat, lon, dtm_m[, obj_height_m]}, ...] → flat int list.

    Layout: ``[flags, n, dlat0, dlon0, dtm0[, oh0], dlat1, ...]`` with
    lat/lon delta-coded (1e-7 deg) and heights in cm.  ``flags`` bit 0 =
    rows carry ``obj_height_m``.
    """
    if vhs is None:
        return None
    if not isinstance(vhs, list):
        return vhs  # unknown shape — leave untouched
    has_oh = any(isinstance(v, dict) and "obj_height_m" in v for v in vhs)
    out = [1 if has_oh else 0, len(vhs)]
    plat = plon = 0
    for v in vhs:
        if not isinstance(v, dict):
            return vhs
        la = _q(v.get("lat"), 10_000_000)
        lo = _q(v.get("lon"), 10_000_000)
        if la is None or lo is None:
            return vhs  # can't pack losslessly — keep verbose
        out.append(la - plat)
        out.append(lo - plon)
        plat, plon = la, lo
        out.append(_q(v.get("dtm_m"), 100))
        if has_oh:
            out.append(_q(v.get("obj_height_m"), 100))
    return out


def unpack_vertices(packed: list | None) -> list | None:
    if packed is None:
        return None
    if not isinstance(packed, list) or len(packed) < 2 or not isinstance(packed[0], int) \
            or (packed and isinstance(packed[0], dict)):
        return packed
    flags, n = packed[0], packed[1]
    has_oh = bool(flags & 1)
    stride = 4 if has_oh else 3
    if len(packed) != 2 + n * stride:
        return packed  # not ours
    out = []
    la = lo = 0
    i = 2
    for _ in range(n):
        la += packed[i]
        lo += packed[i + 1]
        d = {"lat": round(la / 1e7, 7), "lon": round(lo / 1e7, 7)}
        dtm = packed[i + 2]
        d["dtm_m"] = None if dtm is None else round(dtm / 100.0, 2)
        if has_oh:
            oh = packed[i + 3]
            d["obj_height_m"] = None if oh is None else round(oh / 100.0, 2)
        out.append(d)
        i += stride
    return out


# --------------------------------------------------------------------------
# columnar tables
# --------------------------------------------------------------------------

def _is_record_list(v: Any) -> bool:
    return (isinstance(v, list) and len(v) >= _MIN_ROWS_FOR_TABLE
            and all(isinstance(r, dict) for r in v))


def to_table(rows: list[dict]) -> dict:
    n = len(rows)
    cols: dict[str, list] = {}

    def _col(k):
        c = cols.get(k)
        if c is None:
            c = cols[k] = [None] * n
        return c

    for i, r in enumerate(rows):
        for k, v in r.items():
            if k in _VERTEX_KEYS:
                _col(k)[i] = pack_vertices(v)
            elif k in _FLATTEN_KEYS and isinstance(v, dict) and all(
                    not isinstance(x, (dict, list)) for x in v.values()):
                for kk, vv in v.items():
                    _col(f"{k}.{kk}")[i] = vv
                _col(f"{k}.")[i] = 1  # presence marker (dict existed, maybe empty)
            else:
                _col(k)[i] = v
    # Drop presence markers when every row has the dict and at least one field.
    for k in list(cols):
        if k.endswith(".") and all(x == 1 for x in cols[k]):
            del cols[k]
    return {"_cols": cols, "_n": n}


def from_table(t: dict) -> list[dict]:
    n = int(t.get("_n", 0))
    cols = t.get("_cols") or {}
    rows: list[dict] = [dict() for _ in range(n)]
    # Presence markers first so flattened dicts are recreated even when empty.
    markers = {k[:-1]: v for k, v in cols.items() if k.endswith(".")}
    for base, pres in markers.items():
        for i in range(n):
            if pres[i] == 1:
                rows[i][base] = {}
    for k, col in cols.items():
        if k.endswith("."):
            continue
        if "." in k:
            base, sub = k.split(".", 1)
            for i in range(n):
                v = col[i]
                if v is None and base not in markers and base not in rows[i]:
                    # absent field on an absent dict → skip; a real None
                    # inside an existing dict is rare and lost here — v1
                    # never emits None inside centroid/classification.
                    continue
                if v is None and base not in rows[i]:
                    continue
                rows[i].setdefault(base, {})[sub] = v
            continue
        if k in _VERTEX_KEYS:
            for i in range(n):
                v = col[i]
                if v is not None:
                    rows[i][k] = unpack_vertices(v)
            continue
        for i in range(n):
            v = col[i]
            if v is not None:
                rows[i][k] = v
    return rows


def _is_table(v: Any) -> bool:
    return isinstance(v, dict) and "_cols" in v and "_n" in v


def _walk_paths(doc: dict, path: tuple, fn):
    """Apply fn(parent, key) for every concrete location matching *path*."""
    def rec(node, i):
        if i == len(path) - 1:
            k = path[i]
            if isinstance(node, dict):
                if k == "*":
                    for kk in list(node.keys()):
                        fn(node, kk)
                elif k in node:
                    fn(node, k)
            return
        k = path[i]
        if not isinstance(node, dict):
            return
        if k == "*":
            for kk in list(node.keys()):
                rec(node[kk], i + 1)
        elif k in node:
            rec(node[k], i + 1)
    rec(doc, 0)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def to_v2_dict(doc: dict) -> dict:
    """Return a v2 container dict (no compression) — deep-copies via JSON."""
    # Deep copy + normalise in one pass: live processor docs carry numpy
    # scalars / datetimes (v1 wrote them with ``default=str``) and NaN/inf
    # floats (v1 wrote bare ``NaN`` tokens).  NaN/inf → null so the compact
    # output is strict JSON.
    out = json.loads(json.dumps(doc, separators=(",", ":"), allow_nan=True,
                                default=_json_default),
                     parse_constant=lambda _c: None)

    def _tab(parent, key):
        v = parent.get(key)
        if _is_record_list(v):
            parent[key] = to_table(v)

    for p in _TABLE_PATHS:
        _walk_paths(out, p, _tab)
    out["_codec"] = CODEC
    return out


def from_v2_dict(v2: dict) -> dict:
    out = json.loads(json.dumps(v2, separators=(",", ":"), allow_nan=True))
    out.pop("_codec", None)

    def _untab(parent, key):
        v = parent.get(key)
        if _is_table(v):
            parent[key] = from_table(v)

    for p in _TABLE_PATHS:
        _walk_paths(out, p, _untab)
    return out


def is_v2(obj: Any) -> bool:
    if isinstance(obj, dict):
        return obj.get("_codec") == CODEC
    if isinstance(obj, (bytes, bytearray)):
        return obj[:2] == b"\x1f\x8b"
    return False


def encode(doc: dict, level: int = GZIP_LEVEL) -> bytes:
    """v1-shape dict → gzip'd compact v2 bytes."""
    v2 = to_v2_dict(doc)
    raw = json.dumps(v2, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    return gzip.compress(raw, compresslevel=level, mtime=0)


def _json_default(o: Any):
    try:
        import numpy as _np
        if isinstance(o, _np.generic):
            return o.item()
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def decode(blob: bytes | bytearray | dict | str) -> dict:
    """gzip'd v2 bytes / plain v2 JSON bytes / v2 dict → v1-shape dict.

    Also accepts a plain v1 document (bytes or dict) and returns it as-is,
    so callers can feed either product version.
    """
    if isinstance(blob, dict):
        return from_v2_dict(blob) if is_v2(blob) else blob
    if isinstance(blob, str):
        blob = blob.encode("utf-8")
    if blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    d = json.loads(blob)
    return from_v2_dict(d) if is_v2(d) else d


def load(path: str | Path) -> dict:
    return decode(Path(path).read_bytes())


def dump(doc: dict, path: str | Path) -> int:
    """Write ``doc`` as v2 to *path* (atomic).  Returns bytes written."""
    p = Path(path)
    data = encode(doc)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(p)
    return len(data)


def _strip_none(o: Any) -> Any:
    """Drop None-valued dict keys recursively.  The columnar codec stores
    None as *absent* (a null cell decodes to a missing key), so equality
    is defined modulo ``None ≡ absent`` — every consumer reads records
    with ``.get``."""
    if isinstance(o, dict):
        return {k: _strip_none(v) for k, v in o.items() if v is not None}
    if isinstance(o, list):
        return [_strip_none(v) for v in o]
    return o


def _normalise(o: Any) -> Any:
    """JSON-normalise for equality (tuples→lists, -0.0→0.0, NaN→None→absent)."""
    return _strip_none(json.loads(json.dumps(o, separators=(",", ":"), allow_nan=True,
                                             default=_json_default),
                                  parse_constant=lambda _c: None))


def roundtrip_ok(doc: dict) -> tuple[bool, str]:
    """Encode+decode *doc* and compare.  Vertex heights are compared after
    quantisation (7 dp / 2 dp, which v1 already applies).  Returns
    ``(ok, detail)``."""
    try:
        back = decode(encode(doc))
    except Exception as e:  # noqa: BLE001
        return False, f"codec error: {e}"
    a = _normalise(doc)
    b = _normalise(back)

    def _quant(o):
        if isinstance(o, dict):
            if "lat" in o and "lon" in o and ("dtm_m" in o or "obj_height_m" in o) and len(o) <= 4:
                q = {"lat": round(float(o["lat"]), 7), "lon": round(float(o["lon"]), 7)}
                for k in ("dtm_m", "obj_height_m"):
                    if k in o:
                        q[k] = None if o[k] is None else round(float(o[k]), 2)
                return q
            return {k: _quant(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_quant(v) for v in o]
        return o
    a = _quant(a)
    if a == b:
        return True, "ok"
    # locate first differing top-level section for the error message
    for k in set(a) | set(b):
        if a.get(k) != b.get(k):
            return False, f"mismatch in section '{k}'"
    return False, "mismatch"


def sizes(doc: dict) -> dict:
    """Diagnostics: bytes for pretty-v1 / compact-v1 / compact-v2 / gz-v2."""
    pretty = len(json.dumps(doc, indent=2).encode())
    compact = len(json.dumps(doc, separators=(",", ":")).encode())
    v2 = to_v2_dict(doc)
    v2c = len(json.dumps(v2, separators=(",", ":")).encode())
    gz = len(encode(doc))
    return {"v1_pretty": pretty, "v1_compact": compact, "v2_compact": v2c,
            "v2_gz": gz, "ratio_pretty_to_gz": round(pretty / max(gz, 1), 1)}


if __name__ == "__main__":  # pragma: no cover
    import sys
    import time
    for f in sys.argv[1:]:
        t0 = time.time()
        doc = load(f) if f.endswith(".gz") else json.loads(Path(f).read_bytes())
        ok, why = roundtrip_ok(doc)
        s = sizes(doc)
        print(f"{f}: roundtrip={ok} ({why}) "
              f"pretty={s['v1_pretty']/1e6:.2f}MB compact={s['v1_compact']/1e6:.2f}MB "
              f"v2={s['v2_compact']/1e6:.2f}MB gz={s['v2_gz']/1e6:.3f}MB "
              f"x{s['ratio_pretty_to_gz']} in {time.time()-t0:.1f}s")
