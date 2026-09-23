"""product_repair — repairable-layer gaps in KG products (docs/product-repair.md).

A KG product (``_json`` / ``_json_v2``) is *partial* when at least one
segmented tile lacks a layer that comes from the shared Zenodo tile cache
(Copernicus NDVI / WorldCover / SAR / NDVI harmonics, Hansen).  Such gaps
are *not* the KG's fault: an openEO read-timeout, a credit rotation or a
Hansen hiccup left the 0.1° cell uncached and the tile was baked without
the layer.  They are repairable without BEV or openEO work on the peer:
once the cell is in the cache, a v2 re-upgrade from the KG's own full GPKG
(``--v2-upgrade``, ``v2_source._cache_only_fetch``) picks the layer up.

Four small pieces, all reusable for any future cache-fillable layer:

1. **Label** — the producer stamps ``version = "<base>-partial"`` on the
   JSON manifest entry (``label_version``).  ``v21_products.
   v2_products_complete`` then reads the pair as *not current*, so the
   existing re-upgrade machinery owns the fix.  No tombstones, no
   requeue, the partial product stays live and served.
2. **Registry** — ``data/austria_processor/product_repair.json``
   ``{code: {defects, bbox, product_ts, version, detected_at, attempts,
   state}}``, fed by ``observe()`` from the two primary ingest points
   (fresh v1 JSON → ``search_index.update_kg``; ``_json_v2`` →
   ``v2_ingest``) and by the one-shot ``python3 product_repair.py scan
   --apply`` over the existing corpus.
3. **Fixable gate** — ``fixable(code)`` asks the Zenodo cache index
   (``tile_cache.CopernicusTileCache.has_cached(local_ok=False)`` /
   ``HansenTileCache``) whether every missing product is now cached for
   every 0.1° cell of the KG.  The director only ranks a partial code as
   an upgrade candidate when it is fixable; a re-upgrade that comes back
   partial again counts an *attempt* and two attempts park the code as
   ``stuck`` (loop guard).
4. **Cell fill** — ``missing_cells()`` lists the exact (cell, product)
   pairs still uncached; the director ships them to frontier peers as
   ``prewarm.repair_cells`` in ``cache_manifest.json`` (existing 5-min
   sync, no new endpoint) whenever openEO is healthy, and
   ``austria_processor.prewarm_cell_tiles`` fetches + flushes them.

States: ``pending`` (cells missing) → ``fixable`` (all cached, waiting
for the re-upgrade) → ``done`` (a non-partial product arrived) or
``stuck`` (``MAX_ATTEMPTS`` re-upgrades still partial).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("product_repair")

DATA_DIR = Path("data/austria_processor")
REGISTRY_FILE = DATA_DIR / "product_repair.json"

#: manifest ``version`` suffix marking a product with repairable gaps
PARTIAL_SUFFIX = "-partial"

#: tile-availability key (``data_quality.tiles[].<key>``) → cache product
#: name (``tile_cache`` product / ``prewarm`` fetch).  Only layers that
#: live in the shared Zenodo tile cache belong here — a missing DTM or
#: ortho is a BEV problem and is handled by partial_kgs / upstream_fail.
REPAIRABLE_LAYERS: dict[str, str] = {
    "copernicus_ndvi": "ndvi",
    "worldcover": "worldcover",
    "sar": "sar",
    "harmonics": "harmonics",
    "hansen": "hansen",
}

#: re-upgrades that still come back partial before the code is parked
MAX_ATTEMPTS = 2
#: registry entries in a terminal state are dropped after this
DONE_TTL_S = 7 * 86400

_LOCK = threading.Lock()
_FIXABLE_CACHE: dict[str, tuple[float, bool, list]] = {}
FIXABLE_TTL_S = 600


# === SECTION: label ===

def base_version(v: str | None) -> str:
    """``'v2.3-partial'`` → ``'v2.3'`` (identity for plain labels)."""
    v = str(v or "")
    return v[: -len(PARTIAL_SUFFIX)] if v.endswith(PARTIAL_SUFFIX) else v


def is_partial(v: str | None) -> bool:
    return str(v or "").endswith(PARTIAL_SUFFIX)


def label_version(base: str, defects) -> str:
    """Manifest/Zenodo ``version`` for a product with *defects* (list)."""
    return f"{base}{PARTIAL_SUFFIX}" if defects else base


# === SECTION: defect detection ===

def _tile_active(t: dict) -> bool:
    if t.get("outside_austria") or t.get("ortho_outside_austria"):
        return False
    if t.get("upstream_fail") or t.get("no_valid_pixels"):
        return False
    return bool(t.get("segmentation")) or int(t.get("valid_pixels") or 0) > 0


def defects_from_quality(dq: dict | None) -> list[str]:
    """Sorted repairable layer names missing on ≥1 segmented tile.

    Per-tile, not ``layers_summary``: an upstream-failed (BEV) tile counts
    as active with every layer False and would otherwise flag all five.
    """
    tiles = (dq or {}).get("tiles") or []
    out: set[str] = set()
    for t in tiles:
        if not isinstance(t, dict) or not _tile_active(t):
            continue
        for key in REPAIRABLE_LAYERS:
            if key in t and not t.get(key):
                out.add(key)
    return sorted(out)


def defects(doc: dict | None) -> list[str]:
    return defects_from_quality((doc or {}).get("data_quality"))


def defect_tiles(dq: dict | None) -> int:
    """Number of active tiles carrying ≥1 repairable gap."""
    n = 0
    for t in (dq or {}).get("tiles") or []:
        if isinstance(t, dict) and _tile_active(t) and any(
                k in t and not t.get(k) for k in REPAIRABLE_LAYERS):
            n += 1
    return n


# === SECTION: registry ===

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> dict:
    try:
        d = json.loads(REGISTRY_FILE.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save(d: dict) -> None:
    try:
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = REGISTRY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
        os.replace(tmp, REGISTRY_FILE)
    except Exception as e:
        log.warning("product_repair save: %s", e)


def _bbox_of(doc: dict) -> list | None:
    bb = (doc or {}).get("bbox") or {}
    try:
        if {"min_lon", "min_lat", "max_lon", "max_lat"} <= set(bb):
            return [float(bb["min_lon"]), float(bb["min_lat"]),
                    float(bb["max_lon"]), float(bb["max_lat"])]
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            return [float(x) for x in bb]
    except Exception:
        pass
    return None


def observe(code: str, doc: dict | None, product_ts: str = "",
            version: str = "", *, _d: dict | None = None,
            _save: bool = True) -> dict | None:
    """Record what the newest ingested product of *code* looks like.

    Idempotent per ``product_ts``.  A clean product closes any open
    entry (``done``); a partial one opens / refreshes it.  When the
    previous entry was already ``fixable`` (we believed the cells were
    cached and a re-upgrade ran) and the new product is still partial,
    ``attempts`` is bumped — ``MAX_ATTEMPTS`` → ``stuck``.
    Returns the entry (or None when nothing changed / no defects).
    """
    rec = {"defects": defects(doc), "tiles": defect_tiles((doc or {}).get("data_quality")),
           "bbox": _bbox_of(doc), "ver": str((doc or {}).get("version") or "")}
    return _observe_rec(code, rec, product_ts, version, _d=_d, _save=_save)


def _observe_rec(code: str, rec: dict, product_ts: str = "", version: str = "",
                 *, _d: dict | None = None, _save: bool = True) -> dict | None:
    """``observe`` on a pre-computed slim record ``{defects,tiles,bbox,ver}``."""
    code = str(code)
    dfx = rec.get("defects") or []
    with _LOCK:
        d = _d if _d is not None else load()
        cur = d.get(code)
        if cur and product_ts and str(cur.get("product_ts") or "") >= str(product_ts) \
                and cur.get("state") not in ("done",):
            return cur  # already seen this (or a newer) product
        if not dfx:
            if cur and cur.get("state") not in ("done",):
                cur.update(state="done", done_at=_now(), product_ts=product_ts or cur.get("product_ts"))
                if _save:
                    save(d)
                log.info("product_repair %s: repaired (%s) — %s", code,
                         ",".join(cur.get("defects") or []), product_ts)
                return cur
            return None
        ent = dict(cur or {})
        attempts = int(ent.get("attempts") or 0)
        if cur and cur.get("state") == "fixable":
            attempts += 1  # a re-upgrade ran with the cells cached and still came back partial
        elif cur and cur.get("state") == "done":
            attempts = 0   # regression after a clean product: start over
        ent.update({
            "defects": dfx,
            "tiles": rec.get("tiles"),
            "bbox": rec.get("bbox") or ent.get("bbox"),
            "product_ts": product_ts or ent.get("product_ts") or "",
            "version": version or rec.get("ver") or ent.get("version") or "",
            "attempts": attempts,
            "state": "stuck" if attempts >= MAX_ATTEMPTS else "pending",
            "updated_at": _now(),
        })
        ent.setdefault("detected_at", _now())
        ent.pop("done_at", None)
        d[code] = ent
        if _save:
            save(d)
        if ent["state"] == "stuck":
            log.warning("product_repair %s: still partial after %d re-upgrade(s) (%s) — parked",
                        code, attempts, ",".join(dfx))
        return ent


def prune(d: dict) -> int:
    """Drop terminal entries older than ``DONE_TTL_S``; returns count."""
    cutoff = time.time() - DONE_TTL_S
    gone = []
    for c, e in d.items():
        if e.get("state") != "done":
            continue
        try:
            ts = datetime.fromisoformat(e.get("done_at") or e.get("updated_at")).timestamp()
        except Exception:
            ts = 0
        if ts < cutoff:
            gone.append(c)
    for c in gone:
        d.pop(c, None)
    return len(gone)


def reset(codes) -> int:
    """Operator: put ``stuck`` codes back to ``pending`` with 0 attempts."""
    with _LOCK:
        d = load()
        n = 0
        for c in codes:
            e = d.get(str(c))
            if e:
                e.update(state="pending", attempts=0, updated_at=_now())
                n += 1
        save(d)
    return n


# === SECTION: fixable gate + cell fill ===

def _bbox_dict(bb) -> dict | None:
    if not bb or len(bb) != 4:
        return None
    w, s, e, n = bb
    return {"west": w, "south": s, "east": e, "north": n}


def missing_cells(entry: dict, year: int = 2024) -> list[dict]:
    """``[{w,s,e,n,products:[…]}]`` still uncached on Zenodo for *entry*.

    Uses the Zenodo ZIP index only (``local_ok=False``) — the primary's
    own disk says nothing about what a peer can fetch.
    """
    bb = _bbox_dict(entry.get("bbox"))
    if not bb:
        return []
    products = sorted({REPAIRABLE_LAYERS[k] for k in entry.get("defects") or []
                       if k in REPAIRABLE_LAYERS})
    if not products:
        return []
    from tile_cache import CopernicusTileCache, HansenTileCache
    cop = CopernicusTileCache()
    out: dict[tuple, dict] = {}
    cop_products = [p for p in products if p != "hansen"]
    for cw, cs, ce, cn in cop._iter_cells(bb):
        cell = {"west": cw, "south": cs, "east": ce, "north": cn}
        miss = []
        for p in cop_products:
            kw = {"ndvi": False, "landcover": False, "sar": False, "harmonics": False}
            kw["landcover" if p == "worldcover" else p] = True
            try:
                ok = cop.has_cached(cell, year=year, local_ok=False, **kw)
            except Exception:
                ok = False
            if not ok:
                miss.append(p)
        if "hansen" in products:
            try:
                ok = HansenTileCache().has_cached((cw, cs, ce, cn), local_ok=False)
            except Exception:
                ok = False
            if not ok:
                miss.append("hansen")
        if miss:
            out[(cw, cs)] = {"w": cw, "s": cs, "e": ce, "n": cn, "products": miss}
    return list(out.values())


def fixable(code: str, entry: dict | None = None, *, ttl: float = FIXABLE_TTL_S) -> bool:
    """True iff every missing layer of *code* is now cached for every cell.
    TTL-cached (the ZIP-index walk costs a few ms per cell, the director
    asks for hundreds of codes per candidate scan)."""
    code = str(code)
    now = time.time()
    hit = _FIXABLE_CACHE.get(code)
    if hit and now - hit[0] < ttl:
        return hit[1]
    if entry is None:
        entry = load().get(code)
    if not entry or not entry.get("bbox"):
        res, miss = False, []
    else:
        try:
            miss = missing_cells(entry)
        except Exception as e:
            log.debug("product_repair fixable %s: %s", code, e)
            miss = [{"error": str(e)}]
        res = not miss
    _FIXABLE_CACHE[code] = (now, res, miss)
    return res


def invalidate_fixable_cache() -> None:
    _FIXABLE_CACHE.clear()


def sweep(max_cells: int = 40, year: int = 2024) -> dict:
    """Director tick: refresh states, return the cell-fill plan.

    ``pending`` → ``fixable`` when nothing is missing any more; collects
    up to *max_cells* distinct ``(cell, products)`` requests across the
    still-pending codes (oldest first).  Returns ``{cells:[…],
    pending, fixable, stuck, done, promoted:[codes]}``.
    """
    with _LOCK:
        d = load()
        prune(d)
        cells: dict[tuple, dict] = {}
        promoted = []
        order = sorted((c for c, e in d.items() if e.get("state") in ("pending", "fixable")),
                       key=lambda c: str(d[c].get("detected_at") or ""))
        for c in order:
            e = d[c]
            try:
                miss = missing_cells(e, year=year)
            except Exception as ex:
                log.debug("product_repair sweep %s: %s", c, ex)
                continue
            _FIXABLE_CACHE[c] = (time.time(), not miss, miss)
            if not miss:
                if e.get("state") != "fixable":
                    e.update(state="fixable", fixable_at=_now(), updated_at=_now())
                    promoted.append(c)
                continue
            if e.get("state") == "fixable":
                # cell vanished again (tombstoned ZIP) — back to pending
                e.update(state="pending", updated_at=_now())
            e["missing_cells"] = len(miss)
            for m in miss:
                if len(cells) >= max_cells:
                    break
                k = (m["w"], m["s"])
                if k in cells:
                    cells[k]["products"] = sorted(set(cells[k]["products"]) | set(m["products"]))
                else:
                    cells[k] = dict(m)
        save(d)
        counts = _counts(d)
    return {"cells": list(cells.values()), "promoted": promoted, **counts}


def _counts(d: dict) -> dict:
    c = {"pending": 0, "fixable": 0, "stuck": 0, "done": 0}
    by_defect: dict[str, int] = {}
    for e in d.values():
        st = e.get("state") or "pending"
        c[st] = c.get(st, 0) + 1
        if st in ("pending", "fixable", "stuck"):
            for k in e.get("defects") or []:
                by_defect[k] = by_defect.get(k, 0) + 1
    c["by_defect"] = by_defect
    c["total"] = len(d)
    return c


def summary(rows: int = 8) -> dict:
    d = load()
    s = _counts(d)
    open_codes = sorted((c for c, e in d.items() if e.get("state") in ("pending", "fixable", "stuck")),
                        key=lambda c: (d[c].get("state") != "fixable", str(d[c].get("detected_at") or "")))
    s["rows"] = [{"code": c, "state": d[c].get("state"), "defects": d[c].get("defects"),
                  "tiles": d[c].get("tiles"), "attempts": d[c].get("attempts", 0),
                  "missing_cells": d[c].get("missing_cells")} for c in open_codes[:rows]]
    s["open"] = len(open_codes)
    return s


def process_txt_line() -> str | None:
    """``repair:`` line for ``/process.txt`` (None when registry empty)."""
    s = summary()
    if not s.get("total"):
        return None
    bd = " ".join(f"{k}={v}" for k, v in sorted(s["by_defect"].items()))
    line = (f"repair:   pending={s['pending']} fixable={s['fixable']} stuck={s['stuck']} "
            f"done={s['done']} · by_defect[{bd}] "
            f"(label '<ver>{PARTIAL_SUFFIX}' → re-upgrade once cells cached; "
            f"cells ride prewarm.repair_cells when openEO healthy)")
    if s["rows"]:
        line += " · " + " ".join(
            f"{r['code']}@{r['state']}[{','.join(REPAIRABLE_LAYERS.get(k, k) for k in r['defects'] or [])}"
            f"{'/' + str(r['tiles']) + 't' if r.get('tiles') else ''}"
            f"{'/' + str(r['missing_cells']) + 'c' if r.get('missing_cells') else ''}"
            f"{'/a' + str(r['attempts']) if r.get('attempts') else ''}]"
            for r in s["rows"])
        if s["open"] > len(s["rows"]):
            line += f" … +{s['open'] - len(s['rows'])}"
    return line


# === SECTION: one-shot corpus scan (CLI) ===

def scan_corpus(apply: bool = False, json_dir: Path | None = None,
                relabel_manifest: bool = True) -> dict:
    """Walk every local product doc (v1 JSON files + kg_v2_store) once,
    ``observe`` the partial ones and — with *apply* — relabel the
    primary's manifest ``_json`` / ``_json_v2`` entries to
    ``<ver>-partial`` so the director's candidate scan sees them.

    The relabel is local to this manifest copy (peer copies carry equal
    ``uploaded_at`` and are never overwritten by the merge); that is
    enough — dispatch is decided on the primary, and the peer-side
    eligibility check is bypassed for director-whitelisted repair codes
    (see ``austria_processor.v2_upgrade_eligible``).
    """
    import gc
    import glob
    json_dir = json_dir or (DATA_DIR / "json")
    found: dict[str, dict] = {}   # code -> slim record (docs are 5-10 MB each; never retain)
    n_scanned = 0

    def _slim(doc, ts, src):
        return {"defects": defects(doc), "tiles": defect_tiles(doc.get("data_quality")),
                "bbox": _bbox_of(doc), "ver": str(doc.get("version") or ""), "ts": ts, "src": src}

    files = sorted(glob.glob(str(json_dir / "*.json")))
    for i, f in enumerate(files):
        code = Path(f).stem
        try:
            with open(f) as fh:
                doc = json.load(fh)
        except Exception:
            continue
        n_scanned += 1
        rec = _slim(doc, "", "v1_file")
        del doc
        if rec["defects"]:
            found[code] = rec
        if i % 200 == 0:
            log.info("scan: %d/%d v1 files, %d partial so far", i, len(files), len(found))
            gc.collect()
    try:
        import kg_v2_store as S
        codes = sorted(S.all_codes())
        for i, code in enumerate(codes):
            doc = S.get(code)
            if not doc:
                continue
            n_scanned += 1
            rec = _slim(doc, S.timestamp(code) or "", "v2_store")
            del doc
            if rec["defects"]:
                found[code] = rec
            elif code in found:
                found.pop(code)  # store copy is newer than the v1 file
            if i % 200 == 0:
                log.info("scan: %d/%d v2 store docs, %d partial so far", i, len(codes), len(found))
                gc.collect()
    except Exception as e:
        log.warning("scan: kg_v2_store: %s", e)

    mf_path = DATA_DIR / "zenodo_manifest.json"
    relabelled = []
    try:
        raw = json.loads(mf_path.read_text()) if mf_path.exists() else {}
        ent = raw.get("entries", raw) or {}
    except Exception:
        raw, ent = {}, {}
    out = {}
    if apply:
        d = load()
    for code, f in found.items():
        dfx = f["defects"]
        j2 = ent.get(f"{code}_json_v2")
        j1 = ent.get(f"{code}_json")
        newest = j2 if isinstance(j2, dict) else (j1 if isinstance(j1, dict) else None)
        ts = (newest or {}).get("uploaded_at") or f["ts"]
        ver = (newest or {}).get("version") or f["ver"]
        out[code] = {"defects": dfx, "tiles": f["tiles"], "version": ver, "ts": ts, "src": f["src"]}
        if apply:
            _observe_rec(code, f, ts, label_version(base_version(ver), dfx), _d=d, _save=False)
            if relabel_manifest:
                for key in (f"{code}_json_v2", f"{code}_json"):
                    e = ent.get(key)
                    if isinstance(e, dict) and e.get("version") and not is_partial(e.get("version")) \
                            and "error" not in str(e.get("status") or ""):
                        e["version"] = label_version(e["version"], dfx)
                        relabelled.append(key)
    if apply:
        save(d)
        if relabel_manifest and relabelled:
            tmp = mf_path.with_suffix(".repair.tmp")
            tmp.write_text(json.dumps({"entries": ent} if "entries" in raw or not raw else raw,
                                      indent=2, sort_keys=True))
            os.replace(tmp, mf_path)
    return {"scanned": n_scanned, "partial": len(found), "relabelled": relabelled,
            "codes": out}


def _cli(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cmd = argv[0] if argv else "status"
    if cmd == "scan":
        r = scan_corpus(apply="--apply" in argv, relabel_manifest="--no-relabel" not in argv)
        import collections
        by = collections.Counter(tuple(v["defects"]) for v in r["codes"].values())
        print(f"scanned={r['scanned']} partial={r['partial']} relabelled={len(r['relabelled'])}")
        for k, n in by.most_common():
            print(f"  {','.join(k)}: {n}")
        if "--list" in argv:
            for c, v in sorted(r["codes"].items(), key=lambda kv: kv[1]["ts"]):
                print(f"  {c} {v['version']} {v['ts'][:19]} tiles={v['tiles']} {','.join(v['defects'])}")
        if "--apply" not in argv:
            print("(dry run — add --apply to record + relabel)")
        return 0
    if cmd == "sweep":
        r = sweep()
        print(json.dumps({k: v for k, v in r.items() if k != "cells"}), f"cells={len(r['cells'])}")
        for c in r["cells"][:20]:
            print("  ", c)
        return 0
    if cmd == "reset":
        print(reset(argv[1:]))
        return 0
    print(process_txt_line() or "repair: registry empty")
    print(json.dumps(summary(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
