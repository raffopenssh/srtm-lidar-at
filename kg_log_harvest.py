"""kg_log_harvest — fold the fleet's merged log into per-KG history rings.

The director's merged log (``data/combined_log_24h.jsonl`` live ring +
``data/log_archive/YYYY-MM-DD.jsonl.gz`` per-day archive) carries every
processing event with a ``kg`` field: starts, tile steps, warnings,
uploads, ``v2up:`` / ``v2verify:`` lines, chkpt restores, requeues.  Once
the ring prunes (24 h) that per-KG provenance is only greppable through
the archive.  This module flattens it into ``kg_v2_store.kg_log`` — one
bounded, gzip'd ring per product code — so ``kg_docs.history(code)`` and
``/api/v1/kg/<code>/history`` answer instantly and the history survives
the v1 JSON deletion.

Incremental: ``meta.log_harvest_live_ts`` remembers the newest live-ring
line folded; ``meta.log_harvest_archive_done`` lists archive days already
folded (one day per tick to bound work; ~200 days ≈ 16 MB gz total).
"""
from __future__ import annotations

import gzip
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import kg_v2_store as store

log = logging.getLogger(__name__)

LIVE_RING = Path("data/combined_log_24h.jsonl")
ARCHIVE_DIR = Path("data/log_archive")
_CODE_RX = re.compile(r"\b(\d{5}(?:-[a-z][-a-z0-9]*)?)\b")
_KG_MSG_RX = re.compile(r"\bKG\s+(\d{5}(?:-[a-z][-a-z0-9]*)?)\b")
_LVL = {"info": "i", "warning": "w", "error": "e", "success": "s"}
# Per-tile step heartbeats ("KG 62015: lidar — tile 3/16 — reading DTM/DSM",
# "…: terrain — tile 3/16", …) are >90 %% of a KG's log volume.  We keep ONE
# row per (code, tile) — the first heartbeat, rewritten as "tile 3/16 ▶
# lidar" — so the ring holds whole runs, not 600 tile sub-steps.
_HEARTBEAT_RX = re.compile(r"^KG (\S+): ([a-z_]+) — tile (\d+/\d+)(?: — .*)?$")


def _code_of(entry: dict) -> str | None:
    kg = (entry.get("kg") or "").strip()
    if kg and _CODE_RX.fullmatch(kg):
        return kg
    m = _KG_MSG_RX.search(entry.get("msg") or "")
    return m.group(1) if m else None


def _fold(lines, since_ts: str = "") -> tuple[dict, str]:
    """lines: iterable of JSON strings → ({code: [rows]}, newest_ts)."""
    by: dict[str, list] = defaultdict(list)
    newest = since_ts
    seen_tiles: set[tuple] = set()
    for ln in lines:
        try:
            e = json.loads(ln)
        except Exception:
            continue
        ts = e.get("ts") or ""
        if not ts or ts <= since_ts:
            continue
        code = _code_of(e)
        if not code:
            continue
        msg = (e.get("msg") or "")[:400]
        lvl = _LVL.get(e.get("level"), "i")
        if lvl == "i":
            hb = _HEARTBEAT_RX.match(msg)
            if hb:
                key = (code, e.get("peer") or "", hb.group(3))
                if key in seen_tiles:
                    continue
                seen_tiles.add(key)
                msg = f"tile {hb.group(3)} ▶ {hb.group(2)}"
        by[code].append([ts[:32], e.get("peer") or "", lvl, msg])
        if ts > newest:
            newest = ts
    return by, newest


def fold_lines(lines) -> dict:
    """Fold already-serialised JSONL lines (the batch app._archive_lines is
    about to append to the archive) into the per-code rings.  Event-driven
    twin of harvest_live: called at ring-prune time so every line is folded
    exactly once as it leaves the live ring; no second read of the archive."""
    by, _ = _fold(lines, "")
    n = 0
    for code, rows in by.items():
        store.append_log(code, rows)
        n += len(rows)
    return {"codes": len(by), "rows": n}


def harvest_live(ring: Path = LIVE_RING) -> dict:
    """Fold live-ring lines newer than the last harvest."""
    if not ring.exists():
        return {"codes": 0, "rows": 0}
    since = store.meta_get("log_harvest_live_ts", "") or ""
    with ring.open() as f:
        by, newest = _fold(f, since)
    n = 0
    for code, rows in by.items():
        store.append_log(code, rows)
        n += len(rows)
    if newest and newest != since:
        store.meta_set("log_harvest_live_ts", newest)
    return {"codes": len(by), "rows": n}


def harvest_archive_day(day_file: Path) -> dict:
    with gzip.open(day_file, "rt") as f:
        by, _ = _fold(f, "")
    n = 0
    for code, rows in by.items():
        store.append_log(code, rows)
        n += len(rows)
    return {"codes": len(by), "rows": n}


def harvest_archive_step(archive_dir: Path = ARCHIVE_DIR, max_days: int = 1) -> dict:
    """Fold up to ``max_days`` not-yet-folded archive days (oldest first)."""
    if not archive_dir.is_dir():
        return {"days": 0, "remaining": 0}
    done = set((store.meta_get("log_harvest_archive_done", "") or "").split(",")) - {""}
    todo = sorted(p for p in archive_dir.glob("*.jsonl.gz") if p.name[:10] not in done)
    out = {"days": 0, "rows": 0, "remaining": len(todo)}
    for p in todo[:max_days]:
        try:
            r = harvest_archive_day(p)
            out["rows"] += r["rows"]
        except Exception as e:  # noqa: BLE001
            log.warning("kg_log_harvest: %s: %s", p.name, e)
        done.add(p.name[:10])
        out["days"] += 1
    out["remaining"] = max(0, len(todo) - out["days"])
    store.meta_set("log_harvest_archive_done", ",".join(sorted(done)))
    return out


def harvest_for_code(code: str, days_back: int = 60) -> int:
    """Targeted fold for one code across live ring + recent archive days
    (used at ingest so the ring is complete even before the archive sweep
    reached those days)."""
    rows = []
    srcs = [LIVE_RING] if LIVE_RING.exists() else []
    if ARCHIVE_DIR.is_dir():
        srcs += sorted(ARCHIVE_DIR.glob("*.jsonl.gz"))[-days_back:]
    parent = code.split("-", 1)[0]
    for p in srcs:
        try:
            opener = gzip.open(p, "rt") if p.suffix == ".gz" else p.open()
            with opener as f:
                for ln in f:
                    if code not in ln and parent not in ln:
                        continue
                    by, _ = _fold([ln], "")
                    rows += by.get(code, [])
        except Exception as e:  # noqa: BLE001
            log.debug("harvest_for_code %s %s: %s", code, p, e)
    return store.append_log(code, rows) if rows else 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        for c in sys.argv[1:]:
            print(c, harvest_for_code(c))
            for r in store.get_log(c)[-5:]:
                print("  ", r)
    else:
        print("live", harvest_live())
        print("archive", harvest_archive_step(max_days=3))
        print(store.log_stats())
