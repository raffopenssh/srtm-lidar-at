"""KG Block Splitter — split large Katastralgemeinden into contiguous blocks.

Large KGs (>MAX_TILES_PER_BLOCK tiles) are split into multiple blocks,
each containing a contiguous group of parcels.  Blocks are named with
a directional suffix based on their centroid position within the KG:

    49006-north-1, 49006-south-1, 49006-center-1, etc.

Splitting respects parcel boundaries: every parcel belongs to exactly
one block, so users can stitch blocks together in QGIS without overlap.

Usage:
    from kg_splitter import maybe_split_kg, is_block_code, parent_kg_code

    blocks = maybe_split_kg(kg_dict)  # returns [kg_dict] or [block1, block2, ...]
"""

import logging
import math
import numpy as np
from collections import defaultdict

log = logging.getLogger(__name__)

# Strike counter file — written by austria_processor when a KG is
# interrupted (silent subprocess death) or records a non-permanent
# failure. Read on every maybe_split_kg() call so the splitter adapts
# dynamically without code changes.
_STRIKES_FILE = None


def _strikes_file_path():
    global _STRIKES_FILE
    if _STRIKES_FILE is not None:
        return _STRIKES_FILE
    import os
    from pathlib import Path
    base = os.environ.get("SRTM_DATA_DIR")
    if base:
        _STRIKES_FILE = Path(base) / "austria_processor" / "kg_strikes.json"
    else:
        _STRIKES_FILE = Path(__file__).resolve().parent / "data" / "austria_processor" / "kg_strikes.json"
    return _STRIKES_FILE


def _get_strike_count(kg_code: str) -> int:
    """Return the number of strikes (interruptions + non-perm failures)
    recorded for a KG. 0 if the file doesn't exist or the KG isn't tracked.
    """
    import json
    p = _strikes_file_path()
    try:
        if p.exists():
            data = json.loads(p.read_text())
            return int(data.get(str(kg_code), 0))
    except Exception:
        pass
    return 0


def bump_strike(kg_code: str) -> int:
    """Atomically increment the strike count for a KG; returns new value."""
    import json
    p = _strikes_file_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except Exception:
                data = {}
        data[str(kg_code)] = int(data.get(str(kg_code), 0)) + 1
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(p)
        return data[str(kg_code)]
    except Exception:
        return 0


def clear_single_strike(code: str):
    """Remove exactly one code's strike entry (no prefix cascade).

    Used when a single block completes but sibling blocks are still
    pending — the parent's strike count must survive so the split
    stays in force until the whole family is done.
    """
    import json
    p = _strikes_file_path()
    try:
        if not p.exists():
            return
        data = json.loads(p.read_text())
        if data.pop(str(code), None) is None:
            return
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(p)
    except Exception:
        pass


def clear_strikes(kg_code: str):
    """Remove a KG (and any of its block codes) from strike file on success."""
    import json
    p = _strikes_file_path()
    try:
        if not p.exists():
            return
        data = json.loads(p.read_text())
        keys_to_drop = [k for k in data
                        if k == str(kg_code) or k.startswith(str(kg_code) + "-")]
        if not keys_to_drop:
            return
        for k in keys_to_drop:
            data.pop(k, None)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(p)
    except Exception:
        pass


# Maximum tiles per block — tuned for hardware/time constraints
MAX_TILES_PER_BLOCK = 22

# After this many silent interruptions / explicit failures we force-split
# a KG that would otherwise be processed as a single block. The signal
# typically means an OOM kill or watchdog-killed subprocess during the
# heavy gpkg_full / segmentation step — splitting halves the per-child
# working set without operator intervention.
ADAPTIVE_SPLIT_THRESHOLD = 2

# Runtime oversize tolerance. The splitter estimates tiles from the KG
# *API* bbox, but process_one_kg expands the bbox to the cadastre union
# before computing the real grid — some KGs (91017 Schröcken: 20 est →
# 30 runtime) blow well past MAX_TILES_PER_BLOCK only at runtime. When
# the runtime grid exceeds MAX * this factor, the subprocess aborts with
# needs_split instead of grinding into a gpkg_full it cannot finish.
# The 1.3 slack means moderately-over KGs (23-28 tiles) that complete
# fine today are left alone — no churn.
RUNTIME_SPLIT_FACTOR = 1.3


def runtime_tile_limit() -> int:
    """Tile count above which a runtime (post-expansion) grid must split."""
    return math.ceil(MAX_TILES_PER_BLOCK * RUNTIME_SPLIT_FACTOR)


def force_split(kg_code: str) -> int:
    """Raise a KG's strike count to at least ADAPTIVE_SPLIT_THRESHOLD so
    the next maybe_split_kg() call splits it. Returns the resulting count.

    Used by the runtime-oversize abort (needs_split): one observation of
    a too-large runtime grid is deterministic evidence — no need to burn
    two more failed attempts accumulating strikes organically.
    """
    import json
    p = _strikes_file_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except Exception:
                data = {}
        cur = int(data.get(str(kg_code), 0))
        new = max(cur, ADAPTIVE_SPLIT_THRESHOLD)
        if new != cur:
            data[str(kg_code)] = new
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(p)
        return new
    except Exception:
        return 0


def _compute_n_tiles(west, south, east, north, tile_km=1.5, overlap_km=0.1):
    """Count how many tiles a bbox would produce (without building the list)."""
    cos_lat = math.cos(math.radians((south + north) / 2))
    step_x = (tile_km - overlap_km) / (111 * cos_lat)
    step_y = (tile_km - overlap_km) / 111
    dx_deg = tile_km / (111 * cos_lat)
    dy_deg = tile_km / 111
    nx = max(1, math.ceil((east - west) / step_x)) if east > west else 1
    ny = max(1, math.ceil((north - south) / step_y)) if north > south else 1
    return nx * ny


def _directional_label(cx, cy, kg_cx, kg_cy, n_blocks):
    """Return a human-readable directional label for a block centroid.

    Uses N/S/E/W/NE/NW/SE/SW/center based on the block's position
    relative to the KG centroid.
    """
    if n_blocks <= 2:
        # Simple N/S or E/W split
        dx = cx - kg_cx
        dy = cy - kg_cy
        if abs(dx) > abs(dy):
            return "east" if dx > 0 else "west"
        else:
            return "north" if dy > 0 else "south"

    dx = cx - kg_cx
    dy = cy - kg_cy
    # Threshold for "center"
    # Use 20% of the KG span
    if abs(dx) < 0.002 and abs(dy) < 0.002:
        return "center"

    ns = ""
    ew = ""
    if abs(dy) > 0.001:
        ns = "north" if dy > 0 else "south"
    if abs(dx) > 0.001:
        ew = "east" if dx > 0 else "west"

    if ns and ew:
        return f"{ns}{ew}"
    return ns or ew or "center"


def maybe_split_kg(kg: dict) -> list:
    """Check if a KG needs splitting and return block dicts.

    Args:
        kg: KG dict with bbox, kg_code, kg_name, etc.

    Returns:
        List of kg-like dicts. If no split needed, returns [kg] unchanged.
        Each block dict has:
          - kg_code: "49006-north" (block code)
          - _parent_kg_code: "49006" (original code)
          - _block_index: 0, 1, 2, ...
          - _block_label: "north"
          - _n_blocks: total number of blocks
          - bbox: adjusted bbox for this block
          - All other original kg fields preserved

        Parcel filtering to each block's bbox happens at runtime in
        process_one_kg() — no parcel data needed at split time.
    """
    bb = kg.get("bbox", {})
    if not bb or "min_lon" not in bb:
        return [kg]

    west, south = bb["min_lon"], bb["min_lat"]
    east, north = bb["max_lon"], bb["max_lat"]

    kg_code = kg["kg_code"]
    kg_name = kg.get("kg_name", "")

    n_tiles = _compute_n_tiles(west, south, east, north)

    # Adaptive force-split: KGs that have been interrupted/failed multiple
    # times without succeeding are split even when below the tile limit.
    # The strike count comes from data/austria_processor/kg_strikes.json,
    # written by austria_processor when it detects an interrupted run or
    # records a non-permanent failure.
    strikes = _get_strike_count(str(kg_code))
    forced = strikes >= ADAPTIVE_SPLIT_THRESHOLD

    if n_tiles <= MAX_TILES_PER_BLOCK and not forced:
        return [kg]

    if forced and n_tiles <= MAX_TILES_PER_BLOCK:
        # Each strike doubles the block count, capped so each block has ≥2 tiles
        n_blocks = min(2 ** (strikes - ADAPTIVE_SPLIT_THRESHOLD + 1),
                       max(2, n_tiles // 2))
        # Adaptive (strike-driven) splits stay at INFO — they're a
        # genuine state change worth seeing.
        log.info("KG %s (%s): adaptive split into %d blocks after %d strikes "
                 "(%d tiles, below %d-tile limit)",
                 kg_code, kg_name, n_blocks, strikes, n_tiles,
                 MAX_TILES_PER_BLOCK)
    else:
        n_blocks = math.ceil(n_tiles / MAX_TILES_PER_BLOCK)
        if forced:
            n_blocks = max(n_blocks, n_blocks * 2)
            log.info("KG %s (%s): %d tiles > %d limit, %d strikes → splitting into %d blocks (doubled)",
                     kg_code, kg_name, n_tiles, MAX_TILES_PER_BLOCK, strikes, n_blocks)
        else:
            # Plain bbox-driven split decisions are deterministic from
            # the KG's geometry — logging them on every call (status
            # poller hits this thousands of times per minute) just
            # spams the journal. Demote to DEBUG; the *commit* of a
            # split (austria_processor / queue resolver) keeps INFO.
            log.debug("KG %s (%s): %d tiles > %d limit → splitting into %d blocks",
                      kg_code, kg_name, n_tiles, MAX_TILES_PER_BLOCK, n_blocks)

    return _split_bbox_grid(kg, n_blocks)


def _split_bbox_grid(kg, n_blocks):
    """Split KG bbox into a grid of blocks (fallback when no parcels available)."""
    kg_code = kg["kg_code"]
    bb = kg["bbox"]
    west, south = bb["min_lon"], bb["min_lat"]
    east, north = bb["max_lon"], bb["max_lat"]

    kg_cx = (west + east) / 2
    kg_cy = (south + north) / 2

    # Determine grid dimensions
    cos_lat = math.cos(math.radians((south + north) / 2))
    span_x_km = (east - west) * 111 * cos_lat
    span_y_km = (north - south) * 111

    if span_y_km >= span_x_km:
        ny = n_blocks
        nx = 1
    else:
        nx = n_blocks
        ny = 1

    # Try 2D grid for large counts
    if n_blocks >= 4:
        nx = math.ceil(math.sqrt(n_blocks * span_x_km / max(span_y_km, 0.01)))
        ny = math.ceil(n_blocks / nx)
        while nx * ny < n_blocks:
            ny += 1

    # Verify all blocks fit within tile limit; increase grid if needed
    for _ in range(5):
        _dx = (east - west) / nx
        _dy = (north - south) / ny
        _max_bt = 0
        for _iy in range(ny):
            for _ix in range(nx):
                _bw = west + _ix * _dx
                _bs = south + _iy * _dy
                _be = west + (_ix + 1) * _dx
                _bn = south + (_iy + 1) * _dy
                _nt = _compute_n_tiles(_bw, _bs, _be, _bn)
                _max_bt = max(_max_bt, _nt)
        if _max_bt <= MAX_TILES_PER_BLOCK:
            break
        # Increase the larger dimension
        if span_y_km / ny >= span_x_km / nx:
            ny += 1
        else:
            nx += 1

    dx = (east - west) / nx
    dy = (north - south) / ny

    blocks = []
    label_counts = defaultdict(int)

    for iy in range(ny):
        for ix in range(nx):
            bw = west + ix * dx
            bs = south + iy * dy
            be = west + (ix + 1) * dx
            bn = south + (iy + 1) * dy

            block_bbox = {
                "min_lon": bw, "min_lat": bs,
                "max_lon": be, "max_lat": bn,
            }
            bcx = (bw + be) / 2
            bcy = (bs + bn) / 2
            direction = _directional_label(bcx, bcy, kg_cx, kg_cy, nx * ny)
            label_counts[direction] += 1
            label = f"{direction}-{label_counts[direction]}"

            block = dict(kg)
            block["kg_code"] = f"{kg_code}-{label}"
            block["kg_name"] = f"{kg.get('kg_name', '')} ({label})"
            block["bbox"] = block_bbox
            block["_parent_kg_code"] = kg_code
            block["_block_index"] = len(blocks)
            block["_block_label"] = label
            block["_n_blocks"] = nx * ny
            blocks.append(block)

    # Simplify single-occurrence labels
    for b in blocks:
        label = b["_block_label"]
        direction = label.rsplit("-", 1)[0]
        if label_counts[direction] == 1:
            simple = direction
            b["kg_code"] = f"{kg_code}-{simple}"
            b["kg_name"] = f"{kg.get('kg_name', '')} ({simple})"
            b["_block_label"] = simple

    for b in blocks:
        b["_n_blocks"] = len(blocks)

    return blocks


# --- Utility functions for block code handling ---

def is_block_code(code: str) -> bool:
    """Check if a code is a block code (e.g. '49006-north-1')."""
    if not code or "-" not in code:
        return False
    # The parent part must be numeric (KG codes are always numeric)
    parent = code.split("-", 1)[0]
    return parent.isdigit()


def parent_kg_code(code: str) -> str:
    """Extract parent KG code from a block code.

    '49006-north-1' → '49006'
    '49006' → '49006'
    """
    if "-" not in code:
        return code
    parent = code.split("-", 1)[0]
    if parent.isdigit():
        return parent
    return code


def block_label(code: str) -> str:
    """Extract block label from a block code.

    '49006-north-1' → 'north-1'
    '49006' → ''
    """
    if "-" not in code:
        return ""
    parts = code.split("-", 1)
    if parts[0].isdigit():
        return parts[1]
    return ""


def all_block_codes_for_parent(parent_code: str, completed_codes: set) -> list:
    """Find all completed block codes belonging to a parent KG."""
    prefix = f"{parent_code}-"
    return sorted(c for c in completed_codes if c.startswith(prefix))


def is_parent_fully_done(parent_code: str, completed_codes: set, expected_blocks: int = None) -> bool:
    """Check if all blocks of a parent KG are complete.

    If expected_blocks is None, we can't know the total — return True
    if at least one block is complete (conservative).
    """
    blocks = all_block_codes_for_parent(parent_code, completed_codes)
    if expected_blocks is not None:
        return len(blocks) >= expected_blocks
    # Without knowing expected count, check if parent itself is done
    return parent_code in completed_codes or len(blocks) > 0
