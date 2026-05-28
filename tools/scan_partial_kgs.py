#!/usr/bin/env python3
"""One-shot scan: detect partial KGs from existing JSON outputs.

A tile counts as "upstream-fail" if its bbox overlaps the parent KG bbox
by more than --overlap (default 0.10) AND the tile has zero valid pixels
and no DTM. Boundary tiles (overlap below threshold) are NOT counted.

Usage:
  python3 tools/scan_partial_kgs.py [--overlap 0.10] [--seed]

  --seed   merge the detected KGs into data/austria_processor/partial_kgs.json
"""
import json, argparse
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JSON_DIR = REPO / "data/austria_processor/json"
PARTIAL_FILE = REPO / "data/austria_processor/partial_kgs.json"


def rect_overlap_frac(t_w, t_s, t_e, t_n, k_w, k_s, k_e, k_n):
    iw = max(0.0, min(t_e, k_e) - max(t_w, k_w))
    ih = max(0.0, min(t_n, k_n) - max(t_s, k_s))
    inter = iw * ih
    ta = (t_e - t_w) * (t_n - t_s)
    return inter / ta if ta > 0 else 0.0


def scan(overlap_thresh: float):
    out = []
    for p in sorted(JSON_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        bbox = d.get("bbox") or {}
        try:
            k_w = float(bbox["min_lon"]); k_s = float(bbox["min_lat"])
            k_e = float(bbox["max_lon"]); k_n = float(bbox["max_lat"])
        except Exception:
            continue
        dq = d.get("data_quality") or {}
        tiles = dq.get("tiles") or []
        n_up = 0
        ups = []
        n_bnd = 0
        for t in tiles:
            bb = t.get("bbox_wgs") or []
            if len(bb) != 4:
                continue
            try:
                t_w, t_s, t_e, t_n = map(float, bb)
            except Exception:
                continue
            of = rect_overlap_frac(t_w, t_s, t_e, t_n, k_w, k_s, k_e, k_n)
            empty = (int(t.get("valid_pixels", 0) or 0) == 0 and not t.get("dtm"))
            if empty:
                if of > overlap_thresh:
                    n_up += 1
                    ups.append({"tile_index": t.get("tile_index"),
                                "bbox_wgs": bb,
                                "overlap_frac": round(of, 3),
                                "reason": "historic_scan"})
                else:
                    n_bnd += 1
        if n_up > 0:
            out.append({
                "code": d.get("kg_code") or p.stem,
                "name": d.get("kg_name", ""),
                "quality_score": dq.get("quality_score"),
                "quality_grade": dq.get("quality_grade"),
                "n_tiles": len(tiles),
                "n_upstream_failed_tiles": n_up,
                "n_boundary_tiles": n_bnd,
                "upstream_failed_tiles": ups,
                "missing_layers": dq.get("missing_layers", []),
                "json_path": str(p),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", type=float, default=0.10)
    ap.add_argument("--seed", action="store_true",
                    help="merge into partial_kgs.json")
    args = ap.parse_args()
    res = scan(args.overlap)
    res.sort(key=lambda r: (-r["n_upstream_failed_tiles"], r["quality_score"] or 0))
    print(f"detected {len(res)} partial KG(s)")
    print(f"{'code':<18} {'grade':<5} {'score':<6} {'up':<3} {'bnd':<3} name")
    for r in res:
        print(f"{r['code']:<18} {str(r['quality_grade']):<5} "
              f"{str(r['quality_score']):<6} "
              f"{r['n_upstream_failed_tiles']:<3} {r['n_boundary_tiles']:<3} "
              f"{r['name']}")
    if args.seed:
        cur = {}
        if PARTIAL_FILE.exists():
            try:
                cur = json.loads(PARTIAL_FILE.read_text()) or {}
            except Exception:
                cur = {}
        ts = datetime.now(timezone.utc).isoformat()
        for r in res:
            cur[r["code"]] = {
                "code": r["code"],
                "name": r["name"],
                "quality_score": r["quality_score"],
                "quality_grade": r["quality_grade"],
                "n_upstream_failed_tiles": r["n_upstream_failed_tiles"],
                "upstream_failed_tiles": r["upstream_failed_tiles"],
                "missing_layers": r["missing_layers"],
                "elapsed_s": 0.0,
                "ts": ts,
                "source": "historic_scan",
            }
        PARTIAL_FILE.write_text(json.dumps(cur, indent=2, sort_keys=True))
        print(f"\nseeded -> {PARTIAL_FILE}")


if __name__ == "__main__":
    main()
