#!/usr/bin/env python3
"""Fix inflated per-parcel area_summary / frav in existing KG JSONs.

Bug: STRtree centroid-in-polygon assignment credits whole-segment area to a
parcel even when only part overlaps. Result: ~20% of parcels had per-type m²
values totalling 1.5–4× the cadastre parcel area. Fractions were correct;
absolute m² in `area_summary[t].area_sqm` and `frav[letter]` were wrong.

This tool rescales any parcel where Σarea_summary > 1.05 × area_sqm, so the
sum equals the parcel area. Fractions are preserved (recomputed from scaled
areas, which gives the same value).

Usage:
    python3 fix_frav_inflation.py --dry-run    # report how many need fixing
    python3 fix_frav_inflation.py              # rewrite in place (atomic)
    python3 fix_frav_inflation.py --kg 91109   # one KG

Only touches local data/austria_processor/json/*.json. Already-uploaded
Zenodo JSONs are fixed by the next backfill run
(POST /api/v1/admin/run_backfill) which downloads + rewrites + re-uploads.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, tempfile
from pathlib import Path

TYPE_LETTER = {
    'tree':'t','shrub':'s','grass':'g','hedge':'h','water':'w',
    'roof':'R','greenhouse':'G','solar_panel':'P','fence':'F','wall':'W',
    'mast':'M','wind_turbine':'T','substation':'X',
    'road':'r','path':'p','parking':'k','bridge':'b',
    'crop':'c','orchard':'o','vineyard':'v','garden':'a',
    'bare_soil':'B','rock':'K','excavation':'E','fill':'L','tree_loss':'l',
    'construction':'C','earthwork':'e','unclassified':'u',
}

JSON_DIR = Path('data/austria_processor/json')


def _rescale_parcel(p: dict) -> bool:
    """Rescale a single parcel's area_summary + frav. Returns True if changed."""
    parea = float(p.get('area_sqm') or 0)
    if parea <= 0:
        return False
    as_ = p.get('area_summary') or {}
    if not as_:
        return False
    tot = sum(float((info or {}).get('area_sqm', 0)) for info in as_.values())
    if tot <= parea * 1.05:
        return False  # already within 5% → leave alone
    scale = parea / tot
    new_as = {}
    scaled_pairs = []
    for t, info in as_.items():
        a = float((info or {}).get('area_sqm', 0)) * scale
        scaled_pairs.append((t, a))
    ssum = sum(a for _, a in scaled_pairs) or 1.0
    # Sort largest first to keep dominant_type stable on equal-share edge cases
    for t, a in sorted(scaled_pairs, key=lambda kv: -kv[1]):
        new_as[t] = {'area_sqm': int(round(a)), 'fraction': round(a / ssum, 4)}
    p['area_summary'] = new_as
    # Rebuild frav from rescaled area_summary
    if 'frav' in p:
        p['frav'] = {TYPE_LETTER.get(t, '?'): info['area_sqm']
                     for t, info in new_as.items() if info['area_sqm'] > 0}
    return True


def _atomic_write(path: Path, data) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except OSError: pass
        raise


def process_file(path: Path, dry_run: bool) -> tuple[int, int]:
    """Returns (n_parcels, n_fixed)."""
    try:
        data = json.load(open(path))
    except Exception as e:
        print(f'!! {path.name}: {e}', file=sys.stderr)
        return (0, 0)
    pdets = (data.get('parcels') or {}).get('details') or []
    n = len(pdets)
    fixed = sum(1 for p in pdets if _rescale_parcel(p))
    if fixed and not dry_run:
        _atomic_write(path, data)
    return (n, fixed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--kg', help='only process this KG code')
    args = ap.parse_args()

    if args.kg:
        files = list(JSON_DIR.glob(f'{args.kg}*.json'))
    else:
        files = sorted(JSON_DIR.glob('*.json'))
    print(f'scanning {len(files)} KG JSONs in {JSON_DIR}'
          f'{" (dry run)" if args.dry_run else ""}')

    total_p = total_fixed = files_touched = 0
    for f in files:
        n, fx = process_file(f, args.dry_run)
        total_p += n
        total_fixed += fx
        if fx:
            files_touched += 1
            print(f'  {f.name}: {fx}/{n} parcels rescaled')
    print(f'\n{"would fix" if args.dry_run else "fixed"} '
          f'{total_fixed} parcels across {files_touched} KGs '
          f'(out of {total_p} total parcels in {len(files)} KGs)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
