#!/usr/bin/env python3
"""Backfill per-parcel top_10_objects and top_10_trees in existing KG JSONs.

Reads each KG JSON in data/austria_processor/json/. For each parcel that lacks
top_10_objects, derives them by spatial-joining segment_points from the
corresponding light GPKG to the parcel polygon. The light GPKG is fetched
from Zenodo if not already cached locally.

After updating a JSON, the script re-uploads it to its existing Zenodo
draft deposition (replacing the old file in-place) and refreshes the
local manifest entry. This works because:
  * All peers share the same Zenodo token.
  * KG depositions are still drafts (state='unsubmitted'); files in drafts
    can be replaced via DELETE+PUT to the bucket URL.

Usage:
    python3 backfill_parcel_top10.py             # process all local JSONs
    python3 backfill_parcel_top10.py --dry-run   # don't upload, just rewrite JSONs
    python3 backfill_parcel_top10.py --kg 49006-south
    python3 backfill_parcel_top10.py --no-upload
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from zenodo_client import Client, Manifest, DEFAULT_TOKEN, landscape_metadata  # noqa: E402
from parcel_compact import TYPE_LETTER, TOP_N  # noqa: E402

log = logging.getLogger('backfill')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

JSON_DIR = ROOT / 'data' / 'austria_processor' / 'json'
GPKG_CACHE = ROOT / 'data' / 'austria_processor' / 'backfill_gpkg_cache'
MANIFEST_PATH = ROOT / 'data' / 'austria_processor' / 'zenodo_manifest.json'

GPKG_CACHE.mkdir(parents=True, exist_ok=True)

# Per-parcel top item count is fixed in parcel_compact.TOP_N (5).


def _download_light_gpkg(manifest: dict, kg_code: str) -> Path | None:
    """Locate or download the light GPKG for *kg_code*.

    Returns the local path or None if the manifest has no entry.
    """
    # Local first (austria_processor produces these in data/.../gpkg/)
    local_proc = ROOT / 'data' / 'austria_processor' / 'gpkg' / f'{kg_code}_light.gpkg'
    if local_proc.exists():
        return local_proc
    cached = GPKG_CACHE / f'{kg_code}_light.gpkg'
    if cached.exists() and cached.stat().st_size > 1024:
        return cached
    entry = manifest.get(f'{kg_code}_light_gpkg')
    if not entry:
        log.warning('%s: no light_gpkg manifest entry; skipping', kg_code)
        return None
    bucket = entry.get('bucket_url')
    fname = entry.get('filename')
    if not bucket or not fname:
        log.warning('%s: light_gpkg entry missing bucket/filename', kg_code)
        return None
    url = f'{bucket}/{fname}'
    size_mb = (entry.get('size') or 0) / 1e6
    log.info('%s: downloading light GPKG (%.1f MB)', kg_code, size_mb)
    tmp = cached.with_suffix('.tmp')
    try:
        r = requests.get(url, params={'access_token': DEFAULT_TOKEN}, stream=True, timeout=600)
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
        tmp.rename(cached)
        return cached
    except Exception as e:
        log.error('%s: download failed: %s', kg_code, e)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        return None


def _load_segment_points_and_parcels(gpkg_path: Path):
    """Load segment_points and parcels layers from a light GPKG.

    Returns ``(seg_records, parcel_geoms_wgs)`` where seg_records is a list
    of dicts (one per segment, all properties + lon/lat), and
    parcel_geoms_wgs is a list of (parcel_id, shapely_polygon_wgs).
    """
    import fiona
    from shapely.geometry import shape as _shape

    seg_records = []
    with fiona.open(str(gpkg_path), layer='segment_points') as src:
        for f in src:
            props = dict(f['properties'])
            geom = f['geometry']
            if geom and geom.get('type') == 'Point':
                lon, lat = geom['coordinates'][:2]
                props['_lon'] = lon
                props['_lat'] = lat
                seg_records.append(props)

    parcels = []
    with fiona.open(str(gpkg_path), layer='parcels') as src:
        for f in src:
            props = dict(f['properties'])
            pid = props.get('parcel_id', '')
            try:
                geom = _shape(f['geometry'])
                if geom.is_empty:
                    continue
                parcels.append((pid, geom))
            except Exception:
                continue
    return seg_records, parcels


def _seg_to_top_obj_compact(s: dict) -> list:
    """Convert a segment_points record to a compact top_objs array entry.

    Format mirrors parcel_compact.TOP_OBJS_KEYS.
    """
    otype = s.get('type') or ''
    return [
        TYPE_LETTER.get(otype, '?'),
        round(float(s.get('height_max_m') or 0), 1),
        round(float(s.get('height_mean_m') or 0), 1),
        int(float(s.get('area_sqm') or 0)),
        round(float(s['_lon']), 7),
        round(float(s['_lat']), 7),
        round(float(s.get('confidence') or 0), 2),
        round(float(s.get('rf_confidence') or 0), 2),
        1 if s.get('is_manmade') else 0,
    ]


def _seg_to_top_tree_compact(s: dict) -> list:
    """Convert a tree segment_points record to a compact top_trees array entry."""
    return [
        round(float(s.get('height_max_m') or 0), 1),
        round(float(s.get('height_mean_m') or 0), 1),
        round(float(s.get('height_p90_m') or 0), 1),
        int(float(s.get('area_sqm') or 0)),
        round(float(s['_lon']), 7),
        round(float(s['_lat']), 7),
        round(float(s.get('ndvi_mean') or 0), 3),
        round(float(s.get('ndvi_fused') or 0), 3),
        round(float(s.get('height_change_m') or 0), 2),
        s.get('phenology_class', '') or '',
        round(float(s.get('confidence') or 0), 2),
        round(float(s.get('rf_confidence') or 0), 2),
    ]


def _enrich_parcels(js: dict, gpkg_path: Path) -> int:
    """Mutate js['parcels']['details'] in-place with per-parcel top10 lists.

    Returns the number of parcels enriched (those that had at least one segment).
    """
    import numpy as np
    from shapely import STRtree
    from shapely.geometry import Point

    obs_year = js.get('observation_period', {}).get('year') or 0
    pdetails = (js.get('parcels') or {}).get('details') or []
    if not pdetails:
        return 0

    seg_records, parcel_geoms = _load_segment_points_and_parcels(gpkg_path)
    if not seg_records or not parcel_geoms:
        log.warning('  empty segment_points or parcels in gpkg')
        return 0

    # Build STRtree of parcel polygons (WGS84) for centroid-in-polygon lookup.
    geoms = [g for _, g in parcel_geoms]
    pids = [pid for pid, _ in parcel_geoms]
    tree = STRtree(geoms)

    # parcel_id -> list of segment records
    by_parcel: dict[str, list[dict]] = {}
    for s in seg_records:
        try:
            pt = Point(s['_lon'], s['_lat'])
            idx = tree.nearest(pt)
            if geoms[idx].contains(pt):
                by_parcel.setdefault(pids[idx], []).append(s)
        except Exception:
            continue

    enriched = 0
    n_frav = 0
    for pd in pdetails:
        pid = pd.get('parcel_id', '')
        segs = by_parcel.get(pid, [])
        if not segs:
            # No segment-points fell inside the parcel — still try to
            # synthesise a frav from any pre-existing area_summary so
            # the parcel becomes queryable.
            as_ = pd.get('area_summary') or {}
            if as_:
                frav = {TYPE_LETTER.get(t, '?'): int(info.get('area_sqm', 0))
                        for t, info in as_.items() if info.get('area_sqm', 0) > 0}
                if frav:
                    pd['frav'] = frav
                    n_frav += 1
            continue
        # frav: prefer rasterised area_summary (more accurate); else aggregate segments.
        as_ = pd.get('area_summary') or {}
        if as_:
            frav = {TYPE_LETTER.get(t, '?'): int(info.get('area_sqm', 0))
                    for t, info in as_.items() if info.get('area_sqm', 0) > 0}
        else:
            agg = {}
            for s in segs:
                otype = s.get('type') or ''
                k = TYPE_LETTER.get(otype, '?')
                agg[k] = agg.get(k, 0) + int(float(s.get('area_sqm') or 0))
            frav = {k: v for k, v in agg.items() if v > 0}
        if frav:
            pd['frav'] = frav
            n_frav += 1
        # top objects by height_max
        top_objs = sorted(
            segs,
            key=lambda r: (float(r.get('height_max_m') or 0), float(r.get('area_sqm') or 0)),
            reverse=True,
        )[:TOP_N]
        if top_objs:
            pd['top_objs'] = [_seg_to_top_obj_compact(s) for s in top_objs]
        trees = [s for s in segs if (s.get('type') == 'tree')]
        if trees:
            top_trees = sorted(
                trees,
                key=lambda r: float(r.get('height_max_m') or 0),
                reverse=True,
            )[:TOP_N]
            pd['top_trees'] = [_seg_to_top_tree_compact(s) for s in top_trees]
        # Drop any legacy verbose form from a prior backfill run.
        pd.pop('top_10_objects', None)
        pd.pop('top_10_trees', None)
        enriched += 1
    log.info('  enriched %d parcels with top items, %d with frav', enriched, n_frav)
    return enriched


def _validate_enriched_json(js: dict, kg_code: str) -> tuple[bool, list[str]]:
    """Sanity-check an enriched JSON before re-uploading.

    Returns (ok, issues). Hard-fail if any of:
      - top-level structure dropped (lost top_10_objects, top_10_trees, top_by_type)
      - parcels.details count changed
      - 0 parcels enriched (no point in uploading)
      - any per-parcel top_10_objects entry has impossible values
        (negative area, negative height, height > 200m)
      - parcel had non-trivial area_summary but produced 0 top objects
        (would indicate STRtree miss — likely an EPSG/CRS bug)
      - dominant type 'tree' parcels with forested_fraction > 0.2
        but no top_10_trees
    Soft warnings are returned but don't block.
    """
    issues: list[str] = []
    pdetails = (js.get('parcels') or {}).get('details') or []
    if not pdetails:
        return False, ['no parcels.details']

    # Top-level invariants must be preserved.
    for k in ('top_10_objects', 'top_10_trees', 'top_by_type', 'kg_code', 'kg_name'):
        if k not in js:
            issues.append(f'top-level field missing: {k}')

    n_with_obj = sum(1 for p in pdetails if p.get('top_objs') or p.get('top_10_objects'))
    n_with_tree = sum(1 for p in pdetails if p.get('top_trees') or p.get('top_10_trees'))
    n_with_frav = sum(1 for p in pdetails if p.get('frav'))
    if n_with_obj == 0:
        issues.append('0 parcels have top_objs after enrichment')
    if n_with_frav == 0:
        issues.append('0 parcels have frav after enrichment')

    # Parcels with substantial area_summary should have a top object.
    # Use a generous threshold — segment_points are centroids of segments
    # which may straddle parcel boundaries, so small parcels with rasterised
    # area_summary won't always pick up a centroid.
    n_expected_obj = 0
    n_missing_obj = 0
    for p in pdetails:
        as_ = p.get('area_summary') or {}
        total_area = sum(v.get('area_sqm', 0) for v in as_.values())
        if total_area >= 500 and (p.get('area_sqm') or 0) >= 1000:
            n_expected_obj += 1
            if not (p.get('top_objs') or p.get('top_10_objects')):
                n_missing_obj += 1
    if n_expected_obj > 10 and n_missing_obj / n_expected_obj > 0.5:
        issues.append(
            f'{n_missing_obj}/{n_expected_obj} parcels with substantial area_summary '
            f'lack top_10_objects (>50%, likely spatial-join bug)'
        )

    # Tree-bearing parcels should have top_10_trees. Use area_summary['tree']
    # specifically (forested_fraction also includes shrub).
    n_tree_bearing = 0
    n_tree_bearing_no_trees = 0
    for p in pdetails:
        as_ = p.get('area_summary') or {}
        tree_area = (as_.get('tree') or {}).get('area_sqm', 0)
        if tree_area > 200:  # at least 200 m² of tree segments
            n_tree_bearing += 1
            if not (p.get('top_trees') or p.get('top_10_trees')):
                n_tree_bearing_no_trees += 1
    if n_tree_bearing > 5 and n_tree_bearing_no_trees / n_tree_bearing > 0.3:
        issues.append(
            f'{n_tree_bearing_no_trees}/{n_tree_bearing} tree-bearing parcels lack top_10_trees '
            f'(>30%, likely spatial-join bug)'
        )

    # Per-entry sanity bounds. Compact entries are positional arrays.
    bad_entries = 0
    for p in pdetails:
        for raw in (p.get('top_objs') or []):
            if isinstance(raw, list):
                h = raw[1] if len(raw) > 1 else 0
                a = raw[3] if len(raw) > 3 else 0
            else:
                h = raw.get('height_max_m', 0); a = raw.get('area_sqm', 0)
            if (h or 0) < 0 or (h or 0) > 200 or (a or 0) < 0:
                bad_entries += 1
        for raw in (p.get('top_trees') or []):
            if isinstance(raw, list):
                h = raw[0] if raw else 0
            else:
                h = raw.get('height_m', 0)
            if (h or 0) < 0 or (h or 0) > 100:
                bad_entries += 1
        # Legacy verbose form
        for o in (p.get('top_10_objects') or []):
            h = o.get('height_max_m') or 0; a = o.get('area_sqm') or 0
            if h < 0 or h > 200 or a < 0:
                bad_entries += 1
        for t in (p.get('top_10_trees') or []):
            h = t.get('height_m') or 0
            if h < 0 or h > 100:
                bad_entries += 1
    if bad_entries > 0:
        issues.append(f'{bad_entries} per-parcel entries have implausible values')

    # The KG-level top_10_trees should still be present and consistent.
    kg_top_trees = js.get('top_10_trees') or []
    parcel_max_tree = 0.0
    for p in pdetails:
        for raw in (p.get('top_trees') or []):
            h = raw[0] if isinstance(raw, list) and raw else (raw.get('height_m', 0) or 0)
            if h and h > parcel_max_tree:
                parcel_max_tree = h
        for t in (p.get('top_10_trees') or []):
            h = t.get('height_m') or 0
            if h > parcel_max_tree:
                parcel_max_tree = h
    if kg_top_trees and parcel_max_tree > 0:
        kg_max = kg_top_trees[0].get('height_m') or 0
        # Any parcel-level tree should be ≤ the KG-level max (within rounding).
        if parcel_max_tree > kg_max + 0.1:
            issues.append(
                f'parcel tree height {parcel_max_tree:.1f}m exceeds KG max {kg_max:.1f}m'
            )

    log.info('  validation: %d/%d w/objs, %d/%d w/trees, %d/%d w/frav, %d expected-obj missing, %d bad',
             n_with_obj, len(pdetails), n_with_tree, len(pdetails),
             n_with_frav, len(pdetails), n_missing_obj, bad_entries)

    # Hard-fail if any issue.
    return (len(issues) == 0), issues


def _save_json_atomic(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.bf_', suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _upload_replacement(client: Client, manifest: Manifest,
                        kg_code: str, json_path: Path) -> bool:
    """Replace the *_json file in the existing draft Zenodo deposition.

    Uses Client.upload() which automatically takes the update path when
    the manifest already has an entry with depo_id+bucket_url.
    """
    key = f'{kg_code}_json'
    if key not in manifest:
        log.info('  %s not in manifest yet; skipping upload', key)
        return False
    log.info('  uploading replacement %s -> Zenodo', json_path.name)
    # Read kg_name from the JSON itself for nicer metadata.
    try:
        kg_name = json.loads(json_path.read_text()).get('kg_name', '')
    except Exception:
        kg_name = ''
    def _meta(k, fn, ver):
        return landscape_metadata(
            kg_code=kg_code, kg_name=kg_name, version=ver, file_type='json',
        )
    try:
        client.upload(
            key=key,
            local_path=str(json_path),
            version='v1',
            meta_func=_meta,
            manifest=manifest,
        )
        return True
    except Exception as e:
        log.error('  upload failed for %s: %s', kg_code, e)
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--kg', help='process only this KG code (else: all)')
    p.add_argument('--dry-run', action='store_true',
                   help="don't write JSON, don't upload")
    p.add_argument('--no-upload', action='store_true',
                   help='write JSON locally but skip Zenodo replacement')
    p.add_argument('--force', action='store_true',
                   help='re-derive even if parcel already has top_10_objects')
    p.add_argument('--allow-warnings', action='store_true',
                   help='upload even if validation flags issues (logs them)')
    args = p.parse_args()

    manifest_obj = Manifest(MANIFEST_PATH)
    manifest_dict = {k: e.to_dict() for k, e in manifest_obj.entries().items()}

    json_files = sorted(JSON_DIR.glob('*.json'))
    if args.kg:
        json_files = [JSON_DIR / f'{args.kg}.json']
    if not json_files:
        log.error('no JSONs to process')
        return 1

    client = Client(token=DEFAULT_TOKEN) if not (args.dry_run or args.no_upload) else None

    n_total = n_done = n_skipped = n_failed = 0
    for jp in json_files:
        n_total += 1
        kg_code = jp.stem
        try:
            js = json.loads(jp.read_text())
        except Exception as e:
            log.error('%s: cannot parse JSON: %s', kg_code, e)
            n_failed += 1
            continue

        pdetails = (js.get('parcels') or {}).get('details') or []
        if not pdetails:
            log.info('%s: no parcels.details — skipping', kg_code)
            n_skipped += 1
            continue
        if not args.force and any((p.get('top_objs') or p.get('top_10_objects')) and p.get('frav') for p in pdetails):
            log.info('%s: already has per-parcel top_objs+frav — skipping (use --force to override)', kg_code)
            n_skipped += 1
            continue

        gpkg = _download_light_gpkg(manifest_dict, kg_code)
        if gpkg is None:
            n_failed += 1
            continue

        t0 = time.time()
        try:
            n_enriched = _enrich_parcels(js, gpkg)
        except Exception as e:
            log.exception('%s: enrich failed: %s', kg_code, e)
            n_failed += 1
            continue
        log.info('%s: enriched %d/%d parcels in %.1fs', kg_code, n_enriched, len(pdetails), time.time() - t0)
        if n_enriched == 0:
            log.warning('%s: 0 parcels enriched (no segments inside any parcel)', kg_code)
            n_skipped += 1
            continue

        # --- Validation gate (always runs, even in dry-run) ---
        ok, issues = _validate_enriched_json(js, kg_code)
        if not ok:
            log.error('%s: validation FAILED:', kg_code)
            for iss in issues:
                log.error('  - %s', iss)
            if not args.allow_warnings:
                log.error('%s: refusing to write/upload (use --allow-warnings to override)', kg_code)
                n_failed += 1
                continue
            log.warning('%s: proceeding despite issues (--allow-warnings)', kg_code)
        else:
            log.info('%s: validation passed', kg_code)

        if args.dry_run:
            n_done += 1
            continue

        _save_json_atomic(jp, js)

        if args.no_upload:
            n_done += 1
            continue

        if _upload_replacement(client, manifest_obj, kg_code, jp):
            n_done += 1
        else:
            n_failed += 1

    log.info('done: %d processed, %d skipped, %d failed (of %d)', n_done, n_skipped, n_failed, n_total)
    return 0 if n_failed == 0 else 2


if __name__ == '__main__':
    sys.exit(main())
