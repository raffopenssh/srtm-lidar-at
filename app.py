"""Austrian LIDAR & Orthophoto Analysis API.

Endpoints:
  1. POST /api/v1/elevation        — Enrich features with DSM/DTM elevation
  2. POST /api/v1/terrain          — Terrain characterisation (slope, ruggedness, …)
  3. POST /api/v1/changes          — Temporal change detection between ALS dates
  6. POST /api/v1/changes/trees    — Per-tree growth / felling analysis
  7. POST /api/v1/changes/summary  — Multi-epoch change summary
  8. GET  /api/v1/info             — Datasets, object types, event types
  9. GET  /api/v1/docs/llm.txt     — Machine-readable API reference
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import pickle
import sys
import re
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from flask import Flask, request, jsonify, send_file, send_from_directory, Response, redirect
from shapely.geometry import mapping, shape, Point, LineString as SLineString

import tile_index as ti
import raster_io
import terrain_analysis as ta
import object_segmentation as seg  # watershed-based segmentation
import hansen  # Hansen Global Forest Change calibration
import temporal_analysis as tca
import geo_parse
import search_index as si
import cadastre_bridge as cb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='')

# Initialize search index + watch for new KG JSON files
def _init_search_index():
    try:
        idx = si.init_index()
        # Watch for new JSON files every 60s
        json_dir = Path('data/austria_processor/json')
        known = set(f.stem for f in json_dir.glob('*.json')) if json_dir.exists() else set()
        while True:
            time.sleep(60)
            try:
                if not json_dir.exists():
                    continue
                current = set(f.stem for f in json_dir.glob('*.json'))
                new_kgs = current - known
                if new_kgs:
                    log.info('🔍 New KGs detected: %s, rebuilding index', new_kgs)
                    idx.build()
                    known = current
            except Exception as e:
                log.warning('Search index watch: %s', e)
    except Exception as e:
        log.warning('Search index init failed: %s', e)
threading.Thread(target=_init_search_index, daemon=True).start()


def _get_peer_urls():
    """Read peer URLs from peer_urls.txt or from systemd service --peers flag."""
    pf = Path('data/austria_processor/peer_urls.txt')
    if pf.exists():
        return [u.strip() for u in pf.read_text().splitlines() if u.strip() and not u.startswith('#')]
    # Fallback: parse from systemd service / drop-in configs
    svc_paths = list(Path('/etc/systemd/system/austria_processor.service.d').glob('*.conf')) \
        if Path('/etc/systemd/system/austria_processor.service.d').exists() else []
    svc_paths.append(Path('/etc/systemd/system/austria_processor.service'))
    for svc in svc_paths:
        try:
            if not svc.exists():
                continue
            for line in svc.read_text().splitlines():
                if '--peers' in line:
                    parts = line.split('--peers')[1].strip().split()
                    urls = []
                    for p in parts:
                        if p.startswith('http'):
                            urls.append(p)
                        elif p.startswith('--'):
                            break
                    if urls:
                        return urls
        except PermissionError:
            continue
    return []


def _sync_peer_data():
    """Background thread: sync KG JSONs and manifest entries from peers."""
    import requests as req
    time.sleep(30)  # Wait for startup

    json_dir = Path('data/austria_processor/json')
    json_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path('data/austria_processor/zenodo_manifest.json')

    while True:
        try:
            peer_urls = _get_peer_urls()
            if not peer_urls:
                time.sleep(300)
                continue

            new_count = 0
            merged_manifest_entries = {}

            for peer_url in peer_urls:
                try:
                    r = req.get(peer_url.rstrip('/') + '/api/v1/processing/peers', timeout=15)
                    r.raise_for_status()
                    peer_data = r.json()
                except Exception as e:
                    log.debug('Peer sync: %s unreachable: %s', peer_url, e)
                    continue

                peer_manifest = peer_data.get('manifest', {})

                # Download KG JSONs we don't have
                for key, entry in peer_manifest.items():
                    if not key.endswith('_json'):
                        continue
                    code = key.replace('_json', '')
                    local_path = json_dir / f'{code}.json'
                    if local_path.exists():
                        continue

                    # Construct download URL from bucket_url + filename
                    link = entry.get('link', '')
                    if not link and entry.get('bucket_url') and entry.get('filename'):
                        link = f"{entry['bucket_url']}/{entry['filename']}"
                    if not link:
                        continue

                    try:
                        jr = req.get(link, timeout=120, stream=True)
                        jr.raise_for_status()
                        # Atomic write: temp file then rename
                        tmp_path = local_path.with_suffix('.tmp')
                        with open(tmp_path, 'wb') as f:
                            for chunk in jr.iter_content(chunk_size=65536):
                                f.write(chunk)
                        tmp_path.rename(local_path)
                        new_count += 1
                        log.info('Peer sync: downloaded %s.json (%s bytes) from peer',
                                 code, local_path.stat().st_size)
                    except Exception as e:
                        # Clean up partial download
                        local_path.with_suffix('.tmp').unlink(missing_ok=True)
                        log.warning('Peer sync: failed to download %s from %s: %s', code, link, e)

                # Collect manifest entries to merge
                for key, entry in peer_manifest.items():
                    if key not in merged_manifest_entries:
                        merged_manifest_entries[key] = entry

            # Merge into local manifest (atomic read-modify-write)
            if merged_manifest_entries:
                try:
                    local_manifest = {}
                    if manifest_path.exists():
                        md = json.loads(manifest_path.read_text())
                        local_manifest = md.get('entries', md)

                    added = 0
                    for key, entry in merged_manifest_entries.items():
                        if key not in local_manifest:
                            local_manifest[key] = entry
                            added += 1

                    if added > 0:
                        # Atomic write: temp file then rename (same pattern as Manifest.save())
                        import tempfile as _tf
                        manifest_path.parent.mkdir(parents=True, exist_ok=True)
                        fd, tmp = _tf.mkstemp(dir=manifest_path.parent, suffix='.tmp', prefix='.manifest_')
                        try:
                            with os.fdopen(fd, 'w') as f:
                                json.dump({'entries': local_manifest}, f, indent=2, sort_keys=True)
                            os.replace(tmp, manifest_path)
                        except BaseException:
                            try: os.unlink(tmp)
                            except OSError: pass
                            raise
                        log.info('Peer sync: merged %d manifest entries from peers', added)
                except Exception as e:
                    log.warning('Peer sync: manifest merge failed: %s', e)

            if new_count > 0:
                log.info('Peer sync: %d new KG JSONs downloaded, triggering index rebuild', new_count)
                try:
                    idx = si.get_index()
                    idx.build()
                except Exception as e:
                    log.warning('Peer sync: index rebuild failed: %s', e)

        except Exception as e:
            log.warning('Peer sync error: %s', e)

        time.sleep(300)  # Every 5 minutes

threading.Thread(target=_sync_peer_data, daemon=True, name='peer-sync').start()

MAX_AREA_SQM = 25_000_000  # 25 km²

# === SECTION: Processing queue (semaphore for concurrent heavy tasks) ===
_TASK_SEMAPHORE = threading.Semaphore(2)   # max 2 concurrent heavy tasks
_TASK_QUEUE_LOCK = threading.Lock()
_TASK_QUEUE_SIZE = 0  # current waiting count
MAX_QUEUE_SIZE = 4    # reject if more than 4 waiting


def _height_class(h):
    """Classify height (m) into a forestry-relevant height class string."""
    if h is None or h < 0.5:
        return 'ground'
    if h < 2:
        return 'low (<2m)'
    if h < 5:
        return 'shrub (2-5m)'
    if h < 10:
        return 'young (5-10m)'
    if h < 15:
        return 'pole (10-15m)'
    if h < 20:
        return 'mid (15-20m)'
    if h < 25:
        return 'mature (20-25m)'
    if h < 30:
        return 'tall (25-30m)'
    if h < 35:
        return 'very tall (30-35m)'
    if h < 40:
        return 'old growth (35-40m)'
    return 'old growth (40m+)'

def _parse_height_filter(params: dict):
    """Parse height filter params. Returns a filter function or None.

    Supports:
      height_min=X          → height >= X  (shorthand: height_op=gt)
      height_max=X          → height <= X  (shorthand: height_op=lt)
      height_min=X&height_max=Y → X <= height <= Y  (shorthand: height_op=between)
      height_op=gt&height_min=X   → height > X  (alias for consistency)
      height_op=lt&height_max=X   → height < X
      height_op=between&height_min=X&height_max=Y
    """
    h_min = params.get('height_min')
    h_max = params.get('height_max')
    h_op = (params.get('height_op') or '').lower().strip()

    if h_min is not None:
        try:
            h_min = float(h_min)
        except (ValueError, TypeError):
            h_min = None
    if h_max is not None:
        try:
            h_max = float(h_max)
        except (ValueError, TypeError):
            h_max = None

    if h_min is None and h_max is None:
        return None

    # Infer operator from which params are present
    if not h_op:
        if h_min is not None and h_max is not None:
            h_op = 'between'
        elif h_min is not None:
            h_op = 'gt'
        elif h_max is not None:
            h_op = 'lt'

    if h_op == 'gt' and h_min is not None:
        return lambda h: (h or 0) >= h_min
    elif h_op == 'lt' and h_max is not None:
        return lambda h: (h or 0) <= h_max
    elif h_op == 'between' and h_min is not None and h_max is not None:
        return lambda h: h_min <= (h or 0) <= h_max
    return None


def _apply_height_filter_features(features: list, params: dict) -> list:
    """Apply height filter to a list of GeoJSON feature dicts."""
    hf = _parse_height_filter(params)
    if not hf:
        return features
    return [f for f in features
            if hf(f.get('properties', {}).get('height_max_m')
                  or f.get('properties', {}).get('height_after_m') or 0)]


def _apply_height_filter_objects(objects: list, params: dict) -> list:
    """Apply height filter to a list of segment objects (with .height_max)."""
    hf = _parse_height_filter(params)
    if not hf:
        return objects
    return [o for o in objects if hf(o.height_max)]


# === SECTION: Async task system (file-backed progress + result storage) ===
_PROGRESS_DIR = Path('/tmp/segment_progress')
_PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
_RESULTS_DIR = Path('/tmp/segment_results')
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _progress_set(task_id: str, step: str, detail: str = ""):
    """Update progress for a running segment task."""
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    try:
        t0 = 0.0
        if p.exists():
            try:
                t0 = json.loads(p.read_text()).get('t0', 0.0)
            except Exception:
                pass
        p.write_text(json.dumps(dict(step=step, detail=detail, t0=t0,
                                     updated=time.time())))
    except Exception:
        pass

def _progress_start(task_id: str):
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    p.write_text(json.dumps(dict(step='starting', detail='',
                                 t0=time.time(), updated=time.time())))

def _progress_done(task_id: str, auto_share_id: str = None):
    """Mark task as completed (keeps file so polling sees 'done')."""
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    try:
        t0 = 0.0
        if p.exists():
            try:
                t0 = json.loads(p.read_text()).get('t0', 0.0)
            except Exception:
                pass
        d = dict(step='done', detail='', t0=t0, updated=time.time())
        if auto_share_id:
            d['auto_share_id'] = auto_share_id
        p.write_text(json.dumps(d))
    except Exception:
        pass

def _progress_error(task_id: str, error: str):
    """Mark task as failed."""
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    try:
        t0 = 0.0
        if p.exists():
            try:
                t0 = json.loads(p.read_text()).get('t0', 0.0)
            except Exception:
                pass
        p.write_text(json.dumps(dict(step='error', detail=error,
                                     t0=t0, updated=time.time())))
    except Exception:
        pass

def _progress_end(task_id: str):
    """Clean up progress file (legacy, used for non-async)."""
    if not task_id:
        return
    p = _PROGRESS_DIR / f"{task_id}.json"
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

def _store_result(task_id: str, result: dict):
    """Store JSON result for async retrieval."""
    p = _RESULTS_DIR / f"{task_id}.json.gz"
    data = json.dumps(result).encode()
    with gzip.open(str(p), 'wb') as f:
        f.write(data)

def _get_result(task_id: str) -> dict | None:
    """Retrieve stored result, or None."""
    p = _RESULTS_DIR / f"{task_id}.json.gz"
    if not p.exists():
        return None
    with gzip.open(str(p), 'rb') as f:
        return json.loads(f.read())

def _cleanup_old_results(max_age_s: int = 14400):
    """Remove results older than max_age_s."""
    try:
        cutoff = time.time() - max_age_s
        for f in _RESULTS_DIR.glob('*.json.gz'):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
        for f in _RESULTS_DIR.glob('*.gpkg'):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
        for f in _PROGRESS_DIR.glob('*.json'):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
        for f in _QUERY_RESULTS_DIR.glob('*.json.gz'):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except Exception:
        pass

@app.route('/api/v1/segment/progress')
def segment_progress():
    """Poll progress of a running segment task.

    Returns {active, step, detail, elapsed, done, error}.
    When step=='done', the result is available at /api/v1/segment/result?task_id=...
    When step=='error', detail contains the error message.
    """
    task_id = request.args.get('task_id', '')
    p = _PROGRESS_DIR / f"{task_id}.json"
    if not task_id or not p.exists():
        # Check if result already exists (progress file cleaned up)
        if task_id and (_RESULTS_DIR / f"{task_id}.json.gz").exists():
            return jsonify(dict(active=False, step='done', detail='', elapsed=0, done=True, error=None))
        return jsonify(dict(active=False, step='', detail='', elapsed=0, done=False, error=None))
    try:
        info = json.loads(p.read_text())
        step = info.get('step', '')
        elapsed = round(time.time() - info.get('t0', time.time()), 1)
        is_done = step == 'done'
        is_error = step == 'error'
        return jsonify(dict(
            active=not is_done and not is_error,
            step=step,
            detail=info.get('detail', ''),
            elapsed=elapsed,
            done=is_done,
            error=info.get('detail', '') if is_error else None,
            auto_share_id=info.get('auto_share_id'),
        ))
    except Exception:
        return jsonify(dict(active=False, step='', detail='', elapsed=0, done=False, error=None))


@app.route('/api/v1/segment/abort', methods=['POST'])
def segment_abort():
    """Abort a running segment task. Marks it as cancelled."""
    task_id = request.args.get('task_id', '') or (request.get_json(silent=True) or {}).get('task_id', '')
    if not task_id:
        return _error('task_id required')
    p = _PROGRESS_DIR / f"{task_id}.json"
    if p.exists():
        try:
            info = json.loads(p.read_text())
            if info.get('step') not in ('done', 'error'):
                _progress_error(task_id, 'Cancelled by user')
        except Exception:
            _progress_error(task_id, 'Cancelled by user')
    return jsonify({"ok": True, "task_id": task_id})


@app.route('/api/v1/segment/result')
def segment_result():
    """Retrieve the result of an async segment task."""
    task_id = request.args.get('task_id', '')
    if not task_id:
        return _error('task_id required')
    result = _get_result(task_id)
    if result is None:
        return _error('Result not found or not ready', 404)
    # Don't delete after retrieval — let _cleanup_old_results() handle it
    return jsonify(result)


@app.route('/api/v1/training/status')
def training_status():
    """Return RF training job status: running, current KG, model info, resource usage.

    Uses cgroup/proc filesystem reads instead of subprocess to avoid
    fork failures when the srv cgroup is under memory pressure.
    """
    import re, pathlib

    result = dict(running=False, current_kg=None, progress=None,
                  model=None, pid=None, ram_mb=None,
                  service_state=None)

    # Check rf_train service state via cgroup filesystem (no subprocess needed)
    cgroup_base = pathlib.Path('/sys/fs/cgroup/system.slice/rf_train.service')
    try:
        events_text = (cgroup_base / 'cgroup.events').read_text()
        populated = 'populated 1' in events_text
        frozen = 'frozen 1' in events_text
        if populated and not frozen:
            result['running'] = True
            result['service_state'] = 'active'
        elif populated and frozen:
            result['running'] = True
            result['service_state'] = 'activating'  # possibly restarting
        else:
            result['service_state'] = 'inactive'
    except FileNotFoundError:
        result['service_state'] = 'not-found'
    except Exception:
        pass

    # Get memory usage and PID from cgroup/proc (no subprocess needed)
    try:
        mem_bytes = int((cgroup_base / 'memory.current').read_text().strip())
        result['ram_mb'] = round(mem_bytes / (1024 * 1024))
    except Exception:
        pass
    try:
        pids = (cgroup_base / 'cgroup.procs').read_text().strip().splitlines()
        for pid_str in pids:
            pid = int(pid_str)
            # Read null-separated args; first arg is the executable
            args = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode('utf-8', errors='replace').split('\x00')
            if len(args) >= 2 and 'python' in args[0] and 'train_rf_4000kg' in args[1]:
                result['pid'] = pid
                break
    except Exception:
        pass

    # Check oom_kill events for this cgroup
    try:
        mem_events = (cgroup_base / 'memory.events').read_text()
        for line in mem_events.splitlines():
            if line.startswith('oom_kill '):
                oom_kills = int(line.split()[1])
                if oom_kills > 0:
                    result['oom_kills'] = oom_kills
    except Exception:
        pass

    # Parse last log lines for current KG, progress, and last activity
    log_path = pathlib.Path('/tmp/rf_train_4000kg.log')
    if log_path.exists():
        try:
            # Read last 32KB of log (enough to find current KG line)
            with open(log_path, 'rb') as f:
                f.seek(max(0, f.seek(0, 2) - 32768))
                tail = f.read().decode('utf-8', errors='replace')
            lines = tail.strip().splitlines()
            # Find last "Processing KG" line
            for line in reversed(lines):
                m = re.search(r'\[(\d+)/(\d+)\]\s+Processing KG (\d+)\s+\(([^)]+)\)', line)
                if m:
                    result['current_kg'] = dict(
                        index=int(m.group(1)), total=int(m.group(2)),
                        kg_code=m.group(3), kg_name=m.group(4))
                    result['progress'] = f"{m.group(1)}/{m.group(2)}"
                    break
            # Last log line with timestamp → current step
            for line in reversed(lines):
                tm = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if tm:
                    result['last_log_time'] = tm.group(1)
                    # Extract step info from the line
                    step = re.search(r'(Step \d+\S*:.*?)$', line)
                    if step:
                        result['current_step'] = step.group(1).strip()[:80]
                    else:
                        # Just the message part after the log prefix
                        msg = re.sub(r'^.*?(INFO|WARNING|ERROR)\s+\S+:\s*', '', line)
                        if msg:
                            result['current_step'] = msg.strip()[:80]
                    break
        except Exception:
            pass

    # Checkpoint count + failed KGs + next model checkpoint
    MODEL_CP_INTERVAL = 10
    ckpt_dir = pathlib.Path('/home/exedev/srtm-lidar/rf_training_data/checkpoints')
    n_success = 0
    if ckpt_dir.exists():
        n_success = len(list(ckpt_dir.glob('kg_*.npz')))
        result['n_checkpoints'] = n_success
    failed_file = pathlib.Path('/home/exedev/srtm-lidar/rf_training_data/failed_kgs.txt')
    n_fail = 0
    if failed_file.exists():
        failed = [l.strip() for l in failed_file.read_text().strip().splitlines() if l.strip()]
        n_fail = len(failed)
        result['n_failed_kgs'] = n_fail
        result['failed_kgs'] = failed
    result['n_completed'] = n_success + n_fail
    next_cp = ((n_success // MODEL_CP_INTERVAL) + 1) * MODEL_CP_INTERVAL
    result['next_model_checkpoint'] = next_cp
    result['to_next_model'] = max(0, next_cp - n_success)

    # Credits paused?
    pause_file = pathlib.Path('/home/exedev/srtm-lidar/rf_training_data/credits_paused.txt')
    if pause_file.exists():
        result['credits_paused'] = True
        result['credits_paused_since'] = pause_file.read_text().strip().split('\n')[0]
    else:
        result['credits_paused'] = False

    # Model info (prefer best_model from curve eval)
    from learned_classifier import BEST_META_PATH
    active_meta_path = BEST_META_PATH if BEST_META_PATH.exists() else pathlib.Path('/tmp/learned_classifier/rf_meta.json')
    if active_meta_path.exists():
        try:
            meta = json.loads(active_meta_path.read_text())
            result['model'] = dict(
                oob_score=round(meta.get('oob_score', 0), 4),
                composite_score=round(meta.get('composite_score', 0), 4),
                n_train=meta.get('n_train', 0),
                n_kgs=meta.get('n_kgs', 0),
                best_seed=meta.get('best_seed'),
                n_classes=len(meta.get('classes', [])),
                trained_at=meta.get('trained_at', ''),
                source='best_model' if active_meta_path == BEST_META_PATH else 'live')
        except Exception:
            pass

    # OOB learning curve (from checkpoint evaluation, multi-seed)
    curve_path = pathlib.Path('data/oob_curve.csv')
    if curve_path.exists():
        try:
            import csv as _csv
            from collections import defaultdict
            import statistics
            by_kgs = defaultdict(list)
            with open(curve_path) as _f:
                reader = _csv.DictReader(_f)
                for r in reader:
                    n_kgs = int(r['n_kgs'])
                    by_kgs[n_kgs].append({
                        'oob': float(r['oob']),
                        'composite': float(r['composite']) if 'composite' in r and r['composite'] else float(r['oob']),
                        'mean_class_oob': float(r['mean_class_oob']) if r.get('mean_class_oob') else None,
                        'min_class_oob': float(r['min_class_oob']) if r.get('min_class_oob') else None,
                        'n_samples': int(r['n_samples']),
                        'n_classes': int(r['n_classes']),
                    })
            curve = []
            for n_kgs in sorted(by_kgs):
                runs = by_kgs[n_kgs]
                oobs = [r['oob'] for r in runs]
                comps = [r['composite'] for r in runs]
                mean_cls = [r['mean_class_oob'] for r in runs if r['mean_class_oob'] is not None]
                min_cls = [r['min_class_oob'] for r in runs if r['min_class_oob'] is not None]
                entry = {
                    'n_kgs': n_kgs,
                    'n_samples': runs[0]['n_samples'],
                    'n_classes': runs[0]['n_classes'],
                    'n_seeds': len(oobs),
                    'oob_median': round(statistics.median(oobs), 6),
                    'oob_min': round(min(oobs), 6),
                    'oob_max': round(max(oobs), 6),
                    'oob_all': sorted(round(o, 6) for o in oobs),
                    'composite_median': round(statistics.median(comps), 6),
                    'composite_min': round(min(comps), 6),
                    'composite_max': round(max(comps), 6),
                    'composite_all': sorted(round(c, 6) for c in comps),
                }
                if mean_cls:
                    entry['mean_class_median'] = round(statistics.median(mean_cls), 6)
                if min_cls:
                    entry['min_class_median'] = round(statistics.median(min_cls), 6)
                curve.append(entry)
            result['oob_curve'] = curve
        except Exception:
            pass

    # Live history (monitor cron)
    hist_path = pathlib.Path('data/oob_history.csv')
    if hist_path.exists():
        try:
            import csv as _csv
            with open(hist_path) as _f:
                reader = _csv.DictReader(_f)
                result['oob_history'] = [dict(r) for r in reader]
        except Exception:
            pass

    # Monitor state (convergence tracking)
    state_path = pathlib.Path('data/monitor_state.json')
    if state_path.exists():
        try:
            result['monitor'] = json.loads(state_path.read_text())
        except Exception:
            pass

    # Curve detail (per-checkpoint expanded info from JSONL)
    detail_path = pathlib.Path('data/oob_curve_detail.jsonl')
    if detail_path.exists():
        try:
            import statistics
            from collections import defaultdict
            # Parse JSONL: group by n_kgs
            by_kgs = defaultdict(list)
            with open(detail_path) as _f:
                for line in _f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        by_kgs[int(obj['n_kgs'])].append(obj)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            curve_detail = {}
            for n_kgs in sorted(by_kgs):
                entries = by_kgs[n_kgs]
                n_seeds = len(entries)
                # Hyperparams (same for all seeds, take first)
                entry0 = entries[0]
                # Median feature importances across seeds
                all_feat_keys = set()
                for e in entries:
                    all_feat_keys.update(e.get('all_importances', {}).keys())
                feat_medians = {}
                for fk in all_feat_keys:
                    vals = [e['all_importances'][fk] for e in entries
                            if fk in e.get('all_importances', {})]
                    if vals:
                        feat_medians[fk] = round(statistics.median(vals), 6)
                # Median per-class OOB across seeds (only where class present)
                all_classes = set()
                for e in entries:
                    all_classes.update(e.get('per_class_oob', {}).keys())
                class_medians = {}
                classes_present = sorted(all_classes)
                for cls in all_classes:
                    vals = [e['per_class_oob'][cls] for e in entries
                            if cls in e.get('per_class_oob', {})]
                    if vals:
                        class_medians[cls] = round(statistics.median(vals), 4)
                # Composite stats across seeds
                comp_vals = [e.get('composite', e.get('oob', 0)) for e in entries]
                mean_cls_vals = [e.get('mean_class_oob') for e in entries if e.get('mean_class_oob') is not None]
                min_cls_vals = [e.get('min_class_oob') for e in entries if e.get('min_class_oob') is not None]
                curve_detail[str(n_kgs)] = {
                    'n_estimators': entry0.get('n_estimators', 200),
                    'max_depth': entry0.get('max_depth', 20),
                    'min_samples_leaf': entry0.get('min_samples_leaf', 5),
                    'feature_importances': feat_medians,
                    'per_class_oob': class_medians,
                    'classes_present': classes_present,
                    'n_seeds': n_seeds,
                    'composite': round(statistics.median(comp_vals), 6) if comp_vals else None,
                    'mean_class_oob': round(statistics.median(mean_cls_vals), 6) if mean_cls_vals else None,
                    'min_class_oob': round(statistics.median(min_cls_vals), 6) if min_cls_vals else None,
                }
            result['curve_detail'] = curve_detail

            # Expose the exact deployed model's detail (specific seed)
            if result.get('model') and result['model'].get('best_seed') is not None:
                dep_kgs = result['model']['n_kgs']
                dep_seed = result['model']['best_seed']
                for e in by_kgs.get(dep_kgs, []):
                    if e.get('seed') == dep_seed:
                        result['deployed_detail'] = {
                            'n_kgs': dep_kgs,
                            'seed': dep_seed,
                            'n_samples': e.get('n_samples'),
                            'n_classes': e.get('n_classes'),
                            'oob': e.get('oob'),
                            'composite': e.get('composite'),
                            'mean_class_oob': e.get('mean_class_oob'),
                            'min_class_oob': e.get('min_class_oob'),
                            'n_estimators': e.get('n_estimators', 200),
                            'max_depth': e.get('max_depth', 20),
                            'min_samples_leaf': e.get('min_samples_leaf', 5),
                            'per_class_oob': e.get('per_class_oob', {}),
                            'feature_importances': e.get('all_importances', {}),
                            'classes_present': sorted(e.get('per_class_oob', {}).keys()),
                        }
                        break
        except Exception:
            pass

    return jsonify(result)


def _is_curve_eval_running():
    """Check lockfile to see if any curve eval (cron or triggered) is running."""
    import fcntl
    from pathlib import Path
    lockfile = Path('/tmp/rf_curve_eval.lock')
    if not lockfile.exists():
        return False
    try:
        fd = os.open(str(lockfile), os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except (OSError, IOError):
            return True
        finally:
            os.close(fd)
    except FileNotFoundError:
        return False


@app.route('/api/v1/training/evaluate', methods=['POST'])
def trigger_curve_eval():
    """Trigger OOB curve re-evaluation in background."""
    if _is_curve_eval_running():
        return jsonify(running=True, msg='Already running (cron or previous trigger)'), 409
    import subprocess
    # Detach from gunicorn process group so it survives service restarts
    subprocess.Popen(
        ['python3', 'evaluate_checkpoints.py', 'curve', '--step', '5'],
        cwd='/home/exedev/srtm-lidar',
        start_new_session=True,
        stdout=open('/tmp/rf_curve_eval.log', 'a'),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    return jsonify(running=True, msg='Curve evaluation started')


@app.route('/api/v1/training/evaluate', methods=['GET'])
def curve_eval_status():
    """Check if curve evaluation is running, with progress detail."""
    running = _is_curve_eval_running()
    info = {'running': running}
    if running:
        # Read CSV to figure out progress (adaptive seed count)
        try:
            import csv
            from pathlib import Path
            from evaluate_checkpoints import seeds_for_n_kgs
            csv_path = Path('data/oob_curve.csv')
            n_ckpt = len(list(Path('rf_training_data/checkpoints').glob('kg_*.npz')))
            steps = list(range(5, n_ckpt + 1, 5))
            all_combos = [(n, s) for n in steps for s in seeds_for_n_kgs(n)]
            existing = set()
            if csv_path.exists():
                with open(csv_path) as f:
                    for r in csv.DictReader(f):
                        existing.add((int(r['n_kgs']), int(r.get('seed', 0))))
            work = [(n, s) for n, s in all_combos if (n, s) not in existing]
            done_combos = len(all_combos) - len(work)
            if work:
                info['current_kgs'] = work[0][0]
                info['current_seed'] = work[0][1]
            info['done'] = done_combos
            info['total'] = len(all_combos)
        except Exception:
            pass
    return jsonify(info)


@app.route('/api/v1/training/evaluate/stop', methods=['POST'])
def stop_curve_eval():
    """Kill running curve evaluation subprocess."""
    import signal
    if not _is_curve_eval_running():
        return jsonify(running=False, msg='Not running')
    # Find the evaluate_checkpoints process and kill its process group
    try:
        import subprocess
        result = subprocess.run(
            ['pgrep', '-f', 'evaluate_checkpoints.py curve'],
            capture_output=True, text=True
        )
        for pid in result.stdout.strip().split('\n'):
            if pid:
                os.kill(int(pid), signal.SIGTERM)
        return jsonify(running=False, msg='Stopped')
    except Exception as e:
        return jsonify(running=_is_curve_eval_running(), msg=str(e)), 500


# === SECTION: Austria Processor endpoints (proxy to processor state) ===

_processor_process = None  # subprocess.Popen for the processor

def _is_processor_running() -> bool:
    """Check if Austria processor is actively running (systemd or subprocess)."""
    try:
        p = Path('data/austria_processor/progress.json')
        if p.exists():
            d = json.loads(p.read_text())
            if d.get('state') == 'running':
                return True
    except Exception:
        pass
    return False

# Git commit hash (read once at startup)
try:
    import subprocess as _sp
    _GIT_COMMIT = _sp.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        cwd=str(Path(__file__).parent), stderr=_sp.DEVNULL
    ).decode().strip()
except Exception:
    _GIT_COMMIT = 'unknown'

@app.route('/api/v1/processing/status')
def processing_status():
    """Return Austria processor progress (read from progress.json)."""
    progress_file = Path('data/austria_processor/progress.json')
    if not progress_file.exists():
        return jsonify({
            'state': 'idle', 'total_kgs': 0, 'completed': 0,
            'success': 0, 'failed': 0, 'uploaded': 0,
            'upload_size_bytes': 0, 'current_kg': None,
            'rate_kgs_per_hour': 0, 'avg_seconds_per_kg': 0,
            'elapsed_seconds': 0, 'eta_seconds': 0,
            'recent_log': [], 'failed_kgs': [],
            'parcels_total': 0, 'buildings_total': 0, 'started_at': None,
        })
    try:
        data = json.loads(progress_file.read_text())
        # Check if processor is actually running
        global _processor_process
        if _processor_process is not None and _processor_process.poll() is not None:
            data['state'] = 'stopped'
            _processor_process = None
        # Always add fresh system metrics (even if processor writes them too)
        if 'system' not in data:
            data['system'] = {}
        try:
            import pathlib as _pl
            # Live RAM
            mi = open('/proc/meminfo').read()
            mt = ma = 0
            for line in mi.splitlines():
                if line.startswith('MemTotal:'):
                    mt = int(line.split()[1]) // 1024
                elif line.startswith('MemAvailable:'):
                    ma = int(line.split()[1]) // 1024
            if mt:
                data['system']['ram_total_mb'] = mt
                data['system']['ram_used_mb'] = mt - ma
                data['system']['ram_pct'] = round(100 * (mt - ma) / mt, 1)
            # Disk
            st = os.statvfs('/')
            fg = (st.f_bavail * st.f_frsize) / (1024**3)
            tg = (st.f_blocks * st.f_frsize) / (1024**3)
            data['system']['disk_free_gb'] = round(fg, 1)
            data['system']['disk_used_pct'] = round(100*(1 - fg/tg), 1)
            # CPU
            data['system']['cpu_pct'] = round(100 * os.getloadavg()[0] / max(os.cpu_count() or 1, 1), 1)
            # Processor PID check — try managed process first, then find via systemd/pidfile
            _proc_pid = None
            if _processor_process and _processor_process.poll() is None:
                _proc_pid = _processor_process.pid
            else:
                # Find processor started via systemd
                try:
                    import subprocess as _sp
                    _pids = _sp.check_output(
                        ['pgrep', '-f', 'austria_processor.py'],
                        text=True, timeout=2
                    ).strip().split('\n')
                    if _pids and _pids[0]:
                        _proc_pid = int(_pids[0])
                except Exception:
                    pass
            if _proc_pid:
                data['system']['proc_pid'] = _proc_pid
                try:
                    rss_kb = int(open(f'/proc/{_proc_pid}/status').read().split('VmRSS:')[1].split()[0])
                    data['system']['proc_ram_mb'] = rss_kb // 1024
                except Exception:
                    pass
            # Tile caches
            try:
                from tile_cache import cache_summary
                data['system']['tile_caches'] = cache_summary()
            except Exception:
                pass
            # Zenodo persistent cache
            try:
                from zenodo_cache import ZenodoCache
                data['system']['zenodo_cache'] = ZenodoCache().status()
            except Exception:
                pass
            # Manifest summary
            mf = _pl.Path('data/austria_processor/zenodo_manifest.json')
            if mf.exists():
                md = json.loads(mf.read_text())
                ents = md.get('entries', {})
                data['manifest'] = {
                    'count': len(ents),
                    'total_size_bytes': sum(e.get('size', 0) for e in ents.values()),
                }
        except Exception:
            pass
        data['git_commit'] = _GIT_COMMIT
        # Include persisted tile history for all completed/failed KGs
        try:
            th_path = Path('data/austria_processor/tile_history.json')
            if th_path.exists():
                data['tile_history'] = json.loads(th_path.read_text())
        except Exception:
            pass
        data['throttle'] = Path('data/austria_processor/upload_throttle').exists()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/processing/throttle', methods=['GET', 'POST'])
def processing_throttle():
    """Toggle bandwidth throttle mode (skip GPKG uploads to Zenodo)."""
    throttle_file = Path('data/austria_processor/upload_throttle')
    if request.method == 'GET':
        return jsonify({'throttle': throttle_file.exists()})
    # POST toggles
    if throttle_file.exists():
        throttle_file.unlink()
        return jsonify({'throttle': False, 'message': 'Throttle disabled \u2014 full uploads resume'})
    else:
        throttle_file.write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ'))
        return jsonify({'throttle': True, 'message': 'Throttle enabled \u2014 skipping GPKG uploads'})


@app.route('/api/v1/processing/peers', methods=['GET'])
def processing_peers_status():
    """Return compact state for peer coordination.

    Returns completed KG codes, current KG, priority queue, and failed KGs
    so a peer instance can avoid duplicate work.
    """
    data_dir = Path('data/austria_processor')
    result = {'instance': os.environ.get('INSTANCE_ID', 'primary')}

    # Completed KGs
    result['completed'] = sorted(_get_completed_kgs())

    # Current KG being processed
    current = None
    pf = data_dir / 'progress.json'
    if pf.exists():
        try:
            pd = json.loads(pf.read_text())
            ckg = pd.get('current_kg') or {}
            current = ckg.get('code')
        except Exception:
            pass
    result['current'] = current

    # In-progress marker (crash recovery file)
    ipf = data_dir / 'in_progress_kg.txt'
    if ipf.exists():
        try:
            ip = ipf.read_text().strip()
            if ip and ip != current:
                result['in_progress'] = ip
        except Exception:
            pass

    # Priority queue
    retry_path = data_dir / 'retry_queue.json'
    if retry_path.exists():
        try:
            result['priority'] = json.loads(retry_path.read_text())
        except Exception:
            result['priority'] = []
    else:
        result['priority'] = []

    # Failed KGs (permanent)
    failed_path = data_dir / 'failed_kgs.json'
    if failed_path.exists():
        try:
            result['failed'] = json.loads(failed_path.read_text())
        except Exception:
            result['failed'] = []
    else:
        result['failed'] = []

    # Manifest entries (Zenodo URLs) — so peers can discover our uploads
    manifest_path = data_dir / 'zenodo_manifest.json'
    if manifest_path.exists():
        try:
            md = json.loads(manifest_path.read_text())
            result['manifest'] = md.get('entries', md)
        except Exception:
            result['manifest'] = {}
    else:
        result['manifest'] = {}

    return jsonify(result)


@app.route('/api/v1/processing/peers/status')
def processing_peers_combined():
    """Combined processing status across all peers + this instance."""
    import requests as req

    peer_urls = _get_peer_urls()

    instances = []

    # This instance
    local_completed = _get_completed_kgs()
    progress_file = Path('data/austria_processor/progress.json')
    local_state = 'idle'
    local_current = None
    local_rate = 0
    if progress_file.exists():
        try:
            pd = json.loads(progress_file.read_text())
            local_state = pd.get('state', 'idle')
            ckg = pd.get('current_kg') or {}
            local_current = ckg.get('code')
            local_rate = pd.get('rate_kgs_per_hour', 0)
        except Exception:
            pass

    instances.append({
        'instance': os.environ.get('INSTANCE_ID', 'primary'),
        'url': None,  # self
        'state': local_state,
        'current': local_current,
        'completed': len(local_completed),
        'completed_codes': sorted(local_completed),
        'rate': local_rate,
        'online': True,
    })

    all_completed = set(local_completed)

    # Peers
    for peer_url in peer_urls:
        entry = {
            'instance': '?',
            'url': peer_url,
            'state': 'unknown',
            'current': None,
            'completed': 0,
            'completed_codes': [],
            'rate': 0,
            'online': False,
        }
        try:
            r = req.get(peer_url.rstrip('/') + '/api/v1/processing/status', timeout=10)
            r.raise_for_status()
            pd = r.json()
            entry['state'] = pd.get('state', 'unknown')
            ckg = pd.get('current_kg') or {}
            entry['current'] = ckg.get('code')
            entry['completed'] = pd.get('completed', 0)
            entry['rate'] = pd.get('rate_kgs_per_hour', 0)
            entry['online'] = True

            # Also get peer identity from /peers
            try:
                r2 = req.get(peer_url.rstrip('/') + '/api/v1/processing/peers', timeout=5)
                r2.raise_for_status()
                pd2 = r2.json()
                entry['instance'] = pd2.get('instance', peer_url)
                entry['completed_codes'] = pd2.get('completed', [])
                all_completed.update(entry['completed_codes'])
            except Exception:
                pass
        except Exception as e:
            entry['error'] = str(e)

        instances.append(entry)

    return jsonify({
        'instances': instances,
        'total_kgs': 8440,
        'combined_completed': len(all_completed),
        'combined_completed_codes': sorted(all_completed),
        'combined_rate': sum(i.get('rate', 0) for i in instances if i.get('online')),
    })


@app.route('/api/v1/processing/start', methods=['POST'])
def processing_start():
    """Start the Austria processor as a background process."""
    global _processor_process
    if _processor_process is not None and _processor_process.poll() is None:
        return jsonify({'error': 'Processor already running', 'pid': _processor_process.pid}), 409

    args = [sys.executable, 'austria_processor.py']
    state = request.args.get('state') or request.json.get('state', '') if request.is_json else ''
    kg = request.args.get('kg') or (request.json.get('kg', '') if request.is_json else '')
    no_cop = request.args.get('no_copernicus', 'false').lower() in ('true', '1')

    if kg:
        args.extend(['--kg', kg])
    elif state:
        args.extend(['--state', state])
    if no_cop:
        args.append('--no-copernicus')

    log_file = Path('data/austria_processor/logs/processor.log')
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fd = open(log_file, 'a')

    import subprocess
    _processor_process = subprocess.Popen(
        args, stdout=log_fd, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.info('Austria processor started: PID %d, args=%s', _processor_process.pid, args)
    return jsonify({'status': 'started', 'pid': _processor_process.pid})


@app.route('/api/v1/processing/pause', methods=['POST'])
def processing_pause():
    """Pause the processor (sends SIGSTOP)."""
    global _processor_process
    if _processor_process is None or _processor_process.poll() is not None:
        return jsonify({'error': 'Processor not running'}), 404
    import signal as _sig
    os.kill(_processor_process.pid, _sig.SIGSTOP)
    # Update progress file
    pf = Path('data/austria_processor/progress.json')
    if pf.exists():
        d = json.loads(pf.read_text())
        d['state'] = 'paused'
        pf.write_text(json.dumps(d, indent=2, default=str))
    return jsonify({'status': 'paused', 'pid': _processor_process.pid})


@app.route('/api/v1/processing/resume', methods=['POST'])
def processing_resume():
    """Resume the processor (sends SIGCONT)."""
    global _processor_process
    if _processor_process is None or _processor_process.poll() is not None:
        return jsonify({'error': 'Processor not running'}), 404
    import signal as _sig
    os.kill(_processor_process.pid, _sig.SIGCONT)
    pf = Path('data/austria_processor/progress.json')
    if pf.exists():
        d = json.loads(pf.read_text())
        d['state'] = 'running'
        pf.write_text(json.dumps(d, indent=2, default=str))
    return jsonify({'status': 'resumed', 'pid': _processor_process.pid})


@app.route('/api/v1/processing/stop', methods=['POST'])
def processing_stop():
    """Stop the processor (sends SIGTERM)."""
    global _processor_process
    if _processor_process is None or _processor_process.poll() is not None:
        return jsonify({'error': 'Processor not running'}), 404
    import signal as _sig
    os.kill(_processor_process.pid, _sig.SIGTERM)
    _processor_process = None
    pf = Path('data/austria_processor/progress.json')
    if pf.exists():
        d = json.loads(pf.read_text())
        d['state'] = 'stopped'
        pf.write_text(json.dumps(d, indent=2, default=str))
    return jsonify({'status': 'stopped'})


@app.route('/api/v1/processing/postpone', methods=['POST'])
def processing_postpone():
    """Postpone current KG — kill subprocess, re-queue 5 KGs later, no fail count bump."""
    postpone_file = Path('data/austria_processor/postpone_signal.json')
    # Read current KG from progress
    pf = Path('data/austria_processor/progress.json')
    kg_code = None
    if pf.exists():
        try:
            d = json.loads(pf.read_text())
            ckg = d.get('current_kg', {})
            kg_code = ckg.get('code')
        except Exception:
            pass
    if not kg_code:
        return jsonify({'error': 'No current KG to postpone'}), 404
    # Write signal file — processor main loop picks this up
    import datetime as _dt
    postpone_file.write_text(json.dumps({
        'kg_code': kg_code,
        'ts': _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }))
    return jsonify({'status': 'postpone_requested', 'kg_code': kg_code})


@app.route('/api/v1/processing/single', methods=['POST'])
def processing_single():
    """Process a single KG (async in background)."""
    kg = request.args.get('kg') or (request.json.get('kg', '') if request.is_json else '')
    if not kg:
        return jsonify({'error': 'kg parameter required'}), 400
    return processing_start()  # reuse start with kg param


@app.route('/api/v1/processing/retry', methods=['POST'])
def processing_retry():
    """Retry a specific failed KG.

    Writes the KG code to ``retry_queue.json``.  The running processor
    picks this up each iteration and inserts the KG as the very next
    item in its processing queue.  Also cleans up the failed/progress
    state so the KG is no longer shown as failed.

    If the processor is NOT running, it is started automatically.
    """
    kg = request.args.get('kg') or (request.json.get('kg', '') if request.is_json else '')
    if not kg:
        return jsonify({'error': 'kg parameter required'}), 400

    data_dir = Path('data/austria_processor')
    actions = []

    # Write to retry queue (processor reads + clears each iteration)
    retry_path = data_dir / 'retry_queue.json'
    try:
        existing = []
        if retry_path.exists():
            existing = json.loads(retry_path.read_text())
        if kg not in existing:
            existing.append(kg)
        retry_path.write_text(json.dumps(existing))
        actions.append('added to retry_queue')
    except Exception as e:
        log.warning('retry: retry_queue.json: %s', e)

    # Remove from failed_kgs.json
    failed_path = data_dir / 'failed_kgs.json'
    if failed_path.exists():
        try:
            codes = set(json.loads(failed_path.read_text()))
            if kg in codes:
                codes.discard(kg)
                failed_path.write_text(json.dumps(sorted(codes), indent=2))
                actions.append('removed from failed_kgs')
        except Exception as e:
            log.warning('retry: failed_kgs.json: %s', e)

    # Remove from retried_kgs.json so it gets a fresh retry pass
    retried_path = data_dir / 'retried_kgs.json'
    if retried_path.exists():
        try:
            codes = set(json.loads(retried_path.read_text()))
            if kg in codes:
                codes.discard(kg)
                retried_path.write_text(json.dumps(sorted(codes), indent=2))
                actions.append('removed from retried_kgs')
        except Exception as e:
            log.warning('retry: retried_kgs.json: %s', e)

    # Reset failure count
    fc_path = data_dir / 'failure_counts.json'
    if fc_path.exists():
        try:
            fc = json.loads(fc_path.read_text())
            if kg in fc:
                del fc[kg]
                fc_path.write_text(json.dumps(fc, indent=2))
                actions.append('reset failure count')
        except Exception as e:
            log.warning('retry: failure_counts.json: %s', e)

    # Remove from progress tracker failed_kgs list
    progress_path = data_dir / 'progress.json'
    if progress_path.exists():
        try:
            d = json.loads(progress_path.read_text())
            flist = d.get('failed_kgs', [])
            d['failed_kgs'] = [f for f in flist if f.get('code') != kg]
            if len(d['failed_kgs']) != len(flist):
                d['failed'] = max(0, d.get('failed', 0) - 1)
                progress_path.write_text(json.dumps(d, indent=2, default=str))
                actions.append('removed from progress')
        except Exception as e:
            log.warning('retry: progress.json: %s', e)

    # Remove error entry from manifest
    manifest_path = data_dir / 'zenodo_manifest.json'
    if manifest_path.exists():
        try:
            from zenodo_client import Manifest
            m = Manifest(str(manifest_path))
            m.delete(f'{kg}_error')
            m.save()
            actions.append('removed from manifest')
        except Exception:
            pass

    # If processor is not running (neither systemd nor subprocess), start it
    processor_running = ((_processor_process is not None
                          and _processor_process.poll() is None)
                         or _is_processor_running())
    if not processor_running:
        return processing_start()

    # Processor is running — it will pick up the retry queue entry
    return jsonify({
        'status': 'queued_for_retry',
        'kg': kg,
        'actions': actions,
        'note': 'KG will be processed next by the running processor.',
    })


@app.route('/api/v1/processing/prioritize', methods=['POST'])
def processing_prioritize():
    """Prioritize KGs for processing — by bbox or explicit codes.

    Accepts JSON body with either:
      - bbox: {west, south, east, north} — resolves to KG codes via cadastre API
      - kgs: ["63349", "63350", ...] — explicit KG codes

    Filters out already-processed KGs and writes remaining to retry_queue.json.
    The running processor picks these up as next-in-queue.
    """
    data = request.get_json(silent=True) or {}
    # Also accept query params for bbox
    bbox = data.get('bbox')
    kgs = data.get('kgs', [])

    if not bbox and not kgs:
        # Try query params
        try:
            bbox = {
                'west': float(request.args['west']),
                'south': float(request.args['south']),
                'east': float(request.args['east']),
                'north': float(request.args['north']),
            }
        except (KeyError, ValueError):
            pass

    if not bbox and not kgs:
        return jsonify({'error': 'Provide bbox (west/south/east/north) or kgs array'}), 400

    # Resolve bbox to KG codes via cadastre API
    if bbox:
        try:
            import requests as req
            r = req.get(
                'https://cadastre-process-api.exe.xyz/api/v1/spatial/kgs',
                params={
                    'west': bbox['west'], 'south': bbox['south'],
                    'east': bbox['east'], 'north': bbox['north'],
                    'fields': 'kg_code,kg_name',
                },
                timeout=15,
            )
            r.raise_for_status()
            resp = r.json()
            kgs = resp.get('data', {}).get('kg_codes', [])
        except Exception as e:
            return jsonify({'error': f'Failed to resolve bbox to KGs: {e}'}), 502

    if not kgs:
        return jsonify({'error': 'No KGs found in the given area'}), 404

    # Filter out already-processed KGs
    data_dir = Path('data/austria_processor')
    processed = _get_completed_kgs()
    unprocessed = [k for k in kgs if k not in processed]
    already_done = [k for k in kgs if k in processed]

    # Write unprocessed to front of retry queue (priority = first)
    queued = []
    if unprocessed:
        retry_path = data_dir / 'retry_queue.json'
        try:
            existing = []
            if retry_path.exists():
                existing = json.loads(retry_path.read_text())
            existing_set = set(existing)
            for code in unprocessed:
                if code not in existing_set:
                    queued.append(code)
            # Prepend new KGs to front of queue
            existing = queued + existing
            retry_path.write_text(json.dumps(existing))
        except Exception as e:
            return jsonify({'error': f'Failed to write retry queue: {e}'}), 500

    return jsonify({
        'status': 'prioritized',
        'total_kgs': len(kgs),
        'queued': len(queued),
        'already_processed': len(already_done),
        'already_in_queue': len(unprocessed) - len(queued),
        'queued_codes': queued,
        'already_done_codes': already_done,
        'note': f'{len(queued)} KGs queued for priority processing.' if queued else 'All KGs already processed.',
    })


def _get_completed_kgs() -> set:
    """Return set of KG codes that have been successfully processed."""
    data_dir = Path('data/austria_processor')
    completed = set()
    manifest_path = data_dir / 'zenodo_manifest.json'
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            entries = m.get('entries', m)
            for key in entries:
                if key.endswith('_json') and 'error' not in entries[key].get('status', ''):
                    completed.add(key.replace('_json', ''))
        except Exception:
            pass
    json_dir = data_dir / 'json'
    if json_dir.exists():
        for jf in json_dir.glob('*.json'):
            completed.add(jf.stem)
    return completed


@app.route('/api/v1/processing/queue')
def processing_queue_get():
    """Read the priority queue with KG names and failure counts."""
    data_dir = Path('data/austria_processor')
    retry_path = data_dir / 'retry_queue.json'
    codes = []
    if retry_path.exists():
        try:
            codes = json.loads(retry_path.read_text())
        except Exception:
            pass
    # Filter out already-completed KGs
    completed = _get_completed_kgs()
    dirty = len(codes)
    codes = [c for c in codes if c not in completed]
    if len(codes) < dirty:
        # Persist the cleaned list
        try:
            retry_path.write_text(json.dumps(codes))
        except Exception:
            pass
    # Also filter out the currently-processing KG for display
    # (it's already shown in the Current KG card)
    current_kg = None
    pf = data_dir / 'progress.json'
    if pf.exists():
        try:
            pd = json.loads(pf.read_text())
            current_kg = (pd.get('current_kg') or {}).get('code')
        except Exception:
            pass
    if current_kg:
        codes = [c for c in codes if c != current_kg]
    # Load failure counts
    failure_counts = {}
    fc_path = data_dir / 'failure_counts.json'
    if fc_path.exists():
        try:
            failure_counts = json.loads(fc_path.read_text())
        except Exception:
            pass
    # Load permanently failed KGs — also filter out completed
    perm_failed = []
    failed_path = data_dir / 'failed_kgs.json'
    if failed_path.exists():
        try:
            all_failed = json.loads(failed_path.read_text())
            perm_failed = [c for c in all_failed if c not in completed]
            if len(perm_failed) < len(all_failed):
                failed_path.write_text(json.dumps(sorted(perm_failed), indent=2))
        except Exception:
            pass
    # Resolve names from search index
    items = []
    perm_failed_items = []
    try:
        import math as _math
        idx = si.get_index()
        conn = idx._conn()
        def _est_tiles(min_lon, min_lat, max_lon, max_lat, tile_km=1.5, overlap_km=0.1):
            """Estimate number of processing tiles from KG bbox."""
            if min_lon is None or min_lat is None:
                return None
            cos_lat = _math.cos(_math.radians((min_lat + max_lat) / 2))
            step_x = (tile_km - overlap_km) / (111 * cos_lat)
            step_y = (tile_km - overlap_km) / 111
            nx = max(1, _math.ceil((max_lon - min_lon) / step_x))
            ny = max(1, _math.ceil((max_lat - min_lat) / step_y))
            return nx * ny
        def _resolve(code):
            row = conn.execute(
                'SELECT kg_name, gemeinde_name, district_name, min_lon, min_lat, max_lon, max_lat FROM kg WHERE kg_code=?',
                (code,)
            ).fetchone()
            if row:
                return {'code': code, 'name': row['kg_name'],
                        'gemeinde': row['gemeinde_name'],
                        'district': row['district_name'],
                        'failures': failure_counts.get(code, 0),
                        'est_tiles': _est_tiles(row['min_lon'], row['min_lat'], row['max_lon'], row['max_lat'])}
            return {'code': code, 'name': code, 'failures': failure_counts.get(code, 0)}
        items = [_resolve(c) for c in codes]
        perm_failed_items = [_resolve(c) for c in perm_failed]
    except Exception:
        items = [{'code': c, 'name': c, 'failures': failure_counts.get(c, 0)} for c in codes]
        perm_failed_items = [{'code': c, 'name': c, 'failures': failure_counts.get(c, 0)} for c in perm_failed]
    return jsonify({
        'queue': items, 'count': len(items),
        'permanently_failed': perm_failed_items,
        'permanently_failed_count': len(perm_failed_items),
    })


@app.route('/api/v1/processing/queue', methods=['POST'])
def processing_queue_add():
    """Add KG codes to the priority queue at a specific position.

    Body JSON:
      kgs: list of KG codes to add (required)
      position: 0-based insertion index (default 0 = front)
               Use -1 or omit to append at end.
      skip_processed: if true (default), silently drop already-processed KGs

    Duplicates already in the queue are moved to the new position.
    """
    data = request.get_json(silent=True) or {}
    new_codes = data.get('kgs') or data.get('kg_codes') or data.get('queue', [])
    if isinstance(new_codes, str):
        new_codes = [new_codes]
    if not isinstance(new_codes, list) or not new_codes:
        return jsonify({'error': 'kgs must be a non-empty array of KG codes'}), 400
    new_codes = [str(c).strip() for c in new_codes if str(c).strip()]
    position = data.get('position', -1)
    skip_processed = data.get('skip_processed', True)

    retry_path = Path('data/austria_processor/retry_queue.json')
    try:
        codes = json.loads(retry_path.read_text()) if retry_path.exists() else []
    except Exception:
        codes = []

    # Optionally filter out already-processed
    skipped = []
    if skip_processed:
        completed = _get_completed_kgs()
        kept = []
        for c in new_codes:
            if c in completed:
                skipped.append(c)
            else:
                kept.append(c)
        new_codes = kept

    if not new_codes:
        return jsonify({
            'status': 'nothing_to_add',
            'skipped_processed': skipped,
            'queue_length': len(codes),
        })

    # Remove duplicates from existing queue (they'll be re-inserted)
    moved = [c for c in new_codes if c in codes]
    codes = [c for c in codes if c not in set(new_codes)]

    # Insert at position
    if position < 0 or position >= len(codes):
        codes.extend(new_codes)
        actual_pos = len(codes) - len(new_codes)
    else:
        for i, c in enumerate(new_codes):
            codes.insert(position + i, c)
        actual_pos = position

    retry_path.write_text(json.dumps(codes))

    # Resolve names
    added_info = []
    try:
        idx = si.get_index()
        conn = idx._conn()
        for c in new_codes:
            row = conn.execute(
                'SELECT kg_name, gemeinde_name, district_name FROM kg WHERE kg_code=?',
                (c,)
            ).fetchone()
            added_info.append({
                'code': c, 'name': row['kg_name'] if row else c,
                'gemeinde': row['gemeinde_name'] if row else None,
                'position': codes.index(c),
            })
    except Exception:
        added_info = [{'code': c, 'position': codes.index(c)} for c in new_codes]

    return jsonify({
        'status': 'added',
        'added': added_info,
        'added_count': len(new_codes),
        'moved_from_existing': moved,
        'skipped_processed': skipped,
        'queue_length': len(codes),
    })


@app.route('/api/v1/processing/queue', methods=['PUT'])
def processing_queue_put():
    """Replace the entire priority queue."""
    data = request.get_json(silent=True) or {}
    codes = data.get('queue', [])
    if not isinstance(codes, list):
        return jsonify({'error': 'queue must be an array of KG codes'}), 400
    retry_path = Path('data/austria_processor/retry_queue.json')
    retry_path.write_text(json.dumps(codes))
    return jsonify({'status': 'saved', 'count': len(codes)})


@app.route('/api/v1/processing/queue', methods=['DELETE'])
def processing_queue_delete():
    """Remove a KG from the priority queue."""
    code = request.args.get('kg') or (request.get_json(silent=True) or {}).get('kg', '')
    if not code:
        return jsonify({'error': 'kg parameter required'}), 400
    retry_path = Path('data/austria_processor/retry_queue.json')
    try:
        codes = json.loads(retry_path.read_text()) if retry_path.exists() else []
        if code in codes:
            codes.remove(code)
            retry_path.write_text(json.dumps(codes))
            return jsonify({'status': 'removed', 'kg': code, 'remaining': len(codes)})
        return jsonify({'status': 'not_found', 'kg': code}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/processing/log')
def processing_log():
    """Return recent processor log lines."""
    log_file = Path('data/austria_processor/logs/processor.log')
    n = int(request.args.get('lines', 200))
    if not log_file.exists():
        return jsonify({'lines': [], 'total': 0})
    try:
        lines = log_file.read_text().splitlines()[-n:]
        return jsonify({'lines': lines, 'total': len(lines)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/processing/tiles')
def processing_tiles():
    """Return Zenodo cache tile bboxes for map overlay in process.html.

    Reads the locally-cached ZIP central-directory indices written by
    ``zenodo_cache.ZipIndex``.  Each entry name encodes the grid bbox as
    ``{product}_{s}_{w}_{n}_{e}[_{year}].npz``.
    """
    try:
        idx_dir = Path('data/austria_processor/zenodo_zip_index')
        if not idx_dir.exists():
            return jsonify({'copernicus': [], 'hansen': []})
        cop_seen: set = set()
        han_seen: set = set()
        cop_tiles: list = []
        han_tiles: list = []
        for fp in idx_dir.iterdir():
            if not fp.suffix == '.json':
                continue
            try:
                raw = json.loads(fp.read_text())
            except Exception:
                continue
            for name in raw:
                base = name.replace('.npz', '')
                parts = base.split('_')
                product = parts[0]
                floats = []
                for p in parts[1:]:
                    try:
                        floats.append(float(p))
                    except ValueError:
                        break
                if len(floats) < 4:
                    continue
                s, w, n, e = floats[0], floats[1], floats[2], floats[3]
                if product == 'hansen':
                    key = (w, s, e, n)
                    if key not in han_seen:
                        han_seen.add(key)
                        han_tiles.append({'w': w, 's': s, 'e': e, 'n': n})
                else:
                    key = (w, s, e, n)
                    if key not in cop_seen:
                        cop_seen.add(key)
                        cop_tiles.append({'w': w, 's': s, 'e': e, 'n': n})
        return jsonify({'copernicus': cop_tiles, 'hansen': han_tiles})
    except Exception as e:
        return jsonify({'error': str(e), 'copernicus': [], 'hansen': []})


@app.route('/api/v1/processing/manifest')
def processing_manifest():
    """Return the Zenodo manifest for the processor."""
    manifest_path = Path('data/austria_processor/zenodo_manifest.json')
    if not manifest_path.exists():
        return jsonify({'entries': {}, 'count': 0})
    try:
        data = json.loads(manifest_path.read_text())
        entries = data.get('entries', {})
        return jsonify({
            'count': len(entries),
            'entries': entries,
            'total_size_bytes': sum(e.get('size', 0) for e in entries.values()),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# === SECTION: Search index API endpoints ===

@app.route('/api/v1/index/status')
def index_status():
    """Search index status and statistics."""
    try:
        idx = si.get_index()
        return jsonify(idx.stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/index/rebuild', methods=['POST'])
def index_rebuild():
    """Rebuild the search index from scratch."""
    try:
        idx = si.get_index()
        idx.build()
        return jsonify(idx.stats())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/kg/<kg_code>')
def api_kg(kg_code):
    """Return KG info from index. If JSON exists locally, serves the full JSON.
    Otherwise returns index data with Zenodo download links."""
    # If full JSON exists locally, serve it directly (backwards-compatible)
    json_path = Path(f'data/austria_processor/json/{kg_code}.json')
    if json_path.exists() and not request.args.get('index_only'):
        return send_file(str(json_path), mimetype='application/json')
    # Fall back to search index
    try:
        idx = si.get_index()
        result = idx.query_kg(kg_code)
        if result:
            return jsonify(result)
    except Exception as e:
        log.warning('index query_kg %s: %s', kg_code, e)
    return jsonify({'error': f'KG {kg_code} not found'}), 404


@app.route('/api/v1/parcel/<path:parcel_id>')
def api_parcel(parcel_id):
    """Look up a parcel. Returns KG summary + parcel detail if available."""
    if '-' not in parcel_id:
        return jsonify({'error': 'Invalid parcel_id format, expected KGCODE-GNR'}), 400
    try:
        idx = si.get_index()
        result = idx.query_parcel(parcel_id)
        if result:
            return jsonify(result)
    except Exception as e:
        log.warning('index query_parcel %s: %s', parcel_id, e)
    return jsonify({'error': f'Parcel {parcel_id} not found'}), 404


# === SECTION: GPKG detail lazy-load endpoints ===

def _parse_bbox_param(s):
    """Parse 'w,s,e,n' bbox string to tuple, or None."""
    if not s:
        return None
    try:
        parts = [float(x) for x in s.split(',')]
        if len(parts) == 4:
            return tuple(parts)
    except (ValueError, TypeError):
        pass
    return None


@app.route('/api/v1/kg/<kg_code>/buildings')
def api_kg_buildings(kg_code):
    """Height-enriched building footprints from the light GPKG.
    Lazy-loads from Zenodo if not cached locally.

    Params: bbox=w,s,e,n  limit=N  offset=N
    """
    try:
        idx = si.get_index()
        bbox = _parse_bbox_param(request.args.get('bbox'))
        limit = min(int(request.args.get('limit', 500)), 5000)
        offset = int(request.args.get('offset', 0))
        result = idx.query_buildings(kg_code, bbox=bbox, limit=limit, offset=offset)
        if result is None:
            return jsonify({'error': f'No building data available for KG {kg_code}'}), 404
        return jsonify(result)
    except Exception as e:
        log.warning('api_kg_buildings %s: %s', kg_code, e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/kg/<kg_code>/new_buildings')
def api_kg_new_buildings(kg_code):
    """Detected new buildings from the light GPKG.

    Params: bbox=w,s,e,n  limit=N  offset=N
    """
    try:
        idx = si.get_index()
        bbox = _parse_bbox_param(request.args.get('bbox'))
        limit = min(int(request.args.get('limit', 500)), 5000)
        offset = int(request.args.get('offset', 0))
        result = idx.query_new_buildings_detail(kg_code, bbox=bbox, limit=limit, offset=offset)
        if result is None:
            return jsonify({'error': f'No new building data available for KG {kg_code}'}), 404
        return jsonify(result)
    except Exception as e:
        log.warning('api_kg_new_buildings %s: %s', kg_code, e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/kg/<kg_code>/infrastructure')
def api_kg_infrastructure(kg_code):
    """Detected infrastructure from the light GPKG.

    Params: bbox=w,s,e,n  limit=N  offset=N
    """
    try:
        idx = si.get_index()
        bbox = _parse_bbox_param(request.args.get('bbox'))
        limit = min(int(request.args.get('limit', 500)), 5000)
        offset = int(request.args.get('offset', 0))
        result = idx.query_infrastructure_detail(kg_code, bbox=bbox, limit=limit, offset=offset)
        if result is None:
            return jsonify({'error': f'No infrastructure data available for KG {kg_code}'}), 404
        return jsonify(result)
    except Exception as e:
        log.warning('api_kg_infrastructure %s: %s', kg_code, e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/buildings/search')
def api_buildings_search():
    """Search building records from the index (fast, no GPKG).

    Params: kg=CODE  min_height=N  max_height=N  min_stories=N  max_stories=N
            roof_type=flat|pitched  min_area=N  max_area=N  limit=N  offset=N
    """
    try:
        idx = si.get_index()
        result = idx.query_buildings_index(
            kg_code=request.args.get('kg'),
            min_height=_float_or_none(request.args.get('min_height')),
            max_height=_float_or_none(request.args.get('max_height')),
            min_stories=_int_or_none(request.args.get('min_stories')),
            max_stories=_int_or_none(request.args.get('max_stories')),
            roof_type=request.args.get('roof_type'),
            min_area=_float_or_none(request.args.get('min_area')),
            max_area=_float_or_none(request.args.get('max_area')),
            limit=min(int(request.args.get('limit', 500)), 5000),
            offset=int(request.args.get('offset', 0)),
        )
        return jsonify(result)
    except Exception as e:
        log.warning('api_buildings_search: %s', e)
        return jsonify({'error': str(e)}), 500


def _float_or_none(v):
    if v is None: return None
    try: return float(v)
    except (ValueError, TypeError): return None

def _int_or_none(v):
    if v is None: return None
    try: return int(v)
    except (ValueError, TypeError): return None


@app.route('/api/v1/kg/<kg_code>/segments')
def api_kg_segments(kg_code):
    """Segment polygons from the light GPKG.

    Params: bbox=w,s,e,n  type=t1,t2,...  limit=N  offset=N
    """
    try:
        idx = si.get_index()
        bbox = _parse_bbox_param(request.args.get('bbox'))
        type_str = request.args.get('type', '')
        type_filter = [t.strip() for t in type_str.split(',') if t.strip()] or None
        limit = min(int(request.args.get('limit', 500)), 5000)
        offset = int(request.args.get('offset', 0))
        result = idx.query_segments_detail(kg_code, bbox=bbox, type_filter=type_filter,
                                           limit=limit, offset=offset)
        if result is None:
            return jsonify({'error': f'No segment data available for KG {kg_code}'}), 404
        return jsonify(result)
    except Exception as e:
        log.warning('api_kg_segments %s: %s', kg_code, e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/kg/<kg_code>/layers')
def api_kg_layers(kg_code):
    """List available vector layers in a KG's GPKG.

    Params: variant=light|full (default: light)
    """
    try:
        idx = si.get_index()
        variant = request.args.get('variant', 'light')
        if variant not in ('light', 'full'):
            return jsonify({'error': 'variant must be light or full'}), 400
        result = idx.gpkg_layers(kg_code, variant=variant)
        if result is None:
            return jsonify({'error': f'No GPKG available for KG {kg_code}'}), 404
        return jsonify({'kg_code': kg_code, 'variant': variant, 'layers': result})
    except Exception as e:
        log.warning('api_kg_layers %s: %s', kg_code, e)
        return jsonify({'error': str(e)}), 500


# === SECTION: GET query param parsers for compound/parcel filters ===

def _parse_compound_from_args(args):
    """Parse compound filter dict from flat GET query params.

    Supports all query_compound() filter keys as query params.
    Special handling:
      - bbox=w,s,e,n → [float, float, float, float]
      - aspect=S,SW,W → ["S","SW","W"]
      - type_filter=tree:0.8:800 (repeatable) → type_filters list
      - landcover_filter=grass:1300:0.1 (repeatable) → landcover_filters list
      - All min_*/max_* numeric params parsed as float/int
    """
    filters = {}

    # bbox
    if 'bbox' in args:
        try:
            filters['bbox'] = [float(x) for x in args['bbox'].split(',')]
        except ValueError:
            pass

    # String params (exact match)
    for k in ('state', 'district', 'gemeinde', 'dominant_type', 'phenology',
              'terrain_class', 'quality_grade', 'sort', 'sort_dir'):
        if k in args:
            filters[k] = args[k]

    # aspect — comma-separated list
    if 'aspect' in args:
        filters['aspect'] = [x.strip() for x in args['aspect'].split(',') if x.strip()]

    # All numeric range params (min_*/max_*)
    _numeric_keys = [
        'min_slope', 'max_slope', 'min_roughness', 'min_elevation', 'max_elevation',
        'min_elevation_min', 'max_elevation_min', 'min_elevation_max', 'max_elevation_max',
        'min_elevation_range', 'min_steepness_max', 'min_tri', 'max_tri',
        'min_total_area', 'max_total_area', 'min_parcels', 'max_parcels',
        'min_segments', 'max_segments',
        'min_buildings', 'max_buildings', 'min_new_buildings', 'min_infrastructure',
        'min_building_height', 'max_building_height',
        'min_building_max_height', 'max_building_max_height',
        'min_building_stories', 'max_building_stories',
        'min_building_stories_max', 'max_building_stories_max',
        'min_building_pitched_pct', 'max_building_pitched_pct',
        'min_building_footprint', 'max_building_footprint',
        'min_new_building_footprint', 'min_new_building_height', 'min_new_building_stories',
        'min_building_height_coverage',
        'min_tree_count', 'min_tree_height', 'min_tree_canopy_sqm', 'min_tree_volume',
        'min_ndvi', 'max_ndvi', 'min_vegetated_fraction', 'max_vegetated_fraction',
        'min_shannon_diversity',
        'min_ndvi_amplitude', 'min_ndvi_harm_mean', 'max_ndvi_harm_mean',
        'min_ndvi_phase', 'max_ndvi_phase',
        'min_sar_vv', 'max_sar_vv', 'min_sar_vh', 'max_sar_vh',
        'min_dtm_change', 'max_dtm_change', 'min_volume_change', 'max_volume_change',
        'min_changed_segments', 'min_disturbed_volume', 'min_temporal_stability',
        'min_confidence', 'min_rf_confidence', 'max_diverged_pct',
        'max_rf_diverged_count', 'min_rf_classified_pct', 'min_quality_score',
    ]
    for k in _numeric_keys:
        if k in args:
            try:
                v = float(args[k])
                filters[k] = int(v) if v == int(v) and 'pct' not in k and 'fraction' not in k and 'confidence' not in k and 'slope' not in k and 'roughness' not in k and 'elevation' not in k and 'height' not in k and 'ndvi' not in k and 'sar' not in k and 'tri' not in k and 'stability' not in k and 'score' not in k and 'change' not in k and 'volume' not in k and 'canopy' not in k and 'diversity' not in k and 'amplitude' not in k and 'phase' not in k and 'steepness' not in k and 'area' not in k and 'footprint' not in k and 'stories' not in k else v
            except ValueError:
                pass

    # type_filter — repeatable: type_filter=tree:0.8:800&type_filter=grass:0.8:1300
    tf_raw = args.getlist('type_filter')
    if tf_raw:
        type_filters = []
        for raw in tf_raw:
            parts = raw.split(':')
            tf = {'type': parts[0]}
            if len(parts) > 1 and parts[1]:
                try: tf['min_confidence'] = float(parts[1])
                except ValueError: pass
            if len(parts) > 2 and parts[2]:
                try: tf['min_area_sqm'] = float(parts[2])
                except ValueError: pass
            type_filters.append(tf)
        filters['type_filters'] = type_filters

    # landcover_filter — repeatable: landcover_filter=grass:1300:0.1
    lf_raw = args.getlist('landcover_filter')
    if lf_raw:
        landcover_filters = []
        for raw in lf_raw:
            parts = raw.split(':')
            lf = {'type': parts[0]}
            if len(parts) > 1 and parts[1]:
                try: lf['min_area_sqm'] = float(parts[1])
                except ValueError: pass
            if len(parts) > 2 and parts[2]:
                try: lf['min_fraction'] = float(parts[2])
                except ValueError: pass
            if len(parts) > 3 and parts[3]:
                try: lf['min_height_mean'] = float(parts[3])
                except ValueError: pass
            if len(parts) > 4 and parts[4]:
                try: lf['max_height_mean'] = float(parts[4])
                except ValueError: pass
            landcover_filters.append(lf)
        filters['landcover_filters'] = landcover_filters

    return filters


def _parse_parcel_filters_from_args(args):
    """Parse parcel_filters dict from GET query params with pf_ prefix.

    All parcel filter params use a 'pf_' prefix to avoid collision with compound filters.
    Examples:
      pf_aspect=E,SE → aspect: ["E","SE"]
      pf_terrain_class=level → terrain_class: "level"
      pf_min_vegetated_fraction=0.5 → min_vegetated_fraction: 0.5
      pf_types=tree,grass → types: ["tree","grass"]
      pf_type_confidence=tree:0.7:500 (repeatable)
      pf_sort=conservation_score, pf_sort_dir=desc
    """
    pf = {}

    # String params
    for k in ('terrain_class', 'sort', 'sort_dir', 'cadastre_landuse', 'roof_type', 'dominant_type'):
        pk = f'pf_{k}'
        if pk in args:
            pf[k] = args[pk]

    # Bool params
    for k in ('is_vegetated', 'cadastre_has_buildings'):
        pk = f'pf_{k}'
        if pk in args:
            pf[k] = args[pk].lower() in ('true', '1', 'yes')

    # Comma-separated list params
    for k in ('aspect', 'types'):
        pk = f'pf_{k}'
        if pk in args:
            pf[k] = [x.strip() for x in args[pk].split(',') if x.strip()]

    # Numeric params
    _pf_numeric = [
        'min_vegetated_fraction', 'max_vegetated_fraction',
        'min_forested_fraction', 'max_forested_fraction',
        'min_elevation', 'max_elevation',
        'min_type_fraction', 'min_ndsm_max', 'max_ndsm_max',
        'min_parcel_area', 'max_parcel_area',
        'min_slope', 'max_slope', 'min_tri', 'max_tri',
        'min_confidence', 'min_rf_confidence',
        'cadastre_min_area', 'cadastre_max_area',
        'min_stories', 'max_stories',
        'min_hansen_recent_5yr', 'max_hansen_recent_5yr',
        'min_hansen_total', 'max_hansen_total',
    ]
    _pf_int = {'min_stories', 'max_stories', 'min_hansen_recent_5yr',
               'max_hansen_recent_5yr', 'min_hansen_total', 'max_hansen_total'}
    for k in _pf_numeric:
        pk = f'pf_{k}'
        if pk in args:
            try:
                pf[k] = int(args[pk]) if k in _pf_int else float(args[pk])
            except ValueError:
                pass

    # pf_type_confidence — repeatable: pf_type_confidence=tree:0.7:500
    tc_raw = args.getlist('pf_type_confidence')
    if tc_raw:
        type_confidence = []
        for raw in tc_raw:
            parts = raw.split(':')
            tc = {'type': parts[0]}
            if len(parts) > 1 and parts[1]:
                try: tc['min_confidence'] = float(parts[1])
                except ValueError: pass
            if len(parts) > 2 and parts[2]:
                try: tc['min_area_sqm'] = float(parts[2])
                except ValueError: pass
            if len(parts) > 3 and parts[3]:
                try: tc['min_rf_confidence'] = float(parts[3])
                except ValueError: pass
            type_confidence.append(tc)
        pf['type_confidence'] = type_confidence

    return pf


@app.route('/api/v1/query/parcels')
def api_query_parcels():
    """Query parcels from the search index (fast SQL, no GPKG/JSON load).

    Supports all per-parcel landscape attributes plus building join filters.
    This is the endpoint for complex cross-KG parcel queries like:
    "Show parcels with a 1-storey pitched-roof building, south-facing >30% slope,
     <5000 sqm, above 900m, with little deforestation in the last 5 years."

    Params:
      kg=<code>                   Filter by KG code
      state=<name|code>           Filter by Bundesland
      district=<name|code>        Filter by Bezirk
      gemeinde=<name|code>        Filter by Gemeinde
      bbox=<w,s,e,n>              Spatial bbox filter
      min_area / max_area         Parcel area m²
      min_elevation / max_elevation  Elevation m
      min_slope / max_slope       Slope degrees
      min_tri / max_tri           Terrain Roughness Index
      terrain_class=<str>         level / nearly_level / slightly_rugged / ...
      aspect=<N,NE,E,SE,S,SW,W,NW>  Comma-separated aspect filter
      dominant_type=<str>         Dominant segment type (tree/grass/roof/...)
      min_vegetated_fraction / max_vegetated_fraction   0-1
      min_forested_fraction / max_forested_fraction     0-1
      min_ndsm_max / max_ndsm_max                      nDSM max height m
      is_vegetated=true/false     Vegetation boolean
      min_confidence              Min mean classification confidence
      min_rf_confidence           Min RF classification confidence
      max_hansen_recent_5yr       Max recent 5-year forest loss pixels
      min_hansen_recent_5yr       Min recent 5-year forest loss pixels
      max_hansen_total            Max total forest loss pixels
      min_hansen_total            Min total forest loss pixels
      building_roof_type=pitched/flat  Parcel must contain matching building
      building_min_stories=<N>    Building stories filter
      building_max_stories=<N>    Building stories filter
      sort=<col>                  Sort column (default: elevation_m)
      sort_dir=asc/desc           Sort direction (default: desc)
      limit=<N>                   Max results (default 100, max 1000)
      offset=<N>                  Pagination offset
    """
    try:
        idx = si.get_index()
        args = request.args
        limit = min(int(args.get('limit', 100)), 1000)
        offset = int(args.get('offset', 0))

        # Parse bbox
        bbox = None
        if args.get('bbox'):
            parts = [float(x) for x in args['bbox'].split(',')]
            if len(parts) == 4:
                bbox = tuple(parts)

        # Numeric params
        _num = {
            'min_area': float, 'max_area': float,
            'min_elevation': float, 'max_elevation': float,
            'min_slope': float, 'max_slope': float,
            'min_tri': float, 'max_tri': float,
            'min_vegetated_fraction': float, 'max_vegetated_fraction': float,
            'min_forested_fraction': float, 'max_forested_fraction': float,
            'min_ndsm_max': float, 'max_ndsm_max': float,
            'min_confidence': float, 'min_rf_confidence': float,
            'max_hansen_recent_5yr': int, 'min_hansen_recent_5yr': int,
            'max_hansen_total': int, 'min_hansen_total': int,
            'building_min_stories': int, 'building_max_stories': int,
        }
        kwargs = {}
        for k, conv in _num.items():
            if k in args:
                try:
                    kwargs[k] = conv(args[k])
                except ValueError:
                    pass

        result = idx.query_parcels_index(
            kg_code=args.get('kg'),
            terrain_class=args.get('terrain_class'),
            aspect=args.get('aspect'),
            dominant_type=args.get('dominant_type'),
            building_roof_type=args.get('building_roof_type'),
            is_vegetated=args.get('is_vegetated', '').lower() in ('true', '1') if 'is_vegetated' in args else None,
            state=args.get('state'),
            district=args.get('district'),
            gemeinde=args.get('gemeinde'),
            bbox=bbox,
            sort=args.get('sort', 'elevation_m'),
            sort_dir=args.get('sort_dir', 'desc'),
            limit=limit,
            offset=offset,
            **kwargs,
        )
        return jsonify(result)
    except Exception as e:
        log.exception('query_parcels error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/query/compound', methods=['GET', 'POST'])
def api_query_compound():
    """Compound query: filter KGs by any combination of attributes.

    Accepts both POST (JSON body) and GET (query params).

    GET query params map directly to filter keys. Special syntax:
      - bbox=w,s,e,n
      - aspect=S,SW,W (comma-separated)
      - type_filter=tree:0.8:800 (repeatable, type:min_confidence:min_area_sqm)
      - landcover_filter=grass:1300:0.1 (repeatable, type:min_area_sqm:min_fraction)
      - All min_*/max_* numeric params
      - limit, offset, async, task_id as query params

    POST JSON body with filter keys (all optional):
      bbox: [w, s, e, n]
      state, district, gemeinde: str
      aspect: ["S","SW","W"]           dominant aspect direction
      dominant_type, phenology, terrain_class, quality_grade: str (exact match)

      Numeric ranges (min_X / max_X):
        --- Terrain ---
        slope, roughness, elevation, elevation_min, elevation_max,
        elevation_range, steepness_max, tri
        --- Area / parcels / segments ---
        total_area, parcels, segments
        --- Buildings ---
        buildings, new_buildings, infrastructure,
        building_height, building_max_height,
        building_stories, building_stories_max,
        building_pitched_pct, building_footprint,
        new_building_footprint, new_building_height, new_building_stories,
        building_height_coverage
        --- Trees ---
        tree_count, tree_height, tree_canopy_sqm, tree_volume
        --- Vegetation ---
        ndvi, vegetated_fraction, shannon_diversity
        --- NDVI harmonics ---
        ndvi_amplitude, ndvi_harm_mean, ndvi_phase
        --- SAR ---
        sar_vv, sar_vh
        --- Temporal change ---
        dtm_change, volume_change, changed_segments, disturbed_volume,
        temporal_stability
        --- Classification quality ---
        confidence, rf_confidence, diverged_pct (max),
        rf_diverged_count (max), rf_classified_pct, quality_score

      type_filters: [{"type": "tree", "min_confidence": 0.8, "min_area_sqm": 800}, ...]
        Filter by RF classification confidence + area per object type.
      landcover_filters: [{"type": "grass", "min_area_sqm": 1300, "min_fraction": 0.1}, ...]
        Filter by landcover area/fraction/height per object type.
      sort: str (any kg column or type-derived column)
      sort_dir: "asc" | "desc"
      limit: int (default 50, max 1000), offset: int

    Example:
      POST /api/v1/query/compound
      {
        "type_filters": [
          {"type": "tree", "min_confidence": 0.8, "min_area_sqm": 800},
          {"type": "grass", "min_confidence": 0.8, "min_area_sqm": 1300}
        ],
        "max_buildings": 0,
        "aspect": ["S", "SW", "W"],
        "min_roughness": 2.0,
        "sort": "tree_area_sqm",
        "sort_dir": "desc"
      }
    """
    try:
        idx = si.get_index()
        # Accept both POST JSON body and GET query params
        if request.method == 'POST' and request.content_type and 'json' in request.content_type:
            body = request.get_json(force=True) or {}
        elif request.method == 'POST':
            # POST with no JSON — try to parse, fall back to query args
            try:
                body = request.get_json(force=True) or {}
            except Exception:
                body = _parse_compound_from_args(request.args)
        else:
            # GET — parse from query params
            body = _parse_compound_from_args(request.args)
        limit = min(int(body.pop('limit', None) or request.args.get('limit', 50)), 1000)
        offset = int(body.pop('offset', None) or request.args.get('offset', 0))
        _async_val = body.pop('async', None) or request.args.get('async', '')
        do_async = str(_async_val).lower() in ('true', '1', 'yes')
        # Poll existing task
        _task_id_val = body.pop('task_id', None) or request.args.get('task_id')
        if _task_id_val:
            task_id = _task_id_val
            p = _PROGRESS_DIR / f"{task_id}.json"
            if not p.exists():
                return jsonify({'error': 'Unknown task_id'}), 404
            try:
                info = json.loads(p.read_text())
            except Exception:
                return jsonify({'active': False})
            step = info.get('step', '')
            elapsed = round(time.time() - info.get('t0', time.time()), 1)
            if step == 'done':
                result = _get_query_result(task_id)
                if result is not None:
                    try: p.unlink(missing_ok=True)
                    except Exception: pass
                    return jsonify(result)
                return jsonify({'active': False, 'done': True, 'elapsed': elapsed})
            if step == 'error':
                try: p.unlink(missing_ok=True)
                except Exception: pass
                return jsonify({'error': info.get('detail', 'Query failed')}), 500
            return jsonify({'active': True, 'task_id': task_id, 'step': step,
                            'detail': info.get('detail', ''), 'elapsed': elapsed}), 202
        if do_async:
            task_id = str(uuid.uuid4())
            t = threading.Thread(target=_query_worker, daemon=True,
                args=(task_id, idx.query_compound, (body,),
                      dict(limit=limit, offset=offset)))
            t.start()
            return jsonify({'task_id': task_id, 'status': 'running',
                            'poll': f'/api/v1/query/compound'}), 202
        result = idx.query_compound(body, limit=limit, offset=offset)
        return jsonify(result)
    except Exception as e:
        log.warning('api_query_compound: %s', e)
        return jsonify({'error': str(e)}), 500


# === SECTION: Async query task support ===
_QUERY_RESULTS_DIR = Path('/tmp/query_results')
_QUERY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def _query_worker(task_id, query_fn, args, kwargs):
    """Run a slow query in a background thread."""
    try:
        _progress_start(task_id)
        _progress_set(task_id, 'running', f'Scanning KG JSONs...')
        result = query_fn(*args, **kwargs)
        p = _QUERY_RESULTS_DIR / f"{task_id}.json.gz"
        data = json.dumps(result).encode()
        with gzip.open(str(p), 'wb') as f:
            f.write(data)
        _progress_done(task_id)
    except Exception as e:
        log.exception('async query %s failed', task_id)
        _progress_error(task_id, str(e))

def _get_query_result(task_id):
    """Retrieve stored query result."""
    p = _QUERY_RESULTS_DIR / f"{task_id}.json.gz"
    if not p.exists():
        return None
    with gzip.open(str(p), 'rb') as f:
        return json.loads(f.read())

@app.route('/api/v1/query/progress')
def query_progress():
    """Poll progress of an async query task."""
    task_id = request.args.get('task_id', '')
    if not task_id:
        return jsonify({'error': 'task_id required'}), 400
    p = _PROGRESS_DIR / f"{task_id}.json"
    if not p.exists():
        return jsonify({'error': 'Unknown task_id'}), 404
    try:
        info = json.loads(p.read_text())
    except Exception:
        return jsonify({'active': False})
    step = info.get('step', '')
    elapsed = round(time.time() - info.get('t0', time.time()), 1)
    if step == 'done':
        result = _get_query_result(task_id)
        if result is not None:
            # Clean up progress file
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
            return jsonify(result)
        return jsonify({'active': False, 'done': True, 'elapsed': elapsed})
    if step == 'error':
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({'error': info.get('detail', 'Query failed')}), 500
    return jsonify({'active': True, 'step': step, 'detail': info.get('detail', ''),
                    'elapsed': elapsed}), 202


@app.route('/api/v1/query')
def api_query():
    """Unified query endpoint. Supports multiple query modes via params.

    Params:
      q=<text>          Full-text search (KG/gemeinde/district/state names)
      kg=<code>         Exact KG code lookup
      parcel=<id>       Parcel lookup (KGCODE-GNR)
      bbox=<w,s,e,n>    Spatial bbox query
      point=<lon,lat>   Point proximity query
      radius=<km>       Radius for point query (default 5)
      state=<code|name>         Filter/aggregate by Bundesland
      district=<code>           Filter/aggregate by Bezirk
      gemeinde=<code>           Filter/aggregate by Gemeinde
      type=<object_type>        Rank KGs by object type
      metric=<area|fraction|count|height>  Ranking metric (with type=)
      hansen=true               Hansen forest loss query
      year_from=<YYYY>          Start year for hansen/temporal filter
      year_to=<YYYY>            End year for hansen/temporal filter
      new_buildings=true         Query KGs with new buildings
      min_count=<N>             Min count for new_buildings
      divergence=true           Query KGs ranked by RF→final type divergence
      min_divergence=<pct>      Min divergence % (0-100, with divergence=true)
      rf_type=<type>            Filter divergences FROM this RF type
      final_type=<type>         Filter divergences TO this final type
      low_confidence=true       Query KGs with lowest confidence
      max_confidence=<float>    Max confidence threshold (with low_confidence=true)
      confidence_rank=asc|desc  Rank KGs by classification confidence
      type_confidence=<type>    Rank KGs by RF confidence for specific type
      divergence_pairs=true     Get most common RF→final divergence pairs
      high_confidence_type=<type>  KGs where type has high RF confidence
      parcels_by_type=<type>    Per-parcel filter by type RF confidence+area
      top_features=<trees|objects|new_buildings|infrastructure>  Cross-KG top features
      min_confidence=<float>    Min RF confidence (with high_confidence_type/parcels_by_type/top_features)
      min_area_sqm=<float>      Min area m² (with high_confidence_type/parcels_by_type)
      processed_only=true       Only return processed KGs
      aggregate=true            Return aggregate stats instead of KG list
      limit=<N>                 Max results (default 100)
      offset=<N>                Offset for pagination
    """
    try:
        idx = si.get_index()
        args = request.args
        limit = min(int(args.get('limit', 100)), 1000)
        offset = int(args.get('offset', 0))
        processed_only = args.get('processed_only', '').lower() in ('true', '1', 'yes')
        do_aggregate = args.get('aggregate', '').lower() in ('true', '1', 'yes')
        do_async = args.get('async', '').lower() in ('true', '1', 'yes')

        # Poll existing async query task
        if args.get('task_id') and not args.get('q') and not args.get('kg'):
            task_id = args['task_id']
            p = _PROGRESS_DIR / f"{task_id}.json"
            if not p.exists():
                return jsonify({'error': 'Unknown task_id'}), 404
            try:
                info = json.loads(p.read_text())
            except Exception:
                return jsonify({'active': False})
            step = info.get('step', '')
            elapsed = round(time.time() - info.get('t0', time.time()), 1)
            if step == 'done':
                result = _get_query_result(task_id)
                if result is not None:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return jsonify(result)
                return jsonify({'active': False, 'done': True, 'elapsed': elapsed})
            if step == 'error':
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
                return jsonify({'error': info.get('detail', 'Query failed')}), 500
            return jsonify({'active': True, 'task_id': task_id, 'step': step,
                            'detail': info.get('detail', ''), 'elapsed': elapsed}), 202

        # Single KG lookup
        if args.get('kg'):
            result = idx.query_kg(args['kg'])
            return jsonify(result) if result else (jsonify({'error': 'KG not found'}), 404)

        # Parcel lookup
        if args.get('parcel'):
            result = idx.query_parcel(args['parcel'])
            return jsonify(result) if result else (jsonify({'error': 'Parcel not found'}), 404)

        # Aggregate queries
        # Aggregate queries (only when aggregate=true)
        if do_aggregate:
            if args.get('state'):
                return jsonify(idx.aggregate_state(args['state']))
            if args.get('district'):
                return jsonify(idx.aggregate_district(args['district']))
            if args.get('gemeinde'):
                return jsonify(idx.aggregate_gemeinde(args['gemeinde']))
            return jsonify(idx.aggregate_country())

        # Admin hierarchy — list KGs in a state/district/gemeinde
        if args.get('state'):
            sc = args['state']
            # Accept name or code
            if not sc.isdigit():
                for code, name in si.STATE_CODES.items():
                    if name.lower() == sc.lower():
                        sc = code
                        break
            return jsonify(idx.query_admin('state', sc, processed_only=processed_only, limit=limit, offset=offset))
        if args.get('district'):
            dc = args['district']
            # Accept name or code
            if not dc.isdigit():
                for code, name in si.DISTRICT_NAMES.items():
                    if name.lower() == dc.lower():
                        dc = code
                        break
            return jsonify(idx.query_admin('district', dc,
                                           processed_only=processed_only, limit=limit, offset=offset))
        if args.get('gemeinde'):
            gm = args['gemeinde']
            # Accept name or code — look up in DB if name given
            if not gm.isdigit():
                c = idx._conn()
                row = c.execute('SELECT gemeinde_code FROM kg WHERE gemeinde_name = ? LIMIT 1', (gm,)).fetchone()
                if row:
                    gm = row[0]
            return jsonify(idx.query_admin('gemeinde', gm,
                                           processed_only=processed_only, limit=limit, offset=offset))

        # Object type ranking
        if args.get('type'):
            metric = args.get('metric', 'area')
            return jsonify(idx.query_type_ranking(args['type'], metric=metric, limit=limit, offset=offset))

        # Hansen forest loss
        if args.get('hansen', '').lower() in ('true', '1', 'yes'):
            yf = int(args['year_from']) if args.get('year_from') else None
            yt = int(args['year_to']) if args.get('year_to') else None
            ml = int(args.get('min_loss', 0))
            return jsonify(idx.query_hansen_loss(year_from=yf, year_to=yt,
                                                 min_loss=ml, limit=limit, offset=offset))

        # New buildings
        if args.get('new_buildings', '').lower() in ('true', '1', 'yes'):
            mc = int(args.get('min_count', 1))
            return jsonify(idx.query_new_buildings(min_count=mc, limit=limit, offset=offset))

        # Classification divergence
        if args.get('divergence', '').lower() in ('true', '1', 'yes'):
            min_div = float(args.get('min_divergence', 0))
            rf_t = args.get('rf_type')
            fin_t = args.get('final_type')
            return jsonify(idx.query_divergence(
                min_pct=min_div, rf_type=rf_t, final_type=fin_t,
                limit=limit, offset=offset))

        # Divergence pairs (most common RF→final mismatches across all KGs)
        if args.get('divergence_pairs', '').lower() in ('true', '1', 'yes'):
            return jsonify(idx.query_divergence_pairs(limit=limit, offset=offset))

        # Low confidence KGs
        if args.get('low_confidence', '').lower() in ('true', '1', 'yes'):
            mc = float(args.get('max_confidence', 0.5))
            return jsonify(idx.query_low_confidence(
                max_confidence=mc, limit=limit, offset=offset))

        # Confidence ranking
        if args.get('confidence_rank'):
            order = args['confidence_rank']
            return jsonify(idx.query_confidence_ranking(
                order=order, limit=limit, offset=offset))

        # Per-type RF confidence ranking
        if args.get('type_confidence'):
            return jsonify(idx.query_type_confidence(
                args['type_confidence'], limit=limit, offset=offset))

        # High-confidence type filter (KG-level)
        # e.g. ?high_confidence_type=tree&min_confidence=0.8&min_area_sqm=1500
        if args.get('high_confidence_type'):
            mc = float(args.get('min_confidence', 0.7))
            ma = float(args.get('min_area_sqm', 0))
            return jsonify(idx.query_high_confidence_type(
                args['high_confidence_type'], min_confidence=mc,
                min_area_sqm=ma, limit=limit, offset=offset))

        # Per-parcel type+confidence filter (scans KG JSONs — supports async)
        # e.g. ?parcels_by_type=tree&min_confidence=0.8&min_area_sqm=1500
        if args.get('parcels_by_type'):
            mc = float(args.get('min_confidence', 0.7))
            ma = float(args.get('min_area_sqm', 0))
            if do_async:
                task_id = args.get('task_id') or str(uuid.uuid4())
                t = threading.Thread(target=_query_worker, daemon=True,
                    args=(task_id, idx.query_parcels_by_type_confidence,
                          (args['parcels_by_type'],),
                          dict(min_confidence=mc, min_area_sqm=ma,
                               limit=limit, offset=offset)))
                t.start()
                return jsonify({'task_id': task_id, 'status': 'running',
                                'poll': f'/api/v1/query?task_id={task_id}'}), 202
            return jsonify(idx.query_parcels_by_type_confidence(
                args['parcels_by_type'], min_confidence=mc,
                min_area_sqm=ma, limit=limit, offset=offset))

        # Cross-KG top features (trees/objects/new_buildings/infrastructure — supports async)
        # e.g. ?top_features=trees&min_confidence=0.9
        # e.g. ?top_features=new_buildings&min_confidence=0.75
        # e.g. ?top_features=infrastructure&type=mast&min_confidence=0.8
        if args.get('top_features'):
            mc = float(args.get('min_confidence', 0))
            otype = args.get('type')
            tf_bbox = None
            if args.get('bbox'):
                bp = [float(x) for x in args['bbox'].split(',')]
                if len(bp) == 4:
                    tf_bbox = tuple(bp)
            if do_async:
                task_id = args.get('task_id') or str(uuid.uuid4())
                t = threading.Thread(target=_query_worker, daemon=True,
                    args=(task_id, idx.query_top_features,
                          (args['top_features'],),
                          dict(object_type=otype, min_confidence=mc,
                               bbox=tf_bbox, limit=limit, offset=offset)))
                t.start()
                return jsonify({'task_id': task_id, 'status': 'running',
                                'poll': f'/api/v1/query?task_id={task_id}'}), 202
            return jsonify(idx.query_top_features(
                args['top_features'], object_type=otype,
                min_confidence=mc, bbox=tf_bbox,
                limit=limit, offset=offset))

        # Cross-KG segment-level power queries
        # e.g. ?segments=true&object_type=tree&min_rf_confidence=0.9&percentile=0.01
        # e.g. ?segments=true&object_type=excavation&sort=volume&min_rf_confidence=0.7
        # e.g. ?segments=true&object_type=tree_loss&min_rf_confidence=0.8&percentile=0.05
        if args.get('segments', '').lower() in ('true', '1', 'yes'):
            seg_bbox = None
            if args.get('bbox'):
                bp = [float(x) for x in args['bbox'].split(',')]
                if len(bp) == 4:
                    seg_bbox = tuple(bp)
            seg_kwargs = dict(
                object_type=args.get('object_type') or args.get('type'),
                min_rf_confidence=float(args['min_rf_confidence']) if args.get('min_rf_confidence') else None,
                max_rf_confidence=float(args['max_rf_confidence']) if args.get('max_rf_confidence') else None,
                min_confidence=float(args['min_confidence']) if args.get('min_confidence') else None,
                max_confidence=float(args['max_confidence']) if args.get('max_confidence') else None,
                min_area_sqm=float(args['min_area_sqm']) if args.get('min_area_sqm') else None,
                max_area_sqm=float(args['max_area_sqm']) if args.get('max_area_sqm') else None,
                min_height=float(args['min_height']) if args.get('min_height') else None,
                max_height=float(args['max_height']) if args.get('max_height') else None,
                min_volume=float(args['min_volume']) if args.get('min_volume') else None,
                max_volume=float(args['max_volume']) if args.get('max_volume') else None,
                bbox=seg_bbox,
                state=args.get('state'),
                district=args.get('district'),
                sort=args.get('sort', 'height_max_m'),
                sort_dir=args.get('sort_dir', 'desc'),
                percentile=float(args['percentile']) if args.get('percentile') else None,
                limit=limit, offset=offset,
            )
            return jsonify(idx.query_segments(**seg_kwargs))

        # Spatial bbox
        if args.get('bbox'):
            parts = [float(x) for x in args['bbox'].split(',')]
            if len(parts) == 4:
                return jsonify(idx.query_bbox(*parts, processed_only=processed_only, limit=limit, offset=offset))
            return jsonify({'error': 'bbox must be w,s,e,n'}), 400

        # Point proximity
        if args.get('point'):
            parts = [float(x) for x in args['point'].split(',')]
            if len(parts) == 2:
                radius = float(args.get('radius', 5))
                return jsonify(idx.query_point(parts[0], parts[1], radius_km=radius, limit=limit, offset=offset))
            return jsonify({'error': 'point must be lon,lat'}), 400

        # Full-text search
        if args.get('q'):
            return jsonify(idx.query_text(args['q'], limit=limit, offset=offset))

        # Default: list processed KGs
        return jsonify(idx.query_processed(limit=limit, offset=offset))

    except Exception as e:
        log.exception('query error')
        return jsonify({'error': str(e)}), 500


# === SECTION: Cross-API bridge (cadastre + landscape) ===

@app.route('/api/v1/lookup')
def api_lookup():
    """Proxy to cadastre lookup — fuzzy diacritics-insensitive search.

    Searches Austria's federal register (EDM): Gemeinden, KGs, Ortschaften, PLZ.
    Handles umlauts gracefully ("Kofla" → "Köflach").

    Params:
      q (required): Search text (name, PLZ, code)
      type: Filter by entity type (plz|gemeinde|kg|ortschaft)
      limit: Max results (default 20, max 200)
    """
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'error': 'q parameter required'}), 400
    try:
        result = cb.lookup(
            q=q,
            type=request.args.get('type'),
            limit=int(request.args.get('limit', 20)),
        )
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_lookup error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/parcels/batch', methods=['GET', 'POST'])
def api_parcels_batch():
    """Batch parcel landscape enrichment — 3 modes.

    Accepts both POST (JSON body) and GET (query params).

    GET maps to Mode 3 (compound → parcels). Compound filters use direct
    query params; parcel filters use pf_ prefix.

    GET examples:
      /api/v1/parcels/batch?state=Vorarlberg&pf_aspect=E&pf_terrain_class=level&limit=100
      /api/v1/parcels/batch?min_tree_count=50&pf_min_vegetated_fraction=0.5&pf_sort=conservation_score
      /api/v1/parcels/batch?type_filter=tree:0.8:800&pf_types=tree,grass&limit=50
      /api/v1/parcels/batch?parcel_ids=63349-505/3,75414-1314/1  (Mode 1 via GET)
      /api/v1/parcels/batch?min_elevation=900&aspect=SE,S,SW&min_slope=17&max_building_stories_max=1&min_building_pitched_pct=80&pf_aspect=SE,S,SW&pf_min_slope=17&pf_min_elevation=900&pf_max_parcel_area=5000&pf_roof_type=pitched&pf_max_stories=1&pf_cadastre_has_buildings=true&limit=100

    POST modes:

    Mode 1 — Explicit IDs:
      Body: {"parcel_ids": ["63349-505/3", "75414-1314/1", ...]}
      Max 200 parcel IDs per request.

    Mode 2 — Cadastre query:
      Body: {
        "query": {<cadastre /query params>},
        "landscape_filters": {<landscape post-filters>},
        "limit": 50, "offset": 0
      }

    Mode 3 — Landscape-first (compound → parcels):
      Start from our landscape index, find KGs matching compound filters,
      then expand to individual parcels from our KG JSONs.
      This is the power query for nature conservation screening.

      Body: {
        "compound": {                 // SearchIndex.query_compound() filters
          "type_filters": [
            {"type": "tree", "min_confidence": 0.8, "min_area_sqm": 800},
            {"type": "grass", "min_confidence": 0.8, "min_area_sqm": 1300}
          ],
          "max_buildings": 0,
          "aspect": ["S", "SW", "W"],
          "min_roughness": 2.0,
          // also: bbox, state, district, gemeinde, min_slope, min_elevation,
          // min_ndvi, min_vegetated_fraction, min_tree_count, phenology,
          // landcover_filters, sort, sort_dir, etc.
        },
        "parcel_filters": {           // Per-parcel post-filters on our JSON data
          "min_vegetated_fraction": 0.5,
          "min_elevation": 500,
          "max_elevation": 2000,
          "types": ["tree", "grass"],  // require these types in parcel
          "min_type_fraction": 0.1,   // min fraction per required type
          "min_ndsm_max": 5,          // min height above ground (m)
          "min_parcel_area": 1000,    // min parcel area (sqm)
          "is_vegetated": true,
          // Per-type classification confidence:
          "type_confidence": [
            {"type": "tree", "min_confidence": 0.7, "min_area_sqm": 500},
            {"type": "grass", "min_rf_confidence": 0.8},
            // Also: min_rf_count, min_rules_count, min_fraction, max_diverged_pct
          ],
          "min_confidence": 0.6,       // overall combined confidence
          "min_rf_confidence": 0.7,    // overall RF confidence
          // Building attribute filters (centroid → /spatial/points point-in-polygon):
          "roof_type": "pitched",      // "pitched" or "flat"
          "min_stories": 1,             // building stories_est >= N
          "max_stories": 1,             // building stories_est <= N
          // Cadastre-side filters (applied after enrichment):
          "cadastre_has_buildings": false,
          "cadastre_landuse": "W",
          "cadastre_min_area": 5000,
          "sort": "conservation_score", // |vegetated_fraction|elevation|ndsm_max
          "sort_dir": "desc"
        },
        "cadastre_enrich": true,      // fetch cadastre data (default true)
        "limit": 100, "offset": 0
      }

    Returns:
      {"results": [{"parcel_id": ..., "kg_code": ..., "landscape": {...},
                     "cadastre": {...}|null, "conservation_score": N}, ...],
       "total": N, "offset": N, "limit": N, "meta": {...}}
    """
    # --- Parse body: POST JSON or GET query params ---
    if request.method == 'POST':
        try:
            body = request.get_json(force=True) or {}
        except Exception:
            return jsonify({'error': 'Invalid JSON body'}), 400
    else:
        # GET — parse from query params
        args = request.args
        body = {}
        # Mode 1 via GET: parcel_ids=id1,id2,...
        if 'parcel_ids' in args:
            body['parcel_ids'] = [x.strip() for x in args['parcel_ids'].split(',') if x.strip()]
        else:
            # Mode 3 via GET: compound filters from query params, parcel filters from pf_ prefix
            compound = _parse_compound_from_args(args)
            if compound:
                body['compound'] = compound
                pf = _parse_parcel_filters_from_args(args)
                if pf:
                    body['parcel_filters'] = pf
                if 'cadastre_enrich' in args:
                    body['cadastre_enrich'] = args['cadastre_enrich'].lower() in ('true', '1', 'yes')
            # Check for Mode 2 via GET: query params with cq_ prefix
            # (not commonly used via GET, but supported)
        if 'limit' in args:
            body.setdefault('limit', int(args['limit']))
        if 'offset' in args:
            body.setdefault('offset', int(args['offset']))

    try:
        # Mode 1: Explicit parcel IDs
        if 'parcel_ids' in body:
            parcel_ids = body['parcel_ids']
            if isinstance(parcel_ids, str):
                parcel_ids = [x.strip() for x in parcel_ids.split(',') if x.strip()]
            if not isinstance(parcel_ids, list):
                return jsonify({'error': 'parcel_ids must be a list'}), 400
            if len(parcel_ids) > 200:
                return jsonify({'error': f'Max 200 parcels per request, got {len(parcel_ids)}'}), 400
            if len(parcel_ids) == 0:
                return jsonify({'results': [], 'total': 0})
            result = cb.batch_parcel_landscape(parcel_ids)
            return jsonify(result)

        # Mode 2: Query-based batch
        if 'query' in body:
            query_filters = body['query']
            if not isinstance(query_filters, dict) or not query_filters:
                return jsonify({'error': 'query must be a non-empty dict of cadastre filter params'}), 400
            landscape_filters = body.get('landscape_filters') or {}
            limit = min(int(body.get('limit', 50)), 500)
            offset = int(body.get('offset', 0))
            result = cb.batch_parcel_landscape_by_query(
                query_filters=query_filters,
                landscape_filters=landscape_filters,
                limit=limit,
                offset=offset,
            )
            return jsonify(result)

        # Mode 3: Landscape-first (compound query → parcels from our KG JSONs)
        if 'compound' in body:
            compound_filters = body['compound']
            if not isinstance(compound_filters, dict) or not compound_filters:
                return jsonify({'error': 'compound must be a non-empty dict of landscape filter params'}), 400
            parcel_filters = body.get('parcel_filters') or {}
            cadastre_enrich = body.get('cadastre_enrich', True)
            limit = min(int(body.get('limit', 100)), 1000)
            offset = int(body.get('offset', 0))
            result = cb.landscape_parcel_query(
                compound_filters=compound_filters,
                parcel_filters=parcel_filters,
                cadastre_enrich=cadastre_enrich,
                limit=limit,
                offset=offset,
            )
            return jsonify(result)

        return jsonify({'error': 'No filters provided. Use query params (state=, min_slope=, etc.) or POST JSON body with parcel_ids/query/compound.'}), 400

    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_parcels_batch error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/parcels/landscape')
def api_parcels_landscape():
    """Query parcels with landscape analysis context.

    Forwards cadastre query params to the cadastre /query endpoint, then enriches
    results with landscape data from our index/JSON files.

    Cadastre params (forwarded to cadastre /query):
      q, kg, gemeinde, district, state, plz, landuse, min_area, max_area,
      has_buildings, status, ez, has_legal_refs, legal_context,
      min_lon, min_lat, max_lon, max_lat, sort

    Landscape params (post-filter on our data):
      min_vegetated_fraction, max_vegetated_fraction: 0-1
      min_ndvi, max_ndvi: float
      min_tree_canopy_sqm: float
      min_elevation, max_elevation: float
      min_conservation_score: 0-100
      dominant_type: landscape object type
      landscape_sort: conservation_score|area|ndvi|tree_canopy|vegetated_fraction
      landscape_sort_dir: asc|desc

    Pagination: limit (default 50, max 500), offset
    """
    args = request.args

    # Separate cadastre params from landscape params
    cadastre_keys = {
        'q', 'kg', 'gemeinde', 'district', 'state', 'plz', 'landuse',
        'min_area', 'max_area', 'has_buildings', 'status', 'ez',
        'has_legal_refs', 'legal_context',
        'min_lon', 'min_lat', 'max_lon', 'max_lat', 'sort',
    }
    landscape_keys = {
        'min_vegetated_fraction', 'max_vegetated_fraction',
        'min_ndvi', 'max_ndvi', 'min_tree_canopy_sqm',
        'min_elevation', 'max_elevation', 'min_conservation_score',
        'dominant_type', 'landscape_sort', 'landscape_sort_dir',
    }

    query_filters = {k: args[k] for k in cadastre_keys if k in args}
    if not query_filters:
        return jsonify({'error': 'At least one cadastre filter param required'}), 400

    landscape_filters = {}
    for k in landscape_keys:
        if k in args:
            # Convert numeric landscape filters
            if k.startswith('min_') or k.startswith('max_'):
                try:
                    landscape_filters[k] = float(args[k])
                except ValueError:
                    return jsonify({'error': f'Invalid numeric value for {k}'}), 400
            elif k == 'landscape_sort':
                landscape_filters['sort'] = args[k]
            elif k == 'landscape_sort_dir':
                landscape_filters['sort_dir'] = args[k]
            else:
                landscape_filters[k] = args[k]

    limit = min(int(args.get('limit', 50)), 500)
    offset = int(args.get('offset', 0))

    try:
        result = cb.batch_parcel_landscape_by_query(
            query_filters=query_filters,
            landscape_filters=landscape_filters,
            limit=limit,
            offset=offset,
        )
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_parcels_landscape error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/query/nature')
def api_query_nature():
    """Nature conservation opportunity finder.

    Cross-references cadastre, protected areas, legal refs, and landscape
    analysis to find and rank parcels by conservation value.

    Params:
      bbox=w,s,e,n            Spatial filter (WGS84)
      state                   State name or code
      district                District code
      gemeinde                Gemeinde code or name
      protected_area=<name>   Near/in a WDPA protected area
      legal_context=<ctx>     Filter by legal context
                              (national_park, nature_protection, landscape_protection,
                               water_protection, species_protection, etc.)
      min_vegetated_fraction  Min vegetation fraction from landscape (0-1)
      min_ndvi                Min NDVI from landscape analysis
      min_tree_canopy_sqm     Min tree canopy area (sq metres)
      min_area_sqm            Min parcel area
      max_area_sqm            Max parcel area
      landuse                 Cadastre landuse code or abbreviation (W=Wald, LN=Landwirtschaft)
      has_buildings           Building presence filter (true|false)
      sort                    Sort key: conservation_score|area|ndvi|tree_canopy|vegetated_fraction
      limit (default 50), offset
    """
    args = request.args
    try:
        result = cb.nature_conservation_screen(
            bbox=args.get('bbox'),
            state=args.get('state'),
            district=args.get('district'),
            gemeinde=args.get('gemeinde'),
            protected_area=args.get('protected_area'),
            legal_context=args.get('legal_context'),
            min_vegetated_fraction=_float_or_none(args.get('min_vegetated_fraction')),
            min_ndvi=_float_or_none(args.get('min_ndvi')),
            min_tree_canopy_sqm=_float_or_none(args.get('min_tree_canopy_sqm')),
            min_area_sqm=_float_or_none(args.get('min_area_sqm')),
            max_area_sqm=_float_or_none(args.get('max_area_sqm')),
            landuse=args.get('landuse'),
            has_buildings=_bool_or_none(args.get('has_buildings')),
            sort=args.get('sort', 'conservation_score'),
            limit=min(int(args.get('limit', 50)), 500),
            offset=int(args.get('offset', 0)),
        )
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_query_nature error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/parcel/<path:parcel_id>/detail')
def api_parcel_detail(parcel_id):
    """Full combined detail for a single parcel (both APIs).

    Combines cadastre data (area, landuse, EZ, buildings, legal refs) with
    landscape analysis (elevation, NDVI, vegetation, classification, heights).
    Also checks protected area containment and computes conservation score.
    """
    try:
        result = cb.parcel_landscape_detail(parcel_id)
        if 'error' in result:
            return jsonify(result), 400
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_parcel_detail error')
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/kg/<kg_code>/profile')
def api_kg_profile(kg_code):
    """Combined KG profile from both APIs.

    Merges cadastre data (parcels, buildings, landuse distribution, legal refs)
    with landscape analysis (landcover, elevation, NDVI, trees, new buildings).
    """
    try:
        result = cb.kg_combined_profile(kg_code)
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        log.exception('api_kg_profile error')
        return jsonify({'error': str(e)}), 500


# --- Cadastre proxy endpoints ---

@app.route('/api/v1/cadastre/legal/search')
def api_cadastre_legal_search():
    """Proxy to cadastre legal search — find parcels referenced in Austrian law.

    Params: q (text), context (legal_context), type (listed|boundary_walk),
            bundesland, kg (KG code), limit, offset
    """
    try:
        params = {k: v for k, v in request.args.items()}
        result = cb.cadastre_proxy('/legal/search', params=params)
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/cadastre/protected_areas')
def api_cadastre_protected_areas():
    """Proxy to cadastre protected area search — 43 WDPA areas in Austria.

    Params: q (text), near_lon+near_lat (sort by proximity),
            contains_lon+contains_lat (point-in-polygon), limit, offset
    """
    try:
        params = {k: v for k, v in request.args.items()}
        result = cb.cadastre_proxy('/search/protected_area', params=params)
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/cadastre/landuse/distribution')
def api_cadastre_landuse_distribution():
    """Proxy to cadastre landuse distribution — aggregated by geography.

    Params: kg, gemeinde, district, state, code, abbr, group_by, limit, offset
    """
    try:
        params = {k: v for k, v in request.args.items()}
        result = cb.cadastre_proxy('/landuse/distribution', params=params)
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/cadastre/landuse/codes')
def api_cadastre_landuse_codes():
    """Proxy to cadastre landuse codes — reference table of all Austrian codes."""
    try:
        result = cb.cadastre_proxy('/landuse/codes')
        return jsonify(result)
    except cb.CadastreError as e:
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _float_or_none(val: str | None) -> float | None:
    """Parse a float from a query param, or return None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _bool_or_none(val: str | None) -> bool | None:
    """Parse a bool from a query param, or return None."""
    if val is None:
        return None
    return val.lower() in ('true', '1', 'yes')


# === SECTION: Geometry + parameter helpers ===

def _get_geometry():
    """Extract geometry from request. Supports JSON body, form data, file upload.

    Also validates each feature is within Austria and unions multiple features
    into a single convex-hull geometry.
    """
    if 'file' in request.files:
        f = request.files['file']
        content = f.read().decode('utf-8')
        features = geo_parse.parse_input(content)
    elif request.is_json:
        body = request.get_json()
        if 'geometry' in body:
            geom_input = body['geometry']
        elif 'type' in body:
            geom_input = body
        else:
            raise ValueError("JSON body must contain 'geometry' or be a valid GeoJSON")
        features = geo_parse.parse_input(geom_input)
    elif request.form.get('geometry'):
        features = geo_parse.parse_input(request.form['geometry'])
    else:
        data = request.get_data(as_text=True)
        if data:
            features = geo_parse.parse_input(data)
        else:
            raise ValueError("No geometry provided. Send GeoJSON, KML, or coordinates.")

    # Validate each feature is within Austria
    for feat in features:
        geo_parse.validate_austria_bounds(feat['geometry'])

    # Union multiple features into one
    if len(features) > 1:
        features = geo_parse.union_features(features)

    # Convert non-polygon geometries to polygon via buffer + union
    for feat in features:
        geom = feat['geometry']
        if geom.geom_type not in ('Polygon', 'MultiPolygon'):
            feat['geometry'] = _non_polygon_to_polygon(geom)

    return features


def _clean_polygon(geom, min_hole_area=500, min_part_area=100):
    """Remove small holes and tiny polygon slivers from a geometry (metric CRS)."""
    from shapely.geometry import Polygon as ShpPolygon, MultiPolygon as ShpMultiPolygon
    if geom.geom_type == 'Polygon':
        kept = [h for h in geom.interiors if ShpPolygon(h).area >= min_hole_area]
        return ShpPolygon(geom.exterior, kept)
    elif geom.geom_type == 'MultiPolygon':
        parts = [_clean_polygon(p, min_hole_area, min_part_area)
                 for p in geom.geoms if p.area >= min_part_area]
        if not parts:
            return geom
        return parts[0] if len(parts) == 1 else ShpMultiPolygon(parts)
    return geom


def _non_polygon_to_polygon(geom):
    """Convert non-polygon geometry to Polygon.

    For lines: polygonize the network, union, buffer to close gaps, simplify.
    For points: buffer 10m.
    Falls back to convex hull on error.
    """
    from shapely.ops import transform as shp_transform, linemerge, polygonize, unary_union
    import pyproj

    try:
        to_m = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True).transform
        to_ll = pyproj.Transformer.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True).transform

        if geom.geom_type in ('LineString', 'MultiLineString'):
            # Polygonize the line network first
            merged_lines = linemerge(geom)
            polys = list(polygonize(merged_lines))
            if polys:
                result = unary_union(polys)
                # Project to metric; gently close road-width gaps while
                # preserving boundary detail: expand 5m to bridge narrow
                # road gaps, remove small holes (< 2000 m² = road strips),
                # then shrink back and lightly simplify.
                result_m = shp_transform(to_m, result)
                result_m = result_m.buffer(5)
                result_m = _clean_polygon(result_m, min_hole_area=2000)
                result_m = result_m.buffer(-5)
                result_m = _clean_polygon(result_m, min_hole_area=500)
                result_m = result_m.simplify(0.5)
                result = shp_transform(to_ll, result_m)
                if result.geom_type in ('Polygon', 'MultiPolygon') and not result.is_empty:
                    n_verts = (len(result.exterior.coords) if result.geom_type == 'Polygon'
                               else sum(len(p.exterior.coords) for p in result.geoms))
                    log.info("_parse_geometry: polygonize+union %s → %s (%d polygonized, %d vertices)",
                             geom.geom_type, result.geom_type, len(polys), n_verts)
                    return result

            # Polygonize failed — buffer lines directly
            geom_m = shp_transform(to_m, geom)
            result_m = geom_m.buffer(10).simplify(1)
            result = shp_transform(to_ll, result_m)
            if result.geom_type in ('Polygon', 'MultiPolygon') and not result.is_empty:
                log.info("_parse_geometry: line buffer fallback %s → %s", geom.geom_type, result.geom_type)
                return result
        else:
            # Points/other: buffer 10m
            geom_m = shp_transform(to_m, geom)
            result_m = geom_m.buffer(10).simplify(1)
            result = shp_transform(to_ll, result_m)
            if result.geom_type in ('Polygon', 'MultiPolygon') and not result.is_empty:
                log.info("_parse_geometry: buffer %s → %s", geom.geom_type, result.geom_type)
                return result
    except Exception:
        log.warning("_parse_geometry: polygonize/buffer failed, falling back", exc_info=True)

    hull = geom.convex_hull
    if hull.geom_type == 'Polygon' and not hull.is_empty:
        return hull
    return geom.buffer(0.001)


def _get_params():
    params = {}
    if request.is_json:
        body = request.get_json()
        params = {k: v for k, v in body.items()
                  if k not in ('geometry', 'type', 'features', 'coordinates')}
    for key in ('dataset', 'date_a', 'date_b', 'dates',
                'min_height', 'max_height', 'min_area', 'min_change',
                'object_types', 'resolution', 'format',
                'include_ortho', 'include_temporal',
                'include_copernicus', 'include_cadastre',
                'include_hansen', 'include_infra', 'mark_uncertain', 'color_mode', 'types',
                'ortho_year', 'min_object_size',
                'felz_scale', 'rag_threshold', 'groups',
                'include_dtm', 'include_dsm', 'include_segments', 'include_segments_vector',
                'ortho_years', 'raster_layers',
                'top_n_classes', 'top_n_objects', 'min_height_m',
                'layer', 'min_zoom', 'max_zoom',
                'share_id',
                'height_min', 'height_max', 'height_op'):
        val = request.args.get(key)
        if val is not None:
            params[key] = val
    return params


def _validate_area(geom_3035):
    area = geom_3035.area
    if area > MAX_AREA_SQM:
        raise ValueError(
            f"The selected area is too large ({area/1e6:.1f} km²). "
            f"Please choose a smaller region — the maximum allowed is "
            f"{MAX_AREA_SQM/1e6:.0f} km²."
        )


def _error(msg, code=400):
    return jsonify({"error": str(msg)}), code


def _rf_model_meta() -> dict:
    """Return RF model version info for response metadata."""
    try:
        import learned_classifier as lc
        clf = lc.get_classifier()
        if clf.is_trained:
            return {
                "rf_trained_at": clf.trained_at,
                "rf_n_kgs": clf.n_kgs,
                "rf_oob": round(clf.oob_score, 4),
                "rf_n_train": clf.n_train,
            }
    except Exception:
        pass
    return {}


def _try_read_ortho(data: dict) -> tuple:
    """Attempt to read RGB+NIR ortho aligned to ALS data.

    Returns (rgb, spectral) or (None, None).  *spectral* will include
    an ``"ndvi"`` key when NIR is available from an RGBI operate.
    """
    try:
        import ortho_io
        rgb, nir = ortho_io.read_ortho_for_als(data)
        spectral = ortho_io.compute_spectral_indices(rgb, nir=nir)
        # Add raw bands for object_segmentation fused gradient + classification
        if rgb is not None:
            spectral["red"] = rgb[0].astype(np.float32)
            spectral["green"] = rgb[1].astype(np.float32)
            spectral["blue"] = rgb[2].astype(np.float32)
        if nir is not None:
            spectral["nir"] = nir.astype(np.float32)
        return rgb, spectral
    except Exception as e:
        log.warning("Ortho read failed (non-fatal): %s", e)
        return None, None


def _try_copernicus(geom_wgs84, *, ndvi=True, landcover=True, sar=False,
                    harmonics=False, year: int = 2023) -> dict | None:
    """Attempt to fetch Copernicus data for a geometry.

    Always tries the tile cache (local files + Zenodo) first.  Falls back
    to the live openEO API only when cache misses AND the Austria processor
    is not running (to avoid credential conflicts).

    Parameters
    ----------
    year : int
        Observation year.  NDVI composite and SAR backscatter are fetched
        for the growing season (Apr–Sep) of this year.
    """
    bbox = geom_wgs84.bounds  # (minx, miny, maxx, maxy)
    bbox_dict = {"west": bbox[0], "south": bbox[1], "east": bbox[2], "north": bbox[3]}

    # --- Fast path: serve from tile cache (local + Zenodo) ---
    cached = _try_copernicus_cached(bbox_dict, ndvi=ndvi, landcover=landcover,
                                    sar=sar, harmonics=harmonics, year=year)
    if cached is not None:
        return cached

    # Cache miss — need live API.  Block if processor is running.
    if _is_processor_running():
        log.info('Copernicus cache miss — skipping (processor running)')
        return None

    # --- Slow path: live openEO API ---
    try:
        import copernicus
        result = {}
        if ndvi:
            try:
                ndvi_data = copernicus.get_ndvi_composite(bbox_dict, year=year)
                result["ndvi"] = ndvi_data["ndvi"]
                result["transform"] = ndvi_data["transform"]
                result["crs"] = ndvi_data["crs"]
            except Exception as e:
                log.warning("Copernicus NDVI failed: %s", e)
        if landcover:
            try:
                lc = copernicus.get_land_cover(bbox_dict)
                result["landcover"] = lc
            except Exception as e:
                log.warning("Copernicus land cover failed: %s", e)
        if sar:
            try:
                sar_start = f"{year}-06-01"
                sar_end   = f"{year}-09-30"
                sar_data = copernicus.get_sar_backscatter(bbox_dict, sar_start, sar_end)
                result["vv"] = sar_data["vv"]
                result["vh"] = sar_data["vh"]
                result["sar_transform"] = sar_data["transform"]
                result["sar_crs"] = sar_data["crs"]
            except Exception as e:
                log.warning("Copernicus SAR failed: %s", e)
        if harmonics:
            try:
                import ndvi_harmonics
                harm = ndvi_harmonics.get_harmonic_features(bbox_dict, year=year)
                if harm:
                    result["harmonics"] = harm
                    log.info("NDVI harmonics: mean amp=%.3f",
                             float(np.nanmean(harm.get("h_amplitude", [0]))))
            except Exception as e:
                log.warning("NDVI harmonics failed: %s", e)
        return result if result else None
    except ImportError:
        log.info("Copernicus module not available")
        return None
    except Exception as e:
        log.warning("Copernicus data failed: %s", e)
        return None


def _try_copernicus_cached(bbox_dict: dict, *, ndvi=True, landcover=True,
                           sar=False, harmonics=False,
                           year: int = 2023) -> dict | None:
    """Serve Copernicus data from tile cache (local + Zenodo) only.

    Reads per-cell (0.1°) cached tiles and mosaics them if the request
    spans multiple cells.  Returns None on any miss so the caller can
    fall back to the live API.
    """
    try:
        from tile_cache import CopernicusTileCache
        cache = CopernicusTileCache()

        if not cache.has_cached(bbox_dict, ndvi=ndvi, landcover=landcover,
                                sar=sar, harmonics=harmonics, year=year):
            return None

        log.info("Copernicus serving from tile cache")
        result = {}
        if ndvi:
            d = cache.read_cached_product(bbox_dict, "ndvi", year=year)
            if d and "ndvi" in d:
                result["ndvi"] = d["ndvi"]
                result["transform"] = d["transform"]
                result["crs"] = d["crs"]
            else:
                return None
        if landcover:
            d = cache.read_cached_product(bbox_dict, "worldcover", year=year)
            if d:
                result["landcover"] = d
            else:
                return None
        if sar:
            d = cache.read_cached_product(bbox_dict, "sar", year=year)
            if d and "vv" in d:
                result["vv"] = d["vv"]
                result["vh"] = d["vh"]
                result["sar_transform"] = d["transform"]
                result["sar_crs"] = d["crs"]
            else:
                return None
        if harmonics:
            d = cache.read_cached_product(bbox_dict, "harmonics", year=year)
            if d:
                result["harmonics"] = d
            else:
                return None
        return result if result else None
    except Exception as e:
        log.debug("Copernicus cache read failed: %s", e)
        return None


def _try_cadastre(geom_wgs84, transform, shape) -> np.ndarray | None:
    """Attempt to fetch building footprints from cadastre."""
    try:
        import cadastre
        bbox = geom_wgs84.bounds
        return cadastre.get_building_mask(bbox, transform, shape)
    except ImportError:
        log.info("Cadastre module not available")
        return None
    except Exception as e:
        log.warning("Cadastre fetch failed: %s", e)
        return None


def _try_hansen(geom_wgs84, transform, shape) -> dict | None:
    """Attempt to load Hansen forest prior for segment_and_classify."""
    try:
        bbox = geom_wgs84.bounds
        return hansen.get_forest_prior(bbox, transform, shape)
    except Exception as e:
        log.warning("Hansen prior failed: %s", e)
        return None


def _clear_raster_caches():
    """Delete cached .npz / .tif / batch dirs to reclaim memory after training."""
    import pathlib, shutil
    cleared = 0
    for cache_dir in [
        pathlib.Path("/tmp/copernicus_cache"),
        pathlib.Path("/tmp/hansen_cache"),
    ]:
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                try:
                    if f.is_dir():
                        shutil.rmtree(f)
                    else:
                        f.unlink()
                    cleared += 1
                except Exception:
                    pass
    if cleared:
        log.info("Cleared %d cached raster entries after training", cleared)


# === SECTION: /api/v1/elevation endpoint ===

@app.route('/api/v1/elevation', methods=['POST'])
def elevation():
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)

        result_features = []
        for feat in features:
            geom = feat['geometry']
            geom_3035 = ti.geometry_to_3035(geom)

            if geom.geom_type == 'Point':
                e, n = geom_3035.coords[0][:2]
                bounds = (e - 5, n - 5, e + 5, n + 5)
                dsm_data, tf, _ = raster_io.read_window_bbox('DSM', *bounds, dataset, pad=0)
                dtm_data, _, _ = raster_io.read_window_bbox('DTM', *bounds, dataset, pad=0)
                row = max(0, min(int((tf.f - n) / abs(tf.e)), dsm_data.shape[0]-1))
                col = max(0, min(int((e - tf.c) / tf.a), dsm_data.shape[1]-1))
                dsm_val = round(float(dsm_data[row, col]), 2)
                dtm_val = round(float(dtm_data[row, col]), 2)
                props = dict(feat.get('properties', {}))
                props['dsm_elevation_m'] = dsm_val
                props['dtm_elevation_m'] = dtm_val
                props['object_height_m'] = round(float(dsm_val - dtm_val), 2)
                # Add DSM altitude as Z coordinate
                geom_z = Point(geom.x, geom.y, dsm_val)
                result_features.append({"type": "Feature", "properties": props, "geometry": mapping(geom_z)})

            elif geom.geom_type in ('LineString', 'MultiLineString'):
                _validate_area(geom_3035.buffer(10))
                coords_3035 = list(geom_3035.coords) if geom.geom_type == 'LineString' else \
                    [c for ls in geom_3035.geoms for c in ls.coords]
                bounds = geom_3035.bounds
                dsm_data, tf, _ = raster_io.read_window_bbox('DSM', *bounds, dataset)
                dtm_data, _, _ = raster_io.read_window_bbox('DTM', *bounds, dataset)
                enriched_coords = []
                coords_wgs_3d = []
                for e, n in coords_3035:
                    row = max(0, min(int((tf.f - n) / abs(tf.e)), dsm_data.shape[0]-1))
                    col = max(0, min(int((e - tf.c) / tf.a), dsm_data.shape[1]-1))
                    dsm_val = round(float(dsm_data[row, col]), 2)
                    dtm_val = round(float(dtm_data[row, col]), 2)
                    pt_wgs = ti.geometry_from_3035(Point(e, n))
                    enriched_coords.append({
                        "lon": round(pt_wgs.x, 8), "lat": round(pt_wgs.y, 8),
                        "dsm_elevation_m": dsm_val,
                        "dtm_elevation_m": dtm_val,
                        "object_height_m": round(float(dsm_val - dtm_val), 2),
                    })
                    coords_wgs_3d.append((round(pt_wgs.x, 8), round(pt_wgs.y, 8), dsm_val))
                props = dict(feat.get('properties', {}))
                props['elevation_profile'] = enriched_coords
                props['dsm_elevation_min'] = min(p['dsm_elevation_m'] for p in enriched_coords)
                props['dsm_elevation_max'] = max(p['dsm_elevation_m'] for p in enriched_coords)
                # Return geometry with DSM altitude as Z coordinate
                geom_z = SLineString(coords_wgs_3d)
                result_features.append({"type": "Feature", "properties": props, "geometry": mapping(geom_z)})

            else:
                _validate_area(geom_3035)
                data = raster_io.read_dtm_dsm(geom_3035, dataset)
                dtm_valid = data['dtm'][data['mask']]
                dsm_valid = data['dsm'][data['mask']]
                ndsm_valid = data['ndsm'][data['mask']]
                props = dict(feat.get('properties', {}))
                for name, arr in [('dsm_elevation', dsm_valid), ('dtm_elevation', dtm_valid), ('object_heights', ndsm_valid)]:
                    props[name] = {'min': round(float(np.nanmin(arr)), 2), 'max': round(float(np.nanmax(arr)), 2), 'mean': round(float(np.nanmean(arr)), 2)}
                props['area_sqm'] = int(np.sum(data['mask']))
                # Add mean DSM altitude as Z coordinate on polygon exterior
                dsm_mean = round(float(np.nanmean(dsm_valid)), 2)
                geom_dict = mapping(geom)
                if geom_dict.get('type') == 'Polygon' and geom_dict.get('coordinates'):
                    geom_dict['coordinates'] = tuple(
                        tuple((x, y, dsm_mean) for x, y, *_ in ring)
                        for ring in geom_dict['coordinates']
                    )
                result_features.append({"type": "Feature", "properties": props, "geometry": geom_dict})

        return jsonify({"type": "FeatureCollection", "features": result_features,
                        "meta": {"dataset": dataset, "processing_time_s": round(time.time()-t0, 2)}})
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# === SECTION: /api/v1/terrain endpoint ===

@app.route('/api/v1/terrain', methods=['POST'])
def terrain():
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        results = []
        for feat in features:
            geom = feat['geometry']
            geom_3035 = ti.geometry_to_3035(geom)
            _validate_area(geom_3035.buffer(10) if geom.geom_type == 'Point' else geom_3035)
            if geom.geom_type == 'Point':
                geom_3035 = geom_3035.buffer(50)
            dtm_data, mask, tf, crs = raster_io.read_masked('DTM', geom_3035, dataset)
            terrain_stats = ta.characterise_terrain(dtm_data, mask)
            props = dict(feat.get('properties', {}))
            props['terrain'] = terrain_stats
            results.append({"type": "Feature", "properties": props, "geometry": mapping(feat['geometry'])})
        return jsonify({"type": "FeatureCollection", "features": results,
                        "meta": {"dataset": dataset, "processing_time_s": round(time.time()-t0, 2)}})
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)





# === SECTION: /api/v1/segment endpoint (async segmentation pipeline) ===

@app.route('/api/v1/segment', methods=['POST'])
def segment_objects():
    """Watershed-based object segmentation and classification.

    Fused gradient → Felzenszwalb → RAG merge → per-object features → classify → group.
    Returns individual objects (tree, roof, road_surface, …) AND groups (forest, building, …).

    When async=true is passed, returns 202 immediately with {task_id, status: 'running'}.
    Poll /api/v1/segment/progress?task_id=... for status.
    Fetch /api/v1/segment/result?task_id=... when done.
    """
    task_id = request.args.get('task_id', '')
    run_async = str(request.args.get('async', 'false')).lower() in ('true', '1', 'yes')

    # Parse request data upfront (must happen in request context)
    try:
        features = _get_geometry()
        params = _get_params()
    except Exception as e:
        return _error(str(e))

    # Capture raw geometry text for auto-save share recovery
    geometry_text = ''
    try:
        from shapely.geometry import mapping
        if features:
            geom_dict = mapping(features[0]['geometry'])
            geometry_text = json.dumps(geom_dict)
    except Exception:
        pass

    if run_async:
        if not task_id:
            task_id = str(uuid.uuid4())
        # Queue check: reject early if too many waiting
        with _TASK_QUEUE_LOCK:
            if _TASK_QUEUE_SIZE >= MAX_QUEUE_SIZE:
                return _error('Server busy — too many tasks queued. Try again in a few minutes.', 503)
        _progress_start(task_id)
        _cleanup_old_results()
        thread = threading.Thread(
            target=_segment_worker,
            args=(task_id, features, params, geometry_text),
            daemon=True,
        )
        thread.start()
        return jsonify({"task_id": task_id, "status": "running", "queued": _TASK_QUEUE_SIZE}), 202

    return _segment_sync(task_id, features, params)


def _segment_worker(task_id: str, features: list, params: dict, geometry_text: str = ''):
    """Background worker for async segment processing (queue-aware)."""
    global _TASK_QUEUE_SIZE
    with _TASK_QUEUE_LOCK:
        _TASK_QUEUE_SIZE += 1
    _progress_set(task_id, 'queued', f'Waiting for slot ({_TASK_QUEUE_SIZE} in queue)…')
    try:
        acquired = _TASK_SEMAPHORE.acquire(timeout=300)  # wait up to 5 min for a slot
        with _TASK_QUEUE_LOCK:
            _TASK_QUEUE_SIZE = max(0, _TASK_QUEUE_SIZE - 1)
        if not acquired:
            _progress_error(task_id, 'Server busy — timed out waiting in queue. Try again later.')
            return
        try:
            resp = _segment_core(task_id, features, params)
            _store_result(task_id, resp)
            import gc; gc.collect()
            share_id = _auto_save_share(task_id, resp, geometry_text, params)
            _progress_done(task_id, auto_share_id=share_id)
            log.info("Async segment task %s completed (auto-share=%s)", task_id, share_id)
        except Exception as e:
            log.error("Async segment task %s failed: %s", task_id, traceback.format_exc())
            _progress_error(task_id, str(e))
        finally:
            _TASK_SEMAPHORE.release()
    except Exception as e:
        with _TASK_QUEUE_LOCK:
            _TASK_QUEUE_SIZE = max(0, _TASK_QUEUE_SIZE - 1)
        _progress_error(task_id, str(e))


def _unique_share_id(base_name: str) -> str:
    """Return base_name if unused, otherwise append date + counter (e.g. MyArea-0415-2)."""
    candidate = base_name
    if not (SHARE_DIR / f"{candidate}.json.gz").exists():
        return candidate
    # Collision — append date stamp
    date_suffix = time.strftime('%m%d')
    candidate = f"{base_name}-{date_suffix}"
    if not (SHARE_DIR / f"{candidate}.json.gz").exists():
        return candidate
    # Still collides — add counter
    for i in range(2, 100):
        candidate = f"{base_name}-{date_suffix}-{i}"
        if not (SHARE_DIR / f"{candidate}.json.gz").exists():
            return candidate
    # Fallback
    return f"{base_name}-{date_suffix}-{int(time.time()) % 10000}"


def _auto_save_share(task_id: str, result: dict, geometry_text: str, params: dict):
    """Auto-save completed analysis as a share for recovery."""
    try:
        # Check if onestop meta specifies a custom save name
        custom_name = None
        onestop_meta_path = _PROGRESS_DIR / f"{task_id}.onestop.json"
        if onestop_meta_path.exists():
            try:
                meta = json.loads(onestop_meta_path.read_text())
                custom_name = meta.get('params', {}).get('name', '')
                if custom_name:
                    custom_name = custom_name.strip().strip("'").strip('"')
            except Exception:
                pass
        if custom_name and _valid_share_id(custom_name):
            share_id = _unique_share_id(custom_name)
        else:
            share_id = f"auto-{task_id[:8]}"
        state = {
            'v': 1, 'center': [47.3, 15.3], 'zoom': 14,
            'endpoint': 'segment',
            'min_area': int(params.get('min_object_size', 30)),
            'ortho': str(params.get('include_ortho', 'true')).lower() in ('true', '1', 'yes'),
            'temporal': str(params.get('include_temporal', 'false')).lower() in ('true', '1', 'yes'),
            'copernicus': str(params.get('include_copernicus', 'false')).lower() in ('true', '1', 'yes'),
            'cadastre': str(params.get('include_cadastre', 'false')).lower() in ('true', '1', 'yes'),
            'hansen': str(params.get('include_hansen', 'false')).lower() in ('true', '1', 'yes'),
            'mark_uncertain': str(params.get('mark_uncertain', 'false')).lower() in ('true', '1', 'yes'),
            'geometry': geometry_text,
        }
        # Compute center from result bbox if available
        if result.get('features'):
            lats, lngs = [], []
            for f in result['features'][:100]:
                coords = f.get('geometry', {}).get('coordinates')
                if coords:
                    def _extract(c):
                        if isinstance(c, (list, tuple)) and len(c) >= 2 and isinstance(c[0], (int, float)):
                            lngs.append(c[0]); lats.append(c[1])
                        elif isinstance(c, (list, tuple)):
                            for x in c: _extract(x)
                    _extract(coords)
            if lats and lngs:
                state['center'] = [round(sum(lats)/len(lats), 6), round(sum(lngs)/len(lngs), 6)]
                state['zoom'] = 15

        name = custom_name if custom_name and _valid_share_id(custom_name) else f"Auto-save {time.strftime('%Y-%m-%d %H:%M')}"
        payload = {'state': state, 'result': result, 'name': name}

        # Tag one-stop shares with their onestop params for direct-download UX
        onestop_meta_path = _PROGRESS_DIR / f"{task_id}.onestop.json"
        if onestop_meta_path.exists():
            try:
                payload['onestop'] = json.loads(onestop_meta_path.read_text())
            except Exception:
                pass
        data_json = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        data = gzip.compress(data_json.encode())

        SHARE_DIR.mkdir(parents=True, exist_ok=True)
        (SHARE_DIR / f'{share_id}.json.gz').write_bytes(data)
        _share_evict()
        log.info("auto-save: saved share %s (%d KB)", share_id, len(data) // 1024)
        return share_id
    except Exception as e:
        log.error("auto-save failed: %s", e)
        return None


def _segment_sync(task_id: str, features: list, params: dict):
    """Synchronous segment processing (original behavior)."""
    try:
        resp = _segment_core(task_id, features, params)
        _progress_end(task_id)
        return jsonify(resp)
    except ValueError as e:
        _progress_end(task_id)
        return _error(str(e))
    except Exception as e:
        _progress_end(task_id)
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


def _segment_core(task_id: str, features: list, params: dict) -> dict:
    """Core segment processing logic. Returns response dict."""
    def _prog(step, detail=''):
        if task_id:
            _progress_set(task_id, step, detail)

    t0 = time.time()
    if task_id:
        _progress_start(task_id)
    _prog('Parsing geometry')
    dataset = params.get('dataset', ti.DEFAULT_DATASET)
    min_object_size = int(params.get('min_object_size', 30))
    felz_scale = float(params.get('felz_scale', 150))
    rag_threshold = float(params.get('rag_threshold', 0.12))
    include_ortho = str(params.get('include_ortho', 'true')).lower() in ('true', '1', 'yes')
    include_temporal = str(params.get('include_temporal', 'false')).lower() in ('true', '1', 'yes')
    include_copernicus = str(params.get('include_copernicus', 'false')).lower() in ('true', '1', 'yes')
    include_cadastre = str(params.get('include_cadastre', 'false')).lower() in ('true', '1', 'yes')
    include_hansen = str(params.get('include_hansen', 'false')).lower() in ('true', '1', 'yes')
    include_infra = str(params.get('include_infra', 'true')).lower() in ('true', '1', 'yes')
    mark_uncertain = str(params.get('mark_uncertain', 'false')).lower() in ('true', '1', 'yes')
    type_filter = params.get('types', None)
    if isinstance(type_filter, str):
        type_filter = [t.strip() for t in type_filter.split(',')]
    group_filter = params.get('groups', None)
    if isinstance(group_filter, str):
        group_filter = [g.strip() for g in group_filter.split(',')]

    all_objects = []
    all_stats = None
    all_evaluation = None
    all_labels = None
    all_transform = None
    all_shape = None
    all_mask = None
    hansen_evaluation = None

    for feat in features:
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(100)
        _validate_area(geom_3035)

        # Load DTM/DSM
        _prog('Loading DTM/DSM', 'remote raster reads')
        dtm_dates, dsm_dates = None, None
        if include_temporal:
            _prog('Loading DTM/DSM', 'multi-temporal (3 dates)')
            try:
                multi = raster_io.read_multi_date_ndsm(geom_3035)
                data = {
                    'dtm': multi['dtm'], 'dsm': multi['dsm'],
                    'ndsm': multi['ndsm'], 'mask': multi['mask'],
                    'transform': multi['transform'], 'crs': multi['crs'],
                    'shape': multi['shape'],
                }
                dtm_dates, dsm_dates = {}, {}
                for d in multi['dates_loaded']:
                    try:
                        _prog('Loading DTM/DSM', f'date {d}')
                        dd = raster_io.read_dtm_dsm(geom_3035, dataset=d)
                        mh = min(dd['shape'][0], data['shape'][0])
                        mw = min(dd['shape'][1], data['shape'][1])
                        dtm_dates[d] = dd['dtm'][:mh, :mw]
                        dsm_dates[d] = dd['dsm'][:mh, :mw]
                    except Exception as e:
                        log.warning("Date %s load failed: %s", d, e)
            except Exception as e:
                log.warning("Multi-temporal failed, single date: %s", e)
                data = raster_io.read_dtm_dsm(geom_3035, dataset)
        else:
            data = raster_io.read_dtm_dsm(geom_3035, dataset)

        rgb, spectral = (None, None)
        if include_ortho:
            _prog('Loading orthophoto', 'RGBI 20cm')
            rgb, spectral = _try_read_ortho(data)

        copernicus_data = None
        if include_copernicus:
            _prog('Loading Copernicus', 'NDVI + landcover + SAR + harmonics')
            copernicus_data = _try_copernicus(geom, sar=True, harmonics=True, year=ti.dataset_to_year(dataset))

        building_footprints = None
        if include_cadastre:
            _prog('Loading cadastre', 'building footprints')
            building_footprints = _try_cadastre(
                geom, data['transform'], data['shape'],
            )

        hansen_data = None
        if include_hansen:
            _prog('Loading Hansen', 'forest change data')
            hansen_data = _try_hansen(geom, data['transform'], data['shape'])

        # Infrastructure lookup (for rule-based solar/wind/substation/mast)
        _infra_lookup = None
        if include_infra:
            try:
                from infrastructure_lookup import InfrastructureLookup
                from pyproj import Transformer as _Tx
                _tx = _Tx.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
                bounds_3035 = geom.bounds  # (minx, miny, maxx, maxy) in EPSG:3035
                w4, s4 = _tx.transform(bounds_3035[0], bounds_3035[1])
                e4, n4 = _tx.transform(bounds_3035[2], bounds_3035[3])
                _infra_lookup = InfrastructureLookup.for_bbox(w4, s4, e4, n4)
                log.info('Infrastructure lookup: %d features', len(_infra_lookup))
            except Exception as _ie:
                log.warning('Infrastructure lookup failed: %s', _ie)

        # Run segmentation pipeline
        _prog('Segmenting & classifying', 'watershed + classification')
        obs_year = ti.dataset_to_year(dataset)
        result = seg.segment_and_classify(
            data['dtm'], data['dsm'], data['mask'], data['transform'],
            dtm_dates=dtm_dates,
            dsm_dates=dsm_dates,
            spectral=spectral,
            copernicus=copernicus_data,
            building_footprints=building_footprints,
            hansen=hansen_data,
            min_object_size=min_object_size,
            felz_scale=felz_scale,
            rag_threshold=rag_threshold,
            observation_year=obs_year,
            infra_lookup=_infra_lookup,
            mark_uncertain=mark_uncertain,
        )

        objects = result['objects']
        labels = result['labels']

        # Free heavy intermediates that segment_and_classify already consumed
        del dtm_dates, dsm_dates, spectral, copernicus_data, building_footprints
        import gc; gc.collect()

        # Hansen forest loss calibration
        hansen_evaluation = None
        if include_hansen and hansen_data:
            try:
                objects = hansen.calibrate_tree_loss(objects, labels, hansen_data, observation_year=obs_year)
                hansen_evaluation = hansen.evaluate_forest_loss(objects, labels, hansen_data, observation_year=obs_year)
            except Exception as e:
                log.warning("Hansen calibration failed: %s", e)

        # Populate seg_cache so overlay/gpkg endpoints can reuse results
        seg_cache_key = f"{geom_3035.bounds}_{dataset}_{include_ortho}_{include_copernicus}_{include_cadastre}_{include_hansen}_{include_infra}_{mark_uncertain}_temporal"
        _seg_cache.update({
            "labels": labels, "objects": objects,
            "mask": data['mask'], "transform": data['transform'],
            "shape": data['shape'], "ndsm": data.get('ndsm'), "key": seg_cache_key,
        })
        # Populate raster data cache so overlay endpoints don't re-fetch
        raster_cache_key = f"{geom_3035.bounds}_{dataset}"
        _raster_cache.update({"key": raster_cache_key, "data": data})
        if rgb is not None:
            _raster_cache.update({"ortho": rgb, "ortho_key": raster_cache_key})
        log.info("segment: cached results for overlay reuse")
        # Persist to disk so other gunicorn workers can reuse
        _seg_cache_save(seg_cache_key, labels, objects, data['mask'],
                        data['transform'], data['shape'], data.get('ndsm'))

        # Filters
        if type_filter:
            objects = [o for o in objects if o.obj_type in type_filter]
        if group_filter:
            objects = [o for o in objects if o.group_type in group_filter]

        # top_n_classes: keep only top N most frequent types
        top_n_classes = params.get('top_n_classes')
        if top_n_classes:
            top_n_classes = int(top_n_classes)
            from collections import Counter
            type_counts = Counter(o.obj_type for o in objects)
            top_types = set(t for t, _ in type_counts.most_common(top_n_classes))
            objects = [o for o in objects if o.obj_type in top_types]

        # top_n_objects: keep only the N tallest objects
        top_n_objects = params.get('top_n_objects')
        if top_n_objects:
            top_n_objects = int(top_n_objects)
            objects = sorted(objects, key=lambda o: o.height_max, reverse=True)[:top_n_objects]

        # min_height_m: keep only objects taller than Y metres
        min_height_m = params.get('min_height_m')
        if min_height_m:
            min_height_m = float(min_height_m)
            objects = [o for o in objects if o.height_max >= min_height_m]

        all_objects.extend(objects)
        all_stats = result.get('stats')
        all_labels = labels
        all_transform = data['transform']
        all_shape = data['shape']
        all_mask = data['mask']
        if result.get('evaluation'):
            all_evaluation = result['evaluation']

    # Build GeoJSON response
    obj_features = []
    for obj in all_objects:
        centroid_wgs = ti.geometry_from_3035(Point(obj.centroid_e, obj.centroid_n))
        props = {
            "id": obj.obj_id,
            "type": obj.obj_type,
            "type_code": obj.type_code,
            "group_id": obj.group_id,
            "group_type": obj.group_type,
            "height_max_m": obj.height_max,
            "height_mean_m": obj.height_mean,
            "height_p90_m": obj.height_p90,
            "area_sqm": obj.area_sqm,
            "compactness": obj.compactness,
            "elongation": obj.elongation,
            "solidity": obj.solidity,
            "extent": obj.extent,
            "dsm_edge_strength": obj.dsm_edge_strength,
            "slope_mean": obj.slope_mean,
            "aspect_mean": obj.aspect_mean,
            "aspect_dominant": obj.aspect_dominant,
            "roughness": obj.roughness,
            "elevation_mean": obj.elevation_mean,
            "elevation_min": obj.elevation_min,
            "elevation_max": obj.elevation_max,
            "tri_mean": obj.tri_mean,
            "tpi_mean": obj.tpi_mean,
            "curvature_mean": obj.curvature_mean,
            "terrain_class": obj.terrain_class,
            "is_manmade": obj.is_manmade,
            "confidence": obj.confidence,
            "classifier_source": obj.classifier_source,
            "rf_type": obj.rf_type,
            "rf_confidence": obj.rf_confidence,
        }
        if include_ortho or include_copernicus:
            props["ndvi_mean"] = obj.ndvi_mean
            props["ndvi_fused"] = obj.ndvi_fused
            props["brightness_mean"] = obj.brightness_mean
            props["nir_mean"] = obj.nir_mean
        if include_temporal:
            props["height_change"] = obj.height_change
            props["dtm_change"] = obj.dtm_change
            props["temporal_stability"] = obj.temporal_stability
            props["volume_change_m3"] = obj.volume_change_m3
            props["volume_change_abs_m3"] = obj.volume_change_abs_m3
            props["dtm_change_max"] = obj.dtm_change_max
        # Texture features
        if obj.glcm_entropy > 0:
            props["glcm_entropy"] = obj.glcm_entropy
            props["glcm_homogeneity"] = obj.glcm_homogeneity
            props["texture_complexity"] = obj.texture_complexity
        # SAR features
        if obj.sar_vv > 0:
            props["sar_vv"] = obj.sar_vv
            props["sar_vh"] = obj.sar_vh
        # Phenology features
        if obj.harm_amplitude > 0:
            props["harm_amplitude"] = obj.harm_amplitude
            props["harm_phase"] = obj.harm_phase
            props["phenology_class"] = obj.phenology_class
        obj_features.append({
            "type": "Feature",
            "properties": props,
            "geometry": mapping(centroid_wgs),
        })

    resp = {
        "type": "FeatureCollection",
        "features": obj_features,
        "stats": all_stats,
        "meta": {
            "classifier": "watershed_v1",
            "pipeline": "Sobel→Felzenszwalb→RAG→classify→group",
            "dataset": dataset,
            "min_object_size": min_object_size,
            "felz_scale": felz_scale,
            "rag_threshold": rag_threshold,
            "include_ortho": include_ortho,
            "include_temporal": include_temporal,
            "include_copernicus": include_copernicus,
            "include_cadastre": include_cadastre,
            "include_hansen": include_hansen,
            "include_infra": include_infra,
            "mark_uncertain": mark_uncertain,
            "processing_time_s": round(time.time() - t0, 2),
            **_rf_model_meta(),
        },
    }
    if all_evaluation:
        resp["cadastre_evaluation"] = all_evaluation
    if hansen_evaluation:
        resp["hansen_evaluation"] = hansen_evaluation
    if include_copernicus and copernicus_data is None and _is_processor_running():
        resp.setdefault('warnings', []).append(
            'Copernicus data not in cache — served without Sentinel-2/SAR (processor running)')

    return resp


# === SECTION: /api/v1/segment/overlay + raster rendering ===

# Cache for last segmentation result so legend filter re-renders are instant
_seg_cache = {"labels": None, "objects": None, "mask": None,
              "transform": None, "shape": None, "ndsm": None, "key": None}

# File-backed seg cache so all gunicorn workers share segmentation results
_SEG_CACHE_DIR = Path('/tmp/seg_cache')
_SEG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _seg_cache_path(cache_key: str) -> Path:
    """Deterministic filename for a seg cache key."""
    h = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    return _SEG_CACHE_DIR / f"seg_{h}.pkl"

def _seg_cache_save(cache_key: str, labels, objects, mask, transform, shape_hw, ndsm):
    """Persist seg cache to disk for cross-worker reuse.

    Snapshots all data into a single dict up-front (in the calling thread),
    then pickles + writes to disk in a background thread so the API response
    is not blocked.
    """
    # Capture everything NOW — numpy arrays are refcounted, not copied
    payload = {
        'key': cache_key,
        'labels': labels,
        'objects': objects,
        'mask': mask,
        'transform': tuple(transform)[:6] if hasattr(transform, '__iter__') else transform,
        'shape': shape_hw,
        'ndsm': ndsm,
        'ts': time.time(),
    }

    def _do_save():
        try:
            p = _seg_cache_path(cache_key)
            tmp = p.with_suffix('.tmp')
            with open(tmp, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.rename(p)
            log.info("seg_cache: saved to disk (%s, %.1f MB)", p.name, p.stat().st_size / 1e6)
            # Evict old entries (keep at most 3)
            entries = sorted(_SEG_CACHE_DIR.glob('seg_*.pkl'), key=lambda f: f.stat().st_mtime)
            for old in entries[:-3]:
                old.unlink(missing_ok=True)
        except Exception as e:
            log.warning("seg_cache: disk save failed: %s", e)

    threading.Thread(target=_do_save, daemon=True).start()


def _seg_cache_load(cache_key: str):
    """Try to load seg cache from disk. Returns dict or None."""
    try:
        p = _seg_cache_path(cache_key)
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > 3600:
            p.unlink(missing_ok=True)
            return None
        with open(p, 'rb') as f:
            data = pickle.load(f)
        if data.get('key') != cache_key:
            return None
        # Restore transform as rasterio Affine
        from rasterio.transform import Affine
        t = data['transform']
        if isinstance(t, (list, tuple)):
            data['transform'] = Affine(*t[:6])
        log.info("seg_cache: loaded from disk (%s)", p.name)
        return data
    except Exception as e:
        log.warning("seg_cache: disk load failed: %s", e)
        return None


def _seg_cache_scan(cache_key_substring: str):
    """Scan all disk cache files for one whose key contains the substring.

    Used by GeoPackage/MBTiles which match on a partial key (bounds+dataset).
    Returns dict with labels/objects/mask/ndsm or None.
    """
    for p in sorted(_SEG_CACHE_DIR.glob('seg_*.pkl'),
                    key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            if time.time() - p.stat().st_mtime > 3600:
                continue
            with open(p, 'rb') as f:
                data = pickle.load(f)
            if data.get('key') and cache_key_substring in data['key']:
                from rasterio.transform import Affine
                t = data['transform']
                if isinstance(t, (list, tuple)):
                    data['transform'] = Affine(*t[:6])
                log.info("seg_cache: scan hit (%s)", p.name)
                return data
        except Exception:
            pass
    return None

# Cache for raster data (DTM/DSM/nDSM/ortho) keyed by (bounds, dataset)
# so overlay endpoints don't re-fetch from remote after segment has loaded them
_raster_cache = {"key": None, "data": None, "ortho": None, "ortho_key": None}

# Segment type → RGBA colour (matches frontend TYPE_SHAPES)
SEGMENT_COLORS = {
    "tree":         (0, 100, 0, 180),
    "shrub":        (34, 139, 34, 180),
    "grass":        (124, 252, 0, 150),
    "hedge":        (46, 139, 87, 170),
    "water":        (30, 144, 255, 180),
    "roof":         (220, 20, 60, 200),
    "greenhouse":   (255, 105, 180, 180),
    "solar_panel":  (65, 105, 225, 200),
    "fence":        (160, 82, 45, 170),
    "wall":         (139, 69, 19, 170),
    "mast":         (64, 64, 64, 200),
    "wind_turbine": (21, 101, 192, 200),
    "substation":   (255, 111, 0, 200),
    "road":         (128, 128, 128, 160),
    "path":         (169, 169, 169, 150),
    "parking":      (105, 105, 105, 160),
    "bridge":       (112, 128, 144, 170),
    "crop":         (218, 165, 32, 160),
    "orchard":      (107, 142, 35, 170),
    "vineyard":     (147, 112, 219, 170),
    "garden":       (60, 179, 113, 160),
    "bare_soil":    (210, 180, 140, 140),
    "rock":         (139, 134, 130, 160),
    "excavation":   (139, 0, 0, 200),
    "fill":         (255, 140, 0, 200),
    "tree_loss":    (255, 0, 255, 200),
    "construction": (255, 69, 0, 200),
    "unclassified": (190, 190, 190, 120),
}


def _viridis_rgb(t):
    """Return (R,G,B) for t in [0,1] on the viridis scale."""
    VIRIDIS = [
        (68,1,84),(72,35,116),(64,67,135),(52,94,141),(41,120,142),
        (32,144,140),(34,167,132),(68,190,112),(121,209,81),(189,222,38),(253,231,37)
    ]
    t = max(0.0, min(1.0, t))
    idx = t * (len(VIRIDIS) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(VIRIDIS) - 1)
    f = idx - lo
    return tuple(int(VIRIDIS[lo][c] + f * (VIRIDIS[hi][c] - VIRIDIS[lo][c])) for c in range(3))


def _diverging_rgb(t):
    """Blue-white-red diverging scale, t in [-1,1] mapped to [0,1]."""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        # Blue to white
        f = t * 2
        return (int(33 + f * 222), int(102 + f * 153), int(172 + f * 83))
    else:
        # White to red
        f = (t - 0.5) * 2
        return (int(255 - f * 37), int(255 - f * 192), int(255 - f * 192))


def _segment_rgba(labels, objects, mask, type_filter=None, color_mode='type', ndsm=None, type_overrides=None, height_filter=None):
    """Render segmentation labels as RGBA image.

    color_mode: 'type' = categorical colors, 'height' = viridis by height
    type_overrides: optional dict {obj_id: type_name} to override object types
                    (used when rendering a share's stored result over cached labels)
    """
    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    obj_map = {o.obj_id: o for o in objects}

    def _effective_type(obj_id, obj):
        if type_overrides and obj_id in type_overrides:
            return type_overrides[obj_id]
        return obj.obj_type

    if color_mode == 'height' and ndsm is not None:
        # Per-pixel viridis coloring from actual nDSM values
        included = np.zeros((h, w), dtype=bool)
        for obj_id, obj in obj_map.items():
            etype = _effective_type(obj_id, obj)
            if type_filter and etype not in type_filter:
                continue
            if height_filter and not height_filter(obj.height_max):
                continue
            included |= (labels == obj_id)
        # Build viridis LUT (256 entries)
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            lut[i] = _viridis_rgb(i / 255.0)
        idx = np.clip((np.clip(np.sqrt(np.clip(ndsm, 0, 45) / 45.0), 0, 1) * 255).astype(np.uint8), 0, 255)
        for c in range(3):
            rgba[:, :, c] = lut[idx, c]
        rgba[:, :, 3] = np.where(included & mask, np.where(ndsm > 0.3, 180, 60).astype(np.uint8), 0)
    else:
        for obj_id, obj in obj_map.items():
            etype = _effective_type(obj_id, obj)
            if type_filter and etype not in type_filter:
                continue
            if height_filter and not height_filter(obj.height_max):
                continue
            seg_mask = labels == obj_id
            color = SEGMENT_COLORS.get(etype, (128, 128, 128, 120))
            for c in range(4):
                rgba[:, :, c][seg_mask] = color[c]

    # Transparent where no data
    rgba[:, :, 3][~mask] = 0
    return rgba


def _render_seg_overlay(labels, objects, mask, transform, shape_hw, type_filter=None, color_mode='type', ndsm=None, type_overrides=None):
    """Render segmentation as RGBA, reproject to WGS84, return overlay response."""
    from rasterio.warp import calculate_default_transform, reproject as rp, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds

    rgba_3035 = _segment_rgba(labels, objects, mask, type_filter, color_mode=color_mode, ndsm=ndsm, type_overrides=type_overrides)

    src_crs = CRS.from_epsg(3035)
    dst_crs = CRS.from_epsg(4326)
    h, w = shape_hw
    dst_tf, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, w, h, *array_bounds(h, w, transform),
    )
    rgba_wgs = np.zeros((4, dst_h, dst_w), dtype=np.uint8)
    for band in range(4):
        rp(
            source=rgba_3035[:, :, band],
            destination=rgba_wgs[band],
            src_transform=transform,
            src_crs=src_crs,
            dst_transform=dst_tf,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    rgba_out = np.transpose(rgba_wgs, (1, 2, 0))  # (H,W,4)
    bounds = array_bounds(dst_h, dst_w, dst_tf)
    bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])  # south,west,north,east
    return _send_rgba_overlay(rgba_out, bounds_wgs)


def _load_share_type_overrides(share_id: str) -> dict | None:
    """Load a share's features and return {obj_id: type_name} mapping.

    This lets the overlay endpoint render with the share's stored classification
    instead of whatever the current model or cache contains.
    """
    try:
        resolved_id, path = _resolve_share(share_id)
        if not path or not path.exists():
            log.warning("share type overrides: share %s not found", share_id)
            return None
        raw = gzip.decompress(path.read_bytes())
        share_data = json.loads(raw)
        result = share_data.get('result', {})
        features = result.get('features', [])
        if not features:
            return None
        overrides = {}
        for f in features:
            props = f.get('properties', {})
            obj_id = props.get('id')
            obj_type = props.get('type')
            if obj_id is not None and obj_type:
                overrides[int(obj_id)] = obj_type
        return overrides if overrides else None
    except Exception as e:
        log.warning("share type overrides: failed for %s: %s", share_id, e)
        return None


@app.route('/api/v1/segment/overlay', methods=['POST'])
def segment_overlay():
    """Return segment classification as a coloured PNG overlay (reprojected to WGS84).

    First call runs full segmentation and caches the result.
    Subsequent calls with only ?types= changed use the cache for instant re-renders.

    When share_id is provided, the overlay uses the share's stored type assignments
    instead of the current model's classification, so historical shares render correctly.
    """
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        include_ortho = str(params.get('include_ortho', 'true')).lower() in ('true', '1', 'yes')
        include_copernicus = str(params.get('include_copernicus', 'false')).lower() in ('true', '1', 'yes')
        include_cadastre = str(params.get('include_cadastre', 'false')).lower() in ('true', '1', 'yes')
        include_hansen = str(params.get('include_hansen', 'false')).lower() in ('true', '1', 'yes')
        include_infra = str(params.get('include_infra', 'true')).lower() in ('true', '1', 'yes')
        mark_uncertain = str(params.get('mark_uncertain', 'false')).lower() in ('true', '1', 'yes')
        type_filter_str = params.get('types', None)
        type_filter = None
        if type_filter_str:
            type_filter = set(t.strip() for t in type_filter_str.split(','))
        color_mode = params.get('color_mode', 'type')  # 'type' or 'height'
        top_n_classes = params.get('top_n_classes')
        share_id = params.get('share_id')

        # Build type overrides from share's stored result features
        type_overrides = None
        if share_id:
            type_overrides = _load_share_type_overrides(share_id)
            if type_overrides:
                log.info("segment overlay: using type overrides from share %s (%d mappings)",
                         share_id, len(type_overrides))

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(100)
        _validate_area(geom_3035)

        # Build a cache key from geometry bounds + dataset + analysis options
        cache_key = f"{geom_3035.bounds}_{dataset}_{include_ortho}_{include_copernicus}_{include_cadastre}_{include_hansen}_{include_infra}_{mark_uncertain}_temporal"

        # When share_id is provided, try to find ANY cached labels for this geometry
        # (regardless of analysis options) — we only need the pixel labels, not the types
        if type_overrides:
            bounds_prefix = str(geom_3035.bounds)
            cached = None
            # Check in-process cache first (any key with same bounds)
            if _seg_cache["labels"] is not None and _seg_cache.get("key", "").startswith(bounds_prefix):
                cached = _seg_cache
            if cached is None:
                # Try exact key on disk
                cached = _seg_cache_load(cache_key)
            if cached is None:
                # Scan disk caches for any entry with matching geometry bounds
                cached = _seg_cache_scan(bounds_prefix)
            if cached is not None:
                log.info("segment overlay: rendering with share type overrides (share=%s)", share_id)
                return _render_seg_overlay(
                    cached["labels"], cached["objects"],
                    cached["mask"], cached["transform"],
                    cached["shape"], type_filter, color_mode,
                    ndsm=cached.get("ndsm"), type_overrides=type_overrides,
                )
            # Fall through to full pipeline if no cached labels found
            log.info("segment overlay: no cached labels for share %s, running full pipeline", share_id)

        if _seg_cache["key"] == cache_key:
            # Re-render from cache — instant
            log.info("segment overlay: re-render from cache (filter=%s)", type_filter)
            if top_n_classes:
                from collections import Counter
                _tn = int(top_n_classes)
                _tc = Counter(o.obj_type for o in _seg_cache["objects"])
                _top = set(t for t, _ in _tc.most_common(_tn))
                type_filter = (type_filter & _top) if type_filter else _top
            return _render_seg_overlay(
                _seg_cache["labels"], _seg_cache["objects"],
                _seg_cache["mask"], _seg_cache["transform"],
                _seg_cache["shape"], type_filter, color_mode,
                ndsm=_seg_cache.get("ndsm"), type_overrides=type_overrides,
            )

        # Check file-backed cache (shared across gunicorn workers)
        _disk = _seg_cache_load(cache_key)
        if _disk is not None:
            log.info("segment overlay: loaded from disk cache (filter=%s)", type_filter)
            # Populate in-process cache for subsequent re-renders
            _seg_cache.update({
                "labels": _disk["labels"], "objects": _disk["objects"],
                "mask": _disk["mask"], "transform": _disk["transform"],
                "shape": _disk["shape"], "ndsm": _disk.get("ndsm"), "key": cache_key,
            })
            if top_n_classes:
                from collections import Counter
                _tn = int(top_n_classes)
                _tc = Counter(o.obj_type for o in _disk["objects"])
                _top = set(t for t, _ in _tc.most_common(_tn))
                type_filter = (type_filter & _top) if type_filter else _top
            return _render_seg_overlay(
                _disk["labels"], _disk["objects"],
                _disk["mask"], _disk["transform"],
                _disk["shape"], type_filter, color_mode,
                ndsm=_disk.get("ndsm"), type_overrides=type_overrides,
            )

        # Full segmentation pipeline
        # Always load temporal data for stability-based building detection
        dtm_dates, dsm_dates = None, None
        try:
            multi = raster_io.read_multi_date_ndsm(geom_3035)
            data = {
                'dtm': multi['dtm'], 'dsm': multi['dsm'],
                'ndsm': multi['ndsm'], 'mask': multi['mask'],
                'transform': multi['transform'], 'crs': multi['crs'],
                'shape': multi['shape'],
            }
            dtm_dates, dsm_dates = {}, {}
            for d in multi['dates_loaded']:
                try:
                    dd = raster_io.read_dtm_dsm(geom_3035, dataset=d)
                    mh = min(dd['shape'][0], data['shape'][0])
                    mw = min(dd['shape'][1], data['shape'][1])
                    dtm_dates[d] = dd['dtm'][:mh, :mw]
                    dsm_dates[d] = dd['dsm'][:mh, :mw]
                except Exception as e:
                    log.warning("overlay: date %s load failed: %s", d, e)
        except Exception as e:
            log.warning("overlay: multi-temporal failed, single date: %s", e)
            data = raster_io.read_dtm_dsm(geom_3035, dataset)

        rgb, spectral = (None, None)
        if include_ortho:
            rgb, spectral = _try_read_ortho(data)

        copernicus_data = None
        if include_copernicus:
            copernicus_data = _try_copernicus(geom, sar=True, harmonics=True, year=ti.dataset_to_year(dataset))

        building_footprints = None
        if include_cadastre:
            building_footprints = _try_cadastre(geom, data['transform'], data['shape'])

        hansen_data = None
        if include_hansen:
            hansen_data = _try_hansen(geom, data['transform'], data['shape'])

        _infra_lookup = None
        if include_infra:
            try:
                from infrastructure_lookup import InfrastructureLookup
                from pyproj import Transformer as _Tx
                _tx = _Tx.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
                b = geom.bounds
                w4, s4 = _tx.transform(b[0], b[1])
                e4, n4 = _tx.transform(b[2], b[3])
                _infra_lookup = InfrastructureLookup.for_bbox(w4, s4, e4, n4)
            except Exception:
                pass

        obs_year = ti.dataset_to_year(dataset)
        result = seg.segment_and_classify(
            data['dtm'], data['dsm'], data['mask'], data['transform'],
            dtm_dates=dtm_dates,
            dsm_dates=dsm_dates,
            spectral=spectral,
            copernicus=copernicus_data,
            building_footprints=building_footprints,
            hansen=hansen_data,
            observation_year=obs_year,
            infra_lookup=_infra_lookup,
            mark_uncertain=mark_uncertain,
        )

        objects = result['objects']
        labels = result['labels']

        # Hansen calibration
        if include_hansen and hansen_data:
            try:
                objects = hansen.calibrate_tree_loss(objects, labels, hansen_data, observation_year=obs_year)
            except Exception as e:
                log.warning("Hansen calibration failed in overlay: %s", e)

        # Store in cache for fast re-renders with different type filters
        _seg_cache.update({
            "labels": labels, "objects": objects,
            "mask": data['mask'], "transform": data['transform'],
            "shape": data['shape'], "ndsm": data['ndsm'], "key": cache_key,
        })
        log.info("segment overlay: full pipeline %.1fs, cached for re-renders", time.time() - t0)
        _seg_cache_save(cache_key, labels, objects, data['mask'],
                        data['transform'], data['shape'], data['ndsm'])

        if top_n_classes:
            from collections import Counter
            _tn = int(top_n_classes)
            _tc = Counter(o.obj_type for o in objects)
            _top = set(t for t, _ in _tc.most_common(_tn))
            type_filter = (type_filter & _top) if type_filter else _top

        return _render_seg_overlay(labels, objects, data['mask'],
                                   data['transform'], data['shape'], type_filter, color_mode,
                                   ndsm=data['ndsm'], type_overrides=type_overrides)
    except Exception as e:
        log.error("segment overlay: %s", traceback.format_exc())
        return _error(str(e))


# === SECTION: /api/v1/export/geopackage endpoint ===

@app.route('/api/v1/export/geopackage', methods=['POST'])
def export_geopackage():
    """Export selected raster layers as a GeoPackage.

    Query params:
      include_dtm=true       Include DTM/DSM/nDSM bands
      include_dsm=true       (same as above, kept for compat)
      include_segments=true   Include segment_type + segment_height raster bands
      include_segments_vector=true  Include vector polygon layer with segment outlines
      ortho_years=2024,2023   Ortho RGB(I) for listed years
      raster_layers=dtm-2024,dsm-2023,hansen,raster  RGBA overlay renders
      types=tree,road,...     Segment type filter
      height_min=X            Filter segments: height >= X metres
      height_max=X            Filter segments: height <= X metres
      height_op=gt|lt|between Explicit operator (inferred if omitted)
      color_mode=type|height  Segment raster color mode
      include_ortho=true      Legacy: same as ortho_years=default
      async=true              Return 202 with task_id; poll progress; download via GET
    """
    try:
        import fiona
        from fiona.crs import from_epsg
    except ImportError:
        pass

    try:
        features = _get_geometry()
        params = _get_params()

        run_async = str(request.args.get('async', params.get('async', 'false'))).lower() in ('true', '1', 'yes')
        task_id = request.args.get('task_id', params.get('task_id', ''))

        if run_async:
            if not task_id:
                task_id = str(uuid.uuid4())
            _progress_start(task_id)
            thread = threading.Thread(
                target=_gpkg_worker,
                args=(task_id, features, params),
                daemon=True,
            )
            thread.start()
            return jsonify({"task_id": task_id, "status": "running"}), 202

        # Synchronous path
        tmp_path, table_count, elapsed = _gpkg_core(features, params)
        if table_count == 0:
            return _error('No layers selected')
        log.info("GeoPackage export: %d tables, %.1fs", table_count, elapsed)
        return send_file(
            tmp_path, mimetype='application/geopackage+sqlite3',
            as_attachment=True, download_name='landscape_export.gpkg',
        )
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error("geopackage export: %s", traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


def _gpkg_worker(task_id: str, features: list, params: dict):
    """Background worker for async GeoPackage export."""
    try:
        _progress_set(task_id, 'gpkg_export', 'Building GeoPackage…')
        tmp_path, table_count, elapsed = _gpkg_core(features, params, task_id=task_id)
        if table_count == 0:
            _progress_error(task_id, 'No layers selected')
            return
        # Move result to well-known location
        dest = _RESULTS_DIR / f"{task_id}.gpkg"
        import shutil
        shutil.move(tmp_path, str(dest))
        _progress_done(task_id)
        log.info("Async GPKG task %s completed: %d tables, %.1fs", task_id, table_count, elapsed)
    except Exception as e:
        log.error("Async GPKG task %s failed: %s", task_id, traceback.format_exc())
        _progress_error(task_id, str(e))


@app.route('/api/v1/export/geopackage/download/<task_id>', methods=['GET'])
def download_gpkg(task_id):
    """Download a completed async GeoPackage export."""
    if not task_id or not re.match(r'^[a-f0-9\-]+$', task_id):
        return _error('Invalid task_id', 400)
    dest = _RESULTS_DIR / f"{task_id}.gpkg"
    if not dest.exists():
        return _error('GeoPackage not found or not ready yet', 404)
    try:
        resp = send_file(
            str(dest), mimetype='application/geopackage+sqlite3',
            as_attachment=True, download_name='landscape_export.gpkg',
        )
        # Clean up after download
        @resp.call_on_close
        def _cleanup():
            try:
                dest.unlink(missing_ok=True)
                (_PROGRESS_DIR / f"{task_id}.json").unlink(missing_ok=True)
            except Exception:
                pass
        return resp
    except Exception as e:
        log.error("GPKG download failed: %s", e)
        return _error(f"Download error: {e}", 500)


# === SECTION: /api/v1/export/kml endpoint ===

@app.route('/api/v1/export/kml', methods=['POST'])
def export_kml():
    """Export segment features as KML with type/height_class folders.

    Query params:
      types=tree,grass,...    Filter object types (default: all)
      group_by=type|height_class  Folder grouping (default: type)
      height_min=X            Filter: height >= X metres
      height_max=X            Filter: height <= X metres
      height_op=gt|lt|between Explicit operator (inferred from min/max if omitted)
    """
    try:
        features = _get_geometry()
        params = _get_params()
        result_json = params.get('result_json', '')

        type_filter_str = params.get('types', None)
        type_filter = set(t.strip() for t in type_filter_str.split(',')) if type_filter_str else None
        group_by = params.get('group_by', 'type')
        seg_geom = params.get('segment_geometry', 'point').lower().strip()

        # Get features from posted result or from cache
        if result_json:
            try:
                result_data = json.loads(result_json)
                obj_features = result_data.get('features', [])
            except Exception:
                obj_features = []
        else:
            obj_features = []

        if not obj_features:
            return _error('No features to export. Include result_json in the request body.')

        # Filter by type
        if type_filter:
            obj_features = [f for f in obj_features
                           if f.get('properties', {}).get('type') in type_filter]

        # Filter by height
        obj_features = _apply_height_filter_features(obj_features, params)

        # Polygon vectorisation if requested
        kml_features = obj_features
        if seg_geom == 'polygon' and features:
            try:
                geom_wgs = features[0]['geometry']
                dataset = params.get('dataset', ti.DEFAULT_DATASET)
                poly_features = _vectorise_segments_to_geojson(
                    geom_wgs, dataset, type_filter=type_filter,
                    height_params=params)
                if poly_features:
                    kml_features = poly_features
                else:
                    log.warning('kml export polygon: vectorisation returned 0 features, using points')
            except Exception as e:
                log.warning('kml export polygon fallback to points: %s', e)

        # Build KML
        style_mode = params.get('segment_geometry_style', 'type').lower().strip()
        kml = _build_kml(kml_features, group_by, style_mode=style_mode)

        tmp = tempfile.NamedTemporaryFile(suffix='.kml', delete=False, mode='w', encoding='utf-8')
        tmp.write(kml)
        tmp.close()

        return send_file(
            tmp.name, mimetype='application/vnd.google-earth.kml+xml',
            as_attachment=True, download_name='landscape_export.kml',
        )
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error("kml export: %s", traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


def _vectorise_segments_to_geojson(geom_wgs, dataset: str, type_filter=None,
                                    height_params: dict = None,
                                    style_mode: str = 'type') -> list:
    """Vectorise cached segment labels into GeoJSON polygon features (WGS84).

    Returns a list of GeoJSON Feature dicts with Polygon geometry and rich
    properties.  Falls back to empty list if segment cache is unavailable.
    """
    from shapely.ops import transform as shapely_transform
    import pyproj

    geom_3035 = ti.geometry_to_3035(geom_wgs)
    bounds_prefix = f"{geom_3035.bounds}_{dataset}"

    # Try to load from seg cache (in-memory → disk scan)
    cached = None
    if (_seg_cache.get('labels') is not None and
            _seg_cache.get('key') and bounds_prefix in _seg_cache['key']):
        cached = _seg_cache
    else:
        cached = _seg_cache_load(bounds_prefix) or _seg_cache_scan(bounds_prefix)
    if cached is None or cached.get('labels') is None:
        return []

    v_labels = cached['labels']
    v_objects = cached['objects']
    v_mask = cached['mask']
    v_tf = cached['transform']
    from rasterio.transform import Affine
    if isinstance(v_tf, (list, tuple)):
        v_tf = Affine(*v_tf[:6])

    obj_map = {o.obj_id: o for o in v_objects}
    # Apply type filter
    if type_filter:
        v_filtered = [o for o in v_objects if o.obj_type in type_filter]
    else:
        v_filtered = list(v_objects)
    # Apply height filter
    if height_params:
        v_filtered = _apply_height_filter_objects(v_filtered, height_params)
    filtered_ids = {o.obj_id for o in v_filtered}
    if not filtered_ids:
        return []

    from rasterio.features import shapes as rasterize_shapes
    label_int = v_labels.astype(np.int32)
    seg_mask = v_mask & np.isin(label_int, list(filtered_ids))

    # CRS transform: EPSG:3035 → EPSG:4326
    proj = pyproj.Transformer.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)

    out_features = []
    for geom_dict, val in rasterize_shapes(
        label_int, mask=seg_mask, transform=v_tf, connectivity=4,
    ):
        oid = int(val)
        obj = obj_map.get(oid)
        if obj is None:
            continue
        # Reproject polygon coords 3035→4326
        from shapely.geometry import shape, mapping
        poly_3035 = shape(geom_dict)
        poly_wgs = shapely_transform(proj.transform, poly_3035)

        tc = SEGMENT_COLORS.get(obj.obj_type, (128, 128, 128, 120))
        hex_type = '#{:02X}{:02X}{:02X}'.format(tc[0], tc[1], tc[2])
        hv = _viridis_rgb(min(1.0, (max(0, obj.height_max) / 45.0) ** 0.5))
        hex_height = '#{:02X}{:02X}{:02X}'.format(*hv)

        out_features.append({
            'type': 'Feature',
            'geometry': mapping(poly_wgs),
            'properties': {
                'id': oid,
                'type': obj.obj_type,
                'group_type': obj.group_type or '',
                'height_class': _height_class(obj.height_max),
                'height_max_m': round(obj.height_max, 2),
                'height_mean_m': round(obj.height_mean, 2),
                'area_sqm': round(obj.area_sqm, 1),
                'confidence': round(obj.confidence, 2),
                'is_manmade': int(obj.is_manmade) if obj.is_manmade else 0,
                'ndvi_mean': round(obj.ndvi_mean, 3) if obj.ndvi_mean else 0.0,
                'color': hex_type,
                'color_height': hex_height,
            },
        })
    log.info('vectorise_to_geojson: %d polygon features', len(out_features))
    return out_features


def _build_kml(features, group_by='type', style_mode='type'):
    """Build a KML string from GeoJSON features (points or polygons), organised into folders.

    style_mode: 'type' — colour by object type (default)
                'height' — colour by viridis height ramp
    """
    import xml.etree.ElementTree as ET

    # Derive KML colours (AABBGGRR) from the canonical SEGMENT_COLORS
    TYPE_KML_COLORS = {
        t: 'ff{:02x}{:02x}{:02x}'.format(rgba[2], rgba[1], rgba[0])
        for t, rgba in SEGMENT_COLORS.items()
    }

    # Group features
    groups = {}
    for f in features:
        p = f.get('properties', {})
        h_max = p.get('height_max_m') or p.get('height_after_m') or 0
        hc = _height_class(h_max)
        key = p.get('type', 'unknown') if group_by == 'type' else hc
        groups.setdefault(key, []).append((f, p, hc))

    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<kml xmlns="http://www.opengis.net/kml/2.2">',
             '<Document>',
             '<name>Landscape Analysis Export</name>']

    # Styles — include IconStyle (points), PolyStyle (fill), LineStyle (outline)
    for tname, color in TYPE_KML_COLORS.items():
        # Semi-transparent fill: replace 'ff' alpha with '80' (~50% opacity)
        fill_color = '80' + color[2:]
        lines.append(
            f'<Style id="style-{tname}">'
            f'<IconStyle><color>{color}</color><scale>0.8</scale>'
            '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>'
            '</IconStyle>'
            f'<LineStyle><color>{color}</color><width>1.5</width></LineStyle>'
            f'<PolyStyle><color>{fill_color}</color></PolyStyle>'
            '</Style>'
        )

    # Default style
    lines.append(
        '<Style id="style-default">'
        '<IconStyle><color>ff888888</color><scale>0.6</scale>'
        '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>'
        '</IconStyle>'
        '<LineStyle><color>ff888888</color><width>1</width></LineStyle>'
        '<PolyStyle><color>80888888</color></PolyStyle>'
        '</Style>'
    )

    # Height-based viridis styles (10 buckets: 0-5, 5-10, ..., 40-45, 45+)
    HEIGHT_BUCKETS = 10
    for i in range(HEIGHT_BUCKETS):
        t_val = (i + 0.5) / HEIGHT_BUCKETS  # midpoint fraction 0..1
        rgb = _viridis_rgb(t_val)
        color = 'ff{:02x}{:02x}{:02x}'.format(rgb[2], rgb[1], rgb[0])  # KML AABBGGRR
        fill_color = '80' + color[2:]
        lines.append(
            f'<Style id="style-h{i}">'
            f'<IconStyle><color>{color}</color><scale>0.8</scale>'
            '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>'
            '</IconStyle>'
            f'<LineStyle><color>{color}</color><width>1.5</width></LineStyle>'
            f'<PolyStyle><color>{fill_color}</color></PolyStyle>'
            '</Style>'
        )

    def _coords_to_kml_ring(ring_coords):
        """Convert a GeoJSON ring [[lon,lat], ...] to KML coordinate string."""
        parts = []
        for c in ring_coords:
            lon, lat = c[0], c[1]
            alt = c[2] if len(c) > 2 else 0
            parts.append(f'{lon},{lat},{alt}')
        return ' '.join(parts)

    for group_name in sorted(groups.keys()):
        items = groups[group_name]
        lines.append(f'<Folder><name>{_xml_escape(group_name)} ({len(items)})</name>')
        for feat, props, hc in items:
            geom = feat.get('geometry', {})
            geom_type = geom.get('type', '')
            coords = geom.get('coordinates', [])
            if not coords:
                continue
            tname = props.get('type', 'unknown')
            if style_mode == 'height':
                h_bucket = min(int((h_max / 45.0) ** 0.5 * HEIGHT_BUCKETS),
                               HEIGHT_BUCKETS - 1)
                if h_bucket < 0: h_bucket = 0
                style_id = f'style-h{h_bucket}'
            else:
                style_id = f'style-{tname}' if tname in TYPE_KML_COLORS else 'style-default'
            h_max = props.get('height_max_m', 0) or 0
            area = props.get('area_sqm', 0) or 0
            rgba = SEGMENT_COLORS.get(tname, (128, 128, 128, 120))
            hex_color = '#{:02X}{:02X}{:02X}'.format(rgba[0], rgba[1], rgba[2])
            desc_parts = [f'Type: {tname}', f'Color: {hex_color}',
                         f'Height class: {hc}',
                         f'Height max: {h_max:.1f}m', f'Area: {area:.0f} m\u00b2']
            if props.get('confidence'):
                desc_parts.append(f'Confidence: {props["confidence"]:.0%}')
            if props.get('group_type'):
                desc_parts.append(f'Group: {props["group_type"]}')
            desc = '<br/>'.join(desc_parts)
            name_str = f'{tname} ({h_max:.1f}m, {area:.0f}m\u00b2)'

            # Build geometry KML
            if geom_type == 'Polygon' and coords:
                rings_kml = ''
                for i, ring in enumerate(coords):
                    tag = 'outerBoundaryIs' if i == 0 else 'innerBoundaryIs'
                    rings_kml += (f'<{tag}><LinearRing>'
                                  f'<coordinates>{_coords_to_kml_ring(ring)}</coordinates>'
                                  f'</LinearRing></{tag}>')
                geom_kml = f'<Polygon><tessellate>1</tessellate>{rings_kml}</Polygon>'
            elif geom_type == 'MultiPolygon' and coords:
                polys_kml = ''
                for poly_coords in coords:
                    rings_kml = ''
                    for i, ring in enumerate(poly_coords):
                        tag = 'outerBoundaryIs' if i == 0 else 'innerBoundaryIs'
                        rings_kml += (f'<{tag}><LinearRing>'
                                      f'<coordinates>{_coords_to_kml_ring(ring)}</coordinates>'
                                      f'</LinearRing></{tag}>')
                    polys_kml += f'<Polygon><tessellate>1</tessellate>{rings_kml}</Polygon>'
                geom_kml = f'<MultiGeometry>{polys_kml}</MultiGeometry>'
            elif geom_type == 'Point' and len(coords) >= 2:
                lon, lat = coords[0], coords[1]
                alt = coords[2] if len(coords) > 2 else 0
                geom_kml = f'<Point><coordinates>{lon},{lat},{alt}</coordinates></Point>'
            else:
                continue

            lines.append(f'<Placemark><name>{_xml_escape(name_str)}</name>'
                         f'<description><![CDATA[{desc}]]></description>'
                         f'<styleUrl>#{style_id}</styleUrl>'
                         f'<ExtendedData>'
                         f'<Data name="color"><value>{hex_color}</value></Data>'
                         f'<Data name="type"><value>{_xml_escape(tname)}</value></Data>'
                         f'</ExtendedData>'
                         f'{geom_kml}'
                         '</Placemark>')
        lines.append('</Folder>')

    lines.append('</Document></kml>')
    return '\n'.join(lines)


def _xml_escape(s):
    """Escape XML special characters."""
    return (str(s).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;'))


def _fix_gpkg_raster_crs(gpkg_path: str):
    """Ensure GPKG raster layers are properly registered for QGIS.

    Layers written by rasterio fall into two categories:
    - float32 (DTM/DSM/etc): already '2d-gridded-coverage' with proper ancillary
    - uint8 (Ortho/CIR/segment_type/WorldCover/Hansen): registered as 'tiles'

    The 'tiles' data_type is the correct GPKG standard for PNG/JPEG tile pyramids.
    GDAL and QGIS both read CRS from 'tiles' layers via gpkg_tile_matrix_set.srs_id.
    Do NOT convert 'tiles' to '2d-gridded-coverage' — that extension is for
    single-band gridded elevation data (TIFF tiles) and breaks multi-band
    JPEG/PNG layers (Ortho, CIR) in QGIS, causing 'h_band null' errors and
    the raster to appear at 0,0 without CRS.

    This function:
    1. Ensures the gpkg_spatial_ref_sys entry is complete (WKT1 + WKT2)
    2. Repairs uint8 layers that were wrongly converted to '2d-gridded-coverage'
       by an older version of this code (reverts them to 'tiles' and cleans up
       bogus extension/ancillary entries)
    """
    import sqlite3 as _sq
    if not os.path.exists(gpkg_path) or os.path.getsize(gpkg_path) == 0:
        return
    conn = _sq.connect(gpkg_path)
    try:
        # --- 1. Ensure EPSG:3035 has both WKT1 and WKT2 definitions ---
        row = conn.execute(
            "SELECT definition, definition_12_063 FROM gpkg_spatial_ref_sys "
            "WHERE srs_id = 3035"
        ).fetchone()
        if row:
            wkt1, wkt2 = row
            needs_update = False
            if not wkt1 or wkt1 == 'undefined':
                from pyproj import CRS
                wkt1 = CRS.from_epsg(3035).to_wkt('WKT1_GDAL')
                needs_update = True
            if not wkt2 or wkt2 == 'undefined':
                from pyproj import CRS
                wkt2 = CRS.from_epsg(3035).to_wkt('WKT2_2019')
                needs_update = True
            if needs_update:
                conn.execute(
                    "UPDATE gpkg_spatial_ref_sys "
                    "SET definition = ?, definition_12_063 = ? "
                    "WHERE srs_id = 3035",
                    (wkt1, wkt2)
                )
                log.info("Updated EPSG:3035 WKT definitions in %s", gpkg_path)

        # --- 2. Repair uint8 layers wrongly registered as 2d-gridded-coverage ---
        # Legitimate float32 gridded layers have column_name='tile_data' in
        # gpkg_extensions.  Bogus uint8 layers (from old code) have
        # column_name IS NULL.  Revert those to 'tiles' data_type and clean
        # up the spurious extension + ancillary rows.
        try:
            bogus = conn.execute(
                "SELECT table_name FROM gpkg_extensions "
                "WHERE extension_name = 'gpkg_2d_gridded_coverage' "
                "AND column_name IS NULL "
                "AND table_name NOT IN "
                "  ('gpkg_2d_gridded_coverage_ancillary', "
                "   'gpkg_2d_gridded_tile_ancillary')"
            ).fetchall()
            if bogus:
                bogus_names = [r[0] for r in bogus]
                log.info("Repairing %d uint8 layers wrongly registered as "
                         "2d-gridded-coverage: %s", len(bogus_names), bogus_names)
                for tname in bogus_names:
                    conn.execute(
                        "UPDATE gpkg_contents SET data_type = 'tiles' "
                        "WHERE table_name = ? AND data_type = '2d-gridded-coverage'",
                        (tname,)
                    )
                    conn.execute(
                        "DELETE FROM gpkg_extensions "
                        "WHERE table_name = ? "
                        "AND extension_name = 'gpkg_2d_gridded_coverage' "
                        "AND column_name IS NULL",
                        (tname,)
                    )
                    try:
                        conn.execute(
                            "DELETE FROM gpkg_2d_gridded_coverage_ancillary "
                            "WHERE tile_matrix_set_name = ?",
                            (tname,)
                        )
                    except Exception:
                        pass
                    try:
                        conn.execute(
                            "DELETE FROM gpkg_2d_gridded_tile_ancillary "
                            "WHERE tpudt_name = ?",
                            (tname,)
                        )
                    except Exception:
                        pass
        except Exception as e:
            log.debug("GPKG repair check skipped (no extensions table?): %s", e)

        conn.commit()
    except Exception as e:
        log.warning("GPKG CRS fix failed for %s: %s", gpkg_path, e)
    finally:
        conn.close()


def _gpkg_core(features: list, params: dict, task_id: str = '') -> tuple:
    """Core GeoPackage building logic. Returns (tmp_path, table_count, elapsed_s)."""
    t0 = time.time()
    dataset = params.get('dataset', ti.DEFAULT_DATASET)

    # --- Parse layer selection params ---
    _bool = lambda k, d='false': str(params.get(k, d)).lower() in ('true', '1', 'yes')
    include_dtm = _bool('include_dtm', 'true') or _bool('include_dsm', 'true')
    include_segments = _bool('include_segments', 'false')
    include_segments_vector = _bool('include_segments_vector', 'false')
    include_ortho_legacy = _bool('include_ortho', 'false')

    ortho_years_str = params.get('ortho_years', '')
    ortho_years = [int(y.strip()) for y in ortho_years_str.split(',') if y.strip().isdigit()] if ortho_years_str else []
    if include_ortho_legacy and not ortho_years:
        ortho_years = [ti.dataset_to_year(dataset)]

    raster_layers_str = params.get('raster_layers', '')
    raster_layers = [x.strip() for x in raster_layers_str.split(',') if x.strip()] if raster_layers_str else []

    type_filter_str = params.get('types', None)
    type_filter = set(t.strip() for t in type_filter_str.split(',')) if type_filter_str else None
    color_mode = params.get('color_mode', 'type')

    feat = features[0]
    geom = feat['geometry']
    geom_3035 = ti.geometry_to_3035(geom)
    if geom.geom_type == 'Point':
        geom_3035 = geom_3035.buffer(100)
    _validate_area(geom_3035)

    _progress_set(task_id, 'gpkg_raster', 'Reading DTM/DSM…')
    data = raster_io.read_dtm_dsm(geom_3035, dataset)
    dtm = data['dtm']
    dsm = data['dsm']
    ndsm = data['ndsm']
    tf = data['transform']
    h, w = data['shape']
    mask = data['mask']

    tmp = tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False)
    tmp_path = tmp.name
    tmp.close()
    os.unlink(tmp_path)  # GPKG driver needs a fresh path

    table_count = 0

    def _write_gpkg_table(name, arrays_list, dtype='float32', descriptions=None):
        """Write one raster table to the GPKG. First call creates, rest append."""
        nonlocal table_count
        n = len(arrays_list)
        opts = dict(
            driver='GPKG', width=w, height=h, count=n,
            dtype=dtype, crs='EPSG:3035', transform=tf,
            RASTER_TABLE=name, RASTER_IDENTIFIER=name,
        )
        if dtype == 'float32':
            opts['nodata'] = float('nan')
        if table_count > 0:
            opts['APPEND_SUBDATASET'] = 'YES'
        with rasterio.open(tmp_path, 'w', **opts) as dst:
            for i, arr in enumerate(arrays_list, 1):
                out = arr[:h, :w] if arr.shape[0] >= h and arr.shape[1] >= w else arr
                dst.write(out, i)
                if descriptions and i <= len(descriptions):
                    dst.set_band_description(i, descriptions[i - 1])
        table_count += 1
        log.info("GPKG table %s: %d bands (%s)", name, n, dtype)

    # --- Core DTM/DSM/nDSM (1 band each = cleaner in QGIS) ---
    if include_dtm:
        _progress_set(task_id, 'gpkg_dtm', 'Writing DTM/DSM/nDSM…')
        _write_gpkg_table('DTM', [dtm.astype(np.float32)])
        _write_gpkg_table('DSM', [dsm.astype(np.float32)])
        _write_gpkg_table('nDSM', [ndsm.astype(np.float32)])

    # --- Orthophoto for each requested year (RGBA uint8) ---
    rgb = None
    for year in ortho_years:
        try:
            import ortho_io
            _progress_set(task_id, 'gpkg_ortho', f'Writing ortho {year}…')
            rgb_arr, nir = ortho_io.read_ortho_for_als(data, year=year)
            if rgb_arr is not None:
                bands_u8 = [rgb_arr[0], rgb_arr[1], rgb_arr[2]]
                descs = ['Red', 'Green', 'Blue']
                if nir is not None:
                    bands_u8.append(nir)
                    descs.append('NIR')
                _write_gpkg_table(f'Ortho_{year}', bands_u8,
                                  dtype='uint8', descriptions=descs)
                if rgb is None:
                    rgb = rgb_arr
        except Exception as e:
            log.warning("Ortho %d for gpkg failed: %s", year, e)

    # --- Segment type/height rasters ---
    if include_segments:
        try:
            _progress_set(task_id, 'gpkg_segments', 'Building segment rasters…')
            labels = None
            objects = None
            cache_key_check = f"{geom_3035.bounds}_{dataset}"
            if _seg_cache.get("labels") is not None and _seg_cache["key"] and cache_key_check in _seg_cache["key"]:
                log.info("GeoPackage: using cached segmentation (in-process)")
                labels = _seg_cache["labels"]
                objects = _seg_cache["objects"]
            else:
                # Try file-backed cache (cross-worker)
                _dc = _seg_cache_scan(cache_key_check)
                if _dc is not None:
                    log.info("GeoPackage: using cached segmentation (disk)")
                    labels = _dc['labels']
                    objects = _dc['objects']
            if labels is None:
                spectral = None
                if rgb is not None:
                    import ortho_io
                    _, nir_arr = ortho_io.read_ortho_for_als(data)
                    spectral = ortho_io.compute_spectral_indices(rgb, nir=nir_arr)
                    spectral["red"] = rgb[0].astype(np.float32)
                    spectral["green"] = rgb[1].astype(np.float32)
                    spectral["blue"] = rgb[2].astype(np.float32)
                    if nir_arr is not None:
                        spectral["nir"] = nir_arr.astype(np.float32)
                _il = None
                _incl_infra = str(params.get('include_infra', 'true')).lower() in ('true', '1', 'yes')
                if _incl_infra:
                    try:
                        from infrastructure_lookup import InfrastructureLookup
                        from pyproj import Transformer as _Tx2
                        _tx2 = _Tx2.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
                        _b = geom_3035.bounds
                        _w4, _s4 = _tx2.transform(_b[0], _b[1])
                        _e4, _n4 = _tx2.transform(_b[2], _b[3])
                        _il = InfrastructureLookup.for_bbox(_w4, _s4, _e4, _n4)
                    except Exception:
                        pass
                _mark_uncertain = str(params.get('mark_uncertain', 'false')).lower() in ('true', '1', 'yes')
                result = seg.segment_and_classify(
                    dtm, dsm, mask, tf, spectral=spectral,
                    observation_year=ti.dataset_to_year(dataset),
                    infra_lookup=_il,
                    mark_uncertain=_mark_uncertain,
                )
                objects = result['objects']
                labels = result['labels']

            if labels is not None and objects is not None:
                if type_filter:
                    seg_filtered = [o for o in objects if o.obj_type in type_filter]
                else:
                    seg_filtered = list(objects)
                seg_filtered = _apply_height_filter_objects(seg_filtered, params)
                filtered_ids = {o.obj_id for o in seg_filtered}

                type_raster = np.zeros((h, w), dtype=np.float32)
                obj_map = {o.obj_id: o for o in objects}
                for oid in filtered_ids:
                    if oid in obj_map:
                        type_raster[labels == oid] = float(obj_map[oid].type_code)
                height_raster = np.where(ndsm > 0, ndsm, 0).astype(np.float32)

                _write_gpkg_table('segment_type', [type_raster])
                _write_gpkg_table('segment_height', [height_raster])
            else:
                log.info("GeoPackage: skipping segment layers (no segment data available)")
        except Exception as e:
            log.warning("Segments for gpkg failed: %s", e)

    # --- Vector segment polygons ---
    if include_segments_vector:
        try:
            _progress_set(task_id, 'gpkg_vector', 'Vectorising segments…')
            # Get labels + objects (reuse from raster segments if already loaded)
            v_labels = None
            v_objects = None
            cache_key_check = f"{geom_3035.bounds}_{dataset}"
            if include_segments and labels is not None:
                v_labels, v_objects = labels, objects
            else:
                if _seg_cache.get('labels') is not None and _seg_cache.get('key') and cache_key_check in _seg_cache['key']:
                    v_labels = _seg_cache['labels']
                    v_objects = _seg_cache['objects']
                else:
                    _dc = _seg_cache_scan(cache_key_check)
                    if _dc is not None:
                        v_labels = _dc['labels']
                        v_objects = _dc['objects']
            if v_labels is not None and v_objects is not None:
                from rasterio.features import shapes as rasterize_shapes
                import fiona
                from fiona.crs import from_epsg

                obj_map = {o.obj_id: o for o in v_objects}
                # Apply type filter
                if type_filter:
                    v_filtered = [o for o in v_objects if o.obj_type in type_filter]
                else:
                    v_filtered = list(v_objects)
                # Apply height filter
                v_filtered = _apply_height_filter_objects(v_filtered, params)
                filtered_ids = {o.obj_id for o in v_filtered}

                # Vectorise the labels raster
                label_int = v_labels.astype(np.int32)
                seg_mask = mask & np.isin(label_int, list(filtered_ids))

                schema = {
                    'geometry': 'Polygon',
                    'properties': [
                        ('id', 'int'),
                        ('type', 'str'),
                        ('group_type', 'str'),
                        ('height_class', 'str'),
                        ('height_max_m', 'float'),
                        ('height_mean_m', 'float'),
                        ('area_sqm', 'float'),
                        ('slope_mean_deg', 'float'),
                        ('aspect_mean_deg', 'float'),
                        ('aspect_dominant', 'str'),
                        ('elevation_mean_m', 'float'),
                        ('tri_mean', 'float'),
                        ('terrain_class', 'str'),
                        ('confidence', 'float'),
                        ('is_manmade', 'int'),
                        ('ndvi_mean', 'float'),
                        ('color', 'str'),
                        ('color_height', 'str'),
                    ],
                }
                vec_path = tmp_path  # append to same GPKG
                with fiona.open(vec_path, 'w', driver='GPKG', layer='segments',
                                schema=schema, crs=from_epsg(3035)) as dst:
                    written = 0
                    for geom_dict, val in rasterize_shapes(
                        label_int, mask=seg_mask, transform=tf,
                        connectivity=4,
                    ):
                        oid = int(val)
                        obj = obj_map.get(oid)
                        if obj is None:
                            continue
                        # Type-based color
                        tc = SEGMENT_COLORS.get(obj.obj_type, (128, 128, 128, 120))
                        hex_type = '#{:02X}{:02X}{:02X}'.format(tc[0], tc[1], tc[2])
                        # Height-based viridis color (sqrt-scaled 0-45m)
                        hv = _viridis_rgb(min(1.0, (max(0, obj.height_max) / 45.0) ** 0.5))
                        hex_height = '#{:02X}{:02X}{:02X}'.format(*hv)
                        dst.write({
                            'geometry': geom_dict,
                            'properties': {
                                'id': oid,
                                'type': obj.obj_type,
                                'group_type': obj.group_type or '',
                                'height_class': _height_class(obj.height_max),
                                'height_max_m': round(obj.height_max, 2),
                                'height_mean_m': round(obj.height_mean, 2),
                                'area_sqm': round(obj.area_sqm, 1),
                                'slope_mean_deg': round(obj.slope_mean, 1),
                                'aspect_mean_deg': round(obj.aspect_mean, 1),
                                'aspect_dominant': obj.aspect_dominant or 'flat',
                                'elevation_mean_m': round(obj.elevation_mean, 2),
                                'tri_mean': round(obj.tri_mean, 3),
                                'terrain_class': obj.terrain_class or 'level',
                                'confidence': round(obj.confidence, 2),
                                'is_manmade': int(obj.is_manmade) if obj.is_manmade else 0,
                                'ndvi_mean': round(obj.ndvi_mean, 3) if obj.ndvi_mean else 0.0,
                                'color': hex_type,
                                'color_height': hex_height,
                            },
                        })
                        written += 1
                table_count += 1
                log.info("GPKG vector layer 'segments': %d polygons", written)
                # Write QGIS layer_styles table for auto-rendering
                try:
                    _write_gpkg_categorized_style(tmp_path, 'segments', color_mode)
                except Exception as e:
                    log.warning('GPKG style table failed: %s', e)
            else:
                log.info('GeoPackage: skipping vector segments (no segment data available)')
        except Exception as e:
            log.warning('Vector segments for gpkg failed: %s', e)

    # --- Rendered RGBA raster overlays ---
    geom_wgs = _extract_single_geom(features)
    for rlayer in raster_layers:
        try:
            _progress_set(task_id, 'gpkg_overlay', f'Rendering {rlayer}…')
            rgba = _render_overlay_for_gpkg(
                rlayer, data, geom_3035, geom_wgs, dataset,
                type_filter, color_mode, params,
            )
            if rgba is not None:
                bands_u8 = [rgba[:, :, i] for i in range(4)]
                _write_gpkg_table(rlayer, bands_u8, dtype='uint8',
                                  descriptions=['R', 'G', 'B', 'A'])
        except Exception as e:
            log.warning("Raster overlay %s for gpkg failed: %s", rlayer, e)

    # Fix CRS for uint8 raster layers (GDAL registers as 'tiles' not '2d-gridded-coverage')
    _fix_gpkg_raster_crs(tmp_path)

    elapsed = round(time.time() - t0, 2)
    return tmp_path, table_count, elapsed


def _write_gpkg_categorized_style(gpkg_path: str, layer_name: str, color_mode: str = 'type'):
    """Write a QGIS-compatible layer_styles table so the GPKG auto-renders.

    Uses the ``color`` (type) or ``color_height`` field as a data-defined
    fill colour, depending on *color_mode*.
    """
    import sqlite3
    color_field = 'color_height' if color_mode == 'height' else 'color'

    # Build a QML style XML that uses data-defined colour from the color field
    qml = (
        '<!DOCTYPE qgis PUBLIC "http://mrcc.com/qgis.dtd" "SYSTEM">'
        '<qgis version="3.34">'
        '<renderer-v2 type="singleSymbol" symbollevels="0" enableorderby="0">'
        '<symbols>'
        '<symbol type="fill" name="0" clip_to_extent="1" alpha="0.7">'
        '<layer class="SimpleFill" enabled="1" locked="0" pass="0">'
        '<Option type="Map">'
        '<Option type="QString" value="solid" name="style"/>'
        '<Option type="QString" value="0.35,0.35,0.35,255,rgb:0,0,0,1" name="outline_color"/>'
        '<Option type="QString" value="0.2" name="outline_width"/>'
        '</Option>'
        f'<data_defined_properties><Property><Option type="Map">'
        f'<Option type="Map" name="properties"><Option type="Map" name="fillColor">'
        f'<Option type="bool" value="true" name="active"/>'
        f'<Option type="QString" value="&quot;{color_field}&quot;" name="expression"/>'
        f'<Option type="int" value="3" name="type"/>'
        f'</Option></Option></Option></Property></data_defined_properties>'
        '</layer></symbol></symbols></renderer-v2></qgis>'
    )

    conn = sqlite3.connect(gpkg_path)
    try:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS layer_styles ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT,'
            'f_table_catalog TEXT DEFAULT \'\','
            'f_table_schema TEXT DEFAULT \'\','
            'f_table_name TEXT,'
            'f_geometry_column TEXT,'
            'styleName TEXT,'
            'styleQML TEXT,'
            'styleSLD TEXT,'
            'useAsDefault BOOLEAN,'
            'description TEXT,'
            'owner TEXT,'
            'ui TEXT,'
            'update_time TIMESTAMP DEFAULT (strftime(\'%Y-%m-%dT%H:%M:%fZ\',\'now\'))'
            ')'
        )
        conn.execute(
            'INSERT INTO layer_styles '
            '(f_table_name, f_geometry_column, styleName, styleQML, useAsDefault, description) '
            'VALUES (?, ?, ?, ?, 1, ?)',
            (layer_name, 'geom', f'Segment {color_mode}', qml,
             f'Auto-generated colour-by-{color_mode} style'),
        )
        conn.commit()
    finally:
        conn.close()
    log.info('GPKG style written for %s (color_mode=%s)', layer_name, color_mode)


def _render_overlay_for_gpkg(layer_id, data, geom_3035, geom_wgs, dataset,
                              type_filter, color_mode, params):
    """Render a single overlay layer as RGBA array in EPSG:3035 space.

    Returns (h, w, 4) uint8 array or None.
    """
    h, w = data['shape']
    tf = data['transform']
    mask = data['mask']

    ds_map = {
        'dtm-2024': '20240915', 'dtm-2023': '20230915', 'dtm-2022': '20220915',
        'dsm-2024': '20240915', 'dsm-2023': '20230915', 'dsm-2022': '20220915',
    }

    if layer_id.startswith('dtm-'):
        ds = ds_map.get(layer_id, dataset)
        d = raster_io.read_dtm_dsm(geom_3035, ds) if ds != dataset else data
        return _dtm_rgba(d['dtm'], d['mask'])

    elif layer_id.startswith('dsm-'):
        ds = ds_map.get(layer_id, dataset)
        d = raster_io.read_dtm_dsm(geom_3035, ds) if ds != dataset else data
        return _ndsm_rgba(d['ndsm'], d['dsm'], d['mask'])

    elif layer_id.startswith('cir-'):
        # CIR false-color: NIR→R, Red→G, Green→B
        year_str = layer_id.split('-', 1)[1]
        year = int(year_str) if year_str.isdigit() else None
        import ortho_io
        rgb_arr, nir = ortho_io.read_ortho_for_als(data, year=year)
        if nir is None or rgb_arr is None:
            return None
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = nir[:h, :w]
        rgba[:, :, 1] = rgb_arr[0, :h, :w]
        rgba[:, :, 2] = rgb_arr[1, :h, :w]
        rgba[:, :, 3] = np.where(mask[:h, :w], 255, 0)
        return rgba

    elif layer_id == 'hansen':
        bbox_wgs = geom_wgs.bounds if hasattr(geom_wgs, 'bounds') else _extract_single_geom([{'geometry': geom_wgs}]).bounds
        prior = hansen.get_forest_prior(bbox_wgs, tf, (h, w))
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        forest = prior['current_forest']
        rgba[forest] = [20, 120, 20, 140]
        gain = prior['gain']
        rgba[gain] = [0, 200, 200, 180]
        ly = prior['loss_year']
        loss = ly > 0
        brightness = np.clip(80 + (ly.astype(np.float32) / 24.0) * 175, 80, 255).astype(np.uint8)
        rgba[:, :, 0][loss] = brightness[loss]
        rgba[:, :, 1][loss] = 0
        rgba[:, :, 2][loss] = brightness[loss]
        rgba[:, :, 3][loss] = 200
        rgba[:, :, 3][~mask] = 0
        return rgba

    elif layer_id == 'raster':
        # Segment classification raster — needs segmentation
        import object_segmentation as oseg
        labels = None
        objects = None
        seg_mask = None
        seg_ndsm = None
        cache_key_check = f"{geom_3035.bounds}_{dataset}"
        if _seg_cache["key"] and cache_key_check in _seg_cache["key"]:
            labels = _seg_cache["labels"]
            objects = _seg_cache["objects"]
            seg_mask = _seg_cache["mask"]
            seg_ndsm = _seg_cache.get("ndsm")
        else:
            # Try file-backed cache (cross-worker)
            _dc = _seg_cache_scan(cache_key_check)
            if _dc is not None:
                log.info("MBTiles raster: using cached segmentation (disk)")
                labels = _dc['labels']
                objects = _dc['objects']
                seg_mask = _dc['mask']
                seg_ndsm = _dc.get('ndsm')
        if not (labels is not None and objects is not None):
            log.info("gpkg raster overlay: running segmentation")
            spectral = None
            try:
                import ortho_io
                rgb_arr, nir = ortho_io.read_ortho_for_als(data)
                if rgb_arr is not None:
                    spectral = ortho_io.compute_spectral_indices(rgb_arr, nir=nir)
                    spectral["red"] = rgb_arr[0].astype(np.float32)
                    spectral["green"] = rgb_arr[1].astype(np.float32)
                    spectral["blue"] = rgb_arr[2].astype(np.float32)
                    if nir is not None:
                        spectral["nir"] = nir.astype(np.float32)
            except Exception:
                pass
            _il2 = None
            _incl_infra2 = str(params.get('include_infra', 'true')).lower() in ('true', '1', 'yes')
            if _incl_infra2:
                try:
                    from infrastructure_lookup import InfrastructureLookup
                    from pyproj import Transformer as _Tx3
                    _tx3 = _Tx3.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
                    _b2 = geom_3035.bounds
                    _w42, _s42 = _tx3.transform(_b2[0], _b2[1])
                    _e42, _n42 = _tx3.transform(_b2[2], _b2[3])
                    _il2 = InfrastructureLookup.for_bbox(_w42, _s42, _e42, _n42)
                except Exception:
                    pass
            _mark_uncertain2 = str(params.get('mark_uncertain', 'false')).lower() in ('true', '1', 'yes')
            result = seg.segment_and_classify(
                data['dtm'], data['dsm'], mask, tf, spectral=spectral,
                observation_year=ti.dataset_to_year(dataset),
                infra_lookup=_il2,
                mark_uncertain=_mark_uncertain2,
            )
            labels = result['labels']
            objects = result['objects']
            seg_mask = mask
            seg_ndsm = data.get('ndsm')

        hf = _parse_height_filter(params) if params else None
        rgba = _segment_rgba(
            labels, objects, seg_mask,
            type_filter, color_mode=color_mode, ndsm=seg_ndsm,
            height_filter=hf,
        )
        return rgba

    return None


# === SECTION: /api/v1/export/mbtiles endpoint ===

@app.route('/api/v1/export/mbtiles', methods=['POST'])
def export_mbtiles():
    """Export a single raster layer as MBTiles for offline mobile use.

    Query params:
      layer=dtm-2024|ortho-2024|cir-2024|... (required)
      min_zoom=10 (default 10)
      max_zoom=18 (default 18)
      async=true  Return 202 with task_id
    """
    try:
        features = _get_geometry()
        params = _get_params()
        layer = request.args.get('layer', params.get('layer', ''))
        if not layer:
            return _error('layer parameter required')

        run_async = str(request.args.get('async', 'false')).lower() in ('true', '1', 'yes')
        task_id = request.args.get('task_id', str(uuid.uuid4()))

        if run_async:
            _progress_start(task_id)
            thread = threading.Thread(
                target=_mbtiles_worker,
                args=(task_id, features, params, layer),
                daemon=True,
            )
            thread.start()
            return jsonify({"task_id": task_id, "status": "running"}), 202

        # Sync
        path, elapsed = _mbtiles_core(features, params, layer)
        return send_file(path, mimetype='application/x-sqlite3',
                        as_attachment=True, download_name=f'{layer}.mbtiles')
    except Exception as e:
        log.error("mbtiles export: %s", traceback.format_exc())
        return _error(str(e))


def _mbtiles_worker(task_id, features, params, layer):
    try:
        _progress_set(task_id, 'mbtiles_export', f'Building {layer} MBTiles\u2026')
        path, elapsed = _mbtiles_core(features, params, layer, task_id=task_id)
        dest = _RESULTS_DIR / f"{task_id}.mbtiles"
        import shutil
        shutil.move(path, str(dest))
        _progress_done(task_id)
        log.info("MBTiles %s completed: %.1fs", task_id, elapsed)
    except Exception as e:
        log.error("MBTiles %s failed: %s", task_id, traceback.format_exc())
        _progress_error(task_id, str(e))


@app.route('/api/v1/export/mbtiles/download/<task_id>', methods=['GET'])
def download_mbtiles(task_id):
    if not task_id or not re.match(r'^[a-f0-9\\-]+$', task_id):
        return _error('Invalid task_id', 400)
    dest = _RESULTS_DIR / f"{task_id}.mbtiles"
    if not dest.exists():
        return _error('MBTiles not found or not ready yet', 404)
    layer_name = request.args.get('layer', 'layer')
    resp = send_file(str(dest), mimetype='application/x-sqlite3',
                    as_attachment=True, download_name=f'{layer_name}.mbtiles')
    @resp.call_on_close
    def _cleanup():
        try:
            dest.unlink(missing_ok=True)
            (_PROGRESS_DIR / f"{task_id}.json").unlink(missing_ok=True)
        except Exception:
            pass
    return resp


def _mbtiles_core(features, params, layer, task_id=''):
    """Generate MBTiles for a single raster layer."""
    import sqlite3, math
    t0 = time.time()

    feat = features[0]
    geom = feat['geometry']
    geom_3035 = ti.geometry_to_3035(geom)
    if geom.geom_type == 'Point':
        geom_3035 = geom_3035.buffer(100)
    _validate_area(geom_3035)

    dataset = params.get('dataset', ti.DEFAULT_DATASET)
    min_zoom = int(params.get('min_zoom', '10'))
    max_zoom = int(params.get('max_zoom', '16'))

    # Generate the overlay image
    _progress_set(task_id, 'mbtiles_render', f'Rendering {layer}\u2026')

    # Get the overlay PNG + bounds (reuse existing overlay logic)
    geom_wgs = geom
    data = _get_cached_raster(geom_3035, dataset)
    if data is None:
        data = raster_io.read_dtm_dsm(geom_3035, dataset)

    rgba, bounds_wgs = _render_overlay_for_mbtiles(layer, data, geom_3035, geom_wgs, dataset, params, task_id)
    if rgba is None:
        raise ValueError(f'Could not render layer {layer}')

    # bounds_wgs = (south, west, north, east)
    s, w, n, e = bounds_wgs

    # Create MBTiles
    tmp = tempfile.NamedTemporaryFile(suffix='.mbtiles', delete=False)
    tmp_path = tmp.name
    tmp.close()

    conn = sqlite3.connect(tmp_path)
    c = conn.cursor()
    c.execute('CREATE TABLE metadata (name TEXT, value TEXT)')
    c.execute('CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)')
    c.execute('CREATE UNIQUE INDEX tile_index ON tiles (zoom_level, tile_column, tile_row)')

    c.execute("INSERT INTO metadata VALUES ('name', ?)", (layer,))
    c.execute("INSERT INTO metadata VALUES ('format', 'png')")
    c.execute("INSERT INTO metadata VALUES ('bounds', ?)", (f'{w},{s},{e},{n}',))
    c.execute("INSERT INTO metadata VALUES ('minzoom', ?)", (str(min_zoom),))
    c.execute("INSERT INTO metadata VALUES ('maxzoom', ?)", (str(max_zoom),))
    c.execute("INSERT INTO metadata VALUES ('type', 'overlay')")

    from PIL import Image
    # rgba is a numpy array (H, W, 4)
    src_img = Image.fromarray(rgba)
    src_w, src_h = src_img.size

    total_tiles = 0
    for zoom in range(min_zoom, max_zoom + 1):
        _progress_set(task_id, 'mbtiles_tiles', f'Zoom {zoom}/{max_zoom}')
        n_tiles = 2 ** zoom

        # Tile bounds in tile coordinates
        x_min = int((w + 180) / 360 * n_tiles)
        x_max = int((e + 180) / 360 * n_tiles)
        y_min_merc = int((1 - math.log(math.tan(math.radians(n)) + 1/math.cos(math.radians(n))) / math.pi) / 2 * n_tiles)
        y_max_merc = int((1 - math.log(math.tan(math.radians(s)) + 1/math.cos(math.radians(s))) / math.pi) / 2 * n_tiles)

        x_min = max(0, x_min)
        x_max = min(n_tiles - 1, x_max)
        y_min_merc = max(0, y_min_merc)
        y_max_merc = min(n_tiles - 1, y_max_merc)

        for tx in range(x_min, x_max + 1):
            for ty_merc in range(y_min_merc, y_max_merc + 1):
                # Tile bounds in WGS84
                tile_w = tx / n_tiles * 360 - 180
                tile_e = (tx + 1) / n_tiles * 360 - 180
                tile_n_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty_merc / n_tiles)))
                tile_s_rad = math.atan(math.sinh(math.pi * (1 - 2 * (ty_merc + 1) / n_tiles)))
                tile_n_deg = math.degrees(tile_n_rad)
                tile_s_deg = math.degrees(tile_s_rad)

                # Map source image pixels to this tile
                # Source image covers (s,w)->(n,e)
                px_left = (tile_w - w) / (e - w) * src_w
                px_right = (tile_e - w) / (e - w) * src_w
                px_top = (n - tile_n_deg) / (n - s) * src_h
                px_bottom = (n - tile_s_deg) / (n - s) * src_h

                # Crop and resize to 256x256
                crop_box = (int(px_left), int(px_top), int(px_right), int(px_bottom))
                # Skip fully out-of-bounds tiles
                if crop_box[2] <= 0 or crop_box[0] >= src_w or crop_box[3] <= 0 or crop_box[1] >= src_h:
                    continue

                # Clamp
                crop_box = (max(0, crop_box[0]), max(0, crop_box[1]),
                           min(src_w, crop_box[2]), min(src_h, crop_box[3]))
                if crop_box[2] - crop_box[0] < 1 or crop_box[3] - crop_box[1] < 1:
                    continue

                tile_img = src_img.crop(crop_box).resize((256, 256), Image.LANCZOS)

                import io as _io
                buf = _io.BytesIO()
                tile_img.save(buf, format='PNG', optimize=True)
                tile_data = buf.getvalue()

                # TMS y-flip
                tms_y = n_tiles - 1 - ty_merc
                c.execute('INSERT OR REPLACE INTO tiles VALUES (?,?,?,?)',
                         (zoom, tx, tms_y, tile_data))
                total_tiles += 1

    conn.commit()
    conn.close()
    elapsed = time.time() - t0
    log.info("MBTiles %s: %d tiles, %.1fs", layer, total_tiles, elapsed)
    return tmp_path, elapsed


def _render_overlay_for_mbtiles(layer, data, geom_3035, geom_wgs, dataset, params, task_id=''):
    """Render a layer overlay and return (rgba_array, bounds_wgs).
    bounds_wgs = (south, west, north, east)
    """
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import array_bounds

    tf = data['transform']
    h, w_px = data['shape']
    mask = data['mask']

    def _reproject_rgba_to_wgs84(rgba_3035):
        """Reproject an (H, W, 4) uint8 RGBA from EPSG:3035 → EPSG:4326.

        Returns (rgba_wgs, bounds_wgs) where bounds_wgs = (south, west, north, east).
        """
        src_crs = 'EPSG:3035'
        dst_crs = 'EPSG:4326'
        bounds_3035 = array_bounds(h, w_px, tf)
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, w_px, h, *bounds_3035)
        rgba_wgs = np.zeros((dst_h, dst_w, 4), dtype=np.uint8)
        for band in range(4):
            reproject(
                rgba_3035[:, :, band], rgba_wgs[:, :, band],
                src_transform=tf, src_crs=src_crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
        # (left, bottom, right, top) → (south, west, north, east)
        b = array_bounds(dst_h, dst_w, dst_transform)
        bounds_wgs = (b[1], b[0], b[3], b[2])
        return rgba_wgs, bounds_wgs

    try:
        # Extract params needed by _render_overlay_for_gpkg
        type_filter_str = params.get('types', None)
        type_filter = set(t.strip() for t in type_filter_str.split(',')) if type_filter_str else None
        color_mode = params.get('color_mode', 'type')

        # Delegate to _render_overlay_for_gpkg which handles all layer types
        # in EPSG:3035, then reproject the result to WGS84.
        rgba_3035 = _render_overlay_for_gpkg(
            layer, data, geom_3035, geom_wgs, dataset,
            type_filter, color_mode, params,
        )

        if rgba_3035 is None:
            # _render_overlay_for_gpkg doesn't handle ortho-* layers;
            # render them here directly.
            if layer.startswith('ortho-'):
                year_str = layer.split('-', 1)[1]
                year = int(year_str) if year_str.isdigit() else None
                import ortho_io
                rgb_arr, nir = ortho_io.read_ortho_for_als(data, year=year)
                if rgb_arr is None:
                    log.error("MBTiles render %s: no ortho data", layer)
                    return None, None
                rgba_3035 = np.zeros((h, w_px, 4), dtype=np.uint8)
                rgba_3035[:, :, 0] = rgb_arr[0, :h, :w_px]
                rgba_3035[:, :, 1] = rgb_arr[1, :h, :w_px]
                rgba_3035[:, :, 2] = rgb_arr[2, :h, :w_px]
                rgba_3035[:, :, 3] = np.where(mask[:h, :w_px], 255, 0)
            else:
                # Unknown layer – fall back to DTM hillshade
                log.warning("MBTiles render %s: unknown layer, falling back to DTM", layer)
                rgba_3035 = _dtm_rgba(data['dtm'], mask)

        if rgba_3035 is None:
            return None, None

        return _reproject_rgba_to_wgs84(rgba_3035)

    except Exception as e:
        log.error("MBTiles render %s failed: %s", layer, e)
        return None, None


# === SECTION: /api/v1/changes endpoint (temporal analysis) ===

@app.route('/api/v1/changes', methods=['POST'])
def changes():
    """Detect changes between two ALS dates. Returns classified change events."""
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        date_a = params.get('date_a', '20220915')
        date_b = params.get('date_b', '20240915')
        min_change = float(params.get('min_change', 1.0))

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(200)
        _validate_area(geom_3035)

        comparison = tca.compare_dates(geom_3035, date_a, date_b)
        events = tca.detect_changes(geom_3035, date_a, date_b,
                                    min_change=min_change, comparison=comparison)

        event_features = []
        for ev in events:
            centroid_wgs = ti.geometry_from_3035(Point(ev.centroid_e, ev.centroid_n))
            props = {
                "event_type": ev.event_type,
                "area_sqm": ev.area_sqm,
                "height_before_m": ev.height_before,
                "height_after_m": ev.height_after,
                "height_change_mean_m": ev.height_change_mean,
                "height_change_max_m": ev.height_change_max,
                "dtm_change_mean_m": ev.dtm_change_mean,
                "dtm_change_max_m": ev.dtm_change_max,
                "dsm_change_mean_m": ev.dsm_change_mean,
                "confidence": ev.confidence,
                "detail": ev.detail,
            }
            event_features.append({"type": "Feature", "properties": props,
                                   "geometry": mapping(centroid_wgs)})

        # Summarise by type
        by_type = {}
        for ev in events:
            t2 = ev.event_type
            by_type.setdefault(t2, {"count": 0, "total_area_sqm": 0})
            by_type[t2]["count"] += 1
            by_type[t2]["total_area_sqm"] += ev.area_sqm
        for v in by_type.values():
            v["total_area_sqm"] = round(v["total_area_sqm"], 1)

        return jsonify({
            "type": "FeatureCollection", "features": event_features,
            "summary": {"total_events": len(events), "by_type": by_type},
            "comparison_stats": comparison["stats"],
            "meta": {"date_a": date_a, "date_b": date_b, "min_change_m": min_change,
                     "processing_time_s": round(time.time()-t0, 2)},
        })
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# === SECTION: /api/v1/changes/trees endpoint ===

@app.route('/api/v1/changes/trees', methods=['POST'])
def changes_trees():
    """Per-tree growth / felling analysis between two dates."""
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        date_a = params.get('date_a', '20220915')
        date_b = params.get('date_b', '20240915')

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(200)
        _validate_area(geom_3035)

        tree_changes = tca.detect_tree_growth(geom_3035, date_a, date_b)

        tree_features = []
        for tc in tree_changes:
            centroid_wgs = ti.geometry_from_3035(Point(tc.centroid_e, tc.centroid_n))
            tree_features.append({"type": "Feature", "properties": {
                "tree_id": tc.tree_id, "status": tc.status,
                "height_before_m": tc.height_before, "height_after_m": tc.height_after,
                "height_change_m": tc.height_change,
                "crown_area_before_sqm": tc.crown_area_before,
                "crown_area_after_sqm": tc.crown_area_after,
            }, "geometry": mapping(centroid_wgs)})

        by_status = {}
        for tc in tree_changes:
            by_status.setdefault(tc.status, {"count": 0, "mean_dh": []})
            by_status[tc.status]["count"] += 1
            by_status[tc.status]["mean_dh"].append(tc.height_change)
        for v in by_status.values():
            dh_list = v.pop("mean_dh")
            v["height_change_mean_m"] = round(float(np.mean(dh_list)), 2) if dh_list else 0

        return jsonify({
            "type": "FeatureCollection", "features": tree_features,
            "summary": {"total_trees": len(tree_changes), "by_status": by_status},
            "meta": {"date_a": date_a, "date_b": date_b,
                     "processing_time_s": round(time.time()-t0, 2)},
        })
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# === SECTION: Multi-epoch summary ===

@app.route('/api/v1/changes/summary', methods=['POST'])
def changes_summary():
    """Multi-epoch change summary across all available dates."""
    try:
        t0 = time.time()
        features = _get_geometry()
        params = _get_params()
        dates = params.get('dates', None)
        if isinstance(dates, str):
            dates = [d.strip() for d in dates.split(',')]

        feat = features[0]
        geom = feat['geometry']
        geom_3035 = ti.geometry_to_3035(geom)
        if geom.geom_type == 'Point':
            geom_3035 = geom_3035.buffer(200)
        _validate_area(geom_3035)

        result = tca.temporal_summary(geom_3035, dates=dates)
        result["meta"] = {"processing_time_s": round(time.time()-t0, 2)}
        return jsonify(result)
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error(traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# === SECTION: /api/v1/info endpoint ===

# === SECTION: LiDAR/ortho/DTM/CIR/Hansen overlay + GeoTIFF download ===

def _extract_single_geom(features_or_geom):
    """Extract a single shapely geometry from _get_geometry() output."""
    if isinstance(features_or_geom, list):
        if not features_or_geom:
            raise ValueError("No geometry provided.")
        feat = features_or_geom[0]
        if isinstance(feat, dict) and 'geometry' in feat:
            return feat['geometry']
        return shape(feat) if isinstance(feat, dict) else feat
    return features_or_geom


def _geometry_to_3035_bbox(features_or_geom):
    """Convert WGS84 geometry to EPSG:3035 and return (geom_3035, bbox_3035, bbox_wgs84)."""
    geom_wgs84 = _extract_single_geom(features_or_geom)
    geom_3035 = ti.geometry_to_3035(geom_wgs84)
    _validate_area(geom_3035)
    b = geom_3035.bounds  # (minx, miny, maxx, maxy)
    bw = geom_wgs84.bounds
    return geom_3035, b, bw


def _get_cached_raster(geom_3035, dataset):
    """Return DTM/DSM data from cache if available, else read from remote."""
    cache_key = f"{geom_3035.bounds}_{dataset}"
    if _raster_cache["key"] == cache_key and _raster_cache["data"] is not None:
        log.info("raster cache hit for %s", cache_key)
        return _raster_cache["data"]
    log.info("raster cache miss, reading from remote")
    data = raster_io.read_dtm_dsm(geom_3035, dataset)
    _raster_cache.update({"key": cache_key, "data": data})
    return data


def _hillshade(elevation, azimuth=315, altitude=45, z_factor=1.0):
    """Compute hillshade from an elevation array."""
    dy, dx = np.gradient(elevation, 1.0)
    dx *= z_factor
    dy *= z_factor
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    hs = (np.sin(alt_rad) * np.cos(slope) +
          np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return np.clip(hs, 0, 1).astype(np.float32)


def _dtm_rgba(dtm, mask):
    """Render DTM as hillshade relief with hypsometric tinting.

    Shows actual terrain: valleys, ridges, slopes — the relief you see on
    a good topographic map.
    """
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap

    h, w = dtm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # --- Multi-directional hillshade for rich relief ---
    hs1 = _hillshade(dtm, azimuth=315, altitude=40)
    hs2 = _hillshade(dtm, azimuth=90, altitude=55)
    hs_combined = np.clip(0.65 * hs1 + 0.35 * hs2, 0, 1)

    # --- Hypsometric tint based on DTM elevation ---
    vmin = float(np.nanpercentile(dtm[mask], 2)) if mask.any() else 0
    vmax = float(np.nanpercentile(dtm[mask], 98)) if mask.any() else 1000
    if vmax - vmin < 10:
        vmax = vmin + 10

    hypso_colors = [
        (0.0, '#2d6a2e'),   # valley: deep green
        (0.15, '#5a9e3c'),  # lower slopes: green
        (0.3, '#8ebb4a'),   # mid-low: yellow-green
        (0.45, '#c4b85c'),  # mid: olive/tan
        (0.6, '#b8956a'),   # upper mid: brown
        (0.75, '#a08070'),  # upper slopes: grey-brown
        (0.9, '#c8c0b8'),   # near summit: light grey
        (1.0, '#f0ece8'),   # summit: near-white
    ]
    cmap_hypso = LinearSegmentedColormap.from_list('hypso', hypso_colors)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    hypso = cmap_hypso(norm(np.clip(dtm, vmin, vmax)))[:, :, :3]  # (H,W,3) float

    # Modulate by hillshade
    for c in range(3):
        hypso[:, :, c] = hypso[:, :, c] * (0.25 + 0.75 * hs_combined)

    rgba[:, :, :3] = (np.clip(hypso, 0, 1) * 255).astype(np.uint8)
    rgba[:, :, 3] = np.where(mask, 255, 0)
    return rgba


def _ndsm_rgba(ndsm, dsm, mask, vmax=45):
    """Render nDSM as viridis height coloring with DSM hillshade for 3D effect.

    Uses sqrt scaling to better distinguish old-growth forest (25-45m).
    Ground (ndsm < 0.3) is transparent so it can layer over the DTM relief.
    """
    import matplotlib
    import matplotlib.cm as cm

    h, w = ndsm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    hs_dsm = _hillshade(dsm, azimuth=315, altitude=40)

    colormap = cm.get_cmap('viridis')
    ndsm_clamped = np.clip(ndsm, 0, vmax)
    # sqrt scaling: spreads upper range for better old-growth forest distinction
    ndsm_sqrt = np.sqrt(ndsm_clamped / vmax)
    rgb_float = colormap(ndsm_sqrt)[:, :, :3]  # (H,W,3)

    # Modulate by DSM hillshade
    for c in range(3):
        rgb_float[:, :, c] = rgb_float[:, :, c] * (0.3 + 0.7 * hs_dsm)

    elevated = mask & (ndsm > 0.3)
    rgba[:, :, :3] = (np.clip(rgb_float, 0, 1) * 255).astype(np.uint8)
    rgba[:, :, 3] = np.where(elevated, 220, 0)
    return rgba


def _reproject_rasters_to_wgs84(arrays_3035, transform_3035, shape_3035, mask_3035=None):
    """Reproject raw float32 rasters from EPSG:3035 to EPSG:4326.

    Parameters
    ----------
    arrays_3035 : dict[str, np.ndarray]
        Named 2D float32 arrays in EPSG:3035 (e.g. dtm, dsm, ndsm).
    transform_3035 : Affine
        Source rasterio transform.
    shape_3035 : (int, int)
        Source (rows, cols).
    mask_3035 : np.ndarray or None
        Boolean mask. Reprojected as nearest-neighbour.

    Returns
    -------
    (arrays_wgs, mask_wgs, transform_wgs, bounds_wgs)
        arrays_wgs: dict with same keys, reprojected.
        mask_wgs: boolean mask in WGS84 grid.
        transform_wgs: rasterio Affine for the WGS84 grid.
        bounds_wgs: (south, west, north, east) for Leaflet.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds

    src_crs = CRS.from_epsg(3035)
    dst_crs = CRS.from_epsg(4326)
    h, w = shape_3035

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, w, h, *array_bounds(h, w, transform_3035),
    )

    arrays_wgs = {}
    for name, src in arrays_3035.items():
        dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=src.astype(np.float32),
            destination=dst,
            src_transform=transform_3035,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
        arrays_wgs[name] = dst

    # Reproject mask
    if mask_3035 is not None:
        mask_src = mask_3035.astype(np.uint8)
        mask_dst = np.zeros((dst_height, dst_width), dtype=np.uint8)
        reproject(
            source=mask_src,
            destination=mask_dst,
            src_transform=transform_3035,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        mask_wgs = mask_dst > 0
    else:
        # Derive mask from any reprojected array (non-NaN)
        first = next(iter(arrays_wgs.values()))
        mask_wgs = ~np.isnan(first)

    bounds = array_bounds(dst_height, dst_width, dst_transform)
    # (left, bottom, right, top) = (west, south, east, north)
    bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])  # south, west, north, east

    return arrays_wgs, mask_wgs, dst_transform, bounds_wgs


def _reproject_rgb_to_wgs84(rgb_3035, transform_3035, shape_3035):
    """Reproject a (3,H,W) uint8 RGB array from EPSG:3035 to EPSG:4326.

    Returns (rgb_wgs, mask_wgs, bounds_wgs).
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds

    src_crs = CRS.from_epsg(3035)
    dst_crs = CRS.from_epsg(4326)
    h, w = shape_3035

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, w, h, *array_bounds(h, w, transform_3035),
    )

    rgb_wgs = np.zeros((3, dst_height, dst_width), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=rgb_3035[band],
            destination=rgb_wgs[band],
            src_transform=transform_3035,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    mask_wgs = (rgb_wgs[0] > 0) | (rgb_wgs[1] > 0) | (rgb_wgs[2] > 0)

    bounds = array_bounds(dst_height, dst_width, dst_transform)
    bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])

    return rgb_wgs, mask_wgs, bounds_wgs


def _send_rgba_overlay(rgba, bounds_wgs):
    """Encode RGBA array as PNG and return Flask response with bounds header."""
    from PIL import Image
    img = Image.fromarray(rgba, 'RGBA')
    buf = io.BytesIO()
    img.save(buf, 'PNG', optimize=True)
    buf.seek(0)
    resp = send_file(buf, mimetype='image/png')
    south, west, north, east = bounds_wgs
    resp.headers['X-Bounds'] = f'{south},{west},{north},{east}'
    resp.headers['Access-Control-Expose-Headers'] = 'X-Bounds'
    return resp


@app.route('/api/v1/dtm/overlay', methods=['POST'])
def dtm_overlay():
    """Return DTM hillshade relief as a PNG overlay (reprojected to WGS84)."""
    try:
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        data = _get_cached_raster(geom_3035, dataset)

        # Reproject raw elevation to WGS84 *before* computing hillshade
        arrays_wgs, mask_wgs, tf_wgs, bounds_wgs = _reproject_rasters_to_wgs84(
            {'dtm': data['dtm']},
            data['transform'], data['shape'], data['mask'],
        )
        dtm_wgs = arrays_wgs['dtm']
        # Fill NaN for hillshade computation
        dtm_wgs = np.nan_to_num(dtm_wgs, nan=float(np.nanmedian(dtm_wgs[mask_wgs])) if mask_wgs.any() else 0)

        rgba = _dtm_rgba(dtm_wgs, mask_wgs)
        return _send_rgba_overlay(rgba, bounds_wgs)
    except Exception as e:
        log.error("dtm overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/lidar/overlay', methods=['POST'])
def lidar_overlay():
    """Return nDSM as a PNG overlay (viridis + hillshade, reprojected to WGS84)."""
    try:
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        data = _get_cached_raster(geom_3035, dataset)

        # Reproject raw elevation to WGS84 before rendering
        arrays_wgs, mask_wgs, tf_wgs, bounds_wgs = _reproject_rasters_to_wgs84(
            {'ndsm': data['ndsm'], 'dsm': data['dsm']},
            data['transform'], data['shape'], data['mask'],
        )
        ndsm_wgs = np.nan_to_num(arrays_wgs['ndsm'], nan=0)
        dsm_wgs = np.nan_to_num(arrays_wgs['dsm'], nan=0)

        rgba = _ndsm_rgba(ndsm_wgs, dsm_wgs, mask_wgs)
        return _send_rgba_overlay(rgba, bounds_wgs)
    except Exception as e:
        log.error("lidar overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/ortho/overlay', methods=['POST'])
def ortho_overlay():
    """Return orthophoto as a PNG overlay (reprojected to WGS84)."""
    try:
        import ortho_io
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        ortho_year = params.get('ortho_year')
        if ortho_year:
            ortho_year = int(ortho_year)
        data = _get_cached_raster(geom_3035, dataset)

        # Use cached ortho if available and matches (same geometry, default year)
        raster_cache_key = f"{geom_3035.bounds}_{dataset}"
        if (not ortho_year and _raster_cache.get("ortho") is not None
                and _raster_cache.get("ortho_key") == raster_cache_key):
            log.info("ortho overlay: using cached ortho")
            rgb = _raster_cache["ortho"]
            nir = None  # nir not needed for RGB overlay
        else:
            rgb, nir = ortho_io.read_ortho_for_als(data, year=ortho_year)

        # Reproject RGB to WGS84
        rgb_wgs, mask_wgs, bounds_wgs = _reproject_rgb_to_wgs84(
            rgb, data['transform'], data['shape'],
        )

        h, w = rgb_wgs.shape[1], rgb_wgs.shape[2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = rgb_wgs[0]
        rgba[:, :, 1] = rgb_wgs[1]
        rgba[:, :, 2] = rgb_wgs[2]
        rgba[:, :, 3] = np.where(mask_wgs, 255, 0)

        return _send_rgba_overlay(rgba, bounds_wgs)
    except Exception as e:
        log.error("ortho overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/cir/overlay', methods=['POST'])
def cir_overlay():
    """Return Color Infrared (CIR) false-color overlay.

    CIR maps NIR→Red, Red→Green, Green→Blue, highlighting vegetation
    health (bright red = vigorous vegetation, dark = bare/water).
    """
    try:
        import ortho_io
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        ortho_year = params.get('ortho_year')
        if ortho_year:
            ortho_year = int(ortho_year)
        data = _get_cached_raster(geom_3035, dataset)

        rgb, nir = ortho_io.read_ortho_for_als(data, year=ortho_year)
        if nir is None:
            return _error('NIR band not available for this area (no RGBI operate coverage)', 404)

        # CIR: NIR→R, Red→G, Green→B
        cir = np.stack([nir, rgb[0], rgb[1]], axis=0)  # (3,H,W) uint8

        cir_wgs, mask_wgs, bounds_wgs = _reproject_rgb_to_wgs84(
            cir, data['transform'], data['shape'],
        )

        h, w = cir_wgs.shape[1], cir_wgs.shape[2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, 0] = cir_wgs[0]
        rgba[:, :, 1] = cir_wgs[1]
        rgba[:, :, 2] = cir_wgs[2]
        rgba[:, :, 3] = np.where(mask_wgs, 255, 0)

        return _send_rgba_overlay(rgba, bounds_wgs)
    except Exception as e:
        log.error("cir overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/hansen/overlay', methods=['POST'])
def hansen_overlay():
    """Return Hansen forest loss as a coloured PNG overlay.

    Green = current forest, magenta = loss (brighter = more recent),
    cyan = forest gain.
    """
    try:
        geom_wgs84 = _get_geometry()
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035, b3035, bwgs = _geometry_to_3035_bbox(geom_wgs84)

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        geom_wgs = _extract_single_geom(geom_wgs84)
        bbox_wgs = geom_wgs.bounds

        prior = hansen.get_forest_prior(
            bbox_wgs, data['transform'], data['shape'],
        )

        h, w = data['shape']
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Current forest: green
        forest = prior['current_forest']
        rgba[:, :, 0][forest] = 20
        rgba[:, :, 1][forest] = 120
        rgba[:, :, 2][forest] = 20
        rgba[:, :, 3][forest] = 140

        # Forest gain: cyan
        gain = prior['gain']
        rgba[:, :, 0][gain] = 0
        rgba[:, :, 1][gain] = 200
        rgba[:, :, 2][gain] = 200
        rgba[:, :, 3][gain] = 180

        # Loss: magenta, brightness by recency (year 1=2001 dark, 24=2024 bright)
        ly = prior['loss_year']
        loss = ly > 0
        brightness = np.clip(80 + (ly.astype(np.float32) / 24.0) * 175, 80, 255).astype(np.uint8)
        rgba[:, :, 0][loss] = brightness[loss]
        rgba[:, :, 1][loss] = 0
        rgba[:, :, 2][loss] = brightness[loss]
        rgba[:, :, 3][loss] = 200

        # Transparent where no data
        rgba[:, :, 3][~data['mask']] = 0

        from rasterio.warp import calculate_default_transform, reproject as rp, Resampling
        from rasterio.crs import CRS
        from rasterio.transform import array_bounds

        src_crs = CRS.from_epsg(3035)
        dst_crs = CRS.from_epsg(4326)
        dst_tf, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, w, h, *array_bounds(h, w, data['transform']),
        )
        rgba_wgs = np.zeros((4, dst_h, dst_w), dtype=np.uint8)
        for band in range(4):
            rp(
                source=rgba[:, :, band],
                destination=rgba_wgs[band],
                src_transform=data['transform'],
                src_crs=src_crs,
                dst_transform=dst_tf,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
        rgba_out = np.transpose(rgba_wgs, (1, 2, 0))
        bounds = array_bounds(dst_h, dst_w, dst_tf)
        bounds_wgs = (bounds[1], bounds[0], bounds[3], bounds[2])
        return _send_rgba_overlay(rgba_out, bounds_wgs)
    except Exception as e:
        log.error("hansen overlay: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/lidar/geotiff', methods=['POST'])
def lidar_geotiff():
    """Download nDSM as a georeferenced GeoTIFF."""
    try:
        geom_wgs84 = _extract_single_geom(_get_geometry())
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035 = ti.geometry_to_3035(geom_wgs84)
        _validate_area(geom_3035)

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        ndsm = data['ndsm']
        dtm = data['dtm']
        dsm = data['dsm'] if 'dsm' in data else dtm + ndsm
        tf = data['transform']
        h, w = data['shape']

        tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        with rasterio.open(tmp.name, 'w', driver='GTiff', width=w, height=h,
                           count=3, dtype='float32', crs='EPSG:3035',
                           transform=tf, compress='deflate') as dst:
            dst.write(dtm.astype(np.float32), 1)
            dst.write(dsm.astype(np.float32), 2)
            dst.write(ndsm.astype(np.float32), 3)
            dst.set_band_description(1, 'DTM')
            dst.set_band_description(2, 'DSM')
            dst.set_band_description(3, 'nDSM')

        return send_file(tmp.name, mimetype='image/tiff', as_attachment=True,
                         download_name='lidar_dtm_dsm_ndsm.tif')
    except Exception as e:
        log.error("lidar geotiff: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/ortho/geotiff', methods=['POST'])
def ortho_geotiff():
    """Download orthophoto as a georeferenced GeoTIFF (RGB + NIR if available)."""
    try:
        import ortho_io
        geom_wgs84 = _extract_single_geom(_get_geometry())
        params = _get_params()
        dataset = params.get('dataset', '20240915')
        geom_3035 = ti.geometry_to_3035(geom_wgs84)
        _validate_area(geom_3035)

        data = raster_io.read_dtm_dsm(geom_3035, dataset)
        rgb, nir = ortho_io.read_ortho_for_als(data)
        tf = data['transform']
        h, w = data['shape']

        n_bands = 4 if nir is not None else 3
        tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
        with rasterio.open(tmp.name, 'w', driver='GTiff', width=w, height=h,
                           count=n_bands, dtype='uint8', crs='EPSG:3035',
                           transform=tf, compress='deflate') as dst:
            dst.write(rgb[0], 1)
            dst.write(rgb[1], 2)
            dst.write(rgb[2], 3)
            dst.set_band_description(1, 'Red')
            dst.set_band_description(2, 'Green')
            dst.set_band_description(3, 'Blue')
            if nir is not None:
                dst.write(nir, 4)
                dst.set_band_description(4, 'NIR')

        return send_file(tmp.name, mimetype='image/tiff', as_attachment=True,
                         download_name='orthophoto_rgbi.tif')
    except Exception as e:
        log.error("ortho geotiff: %s", traceback.format_exc())
        return _error(str(e))


# === SECTION: RF classifier training endpoints ===

@app.route('/api/v1/classifier/train', methods=['POST'])
def train_classifier():
    """Train RF classifier from cadastre ground truth over a bbox.

    Params: geometry (bbox/geojson), dataset, include_ortho, include_copernicus,
            include_temporal, include_hansen.
    Fetches segment features + cadastre parcel codes, trains RF model.
    All data sources (ortho, copernicus, hansen, temporal) are included by
    default so the RF sees the same features it will use at inference time.
    """
    try:
        import learned_classifier as lc
        import object_segmentation as oc
        import cadastre

        params = _parse_params()
        geom, geom_3035 = _parse_geometry(params)
        dataset = params.get('dataset', ti.DEFAULT_DATASET)
        obs_year = ti.dataset_to_year(dataset)

        include_temporal = str(params.get('include_temporal', 'true')).lower() in ('true', '1', 'yes')
        include_hansen = str(params.get('include_hansen', 'true')).lower() in ('true', '1', 'yes')

        # Read LIDAR
        data = raster_io.read_dtm_dsm(geom_3035, dataset)

        # Multi-temporal DTM/DSM
        dtm_dates, dsm_dates = None, None
        if include_temporal:
            try:
                multi = raster_io.read_multi_date_ndsm(geom_3035)
                dtm_dates, dsm_dates = {}, {}
                for d in multi['dates_loaded']:
                    try:
                        dd = raster_io.read_dtm_dsm(geom_3035, dataset=d)
                        mh = min(dd['shape'][0], data['shape'][0])
                        mw = min(dd['shape'][1], data['shape'][1])
                        dtm_dates[d] = dd['dtm'][:mh, :mw]
                        dsm_dates[d] = dd['dsm'][:mh, :mw]
                    except Exception as e:
                        log.warning("Train: date %s load failed: %s", d, e)
            except Exception as e:
                log.warning("Train: multi-temporal failed: %s", e)

        # Read ortho
        rgb, spectral = _try_read_ortho(data)

        # Copernicus (NDVI, land cover, SAR, harmonics)
        copernicus_data = None
        if not _is_processor_running():
            copernicus_data = _try_copernicus(geom, sar=True, harmonics=True, year=ti.dataset_to_year(dataset))

        # Hansen forest prior
        hansen_data = None
        if include_hansen:
            hansen_data = _try_hansen(geom, data['transform'], data['shape'])

        # Building footprints from cadastre (for calibration features)
        building_footprints = _try_cadastre(geom, data['transform'], data['shape'])

        # Infrastructure lookup
        _il3 = None
        include_infra = str(params.get('include_infra', 'true')).lower() in ('true', '1', 'yes')
        if include_infra:
            try:
                from infrastructure_lookup import InfrastructureLookup
                from pyproj import Transformer as _Tx4
                _tx4 = _Tx4.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
                _b3 = geom.bounds
                _w43, _s43 = _tx4.transform(_b3[0], _b3[1])
                _e43, _n43 = _tx4.transform(_b3[2], _b3[3])
                _il3 = InfrastructureLookup.for_bbox(_w43, _s43, _e43, _n43)
            except Exception:
                pass

        # Segment (feature extraction) — pass ALL data sources
        result = oc.segment_and_classify(
            data['dtm'], data['dsm'], data['mask'], data['transform'],
            dtm_dates=dtm_dates,
            dsm_dates=dsm_dates,
            spectral=spectral,
            copernicus=copernicus_data,
            building_footprints=building_footprints,
            hansen=hansen_data,
            ortho_year=obs_year,
            observation_year=obs_year,
            infra_lookup=_il3,
        )
        objects = result['objects']
        labels_arr = result['labels']

        # Hansen tree-loss calibration
        if include_hansen and hansen_data:
            try:
                objects = hansen.calibrate_tree_loss(
                    objects, labels_arr, hansen_data,
                    observation_year=obs_year,
                )
            except Exception as e:
                log.warning("Train: Hansen calibration failed: %s", e)

        features = [obj.features for obj in objects]

        # Fetch cadastre parcel codes
        bbox_wgs = geom.bounds
        try:
            parcels = cadastre.fetch_parcel_land_use(
                (bbox_wgs[0], bbox_wgs[1], bbox_wgs[2], bbox_wgs[3]))
        except Exception:
            parcels = None

        if parcels is None:
            return _error("Could not fetch cadastre parcel codes")

        # Match segments to cadastre labels
        train_features = []
        train_labels = []
        for feat in features:
            code = _dominant_cadastre_code(feat, parcels)
            if code and code in lc.CADASTRE_TO_TYPE:
                train_features.append(feat)
                train_labels.append(lc.CADASTRE_TO_TYPE[code])

        if len(train_features) < 20:
            return _error(f"Only {len(train_features)} labelled segments, need >= 20")

        clf = lc.LearnedClassifier()
        stats = clf.train(train_features, train_labels)

        # Clear cached raster data to reclaim memory
        _clear_raster_caches()

        return jsonify({
            "status": "trained",
            "training_stats": stats,
            "n_segments_total": len(features),
            "n_segments_labelled": len(train_features),
            "dataset": dataset,
            "observation_year": obs_year,
            "data_sources": {
                "temporal": dtm_dates is not None and len(dtm_dates) >= 2,
                "ortho": spectral is not None,
                "copernicus": copernicus_data is not None,
                "hansen": hansen_data is not None,
                "building_footprints": building_footprints is not None,
            },
        })

    except Exception as e:
        log.error("classifier train: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/classifier/status', methods=['GET'])
def classifier_status():
    """Check if a trained RF model exists."""
    try:
        import learned_classifier as lc
        clf = lc.get_classifier()
        # Report which model source is active
        if lc.BEST_MODEL_PATH.exists() and lc.BEST_META_PATH.exists():
            model_source = "best_model"
        elif lc.MODEL_PATH.exists():
            model_source = "live"
        else:
            model_source = "none"
        return jsonify({
            "trained": clf.is_trained,
            "trained_at": clf.trained_at,
            "n_kgs": clf.n_kgs,
            "n_train": clf.n_train,
            "oob_score": clf.oob_score,
            "n_classes": len(clf.classes),
            "classes": clf.classes,
            "model_source": model_source,
            "top_features": dict(sorted(
                clf.feature_importances.items(),
                key=lambda x: -x[1]
            )[:15]) if clf.feature_importances else {},
        })
    except Exception as e:
        return _error(str(e))


def _dominant_cadastre_code(feat, parcels):
    """Find the most common cadastre code overlapping a segment."""
    # parcels is expected to be a list of {geometry, code} or similar
    # For now, find parcels whose centroid falls within segment bbox
    ce = feat.get("centroid_e", 0)
    cn = feat.get("centroid_n", 0)
    if not parcels:
        return None
    # Simple: find nearest parcel
    best = None
    best_dist = float('inf')
    for p in parcels:
        pc = p.get("centroid")
        code = p.get("code")
        if pc and code:
            d = ((pc[0] - ce)**2 + (pc[1] - cn)**2)**0.5
            if d < best_dist:
                best_dist = d
                best = code
    return best


# === SECTION: /api/v1/docs + share endpoints ===

@app.route('/api/v1/info', methods=['GET'])
def info():
    return jsonify({
        "name": "Austrian LIDAR & Orthophoto Analysis API",
        "version": "3.0.0",
        "description": "Landscape transformation analysis: LIDAR DTM/DSM time series + Sentinel-2 + cadastre",
        "classifier": "landscape_v2 — 10 types focused on human transformation of terrain",
        "source": "data.bev.gv.at",
        "resolution_lidar": "1m",
        "resolution_ortho": "0.2m (1m for analysis, 0.5m for GLCM texture)",
        "crs": "EPSG:3035",
        "datasets_als": {k: {"dtm": True, "dsm": True} for k in sorted(ti.DATASETS.keys())},
        "datasets_ortho": ["20220128 (RGB 50km tiles)"],
        "tiles": len(ti.TILE_COORDS),
        "landscape_types": seg.OBJECT_TYPES,
        "data_sources": {
            "bev_als_dtm_dsm": "1m resolution, 3 dates (2022/2023/2024)",
            "bev_dop_rgbi": "0.2m orthophoto, 47 RGBI operates",
            "copernicus_sentinel2": "10m NDVI growing-season composite (via openEO)",
            "copernicus_worldcover": "10m ESA land cover classification",
            "copernicus_sentinel1_sar": "SAR backscatter (VV+VH)",
            "cadastre_footprints": "mm-precision building polygons (ground truth)",
        },
        "change_event_types": tca.EVENT_TYPES,
        "endpoints": {
            "POST /api/v1/elevation": "Enrich features with DSM/DTM elevation",
            "POST /api/v1/terrain": "Terrain characterisation (slope, ruggedness, etc.)",
            "POST /api/v1/segment": "Watershed segmentation: 25 object types + 11 group types (Felzenszwalb+RAG)",
            "POST /api/v1/changes": "Temporal change detection (earthworks, trees, buildings, roads)",
            "POST /api/v1/changes/trees": "Per-tree growth / felling analysis",
            "POST /api/v1/changes/summary": "Multi-epoch change summary (2022→2023→2024)",
            "GET /api/v1/info": "This endpoint",
            "GET /api/v1/docs/llm.txt": "Machine-readable API reference",
        },
        "max_area_sqkm": MAX_AREA_SQM / 1e6,
        "proxy_pool": __import__('bev_proxy').status(),
    })


# === SECTION: /api/v1/layers endpoint (layer availability check) ===

@app.route('/api/v1/layers', methods=['GET'])
def layers_availability():
    """Return which data layers are available for a given bounding box.

    Query params:
      bbox=lon_min,lat_min,lon_max,lat_max   (WGS84)
    """
    try:
        import ortho_io
        bbox_str = request.args.get('bbox', '')
        if not bbox_str:
            return _error('bbox parameter required (lon_min,lat_min,lon_max,lat_max)')
        parts = [float(x.strip()) for x in bbox_str.split(',')]
        if len(parts) != 4:
            return _error('bbox must have 4 values: lon_min,lat_min,lon_max,lat_max')
        lon_min, lat_min, lon_max, lat_max = parts

        # Tile coverage check
        from shapely.geometry import box as shp_box
        bbox_wgs = shp_box(lon_min, lat_min, lon_max, lat_max)
        geom_3035 = ti.geometry_to_3035(bbox_wgs)
        b = geom_3035.bounds
        tiles = ti.find_tiles_for_bbox(b[0], b[1], b[2], b[3])
        has_tiles = len(tiles) > 0

        # RGBI operates per year (needed for CIR; ortho falls back to DOP)
        rgbi_by_year = {}
        for year in (2024, 2023):
            try:
                operates = ortho_io.find_rgbi_operates(
                    lat_min, lon_min, lat_max, lon_max, year=year,
                )
                rgbi_by_year[str(year)] = len(operates) > 0
            except Exception:
                rgbi_by_year[str(year)] = False
        # The "~2020" slot covers the 20221027 series (years 2018-2021)
        try:
            old_ops = ortho_io.find_rgbi_operates(
                lat_min, lon_min, lat_max, lon_max,
            )
            # Filter to operates from the 20221027 series
            rgbi_by_year["2020"] = any(
                ortho_io.RGBI_OPERATES[o]["series"] == "20221027"
                for o in old_ops
            )
        except Exception:
            rgbi_by_year["2020"] = False

        # Ortho: RGBI operates preferred, but DOP (2022) is the fallback
        # So ortho is available if tiles exist (DOP covers all Austria)
        # We mark the year that has RGBI as "best" but the ~2020 slot
        # also covers the DOP 2022 fallback.
        ortho_result = {
            "2024": rgbi_by_year.get("2024", False) or has_tiles,
            "2023": rgbi_by_year.get("2023", False),
            "2020": has_tiles,  # DOP ~2022 always available where tiles exist
        }

        # CIR requires NIR from RGBI operates — no DOP fallback
        cir_result = {
            "2024": rgbi_by_year.get("2024", False),
            "2023": rgbi_by_year.get("2023", False),
            "2020": rgbi_by_year.get("2020", False),
        }

        dtm_result = {y: has_tiles for y in ('2024', '2023', '2022')}
        dsm_result = {y: has_tiles for y in ('2024', '2023', '2022')}
        hansen_available = has_tiles

        resp = {
            "ortho": ortho_result,
            "cir": cir_result,
            "dtm": dtm_result,
            "dsm": dsm_result,
            "hansen": hansen_available,
        }
        if _is_processor_running():
            # Check if tile cache covers this bbox — if so, Copernicus is
            # still available from cache even while processor runs.
            try:
                from tile_cache import CopernicusTileCache
                _cop_cache = CopernicusTileCache()
                _cop_bbox = {"west": lon_min, "south": lat_min,
                             "east": lon_max, "north": lat_max}
                if not _cop_cache.has_cached(_cop_bbox, ndvi=True, landcover=True,
                                            sar=True, harmonics=True):
                    resp["copernicus_disabled"] = True
            except Exception:
                resp["copernicus_disabled"] = True
        return jsonify(resp)
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error("layers endpoint: %s", traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


# === SECTION: /api/v1/onestop endpoint (single-URL segment + download) ===

@app.route('/api/v1/onestop', methods=['GET'])
def onestop():
    """One-stop URL: trigger segmentation and get result/download from a single GET.

    Designed for users on limited connections who want a bookmarkable URL that
    runs analysis and produces a downloadable result.

    Query params (all via URL):
      bbox=lon_min,lat_min,lon_max,lat_max   Bounding box (required)
      name=MySave                            Save name / share ID (slug, 1-80 chars, [A-Za-z0-9_-])
      min_object_size=10                     Min segment area in m² (default: 10)
      include_ortho=true                     Include orthophoto (default: true)
      include_temporal=false                 Include temporal analysis
      include_copernicus=false               Include Sentinel-2/SAR
      include_cadastre=false                 Include cadastre
      include_hansen=false                   Include Hansen forest change
      include_infra=true                     Include infrastructure spatial match
      types=tree,road                        Object type filter
      height_min=X                           Height filter: >= X metres
      height_max=X                           Height filter: <= X metres
      height_op=gt|lt|between                Height filter operator
      format=json|gpkg|kml                   Output format (default: json)
      layers=segments                        GPKG layers (default: segments)
      include_segments_vector=true            GPKG vector polygons (default: true)
      segment_geometry=point|polygon         Feature geometry in KML/GPKG (default: point for KML, polygon for GPKG)
      segment_geometry_style=type|height      Colour scheme: by object type or height ramp (default: type)
      group_by=type|height_class             KML folder grouping

    Returns:
      - If no task running: starts analysis, returns 202 JSON with task_id + poll URL
      - Poll with same URL + task_id=X or via /api/v1/segment/progress?task_id=X
      - When done: returns the file (gpkg/kml) or JSON, plus auto_share_id
      - Add &task_id=X to check/download a previously started task

    The result is auto-saved as a share for later access.
    Processing is queued (max 2 concurrent, 4 in queue) to prevent overload.

    Timing estimates (< 1 km², ortho=true):
      ~30-60s with ortho only, ~60-90s with ortho+temporal,
      ~90-120s with all sources. GPKG/KML adds ~5-10s.
    """
    try:
        bbox_str = request.args.get('bbox', '')
        task_id = request.args.get('task_id', '')
        fmt = request.args.get('format', 'json').lower().strip()

        # If task_id provided, check its status
        if task_id:
            return _onestop_check(task_id, fmt, dict(request.args))

        # Parse bbox
        if not bbox_str:
            return _error('bbox parameter required (lon_min,lat_min,lon_max,lat_max)', 400)
        parts = [x.strip() for x in bbox_str.split(',')]
        if len(parts) != 4:
            return _error('bbox must have 4 values: lon_min,lat_min,lon_max,lat_max', 400)
        try:
            lon_min, lat_min, lon_max, lat_max = [float(x) for x in parts]
        except ValueError:
            return _error('bbox values must be numbers', 400)

        # Build polygon from bbox
        from shapely.geometry import box as shapely_box
        geom = shapely_box(lon_min, lat_min, lon_max, lat_max)
        features = [{'geometry': geom}]
        geo_parse.validate_austria_bounds(geom)
        geom_3035 = ti.geometry_to_3035(geom)
        _validate_area(geom_3035)

        # Build params
        params = {}
        for key in ('dataset', 'min_object_size', 'include_ortho', 'include_temporal',
                    'include_copernicus', 'include_cadastre', 'include_hansen', 'include_infra',
                    'types', 'groups', 'felz_scale', 'rag_threshold',
                    'height_min', 'height_max', 'height_op',
                    'top_n_classes', 'top_n_objects', 'min_height_m'):
            val = request.args.get(key)
            if val is not None:
                params[key] = val
        # Sensible defaults for one-stop
        params.setdefault('min_object_size', '10')
        params.setdefault('include_ortho', 'true')

        geometry_text = json.dumps(mapping(geom))

        # Queue check
        with _TASK_QUEUE_LOCK:
            if _TASK_QUEUE_SIZE >= MAX_QUEUE_SIZE:
                return jsonify({
                    'error': 'Server busy — too many tasks queued. Try again in a few minutes.',
                    'queue_size': _TASK_QUEUE_SIZE,
                    'retry_after_seconds': 120,
                }), 503

        # Start async task
        task_id = str(uuid.uuid4())
        _progress_start(task_id)
        _cleanup_old_results()

        # Store onestop params so we can produce the right format on completion
        onestop_meta = {
            'format': fmt,
            'params': {k: request.args.get(k) for k in request.args if k != 'task_id'},
        }
        (_PROGRESS_DIR / f"{task_id}.onestop.json").write_text(
            json.dumps(onestop_meta))

        thread = threading.Thread(
            target=_segment_worker,
            args=(task_id, features, params, geometry_text),
            daemon=True,
        )
        thread.start()

        est_seconds = _estimate_time(geom_3035, params)

        # Browser: return auto-polling HTML page
        if _wants_html():
            return _onestop_poll_page(task_id, fmt, step='queued',
                                     detail='Starting analysis…', est=est_seconds)

        host = request.headers.get('X-Forwarded-Host', request.host)
        proto = request.headers.get('X-Forwarded-Proto', request.scheme)
        base = f"{proto}://{host}"
        poll_url = f"{base}/api/v1/onestop?task_id={task_id}&format={fmt}"
        progress_url = f"{base}/api/v1/segment/progress?task_id={task_id}"

        return jsonify({
            'task_id': task_id,
            'status': 'running',
            'queue_size': _TASK_QUEUE_SIZE,
            'poll_url': poll_url,
            'progress_url': progress_url,
            'estimated_seconds': est_seconds,
            'message': 'Analysis started. Poll poll_url until status=done, then the final response will contain the download.',
        }), 202

    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error("onestop: %s", traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


def _estimate_time(geom_3035, params: dict) -> int:
    """Rough estimate of processing time in seconds for a given area + options."""
    area_sqm = geom_3035.area
    area_ha = area_sqm / 10000
    base = 20 + area_ha * 0.5  # ~20s base + 0.5s per hectare
    if str(params.get('include_ortho', 'true')).lower() in ('true', '1', 'yes'):
        base += 10 + area_ha * 0.3
    if str(params.get('include_temporal', 'false')).lower() in ('true', '1', 'yes'):
        base += 15 + area_ha * 0.4
    if str(params.get('include_copernicus', 'false')).lower() in ('true', '1', 'yes'):
        base += 20
    if str(params.get('include_cadastre', 'false')).lower() in ('true', '1', 'yes'):
        base += 5
    if str(params.get('include_hansen', 'false')).lower() in ('true', '1', 'yes'):
        base += 10
    return int(base)


def _wants_html():
    """Return True if the client prefers HTML (i.e. a browser)."""
    accept = request.headers.get('Accept', '')
    # Browsers send text/html first; curl/programmatic clients send */* or application/json
    return 'text/html' in accept and 'application/json' not in accept


def _onestop_poll_page(task_id: str, fmt: str, step: str = '', detail: str = '',
                       elapsed: int = 0, est: int = 0):
    """Return an HTML page that auto-polls onestop and redirects to the download."""
    host = request.headers.get('X-Forwarded-Host', request.host)
    proto = request.headers.get('X-Forwarded-Proto', request.scheme)
    base = f"{proto}://{host}"
    progress_url = f"{base}/api/v1/segment/progress?task_id={task_id}"
    download_url = f"{base}/api/v1/onestop?task_id={task_id}&format={fmt}"
    pct = min(95, int(elapsed / max(est, 1) * 100)) if est else 0
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Processing…</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         display: flex; justify-content: center; align-items: center; min-height: 100vh;
         margin: 0; background: #f5f7fa; color: #333; }}
  .card {{ background: #fff; border-radius: 12px; padding: 40px 48px; text-align: center;
           box-shadow: 0 2px 12px rgba(0,0,0,.08); max-width: 480px; width: 90%; }}
  h2 {{ margin: 0 0 8px; font-size: 20px; }}
  .sub {{ color: #888; font-size: 14px; margin-bottom: 24px; }}
  .bar-bg {{ background: #e9ecef; border-radius: 8px; height: 8px; overflow: hidden; margin: 16px 0; }}
  .bar {{ background: linear-gradient(90deg, #4361ee, #3a86ff); height: 100%; border-radius: 8px;
          transition: width .5s ease; }}
  .status {{ font-size: 14px; color: #555; min-height: 20px; }}
  .elapsed {{ font-size: 12px; color: #aaa; margin-top: 8px; }}
  .err {{ color: #dc3545; font-weight: 600; }}
</style>
</head><body><div class="card">
  <h2>⚙️ Processing</h2>
  <p class="sub">Your analysis is running. This page will auto-download when ready.</p>
  <div class="bar-bg"><div class="bar" id="bar" style="width:{pct}%"></div></div>
  <div class="status" id="status">{_html_esc(step)}: {_html_esc(detail)}</div>
  <div class="elapsed" id="elapsed">{elapsed}s elapsed{f' / ~{est}s est.' if est else ''}</div>
</div>
<script>
const progressUrl = "{progress_url}";
const downloadUrl = "{download_url}";
const est = {est};
let t0 = Date.now() - {elapsed * 1000};

async function poll() {{
  try {{
    const r = await fetch(progressUrl);
    const d = await r.json();
    const el = Math.round((Date.now() - t0) / 1000);
    document.getElementById('elapsed').textContent = el + 's elapsed' + (est ? ' / ~' + est + 's est.' : '');
    if (d.error) {{
      document.getElementById('status').innerHTML = '<span class="err">❌ ' + d.error + '</span>';
      document.getElementById('bar').style.background = '#dc3545';
      return;
    }}
    if (d.done) {{
      document.getElementById('bar').style.width = '100%';
      document.getElementById('status').textContent = 'Downloading…';
      window.location.href = downloadUrl;
      setTimeout(function() {{
        document.querySelector('h2').textContent = '✅ Done';
        document.getElementById('status').textContent = 'Download started. You can close this page.';
        document.querySelector('.sub').textContent = el + 's total processing time.';
      }}, 2000);
      return;
    }}
    const pct = Math.min(95, est ? Math.round(el / est * 100) : 50);
    document.getElementById('bar').style.width = pct + '%';
    document.getElementById('status').textContent = (d.step || '') + (d.detail ? ': ' + d.detail : '');
  }} catch(e) {{ /* retry */ }}
  setTimeout(poll, 3000);
}}
setTimeout(poll, 2000);
</script></body></html>"""
    return html, 200, {'Content-Type': 'text/html; charset=utf-8'}


def _html_esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def _onestop_check(task_id: str, fmt: str, url_params: dict):
    """Check status of a one-stop task. Return result when done."""
    if not task_id or not re.match(r'^[a-f0-9\-]+$', task_id):
        return _error('Invalid task_id', 400)

    p = _PROGRESS_DIR / f"{task_id}.json"
    if not p.exists():
        return _error('Task not found', 404)

    progress = json.loads(p.read_text())
    t0 = progress.get('t0', 0)
    elapsed = round(time.time() - t0) if t0 else 0

    if progress.get('step') == 'error':
        return jsonify({
            'task_id': task_id,
            'status': 'error',
            'error': progress.get('detail', 'Unknown error'),
            'elapsed_seconds': elapsed,
        }), 500

    host = request.headers.get('X-Forwarded-Host', request.host)
    proto = request.headers.get('X-Forwarded-Proto', request.scheme)
    base = f"{proto}://{host}"

    if progress.get('step') != 'done':
        if _wants_html():
            return _onestop_poll_page(task_id, fmt,
                                     step=progress.get('step', ''),
                                     detail=progress.get('detail', ''),
                                     elapsed=elapsed)
        return jsonify({
            'task_id': task_id,
            'status': 'running',
            'step': progress.get('step', ''),
            'detail': progress.get('detail', ''),
            'elapsed_seconds': elapsed,
            'poll_url': f"{base}/api/v1/onestop?task_id={task_id}&format={fmt}",
        }), 202

    # Done! Return result in requested format
    auto_share_id = progress.get('auto_share_id')
    result = _get_result(task_id)
    if not result:
        if auto_share_id:
            # Load from auto-save share
            share_path = SHARE_DIR / f"{auto_share_id}.json.gz"
            if share_path.exists():
                share_data = json.loads(gzip.decompress(share_path.read_bytes()))
                result = share_data.get('result')
        if not result:
            return _error('Result expired. Re-run the analysis.', 410)

    share_url = f"{base}/?share={auto_share_id}" if auto_share_id else None

    if fmt == 'json':
        result['_onestop'] = {
            'task_id': task_id,
            'elapsed_seconds': elapsed,
            'auto_share_id': auto_share_id,
            'share_url': share_url,
        }
        return jsonify(result)

    # For gpkg/kml, we need to build the export from the result
    obj_features = result.get('features', [])

    # Apply type filter
    type_filter_str = url_params.get('types')
    if type_filter_str:
        type_set = set(t.strip() for t in type_filter_str.split(','))
        obj_features = [f for f in obj_features
                       if f.get('properties', {}).get('type') in type_set]

    # Apply height filter
    obj_features = _apply_height_filter_features(obj_features, url_params)

    if fmt == 'kml':
        group_by = url_params.get('group_by', 'type')
        seg_geom = url_params.get('segment_geometry', 'point').lower().strip()

        kml_features = obj_features
        if seg_geom == 'polygon':
            # Vectorise segment raster into polygon features
            try:
                onestop_meta_path = _PROGRESS_DIR / f"{task_id}.onestop.json"
                meta_params = {}
                if onestop_meta_path.exists():
                    meta_params = json.loads(onestop_meta_path.read_text()).get('params', {})
                bbox_str = meta_params.get('bbox', url_params.get('bbox', ''))
                parts = [float(x.strip()) for x in bbox_str.split(',')]
                lon_min, lat_min, lon_max, lat_max = parts
                from shapely.geometry import box as shapely_box
                geom_wgs = shapely_box(lon_min, lat_min, lon_max, lat_max)
                dataset = meta_params.get('dataset', ti.DEFAULT_DATASET)
                type_filter = set(t.strip() for t in url_params['types'].split(',')) if url_params.get('types') else None
                poly_features = _vectorise_segments_to_geojson(
                    geom_wgs, dataset, type_filter=type_filter,
                    height_params=url_params)
                if poly_features:
                    kml_features = poly_features
                else:
                    log.warning('onestop kml polygon: vectorisation returned 0 features, falling back to points')
            except Exception as e:
                log.warning('onestop kml polygon fallback to points: %s', e)

        style_mode = url_params.get('segment_geometry_style', 'type').lower().strip()
        kml = _build_kml(kml_features, group_by, style_mode=style_mode)
        tmp = tempfile.NamedTemporaryFile(suffix='.kml', delete=False, mode='w', encoding='utf-8')
        tmp.write(kml)
        tmp.close()
        resp = send_file(tmp.name, mimetype='application/vnd.google-earth.kml+xml',
                        as_attachment=True, download_name='onestop_export.kml')
        resp.headers['X-Auto-Share-Id'] = auto_share_id or ''
        resp.headers['X-Elapsed-Seconds'] = str(elapsed)
        return resp

    if fmt == 'gpkg':
        # Re-run GPKG export using the segment result's geometry
        try:
            onestop_meta_path = _PROGRESS_DIR / f"{task_id}.onestop.json"
            meta_params = {}
            if onestop_meta_path.exists():
                meta_params = json.loads(onestop_meta_path.read_text()).get('params', {})

            bbox_str = meta_params.get('bbox', url_params.get('bbox', ''))
            parts = [float(x.strip()) for x in bbox_str.split(',')]
            lon_min, lat_min, lon_max, lat_max = parts
            from shapely.geometry import box as shapely_box
            geom = shapely_box(lon_min, lat_min, lon_max, lat_max)
            features = [{'geometry': geom}]

            # Merge: meta_params as defaults, url_params as overrides
            def _mp(key, default=''):
                return url_params.get(key) or meta_params.get(key, default)

            # Map segment_geometry / segment_geometry_style to GPKG params
            seg_geom = _mp('segment_geometry', 'polygon').lower().strip()
            seg_style = _mp('segment_geometry_style', '') or _mp('color_mode', 'type')

            gpkg_params = {
                'include_segments': 'true',
                'include_segments_vector': _mp('include_segments_vector',
                    'false' if seg_geom == 'point' else 'true'),
                'include_dtm': _mp('include_dtm', 'false'),
                'types': _mp('types'),
                'height_min': _mp('height_min'),
                'height_max': _mp('height_max'),
                'height_op': _mp('height_op'),
                'color_mode': seg_style if seg_style in ('type', 'height') else 'type',
            }
            # Override layers param
            layers_str = _mp('layers', 'segments')
            if layers_str and layers_str != 'segments':
                layer_set = set(l.strip() for l in layers_str.split(',') if l.strip())
                _resolve_layer_set(layer_set, gpkg_params)
            else:
                gpkg_params['include_segments'] = 'true'
                if seg_geom != 'point':
                    gpkg_params['include_segments_vector'] = 'true'

            tmp_path, table_count, gpkg_elapsed = _gpkg_core(features, gpkg_params)
            if table_count == 0:
                return _error('No layers produced')

            resp = send_file(tmp_path, mimetype='application/geopackage+sqlite3',
                            as_attachment=True, download_name='onestop_export.gpkg')
            resp.headers['X-Auto-Share-Id'] = auto_share_id or ''
            resp.headers['X-Elapsed-Seconds'] = str(elapsed)
            return resp
        except Exception as e:
            log.error("onestop gpkg: %s", traceback.format_exc())
            return _error(f"GPKG export failed: {e}", 500)

    return _error(f"Unknown format: {fmt}. Use json, gpkg, or kml.", 400)


@app.route('/docs')
def docs_page():
    return send_from_directory('static', 'docs.html')

@app.route('/api/v1/docs/llm.txt', methods=['GET'])
def llm_docs():
    p = Path(__file__).parent / 'llm.txt'
    if p.exists():
        return Response(p.read_text(), mimetype='text/plain')
    return Response("Documentation not yet generated.", mimetype='text/plain')


# === SECTION: /api/v1/parse-geometry endpoint ===

@app.route('/api/v1/parse-geometry', methods=['POST'])
def parse_geometry_file():
    """Parse an uploaded geometry file (Shapefile ZIP, GeoPackage, GeoJSON, KML, GPX, WKT, etc).
    Returns GeoJSON FeatureCollection with all features.
    """
    import fiona
    import fiona.io
    import zipfile
    from shapely.geometry import shape as shp_shape, mapping as shp_mapping
    from shapely.ops import unary_union
    try:
        if 'file' not in request.files:
            return _error('No file uploaded', 400)
        f = request.files['file']
        fname = (f.filename or '').lower()
        raw = f.read()
        if not raw:
            return _error('Empty file', 400)

        # Auto-decompress gzipped uploads (client may gzip text files)
        if fname.endswith('.gz') or (len(raw) >= 2 and raw[:2] == b'\x1f\x8b'):
            import gzip as _gzip
            try:
                raw = _gzip.decompress(raw)
                if fname.endswith('.gz'):
                    fname = fname[:-3]
            except Exception:
                pass  # not actually gzipped, use raw

        features = []

        # Content-sniff: detect KML/GeoJSON/WKT regardless of file extension
        text_raw = None
        try:
            text_raw = raw.decode('utf-8', errors='replace')
        except Exception:
            pass

        is_kml = fname.endswith(('.kml', '.xml'))
        is_json = fname.endswith(('.geojson', '.json'))
        is_wkt = fname.endswith('.wkt')

        # Sniff content for unknown extensions (e.g. .txt)
        if text_raw and not (is_kml or is_json or is_wkt):
            stripped = text_raw.strip()[:500]
            if '<?xml' in stripped or '<kml' in stripped.lower():
                is_kml = True
            elif stripped.startswith('{') and '"type"' in stripped:
                is_json = True
            elif re.match(r'^(POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON|GEOMETRYCOLLECTION)\s*\(', stripped, re.IGNORECASE):
                is_wkt = True

        # Try text-based formats first
        if is_json:
            gj = json.loads(raw)
            return jsonify(gj if gj.get('type') == 'FeatureCollection' else {
                'type': 'FeatureCollection',
                'features': gj.get('features', [{'type': 'Feature', 'geometry': gj, 'properties': {}}])
            })

        if is_kml:
            parsed = geo_parse.parse_input(text_raw or raw.decode('utf-8', errors='replace'))
            # Convert non-polygon geometries (lines from road boundaries etc)
            for feat in parsed:
                geom = feat['geometry']
                if geom.geom_type not in ('Polygon', 'MultiPolygon'):
                    feat['geometry'] = _non_polygon_to_polygon(geom)
            return jsonify(geo_parse.features_to_geojson(parsed))

        if is_wkt:
            from shapely import wkt
            geom = wkt.loads(raw.decode('utf-8', errors='replace').strip())
            return jsonify({'type': 'FeatureCollection', 'features': [
                {'type': 'Feature', 'geometry': shp_mapping(geom), 'properties': {}}
            ]})

        # Binary formats via fiona
        tmp_dir = tempfile.mkdtemp(prefix='geo_upload_')
        try:
            # Shapefile in ZIP
            if fname.endswith('.zip'):
                zpath = os.path.join(tmp_dir, 'upload.zip')
                with open(zpath, 'wb') as wf:
                    wf.write(raw)
                # Find .shp inside zip
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(tmp_dir)
                # Find shp file
                shp_files = []
                for root, dirs, files in os.walk(tmp_dir):
                    for fn in files:
                        if fn.lower().endswith('.shp'):
                            shp_files.append(os.path.join(root, fn))
                if not shp_files:
                    # Try as GPKG or other format inside zip
                    for root, dirs, files in os.walk(tmp_dir):
                        for fn in files:
                            if fn.lower().endswith(('.gpkg', '.geojson', '.json', '.kml', '.gpx')):
                                shp_files.append(os.path.join(root, fn))
                if not shp_files:
                    return _error('No shapefile (.shp) or supported file found in ZIP', 400)
                src_path = shp_files[0]
            else:
                # Single file (gpkg, gpx, shp, etc)
                ext = os.path.splitext(fname)[1] or '.gpkg'
                src_path = os.path.join(tmp_dir, 'upload' + ext)
                with open(src_path, 'wb') as wf:
                    wf.write(raw)

            with fiona.open(src_path) as src:
                src_crs = src.crs
                # Reproject to WGS84 if needed
                need_reproject = False
                if src_crs and str(src_crs).upper() not in ('EPSG:4326', '{"INIT": "EPSG:4326"}'):
                    try:
                        from pyproj import CRS, Transformer
                        c = CRS(src_crs)
                        if c.to_epsg() != 4326:
                            need_reproject = True
                            transformer = Transformer.from_crs(c, CRS.from_epsg(4326), always_xy=True)
                    except Exception:
                        pass

                for feat in src:
                    geom = shp_shape(feat['geometry'])
                    if need_reproject:
                        from shapely.ops import transform
                        geom = transform(transformer.transform, geom)
                    props = dict(feat.get('properties', {}))
                    # Convert non-serializable values
                    for k, v in list(props.items()):
                        if v is not None and not isinstance(v, (str, int, float, bool)):
                            props[k] = str(v)
                    features.append({
                        'type': 'Feature',
                        'geometry': shp_mapping(geom),
                        'properties': props,
                    })
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if not features:
            return _error('No features found in file', 400)

        log.info("parse-geometry: %s → %d features", fname, len(features))
        return jsonify({'type': 'FeatureCollection', 'features': features})

    except Exception as e:
        log.error("parse-geometry: %s", traceback.format_exc())
        return _error(f'Failed to parse file: {e}')


# === SECTION: /api/v1/share endpoints (save/load/rename/list) ===

SHARE_DIR = Path('data/shares')
SHARE_DIR.mkdir(parents=True, exist_ok=True)
SHARE_MAX_BYTES = 2_000_000_000  # 2 GB

_SHARE_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,80}$')  # hex hashes or named slugs
def _valid_share_id(s):
    return bool(s and _SHARE_ID_RE.match(s))

def _resolve_share(share_id):
    """Resolve a share ID, following redirect stubs from renames.
    Returns (resolved_id, path) or (None, None) if not found."""
    p = SHARE_DIR / f'{share_id}.json.gz'
    if not p.exists():
        return None, None
    # Check for redirect alias (small file left after rename)
    if p.stat().st_size < 200:
        try:
            stub = json.loads(gzip.decompress(p.read_bytes()))
            if 'redirect' in stub:
                new_id = stub['redirect']
                new_p = SHARE_DIR / f'{new_id}.json.gz'
                if new_p.exists():
                    return new_id, new_p
                return None, None
        except Exception:
            pass
    return share_id, p


def _share_eviction_tier(path: Path) -> int:
    """Classify a share into eviction tiers (lower = evicted first).

    Tier 0: unnamed auto-saves and hex-hash shares (evict first)
    Tier 1: redirect stubs (tiny, but keep them to preserve old links)
    Tier 2: shares with a user-set name inside the payload (e.g. name='Westbahnhof')
    Tier 3: shares with a user-renamed ID (e.g. 'Wienwest', 'Kohlschwarz90') — evict last
    """
    try:
        share_id = path.stem.replace('.json', '')
        sz = path.stat().st_size
        id_is_hex = bool(re.fullmatch(r'[0-9a-f]{12}', share_id))
        id_is_auto = share_id.startswith('auto-')

        # Redirect stubs: small files that alias old IDs → new named share.
        # Protect them (tier 1) so old links keep working.
        if sz < 200:
            try:
                stub = json.loads(gzip.decompress(path.read_bytes()))
                if 'redirect' in stub:
                    return 1
            except Exception:
                pass
            # Other tiny files (empty state etc.) are low-value
            return 0

        # User-renamed ID: the share filename is a human-readable slug.
        # These are the most valuable — user deliberately chose the name.
        if not id_is_hex and not id_is_auto:
            return 3

        # Check the name field inside the payload (only first ~2KB for speed)
        # Shares that have a name set but still have hex/auto IDs (user set name
        # but didn't rename the ID).
        try:
            raw = gzip.decompress(path.read_bytes())
            data = json.loads(raw)
            name = data.get('name', '')
            if name and name != share_id and not name.startswith('Auto-save '):
                return 2
        except Exception:
            pass

        return 0
    except Exception:
        return 0


MAX_AUTO_SAVES = 15  # keep at most this many tier-0 auto-saves

def _share_evict():
    """Remove shares until total size < SHARE_MAX_BYTES.

    Also caps the number of tier-0 (auto-save / hex-hash) shares to
    MAX_AUTO_SAVES so they don't crowd out named shares in the UI.

    Eviction tiers (lower evicted first):
      0: unnamed auto-saves and hex-hash shares
      1: redirect stubs (preserve old links)
      2: shares with user-set name field
      3: shares with user-renamed ID slug (most protected)
    Within each tier, oldest (by mtime) evicted first.
    """
    all_files = list(SHARE_DIR.glob('*.json.gz'))
    total = sum(f.stat().st_size for f in all_files)

    # --- Phase 1: cap tier-0 auto-saves to MAX_AUTO_SAVES ---
    tier0 = sorted(
        [f for f in all_files if _share_eviction_tier(f) == 0],
        key=lambda f: f.stat().st_mtime, reverse=True  # newest first
    )
    for victim in tier0[MAX_AUTO_SAVES:]:
        total -= victim.stat().st_size
        victim.unlink(missing_ok=True)
        all_files.remove(victim)
        log.info("share: evicted excess auto-save %s (total now %d MB)", victim.name, total // 1_000_000)

    # --- Phase 2: size-based eviction ---
    if total <= SHARE_MAX_BYTES:
        return
    # Sort by (tier, mtime) — lowest tier and oldest files evicted first
    evict_order = sorted(all_files, key=lambda f: (_share_eviction_tier(f), f.stat().st_mtime))
    while total > SHARE_MAX_BYTES and evict_order:
        victim = evict_order.pop(0)
        tier = _share_eviction_tier(victim)
        total -= victim.stat().st_size
        victim.unlink(missing_ok=True)
        log.info("share: evicted %s (tier %d, total now %d MB)", victim.name, tier, total // 1_000_000)


@app.route('/api/v1/shares', methods=['GET'])
def share_list():
    """List saved shares with metadata.

    Ordering: user-renamed shares first (tier 3), then named (tier 2),
    then auto-saves/hex (tier 0).  Within each group, most recent first.
    This ensures manually-named shares are always visible regardless of
    how many auto-saves accumulate.
    """
    try:
        all_files = list(SHARE_DIR.glob('*.json.gz'))
        # Sort by (-tier, -mtime) so high-tier (named) shares come first
        files = sorted(all_files, key=lambda f: (-_share_eviction_tier(f), -f.stat().st_mtime))
        limit = int(request.args.get('limit', 30))
        items = []
        for f in files[:limit]:
            share_id = f.stem.replace('.json', '')
            try:
                data = json.loads(gzip.decompress(f.read_bytes()))
                # Skip redirect stubs (left after rename for old-link compat)
                if 'redirect' in data and 'state' not in data:
                    continue
                state = data.get('state', {})
                stored_name = data.get('name', '')
                # Prefer share ID as display name if it's a human-readable slug
                # (i.e. user-renamed, not a hex hash or auto- prefix)
                id_is_hex = bool(re.fullmatch(r'[0-9a-f]{12}', share_id))
                id_is_auto = share_id.startswith('auto-')
                if not id_is_hex and not id_is_auto:
                    name = share_id  # user-renamed ID IS the name
                else:
                    name = stored_name or share_id
                endpoint = state.get('endpoint', '')
                has_result = 'result' in data and bool(data['result'])
                has_geometry = bool(state.get('geometry', ''))
                # Build a short description
                tags = []
                if endpoint:
                    tags.append(endpoint)
                if has_result:
                    # Try to get object count from result
                    result = data.get('result', {})
                    n_features = len(result.get('features', []))
                    if n_features:
                        tags.append(f"{n_features} objects")
                elif has_geometry:
                    tags.append('geometry only')
                is_onestop = 'onestop' in data
                items.append(dict(
                    id=share_id,
                    name=name,
                    description=', '.join(tags) if tags else '',
                    has_result=has_result,
                    has_geometry=has_geometry,
                    endpoint=endpoint,
                    updated=f.stat().st_mtime,
                    size_kb=round(f.stat().st_size / 1024, 1),
                    onestop=is_onestop,
                ))
            except Exception:
                items.append(dict(id=share_id, name=share_id, description='', has_result=False,
                                  has_geometry=False, endpoint='', updated=f.stat().st_mtime, size_kb=0))
        return jsonify(items)
    except Exception as e:
        return _error(str(e))


@app.route('/api/v1/share', methods=['POST'])
def share_save():
    """Save analysis result + UI state for sharing. Returns {id, url}.
    
    Content-hash dedup: if payload matches an existing share, reuse its ID.
    Client can also send {reuse_id: "abc123"} to update an existing share in-place.
    Overlays (base64 PNG images) are included in the stored payload for instant restore.
    """
    try:
        import hashlib
        payload = request.get_json(force=True)
        if not payload:
            return _error('Empty payload')
        
        host = request.headers.get('X-Forwarded-Host', request.host)
        proto = request.headers.get('X-Forwarded-Proto', 'https')
        
        # Extract reuse_id (not stored)
        reuse_id = payload.pop('reuse_id', None)
        
        # Hash based on state+result only (exclude overlays and name — they're metadata)
        hash_payload = {k: v for k, v in payload.items() if k not in ('overlays', 'name')}
        hash_json = json.dumps(hash_payload, separators=(',', ':'), sort_keys=True)
        content_hash = hashlib.sha256(hash_json.encode()).hexdigest()[:24]
        
        # Full payload including overlays for storage
        data_json = json.dumps(payload, separators=(',', ':'), sort_keys=True)
        data = gzip.compress(data_json.encode())
        
        # Check if client wants to reuse an existing share ID (update in-place)
        if reuse_id and _valid_share_id(reuse_id):
            existing = SHARE_DIR / f'{reuse_id}.json.gz'
            if existing.exists():
                # Merge: keep existing result/overlays if not provided in new payload
                try:
                    existing_obj = json.loads(gzip.decompress(existing.read_bytes()).decode())
                except Exception:
                    existing_obj = {}
                if 'result' not in payload and 'result' in existing_obj:
                    payload['result'] = existing_obj['result']
                if 'overlays' not in payload and 'overlays' in existing_obj:
                    payload['overlays'] = existing_obj['overlays']
                if 'name' not in payload and 'name' in existing_obj:
                    payload['name'] = existing_obj['name']
                merged_json = json.dumps(payload, separators=(',', ':'), sort_keys=True)
                existing.write_bytes(gzip.compress(merged_json.encode()))
                existing.touch()
                url = f'{proto}://{host}/?share={reuse_id}'
                ovl_count = len(payload.get('overlays', {}))
                log.info("share: updated existing %s (%d KB, %d overlays)",
                         reuse_id, len(gzip.compress(merged_json.encode())) // 1024, ovl_count)
                return jsonify({'id': reuse_id, 'url': url, 'reused': True})
        
        # Content-hash dedup: check existing shares (state+result match)
        # If overlays differ but state+result same, update overlays in existing share
        for existing_file in SHARE_DIR.glob('*.json.gz'):
            try:
                existing_data = gzip.decompress(existing_file.read_bytes()).decode()
                existing_obj = json.loads(existing_data)
                existing_hash_payload = {k: v for k, v in existing_obj.items() if k not in ('overlays', 'name')}
                existing_hash = hashlib.sha256(
                    json.dumps(existing_hash_payload, separators=(',', ':'), sort_keys=True).encode()
                ).hexdigest()[:24]
                if existing_hash == content_hash:
                    share_id = existing_file.stem.split('.')[0]
                    # If new payload has overlays/name changes, update stored data
                    new_ovl = payload.get('overlays', {})
                    old_ovl = existing_obj.get('overlays', {})
                    new_name = payload.get('name')
                    old_name = existing_obj.get('name')
                    if len(new_ovl) > len(old_ovl) or new_name != old_name:
                        existing_file.write_bytes(data)
                        log.info("share: dedup hit %s, updated (overlays %d→%d, name=%s)",
                                 share_id, len(old_ovl), len(new_ovl), new_name)
                    else:
                        log.info("share: dedup hit %s", share_id)
                    existing_file.touch()
                    url = f'{proto}://{host}/?share={share_id}'
                    return jsonify({'id': share_id, 'url': url, 'reused': True, 'name': new_name})
            except Exception:
                continue
        
        # New share
        share_id = uuid.uuid4().hex[:12]
        (SHARE_DIR / f'{share_id}.json.gz').write_bytes(data)
        _share_evict()
        ovl_count = len(payload.get('overlays', {}))
        url = f'{proto}://{host}/?share={share_id}'
        log.info("share: saved %s (%d KB, %d overlays)", share_id, len(data) // 1024, ovl_count)
        return jsonify({'id': share_id, 'url': url, 'reused': False})
    except Exception as e:
        log.error("share save: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/api/v1/share/<share_id>', methods=['GET'])
def share_load(share_id):
    """Retrieve a saved share. Serves gzip-compressed if client accepts it.
    
    Follows redirect aliases: if a share was renamed, the old ID contains
    a small {"redirect": "new_id"} stub that points to the new location.
    """
    try:
        if not _valid_share_id(share_id):
            return _error('Invalid share ID', 400)
        share_id, p = _resolve_share(share_id)
        if not p or not p.exists():
            return _error('Share not found', 404)
        raw = p.read_bytes()
        # Touch file to keep it alive (LRU)
        p.touch()
        # Header to let the client know the resolved ID (may differ if redirect followed)
        extra_headers = {'X-Share-Id': share_id}
        # Serve pre-compressed gzip if client accepts it (saves bandwidth, critical for mobile)
        accept_enc = request.headers.get('Accept-Encoding', '')
        if 'gzip' in accept_enc:
            return Response(raw, mimetype='application/json',
                            headers={'Content-Encoding': 'gzip',
                                     'Vary': 'Accept-Encoding',
                                     **extra_headers})
        data = gzip.decompress(raw)
        return Response(data, mimetype='application/json', headers=extra_headers)
    except Exception as e:
        log.error("share load: %s", traceback.format_exc())
        return _error(str(e))


def _resolve_layer_set(layer_set: set, params: dict, state: dict = None):
    """Translate UI layer IDs into _gpkg_core params.

    Layer IDs match the UI data-lyr attributes:
      dtm, segments, raster, hansen,
      ortho-YYYY, cir-YYYY, dtm-YYYY, dsm-YYYY
    """
    if 'dtm' in layer_set:
        params['include_dtm'] = 'true'
        layer_set.discard('dtm')
    if 'segments' in layer_set:
        params['include_segments'] = 'true'
        layer_set.discard('segments')
    layer_set.discard('base')  # basemap tile layer, not exportable

    # ortho-YYYY → ortho_years (raw RGBI raster)
    ortho_yrs = []
    for lid in list(layer_set):
        if lid.startswith('ortho-') and lid[6:].isdigit():
            ortho_yrs.append(lid[6:])
            layer_set.discard(lid)
    if ortho_yrs:
        params['ortho_years'] = ','.join(sorted(ortho_yrs))

    # Everything else (raster, hansen, dtm-YYYY, dsm-YYYY, cir-YYYY)
    # goes to raster_layers for RGBA overlay rendering
    if layer_set:
        params['raster_layers'] = ','.join(sorted(layer_set))


@app.route('/api/v1/share/<share_id>/download.gpkg', methods=['GET'])
def share_download_gpkg(share_id):
    """Direct GeoPackage download from a share — usable as QGIS data source URL.

    Reads the share's geometry, runs GPKG export.

    Query params:
      layers=all         Include DTM/DSM/nDSM + ortho + segments (default)
      layers=active      Only layers that were active in the share's UI state
      layers=dtm,ortho-2024,segments,raster,hansen  Comma-separated layer IDs:
                         dtm        → DTM + DSM + nDSM (raw 1m float32)
                         segments   → Segment type + height rasters
                         ortho-YYYY → Orthophoto RGBI for that year
                         cir-YYYY   → CIR false-colour for that year
                         raster     → Coloured segment overlay (RGBA)
                         hansen     → Hansen forest change overlay (RGBA)
                         dtm-YYYY   → DTM hillshade overlay (RGBA)
                         dsm-YYYY   → DSM hillshade overlay (RGBA)
      types=tree,road    Filter segment types
      color_mode=type    Segment colour mode (type or height)
    """
    try:
        if not _valid_share_id(share_id):
            return _error('Invalid share ID', 400)
        share_id, p = _resolve_share(share_id)
        if not p or not p.exists():
            return _error('Share not found', 404)
        share_data = json.loads(gzip.decompress(p.read_bytes()).decode())
        state = share_data.get('state', {})
        geom_str = state.get('geometry', '')
        if not geom_str:
            return _error('Share has no geometry', 400)

        # Parse geometry
        features = geo_parse.parse_input(geom_str)
        features = geo_parse.union_features(features)
        for feat in features:
            geom = feat['geometry']
            if geom.geom_type not in ('Polygon', 'MultiPolygon'):
                feat['geometry'] = _non_polygon_to_polygon(geom)

        # --- Resolve layers param ---
        layers_raw = request.args.get('layers', 'all').strip().lower()
        params = {}

        if layers_raw == 'all':
            params['include_dtm'] = 'true'
            params['include_segments'] = 'true'
            params['ortho_years'] = str(ti.dataset_to_year(ti.DEFAULT_DATASET))

        elif layers_raw == 'active':
            # Use only layers the user had enabled in the share's UI
            active = set(state.get('layers', []))
            _resolve_layer_set(active, params, state)

        else:
            # Explicit comma-separated list
            layer_set = set(l.strip() for l in layers_raw.split(',') if l.strip())
            _resolve_layer_set(layer_set, params, state)

        # Allow extra query-string overrides
        for key in ('types', 'color_mode', 'dataset', 'height_min', 'height_max', 'height_op'):
            val = request.args.get(key)
            if val is not None:
                params[key] = val

        tmp_path, table_count, elapsed = _gpkg_core(features, params)
        if table_count == 0:
            return _error('No layers produced')
        log.info("share GPKG download %s: %d tables, %.1fs, layers=%s",
                 share_id, table_count, elapsed, layers_raw)
        return send_file(
            tmp_path, mimetype='application/geopackage+sqlite3',
            as_attachment=True, download_name=f'{share_id}.gpkg',
        )
    except ValueError as e:
        return _error(str(e))
    except Exception as e:
        log.error("share gpkg download: %s", traceback.format_exc())
        return _error(f"Internal error: {e}", 500)


@app.route('/api/v1/share/<old_id>/rename', methods=['POST'])
def share_rename(old_id):
    """Rename a share: move file from old_id to new slug.
    Body: {"new_id": "MySlug"}
    Returns: {"id": "MySlug", "old_id": "abc123def456"}
    """
    try:
        if not _valid_share_id(old_id):
            return _error('Invalid share ID', 400)
        body = request.get_json(force=True)
        new_id = (body.get('new_id') or '').strip()
        if not new_id:
            return _error('new_id required', 400)
        # Sanitise: replace spaces/special chars with hyphens
        new_id = re.sub(r'[^A-Za-z0-9_-]+', '-', new_id).strip('-')[:80]
        if not _valid_share_id(new_id):
            return _error('Invalid new ID', 400)
        old_path = SHARE_DIR / f'{old_id}.json.gz'
        if not old_path.exists():
            return _error('Share not found', 404)
        new_path = SHARE_DIR / f'{new_id}.json.gz'
        if new_path.exists() and new_id != old_id:
            return _error('Name already taken', 409)
        # Update the name field inside the JSON to match the new ID
        try:
            payload = json.loads(gzip.decompress(old_path.read_bytes()))
            payload['name'] = new_id
            # Track rename history so old links keep working
            aliases = payload.get('aliases', [])
            if old_id not in aliases:
                aliases.append(old_id)
            payload['aliases'] = aliases
            new_data = gzip.compress(json.dumps(payload, separators=(',', ':'), sort_keys=True).encode())
            new_path.write_bytes(new_data)
            if old_path != new_path:
                old_path.unlink()
        except Exception:
            # Fallback: just move the file
            if not new_path.exists():
                old_path.rename(new_path)
        # Leave a small redirect file at old path so old links still resolve
        if old_id != new_id:
            try:
                alias_data = json.dumps({'redirect': new_id}, separators=(',', ':'))
                (SHARE_DIR / f'{old_id}.json.gz').write_bytes(
                    gzip.compress(alias_data.encode()))
            except Exception:
                pass  # best-effort
        new_path.touch()
        log.info("share: renamed %s → %s", old_id, new_id)
        return jsonify({'id': new_id, 'old_id': old_id})
    except Exception as e:
        log.error("share rename: %s", traceback.format_exc())
        return _error(str(e))


@app.route('/')
def index():
    # ?share=X&dl=gpkg → redirect to GPKG download
    share_id = request.args.get('share', '')
    dl = request.args.get('dl', '').lower()
    if share_id and dl in ('gpkg', 'geopackage'):
        qs = '&'.join(f'{k}={v}' for k, v in request.args.items()
                       if k not in ('share', 'dl'))
        url = f'/api/v1/share/{share_id}/download.gpkg'
        if qs:
            url += '?' + qs
        return redirect(url)
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
