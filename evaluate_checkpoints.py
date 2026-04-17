#!/usr/bin/env python3
"""Evaluate RF model quality across checkpoint accumulation curves.

Loads all KG checkpoint .npz files and trains RF models at incremental
KG counts (every 5 KGs) to trace OOB score as a function of training
data volume. Identifies the optimal checkpoint count.

Also used by the monitoring cron (--monitor mode) to track the live
model, detect convergence/degradation, and preserve the best model.
"""
import argparse
import csv
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger("eval_ckpt")

CHECKPOINT_DIR = Path("rf_training_data/checkpoints")
DATA_DIR = Path("data")
BEST_DIR = DATA_DIR / "best_model"
HISTORY_CSV = DATA_DIR / "oob_history.csv"
CURVE_CSV = DATA_DIR / "oob_curve.csv"
MONITOR_STATE = DATA_DIR / "monitor_state.json"
LIVE_META = Path("/tmp/learned_classifier/rf_meta.json")
LIVE_MODEL = Path("/tmp/learned_classifier/rf_model.joblib")
CURVE_LOCKFILE = Path("/tmp/rf_curve_eval.lock")
DETAIL_JSONL = DATA_DIR / "oob_curve_detail.jsonl"
N_SEEDS = 5  # max seeds (used for early KGs)
SEEDS = list(range(N_SEEDS))

def seeds_for_n_kgs(n_kgs):
    """Fewer seeds as spread narrows with more data."""
    if n_kgs <= 30:
        return list(range(5))
    if n_kgs <= 60:
        return list(range(3))
    return [0]


def composite_score(oob, per_class_oob):
    """Composite quality score: rewards balanced per-class accuracy.

    Formula: 0.4 * OOB + 0.35 * mean_per_class + 0.25 * worst_class

    A model with 75% OOB but all classes >= 50% beats one with 77% OOB
    where rock/earthwork are 0%. The worst-class term penalises models
    that sacrifice rare classes for overall accuracy.
    """
    if not per_class_oob:
        return oob  # fallback to raw OOB
    vals = list(per_class_oob.values())
    mean_cls = sum(vals) / len(vals)
    min_cls = min(vals)
    return 0.4 * oob + 0.35 * mean_cls + 0.25 * min_cls



def load_checkpoints():
    """Load all checkpoint files, return list of (kg_code, features, labels)."""
    checkpoints = []
    for f in sorted(CHECKPOINT_DIR.glob("kg_*.npz")):
        try:
            d = np.load(f, allow_pickle=True)
            kg_code = str(d["kg_code"]) if "kg_code" in d else f.stem
            checkpoints.append({
                "kg_code": kg_code,
                "features": d["features"].tolist(),
                "labels": d["labels"].tolist(),
            })
        except Exception as e:
            log.warning("Failed to load %s: %s", f, e)
    return checkpoints


def train_at_n_kgs(checkpoints, n, n_estimators=200, max_depth=20,
                   min_samples_leaf=5, random_state=42):
    """Train RF on `n` checkpoints (shuffled by random_state). Returns stats dict."""
    import random as _rnd
    from sklearn.ensemble import RandomForestClassifier
    from learned_classifier import FEATURE_KEYS, TYPE_CLASSES, feature_vector, RF_EXCLUDED_CLASSES

    # Shuffle KG order per seed — measures sensitivity to data composition
    shuffled = list(checkpoints)
    _rnd.Random(random_state).shuffle(shuffled)
    subset = shuffled[:n]
    X_list, y_list = [], []
    for ckpt in subset:
        for feat, label in zip(ckpt["features"], ckpt["labels"]):
            # Remap excavation/fill → earthwork
            if label in ("excavation", "fill"):
                label = "earthwork"
            # Drop RF-excluded classes
            if label in RF_EXCLUDED_CLASSES:
                continue
            if label not in TYPE_CLASSES:
                continue
            X_list.append(feature_vector(feat))
            y_list.append(label)

    if len(X_list) < 20:
        return None

    X = np.stack(X_list)
    y = np.array(y_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Break tree_loss label circularity (same fix as learned_classifier.train)
    _circ_cols = [
        FEATURE_KEYS.index("hansen_recent_loss_frac"),
        FEATURE_KEYS.index("hansen_treecover2000"),
    ]
    tl_mask = (y == "tree_loss")
    if tl_mask.any():
        X[np.ix_(tl_mask, _circ_cols)] = 0.0

    # Downsample dominant classes
    from learned_classifier import _downsample
    X, y = _downsample(X, y)

    classes = sorted(set(y))
    n_classes = len(classes)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,
        n_jobs=1,
        random_state=random_state,
        class_weight="balanced",
    )
    rf.fit(X, y)

    importances = {k: float(v) for k, v in zip(FEATURE_KEYS, rf.feature_importances_)}
    top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]

    # Per-class OOB accuracy
    oob_preds = rf.oob_decision_function_
    pred_labels = rf.classes_[np.argmax(oob_preds, axis=1)]
    per_class = {}
    for c in classes:
        mask = y == c
        if mask.sum() > 0:
            per_class[c] = float((pred_labels[mask] == c).mean())

    oob_val = float(rf.oob_score_)
    comp = composite_score(oob_val, per_class)

    return {
        "n_kgs": n,
        "n_samples": len(X),
        "n_classes": n_classes,
        "oob": oob_val,
        "composite": comp,
        "mean_class_oob": sum(per_class.values()) / len(per_class) if per_class else 0.0,
        "min_class_oob": min(per_class.values()) if per_class else 0.0,
        "top5_features": top5,
        "all_importances": importances,
        "per_class_oob": per_class,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "model": rf,
        "classes": classes,
    }


def is_curve_running():
    """Check if a curve evaluation is already in progress."""
    import fcntl
    if not CURVE_LOCKFILE.exists():
        return False
    try:
        fd = os.open(str(CURVE_LOCKFILE), os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False  # lock is free → not running
        except (OSError, IOError):
            return True  # lock held → running
        finally:
            os.close(fd)
    except FileNotFoundError:
        return False


def run_curve(step=5):
    """Train models at every `step` KGs and write OOB curve."""
    import fcntl
    CURVE_LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(CURVE_LOCKFILE), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError):
        log.warning("Curve evaluation already running (lockfile held), skipping.")
        os.close(lock_fd)
        return []
    try:
        return _run_curve_locked(step)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _save_best_model(rf, stats, seed):
    """Save RF model as live model and backup best."""
    import joblib
    import datetime
    from learned_classifier import MODEL_PATH, META_PATH, FEATURE_KEYS

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    # Save live model
    joblib.dump(rf, MODEL_PATH)
    meta = {
        'classes': stats['classes'],
        'n_train': stats['n_samples'],
        'oob_score': stats['oob'],
        'composite_score': stats['composite'],
        'mean_class_oob': stats['mean_class_oob'],
        'min_class_oob': stats['min_class_oob'],
        'per_class_oob': stats['per_class_oob'],
        'feature_importances': stats['all_importances'],
        'feature_keys': FEATURE_KEYS,
        'trained_at': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M'),
        'n_kgs': stats['n_kgs'],
        'best_seed': seed,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    log.info('  Saved live model to %s (composite=%.4f, OOB=%.4f, %d KGs, seed=%d)',
             MODEL_PATH, stats['composite'], stats['oob'], stats['n_kgs'], seed)

    # Backup copy
    shutil.copy2(MODEL_PATH, BEST_DIR / 'rf_model.joblib')
    shutil.copy2(META_PATH, BEST_DIR / 'rf_meta.json')

    # Append to best.log
    now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(BEST_DIR / 'best.log', 'a') as f:
        f.write(f"{now} NEW BEST: composite={stats['composite']:.6f} OOB={stats['oob']:.6f} "
                f"mean_cls={stats['mean_class_oob']:.4f} min_cls={stats['min_class_oob']:.4f} "
                f"kgs={stats['n_kgs']} samples={stats['n_samples']} "
                f"classes={stats['n_classes']} seed={seed}\n")
    log.info('  Backup saved to %s/', BEST_DIR)


def _run_curve_locked(step=5):
    """Actual curve evaluation (called with lock held).

    Runs N_SEEDS random seeds per KG step. Incremental: reads existing CSV,
    skips (n_kgs, seed) combos already evaluated, only trains missing ones.
    Also tracks the best model (highest OOB at max class count) and saves it.
    """
    checkpoints = load_checkpoints()
    n_total = len(checkpoints)
    log.info("Loaded %d checkpoints", n_total)

    CURVE_CSV.parent.mkdir(parents=True, exist_ok=True)

    # CSV columns now include seed, composite, per-class summary
    HEADER = ["n_kgs", "seed", "n_samples", "n_classes", "oob",
              "composite", "mean_class_oob", "min_class_oob",
              "top_feature", "top_importance", "train_time_s"]

    # Read existing rows to know what's done
    existing = set()  # (n_kgs, seed)
    if CURVE_CSV.exists():
        try:
            with open(CURVE_CSV) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    seed = int(r["seed"]) if "seed" in r else 0
                    existing.add((int(r["n_kgs"]), seed))
        except Exception:
            pass

    # Also load existing JSONL detail to know what's already saved there
    existing_detail = set()  # (n_kgs, seed)
    if DETAIL_JSONL.exists():
        try:
            with open(DETAIL_JSONL) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        existing_detail.add((int(obj["n_kgs"]), int(obj["seed"])))
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception:
            pass

    # Migrate old CSV without seed or composite columns: rewrite
    if CURVE_CSV.exists() and existing:
        try:
            with open(CURVE_CSV) as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
                if "seed" not in fields or "composite" not in fields:
                    rows = list(csv.DictReader(open(CURVE_CSV)))
                    log.info("Migrating %d old rows (adding missing columns)", len(rows))
                    with open(CURVE_CSV, "w", newline="") as wf:
                        w = csv.writer(wf)
                        w.writerow(HEADER)
                        for r in rows:
                            w.writerow([
                                r["n_kgs"], r.get("seed", 0), r["n_samples"],
                                r["n_classes"], r["oob"],
                                r.get("composite", r["oob"]),  # fallback
                                r.get("mean_class_oob", ""),
                                r.get("min_class_oob", ""),
                                r.get("top_feature", ""),
                                r.get("top_importance", ""),
                                r.get("train_time_s", "")])
                    existing = {(int(r["n_kgs"]), int(r.get("seed", 0))) for r in rows}
        except Exception as e:
            log.warning("Migration check failed: %s", e)

    # Ensure header exists
    if not CURVE_CSV.exists() or CURVE_CSV.stat().st_size == 0:
        with open(CURVE_CSV, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)

    steps = list(range(step, n_total + 1, step))

    # Build work list: adaptive seed count per step
    work = []
    for n in steps:
        for seed in seeds_for_n_kgs(n):
            if (n, seed) not in existing:
                work.append((n, seed))

    if not work:
        log.info("All steps already evaluated, nothing to do")
        return []

    log.info("%d existing, %d new (step, seed) combos to evaluate",
             len(existing), len(work))

    # Best-model tracking: highest COMPOSITE score at maximum class count.
    # Composite = 0.4*OOB + 0.35*mean_class + 0.25*min_class — rewards balance.
    best_composite = 0.0
    best_n_classes = 0
    best_stats = None
    best_model = None
    best_seed = None

    # Seed from existing CSV data so we don't regress
    if existing:
        try:
            max_cls_seen = 0
            with open(CURVE_CSV) as f:
                for r in csv.DictReader(f):
                    nc = int(r['n_classes'])
                    if nc > max_cls_seen:
                        max_cls_seen = nc
            with open(CURVE_CSV) as f:
                for r in csv.DictReader(f):
                    nc = int(r['n_classes'])
                    comp = float(r.get('composite', r['oob']))  # fallback
                    if nc >= max_cls_seen and comp > best_composite:
                        best_composite = comp
                        best_n_classes = nc
            if best_composite > 0:
                log.info("Historical best from CSV: composite=%.4f at %d classes",
                         best_composite, best_n_classes)
        except Exception:
            pass

    results = []
    for n, seed in work:
        t0 = time.time()
        stats = train_at_n_kgs(checkpoints, n, random_state=seed)
        dt = time.time() - t0
        if stats is None:
            continue
        stats["train_time_s"] = round(dt, 1)
        stats["seed"] = seed
        results.append(stats)
        log.info("n_kgs=%3d  seed=%d  samples=%6d  classes=%2d  OOB=%.4f  composite=%.4f  mean_cls=%.4f  min_cls=%.4f  (%.1fs)",
                 n, seed, stats["n_samples"], stats["n_classes"], stats["oob"],
                 stats["composite"], stats["mean_class_oob"], stats["min_class_oob"], dt)

        # Best-model check: compare COMPOSITE score at max class count.
        # Composite rewards balanced per-class accuracy, not just overall OOB.
        nc = stats["n_classes"]
        comp = stats["composite"]
        if nc > best_n_classes:
            # New max class count — reset best (previous was inflated)
            log.info("  Class count grew %d → %d, resetting best", best_n_classes, nc)
            best_composite = 0.0
            best_n_classes = nc
        if nc >= best_n_classes and comp > best_composite:
            best_composite = comp
            best_stats = stats
            best_model = stats["model"]
            best_seed = seed
            log.info("  🏆 NEW BEST: composite=%.4f OOB=%.4f mean_cls=%.4f min_cls=%.4f (%d KGs, seed=%d, %d classes)",
                     comp, stats["oob"], stats["mean_class_oob"], stats["min_class_oob"], n, seed, nc)
            _save_best_model(best_model, best_stats, best_seed)

        # Free the model from stats to avoid holding all models in memory
        stats.pop("model", None)

        # Append row to CSV immediately
        with open(CURVE_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                stats["n_kgs"], seed, stats["n_samples"], stats["n_classes"],
                f"{stats['oob']:.6f}",
                f"{stats['composite']:.6f}",
                f"{stats['mean_class_oob']:.6f}",
                f"{stats['min_class_oob']:.6f}",
                stats["top5_features"][0][0],
                f"{stats['top5_features'][0][1]:.4f}",
                stats["train_time_s"],
            ])

        # Append detail to JSONL (full importances, per-class OOB, hyperparams)
        if (n, seed) not in existing_detail:
            detail_row = {
                "n_kgs": stats["n_kgs"],
                "seed": seed,
                "n_samples": stats["n_samples"],
                "n_classes": stats["n_classes"],
                "oob": stats["oob"],
                "composite": stats["composite"],
                "mean_class_oob": stats["mean_class_oob"],
                "min_class_oob": stats["min_class_oob"],
                "n_estimators": stats["n_estimators"],
                "max_depth": stats["max_depth"],
                "min_samples_leaf": stats["min_samples_leaf"],
                "all_importances": stats["all_importances"],
                "per_class_oob": stats["per_class_oob"],
                "train_time_s": stats["train_time_s"],
            }
            with open(DETAIL_JSONL, "a") as f:
                f.write(json.dumps(detail_row) + "\n")
            existing_detail.add((n, seed))

    # Summary
    if results:
        log.info("")
        log.info("=" * 60)
        log.info("Added %d new points.", len(results))
        if best_stats:
            log.info("Best model: composite=%.4f OOB=%.4f at %d KGs (seed=%d, %d classes)",
                     best_composite, best_stats["oob"], best_stats["n_kgs"], best_seed, best_n_classes)
    else:
        log.info("No new points added.")

    return results


def monitor():
    """Monitor mode: check live model, log history, preserve best.

    Called every 5 min by cron. Tracks:
    - OOB score over time
    - Convergence detection (rolling window of last N checkpoints)
    - Best model preservation
    - Degradation warnings
    """
    import datetime

    BEST_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Init CSV
    if not HISTORY_CSV.exists():
        with open(HISTORY_CSV, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "oob", "n_train", "n_kgs", "n_classes",
                "trained_at", "top_feature", "top_importance",
                "delta_oob", "rolling_avg_5", "status",
            ])

    # Exit if no live model
    if not LIVE_META.exists():
        return

    meta = json.loads(LIVE_META.read_text())
    oob = meta["oob_score"]
    n_train = meta["n_train"]
    n_kgs = meta["n_kgs"]
    n_classes = len(meta["classes"])
    trained_at = meta["trained_at"]
    importances = meta.get("feature_importances", {})

    # Dedup: skip if trained_at unchanged
    existing_rows = []
    if HISTORY_CSV.exists():
        with open(HISTORY_CSV) as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
        if existing_rows and existing_rows[-1].get("trained_at") == trained_at:
            return  # no new model

    # Top feature
    top_feat = ""
    top_imp = 0.0
    if importances:
        top = max(importances.items(), key=lambda x: x[1])
        top_feat, top_imp = top[0], top[1]

    # Compute delta and rolling avg
    recent_oobs = [float(r["oob"]) for r in existing_rows[-10:] if r.get("oob")]
    delta_oob = oob - recent_oobs[-1] if recent_oobs else 0.0
    rolling_5 = [float(r["oob"]) for r in existing_rows[-4:] if r.get("oob")]
    rolling_5.append(oob)
    rolling_avg = sum(rolling_5) / len(rolling_5)

    # Load monitor state
    state = {}
    if MONITOR_STATE.exists():
        state = json.loads(MONITOR_STATE.read_text())
    best_oob = state.get("best_oob", 0.0)
    best_n_kgs = state.get("best_n_kgs", 0)
    peak_oob = state.get("peak_oob", 0.0)
    peak_n_kgs = state.get("peak_n_kgs", 0)
    consecutive_decline = state.get("consecutive_decline", 0)
    plateau_count = state.get("plateau_count", 0)

    # Determine status — within the mature regime (same n_classes).
    # Early checkpoints with fewer classes have inflated OOB; don't compare.
    mature_n_classes = state.get("mature_n_classes", n_classes)
    if n_classes > mature_n_classes:
        # Class count grew — previous peak was with fewer classes (inflated OOB).
        # Reset mature peak so we only compare like-for-like.
        log.info("Mature class count grew %d → %d, resetting mature peak",
                 mature_n_classes, n_classes)
        mature_n_classes = n_classes
        state["mature_peak_oob"] = 0.0
        state["mature_peak_n_kgs"] = 0

    # Compare only against same-class-count history
    mature_oobs = [float(r["oob"]) for r in existing_rows[-10:]
                   if r.get("oob") and int(r.get("n_classes", 0)) == n_classes]

    status = "improving"
    if len(mature_oobs) >= 2:
        if oob < mature_oobs[-1] - 0.001:
            consecutive_decline = state.get("consecutive_decline", 0) + 1
            plateau_count = 0
            if consecutive_decline >= 5:
                status = "degrading"
            else:
                status = "declining"
        elif abs(oob - mature_oobs[-1]) <= 0.001:
            plateau_count = state.get("plateau_count", 0) + 1
            consecutive_decline = 0
            if plateau_count >= 8:
                status = "converged"
            else:
                status = "plateau"
        else:
            consecutive_decline = 0
            plateau_count = 0
            status = "improving"

    # Track peak within mature regime (same n_classes)
    mature_peak_oob = state.get("mature_peak_oob", 0.0)
    mature_peak_n_kgs = state.get("mature_peak_n_kgs", 0)
    if n_classes == mature_n_classes and oob > mature_peak_oob:
        mature_peak_oob = oob
        mature_peak_n_kgs = n_kgs

    # Track absolute peak (any class count)
    if oob > peak_oob:
        peak_oob = oob
        peak_n_kgs = n_kgs

    # Preserve best model — best within mature regime
    is_new_best = (n_classes >= mature_n_classes and oob > best_oob)
    if is_new_best:
        best_oob = oob
        best_n_kgs = n_kgs
        if LIVE_MODEL.exists():
            shutil.copy2(LIVE_MODEL, BEST_DIR / "rf_model.joblib")
        shutil.copy2(LIVE_META, BEST_DIR / "rf_meta.json")
        # Append to best.log
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(BEST_DIR / "best.log", "a") as f:
            f.write(f"{now} NEW BEST: OOB={oob:.6f} kgs={n_kgs} "
                    f"samples={n_train} classes={n_classes}\n")

    # Write history row
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(HISTORY_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            now, f"{oob:.6f}", n_train, n_kgs, n_classes,
            trained_at, top_feat, f"{top_imp:.4f}",
            f"{delta_oob:+.6f}", f"{rolling_avg:.6f}", status,
        ])

    # Save state
    kgs_since_peak = n_kgs - mature_peak_n_kgs if mature_peak_n_kgs else 0
    should_stop = (status == "degrading" or
                   (status == "converged" and kgs_since_peak > 50))
    state = {
        "best_oob": best_oob,
        "best_n_kgs": best_n_kgs,
        "peak_oob": peak_oob,
        "peak_n_kgs": peak_n_kgs,
        "mature_peak_oob": mature_peak_oob,
        "mature_peak_n_kgs": mature_peak_n_kgs,
        "mature_n_classes": mature_n_classes,
        "consecutive_decline": consecutive_decline,
        "plateau_count": plateau_count,
        "last_oob": oob,
        "last_n_kgs": n_kgs,
        "last_n_classes": n_classes,
        "last_status": status,
        "kgs_since_peak": kgs_since_peak,
        "should_stop": should_stop,
        "updated_at": now,
    }
    MONITOR_STATE.write_text(json.dumps(state, indent=2))

    # Log to syslog for visibility
    flag = "🏆" if is_new_best else "📊"
    msg = (f"{flag} OOB={oob:.4f} kgs={n_kgs} samples={n_train} "
           f"Δ={delta_oob:+.4f} avg5={rolling_avg:.4f} [{status}]")
    if is_new_best:
        msg += " ← NEW BEST"
    if state["should_stop"]:
        msg += f" ⚠️ RECOMMEND STOP (mature peak {mature_peak_oob:.4f} at {mature_peak_n_kgs} KGs, "
        msg += f"{kgs_since_peak} KGs past peak)"
    os.system(f'logger -t rf_monitor "{msg}"')
    print(msg)


def report():
    """Print current monitoring status."""
    if MONITOR_STATE.exists():
        state = json.loads(MONITOR_STATE.read_text())
        print(json.dumps(state, indent=2))
    else:
        print("No monitor state yet.")

    if HISTORY_CSV.exists():
        print(f"\nHistory ({HISTORY_CSV}):")
        with open(HISTORY_CSV) as f:
            for line in f:
                print(line.rstrip())
    if CURVE_CSV.exists():
        print(f"\nCurve ({CURVE_CSV}):")
        with open(CURVE_CSV) as f:
            for line in f:
                print(line.rstrip())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="RF model evaluation & monitoring")
    sub = parser.add_subparsers(dest="command")

    p_curve = sub.add_parser("curve", help="Build OOB curve across all checkpoints")
    p_curve.add_argument("--step", type=int, default=5, help="KG step size")

    p_monitor = sub.add_parser("monitor", help="Check live model (cron mode)")
    p_report = sub.add_parser("report", help="Print current status")

    args = parser.parse_args()
    if args.command == "curve":
        run_curve(step=args.step)
    elif args.command == "monitor":
        monitor()
    elif args.command == "report":
        report()
    else:
        parser.print_help()
