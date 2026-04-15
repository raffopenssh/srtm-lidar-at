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
N_SEEDS = 5
SEEDS = list(range(N_SEEDS))


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
    from learned_classifier import FEATURE_KEYS, TYPE_CLASSES, feature_vector

    # Shuffle KG order per seed — measures sensitivity to data composition
    shuffled = list(checkpoints)
    _rnd.Random(random_state).shuffle(shuffled)
    subset = shuffled[:n]
    X_list, y_list = [], []
    for ckpt in subset:
        for feat, label in zip(ckpt["features"], ckpt["labels"]):
            if label not in TYPE_CLASSES:
                continue
            X_list.append(feature_vector(feat))
            y_list.append(label)

    if len(X_list) < 20:
        return None

    X = np.stack(X_list)
    y = np.array(y_list)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

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

    return {
        "n_kgs": n,
        "n_samples": len(X),
        "n_classes": n_classes,
        "oob": float(rf.oob_score_),
        "top5_features": top5,
        "per_class_oob": per_class,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
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


def _run_curve_locked(step=5):
    """Actual curve evaluation (called with lock held).

    Runs N_SEEDS random seeds per KG step. Incremental: reads existing CSV,
    skips (n_kgs, seed) combos already evaluated, only trains missing ones.
    """
    checkpoints = load_checkpoints()
    n_total = len(checkpoints)
    log.info("Loaded %d checkpoints", n_total)

    CURVE_CSV.parent.mkdir(parents=True, exist_ok=True)

    # CSV columns now include seed
    HEADER = ["n_kgs", "seed", "n_samples", "n_classes", "oob",
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

    # Migrate old CSV without seed column: rewrite with seed=0
    if CURVE_CSV.exists() and existing:
        try:
            with open(CURVE_CSV) as f:
                reader = csv.DictReader(f)
                if "seed" not in (reader.fieldnames or []):
                    rows = list(csv.DictReader(open(CURVE_CSV)))
                    log.info("Migrating %d old rows (adding seed=0)", len(rows))
                    with open(CURVE_CSV, "w", newline="") as wf:
                        w = csv.writer(wf)
                        w.writerow(HEADER)
                        for r in rows:
                            w.writerow([r["n_kgs"], 0, r["n_samples"],
                                        r["n_classes"], r["oob"],
                                        r.get("top_feature", ""),
                                        r.get("top_importance", ""),
                                        r.get("train_time_s", "")])
                    existing = {(int(r["n_kgs"]), 0) for r in rows}
        except Exception as e:
            log.warning("Migration check failed: %s", e)

    # Ensure header exists
    if not CURVE_CSV.exists() or CURVE_CSV.stat().st_size == 0:
        with open(CURVE_CSV, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)

    steps = list(range(step, n_total + 1, step))

    # Build work list: all (step, seed) combos not yet done
    work = []
    for n in steps:
        for seed in SEEDS:
            if (n, seed) not in existing:
                work.append((n, seed))

    if not work:
        log.info("All %d steps × %d seeds already evaluated, nothing to do",
                 len(steps), N_SEEDS)
        return []

    log.info("%d existing, %d new (step, seed) combos to evaluate",
             len(existing), len(work))

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
        log.info("n_kgs=%3d  seed=%d  samples=%6d  classes=%2d  OOB=%.4f  (%.1fs)",
                 n, seed, stats["n_samples"], stats["n_classes"], stats["oob"], dt)

        # Append row to CSV immediately
        with open(CURVE_CSV, "a", newline="") as f:
            csv.writer(f).writerow([
                stats["n_kgs"], seed, stats["n_samples"], stats["n_classes"],
                f"{stats['oob']:.6f}",
                stats["top5_features"][0][0],
                f"{stats['top5_features'][0][1]:.4f}",
                stats["train_time_s"],
            ])

    # Summary
    if results:
        log.info("")
        log.info("=" * 60)
        log.info("Added %d new points.", len(results))
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
        mature_n_classes = n_classes

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
        now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(BEST_DIR / "best.log", "a") as f:
            f.write(f"{now} NEW BEST: OOB={oob:.6f} kgs={n_kgs} "
                    f"samples={n_train} classes={n_classes}\n")

    # Write history row
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
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
