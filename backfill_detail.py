#!/usr/bin/env python3
"""Backfill oob_curve_detail.jsonl for CSV combos missing detail.

Holds the curve eval lock so cron/UI triggers don't conflict.
"""
import csv, fcntl, json, os, sys, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_checkpoints import (
    load_checkpoints, train_at_n_kgs, DETAIL_JSONL, CURVE_LOCKFILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("backfill")

CSV_PATH = Path("data/oob_curve.csv")


def main():
    # Acquire the same lock that curve eval + cron use
    CURVE_LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(CURVE_LOCKFILE), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError):
        log.error("Curve eval lock held — another eval is running. Aborting.")
        os.close(lock_fd)
        sys.exit(1)

    try:
        _run()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        log.info("Lock released.")


def _run():
    log.info("Lock acquired.")

    # Load existing JSONL
    existing = set()
    if DETAIL_JSONL.exists():
        with open(DETAIL_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    existing.add((int(d["n_kgs"]), int(d["seed"])))
                except (json.JSONDecodeError, KeyError):
                    continue

    # Load CSV combos
    needed = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            combo = (int(r["n_kgs"]), int(r["seed"]))
            if combo not in existing:
                needed.append(combo)
    needed.sort()

    log.info("%d combos to backfill (%d already done)", len(needed), len(existing))
    if not needed:
        return

    ckpts = load_checkpoints()
    log.info("Loaded %d checkpoints", len(ckpts))

    for i, (n_kgs, seed) in enumerate(needed):
        t0 = time.time()
        stats = train_at_n_kgs(ckpts, n_kgs, random_state=seed)
        dt = time.time() - t0
        if stats is None:
            log.warning("[%d/%d] n_kgs=%d seed=%d: skipped (too few samples)",
                        i + 1, len(needed), n_kgs, seed)
            continue
        stats["train_time_s"] = round(dt, 1)
        stats["seed"] = seed

        row = {
            "n_kgs": stats["n_kgs"],
            "seed": seed,
            "n_samples": stats["n_samples"],
            "n_classes": stats["n_classes"],
            "oob": stats["oob"],
            "n_estimators": stats["n_estimators"],
            "max_depth": stats["max_depth"],
            "min_samples_leaf": stats["min_samples_leaf"],
            "all_importances": stats["all_importances"],
            "per_class_oob": stats["per_class_oob"],
            "train_time_s": stats["train_time_s"],
        }
        with open(DETAIL_JSONL, "a") as f:
            f.write(json.dumps(row) + "\n")

        log.info("[%d/%d] n_kgs=%d seed=%d OOB=%.4f classes=%d (%.0fs)",
                 i + 1, len(needed), n_kgs, seed, stats["oob"],
                 stats["n_classes"], dt)

    log.info("Done. Backfilled %d combos.", len(needed))


if __name__ == "__main__":
    main()
