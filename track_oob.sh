#!/bin/bash
# Track RF model OOB score over time, preserve best model
set -euo pipefail

META=/tmp/learned_classifier/rf_meta.json
LOG=/home/exedev/srtm-lidar/data/oob_history.csv
BEST_DIR=/home/exedev/srtm-lidar/data/best_model

# Ensure dirs exist
mkdir -p "$(dirname "$LOG")" "$BEST_DIR"

# Init CSV header if needed
[ -f "$LOG" ] || echo "timestamp,oob,n_train,n_kgs,n_classes" > "$LOG"

# Exit silently if no model yet
[ -f "$META" ] || exit 0

# Read current stats
OOB=$(python3 -c "import json; print(json.load(open('$META'))['oob_score'])")
N_TRAIN=$(python3 -c "import json; print(json.load(open('$META'))['n_train'])")
N_KGS=$(python3 -c "import json; print(json.load(open('$META'))['n_kgs'])")
N_CLASSES=$(python3 -c "import json; print(len(json.load(open('$META'))['classes']))")
TRAINED_AT=$(python3 -c "import json; print(json.load(open('$META'))['trained_at'])")
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Deduplicate: skip if last line has same trained_at
if [ -f "$LOG" ]; then
    LAST_TRAINED=$(tail -1 "$LOG" 2>/dev/null | cut -d, -f6)
    [ "$LAST_TRAINED" = "$TRAINED_AT" ] && exit 0
fi

# Append to history
echo "$NOW,$OOB,$N_TRAIN,$N_KGS,$N_CLASSES,$TRAINED_AT" >> "$LOG"

# Check if this is a new best
BEST_OOB=0
[ -f "$BEST_DIR/rf_meta.json" ] && BEST_OOB=$(python3 -c "import json; print(json.load(open('$BEST_DIR/rf_meta.json'))['oob_score'])")

IS_BETTER=$(python3 -c "print(1 if $OOB > $BEST_OOB else 0)")
if [ "$IS_BETTER" = "1" ]; then
    cp /tmp/learned_classifier/rf_model.joblib "$BEST_DIR/rf_model.joblib"
    cp /tmp/learned_classifier/rf_meta.json "$BEST_DIR/rf_meta.json"
    echo "$NOW NEW BEST: OOB=$OOB (was $BEST_OOB) kgs=$N_KGS samples=$N_TRAIN" >> "$BEST_DIR/best.log"
    logger -t track_oob "New best RF model: OOB=$OOB kgs=$N_KGS samples=$N_TRAIN"
fi
