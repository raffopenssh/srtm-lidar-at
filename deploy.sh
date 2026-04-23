#!/bin/bash
set -euo pipefail

# =============================================================================
# deploy.sh — Deploy srtm-lidar-at to a new exe.dev VM instance
#
# Usage (from a fresh exe.dev VM):
#   curl -sL https://raw.githubusercontent.com/raffopenssh/srtm-lidar-at/main/deploy.sh | bash
#
# Or clone first and run locally:
#   git clone https://github.com/raffopenssh/srtm-lidar-at.git
#   cd srtm-lidar-at && bash deploy.sh
#
# Environment variables (set before running, or edit below):
#   PEER_URL     — URL of the primary instance for coordination
#                  (e.g. https://srtm-lidar-at.exe.xyz:8000)
#   INSTANCE_ID  — Name for this instance (default: hostname)
#   GIT_REPO     — Git clone URL (default: the project repo)
# =============================================================================

PEER_URL="${PEER_URL:-https://srtm-lidar-at.exe.xyz:8000}"
INSTANCE_ID="${INSTANCE_ID:-$(hostname)}"
GIT_REPO="${GIT_REPO:-https://github.com/raffopenssh/srtm-lidar-at.git}"
PROJECT_DIR="/home/exedev/srtm-lidar"

echo "══════════════════════════════════════════════════════"
echo "  srtm-lidar-at deployment"
echo "  Instance: ${INSTANCE_ID}"
echo "  Peer:     ${PEER_URL}"
echo "══════════════════════════════════════════════════════"

# ── 1. Clone repo ────────────────────────────────────────
if [ ! -d "${PROJECT_DIR}/.git" ]; then
    echo "\n[1/7] Cloning repository..."
    git clone "${GIT_REPO}" "${PROJECT_DIR}"
else
    echo "\n[1/7] Repository exists — pulling latest..."
    cd "${PROJECT_DIR}" && git pull --ff-only || true
fi
cd "${PROJECT_DIR}"

# ── 2. System dependencies ───────────────────────────────
echo "\n[2/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv python3-dev \
    libgdal-dev gdal-bin \
    libgeos-dev libproj-dev \
    libspatialindex-dev \
    vnstat \
    xz-utils \
    > /dev/null 2>&1

# ── 3. Python dependencies ───────────────────────────────
echo "\n[3/7] Installing Python packages..."
pip install --quiet --break-system-packages -r requirements.txt 2>/dev/null \
  || pip install --quiet -r requirements.txt

# ── 4. Decompress RF model ───────────────────────────────
echo "\n[4/7] Preparing RF model..."
if [ ! -f data/best_model/rf_model.joblib ]; then
    if [ -f data/best_model/rf_model.joblib.xz ]; then
        echo "  Decompressing rf_model.joblib.xz (103 MB → 364 MB)..."
        xz -dk data/best_model/rf_model.joblib.xz
    else
        echo "  WARNING: No RF model found! Check git checkout."
    fi
else
    echo "  RF model already exists."
fi

# ── 5. Create data directories ───────────────────────────
echo "\n[5/7] Creating data directories..."
mkdir -p data/austria_processor/{json,gpkg,logs,tile_checkpoints,zenodo_zip_index}
mkdir -p data/shares data/gpkg_cache
mkdir -p /tmp/learned_classifier /tmp/segment_progress /tmp/segment_results
mkdir -p /tmp/copernicus_cache /tmp/hansen_cache

# Initialize empty files if missing
[ -f data/austria_processor/retry_queue.json ] || echo '[]' > data/austria_processor/retry_queue.json
[ -f data/austria_processor/failed_kgs.json ]  || echo '[]' > data/austria_processor/failed_kgs.json

# Write peer URLs for the web API's peer sync thread
echo "${PEER_URL}" > data/austria_processor/peer_urls.txt
echo "  ✓ Peer URL written to data/austria_processor/peer_urls.txt"

# ── 6. Install systemd services ──────────────────────────
echo "\n[6/7] Installing systemd services..."

# Generate processor service with peer coordination
cat > /tmp/austria_processor.service << EOSVC
[Unit]
Description=Austria Landscape Processor (${INSTANCE_ID})
After=network.target srv.service

[Service]
Type=simple
User=exedev
WorkingDirectory=${PROJECT_DIR}
ExecStart=/usr/bin/python3 austria_processor.py --mark-uncertain --peers ${PEER_URL} --instance-id ${INSTANCE_ID}
Restart=on-failure
RestartSec=60
MemoryMax=8G
MemoryHigh=7G
OOMScoreAdjust=100
OOMPolicy=stop

Environment=PYTHONUNBUFFERED=1
Environment=INSTANCE_ID=${INSTANCE_ID}

StandardOutput=append:${PROJECT_DIR}/data/austria_processor/logs/processor.log
StandardError=append:${PROJECT_DIR}/data/austria_processor/logs/processor.log

[Install]
WantedBy=multi-user.target
EOSVC

sudo cp srv.service /etc/systemd/system/srv.service
sudo cp /tmp/austria_processor.service /etc/systemd/system/austria_processor.service
sudo systemctl daemon-reload

# ── 7. Start services ────────────────────────────────────
echo "\n[7/7] Starting services..."
sudo systemctl enable --now srv
echo "  ✓ Web API started on port 8000"

# Test peer connectivity before starting processor
echo "\n  Testing peer connectivity..."
if curl -sf "${PEER_URL}/api/v1/processing/peers" > /dev/null 2>&1; then
    echo "  ✓ Peer reachable at ${PEER_URL}"
    PEER_DATA=$(curl -sf "${PEER_URL}/api/v1/processing/peers" 2>/dev/null)
    PEER_COMPLETED=$(echo "${PEER_DATA}" | python3 -c "import json,sys; print(len(json.load(sys.stdin).get('completed',[])))" 2>/dev/null || echo "?")
    echo "  ✓ Peer reports ${PEER_COMPLETED} completed KGs"
else
    echo "  ⚠ Peer unreachable — processor will run independently"
fi

# Enable throttle mode by default on new instances (save bandwidth)
touch data/austria_processor/upload_throttle
echo "  ✓ Throttle mode enabled (GPKG uploads skipped — saves ~700 MB/KG)"

sudo systemctl enable --now austria_processor
echo "  ✓ Processor started with peer coordination"

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo ""
echo "  Web UI:     https://$(hostname).exe.xyz:8000/"
echo "  Dashboard:  https://$(hostname).exe.xyz:8000/process.html"
echo "  Peer:       ${PEER_URL}"
echo "  Throttle:   ON (toggle via dashboard)"
echo ""
echo "  Monitor:    journalctl -u srv -f"
echo "              tail -f data/austria_processor/logs/processor.log"
echo ""
echo "  To also coordinate the PRIMARY instance with this one,"
echo "  update the primary's systemd unit:"
echo "    ExecStart=... --peers https://$(hostname).exe.xyz:8000"
echo "══════════════════════════════════════════════════════"
