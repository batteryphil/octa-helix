#!/bin/bash
# ==============================================================================
# MAMBA 3 TITAN — MASTER LAUNCHER & UI HOST
# ==============================================================================

echo "============================================================================="
echo "  JARVIS V5 // PRE-FLIGHT INITIALIZATION"
echo "============================================================================="

# 1. Setup Virtual Environment
if [ ! -d "titan_venv" ]; then
    echo "[1/4] Creating isolated Python environment (titan_venv)..."
    python3 -m venv titan_venv
fi

echo "[2/4] Activating environment..."
source titan_venv/bin/activate

# 2. Install Dependencies safely
echo "[3/4] Installing dependencies... (This may take several minutes for DeepSpeed)"
pip install --upgrade pip
pip install torch transformers datasets huggingface_hub
# DeepSpeed is installed separately to catch errors cleanly
pip install deepspeed

# 3. Launch UI Server in background
echo "[4/4] Starting Local Web UI on port 8080..."
cd monitor_ui
# Check if python3 or python
if command -v python3 &>/dev/null; then
    nohup python3 -m http.server 8080 > server.log 2>&1 &
else
    nohup python -m http.server 8080 > server.log 2>&1 &
fi
UI_PID=$!
cd ..
echo "  -> Monitor UI Live: http://localhost:8080"

# 4. Launch Titan
echo "============================================================================="
echo "  IGNITING TITAN PHASE 1 (BASE PRE-TRAINING)"
echo "============================================================================="
echo "Training logs will be written to training_log.txt. Watch the UI!"

# Initialize default telemetry file so UI doesn't crash immediately
echo '{"phase": "waiting", "step": 0, "lm_loss": 0.0, "domain_loss": 0.0, "gate_score": 0.0, "entropy": 0.0}' > monitor_ui/telemetry.json

nohup ./run_titan.sh --phase 1 > training_log.txt 2>&1 &
TRAIN_PID=$!

echo "Titan Process ID: $TRAIN_PID"
echo "To stop training, run: kill $TRAIN_PID"
echo "To stop UI server, run: kill $UI_PID"
