#!/bin/bash
# ==============================================================================
# MAMBA 3 BEHEMOTH (800M) — DEEPSPEED LAUNCHER
# ==============================================================================
# This script wraps the train_behemoth.py script in the deepspeed launcher,
# applying the ZeRO-2 CPU offload configuration.
# ==============================================================================

# Ensure deepspeed is available in the environment
if ! command -v deepspeed &> /dev/null; then
    echo "ERROR: deepspeed command could not be found."
    echo "Please ensure you have activated your Python environment and installed DeepSpeed:"
    echo "  pip install deepspeed"
    exit 1
fi

echo "Launching Behemoth (800M) via DeepSpeed..."
deepspeed train_behemoth.py --deepspeed_config ds_config.json
