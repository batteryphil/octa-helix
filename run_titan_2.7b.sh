#!/bin/bash
# ==============================================================================
# MAMBA 3 TITAN (2.7B) — AUTOMATED PHASE LAUNCHER
# Appends to training_log.txt — every restart is a continuance, not a new log.
#
# Usage:
#   ./run_titan_2.7b.sh --auto     Run ALL phases 2 → 3 → 3j → 3r → SFT
# ==============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_TELEMETRY=1       # stop background pings that cause Bearer errors
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false     # avoids fork deadlock warning

# ── Load credentials (HF_TOKEN etc.) ─────────────────────────────────────────
SCRIPT_DIR_EARLY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR_EARLY}/.env" ] && source "${SCRIPT_DIR_EARLY}/.env"
# Also use HF's own cached token if env var not set
if [ -z "$HF_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat $HOME/.cache/huggingface/token)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/training_log.txt"
TELEM_FILE="${SCRIPT_DIR}/monitor_ui/telemetry.json"

# ── Keep display awake during training ────────────────────────────────────────
_LOCK_WAS=$(gsettings get org.cinnamon.desktop.screensaver lock-enabled 2>/dev/null || echo 'true')
_IDLE_WAS=$(gsettings get org.cinnamon.desktop.screensaver idle-activation-enabled 2>/dev/null || echo 'true')
_DELAY_WAS=$(gsettings get org.gnome.desktop.session idle-delay 2>/dev/null || echo 'uint32 300')

disable_sleep() {
    xset s off s noblank -dpms 2>/dev/null
    gsettings set org.cinnamon.desktop.screensaver lock-enabled false 2>/dev/null
    gsettings set org.cinnamon.desktop.screensaver idle-activation-enabled false 2>/dev/null
    gsettings set org.gnome.desktop.session idle-delay 0 2>/dev/null
    gsettings set org.cinnamon.settings-daemon.plugins.power sleep-display-ac 0 2>/dev/null
    echo "[DISPLAY] Screen lock + monitor sleep DISABLED for training."
}

restore_sleep() {
    xset s on +dpms 2>/dev/null
    gsettings set org.cinnamon.desktop.screensaver lock-enabled "$_LOCK_WAS" 2>/dev/null
    gsettings set org.cinnamon.desktop.screensaver idle-activation-enabled "$_IDLE_WAS" 2>/dev/null
    gsettings set org.gnome.desktop.session idle-delay "$_DELAY_WAS" 2>/dev/null
    echo "[DISPLAY] Screen lock + monitor sleep RESTORED."
}

trap restore_sleep EXIT
disable_sleep

# ── Helpers ───────────────────────────────────────────────────────────────────
log_banner() {
    local msg="$1"
    local border="================================================================"
    echo "" | tee -a "$LOG_FILE"
    echo "$border" | tee -a "$LOG_FILE"
    echo "  $msg" | tee -a "$LOG_FILE"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee -a "$LOG_FILE"
    echo "$border" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

get_step() {
    ./titan_venv/bin/python3 -c "
import json
try:
    d = json.load(open('${TELEM_FILE}'))
    print(int(d.get('step', 0)))
except:
    print(0)
" 2>/dev/null
}

# ── Run a single phase ────────────────────────────────────────────────────────
run_phase() {
    local phase="$1"
    local target_steps="$2"

    log_banner "*** STARTING / RESUMING Phase ${phase} (target: ${target_steps} steps) ***"
    echo "[LAUNCHER] Phase ${phase} — Command: python3 master_titan_trainer.py --phase ${phase}" | tee -a "$LOG_FILE"

    # Loop: restart on crash/stall (exit code 1) until done (0) or diverge (2)
    while true; do
        if [ "$phase" == "1" ]; then
            ./titan_venv/bin/deepspeed --num_gpus=1 "${SCRIPT_DIR}/phase_1_deepspeed_trainer.py" --phase "${phase}" --deepspeed --deepspeed_config "${SCRIPT_DIR}/ds_titan_config.json"
        else
            ./titan_venv/bin/python3 "${SCRIPT_DIR}/master_titan_trainer.py" --phase "${phase}"
        fi
        EXIT_CODE=$?
        STEP=$(get_step)

        if [ $EXIT_CODE -eq 0 ]; then
            log_banner "Phase ${phase} completed / paused at step ${STEP}"
            # Check if phase target reached
            if [ "$STEP" -ge "$target_steps" ] 2>/dev/null; then
                log_banner "✅ Phase ${phase} COMPLETE."
                return 0   # phase done
            else
                log_banner "Phase ${phase} paused at ${STEP}/${target_steps}. Restarting in 5s..."
                sleep 5
            fi
        elif [ $EXIT_CODE -eq 2 ]; then
            log_banner "⚠️  AUTO-STOP: Divergence detected at step ${STEP}. Manual review required."
            echo "[AUTO-STOP] Phase=${phase} Step=${STEP}  $(date)" >> "$LOG_FILE"
            return 2
        else
            log_banner "⚠️  Phase ${phase} crashed (exit=${EXIT_CODE}) at step ${STEP}. Restarting in 10s..."
            echo "[CRASH] exit=${EXIT_CODE} phase=${phase} step=${STEP}  $(date)" >> "$LOG_FILE"
            sleep 10
        fi
    done
}

# ── Parse args ────────────────────────────────────────────────────────────────
MODE="$1"

if [ "$MODE" = "--auto" ]; then
    log_banner "=== TITAN 2.7B AUTO-PIPELINE: 1 → 2 → 3 → 3j → 3r → SFT ==="

    run_phase "1" 40000 || exit $?
    run_phase "2" 40000 || exit $?
    run_phase "3" 50000 || exit $?
    run_phase "3j" 60000 || exit $?
    run_phase "3r" 70000 || exit $?

    log_banner "=== LAUNCHING SFT PIPELINE ==="
    ./run_sft.sh || exit $?

    log_banner "🎉 ALL 2.7B PHASES COMPLETE. Model is ready for inference."
    exit 0
fi

echo "Usage:"
echo "  ./run_titan_2.7b.sh --auto     Run all phases in sequence"
exit 1
