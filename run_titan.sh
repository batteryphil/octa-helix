#!/bin/bash
# ==============================================================================
# MAMBA 3 TITAN (2.5B) — AUTOMATED PHASE LAUNCHER
# Appends to training_log.txt — every restart is a continuance, not a new log.
#
# Usage:
#   ./run_titan.sh --phase <1|2|3|3j|sft>     Run a specific phase
#   ./run_titan.sh --auto                      Run ALL phases 1→2→3→3j→sft
#   ./run_titan.sh --serve                     Start inference server + chat UI
# ==============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    python3 -c "
import json, sys
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
        if [ "$phase" = "sft" ]; then
            python3 "${SCRIPT_DIR}/sft_titan_trainer.py" --steps "${target_steps}"
        else
            python3 "${SCRIPT_DIR}/master_titan_trainer.py" --phase "${phase}"
        fi
        EXIT_CODE=$?
        STEP=$(get_step)

        if [ $EXIT_CODE -eq 0 ]; then
            log_banner "Phase ${phase} completed / paused at step ${STEP}"
            # Check if phase target reached
            if [ "$STEP" -ge "$target_steps" ] 2>/dev/null || [ "$phase" = "sft" ]; then
                log_banner "✅ Phase ${phase} COMPLETE. Running auto-eval..."
                python3 "${SCRIPT_DIR}/auto_eval.py" --phase "${phase}" 2>&1 | tee -a "$LOG_FILE"
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

# ── Serve mode ────────────────────────────────────────────────────────────────
serve() {
    log_banner "Starting Titan Inference Server + Chat UI"
    echo "[SERVER] API: http://localhost:8000"
    echo "[SERVER] Chat UI: open chat_ui/index.html in browser"
    echo "[SERVER] Health: http://localhost:8000/health"
    cd "${SCRIPT_DIR}" && source titan_venv/bin/activate 2>/dev/null
    python3 "${SCRIPT_DIR}/titan_server.py"
}

# ── Parse args ────────────────────────────────────────────────────────────────
MODE="$1"
PHASE_ARG="$2"

if [ "$MODE" = "--serve" ]; then
    serve
    exit $?
fi

if [ "$MODE" = "--auto" ]; then
    log_banner "=== TITAN AUTO-PIPELINE: 1 → 2 → 3 → 3j → sft ==="

    run_phase "1" 50000 || exit $?
    run_phase "2" 60000 || exit $?
    run_phase "3" 70000 || exit $?
    run_phase "3j" 80000 || exit $?
    run_phase "sft" 10000 || exit $?

    log_banner "🎉 ALL PHASES COMPLETE. Model is ready. Run: ./run_titan.sh --serve"
    exit 0
fi

if [ "$MODE" = "--phase" ]; then
    if [ -z "$PHASE_ARG" ]; then
        echo "ERROR: Specify phase with --phase <1|2|3|3j|sft>"
        exit 1
    fi

    # Phase step targets
    case "$PHASE_ARG" in
        1)   TARGET=50000 ;;
        2)   TARGET=60000 ;;
        3)   TARGET=70000 ;;
        3j)  TARGET=80000 ;;
        sft) TARGET=10000 ;;
        *)   echo "ERROR: Unknown phase '$PHASE_ARG'"; exit 1 ;;
    esac

    run_phase "$PHASE_ARG" "$TARGET"
    exit $?
fi

echo "Usage:"
echo "  ./run_titan.sh --phase <1|2|3|3j|sft>   Run specific phase"
echo "  ./run_titan.sh --auto                    Run all phases in sequence"
echo "  ./run_titan.sh --serve                   Start inference server"
exit 1
