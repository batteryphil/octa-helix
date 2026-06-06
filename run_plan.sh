#!/bin/bash
# run_plan.sh — Phase 3j (arm specialization) then Phase 3r (router training)
# Usage: bash run_plan.sh
# Runs both phases sequentially. If 3j already has a checkpoint, resumes from it.

set -e
PROJ=/home/phil/.gemini/antigravity/scratch/analysis_project
VENV=$PROJ/titan_venv/bin/python3
LOG=$PROJ/training_log.txt
CKPTS=$PROJ/titan_checkpoints

export HF_TOKEN="os.environ.get("HF_TOKEN","")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd $PROJ

echo "============================================"
echo "  TITAN TRAINING PLAN"
echo "  Phase 3j: arm specialization (75K steps)"
echo "  Phase 3r: router training    (30K steps)"
echo "============================================"
date

# ── Phase 3j ──────────────────────────────────────────────────────────────────
echo ""
echo "[PLAN] Starting Phase 3j — arm specialization via hard domain routing"
echo "[PLAN] Backbone frozen | Arms train | Hard 1-hot routing | 75K steps"
echo ""

$VENV master_titan_trainer.py \
    --phase 3j \
    --ckpt mamba14b_transfer.pt \
    2>&1 | tee -a $LOG

echo ""
echo "[PLAN] Phase 3j complete."
echo "[PLAN] Checkpoint: $CKPTS/phase_3j.pt"
date

# ── Phase 3r ──────────────────────────────────────────────────────────────────
echo ""
echo "[PLAN] Starting Phase 3r — router training via LM loss"
echo "[PLAN] Backbone + arms FROZEN | Router trains | Soft top-2 routing | 30K steps"
echo ""

$VENV master_titan_trainer.py \
    --phase 3r \
    2>&1 | tee -a $LOG

echo ""
echo "[PLAN] Phase 3r complete."
echo "[PLAN] Final checkpoint: $CKPTS/phase_3r.pt"
date
echo ""
echo "============================================"
echo "  TRAINING COMPLETE"
echo "  Run inference test:"
echo "  python3 -c \"...\" --ckpt phase_3r.pt"
echo "============================================"
