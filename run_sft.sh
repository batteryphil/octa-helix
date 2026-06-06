#!/bin/bash
# run_sft.sh — Executes the Phase 4 SFT curriculum sequentially.
# This assumes Phase 3r has completed and titan_checkpoints/phase_3r.pt exists.

set -e
set -o pipefail
PROJ="/home/phil/.gemini/antigravity/scratch/analysis_project"
VENV="$PROJ/titan_venv/bin/python3"
LOG="$PROJ/training_log.txt"

export HF_TOKEN="os.environ.get("HF_TOKEN","")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd $PROJ

echo "============================================"
echo "  TITAN SFT CURRICULUM (PHASE 4)"
echo "============================================"
date

# 1. Backbone Unfreeze & Factual Injection
echo ""
echo "[SFT STEP 1] Backbone Unfreeze (sft15)"
echo "[PLAN] Teaches the Router to bypass reasoning for trivial facts."
echo ""
$VENV sft15_2.7b_backbone.py 2>&1 | tee -a $LOG

# 2. Targeted Reasoning
echo ""
echo "[SFT STEP 2] Targeted Reasoning (sft20)"
echo "[PLAN] High-complexity datasets (Algebra, Logic) with THINK_MIN masking."
echo ""
$VENV sft20_2.7b_reasoning.py 2>&1 | tee -a $LOG

# 3. Termination Sprint
echo ""
echo "[SFT STEP 3] Termination Sprint (sft3)"
echo "[PLAN] Flat LR sprint to optimize the P(</think>) closure threshold."
echo ""
$VENV sft3_2.7b_sprint.py 2>&1 | tee -a $LOG

echo ""
echo "============================================"
echo "  SFT CURRICULUM COMPLETE"
echo "  The model is now ready for CAAI Governor Inference."
echo "============================================"
date
