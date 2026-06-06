#!/bin/bash
# Wait for sft6 to finish then continue from its checkpoint
while ps aux | grep -q "sft6_targeted.py" | grep -v grep; do sleep 5; done
sleep 3
# Update LOAD_FROM to use sft6 checkpoint, run 1000 more steps
cd /home/phil/.gemini/antigravity/scratch/analysis_project
sed 's|LOAD_FROM = "titan_checkpoints/phase_sft2.pt"|LOAD_FROM = "titan_checkpoints/phase_sft6_targeted.pt"|; s|TARGET_STEPS     = 500|TARGET_STEPS     = 1000|; s|SAVE_AS   = "titan_checkpoints/phase_sft6_targeted.pt"|SAVE_AS   = "titan_checkpoints/phase_sft6b_targeted.pt"|; s|LOG_PATH  = "sft6.log"|LOG_PATH  = "sft6b.log"|' sft6_targeted.py > sft6b_targeted.py
echo "sft6b ready — launching..."
nohup ./titan_venv/bin/python3 -u sft6b_targeted.py >> sft6b.log 2>&1 &
echo "sft6b PID: $!"
