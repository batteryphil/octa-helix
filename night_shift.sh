#!/bin/bash
# night_shift.sh — Waits for Phase 3r to finish, then runs SFT and Benchmarks.

PID=1305865
PROJ="/home/phil/.gemini/antigravity/scratch/analysis_project"
LOG="$PROJ/night_shift.log"

cd $PROJ

echo "============================================" | tee -a $LOG
echo "  NIGHT SHIFT AUTOMATION STARTED" | tee -a $LOG
date | tee -a $LOG
echo "  Waiting for Phase 3r (PID: $PID) to complete..." | tee -a $LOG
echo "============================================" | tee -a $LOG

# Wait until the deepspeed process exits
while kill -0 $PID 2>/dev/null; do
    sleep 60
done

echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "  Phase 3r has finished! Proceeding to SFT." | tee -a $LOG
date | tee -a $LOG
echo "============================================" | tee -a $LOG

# Run SFT Pipeline
./run_sft.sh 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "  SFT Pipeline has finished! Proceeding to Benchmarks." | tee -a $LOG
date | tee -a $LOG
echo "============================================" | tee -a $LOG

# Run Benchmark Pipeline
./run_benchmarks.sh 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "  NIGHT SHIFT COMPLETE! ALL TASKS FINISHED." | tee -a $LOG
date | tee -a $LOG
echo "============================================" | tee -a $LOG
