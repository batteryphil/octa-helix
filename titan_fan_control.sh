#!/bin/bash
# Titan training startup — GPU power cap + fan max
# Case fans left to Dell EC (automatic)

# Max out GPU fan via nvidia-settings
nvidia-settings -a '[gpu:0]/GPUFanControlState=1' -a '[fan:0]/GPUTargetFanSpeed=100' 2>/dev/null
echo "GPU fan set to 100%"

# Cap GPU power to 140W to keep temps under control during long training
nvidia-smi -pl 140 2>/dev/null
echo "GPU power capped to 140W"
