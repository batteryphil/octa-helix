import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'thalamic-bloom-main', 'src'))
from mamba3_titan_builder import Mamba3Titan, LowRankBridge

CKPT = "/home/phil/.gemini/antigravity/scratch/analysis_project/titan_checkpoints/phase_3j.pt"

model = Mamba3Titan(vocab_size=50304, d_model=2048, n_layers=48, mimo_paths=8, use_gradient_checkpointing=False)
model.bridge = LowRankBridge(d_model=2048, bottleneck=512)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
state = ckpt["model"] if "model" in ckpt else ckpt
model.load_state_dict(state, strict=False)

print("Measuring L2 distance between Arm 0 and other arms for out_proj.weight:")
arm0_w = model.mimo_reasoning_blocks[0].ssm.out_proj.weight
for i in range(1, 8):
    arm_i_w = model.mimo_reasoning_blocks[i].ssm.out_proj.weight
    dist = torch.norm(arm0_w - arm_i_w).item()
    print(f"Arm 0 vs Arm {i}: {dist:.4f}")

print("\nMeasuring L2 distance between Arm 0 and other arms for in_proj.weight:")
arm0_in = model.mimo_reasoning_blocks[0].ssm.in_proj.weight
for i in range(1, 8):
    arm_i_in = model.mimo_reasoning_blocks[i].ssm.in_proj.weight
    dist = torch.norm(arm0_in - arm_i_in).item()
    print(f"Arm 0 vs Arm {i}: {dist:.4f}")
