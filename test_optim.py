import torch
from mamba3_titan_builder import Mamba3Titan
model = Mamba3Titan(vocab_size=50304, d_model=2048, n_layers=48, mimo_paths=8, use_gradient_checkpointing=False)
model.set_phase('3j')
temp_head_params = [p for p in model.lm_head.parameters() if p.requires_grad]
temp_head_set = set(id(p) for p in temp_head_params)
temp_core_params = [p for p in model.parameters() if id(p) not in temp_head_set and p.requires_grad]
print("temp_core_params:", len(temp_core_params))
print("temp_head_params:", len(temp_head_params))
