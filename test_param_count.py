import sys
sys.path.append("thalamic-bloom-main/src")
from mamba3_titan_builder import Mamba3Titan
model = Mamba3Titan(vocab_size=50304, d_model=2560, n_layers=64, mimo_paths=8)
print("Params:", sum(p.numel() for p in model.parameters()))
