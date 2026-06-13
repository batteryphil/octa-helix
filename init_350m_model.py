import os
import torch
from mamba3_titan_builder import Mamba3Titan

# 350M Parameter configuration
VOCAB_SIZE = 50304
D_MODEL = 1024
N_LAYERS = 24
MIMO_PATHS = 8

def init_new_model():
    print(f"Initializing new {D_MODEL} d_model, {N_LAYERS} layer Mamba3 MIMO model...")
    model = Mamba3Titan(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        n_layers=N_LAYERS,
        mimo_paths=MIMO_PATHS, 
        arm_expand=2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
    
    out_dir = "checkpoints_350m"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase_0_init.pt")
    
    torch.save(model.state_dict(), out_path)
    print(f"Model saved to {out_path}")

if __name__ == "__main__":
    init_new_model()
