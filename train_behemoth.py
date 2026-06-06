import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from mamba3_behemoth_builder import Mamba3Behemoth
import time
import argparse

# --- MOCK DATASET FOR MASSIVE RAM SHUFFLING ---
class CognitiveCocktailDataset(Dataset):
    """
    Mock dataset that generates random sequences to simulate the 
    Cognitive Cocktail (GSM8K + ARC + Premium Reasoning) in RAM.
    """
    def __init__(self, num_samples=100, seq_len=1024, vocab_size=50304):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random token sequence
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # Target is the next token (shifted by 1)
        labels = input_ids.clone()
        return input_ids, labels

def train():
    print("================================================================")
    print("  JARVIS V5 // MAMBA 3 BEHEMOTH (800M) — DEEPSPEED CPU OFFLOAD  ")
    print("================================================================")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    # DeepSpeed configuration argument
    try:
        import deepspeed
        parser = deepspeed.add_config_arguments(parser)
        HAS_DEEPSPEED = True
    except ImportError:
        HAS_DEEPSPEED = False
        print("WARNING: DeepSpeed not found. Will run a dry-run mock without it.")
        
    cmd_args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target Base Device: {device}")
    
    # 1. Initialize Behemoth Architecture
    # 40 Layers, 10 MIMO Arms, d_model=1536
    print("Building Mamba3 Behemoth Architecture (~799M parameters)...")
    model = Mamba3Behemoth(
        vocab_size=50304,
        d_model=1536,
        n_layers=40,
        mimo_paths=10,
        use_gradient_checkpointing=True # CRITICAL for 12GB VRAM
    )
    model.initialize_asymmetric_arms()
    print("Behemoth initialized successfully.")
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # 2. Setup Optimizer (AdamW)
    head_params_set = set(model.lm_head.parameters())
    core_params_list = [p for p in model.parameters() if p not in head_params_set]
    
    optimizer = torch.optim.AdamW([
        {'params': core_params_list, 'lr': 1e-4}, 
        {'params': list(head_params_set), 'lr': 2e-4}
    ], weight_decay=0.01)
    
    criterion = nn.CrossEntropyLoss()
    
    # 3. Massive RAM Dataloader
    print("Loading Cognitive Cocktail into 124GB RAM...")
    dataset = CognitiveCocktailDataset(num_samples=32, seq_len=1024)
    # The batch size is handled by DeepSpeed config, but we provide a default dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 4. DeepSpeed Initialization
    if HAS_DEEPSPEED:
        print("Initializing DeepSpeed Engine...")
        model_engine, optimizer, dataloader, _ = deepspeed.initialize(
            args=cmd_args,
            model=model,
            optimizer=optimizer,
            model_parameters=model.parameters(),
            training_data=dataset
        )
        print("DeepSpeed Engine Online. Optimizer state offloaded to CPU.")
    else:
        model_engine = model.to(device)
        print("Running in raw PyTorch fallback mode.")
    
    # 5. Dummy Training Loop
    print("\nStarting Warmup Epoch...")
    model_engine.train()
    
    start_time = time.time()
    for step, batch in enumerate(dataloader):
        if HAS_DEEPSPEED:
            # DeepSpeed dataloader yields the entire batch natively
            input_ids, labels = batch
            input_ids = input_ids.to(model_engine.local_rank)
            labels = labels.to(model_engine.local_rank)
        else:
            input_ids, labels = batch
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
        
        # Forward pass (handles latent forcing and gradient checkpointing internally)
        logits = model_engine(input_ids, loop_idx=0)
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate loss
        loss = criterion(shift_logits.view(-1, 50304), shift_labels.view(-1))
        
        if HAS_DEEPSPEED:
            # DeepSpeed handles backward and stepping
            model_engine.backward(loss)
            model_engine.step()
        else:
            loss.backward()
            optimizer.step()
        
        # Read telemetry (from base model)
        base_model = model_engine.module if HAS_DEEPSPEED else model_engine
        telem = base_model.last_telemetry
        gate_score = telem.get('gate_score', 0.0)
        entropy = telem.get('entropy', 0.0)
        
        print(f"Step {step+1:03d} | Loss: {loss.item():.4f} | Gate Score: {gate_score:.4f} | Entropy: {entropy:.4f}")
        
    end_time = time.time()
    print(f"\nTraining complete. Time elapsed: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    train()
