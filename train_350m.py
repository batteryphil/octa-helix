import torch
from torch.utils.data import DataLoader
from mamba3_titan_builder import Mamba3Titan
from datasets import load_dataset
from transformers import AutoTokenizer
import os

def train():
    print("Loading 350M Mamba3 model...")
    model = Mamba3Titan(
        vocab_size=50304,
        d_model=1024,
        n_layers=24,
        mimo_paths=8,
        arm_expand=2
    ).cuda()
    
    ckpt_path = "checkpoints_350m/phase_0_init.pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print("Loaded initial weights.")
        
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    model.train()
    step = 0
    
    print("Starting Phase 1 (Cold Start) Training...")
    
    for batch in dataset:
        text = batch["text"]
        tokens = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        input_ids = tokens["input_ids"].cuda()
        
        if input_ids.shape[1] < 2: continue
        
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x, loop_idx=0)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, 50304), y.view(-1))
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")
            
        if step % 500 == 0 and step > 0:
            out_path = f"checkpoints_350m/phase_1_step_{step}.pt"
            torch.save(model.state_dict(), out_path)
            print(f"Saved checkpoint to {out_path}")
            break
            
        step += 1

if __name__ == "__main__":
    train()
