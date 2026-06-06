import torch
import json
import sys
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), 'thalamic-bloom-main', 'src'))
try:
    from mamba3_titan_builder import Mamba3Titan, LowRankBridge
except ImportError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_MIMO = "titan_checkpoints/phase_sft20_best.pt"
BASE_MODEL_NAME = "state-spaces/mamba-1.4b-hf"
N_SAMPLES = 3

def load_mimo():
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tok.eos_token_id = tok.eos_token_id or 0
    tok.add_special_tokens({"additional_special_tokens": ["<think>","</think>"]})
    
    model = Mamba3Titan(vocab_size=50304, d_model=2048, n_layers=48, mimo_paths=8, use_gradient_checkpointing=False)
    model.bridge = LowRankBridge(d_model=2048, bottleneck=512)
    model.set_phase("sft")
    
    ckpt = torch.load(CKPT_MIMO, map_location="cpu", weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model = model.to(torch.bfloat16).to(DEVICE)
    model.eval()
    return model, tok

def generate_patch(model, tok, prompt, is_mimo):
    ids = tok.encode(prompt, return_tensors='pt').to(DEVICE)
    if ids.shape[1] > 1800:
        # Truncate context to save room for generation
        ids = ids[:, -1800:]
        
    gen = []
    max_tokens = 300
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for _ in range(max_tokens):
            out = model(ids)
            logits = out[0] if is_mimo else out.logits
            nxt = logits[0, -1].float()
            t = nxt.argmax(dim=-1).item()
            gen.append(t)
            ids = torch.cat([ids, torch.tensor([[t]], device=DEVICE)], dim=-1)
            if t == tok.eos_token_id:
                break
                
    response = tok.decode(gen, skip_special_tokens=False)
    if is_mimo and "</think>" in response:
        patch = response.split("</think>")[-1].strip()
    else:
        patch = response.strip()
    
    if not patch:
        patch = "diff --git a/null b/null" # Fallback empty patch
    return patch

def main():
    print(f"Loading SWE-bench Lite (Evaluating {N_SAMPLES} samples)...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    
    model, tok = load_mimo()
    
    predictions = []
    out_file = "predictions.jsonl"
    
    with open(out_file, "w") as f:
        pass # clear
        
    for i, item in enumerate(ds):
        if i >= N_SAMPLES: break
        print(f"Processing {i+1}/{N_SAMPLES}: {item['instance_id']}")
        
        prompt = f"User: Fix the following issue by providing a standard unified git diff patch.\n\nIssue:\n{item['problem_statement']}\n\nAssistant: <think>\n"
        
        patch = generate_patch(model, tok, prompt, is_mimo=True)
        
        # Format required by swebench harness
        pred = {
            "instance_id": item['instance_id'],
            "model_patch": patch,
            "model_name_or_path": "mimo-1.4b-8arm"
        }
        predictions.append(pred)
        
        with open(out_file, "a") as f:
            f.write(json.dumps(pred) + "\n")
            
    print(f"Saved {N_SAMPLES} predictions to {out_file}")

if __name__ == "__main__":
    main()
