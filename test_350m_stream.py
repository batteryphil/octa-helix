import torch
from transformers import AutoTokenizer
from mamba3_titan_builder import Mamba3Titan
import time

def main():
    print("Loading 350M Mamba3 model...")
    model = Mamba3Titan(
        vocab_size=50304,
        d_model=1024,
        n_layers=24,
        mimo_paths=8,
        arm_expand=2
    ).cuda()
    
    ckpt_path = "checkpoints_350m/phase_1_step_500.pt"
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
        print("Loaded Phase 1 weights.")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return
        
    model = model.bfloat16().eval()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    prompt = "Once upon a time, there was a little girl named Lily. She loved to"
    print(f"\nPrompt: {prompt}\n")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    
    print("Generating...")
    start_time = time.time()
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Very basic generation loop
            out_ids = input_ids[0].tolist()
            for _ in range(50):
                x = torch.tensor([out_ids], dtype=torch.long).cuda()
                logits, _ = model(x, loop_idx=0)
                next_id = torch.argmax(logits[0, -1, :]).item()
                out_ids.append(next_id)
                if next_id == tokenizer.eos_token_id:
                    break
                    
    generated_text = tokenizer.decode(out_ids)
    elapsed = time.time() - start_time
    
    print(f"\n--- Output ({50 / elapsed:.2f} tokens/sec) ---")
    print(generated_text)
    print("---------------------\n")

if __name__ == "__main__":
    main()
