import titan_inference
import torch

engine = titan_inference.get_engine("legacy_1.4b_project/titan_checkpoints/phase_sft18_best.pt")

prompt = "<|system|>\nYou are a helpful assistant.\n<|endoftext|>\n<|user|>\nWhat is the capital of France?\n<|endoftext|>\n<|assistant|>\n"
print("Prompting with proper format...")
for tok, info in engine.stream(prompt, max_new_tokens=20, temperature=0.1):
    print(tok, end="", flush=True)
print("\nDone.")
