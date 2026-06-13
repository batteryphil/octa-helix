"""
Test Falcon-Mamba-7B-Instruct in 4-bit quantization.
Pure Mamba — zero KV cache, constant memory regardless of sequence length.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "tiiuae/falcon-mamba-7b-instruct"
CACHE_DIR = "/home/phil/.gemini/antigravity/scratch/analysis_project/hf_cache"

print(f"Loading {MODEL_ID} in 4-bit NF4...")
print(f"Cache dir: {CACHE_DIR}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # saves ~0.4GB extra
    bnb_4bit_quant_type="nf4",        # NormalFloat4 — best quality for LLM weights
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=CACHE_DIR,
)

vram = torch.cuda.memory_allocated() / 1e9
print(f"\n✅ Model loaded! VRAM used: {vram:.2f} GB\n")

# Test prompts
prompts = [
    "Hello! Who are you?",
    "What is 17 times 24?",
    "Write a 3-line Python function that reverses a string.",
]

for user_msg in prompts:
    messages = [{"role": "user", "content": user_msg}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,   # Mamba uses recurrent state, NOT KV cache — safe
        )

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
    )
    print(f"Q: {user_msg}")
    print(f"A: {response.strip()}")
    print("-" * 60)

vram_peak = torch.cuda.max_memory_allocated() / 1e9
print(f"\nPeak VRAM: {vram_peak:.2f} GB")
print("Done.")
