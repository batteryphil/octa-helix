import time
import torch
from titan_inference import get_engine

engine = get_engine("auto")
print("Engine loaded.")
prompt = "User: What is 2+2?\nAssistant: "

# Force disable caching
tokens = []
for tok, info in engine.stream(prompt, max_new_tokens=10, use_cache=False):
    tokens.append(tok)
print(f"Output: {''.join(tokens)}")
