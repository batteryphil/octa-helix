import time
import torch
from titan_inference import get_engine

engine = get_engine("auto")
print("Engine loaded.")
prompt = "User: What is 2+2?\nAssistant:"

t0 = time.time()
print(f"Generating for prompt: {prompt!r}")

try:
    for tok, info in engine.stream(prompt, max_new_tokens=20):
        print(f"TOKEN: {tok!r}")
except Exception as e:
    print(f"Exception: {e}")

print(f"Time: {time.time()-t0:.2f}s")
