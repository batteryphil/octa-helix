import time
import torch
from titan_inference import get_engine

engine = get_engine("auto")
print("Engine loaded.")
prompt = "User: What is 2+2?\nAssistant: "

# Warmup
print("Warming up...")
_ = engine.generate(prompt, max_new_tokens=2)

print("Profiling...")
t0 = time.time()
tokens = []
times = []
last_t = time.time()
for tok, info in engine.stream(prompt, max_new_tokens=10):
    tokens.append(tok)
    t = time.time()
    times.append(t - last_t)
    last_t = t

print(f"\nTokens: {tokens}")
print(f"Times: {times}")
print(f"Total time for 10 tokens: {sum(times):.2f}s")
