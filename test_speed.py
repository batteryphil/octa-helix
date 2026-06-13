import time
from titan_inference import get_engine
engine = get_engine("auto")
print("Engine loaded.")
prompt = "User: What is 2+2?\nAssistant: "
t0 = time.time()
tokens = []
for tok, info in engine.stream(prompt, max_new_tokens=10):
    tokens.append(tok)
    print(tok, end="", flush=True)
t1 = time.time()
print(f"\nTime: {t1-t0:.2f}s, {len(tokens)/(t1-t0):.2f} tok/s")
