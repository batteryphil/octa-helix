"""Test the model with the exact training prompt format."""
import titan_inference

engine = titan_inference.get_engine()

prompts = [
    "User: What is the capital of France?\nAssistant: <think>\n",
    "User: What is 2 + 2?\nAssistant: <think>\n",
    "User: Hello, how are you?\nAssistant: <think>\n",
]

for prompt in prompts:
    print(f"\nPrompt: {prompt.strip()}")
    print("Response: ", end="", flush=True)
    tokens = []
    for tok, info in engine.stream(prompt, max_new_tokens=60, temperature=0.75, top_p=0.90):
        if "<|endoftext|>" in tok:
            break
        print(tok, end="", flush=True)
        tokens.append(tok)
    print(f"\n--- ({len(tokens)} tokens) ---")
