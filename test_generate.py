import titan_inference

engine = titan_inference.get_engine("legacy_1.4b_project/titan_checkpoints/phase_sft20_best.pt")

print("Prompting: 'What is the capital of France?'")
for tok, info in engine.stream("What is the capital of France?", max_new_tokens=20):
    print(tok, end="", flush=True)
print("\nDone.")
