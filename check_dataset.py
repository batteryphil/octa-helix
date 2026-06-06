from datasets import load_dataset
ds = load_dataset("allenai/big-reasoning-traces", "DeepSeek", split="train", streaming=True)
for item in ds:
    print(item)
    break
