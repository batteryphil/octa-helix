from datasets import load_dataset
ds = load_dataset("allenai/big-reasoning-traces", "DeepSeek", split="train", streaming=True)
it = iter(ds)
item = next(it)
print(item.keys())
print("system:", item.get('system', '')[:50])
print("messages:")
for m in item.get('messages', []):
    print(f"  {m['role']}: {m['content'][:50]}...")
