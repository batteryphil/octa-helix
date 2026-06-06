import torch, glob
for path in glob.glob("legacy_1.4b_project/titan_checkpoints/*.pt"):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model_dict = ckpt.get("model", ckpt)
        if not isinstance(model_dict, dict): continue
        layers = [k for k in model_dict.keys() if k.startswith("layers.")]
        if not layers: continue
        max_layer = max([int(k.split(".")[1]) for k in layers])
        d_model = model_dict.get("embedding.weight", torch.empty(0)).shape[-1] if "embedding.weight" in model_dict else 0
        if max_layer > 47 or d_model > 2048:
            print(f"FOUND 2.7B! {path} -> max_layer={max_layer}, d_model={d_model}")
    except Exception as e:
        pass
print("Done scanning.")
