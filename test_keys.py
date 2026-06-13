import torch
import titan_inference
engine = titan_inference.get_engine("legacy_1.4b_project/titan_checkpoints/phase_sft20_best.pt")
ckpt = torch.load("legacy_1.4b_project/titan_checkpoints/phase_sft20_best.pt", map_location='cpu')
sd = ckpt.get('model', ckpt.get('model_state_dict', ckpt))

missing, unexpected = engine.model.load_state_dict(sd, strict=False)
print("MISSING KEYS:", len(missing), missing[:5])
print("UNEXPECTED KEYS:", len(unexpected), unexpected[:5])
