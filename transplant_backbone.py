import torch
import sys

def main():
    p1_path = "titan_checkpoints/phase_1.pt"
    p2_path = "titan_checkpoints/phase_2.pt"
    out_path = "titan_checkpoints/phase_2_repaired.pt"

    print(f"Loading {p1_path}...")
    p1 = torch.load(p1_path, map_location="cpu", weights_only=True)
    p1_model = p1.get("model", p1)

    print(f"Loading {p2_path}...")
    p2 = torch.load(p2_path, map_location="cpu", weights_only=True)
    p2_model = p2.get("model", p2)

    print("\nPerforming transplant...")
    repaired_model = {}
    transplant_count = 0
    keep_count = 0

    for k, v in p2_model.items():
        if k.startswith("embedding.") or k.startswith("layers.") or k.startswith("lm_head.") or k.startswith("norm_f."):
            if k in p1_model:
                repaired_model[k] = p1_model[k].clone()
                transplant_count += 1
            else:
                print(f"WARNING: {k} missing from phase_1.pt! Keeping phase_2 version.")
                repaired_model[k] = v.clone()
                keep_count += 1
        else:
            repaired_model[k] = v.clone()
            keep_count += 1

    print(f"Transplanted {transplant_count} tensors from Phase 1.")
    print(f"Kept {keep_count} tensors from Phase 2 (Arms, Router, etc).")

    new_ckpt = {
        'model': repaired_model,
        'step': p2.get('step', 120000),
        'phase': '2'
    }

    print(f"\nSaving to {out_path}...")
    torch.save(new_ckpt, out_path)
    print("Done!")

if __name__ == "__main__":
    main()
