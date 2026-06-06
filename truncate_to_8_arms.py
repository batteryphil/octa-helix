import torch
import os

def truncate_checkpoint():
    ckpt_path = 'checkpoints_2.7b/phase_1.pt'
    opt_path = 'checkpoints_2.7b/phase_1_optim.pt'
    bak_ckpt_path = 'checkpoints_2.7b/phase_1_16arms.pt.bak'
    bak_opt_path = 'checkpoints_2.7b/phase_1_16arms_optim.pt.bak'
    
    if not os.path.exists(ckpt_path):
        print(f"File not found: {ckpt_path}")
        return

    # Backup originals
    if not os.path.exists(bak_ckpt_path):
        os.rename(ckpt_path, bak_ckpt_path)
        print(f"Backed up checkpoint to {bak_ckpt_path}")
    else:
        print(f"Backup already exists at {bak_ckpt_path}, using it as source.")
        
    if os.path.exists(opt_path) and not os.path.exists(bak_opt_path):
        os.rename(opt_path, bak_opt_path)
        print(f"Backed up optimizer to {bak_opt_path}")

    # Load 16-arm checkpoint
    print("Loading 16-arm checkpoint...")
    ckpt = torch.load(bak_ckpt_path, map_location='cpu', weights_only=False)
    
    # Truncate model weights
    new_model = {}
    for k, v in ckpt['model'].items():
        if k.startswith('mimo_reasoning_blocks.'):
            arm_idx = int(k.split('.')[1])
            if arm_idx < 8:
                new_model[k] = v
        elif 'domain_router' in k:
            if 'weight' in k:
                # Truncate router output from 16 to 8
                new_model[k] = v[:8, :]
            elif 'bias' in k:
                new_model[k] = v[:8]
        else:
            new_model[k] = v
            
    ckpt['model'] = new_model
    
    print(f"Saving truncated checkpoint to {ckpt_path}...")
    torch.save(ckpt, ckpt_path)
    
    # Load and truncate optimizer
    if os.path.exists(bak_opt_path):
        print("Loading 16-arm optimizer...")
        opt = torch.load(bak_opt_path, map_location='cpu', weights_only=False)
        
        # We need to map state items. If a state key belongs to a parameter we deleted, remove it.
        # But wait, opt['state'] is a dictionary where keys are parameter IDs OR indices.
        # If the keys are integers, they correspond to the param_groups.
        # It's usually better to just flush the optimizer state for the arms/router,
        # or completely delete the optimizer to start fresh in Phase 1 if we changed topology.
        # Since Phase 1 is a foundation phase, deleting the optimizer state for the arms is safest,
        # but the backbone state is huge and valuable.
        # To be completely safe and avoid tensor dimension mismatches in Adam states,
        # we will simply NOT write the truncated optimizer back, forcing a fresh momentum buffer.
        # Phase 1 has 250k steps, momentum rebuilds quickly.
        print("Skipping optimizer truncation to avoid topology mismatches. Phase 1 will rebuild momentum.")
        
    print("Done!")

if __name__ == '__main__':
    truncate_checkpoint()
