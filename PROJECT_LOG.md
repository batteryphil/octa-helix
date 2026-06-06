# Mamba3 Titan — Complete Project Log
*Last updated: 2026-05-26 — use this to resume after any restart*

---

## TL;DR — Current State

- **Model**: Mamba3 Titan (Mamba1 1.4B backbone + 8 MIMO arms)
- **Phase**: 3j (arm specialization via domain routing)
- **Baseline**: `mamba14b_transfer.pt` — Mamba1 1.4B pretrained weights (NOT scratch)
- **Training cmd**: `master_titan_trainer.py --phase 3j --ckpt .../mamba14b_transfer.pt`
- **After first save**: resume with just `--phase 3j` (no --ckpt)
- **LM loss at step 1**: ~4.9 (vs ~10.8 for scratch — Mamba1 already knows language)
- **Target**: 200,000 steps Phase 3j → 30,000 steps Phase 3r

---

## Key File Locations

| File | Purpose |
|---|---|
| `analysis_project/mamba3_titan_builder.py` | Model architecture |
| `analysis_project/master_titan_trainer.py` | Training loop |
| `analysis_project/training_log.txt` | Live training output (append-only) |
| `analysis_project/monitor_ui/telemetry.json` | Live metrics |
| `analysis_project/monitor_ui/word_salad.json` | Generation samples every 250 steps |
| `titan_checkpoints/mamba14b_transfer.pt` | **CORRECT START CHECKPOINT** (6.5GB, Mamba1 1.4B) |
| `titan_checkpoints/phase_3j.pt` | Running Phase 3j checkpoint (auto-saved every 500 steps) |
| `titan_checkpoints/phase_1.pt` | Scratch Phase 1 (DO NOT USE) |
| `titan_checkpoints/phase_2.pt` | Scratch Phase 2 (DO NOT USE) |

---

## How to Restart Training

```bash
cd /home/phil/.gemini/antigravity/scratch/analysis_project
export HF_TOKEN="os.environ.get("HF_TOKEN","")"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- FIRST RUN (from Mamba1 1.4B baseline) ---
nohup ./titan_venv/bin/python3 master_titan_trainer.py \
  --phase 3j \
  --ckpt /home/phil/.gemini/antigravity/scratch/analysis_project/titan_checkpoints/mamba14b_transfer.pt \
  >> training_log.txt 2>&1 &

# --- RESUME (once phase_3j.pt exists with steps saved) ---
nohup ./titan_venv/bin/python3 master_titan_trainer.py \
  --phase 3j \
  >> training_log.txt 2>&1 &
```

---

## Architecture

### Model Parameters
- **Backbone**: 48 Mamba layers, d_model=2048, split at layer 24 for routing
- **MIMO Arms**: 8 parallel MambaLayer specialists
- **Router**: Linear(2048→8) + learnable temperature, reads mid-backbone hidden state
- **Blackboard**: 64-dim sparse IPC bus (inter-arm communication)
- **ConceptPerceptron**: Global context injected every 6 backbone layers
- **LowRankBridge**: 2048→64→2048 bottleneck before arms
- **Vocab**: 50304 (GPT-NeoX tokenizer, EleutherAI/gpt-neox-20b)

### 8 MIMO Arms

| Arm | Role | Phase 3j Dataset |
|---|---|---|
| 0 | General Language | TinyStories |
| 1 | Symbolic Math | MetaMathQA |
| 2 | Logical Reasoning | ARC-Challenge |
| 3 | Code Syntax | CodeAlpaca |
| 4 | Factual Recall | Wikipedia |
| 5 | Summarization | CNN/DailyMail |
| 6 | Creative/Chat | OpenHermes-2.5 |
| 7 | Instruction Following | OpenWebText |

Equal weights (1 each) — every arm gets identical gradient update count.

---

## Phase History

### Phase 1–2 Scratch: ABANDONED
Trained from scratch, ~50K+ steps total. Problems:
- Too slow to learn real language (word salad throughout)
- Loss plateaued early on OpenHermes chat data (wrong dataset for pre-training)
- Arms cloned at Phase 2 start but never diverged
**Decision**: Switch to Mamba1 1.4B pretrained backbone instead.

### Phase 3j Scratch: ABANDONED at step 23,618
Ran on scratch Phase 2 weights. Problems:
- arm_collapse_metric stuck at 1.0 (arms never diverged)
- Three bugs discovered (see Bugs section)
- Wrong baseline — should have used Mamba1 transfer
**Decision**: Stop and restart from mamba14b_transfer.pt

### Phase 3j Mamba1 1.4B: CURRENT
- Source: `state-spaces/mamba-1.4b-hf` backbone weights
- Arms initialized from Mamba1 layers 40-47 + small noise
- Router initialized fresh
- Backbone frozen, arms + router + blackboard train
- LM loss starts ~4.9 (real language knowledge from day 1)

---

## Why Mamba1 1.4B Transfer?

Mamba1 1.4B is already pretrained on 300B tokens from The Pile. Instead of burning days getting the scratch model to learn basic language, we:
1. Downloaded `state-spaces/mamba-1.4b-hf`
2. Mapped its 48-layer, d_model=2048 backbone weights into Titan's backbone
3. Initialized 8 arms from Mamba1's last 8 layers (layers 40-47) + small noise
4. Grafted Titan's router, blackboard, LM head on fresh
5. Saved as `mamba14b_transfer.pt`

This means Phase 3j starts with a model that already knows English — arms just need to specialize into their domains.

---

## Bugs Found and Fixed (2026-05-26)

### Bug 1: Diversity Loss Was a Dead Scalar
**Where**: `mamba3_titan_builder.py`, arm diversity loss block (~line 446)
**Problem**: The diversity loss was wrapped in `torch.no_grad()` and the result `.detach()`'d. It was adding a constant to the loss function — no gradient ever reached arm weights. Arms had zero push to diverge from each other.
**Fix**: Removed `no_grad` and `detach`. Now live gradient flows through cosine similarity of arm outputs → arms are actively pushed apart each training step.

### Bug 2: Arm Telemetry Loop Was Wrong Size
**Where**: `master_titan_trainer.py`, arm collapse computation (~line 1220)
**Problem**: `for i in range(16)` — model has 8 arms. Index 8 raises IndexError, caught silently by `except: pass`. `arm_sims` returned as `[]`, fell back to `arm_collapse_metric=1.0` always. We could not tell if arms were actually diverging.
**Fix**: Changed to `range(model.mimo_paths)`. Also fixed `torch.eye(16)` → `torch.eye(model.mimo_paths)` and `/15.0` → `/max(1, model.mimo_paths-1)`.

### Bug 3: _arm_snapshot Never Assigned
**Where**: `mamba3_titan_builder.py`, forward pass (~line 468)
**Problem**: Telemetry tried to read `self._arm_snapshot` but it was never set in the forward pass → `AttributeError` every call, silently caught. Glass-brain collapse metric always failed.
**Fix**: Added `self._arm_snapshot = stacked_states.detach().float().mean(dim=(0,1))` after computing arm outputs.

### Bug 4: Training from Wrong Checkpoint
**Where**: trainer invocation
**Problem**: This session was running `phase_3j.pt` (from the slow scratch chain) instead of `mamba14b_transfer.pt` (Mamba1 pretrained). The transfer checkpoint had been created in a previous session but this session didn't load it.
**Fix**: Stopped trainer, restarted with `--ckpt /path/to/mamba14b_transfer.pt`.

---

## Metrics Reference

| Metric | Meaning | Target by 10k steps |
|---|---|---|
| LM Loss | Language model cross-entropy | < 2.5 |
| Dom Loss | Router CE loss (domain classification) | < 1.5 |
| Div | Arm output cosine similarity | Decreasing from ~0.15 |
| arm_collapse_metric | 1=clones, 0=orthogonal | < 0.90 |
| Entropy | Routing distribution entropy | 0.0 (1-hot routing, correct) |
| GNorm | Gradient norm | < 10 normally |
| TPS | Tokens per second | ~1,400–1,500 |
| GPU temp | RTX 3060 temperature | < 85°C |

---

## Hardware

- **Machine**: Dell Precision 7820 workstation
- **CPU**: Xeon Gold 6136 @ 3.0GHz (12c/24t) — **1 of 2 sockets populated**
- **RAM**: 124GB ECC DDR4
- **GPU**: NVIDIA RTX 3060 12GB (training)
- **PSU**: 1400W (confirmed — sufficient for planned V100 upgrade)

### Planned Upgrade: 3× Tesla V100 PCIe 16GB
- Buy on eBay: `"Tesla V100 PCIe 16GB"` — NOT SXM2 form factor
- Also need: 2nd Xeon Gold 6136 LGA3647 (~$50–80) + heatsink (~$20–40)
- PSU already sufficient (1400W vs ~1150W needed)
- When hardware arrives: add DDP to trainer (`torchrun --nproc_per_node=3`)
- Estimated speedup: ~3× (~4,500 TPS vs ~1,450 now)

---

## HuggingFace Token
`os.environ.get("HF_TOKEN","")`
Set as env var `HF_TOKEN` before launching trainer.

---

## Next Steps After Phase 3j (200k steps)

```bash
# Phase 3r — Router refinement via LM loss (30k steps)
nohup ./titan_venv/bin/python3 master_titan_trainer.py --phase 3r >> training_log.txt 2>&1 &
```

In Phase 3r:
- Arms are frozen (protecting 3j specialization)
- Router trains via LM loss (learns which arm actually reduces loss per domain)
- 30,000 steps
- Then: SFT (supervised fine-tuning) — format TBD
