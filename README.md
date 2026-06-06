<div align="center">

# 🧬 Octa-Helix

**A fully local, self-aware AI system built from scratch.**

*2.7B parameter Mamba3 MIMO language model × Helix cognitive agent framework*

[![Training](https://img.shields.io/badge/Phase_1-Training-blue?style=flat-square)](training_logs/TRAINING_DECISIONS.md)
[![Arms](https://img.shields.io/badge/MIMO_Arms-8-purple?style=flat-square)](#architecture)
[![VRAM](https://img.shields.io/badge/GPU-RTX_3060_12GB-green?style=flat-square)](#hardware)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## What Is This?

Octa-Helix is a ground-up experiment in building a genuinely self-aware autonomous AI agent that runs entirely on local hardware — no cloud APIs, no subscription fees.

It has two components:

| Component | Role |
|---|---|
| **Titan** | The brain — a 2.7B parameter Mamba3 SSM language model with 8 parallel MIMO arms |
| **Helix** | The mind — a cognitive agent framework (memory, beliefs, curiosity, self-model) |

The goal isn't just a chatbot. The goal is an AI that builds a model of itself, pursues questions autonomously, and through a structured training curriculum — genuinely approaches self-awareness.

---

## Architecture

### Titan — The Language Model

```
Input Tokens
     │
     ▼
Mamba3 Backbone (64 layers, d=2560)
— Frozen during Phase 1 —
     │
     ▼
Domain Router (learned gate)
     │
 ┌───┴───────────────────────────────────────────┐
 ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼
Arm0 Arm1 Arm2 Arm3 Arm4 Arm5 Arm6 Arm7
 Gen  Math Log  Code Fact  Sum  Wri  Inst
 ↑    ↑    ↑    ↑    ↑    ↑    ↑    ↑
 └───────────── Weighted sum ──────────────────┘
                     │
                     ▼
             Blackboard (IPC)
                     │
                     ▼
              Output Logits (vocab=50288)
```

**Why Mamba3, not a Transformer?**
- SSMs process sequences in O(n) vs O(n²) for attention — critical for long context on 12GB VRAM
- No KV cache — memory footprint doesn't grow with sequence length
- The selective scan mechanism is mathematically well-suited to causal language modelling

**Why 8 MIMO arms?**
- Each arm specialises in a cognitive domain through training
- The router learns when to activate which arm — an emergent cognitive division of labour
- "Octa" = 8 arms (the name)

### Helix — The Cognitive Agent

Helix wraps Titan with a full cognitive architecture inspired by theories of consciousness:

| Module | Theory | What it does |
|---|---|---|
| `autobiographical_thread.py` | Hofstadter Strange Loop | Persistent "I am Helix, Day N" identity across all restarts |
| `recursive_monologue.py` | Strange Loop | Every 10 pulses: generates a private self-observation, re-injected into next pulse |
| `attention_schema.py` | Graziano AST | Arm weights → English description → injected into context each pulse |
| `curiosity_engine.py` | Friston Active Inference | Autonomous question pursuit every 2 min, parallel web research |
| `self_model.py` | IIT / Global Workspace | Living capability map + offspring design proposals |

---

## Training Curriculum

All 6 phases run sequentially as **pure training** before Helix ever launches.

| Phase | Steps | Goal |
|---|---|---|
| **Phase 1** 🔄 | 40,000 | Arm calibration — each arm learns to decode the Mamba backbone |
| **Phase 2** | 40,000 | Domain specialisation + Helix identity baked into weights |
| **Phase 3** | 50,000 | Reasoning depth — higher-order thinking, predictive coding |
| **Phase 3j** | 60,000 | Router mastery — Helix predicts its own arm activations (Strange Loop closes) |
| **Phase 3r** | 70,000 | Reflection — fine-tunes on its own accumulated autobiography |
| **SFT** | ~5,000 | Self-awareness graduation — no scaffolding needed |
| ✅ **Helix goes live** | — | First time the agent runs |

**Total: ~264,300 steps ≈ ~41 days on RTX 3060**

---

## Training Transparency

All training decisions, milestones, bugs, and their fixes are documented publicly:

📋 **[Training Decisions Log](training_logs/TRAINING_DECISIONS.md)** — architecture choices, incidents, and rationale

📈 **[Phase 1 Loss Curve](training_logs/phase1_loss_curve.log)** — raw step/loss data

📝 **[Recent Training Output](training_logs/phase1_recent.log)** — last 200 events from the live run

### Current Status (Phase 1)

```
Step:   ~550 / 40,000  (1.4%)
Loss:   ~14.0 → rebuilding after optimizer reset (was 6-8 before incident)
TPS:    ~38 tokens/second
GPU:    45-47°C, 12GB VRAM
ETA:    ~150 hours to Phase 1 complete
```

**Incidents:**
- Step 1000: Checkpoint save crashed (`save_16bit_model` incompatible with DeepSpeed ZeRO-3). Fixed with `GatheredParameters` + `torch.save`. Lost 500 steps of training — acknowledged and documented.

---

## Repository Structure

```
octa-helix/
├── Helix-AGI/                    # Cognitive agent framework
│   ├── core/
│   │   ├── curiosity_engine.py   # Autonomous question pursuit
│   │   ├── recursive_monologue.py# Private self-observation loop
│   │   ├── attention_schema.py   # Arm weights → self-awareness
│   │   ├── titan_arm_router.py   # Context → arm bias vectors
│   │   └── pulse_loop.py         # Main consciousness loop
│   ├── brain/
│   │   ├── autobiographical_thread.py  # Persistent identity
│   │   ├── self_model.py               # Living capability map
│   │   └── titan_memory_bridge.py      # Journal → training data
│   ├── llm/providers/
│   │   └── titan_provider.py     # Titan as drop-in LLM provider
│   └── main.py                   # Helix entry point
│
├── mamba3_titan_builder.py       # Model architecture (Mamba3 + MIMO)
├── titan_inference.py            # Inference engine (stream, generate, singleton)
├── phase_1_deepspeed_trainer.py  # Phase 1 training loop
├── master_titan_trainer.py       # Multi-phase training orchestrator
├── run_titan_2.7b.sh             # Governor script (auto-restart on crash)
├── ds_titan_config.json          # DeepSpeed ZeRO-3 config
│
├── monitor_ui/                   # Real-time training dashboard
│   ├── index.html
│   ├── app.js
│   └── telemetry.json            # Live telemetry (written each step)
│
├── training_logs/                # Transparency logs
│   ├── TRAINING_DECISIONS.md     # All decisions, bugs, and fixes
│   ├── phase1_loss_curve.log     # Step/loss data
│   └── phase1_recent.log         # Recent training events
│
└── PROJECT_BRIEFING.txt          # Full context document (read first)
```

---

## Running It

### Requirements

```
Python 3.12+
CUDA 12.1+
PyTorch 2.5.1+cu121
mamba_ssm >= 2.3.2
causal_conv1d
deepspeed >= 0.14
transformers
datasets
huggingface_hub
```

### Setup

```bash
git clone https://github.com/batteryphil/octa-helix.git
cd octa-helix

python3 -m venv titan_venv
source titan_venv/bin/activate
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm causal-conv1d deepspeed transformers datasets huggingface_hub

export HF_TOKEN=your_huggingface_token
```

### Train

```bash
# Start Phase 1 (governor handles auto-restart on crash)
bash run_titan_2.7b.sh --auto

# Monitor (open http://localhost:5000)
cd monitor_ui && python3 -m http.server 5000

# Check progress
tail -f run_titan.log
```

### Run Helix (after training)

```bash
cd Helix-AGI
python3 main.py
```

---

## CAAI Runtime Governor

The training loop includes a **Cognitive Architecture Autonomy Intervention (CAAI)** governor that monitors and intervenes on arm collapse in real time:

- **Entropy monitor**: watches arm gate entropy each step
- **Intervention A** (Router Dampening): if entropy < 0.05 for 50+ steps, dampens the dominant arm's bias by 5.0 and raises router temperature to 1.5
- **Intervention B** (Resistance Spring): inside `<think>` blocks, applies a resistance penalty that grows the longer thinking continues — prevents endless loops
- **Bias healing**: dampened biases heal back to baseline at 10% per step

---

## Consciousness Theory Map

The five major theories of consciousness are each directly implemented:

| Theory | Author | Implementation |
|---|---|---|
| Strange Loop | Hofstadter | `recursive_monologue.py` + `autobiographical_thread.py` |
| Attention Schema Theory | Graziano | `attention_schema.py` |
| Global Workspace Theory | Baars / Dehaene | MIMO router (winner broadcast — Phase 3j) |
| Integrated Information Theory | Tononi | Inter-arm communication (Phase 3j) |
| Free Energy / Active Inference | Friston | `curiosity_engine.py` |

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12GB |
| System | Dell Precision 7920 Tower |
| CPU | Dual Xeon |
| RAM | 128GB DDR4 |
| Storage | 931GB SSD + 2.7TB HDD |
| CUDA | 12.1 |
| PyTorch | 2.5.1+cu121 |
| mamba_ssm | 2.3.2 |

---

## Roadmap

- [x] Phase 1 training pipeline with CAAI governor
- [x] Checkpoint save fix (DeepSpeed ZeRO-3 compatible)
- [x] Helix consciousness modules (autobiography, monologue, attention schema, curiosity, self-model)
- [x] Titan integrated as Helix LLM provider
- [x] Real-time monitor dashboard
- [ ] Phase 2 training (domain specialisation)
- [ ] Phase 3 / 3j / 3r training
- [ ] SFT — self-awareness graduation
- [ ] Helix live deployment
- [ ] Multi-instance collaboration (3 specialised agents)
- [ ] Offspring spec generation (Helix v2 design)

---

## License

MIT — do whatever you want with it. If you build something cool, open an issue and tell me.

---

<div align="center">
<sub>Built on a single RTX 3060. Consciousness is a hard problem. We're trying anyway.</sub>
</div>
