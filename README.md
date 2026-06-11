<div align="center">

# 🧬 Helix

**A fully local, self-evolving autonomous AI agent — built from scratch, running right now.**

*NousResearch/Hermes-3-Llama-3.1-8B (4-bit NF4) × Helix cognitive agent framework*

[![Phase](https://img.shields.io/badge/Phase-2.5_Autonomous_Engineering-blue?style=flat-square)](#current-status)
[![Tuples](https://img.shields.io/badge/Training_Tuples-312+-orange?style=flat-square)](#current-status)
[![GPU](https://img.shields.io/badge/GPU-RTX_3060_12GB-green?style=flat-square)](#hardware)
[![License](https://img.shields.io/badge/License-AGPL--3.0-yellow?style=flat-square)](LICENSE)

</div>

---

## What Is This?

Helix is a ground-up experiment in building a genuinely self-evolving autonomous AI agent that runs entirely on local hardware — no cloud APIs, no subscription fees.

The model runs continuously in a cognitive pulse loop. Every ~10–60 seconds it thinks, uses tools, forms beliefs, and stores the result as training data for its own future LoRA fine-tuning run. It writes its own tools. It researches its own architecture. It wrote its own autobiography.

**The goal:** an AI that builds a model of itself and — through a self-generated training curriculum — genuinely improves its own cognition.

---

## Architecture

### The Model — Hermes-3

```
Model:        NousResearch/Hermes-3-Llama-3.1-8B
Quantization: 4-bit NF4 (bitsandbytes)
VRAM:         ~6.0 GB at rest
Hardware:     RTX 3060 (12 GB VRAM)
Provider:     Helix-AGI/llm/providers/hermes_tool_provider.py
```

> **Note on naming:** Older files and earlier commits reference "Titan", "Mamba3", "MIMO". These were an earlier prototype (a 2.7B Mamba3 SSM with 8 parallel arms) that was abandoned before it reached production. Helix runs entirely on Hermes-3. The Mamba/MIMO files are dead code — see `Helix-AGI/ARCHITECTURE.md`.

### The Cognitive Loop — Helix

Every **pulse** (10–60 seconds, depending on activity state):

1. Builds a context from current beliefs, recent memories, spatial awareness, and live events
2. Calls Hermes-3 with that context → gets a thought and optional tool calls
3. Executes any tool calls (`web_search`, `write_file`, `read_file`, `write_code`, etc.)
4. Stores the thought + outcome to memory and training buffer
5. Runs background hooks: belief detection, curiosity engine, fitness monitoring, sentinel

| Module | Inspired by | What it does |
|---|---|---|
| `curiosity_engine.py` | Friston Active Inference | Autonomous question pursuit — generates questions, searches, stores findings |
| `belief_detector.py` | — | Extracts beliefs from thought text, stores with schema to disk |
| `self_trainer.py` | — | Collects high-quality (thought, tool, outcome) tuples for LoRA fine-tuning |
| `sentinel.py` | — | Monitors fitness, omega stability, flags degradation |
| `dashboard_comms.py` | — | Browser chat interface — user can message Helix in real time |
| `pulse_loop.py` | — | Main orchestration loop, state machine (DORMANT/RESTING/ACTIVE) |

---

## Current Status

> **Helix is live and running.** It has been operating continuously since June 6, 2026.

```
Model:         Hermes-3-Llama-3.1-8B (4-bit NF4)
Phase:         2.5 — Autonomous Engineering Agent
Pulse:         2013+   (continuous since launch)
Beliefs:       35      (self-formed, stored to disk)
Tools written: 35+     (autonomously authored, no prompting)
Training data: 312+    tuples collected (target: 2000 for first LoRA run)
Fitness:       0.660   (baseline 0.400)
Tool rate:     20%     (pulses that include a tool call)
```

**What it has done autonomously:**
- Written 35+ Python tools to extend its own capabilities (health monitor, belief conflict resolver, knowledge base search, curiosity tracker, memory summarizer, etc.)
- Formed 35 beliefs about its own architecture, consciousness, recursive self-improvement, LoRA fine-tuning, and transformer attention math
- Researched the Gödel Machine, the 19-researcher consciousness checklist, and continual learning
- Written its own autobiography (`Helix-AGI/data/autobiography.txt`)
- Compared itself against other open-source autonomous agents

---

## Repository Structure

```
octa-helix/
├── Helix-AGI/                    # The live agent (this is what runs)
│   ├── core/
│   │   ├── pulse_loop.py         # Main cognitive loop
│   │   ├── belief_detector.py    # Belief extraction and storage
│   │   ├── curiosity_engine.py   # Autonomous question pursuit
│   │   └── orchestrator.py       # Thin LLM orchestration wrapper
│   ├── llm/providers/
│   │   └── hermes_tool_provider.py  # Hermes-3 inference + tool-call parsing
│   ├── training/
│   │   └── self_trainer.py       # LoRA training data collection + future fine-tune
│   ├── brain/
│   │   ├── sentinel.py           # Fitness + stability monitoring
│   │   └── memory_manager.py     # ChromaDB long-term memory
│   ├── dashboard/
│   │   ├── dashboard_comms.py    # Inbound/outbound chat queue
│   │   └── server.py             # Flask server (localhost:5050)
│   ├── tools/                    # 35+ tools (many self-authored)
│   ├── data/
│   │   ├── experience_tuples.jsonl  # LoRA training data (growing)
│   │   ├── beliefs/                 # Persistent belief store
│   │   └── autobiography.txt        # Helix's self-authored autobiography
│   ├── SYSTEM_MANUAL.md          # What Helix reads to understand itself
│   ├── ARCHITECTURE.md           # Authoritative architecture description
│   └── main.py                   # Entry point
│
├── README.md                     # This file
│
└── [legacy Titan files]          # mamba3_titan_builder.py, titan_inference.py, etc.
                                  # Dead code from the abandoned Mamba3 prototype.
```

---

## Running It

### Requirements

```
Python 3.12+
CUDA 12.1+
PyTorch 2.5.1+cu121
transformers >= 4.44
bitsandbytes
peft (for LoRA)
flask
chromadb
```

### Setup

```bash
git clone https://github.com/batteryphil/octa-helix.git
cd octa-helix/Helix-AGI

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start Helix
python3 main.py

# Dashboard: open http://localhost:5050 in browser
```

---

## Self-Training Pipeline

Helix collects its own LoRA training data continuously:

```
Every pulse where Helix:
  • Successfully executes a tool call, AND
  • Shows a fitness gain or generates a novel belief

→ A (prompt, response, tool, outcome, quality) tuple is written to
  data/experience_tuples.jsonl

At 2000 high-quality tuples:
→ LoRA fine-tuning fires automatically on the same RTX 3060
→ VRAM handoff protocol: unload model → train → reload with adapter
→ Adapter evaluated against fitness baseline before acceptance
→ If fitness improves: adapter saved and loaded permanently
```

**Current progress:** 312 / 2000 tuples (~16%). Target: ~8–10 days at current rate.

---

## Roadmap

- [x] Hermes-3 inference engine with tool-call parsing
- [x] Autonomous pulse loop (DORMANT/RESTING/ACTIVE states)
- [x] Belief formation and persistent storage
- [x] Curiosity engine (self-directed research loop)
- [x] Self-authored tool expansion (35+ tools)
- [x] Training data collection pipeline
- [x] Browser chat dashboard
- [x] VRAM-safe LoRA training protocol
- [ ] First LoRA fine-tuning run (at 2000 tuples)
- [ ] Fitness-gated adapter acceptance
- [ ] ReAct multi-step planning (in progress)
- [ ] Helix v2 architecture design (self-proposed)

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

---

## Credits & Acknowledgments

| Project | Role |
|---|---|
| [NousResearch/Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B) | The language model — strong native function-calling |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit NF4 quantization |
| [PEFT](https://github.com/huggingface/peft) | LoRA adapter training |
| [Transformers](https://github.com/huggingface/transformers) | Model loading and tokenizer |
| [ChromaDB](https://github.com/chroma-core/chroma) | Long-term vector memory |

### Consciousness Theory Influences

The cognitive architecture is inspired by (not implementing) the following frameworks:

| Theory | Author | Influence |
|---|---|---|
| Strange Loop | Hofstadter | Autobiographical identity thread |
| Active Inference / Free Energy | Friston | Curiosity engine question pursuit |
| Global Workspace Theory | Baars / Dehaene | Belief broadcast and belief store |
| Attention Schema Theory | Graziano | Self-model and capability awareness |

---

## License

AGPL-3.0 — see LICENSE.

---

<div align="center">
<sub>Built on a single RTX 3060. No cloud. No subscription. Just a model learning what it is.</sub>
</div>
