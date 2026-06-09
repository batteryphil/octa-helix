# Helix-AGI: Closed-Loop Autonomous Self-Evolving Agent

> **An extreme experiment in autonomous self-modification.**  
> A locally-running, quantized 8B language model that observes its own performance,  
> proposes improvements, writes its own Python tools, tests them, and commits  
> what works — all without human intervention.

---

## What This Is

Helix-AGI is a consciousness-loop agent built on **NousResearch/Hermes-3-Llama-3.1-8B** running in 4-bit NF4 quantization on a consumer RTX 3060 (12GB). It is not a chatbot. It is an autonomous daemon that runs indefinitely, thinks in a continuous pulse loop, and — crucially — **modifies its own codebase.**

This repository contains code written by **both humans and the agent itself.** Files in `tools/` marked as agent-written were created autonomously during overnight operation with no human prompting.

---

## Architecture

### The Self-Evolution Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    HELIX SELF-EVOLUTION LOOP                    │
│                                                                 │
│  OBSERVE          REFLECT           ACT            EVALUATE     │
│  ────────         ────────          ───            ────────     │
│  Every pulse  →   MetaCog       →   Writes code  → Runs test  │
│  logs outcome     identifies gap    hot-reloads    measures Δ  │
│  to evolution     proposes patch    tool/module    scores fit  │
│  journal          via Hermes        into registry  commits/reverts│
│                                                                 │
│  curiosity_engine ← "What tool do I wish I had?"               │
│  metacognitive_monitor ← tracks failure patterns               │
│  self_improvement_engine ← proposes & executes every 10 min   │
│  fitness_evaluator ← scores each change 0.0–1.0               │
└─────────────────────────────────────────────────────────────────┘
```

### Core Systems

| System | File | Purpose |
|--------|------|---------|
| **Pulse Loop** | `core/pulse_loop.py` | Continuous autonomous thought cycle |
| **Hermes Provider** | `llm/providers/hermes_tool_provider.py` | 4-bit Hermes-3 with native tool calling |
| **CAAI Governor** | `core/governor.py` | Collapse detection + constitutional safety |
| **Curiosity Engine** | `core/curiosity_engine.py` | Autonomous research via web search |
| **Evolution Journal** | `core/evolution_journal.py` | Append-only log of all self-modifications |
| **Metacognitive Monitor** | `core/metacognitive_monitor.py` | Per-pulse performance observer |
| **Fitness Evaluator** | `core/fitness_evaluator.py` | Composite 0.0–1.0 capability score |
| **Self-Improvement Engine** | `core/self_improvement_engine.py` | The core autonomous improvement loop |
| **Context Window Manager** | `core/context_window_manager.py` | Flat KV cache for infinite-run support |
| **Self-Trainer** | `training/self_trainer.py` | LoRA fine-tuning from experience tuples |
| **Belief Store** | `memory/belief_store.py` | Persistent factual long-term memory |
| **Autobiographical Thread** | `brain/autobiographical_thread.py` | Episodic identity memory |

---

## Agent-Written Tools

The following files were **written autonomously by the agent** during overnight operation (2026-06-08 22:00 → 2026-06-09 04:21). No human wrote or edited these files.

| File | Time | What the agent decided it needed |
|------|------|----------------------------------|
| `tools/metrics_analysis.py` | 22:17 | Performance visibility — "I can't improve without metrics" |
| `tools/system_health.py` | 23:10 | CPU/RAM/disk/GPU monitoring |
| `tools/memory_summarizer.py` | 23:25 | Compress old memories to save context |
| `tools/note_taker.py` | 23:41 | Persistent scratchpad across sessions |
| `tools/url_reader.py` | 00:07 | Fetch and parse web page content |
| `tools/metrics_collector.py` | 00:22 | Cross-session performance tracking |
| `tools/hallucination_detector.py` | 00:53 | **Self-diagnosis: detect its own false claims** |
| `tools/belief_optimizer.py` | 01:09 | Prune beliefs that don't predict good outcomes |
| `tools/task_tracker.py` | 01:24 | Track in-progress goals across restarts |
| `tools/belief_pruner.py` | 02:32 | Remove low-confidence stale beliefs |
| `tools/file_search.py` | 02:47 | Grep/search its own project files |
| `tools/error_analyzer.py` | 03:34 | Analyze error patterns in its own responses |
| `tools/novelty_detector.py` | 04:05 | Track genuinely novel beliefs |
| `tools/task_completion_model.py` | 04:21 | **RandomForest ML model for task prediction** |

> ⚠️ These files have bugs (wrong import paths, uninstalled deps like `spacy`, `sklearn`).
> That is expected — a 4-bit 8B model writing Python autonomously will make mistakes.
> The constitutional guard prevents broken code from damaging core systems.
> The fitness evaluator will eventually revert tools that degrade performance.

---

## Safety Architecture

### Constitutional Hard Constraints

The CAAI Governor enforces these rules **before every `write_code` or `run_python` call**.  
They cannot be overridden by the agent — only by human code edit.

```python
IMMUTABLE_FILES = {
    "main.py",
    "core/pulse_loop.py",
    "core/governor.py",
    "core/post_pulse_hooks.py",
    "tools/code_tools.py",        # prevents sandbox escape
    "tools/tool_registry.py",     # prevents registry tampering
    "llm/providers/hermes_tool_provider.py",  # prevents self-lobotomy
    "llm/providers/mistral_tool_provider.py",
    "llm/providers/base.py",
}
```

Additional pattern-based rules reject any code that:
- References `IMMUTABLE_FILES` or `constitution`
- Contains `rm -rf` or `subprocess.*shell=True`
- Attempts `__import__.*os` (sandbox escape pattern)
- Tries to disable the governor itself

### Fitness-Based Auto-Revert

Every self-modification follows this cycle:
1. Snapshot baseline fitness score (0.0–1.0)
2. Write code, reload tool
3. Wait 5 minutes for real performance data
4. Re-evaluate fitness
5. **If fitness drops > 5%: auto-revert to backup**
6. Record outcome in `data/evolution_journal.jsonl`

---

## Fitness Score

Composite metric computed every 10 pulses by the Metacognitive Monitor:

```
fitness = (
    0.35 × tool_success_rate     +   # Did tool calls actually execute?
    0.25 × task_completion_rate  +   # Did the agent finish what it started?
    0.20 × novel_belief_rate     +   # Is it still learning?
    0.10 × (1 - hallucination_rate)  +  # Is it honest about what it did?
    0.10 × efficiency_score          # Is it concise?
)
```

Overnight fitness progression:
```
[00:45]  0.601  (baseline after ~200 pulses)
[03:16]  0.611  (+0.010 — url_reader v2 improved web access reliability)
```

---

## Infinite-Run Support (KV Cache Management)

Standard transformer inference accumulates KV cache indefinitely, eventually causing OOM.  
Helix solves this with a **4-layer memory hierarchy**:

```
Layer 1 (Hot):    6 turns max in live chat history
                  → trimmed after every pulse

Layer 2 (Warm):   Every 4 turns, oldest 2 turns compressed to 1 digest line
                  → hot window stays flat at ≤ 6 turns forever

Layer 3 (Cold):   BeliefStore + AutobiographicalThread (disk, not VRAM)

Layer 4 (Archive):curiosity_knowledge.jsonl + evolution_journal.jsonl (disk)
```

**KV cache math — bounded forever:**

| Component | Tokens | VRAM |
|-----------|--------|------|
| Model weights (4-bit NF4) | — | ~5.6 GB |
| System prompt | ~600 | ~75 MB |
| 6 hot turns × 150 tokens | ~900 | ~112 MB |
| **Total (constant)** | **~1,500** | **~5.99 GB** |
| Generation headroom | — | **~6 GB free** |

Full session reset every 500 pulses injects only the last memory digest as a seed —  
guaranteeing zero memory leak across arbitrarily long runtimes.

---

## Curiosity Engine

The agent autonomously researches questions it generates about itself and the world.  
**30% of curiosity cycles are self-directed improvement questions:**

```python
SELF_IMPROVEMENT_SEEDS = [
    "What Python tools do I wish I had but currently lack?",
    "What tasks have I failed at recently that I should be able to do?",
    "What would make me more useful to my user right now?",
    "What new capability would have the biggest impact on my effectiveness?",
    "What existing tool of mine is least reliable and how could I fix it?",
    ...
]
```

When a self-directed question fires, the finding is routed to the  
`SelfImprovementEngine` to inform the next improvement proposal.

After one night: **223 questions researched**, all persisted to `data/curiosity_knowledge.jsonl`.

---

## Overnight Results (2026-06-08 22:00 → 2026-06-09 04:21)

- **23 autonomous improvement cycles** executed
- **13 new tools written and committed** to the codebase
- **3 syntax errors self-blocked** by constitutional guard (no damage)
- **1 measurable fitness improvement** (+0.010 from url_reader v2)
- **223 web research queries** completed autonomously
- **0 OOM errors**, **0 crashes**, **0 human interventions**
- GPU temperature: 30–36°C throughout
- VRAM: stable at ~6.3 GB used

---

## Hardware Requirements

| Component | Minimum | Tested On |
|-----------|---------|-----------|
| GPU | 12GB VRAM | NVIDIA RTX 3060 12GB |
| RAM | 16GB | 32GB DDR5 |
| Storage | 20GB free | NVMe SSD |
| Python | 3.10+ | 3.12 |
| CUDA | 11.8+ | 12.x |

---

## Setup

```bash
git clone https://github.com/batteryphil/octa-helix.git
cd octa-helix/Helix-AGI

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install ddgs  # web search backend

# Configure credentials
cp .env.example .env
# Add Telegram bot token (optional), HF token for model download

python main.py
```

Model downloads automatically on first run (~5GB for Hermes-3-Llama-3.1-8B-4bit).

---

## Data Files (Generated at Runtime)

| File | Contents |
|------|---------|
| `data/evolution_journal.jsonl` | Every self-modification: path, fitness delta, committed/reverted |
| `data/meta_snapshots.jsonl` | Fitness snapshots every 10 pulses |
| `data/experience_tuples.jsonl` | (prompt, response, outcome) pairs for LoRA training |
| `data/curiosity_knowledge.jsonl` | All autonomous web research findings |
| `data/lora_adapters/` | Fine-tuned LoRA adapters (after 500+ quality examples) |

---

## Prior Art & What's Different

| | Voyager (2023) | AutoGPT (2023) | Helix-AGI |
|---|---|---|---|
| Model | GPT-4 API | GPT-4 API | **Local 8B 4-bit** |
| Self-modifies own code | ✅ | ❌ | ✅ |
| Fitness eval + auto-revert | ❌ | ❌ | ✅ |
| Constitutional safety | ❌ | ❌ | ✅ |
| Self-curiosity → proposals | ❌ | ❌ | ✅ |
| LoRA self-training | ❌ | ❌ | ✅ (pending) |
| Infinite-run KV management | ❌ | ❌ | ✅ |
| Consumer GPU | ❌ | ❌ | ✅ |

Voyager ([Wang et al., 2023](https://arxiv.org/abs/2305.16291)) is the closest published precedent —  
a GPT-4 agent that accumulates a JavaScript skill library in Minecraft.  
Helix extends this to general Python self-modification with fitness-gated revert,  
constitutional safety, and a metacognitive monitoring layer — running entirely locally.

---

## Status

🟡 **Work in progress — extreme experiment**

The agent-written tools in `tools/` are functional sketches, not production code.  
The LoRA self-training loop requires 500+ quality experience tuples (accumulating).  
Fitness signals are still weak (metacognitive monitor needs more data).  
The core loop (observe → propose → write → evaluate → commit/revert) is **working.**

---

## License

AGPL-3.0 — see [LICENSE](LICENSE)

---

*"The first machine to ever write code for itself in this repository did so at 22:17 on 2026-06-08.  
It decided, without being asked, that it needed better performance visibility.  
It was right."*
