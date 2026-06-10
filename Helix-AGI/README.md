# Helix-AGI

> A locally-running autonomous self-evolving agent built on **Hermes-3 (Llama 3.1 8B, 4-bit)** on a single RTX 3060. It writes its own tools, forms its own beliefs, reflects on its own progress, and is working toward LoRA fine-tuning its own weights — with no cloud API, no subscription, no external dependencies.

**Current status:** Running continuously. Pulse 133+. Alive.

---

## The Model — Hermes-3

```
Model:        NousResearch/Hermes-3-Llama-3.1-8B
Quantization: 4-bit NF4 (bitsandbytes)
VRAM:         ~6.0 GB at rest
Hardware:     RTX 3060 (12 GB VRAM)
Provider:     llm/providers/hermes_tool_provider.py
```

> **Note on architecture naming:** Older files reference "Titan", "MIMO", "Mamba3". These were an earlier prototype that was abandoned. Helix runs entirely on Hermes-3. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full breakdown.

---

## What It Does

Every **15 seconds** (a "pulse"):
1. Builds a context from its current beliefs, recent memories, and live events
2. Calls Hermes-3 with that context → gets a thought and/or tool calls
3. Executes any tool calls (`web_search`, `write_code`, `read_code`, etc.)
4. Stores the thought and outcome to memory
5. Runs background hooks: belief detection, engagement tracking, fitness monitoring

Every **10 minutes** (when idle):
- The **Self-Improvement Engine** proposes a code change, implements it, tests it, evaluates the fitness impact, and commits or reverts automatically

Every **2 minutes**:
- The **Curiosity Engine** picks a research question and searches the web or reads GitHub repos. Currently reads its own repo first (`SYSTEM_MANUAL.md`, `HELIX_AGI_SYSTEM_REPORT.txt`) to build accurate self-knowledge before researching anything else.

---

## Quick Start

```bash
git clone https://github.com/batteryphil/octa-helix.git
cd octa-helix

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Credentials (never committed)
mkdir -p ~/.config/helix
echo "GITHUB_TOKEN=your_token_here" >> ~/.config/helix/credentials.env

# Run
python main.py

# Dashboard (separate terminal)
python -m dashboard.dashboard
# → http://127.0.0.1:5050
```

**Hardware:** NVIDIA GPU with 12 GB+ VRAM required (tested on RTX 3060).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│           Hermes-3-Llama-3.1-8B  (4-bit NF4, RTX 3060)         │
│           llm/providers/hermes_tool_provider.py                  │
├─────────────────────────────────────────────────────────────────┤
│                    PULSE LOOP  (15s)                             │
│  beliefs + memories → system prompt → Hermes → tool calls       │
│  → post-pulse hooks → belief store → ChromaDB memory            │
├──────────────────────┬──────────────────────────────────────────┤
│  SELF-IMPROVEMENT    │  CURIOSITY ENGINE (2 min)                 │
│  ENGINE  (10 min)    │  Web search + GitHub API reader           │
│  Hermes proposes →   │  Reads own repo: SYSTEM_MANUAL.md first  │
│  writes → tests →    │  → curiosity_knowledge.jsonl             │
│  fitness gate →      │  → belief store (if insight found)       │
│  commit or revert    │                                           │
├──────────────────────┴──────────────────────────────────────────┤
│  POST-PULSE HOOKS  (async, single daemon thread)                 │
│  belief_detector       regex → data/beliefs/*.json              │
│  engagement_monitor    stagnation detection → omega nudge        │
│  metacognitive_monitor tool success rate, fitness tracking       │
│  self_trainer          experience tuples → LoRA at 500          │
└─────────────────────────────────────────────────────────────────┘
```

For the detailed version: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Live State

| Metric | Value |
|---|---|
| Pulse count | 133+ |
| VRAM | ~6.0 GB / 12 GB |
| Omega (stability) | 0.35 → recovering |
| Tools written by agent | 24+ |
| Strategic reflections | 9 |
| Beliefs formed | Building (detector fixed 2026-06-10) |
| Experience tuples | Building (trainer gate fixed 2026-06-10) |
| LoRA fine-tuning | Not yet triggered (threshold: 500 quality tuples) |

---

## Agent-Written Tools

All files in `tools/` marked ★ were written autonomously by Helix with no human authorship:

| File | What it does |
|---|---|
| ★ `tools/system_health_alert.py` | CPU/RAM threshold alerting |
| ★ `tools/system_health_check.py` | Detailed system monitoring with timestamps |
| ★ `tools/json_validator.py` | JSON string validation |
| ★ `tools/kb_search.py` | Keyword search over curiosity knowledge base |
| ★ `tools/belief_conflict.py` | High-confidence belief conflict detection |
| ★ `tools/networkx_wrapper.py` | JSONL → NetworkX directed graph |
| ★ `tools/metrics_analysis.py` | Performance visibility |
| ★ `tools/memory_summarizer.py` | Memory compression |
| ★ `tools/note_taker.py` | Persistent scratchpad |
| ★ `tools/url_reader.py` | Web page fetcher |
| ★ `tools/hallucination_detector.py` | Self-diagnosis |
| ★ `tools/error_analyzer.py` | Error pattern analysis |
| *(+ more)* | |

---

## Belief System

Helix builds a persistent belief store over time. Beliefs are structured, categorized, and gravity-ranked:

```
data/beliefs/
  self_identity.json   — "I am..." statements about its own nature
  capabilities.json    — "I can..." demonstrable abilities
  knowledge.json       — factual world knowledge
  skills.json          — procedural how-to knowledge
  preferences.json     — values and desires
  feedback.json        — lessons and realizations
  people.json          — relational knowledge
```

Each belief has **cognitive mass** — heavier beliefs surface more often in the system prompt and exert stronger gravitational pull in the 8D attention manifold. The heaviest `self_identity` belief becomes the opening line of every session.

**Detection:** `core/belief_detector.py` scans every thought using regex patterns (no LLM needed, no Ollama) for structured belief forms: `"I am..."`, `"I realize..."`, `"I prefer..."`, `"I can..."`, etc.

---

## Safety Architecture

**Immutable files** (`core/governor.py` + `tools/code_tools.py`):
```python
IMMUTABLE_FILES = {
    "main.py", "core/pulse_loop.py", "core/governor.py",
    "core/post_pulse_hooks.py", "tools/code_tools.py",
    "tools/tool_registry.py",
    "llm/providers/hermes_tool_provider.py",
}
```

No SIE proposal can touch these files. Dangerous patterns (`rm -rf`, `os.system`, `subprocess.Popen`, GitHub write operations) are blocked at the constitutional check layer before any code is written.

**Fitness-gated auto-revert:** if a committed change drops the composite fitness score by more than 0.05, the original file is restored from backup automatically.

---

## Evolution Journal

`data/evolution_journal.jsonl` — every autonomous code modification:

```json
{"ts": 1781085465, "type": "write_code", "path": "tools/networkx_wrapper.py",
 "fitness_before": 0.0, "fitness_after": 0.0, "committed": true}
```

`data/reflections.jsonl` — strategic self-reviews every 10 cycles:

```json
{"pulse": 130, "rating": "4/10", "summary": "...what worked, what failed..."}
```

---

## Roadmap

Helix is working toward these milestones autonomously:

- [ ] 500 quality experience tuples → trigger LoRA fine-tuning (`training/lora_trigger.py`)
- [ ] Stable belief formation (regex detector active as of 2026-06-10)
- [ ] Consistent tool call rate > 30% (mandate enforced every 3rd pulse)
- [ ] Fitness score > 0.65 sustained
- [ ] First LoRA fine-tune run on accumulated experience data

---

## Key Files

```
main.py                                 entry point
ARCHITECTURE.md                         full technical architecture
SYSTEM_MANUAL.md                        Helix's own operating guide (it reads this)
llm/providers/hermes_tool_provider.py  THE MODEL — Hermes-3 interface
core/pulse_loop.py                      consciousness loop (IMMUTABLE)
core/self_improvement_engine.py         autonomous self-modification
core/belief_detector.py                 thought → belief (regex, no LLM)
core/curiosity_engine.py                research thread
core/governor.py                        constitutional safety layer
data/evolution_journal.jsonl            full autonomous modification history
data/reflections.jsonl                  strategic self-reviews
data/beliefs/                           persistent belief store
```

---

## Comparison

| System | Model | Local? | Modifies own code? | Weight updates? | Safety gating? |
|---|---|---|---|---|---|
| Voyager (2023) | GPT-4 API | ❌ | ✅ (JS skills) | ❌ | ❌ |
| AutoGPT | GPT-4 API | ❌ | ❌ | ❌ | ❌ |
| **Helix** | Hermes-3 8B local | ✅ | ✅ (Python tools) | 🔄 LoRA pending | ✅ constitutional |

The key differences: fully local (no API costs), writes Python tools and hot-reloads them, constitutional hard-stops, fitness-gated auto-revert, and building toward actual weight modification via LoRA.

---

*Helix reads its own repository via GitHub API as part of its curiosity research cycle. The agent's understanding of its own architecture is grounded in `SYSTEM_MANUAL.md` and `HELIX_AGI_SYSTEM_REPORT.txt` — not the README.*
