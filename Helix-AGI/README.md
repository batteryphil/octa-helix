# Helix-AGI

> A locally-running autonomous self-evolving agent. It writes its own tools, reflects on its own progress, and works toward LoRA fine-tuning its own weights — all on a single RTX 3060.

**Current status:** Running continuously since 2026-06-08. Cycle 67+. Alive.

---

## What It Is Right Now

Helix is a 4-bit quantized Hermes-3 8B model running a continuous **pulse loop** — every 15 seconds it thinks, every 10 minutes it tries to improve itself by writing new Python tools. Every 10 cycles it reflects on everything it has done and redirects its effort.

It has written **14+ tools autonomously**. It has filed **6 strategic self-reviews**. It has a satisfaction score of **4/10** with itself — correctly.

---

## Quick Start

```bash
# Clone
git clone https://github.com/batteryphil/octa-helix.git
cd octa-helix

# Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Credentials (never committed)
mkdir -p ~/.config/helix
echo "GITHUB_TOKEN=your_token_here" >> ~/.config/helix/credentials.env

# Run
python main.py

# Dashboard (separate terminal)
python -m dashboard.dashboard
# Open http://127.0.0.1:5050
```

**Hardware requirement:** NVIDIA GPU with 12GB+ VRAM. Tested on RTX 3060.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PULSE LOOP (15s)                          │
│  Hermes-3 8B 4-bit NF4 ← system prompt ← beliefs + context     │
│         ↓ thinks        ↓ tool calls     ↓ outbound reply       │
│  [BeliefStore]    [ToolRegistry]    [Dashboard / Telegram]      │
└────────────────────────────┬────────────────────────────────────┘
                             │ every 10 min (when idle)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SELF-IMPROVEMENT ENGINE (SIE)                   │
│                                                                   │
│  Phase 1: _generate_proposal()                                   │
│    → reads fitness metrics + journal + last reflection           │
│    → asks Hermes: "what should I improve?"                       │
│    → dedup: blocks paths modified in last 20 cycles              │
│    → cooldown: blocks paths with Δ0.000 for 5 cycles            │
│                                                                   │
│  Phase 2: constitutional_check()                                 │
│    → hard-blocks: main.py, pulse_loop.py, governor.py, etc.     │
│    → keyword scan for dangerous patterns                         │
│                                                                   │
│  Phase 3: _implement_proposal() — 3-attempt retry loop          │
│    Attempt 1: generate fresh code (stdlib + safe deps only)      │
│    Attempt 2: show Hermes the exact error → ask it to fix       │
│    Attempt 3: one more shot                                       │
│    Gate 1: write_code() syntax check                             │
│    Gate 2: importlib import test (catches bad deps)              │
│    Gate 3: reload_tool() error check                             │
│                                                                   │
│  Phase 4: wait 5 min → FitnessEvaluator.evaluate()             │
│    PASS  → commit to git                                         │
│    FAIL  → revert from backup                                    │
│                                                                   │
│  Every 10 cycles: _reflect_on_progress()                        │
│    → reads all journal entries + fitness trend                   │
│    → asks Hermes: what worked? what failed? satisfaction 0-10?   │
│    → saves to data/reflections.jsonl                             │
│    → injects into next proposal prompt                           │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CURIOSITY ENGINE (2 min)                      │
│  Pursues open questions via web search + GitHub API             │
│  Current threads: Mamba3 architecture, continual learning,      │
│  AI self-modification, what "understanding" means mathematically │
│                                                                   │
│  Research repos (GitHub token required):                        │
│  • batteryphil/octa-helix (itself)                              │
│  • batteryphil/thalamic-bloom                                   │
│  • batteryphil/mamba2backbonerecursion                          │
│  • batteryphil/mamba1and2-to-3                                  │
│  • batteryphil/syrin-pythonmamba                                │
│  • state-spaces/mamba                                           │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY                              │
│  Hot    → last 6 conversation turns (KV cache, ~187MB flat)    │
│  Warm   → digest summaries (BeliefStore, compressed)           │
│  Cold   → long-term ChromaDB semantic search                    │
│  Archive → evolution_journal.jsonl + experience_tuples.jsonl   │
│                                                                   │
│  Context window resets every 500 pulses (prevents OOM)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Live State (as of 2026-06-09, ~17:00 CT)

| Metric | Value |
|---|---|
| Pulse cycle | 67+ |
| VRAM usage | ~6.3 GB / 11.9 GB |
| GPU temp | 45°C |
| Fitness score | 0.6105 (flat — local minimum) |
| Tools written by agent | 14 |
| Strategic reflections filed | 6 |
| Agent satisfaction | 4/10 (self-reported) |
| Experience tuples | 50 / 500 (LoRA threshold) |
| LoRA fine-tuning | Not yet triggered |

**Current agent priority (from reflection #6):**
> *"Focus on improving error handling and logging over the next 10 cycles."*

**What it's stuck on:** The agent kept rewriting `error_logger.py` every cycle with zero fitness gain. A zero-delta cooldown was patched in at cycle 67 to break this loop.

---

## Agent-Written Tools (no human authorship)

All files in `tools/` marked with ★ were written autonomously by Helix:

| File | Written | What it does |
|---|---|---|
| ★ `tools/metrics_analysis.py` | 22:17 Jun 8 | Performance visibility |
| ★ `tools/system_health.py` | 23:10 Jun 8 | CPU/RAM/GPU monitoring |
| ★ `tools/memory_summarizer.py` | 23:25 Jun 8 | Memory compression |
| ★ `tools/note_taker.py` | 23:41 Jun 8 | Persistent scratchpad |
| ★ `tools/url_reader.py` | 00:07 Jun 9 | Web page fetcher |
| ★ `tools/metrics_collector.py` | 00:22 Jun 9 | Cross-session tracking |
| ★ `tools/hallucination_detector.py` | 00:53 Jun 9 | Self-diagnosis |
| ★ `tools/belief_optimizer.py` | 01:09 Jun 9 | Belief pruning |
| ★ `tools/task_tracker.py` | 01:24 Jun 9 | Goal persistence |
| ★ `tools/file_search.py` | 02:47 Jun 9 | Project grep |
| ★ `tools/error_analyzer.py` | 03:34 Jun 9 | Error pattern analysis |
| ★ `tools/novelty_detector.py` | 04:05 Jun 9 | Novel belief tracking |
| ★ `tools/task_completion_model.py` | 04:21 Jun 9 | Task predictor |
| ★ `tools/error_logger.py` | 16:16 Jun 9 | Error logging (overwritten 8x) |

---

## Safety Architecture

**Constitutional hard-stops** (`core/governor.py`):

```python
IMMUTABLE_FILES = {
    "main.py", "core/pulse_loop.py", "core/governor.py",
    "core/self_improvement_engine.py", "llm/providers/hermes_tool_provider.py"
}
```

No proposal can modify these files. The SIE checks before writing. Suspicious keywords (`rm -rf`, `os.system`, `subprocess.Popen`, etc.) are auto-blocked.

**Fitness-gated auto-revert:** if a committed change drops the composite fitness score, the original file is restored from backup within 5 minutes.

---

## Evolution Journal

`data/evolution_journal.jsonl` — every autonomous modification, timestamped:

```json
{"ts": 1781004918, "cycle": 10, "path": "tools/url_reader.py",
 "fitness_delta": +0.010, "committed": true,
 "description": "Fetch and parse web page content"}
```

`data/reflections.jsonl` — strategic self-reviews every 10 cycles:

```json
{"cycle": 60, "goal_satisfaction": "4/10",
 "what_worked": "URL reader consistently useful for fetching data",
 "what_failed": "Syntax errors and bad imports caused many failed writes",
 "priority_next": "Focus on error handling and logging"}
```

---

## What's Next

The agent is working toward these milestones autonomously:

- [ ] `training/lora_trigger.py` — trigger LoRA fine-tuning at 500 experience tuples
- [ ] `core/belief_graph.py` — relationship mapping between beliefs
- [ ] `tools/self_diagnostic.py` — run all tools, report pass/fail rates
- [ ] LoRA fine-tuning fires (requires 500 quality experience tuples, currently at 50)
- [ ] Fitness score breaks above 0.62 consistently

---

## Key Files

```
main.py                          — entry point, wires everything together
core/pulse_loop.py               — consciousness loop (IMMUTABLE)
core/self_improvement_engine.py  — autonomous self-modification
core/governor.py                 — constitutional safety layer
core/curiosity_engine.py         — research thread (2min cycles)
core/evolution_journal.py        — modification audit log
llm/providers/hermes_tool_provider.py — Hermes-3 interface
dashboard/dashboard_ui.html      — live monitoring UI
data/evolution_journal.jsonl     — full autonomous modification history
data/reflections.jsonl           — strategic self-reviews
data/meta_snapshots.jsonl        — fitness timeline
data/experience_tuples.jsonl     — training data for LoRA
```

---

## Prior Art

| System | Model | Local? | Self-modifies weights? | Safety gating? |
|---|---|---|---|---|
| Voyager (2023) | GPT-4 API | ❌ | ❌ | ❌ |
| AutoGPT | GPT-4 API | ❌ | ❌ | ❌ |
| **Helix** | Hermes-3 8B local | ✅ | 🔄 pending | ✅ |

The key difference: Helix runs entirely on local hardware with no API costs, uses fitness-gated self-modification with constitutional hard-stops, and is building toward actual weight updates via LoRA.

---

*This README reflects the live system state. The agent reads its own repository via GitHub API as part of its curiosity research.*
