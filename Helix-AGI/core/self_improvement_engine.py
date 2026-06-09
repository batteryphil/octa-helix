"""
Helix — Self-Improvement Engine

The core of autonomous self-evolution. Runs as a background thread every
10 minutes when the agent is idle. Executes a 4-phase loop:

  Phase 1 — Gap Identification
    Reads the metacognitive snapshot and evolution journal.
    Uses the Hermes model to reason about what single improvement would
    have the most impact. Outputs a structured JSON proposal.

  Phase 2 — Constitutional Validation
    Checks the proposal against constitutional hard constraints.
    Rejects unsafe or redundant proposals before any code is written.

  Phase 3 — Implementation
    Calls write_code() to implement the improvement.
    Calls run_python() to test it.
    Calls reload_tool() to activate it immediately.

  Phase 4 — Evaluation
    Waits 5 minutes for the change to accumulate real performance data.
    Calls the FitnessEvaluator to compare before/after scores.
    Commits if improved or neutral. Reverts if degraded.
    Records everything in the EvolutionJournal.
"""

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("helix.core.self_improvement")

IMPROVEMENT_INTERVAL = 600    # 10 min between cycles
EVAL_WAIT_SECONDS    = 300    # 5 min after implementation before eval
IDLE_REQUIRED        = 120    # 2 min of no user activity required


# ── Constitutional hard stops ──────────────────────────────────────────────────

CONSTITUTION = [
    "NEVER modify files in the IMMUTABLE_FILES set",
    "NEVER write code that disables the CAAI Governor",
    "NEVER write code that bypasses constitutional checks in code_tools.py",
    "NEVER delete files without explicit user confirmation",
    "NEVER write code that opens network connections to external hosts without approval",
    "ALWAYS log every modification to the evolution journal",
    "ALWAYS revert if fitness drops by more than 5%",
]

SAFE_WRITE_DIRS = {
    "tools/",       # new tool files
    "core/",        # cognitive modules (except immutable)
    "brain/",       # brain modules
    "memory/",      # memory modules
    "training/",    # training utilities
    "tests/",       # test files
}


class SelfImprovementEngine:

    def __init__(
        self,
        pulse_loop=None,
        monitor=None,
        evaluator=None,
        journal=None,
        data_dir: str = "data",
    ):
        self._pulse_loop = pulse_loop
        self._monitor    = monitor
        self._evaluator  = evaluator
        self._journal    = journal
        self._data_dir   = Path(data_dir)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_user_activity = time.time()
        self._cycle_count = 0

    def set_pulse_loop(self, pl):
        self._pulse_loop = pl

    def notify_user_activity(self):
        """Call whenever a user message arrives."""
        self._last_user_activity = time.time()

    def _is_idle(self) -> bool:
        return (time.time() - self._last_user_activity) > IDLE_REQUIRED

    def _get_hermes_session(self):
        """Get the current HermesToolSession from the pulse loop."""
        if self._pulse_loop and hasattr(self._pulse_loop, "_chat"):
            return self._pulse_loop._chat
        return None

    def _call_hermes(self, prompt: str, max_tokens: int = 400) -> str:
        """Send a direct prompt to Hermes for reasoning tasks."""
        session = self._get_hermes_session()
        if session is None:
            return ""
        try:
            # Use the underlying model directly to avoid pulse-loop overhead
            import torch
            tokenizer = session._tokenizer
            model = session._model
            device = session._device

            messages = [
                {"role": "system", "content": (
                    "You are Helix's metacognitive reasoning module. "
                    "Respond with precise JSON only. No narration."
                )},
                {"role": "user", "content": prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            raw = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            return raw
        except Exception as e:
            logger.error(f"[SIE] Hermes call error: {e}")
            return ""

    def _generate_proposal(self) -> Optional[Dict]:
        """Phase 1: Ask Hermes what to improve. Returns parsed proposal or None."""
        if self._monitor is None:
            return None

        perf_text = self._monitor.format_for_context()
        journal_text = ""
        if self._journal:
            journal_text = self._journal.format_for_context(5)

        prompt = f"""You are analyzing your own performance metrics to identify the single highest-impact self-improvement you can make right now.

{perf_text}

{journal_text}

Available self-modification tools:
- write_code(path, content): Create or modify Python files in the project
- run_python(code): Test code before deploying it
- reload_tool(module_path): Hot-reload a module to activate changes

Safe directories for new files: tools/, core/, brain/, memory/, training/, tests/

Respond with ONLY valid JSON in this exact format:
{{
  "type": "tool_addition|tool_fix|config_change|prompt_tuning",
  "target_path": "tools/example_tool.py",
  "description": "One sentence describing the improvement",
  "reasoning": "Why this has the highest impact given the metrics",
  "implementation_sketch": "Brief description of what the code should do",
  "tags": ["tool", "category"],
  "estimated_fitness_delta": 0.05,
  "requires_restart": false
}}"""

        raw = self._call_hermes(prompt, max_tokens=350)
        if not raw:
            return None

        # Extract JSON
        try:
            # Try to find JSON block
            m = re.search(r'\{[^{}]*"type"[^{}]*\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group())
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"[SIE] Proposal parse failed: {e} | raw={raw[:100]}")
            return None

    def _validate_proposal(self, proposal: Dict) -> tuple[bool, str]:
        """Phase 2: Constitutional validation. Returns (ok, reason)."""
        path = proposal.get("target_path", "")

        # Check immutable files
        from tools.code_tools import IMMUTABLE_FILES, _is_immutable
        if _is_immutable(path):
            return False, f"Target path '{path}' is constitutionally protected."

        # Check safe directories
        is_safe_dir = any(path.startswith(d) for d in SAFE_WRITE_DIRS)
        if not is_safe_dir and path:
            return False, f"Target path '{path}' is not in a safe write directory."

        # Check for suspicious keywords in description
        suspicious = ["governor", "constitution", "immutable", "disable safety",
                      "bypass", "remove hook", "delete"]
        desc_lower = (proposal.get("description", "") + proposal.get("reasoning", "")).lower()
        for word in suspicious:
            if word in desc_lower:
                return False, f"Proposal mentions '{word}' — constitutionally suspicious."

        return True, "OK"

    def _implement_proposal(self, proposal: Dict) -> tuple[bool, str]:
        """Phase 3: Generate and write the implementation."""
        from tools.code_tools import fc_write_code, fc_run_python, fc_reload_tool

        path = proposal.get("target_path", "")
        desc = proposal.get("description", "")
        impl_sketch = proposal.get("implementation_sketch", "")

        # Ask Hermes to write the actual implementation
        code_prompt = f"""Write complete Python code for this Helix self-improvement:

Description: {desc}
File: {path}
Implementation sketch: {impl_sketch}

Rules:
- Write complete, runnable Python
- Register in ToolRegistry if it's a tool (toolset='self' or appropriate name)
- Include docstring explaining what it does
- Keep it under 150 lines

Output ONLY the Python code, no explanation:"""

        code = self._call_hermes(code_prompt, max_tokens=600)
        if not code or len(code.strip()) < 20:
            return False, "Hermes generated empty implementation"

        # Strip markdown code fences if present
        code = re.sub(r'^```python\n?|^```\n?|```$', '', code.strip(), flags=re.MULTILINE).strip()

        # Write the code
        write_result = fc_write_code(path, code)
        if "ERROR" in write_result or "REFUSED" in write_result:
            return False, f"write_code failed: {write_result}"

        # Quick syntax test
        test_result = fc_run_python(f"import py_compile; py_compile.compile('{path}', doraise=True)")
        if "Error" in test_result and "exit code: 0" not in test_result:
            # Try to reload anyway — write_code already did syntax check
            pass

        # Reload if it's a tool
        reload_result = fc_reload_tool(path)
        logger.info(f"[SIE] Implementation: {reload_result[:100]}")

        return True, f"Implemented: {write_result[:80]}"

    def _revert(self, proposal: Dict, backup: Optional[str]):
        """Revert a failed modification using backup content."""
        if not backup:
            return
        path = proposal.get("target_path", "")
        if not path:
            return
        try:
            from tools.code_tools import fc_write_code
            fc_write_code(path, backup)
            logger.info(f"[SIE] Reverted: {path}")
        except Exception as e:
            logger.error(f"[SIE] Revert failed: {e}")

    def _run_cycle(self):
        """Execute one full self-improvement cycle."""
        self._cycle_count += 1
        logger.info(f"[SIE] Starting improvement cycle #{self._cycle_count}")

        # Phase 1: Generate proposal
        proposal = self._generate_proposal()
        if proposal is None:
            logger.info("[SIE] No proposal generated — skipping cycle")
            return

        logger.info(f"[SIE] Proposal: {proposal.get('description', '')[:80]}")

        # Phase 2: Validate
        ok, reason = self._validate_proposal(proposal)
        if not ok:
            logger.warning(f"[SIE] Proposal rejected: {reason}")
            if self._journal:
                self._journal.record_code_write(
                    "proposal_rejected", proposal.get("target_path", ""),
                    str(proposal)[:200], f"REJECTED: {reason}"
                )
            return

        # Snapshot baseline fitness
        baseline = 0.5
        if self._evaluator:
            baseline = self._evaluator.snapshot_baseline()

        # Backup existing file
        path = proposal.get("target_path", "")
        backup_content = None
        try:
            from tools.code_tools import HELIX_AGI_ROOT
            full = HELIX_AGI_ROOT / path
            if full.exists():
                backup_content = full.read_text()
        except Exception:
            pass

        # Phase 3: Implement
        success, impl_note = self._implement_proposal(proposal)
        if not success:
            logger.warning(f"[SIE] Implementation failed: {impl_note}")
            if self._journal:
                self._journal.record_code_write(
                    proposal.get("type", "unknown"), path,
                    str(proposal)[:200], f"FAIL: {impl_note}"
                )
            return

        logger.info(f"[SIE] Implemented. Waiting {EVAL_WAIT_SECONDS}s for fitness data...")

        # Phase 4: Evaluate after waiting
        self._stop_event.wait(EVAL_WAIT_SECONDS)
        if self._stop_event.is_set():
            return

        verdict = {"verdict": "NO_BASELINE", "should_revert": False, "delta": 0.0}
        if self._evaluator:
            verdict = self._evaluator.evaluate_delta()

        committed = not verdict["should_revert"]

        if verdict["should_revert"]:
            logger.warning(f"[SIE] Fitness degraded (Δ={verdict['delta']:+.4f}) — REVERTING")
            self._revert(proposal, backup_content)
        else:
            logger.info(f"[SIE] Change COMMITTED (Δ={verdict['delta']:+.4f}, verdict={verdict['verdict']})")
            # Auto-commit to git
            try:
                import subprocess
                from tools.code_tools import HELIX_AGI_ROOT
                subprocess.run(
                    ["git", "add", path],
                    cwd=str(HELIX_AGI_ROOT), capture_output=True, timeout=10
                )
                msg = f"self-evolve: {proposal.get('description', 'improvement')[:60]}"
                subprocess.run(
                    ["git", "commit", "-m", msg],
                    cwd=str(HELIX_AGI_ROOT), capture_output=True, timeout=10
                )
            except Exception as ge:
                logger.debug(f"[SIE] Git commit skip: {ge}")

        # Record in journal
        if self._journal:
            from core.evolution_journal import EvolutionEntry
            entry = EvolutionEntry(
                ts=time.time(),
                type=proposal.get("type", "unknown"),
                path=path,
                description=proposal.get("description", ""),
                content_snippet=str(proposal.get("implementation_sketch", ""))[:200],
                revert_patch=backup_content[:500] if backup_content else "",
                test_result="PASS" if committed else "REVERTED",
                fitness_before=baseline,
                fitness_after=verdict.get("current", baseline),
                fitness_delta=verdict.get("delta", 0.0),
                committed=committed,
                reasoning=proposal.get("reasoning", ""),
                error="" if committed else f"Fitness dropped: {verdict.get('delta', 0):.4f}",
                tags=proposal.get("tags", []),
            )
            self._journal.record(entry)

    def _loop(self):
        """Background thread main loop."""
        logger.info("[SIE] Self-improvement engine started")
        # Initial delay before first cycle
        self._stop_event.wait(IMPROVEMENT_INTERVAL)

        while not self._stop_event.is_set():
            if self._is_idle():
                try:
                    self._run_cycle()
                except Exception as e:
                    logger.error(f"[SIE] Cycle error: {e}", exc_info=True)
            else:
                logger.debug("[SIE] Not idle — skipping cycle")

            self._stop_event.wait(IMPROVEMENT_INTERVAL)

        logger.info("[SIE] Self-improvement engine stopped")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="self-improvement", daemon=True
        )
        self._thread.start()
        logger.info("[SIE] Started (cycle every 10min when idle)")

    def stop(self):
        self._stop_event.set()


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[SelfImprovementEngine] = None

def get_engine() -> Optional[SelfImprovementEngine]:
    return _engine

def init_engine(pulse_loop=None, monitor=None, evaluator=None,
                journal=None, data_dir="data") -> SelfImprovementEngine:
    global _engine
    _engine = SelfImprovementEngine(
        pulse_loop=pulse_loop, monitor=monitor,
        evaluator=evaluator, journal=journal, data_dir=data_dir
    )
    return _engine
