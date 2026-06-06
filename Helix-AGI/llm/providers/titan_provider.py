"""
Helix — Titan MIMO Local Provider

Implements the ChatSession interface backed by the Titan 2.7B MIMO inference engine.
Runs 100% locally on the RTX 3060 — no API calls, no cloud dependency.

Architecture:
  - Parses Helix pulse meta-tags ([JOURNAL:], [REMEMBER:], [BELIEF_FORM], etc.)
  - Maps tags to MIMO arm bias vectors so the right arm activates per cognitive task
  - Handles context compression automatically when history exceeds Titan's window
  - Logs (prompt, response) pairs to a replay buffer for overnight fine-tuning

Arm → Helix subsystem mapping:
  Arm 0  General Language      → pulse_loop internal monologue
  Arm 1  Symbolic Math         → physics_engine Lagrangian calculations
  Arm 2  Logical Reasoning     → belief_detector / belief_consolidator
  Arm 3  Code Syntax           → tool call generation
  Arm 4  Factual Recall        → memory_manager semantic search
  Arm 5  Summarization         → context_compressor
  Arm 6  Creative Writing      → cognitive_journal journaling
  Arm 7  Instruction Following → orchestrator tool dispatch
"""

import logging
import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Optional, List, Dict

from llm.providers.base import ChatSession

logger = logging.getLogger("helix.llm.providers.titan")

# ── Path bootstrap: allow running from any working directory ──────────────────
_HERE = Path(__file__).resolve().parent          # Helix-AGI/llm/providers/
_PROJECT = _HERE.parents[2]                      # analysis_project/
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

# ── Arm bias profiles (8 arms) ───────────────────────────────────────────────
# Additive boosts applied to gate logits before softmax.
# Values: 0.0 = neutral, 2.0 = strong preference.
# Order: [General, Math, Logical, Code, Factual, Summary, Creative, Instruction]
_ARM_PROFILES: Dict[str, List[float]] = {
    "journal":      [0.2, 0.0, 0.0, 0.0, 0.0, 0.5, 2.0, 0.3],
    "remember":     [0.2, 0.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.3],
    "belief":       [0.2, 0.5, 2.0, 0.0, 0.5, 0.3, 0.0, 0.5],
    "tool_call":    [0.2, 0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 2.0],
    "math_physics": [0.2, 2.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.3],
    "compress":     [0.2, 0.0, 0.3, 0.0, 0.5, 2.0, 0.3, 0.5],
    "default":      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}


# ── Tag → profile mapping ────────────────────────────────────────────────────
_TAG_TO_PROFILE = {
    r"\[JOURNAL[:\]]":          "journal",
    r"\[NOTE[:\]]":             "journal",
    r"\[REMEMBER[:\]]":         "remember",
    r"\[BELIEF_FORM\]":         "belief",
    r"\[BELIEF_CONSOLIDAT":     "belief",
    r"tool_call|function_call": "tool_call",
    r"Ω|lagrangian|manifold|curvature|stability": "math_physics",
    r"\[COMPRESS\]|context_limit": "compress",
}


def _detect_arm_profile(text: str) -> str:
    """Scan message text for Helix meta-tags and return best arm profile name."""
    text_lower = text.lower()
    for pattern, profile in _TAG_TO_PROFILE.items():
        if re.search(pattern, text, re.IGNORECASE):
            return profile
    return "default"


def _compress_history(history: List[Dict], max_chars: int = 3000) -> List[Dict]:
    """Trim oldest history turns to stay within Titan's context window.
    Keeps the system turn (index 0) and all recent turns.
    """
    if not history:
        return history
    total = sum(len(m.get("content", "")) for m in history)
    while total > max_chars and len(history) > 2:
        removed = history.pop(1)  # remove oldest non-system turn
        total -= len(removed.get("content", ""))
    return history


class TitanSession(ChatSession):
    """
    Chat session backed by Titan 2.7B MIMO local inference.

    Drop-in replacement for GeminiSession / OllamaSession.
    The pulse loop sees only the ChatSession interface.
    """

    # Max chars of history to keep — Titan context ~512 tokens ≈ 2048 chars.
    # Helix's context_compressor handles deeper compression upstream.
    MAX_HISTORY_CHARS = 2048

    # Replay buffer: log every (prompt, response) pair for overnight fine-tuning
    REPLAY_BUFFER_PATH = _PROJECT / "helix_replay_buffer.jsonl"

    def __init__(
        self,
        system_instruction: str,
        temperature: float = 0.85,
        max_output_tokens: int = 512,
        enable_deep_think: bool = False,
        checkpoint: str = "auto",
    ):
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_deep_think = enable_deep_think

        self.history: List[Dict[str, str]] = []
        self._engine = None  # lazy-loaded on first send_message()
        self._checkpoint = checkpoint

        logger.info(
            f"TitanSession created — temp={temperature}, "
            f"max_tokens={max_output_tokens}, deep_think={enable_deep_think}"
        )

    # ── Lazy engine load ──────────────────────────────────────────────────────
    def _ensure_loaded(self):
        if self._engine is not None:
            return
        try:
            from titan_inference import TitanInference
            logger.info("Loading Titan 2.7B MIMO model into GPU memory...")
            self._engine = TitanInference(checkpoint=self._checkpoint)
            self._engine.load()
            logger.info("Titan loaded successfully ✓")
        except Exception as e:
            logger.error(f"Failed to load Titan: {e}")
            raise RuntimeError(
                f"Titan inference engine failed to initialize: {e}\n"
                "Ensure Phase 1 training has produced a checkpoint at "
                "checkpoints_2.7b/phase_1.pt"
            ) from e

    # ── Core interface ────────────────────────────────────────────────────────
    def send_message(self, message: str) -> str:
        """Send a pulse message to Titan and return generated text."""
        self._ensure_loaded()

        # Detect which cognitive arm profile this pulse needs
        profile_name = _detect_arm_profile(message)
        arm_bias = _ARM_PROFILES[profile_name]
        if profile_name != "default":
            logger.debug(f"Arm profile: {profile_name} | bias applied to gate logits")

        # Build prompt: system + compressed history + new message
        self.history.append({"role": "user", "content": message})
        self.history = _compress_history(self.history, self.MAX_HISTORY_CHARS)

        prompt = self._build_prompt()

        # Generate
        t0 = time.time()
        try:
            response_tokens = []
            for token, arm_info in self._engine.stream(
                prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_output_tokens,
                arm_bias=arm_bias,
            ):
                response_tokens.append(token)

            response = "".join(response_tokens).strip()
            elapsed = time.time() - t0
            tps = len(response_tokens) / max(elapsed, 0.001)
            logger.debug(
                f"Titan generated {len(response_tokens)} tokens "
                f"in {elapsed:.1f}s ({tps:.1f} tok/s) | profile={profile_name}"
            )

        except Exception as e:
            logger.error(f"Titan generation error: {e}")
            response = f"[Titan internal error: {str(e)[:120]}]"

        # Store in history
        self.history.append({"role": "assistant", "content": response})

        # Log to replay buffer for overnight fine-tuning
        self._log_replay(prompt, response, profile_name)

        return response

    def get_history_size(self) -> int:
        total = len(self.system_instruction)
        for msg in self.history:
            total += len(msg.get("content", ""))
        return total

    # ── Prompt assembly ───────────────────────────────────────────────────────
    def _build_prompt(self) -> str:
        """Assemble a flat prompt string from system instruction + history."""
        parts = [f"[SYSTEM]\n{self.system_instruction}\n[/SYSTEM]\n"]
        for msg in self.history:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")
        parts.append("Assistant:")
        return "\n".join(parts)

    # ── Replay buffer ─────────────────────────────────────────────────────────
    def _log_replay(self, prompt: str, response: str, profile: str):
        """Append (prompt, response) to JSONL replay buffer for nightly fine-tuning."""
        try:
            entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "profile": profile,
                "prompt": prompt[-1000:],  # trim to avoid huge files
                "response": response,
            }
            with open(self.REPLAY_BUFFER_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Replay buffer write failed: {e}")
