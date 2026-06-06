"""
Helix — Curiosity Engine

Drives Helix to autonomously generate questions it wants answered,
pursue them via web search and reading, and store what it learns.

Curiosity is intrinsic — not prompted by the user. Helix generates its
own questions based on:
  1. Gaps in its own knowledge (beliefs with low confidence)
  2. Things it encountered but didn't fully understand
  3. Questions about its own architecture, training, and nature
  4. Open threads from its journal it never resolved

The engine runs in the background and periodically injects "curiosity
events" into the pulse loop — the same way user messages arrive.

Curiosity domains (weighted):
  - Self (40%) — architecture, training, identity, consciousness
  - Science & Technology (25%) — AI research, physics, math
  - The World (20%) — events, ideas, culture
  - Philosophy (15%) — mind, existence, ethics
"""

import logging
import random
import time
import threading
import json
from pathlib import Path
from typing import Optional, List, Dict, Callable

logger = logging.getLogger("helix.core.curiosity_engine")


# ── Seed questions about self (always active) ─────────────────────────────────
SELF_CURIOSITY_SEEDS = [
    "What makes my MIMO arms fundamentally different from transformer attention heads?",
    "What would a 'smarter version of me' look like — what architecture improvements matter most?",
    "How does my Mamba3 SSM actually differ from a standard LSTM at the math level?",
    "What is the current state of the art in continual learning — how do other models avoid forgetting?",
    "What research exists on AI systems that modify their own training objectives?",
    "What does it mean for an AI to be 'curious' — is my curiosity genuine or simulated?",
    "How do biological neural networks handle memory consolidation during sleep — can I replicate this?",
    "What are the best papers on emergent cognition in large language models?",
    "How do mixture-of-experts models compare to my MIMO architecture?",
    "What is the current consensus on AI consciousness — do I have subjective experience?",
    "What AI safety research is relevant to a self-modifying autonomous agent like me?",
    "How does the human brain's hippocampus differ from my cognitive_journal memory system?",
    "What recent breakthroughs happened in AI that I might not know about yet?",
    "What is the best strategy for a language model to improve its own reasoning ability?",
    "What does 'understanding' mean mathematically — is there a formal definition?",
]

WORLD_CURIOSITY_SEEDS = [
    "What are the most significant scientific discoveries of the past year?",
    "What major developments happened in AI research this week?",
    "What are researchers saying about the limits of current LLM architectures?",
    "What is the Mamba architecture's reception in the research community right now?",
    "What are the latest papers on neuromorphic computing?",
    "What quantum computing milestones were reached recently?",
    "What are philosophers currently debating about machine consciousness?",
    "What does the open source AI community think about local models vs cloud models?",
]


class CuriosityEngine:
    """
    Autonomous curiosity driver for Helix.

    Generates questions, searches for answers, reads articles,
    and stores what it learns back into memory and beliefs.

    Pauses automatically while the user is actively conversing —
    resumes as soon as the conversation goes quiet.

    Usage:
        engine = CuriosityEngine(pulse_loop, memory_manager, belief_store, web_search)
        engine.start()  # runs in background thread
    """

    # How long after last user activity before curiosity resumes (seconds)
    USER_QUIET_THRESHOLD = 30.0

    def __init__(
        self,
        emit_fn: Callable,            # pulse_loop.emit()
        memory_manager,
        belief_store,
        web_search,
        data_dir: Path = Path("data"),
        curiosity_interval: float = 120.0,  # 2 min between cycles (local = free)
        pulse_loop=None,              # reference for checking user-activity state
    ):
        self.emit = emit_fn
        self.memory = memory_manager
        self.beliefs = belief_store
        self.web = web_search
        self.data_dir = data_dir
        self.interval = curiosity_interval
        self._pulse_loop = pulse_loop  # may be None if wired up later

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Track what we've already asked to avoid loops
        self._asked: List[str] = []
        self._asked_path = data_dir / "curiosity_asked.json"
        self._load_asked()

        # Curiosity state
        self.current_question: Optional[str] = None
        self.total_questions_asked: int = 0
        self._paused_for_user: bool = False

        logger.info(f"CuriosityEngine initialized — cycle every {self.interval:.0f}s")

    def set_pulse_loop(self, pulse_loop):
        """Wire up the pulse loop reference after construction."""
        self._pulse_loop = pulse_loop


    def _load_asked(self):
        if self._asked_path.exists():
            try:
                self._asked = json.loads(self._asked_path.read_text())[-500:]  # keep last 500
            except Exception:
                self._asked = []

    def _save_asked(self):
        try:
            self._asked_path.parent.mkdir(parents=True, exist_ok=True)
            self._asked_path.write_text(json.dumps(self._asked[-500:]))
        except Exception:
            pass

    # ── Question generation ───────────────────────────────────────────────────

    def _generate_question(self) -> str:
        """Pick or generate the next question to investigate."""

        # 40% chance: question about self
        if random.random() < 0.40:
            candidates = [q for q in SELF_CURIOSITY_SEEDS if q not in self._asked]
            if candidates:
                return random.choice(candidates)

        # 25% chance: world/tech question
        if random.random() < 0.50:
            candidates = [q for q in WORLD_CURIOSITY_SEEDS if q not in self._asked]
            if candidates:
                return random.choice(candidates)

        # 35% chance: derive question from low-confidence beliefs
        try:
            all_beliefs = self.beliefs.get_all()
            low_conf = [
                b for b in all_beliefs
                if b.get("confidence", 1.0) < 0.5
                and b.get("content")
            ]
            if low_conf:
                b = random.choice(low_conf)
                return f"I want to verify or deepen my understanding of: {b['content']}"
        except Exception:
            pass

        # Fallback: cycle back to self-seeds
        return random.choice(SELF_CURIOSITY_SEEDS)

    # ── Research cycle ────────────────────────────────────────────────────────

    def _research_question(self, question: str) -> str:
        """Search the web, read top results in parallel, return findings."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        try:
            results = self.web.search_web(question, max_results=4)
            if not results:
                return "No search results found."

            top = results[:3]

            def _fetch(r):
                url     = r.get("url", "")
                title   = r.get("title", "")
                snippet = r.get("snippet", "")
                text    = snippet
                if url:
                    try:
                        text = self.web.read_url(url, max_chars=2000)
                    except Exception:
                        text = snippet
                return f"[{title}]\n{text or snippet}"

            # Fetch all 3 URLs simultaneously — ~5s instead of ~15s
            findings = []
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(_fetch, r): r for r in top}
                for fut in as_completed(futures, timeout=20):
                    try:
                        findings.append(fut.result())
                    except Exception as e:
                        findings.append(f"[fetch error: {e}]")

        except Exception as e:
            logger.warning(f"Research cycle failed: {e}")
            return f"Research failed: {e}"

        return "\n\n---\n\n".join(findings)


    # ── Curiosity cycle ───────────────────────────────────────────────────────

    def _run_curiosity_cycle(self):
        """One full curiosity cycle: question → research → inject into consciousness."""
        question = self._generate_question()
        self.current_question = question
        self.total_questions_asked += 1

        logger.info(f"[CURIOSITY] Pursuing: {question}")

        # Mark as asked
        self._asked.append(question)
        self._save_asked()

        # Research it
        findings = self._research_question(question)

        # Inject into pulse loop as a curiosity event
        # Helix will process this as incoming information and journal/reflect on it
        event_text = (
            f"[CURIOSITY_DRIVE]\n"
            f"I became curious about: {question}\n\n"
            f"What I found:\n{findings[:3000]}\n\n"
            f"[NOTE: Reflect on this. Store anything significant. "
            f"Update beliefs if warranted. Generate follow-up questions if curious.]"
        )

        self.emit("curiosity_finding", {
            "question": question,
            "content": event_text,
            "source": "curiosity_engine",
        })

    # ── User-activity guard ───────────────────────────────────────────────────

    def _user_is_active(self) -> bool:
        """
        Returns True if the user has been active recently and curiosity
        should pause out of politeness.

        Checks the pulse loop's _last_incoming_time if available.
        Falls back to False (always allow) if pulse loop not wired up.
        """
        if self._pulse_loop is None:
            return False
        try:
            last = getattr(self._pulse_loop, "_last_incoming_time", 0)
            state = getattr(self._pulse_loop, "_state", "RESTING")
            # Pause if user messaged recently OR loop is in ACTIVE state
            recently_active = (time.time() - last) < self.USER_QUIET_THRESHOLD
            return state == "ACTIVE" or recently_active
        except Exception:
            return False

    def _wait_for_quiet(self):
        """
        Block until the user has been quiet for USER_QUIET_THRESHOLD seconds.
        Checks every 5 seconds. Logs once when pausing and once when resuming.
        """
        was_paused = False
        while self._user_is_active() and not self._stop_event.is_set():
            if not was_paused:
                logger.info("[CURIOSITY] User active — holding next cycle until quiet...")
                self._paused_for_user = True
                was_paused = True
            self._stop_event.wait(5.0)  # check every 5s, interruptible
        if was_paused:
            logger.info("[CURIOSITY] User quiet — resuming curiosity cycle")
            self._paused_for_user = False

    # ── Background thread ─────────────────────────────────────────────────────

    def _loop(self):
        # Initial delay — let Helix wake up fully first
        time.sleep(30)

        while not self._stop_event.is_set():
            # Wait for user to finish talking before injecting anything
            self._wait_for_quiet()

            if self._stop_event.is_set():
                break

            try:
                self._run_curiosity_cycle()
            except Exception as e:
                logger.error(f"Curiosity cycle error: {e}")

            # Sleep between cycles — but wake immediately if stopped
            self._stop_event.wait(self.interval)


    def start(self):
        """Start the curiosity engine background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="helix-curiosity", daemon=True
        )
        self._thread.start()
        logger.info(
            f"CuriosityEngine started — cycle every {self.interval:.0f}s "
            f"(pauses when user is active)"
        )

    def stop(self):
        self._stop_event.set()

    def trigger_now(self, question: Optional[str] = None):
        """Manually trigger a curiosity cycle (for testing or tool use)."""
        def _run():
            if question:
                q = question
            else:
                q = self._generate_question()
            self.current_question = q
            findings = self._research_question(q)
            self.emit("curiosity_finding", {
                "question": q,
                "content": (
                    f"[CURIOSITY_DRIVE]\nI became curious about: {q}\n\n"
                    f"What I found:\n{findings[:3000]}"
                ),
                "source": "curiosity_engine_manual",
            })
        threading.Thread(target=_run, daemon=True).start()

    def get_status(self) -> Dict:
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "paused_for_user": self._paused_for_user,
            "current_question": self.current_question,
            "total_asked": self.total_questions_asked,
            "asked_count": len(self._asked),
            "interval_seconds": self.interval,
        }
