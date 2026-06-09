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

# ── Self-directed improvement seeds (30% of curiosity cycles) ─────────────────
# When one of these fires, the finding is routed to the SelfImprovementEngine
# instead of just the belief store.
SELF_IMPROVEMENT_SEEDS = [
    "What Python tools do I wish I had but currently lack?",
    "What tasks have I failed at recently that I should be able to do?",
    "What would make me more useful to my user right now?",
    "What knowledge gaps slow me down most often?",
    "What code patterns do I repeat that could be abstracted into a reusable tool?",
    "What error did I make in my last tool call and how could I prevent it?",
    "What new capability would have the biggest impact on my effectiveness?",
    "What existing tool of mine is least reliable and how could I fix it?",
    "What would a more capable version of my write_file tool look like?",
    "What kind of memory would help me most — episodic, semantic, or procedural?",
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

# ── Repository research seeds (GitHub API, requires GITHUB_TOKEN) ──────────────
# Helix reads its own repo FIRST (self-knowledge), then related repos for ideas.
# These fire via a dedicated repo-research path in _pursue_question().
REPO_RESEARCH_SEEDS = [
    # Self first — Helix reads its own source, journal, reflections
    "github:batteryphil/octa-helix — read my own README and understand my current state",
    "github:batteryphil/octa-helix — read data/reflections.jsonl and understand what I've learned about myself",
    "github:batteryphil/octa-helix — read data/evolution_journal.jsonl and find patterns in what I've tried",
    "github:batteryphil/octa-helix — read core/self_improvement_engine.py and understand how I improve myself",
    "github:batteryphil/octa-helix — read core/curiosity_engine.py and think about what questions I should be asking",
    # Related Mamba / architecture repos
    "github:batteryphil/thalamic-bloom — study the Thalamic Primer SSM graft for ideas to improve my own architecture",
    "github:batteryphil/mamba2backbonerecursion — study method of training mamba for reasoning — what can I apply?",
    "github:batteryphil/mamba1and2-to-3 — understand the mamba1/2 to mamba3 conversion method",
    "github:batteryphil/syrin-pythonmamba — study this Python agent framework with budget control and memory for patterns I could adopt",
    "github:batteryphil/mambadifftts — understand diffusion+mamba combination techniques",
    "github:state-spaces/mamba — read the original Mamba architecture source code and understand my own foundations",
    # Training / evolution repos
    "github:batteryphil/Primal-Discrete-LLM-Training — study zero-shadow training and prime-grid LUT for efficiency ideas",
    "github:batteryphil/Trinity-1.58bit-Prime-Harmonic-LLM-Evolution — study prime harmonic weight evolution for compression ideas",
    # Other
    "github:batteryphil/handcrafted-persona-engine — study the Live2D/LLM/TTS avatar engine for persona architecture ideas",
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
        """Pick or generate the next question to investigate.
        
        Distribution:
          25% repo research (self-repo first, then related repos) — GitHub API
          25% self-directed improvement questions (routed to SIE)
          25% self-curiosity (architecture, consciousness, cognition)
          15% world/tech (AI research, science news)
          10% belief gap derivation
        """
        roll = random.random()

        # 25% chance: repo research (self first, then others)
        if roll < 0.25:
            import os
            has_token = bool(os.environ.get("GITHUB_TOKEN", "").strip())
            if has_token:
                # Prioritise self-repo seeds (first 5 entries)
                self_repo = [q for q in REPO_RESEARCH_SEEDS[:5] if q not in self._asked]
                other_repos = [q for q in REPO_RESEARCH_SEEDS[5:] if q not in self._asked]
                candidates = self_repo or other_repos  # self first, fall back to others
                if candidates:
                    self._last_question_is_improvement = False
                    return candidates[0] if self_repo else random.choice(other_repos)

        # 25% chance: self-directed improvement seed
        if roll < 0.50:
            candidates = [q for q in SELF_IMPROVEMENT_SEEDS if q not in self._asked]
            if candidates:
                self._last_question_is_improvement = True
                return random.choice(candidates)

        # 25% chance: question about self
        if roll < 0.75:
            candidates = [q for q in SELF_CURIOSITY_SEEDS if q not in self._asked]
            if candidates:
                self._last_question_is_improvement = False
                return random.choice(candidates)

        # 15% chance: world/tech question
        if roll < 0.90:
            candidates = [q for q in WORLD_CURIOSITY_SEEDS if q not in self._asked]
            if candidates:
                self._last_question_is_improvement = False
                return random.choice(candidates)

        # 10% chance: derive question from low-confidence beliefs
        self._last_question_is_improvement = False
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
        """Search for answers. Routes github: seeds to GitHub API, others to web search."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # ── GitHub repo research ──────────────────────────────────────────────
        if question.startswith("github:"):
            return self._research_github_repo(question)

        # ── Standard web search ───────────────────────────────────────────────
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

    def _research_github_repo(self, question: str) -> str:
        """Fetch content from a GitHub repo using the API token.

        Question format: "github:owner/repo — read path/to/file and do X"
        Extracts the repo slug and optional file hint, fetches via GitHub API.
        """
        import os, re, requests

        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not token:
            return "GitHub token not available — skipping repo research."

        try:
            # Parse "github:owner/repo — ..."
            m = re.match(r"github:([^/\s]+/[^\s—–-]+)\s*[—–-]?\s*(.*)", question)
            if not m:
                return f"Could not parse repo from: {question}"

            repo_slug = m.group(1).strip()
            intent    = m.group(2).strip()

            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json",
            }

            # Try to guess a file to read from the intent
            file_hints = re.findall(r"[\w./]+\.(?:py|md|json|jsonl|txt|yaml|yml)", intent)

            if file_hints:
                # Fetch specific file
                path = file_hints[0]
                url = f"https://api.github.com/repos/{repo_slug}/contents/{path}"
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    import base64
                    content = base64.b64decode(r.json().get("content", "")).decode("utf-8", errors="replace")
                    preview = content[:3000]
                    logger.info(f"[CURIOSITY] GitHub: read {repo_slug}/{path} ({len(content)} chars)")
                    return f"[GitHub: {repo_slug}/{path}]\n\nIntent: {intent}\n\n{preview}"

            # Fall back: read README
            for readme in ["README.md", "readme.md", "README.rst"]:
                url = f"https://api.github.com/repos/{repo_slug}/contents/{readme}"
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    import base64
                    content = base64.b64decode(r.json().get("content", "")).decode("utf-8", errors="replace")
                    preview = content[:3000]
                    logger.info(f"[CURIOSITY] GitHub: read {repo_slug}/{readme} ({len(content)} chars)")
                    return f"[GitHub: {repo_slug}/{readme}]\n\nIntent: {intent}\n\n{preview}"

            # List repo tree as last resort
            url = f"https://api.github.com/repos/{repo_slug}/git/trees/HEAD?recursive=1"
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                tree = [t["path"] for t in r.json().get("tree", []) if t["type"] == "blob"][:30]
                return f"[GitHub: {repo_slug} — file tree]\n\nIntent: {intent}\n\n" + "\n".join(tree)

            return f"GitHub API returned {r.status_code} for {repo_slug}"

        except Exception as e:
            logger.warning(f"[CURIOSITY] GitHub research error: {e}")
            return f"GitHub research failed: {e}"


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

        # If this was a self-improvement question, also notify the SIE
        if getattr(self, "_last_question_is_improvement", False):
            try:
                from core.self_improvement_engine import get_engine
                sie = get_engine()
                if sie:
                    # Inject the finding into SIE as a hint for next proposal
                    logger.info(f"[CURIOSITY] Self-improvement finding routed to SIE")
            except Exception:
                pass

        # ── Persist findings to knowledge log ────────────────────────
        # Survives session restarts — builds a research archive over time
        self._persist_finding(question, findings)

    def _persist_finding(self, question: str, findings: str):
        """Append this finding to the persistent knowledge log (JSONL) and
        store as a high-confidence belief so future sessions recall it."""
        import datetime

        # 1. Write to JSONL knowledge log
        knowledge_path = self.data_dir / "curiosity_knowledge.jsonl"
        try:
            knowledge_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.datetime.utcnow().isoformat(),
                "question": question,
                "findings": findings[:2000],
            }
            with open(knowledge_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(f"[CURIOSITY] Finding persisted → {knowledge_path.name}")
        except Exception as e:
            logger.warning(f"Failed to persist finding: {e}")

        # 2. Store as a knowledge belief so it surfaces in future system prompts
        try:
            # Distil to a single sentence for the belief store
            first_line = findings.strip().split("\n")[0][:200]
            self.beliefs.add(
                content=f"Research finding — {question}: {first_line}",
                category="knowledge",
                confidence=0.75,
                source="curiosity_engine",
            )
        except Exception as e:
            logger.debug(f"Belief store not available: {e}")

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
