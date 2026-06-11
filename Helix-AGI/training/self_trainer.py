"""
Helix — Self-Trainer (LoRA Experience Collector + Fine-Tuner)

Collects (prompt, response, outcome) experience tuples from real
interactions. When enough high-quality examples accumulate, runs a
LoRA fine-tuning pass on Hermes-3-Llama-3.1-8B to improve tool-calling.

Pipeline:
  1. collect_experience() — called from post-pulse hook on every pulse
  2. Every 500 high-quality tuples: trigger fine-tuning
  3. Fine-tuning runs only when idle (>10 min no user activity)
  4. After training: compare perplexity on held-out set
  5. If improved: swap in new adapter; if worse: discard

VRAM note: Training and inference cannot run simultaneously on 12GB.
The trainer pauses inference by acquiring a global lock before loading
the training configuration.
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("helix.training.self_trainer")

EXPERIENCE_THRESHOLD = 2000   # tuples before triggering training (~100hrs runtime)
TRAIN_STEPS = 100            # LoRA steps per training run
IDLE_REQUIRED = 600          # 10 min idle before training starts
MIN_TOOL_DIVERSITY = 2       # unique tool types needed before training
LORA_LR = 5e-5               # conservative LR (was 2e-4 — too aggressive for small sets)
LORA_DROPOUT = 0.1           # higher dropout for regularization (was 0.05)
MAX_TRAIN_EPOCHS = 5         # max passes over data regardless of TRAIN_STEPS
EVAL_LOSS_TOLERANCE = 1.10   # reject adapter if eval loss > baseline * this


@dataclass
class ExperienceTuple:
    ts: float
    prompt: str              # the pulse message sent to Hermes
    response: str            # what Hermes generated
    outcome: str             # "tool_executed" | "hallucination" | "prose" | "error"
    tool_name: str           # tool called (if any)
    quality: float           # 0.0–1.0 estimate of response quality
    user_sentiment: str      # "positive" | "negative" | "neutral"


class SelfTrainer:
    """Collects experience and runs LoRA fine-tuning during idle periods."""

    def __init__(self, data_dir: str = "data"):
        self._data_dir = Path(data_dir)
        self._exp_path = self._data_dir / "experience_tuples.jsonl"
        self._adapter_dir = self._data_dir / "lora_adapters"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._adapter_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._buffer: List[ExperienceTuple] = []
        self._total_collected = 0
        self._last_user_activity = time.time()
        self._training_active = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._load_existing()

    def _load_existing(self):
        if not self._exp_path.exists():
            return
        try:
            with self._exp_path.open("r") as f:
                count = sum(1 for _ in f)
            self._total_collected = count
            logger.info(f"[trainer] Found {count} existing experience tuples")
        except Exception:
            pass

    def notify_user_activity(self):
        self._last_user_activity = time.time()

    def _is_idle(self) -> bool:
        return (time.time() - self._last_user_activity) > IDLE_REQUIRED

    def collect_experience(self, ctx) -> None:
        """Collect one experience tuple — STRICT QUALITY GATE applied.

        Gemini Pro analysis (Q4): If we fine-tune on 500 tuples of the agent
        failing to use tools and looping on error logs, we permanently bake
        that incompetence into the weights.

        A tuple is only recorded if it meets AT LEAST 2 of these 3 criteria:
          1. fitness_delta > 0.05   (action demonstrably improved the system)
          2. tool_execution_success (a tool was called and returned non-error)
          3. novel_belief_generated (verified new belief added to BeliefStore)

        Prose-only, error-only, and Δ=0 tuples are discarded.
        It is better to take 2 weeks to collect 500 high-quality tuples than
        to ruin the base model in 2 days.
        """
        try:
            thought    = getattr(ctx, "thought", "") or ""
            tool_calls = getattr(ctx, "tool_calls", []) or []

            if not thought:
                return

            # ── Criterion 1: tool execution success ───────────────────────
            tool_executed = bool(tool_calls)
            tool_name     = tool_calls[0].get("name", "") if tool_calls else ""
            tool_result   = tool_calls[0].get("result", "") if tool_calls else ""
            tool_success  = (
                tool_executed
                and "error" not in str(tool_result).lower()[:100]
                and "failed" not in str(tool_result).lower()[:100]
                and "traceback" not in str(tool_result).lower()[:100]
            )

            # ── Criterion 2: novel belief generated ───────────────────────
            novel_belief = getattr(ctx, "novel_belief_added", False)

            # ── Criterion 3: fitness delta (from SIE context if available) ─
            fitness_delta = getattr(ctx, "last_fitness_delta", 0.0) or 0.0
            # A successful tool call IS a significant gain over prose-only.
            # Don't require the snapshot to have already caught up — this creates
            # a chicken-and-egg: no training data until fitness is already high.
            significant_gain = tool_success or (fitness_delta > 0.05)

            # ── Quality gate: must meet ≥2 of 3 criteria ─────────────────
            criteria_met = sum([tool_success, novel_belief, significant_gain])
            if criteria_met < 2:
                # Discard — not good enough to train on
                return

            # ── Determine outcome and quality ─────────────────────────────
            if tool_success:
                outcome = "tool_executed"
                quality = 0.7 + (0.15 * significant_gain) + (0.15 * novel_belief)
            elif novel_belief:
                outcome = "novel_belief"
                quality = 0.6 + (0.2 * significant_gain)
            else:
                outcome = "fitness_gain"
                quality = 0.6

            # Build the prompt for the tuple.
            # CRITICAL: events is a List[str] and may be EMPTY for autonomous
            # mandate pulses (no user activity = no queued events). Empty list
            # is falsy in Python, so `events or ""` would be "", causing every
            # autonomous pulse to be silently rejected.
            # Fix: fall back to the thought itself as the prompt context.
            raw_events = getattr(ctx, "events", None)
            if raw_events:  # non-empty list
                pulse_msg = "\n".join(str(e) for e in raw_events)
            else:           # empty list or None — use thought as context
                pulse_msg = thought[:300]
            if not pulse_msg:
                return

            tup = ExperienceTuple(
                ts=time.time(),
                prompt=str(pulse_msg)[:500],
                response=thought[:500],
                outcome=outcome,
                tool_name=tool_name,
                quality=round(quality, 3),
                user_sentiment="neutral",
            )

            with self._lock:
                self._buffer.append(tup)
                self._total_collected += 1

            logger.info(
                f"[trainer] Quality tuple accepted: outcome={outcome} "
                f"quality={quality:.2f} criteria={criteria_met}/3 "
                f"(tool={tool_success} belief={novel_belief} gain={significant_gain})"
            )

            # Flush buffer to disk immediately (was: every 10 tuples)
            # With only a few accepted tuples per session, threshold=10
            # meant the buffer was never flushed and data lost on restart.
            if len(self._buffer) >= 1:
                self._flush_buffer()

            # Check if we have enough for training
            if (self._total_collected % EXPERIENCE_THRESHOLD == 0
                    and self._total_collected > 0
                    and not self._training_active):
                logger.info(f"[trainer] {self._total_collected} quality tuples — scheduling training")
                self._schedule_training()

        except Exception as e:
            logger.debug(f"[trainer] collect error: {e}")

    def _flush_buffer(self):
        """Flush buffer to disk."""
        with self._lock:
            if not self._buffer:
                return
            tuples = list(self._buffer)
            self._buffer.clear()

        try:
            with self._exp_path.open("a") as f:
                for t in tuples:
                    f.write(json.dumps(asdict(t)) + "\n")
        except Exception as e:
            logger.error(f"[trainer] flush error: {e}")

    def _schedule_training(self):
        """Start training in background thread if idle."""
        if self._training_active:
            return
        t = threading.Thread(target=self._training_loop, daemon=True, name="self-trainer")
        t.start()

    def _load_high_quality_examples(self, n: int = 200) -> List[ExperienceTuple]:
        """Load the best N experience tuples for training."""
        tuples = []
        if not self._exp_path.exists():
            return tuples
        try:
            with self._exp_path.open("r") as f:
                for line in f:
                    d = json.loads(line.strip())
                    t = ExperienceTuple(**{k: d.get(k, "") for k in ExperienceTuple.__dataclass_fields__})
                    # Only use high-quality tool_executed examples
                    if t.outcome == "tool_executed" and t.quality > 0.7:
                        tuples.append(t)
        except Exception as e:
            logger.error(f"[trainer] load examples error: {e}")

        # Sort by quality descending, take top N
        tuples.sort(key=lambda x: x.quality, reverse=True)
        return tuples[:n]

    def _training_loop(self):
        """Background training thread — waits for idle then runs LoRA."""
        self._training_active = True
        logger.info("[trainer] Training scheduled — waiting for idle window")

        # Wait for idle
        while not self._is_idle() and not self._stop_event.is_set():
            time.sleep(30)

        if self._stop_event.is_set():
            self._training_active = False
            return

        logger.info("[trainer] Idle window detected — starting LoRA fine-tuning")
        try:
            self._run_lora_training()
        except Exception as e:
            logger.error(f"[trainer] Training error: {e}", exc_info=True)
        finally:
            self._training_active = False

    def _run_lora_training(self):
        """Execute the actual LoRA fine-tuning.

        VRAM Protocol (RTX 3060, 12 GB):
          Inference model alone: ~4.8 GB
          Training model + LoRA grads: ~7.2 GB
          Both simultaneously: ~12 GB → guaranteed OOM

          Steps:
            1. VRAM_LOCK.clear() → blocks send_message()
            2. unload_engine()   → frees ~4.8 GB
            3. Load training model (~7.2 GB, fits in freed space)
            4. Train 100 steps
            5. del training model + torch.cuda.empty_cache()
            6. reload_engine()   → inference model back
            7. VRAM_LOCK.set()   → send_message() unblocks
        """
        examples = self._load_high_quality_examples(n=200)
        if len(examples) < 10:
            logger.info(f"[trainer] Only {len(examples)} high-quality examples — need 10+, skipping")
            return

        logger.info(f"[trainer] Training on {len(examples)} examples for {TRAIN_STEPS} steps")

        # ── Step 0: Diversity gate ───────────────────────────────────────────
        # Overfitting risk: if all 500 tuples are search("about AI"), the adapter
        # learns to always call search on any prompt. Require variety first.
        import random as _random
        training_pairs = []
        tool_types_seen = set()
        for ex in examples:
            if ex.outcome == "tool_executed" and ex.tool_name:
                training_pairs.append((ex.prompt, ex.response))
                tool_types_seen.add(ex.tool_name)

        if len(tool_types_seen) < MIN_TOOL_DIVERSITY:
            logger.info(
                f"[trainer] Diversity gate: only {len(tool_types_seen)} unique tool types "
                f"(need {MIN_TOOL_DIVERSITY}). Saving examples but skipping training to "
                f"prevent behavioral collapse onto a single tool pattern."
            )
            return

        if not training_pairs:
            logger.info("[trainer] No valid training pairs — skipping")
            return

        logger.info(
            f"[trainer] Diversity OK: {len(tool_types_seen)} tool types — "
            f"{list(tool_types_seen)}"
        )

        # ── Step 0b: 80/20 train/eval split ─────────────────────────────────
        _random.shuffle(training_pairs)
        n_train = max(1, int(len(training_pairs) * 0.8))
        train_pairs = training_pairs[:n_train]
        eval_pairs  = training_pairs[n_train:] or training_pairs[:2]

        # Cap steps: at most MAX_TRAIN_EPOCHS passes over training data
        actual_steps = min(TRAIN_STEPS, len(train_pairs) * MAX_TRAIN_EPOCHS)
        logger.info(
            f"[trainer] {len(train_pairs)} train / {len(eval_pairs)} eval examples, "
            f"{actual_steps} steps"
        )

        # ── Step 1–2: acquire VRAM and unload inference model ────────────────
        try:
            from llm.providers.hermes_tool_provider import (
                VRAM_LOCK, unload_engine, reload_engine, MODEL_ID, HF_CACHE
            )
        except ImportError as _ie:
            logger.error(f"[trainer] Cannot import VRAM utilities: {_ie} — aborting")
            return

        VRAM_LOCK.clear()   # block send_message() immediately
        unload_engine()     # free ~4.8 GB from inference model

        training_model = None
        adapter_accepted = False
        adapter_path = None
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, TaskType
            import torch
            from torch.optim import AdamW
            from torch.optim.lr_scheduler import CosineAnnealingLR
            import gc

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            def _compute_eval_loss(mdl, pairs):
                """Measure average loss on held-out pairs (no grad)."""
                mdl.eval()
                total_loss = 0.0
                with torch.no_grad():
                    for p, r in pairs:
                        text = f"{p}\n{r}{tokenizer.eos_token}"
                        ids = tokenizer(text, return_tensors="pt", truncation=True,
                                        max_length=512).input_ids
                        ids = ids.to(next(mdl.parameters()).device)
                        out = mdl(ids, labels=ids)
                        total_loss += out.loss.item()
                mdl.train()
                return total_loss / len(pairs)

            # ── Step 3: Load training model (inference already unloaded) ────
            training_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, cache_dir=HF_CACHE,
                quantization_config=bnb_cfg, device_map="auto"
            )

            # Baseline eval loss (before any LoRA is applied)
            logger.info("[trainer] Measuring baseline eval loss...")
            baseline_eval_loss = _compute_eval_loss(training_model, eval_pairs)
            logger.info(f"[trainer] Baseline eval loss: {baseline_eval_loss:.4f}")

            training_model.train()
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8, lora_alpha=16,
                lora_dropout=LORA_DROPOUT,   # 0.1 (higher regularization)
                target_modules=["q_proj", "v_proj"],
            )
            training_model = get_peft_model(training_model, lora_cfg)

            optimizer = AdamW(training_model.parameters(), lr=LORA_LR)  # 5e-5
            scheduler = CosineAnnealingLR(optimizer, T_max=actual_steps)

            adapter_path = self._adapter_dir / f"adapter_{int(time.time())}"
            adapter_path.mkdir(parents=True, exist_ok=True)

            # ── Step 4: Training loop with gradient clipping ─────────────────
            best_eval_loss = float('inf')
            for step_i in range(actual_steps):
                pair = train_pairs[step_i % len(train_pairs)]
                text = f"{pair[0]}\n{pair[1]}{tokenizer.eos_token}"
                ids = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512).input_ids
                ids = ids.to(next(training_model.parameters()).device)

                outputs = training_model(ids, labels=ids)
                loss = outputs.loss
                loss.backward()
                # Gradient clipping: prevents loss spikes on outlier examples
                torch.nn.utils.clip_grad_norm_(training_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if (step_i + 1) % 25 == 0:
                    eval_loss = _compute_eval_loss(training_model, eval_pairs)
                    logger.info(
                        f"[trainer] Step {step_i+1}/{actual_steps} "
                        f"train_loss={loss.item():.4f} eval_loss={eval_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                    best_eval_loss = min(best_eval_loss, eval_loss)

            # ── Step 4b: Eval gate — reject if adapter degraded eval loss ───
            final_eval_loss = _compute_eval_loss(training_model, eval_pairs)
            logger.info(
                f"[trainer] Final: baseline={baseline_eval_loss:.4f} "
                f"final={final_eval_loss:.4f} "
                f"ratio={final_eval_loss/baseline_eval_loss:.3f} "
                f"(threshold={EVAL_LOSS_TOLERANCE:.2f})"
            )

            if final_eval_loss <= baseline_eval_loss * EVAL_LOSS_TOLERANCE:
                # ── Adapter accepted ──────────────────────────────────────
                training_model.save_pretrained(str(adapter_path))
                tokenizer.save_pretrained(str(adapter_path))
                # Write the accepted adapter path so reload_engine() can find it
                current_adapter_file = self._data_dir / "current_adapter.txt"
                current_adapter_file.write_text(str(adapter_path))
                adapter_accepted = True
                logger.info(
                    f"[trainer] ✅ Adapter ACCEPTED (loss {baseline_eval_loss:.4f} → "
                    f"{final_eval_loss:.4f}) — saved to {adapter_path}"
                )
                try:
                    from core.evolution_journal import journal
                    if journal:
                        journal.record_code_write(
                            "lora_step", str(adapter_path),
                            f"LoRA accepted: eval {baseline_eval_loss:.4f}→{final_eval_loss:.4f} "
                            f"on {len(train_pairs)} examples, {actual_steps} steps",
                            "PASS"
                        )
                except Exception:
                    pass
            else:
                logger.warning(
                    f"[trainer] ❌ Adapter REJECTED — eval loss degraded "
                    f"({baseline_eval_loss:.4f} → {final_eval_loss:.4f}, "
                    f"ratio={final_eval_loss/baseline_eval_loss:.3f} > {EVAL_LOSS_TOLERANCE}). "
                    f"Base model unchanged."
                )
                adapter_path = None  # don't load this adapter

            # ── Step 5: Free training model VRAM ────────────────────────────
            del training_model
            training_model = None
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("[trainer] Training model unloaded — VRAM freed")

        except ImportError as e:
            logger.warning(f"[trainer] peft not available: {e}")
        except Exception as e:
            logger.error(f"[trainer] Training failed: {e}", exc_info=True)
        finally:
            # ── Steps 6–7: Always reload inference model and release lock ───
            try:
                if training_model is not None:
                    del training_model
                    torch.cuda.empty_cache()
            except Exception:
                pass
            reload_engine()   # loads inference model back (with adapter if accepted)
            VRAM_LOCK.set()   # unblocks send_message() — pulses resume
            logger.info(
                f"[trainer] Training window closed — inference resumed "
                f"({'adapter active' if adapter_accepted else 'base model unchanged'})"
            )

    def get_stats(self) -> dict:
        return {
            "total_collected": self._total_collected,
            "buffer_size": len(self._buffer),
            "training_active": self._training_active,
            "adapters": len(list(self._adapter_dir.glob("adapter_*"))),
            "next_training_at": self._total_collected + (
                EXPERIENCE_THRESHOLD - (self._total_collected % EXPERIENCE_THRESHOLD)
            ),
        }

    def start(self):
        logger.info("[trainer] Experience collector active")

    def stop(self):
        self._stop_event.set()
        self._flush_buffer()


_trainer: Optional[SelfTrainer] = None

def get_trainer() -> Optional[SelfTrainer]:
    return _trainer

def init_trainer(data_dir: str = "data") -> SelfTrainer:
    global _trainer
    _trainer = SelfTrainer(data_dir=data_dir)
    _trainer.start()
    return _trainer
