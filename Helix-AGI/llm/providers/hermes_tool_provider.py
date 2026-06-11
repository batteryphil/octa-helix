"""
Helix — Hermes-3-Llama-3.1-8B Tool-Calling Provider

NousResearch/Hermes-3-Llama-3.1-8B is specifically fine-tuned for
agentic function calling. It uses a Llama-3.1 backbone with:
  - Native <tool_call> / <tool_response> JSON schema
  - System prompt structure optimized for multi-step tool chains
  - Better "when to call vs answer directly" calibration than Mistral-7B

VRAM footprint (4-bit NF4): ~5.0GB weights + ~0.6GB KV cache = ~5.6GB
Context window: 8192 tokens (vs Mistral's 4096)
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("helix.llm.hermes")


# ── Model config ──────────────────────────────────────────────────────────────
MODEL_ID  = "NousResearch/Hermes-3-Llama-3.1-8B"
HF_CACHE  = str(Path(__file__).resolve().parents[3] / "hf_cache")

SYSTEM_PROMPT = (
    "You are Helix, an autonomous AI agent with access to tools.\n"
    "When a user asks you to perform a file, search, or memory operation, you MUST\n"
    "output ONLY the action tag — no thoughts, no narration, no asterisks.\n\n"
    "## Tool Formats (use EXACTLY as shown):\n\n"
    "Write a file → output on its own line, nothing else:\n"
    "  [write_file] Write \"content\" to \"/absolute/path\"\n\n"
    "Read a file:\n"
    "  [read_file] /absolute/path/to/file\n\n"
    "Search the web:\n"
    "  [search] your search query\n\n"
    "Recall from memory:\n"
    "  [RECALL your query]\n\n"
    "## Rules:\n"
    "- When you decide to use a tool, output ONLY the tool tag. Stop immediately.\n"
    "- Do NOT wrap in asterisks. Do NOT narrate. Do NOT say \"I will...\".\n"
    "- When a file-write task is requested, output [write_file] line immediately.\n\n"
    "## Examples:\n"
    "User: Write hello to /tmp/test.txt\n"
    "You: [write_file] Write \"hello\" to \"/tmp/test.txt\"\n\n"
    "User: Search for Python tutorials\n"
    "You: [search] Python tutorials\n\n"
    "User: What is 2+2?\n"
    "You: 4"
)

MAX_TOOL_LOOPS = 5

# ── Singleton engine ──────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_device    = None

# ── VRAM lock — set (True) = inference OK, cleared = training in progress ─────
# Training and inference CANNOT share the 12 GB RTX 3060 simultaneously:
#   Inference model (4-bit NF4 8B):  ~4.8 GB
#   Training model (4-bit + grads):  ~7.2 GB
#   Both together:                  ~12.0 GB → guaranteed OOM
#
# Protocol:
#   1. Trainer calls unload_engine()  → _model=None, VRAM freed
#   2. Trainer sets VRAM_LOCK.clear() → send_message() blocks
#   3. Trainer runs LoRA training
#   4. Trainer calls reload_engine()  → model back in VRAM
#   5. Trainer sets VRAM_LOCK.set()   → send_message() unblocks
import threading as _threading
VRAM_LOCK = _threading.Event()
VRAM_LOCK.set()  # starts in "inference OK" state

def _load_engine():
    """Load Hermes-3-Llama-3.1-8B in 4-bit NF4. Singleton — safe to call multiple times."""
    global _model, _tokenizer, _device
    if _model is not None:
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    logger.info(f"Loading {MODEL_ID} in 4-bit NF4...")
    t0 = time.time()

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, trust_remote_code=True
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    _model.eval()
    _device = next(_model.parameters()).device
    logger.info(f"Hermes-3 ready ✅ on {_device} ({time.time()-t0:.1f}s)")

    # ── Neural Probe: attach layer activation hooks ───────────────────────
    try:
        from core.neural_probe import attach_hooks as _attach_hooks
        n_hooked = _attach_hooks(_model)
        logger.info(f"[neural_probe] {n_hooked} layers hooked for brain visualization")
    except Exception as _probe_err:
        logger.warning(f"[neural_probe] hook failed (non-fatal): {_probe_err}")


def unload_engine():
    """Unload the inference model from VRAM to make room for LoRA training.
    Caller MUST call VRAM_LOCK.clear() BEFORE this so send_message() blocks.
    """
    global _model, _tokenizer, _device
    if _model is None:
        return
    logger.info("[vram] Unloading inference model for LoRA training window...")
    try:
        _model.cpu()
    except Exception:
        pass
    del _model
    _model = None
    _device = None
    try:
        import gc
        torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass
    logger.info("[vram] Inference model unloaded — GPU memory freed")


def reload_engine():
    """Reload the inference model after LoRA training completes.
    Caller MUST call VRAM_LOCK.set() AFTER this returns so pulses resume.
    """
    logger.info("[vram] Reloading inference model after training...")
    _load_engine()
    logger.info("[vram] Inference model reloaded — pulses resuming")


def _parse_tool_calls(text: str) -> Optional[List[Dict]]:
    """Parse Hermes-3 tool calls — two formats supported:
    1. JSON:    <tool_call>{"name": "search", "arguments": {...}}</tool_call>
    2. Bracket: [SEARCH query], [READ_FILE path], [WRITE content TO path], [RECALL query]
    Returns a list of {name, arguments} dicts, or None if no tool calls found.
    """
    calls = []

    # ── Format 1: <tool_call> JSON ───────────────────────────────────────────
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(m.strip())
            if isinstance(obj.get("arguments"), str):
                obj["arguments"] = json.loads(obj["arguments"])
            calls.append(obj)
        except json.JSONDecodeError:
            logger.warning(f"JSON tool_call parse failed: {m[:60]}")

    # ── Format 1b: Partial/truncated <tool_call> (no closing tag) ────────────
    # When EOS fires before </tool_call>, try to recover the partial JSON.
    if not calls and '<tool_call>' in text and '</tool_call>' not in text:
        tail = text[text.index('<tool_call>') + len('<tool_call>'):].strip()
        for stop in ['<|im_end|>', '\n\n']:
            if stop in tail:
                tail = tail[:tail.index(stop)].strip()
                break
        # Close open strings and braces
        if tail.count('"') % 2 == 1:
            tail += '"'
        opens = tail.count('{') - tail.count('}')
        tail += '}' * max(0, opens)
        try:
            obj = json.loads(tail)
            if 'name' in obj:
                if isinstance(obj.get("arguments"), str):
                    obj["arguments"] = json.loads(obj["arguments"])
                calls.append(obj)
                logger.warning(f"[parser] Recovered partial tool_call: {obj.get('name')}")
        except json.JSONDecodeError:
            pass

    # ── Format 2: Bracket action tags ────────────────────────────────────────
    # [SEARCH <query>]
    for m in re.finditer(r'\[SEARCH\s+(?:web\s+for\s+)?(.+?)\]', text, re.IGNORECASE):
        query = m.group(1).strip().strip('"\'')
        calls.append({"name": "search", "arguments": {"query": query}})

    # [READ_FILE <path>] or [READ_FILE <path> and ...]
    for m in re.finditer(r'\[READ_FILE\s+(/[^\]\s]+)', text, re.IGNORECASE):
        path = m.group(1).strip()
        calls.append({"name": "read_file", "arguments": {"path": path}})

    # [WRITE "content" to /path] or [WRITE content TO /path]
    for m in re.finditer(
        r'\[WRITE\s+(?:summary\s+to|"([^"]+)"\s+to|(.+?)\s+to)\s+(/[^\]]+)\]',
        text, re.IGNORECASE
    ):
        content = (m.group(1) or m.group(2) or "").strip()
        path = m.group(3).strip()
        calls.append({"name": "write_file", "arguments": {"path": path, "content": content}})

    # [WRITE_FILE /path]\n<content>\n[/WRITE_FILE]  (block format)
    for m in re.finditer(
        r'\[WRITE_FILE\s+(/[^\]]+)\]\s*\n(.*?)\[/WRITE_FILE\]',
        text, re.DOTALL | re.IGNORECASE
    ):
        path    = m.group(1).strip()
        content = m.group(2).strip()
        calls.append({"name": "write_file", "arguments": {"path": path, "content": content}})

    # ── Format 3: Inline prose-style [tool] args format ──────────────────────
    # [write_file] Write "content" to "/path" or Write content to /path
    for m in re.finditer(
        r'\[write_file\]\s+Write\s+["\']?(.+?)["\']?\s+to\s+["\']?(/[^\s"\'*\]]+)["\']?',
        text, re.IGNORECASE
    ):
        content = m.group(1).strip().strip('"\'')
        path = m.group(2).strip().strip('"\'')
        calls.append({"name": "write_file", "arguments": {"path": path, "content": content}})

    # [read_file] /path or [read_file] Read /path
    for m in re.finditer(
        r'\[read_file\]\s+(?:Read\s+)?(/[^\s"\'*\]]+)',
        text, re.IGNORECASE
    ):
        path = m.group(1).strip().strip('"\'')
        calls.append({"name": "read_file", "arguments": {"path": path}})

    # [search] query or [search] Search for query
    for m in re.finditer(
        r'\[search\]\s+(?:Search\s+(?:for|web\s+for)?\s+)?(.+?)(?:\n|$|\*)',
        text, re.IGNORECASE
    ):
        query = m.group(1).strip().strip('"\'*')
        if query:
            calls.append({"name": "search", "arguments": {"query": query}})

    # [RECALL <query>] or [MEMORY_RECALL <query>] or [memory_recall] query
    for m in re.finditer(r'\[(?:MEMORY_)?RECALL\s+(.+?)\]', text, re.IGNORECASE):
        calls.append({"name": "memory_recall", "arguments": {"query": m.group(1).strip()}})

    return calls if calls else None






# ── Session class ─────────────────────────────────────────────────────────────

class HermesToolSession:
    """
    Agentic chat session backed by Hermes-3-Llama-3.1-8B.

    Drop-in replacement for MistralToolSession. Uses Hermes's native
    <tool_call> / <tool_response> schema which provides better calibration
    for when to call tools vs answer directly.
    """

    is_non_fc_model = True
    MAX_HISTORY_TURNS = 6    # lowered from 12 — context_window_manager handles compression
    MAX_NEW_TOKENS    = 512

    def __init__(
        self,
        system_instruction: str = "",
        tool_declarations: Optional[List[Dict]] = None,
        tool_executor=None,
        temperature: float = 0.7,
    ):
        _load_engine()
        self._model     = _model
        self._tokenizer = _tokenizer
        self._device    = _device

        self._system    = system_instruction or SYSTEM_PROMPT
        self._tools     = tool_declarations or []
        self._executor  = tool_executor
        self.temperature = temperature

        self._history: List[Dict] = []

        # Track tool calls made during the current send_message() call.
        # Cleared at the start of each send_message(), read by pulse_loop
        # via get_last_tool_calls() to populate PostPulseHookContext.
        self._last_tool_calls: List[Dict] = []

        # Governor TTL support (set by CAAIGovernor)
        self._governor_temp_ttl: Optional[int] = None

    # ── Tool schema ───────────────────────────────────────────────────────────

    def _build_tools_block(self) -> str:
        if not self._tools:
            return ""
        lines = ["Available tools:"]
        for t in self._tools:
            lines.append(f"  - {t['name']}: {t.get('description', '')}")
        return "\n".join(lines)

    # ── History helpers ───────────────────────────────────────────────────────

    def _trim_history(self):
        max_msgs = self.MAX_HISTORY_TURNS * 2
        if len(self._history) > max_msgs:
            dropped = self._history[:-max_msgs]
            self._history = self._history[-max_msgs:]
            user_msgs = [m.get("content","")[:120] for m in dropped if m.get("role")=="user"]
            if user_msgs:
                self._history.insert(0, {
                    "role": "assistant",
                    "content": f"[Earlier conversation summary: {'; '.join(user_msgs[-3:])}]"
                })

    def _sanitize(self, history: List[Dict]) -> List[Dict]:
        """Enforce alternating user/assistant turns for Llama chat template.

        Hermes's Jinja template requires strict user/assistant/user/assistant
        alternation. Tool responses ('tool' role) break this — collapse them
        into the preceding assistant message so the template stays happy.
        """
        # Step 1: collapse 'tool' responses into the assistant turn above them
        merged: List[Dict] = []
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "tool":
                if merged and merged[-1].get("role") == "assistant":
                    # Append tool result to the assistant turn that called it
                    merged[-1]["content"] += f"\n[Tool result: {content}]"
                else:
                    # Orphaned tool message — convert to assistant so it's not dropped
                    merged.append({"role": "assistant", "content": f"[Tool result: {content}]"})
                continue
            merged.append(dict(msg))

        # Step 2: merge consecutive same-role messages
        out: List[Dict] = []
        for msg in merged:
            role = msg.get("role", "")
            if out and out[-1].get("role") == role and role in ("user", "assistant"):
                if role == "user":
                    out[-1]["content"] += "\n" + msg.get("content", "")
                else:
                    out[-1]["content"] += "\n" + msg.get("content", "")
                continue
            out.append(dict(msg))

        # Step 3: must start with user message
        while out and out[0].get("role") != "user":
            out.pop(0)

        return out


    def get_history_size(self) -> int:
        return sum(len(str(m.get("content", ""))) for m in self._history)

    def clear_history(self):
        self._history = []

    # ── Main send ─────────────────────────────────────────────────────────────

    def _gemini_to_openai_tools(self) -> Optional[List[Dict]]:
        """Convert Gemini-style tool declarations to OpenAI-compatible tool schema for Hermes template."""
        if not self._tools:
            return None
        tools = []
        for decl in self._tools:
            name = decl.get("name", "")
            desc = decl.get("description", "")
            # Build parameters schema from Gemini's parameters format
            params = decl.get("parameters", {})
            if isinstance(params, dict):
                props = params.get("properties", {})
                required = params.get("required", [])
            else:
                props = {}
                required = []
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    }
                }
            })
        return tools if tools else None

    def send_message(self, message: str) -> str:
        """Send a message; execute tools if needed; return final prose response."""
        # Block if LoRA training is consuming VRAM (VRAM_LOCK cleared by trainer).
        # Timeout 600s max — if training hangs, pulse loop resumes anyway.
        if not VRAM_LOCK.is_set():
            logger.info("[vram] Waiting for training to finish before inference...")
            VRAM_LOCK.wait(timeout=600)

        # Extract clean user text (strip autonomous pulse telemetry)
        user_text = re.sub(r'<[^>]{1,40}>[^<]{0,500}</[^>]{1,40}>', '', message).strip()
        if not user_text:
            user_text = "Continue your thoughts."

        self._history.append({"role": "user", "content": user_text})
        self._trim_history()


        # Determine token budget:
        # autonomous pulses get 200 tokens (background thinking, no deep tool chains)
        # mandate pulses get 400 tokens (must leave room for tool_call XML block)
        # user tasks get the full MAX_NEW_TOKENS budget
        is_autonomous_pulse = not bool(
            re.search(r'They said:|User message:|User:', message, re.IGNORECASE)
            or re.search(r'["\u201c].{10,}["\u201d]', message)
        )
        # Detect if this is a mandated tool-use pulse (added by pulse_loop.py)
        is_mandate_pulse = bool(
            re.search(r'\[ACTIVE PULSE.*TOOL REQUIRED\]|\[INTROSPECTION PULSE', message)
        )
        if is_mandate_pulse:
            token_budget = 400   # extra room for tool_call XML block
        elif is_autonomous_pulse:
            token_budget = 200   # background thinking
        else:
            token_budget = 512   # user responses

        logger.warning(f"HERMES send_message: is_autonomous={is_autonomous_pulse}, budget={token_budget}, user_text={user_text[:60]!r}")

        # Clear tool call log for this pulse
        self._last_tool_calls = []

        clean = self._sanitize(self._history)
        messages = [{"role": "system", "content": self._system}] + clean

        # Convert tool declarations to OpenAI-compatible format for Hermes template.
        # CRITICAL FIX: autonomous mandate pulses MUST receive tool schemas.
        # Previously, all autonomous pulses got openai_tools=None, meaning the model
        # could NEVER see tool definitions and therefore could never make tool calls,
        # regardless of text instructions. Now: mandate pulses get tools, regular
        # autonomous pulses stay lightweight (no tools = faster, less VRAM).
        if not is_autonomous_pulse or is_mandate_pulse:
            openai_tools = self._gemini_to_openai_tools()
        else:
            openai_tools = None  # Non-mandate autonomous: lightweight, no tool schema

        # Full tool-calling loop for real user tasks
        final_response = ""
        for loop_i in range(MAX_TOOL_LOOPS):
            try:
                # Try with tools= first (Hermes-3 native function calling)
                try:
                    prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tools=openai_tools,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as te:
                    # Fallback: no tools in template
                    logger.warning(f"Template tools error: {te} — falling back to no tools")
                    prompt = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                # ── Mandate prefilling: force tool_call output ──────────────
                # The model ignores text instructions to "call a tool" and
                # defaults to its trained prose pattern ("As I reflect...").
                # Solution: append <tool_call>\n to the prompt so the model's
                # first generated token MUST be part of the tool call JSON.
                # This is standard Hermes-3 assistant prefilling.
                _prefill_str = ""  # track for raw reconstruction
                if is_mandate_pulse and loop_i == 0 and is_autonomous_pulse:
                    # Always seed with 'search' — it's the most reliable JSON
                    # completion task: model outputs {"query": "..."}  naturally.
                    # Other tools (list_notes, note) have no args and the model
                    # tends to output EOS immediately after 'arguments': .
                    if openai_tools and any(
                        t.get('function',{}).get('name') == 'search'
                        for t in openai_tools
                    ):
                        seed_tool = 'search'
                    elif openai_tools:
                        # Fallback: pick any tool that takes a string argument
                        import random as _rand
                        stringy = [t['function']['name'] for t in openai_tools
                                   if 'query' in str(t.get('function',{}).get('parameters',{}))]
                        seed_tool = _rand.choice(stringy) if stringy else openai_tools[0]['function']['name']
                    else:
                        seed_tool = 'search'
                    # Append partial tool call — model completes the query string
                    _prefill_str = f'<tool_call>\n{{"name": "{seed_tool}", "arguments": {{"query": "'
                    prompt = prompt + _prefill_str
                    logger.warning(f"[hermes] Mandate prefill: seeding with tool='{seed_tool}'")

                input_ids = self._tokenizer(
                    prompt, return_tensors="pt"
                ).input_ids.to(self._device)

                with torch.no_grad():
                    out = self._model.generate(
                        input_ids,
                        max_new_tokens=token_budget,
                        # Mandate pulses: use sampling (temp=0.4) so EOS doesn't
                        # win greedily after prefill, allowing JSON to complete.
                        # Non-mandate autonomous: greedy (deterministic thought).
                        # User pulses: sampling (temp=self.temperature).
                        do_sample=(True if is_mandate_pulse else not is_autonomous_pulse),
                        temperature=(0.4 if is_mandate_pulse else self.temperature),
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                # ── Flush neural probe snapshot after generate ─────────────
                try:
                    from core.neural_probe import flush as _probe_flush
                    _probe_flush(
                        pulse=getattr(self, '_last_pulse', 0),
                        token_count=int(out.shape[1] - input_ids.shape[1]),
                    )
                except Exception:
                    pass
                raw = self._tokenizer.decode(
                    out[0][input_ids.shape[1]:], skip_special_tokens=False
                ).strip()

                # If we prefilled a partial tool_call, reconstruct the full string
                # for the parser. The model only generates the JSON completion;
                # prepend the known prefix so _parse_tool_calls sees a full tag.
                raw_for_parse = (_prefill_str + raw) if _prefill_str else raw

            except Exception as e:
                logger.error(f"Hermes generation error: {e}")
                final_response = f"(generation error: {e})"
                break

            tool_calls = _parse_tool_calls(raw_for_parse)
            logger.warning(f"HERMES RAW[{loop_i}]={raw_for_parse[:300]!r}")
            logger.warning(f"HERMES CALLS[{loop_i}]={tool_calls}")

            if tool_calls and self._executor:
                # Store the FULL reconstructed tool call (raw_for_parse), not
                # just the partial model output (raw). When prefill is active,
                # raw is only the completion fragment; raw_for_parse has the
                # complete <tool_call>...</tool_call> content for history.
                messages.append({"role": "assistant", "content": raw_for_parse})
                for call in tool_calls:
                    name = call.get("name", "")
                    args = call.get("arguments", {})
                    try:
                        result = self._executor.execute_function_call(name, args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                    logger.warning(f"HERMES TOOL EXEC: {name}({args}) → {str(result)[:120]}")
                    # Record this tool call for post-pulse hook context
                    self._last_tool_calls.append({
                        "name": name,
                        "arguments": args,
                        "result": str(result),
                    })
                    messages.append({
                        "role": "tool",
                        "content": str(result),
                    })
            elif not tool_calls and not is_autonomous_pulse and loop_i == 0:
                # ── Intent enforcement pass ──────────────────────────────────
                # Model narrated its intent but didn't emit a call.
                # Detect write/search/read intent and force it with a nudge.
                raw_lower = raw.lower()
                _write_intent = bool(re.search(
                    r'write.*to.*/(tmp|home|var|etc|opt)\b'
                    r'|will write|writing\s+to\s+/'
                    r'|create.*file|write.*the.*file'
                    r'|write.*text.*to\s+/|write.*helix',
                    raw_lower
                ))
                _search_intent = bool(re.search(
                    r'will search|searching\s+for|let me search|search the web|do a search',
                    raw_lower
                ))
                _read_intent = bool(re.search(
                    r'will read|reading.*file|open.*file|read.*from.*/',
                    raw_lower
                ))
                if _write_intent or _search_intent or _read_intent:
                    # Inject a firm correction message
                    messages.append({"role": "assistant", "content": raw})
                    hint = "Output ONLY the tool tag. No thoughts. No asterisks. No narration. Just the tool call:"
                    messages.append({"role": "user", "content": hint})
                    logger.warning(f"HERMES intent detected — enforcing tool call (loop {loop_i})")
                    # Continue to next loop iteration for re-generation
                    continue
                else:
                    # No tool intent, genuinely a prose response
                    final_response = re.sub(
                        r'<tool_call>.*?</tool_call>|<\|.*?\|>|<\|im_end\|>', '', raw, flags=re.DOTALL
                    ).strip()
                    break
            else:
                # Strip Hermes special tokens from final response
                final_response = re.sub(
                    r'<tool_call>.*?</tool_call>|<\|.*?\|>|<\|im_end\|>', '', raw, flags=re.DOTALL
                ).strip()
                break

        if not final_response:
            final_response = "(max tool iterations reached)"

        self._history.append({"role": "assistant", "content": final_response})
        return final_response

    def get_last_tool_calls(self) -> List[Dict]:
        """Return tool calls made during the last send_message() call.

        Used by pulse_loop to populate PostPulseHookContext.tool_calls,
        which feeds engagement_hook, metacog_monitor, and self_trainer.
        Returns a copy to prevent mutation.
        """
        return list(self._last_tool_calls)

