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
    MAX_HISTORY_TURNS = 12
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
        """Enforce alternating user/assistant turns for Llama chat template."""
        out = []
        for msg in history:
            role = msg.get("role", "")
            if out and out[-1].get("role") == role and role in ("user", "assistant"):
                if role == "user":
                    out[-1]["content"] += "\n" + msg.get("content", "")
                else:
                    out[-1] = msg
                continue
            out.append(dict(msg))
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
        # Extract clean user text (strip autonomous pulse telemetry)
        user_text = re.sub(r'<[^>]{1,40}>[^<]{0,500}</[^>]{1,40}>', '', message).strip()
        if not user_text:
            user_text = "Continue your thoughts."

        self._history.append({"role": "user", "content": user_text})
        self._trim_history()


        # Determine token budget:
        # autonomous pulses get 80 tokens (no need for deep tool chains)
        # user tasks get the full MAX_NEW_TOKENS budget
        is_autonomous_pulse = not bool(
            re.search(r'They said:|User message:|User:', message, re.IGNORECASE)
            or re.search(r'["\u201c].{10,}["\u201d]', message)
        )
        token_budget = 80 if is_autonomous_pulse else self.MAX_NEW_TOKENS

        logger.warning(f"HERMES send_message: is_autonomous={is_autonomous_pulse}, budget={token_budget}, user_text={user_text[:60]!r}")

        clean = self._sanitize(self._history)
        messages = [{"role": "system", "content": self._system}] + clean

        # Convert tool declarations to OpenAI-compatible format for Hermes template
        openai_tools = self._gemini_to_openai_tools() if not is_autonomous_pulse else None

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

                input_ids = self._tokenizer(
                    prompt, return_tensors="pt"
                ).input_ids.to(self._device)

                with torch.no_grad():
                    out = self._model.generate(
                        input_ids,
                        max_new_tokens=token_budget,
                        do_sample=(not is_autonomous_pulse),
                        temperature=self.temperature,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                raw = self._tokenizer.decode(
                    out[0][input_ids.shape[1]:], skip_special_tokens=False
                ).strip()

            except Exception as e:
                logger.error(f"Hermes generation error: {e}")
                final_response = f"(generation error: {e})"
                break

            tool_calls = _parse_tool_calls(raw)
            logger.warning(f"HERMES RAW[{loop_i}]={raw[:300]!r}")
            logger.warning(f"HERMES CALLS[{loop_i}]={tool_calls}")

            if tool_calls and self._executor:
                messages.append({"role": "assistant", "content": raw})
                for call in tool_calls:
                    name = call.get("name", "")
                    args = call.get("arguments", {})
                    try:
                        result = self._executor.execute_function_call(name, args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                    logger.warning(f"HERMES TOOL EXEC: {name}({args}) → {str(result)[:120]}")
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
        logger.info(f"Hermes response ({len(final_response)} chars): {final_response[:80]}")
        return final_response

