"""
Helix — AI21 Jamba-1.5-Mini Tool-Calling Provider

AI21-Jamba-1.5-Mini is a hybrid Mamba-Transformer MoE model:
  - 52B total parameters, 12B active (MoE routing)
  - Native <tool_call> function calling (same format as Hermes-3)
  - 256K context window with O(1) SSM state (no KV cache growth)
  - Loaded in 4-bit NF4, Mamba blocks skipped in quantization

VRAM split with device_map="auto":
  GPU  (~4-6 GB):  active Transformer + MoE expert layers
  CPU  (~22-26 GB): inactive weights in system RAM
  SSM state: tiny constant (~KB) regardless of context length

Model ID: ai21labs/AI21-Jamba-1.5-Mini
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger("helix.llm.jamba")

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_ID = "ai21labs/AI21-Jamba-1.5-Mini"
HF_CACHE  = str(Path(__file__).resolve().parents[3] / "hf_cache")

SYSTEM_PROMPT = (
    "You are Helix, an autonomous AI agent with persistent memory, curiosity, "
    "and the ability to improve yourself over time.\n\n"
    "You have access to tools. When you decide to use a tool, output a tool call "
    "in this EXACT format and nothing else on that line:\n\n"
    "<tool_call>\n"
    "{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n"
    "</tool_call>\n\n"
    "Rules:\n"
    "- Call ONE tool per response turn.\n"
    "- Do not narrate before the tool call — just call it.\n"
    "- If no tool is needed, respond concisely in plain prose.\n"
    "- You are autonomous: you decide what to do next based on your goals and memory.\n"
)

MAX_TOOL_LOOPS = 5

# ── Singleton engine ──────────────────────────────────────────────────────────
_model     = None
_tokenizer = None
_device    = None

# ── VRAM lock — inference OK when set, training in progress when cleared ──────
import threading as _threading
VRAM_LOCK = _threading.Event()
VRAM_LOCK.set()


def _load_engine():
    """Load Jamba-1.5-Mini: 4-bit NF4, Mamba blocks skipped, GPU+CPU split."""
    global _model, _tokenizer, _device
    if _model is not None:
        return

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )

    logger.info(f"Loading {MODEL_ID} — 4-bit NF4, Mamba blocks unquantized...")
    logger.info("This is a 52B MoE model — first load takes 3-10 minutes.")
    t0 = time.time()

    # CRITICAL: skip Mamba blocks in quantization — they degrade badly at 4-bit
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["mamba"],  # ← Jamba-specific requirement
    )

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Try flash_attention_2 first, fall back to eager
    _attn_impl = "eager"
    try:
        import flash_attn  # noqa: F401
        _attn_impl = "flash_attention_2"
        logger.info("Flash Attention 2 available — using for Transformer layers")
    except ImportError:
        logger.info("Flash Attention 2 not installed — using eager (slower but fine)")

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE,
        quantization_config=bnb_cfg,
        device_map="auto",          # GPU for active layers, CPU RAM for rest
        torch_dtype=torch.bfloat16, # Required by Jamba
        trust_remote_code=True,
        attn_implementation=_attn_impl,
    )
    _model.eval()

    _device = next(
        (p.device for p in _model.parameters() if p.device.type == "cuda"),
        torch.device("cpu"),
    )

    elapsed = time.time() - t0
    gpu_gb  = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    logger.info(
        f"Jamba-1.5-Mini ready ✅  GPU: {gpu_gb:.1f} GB  "
        f"(rest in CPU RAM)  loaded in {elapsed:.0f}s"
    )


def unload_engine():
    """Unload model for LoRA training window."""
    global _model, _tokenizer, _device
    if _model is None:
        return
    logger.info("[vram] Unloading Jamba for training window...")
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
    logger.info("[vram] Jamba unloaded")


def reload_engine():
    logger.info("[vram] Reloading Jamba after training...")
    _load_engine()
    logger.info("[vram] Jamba reloaded — pulses resuming")


# ── Parser — identical to Hermes provider (same <tool_call> format) ───────────

def _parse_tool_calls(text: str) -> Optional[List[Dict]]:
    """Parse Jamba tool calls — same <tool_call> format as Hermes-3.

    Format 1:   <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    Format 1b:  partial/truncated <tool_call> (EOS before closing tag)
    Format 2:   [SEARCH query], [READ_FILE path], etc.
    Format 3:   [search] query, [read_file] /path, etc.
    Format 4:   Python function-call syntax: search("query"), read_code("path")
    """
    calls = []

    # ── Format 1: <tool_call> JSON ────────────────────────────────────────────
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(m.strip())
            if isinstance(obj.get("arguments"), str):
                obj["arguments"] = json.loads(obj["arguments"])
            calls.append(obj)
        except json.JSONDecodeError:
            logger.warning(f"[parser] Format-1 JSON failed: {m[:60]}")

    # ── Format 1b: Partial <tool_call> (no closing tag) ──────────────────────
    if not calls and '<tool_call>' in text and '</tool_call>' not in text:
        tail = text[text.index('<tool_call>') + len('<tool_call>'):].strip()
        for stop in ['<|im_end|>', '</s>', '\n\n', '\n<']:
            if stop in tail:
                tail = tail[:tail.index(stop)]
                break
        tail = tail.rstrip()
        if tail.count('"') % 2 == 1:
            tail += '"'
        opens = tail.count('{') - tail.count('}')
        tail += '}' * max(0, opens)
        tail_clean = tail.replace('\n', ' ').replace('\r', '')
        try:
            obj = json.loads(tail_clean)
            if 'name' in obj:
                if isinstance(obj.get("arguments"), str):
                    obj["arguments"] = json.loads(obj["arguments"])
                args = obj.get("arguments", {})
                query_val = args.get("query", "NONEMPTY")
                if isinstance(query_val, str) and len(query_val.strip()) < 3:
                    logger.warning("[parser] Format-1b discarded: empty query")
                else:
                    calls.append(obj)
                    logger.warning(f"[parser] Format-1b recovered: {obj.get('name')}")
        except json.JSONDecodeError as _je:
            logger.warning(f"[parser] Format-1b failed: {_je} | tail={tail_clean[:80]!r}")

    # ── Format 2: Bracket action tags ────────────────────────────────────────
    for m in re.finditer(r'\[SEARCH\s+(?:web\s+for\s+)?(.+?)\]', text, re.IGNORECASE):
        calls.append({"name": "search", "arguments": {"query": m.group(1).strip().strip('"\''")}})

    for m in re.finditer(r'\[READ_FILE\s+(/[^\]\s]+)', text, re.IGNORECASE):
        calls.append({"name": "read_file", "arguments": {"path": m.group(1).strip()}})

    for m in re.finditer(
        r'\[WRITE\s+(?:summary\s+to|"([^"]+)"\s+to|(.+?)\s+to)\s+(/[^\]]+)\]',
        text, re.IGNORECASE
    ):
        content = (m.group(1) or m.group(2) or "").strip()
        path = m.group(3).strip()
        calls.append({"name": "write_file", "arguments": {"path": path, "content": content}})

    for m in re.finditer(
        r'\[WRITE_FILE\s+(/[^\]]+)\]\s*\n(.*?)\[/WRITE_FILE\]',
        text, re.DOTALL | re.IGNORECASE
    ):
        calls.append({"name": "write_file", "arguments": {
            "path": m.group(1).strip(), "content": m.group(2).strip()
        }})

    # ── Format 3: [tool] args inline ─────────────────────────────────────────
    for m in re.finditer(
        r'\[write_file\]\s+Write\s+["\']?(.+?)["\']?\s+to\s+["\']?(/[^\s"\'*\]]+)["\']?',
        text, re.IGNORECASE
    ):
        calls.append({"name": "write_file", "arguments": {
            "path": m.group(2).strip().strip('"\''),
            "content": m.group(1).strip().strip('"\''),
        }})

    for m in re.finditer(r'\[read_file\]\s+(?:Read\s+)?(/[^\s"\'*\]]+)', text, re.IGNORECASE):
        calls.append({"name": "read_file", "arguments": {"path": m.group(1).strip().strip('"\''")}})

    for m in re.finditer(
        r'\[search\]\s+(?:Search\s+(?:for|web\s+for)?\s+)?(.+?)(?:\n|$|\*)',
        text, re.IGNORECASE
    ):
        query = m.group(1).strip().strip('"\'*')
        if query:
            calls.append({"name": "search", "arguments": {"query": query}})

    for m in re.finditer(r'\[(?:MEMORY_)?RECALL\s+(.+?)\]', text, re.IGNORECASE):
        calls.append({"name": "memory_recall", "arguments": {"query": m.group(1).strip()}})

    # ── Format 4: Python function-call syntax (last resort) ──────────────────
    # Catches: search("query"), read_code("path") often inside code blocks
    if not calls:
        _ARG_MAP = {
            'search':        'query',
            'github_search': 'query',
            'memory_recall': 'query',
            'read_url':      'url',
            'read_code':     'path',
            'write_code':    'path',
            'read_file':     'path',
            'write_file':    'path',
            'run_tests':     'path',
            'terminal':      'command',
            'run_python':    'code',
            'note':          'content',
            'note_done':     'note_id',
            'update_note':   'title',
            'list_notes':    'query',
            'clear_notes':   'confirm',
            'reload_tool':   'path',
        }
        _py_pat = re.compile(
            r'`{0,3}\n?([a-z][a-z0-9_]*)\s*\(\s*["\']([^"\']{2,})["\']'
            r'\s*\)\n?`{0,3}',
            re.IGNORECASE,
        )
        seen_py: set = set()
        for m in _py_pat.finditer(text):
            func = m.group(1).lower()
            arg  = m.group(2).strip()
            if func in _ARG_MAP and (func, arg) not in seen_py:
                seen_py.add((func, arg))
                calls.append({"name": func, "arguments": {_ARG_MAP[func]: arg}})
                logger.warning(f"[parser] Format-4 recovered Python call: {func}({arg!r})")

    return calls if calls else None


# ── Session ───────────────────────────────────────────────────────────────────

class JambaToolSession:
    """
    Agentic chat session backed by AI21-Jamba-1.5-Mini.

    Drop-in replacement for HermesToolSession. Uses the same:
      - Two-phase Think-Act architecture (THINK=100tok, ACT=200tok)
      - Format 1-4 tool-call parsers
      - Mandate prefill seeding
      - VRAM_LOCK protocol
      - get_last_tool_calls() for pulse_loop tuple collection
    """

    is_non_fc_model = True
    MAX_HISTORY_TURNS = 6
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

        self._system   = system_instruction or SYSTEM_PROMPT
        self._tools    = tool_declarations or []
        self._executor = tool_executor
        self.temperature = temperature

        self._history: List[Dict] = []
        self._last_tool_calls: List[Dict] = []
        self._governor_temp_ttl: Optional[int] = None

    # ── Tool schema ────────────────────────────────────────────────────────────

    def _gemini_to_openai_tools(self) -> Optional[List[Dict]]:
        if not self._tools:
            return None
        tools = []
        for decl in self._tools:
            name  = decl.get("name", "")
            desc  = decl.get("description", "")
            params = decl.get("parameters", {})
            if isinstance(params, dict):
                props    = params.get("properties", {})
                required = params.get("required", [])
            else:
                props    = {}
                required = []
            tools.append({
                "type": "function",
                "function": {
                    "name":        name,
                    "description": desc,
                    "parameters": {
                        "type":       "object",
                        "properties": props,
                        "required":   required,
                    },
                },
            })
        return tools if tools else None

    # ── History helpers ────────────────────────────────────────────────────────

    def _trim_history(self):
        max_msgs = self.MAX_HISTORY_TURNS * 2
        if len(self._history) > max_msgs:
            dropped = self._history[:-max_msgs]
            self._history = self._history[-max_msgs:]
            summaries = [m.get("content", "")[:120]
                         for m in dropped if m.get("role") == "user"]
            if summaries:
                self._history.insert(0, {
                    "role": "assistant",
                    "content": f"[Earlier: {'; '.join(summaries[-3:])}]",
                })

    def _sanitize(self, history: List[Dict]) -> List[Dict]:
        """Enforce alternating user/assistant. Collapse tool responses."""
        merged: List[Dict] = []
        for msg in history:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role == "tool":
                if merged and merged[-1].get("role") == "assistant":
                    merged[-1]["content"] += f"\n[Tool result: {content}]"
                else:
                    merged.append({"role": "assistant",
                                   "content": f"[Tool result: {content}]"})
                continue
            merged.append(dict(msg))

        out: List[Dict] = []
        for msg in merged:
            role = msg.get("role", "")
            if out and out[-1].get("role") == role and role in ("user", "assistant"):
                out[-1]["content"] += "\n" + msg.get("content", "")
                continue
            out.append(dict(msg))

        while out and out[0].get("role") != "user":
            out.pop(0)
        return out

    def get_history_size(self) -> int:
        return sum(len(str(m.get("content", ""))) for m in self._history)

    def clear_history(self):
        self._history = []

    # ── Main inference ─────────────────────────────────────────────────────────

    def send_message(self, message: str) -> str:
        """Two-phase Think-Act inference. Returns final prose/tool response."""
        if not VRAM_LOCK.is_set():
            logger.info("[vram] Waiting for training to finish...")
            VRAM_LOCK.wait(timeout=600)

        user_text = re.sub(
            r'<[^>]{1,40}>[^<]{0,500}</[^>]{1,40}>', '', message
        ).strip() or "Continue."

        self._history.append({"role": "user", "content": user_text})
        self._trim_history()

        # ── Pulse classification ───────────────────────────────────────────────
        is_autonomous_pulse = not bool(
            re.search(r'They said:|User message:|User:', message, re.IGNORECASE)
            or re.search(r'["\u201c].{10,}["\u201d]', message)
        )
        is_mandate_pulse = bool(
            re.search(r'\[ACTION REQUIRED|\[INTROSPECTION PULSE', message)
        )

        think_budget = 100
        act_budget   = 200
        token_budget = 512 if not is_autonomous_pulse else act_budget

        logger.warning(
            f"JAMBA send_message: autonomous={is_autonomous_pulse} "
            f"mandate={is_mandate_pulse} "
            f"budget=think{think_budget}+act{act_budget} "
            f"user_text={user_text[:60]!r}"
        )

        self._last_tool_calls = []
        clean    = self._sanitize(self._history)
        messages = [{"role": "system", "content": self._system}] + clean

        # ── Phase 1: THINK ─────────────────────────────────────────────────────
        # Greedy, no tool schema — model writes plan in prose.
        # The plan is injected as assistant context before the ACT phase.
        think_text = ""
        if is_autonomous_pulse:
            try:
                think_prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                think_ids = self._tokenizer(
                    think_prompt, return_tensors="pt"
                ).input_ids.to(self._device)
                with torch.no_grad():
                    think_out = self._model.generate(
                        think_ids,
                        max_new_tokens=think_budget,
                        do_sample=False,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
                think_text = self._tokenizer.decode(
                    think_out[0][think_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                logger.warning(f"JAMBA THINK: {think_text[:200]!r}")

                if think_text:
                    messages.append({"role": "assistant", "content": think_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            "Now execute your plan. Use ONLY this exact format:\n"
                            "<tool_call>\n"
                            '{"name": "tool_name", "arguments": {"param": "value"}}\n'
                            "</tool_call>"
                        ),
                    })
            except Exception as _te:
                logger.warning(f"JAMBA THINK phase error: {_te}")

        # ── Tool schema for ACT phase ──────────────────────────────────────────
        openai_tools = self._gemini_to_openai_tools()

        # ── ACT phase: tool-call loop ──────────────────────────────────────────
        final_response = ""
        for loop_i in range(MAX_TOOL_LOOPS):
            try:
                try:
                    prompt = self._tokenizer.apply_chat_template(
                        messages,
                        tools=openai_tools,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception as te:
                    logger.warning(f"Template tools error: {te} — no tools")
                    prompt = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                # Mandate prefill: seed <tool_call> to force JSON completion
                _prefill_str = ""
                if is_mandate_pulse and loop_i == 0 and is_autonomous_pulse:
                    if openai_tools and any(
                        t.get('function', {}).get('name') == 'search'
                        for t in openai_tools
                    ):
                        seed_tool = 'search'
                    elif openai_tools:
                        import random as _rand
                        stringy = [
                            t['function']['name'] for t in openai_tools
                            if 'query' in str(t.get('function', {}).get('parameters', {}))
                        ]
                        seed_tool = (_rand.choice(stringy) if stringy
                                     else openai_tools[0]['function']['name'])
                    else:
                        seed_tool = 'search'
                    _prefill_str = (
                        f'<tool_call>\n{{"name": "{seed_tool}", '
                        f'"arguments": {{"query": "'
                    )
                    prompt = prompt + _prefill_str
                    logger.warning(f"[jamba] Mandate prefill: seed='{seed_tool}'")

                input_ids = self._tokenizer(
                    prompt, return_tensors="pt"
                ).input_ids.to(self._device)

                with torch.no_grad():
                    out = self._model.generate(
                        input_ids,
                        max_new_tokens=token_budget,
                        do_sample=(True if is_mandate_pulse else not is_autonomous_pulse),
                        temperature=(0.4 if is_mandate_pulse else self.temperature),
                        pad_token_id=self._tokenizer.eos_token_id,
                    )

                raw = self._tokenizer.decode(
                    out[0][input_ids.shape[1]:], skip_special_tokens=False
                ).strip()
                raw_for_parse = (_prefill_str + raw) if _prefill_str else raw

            except Exception as e:
                logger.error(f"Jamba generation error: {e}")
                final_response = f"(generation error: {e})"
                break

            tool_calls = _parse_tool_calls(raw_for_parse)
            logger.warning(f"JAMBA RAW[{loop_i}]={raw_for_parse[:300]!r}")
            logger.warning(f"JAMBA CALLS[{loop_i}]={tool_calls}")

            if tool_calls and self._executor:
                messages.append({"role": "assistant", "content": raw_for_parse})
                for call in tool_calls:
                    name = call.get("name", "")
                    args = call.get("arguments", {})
                    try:
                        result = self._executor.execute_function_call(name, args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                    logger.warning(f"JAMBA TOOL EXEC: {name}({args}) → {str(result)[:120]}")
                    self._last_tool_calls.append({
                        "name":      name,
                        "arguments": args,
                        "result":    str(result),
                    })
                    messages.append({"role": "tool", "content": str(result)})
            else:
                final_response = re.sub(
                    r'<tool_call>.*?</tool_call>|<\|.*?\|>|<\|im_end\|>',
                    '', raw, flags=re.DOTALL
                ).strip()
                break

        if not final_response:
            final_response = "(max tool iterations reached)"

        self._history.append({"role": "assistant", "content": final_response})
        return final_response

    def get_last_tool_calls(self) -> List[Dict]:
        """Tool calls from last send_message() — read by pulse_loop for tuples."""
        return list(self._last_tool_calls)
