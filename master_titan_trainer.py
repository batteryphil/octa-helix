import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba3_titan_builder import Mamba3Titan
import argparse
import os
import sys
import json
import time
import math
import signal
import collections
import traceback
import threading
import queue

# Reduce CUDA memory fragmentation — must be set before any CUDA initialization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── HuggingFace authentication ───────────────────────────────────────────────
# Must happen before any load_dataset() call.
# Authenticated requests: higher rate limits, more stable streaming over long runs.
_HF_TOKEN = os.environ.get("HF_TOKEN", "os.environ.get("HF_TOKEN","")")
try:
    from huggingface_hub import login as hf_login
    hf_login(token=_HF_TOKEN, add_to_git_credential=False)
    print(f"[HF] Authenticated as hub user (token: {_HF_TOKEN[:8]}...)")
except Exception as _e:
    print(f"[HF] Warning: could not authenticate — {_e}. Streaming as guest.")

# GPU temperature via NVML
try:
    import pynvml
    pynvml.nvmlInit()
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    def get_gpu_temp():
        return pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
except Exception:
    def get_gpu_temp():
        return None

try:
    from huggingface_hub import login
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("WARNING: huggingface_hub, datasets, or transformers not found.")

HF_TOKEN = os.environ.get("HF_TOKEN", "os.environ.get("HF_TOKEN","")")

# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN — catches SIGTERM (kill) and SIGINT (Ctrl+C)
# Sets a flag that the training loop checks after every step.
# ─────────────────────────────────────────────────────────────────────────────
_shutdown_requested = False

def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n[SIGNAL] {sig_name} received — finishing current step then saving checkpoint...")
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT,  _signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE
# Cosine decay with linear warmup. On resume: brief re-warmup from 10% of the
# scheduled LR back to full, to prevent the optimizer-reset gradient spike.
# ─────────────────────────────────────────────────────────────────────────────
def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int = 500) -> float:
    """
    SGDR cosine annealing with warm restarts.
    Used for Phase 1 only (long runs where escaping local minima matters).
    Resets every 40K steps with peak LR decaying 70% each cycle.
    """
    if step < warmup_steps:
        return base_lr * max(step, 1) / warmup_steps

    cycle_len   = 40_000
    decay_rate  = 0.70
    floor_frac  = 0.05

    steps_after_warmup = step - warmup_steps
    cycle_idx   = steps_after_warmup // cycle_len
    cycle_step  = steps_after_warmup  % cycle_len

    cycle_peak  = base_lr * (decay_rate ** cycle_idx)
    progress    = cycle_step / cycle_len
    return cycle_peak * (floor_frac + (1 - floor_frac) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def simple_cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int = 500) -> float:
    """
    Plain cosine decay with linear warmup. No restarts, no cycle decay.
    Used for Phase 2/3 — shorter runs where stability > exploration.
    Decays from base_lr at warmup end to 5% of base_lr at total_steps.
    """
    if step < warmup_steps:
        return base_lr * max(step, 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return base_lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def get_lr(step: int, total_steps: int, base_lr: float,
           warmup_steps: int = 500,
           resume_step: int = 0, resume_warmup: int = 300,
           use_sgdr: bool = True) -> float:
    """
    Returns the LR for `step`.
    use_sgdr=True  → SGDR with warm restarts (Phase 1)
    use_sgdr=False → simple cosine (Phase 2/3)
    If we just resumed, re-warm from 10% → scheduled over resume_warmup steps.
    """
    if use_sgdr:
        scheduled = cosine_lr(step, total_steps, base_lr, warmup_steps)
    else:
        scheduled = simple_cosine_lr(step, total_steps, base_lr, warmup_steps)
    steps_since_resume = step - resume_step
    if resume_step > 0 and steps_since_resume < resume_warmup:
        frac = steps_since_resume / resume_warmup
        return scheduled * (0.10 + 0.90 * frac)
    return scheduled


def apply_lr(optimizer, lr_core: float, lr_head: float, lr_router: float = 0.0):
    for pg in optimizer.param_groups:
        if pg.get('name') == 'router':
            pg['lr'] = lr_router
        elif pg.get('name') == 'head':
            pg['lr'] = lr_head
        else:
            pg['lr'] = lr_core


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-STOP POLICY
# Tracks a rolling window of recent losses. Triggers stop if:
#   • Catastrophic divergence: rolling mean > DIVERGE_THRESHOLD for
#     DIVERGE_PATIENCE consecutive steps (model has exploded, no recovery)
#   • Plateau stop is intentionally NOT triggered — we let it keep training.
# ─────────────────────────────────────────────────────────────────────────────
DIVERGE_THRESHOLD  = 150.0   # if rolling-1000 mean exceeds this → diverged
DIVERGE_PATIENCE   = 1500    # must stay above threshold this many steps

class AutoStop:
    def __init__(self):
        self._window    = collections.deque(maxlen=1000)
        self._bad_steps = 0

    def update(self, loss: float) -> tuple[bool, str]:
        """Returns (should_stop, reason). Call every step."""
        self._window.append(loss)
        mean = sum(self._window) / len(self._window)

        if mean > DIVERGE_THRESHOLD:
            self._bad_steps += 1
            if self._bad_steps >= DIVERGE_PATIENCE:
                return True, (
                    f"DIVERGENCE: rolling-1000 mean loss {mean:.2f} > "
                    f"{DIVERGE_THRESHOLD} for {self._bad_steps} steps. "
                    f"Training has collapsed — stopping to protect checkpoint."
                )
        else:
            self._bad_steps = 0   # reset if it recovers

        return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# DATA PIPELINE — Fixed
# ─────────────────────────────────────────────────────────────────────────────

def get_text_from_item(item, ds_name):
    """Extract clean plain text from each dataset's schema. No raw dicts."""
    n = ds_name.lower()

    if 'fineweb' in n or 'pile' in n:
        # FineWeb-Edu: {'text': '...', 'score': ...}
        return item.get('text', '')

    elif 'c4' in n:
        # C4: {'text': '...', 'timestamp': ..., 'url': ...}
        return item.get('text', '')

    elif 'books' in n or 'bookcorpus' in n:
        # BookCorpus: {'text': '...'}
        return item.get('text', '')

    elif 'wikipedia' in n:
        # wikimedia/wikipedia: {'title': '...', 'text': '...'}
        title = item.get('title', '')
        text  = item.get('text', '')
        return f"{title}\n\n{text}" if title else text

    elif 'openhermes' in n or 'hermes' in n:
        # OpenHermes-2.5: format as readable chat text, stripping system prompts
        # System prompts are training artifacts — skip them entirely
        convs = item.get('conversations', [])
        turns = []
        for t in convs:
            role  = (t.get('from') or '').lower()
            value = (t.get('value') or '').strip()
            if not value:
                continue
            if role == 'system':
                continue  # ← Key fix: system prompt was causing "world class trivia AI" memorization
            elif role in ('human', 'user'):
                turns.append(f"User: {value}")
            elif role in ('gpt', 'assistant'):
                turns.append(f"Assistant: {value}")
        return '\n'.join(turns)

    elif 'gsm8k' in n:
        # GSM8K: {'question': '...', 'answer': '...'} — answer has step-by-step
        q = item.get('question', '').strip()
        a = item.get('answer',   '').strip()
        return f"Problem: {q}\nSolution: {a}"

    elif 'metamath' in n:
        # MetaMathQA: {'query': '...', 'response': '...'} — augmented math with CoT
        q = item.get('query',    item.get('question', '')).strip()
        a = item.get('response', item.get('answer',   '')).strip()
        return f"Problem: {q}\nSolution: {a}"

    elif 'code' in n or 'alpaca' in n:
        # CodeAlpaca: {'instruction': '...', 'input': '...', 'output': '...'}
        inst = item.get('instruction', '').strip()
        inp  = item.get('input',       '').strip()
        out  = item.get('output',      '').strip()
        body = f"{inst}\n{inp}".strip() if inp else inst
        return f"Task: {body}\nSolution:\n{out}"

    elif 'cnn' in n or 'dailymail' in n:
        # CNN/DailyMail: {'article': '...', 'highlights': '...'}
        art  = item.get('article',    '').strip()
        summ = item.get('highlights', '').strip()
        return f"{art}\n\nSummary: {summ}"

    elif 'arc' in n:
        # AI2-ARC: {'question': '...', 'choices': {...}, 'answerKey': '...'}
        q = item.get('question', '').strip()
        choices = item.get('choices', {})
        opts = ''
        if choices:
            for lbl, txt in zip(choices.get('label', []), choices.get('text', [])):
                opts += f"{lbl}: {txt}\n"
        ans = item.get('answerKey', '').strip()
        return f"Question: {q}\n{opts}Answer: {ans}"

    elif 'reasoning' in n or 'stratos' in n:
        # allenai/big-reasoning-traces format
        if 'messages' in item:
            turns = []
            for msg in item['messages']:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'user':
                    turns.append(f"User: {content}")
                elif role == 'assistant':
                    turns.append(f"Assistant: {content}")
            return '\n'.join(turns)
            
        return ''

    elif 'scienceqa' in n or 'sciq' in n:
        q = item.get('question', '').strip()
        a = item.get('correct_answer', item.get('answer', '')).strip()
        support = item.get('support', '').strip()
        return f"Question: {q}\nAnswer: {a}" + (f"\nExplanation: {support}" if support else '')

    elif 'writingprompts' in n or 'creative' in n or 'story' in n:
        prompt = item.get('prompt', item.get('title', '')).strip()
        story  = item.get('story',  item.get('text', '')).strip()
        return f"Prompt: {prompt}\n\n{story}" if story else prompt

    elif 'science' in n or 'arxiv' in n or 'scientific' in n or 'summarize_from_feedback' in n:
        # openai/summarize_from_feedback: nested under info dict
        info = item.get('info', {})
        title = info.get('title', '') if isinstance(info, dict) else ''
        post  = info.get('post',  item.get('abstract', item.get('text', '')))
        post  = post.strip() if isinstance(post, str) else ''
        return f"{title}\n\n{post}" if post else title

    elif 'ethics' in n or 'moral' in n or 'commonsense_qa' in n:
        q = item.get('question', item.get('text', '')).strip()
        a = item.get('answer',   item.get('label', '')).strip()
        return f"Ethical question: {q}\nJudgment: {a}" if a else q

    elif 'xnli' in n or 'opus' in n or 'ccaligned' in n:
        # Cross-lingual
        src = item.get('translation', {}).get('en', item.get('text', '')).strip()
        return src

    elif 'winogrande' in n or 'winogender' in n:
        sent = item.get('sentence', '').strip()
        ans  = item.get('answer', '').strip()
        return f"{sent} Answer: {ans}" if ans else sent

    elif 'scruples' in n or 'selfaware' in n or 'truthful' in n:
        q = item.get('question', item.get('text', '')).strip()
        a = item.get('best_answer', item.get('answer', '')).strip()
        return f"Q: {q}\nA: {a}" if a else q

    elif 'quora' in n or 'squad' in n or 'trivia' in n or 'naturalq' in n:
        q = item.get('question', item.get('query', '')).strip()
        if isinstance(q, dict): q = q.get('text', '')
        answers = item.get('answers', {})
        if isinstance(answers, dict):
            a = ' '.join(answers.get('text', []))[:300]
        else:
            a = str(answers)[:300]
        return f"Question: {q}\nAnswer: {a}" if a else f"Question: {q}"

    elif 'temporal' in n or 'timex' in n:
        return item.get('text', item.get('sentence', ''))

    elif 'spatial' in n or 'clevr' in n:
        q = item.get('question', '').strip()
        a = item.get('answer', '').strip()
        return f"Spatial question: {q}\nAnswer: {a}" if a else q

    # Fallback: try common text keys
    for key in ('text', 'content', 'document', 'passage', 'body'):
        if key in item and isinstance(item[key], str):
            return item[key]
    return ''


def hf_streaming_generator(datasets_mix, tokenizer, seq_len=1024, start_step=0):
    """
    Weighted random sampling across datasets with auto-restart on exhaustion.
    datasets_mix: list of (hf_dataset, weight_int, ds_name_str, domain_id_int)
    start_step:   number of batches to skip at startup (text-only, no GPU tensors)
    """
    import random
    MAX_CHUNKS_PER_DOC = 3

    sources = []
    for ds, weight, name, domain_id in datasets_mix:
        sources.append({
            'iter':      iter(ds),
            'ds':        ds,
            'weight':    weight,
            'name':      name,
            'domain_id': domain_id,
        })

    weights   = [s['weight'] for s in sources]
    exhausted = [False] * len(sources)
    buf_ids, buf_dom = [], []
    emitted = 0   # total batches yielded — used for start_step skip

    # ── Fast-forward: skip to start_step WITHOUT tokenizing or creating tensors ──
    # This avoids re-downloading and GPU-allocating 29K batches on every restart.
    if start_step > 0:
        print(f"[DATA] Fast-forwarding stream to step {start_step:,} (text-only, no GPU)...")
        skipped = 0
        while skipped < start_step:
            active = [i for i, e in enumerate(exhausted) if not e]
            if not active:
                for s in sources: s['iter'] = iter(s['ds'])
                exhausted = [False] * len(sources)
                active    = list(range(len(sources)))
            active_weights = [weights[i] for i in active]
            idx = random.choices(active, weights=active_weights, k=1)[0]
            src = sources[idx]
            try:
                item = next(src['iter'])
                text = get_text_from_item(item, src['name'])
                if not text or len(text.strip()) < 20:
                    continue
                # Estimate how many chunks this doc would have produced
                word_count  = len(text.split())
                est_tokens  = int(word_count * 1.3)
                est_chunks  = min(max(1, est_tokens // seq_len), MAX_CHUNKS_PER_DOC)
                skipped    += est_chunks
                if skipped % 5000 < est_chunks:
                    print(f"[DATA] Skip progress: {min(skipped, start_step):,}/{start_step:,}")
            except StopIteration:
                exhausted[idx] = True
            except Exception:
                src['iter'] = iter(src['ds'])
        emitted = start_step
        print(f"[DATA] Fast-forward complete — resuming from step {start_step:,}")

    while True:
        # Pick source by weight; skip exhausted
        active = [i for i, e in enumerate(exhausted) if not e]
        if not active:
            # All exhausted — restart all
            print("[DATA] All streams exhausted — restarting.")
            for s in sources:
                s['iter'] = iter(s['ds'])
            exhausted = [False] * len(sources)
            active    = list(range(len(sources)))

        active_weights = [weights[i] for i in active]
        idx = random.choices(active, weights=active_weights, k=1)[0]
        src = sources[idx]

        try:
            item    = next(src['iter'])
            text    = get_text_from_item(item, src['name'])
            if not text or len(text.strip()) < 30:
                continue   # skip empty/trivial samples

            # ── Quality filter: character-level OOD detection ─────────────────
            # Word-splitting fails on dense garbage (hex, base64, minified code)
            # that has very few spaces. Character-level ratios work on everything.
            n = len(text)
            # Signal 1: non-ASCII ratio — English prose < 1%, French/accented > 3%
            non_ascii_r = sum(1 for c in text if ord(c) > 127) / n
            if non_ascii_r > 0.03:
                continue   # >3% non-ASCII chars = non-English
            # Signal 2: dense alphanum with no spaces = base64/hex blob
            alnum_r = sum(1 for c in text if c.isalnum()) / n
            space_r = text.count(' ') / n
            if alnum_r > 0.85 and space_r < 0.05:
                continue   # dense blob, no whitespace = encoded garbage
            # Signal 3: heavy special-punct = markup/template/binary
            special = sum(1 for c in text if c in '{}[]<>|\\=;@#$%^&*`~_')
            if special / n > 0.08:
                continue
            # Signal 4: code-syntax density = minified JS/CSS/code
            # Parens+semicolons+dots in code: ~18% | in prose: <3%
            code_syn = sum(1 for c in text if c in "().;")
            if code_syn / n > 0.10:
                continue   # >10% code-syntax chars = not prose
            # ─────────────────────────────────────────────────────────────────


            tokens  = tokenizer.encode(text)
            if len(tokens) < 8:
                continue   # skip too-short token sequences

            # Cap per-document chunks: prevents long FineWeb docs from flooding
            # 10+ consecutive batches with correlated text (root cause of spike clusters)
            MAX_CHUNKS_PER_DOC = 3
            chunk_count = 0
            for i in range(0, len(tokens), seq_len):
                if chunk_count >= MAX_CHUNKS_PER_DOC:
                    break   # discard remaining chunks of this doc; pick a new doc
                chunk = tokens[i:i + seq_len]
                if len(chunk) < seq_len:
                    pad_id = getattr(tokenizer, 'pad_token_id', 1)
                    if pad_id is None: pad_id = 1
                    chunk = chunk + [pad_id] * (seq_len - len(chunk))
                buf_ids.append(chunk)
                buf_dom.append(src['domain_id'])
                chunk_count += 1
                if len(buf_ids) >= 1:
                    yield (
                        torch.tensor(buf_ids[:1],  dtype=torch.long),
                        torch.tensor(buf_ids[:1],  dtype=torch.long),
                        torch.tensor(buf_dom[:1],  dtype=torch.long),
                    )
                    buf_ids, buf_dom = buf_ids[1:], buf_dom[1:]

        except StopIteration:
            exhausted[idx] = True
            print(f"[DATA] Stream '{src['name']}' exhausted — will restart when all done.")
        except Exception as e:
            # Catch shard download errors / socket timeouts gracefully
            print(f"[DATA] Stream '{src['name']}' error ({type(e).__name__}: {e}) — restarting stream.")
            src['iter'] = iter(src['ds'])  # restart this one stream immediately



def get_dataloader_for_phase(phase, tokenizer, resume_step=0, seq_len=512):
    """
    Load only the datasets needed for this phase, with correct weights and text extraction.

    Phase 1 — Web pre-training (foundation):
        70% FineWeb-Edu (10B token educational web)  ← was missing entirely
        30% OpenHermes-2.5 (chat text, system prompts stripped)

    Phase 2 — Domain injection (math + code + facts):
        40% MetaMathQA (augmented reasoning chains — much larger/richer than GSM8K alone)
        30% Wikipedia (structured encyclopedic facts)
        20% CodeAlpaca (code reasoning)
        10% GSM8K (grade-school math word problems)

    Phase 3 — Cognitive Bloom (chat format + domain mix):
        50% OpenHermes-2.5 (chat format — arms learn conversational structure)
        25% MetaMathQA (preserve math)
        25% CodeAlpaca (preserve code)

    Phase 3j — Arm Specialization (4-domain routing):
        30% OpenHermes (conversation — arm 0 anchor)
        25% MetaMathQA (math arm)
        25% CodeAlpaca (code arm)
        20% CNN/DailyMail (summarization arm)
    """
    if not HAS_HF:
        print("ERROR: HuggingFace libraries not found.")
        sys.exit(1)

    login(token=HF_TOKEN)
    print(f"\n[DATA] Loading Phase {phase} datasets (streaming)...")

    # ── Phase 1: Full-spectrum pre-training (8 streams, all arm domains covered) ──
    if phase == '1':
        print("[DATA] Phase 1: 8-stream curriculum — all 16 arm domains seeded")
        import socket
        socket.setdefaulttimeout(30)

        # High-quality Phase 1 mix — clean, dense, coherent text only.
        # Dropped: CNN/DailyMail (noisy journalism), SQuAD (tiny/repetitive),
        #          OpenHermes (chat format — saves for Phase 3 fine-tuning).
        # Added:   C4 (massive clean CommonCrawl) for raw language volume.
        ds_fineweb  = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True, token=HF_TOKEN)
        ds_wiki     = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, token=HF_TOKEN)
        ds_c4       = load_dataset("allenai/c4", "en", split="train", streaming=True, token=HF_TOKEN)
        ds_metamath = load_dataset("meta-math/MetaMathQA", split="train", streaming=True, token=HF_TOKEN)
        ds_arc      = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True, token=HF_TOKEN)
        ds_code     = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train", streaming=True, token=HF_TOKEN)

        mix = [
            (ds_fineweb,  55, 'fineweb-edu', 0),   # Arm 0: General Language  (+10 — best quality)
            (ds_c4,       20, 'c4',          4),   # Arm 4: Factual Recall     (new — massive volume)
            (ds_wiki,     12, 'wikipedia',   4),   # Arm 4: Factual Recall     (dense factual text)
            (ds_metamath,  7, 'metamath',    1),   # Arm 1: Symbolic Math
            (ds_arc,       4, 'arc_reason',  2),   # Arm 2: Logical Reasoning
            (ds_code,      2, 'codealpaca',  3),   # Arm 3: Code Syntax
        ]

    # ── Phase 2: Domain Injection ────────────────────────────────────────────
    elif phase == '2':
        print("[DATA] Phase 2: Domain Injection — factual + math + code (no chat format yet)")
        # Hermes (chat) saved for Phase 3. Phase 2 injects structured domain knowledge.
        ds_wiki     = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, token=HF_TOKEN)
        ds_metamath = load_dataset("meta-math/MetaMathQA", split="train", streaming=True, token=HF_TOKEN)
        ds_code     = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train", streaming=True, token=HF_TOKEN)
        ds_arc      = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True, token=HF_TOKEN)
        ds_fineweb  = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True, token=HF_TOKEN)

        mix = [
            (ds_wiki,     35, 'wikipedia',   4),   # Dense factual encyclopedic text
            (ds_metamath, 30, 'metamath',    1),   # Math reasoning chains
            (ds_code,     20, 'codealpaca',  3),   # Code structure and logic
            (ds_arc,      10, 'arc_reason',  2),   # Q&A reasoning
            (ds_fineweb,   5, 'fineweb-edu', 0),   # Anchor — keeps language fluency
        ]

    # ── Phase 3 & 3j: Cognitive Bloom (chat + domain mix) ──────────────────────
    elif phase in ('3', '3j'):

        print(f"[DATA] Phase {phase}: Cognitive Bloom — chat + math + code")
        ds_hermes   = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True, token=HF_TOKEN)
        ds_metamath = load_dataset("meta-math/MetaMathQA", split="train", streaming=True, token=HF_TOKEN)
        ds_code     = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train", streaming=True, token=HF_TOKEN)
        ds_arc      = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True, token=HF_TOKEN)
        ds_wiki     = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, token=HF_TOKEN)
        ds_cnn      = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True, token=HF_TOKEN)
        ds_stories  = load_dataset("roneneldan/TinyStories",      split="train", streaming=True, token=HF_TOKEN)
        ds_webtext  = load_dataset("Skylion007/openwebtext",       split="train", streaming=True, token=HF_TOKEN)

        # EQUAL weights: 12.5% per arm so every arm gets the same number of
        # gradient updates. Previous unequal mix (Chat 25%, Facts 7%) caused
        # some arms to over-specialise and others to barely train, producing
        # one dominant arm (Arm5) instead of 8 genuine specialists.
        mix = [
            (ds_stories,  1, 'tinystories', 0),   # Creative/Narrative → Arm 0
            (ds_metamath, 1, 'metamath',    1),   # Math               → Arm 1
            (ds_arc,      1, 'arc_reason',  2),   # Reasoning/Logic    → Arm 2
            (ds_code,     1, 'codealpaca',  3),   # Code               → Arm 3
            (ds_wiki,     1, 'wikipedia',   4),   # Factual            → Arm 4
            (ds_cnn,      1, 'cnn_daily',   5),   # Summarization      → Arm 5
            (ds_hermes,   1, 'openhermes',  6),   # Chat/Instruction   → Arm 6
            (ds_webtext,  1, 'openwebtext', 7),   # General Web        → Arm 7
        ]



    # ── Phase 3r: Router Training via LM loss ─────────────────────────────────
    elif phase == '3r':
        # Same domain mix as 3j — domain variety helps router see all arms.
        # domain_ids still passed so we can monitor routing vs labels,
        # but the router is NOT trained via CE loss — only LM loss drives it.
        print(f"[DATA] Phase 3r: Router training — same mix, LM-loss router")
        ds_hermes   = load_dataset("teknium/OpenHermes-2.5",          split="train", streaming=True, token=HF_TOKEN)
        ds_metamath = load_dataset("meta-math/MetaMathQA",            split="train", streaming=True, token=HF_TOKEN)
        ds_code     = load_dataset("HuggingFaceH4/CodeAlpaca_20K",    split="train", streaming=True, token=HF_TOKEN)
        ds_arc      = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True, token=HF_TOKEN)
        ds_wiki     = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, token=HF_TOKEN)
        ds_cnn      = load_dataset("abisee/cnn_dailymail", "3.0.0",   split="train", streaming=True, token=HF_TOKEN)
        ds_stories  = load_dataset("roneneldan/TinyStories",          split="train", streaming=True, token=HF_TOKEN)
        ds_webtext  = load_dataset("Skylion007/openwebtext",          split="train", streaming=True, token=HF_TOKEN)
        mix = [
            (ds_stories,  10, 'tinystories', 0),
            (ds_metamath, 20, 'metamath',    1),
            (ds_arc,       8, 'arc_reason',  2),
            (ds_code,     15, 'codealpaca',  3),
            (ds_wiki,      7, 'wikipedia',   4),
            (ds_cnn,       7, 'cnn_daily',   5),
            (ds_hermes,   25, 'openhermes',  6),
            (ds_webtext,   8, 'openwebtext', 7),
        ]

    # ── Phase SFT: Reasoning Traces ───────────────────────────────────────────
    elif phase == 'sft':
        print(f"[DATA] Phase SFT: Supervised fine-tuning on reasoning traces")
        # Using a distilled reasoning dataset with explicit CoT traces
        ds_reason = load_dataset("allenai/big-reasoning-traces", "DeepSeek", split="train", streaming=True, token=HF_TOKEN)
        # Weight doesn't matter much with 1 dataset, but we keep the format
        mix = [
            (ds_reason, 1, 'reasoning', 0),
        ]

    else:
        print(f"ERROR: Unknown phase '{phase}'")
        sys.exit(1)

    print(f"[DATA] Mix: {[(w, n) for _, w, n, _ in mix]}")
    return hf_streaming_generator(mix, tokenizer, seq_len=seq_len, start_step=resume_step)



def get_previous_phase(phase):
    return {'1': None, '2': '1', '3': '2', '3j': '2', '3r': '3j', 'sft': '3r'}.get(phase)


# ───────────────────────────────────────────────────────────────────────────
# WORD SALAD — inline generation sanity check every 250 steps
# Uses temperature + top-p + n-gram repetition penalty to get genuine output.
# Picky prompts with known answers so we can tell if the model is learning.
# ───────────────────────────────────────────────────────────────────────────

# Plain-text prompts — Phase 1 is raw web/book pretraining (GPT-NeoX tokenizer).
# Chat-format prompts (im_start etc.) won't be understood until Phase 3 fine-tuning.
WORD_SALAD_PROMPTS = [
    "User: Write a Python function to compute the Fibonacci sequence.\nAssistant: <think>\n",
    "User: If John has 5 apples and eats 2, then buys 3 more, how many does he have?\nAssistant: <think>\n",
    "User: Explain the difference between a list and a tuple in Python.\nAssistant: <think>\n",
    "User: Who was the 16th president of the United States and what is he known for?\nAssistant: <think>\n",
]
WORD_SALAD_TOKENS     = 120    # longer — chat answers need more tokens to form
SALAD_TEMPERATURE     = 0.75   # slightly lower — more focused sampling
SALAD_TOP_P           = 0.90
SALAD_REP_PENALTY     = 1.10   # softened from 1.35 — model not diverse enough yet
                                # 1.35 was pushing it off probable tokens → random output
SALAD_NGRAM_BLOCK     = 3      # 3-gram block (was 4) — still kills loops, less aggressive


def _ngram_block_mask(generated_ids: list, ngram: int, vocab_size: int, device) -> torch.Tensor:
    """Returns a logit mask (-inf on banned tokens) to prevent n-gram repeats."""
    mask = torch.zeros(vocab_size, device=device)
    n = len(generated_ids)
    if n < ngram:
        return mask
    prefix = tuple(generated_ids[-(ngram - 1):])
    for i in range(n - ngram + 1):
        if tuple(generated_ids[i:i + ngram - 1]) == prefix:
            banned = generated_ids[i + ngram - 1]
            mask[banned] = float('-inf')
    return mask


def _sample_next(logits_1d: torch.Tensor, generated_ids: list,
                 temperature: float, top_p: float,
                 rep_penalty: float, ngram: int) -> int:
    """Rep penalty → temperature → n-gram block → top-p nucleus → sample."""
    vocab = logits_1d.shape[0]
    logits = logits_1d.float().clone()

    # Repetition penalty
    for tok in set(generated_ids):
        if logits[tok] > 0:
            logits[tok] /= rep_penalty
        else:
            logits[tok] *= rep_penalty

    logits /= max(temperature, 1e-8)
    logits += _ngram_block_mask(generated_ids, ngram, vocab, logits.device)

    # Top-p nucleus
    sorted_l, sorted_i = torch.sort(logits, descending=True)
    cum = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
    remove = (cum - torch.softmax(sorted_l, dim=-1)) > top_p
    sorted_l[remove] = float('-inf')
    probs = torch.softmax(sorted_l, dim=-1)
    if torch.isnan(probs).any() or probs.sum() <= 0:
        probs = torch.ones_like(probs) / vocab
    return int(sorted_i[torch.multinomial(probs, 1)].item())


@torch.no_grad()
def run_word_salad(model, tokenizer, device, step, phase,
                   save_dir, optimizer, salad_path):
    """
    GPU-resident generation with torch.no_grad().
    mamba_ssm's selective_scan_cuda is GPU-only — CPU offload breaks it.
    We have ~5GB headroom so generation on GPU is safe.
    """
    print(f"\n[SALAD] Step {step}: eval generation...")

    samples   = []
    gen_start = time.time()
    CONTEXT_WINDOW = 32   # short context — keeps peak activation tiny
    MAX_TOKENS     = 60   # enough to judge output quality

    model.eval()
    torch.cuda.empty_cache()
    print(f"[SALAD] VRAM before gen: {torch.cuda.memory_allocated()/1e9:.2f}GB free: "
          f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.2f}GB")

    try:
        for prompt in WORD_SALAD_PROMPTS:
            try:
                prompt_ids = tokenizer.encode(prompt)
                generated  = list(prompt_ids)

                for _ in range(MAX_TOKENS):
                    id_tensor = torch.tensor([generated], dtype=torch.long, device=device)
                    # Clamp IDs to model vocab — safety net for any tokenizer mismatch
                    id_tensor = id_tensor.clamp(0, model.vocab_size - 1)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, _ = model(id_tensor, loop_idx=0)
                    next_id = _sample_next(
                        logits[0, -1, :].float(), generated,
                        SALAD_TEMPERATURE, SALAD_TOP_P,
                        SALAD_REP_PENALTY, SALAD_NGRAM_BLOCK,
                    )
                    del logits, id_tensor
                    generated.append(next_id)
                    if next_id == tokenizer.eos_token_id:
                        break

                output   = tokenizer.decode(generated[len(prompt_ids):], skip_special_tokens=True).strip()
                words    = output.split()
                rep_rate = round(1.0 - len(set(words)) / max(len(words), 1), 3)

            except Exception as e:
                output, rep_rate = f"[error: {e}]", 1.0
                traceback.print_exc()

            quality = "✅" if rep_rate < 0.30 else ("🟡" if rep_rate < 0.60 else "🔴")
            print(f"  {quality} rep={rep_rate:.0%} | {prompt[:50]}")
            print(f"     → {output[:120]}")
            samples.append({"prompt": prompt, "output": output, "rep_rate": rep_rate})

    except Exception as outer_e:
        print(f"[SALAD] Outer error: {outer_e}")
        traceback.print_exc()
        samples = [{"prompt": p, "output": "[salad skipped]", "rep_rate": 1.0} for p in WORD_SALAD_PROMPTS]
    finally:
        torch.cuda.empty_cache()
        model.train()
        print(f"[SALAD] VRAM after gen: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    elapsed = time.time() - gen_start
    avg_rep = round(sum(s['rep_rate'] for s in samples) / len(samples), 3) if samples else 1.0
    quality = "good" if avg_rep < 0.30 else ("fair" if avg_rep < 0.60 else "poor")

    with open(salad_path, "w") as f:
        json.dump({
            "step": step, "phase": phase,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(elapsed, 2),
            "avg_rep": avg_rep, "quality": quality,
            "samples": samples,
        }, f, indent=2)

    print(f"[SALAD] Done in {elapsed:.1f}s | avg_rep={avg_rep:.0%} | {quality}\n")





# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS — always saves model + optimizer + step + LR state
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(save_dir, phase, step, model, optimizer, reason="periodic"):
    ckpt_path = os.path.join(save_dir, f"phase_{phase}.pt")
    tmp_path  = ckpt_path + ".tmp"
    print(f"\n[CKPT] Saving checkpoint at step {step} ({reason})...")
    state = {
        'model': model.state_dict(),
        'step':  step,
        'phase': phase,
    }
    torch.save(state, tmp_path)
    # Save optimizer state separately in CPU pinned format (PagedAdam8bit)
    # This keeps the main checkpoint small and avoids VRAM OOM on restore.
    opt_path = ckpt_path.replace('.pt', '_optim.pt')
    try:
        opt_state = {k: {sk: sv.cpu() if hasattr(sv, 'cpu') else sv
                         for sk, sv in v.items()} if isinstance(v, dict) else v
                     for k, v in optimizer.state_dict().items()}
        torch.save(opt_state, opt_path + '.tmp')
        os.replace(opt_path + '.tmp', opt_path)
    except Exception as _oe:
        pass  # optimizer save failure never blocks training
    os.replace(tmp_path, ckpt_path)   # atomic swap — no corrupt file on kill
    
    # ── Archival Copy ────────────────────────────────────────────────────────
    # The user has 3TB storage; save persistent step checkpoints to prevent overwriting
    if reason == "periodic":
        import shutil
        archival_ckpt = ckpt_path.replace('.pt', f'_step_{step}.pt')
        shutil.copy2(ckpt_path, archival_ckpt)
        if os.path.exists(opt_path):
            archival_opt = opt_path.replace('_optim.pt', f'_step_{step}_optim.pt')
            shutil.copy2(opt_path, archival_opt)
    
    print(f"[CKPT] Saved → {ckpt_path}")
    return ckpt_path


def load_checkpoint(ckpt_path, model, optimizer, device):
    """Load model + optimizer state. Returns resume_step (0 if not found)."""
    if not os.path.exists(ckpt_path):
        return 0
    print(f"[CKPT] Loading checkpoint: {ckpt_path}")
    # Load entire checkpoint to CPU first — avoids VRAM OOM on optimizer restore.
    # PagedAdam8bit keeps moments in CPU RAM anyway, so this is the natural path.
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    
    ckpt_model = ckpt['model']
    current_model_dict = model.state_dict()
    filtered_ckpt = {}
    for k, v in ckpt_model.items():
        if k in current_model_dict:
            if v.shape == current_model_dict[k].shape:
                filtered_ckpt[k] = v
            else:
                print(f"[CKPT] Skipping {k} due to shape mismatch: ckpt={v.shape}, model={current_model_dict[k].shape}")
        else:
            filtered_ckpt[k] = v
            
    model.load_state_dict(filtered_ckpt, strict=False)
    opt_path = ckpt_path.replace('.pt', '_optim.pt')
    if os.path.exists(opt_path):
        try:
            opt_state = torch.load(opt_path, map_location='cpu', weights_only=False)
            optimizer.load_state_dict(opt_state)
            print("[CKPT] Optimizer state restored ✅ (from CPU sidecar file)")
        except Exception as e:
            print(f"[CKPT] Optimizer sidecar load failed ({type(e).__name__}) — fresh optimizer")
    else:
        print("[CKPT] No optimizer sidecar — fresh PagedAdam8bit (300-step re-ramp active)")
    resume_step = int(ckpt.get('step', 0))
    print(f"[CKPT] Resuming from step {resume_step}")
    return resume_step


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--phase', type=str, required=True, choices=['1', '2', '3', '3j', '3r', 'sft'])
    parser.add_argument('--ckpt',  type=str, default=None,
                        help='Path to a specific checkpoint to load (overrides phase-chain cold-start)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints (default: titan_checkpoints/)')
    try:
        import deepspeed
        parser = deepspeed.add_config_arguments(parser)
    except ImportError:
        pass
    cmd_args = parser.parse_args()
    phase    = cmd_args.phase
    ckpt_override = cmd_args.ckpt  # e.g. titan_checkpoints/mamba14b_transfer.pt

    # ── Paths ───────────────────────────────────────────────────────────────────────────
    project_dir  = os.path.dirname(os.path.abspath(__file__))
    save_dir     = cmd_args.save_dir if cmd_args.save_dir else os.path.join(project_dir, "checkpoints_2.7b")
    telem_path   = os.path.join(project_dir, "monitor_ui", "telemetry.json")
    salad_path   = os.path.join(project_dir, "monitor_ui", "word_salad.json")
    log_path     = os.path.join(project_dir, "training_log.txt")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[PATHS] Checkpoint dir: {save_dir}")

    # ── Continuance log banner ───────────────────────────────────────────────
    # Always append; the bash launcher wrote the === RESTART === line already.
    # We write a Python-level header here too for clarity.
    with open(log_path, "a") as lf:
        lf.write(f"\n[TRAINER] Process start  UTC={time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}  phase={phase}\n")

    # ── Device / Model ───────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if HAS_HF:
        print("Loading Tokenizer (EleutherAI/gpt-neox-20b)...")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = 0
            
        if phase in ('3r', 'sft'):
            num_added = tokenizer.add_special_tokens({'additional_special_tokens': ['<think>', '</think>']})
            print(f"Added {num_added} special tokens to tokenizer.")
    else:
        tokenizer = None

    model = Mamba3Titan(vocab_size=50288, d_model=2560, n_layers=64,
                        mimo_paths=8, use_gradient_checkpointing=True)
                        
    if phase in ('3r', 'sft') and tokenizer is not None:
        if len(tokenizer) > model.embedding.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
        else:
            print(f"Tokenizer length ({len(tokenizer)}) fits within model vocab ({model.embedding.weight.shape[0]}). No resize needed.")

    # ~965M params (1B-class). Reduced from 80 layers to fit RTX 3060 12GB.
    # Sequential scan autograd stores h tensors across ALL layers — 40% fewer
    # layers = 40% less activation memory, clearing the scan-loop OOM.
    # NOTE: initialize_asymmetric_arms() removed for Blackboard architecture.
    # Phase 1 uses uniform 1/16 averaging — orthogonal arms cancel each other's
    # gradients in the average, causing a plateau. Phase 2 clones them anyway,
    # so orthogonal init has zero benefit. Standard PyTorch init is correct here.
    model.set_phase(phase)
    model = model.to(torch.bfloat16).to(device)

    # ── HARD GUARD: never silently train with a placeholder SSM ──────────────
    from mamba3_titan_builder import DummyMambaSSM
    for name, mod in model.named_modules():
        if isinstance(mod, DummyMambaSSM):
            print(f"\n[FATAL] DummyMambaSSM detected at '{name}'.")
            print("  The model has NO sequence modeling — training will not converge.")
            print("  Fix: MinimalMambaSSM should be used. Check mamba3_titan_builder.py")
            sys.exit(1)
    print("[OK] SSM check passed — real Mamba S6 selective scan confirmed.")
    print(f"Model loaded on {device}. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── Phase-conditional LR ─────────────────────────────────────────────────
    if phase == '1':
        BASE_LR_CORE  = 3e-5
        BASE_LR_HEAD  = 6e-5
    elif phase == '2':
        BASE_LR_CORE  = 5e-5   # Higher than before: domain shift needs real LR, not micro-LR
        BASE_LR_HEAD  = 1e-4   # Head adapts to new domain structure faster
    elif phase in ('3', '3j'):
        BASE_LR_CORE  = 5e-6
        BASE_LR_HEAD  = 1e-5
    elif phase == '3r':
        # Router needs to learn fast (fresh init learning from LM loss).
        # Arms need a low LR — they have 3j specialisation to preserve.
        # Using separate param groups below.
        BASE_LR_CORE   = 5e-7   # arms: near-frozen — protect 3j knowledge
        BASE_LR_HEAD   = 1e-5   # lm_head: moderate adaptation
        BASE_LR_ROUTER = 5e-5   # router: has domain signal, learn fast
    elif phase == 'sft':
        # SFT teaches the model how to use the <think> token.
        # The new embeddings must learn fast. Arms learn moderately to develop latent algorithms.
        BASE_LR_CORE   = 1e-5   # arms: learn latent algorithms
        BASE_LR_HEAD   = 1e-4   # head & embeddings: fast adaptation to new <think> tokens
        BASE_LR_ROUTER = 5e-5   # router: needs to learn when to route to think arm
    else:
        BASE_LR_CORE  = 1e-5
        BASE_LR_HEAD  = 2e-5

    # ── Optimizer — BEFORE torch.compile ─────────────────────────────────────
    head_params_set   = set(id(p) for p in model.lm_head.parameters())
    router_params_set = set(id(p) for p in model.domain_router.parameters())
    router_params_set.add(id(model.router_temp))
    emb_params_set    = set(id(p) for p in model.embedding.parameters())

    head_params_list   = [p for p in model.lm_head.parameters()  if p.requires_grad]
    router_params_list = [p for p in model.parameters()
                          if id(p) in router_params_set and p.requires_grad]
    emb_params_list    = [p for p in model.parameters()
                          if id(p) in emb_params_set and p.requires_grad]
    core_params_list   = [p for p in model.parameters()
                          if id(p) not in head_params_set
                          and id(p) not in router_params_set
                          and id(p) not in emb_params_set
                          and p.requires_grad]

    trainable_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Trainable parameters: {trainable_M:.0f}M")

    try:
        import bitsandbytes as bnb
        if phase == '3r':
            # Three groups: router (high LR, fresh init) | arms (low LR, preserve 3j)
            optimizer = bnb.optim.Adam8bit([
                {'params': router_params_list, 'lr': BASE_LR_ROUTER, 'name': 'router'},  # learns fast
                {'params': head_params_list,   'lr': BASE_LR_HEAD, 'name': 'head'},    # moderate
                {'params': emb_params_list,    'lr': BASE_LR_CORE, 'name': 'embedding'}, # slow
                {'params': core_params_list,   'lr': BASE_LR_CORE, 'name': 'core'},    # arms: slow + careful
            ], weight_decay=0.01)
            print(f"3r co-training: router LR={BASE_LR_ROUTER:.0e} | "
                  f"arms LR={BASE_LR_CORE:.0e} | head LR={BASE_LR_HEAD:.0e}")
        elif phase == 'sft':
            # SFT groups: emb/head learn fast for new tokens. Router learns moderate. Arms slow.
            optimizer = bnb.optim.Adam8bit([
                {'params': emb_params_list,    'lr': 1e-4, 'name': 'embedding'}, # Fast for <think>
                {'params': router_params_list, 'lr': BASE_LR_ROUTER, 'name': 'router'},
                {'params': head_params_list,   'lr': BASE_LR_HEAD, 'name': 'head'},
                {'params': core_params_list,   'lr': BASE_LR_CORE, 'name': 'core'},
            ], weight_decay=0.01)
            print(f"SFT: Emb LR=1e-4 | Router LR={BASE_LR_ROUTER:.0e} | Arms LR={BASE_LR_CORE:.0e}")
        elif phase == '3j' and trainable_M < 800:
            # Backbone frozen → ~350M trainable → optimizer moments ~1GB in 8-bit.
            optimizer = bnb.optim.Adam8bit([
                {'params': core_params_list, 'lr': BASE_LR_CORE, 'name': 'core'},
                {'params': head_params_list, 'lr': BASE_LR_HEAD, 'name': 'head'},
            ], weight_decay=0.01)
            print(f"Using Adam8bit GPU-only (backbone frozen, {trainable_M:.0f}M trainable params).")
        else:
            optimizer = bnb.optim.PagedAdam8bit([
                {'params': core_params_list, 'lr': BASE_LR_CORE, 'name': 'core'},
                {'params': head_params_list, 'lr': BASE_LR_HEAD, 'name': 'head'},
            ], weight_decay=0.01)
            print(f"Using PagedAdam8bit — moments in CPU RAM ({trainable_M:.0f}M trainable params).")
    except ImportError:
        print("WARNING: bitsandbytes not found — falling back to AdamW.")
        if phase in ('3r', 'sft'):
            optimizer = torch.optim.AdamW([
                {'params': router_params_list, 'lr': BASE_LR_ROUTER, 'name': 'router'},
                {'params': head_params_list,   'lr': BASE_LR_HEAD, 'name': 'head'},
                {'params': core_params_list,   'lr': BASE_LR_CORE, 'name': 'core'},
            ], weight_decay=0.01)
        else:
            optimizer = torch.optim.AdamW([
                {'params': core_params_list, 'lr': BASE_LR_CORE, 'name': 'core'},
                {'params': head_params_list, 'lr': BASE_LR_HEAD, 'name': 'head'},
            ], weight_decay=0.01)

    # torch.compile disabled: builder has inplace ops (cp_gate telemetry, buffer updates)
    # that trigger "tensor modified by inplace op" in compiled autograd graph.
    # Prefetch + non_blocking H2D are the larger wins and remain active.

    pad_id = getattr(tokenizer, 'pad_token_id', 1)
    if pad_id is None: pad_id = 1
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # ── Checkpoint: load phase checkpoint (if exists) for resume ─────────────
    ckpt_path   = os.path.join(save_dir, f"phase_{phase}.pt")
    resume_step = 0

    # --ckpt override: load a specific checkpoint directly (e.g. mamba-1.4b transfer)
    if ckpt_override:
        ckpt_abs = ckpt_override if os.path.isabs(ckpt_override) else os.path.join(save_dir, ckpt_override)
        print(f"[CKPT] --ckpt override: loading {ckpt_abs}")
        _ckpt = torch.load(ckpt_abs, map_location='cpu', weights_only=False)
        model.load_state_dict(_ckpt['model'], strict=False)
        resume_step = int(_ckpt.get('step', 0))
        del _ckpt; import gc; gc.collect()
        with torch.no_grad():
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())
        print(f"[CKPT] Loaded override checkpoint. Resuming from step {resume_step}")
    # If this phase has a checkpoint, load it (model + optimizer)
    elif os.path.exists(ckpt_path):
        resume_step = load_checkpoint(ckpt_path, model, optimizer, device)
    else:
        # Cold start: load previous phase model weights only (no optimizer)
        prev_phase = get_previous_phase(phase)
        if prev_phase:
            prev_ckpt = os.path.join(save_dir, f"phase_{prev_phase}.pt")
            if os.path.exists(prev_ckpt):
                print(f"Loading Phase {prev_phase} weights for cold-start of Phase {phase}...")
                # Load to CPU first — map_location=device would put the full 6GB ckpt
                # (model + optimizer) onto CUDA, exhausting VRAM before training starts
                _ckpt = torch.load(prev_ckpt, map_location='cpu', weights_only=True)
                _model_state = _ckpt['model']
                del _ckpt  # immediately free optimizer tensors from CPU RAM
                model.load_state_dict(_model_state, strict=False)
                del _model_state
                import gc; gc.collect()
                
                # Phase 3r: Retain optimizer state for Embedding/Head but flush Router
                if phase == '3r':
                    opt_path = prev_ckpt.replace('.pt', '_optim.pt')
                    if os.path.exists(opt_path):
                        print(f"[CKPT] Phase 3r Cold Start: Loading previous optimizer state to retain Emb/Head momentum...")
                        opt_state = torch.load(opt_path, map_location='cpu', weights_only=False)
                        
                        print("[CKPT] Temporarily setting phase 3j to recreate optimizer topology...")
                        model.set_phase('3j')
                        temp_head_params = [p for p in model.lm_head.parameters() if p.requires_grad]
                        temp_head_set = set(id(p) for p in temp_head_params)
                        temp_router_params = [p for p in model.domain_router.parameters() if p.requires_grad]
                        temp_router_set = set(id(p) for p in temp_router_params)
                        if hasattr(model, 'router_temp'):
                            temp_router_set.add(id(model.router_temp))
                        
                        temp_core_params = [p for p in model.parameters() if id(p) not in temp_head_set and id(p) not in temp_router_set and p.requires_grad]
                        
                        import bitsandbytes as bnb
                        temp_opt = bnb.optim.Adam8bit([
                            {'params': temp_core_params},
                            {'params': temp_head_params},
                        ])
                        temp_opt.load_state_dict(opt_state)
                        model.set_phase('3r') # Restore Phase 3r
                        
                        print("[CKPT] Migrating states and flushing AdamW buffers for Router parameters...")
                        # Transfer states to new optimizer
                        for p, state in temp_opt.state.items():
                            optimizer.state[p] = state
                            
                        # Router parameters are already empty in the new optimizer, but let's be explicit
                        router_ids = [id(p) for p in model.domain_router.parameters()]
                        for group in optimizer.param_groups:
                            if group.get('name') == 'router':
                                for p in group['params']:
                                    if p in optimizer.state:
                                        optimizer.state[p] = {}
                        del temp_opt
                        del opt_state
                        import gc; gc.collect()
                elif phase == 'sft':
                    print(f"[CKPT] Phase SFT Cold Start: Discarding previous optimizer state. Starting with fresh momentum buffer.")
                    # Do not load opt_state, let optimizer stay fresh
                
                # ── CRITICAL: Break the weight tie that Phase 1 saved ──────────────
                # Phase 1 checkpoint has lm_head.weight == embedding.weight (tied).
                # load_state_dict restores this tie. We must explicitly clone the
                # lm_head tensor to give it its own independent memory so that
                # gradients to lm_head cannot cascade into and corrupt embeddings.
                with torch.no_grad():
                    model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())
                print("Phase weights loaded. Weight tie broken — lm_head is now independent.")
            else:
                print(f"ERROR: Phase {prev_phase} checkpoint not found at {prev_ckpt}.")
                sys.exit(1)

    if phase == '2':
        print(f"[CLONE] Phase 2 active: Enforcing {model.mimo_paths} identical clones (copying Arm 0 to 1-{model.mimo_paths-1})...")
        with torch.no_grad():
            base_arm = model.mimo_reasoning_blocks[0]
            for i in range(1, model.mimo_paths):
                model.mimo_reasoning_blocks[i].load_state_dict(base_arm.state_dict())
        print("[CLONE] Arms perfectly synchronized.")
        # LM head Phase 1 weights are preserved — vocab structure is useful.
        # The higher BASE_LR_HEAD (3e-5) will rapidly adapt it to the new feature space.

    RESUME_WARMUP_STEPS = 300  # full warmup for new phase

    # ── Training config (MUST be before dataloader so seq_len is defined) ────
    if phase == '1':
        target_steps = 250_000  # ~256M tokens — gives model real time to converge
        seq_len      = 512
        WARMUP_STEPS = 2000    # Extended warmup for long run
    elif phase == '2':
        target_steps = 250_000   # Extended domain injection (user request)
        seq_len      = 512
        WARMUP_STEPS = 500      # slightly longer warmup — fresh optimizer on domain shift
    elif phase in ('3', '3j'):
        target_steps = 200_000  # 200K: each arm gets ~25K real updates (was 75K = ~1.7K)
        seq_len      = 768
        WARMUP_STEPS = 1000  # longer warmup: arms start random, need stable LM loss first
    elif phase == '3r':
        target_steps = 30_000   # router is tiny (~4M params), converges fast
        seq_len      = 768      # same context length as 3j
        WARMUP_STEPS = 200      # short warmup — router hasn't seen LM-loss gradient before
    elif phase == 'sft':
        target_steps = 20_000   # Reasoning SFT phase
        seq_len      = 768      # Maintain same context length
        WARMUP_STEPS = 500      # Longer warmup for new <think> embeddings
    else:
        target_steps = 30_000
        seq_len      = 512
        WARMUP_STEPS = 300

    SAVE_EVERY    = 500
    DATA_TIMEOUT  = 300
    # GRAD_ACCUM=1 — GPU has no headroom. Model (6.9GB) + Adam state (3.5GB) +
    # one forward pass activations (3.5GB) = ~14GB, already fragmented to fit in 12GB.
    # Any accumulation OOMs. Data mix improvement below still applies.
    GRAD_ACCUM    = 1


    # ── Data pipeline ────────────────────────────────────────────────────────
    dataloader_generator = get_dataloader_for_phase(
        phase, tokenizer, resume_step=resume_step, seq_len=seq_len
    )

    # OPT 4: Prefetch thread — keeps a 2-batch lookahead queue so the GPU never
    # idles waiting for the CPU to tokenize+assemble the next batch.
    _PREFETCH_SIZE = 2
    _prefetch_q    = queue.Queue(maxsize=_PREFETCH_SIZE)
    _sentinel      = object()  # signals generator exhaustion

    def _prefetch_worker(gen, q):
        try:
            for item in gen:
                # Pin tensors to page-locked memory for faster H2D DMA (OPT 4b)
                pinned = tuple(
                    t.pin_memory() if isinstance(t, torch.Tensor) else t
                    for t in item
                )
                q.put(pinned)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_sentinel)

    _prefetch_thread = threading.Thread(
        target=_prefetch_worker,
        args=(dataloader_generator, _prefetch_q),
        daemon=True,
        name='DataPrefetch'
    )
    _prefetch_thread.start()

    def _iter_prefetch():
        while True:
            item = _prefetch_q.get()
            if item is _sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    dataloader_generator = _iter_prefetch()


    # ── LR cycle restart on new-data resume ──────────────────────────────────
    # If we're resuming mid-phase on a different dataset (FineWeb-Edu injection),
    # the cosine schedule at step 25K+ gives near-zero LR. Instead, treat LR as
    # a fresh cosine cycle anchored to the resume_step so the model gets a
    # proper learning rate for the new data distribution.
    # Effective LR reference step = 0 for the purposes of cosine scheduling,
    # but we shift all step indices by resume_step when computing LR.
    # LR schedule: track actual step, no offset reset on restart.
    # The resume_warmup (300 steps) already provides a gentle re-ramp after
    # each restart. Cosine at step 6K-10K gives 4-5e-6 — no need to reset.
    lr_step_offset = 0
    lr_cycle_steps = target_steps   # cosine spans full phase for Phase 2/3
    use_sgdr = (phase == '1')       # SGDR only for Phase 1; simple cosine for Phase 2/3


    auto_stop = AutoStop()
    model.train()

    header = (
        f"\n{'='*72}\n"
        f"  MAMBA3 TITAN 2.5B  |  Phase {phase}  |  Resume step {resume_step}\n"
        f"  UTC: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n"
        f"  Target: {target_steps:,} steps  |  Remaining: {target_steps - resume_step:,}\n"
        f"{'='*72}"
    )
    print(header)
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    start_time      = time.time()
    step_start_time = time.time()
    last_lr_core    = BASE_LR_CORE

    for step, batch in enumerate(dataloader_generator, start=resume_step):

        if step >= target_steps:
            print(f"\n[PHASE COMPLETE] Reached {target_steps:,} steps for Phase {phase}.")
            save_checkpoint(save_dir, phase, step, model, optimizer, reason="phase_complete")
            break

        # ── LR update: cosine schedule on actual step, gentle resume re-ramp ──
        lr_core = get_lr(step, lr_cycle_steps, BASE_LR_CORE,
                         warmup_steps=WARMUP_STEPS,
                         resume_step=resume_step,
                         resume_warmup=RESUME_WARMUP_STEPS,
                         use_sgdr=use_sgdr)
        lr_head = get_lr(step, lr_cycle_steps, BASE_LR_HEAD,
                         warmup_steps=WARMUP_STEPS,
                         resume_step=resume_step,
                         resume_warmup=RESUME_WARMUP_STEPS,
                         use_sgdr=use_sgdr)
                         
        if phase == '3r':
            # Phase 3r: Custom linear warmup for the router to map unsupervised LM-loss
            router_warmup = 1000
            steps_since_resume = step - resume_step
            if steps_since_resume < router_warmup:
                lr_router = BASE_LR_ROUTER * (steps_since_resume / router_warmup)
            else:
                lr_router = get_lr(step, lr_cycle_steps, BASE_LR_ROUTER,
                                   warmup_steps=WARMUP_STEPS,
                                   resume_step=resume_step,
                                   resume_warmup=RESUME_WARMUP_STEPS,
                                   use_sgdr=False)
        else:
            lr_router = lr_core

        apply_lr(optimizer, lr_core, lr_head, lr_router)
        last_lr_core = lr_core


        # ── Forward / backward with gradient accumulation ────────────────
        input_ids, labels, domain_ids = batch
        input_ids  = input_ids.to(device,  non_blocking=True)
        labels     = labels.to(device,     non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)

        # Micro-step accumulation: zero grad only at start of accumulation window
        micro_step = step % GRAD_ACCUM
        if micro_step == 0:
            optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, domain_loss = model(input_ids, loop_idx=0, domain_ids=domain_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = criterion(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
            if phase == '3r':
                # Pure LM loss — router trains via gradient from prediction quality.
                # domain_loss in 3r is only a tiny load-balance penalty (0.01 scale).
                loss = lm_loss + domain_loss
            elif phase in ('3', '3j'):
                loss = lm_loss + 0.1 * domain_loss
            else:
                loss = lm_loss
            loss    = loss / GRAD_ACCUM   # scale so accumulated gradient == true mean

        loss.backward()

        # Only step optimizer after GRAD_ACCUM micro-steps
        is_update_step = (micro_step == GRAD_ACCUM - 1)
        if is_update_step:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
        else:
            grad_norm = 0.0   # not stepping this micro-step

        if (step + 1) % 20 == 0:
            import gc; gc.collect()   # defrag every 20 steps — not every step (was -35% TPS)
            torch.cuda.empty_cache()


        # ── Metrics ─────────────────────────────────────────────────────────
        step_elapsed    = time.time() - step_start_time
        tps             = (seq_len * GRAD_ACCUM) / step_elapsed if step_elapsed > 0 else 0.0
        step_start_time = time.time()
        gpu_temp        = get_gpu_temp()

        # Stall watchdog
        if step_elapsed > DATA_TIMEOUT:
            print(f"[WATCHDOG] Step {step+1} took {step_elapsed:.1f}s > {DATA_TIMEOUT}s. Saving and aborting.")
            save_checkpoint(save_dir, phase, step + 1, model, optimizer, reason="watchdog_stall")
            sys.exit(1)

        telem     = model.last_telemetry
        dom_l     = domain_loss.item() if isinstance(domain_loss, torch.Tensor) else domain_loss
        div_l     = telem.get('diversity_loss', 0.0)
        gate      = telem.get('gate_score', 0.0)
        entropy   = telem.get('entropy', 0.0)
        temp_str  = f" | GPU: {gpu_temp}°C" if gpu_temp is not None else ""
        lr_str    = f" | LR: {lr_core:.2e}"

        grad_norm_val = float(grad_norm) if is_update_step else 0.0
        log_line = (
            f"Phase {phase} | Step {step+1:05d} | LM Loss: {lm_loss.item():.4f} | "
            f"Dom Loss: {dom_l:.4f} | Div: {div_l:.4f} | Gate: {gate:.4f} | Entropy: {entropy:.4f} | "
            f"GNorm: {grad_norm_val:.2f} | TPS: {tps:.1f}{temp_str}{lr_str}"
        )
        print(log_line)

        # Append to training log (continuance — no overwrite)
        with open(log_path, "a") as lf:
            lf.write(log_line + "\n")

        # ── Arm Divergence: weight-based cosine similarity (every 10 steps) ────
        # Uses normalized flattened in_proj weight vectors per arm.
        # Confirmed: arm weight L2 diffs exist (norm~3.2) — weight cosine reveals it.
        # Collapse score = (cosine_sim + 1) / 2  →  0=orthogonal, 1=clone
        arm_sims_live, col_mean_live, col_max_live = [], 1.0, 1.0
        if (step + 1) % 10 == 0:
            try:
                with torch.no_grad():
                    arm_vecs = []
                    for i in range(model.mimo_paths):  # FIX: was range(16), model has 8 arms
                        arm_mod = model.mimo_reasoning_blocks[i]
                        w = None
                        for name, p in arm_mod.named_parameters():
                            if 'ssm' in name and 'weight' in name and p.dim() >= 2:
                                w = p.detach().float().cpu().view(-1)
                                break
                        if w is None:
                            w = list(arm_mod.parameters())[-1].detach().float().cpu().view(-1)
                        arm_vecs.append(torch.nn.functional.normalize(w, dim=0))
                    arm_mat  = torch.stack(arm_vecs, dim=0)  # [n_arms, D] on CPU
                    sim_mat  = arm_mat @ arm_mat.T            # [n_arms, n_arms]
                    collapse_mat = (sim_mat + 1.0) / 2.0     # [0,1]: 1=clone, 0=diverse
                    n_arms   = model.mimo_paths
                    off_diag = ~torch.eye(n_arms, dtype=torch.bool)  # FIX: was eye(16)
                    per_arm  = (collapse_mat * off_diag.float()).sum(dim=1) / max(1, n_arms - 1)  # FIX: was /15
                    arm_sims_live = [round(v, 4) for v in per_arm.tolist()]
                    col_mean_live = round(per_arm.mean().item(), 4)
                    col_max_live  = round(per_arm.max().item(), 4)
            except Exception:
                pass  # never crash training over telemetry

        # ── Phase 3r: Telemetry for Top-2 Pairings and Utilization ───────────
        util_history = getattr(model, 'util_history', [])
        current_pairings = {}
        if phase == '3r':
            pairs = telem.get('top2_pairings', [])
            if pairs:
                for pair in pairs:
                    k = f"[{pair[0]}, {pair[1]}]"
                    current_pairings[k] = current_pairings.get(k, 0) + 1
            
            # Utilization is sum of weights or just frequency
            # We already have arm_weights in telem
            util_history.append(telem.get('arm_weights', [0.0]*model.mimo_paths))
            if len(util_history) > 50:
                util_history.pop(0)
            model.util_history = util_history
            
            if len(util_history) == 50:
                avg_util = [sum(col)/50.0 for col in zip(*util_history)]
                for i, u in enumerate(avg_util):
                    if u < 0.10:
                        print(f"[WARNING] Expert {i} underutilized: {u:.1%}")
                    elif u > 0.40:
                        print(f"[WARNING] Expert {i} overutilized: {u:.1%}")

        # ── Telemetry for monitor UI ─────────────────────────────────────────
        telemetry_data = {
            "phase":               phase,
            "step":                step + 1,
            "lm_loss":             round(lm_loss.item(), 4),
            "domain_loss":         round(dom_l, 4),
            "gate_score":          round(gate, 4),
            "entropy":             round(entropy, 4),
            "grad_norm":           round(grad_norm_val, 4),
            "tps":                 round(tps, 1),
            "gpu_temp":            gpu_temp,
            "lr":                  round(lr_core, 8),
            "resume_step":         resume_step,
            # Weight-based arm divergence (trainer-side, bypasses grad ckpt)
            "arm_collapse_metric": col_mean_live if arm_sims_live else round(telem.get('arm_collapse_mean', 1.0), 4),
            "arm_collapse_mean":   col_mean_live,
            "arm_collapse_max":    col_max_live,
            "arm_sims":            arm_sims_live if arm_sims_live else telem.get('arm_sims', []),
            "latent_energy":       round(telem.get('latent_energy', 0.0), 4),
            "arm_weights":         telem.get('arm_weights', []),
            "top2_pairings":       current_pairings,
        }
        with open(telem_path, "w") as f:
            json.dump(telemetry_data, f)


        # ── Auto-stop check ──────────────────────────────────────────────────
        should_stop, reason = auto_stop.update(lm_loss.item())
        if should_stop:
            print(f"\n[AUTO-STOP] {reason}")
            save_checkpoint(save_dir, phase, step + 1, model, optimizer, reason="auto_stop_divergence")
            with open(log_path, "a") as lf:
                lf.write(f"[AUTO-STOP] {reason}\n")
            sys.exit(2)  # exit code 2 = diverged; run_titan.sh skips eval

        # ── Word salad every 1000 steps ───────────────────────────────────────
        # CPU-offload generation: model moves to CPU, frees all VRAM, then returns.
        if (step + 1) % 1000 == 0 and tokenizer is not None:
            run_word_salad(model, tokenizer, device,
                           step + 1, phase, save_dir, optimizer, salad_path)
            # model.train() + model.to(device) are called inside run_word_salad on exit

        # ── Periodic checkpoint every 500 steps ───────────────────────────────
        if (step + 1) % SAVE_EVERY == 0:
            save_checkpoint(save_dir, phase, step + 1, model, optimizer, reason="periodic")

        # ── Graceful shutdown (SIGTERM / SIGINT) ─────────────────────────────
        if _shutdown_requested:
            print(f"\n[SHUTDOWN] Saving checkpoint at step {step+1} and exiting cleanly.")
            save_checkpoint(save_dir, phase, step + 1, model, optimizer, reason="graceful_shutdown")
            with open(log_path, "a") as lf:
                lf.write(f"[SHUTDOWN] Graceful save at step {step+1}  "
                         f"UTC={time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n")
            sys.exit(0)

    # End of loop
    elapsed = time.time() - start_time
    print(f"\nExecution Time: {elapsed:.2f}s")
    print("\n" + "="*72)
    print(f"  Phase {phase} COMPLETE — run auto_eval.py --phase {phase} to verify.")
    print("="*72)


if __name__ == "__main__":
    train()
