"""
sft3_sprint.py — Termination Sprint: 2,000 steps @ flat 5e-6
──────────────────────────────────────────────────────────────
Objective: push P(</think>) off zero and into firing range.

Why flat LR vs cosine:
  sft2 cosine decayed to 2.8e-7 by step 27k — gradient too small to
  move the </think> logit.  This sprint holds LR=5e-6 flat for 2,000
  steps so every masked batch delivers a consistent, strong signal on
  the termination token.

Changes from sft2:
  - FLAT LR at 5e-6 (no cosine decay at all)
  - Loads phase_sft2.pt (best reasoning-quality checkpoint)
  - ONLY masked SFT batches (no plain-text regulariser needed — model
    already proved stable English in sft2, and diluting gradient here
    would slow termination convergence)
  - Dataset mix: 50% DeepSeek ≤1200 tok + 30% TriviaQA + 20% GSM8K
    (more factual QA to reinforce short-answer close patterns)
  - Early-exit: stops as soon as P(</think>) > 0.01 on any probe
    (fires a checkpoint and exits cleanly)
  - Probe every 200 steps (was 500) to catch the crossing fast

Run:
  cd /home/phil/.gemini/antigravity/scratch/analysis_project
  nohup ./titan_venv/bin/python3 -u sft3_sprint.py > sft3.log 2>&1 &
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, time, signal, threading, queue, random, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from mamba3_titan_builder import Mamba3Titan

_HF_TOKEN = os.environ.get("HF_TOKEN", "HF_TOKEN_REDACTED")
try:
    from huggingface_hub import login as hf_login
    hf_login(token=_HF_TOKEN, add_to_git_credential=False)
    print(f"[HF] Authenticated ({_HF_TOKEN[:8]}...)")
except Exception as e:
    print(f"[HF] Warning: {e}")

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    import pynvml; pynvml.nvmlInit()
    _nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
    def gpu_temp(): return pynvml.nvmlDeviceGetTemperature(_nvml, pynvml.NVML_TEMPERATURE_GPU)
except Exception:
    def gpu_temp(): return None

# ── CONFIG ────────────────────────────────────────────────────────────────────
TARGET_STEPS     = 2_000
SEQ_LEN          = 768
SAVE_EVERY       = 500
DATA_TIMEOUT     = 300
MAX_THINK_TOKENS = 1200

# FLAT LR — no decay, no warmup
FLAT_LR_CORE   = 5e-6
FLAT_LR_HEAD   = 1e-5
FLAT_LR_ROUTER = 5e-6

# Early-exit threshold: stop as soon as any prompt hits this P(</think>)
EARLY_EXIT_P = 0.01

CKPT_DIR   = "checkpoints_2.7b"
LOAD_FROM  = f"{CKPT_DIR}/phase_sft20_done.pt"
SAVE_AS    = f"{CKPT_DIR}/phase_sft3_sprint.pt"
LOG_PATH   = "sft3.log"
TRAIN_LOG  = "training_log.txt"
TELEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "monitor_ui", "telemetry.json")

# ── SIGNAL HANDLER ────────────────────────────────────────────────────────────
_shutdown = False
def _sig(s, f):
    global _shutdown
    print(f"\n[SIGNAL] Caught — finishing step then saving...")
    _shutdown = True
signal.signal(signal.SIGTERM, _sig)
signal.signal(signal.SIGINT,  _sig)

# ── LOSS MASK ─────────────────────────────────────────────────────────────────
def make_masked_labels(input_ids: torch.Tensor, end_think_id: int) -> torch.Tensor:
    labels = input_ids.clone()
    B, L = labels.shape
    for b in range(B):
        end_pos = None
        for pos in range(L):
            if labels[b, pos].item() == end_think_id:
                end_pos = pos
                break
        labels[b] = -100 if end_pos is None else labels[b]
        if end_pos is not None:
            labels[b, :end_pos] = -100
    return labels

# ── FORMATTERS ────────────────────────────────────────────────────────────────
def format_reasoning(item):
    if 'messages' in item:
        turns = []
        for msg in item['messages']:
            r, c = msg.get('role', ''), msg.get('content', '')
            if r == 'user':        turns.append(f"User: {c}")
            elif r == 'assistant': turns.append(f"Assistant: {c}")
        return '\n'.join(turns)
    return ''

def format_trivia(item):
    q = item.get('question', '').strip()
    ans = item.get('answer', {})
    if isinstance(ans, dict):
        a_list = ans.get('aliases', ans.get('value', ['']))
        a = a_list[0] if isinstance(a_list, list) and a_list else str(a_list)
    else:
        a = str(ans)
    a = a.strip()
    if not q or not a: return ''
    return f"User: {q}\nAssistant: <think>\nThe answer is {a}.\n</think>\n{a}"

def format_gsm8k(item):
    q = item.get('question', '').strip()
    a = item.get('answer', '').strip()
    if not q or not a: return ''
    final = a.split('####')[-1].strip() if '####' in a else a
    steps = a.split('####')[0].strip() if '####' in a else a
    return f"User: {q}\nAssistant: <think>\n{steps}\n</think>\n{final}"

def think_token_count(text, tok):
    s, e = text.find('<think>'), text.find('</think>')
    if s == -1 or e == -1 or e <= s: return 0
    return len(tok.encode(text[s + 7:e]))

# ── DATA GENERATOR ────────────────────────────────────────────────────────────
def make_dataloader(tokenizer, end_think_id):
    """50% DeepSeek + 30% TriviaQA + 20% GSM8K — all masked SFT batches."""
    print("[DATA] Loading sprint datasets...")
    ds_reason = load_dataset("allenai/big-reasoning-traces", "DeepSeek",
                             split="train", streaming=True)
    ds_trivia = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                             split="train", streaming=True)
    ds_gsm    = load_dataset("openai/gsm8k", "main",
                             split="train", streaming=True)

    sources = [
        {'iter': iter(ds_reason), 'ds': ds_reason, 'weight': 50,
         'name': 'reasoning', 'domain_id': 0, 'fmt': format_reasoning},
        {'iter': iter(ds_trivia), 'ds': ds_trivia, 'weight': 30,
         'name': 'trivia',    'domain_id': 4, 'fmt': format_trivia},
        {'iter': iter(ds_gsm),   'ds': ds_gsm,   'weight': 20,
         'name': 'gsm8k',     'domain_id': 1, 'fmt': format_gsm8k},
    ]
    weights   = [s['weight'] for s in sources]
    exhausted = [False] * len(sources)
    pad_id    = getattr(tokenizer, 'pad_token_id', 1) or 1

    while True:
        active = [i for i, e in enumerate(exhausted) if not e]
        if not active:
            print("[DATA] All streams exhausted — restarting.")
            for s in sources: s['iter'] = iter(s['ds'])
            exhausted = [False] * len(sources)
            active    = list(range(len(sources)))

        idx = random.choices(active, weights=[weights[i] for i in active], k=1)[0]
        src = sources[idx]
        try:
            item = next(src['iter'])
            text = src['fmt'](item)
            if not text or len(text.strip()) < 30: continue

            if src['name'] == 'reasoning':
                n = think_token_count(text, tokenizer)
                if n == 0 or n > MAX_THINK_TOKENS: continue

            tokens = tokenizer.encode(text)
            if len(tokens) < 16: continue

            chunk = tokens[:SEQ_LEN]
            if len(chunk) < SEQ_LEN:
                chunk = chunk + [pad_id] * (SEQ_LEN - len(chunk))

            input_ids  = torch.tensor([chunk], dtype=torch.long)
            labels     = make_masked_labels(input_ids, end_think_id)
            if (labels == -100).all(): continue
            domain_ids = torch.tensor([src['domain_id']], dtype=torch.long)
            yield (input_ids, labels, domain_ids)

        except StopIteration:
            exhausted[idx] = True
        except Exception as e:
            print(f"[DATA] '{src['name']}' error: {e} — restarting.")
            src['iter'] = iter(src['ds'])

# ── CHECKPOINT ────────────────────────────────────────────────────────────────
def save_ckpt(step, model, optimizer, reason=""):
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({'step': step, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, SAVE_AS)
    print(f"[CKPT] Saved {SAVE_AS} at step {step:,} ({reason})")

# ── TERMINATION PROBE ─────────────────────────────────────────────────────────
PROBE_PROMPTS = [
    "User: Who was the 16th president of the United States?\nAssistant: <think>\n",
    "User: Solve for x: 3x + 7 = 22\nAssistant: <think>\n",
    "User: What is 17 multiplied by 23?\nAssistant: <think>\n",
]

def probe(model, tok, device, end_think_id):
    """Returns list of (p_end, fired, text) and max P(</think>) seen."""
    model.eval()
    results = []
    max_p = 0.0
    with torch.no_grad():
        for prompt in PROBE_PROMPTS:
            ids = tok.encode(prompt, return_tensors='pt').to(device)
            gen, fired = [], False
            for _ in range(300):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(ids)
                nxt   = logits[0, -1].float() / 0.72
                probs = F.softmax(nxt, dim=-1)
                p_end = float(probs[end_think_id])
                tok_id = torch.multinomial(probs, 1).item()
                if tok_id == end_think_id:
                    fired = True; break
                if tok_id == tok.eos_token_id: break
                gen.append(tok_id)
                ids = torch.cat([ids, torch.tensor([[tok_id]], device=device)], dim=-1)
            p_end = float(F.softmax(logits[0, -1].float(), dim=-1)[end_think_id])
            max_p = max(max_p, p_end)
            text  = tok.decode(gen, skip_special_tokens=False)
            results.append((p_end, fired, text[:160].strip()))
    model.train()
    return results, max_p

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tok.eos_token_id = tok.eos_token_id or 0
    tok.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    end_think_id = tok.convert_tokens_to_ids("</think>")
    print(f"  </think>={end_think_id}")

    print(f"Loading checkpoint: {LOAD_FROM}")
    model = Mamba3Titan(vocab_size=50304, d_model=2560, n_layers=64,
                        mimo_paths=16, use_gradient_checkpointing=False)
    if len(tok) > model.embedding.weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    model.set_phase("sft")

    ckpt = torch.load(LOAD_FROM, map_location="cpu", weights_only=True)
    base_step = ckpt.get("step", 0)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(torch.bfloat16).to(device)
    print(f"  Loaded step {base_step:,}  VRAM={torch.cuda.memory_allocated(device)/1e9:.2f}GB")

    # ── FREEZE BACKBONE TO PREVENT OOM & CATASTROPHIC FORGETTING ──────────────
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.mimo_reasoning_blocks.parameters(): p.requires_grad_(True)
    for p in model.bridge.parameters(): p.requires_grad_(True)
    for p in model.lm_head.parameters(): p.requires_grad_(True)
    for p in model.domain_router.parameters(): p.requires_grad_(True)
    for p in model.bb_write.parameters(): p.requires_grad_(True)
    for p in model.bb_read.parameters(): p.requires_grad_(True)
    model.moe_scale.requires_grad_(True); model.cp_gate.requires_grad_(True)
    model.backbone_gate.requires_grad_(True); model.router_temp.requires_grad_(True)

    # ── Optimizer — fresh momentum, FLAT LR ───────────────────────────────────
    head_ids   = set(id(p) for p in model.lm_head.parameters())
    router_ids = set(id(p) for p in model.domain_router.parameters())
    router_ids.add(id(model.router_temp))
    emb_ids    = set(id(p) for p in model.embedding.parameters())

    head_p   = [p for p in model.lm_head.parameters()   if p.requires_grad]
    router_p = [p for p in model.parameters() if id(p) in router_ids and p.requires_grad]
    emb_p    = [p for p in model.parameters() if id(p) in emb_ids    and p.requires_grad]
    core_p   = [p for p in model.parameters()
                if id(p) not in head_ids and id(p) not in router_ids
                and id(p) not in emb_ids and p.requires_grad]

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.Adam8bit([
            {'params': emb_p,    'lr': FLAT_LR_HEAD,   'name': 'embedding'},
            {'params': head_p,   'lr': FLAT_LR_HEAD,   'name': 'head'},
            {'params': router_p, 'lr': FLAT_LR_ROUTER, 'name': 'router'},
            {'params': core_p,   'lr': FLAT_LR_CORE,   'name': 'core'},
        ], weight_decay=0.01)
        print(f"Optimizer: Adam8bit (flat LR={FLAT_LR_CORE:.0e}, fresh state)")
    except ImportError:
        optimizer = torch.optim.AdamW([
            {'params': emb_p,    'lr': FLAT_LR_HEAD,   'name': 'embedding'},
            {'params': head_p,   'lr': FLAT_LR_HEAD,   'name': 'head'},
            {'params': router_p, 'lr': FLAT_LR_ROUTER, 'name': 'router'},
            {'params': core_p,   'lr': FLAT_LR_CORE,   'name': 'core'},
        ], weight_decay=0.01)
        print(f"Optimizer: AdamW (flat LR={FLAT_LR_CORE:.0e}, fresh state)")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ── Data pipeline ─────────────────────────────────────────────────────────
    gen = make_dataloader(tok, end_think_id)
    _q  = queue.Queue(maxsize=2)
    _sentinel = object()
    def _worker(g, q):
        try:
            for item in g:
                q.put(tuple(t.pin_memory() if isinstance(t, torch.Tensor) else t
                            for t in item))
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_sentinel)
    threading.Thread(target=_worker, args=(gen, _q), daemon=True, name='DataPrefetch').start()
    def _iter():
        while True:
            item = _q.get()
            if item is _sentinel: break
            if isinstance(item, Exception): raise item
            yield item

    print(f"\n{'='*72}")
    print(f"  TERMINATION SPRINT  |  Base: {LOAD_FROM}  |  Steps: {TARGET_STEPS:,}")
    print(f"  FLAT LR: core={FLAT_LR_CORE:.0e}  head={FLAT_LR_HEAD:.0e}")
    print(f"  Early exit: P(</think>) > {EARLY_EXIT_P}  |  Probe every 200 steps")
    print(f"  Mix: 50% DeepSeek + 30% TriviaQA + 20% GSM8K (all masked)")
    print(f"{'='*72}\n")

    step_time  = time.time()
    local_step = 0
    model.train()

    for step, batch in enumerate(_iter(), start=base_step):
        if local_step >= TARGET_STEPS:
            print(f"\n[COMPLETE] {TARGET_STEPS:,} sprint steps done.")
            save_ckpt(step, model, optimizer, "sprint_complete")
            break

        input_ids, labels, domain_ids = batch
        input_ids  = input_ids.to(device,  non_blocking=True)
        labels     = labels.to(device,     non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(input_ids, loop_idx=0, domain_ids=domain_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = criterion(shift_logits.view(-1, 50304), shift_labels.view(-1))

        lm_loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()

        if (local_step + 1) % 20 == 0:
            import gc; gc.collect()
            torch.cuda.empty_cache()

        elapsed    = time.time() - step_time
        tps        = SEQ_LEN / max(elapsed, 1e-6)
        step_time  = time.time()
        temp       = gpu_temp()
        active_tok = int((labels != -100).sum().item())
        telem      = model.last_telemetry
        loss_val   = lm_loss.item()

        if elapsed > DATA_TIMEOUT:
            print(f"[WATCHDOG] {elapsed:.0f}s — saving & aborting.")
            save_ckpt(step + 1, model, optimizer, "watchdog"); sys.exit(1)

        line = (f"Phase sft3 | Step {step+1:05d} | LM Loss: {loss_val:.4f} | "
                f"Dom Loss: 0.0000 | Div: 0.0000 | Gate: {telem.get('gate_score',0):.4f} | "
                f"Entropy: {telem.get('entropy',0):.4f} | GNorm: {gnorm:.2f} | "
                f"TPS: {tps:.1f}{(' | GPU:'+str(temp)+'°C') if temp else ''} | "
                f"LR: {FLAT_LR_CORE:.0e} | ActTok: {active_tok}")
        print(line, flush=True)
        with open(LOG_PATH,  'a') as f: f.write(line + '\n')
        with open(TRAIN_LOG, 'a') as f: f.write(line + '\n')

        try:
            with open(TELEM_PATH, 'w') as f:
                json.dump({"phase": "sft3_sprint", "step": step+1,
                           "lm_loss": round(loss_val, 4), "domain_loss": 0.0,
                           "gate_score": round(telem.get('gate_score',0), 4),
                           "entropy": round(telem.get('entropy',0), 4),
                           "grad_norm": round(gnorm, 4), "tps": round(tps, 1),
                           "gpu_temp": temp, "lr": FLAT_LR_CORE,
                           "resume_step": base_step, "active_tokens": active_tok,
                           "arm_weights": telem.get('arm_weights', []),
                           "arm_collapse_mean": 0.0, "arm_collapse_max": 0.0,
                           "arm_sims": [], "latent_energy": 0.0,
                           "top2_pairings": {}}, f)
        except Exception:
            pass

        if (step + 1) % SAVE_EVERY == 0:
            save_ckpt(step + 1, model, optimizer, "periodic")

        # ── Probe every 200 steps ─────────────────────────────────────────────
        if (step + 1) % 200 == 0:
            print(f"\n[PROBE @ step {step+1}]")
            results, max_p = probe(model, tok, device, end_think_id)
            for p_end, fired, text in results:
                print(f"  P(</think>)={p_end:.5f} fired={fired} | {text}")
            print(f"  >> max P(</think>)={max_p:.5f}\n")

            # Early exit if termination signal is firing
            if max_p >= EARLY_EXIT_P:
                print(f"\n🎯 [EARLY EXIT] P(</think>)={max_p:.5f} >= {EARLY_EXIT_P}")
                print(f"   Termination signal is live — saving final checkpoint.")
                save_ckpt(step + 1, model, optimizer, "early_exit_termination_live")
                break

        if _shutdown:
            print("[SHUTDOWN] Saving.")
            save_ckpt(step + 1, model, optimizer, "signal_shutdown")
            break

        local_step += 1

    print(f"\n[SPRINT DONE]  Checkpoint: {SAVE_AS}")

if __name__ == "__main__":
    main()
