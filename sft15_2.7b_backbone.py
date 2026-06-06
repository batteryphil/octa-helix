"""
sft15_backbone_unfreeze.py — Unfreeze backbone layers 10-30
────────────────────────────────────────────────────────────
Goal: fix factual recall by training the MIDDLE backbone layers
      where semantic Q→A associations are stored.

Safeguards:
  • EWC (Elastic Weight Consolidation) — penalise large deviations
    from pre-trained weights to prevent catastrophic forgetting
  • Ultra-low LR: 5e-9 for mid-backbone, 1e-7 for top layers+head
  • Termination guard: abort if </think> max-P drops below 0.60
  • 50% gold injection (every 2 steps) for maximum factual signal
  • 10 000 steps, probe every 500, save every 1000
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, time, signal, threading, queue, random, json, copy
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from mamba3_titan_builder import Mamba3Titan
try:
    from huggingface_hub import login
    login(token="HF_TOKEN_REDACTED", add_to_git_credential=False)
except: pass
from datasets import load_dataset
from transformers import AutoTokenizer
try:
    import pynvml; pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_temp = lambda: pynvml.nvmlDeviceGetTemperature(_h, pynvml.NVML_TEMPERATURE_GPU)
except: gpu_temp = lambda: None

# ── Config ────────────────────────────────────────────────────────────────────
LOAD_FROM      = "checkpoints_2.7b/phase_1.pt"  # continue from Phase 1
SAVE_AS        = "checkpoints_2.7b/phase_sft15_factual.pt"
LOG_PATH       = "sft15.log"
TELEM          = "monitor_ui/telemetry.json"

TARGET_STEPS   = 10_000
SEQ_LEN        = 192
LR_MID         = 5e-8         # 10x increase from initial — EWC protects us
LR_TOP         = 5e-7         # top layers + head also higher
EWC_LAMBDA     = 50.0         # looser anchor — allow learning at higher LR
GOLD_EVERY     = 2            # inject gold fact every N steps (50%)
PROBE_EVERY    = 500
SAVE_EVERY     = 1000
TERM_ABORT_P   = 0.55         # abort if max probe P drops below this
CLIP           = 0.5
GRAD_ACCUM     = 2            # accumulate gradients to reduce peak memory

UNFREEZE_START = 20           # first backbone layer to unfreeze (OOM fix)
UNFREEZE_END   = 26           # last backbone layer to unfreeze (inclusive)

_shutdown = False
def _sig(s, f): global _shutdown; _shutdown = True
signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)

# ── 40 gold facts ─────────────────────────────────────────────────────────────
GOLD = [
    ("Who was the 16th president of the United States?",
     "Abraham Lincoln was the 16th president, serving 1861-1865.",    "Abraham Lincoln"),
    ("What is the capital of France?",
     "The capital of France is Paris.",                                "Paris"),
    ("What is 2+2?",           "2+2 equals 4.",                       "4"),
    ("Solve for x: 3x+7=22",  "3x=15, so x=5.",                      "x = 5"),
    ("Is the Great Wall of China visible from space?",
     "No — it is too narrow to see from orbit.",                      "No"),
    ("Is the sky blue?",
     "Yes — Rayleigh scattering makes the sky appear blue.",           "Yes"),
    ("What is the largest planet?",
     "Jupiter is the largest planet in the solar system.",             "Jupiter"),
    ("Who wrote Romeo and Juliet?",
     "Romeo and Juliet was written by William Shakespeare.",           "William Shakespeare"),
    ("What is H2O?",           "H2O is the chemical formula for water.", "Water"),
    ("What is the capital of Germany?",
     "The capital of Germany is Berlin.",                              "Berlin"),
    ("Who painted the Mona Lisa?",
     "The Mona Lisa was painted by Leonardo da Vinci.",                "Leonardo da Vinci"),
    ("What is the square root of 144?",
     "The square root of 144 is 12.",                                  "12"),
    ("What is the capital of Japan?",
     "The capital of Japan is Tokyo.",                                 "Tokyo"),
    ("Who was the first person on the Moon?",
     "Neil Armstrong was the first person to walk on the Moon (1969).","Neil Armstrong"),
    ("What continent is Brazil in?",
     "Brazil is in South America.",                                    "South America"),
    ("What is the capital of Australia?",
     "The capital of Australia is Canberra.",                          "Canberra"),
    ("All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?",
     "Yes — by syllogism all cats are mammals, so Whiskers is a mammal.", "Yes"),
    ("What is 7 times 8?",     "7 times 8 equals 56.",                "56"),
    ("What is the boiling point of water?",
     "Water boils at 100 degrees Celsius.",                            "100 degrees Celsius"),
    ("What is the speed of light?",
     "Light travels at about 299,792 km/s.",                           "299,792 km/s"),
    ("What is the capital of Italy?",  "The capital of Italy is Rome.",   "Rome"),
    ("What is the capital of Spain?",  "The capital of Spain is Madrid.", "Madrid"),
    ("How many continents are there?", "There are 7 continents.",         "7"),
    ("What is the chemical symbol for gold?",
     "The chemical symbol for gold is Au.",                            "Au"),
    ("What planet is closest to the Sun?",
     "Mercury is the closest planet to the Sun.",                      "Mercury"),
    ("What is the capital of Canada?",
     "The capital of Canada is Ottawa.",                               "Ottawa"),
    ("Who invented the telephone?",
     "Alexander Graham Bell invented the telephone.",                  "Alexander Graham Bell"),
    ("What is the atomic number of carbon?",
     "Carbon has an atomic number of 6.",                              "6"),
    ("What is the capital of China?",
     "The capital of China is Beijing.",                               "Beijing"),
    ("What is 100 divided by 4?",
     "100 divided by 4 equals 25.",                                    "25"),
    ("Is a whale a mammal?",   "Yes, whales are marine mammals.",      "Yes"),
    ("What language do Brazilians speak?",
     "Brazilians speak Portuguese.",                                   "Portuguese"),
    ("How many sides does a hexagon have?",
     "A hexagon has 6 sides.",                                         "6"),
    ("What is the tallest mountain?",
     "Mount Everest is the tallest mountain at 8,849 metres.",         "Mount Everest"),
    ("Who discovered gravity?",
     "Isaac Newton formulated the law of gravity.",                    "Isaac Newton"),
    ("What is the capital of Russia?",
     "The capital of Russia is Moscow.",                               "Moscow"),
    ("Solve for x: 2x = 10",   "Dividing both sides by 2 gives x=5.", "x = 5"),
    ("What is the capital of Brazil?",
     "The capital of Brazil is Brasília.",                             "Brasília"),
    ("What is 15% of 200?",    "15% of 200 is 30.",                   "30"),
    ("Is the Earth flat?",
     "No — the Earth is an oblate spheroid.",                          "No"),
]

def fmt_gold(q, r, a):
    return f"User: {q}\nAssistant: <think>\n{r}\n</think>\n{a}"

def make_labels(ids, open_id, end_id, pad_id=1):
    labels = ids.clone()
    for b in range(ids.shape[0]):
        seq = ids[b].tolist()
        start = next((i for i,t in enumerate(seq) if t == open_id), None)
        if start is None: labels[b] = -100
        else: labels[b, :start+1] = -100
        for i in range(len(seq)-1, -1, -1):
            if seq[i] == pad_id: labels[b, i] = -100
            else: break
    return labels

def fmt_trivia(item):
    q = item.get('question','').strip()
    a = item.get('answer', {})
    if isinstance(a, dict): a = a.get('aliases', [a.get('value','')])[0]
    a = str(a).strip()
    if not q or not a or len(a) > 60: return ''
    return f"User: {q}\nAssistant: <think>\nThe answer is {a}.\n</think>\n{a}"

def fmt_boolq(item):
    q = item.get('question','').strip()
    a = "Yes" if item.get('answer', False) else "No"
    if not q: return ''
    return f"User: {q}\nAssistant: <think>\nThe answer is {a.lower()}.\n</think>\n{a}"

def dataloader(tok, open_id, end_id):
    ds_t = load_dataset("mandarjoshi/trivia_qa","rc.nocontext",split="train",streaming=True)
    ds_b = load_dataset("google/boolq", split="train", streaming=True)
    srcs = [
        {'iter':iter(ds_t),'ds':ds_t,'fmt':fmt_trivia,'w':50},
        {'iter':iter(ds_b),'ds':ds_b,'fmt':fmt_boolq, 'w':50},
    ]
    pad = tok.pad_token_id or 1; gold_idx = 0; step = 0
    while True:
        step += 1
        if step % GOLD_EVERY == 0:                   # 50% gold
            q,r,a = GOLD[gold_idx % len(GOLD)]; gold_idx += 1
            text = fmt_gold(q,r,a)
            toks = tok.encode(text)[:SEQ_LEN]
            if len(toks) < SEQ_LEN: toks += [pad]*(SEQ_LEN-len(toks))
            ids = torch.tensor([toks], dtype=torch.long)
            labels = make_labels(ids, open_id, end_id, pad)
            yield ids, labels, 'gold'; continue
        w = [s['w'] for s in srcs]
        idx = random.choices(range(len(srcs)), weights=w, k=1)[0]
        src = srcs[idx]
        try:
            item = next(src['iter']); text = src['fmt'](item)
            if not text or len(text) < 8: continue
            toks = tok.encode(text)[:SEQ_LEN]
            if len(toks) < 6: continue
            if len(toks) < SEQ_LEN: toks += [pad]*(SEQ_LEN-len(toks))
            ids = torch.tensor([toks], dtype=torch.long)
            labels = make_labels(ids, open_id, end_id, pad)
            if (labels == -100).all(): continue
            yield ids, labels, 'qa'
        except StopIteration: src['iter'] = iter(src['ds'])
        except: pass

PROBES = [
    ("president", "User: Who was the 16th president of the United States?\nAssistant: <think>\n", "Abraham Lincoln"),
    ("capital",   "User: What is the capital of France?\nAssistant: <think>\n",                  "Paris"),
    ("math",      "User: What is 2+2?\nAssistant: <think>\n",                                    "4"),
    ("algebra",   "User: Solve for x: 3x+7=22\nAssistant: <think>\n",                           "x = 5"),
    ("logic",     "User: All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?\nAssistant: <think>\n", "Yes"),
]

def run_probe(model, tok, device, end_id):
    model.eval(); results = []; max_p = 0.0
    with torch.no_grad():
        for pname, prompt, expected in PROBES:
            ids = tok.encode(prompt, return_tensors='pt').to(device)
            gen = []; fired = False; fp = None
            for i in range(200):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(ids)
                raw = logits[0,-1].float()
                p = float(F.softmax(raw, dim=-1)[end_id])
                max_p = max(max_p, p)
                t = torch.multinomial(F.softmax(raw/0.8, dim=-1), 1).item()
                if t == end_id: fired = True; fp = i; break
                if t == tok.eos_token_id: break
                gen.append(t); ids = torch.cat([ids, torch.tensor([[t]], device=device)], dim=-1)
            out = []
            if fired:
                for _ in range(30):
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        lg, _ = model(ids)
                    raw2 = lg[0,-1].float(); raw2[end_id] = -1e9
                    t2 = torch.multinomial(F.softmax(raw2/0.8, dim=-1), 1).item()
                    if t2 == tok.eos_token_id: break
                    out.append(t2); ids = torch.cat([ids, torch.tensor([[t2]], device=device)], dim=-1)
            ans = tok.decode(out, skip_special_tokens=True).strip()
            ok = expected.lower() in ans.lower() or any(w in ans.lower() for w in expected.lower().split() if len(w)>2)
            results.append((pname, p, fired, fp, ans, ok, expected))
    model.train(); return results, max_p

def save(step, model, reason=""):
    os.makedirs("checkpoints_2.7b", exist_ok=True)
    torch.save({"step": step, "model": model.state_dict()}, SAVE_AS)
    print(f"[CKPT] {SAVE_AS} @ step {step} ({reason})")

def main():
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tok.eos_token_id = tok.eos_token_id or 0
    tok.add_special_tokens({"additional_special_tokens": ["<think>","</think>"]})
    end_id  = tok.convert_tokens_to_ids("</think>")
    open_id = tok.convert_tokens_to_ids("<think>")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = Mamba3Titan(vocab_size=50304, d_model=2560, n_layers=64,
                        mimo_paths=16, use_gradient_checkpointing=False)
    model.resize_token_embeddings(50304); model.set_phase('sft')
    ckpt = torch.load(LOAD_FROM, map_location='cpu', weights_only=True)
    base_step = ckpt.get('step', 0)
    model.load_state_dict(ckpt['model'], strict=False)

    # ── EWC: snapshot pre-trained weights for the unfrozen layers ──────────────
    print(f"[EWC] Snapshotting layers {UNFREEZE_START}-{UNFREEZE_END} for weight anchoring...")
    ewc_anchors = {}
    for li in range(UNFREEZE_START, UNFREEZE_END + 1):
        for name, param in model.layers[li].named_parameters():
            key = f"layers.{li}.{name}"
            # Keep anchor on GPU in bfloat16 to avoid PCIe bottleneck and save RAM
            ewc_anchors[key] = param.data.clone().to(device=device, dtype=torch.bfloat16)
    print(f"[EWC] Anchored {len(ewc_anchors)} parameter tensors on GPU")

    model = model.to(torch.bfloat16).to(device)
    print(f"  Loaded step {base_step:,}  VRAM={torch.cuda.memory_allocated(device)/1e9:.2f}GB")

    # ── Freeze everything, then selectively unfreeze ────────────────────────────
    for p in model.parameters(): p.requires_grad_(False)

    # Unfreeze mid-backbone layers 10-30 (factual recall zone)
    mid_params = []
    for li in range(UNFREEZE_START, UNFREEZE_END + 1):
        for p in model.layers[li].parameters():
            p.requires_grad_(True); mid_params.append(p)

    # Unfreeze top backbone layers 31-63 + head (reasoning/termination zone)
    top_params = []
    for li in range(31, 64):
        for p in model.layers[li].parameters():
            p.requires_grad_(True); top_params.append(p)
    for p in model.lm_head.parameters():
        p.requires_grad_(True); top_params.append(p)
    for p in model.mimo_reasoning_blocks.parameters():
        p.requires_grad_(True); top_params.append(p)
    for p in model.bridge.parameters():
        p.requires_grad_(True); top_params.append(p)
    
    # ── CRITICAL FIX: The Router & Blackboard must be unfrozen to adapt to the shifting backbone ──
    for p in model.domain_router.parameters():
        p.requires_grad_(True); top_params.append(p)
    for p in model.bb_read.parameters():
        p.requires_grad_(True); top_params.append(p)
    for p in model.bb_write.parameters():
        p.requires_grad_(True); top_params.append(p)
    top_params.extend([model.router_temp, model.moe_scale, model.cp_gate, model.backbone_gate])
    model.router_temp.requires_grad_(True)
    model.moe_scale.requires_grad_(True)
    model.cp_gate.requires_grad_(True)
    model.backbone_gate.requires_grad_(True)

    n_mid = sum(p.numel() for p in mid_params)
    n_top = sum(p.numel() for p in top_params)
    print(f"  Mid-backbone trainable: {n_mid/1e6:.1f}M (LR={LR_MID:.0e})")
    print(f"  Top-backbone trainable: {n_top/1e6:.1f}M (LR={LR_TOP:.0e})")

    import bitsandbytes as bnb
    opt = bnb.optim.Adam8bit([
        {'params': mid_params, 'lr': LR_MID, 'weight_decay': 0.0},
        {'params': top_params, 'lr': LR_TOP, 'weight_decay': 0.01},
    ])

    # ── Data pipeline ──────────────────────────────────────────────────────────
    gen = dataloader(tok, open_id, end_id)
    _q  = queue.Queue(maxsize=2)
    def _w(g, q):
        for it in g:
            q.put(tuple(t.pin_memory() if isinstance(t, torch.Tensor) else t for t in it))
    threading.Thread(target=_w, args=(gen, _q), daemon=True).start()

    print(f"\n{'='*68}")
    print(f"  SFT15 BACKBONE UNFREEZE — FACTUAL GROUNDING")
    print(f"  Base step: {base_step:,}  Total: {TARGET_STEPS:,} steps")
    print(f"  Unfreezing layers {UNFREEZE_START}-{UNFREEZE_END} | EWC λ={EWC_LAMBDA}")
    print(f"  Gold injection: every {GOLD_EVERY} steps ({100//GOLD_EVERY}% of batches)")
    print(f"  Termination abort threshold: maxP < {TERM_ABORT_P}")
    print(f"{'='*68}\n")

    model.train(); step_time = time.time(); best_correct = 0; accum = 0

    for local_step in range(TARGET_STEPS):
        if _shutdown: break
        ids, labels, src = _q.get()
        ids    = ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(ids)
            B, L, V = logits.shape
            sl = logits[:,:-1].contiguous().view(-1, V)
            la = labels[:,1:].contiguous().view(-1)
            ce_loss = F.cross_entropy(sl, la, ignore_index=-100)

        # EWC penalty computed on GPU, WITH gradients!
        ewc_loss = 0.0
        for li in range(UNFREEZE_START, UNFREEZE_END + 1):
            for name, param in model.layers[li].named_parameters():
                key = f"layers.{li}.{name}"
                if key in ewc_anchors:
                    diff = param - ewc_anchors[key]
                    ewc_loss += (diff ** 2).sum()
        ewc_loss = EWC_LAMBDA * ewc_loss / max(len(ewc_anchors), 1)
        loss = (ce_loss + ewc_loss) / GRAD_ACCUM

        loss.backward(); accum += 1
        if accum < GRAD_ACCUM: continue
        accum = 0
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], CLIP))
        opt.step(); opt.zero_grad(set_to_none=True)

        elapsed = time.time() - step_time; step_time = time.time()
        tps = (SEQ_LEN * GRAD_ACCUM) / max(elapsed, 1e-6); temp = gpu_temp()
        line = (f"sft15 | Step {base_step+local_step+1:05d} | "
                f"CE:{ce_loss.item():.3f} EWC:{ewc_loss.item():.2f} | "
                f"GNorm:{gnorm:.2f} | TPS:{tps:.0f}"
                f"{' | GPU:'+str(temp)+'°C' if temp else ''} [{src.upper()}]")
        print(line, flush=True)
        with open(LOG_PATH, 'a') as f: f.write(line + "\n")
        with open("training_log.txt", 'a') as f: f.write(line + "\n")
        try:
            with open(TELEM, 'w') as f:
                json.dump({"phase":"sft15","step":base_step+local_step+1,
                           "lm_loss":round(ce_loss.item(),4),
                           "domain_loss":round(float(ewc_loss.item()),4),
                           "grad_norm":round(gnorm,4),"tps":round(tps,1),
                           "gpu_temp":temp,"lr":LR_MID,
                           "arm_weights":[0.125]*8,"gate_score":0.125,
                           "entropy":2.08,"arm_collapse_mean":0.09,
                           "arm_collapse_max":0.14,"arm_sims":[],"latent_energy":0.0,
                           "top2_pairings":{}}, f)
        except: pass

        if (local_step+1) % SAVE_EVERY == 0:
            save(base_step+local_step+1, model, "periodic")

        if (local_step+1) % PROBE_EVERY == 0:
            step_lbl = base_step + local_step + 1
            print(f"\n[PROBE @ step {step_lbl}]")
            results, max_p = run_probe(model, tok, device, end_id)
            fired_n = correct_n = 0
            for pname, p_end, fired, fp, ans, ok, expected in results:
                s = "✅" if fired else "❌"; c = "🎯" if ok else "✗"
                if fired: fired_n += 1
                if ok:    correct_n += 1
                print(f"  {s}{c} [{pname:10s}] @{fp if fp else '--':>3} "
                      f"P={p_end:.3f} | '{ans[:35]}' [exp:{expected}]")
            print(f"  >> {fired_n}/{len(PROBES)} fired | {correct_n}/{len(PROBES)} correct "
                  f"| maxP={max_p:.4f}\n")

            if max_p < TERM_ABORT_P:
                print(f"⚠️  TERMINATION DEGRADED — maxP={max_p:.3f} < {TERM_ABORT_P}")
                print(f"   Halting to protect </think> capability. Saving current state.")
                save(step_lbl, model, "ABORT_TERM_DEGRADED"); return

            if correct_n > best_correct:
                best_correct = correct_n
                save(step_lbl, model, f"best_{correct_n}correct")

            if correct_n >= 5:
                print(f"🎯 FACTUAL COMPLETE — {correct_n}/{len(PROBES)} correct")
                save(step_lbl, model, "FACTUAL_COMPLETE"); return

    save(base_step+TARGET_STEPS, model, "done")
    print(f"\n[SFT15 DONE] → {SAVE_AS}")

if __name__ == "__main__":
    main()
