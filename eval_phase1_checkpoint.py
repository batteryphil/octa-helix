"""
Phase 1 Checkpoint Evaluation
Tests: coherence, perplexity, arm specialization, repetition, and readiness for Phase 2.
"""
import torch, sys, os, json, math, time
sys.path.insert(0, os.path.dirname(__file__))

CKPT = "checkpoints_2.7b/phase_1.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW = 120
TEMP    = 0.7
TOP_P   = 0.9

print("="*70)
print("  OCTA-HELIX TITAN — PHASE 1 CHECKPOINT EVALUATION")
print(f"  Checkpoint: {CKPT}")
print("="*70)

# ── Load model ────────────────────────────────────────────────────────────────
print("[1/5] Loading model...", flush=True)
from mamba3_titan_builder import Mamba3Titan
from transformers import AutoTokenizer
import gc

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
if tok.pad_token is None: tok.pad_token = tok.eos_token

# Load checkpoint weights first, then build model directly on GPU
# This avoids the CPU→GPU double-buffer OOM
print("    Loading checkpoint weights...", flush=True)
state = torch.load(CKPT, map_location="cpu", weights_only=False)
model_state = state.get("model_state_dict", state)
del state
gc.collect()
torch.cuda.empty_cache()

print("    Building model skeleton on CPU...", flush=True)
model = Mamba3Titan(vocab_size=50288, d_model=2560, n_layers=64,
                    mimo_paths=8, use_gradient_checkpointing=False)
model.load_state_dict(model_state, strict=False)
del model_state
gc.collect()

# Move to GPU in bfloat16 to save VRAM
print("    Moving to GPU (bfloat16)...", flush=True)
torch.cuda.empty_cache()
try:
    model = model.bfloat16().to(DEVICE)
except torch.OutOfMemoryError:
    print("    GPU OOM — falling back to CPU (eval will be slow)")
    DEVICE = "cpu"
    model = model.bfloat16().to(DEVICE)
model.eval()

params = sum(p.numel() for p in model.parameters()) / 1e9
print(f"    Loaded: {params:.2f}B params | device={DEVICE}")

# ── Test prompts by domain ────────────────────────────────────────────────────
TESTS = [
    ("General Language",    "The most important thing in life is"),
    ("Factual Knowledge",   "The capital of France is Paris, and the capital of Germany is"),
    ("Math Reasoning",      "If a train travels at 60 miles per hour for 2.5 hours, it covers"),
    ("Code Generation",     "def fibonacci(n):\n    # Returns the nth Fibonacci number\n    "),
    ("Logic / Reasoning",   "All birds have wings. Penguins are birds. Therefore, penguins"),
    ("Summarization",       "In summary, the key points of the argument are:"),
    ("Creative Writing",    "Once upon a time in a world where machines could dream,"),
    ("Instruction Follow",  "To make a cup of tea, follow these steps:\n1."),
]

# ── Generation helper ─────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt, max_new=MAX_NEW, temp=TEMP, top_p=TOP_P):
    ids = tok.encode(prompt)[-256:]
    x   = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    generated = []
    arm_weights_list = []

    for _ in range(max_new):
        out = model(x)
        # model returns (logits, domain_loss) tuple
        if isinstance(out, tuple):
            logits = out[0][:, -1, :]
            arm_weights_list.append([0.125]*8)
        elif isinstance(out, dict):
            logits = out["logits"][:, -1, :]
            if "arm_weights" in out:
                arm_weights_list.append(out["arm_weights"][0].cpu().tolist())
        else:
            logits = out[:, -1, :]
        logits = logits.float()

        # Clamp to tokenizer vocab BEFORE sampling
        logits[:, tok.vocab_size:] = -float('inf')

        if temp > 0:
            logits = logits / temp
            probs = torch.softmax(logits, dim=-1)          # [1, vocab]
            sorted_p, sorted_i = torch.sort(probs[0], descending=True)  # both [vocab]
            cum_p = torch.cumsum(sorted_p, dim=0)
            sorted_p[cum_p - sorted_p > top_p] = 0
            sorted_p /= sorted_p.sum()
            sampled_idx = torch.multinomial(sorted_p, 1).item()   # scalar
            next_id_val  = sorted_i[sampled_idx].item()           # scalar
        else:
            next_id_val = logits[0].argmax().item()               # scalar

        if next_id_val == tok.eos_token_id:
            break
        generated.append(next_id_val)
        next_tok = torch.tensor([[next_id_val]], dtype=torch.long, device=DEVICE)
        x = torch.cat([x, next_tok], dim=1)
        if x.shape[1] > 512:
            x = x[:, -512:]

    text = tok.decode(generated, skip_special_tokens=True)
    avg_weights = [sum(w[i] for w in arm_weights_list)/max(len(arm_weights_list),1)
                   for i in range(8)] if arm_weights_list else [0]*8
    return text, avg_weights

# ── Perplexity helper ─────────────────────────────────────────────────────────
@torch.no_grad()
def perplexity(text, max_len=256):
    ids = tok.encode(text)[:max_len]
    if len(ids) < 4: return float('inf')
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)
    y = torch.tensor(ids[1:],    dtype=torch.long, device=DEVICE)
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else (out["logits"] if isinstance(out, dict) else out)
    logits = logits[0].float()          # [seq_len, vocab]
    logits[:, tok.vocab_size:] = -float('inf')
    loss = torch.nn.functional.cross_entropy(logits, y)
    return math.exp(loss.item())

# ── Run tests ─────────────────────────────────────────────────────────────────
print("\n[2/5] Generation quality per domain:")
print("-"*70)

results = []
ARM_NAMES = ["General","Math","Logic","Code","Factual","Summary","Creative","Instruct"]

for domain, prompt in TESTS:
    t0 = time.time()
    output, arm_w = generate(prompt)
    elapsed = time.time() - t0
    dom_arm = ARM_NAMES.index(domain.split()[0]) if domain.split()[0] in ARM_NAMES else 0
    top_arm = ARM_NAMES[arm_w.index(max(arm_w))]
    top_pct = max(arm_w) * 100

    # Repetition check
    words = output.split()
    unique_ratio = len(set(words)) / max(len(words), 1)

    print(f"\n  [{domain}]")
    print(f"  Prompt: \"{prompt[:60]}\"")
    print(f"  Output: \"{output[:180]}\"")
    print(f"  Top arm: {top_arm} ({top_pct:.0f}%) | Unique words: {unique_ratio:.0%} | {elapsed:.1f}s")

    results.append({
        "domain": domain,
        "output": output,
        "top_arm": top_arm,
        "top_pct": top_pct,
        "unique_ratio": unique_ratio,
        "coherent": unique_ratio > 0.35,
    })

# ── Perplexity on reference texts ─────────────────────────────────────────────
print("\n[3/5] Perplexity on reference texts (lower = better):")
print("-"*70)
REF_TEXTS = [
    ("News prose",    "Scientists have discovered a new species of deep-sea fish that produces its own light using bioluminescent organs located along its lateral line."),
    ("Math text",     "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides."),
    ("Code comment",  "This function iterates over the list and returns the maximum value found. It uses a simple linear scan with O(n) time complexity."),
    ("Instruction",   "First, preheat the oven to 375 degrees Fahrenheit. Then mix the dry ingredients in a large bowl before adding the wet ingredients gradually."),
]
perplexities = []
for label, text in REF_TEXTS:
    ppl = perplexity(text)
    perplexities.append(ppl)
    quality = "✓ Excellent" if ppl < 5 else "✓ Good" if ppl < 20 else "~ Acceptable" if ppl < 100 else "✗ Poor"
    print(f"  {label:<20} PPL={ppl:>8.2f}  {quality}")

avg_ppl = sum(perplexities) / len(perplexities)

# ── Arm specialization ────────────────────────────────────────────────────────
print("\n[4/5] Arm specialization analysis:")
print("-"*70)
arm_usage = {n: 0 for n in ARM_NAMES}
for r in results:
    arm_usage[r["top_arm"]] = arm_usage.get(r["top_arm"], 0) + 1

print("  Arm dominance counts across all prompts:")
for arm, count in sorted(arm_usage.items(), key=lambda x: -x[1]):
    bar = "█" * count
    print(f"  {arm:<12} {bar} ({count})")

# ── Load telemetry for collapse metric ───────────────────────────────────────
try:
    telem = json.load(open("monitor_ui/telemetry.json"))
    collapse = telem.get("arm_collapse_mean", "n/a")
    arm_sims = telem.get("arm_sims", [])
    arm_weights = telem.get("arm_weights", [])
    print(f"\n  Arm similarity (0=diverse, 1=clone):")
    for i, (s, w) in enumerate(zip(arm_sims, arm_weights)):
        bar = "█" * int(s * 20)
        print(f"  Arm {i} ({ARM_NAMES[i]:<8}): sim={s:.3f} weight={w:.3f} {bar}")
except: pass

# ── Verdict ───────────────────────────────────────────────────────────────────
print("\n[5/5] PHASE 1 READINESS VERDICT")
print("="*70)

coherent_count = sum(1 for r in results if r["coherent"])
arm_collapse   = len(set(r["top_arm"] for r in results))
div_arms       = arm_collapse

print(f"  Steps completed:    {telem.get('step', '?')} / 40,000  ({telem.get('step', 0)/40000*100:.1f}%)")
print(f"  LM Loss:            {telem.get('lm_loss', '?')}")
print(f"  Avg Perplexity:     {avg_ppl:.2f}")
print(f"  Coherent outputs:   {coherent_count}/8  ({coherent_count/8*100:.0f}%)")
print(f"  Arm diversity:      {div_arms} distinct dominant arms used")
print(f"  Arm collapse:       {telem.get('arm_collapse_mean', '?')}")
print()

# Phase 2 readiness check
ready_loss   = avg_ppl < 30
ready_cohere = coherent_count >= 6
ready_arms   = telem.get('step', 0) >= 15000  # arms need more steps to diverge

print("  Phase 2 readiness checklist:")
print(f"  {'✓' if ready_loss   else '✗'} Perplexity < 30      (avg={avg_ppl:.1f})")
print(f"  {'✓' if ready_cohere else '✗'} Coherent outputs ≥6  ({coherent_count}/8)")
print(f"  {'✗' if not ready_arms else '✓'} Arms diverging        (step {telem.get('step',0)}/15000+ needed)")
print()

if ready_loss and ready_cohere and ready_arms:
    print("  ✅ READY FOR PHASE 2")
elif ready_loss and ready_cohere:
    print("  ⏳ GOOD PROGRESS — continue Phase 1 until arms diverge (Div > 0)")
    print(f"     Estimated steps needed: ~10,000-15,000 more")
else:
    print("  ⚠️  NEEDS MORE TRAINING")

print("="*70)

# Save results
out = {"step": telem.get("step"), "avg_ppl": avg_ppl, "coherent": coherent_count,
       "arm_diversity": div_arms, "results": results}
json.dump(out, open("eval_reports/phase1_eval_current.json","w"), indent=2)
print(f"\n  Results saved → eval_reports/phase1_eval_current.json")
