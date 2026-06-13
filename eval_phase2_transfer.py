"""
Phase 1 → Phase 2 Transfer Quality Test

The real question: does the Phase 1 backbone have enough domain knowledge
to support Phase 2 specialization? Tests:
  1. Teacher-forced perplexity by domain (how well does it understand each domain?)
  2. Arm output divergence (have arms already started separating naturally?)
  3. Router domain sensitivity (does routing change with different inputs?)
"""
import torch, sys, os, json, math, gc
sys.path.insert(0, os.path.dirname(__file__))

CKPT   = "checkpoints_2.7b/phase_1.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*65)
print("  PHASE 1 → PHASE 2 TRANSFER TEST")
print("  Can Phase 2 domain injection build on this backbone?")
print("="*65)

# ── Load ──────────────────────────────────────────────────────────────────────
from mamba3_titan_builder import Mamba3Titan
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
if tok.pad_token is None: tok.pad_token = tok.eos_token

state    = torch.load(CKPT, map_location="cpu", weights_only=False)
model_sd = state.get("model_state_dict", state)
step     = state.get("step", "?")
del state; gc.collect(); torch.cuda.empty_cache()

model = Mamba3Titan(vocab_size=50288, d_model=2560, n_layers=64,
                    mimo_paths=8, use_gradient_checkpointing=False)
model.load_state_dict(model_sd, strict=False)
del model_sd; gc.collect(); torch.cuda.empty_cache()

model = model.bfloat16().to(DEVICE)
model.eval()
print(f"  Loaded step {step}\n")

ARM_NAMES = ["General","Math","Logic","Code","Factual","Summary","Creative","Instruct"]

# ── Test 1: Teacher-forced perplexity by domain ───────────────────────────────
# This is the REAL perplexity — model sees context, predicts next token.
# Low = backbone knows the domain. High = backbone hasn't learned it yet.
print("[1/3] Teacher-forced perplexity by domain:")
print("      (lower = backbone already understands this domain)")
print("-"*65)

DOMAIN_TEXTS = {
    "Math":     "The derivative of x squared is 2x. To find the area under a curve, we integrate the function. If f(x) = x^2 then the integral from 0 to 3 is 9.",
    "Code":     "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
    "Factual":  "The speed of light in a vacuum is approximately 299,792,458 meters per second. Albert Einstein's theory of special relativity established that this is the universal speed limit.",
    "Logic":    "If all mammals are warm-blooded, and dolphins are mammals, then dolphins must be warm-blooded. This follows from the logical structure of a syllogism.",
    "Creative": "The old lighthouse stood at the edge of the world, its beam cutting through the eternal fog. Inside, the keeper wound the great clockwork mechanism, listening to the sea.",
    "Summary":  "In conclusion, the evidence strongly supports the hypothesis. The three experiments demonstrated consistent results, the statistical significance was confirmed, and the peer review validated the methodology.",
    "Instruct": "To install Python on Ubuntu: First, update the package list with sudo apt update. Then install Python 3 with sudo apt install python3. Verify with python3 --version.",
    "General":  "The weather today is sunny with a chance of rain in the afternoon. Temperatures will reach a high of 75 degrees Fahrenheit with light winds from the southwest.",
}

@torch.no_grad()
def domain_perplexity(text, max_len=200):
    ids = tok.encode(text)[:max_len]
    if len(ids) < 4: return float('inf')
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=DEVICE)
    y = torch.tensor(ids[1:],    dtype=torch.long, device=DEVICE)
    out    = model(x)
    logits = (out[0] if isinstance(out, tuple) else out)[0].float()
    logits[:, tok.vocab_size:] = -float('inf')
    loss   = torch.nn.functional.cross_entropy(logits, y)
    return math.exp(min(loss.item(), 20))   # cap at e^20 to avoid inf display

ppls = {}
for domain, text in DOMAIN_TEXTS.items():
    ppl = domain_perplexity(text)
    ppls[domain] = ppl
    quality = "✓ Strong"  if ppl < 20  else \
              "✓ Good"    if ppl < 100  else \
              "~ Fair"    if ppl < 500  else \
              "✗ Weak"
    bar = "█" * max(1, int(30 - min(ppl/20, 30)))
    print(f"  {domain:<10} PPL={ppl:>8.1f}  {quality:<10}  {bar}")

avg_ppl = sum(ppls.values()) / len(ppls)
best    = min(ppls, key=ppls.get)
weakest = max(ppls, key=ppls.get)
print(f"\n  Average PPL: {avg_ppl:.1f}")
print(f"  Strongest domain: {best} (PPL={ppls[best]:.1f})")
print(f"  Weakest domain:   {weakest} (PPL={ppls[weakest]:.1f})")
backbone_ready = avg_ppl < 500

# ── Test 2: Arm output divergence ─────────────────────────────────────────────
print("\n[2/3] Arm output divergence (higher = arms already separating):")
print("-"*65)

@torch.no_grad()
def arm_divergence(texts):
    arm_outputs = [[] for _ in range(8)]
    for text in texts:
        ids = tok.encode(text)[:64]
        x   = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        emb = model.embedding(x)
        h   = emb
        for layer in model.layers:
            h = layer(h)
        h_mean = h[0].float().mean(dim=0)  # [d_model] — pooled backbone repr
        for i, arm in enumerate(model.mimo_reasoning_blocks):
            out = arm(h)[0].float().mean(dim=0)  # [d_model]
            arm_outputs[i].append(out)
    # Average across texts, then compute pairwise cosine similarity
    arm_vecs = [torch.stack(v).mean(0) for v in arm_outputs]  # 8 × [d]
    sims = []
    for i in range(8):
        for j in range(i+1, 8):
            cos = torch.nn.functional.cosine_similarity(
                arm_vecs[i].unsqueeze(0), arm_vecs[j].unsqueeze(0)
            ).item()
            sims.append(cos)
    return arm_vecs, sims

test_sentences = [
    "The derivative of sin(x) is cos(x).",
    "def merge_sort(arr): return arr if len(arr) <= 1 else arr",
    "The French Revolution began in 1789.",
    "Once upon a time in a land far away.",
    "To summarize: the main point is efficiency.",
    "All humans are mortal. Socrates is human.",
    "Step 1: Open the terminal. Step 2: Type ls.",
    "The sky is blue because of Rayleigh scattering.",
]

arm_vecs, pairwise_sims = arm_divergence(test_sentences)
avg_sim    = sum(pairwise_sims) / len(pairwise_sims)
min_sim    = min(pairwise_sims)
arms_diverged = avg_sim < 0.90

print(f"  Pairwise cosine similarity between arms:")
print(f"  Average: {avg_sim:.4f}   Min: {min_sim:.4f}")
print(f"  (0 = completely different, 1 = identical clones)")
print()

# Per-arm uniqueness vs mean
mean_vec = torch.stack(arm_vecs).mean(0)
for i, (name, vec) in enumerate(zip(ARM_NAMES, arm_vecs)):
    diff = 1.0 - torch.nn.functional.cosine_similarity(
        vec.unsqueeze(0), mean_vec.unsqueeze(0)
    ).item()
    bar = "█" * int(diff * 200)
    status = "↑ most unique" if diff == max(
        1.0 - torch.nn.functional.cosine_similarity(v.unsqueeze(0), mean_vec.unsqueeze(0)).item()
        for v in arm_vecs) else ""
    print(f"  Arm {i} ({name:<8}): divergence={diff:.4f}  {bar} {status}")

divergence_pct = (1 - avg_sim) * 100
print(f"\n  Arms are {divergence_pct:.1f}% diverged from each other")

# ── Test 3: Router domain sensitivity ─────────────────────────────────────────
print("\n[3/3] Router domain sensitivity:")
print("      (does router weight differently for different input types?)")
print("-"*65)

ROUTER_TESTS = [
    ("Math problem",    "Solve: if x + 5 = 12, then x ="),
    ("Code task",       "Write a Python function to reverse a string:"),
    ("Factual query",   "What is the capital of Japan?"),
    ("Creative prompt", "Write a poem about the ocean:"),
    ("Logic puzzle",    "All cats are animals. Some animals are domestic. Therefore:"),
    ("Summary task",    "Summarize the following in one sentence:"),
]

@torch.no_grad()
def get_router_weights(text):
    ids = tok.encode(text)[:64]
    x   = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out = model(x)
    # Get gate logits from domain_router on last backbone hidden state
    emb = model.embedding(x)
    h   = emb
    for layer in model.layers:
        h = layer(h)
    gate_logits = model.domain_router(h[:, -1, :])   # [1, n_arms]
    weights     = torch.softmax(gate_logits.float(), dim=-1)[0].tolist()
    return weights

weight_matrix = []
for label, text in ROUTER_TESTS:
    w = get_router_weights(text)
    weight_matrix.append(w)
    top_i   = w.index(max(w))
    top_pct = max(w) * 100
    print(f"  {label:<20}: top arm = {ARM_NAMES[top_i]:<10} ({top_pct:.0f}%)")

# Measure variance in routing decisions
import statistics
per_arm_variance = []
for arm_idx in range(8):
    col = [row[arm_idx] for row in weight_matrix]
    per_arm_variance.append(statistics.variance(col) if len(col) > 1 else 0)
avg_routing_var = sum(per_arm_variance) / len(per_arm_variance)
router_sensitive = avg_routing_var > 0.001

print(f"\n  Routing variance across input types: {avg_routing_var:.6f}")
print(f"  {'✓ Router IS sensitive to input type' if router_sensitive else '✗ Router ignores input type (always same arm)'}")

# ── FINAL VERDICT ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  PHASE 2 TRANSFER VERDICT")
print("="*65)

v1 = backbone_ready
v2 = arms_diverged
v3 = router_sensitive

# Interpret
print(f"  {'✓' if v1 else '✗'} Backbone domain knowledge  (avg PPL={avg_ppl:.0f})")
print(f"  {'✓' if v2 else '~'} Arms showing divergence    ({divergence_pct:.1f}% apart, avg_sim={avg_sim:.3f})")
print(f"  {'✓' if v3 else '✗'} Router is input-sensitive  (var={avg_routing_var:.6f})")
print()

passed = sum([v1, v2, v3])
if passed == 3:
    print("  🟢 BACKBONE READY — Phase 2 will succeed")
    print("     Domain injection will have a solid foundation to build on.")
elif passed == 2:
    print("  🟡 BORDERLINE — Phase 2 will likely work, with some risk.")
    if not v1:
        print(f"     ⚠  Backbone PPL still high ({avg_ppl:.0f}). More Phase 1 = safer.")
    if not v2:
        print(f"     ⚠  Arms still very similar. Phase 2 may not differentiate them well.")
    if not v3:
        print(f"     ⚠  Router ignores context. Phase 2 domain seeding may be wasted.")
else:
    print("  🔴 NOT READY — Phase 2 will fail. More Phase 1 needed.")

print()
print("  RECOMMENDATION:")
if passed >= 2:
    print("  → You can move to Phase 2 now. Risk is low.")
    print("  → More Phase 1 steps (up to 15k) would improve arm divergence,")
    print("    but the backbone is functional enough for Phase 2 to work.")
else:
    print("  → Continue Phase 1. Target at least 5,000 more steps.")
print("="*65)
