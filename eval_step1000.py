"""
Quick sanity test for Titan at step ~1000.
Tests: language coherence, instruction following, self-reference, reasoning.
Loads checkpoint directly — no DeepSpeed needed for inference.
"""
import torch, sys, os, time
sys.path.insert(0, '/home/phil/.gemini/antigravity/scratch/analysis_project')

# Force CPU — GPU VRAM is reserved by desktop processes
# Model is 2.7B @ bfloat16 = ~5.4GB, needs clean VRAM; use CPU for eval
os.environ["CUDA_VISIBLE_DEVICES"] = ""
CKPT = "checkpoints_2.7b/phase_1.pt"
DEVICE = torch.device("cpu")


print("="*60)
print("TITAN STEP-1000 EVALUATION")
print("="*60)

# ── Load ──────────────────────────────────────────────────────────
print("\n[1/3] Loading checkpoint...")
t0 = time.time()
try:
    from titan_inference import TitanInference
    engine = TitanInference(checkpoint=CKPT)
    engine.load()
    print(f"      Loaded in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"      FAILED: {e}")
    sys.exit(1)

# ── Prompts ───────────────────────────────────────────────────────
TESTS = [
    ("Basic language",
     "The sky is blue because"),

    ("Simple instruction",
     "List three colors:\n1."),

    ("Self-reference",
     "I am an AI language model. My purpose is"),

    ("Simple reasoning",
     "If a cat has 4 legs and a spider has 8 legs, together they have"),

    ("World knowledge",
     "The capital of France is"),

    ("Continuation quality",
     "Once upon a time in a land far away, there lived a"),
]

print("\n[2/3] Running inference tests...\n")
results = []

# Load tokenizer + model directly on CPU (bypasses titan_inference.py CUDA assumptions)
from transformers import AutoTokenizer
from mamba3_titan_builder import Mamba3Titan
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tokenizer.pad_token = tokenizer.eos_token

model = Mamba3Titan(vocab_size=50288, d_model=2560, n_layers=64, mimo_paths=8,
                    use_gradient_checkpointing=False)
model.set_phase("sft")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"], strict=False)
model = model.float()  # cast bfloat16 → float32 for CPU stability
model.eval()
print(f"  Model on CPU | step={ckpt['step']}")

@torch.no_grad()
def quick_generate(prompt, max_new=40, temp=0.8):
    ids = tokenizer.encode(prompt, return_tensors="pt")
    for _ in range(max_new):
        with torch.no_grad():
            logits, _ = model(ids, loop_idx=0)
        next_logits = logits[0, -1, :] / temp
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    new_ids = ids[0, ids.shape[1] - max_new:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True)

for label, prompt in TESTS:
    try:
        t0 = time.time()
        output = quick_generate(prompt)
        elapsed = time.time() - t0
        results.append((label, prompt, output, "cpu", 0, True))
        print(f"  [{label}]")
        print(f"    PROMPT : {prompt[:60]}")
        print(f"    OUTPUT : {output[:120]}")
        print(f"    TIME   : {elapsed:.1f}s")
        print()
    except Exception as e:
        results.append((label, prompt, str(e), "?", 0, False))
        print(f"  [{label}] ERROR: {e}\n")


# ── Summary ───────────────────────────────────────────────────────
print("\n[3/3] Assessment\n")
print("-"*60)

# Score heuristics
def score(output):
    if not output or len(output.strip()) < 5:
        return 0, "empty"
    words = output.split()
    if len(words) < 3:
        return 1, "very short"
    # gibberish check: ratio of real-looking words
    real = sum(1 for w in words if w.isalpha() and len(w) > 1)
    ratio = real / max(len(words), 1)
    if ratio < 0.3:
        return 1, f"mostly gibberish ({ratio:.0%} real words)"
    if ratio < 0.6:
        return 2, f"partial coherence ({ratio:.0%} real words)"
    return 3, f"coherent ({ratio:.0%} real words)"

total = 0
for label, prompt, output, arm, pct, ok in results:
    if not ok:
        print(f"  ✗ {label}: CRASHED")
        continue
    s, reason = score(output)
    total += s
    icon = "✓" if s == 3 else ("△" if s == 2 else "✗")
    print(f"  {icon} {label}: {reason}")

max_score = len(TESTS) * 3
pct_score = total / max_score * 100
print(f"\n  Score: {total}/{max_score} ({pct_score:.0f}%)")

if pct_score >= 70:
    verdict = "HEALTHY — model is developing coherent language at step 1000."
elif pct_score >= 40:
    verdict = "EARLY STAGE — partial coherence, expected at step 1000 of 40,000."
else:
    verdict = "CONCERNING — below expected coherence. Review arm entropy."

print(f"\n  Verdict: {verdict}")
print("\n" + "="*60)
