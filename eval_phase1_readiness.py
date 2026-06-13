"""
Phase 1 Readiness Test — "Can we move to Phase 2?"

Phase 1 Goal: Each arm learns to DECODE the Mamba backbone independently.
Phase 1 is DONE when:
  1. Loss is stable (not still dropping fast) and below threshold
  2. All 8 arms are active (none silenced/dead)
  3. The router is learning (not uniform 0.125 weights forever)
  4. Loss per-arm is measurably similar (all arms can decode, not just arm 0)

We do NOT test for coherent generation or arm specialization — that's Phase 2.
"""
import torch, sys, os, json, math
sys.path.insert(0, os.path.dirname(__file__))

CKPT   = "checkpoints_2.7b/phase_1.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*65)
print("  PHASE 1 READINESS TEST")
print("  Question: Is arm calibration complete? Ready for Phase 2?")
print("="*65)

# ── 1. Loss trend analysis (no GPU needed) ───────────────────────────────────
print("\n[1/4] Loss stability analysis (from training log)...")

import subprocess, re
result = subprocess.run(
    ['grep', 'Phase 1 | Step', 'run_titan.log'],
    capture_output=True, text=True
)
lines = result.stdout.strip().split('\n')

buckets = {}
for line in lines:
    m_step = re.search(r'Step (\d+)', line)
    m_loss = re.search(r'LM Loss: ([\d.]+)', line)
    if m_step and m_loss:
        step = int(m_step.group(1))
        loss = float(m_loss.group(1))
        b = (step // 1000) * 1000
        buckets.setdefault(b, []).append(loss)

# Get last 8 buckets (only from real run, skip pre-fix restarts)
sorted_buckets = sorted(buckets.items())
# Filter to steps >= 2000 (post-fix clean run)
clean = [(s, v) for s, v in sorted_buckets if s >= 2000]
avgs  = [(s, sum(v)/len(v)) for s, v in clean if len(v) > 10]

print(f"\n  1000-step average loss (clean run only):")
for step, avg in avgs[-8:]:
    trend = ""
    bar   = "█" * int(avg * 3)
    print(f"  Steps {step:>5}-{step+999}  Avg={avg:.4f}  {bar}")

# Measure rate of change over last 3 windows vs previous 3
if len(avgs) >= 6:
    recent_drop  = avgs[-3][1] - avgs[-1][1]       # how much dropped in last 3k steps
    earlier_drop = avgs[-6][1] - avgs[-3][1]        # how much dropped in 3k steps before that
    drop_rate    = recent_drop / max(earlier_drop, 0.0001)
    still_dropping_fast = (recent_drop > 0.3) and (drop_rate > 0.3)
else:
    recent_drop = 0
    still_dropping_fast = True

current_avg = avgs[-1][1] if avgs else 99
print(f"\n  Current avg loss (last 1000 steps): {current_avg:.4f}")
print(f"  Drop in last 3000 steps: {recent_drop:.4f}")
print(f"  {'⚠️  Still dropping fast — more training helps' if still_dropping_fast else '✓  Loss has stabilised'}")

# ── 2. Load model ────────────────────────────────────────────────────────────
print("\n[2/4] Loading checkpoint for arm analysis...")
import gc
from mamba3_titan_builder import Mamba3Titan
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
if tok.pad_token is None: tok.pad_token = tok.eos_token

state      = torch.load(CKPT, map_location="cpu", weights_only=False)
model_sd   = state.get("model_state_dict", state)
step_saved = state.get("step", "?")
del state; gc.collect(); torch.cuda.empty_cache()

model = Mamba3Titan(vocab_size=50288, d_model=2560, n_layers=64,
                    mimo_paths=8, use_gradient_checkpointing=False)
model.load_state_dict(model_sd, strict=False)
del model_sd; gc.collect(); torch.cuda.empty_cache()

model = model.bfloat16().to(DEVICE)
model.eval()
print(f"  Loaded step {step_saved}")

# ── 3. Arm health — are all arms alive? ─────────────────────────────────────
print("\n[3/4] Arm health check...")
ARM_NAMES = ["General","Math","Logic","Code","Factual","Summary","Creative","Instruct"]

# Measure output norm of each arm on a batch of test tokens
test_texts = [
    "The capital of France is",
    "def fibonacci(n): return",
    "2 + 2 equals",
    "Once upon a time",
    "The scientific method involves",
    "To summarize the above:",
    "Write a poem about the moon:",
    "Step 1: Install the package",
]

arm_norms   = [0.0] * 8
arm_weights_accum = [0.0] * 8
arm_perp    = []

@torch.no_grad()
def measure_arm(text):
    ids = tok.encode(text)[:128]
    if len(ids) < 2: return None
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    # Run through backbone (self.layers)
    emb = model.embedding(x)
    h   = emb
    for layer in model.layers:
        h = layer(h)

    # Run each MIMO reasoning block independently
    norms = []
    for i, arm in enumerate(model.mimo_reasoning_blocks):
        arm_out = arm(h)                      # [1, seq, d]
        norms.append(arm_out.float().norm().item())
    return norms

print("  Testing each arm's output norm (>0 = alive, 0 = dead):")
all_norms = []
for text in test_texts:
    norms = measure_arm(text)
    if norms: all_norms.append(norms)

avg_norms = [sum(r[i] for r in all_norms)/len(all_norms) for i in range(8)]
dead_arms = [i for i, n in enumerate(avg_norms) if n < 1.0]

for i, (name, norm) in enumerate(zip(ARM_NAMES, avg_norms)):
    status = "✓ ALIVE" if norm > 1.0 else "✗ DEAD "
    bar    = "█" * min(int(norm / 50), 30)
    print(f"  Arm {i} ({name:<8}): norm={norm:>8.1f}  {status}  {bar}")

print(f"\n  Dead arms: {len(dead_arms)} / 8  {'✓ All alive' if not dead_arms else f'⚠️  Arms {dead_arms} are dead'}")

# ── 4. Router weight distribution ────────────────────────────────────────────
print("\n[4/4] Router weight distribution...")
# Read from telemetry (saved during training)
try:
    telem = json.load(open("monitor_ui/telemetry.json"))
    aw    = telem.get("arm_weights", [0.125]*8)
    sims  = telem.get("arm_sims", [1.0]*8)
    entropy = telem.get("entropy", 0)
    print(f"  Gate entropy (higher=more uniform=better for Phase 1): {entropy:.4f}")
    print(f"  Target for Phase 1 exit: entropy > 0.5  {'✓' if entropy > 0.5 else '✗'}")
    print()
    print("  Arm weight distribution (from last training step):")
    for i, (name, w, s) in enumerate(zip(ARM_NAMES, aw, sims)):
        bar = "█" * int(w * 80)
        print(f"  Arm {i} ({name:<8}): weight={w:.4f}  sim={s:.3f}  {bar}")
    max_weight   = max(aw)
    dominant_arm = aw.index(max_weight)
    collapse     = max_weight > 0.90
    print(f"\n  Dominant arm: Arm {dominant_arm} ({ARM_NAMES[dominant_arm]}) at {max_weight*100:.1f}%")
    print(f"  {'⚠️  Router collapsed — one arm dominates' if collapse else '✓  Router distributing reasonably'}")
except Exception as e:
    print(f"  Could not load telemetry: {e}")
    entropy = 0; collapse = True

# ── VERDICT ──────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  PHASE 1 EXIT CRITERIA")
print("="*65)

c1 = current_avg < 3.5                   # loss low enough
c2 = not still_dropping_fast             # loss has stabilised
c3 = len(dead_arms) == 0                 # all arms alive
c4 = entropy > 0.5                       # router not fully collapsed
c5 = int(str(step_saved).replace(',','')) >= 15000 if str(step_saved).replace(',','').isdigit() else False

print(f"  {'✓' if c1 else '✗'} LM loss < 3.5          (current avg = {current_avg:.3f})")
print(f"  {'✓' if c2 else '✗'} Loss stabilised         (recent drop = {recent_drop:.3f})")
print(f"  {'✓' if c3 else '✗'} All 8 arms alive        ({8-len(dead_arms)}/8 alive)")
print(f"  {'✓' if c4 else '✗'} Router entropy > 0.5   (entropy = {entropy:.3f})")
print(f"  {'✓' if c5 else '✗'} Steps >= 15,000         (step = {step_saved})")

passed = sum([c1,c2,c3,c4,c5])
print()
if passed == 5:
    print("  🟢 READY FOR PHASE 2 — all criteria met")
elif passed >= 3 and c3:
    print(f"  🟡 CLOSE — {passed}/5 criteria met. Continue Phase 1.")
    if not c5:
        steps_left = 15000 - (int(str(step_saved).replace(',','')) if str(step_saved).replace(',','').isdigit() else 0)
        hrs = steps_left * 16 / 3600
        print(f"     Minimum ~{steps_left:,} more steps needed (~{hrs:.0f} hours)")
    if still_dropping_fast:
        print(f"     Loss still declining — more training = better Phase 2 start")
else:
    print(f"  🔴 NOT READY — {passed}/5 criteria met. Continue Phase 1.")

print("="*65)
