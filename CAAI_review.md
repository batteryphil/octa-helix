# CAAI Runtime Governor — Architecture Review & Implementation Notes

**Reviewer:** Antigravity  
**Date:** 2026-06-08  
**Source:** `CAAI_RUNTIME_GOVERNOR_CONCEPT.txt`

---

## What This Is

A non-differentiable middleware shim that sits between the Mamba3 forward pass and
the token sampler. Its job: detect behavioral collapse in real-time and apply
targeted runtime interventions without touching model weights.

The core insight is sound: **SSM hidden state `h` is writable at inference time**,
which gives you a lever that transformers simply don't have. You can surgically
reset or decay a single arm's recurrent state without affecting any other arm.

---

## What Works Well

### Detection Metrics (Section 3)
All three collapse signals are well-chosen:

- **Router Entropy < 0.05 for 50+ tokens** — correct signal for arm lock-in. The
  50-token window is appropriately chosen; shorter windows would false-positive on
  deliberate focused reasoning.
- **Blackboard Variance** — elegant. The shared context vector averaging to mush
  is the exact symptom of "semantic dead zone" collapse. Variance is cheap to
  compute.
- **N-gram repetition** — table stakes, but necessary as a third independent
  signal. Using all three in conjunction (not OR, not AND) via `collapse_warning_level`
  accumulation is the right call.

### Intervention A — Router Dampening
Applying a temperature spike specifically to `route_logits` (not the output logits)
is clever. It forces arm diversity without degrading token quality, since the
output distribution is unaffected until the next routing decision. The -5.0 bias
penalty on the dominant arm's route logit is aggressive but appropriate for a
confirmed collapse event.

### Intervention C — Memory Purge
This is the most architecturally novel intervention. Selectively decaying `h` for
a stuck arm while preserving all other arms' state is only possible with SSM
architecture. A transformer has no per-head state to reset — the entire KV cache
would need to be invalidated. **This is a genuine architectural advantage of Mamba
that deserves more emphasis.**

### Exposure Bias Masking (the "Resistance Spring")
The decaying soft penalty on `</think>` is the right approach vs. a hard quota.
Hard quotas cause the "Hostage Situation" the doc correctly identifies. The linear
decay formula (`-15.0 + t * 0.1`) is a reasonable first pass.

---

## Identified Gaps & Issues

### 1. Duplicate Intervention Label (Critical Bug in Spec)
Both **Anchor Forcing** and the **Exposure Bias Masking** are labeled
"INTERVENTION B". Anchor Forcing should be **INTERVENTION B**, Exposure Bias
should be **INTERVENTION C**, and Memory Purge should be **INTERVENTION D**.
This needs to be resolved before any implementation — the pseudo-code only
implements A and C, leaving Anchor Forcing completely unimplemented.

### 2. Anchor Forcing Is Undefined (Section 4, INTERVENTION B)
> *"injects a hidden, zero-weight 'Grounding Anchor' token sequence into the
Mamba state"*

This mechanism is described but never specified:
- What is the grounding anchor? A fixed learned vector? A running mean of `h`
  across the first N tokens of the prompt?
- How does the Governor detect hallucination (the trigger for this intervention)
  without a separate verifier or retrieval system?
- "Zero-weight" is ambiguous — does this mean the anchor token contributes 0 to
  the logits, or 0 to the gradient (irrelevant at inference time)?

**Suggestion:** Either ground this in a specific mechanism (e.g., store the
`h` state after processing the factual part of the prompt, and use that as a
reset target when hallucination is detected via perplexity spike on known entities)
or remove it from the spec until it's defined.

### 3. Thresholds Are Untested Magic Numbers
- `router_entropy < 0.05` — why 0.05? What's the entropy distribution of healthy
  routing?
- `bb_variance < 0.1` — what are the typical variance ranges across arms?
- `-15.0` decay start / `0.1` per-token decay — gives a 150-token ramp to 0.
  Was this derived analytically or empirically?
- `collapse_warning_level > THRESHOLD` — `THRESHOLD` is never defined.

**Suggestion:** Before shipping, profile entropy and variance distributions across
1000+ normal generation sequences to establish empirical baselines. The thresholds
should be percentile-based (e.g., "bottom 5th percentile of observed router
entropy") not hand-tuned.

### 4. Telemetry Extraction Is Not Specified
The pseudo-code calls `model(token, state)` and gets back `telemetry`, but the
actual Titan implementation needs to expose this. Specifically:
- `telemetry.router_entropy` — requires softmax entropy computed from `route_logits`
  *before* argmax/sampling, then returned from the forward pass
- `telemetry.bb_variance` — requires the Blackboard module to expose its
  `shared_context` variance at each step
- `telemetry.inside_think_block` — requires token-level tracking of whether a
  `<think>` token has been seen without a matching `</think>`
- `telemetry.think_token_count` — running counter since last `<think>`

None of these are currently returned by a standard `model.forward()`. **This is
the most significant implementation gap** — the Governor cannot exist without
first instrumenting the Titan model to emit per-step telemetry.

### 5. Intervention A Reset Logic Is Incomplete
```python
model.router_temp = 1.5
collapse_warning_level = 0
```
The temperature is spiked to 1.5 but **never restored**. The model will continue
routing at temperature 1.5 for all subsequent tokens until the next collapse event
(which won't come because the warning level was reset to 0). Add:
```python
model.router_temp = 1.5
router_temp_ttl = 20  # tokens
```
Then in the per-token loop:
```python
if router_temp_ttl > 0:
    router_temp_ttl -= 1
else:
    model.router_temp = 1.0  # restore
```

### 6. No Hysteresis on Interventions
The Governor can trigger every time `collapse_warning_level` crosses threshold,
potentially oscillating: spike → variance → entropy drops → spike again → loop.
Add a cooldown period (e.g., no interventions for 30 tokens after any intervention)
to prevent intervention thrashing.

### 7. Multi-Arm Collapse Not Handled
The current spec handles single dominant-arm collapse (one arm taking >90% routing
weight). But "mush" collapse (multiple arms averaging to the same latent vector)
would show high router entropy (looks healthy!) but low Blackboard variance. The
current trigger logic `entropy < 0.05 AND variance < 0.1` would miss this case
because entropy is high. **Suggestion:** Add a separate trigger:
`entropy > 0.8 AND variance < 0.1` → "spread collapse" intervention (purge all
arms simultaneously rather than just the dominant one).

---

## Implementation Priority Order

If building this from scratch:

1. **Instrument Titan's forward pass** to emit per-step telemetry (entropy,
   variance, think-block state). Nothing else works without this.
2. **Implement the warning level accumulator** with hysteresis + cooldown.
3. **Implement Intervention A** (router dampening) with TTL restoration.
4. **Implement Intervention C** (memory purge) — the highest-value, most
   architecturally unique capability.
5. **Calibrate thresholds** empirically from production distributions.
6. **Define and implement Anchor Forcing** — or drop it from v1 scope.
7. **Add spread collapse detection** (the missed mush case).

---

## One Broader Observation

The document frames the Governor as a Mamba-specific solution, which is accurate.
But it's worth noting that you're currently running **Mistral-7B** as the active
inference engine, which is a transformer. The Governor as specified is entirely
inapplicable to Mistral — there's no Blackboard, no arm router, no persistent `h`
state. If Helix migrates back to a Mamba-based backbone (which would be the right
long-term direction for an always-on daemon due to the constant VRAM footprint),
the Governor becomes extremely relevant. For now, the Mistral deployment needs a
different collapse-detection strategy: conversation-level repetition scoring and
context window rotation are the appropriate equivalents.

---

*Review complete. File: `CAAI_review.md`*
