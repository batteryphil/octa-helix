---
license: apache-2.0
language:
- en
tags:
- mamba
- state-space-model
- moe
- reasoning
- custom-architecture
---

# Mamba-Titan-1.4B-Reasoning

**Mamba-Titan-1.4B-Reasoning** is a scratch-built, custom State Space Model (SSM) fine-tuned specifically for step-by-step reasoning. It features a novel **8-Arm Mixture-of-Experts (MoE)** reasoning block, a **Blackboard IPC (Inter-Process Communication)** bus for expert cross-talk, and a **Concept Perceptron** for context stability.

The model was iteratively designed, diagnosed, and trained entirely on a single consumer RTX 3060.

## Architecture Highlights

This is not a standard Transformer. The architecture was explicitly engineered to separate fast semantic retrieval from deep logical reasoning.

*   **Frozen Backbone:** A standard Mamba sequence mixer (d_model=2048) handles token routing and grammar.
*   **8-Arm MIMO Reasoning MoE:** At layer 24, a low-rank bridge routes the residual stream to 8 specialized reasoning "arms" (e.g., Math, Logic, Factual, Instruct).
*   **Blackboard IPC (The Anchor):** The experts communicate via a 64-dimensional internal blackboard. Ablation tests show this blackboard is critical for preventing premature sequence termination; it acts as an "anchor" that holds the model in a reasoning state until computation is complete.
*   **Concept Perceptron (Thalamic Gate):** A dedicated scratchpad state that acts as a thalamic gate to stabilize the context vector across long sequences.
*   **Termination Vault:** The `</think>` token is geometrically isolated (0.0th percentile norm in the vocabulary space), resulting in near 100% deterministic termination of the reasoning loop without drift.

## Training Details
*   **Size:** 1.4B total parameters (455M Trainable during reasoning SFT)
*   **Training phases:** Curriculum learning (cold-start gating → logic unblocking → goldilocks calibration → targeted reasoning SFT). 
*   **Reasoning Format:** The model is strictly trained to output a `<think>`...`</think>` trace before answering.

## Performance & Benchmarks (SFT20)
Evaluated on an 80-prompt custom reasoning/factual benchmark with `repetition_penalty=1.2`:

*   **Overall Accuracy:** 66%
*   **Geography / Extractive QA:** 86%
*   **Format Adherence:** 100% termination rate. Median think trace length: 19 tokens.
*   **Yes/No & Logic:** 66%

### Known Limitations (Honest Assessment)
While the structural routing and reasoning framework is highly validated, the internal experts currently lack deep computational scale:
1.  **Math/Algebra (50%):** The router correctly identifies mathematical queries and delegates to the Math Arm. However, the 64-dim Math expert struggles with simultaneous multi-step arithmetic, sometimes hallucinating numbers inside the `<think>` trace before producing an answer.
2.  **Logic Syllogisms:** Without explicit context, the logic arm can occasionally hallucinate premises rather than applying strict deductive logic.

## How to Load & Use

Because of the custom architecture, you cannot load this using standard `AutoModelForCausalLM`. You must use the included `mamba3_titan_builder.py` file.

```python
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from mamba3_titan_builder import Mamba3Titan

# Load Tokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
tok.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
end_id = tok.convert_tokens_to_ids("</think>")

# Initialize Custom Architecture
model = Mamba3Titan(vocab_size=50304, d_model=2048, n_layers=48, mimo_paths=8)
model.resize_token_embeddings(50304)
model.set_phase('sft')

# Load Weights
ckpt = torch.load("phase_sft20_best.pt", map_location='cpu', weights_only=True)
model.load_state_dict(ckpt['model'], strict=False)
model = model.to(torch.bfloat16).cuda().eval()

# Generate
prompt = "If a box has 5 red balls and 3 blue balls, how many total balls are there?"
ids = tok.encode(f"User: {prompt}\nAssistant: <think>\n", return_tensors='pt').cuda()

for i in range(150):
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, _ = model(ids)
    t = torch.multinomial(F.softmax(logits[0, -1]/0.8, dim=-1), 1).item()
    if t == end_id: break
    ids = torch.cat([ids, torch.tensor([[t]]).cuda()], dim=-1)

# Continue generating final answer after </think>...
```
