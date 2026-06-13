"""
Test Mistral-7B-Instruct-v0.3 in 4-bit with native tool calling.
Mistral v0.3 emits: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
"""
import json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID  = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR = "/home/phil/.gemini/antigravity/scratch/analysis_project/hf_cache"

print(f"Loading {MODEL_ID} in 4-bit NF4...")

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto", cache_dir=CACHE_DIR
)
model.eval()

vram = torch.cuda.memory_allocated() / 1e9
print(f"✅ Loaded  VRAM: {vram:.2f} GB\n")

# ── Define tools in Mistral's format ─────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '17 * 24'"}
                },
                "required": ["expression"]
            }
        }
    }
]

def fake_tool_executor(name, args):
    """Simulate tool responses."""
    if name == "web_search":
        return f"Search results for '{args['query']}': [1] Mamba SSM achieves O(n) complexity vs O(n²) for transformers. [2] State space models use selective scanning to compress long sequences efficiently."
    if name == "calculator":
        try:
            return str(eval(args.get("expression", "0")))
        except:
            return "Error evaluating expression"
    return f"Tool '{name}' not found"

def parse_tool_calls(text):
    """Extract [TOOL_CALLS] JSON from Mistral output."""
    m = re.search(r'\[TOOL_CALLS\]\s*(\[.*?\])', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass
    return None

def run_turn(messages, user_msg):
    """Single turn: generate, check for tool call, execute, re-generate."""
    messages.append({"role": "user", "content": user_msg})

    prompt = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=200, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)

    raw = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    calls = parse_tool_calls(raw)

    if calls:
        print(f"  🔧 Tool call detected: {calls}")
        # Mistral requires 9-char alphanumeric IDs
        import random, string
        for call in calls:
            call["id"] = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
        messages.append({"role": "assistant", "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments", {}))}}
            for c in calls
        ]})
        # Execute each tool and feed result back
        for call in calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            result = fake_tool_executor(name, args)
            print(f"  📦 Tool result: {result[:80]}...")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

        # Re-generate with tool results
        prompt2 = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        ids2 = tokenizer(prompt2, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out2 = model.generate(ids2, max_new_tokens=200, do_sample=True,
                                  temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        final = tokenizer.decode(out2[0][ids2.shape[1]:], skip_special_tokens=True).strip()
        messages.append({"role": "assistant", "content": final})
        return final
    else:
        clean = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        messages.append({"role": "assistant", "content": clean})
        return clean

# ── Tests ─────────────────────────────────────────────────────────────
messages = []
tests = [
    "Search the web for how Mamba state space models differ from transformers.",
    "What is 347 multiplied by 89? Use the calculator.",
    "What is the capital of Japan?",   # Should NOT call a tool
]

for q in tests:
    print(f"\nQ: {q}")
    ans = run_turn(messages, q)
    print(f"A: {ans[:300]}")
    print("-" * 60)

print(f"\nPeak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
