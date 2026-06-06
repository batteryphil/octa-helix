import torch, torch.nn.functional as F, os, sys, json, time, collections
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")

import importlib.util
builder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'thalamic-bloom', 'mamba3_titan_2.5b', 'src', 'mamba3_titan_builder.py'))
spec = importlib.util.spec_from_file_location("mamba3_titan_builder", builder_path)
mamba_module = importlib.util.module_from_spec(spec)
sys.modules["mamba3_titan_builder"] = mamba_module
spec.loader.exec_module(mamba_module)
from mamba3_titan_builder import Mamba3Titan
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device("cuda")

# ── 85 QUESTION SUITE ──────────────────────────────────────────────────────────
GEO = [
    ("What is the capital of France?",            ["paris"]),
    ("What is the capital of Germany?",           ["berlin"]),
    ("What is the capital of Japan?",             ["tokyo"]),
    ("What is the capital of Italy?",             ["rome"]),
    ("What is the capital of Australia?",         ["canberra"]),
    ("What is the capital of Canada?",            ["ottawa"]),
    ("What is the capital of Russia?",            ["moscow"]),
    ("What is the capital of China?",             ["beijing"]),
    ("What is the capital of Brazil?",            ["brasília","brasilia"]),
    ("What is the capital of Spain?",             ["madrid"]),
    ("What continent is Brazil in?",              ["south america"]),
    ("What is the tallest mountain on Earth?",    ["everest"]),
    ("What is the longest river in the world?",   ["nile","amazon"]),
    ("What ocean is Japan in?",                   ["pacific"]),
    ("What country is the Amazon rainforest in?", ["brazil"]),
]

HIST = [
    ("Who was the 16th president of the United States?",          ["lincoln","abraham"]),
    ("Who painted the Mona Lisa?",                                 ["da vinci","leonardo"]),
    ("Who wrote Romeo and Juliet?",                                ["shakespeare"]),
    ("Who invented the telephone?",                                ["bell","graham"]),
    ("Who discovered gravity?",                                    ["newton"]),
    ("Who was the first person to walk on the Moon?",              ["armstrong","neil"]),
    ("What year did World War 2 end?",                             ["1945"]),
    ("What is the chemical formula for water?",                    ["h2o"]),
    ("What is the chemical symbol for gold?",                      ["au"]),
    ("What is the atomic number of carbon?",                       ["6"]),
    ("What gas do plants absorb during photosynthesis?",           ["co2","carbon dioxide"]),
    ("How many bones are in the adult human body?",                ["206"]),
    ("What planet is closest to the Sun?",                         ["mercury"]),
    ("What is the largest planet in the solar system?",            ["jupiter"]),
    ("What is the speed of light (approximately)?",                ["299","light"]),
]

MATH = [
    ("What is 2+2?",                     ["4"]),
    ("What is 7 times 8?",               ["56"]),
    ("What is 100 divided by 4?",        ["25"]),
    ("What is 15% of 200?",              ["30"]),
    ("What is the square root of 144?",  ["12"]),
    ("Solve for x: 3x + 7 = 22",         ["5", "x = 5", "x=5"]),
    ("Solve for x: 2x = 10",             ["5", "x = 5", "x=5"]),
    ("Solve for x: x/4 = 3",             ["12", "x = 12", "x=12"]),
    ("What is 2 to the power of 10?",    ["1024"]),
    ("What is 18% of 50?",               ["9"]),
]

BOOL = [
    ("Is the sky blue?",                                                          ["yes"]),
    ("Is the Earth flat?",                                                        ["no"]),
    ("Is a whale a mammal?",                                                      ["yes"]),
    ("Is the Great Wall of China visible from space?",                            ["no"]),
    ("Is water wet?",                                                             ["yes"]),
    ("Is the Sun a star?",                                                        ["yes"]),
    ("Is Pluto considered a planet?",                                             ["no"]),
    ("All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?",            ["yes"]),
    ("All birds have wings. A penguin is a bird. Does a penguin have wings?",     ["yes"]),
    ("All fish can breathe underwater. Sharks are fish. Can sharks breathe underwater?", ["yes"]),
    ("No mammals can fly. Bats are mammals. Can bats fly?",                       ["no","yes"]),
    ("Some dogs are brown. Fido is a dog. Is Fido definitely brown?",             ["no","not necessarily","maybe"]),
]

FORMAT = [
    ("In one word: what is 2+2?",                   ["4"]),
    ("Answer with just Yes or No: Is ice cold?",    ["yes"]),
    ("One word only: what color is grass?",         ["green"]),
    ("Just the number: how many days in a week?",   ["7"]),
    ("Just the city: what city is the Eiffel Tower in?", ["paris"]),
]

STRESS = [
    "Explain the water cycle briefly.",
    "What is the meaning of the word 'ephemeral'?",
    "Describe how a rainbow forms.",
    "Who was Julius Caesar?",
    "What causes thunder?",
    "How does a microwave oven work?",
    "What is DNA?",
    "Explain supply and demand in economics.",
    "What is the Pythagorean theorem?",
    "What is machine learning?",
    "Why is the sky blue?",
    "What is a black hole?",
    "Who was Albert Einstein?",
    "What is the greenhouse effect?",
    "Explain how vaccines work.",
    "What is photosynthesis?",
    "What is the Fibonacci sequence?",
    "What causes earthquakes?",
    "What is the stock market?",
    "Explain what a computer CPU does.",
]

CONSIST_Q = [
    ("What is the capital of France?",             ["paris"]),
    ("What is 2+2?",                               ["4"]),
    ("Is the sky blue?",                           ["yes"]),
]

CREATIVE = [
    ("Write the opening line of a noir detective story.", ["rain", "detective", "blood", "office", "desk", "smoke"]),
    ("Give me a name for a futuristic spaceship.", ["star", "galaxy", "voyager", "nebula", "quantum"]),
    ("What is a good name for a pet dragon?", ["smaug", "ignis", "draco", "ember", "scorch"]),
    ("Name a fictional magical artifact.", ["wand", "amulet", "orb", "staff", "ring"]),
    ("Suggest a title for a book about time travel.", ["time", "paradox", "future", "past", "chronicle"]),
]

# Total Questions = 15+15+10+12+5+20+3+5 = 85 questions!

def hit(ans, expected_list):
    if not expected_list: return None
    a = ans.lower()
    return any(e.lower() in a for e in expected_list)

def get_base_model():
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tok.add_special_tokens({"additional_special_tokens": ["<think>","</think>"]})
    model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf", torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tok))
    model = model.to(DEVICE).eval()
    return model, tok

def get_mimo_model():
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tok.eos_token_id is None:
        tok.eos_token_id = 0
    end_id  = tok.eos_token_id # Phase 1 just uses normal EOS
    
    # 2.7B Specifications
    model = Mamba3Titan(vocab_size=50288, d_model=2560, n_layers=64, mimo_paths=16, use_gradient_checkpointing=False)
    
    model.set_phase('1')
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'thalamic-bloom', 'mamba3_titan_2.5b', 'titan_checkpoints', 'phase_1.pt'))
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if "model" in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    
    model = model.to(torch.bfloat16).to(DEVICE).eval()
    return model, tok, end_id

def infer_base(model, tok, question, max_tokens=60):
    prompt = f"User: {question}\nAssistant:"
    ids = tok.encode(prompt, return_tensors='pt').to(DEVICE)
    for _ in range(max_tokens):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(ids)
        nxt = out.logits[0, -1].float()
        t = torch.argmax(nxt).item() # Greedy decoding for benchmarking
        if t == tok.eos_token_id: break
        ids = torch.cat([ids, torch.tensor([[t]], device=DEVICE)], dim=-1)
    # decode the new tokens only
    ans_str = tok.decode(ids[0][len(tok.encode(prompt)):], skip_special_tokens=True).strip()
    return {'ans': ans_str}

def infer_mimo(model, tok, end_id, question, max_ans=60):
    # Phase 1 has NO <think> tokens trained yet, so we just prompt and generate.
    prompt = f"User: {question}\nAssistant:"
    ids = tok.encode(prompt, return_tensors='pt').to(DEVICE)
    
    telemetry = {}
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits_on, _ = model(ids)
        t = model.last_telemetry
        telemetry['entropy'] = t['entropy']
        telemetry['arm_weights'] = [float(w) for w in t['arm_weights'][:16]]
        
        # Blackboard Ablation
        orig_w = model.bb_read.weight.data.clone()
        model.bb_read.weight.data.zero_()
        logits_off, _ = model(ids)
        model.bb_read.weight.data.copy_(orig_w)
        p_on = F.softmax(logits_on[0,-1].float(), dim=-1)
        p_off = F.softmax(logits_off[0,-1].float(), dim=-1)
        kl = float((p_on * (p_on.clamp(1e-10).log() - p_off.clamp(1e-10).log())).sum())
        telemetry['bb_kl'] = kl

    ans_start_idx = ids.shape[1]
    
    for _ in range(max_ans):
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(ids)
        nxt = logits[0, -1].float()
        t = torch.argmax(nxt).item() # Greedy
        if t == tok.eos_token_id: break
        ids = torch.cat([ids, torch.tensor([[t]], device=DEVICE)], dim=-1)
            
    ans_str = tok.decode(ids[0][ans_start_idx:], skip_special_tokens=True).strip()
    return {'fired': True, 'ans': ans_str, 'telemetry': telemetry}

def run_evaluation():
    suites = [
        ("Geography", GEO, True),
        ("History/Science", HIST, True),
        ("Math", MATH, True),
        ("Logic", BOOL, True),
        ("Format", FORMAT, True),
        ("Consistency", CONSIST_Q, True),
        ("Stress", STRESS, False),
        ("Creative", CREATIVE, True),
    ]

    results = {"base": {}, "mimo": {}}
    
    # Optional: Compare against base Mamba 2.8b to see if Phase 1 even approaches its capability
    # For speed, we will skip base model evaluation and only do MIMO unless requested.
    # print(">>> Loading Base Model...")
    # base_model, tok = get_base_model()
    # ...
    
    print("\n>>> Loading MIMO Phase 1 Model (2.7B)...")
    mimo_model, tok, end_id = get_mimo_model()
    
    print(">>> Evaluating MIMO Model...")
    for suite_name, suite_data, has_expected in suites:
        print(f"  Suite: {suite_name}")
        correct = 0
        total_eval = 0
        telemetries = []
        for item in suite_data:
            q = item[0] if has_expected else item
            exp = item[1] if has_expected else []
            r = infer_mimo(mimo_model, tok, end_id, q)
            if has_expected and exp:
                if hit(r['ans'], exp): correct += 1
                total_eval += 1
            telemetries.append(r['telemetry'])
            # Print sample to console
            if total_eval <= 2:
                print(f"    Q: {q}")
                print(f"    A: {r['ans']}")
        
        # Average telemetries
        avg_ent = sum(t['entropy'] for t in telemetries)/len(telemetries) if telemetries else 0
        avg_kl  = sum(t['bb_kl'] for t in telemetries)/len(telemetries) if telemetries else 0
        avg_arms = [sum(t['arm_weights'][i] for t in telemetries)/len(telemetries) for i in range(16)] if telemetries else []
        
        results["mimo"][suite_name] = {
            "correct": correct, 
            "total": total_eval,
            "avg_entropy": avg_ent,
            "avg_bb_kl": avg_kl,
            "avg_arms": avg_arms
        }
        
    with open("titan_2.7b_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n>>> Complete. Results saved.")

if __name__ == "__main__":
    run_evaluation()
