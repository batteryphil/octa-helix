"""
sft20_reasoning.py — Algebra + Logic + Extractive QA
Three targeted fixes for complex prompt failures.
"""
import torch, torch.nn.functional as F, os, time, signal, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
from mamba3_titan_builder import Mamba3Titan
from transformers import AutoTokenizer
try:
    from huggingface_hub import login; login(token="HF_TOKEN_REDACTED",add_to_git_credential=False)
except: pass
try:
    import pynvml; pynvml.nvmlInit(); _h=pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_temp=lambda: pynvml.nvmlDeviceGetTemperature(_h,pynvml.NVML_TEMPERATURE_GPU)
except: gpu_temp=lambda: None

LOAD_FROM="checkpoints_2.7b/phase_sft15_factual.pt"
SAVE_BEST="checkpoints_2.7b/phase_sft20_best.pt"
SAVE_DONE="checkpoints_2.7b/phase_sft20_done.pt"
LOG_PATH="sft20.log"; TELEM="monitor_ui/telemetry.json"
TARGET_STEPS=3000; SEQ_LEN=320
LR_ARMS=2e-6; LR_ROUTER=2e-5; LR_GATES=3e-5; LR_HEAD=8e-7
ANS_WEIGHT=10.0; ANS_WINDOW=40; THINK_MIN=25; CLIP=0.6
PROBE_EVERY=300; SAVE_EVERY=500

_shutdown=False
def _sig(s,f): global _shutdown; _shutdown=True
signal.signal(signal.SIGTERM,_sig); signal.signal(signal.SIGINT,_sig)

# ── CATEGORY 1: ALGEBRA ───────────────────────────────────────────────────────
ALGEBRA=[
    ("Solve for x: 3x+7=22",
     "Subtract 7 from both sides: 3x = 22-7 = 15. Divide both sides by 3: x = 15/3 = 5.",
     "x = 5"),
    ("Solve for x: 2x=10",
     "Divide both sides by 2: x = 10/2 = 5.",
     "x = 5"),
    ("Solve for x: x/4=3",
     "Multiply both sides by 4: x = 3*4 = 12.",
     "x = 12"),
    ("Solve for x: 5x-3=22",
     "Add 3 to both sides: 5x = 25. Divide by 5: x = 5.",
     "x = 5"),
    ("Solve for x: x+15=30",
     "Subtract 15 from both sides: x = 30-15 = 15.",
     "x = 15"),
    ("Solve for x: 4x=20",
     "Divide both sides by 4: x = 20/4 = 5.",
     "x = 5"),
    ("Solve for x: x/3=7",
     "Multiply both sides by 3: x = 7*3 = 21.",
     "x = 21"),
    ("Solve for x: 2x+4=16",
     "Subtract 4: 2x = 12. Divide by 2: x = 6.",
     "x = 6"),
    ("Solve for x: 3x-9=0",
     "Add 9: 3x = 9. Divide by 3: x = 3.",
     "x = 3"),
    ("Solve for x: 6x+2=14",
     "Subtract 2: 6x = 12. Divide by 6: x = 2.",
     "x = 2"),
    ("What is 15% of 200?",
     "15% means 15/100. So 15/100 * 200 = 15 * 2 = 30.",
     "30"),
    ("What is 18% of 50?",
     "18% of 50: 0.18 * 50 = 9.",
     "9"),
    ("What is 2 to the power of 10?",
     "2^10: 2,4,8,16,32,64,128,256,512,1024. So 2^10 = 1024.",
     "1024"),
    ("What is 2 to the power of 8?",
     "2^8: 2,4,8,16,32,64,128,256. So 2^8 = 256.",
     "256"),
    ("A train travels 60mph for 3 hours. How far does it travel?",
     "Distance = speed * time = 60 * 3 = 180 miles.",
     "180 miles"),
    ("If a car travels 300 miles in 5 hours, what is its average speed?",
     "Speed = distance / time = 300 / 5 = 60 mph.",
     "60 mph"),
    ("What is 25% of 80?",
     "25% = 1/4. So 80/4 = 20.",
     "20"),
    ("What is 7 squared?",
     "7 squared means 7*7 = 49.",
     "49"),
    ("Solve for x: x-8=12",
     "Add 8 to both sides: x = 12+8 = 20.",
     "x = 20"),
    ("What is the perimeter of a rectangle 5m wide and 8m long?",
     "Perimeter = 2*(length+width) = 2*(8+5) = 2*13 = 26 meters.",
     "26 meters"),
]

# ── CATEGORY 2: LOGIC / SYLLOGISMS ────────────────────────────────────────────
LOGIC=[
    ("All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?",
     "Premise 1: All cats are mammals. Premise 2: Whiskers is a cat. By modus ponens: Whiskers is a mammal.",
     "Yes"),
    ("All birds have wings. A penguin is a bird. Does a penguin have wings?",
     "Premise 1: All birds have wings. Premise 2: A penguin is a bird. Therefore: A penguin has wings.",
     "Yes"),
    ("No fish can breathe air. Sharks are fish. Can sharks breathe air?",
     "Premise 1: No fish can breathe air. Premise 2: Sharks are fish. Therefore: Sharks cannot breathe air.",
     "No"),
    ("All squares are rectangles. All rectangles have 4 sides. Does a square have 4 sides?",
     "Squares are rectangles. Rectangles have 4 sides. Therefore squares have 4 sides.",
     "Yes"),
    ("If it rains, the ground gets wet. It is raining. Is the ground wet?",
     "Premise: If rain then wet ground. Fact: It is raining. By modus ponens: the ground is wet.",
     "Yes"),
    ("All humans are mortal. Socrates is a human. Is Socrates mortal?",
     "All humans are mortal. Socrates is a human. Therefore Socrates is mortal.",
     "Yes"),
    ("No mammals can fly. Bats are mammals. Can bats fly?",
     "The premise says no mammals can fly, but this is actually false — bats can fly. However, given the premises as stated: No mammals fly, bats are mammals, so bats cannot fly.",
     "No, given the premises"),
    ("All Blorbians are Frumps. Zix is a Blorbian. Is Zix a Frump?",
     "Premise: All Blorbians are Frumps. Zix is a Blorbian. Therefore Zix is a Frump.",
     "Yes"),
    ("Some dogs are brown. Fido is a dog. Is Fido definitely brown?",
     "Some dogs are brown — not all. Fido is a dog, but we don't know if Fido is one of the brown ones. We cannot conclude Fido is brown.",
     "Not necessarily"),
    ("All Wazzles are purple. No Frumps are Wazzles. Are any Frumps purple?",
     "All Wazzles are purple. No Frumps are Wazzles — Frumps and Wazzles are disjoint sets. Therefore no Frumps are Wazzles and we cannot conclude Frumps are purple.",
     "No, not from these premises"),
    ("If A then B. If B then C. A is true. Is C true?",
     "A is true. A implies B, so B is true. B implies C, so C is true.",
     "Yes"),
    ("All even numbers are divisible by 2. 14 is even. Is 14 divisible by 2?",
     "14 is even. All even numbers are divisible by 2. Therefore 14 is divisible by 2.",
     "Yes"),
    ("Is the sky blue?",
     "The sky appears blue during the day due to Rayleigh scattering of sunlight by the atmosphere.",
     "Yes"),
    ("Is the Earth flat?",
     "The Earth is an oblate spheroid — roughly spherical. It is not flat.",
     "No"),
    ("Is a whale a mammal?",
     "Whales breathe air, are warm-blooded, give live birth, and nurse young with milk. These are mammal characteristics. Yes.",
     "Yes"),
]

# ── CATEGORY 3: EXTRACTIVE QA (in-context facts) ─────────────────────────────
EXTRACTIVE=[
    ("The Eiffel Tower was built in 1889. It is 330 meters tall. The Empire State Building was built in 1931 and is 443 meters tall. Which is taller?",
     "The prompt states the Empire State Building is 443 meters and the Eiffel Tower is 330 meters. 443 > 330, so the Empire State Building is taller.",
     "The Empire State Building"),
    ("The Eiffel Tower was built in 1889. The Empire State Building was built in 1931. Which came first?",
     "1889 is earlier than 1931. The Eiffel Tower was built in 1889, so it came first.",
     "The Eiffel Tower"),
    ("The Eiffel Tower is 330m tall. The Empire State Building is 443m tall. How much taller is the Empire State Building?",
     "443 - 330 = 113. The Empire State Building is 113 meters taller.",
     "113 meters"),
    ("A box contains 5 red balls, 3 blue balls, and 2 green balls. How many balls are in the box total?",
     "5 red + 3 blue + 2 green = 10 balls total.",
     "10 balls"),
    ("A box has 5 red balls and 3 blue balls. How many more red balls are there than blue?",
     "5 red - 3 blue = 2 more red balls.",
     "2"),
    ("John is 30 years old. Mary is 5 years younger than John. How old is Mary?",
     "Mary is 5 years younger than John. John is 30. So Mary is 30 - 5 = 25 years old.",
     "25"),
    ("A store sells apples for $2 each and oranges for $3 each. Alice buys 4 apples and 2 oranges. How much does she spend?",
     "4 apples * $2 = $8. 2 oranges * $3 = $6. Total = $8 + $6 = $14.",
     "$14"),
    ("Train A travels at 60mph. Train B travels at 80mph. They start 280 miles apart and travel toward each other. How long until they meet?",
     "Combined speed = 60 + 80 = 140 mph. Time = distance / combined speed = 280 / 140 = 2 hours.",
     "2 hours"),
    ("A recipe needs 3 cups of flour for 12 cookies. How many cups are needed for 36 cookies?",
     "36 cookies is 3 times 12 cookies. So we need 3 * 3 = 9 cups of flour.",
     "9 cups"),
    ("The temperature on Monday was 72F. On Tuesday it dropped 8 degrees. On Wednesday it rose 5 degrees. What was Wednesday's temperature?",
     "Monday: 72. Tuesday: 72-8=64. Wednesday: 64+5=69.",
     "69F"),
    ("A car gets 30 miles per gallon. The tank holds 12 gallons. How far can it travel on a full tank?",
     "Range = mpg * gallons = 30 * 12 = 360 miles.",
     "360 miles"),
    ("There are 24 students in a class. 1/3 of them are wearing glasses. How many students wear glasses?",
     "1/3 of 24 = 24/3 = 8 students.",
     "8"),
]

ALL_DATA = (
    [(q,t,a,'algebra') for q,t,a in ALGEBRA] +
    [(q,t,a,'logic')   for q,t,a in LOGIC] +
    [(q,t,a,'extract') for q,t,a in EXTRACTIVE]
)

def make_example(tok, q, think, ans, open_id, end_id, pad_id):
    text = f"User: {q}\nAssistant: <think>\n{think}\n</think>\n{ans}"
    toks = tok.encode(text)[:SEQ_LEN]
    if len(toks)<SEQ_LEN: toks+=[pad_id]*(SEQ_LEN-len(toks))
    ids=torch.tensor([toks],dtype=torch.long)
    labels=torch.full_like(ids,-100)
    weights=torch.ones(SEQ_LEN,dtype=torch.float32)
    start=next((i for i,t in enumerate(toks) if t==open_id),None)
    end  =next((i for i,t in enumerate(toks) if t==end_id), None)
    if start:
        for i in range(start+1,SEQ_LEN):
            if toks[i]==pad_id: break
            labels[0,i]=toks[i]
    if end:
        for j in range(end,min(end+ANS_WINDOW,SEQ_LEN)):
            if toks[j]==pad_id: break
            weights[j]=ANS_WEIGHT
    return ids,labels,weights

PROBES=[
    ("algebra","Solve for x: 3x+7=22","x = 5"),
    ("algebra2","What is 15% of 200?","30"),
    ("logic","All cats are mammals. Whiskers is a cat. Is Whiskers a mammal?","Yes"),
    ("logic2","If A then B. If B then C. A is true. Is C true?","Yes"),
    ("extract","The Eiffel Tower is 330m tall. The Empire State Building is 443m tall. Which is taller?","Empire State"),
    ("extract2","John is 30. Mary is 5 years younger. How old is Mary?","25"),
    ("capital","What is the capital of France?","Paris"),
    ("sky","Is the sky blue?","Yes"),
]

def run_probe(model,tok,device,end_id):
    model.eval(); results=[]; max_p=0.0; REP=1.2
    with torch.no_grad():
        for pname,prompt,expected in PROBES:
            ids=tok.encode(f"User: {prompt}\nAssistant: <think>\n",return_tensors='pt').to(device)
            gen=[]; fired=False; fp=None; pf=0.0; recent=[]
            for i in range(180):
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    logits,_=model(ids)
                raw=logits[0,-1].float()
                for prev in set(recent[-20:]): raw[prev]/=REP
                if i<THINK_MIN: raw[end_id]=-1e9
                p=float(F.softmax(raw,dim=-1)[end_id]); max_p=max(max_p,p)
                t=torch.multinomial(F.softmax(raw/0.8,dim=-1),1).item()
                if t==end_id: fired=True; fp=i; pf=p; break
                recent.append(t); gen.append(t)
                ids=torch.cat([ids,torch.tensor([[t]],device=device)],dim=-1)
            out=[]; recent2=[]
            if fired:
                for _ in range(30):
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        lg,_=model(ids)
                    raw2=lg[0,-1].float(); raw2[end_id]=-1e9
                    for prev in set(recent2[-12:]): raw2[prev]/=REP
                    t2=torch.multinomial(F.softmax(raw2/0.8,dim=-1),1).item()
                    if t2==tok.eos_token_id: break
                    recent2.append(t2); out.append(t2)
                    ids=torch.cat([ids,torch.tensor([[t2]],device=device)],dim=-1)
            ans=tok.decode(out,skip_special_tokens=True).strip()
            ok=(expected.lower() in ans.lower() or
                any(w in ans.lower() for w in expected.lower().split() if len(w)>2))
            results.append((pname,pf,fired,fp,ans,ok,expected))
    model.train(); return results,max_p

def main():
    device=torch.device("cuda")
    tok=AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tok.eos_token_id=tok.eos_token_id or 0
    tok.add_special_tokens({"additional_special_tokens":["<think>","</think>"]})
    end_id=tok.convert_tokens_to_ids("</think>")
    open_id=tok.convert_tokens_to_ids("<think>")
    pad_id=tok.pad_token_id or 1

    model=Mamba3Titan(vocab_size=50304,d_model=2560,n_layers=64,mimo_paths=16,use_gradient_checkpointing=False)
    model.resize_token_embeddings(50304); model.set_phase('sft')
    ckpt=torch.load(LOAD_FROM,map_location='cpu',weights_only=True)
    base_step=ckpt.get('step',0)
    model.load_state_dict(ckpt['model'],strict=False)
    model=model.to(torch.bfloat16).to(device)
    print(f"  Loaded step {base_step:,} | moe={model.moe_scale.item():.2f} | cp={model.cp_gate.item():.3f}")

    for p in model.parameters(): p.requires_grad_(False)
    for p in model.mimo_reasoning_blocks.parameters(): p.requires_grad_(True)
    for p in model.bridge.parameters(): p.requires_grad_(True)
    for p in model.lm_head.parameters(): p.requires_grad_(True)
    for p in model.domain_router.parameters(): p.requires_grad_(True)
    for p in model.bb_write.parameters(): p.requires_grad_(True)
    for p in model.bb_read.parameters(): p.requires_grad_(True)
    model.moe_scale.requires_grad_(True); model.cp_gate.requires_grad_(True)
    model.backbone_gate.requires_grad_(True); model.router_temp.requires_grad_(True)

    opt=torch.optim.AdamW([
        {'params':list(model.domain_router.parameters())+[model.router_temp],'lr':LR_ROUTER,'weight_decay':0.0},
        {'params':[model.moe_scale,model.cp_gate,model.backbone_gate]+
                  list(model.bb_read.parameters())+list(model.bb_write.parameters()),'lr':LR_GATES,'weight_decay':0.0},
        {'params':list(model.mimo_reasoning_blocks.parameters())+list(model.bridge.parameters()),'lr':LR_ARMS,'weight_decay':0.01},
        {'params':list(model.lm_head.parameters()),'lr':LR_HEAD,'weight_decay':0.01},
    ])

    n=len(ALL_DATA)
    cat_counts={'algebra':0,'logic':0,'extract':0}
    print(f"\n{'='*68}")
    print(f"  SFT20 REASONING — {n} examples: {len(ALGEBRA)} algebra + {len(LOGIC)} logic + {len(EXTRACTIVE)} extractive")
    print(f"  SEQ_LEN={SEQ_LEN} | ANS_WINDOW={ANS_WINDOW} | ANS_WEIGHT={ANS_WEIGHT}x | THINK_MIN={THINK_MIN}")
    print(f"{'='*68}\n")

    model.train(); step_t=time.time(); best_correct=0; idx=0

    for local_step in range(TARGET_STEPS):
        if _shutdown: break
        q,think,ans,cat=ALL_DATA[idx%n]; idx+=1; cat_counts[cat]+=1
        ids,labels,weights=make_example(tok,q,think,ans,open_id,end_id,pad_id)
        ids=ids.to(device); labels=labels.to(device); weights=weights.to(device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            logits,_=model(ids)
            B,L,V=logits.shape
            sl=logits[:,:-1].contiguous().view(-1,V)
            la=labels[:,1:].contiguous().view(-1)
            w=weights[1:].view(-1).to(device)
            seq_pos=torch.arange(L-1,device=device).unsqueeze(0).expand(B,-1).reshape(-1)
            label_flat=la.clamp(min=0)
            early_mask=(seq_pos<THINK_MIN)&(label_flat!=end_id)
            sl_m=sl.clone(); sl_m[early_mask,end_id]=-1e4
            loss_per=F.cross_entropy(sl_m,label_flat,reduction='none')
            mask=(la!=-100).float()
            loss=(loss_per*w*mask).sum()/(mask.sum()+1e-6)

        loss.backward()
        gnorm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],CLIP))
        opt.step()

        elapsed=time.time()-step_t; step_t=time.time()
        tps=SEQ_LEN/max(elapsed,1e-6); temp=gpu_temp()
        telem=model.last_telemetry; ent=telem.get('entropy',0)
        moe_s=float(model.moe_scale); cpg=float(model.cp_gate)
        line=(f"sft20 | Step {base_step+local_step+1:05d} | Loss:{loss.item():.4f} | "
              f"GNorm:{gnorm:.2f} | TPS:{tps:.0f} | moe={moe_s:.3f} | cp={cpg:.4f} | ent={ent:.3f} [{cat.upper()}]"
              f"{' | GPU:'+str(temp)+'C' if temp else ''}")
        print(line,flush=True)
        with open(LOG_PATH,'a') as f: f.write(line+"\n")
        with open("training_log.txt",'a') as f: f.write(line+"\n")
        try:
            with open(TELEM,'w') as f:
                json.dump({"phase":"sft20","step":base_step+local_step+1,"lm_loss":round(loss.item(),4),
                           "domain_loss":0.0,"grad_norm":round(gnorm,4),"tps":round(tps,1),
                           "gpu_temp":temp,"lr":LR_ARMS,"moe_scale":round(moe_s,4),"cp_gate":round(cpg,5),
                           "entropy":round(ent,3),"arm_weights":telem.get('arm_weights',[0.0625]*16),
                           "gate_score":telem.get('gate_score',0.0625),"arm_collapse_mean":telem.get('arm_collapse_mean',0),
                           "arm_collapse_max":telem.get('arm_collapse_max',0),"arm_sims":telem.get('arm_sims',[])},f)
        except: pass

        if (local_step+1)%SAVE_EVERY==0:
            torch.save({"step":base_step+local_step+1,"model":model.state_dict()},
                       f"checkpoints_2.7b/phase_sft20_step{base_step+local_step+1}.pt")

        if (local_step+1)%PROBE_EVERY==0:
            step_lbl=base_step+local_step+1
            print(f"\n[PROBE @ step {step_lbl}]")
            print(f"  moe={model.moe_scale.item():.4f} cp={model.cp_gate.item():.5f} rtemp={model.router_temp.item():.3f}")
            arm_w=model.last_telemetry.get('arm_weights',[0.125]*8)
            arm_lbl=["GenLang","Math","Logic","Code","Factual","Summ","Creative","Instruct"]
            print(f"  Router: {dict(zip(arm_lbl,[round(w,3) for w in arm_w]))}")
            results,max_p=run_probe(model,tok,device,end_id)
            fired_n=correct_n=0
            for pname,pf,fired,fp,ans,ok,expected in results:
                s="✅" if fired else "❌"; c="🎯" if ok else "✗"
                if fired: fired_n+=1
                if ok: correct_n+=1
                print(f"  {s}{c} [{pname:10s}] @{fp if fp else '--':>3} P={pf:.3f} | '{ans[:38]}' [exp:{expected}]")
            print(f"  >> {fired_n}/{len(PROBES)} fired | {correct_n}/{len(PROBES)} correct | maxP={max_p:.4f}\n")
            if correct_n>best_correct:
                best_correct=correct_n
                torch.save({"step":step_lbl,"model":model.state_dict()},SAVE_BEST)
                print(f"  [BEST] {SAVE_BEST} @ step {step_lbl} ({correct_n}/{len(PROBES)})")
            if correct_n>=7:
                print(f"🎯 TARGET REACHED — {correct_n}/{len(PROBES)}")
                torch.save({"step":step_lbl,"model":model.state_dict()},SAVE_DONE); return

    torch.save({"step":base_step+TARGET_STEPS,"model":model.state_dict()},SAVE_DONE)
    print(f"\n[SFT20 DONE] best={SAVE_BEST} ({best_correct}/{len(PROBES)}) | final={SAVE_DONE}")

if __name__=="__main__": main()
