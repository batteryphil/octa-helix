import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def main():
    model_name = "state-spaces/mamba-2.8b-hf"
    
    print(f"Downloading/Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    print(f"Downloading/Loading model weights for {model_name}... (This may take a few minutes)")
    # Mamba models don't support device_map="auto" gracefully in all versions of transformers,
    # so we load to CUDA explicitly and use bfloat16 to save memory.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
        model = model.to("cuda")
    except Exception as e:
        print(f"Failed to load model natively via transformers: {e}")
        print("Attempting to load via mamba_ssm package...")
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
            model = MambaLMHeadModel.from_pretrained(model_name, device="cuda", dtype=torch.bfloat16)
        except Exception as e2:
            print(f"Failed to load via mamba_ssm as well: {e2}")
            sys.exit(1)
            
    print("Model loaded successfully! Type 'quit' to exit.")
    print("Note: This is a BASE model, not instruction-tuned. It will try to autocomplete your text.\n")
    
    while True:
        try:
            prompt = input("Prompt: ")
            if prompt.strip().lower() in ['quit', 'exit']:
                break
                
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            
            print("Generating...", flush=True)
            out = model.generate(
                input_ids, 
                max_new_tokens=100, 
                temperature=0.7, 
                top_p=0.9, 
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(out[0], skip_special_tokens=True)
            print("\n--- RESPONSE ---")
            print(response)
            print("----------------\n")
            
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except EOFError:
            print("\nEOF received, exiting.")
            break
        except Exception as e:
            print(f"\nError during generation: {e}")

if __name__ == "__main__":
    main()
