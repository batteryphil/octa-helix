import time
import torch
from titan_inference import get_engine

print("Initializing Titan MIMO inference engine...")
engine = get_engine("auto")
print("Engine loaded. Type 'quit' to exit.\n")

history = []

while True:
    try:
        user_input = input("User: ")
        if user_input.strip().lower() == 'quit':
            break
            
        history.append(f"User: {user_input}")
        prompt = "\n".join(history) + "\nAssistant:"
        
        print("Assistant:", end="", flush=True)
        
        response_tokens = []
        for tok, info in engine.stream(prompt, max_new_tokens=256):
            print(tok, end="", flush=True)
            response_tokens.append(tok)
            
        print() # Newline
        
        response = "".join(response_tokens).strip()
        history.append(f"Assistant: {response}")
        
    except KeyboardInterrupt:
        print("\nInterrupted.")
        break
    except Exception as e:
        print(f"\nError: {e}")
