import os
import time
import subprocess

MODEL_DIR = "/home/phil/.gemini/antigravity/scratch/analysis_project/hf_cache/qwen2.5-coder-32b"
MODEL_FILE = os.path.join(MODEL_DIR, "qwen2.5-coder-32b-instruct-q4_k_m.gguf")

def main():
    print("Waiting for model download to complete...")
    while not os.path.exists(MODEL_FILE):
        time.sleep(10)
        
    print("\nModel downloaded! Executing switchover...")
    
    # 1. Kill the old pipelines
    subprocess.run(["pkill", "-f", "llama_cpp.server"])
    subprocess.run(["pkill", "-f", "run_swe_bounty.py"])
    subprocess.run(["pkill", "-f", "sweagent run"])
    
    time.sleep(5)
    
    # 2. Start the new API server
    print("Starting new llama_cpp.server on 32B model...")
    api_cmd = [
        "/home/phil/.gemini/antigravity/scratch/analysis_project/titan_venv/bin/python",
        "-m", "llama_cpp.server",
        "--model", MODEL_FILE,
        "--n_ctx", "8192",
        "--n_gpu_layers", "0",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    
    # We use Popen so it runs in the background
    with open("api_server.log", "w") as f:
        subprocess.Popen(api_cmd, stdout=f, stderr=subprocess.STDOUT)
        
    time.sleep(10) # Give it time to load
    
    # 3. Restart the orchestrator
    print("Restarting 24/7 bounty orchestrator...")
    orc_cmd = [
        "/home/phil/.gemini/antigravity/scratch/analysis_project/titan_venv/bin/python",
        "/home/phil/.gemini/antigravity/scratch/analysis_project/run_swe_bounty.py"
    ]
    with open("bounty_orchestrator.log", "a") as f:
        subprocess.Popen(orc_cmd, stdout=f, stderr=subprocess.STDOUT)
        
    print("Switchover complete!")

if __name__ == "__main__":
    main()
