import psutil
import json
import time
import os

def check_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    
    if cpu_usage > 80:
        return {
            "status": "warning",
            "cpu_usage": cpu_usage,
            "vram_usage": vram_usage
        }
    elif vram_usage > 10:
        return {
            "status": "warning",
            "cpu_usage": cpu_usage,
            "vram_usage": vram_usage
        }
    else:
        return {
            "status": "ok",
            "cpu_usage": cpu_usage,
            "vram_usage": vram_usage
        }

def main():
    while True:
        health = check_system_health()
        print(json.dumps(health))
        time.sleep(5)

if __name__ == "__main__":
    main()