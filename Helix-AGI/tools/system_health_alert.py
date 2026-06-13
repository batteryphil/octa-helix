import psutil
import json
import os

def get_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    return {
        'cpu_usage': cpu_usage if cpu_usage < 80 else f"CPU usage exceeded 80%: {cpu_usage}%",
        'vram_usage': vram_usage if vram_usage < 10 else f"VRAM usage exceeded 10GB: {vram_usage}GB"
    }

def main():
    health = get_system_health()
    print(json.dumps(health))

if __name__ == '__main__':
    main()