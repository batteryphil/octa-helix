import psutil
import json
import os

def check_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    
    if cpu_usage > 80:
        return json.dumps({"status": "warning", "cpu_usage": cpu_usage, "vram_usage": vram_usage, "message": "CPU usage exceeded 80%"})
    elif vram_usage > 10:
        return json.dumps({"status": "warning", "cpu_usage": cpu_usage, "vram_usage": vram_usage, "message": "VRAM usage exceeded 10GB"})
    else:
        return json.dumps({"status": "ok", "cpu_usage": cpu_usage, "vram_usage": vram_usage})

if __name__ == '__main__':
    print(check_system_health())