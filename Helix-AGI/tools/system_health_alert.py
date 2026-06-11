import psutil
import json
import os

def get_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    return {
        "cpu_usage": cpu_usage,
        "vram_usage": vram_usage,
        "status": "ok"
    }

    if cpu_usage > 80:
        return {
            "cpu_usage": cpu_usage,
            "vram_usage": vram_usage,
            "status": "alert"
        }
    elif vram_usage > 10:
        return {
            "cpu_usage": cpu_usage,
            "vram_usage": vram_usage,
            "status": "alert"
        }

if __name__ == '__main__':
    print(json.dumps(get_system_health(), indent=2))