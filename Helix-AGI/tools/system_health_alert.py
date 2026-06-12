import psutil
import json
import os

def check_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().total - psutil.virtual_memory().available

    if cpu_usage > 80:
        return "alert"
    elif vram_usage > 10 * 1024 * 1024 * 1024:  # 10 GB in bytes
        return "alert"
    else:
        return "ok"

def main():
    health_status = check_system_health()
    print(json.dumps({"system_health": health_status}))

if __name__ == "__main__":
    main()