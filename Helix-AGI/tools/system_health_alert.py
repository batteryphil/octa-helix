import psutil
import json
import os

def check_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().total - psutil.virtual_memory().available

    if cpu_usage > 80:
        return "CPU usage exceeds 80%: {}%".format(cpu_usage)
    elif vram_usage > 10 * 1024 * 1024 * 1024:  # 10 GB
        return "VRAM usage exceeds 10GB: {} bytes".format(vram_usage)
    else:
        return "ok"

def main():
    result = check_system_health()
    print(json.dumps({"status": result}))

if __name__ == "__main__":
    main()