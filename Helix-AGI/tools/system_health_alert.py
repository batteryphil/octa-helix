import psutil
import json
import time
import os
import re
import pathlib

def get_cpu_usage():
    return psutil.cpu_percent()

def get_vram_usage():
    return psutil.virtual_memory().total - psutil.virtual_memory().available

def check_system_health():
    cpu_usage = get_cpu_usage()
    vram_usage = get_vram_usage()
    
    status = {
        "cpu_usage": cpu_usage,
        "vram_usage": vram_usage,
        "alert": False
    }
    
    if cpu_usage > 80:
        status["alert"] = "High CPU usage"
    if vram_usage > 10737418240: # 10GB in bytes
        status["alert"] = "High VRAM usage"
    
    return status

def main():
    while True:
        status = check_system_health()
        print(json.dumps(status))
        time.sleep(5)

if __name__ == "__main__":
    main()