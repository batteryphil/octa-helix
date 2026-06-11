import psutil
import json

def check_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    
    if cpu_usage > 80:
        return "alert: CPU usage exceeds 80% - current usage: {}%".format(cpu_usage)
    elif vram_usage > 10:
        return "alert: VRAM usage exceeds 10GB - current usage: {}GB".format(vram_usage)
    else:
        return "ok"

if __name__ == '__main__':
    print(check_system_health())