import psutil
import json

def check_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().total - psutil.virtual_memory().available
    vram_usage_gb = vram_usage / (1024 * 1024 * 1024)

    if cpu_usage > 80:
        return 'cpu_high'
    elif vram_usage_gb > 10:
        return 'vram_high'
    else:
        return 'ok'

if __name__ == '__main__':
    print(json.dumps({'status': check_system_health()}))