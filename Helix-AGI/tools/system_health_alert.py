import psutil
import json
import os

def system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    
    if cpu_usage > 80:
        return 'cpu_high'
    elif vram_usage > 10:
        return 'vram_high'
    else:
        return 'ok'

def main():
    health_status = system_health()
    print(json.dumps({'status': health_status}))

if __name__ == '__main__':
    main()