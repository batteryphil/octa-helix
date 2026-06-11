import psutil
import json
import os

def check_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    
    if cpu_usage > 80:
        return 'cpu_alert'
    elif vram_usage > 10:
        return 'vram_alert'
    else:
        return 'ok'

def main():
    health_status = check_system_health()
    print(json.dumps({'status': health_status}))

if __name__ == '__main__':
    main()