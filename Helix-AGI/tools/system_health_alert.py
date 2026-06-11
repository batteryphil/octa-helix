import psutil
import json
import os

def system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().total - psutil.virtual_memory().available
    status = 'ok'
    
    if cpu_usage > 80:
        status = 'cpu_high'
    if vram_usage > 10 * 1024 * 1024 * 1024:  # 10 GB
        status = 'vram_high'
    
    return {'status': status, 'cpu_usage': cpu_usage, 'vram_usage': vram_usage}

def log_alert(status, cpu_usage, vram_usage):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp} - Status: {status}, CPU: {cpu_usage}%, VRAM: {vram_usage / (1024 * 1024 * 1024):.2f} GB\n"
    with open('system_health_log.txt', 'a') as f:
        f.write(log_entry)

if __name__ == '__main__':
    print(json.dumps(system_health()))
    log_alert(*system_health().values())