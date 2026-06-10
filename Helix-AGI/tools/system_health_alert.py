import psutil
import json

def system_health_alert():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().percent
    
    if cpu_usage > 80 or vram_usage > 80:
        return json.dumps({"status": "alert", "cpu_usage": cpu_usage, "vram_usage": vram_usage})
    else:
        return json.dumps({"status": "ok", "cpu_usage": cpu_usage, "vram_usage": vram_usage})

if __name__ == '__main__':
    print(system_health_alert())