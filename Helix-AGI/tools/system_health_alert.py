import psutil
import json

def check_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    
    if cpu_usage > 80:
        return "alert", f"CPU usage exceeded 80%: {cpu_usage}%"
    elif vram_usage > 10:
        return "alert", f"VRAM usage exceeded 10GB: {vram_usage:.2f} GB"
    else:
        return "ok", ""

if __name__ == "__main__":
    status, message = check_system_health()
    print(json.dumps({"status": status, "message": message}))