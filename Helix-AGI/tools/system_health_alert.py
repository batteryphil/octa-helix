import psutil
import json

def get_system_health():
    cpu_usage = psutil.cpu_percent()
    vram_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)
    return {
        'cpu_usage': cpu_usage,
        'vram_usage': vram_usage
    }

def main():
    health = get_system_health()
    if health['cpu_usage'] > 80 or health['vram_usage'] > 10:
        print(json.dumps({'status': 'alert', 'reason': 'CPU usage exceeds 80% or VRAM usage exceeds 10GB', 'data': health}))
    else:
        print(json.dumps({'status': 'ok', 'data': health}))

if __name__ == '__main__':
    main()