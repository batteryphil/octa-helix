import psutil
import json
import os

def check_system_health():
    memory_usage = psutil.virtual_memory()
    memory_info = {
        'memory_total': memory_usage.total,
        'memory_available': memory_usage.available,
        'memory_percent': memory_usage.percent,
        'memory_used': memory_usage.used
    }
    
    cpu_usage = psutil.cpu_percent()
    memory_info['cpu_percent'] = cpu_usage
    
    if memory_info['memory_percent'] > 90:
        memory_info['status'] = 'critical'
    elif memory_info['memory_percent'] > 80:
        memory_info['status'] = 'warning'
    else:
        memory_info['status'] = 'ok'
    
    return memory_info

def main():
    health_info = check_system_health()
    print(json.dumps(health_info, indent=2))

if __name__ == '__main__':
    main()