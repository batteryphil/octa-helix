import psutil
import json
import os
import time
from pathlib import Path
from datetime import datetime

def check_system_health():
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    cpu_threshold = 80
    memory_threshold = 10 * 1024 * 1024 * 1024  # 10 GB
    
    health_data = {
        "timestamp": datetime.now().isoformat(),
        "cpu_percent": cpu_percent,
        "memory_total": memory_info.total,
        "memory_available": memory_info.available,
        "memory_percent": memory_info.percent,
        "memory_used": memory_info.used
    }
    
    if cpu_percent > cpu_threshold:
        health_data["cpu_alert"] = f"CPU usage exceeded threshold: {cpu_percent}% > {cpu_threshold}%"
    
    if memory_info.used > memory_threshold:
        health_data["memory_alert"] = f"Memory usage exceeded threshold: {memory_info.used} bytes > {memory_threshold} bytes"
    
    return json.dumps(health_data)

if __name__ == "__main__":
    print(check_system_health())