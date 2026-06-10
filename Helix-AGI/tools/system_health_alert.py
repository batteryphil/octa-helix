import psutil
import json
import os

def check_system_health():
    cpu_percent = psutil.cpu_percent()
    mem_usage = psutil.virtual_memory()
    cpu_threshold = 80
    mem_threshold = 80

    health_status = {
        "cpu_usage": cpu_percent,
        "memory_usage": mem_usage.percent,
        "status": "ok"
    }

    if cpu_percent > cpu_threshold:
        health_status["status"] = "alert"
        health_status["message"] = f"High CPU usage: {cpu_percent}%"

    if mem_usage.percent > mem_threshold:
        health_status["status"] = "alert"
        health_status["message"] = f"High memory usage: {mem_usage.percent}%"

    return health_status

def main():
    health = check_system_health()
    print(json.dumps(health))

if __name__ == "__main__":
    main()