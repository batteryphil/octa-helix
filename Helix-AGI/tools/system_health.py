"""
Monitor system health metrics such as CPU, RAM, disk, and GPU usage and store them for analysis.
"""

import psutil
import time
import json
import os
from helix.registry import ToolRegistry

class SystemHealthMonitor:
    def __init__(self, interval=60):
        self.interval = interval
        self.data = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'gpu': []
        }
        ToolRegistry.register_tool(self, toolset='self')

    def collect_metrics(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        gpu_usage = psutil.gpu_percent()

        self.data['cpu'].append(cpu_usage)
        self.data['memory'].append({
            'total': memory_usage.total,
            'available': memory_usage.available,
            'percent': memory_usage.percent
        })
        self.data['disk'].append({
            'total': disk_usage.total,
            'used': disk_usage.used,
            'free': disk_usage.free,
            'percent': disk_usage.percent
        })
        self.data['gpu'].append(gpu_usage)

    def save_data(self):
        timestamp = int(time.time())
        filename = f'system_health_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)

    def run(self):
        while True:
            self.collect_metrics()
            self.save_data()
            time.sleep(self.interval)

if __name__ == '__main__':
    monitor = SystemHealthMonitor()
    monitor.run()