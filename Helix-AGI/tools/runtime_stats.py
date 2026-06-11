import json
import psutil
import time
import os
from pathlib import Path
from datetime import datetime

class RuntimeStats:
    def __init__(self, output_file):
        self.output_file = output_file
        self.stats = []

    def collect_stats(self):
        proc = psutil.Process()
        cpu_percent = psutil.cpu_percent()
        memory_info = proc.memory_full_info()
        memory_percent = memory_info.rss / (1024 * 1024 * 1024)  # in GB
        execution_time = time.time()

        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'execution_time': execution_time
        }

    def log_stats(self, stats):
        self.stats.append(stats)

    def save_stats(self):
        Path(self.output_file).write_text(json.dumps(self.stats, indent=2))

def main():
    stats_tool = RuntimeStats(output_file='runtime_stats.json')
    for _ in range(5):
        stats = stats_tool.collect_stats()
        stats_tool.log_stats(stats)
        time.sleep(1)
    stats_tool.save_stats()

if __name__ == '__main__':
    main()