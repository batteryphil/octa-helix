#!/usr/bin/env python3
# -*- coding: utf-8 –*

"""
This tool monitors system health metrics like CPU, RAM, disk, and GPU usage.
It periodically collects the metrics and provides a summary of the current resource usage.
It also logs any critical issues for later analysis.
"""

import psutil
import time
import json
import os
import logging
from collections import defaultdict

TOOL_ID = 'system_health'
TOOL_NAME = 'System Health Monitor'
TOOL_DESCRIPTION = 'Monitors system health metrics like CPU, RAM, disk, and GPU usage.'
TOOL_VERSION = '1.0'
TOOL_AUTHOR = 'Helix AI'
TOOL_LICENSE = 'MIT'

class SystemHealthMonitor:
    def __init__(self, interval=60):
        self.interval = interval
        self.last_check = time.time()
        self.metrics = defaultdict(dict)
        self.critical_issues = []
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(filename='system_health.log', level=logging.INFO)

    def collect_metrics(self):
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent
        gpu_usage = psutil.gpu_percent()
        return {
            'cpu': cpu_usage,
            'ram': ram_usage,
            'disk': disk_usage,
            'gpu': gpu_usage
        }

    def check_for_critical_issues(self, metrics):
        issues = []
        if metrics['cpu'] > 90:
            issues.append('High CPU usage')
        if metrics['ram'] > 90:
            issues.append('High RAM usage')
        if metrics['disk'] > 90:
            issues.append('High disk usage')
        if metrics['gpu'] > 90:
            issues.append('High GPU usage')
        return issues

    def run(self):
        while True:
            current_time = time.time()
            if current_time - self.last_check >= self.interval:
                metrics = self.collect_metrics()
                self.metrics[current_time] = metrics
                issues = self.check_for_critical_issues(metrics)
                if issues:
                    self.critical_issues.extend(issues)
                    logging.warning(f'Critical issues at {current_time}: {", ".join(issues)}')
                self.last_check = current_time
            time.sleep(self.interval)

    def get_metrics(self):
        return self.metrics

    def get_critical_issues(self):
        return self.critical_issues

if __name__ == '__main__':
    tool = SystemHealthMonitor()
    tool.run()