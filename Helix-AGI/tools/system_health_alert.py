import json
import os
import psutil
import re
import sys
import time
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
import requests

class SystemHealthAlert:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        self.logs_path = Path("system_health_alert_logs")
        self.logs_path.mkdir(exist_ok=True)

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            return json.load(f)

    def get_cpu_usage(self):
        return psutil.cpu_percent()

    def get_memory_usage(self):
        return psutil.virtual_memory().percent

    def get_disk_usage(self):
        return psutil.disk_usage('/').percent

    def log_data(self, data):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file = self.logs_path / f"system_health_alert_{timestamp}.log"
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)