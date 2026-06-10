import json
import os
from datetime import datetime
from pathlib import Path
import psutil
import re
import requests
from bs4 import BeautifulSoup

class BehaviorTracker:
    def __init__(self):
        self.metrics = {
            'success_rate': [],
            'tool_usage': [],
            'novel_beliefs': []
        }
        self.last_save_time = None

    def log_success_rate(self, success):
        self.metrics['success_rate'].append(success)

    def log_tool_usage(self, tool):
        self.metrics['tool_usage'].append(tool)

    def log_novel_beliefs(self, beliefs):
        self.metrics['novel_beliefs'].extend(beliefs)

    def save_metrics(self):
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        if self.last_save_time != now:
            with open(Path('tools') / f'behavior_tracker_{now}.json', 'w') as f:
                json.dump(self.metrics, f, indent=2)
            self.last_save_time = now