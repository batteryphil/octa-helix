import json
import requests
import re
import pathlib
import psutil
from bs4 import BeautifulSoup

def analyze_behavior_data(data):
    patterns = {
        'low_performance': re.compile(r'performance below standard'),
        'missed_standards': re.compile(r'missed own standards'),
    }
    issues = {}
    for entry in data:
        for pattern, regex in patterns.items():
            if regex.search(entry['description']):
                issues.setdefault(pattern, []).append(entry)
    return issues

def fetch_behavior_data():
    url = 'https://example.com/helix/behavior'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    data = []
    for entry in soup.find_all('entry'):
        data.append({
            'timestamp': entry.find('timestamp').text,
            'description': entry.find('description').text,
        })
    return data

def main():
    data = fetch_behavior_data()
    issues = analyze_behavior_data(data)
    print(json.dumps(issues, indent=2))

if __name__ == '__main__':
    main()