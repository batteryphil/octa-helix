import json
import re
import pathlib
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from pathlib import Path
from json import JSONDecodeError

def parse_log_file(log_file_path: str) -> List[Dict]:
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    error_lines = []
    for i in range(-1, -21, -1):
        if 'ERROR' in lines[i]:
            error_lines.append({'line': i, 'content': lines[i].strip()})
    return error_lines

def extract_error_details(url: str) -> Dict:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').text
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
        return {'title': title, 'memory_usage': memory_usage}
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}
    except JSONDecodeError:
        return {'error': 'Invalid JSON'}

def main():
    log_file_path = 'path/to/helix/log/file.log'
    error_lines = parse_log_file(log_file_path)
    for error_line in error_lines:
        error_details = extract_error_details(error_line['content'])
        print(json.dumps({**error_line, **error_details}, indent=2))

if __name__ == '__main__':
    main()