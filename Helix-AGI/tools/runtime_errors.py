import json
import re
import pathlib
from typing import List, Dict

def parse_helix_log_file(log_file_path: str) -> List[Dict]:
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    
    error_lines = [line.strip() for line in lines if 'ERROR' in line]
    last_20_errors = error_lines[-20:]
    
    return [{'line_number': i+1, 'error_message': error} for i, error in enumerate(last_20_errors)]

def main():
    log_file_path = 'path/to/helix/log/file.log'
    errors = parse_helix_log_file(log_file_path)
    print(json.dumps(errors, indent=2))

if __name__ == '__main__':
    main()