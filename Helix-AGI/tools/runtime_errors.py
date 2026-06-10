import json
import re
from pathlib import Path
from typing import List

def extract_last_20_error_logs(log_file_path: Path) -> List[str]:
    with open(log_file_path, 'r') as file:
        logs = file.readlines()
    
    error_logs = [log for log in logs if 'ERROR' in log]
    last_20_errors = error_logs[-20:]
    
    return last_20_errors

def main():
    log_file_path = Path('/path/to/helix/log/file.log')
    last_20_error_logs = extract_last_20_error_logs(log_file_path)
    
    for log in last_20_error_logs:
        print(log.strip())

if __name__ == '__main__':
    main()