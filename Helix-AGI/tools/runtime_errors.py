import os
import json
import re
from pathlib import Path
from typing import List

def extract_last_20_error_logs(log_file: Path) -> List[str]:
    with open(log_file, 'r') as file:
        lines = file.readlines()
    error_lines = [line for line in lines if 'ERROR' in line]
    return error_lines[-20:]

if __name__ == '__main__':
    log_file = Path('/path/to/helix.log')
    last_20_errors = extract_last_20_error_logs(log_file)
    print(json.dumps(last_20_errors, indent=2))