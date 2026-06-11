import os
import re
from pathlib import Path
from typing import List

def extract_last_20_error_logs(log_file: str) -> List[str]:
    with open(log_file, 'r') as f:
        lines = f.readlines()
    error_lines = [line for line in lines if 'ERROR' in line]
    return error_lines[-20:]

def main():
    log_file = 'helix.log'
    last_20_errors = extract_last_20_error_logs(log_file)
    for error in last_20_errors:
        print(error, end='')

if __name__ == '__main__':
    main()