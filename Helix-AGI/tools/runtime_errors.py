import os
import re
from pathlib import Path

def extract_last_20_error_logs():
    log_file_path = Path(__file__).parent / 'helix.log'
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
    error_lines = [line for line in lines if 'ERROR' in line]
    last_20_errors = error_lines[-20:]
    return last_20_errors

def main():
    error_logs = extract_last_20_error_logs()
    for error in error_logs:
        print(error, end='')

if __name__ == '__main__':
    main()