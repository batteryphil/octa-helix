import os
import re
import sys
import json
from pathlib import Path

def get_last_20_error_logs():
    log_file = Path('/var/log/helix.log')
    if not log_file.exists():
        print("Error log file does not exist.")
        return []
    
    try:
        with log_file.open() as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading log file: {e}")
        return []
    
    error_lines = [line for line in lines if 'ERROR' in line]
    last_20_errors = error_lines[-20:]
    
    return last_20_errors

def main():
    errors = get_last_20_error_logs()
    print(json.dumps(errors))

if __name__ == '__main__':
    main()