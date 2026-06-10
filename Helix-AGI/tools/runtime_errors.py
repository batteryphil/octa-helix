import os
import re
from pathlib import Path
from typing import List

def get_last_20_error_lines(log_file: Path) -> List[str]:
    with open(log_file, 'r') as file:
        lines = file.readlines()
    error_lines = [line.strip() for line in lines if 'ERROR' in line]
    return error_lines[-20:] if len(error_lines) > 20 else error_lines

if __name__ == '__main__':
    log_file = Path('/path/to/helix/log/file.log')
    last_20_errors = get_last_20_error_lines(log_file)
    for error in last_20_errors:
        print(error)