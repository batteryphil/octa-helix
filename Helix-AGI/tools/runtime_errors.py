import re
import json
import pathlib
from typing import List, Dict

def extract_last_n_errors(log_file_path: str, n: int = 20) -> List[str]:
    with open(log_file_path, 'r') as file:
        content = file.read()
    
    error_lines = re.findall(r'ERROR.*', content)[-n:]
    return error_lines

def main() -> None:
    log_file_path = pathlib.Path(__file__).resolve().parent / 'helix.log'
    last_errors = extract_last_n_errors(log_file_path)
    print(json.dumps(last_errors))

if __name__ == '__main__':
    main()