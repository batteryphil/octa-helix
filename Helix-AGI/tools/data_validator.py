import json
import re
import pathlib
from pathlib import Path
from typing import Any, Dict

def validate_json_file(file_path: Path) -> str:
    try:
        with file_path.open('r') as f:
            data: Dict[str, Any] = json.load(f)
        return 'valid'
    except json.JSONDecodeError as e:
        return 'invalid', str(e)

def validate_file_format(file_path: Path) -> str:
    if file_path.suffix == '.json':
        return validate_json_file(file_path)
    else:
        return 'invalid', f'Unsupported file format: {file_path.suffix}'

def main():
    import sys
    if len(sys.argv) != 2:
        print('Usage: python data_validator.py <file_path>')
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.is_file():
        print(f'Error: {file_path} is not a file.')
        sys.exit(1)
    
    status, message = validate_file_format(file_path)
    print(status, message)

if __name__ == '__main__':
    main()