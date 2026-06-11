import json
import sys
import logging
from pathlib import Path

def validate_and_clean_jsonl_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    cleaned_lines = []
    for line in lines:
        try:
            data = json.loads(line)
            cleaned_data = clean_data(data)
            cleaned_lines.append(json.dumps(cleaned_data))
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON line: {line}")
    
    return '\n'.join(cleaned_lines)

def clean_data(data):
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, float):
            if value != value or value == float('inf') or value == float('-inf') or value == float('nan'):
                value = None
        elif isinstance(value, (list, tuple)):
            cleaned_value = []
            for item in value:
                cleaned_item = clean_data(item) if isinstance(item, dict) else item
                cleaned_value.append(cleaned_item)
            value = cleaned_value
        elif isinstance(value, dict):
            value = clean_data(value)
        cleaned_data[key] = value
    return cleaned_data

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python data_validator.py <file_path>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    cleaned_data = validate_and_clean_jsonl_data(file_path)
    print(cleaned_data)