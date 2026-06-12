import json
import jsonlines
import os
import sys

def validate_jsonl(file_path):
    if not os.path.exists(file_path):
        return "File does not exist", False
    
    with open(file_path, "r") as f:
        try:
            for line in f:
                jsonlines.loads(line)
            return "File structure and content validated", True
        except json.JSONDecodeError:
            return "JSONL file structure invalid", False

def validate_json(file_path):
    if not os.path.exists(file_path):
        return "File does not exist", False
    
    with open(file_path, "r") as f:
        try:
            json.load(f)
            return "File structure and content validated", True
        except json.JSONDecodeError:
            return "JSON file structure invalid", False

def data_validator(file_path):
    if file_path.suffix == ".jsonl":
        return validate_jsonl(file_path)
    elif file_path.suffix == ".json":
        return validate_json(file_path)
    else:
        return f"Unsupported file type: {file_path.suffix}", False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_validator.py <file_path>")
        sys.exit(1)
    
    file_path = pathlib.Path(sys.argv[1])
    result, valid = data_validator(file_path)
    print(result)
    sys.exit(valid)