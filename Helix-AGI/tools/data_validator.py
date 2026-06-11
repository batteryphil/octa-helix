import json
import re
from pathlib import Path

def validate_data(data, schema):
    errors = []
    for item in data:
        if not all(key in item for key in schema['keys']):
            errors.append(f"Missing keys: {', '.join(schema['keys'])}")
        for key, value in item.items():
            if key not in schema['keys']:
                errors.append(f"Unknown key: {key}")
            if schema['types'][key] != type(value):
                errors.append(f"Type mismatch: {key} expected {schema['types'][key]}, got {type(value)}")
    return errors

def main():
    import sys
    if len(sys.argv) != 3:
        print("Usage: python data_validator.py <data_file> <schema_file>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    schema_file = sys.argv[2]
    
    with open(data_file) as f:
        data = json.load(f)
    
    with open(schema_file) as f:
        schema = json.load(f)
    
    errors = validate_data(data, schema)
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"- {error}")
    else:
        print("Data validates against schema")

if __name__ == '__main__':
    main()