import json
import sys

def validate_json(data):
    try:
        json.loads(data)
        return 'valid', json.loads(data)
    except json.JSONDecodeError:
        return 'invalid', 'Invalid JSON data'

def main():
    if len(sys.argv) != 2:
        print("Usage: python data_validator.py <json_data>")
        sys.exit(1)

    json_data = sys.argv[1]
    result, parsed_data = validate_json(json_data)
    print(f"Result: {result}")
    if result == 'valid':
        print(parsed_data)

if __name__ == '__main__':
    main()