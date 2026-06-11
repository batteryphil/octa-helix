import json
from json import JSONDecodeError

def validate_json(json_string):
    try:
        json.loads(json_string)
        return 'valid'
    except JSONDecodeError:
        return 'invalid', 'Invalid JSON string'

def main():
    json_string = '{"name": "John", "age": 30, "city": "New York"}'
    result = validate_json(json_string)
    print(result)

if __name__ == '__main__':
    main()