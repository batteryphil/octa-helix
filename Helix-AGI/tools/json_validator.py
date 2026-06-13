import json
from json import JSONDecodeError

def validate_json(json_string):
    try:
        json.loads(json_string)
        return 'valid'
    except JSONDecodeError as e:
        return 'invalid', str(e)

if __name__ == '__main__':
    test_json = '{"name": "John", "age": 30, "city": "New York"}'
    print(validate_json(test_json))

    test_invalid_json = '{"name": "John", "age": 30, "city": "New York"'
    print(validate_json(test_invalid_json))