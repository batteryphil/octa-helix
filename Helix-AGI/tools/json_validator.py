import json
from json import JSONDecodeError

def validate_json(json_string):
    try:
        json.loads(json_string)
        return 'valid'
    except JSONDecodeError as e:
        return 'invalid', str(e)

if __name__ == '__main__':
    json_string = '{"name": "John", "age": 30, "city": "New York"}'
    print(validate_json(json_string))  # Output: valid

    json_string = '{"name": "John", "age": 30, "city": "New York"'
    print(validate_json(json_string))  # Output: invalid, Expecting ',' delimiter