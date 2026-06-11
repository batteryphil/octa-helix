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
    print(validate_json(test_json))  # Output: valid

    test_json_invalid = '{"name": "John", "age": 30, "city": "New York"'
    print(validate_json(test_json_invalid))  # Output: ('invalid', "Expecting ',' delimiter: line 1 column 27 (char 26)")