import json
from json import JSONDecodeError

def validate_json(json_string):
    try:
        json.loads(json_string)
        return 'valid'
    except JSONDecodeError as e:
        return 'invalid', str(e)

if __name__ == '__main__':
    test_json = '{"name": "Helix", "version": 1.0}'
    print(validate_json(test_json))  # Output: valid

    test_json = '{"name": "Helix", "version": 1.0'
    print(validate_json(test_json))  # Output: invalid SyntaxError: Expecting ',' delimiter