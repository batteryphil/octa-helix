import json
import sys

def validate_json(json_str, schema):
    try:
        data = json.loads(json_str)
        return jsonschema.validate(data, schema), None
    except json.JSONDecodeError as e:
        return None, str(e)
    except jsonschema.ValidationError as e:
        return None, str(e)

def main():
    if len(sys.argv) != 3:
        print("Usage: python data_validator.py <json_string> <schema_file>")
        sys.exit(1)

    json_str = sys.argv[1]
    schema_file = sys.argv[2]

    with open(schema_file) as f:
        schema = json.load(f)

    result, error = validate_json(json_str, schema)

    if result:
        print("valid")
    else:
        print(f"invalid: {error}")

if __name__ == "__main__":
    main()