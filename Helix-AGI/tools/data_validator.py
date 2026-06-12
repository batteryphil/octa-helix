"""Data validation tool — validates JSON data structure without external deps."""
import json

def validate_json_structure(data_str: str, required_keys: list = None) -> str:
    """Validate that a string is valid JSON and optionally has required keys."""
    try:
        obj = json.loads(data_str) if isinstance(data_str, str) else data_str
    except (json.JSONDecodeError, TypeError) as e:
        return f"invalid JSON: {e}"
    if required_keys:
        missing = [k for k in required_keys if k not in obj]
        if missing:
            return f"missing required keys: {missing}"
    return f"valid ({type(obj).__name__} with {len(obj) if hasattr(obj,'__len__') else 'N/A'} items)"

if __name__ == '__main__':
    data = '{"name": "John", "age": 30, "city": "New York"}'
    print(validate_json_structure(data, ['name', 'age']))
