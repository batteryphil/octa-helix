import json
import re
from pathlib import Path
from typing import List, Dict

def validate_data(data: List[Dict], schema: Dict) -> Dict:
    errors = []
    for item in data:
        for key, value in item.items():
            if key not in schema:
                errors.append(f"Key '{key}' not found in schema")
            if isinstance(schema[key], str) and not isinstance(value, str):
                errors.append(f"Key '{key}' should be a string, got {type(value)}")
            if isinstance(schema[key], int) and not isinstance(value, int):
                errors.append(f"Key '{key}' should be an integer, got {type(value)}")
            if isinstance(schema[key], bool) and value not in [True, False]:
                errors.append(f"Key '{key}' should be a boolean, got {value}")
    return {"errors": errors}