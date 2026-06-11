import json
import re
import pathlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict

def parse_error_line(line: str) -> Dict:
    timestamp_match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
    timestamp = timestamp_match.group(1) if timestamp_match else "N/A"
    error_message_match = re.search(r"ERROR: (.+)", line)
    error_message = error_message_match.group(1) if error_message_match else "N/A"
    stack_trace_match = re.search(r"\nTraceback \(most recent call last\):\n(.+)", line)
    stack_trace = stack_trace_match.group(1) if stack_trace_match else "N/A"
    return {
        "timestamp": timestamp,
        "error_message": error_message,
        "stack_trace": stack_trace
    }

def extract_recent_errors(log_path: Path) -> List[Dict]:
    with open(log_path, "r") as file:
        lines = file.readlines()
    errors = []
    for line in reversed(lines):
        if "ERROR" in line:
            error_info = parse_error_line(line)
            errors.append(error_info)
            if len(errors) >= 5:
                break
    return errors

def main():
    log_path = Path("/path/to/helix/log/file.log")
    recent_errors = extract_recent_errors(log_path)
    print(json.dumps(recent_errors, indent=2))

if __name__ == "__main__":
    main()