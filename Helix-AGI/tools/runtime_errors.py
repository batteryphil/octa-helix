import json
import re
import pathlib
from pathlib import Path
from typing import List, Dict

def parse_log_entry(entry: str) -> Dict[str, str]:
    """
    Parse a single log entry and extract the error message.
    """
    error_pattern = re.compile(r"ERROR: (.+)")
    match = error_pattern.search(entry)
    if match:
        return {"error": match.group(1)}
    return {}

def summarize_errors(log_path: Path, num_entries: int = 100) -> List[Dict[str, int]]:
    """
    Summarize the most recent runtime errors from the log file.
    """
    with open(log_path, "r") as file:
        log_content = file.read()
    
    lines = log_content.split("\n")[-num_entries:]
    errors = (parse_log_entry(line) for line in lines if "ERROR" in line)
    error_summary = {}
    for error in errors:
        error_message = error.get("error")
        if error_message:
            error_summary[error_message] = error_summary.get(error_message, 0) + 1
    
    return [{"message": k, "count": v} for k, v in error_summary.items()]

if __name__ == "__main__":
    log_file_path = pathlib.Path("helix.log")
    num_recent_errors = 10
    error_summary = summarize_errors(log_file_path, num_recent_errors)
    print(json.dumps(error_summary, indent=2))