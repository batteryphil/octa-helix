import os
import json
import re
from pathlib import Path
from typing import List, Dict

def get_last_20_error_lines(log_file: Path) -> List[Dict]:
    with open(log_file, "r") as f:
        lines = f.readlines()
    error_lines = [line.strip() for line in lines[-20:] if "ERROR" in line]
    return [{"line": line, "timestamp": str(Path(log_file).stat().st_mtime)} for line in error_lines]

def main():
    log_dir = Path("/var/log/helix")
    log_files = list(log_dir.glob("*.log"))
    if log_files:
        most_recent_log = max(log_files, key=lambda f: f.stat().st_mtime)
        errors = get_last_20_error_lines(most_recent_log)
        print(json.dumps(errors, indent=2))
    else:
        print("No log files found in Helix log directory.")

if __name__ == "__main__":
    main()