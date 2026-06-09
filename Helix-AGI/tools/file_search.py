"""
Search and grep files in the project.

Usage:
    $ python -m tools.file_search --query "search query" [--case-sensitive] [--whole-word] [--file-type txt]

Search for lines matching the given query in all files in the project. Options:
    --case-sensitive: Perform a case-sensitive search.
    --whole-word: Match only whole words.
    --file-type: Filter files by type (e.g., txt, py, md).
"""

import os
import sys
import argparse
from pathlib import Path

def search_files(query, case_sensitive, whole_word, file_type, path="."):
    matches = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file_type and not file.endswith(file_type):
                continue
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                with open(file_path, "r" if case_sensitive else "rU") as f:
                    if whole_word:
                        matches.extend(line for line in f if query in line and not line.split(query)[1])
                    else:
                        matches.extend(line for line in f if query in line.lower() if case_sensitive or query.lower() in line)
    return matches

def main():
    parser = argparse.ArgumentParser(description="Search and grep files in the project.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--case-sensitive", action="store_true", help="Perform a case-sensitive search")
    parser.add_argument("--whole-word", action="store_true", help="Match only whole words")
    parser.add_argument("--file-type", help="Filter files by type (e.g., txt, py, md)")
    args = parser.parse_args(sys.argv[1:])
    matches = search_files(args.query, args.case_sensitive, args.whole_word, args.file_type)
    for match in matches:
        print(match.strip())

if __name__ == "__main__":
    main()