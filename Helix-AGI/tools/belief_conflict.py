import json
import os
from pathlib import Path
from typing import List, Tuple

from bs4 import BeautifulSoup
import requests

# Mock belief_store.json
BELIEF_STORE_FILE = "belief_store.json"

def load_beliefs(file_path: str) -> List[Tuple[str, float]]:
    with open(file_path, "r") as f:
        beliefs = json.load(f)
    return [(belief["belief"], belief["confidence"]) for belief in beliefs if belief["confidence"] > 0.8]

def find_conflicts(beliefs: List[Tuple[str, float]]) -> List[Tuple[str, str]]:
    conflicts = []
    for i, (belief1, _) in enumerate(beliefs):
        for belief2, _ in beliefs[i+1:]:
            if belief1 == belief2:
                continue
            if any(map(belief1.__contains__, ["not", "no", "never", "none"]) and any(map(belief2.__contains__, ["not", "no", "never", "none"])) or 
                any(map(belief2.__contains__, ["not", "no", "never", "none"]) and any(map(belief1.__contains__, ["not", "no", "never", "none"]))) or
                belief1.startswith("The belief") and belief2.startswith("The belief")):
                conflicts.append((belief1, belief2))
    return conflicts

def main():
    beliefs = load_beliefs(BELIEF_STORE_FILE)
    conflicts = find_conflicts(beliefs)
    print(json.dumps(conflicts, indent=2))

if __name__ == "__main__":
    main()