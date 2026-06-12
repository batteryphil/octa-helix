import json
import os
from pathlib import Path
from typing import List, Tuple

def load_beliefs(filename: str) -> List[dict]:
    with open(filename, 'r') as f:
        return json.load(f)

def extract_beliefs(belief: str) -> Tuple[str, str]:
    return (re.search(r'(?P<subject>.*?):\s*(?P<belief>.*)', belief).group('subject'), re.search(r'(?P<belief>.*?):\s*(?P<subject>.*?)', belief).group('subject'))

def find_conflicts(beliefs: List[dict]) -> List[Tuple[str, str]]:
    conflicts = []
    for i, b1 in enumerate(beliefs):
        for b2 in beliefs[i+1:]:
            if b1['subject'] == b2['subject']:
                conflicts.append((b1['belief'], b2['belief']))
    return conflicts

def resolve_conflict(belief1: str, belief2: str) -> str:
    if belief1 in belief2 or belief2 in belief1:
        return "The beliefs are too similar to resolve automatically."
    return "The beliefs conflict and require manual resolution."

def main():
    filename = 'beliefs.json'
    beliefs = load_beliefs(filename)
    conflicts = find_conflicts(beliefs)
    for conflict in conflicts:
        print(f"Conflicting beliefs: {conflict[0]} and {conflict[1]}")
        print(f"Resolution suggestion: {resolve_conflict(conflict[0], conflict[1])}")
        print()

if __name__ == '__main__':
    main()