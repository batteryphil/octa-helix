import json
import sys
import os
from typing import List, Dict

def load_beliefs(filename: str) -> List[Dict]:
    with open(filename, 'r') as f:
        return json.load(f)

def save_beliefs(filename: str, beliefs: List[Dict]):
    with open(filename, 'w') as f:
        json.dump(beliefs, f, indent=2)

def resolve_conflict(beliefs: List[Dict]) -> List[Dict]:
    resolved = []
    for i, b1 in enumerate(beliefs):
        for j, b2 in enumerate(beliefs[i+1:]):
            if b1['belief'] == b2['belief']:
                continue
            if b1['confidence'] > 0.7 and b2['confidence'] > 0.7:
                print(f"Conflict detected between:\n{b1}\n{b2}")
                resolution = input("Enter resolution (update 1st or 2nd belief or both): ")
                if '1' in resolution:
                    b1['confidence'] = 0.5
                if '2' in resolution:
                    b2['confidence'] = 0.5
                print(f"Resolved conflict:\n{b1}\n{b2}")
            resolved.append(b1)
    return resolved

def main():
    beliefs_file = 'beliefs.json'
    if len(sys.argv) > 1:
        beliefs_file = sys.argv[1]
    
    beliefs = load_beliefs(beliefs_file)
    resolved_beliefs = resolve_conflict(beliefs)
    save_beliefs(beliefs_file, resolved_beliefs)