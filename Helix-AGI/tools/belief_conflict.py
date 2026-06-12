import json
import os
from pathlib import Path
from typing import List, Dict, Any

def load_beliefs(filename: str) -> List[Dict[str, Any]]:
    with open(filename, 'r') as f:
        return json.load(f)

def save_beliefs(filename: str, beliefs: List[Dict[str, Any]]):
    with open(filename, 'w') as f:
        json.dump(beliefs, f, indent=2)

def resolve_conflicts(beliefs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    resolved = []
    conflicts = []
    for i, b1 in enumerate(beliefs):
        for j, b2 in enumerate(beliefs[i+1:]):
            if b1['belief'] == b2['belief']:
                continue
            if b1['confidence'] > 0.5 and b2['confidence'] > 0.5:
                conflicts.append((b1, b2))
                resolved.append({'belief': b1['belief'], 'confidence': (b1['confidence'] + b2['confidence']) / 2})
    return resolved, conflicts

def main():
    beliefs_file = 'beliefs.json'
    if os.path.exists(beliefs_file):
        beliefs = load_beliefs(beliefs_file)
    else:
        beliefs = [
            {'belief': 'The sky is blue', 'confidence': 0.8},
            {'belief': 'The sky is green', 'confidence': 0.2},
            {'belief': '2 + 2 = 4', 'confidence': 1.0}
        ]
    
    resolved, conflicts = resolve_conf