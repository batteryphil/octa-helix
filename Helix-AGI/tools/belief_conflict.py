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

def belief_conflicts(beliefs: List[Dict[str, Any]]) -> List[str]:
    conflicts = []
    for i, belief1 in enumerate(beliefs):
        for belief2 in beliefs[i+1:]:
            if belief1['name'] == belief2['name']:
                if belief1['confidence'] > 0.7 and belief2['confidence'] > 0.7:
                    conflicts.append((belief1, belief2))
    return conflicts

def resolve_conflicts(conflicts: List[str]) -> List[str]:
    resolved = []
    for conflict in conflicts:
        belief1, belief2 = conflict
        resolved.append(f"Conflicting beliefs: {belief1['name']} (confidence: {belief1['confidence']}) and {belief2['name']} (confidence: {belief2['confidence']})")
    return resolved

def main():
    beliefs_file = 'beliefs.json'
    if not os.path.exists(beliefs_file):
        print("Beliefs file not found. Please create a beliefs.json file.")
        return
    
    beliefs = load_beliefs(beliefs_file)
    conflicts = belief_conflicts(beliefs)
    resolved_conflicts = resolve_conflicts(conflicts)