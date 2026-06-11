import json
import os
from pathlib import Path
from typing import List, Tuple

def load_beliefs(filename: str) -> List[dict]:
    with open(filename, 'r') as f:
        return json.load(f)

def belief_conflicts(beliefs: List[dict]) -> List[Tuple[str, str]]:
    conflicts = []
    for i, belief1 in enumerate(beliefs):
        for belief2 in beliefs[i+1:]:
            if belief1['confidence'] > 0.7 and belief2['confidence'] > 0.7:
                if belief1['text'].strip().lower() != belief2['text'].strip().lower():
                    conflicts.append((belief1['id'], belief2['id']))
    return conflicts

def reconcile_conflict(conflict: Tuple[str, str]) -> str:
    belief1_id, belief2_id = conflict
    belief1 = next(b for b in beliefs if b['id'] == belief1_id)
    belief2 = next(b for b in beliefs if b['id'] == belief2_id)
    return f"{belief1['text']} and {belief2['text']} are in conflict"

def main():
    filename = Path("beliefs.json")
    beliefs = load_beliefs(filename)
    conflicts = belief_conflicts(beliefs)
    for conflict in conflicts:
        print(reconcile_conflict(conflict))

if __name__ == "__main__":
    main()