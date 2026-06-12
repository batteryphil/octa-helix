import json
import os
from typing import List, Tuple

def load_beliefs(filename: str) -> List[Tuple[str, float]]:
    with open(filename, 'r') as f:
        beliefs = json.load(f)
    return [(belief, confidence) for belief, confidence in beliefs.items() if confidence > 0.8]

def find_conflicts(beliefs: List[Tuple[str, float]]) -> List[Tuple[str, str]]:
    conflicts = []
    for i, (belief1, _) in enumerate(beliefs):
        for belief2, _ in beliefs[i+1:]:
            if belief1[0] != belief2[0] and belief1[0] != 'meta' and belief2[0] != 'meta':
                if belief1[0] == 'meta' or belief2[0] == 'meta':
                    conflicts.append((belief1[0], belief2[0]))
                elif belief1[0] != belief2[0]:
                    conflicts.append((belief1[0], belief2[0]))
    return conflicts

# Example usage
if __name__ == "__main__":
    filename = "beliefs.json"
    beliefs = load_beliefs(filename)
    conflicts = find_conflicts(beliefs)
    print(conflicts)