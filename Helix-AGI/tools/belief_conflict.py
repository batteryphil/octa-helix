import json
import os
from pathlib import Path
from typing import List, Tuple

def load_beliefs(filename: str) -> List[Tuple[str, float]]:
    with open(filename, 'r') as f:
        beliefs = json.load(f)
    return [(belief['statement'], belief['confidence']) for belief in beliefs if belief['confidence'] > 0.8]

def find_conflicts(beliefs: List[Tuple[str, float]]) -> List[Tuple[str, str, float, float]]:
    conflicts = []
    for i in range(len(beliefs)):
        for j in range(i+1, len(beliefs)):
            if beliefs[i][0] != beliefs[j][0] and (beliefs[i][0] or beliefs[j][0]).lower() in (beliefs[j][0] or beliefs[i][0]).lower():
                conflicts.append((beliefs[i][0], beliefs[j][0], beliefs[i][1], beliefs[j][1]))
    return conflicts

def save_conflicts(conflicts: List[Tuple[str, str, float, float]], filename: str):
    Path(filename).write_text(json.dumps(conflicts, indent=2))

def main():
    beliefs_file = 'beliefs.json'
    conflicts_file = 'belief_conflicts.json'
    
    beliefs = load_beliefs(beliefs_file)
    conflicts = find_conflicts(beliefs)
    save_conflicts(conflicts, conflicts_file)
    
    print(f"Found {len(conflicts)} belief conflicts and saved to {conflicts_file}")

if __name__ == '__main__':
    main()