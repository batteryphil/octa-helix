import json
from pathlib import Path
from typing import List, Tuple

def load_beliefs(filename: Path) -> List[Tuple[float, str]]:
    with open(filename, 'r') as f:
        beliefs = json.load(f)
    return [(float(belief['confidence']), belief['text']) for belief in beliefs]

def resolve_conflict(beliefs: List[Tuple[float, str]]) -> Tuple[float, str]:
    for i in range(len(beliefs) - 1):
        for j in range(i + 1, len(beliefs)):
            if beliefs[i][0] > 0.8 and beliefs[j][0] > 0.8 and beliefs[i][1] != beliefs[j][1]:
                return (1.0, f"Belief: {beliefs[i][1]}, Conflict: {beliefs[j][1]}")
    return (0.0, "No conflicting beliefs found")

def main():
    beliefs_file = Path('beliefs.json')
    beliefs = load_beliefs(beliefs_file)
    result = resolve_conflict(beliefs)
    print(result)

if __name__ == '__main__':
    main()