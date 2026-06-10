import json
from pathlib import Path
from typing import List, Tuple

def load_beliefs(file_path: Path) -> List[dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def belief_conflicts(confidences: List[Tuple[str, float]]) -> List[Tuple[str, str, float]]:
    conflicts = []
    for i, (belief1, c1) in enumerate(confidences):
        for belief2, c2 in confidences[i+1:]:
            if c1 > 0.7 and c2 > 0.7 and belief1 != belief2 and not belief1 == belief2:
                conflicts.append((belief1, belief2, (c1 + c2) / 2))
    return conflicts

def main():
    beliefs_file = Path('beliefs.json')
    beliefs = load_beliefs(beliefs_file)

    confidences = [(belief, float(confidence)) for belief, confidence in beliefs.items()]
    conflicts = belief_conflicts(confidences)

    print(json.dumps(conflicts, indent=2))

if __name__ == '__main__':
    main()