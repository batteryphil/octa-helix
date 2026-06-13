import json
from pathlib import Path
from typing import List, Tuple

def load_beliefs(file_path: Path) -> List[Tuple[str, float]]:
    with open(file_path, 'r') as f:
        beliefs = json.load(f)
    return [(belief['statement'], belief['confidence']) for belief in beliefs if 'confidence' in belief]

def find_conflict_pairs(beliefs: List[Tuple[str, float]], confidence_threshold: float) -> List[Tuple[str, str, float]]:
    conflicts = []
    for i in range(len(beliefs)):
        for j in range(i+1, len(beliefs)):
            if beliefs[i][1] > confidence_threshold and beliefs[j][1] > confidence_threshold and not beliefs[i][0] == beliefs[j][0]:
                conflicts.append((beliefs[i][0], beliefs[j][0], max(beliefs[i][1], beliefs[j][1])))
    return conflicts

def main():
    beliefs_file = Path('beliefs.json')
    confidence_threshold = 0.8
    beliefs = load_beliefs(beliefs_file)
    conflicts = find_conflict_pairs(beliefs, confidence_threshold)
    print(json.dumps(conflicts, indent=2))

if __name__ == '__main__':
    main()