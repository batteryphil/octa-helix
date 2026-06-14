import json
from pathlib import Path
from typing import List, Tuple

def load_beliefs(file_path: Path) -> List[Tuple[str, float]]:
    with open(file_path, 'r') as f:
        beliefs = json.load(f)
    return [(belief['statement'], belief['confidence']) for belief in beliefs if 'confidence' in belief]

def find_conflicting_beliefs(beliefs: List[Tuple[str, float]], confidence_threshold: float) -> List[Tuple[Tuple[str, float], Tuple[str, float]]]:
    conflicting_pairs = []
    for i, (belief1, confidence1) in enumerate(beliefs):
        for j, (belief2, confidence2) in enumerate(beliefs[i+1:], start=i+1):
            if confidence1 > confidence_threshold and confidence2 > confidence_threshold and not belief1 == belief2:
                conflicting_pairs.append(((belief1, confidence1), (belief2, confidence2)))
    return conflicting_pairs

def main():
    beliefs_file = Path('beliefs.json')
    confidence_threshold = 0.8
    
    beliefs = load_beliefs(beliefs_file)
    conflicting_pairs = find_conflicting_beliefs(beliefs, confidence_threshold)
    
    print(json.dumps([{'belief1': (statement, confidence), 'belief2': (statement, confidence)} for (statement, confidence), (statement, confidence) in conflicting_pairs], indent=2))

if __name__ == '__main__':
    main()