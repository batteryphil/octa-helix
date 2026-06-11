import json
from pathlib import Path
from typing import List, Tuple

def load_beliefs(filename: Path) -> List[dict]:
    with open(filename, 'r') as f:
        return json.load(f)

def belief_confidence(belief: dict) -> float:
    return belief['confidence']

def resolve_conflict(belief1: dict, belief2: dict) -> dict:
    merged_belief = {
        **belief1,
        **belief2,
        'confidence': (belief_confidence(belief1) + belief_confidence(belief2)) / 2
    }
    return merged_belief

def find_conflicts_and_resolve(beliefs: List[dict]) -> List[dict]:
    conflicts = []
    resolved_beliefs = []
    for i in range(len(beliefs)):
        for j in range(i + 1, len(beliefs)):
            if belief_confidence(beliefs[i]) > 0.8 and belief_confidence(beliefs[j]) > 0.8:
                if beliefs[i]['belief'] == beliefs[j]['belief']:
                    conflicts.append((beliefs[i], beliefs[j]))
                    resolved_beliefs.append(resolve_conflict(beliefs[i], beliefs[j]))
    return conflicts, resolved_beliefs

def save_beliefs(filename: Path, beliefs: List[dict]):
    with open(filename, 'w') as f:
        json.dump(beliefs, f, indent=2)

if __name__ == '__main__':
    beliefs_file = Path('beliefs.json')
    beliefs = load_beliefs(beliefs_file)
    conflicts, resolved = find_conflicts_and_resolve(beliefs)