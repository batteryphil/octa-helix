import json
import os
from pathlib import Path
from typing import List, Tuple, Dict

def load_beliefs(file_path: Path) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def resolve_conflict(belief1: Dict, belief2: Dict) -> Dict:
    merged = {**belief1, **belief2}
    merged['confidence'] = (belief1['confidence'] + belief2['confidence']) / 2
    return merged

def find_conflicting_pairs(beliefs: List[Dict]) -> List[Tuple[Dict, Dict]]:
    pairs = []
    for i, belief1 in enumerate(beliefs):
        for belief2 in beliefs[i+1:]:
            if belief1['name'] == belief2['name'] and belief1['confidence'] > 0.7 and belief2['confidence'] > 0.7:
                pairs.append((belief1, belief2))
    return pairs

def resolve_belief_conflicts(beliefs: List[Dict]) -> List[Dict]:
    conflicts = find_conflicting_pairs(beliefs)
    resolved = []
    for pair in conflicts:
        resolved.append(resolve_conflict(pair[0], pair[1]))
    return resolved