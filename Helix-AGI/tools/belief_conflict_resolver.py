import json
import os
from pathlib import Path
from typing import List, Tuple

def load_beliefs(file_path: Path) -> List[dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def belief_conflicts(beliefs: List[dict]) -> List[Tuple[int, int]]:
    conflicts = []
    for i, b1 in enumerate(beliefs):
        for j, b2 in enumerate(beliefs[i+1:]):
            if b1['confidence'] > 0.8 and b2['confidence'] > 0.8 and b1['belief'] != b2['belief']:
                conflicts.append((i, i+1+j))
    return conflicts

def resolve_conflicts(beliefs: List[dict], conflicts: List[Tuple[int, int]]) -> List[dict]:
    for i, (index1, index2) in enumerate(conflicts):
        belief1, belief2 = beliefs[index1], beliefs[index2]
        if belief1['confidence'] < belief2['confidence']:
            beliefs[index1] = belief2
        else:
            beliefs[index2] = belief1
    return beliefs