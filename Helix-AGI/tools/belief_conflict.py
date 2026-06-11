import json
from pathlib import Path

def load_beliefs(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_beliefs(file_path, beliefs):
    with open(file_path, 'w') as f:
        json.dump(beliefs, f, indent=2)

def belief_conflict(beliefs):
    conflicts = []
    for i, belief1 in enumerate(beliefs):
        for belief2 in beliefs[i+1:]:
            if belief1['confidence'] > 0.8 and belief2['confidence'] > 0.8:
                if belief1['statement'] == 'not ' + belief2['statement']:
                    conflicts.append((belief1, belief2))
                elif belief2['statement'] == 'not ' + belief1['statement']:
                    conflicts.append((belief2, belief1))
                elif belief1['statement'].startswith('and ') and belief2['statement'].startswith('and '):
                    if belief1['statement'].endswith(' not') and belief2['statement'].endswith(' not'):
                        conflicts.append((belief1, belief2))
                    elif belief2['statement'].endswith(' not'):
                        conflicts.append((belief2, belief1))
                    elif belief1['statement'].endswith(' not'):
                        conflicts.append((belief1, belief2))
                elif belief1['statement'].endswith(' not') and belief2['statement'].endswith(' not'):
                    conflicts.append((belief1, belief2))
    return conflicts