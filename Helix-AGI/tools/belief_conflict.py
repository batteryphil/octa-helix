import json
from typing import List, Tuple

def belief_conflict(beliefs: List[dict]) -> List[Tuple[str, str]]:
    conflicts = []
    for i, belief1 in enumerate(beliefs):
        for belief2 in beliefs[i+1:]:
            if belief1['confidence'] > 0.8 and belief2['confidence'] > 0.8:
                if belief1['statement'] != belief2['statement'] and belief1['statement'] != 'NOT ' + belief2['statement'] and belief2['statement'] != 'NOT ' + belief1['statement']:
                    conflicts.append((belief1['statement'], belief2['statement']))
    return conflicts

def load_beliefs(file_path: str) -> List[dict]:
    with open(file_path, 'r') as f:
        return json.load(f)

def save_beliefs(file_path: str, beliefs: List[dict]):
    with open(file_path, 'w') as f:
        json.dump(beliefs, f, indent=2)

def main():
    beliefs_file = 'beliefs.json'
    beliefs = load_beliefs(beliefs_file)
    conflicts = belief_conflict(beliefs)
    if conflicts:
        print("Belief conflicts found:")
        for conflict in conflicts:
            print(conflict)
    else:
        print("No belief conflicts found.")
    save_beliefs(beliefs_file, beliefs)

if __name__ == '__main__':
    main()