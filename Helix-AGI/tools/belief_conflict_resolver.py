import json
from typing import List, Dict

class BeliefConflictResolver:
    def __init__(self, resolution_strategy: str):
        self.resolution_strategy = resolution_strategy
        self.conflicts = []

    def add_conflict(self, belief1: Dict, belief2: Dict):
        self.conflicts.append((belief1, belief2))

    def resolve_conflicts(self) -> List[Dict]:
        resolved_beliefs = []
        for belief1, belief2 in self.conflicts:
            if self.resolution_strategy == 'favor_higher_confidence':
                if belief1['confidence'] > belief2['confidence']:
                    resolved_beliefs.append(belief1)
                else:
                    resolved_beliefs.append(belief2)
            else:
                raise ValueError(f"Unknown resolution strategy: {self.resolution_strategy}")
        return resolved_beliefs

    def log_resolved_beliefs(self, resolved_beliefs: List[Dict]):
        with open('resolved_beliefs.json', 'w') as f:
            json.dump(resolved_beliefs, f, indent=2)

def main():
    resolver = BeliefConflictResolver('favor_higher_confidence')
    belief1 = {'name': 'belief1', 'confidence': 0.8}
    belief2 = {'name': 'belief2', 'confidence': 0.6}
    resolver.add_conflict(belief1, belief2)
    resolved_beliefs = resolver.resolve_conflicts()
    resolver.log_resolved_beliefs(resolved_beliefs)
    print("Resolved beliefs logged successfully.")

if __name__ == '__main__':
    main()