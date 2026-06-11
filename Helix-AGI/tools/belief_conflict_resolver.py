import json
import os
import sys

class BeliefConflictResolver:
    def __init__(self, belief_store):
        self.belief_store = belief_store

    def resolve_conflict(self, belief1, belief2):
        print(f"Conflict between {belief1} and {belief2}.")
        update1 = input(f"Update belief1 to: ").strip()
        update2 = input(f"Update belief2 to: ").strip()
        return update1, update2

    def resolve_conflicts(self):
        conflicts = []
        for i in range(len(self.belief_store)):
            for j in range(i+1, len(self.belief_store)):
                if self.belief_store[i]['confidence'] > 0.5 and self.belief_store[j]['confidence'] > 0.5:
                    if self.belief_store[i]['belief'] != self.belief_store[j]['belief']:
                        conflicts.append((self.resolve_conflict(self.belief_store[i]['belief'], self.belief_store[j]['belief'])))
        return conflicts

# Example usage
if __name__ == "__main__":
    with open('belief_store.json') as f:
        belief_store = json.load(f)

    resolver = BeliefConflictResolver(belief_store)
    conflicts = resolver.resolve_conflicts()
    print("Conflicts:")
    for conflict in conflicts:
        print(conflict)