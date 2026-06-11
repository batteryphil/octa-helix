import json
import sys
import os

class BeliefConflictResolver:
    def __init__(self, belief_store):
        self.belief_store = belief_store

    def find_conflicts(self):
        conflicts = []
        beliefs = list(self.belief_store.items())
        for i in range(len(beliefs)):
            for j in range(i+1, len(beliefs)):
                if beliefs[i].confidence > 0.7 and beliefs[j].confidence > 0.7:
                    if self._are_conflicting(beliefs[i], beliefs[j]):
                        conflicts.append((beliefs[i], beliefs[j]))
        return conflicts

    def _are_conflicting(self, belief1, belief2):
        if belief1.topic == belief2.topic and belief1.aspect == belief2.aspect:
            return True
        return False

    def resolve_conflict(self, conflict):
        print(f"Conflict found: {conflict[0]} vs {conflict[1]}")
        choice = input("Choose which belief to update (0 or 1): ")
        if choice == '0':
            self.belief_store.update_confidence(conflict[0].topic, conflict[0].aspect, conflict[0].confidence * 0.9)
        elif choice == '1':
            self.belief_store.update_confidence(conflict[1].topic, conflict[1].aspect, conflict[1].confidence * 0.9)
        else:
            print("Invalid choice")

def main():
    with open('belief_store.json', 'r') as f:
        belief_store = json.load(f)

    resolver = BeliefConflictResolver(belief_store)
    conflicts = resolver.find