import json
from pathlib import Path

def resolve_conflict(belief1, belief2):
    if belief1['confidence'] > belief2['confidence']:
        return belief1
    else:
        return belief2

def resolve_conflict_resolver(belief_store):
    resolved_store = {}
    for topic, beliefs in belief_store.items():
        resolved_beliefs = []
        for belief in beliefs:
            conflicting_beliefs = [b for b in beliefs if b['id'] != belief['id'] and b['conflict_with'] == belief['id']]
            if conflicting_beliefs:
                most_confident = max(conflicting_beliefs, key=lambda b: b['confidence'])
                resolved_beliefs.append(resolve_conflict(belief, most_confident))
            else:
                resolved_beliefs.append(belief)
        resolved_store[topic] = resolved_beliefs