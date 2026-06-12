import json
from datetime import datetime
from typing import List, Dict

class Belief:
    def __init__(self, text: str, confidence: float, timestamp: str):
        self.text = text
        self.confidence = confidence
        self.timestamp = datetime.fromisoformat(timestamp)

    def __lt__(self, other: 'Belief'):
        return self.timestamp > other.timestamp

def resolve_conflicts(beliefs: List[Dict]) -> List[str]:
    resolved = {}
    for belief in sorted(beliefs, key=lambda b: (-b['confidence'], b['timestamp'])):
        key = belief['key']
        if key not in resolved:
            resolved[key] = belief
        else:
            if belief['confidence'] > resolved[key]['confidence'] or \
               (belief['confidence'] == resolved[key]['confidence'] and
                resolved[key]['timestamp'] > belief['timestamp']):
                resolved[key] = belief
    return list(resolved.values())