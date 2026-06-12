import json
from typing import List, Tuple

def resolve_conflicts(beliefs: List[Tuple[str, float]]) -> List[str]:
    resolved_beliefs = []
    for belief, confidence in beliefs:
        for resolved in resolved_beliefs:
            resolved_confidence, _ = resolved
            if confidence > resolved_confidence:
                resolved[0] = belief
                resolved[1] = confidence
                break
        else:
            resolved_beliefs.append((belief, confidence))
    return [belief for belief, _ in resolved_beliefs] if resolved_beliefs else ['conflict']

def main():
    beliefs = [
        ('The sky is blue', 0.8),
        ('The sky is green', 0.2),
        ('The sky is blue', 0.6),
        ('The sky is green', 0.4),
    ]
    resolved = resolve_conflicts(beliefs)
    print(json.dumps(resolved))

if __name__ == '__main__':
    main()