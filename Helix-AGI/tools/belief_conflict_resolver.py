import json
import random

def resolve_conflicts(beliefs):
    confidences = [belief['confidence'] for belief in beliefs]
    avg_confidence = sum(confidences) / len(confidences)
    return [belief for belief in beliefs if belief['confidence'] > avg_confidence]

def main():
    beliefs = [
        {"belief": "The sky is blue", "confidence": 0.8},
        {"belief": "The sky is green", "confidence": 0.2},
        {"belief": "The sky is yellow", "confidence": 0.3},
    ]
    resolved_beliefs = resolve_conflicts(beliefs)
    print(json.dumps(resolved_beliefs, indent=2))

if __name__ == '__main__':
    main()