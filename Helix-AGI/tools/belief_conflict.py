import json
from typing import List, Dict, Tuple

class Belief:
    def __init__(self, label: str, confidence: float):
        self.label = label
        self.confidence = confidence

    def __str__(self) -> str:
        return f"{self.label} ({self.confidence:.2f})"

def load_beliefs(file_path: str) -> List[Belief]:
    with open(file_path, "r") as f:
        beliefs = json.load(f)
    return [Belief(label=b["label"], confidence=b["confidence"]) for b in beliefs]

def find_conflicts(beliefs: List[Belief]) -> List[Tuple[Belief, Belief]]:
    conflicts = []
    for i, b1 in enumerate(beliefs):
        for j, b2 in enumerate(beliefs[i+1:]):
            if abs(b1.confidence - b2.confidence) < 0.1 and b1.label != b2.label:
                conflicts.append((b1, b2))
    return conflicts

def resolve_conflict(b1: Belief, b2: Belief) -> Belief:
    return Belief(label=f"{b1.label} and {b2.label}", confidence=(b1.confidence + b2.confidence) / 2)

def main():
    beliefs_file = "beliefs.json"
    beliefs = load_beliefs(beliefs_file)
    conflicts = find_conflicts(beliefs)
    if conflicts:
        print("Conflicts found:")
        for c in conflicts:
            print(f"{c[0]} and {c[1]}")
        new_belief = resolve_conflict