import json
from typing import List, Tuple

class Belief:
    def __init__(self, text: str, confidence: float):
        self.text = text
        self.confidence = confidence

    def __str__(self) -> str:
        return f"{self.text} ({self.confidence:.2f})"

def load_beliefs(file_path: str) -> List[Belief]:
    with open(file_path, "r") as f:
        beliefs = json.load(f)
    return [Belief(belief["text"], belief["confidence"]) for belief in beliefs]

def find_conflicting_pairs(beliefs: List[Belief], confidence_threshold: float) -> List[Tuple[Belief, Belief]]:
    conflicting_pairs = []
    for i, belief1 in enumerate(beliefs):
        for j in range(i + 1, len(beliefs)):
            belief2 = beliefs[j]
            if belief1.confidence > confidence_threshold and belief2.confidence > confidence_threshold:
                conflicting_pairs.append((belief1, belief2))
    return conflicting_pairs