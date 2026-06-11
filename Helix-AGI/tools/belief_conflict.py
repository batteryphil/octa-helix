import json
from pathlib import Path
from typing import List, Tuple, Dict

from bs4 import BeautifulSoup
import requests

# Mock belief store
belief_store = [
    {"id": 1, "text": "The sky is blue", "confidence": 0.9},
    {"id": 2, "text": "The sky is green", "confidence": 0.8},
    {"id": 3, "text": "2 + 2 = 4", "confidence": 0.95},
    {"id": 4, "text": "2 + 2 = 5", "confidence": 0.6},
]

def find_conflicting_beliefs(
    beliefs: List[Dict[str, float]], confidence_threshold: float
) -> List[Tuple[int, int]]:
    conflicting_pairs = []
    for i, belief1 in enumerate(beliefs):
        for j in range(i + 1, len(beliefs)):
            belief2 = beliefs[j]
            if belief1["confidence"] > confidence_threshold and belief2["confidence"] > confidence_threshold and belief1["text"] != belief2["text"]:
                conflicting_pairs.append((belief1["id"], belief2["id"]))
    return conflicting_pairs

# Example usage
conflicts = find_conflicting_beliefs(belief_store, 0.7)
print(json.dumps(conflicts, indent=2))