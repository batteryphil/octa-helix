import json
import os
from pathlib import Path
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
import psutil

class BeliefConflictResolver:
    def __init__(self, belief_store_path: str):
        self.belief_store_path = belief_store_path
        self.beliefs = self.load_beliefs()

    def load_beliefs(self) -> List[dict]:
        with open(self.belief_store_path, 'r') as f:
            return json.load(f)

    def resolve_conflicts(self) -> None:
        conflicting_pairs = self.find_conflicting_pairs()
        for pair in conflicting_pairs:
            self.resolve_pair(pair)

    def find_conflicting_pairs(self) -> List[Tuple[int, int]]:
        beliefs = list(enumerate(self.beliefs))
        conflicting_pairs = []
        for i, (index1, belief1) in enumerate(beliefs):
            for index2, belief2 in beliefs[i+1:]:
                if belief1['name'] == belief2['name']:
                    conflicting_pairs.append((index1, index2))
        return conflicting_pairs

    def resolve_pair(self, pair: Tuple[int, int]) -> None:
        index1, index2 = pair
        belief1 = self.beliefs[index1]
        belief2 = self.beliefs[index2]
        if belief1['weight'] > belief2['weight']:
            del self.beliefs[index2]
        else:
            del self.beliefs[index1]