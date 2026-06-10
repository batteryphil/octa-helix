import json
import requests
from bs4 import BeautifulSoup
import psutil
import re
import pathlib
from typing import List, Dict, Tuple

class KnowledgeIntegrator:
    def __init__(self, belief_store: Dict[str, Dict], belief_conflict_resolver, curiosity_tracker, knowledge_searcher):
        self.belief_store = belief_store
        self.belief_conflict_resolver = belief_conflict_resolver
        self.curiosity_tracker = curiosity_tracker
        self.knowledge_searcher = knowledge_searcher

    def resolve_conflicts(self):
        conflicts = self.belief_conflict_resolver.get_conflicts()
        for conflict in conflicts:
            evidence = self.knowledge_searcher.search(conflict)
            if evidence:
                resolution = self.belief_conflict_resolver.resolve(conflict, evidence)
                self.belief_store.update(resolution)
            else:
                print(f"No evidence found for conflict: {conflict}")

    def prioritize_conflicts(self) -> List[Tuple[str, float]]:
        conflicts = self.belief_conflict_resolver.get_conflicts()
        return [(conflict, self.curiosity_tracker.get_curiosity(conflict)) for conflict in conflicts]

def main():
    with open("belief_store.json") as f:
        belief_store = json.load(f)

    with open("belief_conflict_resolver.json") as f:
        belief_conflict_resolver = json.load(f)

    with open("curiosity_tracker.json") as f:
        curiosity_tracker = json.load(f)

    with open("knowledge_searcher.json") as f:
        knowledge_searcher = json.load(f)

    integrator = KnowledgeIntegrator(belief_store, belief_conflict_resolver, curiosity_tracker, knowledge_searcher)