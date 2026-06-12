import json
import jsonlines
import re
from pathlib import Path

def load_curiosity_knowledge():
    file_path = Path(__file__).parent / "curiosity_knowledge.jsonl"
    with file_path.open() as file:
        return [json.load(file) for line in file]

def calculate_relevance(query, entry):
    words = set(query.lower().split())
    entry_words = set(entry['title'].lower().split() + entry['content'].split())
    return len(words & entry_words)

def search_knowledge(query):
    knowledge = load_curiosity_knowledge()
    return sorted(knowledge, key=lambda entry: calculate_relevance(query, entry), reverse=True)[:3]

if __name__ == '__main__':
    query = "What is the capital of France?"
    results = search_knowledge(query)
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Title: {result['title']}")
        print(f"Content: {result['content']}")
        print()