import json
import jsonlines
import re
from pathlib import Path

def search_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        facts = [fact for fact in f if query.lower() in fact['text'].lower()]
    facts = sorted(facts, key=lambda x: re.search(query, x['text']).start(), reverse=True)
    return facts[:3]

def main():
    query = input("Enter a search query: ")
    results = search_knowledge(query)
    for i, fact in enumerate(results, 1):
        print(f"Result {i}: {fact['text']}")

if __name__ == '__main__':
    main()