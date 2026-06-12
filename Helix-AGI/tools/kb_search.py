import json
import jsonlines
import re
from pathlib import Path

def search_curiosity_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        results = []
        for line in f:
            entry = json.loads(line)
            if query.lower() in entry['text'].lower():
                results.append(entry)
            if len(results) >= 3:
                break
    return results

def main():
    query = input("Enter a search query: ")
    results = search_curiosity_knowledge(query)
    for i, result in enumerate(results, start=1):
        print(f"Result {i}: {result['text']}")

if __name__ == '__main__':
    main()