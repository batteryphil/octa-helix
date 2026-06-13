import json
import jsonlines
import re
from pathlib import Path

def search_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        results = []
        for line in f:
            entry = json.loads(line)
            if query.lower() in entry['text'].lower():
                results.append(entry)
        results = sorted(results, key=lambda x: re.search(query, x['text'], re.IGNORECASE), reverse=True)
        return results[:3]

def main():
    query = input("Enter a search query: ")
    results = search_knowledge(query)
    print(f"Top 3 matching knowledge entries:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text']}")

if __name__ == '__main__':
    main()