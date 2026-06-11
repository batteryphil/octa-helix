import json
import jsonlines
import re
from pathlib import Path

def search_curiosity_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        results = []
        for line in f:
            entry = json.loads(line)
            if query.lower() in entry['query_keywords'].lower() or query.lower() in entry['title'].lower():
                results.append(entry)
                if len(results) >= 3:
                    break
        return results

def main():
    query = input("Enter a search query: ")
    results = search_curiosity_knowledge(query)
    print(f"Top 3 relevant results for '{query}':")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']} - {result['description']}")

if __name__ == '__main__':
    main()