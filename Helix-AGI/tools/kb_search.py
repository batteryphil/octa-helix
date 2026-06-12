import json
import jsonlines
import re
from pathlib import Path

def search_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        lines = [line for line in f if any(term in line['text'] for term in query.split())]
        lines.sort(key=lambda x: sum(term in x['text'] for term in query.split()), reverse=True)
        return lines[:3]

def main():
    query = input("Enter a search query: ")
    results = search_knowledge(query)
    for i, result in enumerate(results, 1):
        print(f"Result {i}: {result['text']}")

if __name__ == '__main__':
    main()