import json
import jsonlines
import re
from pathlib import Path

def search_curiosity_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        lines = list(f)
    query = query.lower()
    matches = [line for line in lines if query in line.lower()]
    if len(matches) > 3:
        matches = matches[:3]
    return matches

def main():
    query = input("Enter a search query: ")
    results = search_curiosity_knowledge(query)
    print("\nSearch results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result}")

if __name__ == '__main__':
    main()