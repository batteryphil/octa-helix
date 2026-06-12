import json
import jsonlines
import re
from pathlib import Path

def search_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        lines = list(f)
    results = sorted(enumerate(lines), key=lambda x: re.search(query, x[1]['text'], re.IGNORECASE), reverse=True)[:3]
    return [(lines[i]['id'], lines[i]['text']) for i, _ in results]

if __name__ == '__main__':
    query = input("Enter search query: ")
    top_results = search_knowledge(query)
    print("Top 3 results:")
    for i, (id, text) in enumerate(top_results, 1):
        print(f"{i}. {id} - {text}")