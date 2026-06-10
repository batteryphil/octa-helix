import json
import jsonlines
import sys
from pathlib import Path

def search_curiosity_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        entries = [entry for entry in f]
    relevant_entries = sorted(entries, key=lambda entry: similarity(entry['text'], query), reverse=True)[:3]
    return relevant_entries

def similarity(text1, text2):
    words1 = text1.split()
    words2 = text2.split()
    return sum(1 for word in words1 if word in words2)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
        relevant_entries = search_curiosity_knowledge(query)
        for entry in relevant_entries:
            print(json.dumps(entry, indent=2))
    else:
        print("Please provide a search query.")