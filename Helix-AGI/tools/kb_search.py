import json
import jsonlines
import os
import sys

def search_curiosity_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        for line in f:
            entry = json.loads(line)
            if query.lower() in entry['text'].lower():
                print(json.dumps(entry))
                count += 1
            if count >= 3:
                break

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python kb_search.py <search_query>")
        sys.exit(1)
    search_query = sys.argv[1]
    search_curiosity_knowledge(search_query)