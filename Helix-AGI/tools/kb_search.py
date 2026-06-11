import json
import jsonlines
import sys
from pathlib import Path

def search_knowledge(query):
    with jsonlines.open('curiosity_knowledge.jsonl', 'r') as f:
        knowledge = [entry for entry in f]
    
    # Tokenize the query
    tokens = query.lower().split()
    
    # Calculate cosine similarity between query and each knowledge entry
    scores = []
    for entry in knowledge:
        entry_tokens = entry['text'].lower().split()
        score = sum(1 for token in tokens if token in entry_tokens)
        scores.append((entry, score))
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 3 results
    return [entry for entry, _ in scores[:3]]

if __name__ == '__main__':
    query = ' '.join(sys.argv[1:])
    results = search_knowledge(query)
    print(json.dumps(results, indent=2))