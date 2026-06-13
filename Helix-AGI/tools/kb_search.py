import json
import jsonlines
import re
from pathlib import Path
from typing import List, Dict

def load_curiosity_knowledge() -> List[Dict]:
    file_path = Path(__file__).parent / "curiosity_knowledge.jsonl"
    with jsonlines.open(file_path) as f:
        return [json.loads(line) for line in f]

def preprocess_query(query: str) -> List[str]:
    # Convert to lowercase
    query = query.lower()
    # Remove punctuation
    query = re.sub(r'[^\w\s]', '', query)
    # Tokenize
    return query.split()

def calculate_similarity(query_tokens: List[str], knowledge_entry: Dict) -> float:
    # Implement a simple similarity metric
    # For example, calculate the Jaccard similarity between query tokens and entry keywords
    query_set = set(query_tokens)
    entry_keywords_set = set(knowledge_entry.get('keywords', []))
    return len(query_set & entry_keywords_set) / len(query_set | entry_keywords_set)

def search_knowledge(query: str, knowledge: List[Dict], top_k: int = 3) -> List[Dict]:
    query_tokens = preprocess_query(query)
    return sorted(knowledge, key=lambda entry: calculate_similarity(query_tokens, entry), reverse=True)[:top_k]

def main():
    query = "What is the capital of France?"
    knowledge = load_curiosity_knowledge()
    results = search_knowledge(query, knowledge)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")

if __name__ == '__main__':
    main()