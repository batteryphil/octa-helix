import json
import jsonlines
import re
from pathlib import Path
from typing import List, Dict

def load_knowledge_base(file_path: Path) -> List[Dict]:
    with file_path.open() as f:
        return list(jsonlines.Reader(f))

def preprocess_query(query: str) -> str:
    return re.sub(r'\W+', ' ', query).strip().lower()

def calculate_similarity(query: str, entry: Dict) -> float:
    query_words = set(preprocess_query(query).split())
    entry_words = set(entry['text'].split())
    return len(query_words & entry_words) / len(query_words | entry_words)

def search_knowledge_base(query: str, knowledge_base: List[Dict], top_k: int = 3) -> List[Dict]:
    query = preprocess_query(query)
    return sorted(knowledge_base, key=lambda x: calculate_similarity(query, x), reverse=True)[:top_k]

def main():
    knowledge_base_file = Path('curiosity_knowledge.jsonl')
    query = 'What is the capital of France?'
    knowledge_base = load_knowledge_base(knowledge_base_file)
    results = search_knowledge_base(query, knowledge_base, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"Result {i}: {result['text']}")

if __name__ == '__main__':
    main()