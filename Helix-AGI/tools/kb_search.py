import json
import jsonlines
import re
from pathlib import Path

def load_knowledge_base():
    knowledge_path = Path(__file__).parent / "curiosity_knowledge.jsonl"
    with knowledge_path.open() as file:
        return list(jsonlines.Reader(file))

def preprocess_query(query):
    return re.sub(r"\s+", " ", query.strip().lower())

def calculate_relevance(entry, query):
    entry_text = entry["text"].lower()
    query_text = query.lower()
    return entry_text.count(query_text) / len(entry_text.split())

def search_knowledge_base(knowledge, query):
    query = preprocess_query(query)
    return sorted(knowledge, key=lambda x: calculate_relevance(x["text"], query), reverse=True)[:3]

if __name__ == "__main__":
    knowledge = load_knowledge_base()
    query = "What is the capital of France?"
    results = search_knowledge_base(knowledge, query)
    print(f"Top 3 relevant entries for '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['text']}")