import json
import jsonlines
import re
from pathlib import Path

def load_curated_knowledge(file_path):
    with file_path.open() as f:
        return json.load(f)

def search_knowledge(knowledge, query):
    results = []
    for item in knowledge:
        score = re.search(query, item['title'].lower()) is not None
        results.append((item, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3]

def main():
    curated_knowledge_file = Path('curated_knowledge.jsonl')
    query = 'What is the capital of France?'
    knowledge = load_curated_knowledge(curated_knowledge_file)
    results = search_knowledge(knowledge, query)
    for item, score in results:
        print(f"Title: {item['title']}, Score: {score}")

if __name__ == '__main__':
    main()