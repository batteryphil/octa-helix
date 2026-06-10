import json
import jsonlines
import re
from pathlib import Path

def load_knowledge_base(file_path):
    with file_path.open() as f:
        return json.load(f)

def preprocess_query(query):
    return re.sub(r'\W+', ' ', query).strip().lower()

def search_knowledge_base(knowledge, query):
    query = preprocess_query(query)
    return [line for line in knowledge if query in line.lower()]

def main():
    knowledge_file = Path('curiosity_knowledge.jsonl')
    knowledge = load_knowledge_base(knowledge_file)
    
    query = input("Enter a search query: ")
    results = search_knowledge_base(knowledge, query)
    
    print("Top 3 relevant lines:")
    for i, line in enumerate(results[:3], 1):
        print(f"{i}. {line}")

if __name__ == '__main__':
    main()