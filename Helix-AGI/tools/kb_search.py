import json
import re
import pathlib
from typing import List, Tuple

def load_curiosity_knowledge_file() -> List[str]:
    file_path = pathlib.Path(__file__).parent / 'curiosity_knowledge.txt'
    with open(file_path, 'r') as f:
        return f.read().splitlines()

def preprocess_query(query: str) -> Tuple[str, str]:
    query = query.strip().lower()
    query = re.sub(r'\W+', ' ', query).strip()
    return query, ' '.join(query.split()[0:2])

def search_curiosity_knowledge(query: str) -> List[str]:
    query, context = preprocess_query(query)
    knowledge = load_curiosity_knowledge_file()
    relevant_lines = [line for line in knowledge if query in line.lower() or context in line.lower()]
    return relevant_lines[:3]

def main():
    if __name__ == '__main__':
        query = input("Enter a search query: ")
        results = search_curiosity_knowledge(query)
        print("Top 3 relevant lines:")
        for i, line in enumerate(results, 1):
            print(f"{i}. {line}")

if __name__ == '__main__':
    main()