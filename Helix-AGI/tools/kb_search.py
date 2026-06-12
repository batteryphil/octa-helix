import json
import jsonlines
import re
import pathlib

def load_knowledge_base():
    knowledge_path = pathlib.Path(__file__).parent / "curiosity_knowledge.jsonl"
    with knowledge_path.open() as file:
        return list(jsonlines.Reader(file))

def search_knowledge(knowledge, keyword):
    results = []
    for entry in knowledge:
        if keyword in entry['title'].lower() or keyword in entry['content'].lower():
            results.append(entry)
        if len(results) >= 3:
            break
    return results

def main():
    keyword = input("Enter a search keyword: ")
    knowledge = load_knowledge_base()
    results = search_knowledge(knowledge, keyword)
    print(f"Top 3 relevant entries for '{keyword}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   {result['content']}")

if __name__ == '__main__':
    main()