import json
import jsonlines
import re

def search_knowledge(keyword, file_path):
    with jsonlines.open(file_path) as f:
        results = []
        for line in f:
            entry = json.loads(line)
            if keyword in entry['title'] or keyword in entry['content']:
                results.append(entry)
            if len(results) >= 3:
                break
    return results

def main():
    keyword = input("Enter a search keyword: ")
    file_path = "curiosity_knowledge.jsonl"
    results = search_knowledge(keyword, file_path)
    for i, result in enumerate(results, start=1):
        print(f"Result {i}:")
        print(f"Title: {result['title']}")
        print(f"Content: {result['content']}")
        print()

if __name__ == "__main__":
    main()