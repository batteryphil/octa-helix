import json
import jsonlines
import re

def load_curiosity_knowledge():
    with open('curiosity_knowledge.jsonl', 'r') as f:
        return [json.loads(line) for line in f]

def score_match(query, line):
    query_words = set(query.lower().split())
    line_words = set(line['title'].lower().split())
    return len(query_words & line_words)

def search_knowledge(query):
    knowledge = load_curiosity_knowledge()
    scored_lines = [(score_match(query, line), line) for line in knowledge]
    scored_lines.sort(reverse=True, key=lambda x: x[0])
    return scored_lines[:3]

def main():
    query = input("Enter a search query: ")
    results = search_knowledge(query)
    print("Top 3 matches:")
    for score, line in results:
        print(f"{score}: {line['title']} ({line['url']})")

if __name__ == '__main__':
    main()