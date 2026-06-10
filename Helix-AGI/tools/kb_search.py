import json
import jsonlines
import re
import sys
from pathlib import Path

def load_knowledge_base():
    file_path = Path(__file__).parent / "curiosity_knowledge.jsonl"
    with file_path.open() as file:
        return list(jsonlines.Reader(file))

def search_insights(keyword, knowledge_base):
    def relevance_score(insight):
        return re.search(keyword.lower(), insight['title'].lower()) is not None

    return sorted(filter(relevance_score, knowledge_base), reverse=True)[:3]

def main():
    if len(sys.argv) != 2:
        print("Usage: python kb_search.py <keyword>")
        return
    
    keyword = sys.argv[1]
    knowledge_base = load_knowledge_base()
    relevant_insights = search_insights(keyword, knowledge_base)
    
    print(f"Top 3 relevant insights for '{keyword}':")
    for i, insight in enumerate(relevant_insights, 1):
        print(f"{i}. {insight['title']} - {insight['summary']}")

if __name__ == "__main__":
    main()