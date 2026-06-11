import json
import requests
from bs4 import BeautifulSoup
import psutil
import re
import pathlib

def search_knowledge(query):
    # Search the curiosity knowledge database
    url = f"https://www.example.com/api/search?q={query}"
    response = requests.get(url)
    data = json.loads(response.text)
    
    # Calculate relevance scores for each result
    results = []
    for item in data["results"]:
        title = item["title"]
        content = item["content"]
        score = calculate_similarity(query, title) * 0.7 + calculate_similarity(query, content) * 0.3
        results.append({"title": title, "content": content, "score": score})
    
    # Sort results by relevance score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top 3 results
    return results[:3]

def calculate_similarity(query, text):
    # Calculate similarity between query and text using simple string matching
    query = query.lower()
    text = text.lower()
    words = re.findall(r'\w+', query)
    score = sum(1 for word in words if word in text)
    return score / len(words)

def main():
    query = input("Enter a search query: ")
    results = search_knowledge(query)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} - Score: {result['score']}")
        print(result['content'])

if __name__ == '__main__':
    main()