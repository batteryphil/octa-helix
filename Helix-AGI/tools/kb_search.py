import json
import jsonlines
import requests
from bs4 import BeautifulSoup
import re
import pathlib

def search_curiosity_knowledge(query):
    url = f"https://www.google.com/search?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for g in soup.find_all("div", class_="yuRUdf"):
        title = g.find("h3", class_="r").text.strip()
        link = "https://www.google.com" + g.find("a", class_="yuRUdf-notranslate")["href"]
        snippet = g.find("div", class_="V7UhJnc").text.strip()
        results.append({"title": title, "link": link, "snippet": snippet})
    return results[:3]

def main():
    query = input("Enter a search query: ")
    results = search_curiosity_knowledge(query)
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Title: {result['title']}")
        print(f"  Link: {result['link']}")
        print(f"  Snippet: {result['snippet']}")

if __name__ == "__main__":
    main()