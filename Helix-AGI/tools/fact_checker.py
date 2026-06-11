import requests
import json
from bs4 import BeautifulSoup
import re
import json
import pathlib

def fetch_wikipedia(page_title):
    url = f"https://en.wikipedia.org/wiki/{page_title}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})
    paragraphs = content.find_all("p")
    return "\n".join([p.text for p in paragraphs])

def fact_checker(statement):
    statement = statement.lower()
    statement = re.sub(r"[^a-zA-Z0-9\s]", "", statement)
    
    fact = None
    for word in statement.split():
        fact = fetch_wikipedia(word)
        if fact:
            break
    
    if fact:
        return "verified"
    else:
        return "unverified"

if __name__ == "__main__":
    statement = "The capital of France is Paris"
    print(fact_checker(statement))