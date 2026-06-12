import json
import requests
from bs4 import BeautifulSoup
import re
import pathlib

def get_facts_from_wikipedia(query):
    url = f"https://en.wikipedia.org/wiki/{query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    facts = []
    for p in soup.select("p"):
        text = p.get_text(strip=True)
        if text:
            facts.append(text)
    return facts

def check_fact(statement, query):
    facts = get_facts_from_wikipedia(query)
    score = 0
    for fact in facts:
        if re.search(statement, fact, re.IGNORECASE):
            score += 1
    return score / len(facts)

def main():
    statement = input("Enter a statement to check: ")
    query = input("Enter a topic to check facts about: ")
    confidence = check_fact(statement, query)
    print(f"Confidence score: {confidence}")

if __name__ == "__main__":
    main()