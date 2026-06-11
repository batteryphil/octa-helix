import requests
from bs4 import BeautifulSoup
import json
import re
import pathlib

def fact_check(statement):
    # Query curated knowledge base (example: Wikipedia)
    url = "https://en.wikipedia.org/wiki/Main_Page"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"})

    # Parse and extract relevant facts
    facts = []
    for p in content.find_all("p"):
        fact = re.sub(r"\s+", " ", p.text.strip()).strip()
        if fact:
            facts.append(fact)

    # Compare statement against facts
    for fact in facts:
        if statement.lower() in fact.lower():
            return "fact-checked"
        elif any(disputed in fact.lower() for disputed in ["controversy", "disputed", "debate"]):
            return "fact-checked: disputed"
    return "fact-checked: unknown"

if __name__ == "__main__":
    statement = "The capital of France is Paris."
    print(fact_check(statement))