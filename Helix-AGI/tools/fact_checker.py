import json
import requests
from bs4 import BeautifulSoup
import re
import pathlib

def get_facts_from_wikipedia(query):
    url = f"https://en.wikipedia.org/wiki/{query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    facts = []
    for p in paragraphs:
        text = p.text.strip()
        if text:
            facts.append(text)
    return facts

def calculate_confidence(statement, facts):
    statement_words = statement.lower().split()
    fact_words = [fact.lower() for fact in facts]
    count = sum(1 for word in statement_words if word in fact_words)
    return count / len(statement_words)

def fact_checker(statement):
    query = re.sub(r"[^a-zA-Z0-9 ]", "", statement).capitalize()
    facts = get_facts_from_wikipedia(query)
    confidence = calculate_confidence(statement, facts)
    return {"statement": statement, "query": query, "confidence": confidence, "facts": facts}

if __name__ == "__main__":
    statement = "The capital of France is Paris."
    result = fact_checker(statement)
    print(json.dumps(result, indent=2))