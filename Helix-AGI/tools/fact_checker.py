import json
import requests
from bs4 import BeautifulSoup
import re

def get_facts_from_kb(statement):
    url = "https://api.example.com/facts"
    payload = {"query": statement}
    response = requests.post(url, json=payload)
    facts = response.json()
    return facts

def check_for_contradictions(statement, facts):
    contradictions = []
    for fact in facts:
        if re.search(re.escape(statement), fact["text"], re.IGNORECASE):
            if fact["truthiness"] != statement:
                contradictions.append((fact["text"], fact["truthiness"]))
    return contradictions

def main():
    statement = input("Enter a statement to check for contradictions: ")
    facts = get_facts_from_kb(statement)
    contradictions = check_for_contradictions(statement, facts)
    if contradictions:
        print("Contradictions found:")
        for contradiction in contradictions:
            print(contradiction)
    else:
        print("No contradictions found.")

if __name__ == "__main__":
    main()