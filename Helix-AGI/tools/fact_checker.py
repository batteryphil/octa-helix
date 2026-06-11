import json
import jsonlines
import requests
from bs4 import BeautifulSoup
import psutil
import re
import pathlib

def load_knowledge_base():
    knowledge_base = {}
    for line in jsonlines.open('curiosity_knowledge.jsonl'):
        fact = json.loads(line)
        knowledge_base[fact['id']] = fact
    return knowledge_base

def check_fact_consistency(fact, knowledge_base):
    fact_id = fact['id']
    related_facts = []
    for fact_id_related in fact['related_facts']:
        related_facts.append(knowledge_base[fact_id_related])
    
    for related_fact in related_facts:
        if fact['claim'] == related_fact['claim']:
            return 'consistent', f"Fact {fact_id} is consistent with related fact {related_fact['id']}."
        elif fact['claim'] != related_fact['claim']:
            return 'inconsistent', f"Fact {fact_id} is inconsistent with related fact {related_fact['id']}."

    return 'inconsistent', f"Fact {fact_id} is inconsistent with the knowledge base."

def main():
    fact = {
        'id': 'fact123',
        'claim': 'The Earth revolves around the Sun.',
        'related_facts': ['fact456', 'fact789']
    }
    knowledge_base = load_knowledge_base()
    consistency, explanation = check_fact_consistency(fact, knowledge_base)
    print(f"Consistency: {consistency}\nExplanation: {explanation}")

if __name__ == '__main__':
    main()