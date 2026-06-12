import json
import requests
import jsonlines
from pathlib import Path

def load_facts():
    facts = {}
    for line in jsonlines.open('curiosity_knowledge.jsonl'):
        item = json.loads(line)
        facts[item['id']] = item['text']
    return facts

def check_fact(statement, facts):
    supported = []
    contradicted = []
    for fact_id, fact in facts.items():
        if re.search(statement, fact):
            supported.append(fact_id)
        elif re.search(fact, statement):
            contradicted.append(fact_id)
    return {
        'supported': supported,
        'contradicted': contradicted,
        'confidence': len(supported) - len(contradicted)
    }

def main():
    statement = input("Enter a statement to check: ")
    facts = load_facts()
    result = check_fact(statement, facts)
    print(f"Confidence: {result['confidence']}")
    print("Supported by facts:")
    for fact_id in result['supported']:
        print(fact_id)
    print("Contradicted by facts:")
    for fact_id in result['contradicted']:
        print(fact_id)

if __name__ == '__main__':
    main()