import json
import requests
from bs4 import BeautifulSoup
import re
import pathlib

def validate_data(data_source, knowledge_base_url):
    response = requests.get(knowledge_base_url)
    knowledge_base = json.loads(response.text)

    for item in data_source:
        if item['name'] in knowledge_base:
            if item['value'] == knowledge_base[item['name']]:
                item['valid'] = True
            else:
                item['valid'] = False
                item['discrepancy'] = f"Discrepancy found: {knowledge_base[item['name']]}"
        else:
            item['valid'] = False
            item['discrepancy'] = "Item not found in knowledge base"

    return data_source

def main():
    data_source = [
        {"name": "population", "value": "33000000"},
        {"name": "capital", "value": "New York"},
        {"name": "language", "value": "English"}
    ]

    knowledge_base_url = "https://example.com/knowledge_base.json"

    validated_data = validate_data(data_source, knowledge_base_url)

    for item in validated_data:
        print(f"{item['name']}: {'Valid' if item['valid'] else 'Invalid'} - {item['discrepancy']}")

if __name__ == '__main__':
    main()