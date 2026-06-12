import json
import requests
from bs4 import BeautifulSoup
import re
import jsonlines
import pathlib
import psutil
import time

def fetch_and_parse(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    knowledge = []
    for p in paragraphs:
        text = p.text.strip()
        if text:
            knowledge.append(text)
    return knowledge

def filter_and_format(knowledge):
    filtered_knowledge = []
    for item in knowledge:
        item = re.sub(r'\s+', ' ', item)
        item = re.sub(r'[^a-zA-Z0-9\s]', '', item)
        item = item.strip()
        if item:
            filtered_knowledge.append(item)
    return filtered_knowledge

def save_to_file(knowledge, file_path):
    with jsonlines.open(file_path, mode='a') as f:
        for item in knowledge:
            f.write(item + '\n')

def main():
    urls = [
        'https://example.com/source1',
        'https://example.com/source2',
        # Add more trusted sources here
    ]
    knowledge = []
    for url in urls:
        new_knowledge = fetch_and_parse(url)
        knowledge.extend(filter_and_parse(new_knowledge))
    save_to_file(knowledge, 'curiosity_knowledge.jsonl')
    print(f"Updated knowledge base with {len(knowledge)} new items.")

if __name__ == '__main__':
    main()