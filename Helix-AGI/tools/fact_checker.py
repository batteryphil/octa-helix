import requests
from bs4 import BeautifulSoup
import json
import re
import pathlib

def fact_checker(fact):
    # Query knowledge base 1: Wikipedia
    url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={fact}&format=json"
    response = requests.get(url)
    data = json.loads(response.content.decode('utf-8'))
    if len(data) > 1:
        summary = data[2].split('.')[0].strip()
        if summary:
            return 'verified'
    # Query knowledge base 2: Google
    elif re.search(r'\w+', fact):
        url = f"https://www.google.com/search?q={fact}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        titles = [a.text for a in soup.select('div.yuRUbf > a')]
        if titles:
            return 'verified'
    return 'unverified'

def main():
    fact = input("Enter a fact to check: ")
    result = fact_checker(fact)
    print(result)

if __name__ == '__main__':
    main()