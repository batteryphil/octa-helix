import json
import requests
from bs4 import BeautifulSoup
import re

class KnowledgeBaseSearch:
    def __init__(self, url):
        self.url = url
        self.page = requests.get(url)
        self.soup = BeautifulSoup(self.page.content, 'html.parser')

    def search(self, query):
        query = query.lower()
        search_results = []
        for match in self.soup.find_all(string=lambda text: query in text.lower()):
            search_results.append(str(match))
        return search_results

def main():
    url = 'https://example.com/knowledge-base'
    query = 'example query'
    search = KnowledgeBaseSearch(url)
    results = search.search(query)
    print(json.dumps({'query': query, 'results': results}))

if __name__ == '__main__':
    main()