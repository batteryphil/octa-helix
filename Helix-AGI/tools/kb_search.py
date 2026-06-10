import json
import requests
from bs4 import BeautifulSoup
import re
import json
import os

class KnowledgeBaseSearch:
    def __init__(self, url):
        self.url = url
        self.entries = self.fetch_entries()

    def fetch_entries(self):
        response = requests.get(self.url)
        soup = BeautifulSoup(response.text, 'html.parser')
        entries = soup.find_all('div', class_='entry')
        return [entry.text.strip() for entry in entries]

    def fuzzy_match(self, query, entries):
        query = re.sub(r'\W+', ' ', query).strip().lower()
        matched_entries = []
        for entry in entries:
            entry = re.sub(r'\W+', ' ', entry).strip().lower()
            score = sum(c == q for c, q in zip(entry, query)) / len(query)
            if score > 0.5:
                matched_entries.append((entry, score))
        return matched_entries

    def semantic_analysis(self, query, entries):
        query_words = set(query.split())
        relevant_entries = []
        for entry, score in self.fuzzy_match(query, entries):
            entry_words = set(entry.split())
            intersection = query_words & entry_words
            if intersection:
                relevance = len(intersection) / (len(query_words) + len(entry_words))
                relevant_entries.append((entry, score, relevance))
        return relevant_entries

    def rank_entries(self, entries):
        return sorted(entries, key=lambda x: (-x[1], -x[2]), reverse=True)

    def search(self, query):
        matched_entries = self.fuzzy_match(query, self.entries)
        relevant_entries = self.semantic_analysis