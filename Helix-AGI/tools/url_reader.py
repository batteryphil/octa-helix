"""
Fetches and parses the content of a web page from a given URL.

Usage:
from url_reader import URLReader

url = "https://example.com"
reader = URLReader(url)
content = reader.get_content()
text = reader.get_text()
tables = reader.get_tables()
links = reader.get_links()
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

class URLReader:
    def __init__(self, url):
        self.url = url
        self.response = requests.get(url)
        self.soup = BeautifulSoup(self.response.text, 'html.parser')

    def get_content(self):
        """Returns the raw HTML content of the web page."""
        return self.response.text

    def get_text(self):
        """Returns the text content extracted from the HTML."""
        return self.soup.get_text()

    def get_tables(self):
        """Returns a list of tables found on the page."""
        return self.soup.find_all('table')

    def get_links(self):
        """Returns a list of links found on the page."""
        return [urljoin(self.url, link['href']) for link in self.soup.find_all('a', href=True)]

    def get_elements(self, tag, attributes=None):
        """Returns a list of elements with the specified tag and attributes."""
        if attributes:
            return [elem for elem in self.soup.find_all(tag, **attributes)]
        else:
            return self.soup.find_all(tag)