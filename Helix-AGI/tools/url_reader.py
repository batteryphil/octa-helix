#!/usr/bin/env python3
# url_reader.py

"""
A tool to fetch and parse the content of a web page.

Usage:
  url_reader.py <url>
"""

import requests
from bs4 import BeautifulSoup
from htools import Tool, ToolRegistry

class URLReader(Tool):
    toolset = 'self'

    def __init__(self, url):
        self.url = url

    def run(self):
        response = requests.get(self.url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else 'No title found'
        content = soup.get_text()
        links = [a['href'] for a in soup.find_all('a', href=True)]

        return {
            'title': title,
            'content': content,
            'links': links
        }

def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    url = sys.argv[1]
    tool = URLReader(url)
    result = tool.run()
    print(f"Title: {result['title']}")
    print("Content:")
    print(result['content'])
    print("Links:")
    print('\n'.join(result['links']))

if __name__ == '__main__':
    ToolRegistry.register_tool(URLReader)
    main()