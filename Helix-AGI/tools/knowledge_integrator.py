import json
import requests
import re
import pathlib
from bs4 import BeautifulSoup
from psutil import cpu_percent

class KnowledgeIntegrator:
    def __init__(self, tools):
        self.tools = tools
        self.insights = {}
        self.inconsistencies = []

    def fetch_insights(self, tool):
        url = f"http://localhost:8000/tools/{tool}/insights"
        response = requests.get(url)
        return response.json()

    def compare_insights(self, tool1, tool2):
        insight1 = self.insights.get(tool1)
        insight2 = self.insights.get(tool2)
        if insight1 and insight2:
            if insight1 == insight2:
                return f"{tool1} and {tool2} have consistent insights."
            else:
                self.inconsistencies.append((tool1, tool2, (insight1, insight2)))
        else:
            self.inconsistencies.append((tool1, tool2, (None, None)))
        return None

# Example usage:
integrator = KnowledgeIntegrator(["tool1", "tool2", "tool3"])
integrator.insights["tool1"] = "Insight 1"
integrator.insights["tool2"] = "Insight 2"
integrator.insights["tool3"] = "Insight 3"

print(integrator.compare_insights("tool1", "tool2"))
print(integrator.compare_insights("tool1", "tool3"))
print(integrator.compare_insights("tool2", "tool3"))