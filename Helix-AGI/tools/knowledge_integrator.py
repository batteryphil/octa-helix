import json
import requests
import re
from pathlib import Path
from bs4 import BeautifulSoup
from json import JSONDecodeError

def fetch_tool_output(tool_name):
    try:
        response = requests.get(f"http://localhost:8000/{tool_name}")
        response.raise_for_status()
        return response.json()
    except JSONDecodeError:
        return {}
    except requests.RequestException:
        return {}

def integrate_knowledge(curiosity_tracker_output, belief_conflict_output, belief_dump_output):
    knowledge = {}

    # Merge curiosity_tracker output
    curiosity_tracker = fetch_tool_output("curiosity_tracker")
    for topic, score in curiosity_tracker.items():
        if topic not in knowledge:
            knowledge[topic] = score

    # Merge belief_conflict output
    belief_conflict = fetch_tool_output("belief_conflict")
    for topic, score in belief_conflict.items():
        if topic in knowledge:
            knowledge[topic] += score
        else:
            knowledge[topic] = score

    # Merge belief_dump output
    belief_dump = fetch_tool_output("belief_dump")
    for topic, score in belief_dump.items():
        if topic in knowledge:
            knowledge[topic] += score
        else:
            knowledge[topic] = score

    return knowledge