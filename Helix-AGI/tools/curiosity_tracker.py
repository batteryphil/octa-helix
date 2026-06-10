import json
import os
import datetime
import random
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

DATA_FILE = "curiosity_data.json"
PLOT_FILE = "curiosity_plot.png"

def get_curiosity_score(url: str) -> float:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    score_element = soup.find("div", class_="curiosity-score")
    return float(score_element.text.strip())

def log_curiosity_event(date: datetime.date, score: float) -> None:
    data = Path(DATA_FILE)
    if not data.exists():
        data.write_text("[]")  # Fixed the syntax error here

    with open(DATA_FILE, "r") as file:
        events = json.load(file)

    events.append({"date": str(date), "score": score})  # Fixed the syntax error here

    with open(DATA_FILE, "w") as file:
        json.dump(events, file)