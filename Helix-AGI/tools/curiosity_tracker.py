import json
import requests
import time
import os
import re
import json
import psutil
from pathlib import Path
from bs4 import BeautifulSoup

def get_curiosity_metrics():
    response = requests.get("http://localhost:8000/api/curiosity")
    return json.loads(response.text)

def save_metrics(metrics, filename):
    with open(filename, "a") as f:
        json.dump(metrics, f)
        f.write("\n")

def load_metrics(filename):
    if Path(filename).exists():
        with open(filename, "r") as f:
            return [json.loads(line) for line in f]
    else:
        return []

def calculate_average_response_length(metrics):
    return sum(len(m["response"]) for m in metrics) / len(metrics)

def calculate_tool_call_rate(metrics):
    return len(metrics) / (time.time() - metrics[0]["timestamp"])

def main():
    metrics_filename = "curiosity_metrics.jsonl"
    metrics = load_metrics(metrics_filename)

    while True:
        new_metrics = get_curiosity_metrics()
        metrics.append(new_metrics)
        save_metrics(new_metrics, metrics_filename)

        avg_response_length = calculate_average_response_length(metrics)
        tool_call_rate = calculate_tool_call_rate(metrics)

        print(f"Metrics: {new_metrics}")
        print(f"Average response length: {avg_response_length}")
        print(f"Tool call rate: {tool_call_rate} calls per second")

        time.sleep(5)

if __name__ == "__main__":
    main()