import json
import networkx as nx
from pathlib import Path

def load_jsonl(file_path):
    data = []
    for line in Path(file_path).read_text().split('\n'):
        if line.strip():
            data.append(json.loads(line))
    return data

def get_neighbors(jsonl_file, node):
    G = nx.DiGraph()
    data = load_jsonl(jsonl_file)
    for item in data:
        G.add_edge(item['source'], item['target'])
    return list(G.neighbors(node))

if __name__ == '__main__':
    jsonl_file = 'path/to/your/jsonl/file'
    node = 'node_id'
    neighbors = get_neighbors(jsonl_file, node)
    print(neighbors)