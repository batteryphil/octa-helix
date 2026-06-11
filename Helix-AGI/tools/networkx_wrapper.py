import json
import networkx as nx
from pathlib import Path

def load_jsonl_to_digraph(file_path):
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        for line in f:
            node, *neighbors_str = line.strip().split()
            neighbors = [n for n in neighbors_str]
            G.add_node(node, neighbors=neighbors)
            G.add_edges_from([(node, n) for n in neighbors])
    return G

def get_neighbors(G, node):
    return G.nodes[node].get('neighbors', [])

if __name__ == '__main__':
    file_path = 'example.jsonl'
    G = load_jsonl_to_digraph(file_path)
    print(get_neighbors(G, 'A'))