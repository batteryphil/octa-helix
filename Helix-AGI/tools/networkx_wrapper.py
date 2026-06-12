import json
import networkx as nx
from pathlib import Path

def get_neighbors(graph_file, node):
    G = nx.readwrite.jsonl_graph(graph_file)
    return list(G.neighbors(node))

if __name__ == '__main__':
    graph_file = 'path/to/graph.jsonl'
    node = 'node_id'
    neighbors = get_neighbors(graph_file, node)
    print(neighbors)