import json
import jsonlines
import networkx as nx
from pathlib import Path

def jsonl_to_dictlist(file_path):
    with jsonlines.open(file_path) as f:
        return [json.loads(line) for line in f]

def create_networkx_graph(jsonl_file):
    G = nx.DiGraph()
    data = jsonl_to_dictlist(jsonl_file)
    for item in data:
        G.add_node(item['id'])
        for neighbor in item['neighbors']:
            G.add_edge(item['id'], neighbor)
    return G

def get_neighbors(graph, node_id):
    return list(graph.predecessors(node_id))

def main():
    jsonl_file_path = Path('example.jsonl')
    graph = create_networkx_graph(jsonl_file_path)
    node_id = 'node123'
    neighbors = get_neighbors(graph, node_id)
    print(f"Neighbors of {node_id}: {neighbors}")

if __name__ == '__main__':
    main()