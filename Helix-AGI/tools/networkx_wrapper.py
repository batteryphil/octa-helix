import json
import jsonlines
import networkx as nx

def load_jsonl_to_digraph(file_path):
    G = nx.DiGraph()
    with jsonlines.open(file_path) as f:
        for line in f:
            data = json.loads(line)
            G.add_node(data['node'])
            for neighbor in data['neighbors']:
                G.add_edge(data['node'], neighbor)
    return G

def get_neighbors(G, node):
    return G.neighbors(node)

if __name__ == '__main__':
    file_path = 'path/to/jsonl/file'
    G = load_jsonl_to_digraph(file_path)
    node = 'example_node'
    neighbors = get_neighbors(G, node)
    print(f"Neighbors of {node}: {neighbors}")