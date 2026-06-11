import json
import jsonlines
import networkx as nx

def load_jsonl_to_digraph(file_path):
    G = nx.DiGraph()
    with jsonlines.open(file_path) as f:
        for line in f:
            data = json.loads(line)
            node = data['node']
            neighbors = data.get('neighbors', [])
            G.add_node(node)
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
    return G

def get_neighbors(G, node):
    return list(G.successors(node))

if __name__ == '__main__':
    file_path = 'example.jsonl'
    G = load_jsonl_to_digraph(file_path)
    node = 'example_node'
    neighbors = get_neighbors(G, node)
    print(f"Neighbors of {node}: {neighbors}")