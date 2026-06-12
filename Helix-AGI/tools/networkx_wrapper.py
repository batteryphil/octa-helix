import json
import jsonlines
import networkx as nx

def load_jsonl_as_digraph(file_path):
    G = nx.DiGraph()
    with jsonlines.open(file_path) as f:
        for line in f:
            data = json.loads(line)
            G.add_edge(data['source'], data['target'])
    return G

def get_neighbors(G, node):
    return list(G.neighbors(node))

if __name__ == '__main__':
    file_path = 'example.jsonl'
    node = 'example_node'
    G = load_jsonl_as_digraph(file_path)
    neighbors = get_neighbors(G, node)
    print(neighbors)