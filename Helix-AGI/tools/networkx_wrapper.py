import json
import jsonlines
import networkx as nx

def load_jsonl_to_digraph(file_path):
    G = nx.DiGraph()
    with jsonlines.open(file_path) as f:
        for line in f:
            data = json.loads(line)
            G.add_node(data['node_id'])
            G.add_edges_from([(data['node_id'], neighbor) for neighbor in data.get('neighbors', [])])
    return G

def get_neighbors(digraph, node_id):
    return list(digraph.neighbors(node_id))

if __name__ == '__main__':
    file_path = 'path/to/jsonl/file'
    node_id = 'target_node'
    digraph = load_jsonl_to_digraph(file_path)
    neighbors = get_neighbors(digraph, node_id)
    print(neighbors)