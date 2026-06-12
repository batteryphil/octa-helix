import json
import jsonlines
import networkx as nx

def jsonl_to_dict(jsonl_path):
    with jsonlines.open(jsonl_path) as f:
        return [json.loads(line) for line in f]

def create_graph_from_jsonl(jsonl_data):
    G = nx.DiGraph()
    for item in jsonl_data:
        G.add_node(item['id'])
        for neighbor in item['neighbors']:
            G.add_edge(item['id'], neighbor)
    return G

def get_neighbors(graph, node_id):
    return list(graph.predecessors(node_id))

def main():
    jsonl_path = 'path/to/your/jsonl/file'
    jsonl_data = jsonl_to_dict(jsonl_path)
    graph = create_graph_from_jsonl(jsonl_data)
    node_id = 'some_node_id'
    neighbors = get_neighbors(graph, node_id)
    print(neighbors)

if __name__ == '__main__':
    main()