import json
import jsonlines
from networkx import DiGraph

def jsonl_to_digraph(file_path):
    graph = DiGraph()
    with jsonlines.open(file_path) as reader:
        for line in reader:
            data = json.loads(line)
            node = data['node']
            edges = data['edges']
            graph.add_node(node)
            for edge in edges:
                graph.add_edge(node, edge['target'])
    return graph

def get_neighbors(graph, node):
    return list(graph.successors(node))

if __name__ == '__main__':
    file_path = 'example.jsonl'
    graph = jsonl_to_digraph(file_path)
    node = 'A'
    neighbors = get_neighbors(graph, node)
    print(f"Neighbors of {node}: {neighbors}")