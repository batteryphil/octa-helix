import json
import networkx as nx

def get_neighbors(digraph, node):
    return list(digraph.neighbors(node))

def main():
    with open('graph.jsonl', 'r') as f:
        graph_data = [json.loads(line) for line in f]
    graph = nx.DiGraph()
    for data in graph_data:
        graph.add_edge(data['source'], data['target'])
    node = 'A'
    neighbors = get_neighbors(graph, node)
    print(f"Neighbors of {node}: {neighbors}")

if __name__ == '__main__':
    main()