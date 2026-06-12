import json
import jsonlines
from networkx import DiGraph

class NetworkxWrapper:
    def __init__(self, jsonl_file_path):
        self.graph = DiGraph()
        with jsonlines.open(jsonl_file_path) as f:
            for line in f:
                node, neighbors = line.strip().split('\t', 1)
                neighbors = neighbors.split('\t')
                self.graph.add_node(node)
                self.graph.add_edges_from([(node, n) for n in neighbors])

    def get_neighbors(self, node):
        return self.graph.neighbors(node)

def main():
    jsonl_file_path = 'path/to/graph.jsonl'
    wrapper = NetworkxWrapper(jsonl_file_path)
    node = 'example_node'
    neighbors = wrapper.get_neighbors(node)
    print(f"Neighbors of {node}: {neighbors}")

if __name__ == '__main__':
    main()