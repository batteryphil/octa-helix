import json
import jsonlines
import networkx as nx
from pathlib import Path

class JSONLToNetworkX:
    def __init__(self, jsonl_file):
        self.jsonl_file = jsonl_file
        self.graph = nx.DiGraph()

    def load_jsonl(self):
        with jsonlines.open(self.jsonl_file) as f:
            for line in f:
                yield json.loads(line)

    def add_edges(self):
        for line in self.load_jsonl():
            self.graph.add_edge(line['source'], line['target'])

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))

if __name__ == '__main__':
    jsonl_file = Path('example.jsonl')
    converter = JSONLToNetworkX(jsonl_file)
    converter.add_edges()
    print(converter.get_neighbors('A'))