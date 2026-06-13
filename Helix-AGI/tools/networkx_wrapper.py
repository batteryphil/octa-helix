import json
import networkx as nx
from pathlib import Path

class BeliefGraph:
    def __init__(self, filename):
        self.graph = nx.DiGraph()
        self.load_graph(filename)

    def load_graph(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.graph.add_edge(data['source'], data['target'], **data.get('attributes', {}))

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))

if __name__ == '__main__':
    graph = BeliefGraph('data/beliefs.jsonl')
    print(graph.get_neighbors('belief1'))