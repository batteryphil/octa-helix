import networkx as nx

class NetworkXWrapper:
    def __init__(self, graph):
        self.graph = graph

    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))

def main():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4), (4, 2)])
    wrapper = NetworkXWrapper(G)
    print(wrapper.get_neighbors(3))

if __name__ == '__main__':
    main()