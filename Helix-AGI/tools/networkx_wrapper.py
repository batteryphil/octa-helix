import networkx as nx

def get_neighbors(graph, node):
    return list(graph.neighbors(node))

if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    G.add_edge('C', 'D')
    print(get_neighbors(G, 'A'))