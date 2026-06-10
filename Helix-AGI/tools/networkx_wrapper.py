import networkx as nx

def shortest_path(digraph, node1, node2):
    return nx.shortest_path(digraph, source=node1, target=node2)

if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edge('A', 'B', weight=5)
    G.add_edge('B', 'C', weight=4)
    G.add_edge('A', 'D', weight=2)
    G.add_edge('D', 'C', weight=1)
    G.add_edge('D', 'E', weight=3)
    G.add_edge('E', 'C', weight=6)

    print(shortest_path(G, 'A', 'C'))  # ['A', 'D', 'C']
    print(shortest_path(G, 'A', 'E'))  # ['A', 'D', 'E']
    print(shortest_path(G, 'B', 'E'))  # []