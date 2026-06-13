import networkx as nx

def get_neighbors(node_id, graph):
    return list(graph.neighbors(node_id))

if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    G.add_edge('C', 'A')
    print(get_neighbors('A', G))