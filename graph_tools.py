import numpy as np

from typing import List


class Neighborhood:
    def __init__(self, id_representative_image : int, ids_neighbors : List[int]):
        self.id_representative_image = id_representative_image
        self.ids_neighbors = ids_neighbors

    def __repr__(self):
        return f"({self.id_representative_image}) -> {self.ids_neighbors}"

def compute_covering(graph_adjacency) -> List[dict]:
    """
        Computes greedy covering
    """
    black_nodes = set([]) #nodes not covered yet

    covering_neighborhoods = []

    n = len(graph_adjacency)
    while len(black_nodes) != n:
        white_nodes = get_white_nodes(black_nodes, graph_adjacency)

        d_a = compute_uncovered_images_count(graph_adjacency, white_nodes)

        id_representative_image = np.argmax(d_a)
        neighbors_representative_image = get_neighbors(id_representative_image, graph_adjacency)

        black_nodes.add(id_representative_image)
        for id_neighbor in neighbors_representative_image:
            black_nodes.add(id_neighbor)

        representative_neighborhood = Neighborhood(id_representative_image, neighbors_representative_image)
        covering_neighborhoods.append(representative_neighborhood)

    return covering_neighborhoods

def get_white_nodes(black_nodes, graph_adjacency):
    """
        White Nodes are the complementary to White nodes
    """
    return set([id_node for id_node in range(len(graph_adjacency)) if id_node not in black_nodes])

def compute_uncovered_images_count(graph_adjacency, white_nodes : List[int]):
    """
    Computes d_a in the paper
    """

    d_a = [-1 for _ in range(len(graph_adjacency))]
    for id_white in white_nodes:
        neighbors = get_neighbors(id_white, graph_adjacency)
        white_neighbors = get_white_nodes_in_neighborhood(neighbors, white_nodes)
        d_a[id_white] = len(white_neighbors)
    return d_a

def get_white_nodes_in_neighborhood(neighborhood, white_nodes):
    return [neighbor for neighbor in neighborhood if neighbor in white_nodes]

def get_neighbors(id_query_node, graph_adjacency):
    return [id_node for id_node in range(len(graph_adjacency)) 
        if graph_adjacency[id_query_node, id_node] > 0 and id_query_node != id_node]