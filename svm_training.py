import sklearn.svm.LinearSVC
import numpy as np

from graph_tools import Neighborhood

from typing import List

def train_svms(representative_neighborhoods, all_descriptors, graph_adjacency, edge_threshold = 0):
    #TODO calibrate the SVMs using Pratt's method
    X = all_descriptors
    Y = get_classes_all_nodes(representative_neighborhoods, len(all_descriptors))

    svms = sklearn.svm.LinearSVC()
    svms.fit(X,Y)

    return svms

def get_classes_all_nodes(representative_neighborhoods : Neighborhood, nb_nodes):
    """
        returns multi-label classes of each image
    """
    nb_representative_neighborhoods = len(representative_neighborhoods)

    Y = np.zeros((nb_nodes, nb_representative_neighborhoods))
    for id_neighborhood, neighborhood in enumerate(representative_neighborhoods):
        ids_nodes_neighborhood = [neighborhood.id_representative_image + neighborhood.ids_neighbors]
        Y[ids_nodes_neighborhood, [id_neighborhood]] = 1
    return Y

def get_negative_ids(id_representative : int, graph_adjacency):
    return [id_node for id_node in range(len(graph_adjacency)) if graph_adjacency[id_representative,id_node] == 0]

def get_positive_ids(id_representative : int, graph_adjacency, edge_threshold = 0):
    return [id_node for id_node in range(len(graph_adjacency)) if graph_adjacency[id_representative, id_node] >= edge_threshold]