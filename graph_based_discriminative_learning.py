from graph_tools import compute_covering
from svm_training import train_svms

import numpy as np

from typing import List


nb_descriptors = 50
dim_descriptors = 128

descriptors = np.random.random((nb_descriptors, dim_descriptors))

graph_adjacency = np.array(np.random.rand(nb_descriptors,nb_descriptors) > 0.95, dtype = np.float32)


covering = compute_covering(graph_adjacency)

svms = train_svms(covering, descriptors, graph_adjacency, 0.05)