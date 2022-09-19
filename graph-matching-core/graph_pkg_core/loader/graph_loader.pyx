import os
import re
from glob import glob
from typing import Optional, List, Tuple, Iterable

import networkx as nx
import numpy as np
import pandas
from tqdm import tqdm

from graph_pkg_core.graph.edge cimport Edge
from graph_pkg_core.graph.graph cimport Graph
from graph_pkg_core.graph.label.label_edge cimport LabelEdge
from graph_pkg_core.graph.label.label_node_vector cimport LabelNodeVector
from graph_pkg_core.graph.node cimport Node

def _construct_graph(idx_graph: int,
                     graph: nx.Graph) -> Graph:
    """

    Args:
        idx_graph:
        graph:

    Returns:

    """
    new_graph = Graph(name=str(idx_graph),
                      filename=f'gr_{idx_graph}.graphml',
                      num_nodes=len(graph.nodes))
    node_attr = 'x'

    for idx_node, node_data in graph.nodes(data=True):
        np_data = np.fromstring(node_data[node_attr][1:-1], sep=' ')
        lbl_node = LabelNodeVector(np_data)
        node = Node(int(idx_node), lbl_node)

        new_graph.add_node(node)

    for idx_start, idx_stop in graph.edges:
        edge = Edge(int(idx_start), int(idx_stop), LabelEdge(0))

        new_graph.add_edge(edge)

    return new_graph

def load_graphs(path_dataset: str,
                load_classes: Optional[bool] = False) -> Tuple[List[Graph], Optional[np.ndarray]]:
    """

    Args:
        path_dataset:
        load_classes:
    """
    files = glob(os.path.join(path_dataset, '*.graphml'))
    node_attr = 'x'

    graphs = []

    for _, file in tqdm(enumerate(files),
                        total=len(files),
                        desc='Loading Graphs'):
        # The idx of the graph is retrieved from its filename
        filename = file.split('/')[-1]
        idx_graph = re.findall('[0-9]+', filename)[0]

        # Load the graph with networkx
        nx_graph = nx.read_graphml(file)

        # Construct the graph with the loaded nx.Graph
        new_graph = _construct_graph(idx_graph, nx_graph)

        graphs.append(new_graph)

    # Sort the graphs by their graph idx
    graphs = sorted(graphs, key=lambda x: int(x.name))

    if load_classes:
        classes_file = os.path.join(path_dataset, 'graph_classes.csv')
        df = pandas.read_csv(classes_file)
        classes = df['class'].to_numpy()

    return graphs, classes

from sklearn.model_selection import train_test_split

def train_val_test_split(X: List[Graph],
                         y: Iterable,
                         val_size: float=0.2,
                         test_size: float=0.2):
    """

    Args:
        X:
        y:
        val_size:
        test_size:

    Returns:

    """
    # First get the val split
    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      test_size=val_size,
                                                      random_state=1)

    test_size = test_size / (1 - val_size)
    X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                        y_train,
                                                        test_size=test_size,
                                                        random_state=1)

    return X_train, X_val, X_test, y_train, y_val, y_test