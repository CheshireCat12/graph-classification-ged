from typing import Tuple, List

from graph_pkg_core.edit_cost.edit_cost_vector cimport EditCostVector
from graph_pkg_core.algorithm.graph_edit_distance cimport GED
from graph_pkg_core.loader.graph_loader import load_graphs


class Coordinator:

    def __init__(self,
                 root_dataset: str,
                 parameters_edit_cost: Tuple):

        self.root_dataset = root_dataset
        self.graphs, self.classes = load_graphs(self.root_dataset,
                                                load_classes=True)

        self.parameters_edit_cost = parameters_edit_cost
        self.edit_cost = EditCostVector(*self.parameters_edit_cost)
        self.ged = GED(self.edit_cost)
