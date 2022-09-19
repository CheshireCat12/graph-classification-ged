from typing import Tuple

from graph_pkg_core.algorithm.graph_edit_distance cimport GED
from graph_pkg_core.coordinator.graph_loader import load_graphs
from graph_pkg_core.edit_cost.edit_cost_vector cimport EditCostVector


class Coordinator:

    def __init__(self,
                 parameters_edit_cost: Tuple,
                 root_dataset: str = None):
        """

        Args:
            parameters_edit_cost:
            root_dataset:
        """
        self.parameters_edit_cost = parameters_edit_cost
        self.edit_cost = EditCostVector(*self.parameters_edit_cost)
        self.ged = GED(self.edit_cost)

        if root_dataset is not None:
            self.root_dataset = root_dataset
            self.graphs, self.classes = load_graphs(self.root_dataset,
                                                    load_classes=True)
