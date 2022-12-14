import os

import pytest

from graph_pkg_core.loader.loader_vector import LoaderVector
from graph_pkg_core.loader.graph_loader import load_graphs
from graph_pkg_core.loader.graph_loader import load_graphs

############## VECTOR ##############

FOLDER_DATA = os.path.join(os.path.dirname(__file__),
                           '../test_data/proteins_test')


@pytest.mark.parametrize('folder, num_graphs, size_graphs, name_graphs',
                         [
                             (FOLDER_DATA,
                              22,
                              [42, 27, 10, 24, 11, 336, 108, 154, 19, 67, 12, 32, 11, 98, 32, 10, 31, 14, 13, 16, 24,
                               6],
                              ['gr_0.graphml', 'gr_1.graphml', 'gr_2.graphml', 'gr_3.graphml', 'gr_4.graphml',
                               'gr_5.graphml', 'gr_6.graphml', 'gr_7.graphml', 'gr_8.graphml', 'gr_22.graphml',
                               'gr_23.graphml', 'gr_663.graphml', 'gr_664.graphml', 'gr_665.graphml', 'gr_666.graphml',
                               'gr_667.graphml', 'gr_668.graphml', 'gr_669.graphml', 'gr_671.graphml', 'gr_672.graphml',
                               'gr_673.graphml', 'gr_676.graphml']
                              ),
                         ])
def test_all_letters(folder, num_graphs, size_graphs, name_graphs):
    # loader_vector = LoaderVector(folder)
    # graphs = loader_vector.load()
    graphs = load_graphs(folder)
    print([graph.name for graph in graphs])

    assert len(graphs) == num_graphs
    assert [len(graph) for graph in graphs] == size_graphs
    assert [graph.filename for graph in graphs] == name_graphs
