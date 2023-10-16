import csv
import json
from typing import List

import numpy as np


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def set_global_verbose(verbose: bool = False) -> None:
    """
    Set the global verbose.
    Activate the logging module (use `logging.info('Hello world!')`)
    Activate the tqdm loading bar.

    Args:
        verbose: If `True` activate the global verbose

    Returns:

    """
    import logging
    from functools import partialmethod
    from tqdm import tqdm

    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=not verbose)


class Logger:

    def __init__(self, filename: str):
        """

        Args:
            filename:
        """
        self.filename = filename
        self.__data = {}
        self.lvl_name = None

    @property
    def data(self):
        if self.lvl_name:
            return self.__data[self.lvl_name]
        else:
            return self.__data

    def set_lvl(self, lvl_name: str) -> None:
        """
        Change the level of the logger.
        It is used to log experiment with multiple loops

        Args:
            lvl_name:

        Returns:

        """
        self.lvl_name = lvl_name
        self.__data[lvl_name] = {}

    def save_data(self) -> None:
        """
        Save the current state of the data property.

        Returns:

        """
        with open(self.filename, 'w') as file:
            json.dump(self.__data, file, indent=4)


def write_predictions(filename: str,
                      predictions: List[int],
                      GT_labels: List[int]) -> None:
    """
    Write the predictions and the corresponding GT labels in `filename`.

    Args:
        filename: File where to save the predictions.
        predictions: Iterable of predictions
        GT_labels: Iterable of the GT labels

    Returns:

    """

    with open(filename, 'w') as csv_file:
        fieldnames = ['predictions', 'GT_labels']

        writer = csv.DictWriter(csv_file,
                                fieldnames=fieldnames)

        writer.writeheader()
        for pred, GT_lbl in zip(predictions, GT_labels):
            writer.writerow({'predictions': pred, 'GT_labels': GT_lbl})



def write_times(filename: str, times: List[float]) -> None:
    """
    Save the computation time in `.csv` file

    Args:
        filename: File where to save the GEDs.
        distances: `np.array` containing the GEDs

    Returns:

    """
    with open(filename, 'w') as csv_file:
        fieldnames = ['time']

        writer = csv.DictWriter(csv_file,
                                fieldnames=fieldnames)

        writer.writeheader()
        for time in times:
            writer.writerow({'time': time})


def write_distances(filename: str, distances: np.ndarray) -> None:
    """
    Save the GEDs in `.npy` file

    Args:
        filename: File where to save the GEDs.
        distances: `np.array` containing the GEDs

    Returns:

    """
    with open(filename, 'wb') as file:
        np.save(file, distances)


def write_GT_labels(filename: str,
                    GT_labels: List[int]) -> None:
    """
    Write GT labels in `filename`.

    Args:
        filename: File where to save the predictions.
        GT_labels: Iterable of the GT labels

    Returns:

    """

    with open(filename, 'w') as csv_file:
        fieldnames = ['GT_labels']

        writer = csv.DictWriter(csv_file,
                                fieldnames=fieldnames)

        writer.writeheader()
        for GT_lbl in GT_labels:
            writer.writerow({'GT_labels': GT_lbl})


def save_acc_results(file_results: str, cv_predictions: List) -> None:
    """
    Save the list of cross-validation scores

    Args:
        file_results:
        cv_predictions:

    Returns:

    """
    with open(file_results, 'w') as write_file:
        json.dump(cv_predictions, write_file, indent=4)

from cyged import load_graphs as load_graphml
from os.path import join
import pickle
import pandas

def _load_pkl_graphs(root_dataset: str, load_classes: bool):
    filename = join(root_dataset, 'graphs.pkl')

    with open(filename, 'rb') as f:
        graphs = pickle.load(f)

    if load_classes:
        classes_file = join(root_dataset, 'graph_classes.csv')
        df = pandas.read_csv(classes_file)
        classes = df['class'].to_numpy()

    return graphs, classes


def load_graphs(root_dataset: str, file_extension: str, load_classes: bool):
    if file_extension == 'graphml':
        graphs, lbls = load_grahml(root_dataset=root_dataset,
                                   file_extension=file_extension,
                                   load_classes=load_classes)
        return graphs, lbls

    if file_extension == 'pkl':
        graphs, lbls = _load_pkl_graphs(root_dataset=root_dataset,
                                        load_classes=load_classes)

        return graphs, lbls