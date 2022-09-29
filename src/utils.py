import csv
import json
from typing import Iterable, List

import numpy as np
from sklearn.model_selection import train_test_split


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


def train_val_test_split(X: List,
                         y: Iterable,
                         val_size: float = 0.2,
                         test_size: float = 0.2,
                         random_state=1):
    """

    Args:
        X:
        y:
        val_size:
        test_size:
        random_state:

    Returns:

    """
    # First get the val split
    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      test_size=val_size,
                                                      random_state=random_state)

    test_size = test_size / (1 - val_size)
    X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                        y_train,
                                                        test_size=test_size,
                                                        random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


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
