import argparse
import csv
import logging
import os
import pathlib
from collections import namedtuple
from itertools import product
from time import time
from typing import List, Tuple

import numpy as np
from graph_pkg_core.algorithm.knn import KNNClassifier
from graph_pkg_core.coordinator.coordinator import Coordinator
from graph_pkg_core.coordinator.graph_loader import train_val_test_split
from graph_pkg_core.graph.graph import Graph
from graph_pkg_core.utils.logger import Logger
from tqdm import tqdm

AccuracyTracker = namedtuple('AccuracyTracker',
                             ['acc', 'best_alpha', 'best_k'])


def train(coordinator: Coordinator,
          logger: Logger,
          alphas: List[float],
          ks: List[int],
          X_train: List[Graph],
          X_val: List[Graph],
          y_train: List[int],
          y_val: List[int],
          n_cores: int) -> AccuracyTracker:
    """

    Args:
        coordinator:
        logger:
        alphas:
        ks:
        X_train:
        X_val:
        y_train:
        y_val:
        n_cores:

    Returns:

    """
    is_parallel = n_cores > 0

    knn = KNNClassifier(coordinator.ged, parallel=is_parallel)
    knn.train(X_train, y_train)

    acc_tracker = AccuracyTracker(float('-inf'), None, None)

    # Save all the hyperparameters tested
    logger.data['hyperparameters_tuning'] = {'alphas': alphas,
                                             'ks': ks}
    logger.data['val_accuracies'] = []  # List of tuple (acc, alpha, k)
    logger.data['val_prediction_times'] = []

    for alpha, k in tqdm(product(alphas, ks), total=len(alphas) * len(ks)):
        # Update alpha parameter
        coordinator.edit_cost.update_alpha(alpha)

        # Perform prediction
        start_time = time()
        predictions = knn.predict(X_val, k=k, num_cores=n_cores)
        prediction_time = time() - start_time

        # compute accuracy
        current_acc = 100 * ((np.array(predictions) == y_val).sum() / len(y_val))

        # Keep track of the best acc with the corresponding hyperparameters
        if current_acc > acc_tracker.acc:
            acc_tracker = acc_tracker._replace(acc=current_acc, best_alpha=alpha, best_k=k)

        logger.data['val_accuracies'].append((current_acc, alpha, k))
        logger.data['val_prediction_times'].append(prediction_time)
        logger.save_data()

    logging.info(f'Best val classification accuracy {acc_tracker.acc: .2f}'
                 f'(alpha: {acc_tracker.best_alpha}, k: {acc_tracker.best_k})')

    logger.data['best_acc'] = acc_tracker.acc
    logger.data['best_params'] = {'alpha': acc_tracker.best_alpha,
                                  'k': acc_tracker.best_k}
    logger.save_data()

    return acc_tracker


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


def evaluate(coordinator: Coordinator,
             logger: Logger,
             acc_tracker: AccuracyTracker,
             X_train: List[Graph],
             X_test: List[Graph],
             y_train: List[int],
             y_test: List[int],
             n_cores: int,
             folder_results: str,
             save_predictions: bool,
             save_distances: bool) -> None:
    """

    Args:
        coordinator:
        logger:
        acc_tracker:
        X_train:
        X_test:
        y_train:
        y_test:
        n_cores:
        folder_results:
        save_distances:
        save_predictions:

    Returns:

    """
    is_parallel = n_cores > 0

    knn = KNNClassifier(coordinator.ged, parallel=is_parallel)
    knn.train(X_train, y_train)

    alpha, k = acc_tracker.best_alpha, acc_tracker.best_k

    # Set the best alpha parameter
    coordinator.edit_cost.update_alpha(alpha)

    # Perform prediction
    start_time = time()
    predictions = knn.predict(X_test, k=k, num_cores=n_cores)
    prediction_time = time() - start_time

    # compute accuracy
    current_acc = 100 * ((np.array(predictions) == y_test).sum() / len(y_test))

    logger.data['test_accuracy'] = (current_acc, alpha, k)
    logger.data['test_prediction_time'] = prediction_time
    logger.save_data()

    logging.info(f'Classification accuracy (test) {current_acc: .2f}'
                 f'(alpha: {acc_tracker.best_alpha}, k: {acc_tracker.best_k})')

    if save_predictions:
        file_predictions = os.path.join(folder_results,
                                        'predictions.csv')
        write_predictions(file_predictions, predictions, y_test)

    if save_distances:
        file_distances = os.path.join(folder_results,
                                      'distances.npy')
        write_distances(file_distances, knn.current_distances)


def graph_classifier(root_dataset: str,
                     parameters_edit_cost: Tuple,
                     size_splits: List[float],
                     alphas: List[float],
                     ks: List[int],
                     n_cores: int,
                     folder_results: str,
                     save_predictions: bool,
                     save_distances: bool,
                     verbose: bool,
                     args):
    """

    Args:
        root_dataset:
        parameters_edit_cost:
        size_splits:
        alphas:
        ks:
        n_cores:
        folder_results:
        save_predictions:
        save_distances:
        verbose:
        args:

    Returns:

    """
    set_global_verbose(verbose)

    pathlib.Path(folder_results).mkdir(parents=True, exist_ok=True)

    # Init logger
    logger_filename = os.path.join(folder_results,
                                   'results_general.json')
    logger = Logger(logger_filename)

    # Save all the input parameters
    logger.data['parameters'] = vars(args)
    logger.save_data()

    coordinator = Coordinator(parameters_edit_cost,
                              root_dataset)

    # Split the dataset into train, val and test sets
    size_train, size_val, size_test = size_splits
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(coordinator.graphs,
                                                                          coordinator.classes,
                                                                          val_size=size_val,
                                                                          test_size=size_test)

    # Optimize the hyperparameters
    acc_tracker = train(coordinator,
                        logger,
                        alphas, ks,
                        X_train, X_val,
                        list(y_train), list(y_val),
                        n_cores)

    evaluate(coordinator,
             logger,
             acc_tracker,
             X_train, X_test,
             list(y_train), list(y_test),
             n_cores,
             folder_results,
             save_predictions, save_distances)


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

    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=not verbose)


def main(args):
    graph_classifier(args.root_dataset,
                     args.parameters_edit_cost,
                     args.size_splits,
                     args.alphas,
                     args.ks,
                     args.n_cores,
                     args.folder_results,
                     args.save_predictions,
                     args.save_distances,
                     args.verbose,
                     args)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Graph reduction by coarsening')
    subparser = args_parser.add_subparsers()

    args_parser.add_argument('--root_dataset',
                             type=str,
                             required=True,
                             default='./data',
                             help='Root of the dataset')

    # Hyperparameters to test
    args_parser.add_argument('--alphas',
                             nargs='*',
                             default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                             type=float,
                             help='List of alphas to test')
    args_parser.add_argument('--ks',
                             nargs='*',
                             default=[3, 5, 7],
                             type=int,
                             help='List of ks to test (k being the number of neighbors for the KNN)')

    # Parameters used during the optimization process
    args_parser.add_argument('--size_splits',
                             nargs=3,
                             type=float,
                             default=[0.6, 0.2, 0.2],
                             help='Arguments that set the size of the splits'
                                  '(e.g., --size_split size_train size_val size_test')
    args_parser.add_argument('--parameters_edit_cost',
                             nargs='+',
                             default=(1., 1., 1., 1., 'euclidean'),
                             help='Tuple with the cost for the edit operations')

    args_parser.add_argument('--n_cores',
                             default=0,
                             type=int,
                             help='Set the number of cores to use.'
                                  'If n_cores == 0 then it is run without parallelization.'
                                  'If n_cores > 0 then use this number of cores')

    args_parser.add_argument('--save_predictions',
                             action='store_true',
                             help='save the predicted classes if activated')
    args_parser.add_argument('--save_distances',
                             action='store_true',
                             help='Save the GEDs between the train and test graphs')

    args_parser.add_argument('--folder_results',
                             type=str,
                             required=True,
                             help='Folder where to save the reduced graphs')

    args_parser.add_argument('-v',
                             '--verbose',
                             action='store_true',
                             help='Activate verbose print')

    parse_args = args_parser.parse_args()

    main(parse_args)
