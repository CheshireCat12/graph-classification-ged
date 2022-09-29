import logging
import os
from collections import namedtuple
from itertools import product
from time import time
from typing import List

import numpy as np
from graph_pkg_core.algorithm.knn import KNNClassifier
from graph_pkg_core.graph.graph import Graph
from graph_pkg_core.coordinator.coordinator import Coordinator
from tqdm import tqdm

from src.utils import write_GT_labels, write_distances, write_predictions
from src.utils import Logger

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


def evaluate(coordinator: Coordinator,
             logger: Logger,
             acc_tracker: AccuracyTracker,
             X_train: List[Graph],
             X_test: List[Graph],
             y_train: List[int],
             y_test: List[int],
             n_cores: int,
             folder_results: str,
             save_gt_labels: bool,
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

    if save_gt_labels:
        file_gt_labels = os.path.join(folder_results,
                                      'gt_labels.csv')
        write_GT_labels(file_gt_labels, list(y_train) + list(y_test))

    if save_predictions:
        file_predictions = os.path.join(folder_results,
                                        'predictions.csv')
        write_predictions(file_predictions, predictions, y_test)

    if save_distances:
        file_distances = os.path.join(folder_results,
                                      'distances.npy')
        write_distances(file_distances, knn.current_distances)
