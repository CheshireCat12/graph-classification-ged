import argparse
import logging
import os
from collections import defaultdict, namedtuple
from itertools import product
from time import time
from typing import List, Tuple

import numpy as np
from graph_pkg_core.algorithm.knn import KNNClassifier
from graph_pkg_core.coordinator.coordinator import Coordinator
from graph_pkg_core.coordinator.graph_loader import train_val_test_split
from graph_pkg_core.utils.logger import Logger
from tqdm import tqdm


def train(coordinator,
          logger,
          alphas,
          ks,
          X_train,
          X_val,
          y_train,
          y_val,
          n_cores):
    is_parallel = n_cores > 0

    knn = KNNClassifier(coordinator.ged, parallel=is_parallel)
    knn.train(X_train, list(y_train))

    AccuracyTracker = namedtuple('AccuracyTracker', ['acc', 'best_alpha', 'best_k'])
    logger.data['best_acc'] = float('-inf')
    logger.data['best_params'] = (None, None)
    logger.data['hyperparameters_tuning'] = {
        'alphas': alphas,
        'ks': ks
    }
    logger.data['accuracies'] = []  # Contains tuple (acc, alpha, k)
    logger.data['prediction_times'] = []

    for alpha, k in tqdm(product(alphas, ks), total=len(alphas)*len(ks)):
        # Update alpha parameter
        coordinator.edit_cost.update_alpha(alpha)

        # Perform prediction
        start_time = time()
        predictions = knn.predict(X_val, k=k, num_cores=n_cores)
        prediction_time = time() - start_time

        # compute accuracy
        acc = (np.array(predictions) == y_val).sum() / len(X_val)

        logger.data['prediction_times'].append(prediction_time)
        logger.data['accuracies'].append((acc, alpha, k))



        print(f'alpha: {alpha}; k: {k}; acc: {acc}')
        print(f'Computation time: {prediction_time}')
    pass


def graph_classifier(root_dataset: str,
                     parameters_edit_cost: Tuple,
                     size_splits: List[float],
                     n_cores: int,
                     folder_results: str,
                     verbose: bool):
    # TODO: set the global verbose
    # TODO: Make sure the folder_results exists

    coordinator = Coordinator(parameters_edit_cost,
                              root_dataset)

    size_train, size_val, size_test = size_splits
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(coordinator.graphs,
                                                                          coordinator.classes,
                                                                          val_size=size_val,
                                                                          test_size=size_test)

    pass


def main(args):
    print(args)

    logger_filename = os.path.join(args.folder_results, 'results_general.json')
    logger = Logger(logger_filename)
    logger.data['parameters'] = vars(args)
    logger.save_data()

    # classification(coordinator, args.alphas, args.ks, X_train, X_val, y_train, y_val)

    print(len(y_train))
    pass


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
                             help='Set the number of cores to use.'
                                  'If n_cores == 0 then it is run without parallelization.'
                                  'If n_cores > 0 then use this number of cores')

    args_parser.add_argument('--save_predictions',
                             action='store_true',
                             help='save the predicted classes if activated')

    args_parser.add_argument('--folder_results',
                             type=str,
                             required=True,
                             help='Folder where to save the reduced graphs')

    args_parser.add_argument('-v',
                             '--verbose',
                             action='store_true',
                             help='Activate verbose print')

    parse_args = args_parser.parse_args()

    if parse_args.verbose:
        logging.basicConfig(level=logging.INFO)

    main(parse_args)
