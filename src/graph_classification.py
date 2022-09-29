import os
import pathlib
from typing import List, Tuple

from graph_pkg_core.coordinator.coordinator import Coordinator

from src.train_eval import train, evaluate
from src.utils import set_global_verbose, Logger
from src.utils import train_val_test_split, write_GT_labels

LOGGER_FILE = 'results_general_GED.json'


def graph_classifier(root_dataset: str,
                     parameters_edit_cost: Tuple,
                     size_splits: List[float],
                     alphas: List[float],
                     ks: List[int],
                     seed: int,
                     n_cores: int,
                     folder_results: str,
                     save_gt_labels: bool,
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
        seed:
        n_cores:
        folder_results:
        save_gt_labels:
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
                                   LOGGER_FILE)
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
                                                                          test_size=size_test,
                                                                          random_state=seed)

    # Optimize the hyperparameters
    acc_tracker = train(coordinator,
                        logger,
                        alphas, ks,
                        X_train, X_val,
                        list(y_train), list(y_val),
                        n_cores)

    # Merge train and validation set for the final evaluation
    X_train = X_train + X_val
    y_train = list(y_train) + list(y_val)

    evaluate(coordinator,
             logger,
             acc_tracker,
             X_train, X_test,
             list(y_train), list(y_test),
             n_cores,
             folder_results,
             save_gt_labels, save_predictions, save_distances)
