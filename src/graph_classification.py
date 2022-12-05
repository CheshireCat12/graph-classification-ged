import os
import pathlib
from typing import List, Tuple

import sklearn.metrics
from graph_pkg_core.coordinator.coordinator import Coordinator
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, get_scorer

from src.train_eval import train, evaluate
from src.utils import set_global_verbose, Logger
from src.utils import train_val_test_split, AccuracyTracker

LOGGER_FILE = 'results_general_GED.json'

from graph_pkg_core.algorithm.matrix_distances import MatrixDistances
from tqdm import tqdm
import numpy as np
from typing import List


def reduce_best_params(best_params: List) -> Tuple[float, int, int]:
    best_score = float('-inf')
    best_idx = None
    for idx, (score, k, idx_alpha) in enumerate(best_params):

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_params[best_idx]


from collections import defaultdict


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

    sub_index = 600

    is_parallel = n_cores > 0
    matrix_dist = MatrixDistances(coordinator.ged,
                                  parallel=is_parallel)
    distances = []
    for alpha in tqdm(alphas, desc='Compute GED matrix'):
        coordinator.edit_cost.update_alpha(alpha)
        dist = matrix_dist.calc_matrix_distances(coordinator.graphs[:sub_index],
                                                 coordinator.graphs[:sub_index],
                                                 num_cores=n_cores)
        distances.append(np.array(dist))

    y = coordinator.classes[:sub_index]

    n_trial = 1

    param_grid = {'n_neighbors': ks}
    scoring = {'acc': 'accuracy',
               # 'balanced_acc': 'balanced_accuracy',
               # 'f1_macro': 'f1_macro',
               # 'f1_micro': 'f1_micro',
               # 'precision_macro': 'precision_macro',
               # 'recall_macro': 'recall_macro',
               }
    for c_seed in range(n_trial):
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=c_seed)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=c_seed)

        scores = {'test_' + name_score: [] for name_score in scoring.keys()}
        for train_index, test_index in outer_cv.split(distances[0], y):

            best_params = []
            for alpha_idx, alpha_dist in enumerate(distances):
                clf = GridSearchCV(estimator=KNeighborsClassifier(metric='precomputed'),
                                   param_grid=param_grid,
                                   n_jobs=n_cores,
                                   cv=inner_cv)
                clf.fit(alpha_dist[np.ix_(train_index, train_index)], y[train_index])

                best_params.append((clf.best_score_, clf.best_params_['n_neighbors'], alpha_idx))

            _, best_k, best_idx_alpha = reduce_best_params(best_params)
            alpha_dist = distances[best_idx_alpha]
            knn_test = KNeighborsClassifier(n_neighbors=best_k,
                                            metric='precomputed')

            knn_test.fit(alpha_dist[np.ix_(train_index, train_index)],
                         y[train_index])
            test_predictions = knn_test.predict(alpha_dist[np.ix_(test_index, train_index)])
            for scorer_name, scorer in scoring.items():
                current_scorer = get_scorer(scorer)
                score = current_scorer._score_func(y[test_index], test_predictions)

                scores['test_' + scorer_name].append(score)

        print(scores)
        print('***')
        # knn = KNeighborsClassifier(n_neighbors=5, metric='precomputed')

        # knn.fit(distances[0][np.ix_(train_index, train_index)], y[train_index])

        # knn.predict(distances[0][np.ix_(test_index, train_index)])

    # # Split the dataset into train, val and test sets
    # size_train, size_val, size_test = size_splits
    # X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(coordinator.graphs,
    #                                                                       coordinator.classes,
    #                                                                       val_size=size_val,
    #                                                                       test_size=size_test,
    #                                                                       random_state=seed)

    # # Check if there is only a single value for the alphas and ks
    # if len(alphas) * len(ks) == 1:
    #     acc_tracker = AccuracyTracker(-1, alphas[0], ks[0])
    # else:
    #     # Optimize the hyperparameters
    #     acc_tracker = train(coordinator,
    #                         logger,
    #                         alphas, ks,
    #                         X_train, X_val,
    #                         list(y_train), list(y_val),
    #                         n_cores)

    # # Merge train and validation set for the final evaluation
    # X_train = X_train + X_val
    # y_train = list(y_train) + list(y_val)

    # evaluate(coordinator,
    #          logger,
    #          acc_tracker,
    #          X_train, X_test,
    #          list(y_train), list(y_test),
    #          n_cores,
    #          folder_results,
    #          save_gt_labels, save_predictions, save_distances)
