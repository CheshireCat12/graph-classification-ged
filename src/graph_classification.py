from os.path import join, isfile
from pathlib import Path
from time import time
from typing import List, Tuple

import numpy as np
from cyged import Coordinator
from cyged import MatrixDistances
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from src.utils import set_global_verbose, Logger, write_distances, write_times, write_GT_labels, write_predictions, \
    save_acc_results, \
    seed_everything, load_graphs

PARAMETERS_FILE = 'parameters.json'
RESULTS_FILE = 'acc_results.json'


def load_distances(coordinator: Coordinator,
                   alphas: List[float],
                   n_cores: int,
                   folder_results: str) -> List[np.ndarray]:
    """
    The function loads or computes the GED matrices for each alpha value in the given list of alphas.
    If the GED matrices for a particular alpha value already exist in the specified folder, it loads the matrices from there.
    Otherwise, it computes the GED matrices using the MatrixDistances class and the coordinator.graphs attribute.
    The matrices are then stored in the specified folder for future use.

    Args:
        coordinator:
        alphas: A list of floats representing the alpha values for which the GED matrices need
                to be computed or loaded.
        n_cores: An integer representing the number of cores to be used for parallel computation of GED matrices
        folder_results: A string representing the path of the folder where the GED matrices
                        will be stored or loaded from

    Returns:
        A list of numpy arrays representing the GED matrices for all the alpha values.
    """
    is_parallel = n_cores > 0
    matrix_dist = MatrixDistances(coordinator.ged,
                                  parallel=is_parallel)
    distances = []
    times = []

    for alpha in tqdm(alphas, desc='Load or Compute GED matrices'):
        file_distances = join(folder_results,
                              'distances',
                              f'distances_alpha{alpha}.npy')

        file_times = join(folder_results,
                          f'ged_times.csv')

        # Check if the file containing the distances for the particular alpha exists
        if isfile(file_distances):
            # If yes load the distances
            dist = np.load(file_distances)
        else:
            # Otherwise compute the GEDs
            coordinator.edit_cost.update_alpha(alpha)
            time_a = time()
            dist = np.array(matrix_dist.calc_matrix_distances(coordinator.graphs,
                                                              coordinator.graphs,
                                                              num_cores=n_cores))
            time_b = time()
            times.append(time_b - time_a)

            write_distances(file_distances, dist)
            write_times(file_times, times)

        distances.append(dist)

    return distances


def reduce_best_params(best_params: List) -> Tuple[float, int, int]:
    """
    This function takes in a list of tuples, where each tuple contains a score,
    an integer k, and another integer idx_alpha.
    The function iterates through each tuple in the list, compares the score of
     the current tuple to a variable "best_score" initialized as negative infinity,
     and if the current score is greater, updates "best_score" and "best_idx" to
     the current score and index.

    Args:
        best_params: List of Tuples where each tuple contains a score, an integer k, and another integer idx_alpha.

    Returns:
       A Tuple containing the highest score , k and idx_alpha.
    """
    best_idx, best_score = None, float('-inf')
    for idx, (score, k, idx_alpha) in enumerate(best_params):
        if score > best_score:
            best_idx, best_score = idx, score

    return best_params[best_idx]


def cross_validate(distances: List[np.ndarray],
                   classes: np.ndarray,
                   param_grid: dict,
                   inner_cv, outer_cv,
                   n_cores: int,
                   scoring: dict,
                   save_predictions: bool,
                   current_trial: int,
                   folder_results: str):
    scores = {'test_' + name_score: [] for name_score in scoring.keys()}

    for idx_outer, (train_index, test_index) in enumerate(outer_cv.split(distances[0], classes)):
        best_params = []

        # Perform grid search on all the alphas and ks to select the bests
        for alpha_idx, alpha_dist in enumerate(distances):
            clf = GridSearchCV(estimator=KNeighborsClassifier(metric='precomputed'),
                               param_grid=param_grid,
                               n_jobs=n_cores,
                               cv=inner_cv)
            clf.fit(alpha_dist[np.ix_(train_index, train_index)], classes[train_index])

            best_params.append((clf.best_score_, clf.best_params_['n_neighbors'], alpha_idx))

        # Retrieve the best hyperparameters
        _, best_k, best_idx_alpha = reduce_best_params(best_params)
        alpha_dist = distances[best_idx_alpha]

        # Retrain KNN with the best alpha and k and perform the final classification on test set
        knn_test = KNeighborsClassifier(n_neighbors=best_k,
                                        metric='precomputed')
        knn_test.fit(alpha_dist[np.ix_(train_index, train_index)],
                     classes[train_index])
        test_predictions = knn_test.predict(alpha_dist[np.ix_(test_index, train_index)])

        if save_predictions:
            filename = join(folder_results,
                            'predictions',
                            f'predictions_trial_{current_trial}_outer_{idx_outer}.csv')
            write_predictions(filename, test_predictions, classes[test_index])

        for scorer_name, scorer in scoring.items():
            current_scorer = get_scorer(scorer)
            score = current_scorer._score_func(classes[test_index], test_predictions)

            scores['test_' + scorer_name].append(score)

    return scores


def graph_classifier(root_dataset: str,
                     graph_format: str,
                     parameters_edit_cost: Tuple,
                     alphas: List[float],
                     ks: List[int],
                     n_trial: int,
                     n_outer_cv: int,
                     n_inner_cv: int,
                     n_cores: int,
                     folder_results: str,
                     save_gt_labels: bool,
                     save_predictions: bool,
                     verbose: bool,
                     args):
    """

    Args:
        root_dataset:
        parameters_edit_cost:
        alphas:
        ks:
        n_trial:
        n_outer_cv:
        n_inner_cv:
        n_cores:
        folder_results:
        save_gt_labels:
        save_predictions:
        verbose:
        args:

    Returns:

    """
    set_global_verbose(verbose)
    seed_everything(7)

    # Create folders used later
    Path(folder_results).mkdir(parents=True, exist_ok=True)
    Path(join(folder_results, 'distances')).mkdir(parents=True, exist_ok=True)
    Path(join(folder_results, 'predictions')).mkdir(parents=True, exist_ok=True)

    # Init logger
    logger_filename = join(folder_results,
                           PARAMETERS_FILE)
    logger = Logger(logger_filename)

    # Save all the input parameters
    logger.data['parameters'] = vars(args)
    logger.save_data()

    graphs, lbls = load_graphs(root_dataset=root_dataset,
                               file_extension=graph_format,
                               load_classes=True)

    coordinator = Coordinator(parameters_edit_cost,
                              graphs=graphs,
                              classes=lbls)

    distances = load_distances(coordinator, alphas, n_cores, folder_results)

    if save_gt_labels:
        file_gt_labels = join(folder_results,
                              'distances',
                              'gt_labels.csv')
        write_GT_labels(file_gt_labels, coordinator.classes)

    param_grid = {'n_neighbors': ks}
    scoring = {'acc': 'accuracy',
               'balanced_acc': 'balanced_accuracy',
               }

    trial_results = []
    for c_seed in range(n_trial):
        outer_cv = StratifiedKFold(n_splits=n_outer_cv, shuffle=True, random_state=c_seed)
        inner_cv = StratifiedKFold(n_splits=n_inner_cv, shuffle=True, random_state=c_seed)

        scores = cross_validate(distances=distances,
                                classes=coordinator.classes,
                                param_grid=param_grid,
                                inner_cv=inner_cv,
                                outer_cv=outer_cv,
                                n_cores=n_cores,
                                scoring=scoring,
                                save_predictions=save_predictions,
                                current_trial=c_seed,
                                folder_results=folder_results)
        trial_results.append(scores)
        save_acc_results(join(folder_results, RESULTS_FILE), trial_results)
