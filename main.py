import argparse

from src.graph_classification import graph_classifier


def main(args):
    graph_classifier(args.root_dataset,
                     args.graph_format,
                     args.parameters_edit_cost,
                     args.alphas,
                     args.ks,
                     args.n_trials,
                     args.n_outer_cv,
                     args.n_inner_cv,
                     args.n_cores,
                     args.folder_results,
                     args.save_gt_labels,
                     args.save_predictions,
                     args.verbose,
                     args)


DEFAULT_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_KS = [3, 5, 7]

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser(description='Graph Classification Using KNN with GED')
    subparser = args_parser.add_subparsers()

    args_parser.add_argument('--root-dataset',
                             type=str,
                             required=True,
                             default='./data',
                             help='Root of the dataset')
    args_parser.add_argument('--graph-format',
                             type=str,
                             default='graphml',
                             help='Root of the dataset')

    args_parser.add_argument('--parameters-edit-cost',
                             nargs='+',
                             default=(1., 1., 1., 1., 'euclidean'),
                             help='Tuple with the cost for the edit operations')

    # Hyperparameters to test
    args_parser.add_argument('--alphas',
                             nargs='*',
                             default=DEFAULT_ALPHAS,
                             type=float,
                             help='List of alphas to test')
    args_parser.add_argument('--ks',
                             nargs='*',
                             default=DEFAULT_KS,
                             type=int,
                             help='List of ks to test (k being the number of neighbors for the KNN)')

    # Parameters used during the optimization process
    args_parser.add_argument('--n-trials',
                             default=10,
                             type=int,
                             help='Number of cross-validation to perform')
    args_parser.add_argument('--n-outer-cv',
                             default=10,
                             type=int,
                             help='Number of outer loops in the cross-validation')
    args_parser.add_argument('--n-inner-cv',
                             default=5,
                             type=int,
                             help='Number of inner loops in the cross-validation')
    args_parser.add_argument('--n-cores',
                             default=0,
                             type=int,
                             help='Set the number of cores to use.'
                                  'If n_cores == 0 then it is run without parallelization.'
                                  'If n_cores > 0 then use this number of cores')

    args_parser.add_argument('--save-gt-labels',
                             action='store_true',
                             help='save the ground truth classes if activated')
    args_parser.add_argument('--save-predictions',
                             action='store_true',
                             help='save the predicted classes if activated')

    args_parser.add_argument('--folder-results',
                             type=str,
                             required=True,
                             help='Folder where to save the classification results')

    args_parser.add_argument('-v',
                             '--verbose',
                             action='store_true',
                             help='Activate verbose print')

    parse_args = args_parser.parse_args()

    if parse_args.alphas == [0] and parse_args.ks == [0]:
        parse_args.alphas = DEFAULT_ALPHAS
        parse_args.ks = DEFAULT_KS

    main(parse_args)
