import argparse

from src.graph_classification import graph_classifier


def main(args):
    graph_classifier(args.root_dataset,
                     args.parameters_edit_cost,
                     args.size_splits,
                     args.alphas,
                     args.ks,
                     args.seed,
                     args.n_cores,
                     args.folder_results,
                     args.save_gt_labels,
                     args.save_predictions,
                     args.save_distances,
                     args.verbose,
                     args)


DEFAULT_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_KS = [3, 5, 7]

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
                             default=DEFAULT_ALPHAS,
                             type=float,
                             help='List of alphas to test')
    args_parser.add_argument('--ks',
                             nargs='*',
                             default=DEFAULT_KS,
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

    args_parser.add_argument('--seed',
                             default=1,
                             type=int,
                             help='Choose the random seed')
    args_parser.add_argument('--n_cores',
                             default=0,
                             type=int,
                             help='Set the number of cores to use.'
                                  'If n_cores == 0 then it is run without parallelization.'
                                  'If n_cores > 0 then use this number of cores')

    args_parser.add_argument('--save_gt_labels',
                             action='store_true',
                             help='save the ground truth classes if activated')
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

    if parse_args.alphas == [0] and parse_args.ks == [0]:
        parse_args.alphas = DEFAULT_ALPHAS
        parse_args.ks = DEFAULT_KS

    main(parse_args)
