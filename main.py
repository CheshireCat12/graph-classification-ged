import argparse
import logging

from graph_pkg_core.loader.graph_loader import load_graphs
from graph_pkg_core.loader.graph_loader import train_val_test_split
from graph_pkg_core.coordinator.coordinator import Coordinator


def main(args):
    coordinator = Coordinator(args.root_dataset, (1., 1., 1., 1., 'euclidean'))
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(coordinator.graphs,
                                                                          coordinator.classes,
                                                                          val_size=0.2,
                                                                          test_size=0.2)

    print(len(y_train))
    pass


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Graph reduction by coarsening')
    subparser = args_parser.add_subparsers()

    args_parser.add_argument('--dataset',
                             type=str,
                             help='Graph dataset to reduce (the dataset has to be in TUDataset)')
    args_parser.add_argument('--root_dataset',
                             type=str,
                             required=True,
                             default='./data',
                             help='Root of the dataset')

    args_parser.add_argument('-v',
                             '--verbose',
                             action='store_true',
                             help='Activate verbose print')

    parse_args = args_parser.parse_args()

    if parse_args.verbose:
        logging.basicConfig(level=logging.INFO)

    main(parse_args)