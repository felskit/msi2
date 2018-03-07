import argparse
import pandas

from src.utils.metrics import METRIC_CHOICES, get_metric


class KnnParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Data classification using k nearest neighbors algorithm.')
        self.parser.add_argument(
            'input',
            type=str,
            help='Path to a CSV file containing the training data.'
        )
        self.parser.add_argument(
            '-k',
            type=int,
            default=1,
            help='Number of neighbors to include when classifying points.'
        )
        self.parser.add_argument(
            '-m', '--metric',
            type=str,
            choices=METRIC_CHOICES,
            default=METRIC_CHOICES[0],
            help="""Metric used for calculating distances.
            Supported choices are: Euclidean, Chebyshev, Manhattan and Minkowski norms.
            When using the Minkowski norm, you can supply the value of p using the -p option."""
        )
        self.parser.add_argument(
            '-p',
            type=float,
            default=2.0,
            help="""The value of the argument p used in the Minkowski metric.
            When using other metrics, the value of this argument will be ignored."""
        )

    def parse_args(self):
        args = vars(self.parser.parse_args())
        return args['k'], pandas.read_csv(args['input']), get_metric(args['metric'], args['p'])
