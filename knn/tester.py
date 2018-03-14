import argparse
import os

import pandas as pd

from crossvalidate import generate_partition, cross_validate
from src.utils.metrics import get_metric

SUBSET_COUNT = 5

metrics = [
    (get_metric('manhattan', None), 'manhattan'),
    (get_metric('euclidean', None), 'euclidean'),
    (get_metric('chebyshev', None), 'chebyshev'),
    (get_metric('minkowski', 1.5), 'minkowski1.5'),
    (get_metric('minkowski', 3.0), 'minkowski3.0')
]

ks = list(range(1, 14, 2))

parser = argparse.ArgumentParser(description='k nearest neighbors testing script. '
                                             'Performs cross-validation for various metrics and values of k '
                                             'to determine optimal algorithm parameters for the supplied '
                                             'data set.')
parser.add_argument(
    'input',
    type=str,
    help='Path to a CSV file containing the training data.'
)

args = parser.parse_args()
file = args.input
data = pd.read_csv(file)
results = ['metric,k,result\n']
for metric, name in metrics:
    for k in ks:
        print('Cross-validation of {} for {} with k = {} starting...'.format(file, name, k))
        subsets = generate_partition(data, SUBSET_COUNT)
        result = cross_validate(k, metric, data, subsets)
        results.append('{},{},{}\n'.format(name, k, result))
with open('results.' + os.path.basename(file), 'w') as output:
    for result in results:
        output.write(result)
