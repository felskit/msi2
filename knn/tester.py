import pandas as pd

from crossvalidate import generate_partition, cross_validate
from src.utils.metrics import get_metric

SUBSET_COUNT = 5

files = [
    'data.simple.train.1000.csv',
    'data.three_gauss.train.1000.csv'
]

metrics = [
    (get_metric('manhattan', None), 'manhattan'),
    (get_metric('euclidean', None), 'euclidean'),
    (get_metric('chebyshev', None), 'chebyshev'),
    (get_metric('minkowski', 1.5), 'minkowski1.5'),
    (get_metric('minkowski', 3.0), 'minkowski3.0')
]

ks = list(range(1, 14, 2))

for file in files:
    with open('input/' + file, 'r') as raw_data:
        data = pd.read_csv(raw_data)
        results = ['metric,k,result\n']
        for metric, name in metrics:
            for k in ks:
                print('Cross-validation of {} for {} with k = {} starting...'.format(file, name, k))
                subsets = generate_partition(data, SUBSET_COUNT)
                result = cross_validate(k, metric, data, subsets)
                results.append('{},{},{}\n'.format(name, k, result))
        with open('results.' + file, 'w') as output:
            for result in results:
                output.write(result)
