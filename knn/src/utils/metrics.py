import numpy as np


def minkowski_factory(p):
    def minkowski(p1, p2):
        return np.sum(np.abs(p1 - p2) ** p) ** (1 / p)
    return minkowski


def chebyshev(p1, p2):
    return np.max(np.abs(p1 - p2))


METRIC_CHOICES = [
    'euclidean',
    'chebyshev',
    'manhattan',
    'minkowski'
]


def get_metric(name, p):
    if name == 'euclidean':
        return minkowski_factory(2)
    elif name == 'chebyshev':
        return chebyshev
    elif name == 'manhattan':
        return minkowski_factory(1)
    elif name == 'minkowski':
        return minkowski_factory(p)
    else:
        raise ValueError('Invalid metric choice')
