def minkowski_factory(p):
    def minkowski(p1, p2):
        return (abs(p1[0] - p2[0]) ** p + abs(p1[1] - p2[1]) ** p) ** (1 / p)
    return minkowski


def chebyshev(p1, p2):
    return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))


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
