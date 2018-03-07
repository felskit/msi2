import heapq
import pandas as pd

from src.classes.point import Point


class KnnClassifier:
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.get_dist = KnnClassifier._get_euclidean_dist  # TODO: switch-based metrics (some string parameter?)

    def classify(self, point):
        q = self._queueify(self.data, point)
        labels = self._get_nearest_neighbors_labels(q)
        label_counts = pd.Series(labels).value_counts()
        max_labels = label_counts.index[label_counts == label_counts.max()]
        return int(pd.Series(max_labels).sample(n=1).iloc[0])

    def _queueify(self, data, point):
        q = []
        for _, row in data.iterrows():
            row_point = (row['x'], row['y'])
            q.append(Point(row['cls'], self.get_dist(row_point, point)))
        heapq.heapify(q)
        return q

    def _get_nearest_neighbors_labels(self, q):
        return [x.cls for x in heapq.nsmallest(self.k, q)]

    # TODO: maybe take these out to another class containing all metrics?

    @staticmethod
    def _get_euclidean_dist(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 1 / 2

    @staticmethod
    def _get_taxi_dist(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
