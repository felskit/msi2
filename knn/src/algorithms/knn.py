import heapq
import pandas as pd

from src.classes.point import Point


class KnnClassifier:
    def __init__(self, k, data):  # TODO: different metrics
        self.k = k
        self.data = data

    def _queueify(self, data, point):
        q = []
        for _, row in data.iterrows():
            q.append(Point(row['x'], row['y'], row['cls'], point))
        heapq.heapify(q)
        return q

    def _get_nearest_neighbors_labels(self, q):
        return [x.cls for x in heapq.nsmallest(self.k, q)]

    def classify(self, point):
        q = self._queueify(self.data, point)
        labels = self._get_nearest_neighbors_labels(q)
        counts = pd.Series(labels).value_counts()
        return int(pd.Series(counts.index[counts == counts.max()]).sample(n=1).iloc[0])
