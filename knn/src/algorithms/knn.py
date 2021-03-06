import heapq

import numpy as np
import pandas as pd

from src.classes.point import Point


class KnnClassifier:
    def __init__(self, k, data, metric):
        self.k = k
        self.data = data
        self.get_dist = metric

    def classify(self, point):
        q = self._queueify(self.data, point)
        labels = self._get_nearest_neighbors_labels(q)
        label_counts = pd.Series(labels).value_counts()
        max_labels = label_counts.index[label_counts == label_counts.max()]
        return int(pd.Series(max_labels).sample(n=1).iloc[0])

    def _queueify(self, data, point):
        q = []
        for _, row in data.iterrows():
            row_coords = np.array(row[:-1])
            q.append(Point(row['cls'], self.get_dist(row_coords, point)))
        heapq.heapify(q)
        return q

    def _get_nearest_neighbors_labels(self, q):
        return [x.cls for x in heapq.nsmallest(self.k, q)]
