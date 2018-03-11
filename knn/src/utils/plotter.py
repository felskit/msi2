import numpy as np
import matplotlib.pyplot as plt

from src.algorithms.knn import KnnClassifier


class KnnPlotter:
    def __init__(self, k, data, metric, delta):
        self.k = k
        self.data = data
        self.offset = delta / 2
        self.classifier = KnnClassifier(k, data, metric)

        lower_lim = -1
        upper_lim = 1 + 1e-3
        x_range = np.arange(lower_lim, upper_lim, delta)
        y_range = np.arange(lower_lim, upper_lim, delta)
        self.X, self.Y = np.meshgrid(x_range, y_range)

    def _classify_mesh(self):
        self.Z = np.zeros_like(self.X)
        nrow, ncol = self.X.shape
        for i in range(nrow):
            for j in range(ncol):
                point = (self.X[i, j] + self.offset, self.Y[i, j] + self.offset)
                self.Z[i, j] = self.classifier.classify(point)

    def plot(self):
        self._classify_mesh()
        fig, ax = plt.subplots()
        plt.set_cmap('Set1')
        ax.pcolormesh(self.X, self.Y, self.Z, edgecolors='None', alpha=0.2)
        ax.scatter(self.data['x'], self.data['y'], c=self.data['cls'], zorder=1)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        plt.savefig('plot.pdf')
        # plt.show()
