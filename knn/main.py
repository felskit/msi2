from src.utils.parser import KnnParser
from src.utils.plotter import KnnPlotter

parser = KnnParser()
k, data, metric = parser.parse_args()
plotter = KnnPlotter(k, data, metric, 0.05)
plotter.plot()
