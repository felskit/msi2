from src.utils.parser import KnnParser
from src.utils.plotter import KnnPlotter

parser = KnnParser()
k, data = parser.parse_args()
plotter = KnnPlotter(k, data, 0.05)
plotter.plot()
