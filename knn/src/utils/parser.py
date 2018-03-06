import argparse
import pandas


class KnnParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Data classification using k nearest neighbors algorithm.')
        self.parser.add_argument('-k', type=int, required=True, help='k parameter')
        self.parser.add_argument('-i', '--input', type=str, required=True, help='training data')

    def parse_args(self):
        args = vars(self.parser.parse_args())
        return args['k'], pandas.read_csv(args['input'])
