import argparse

import cv2
import os

from src.classifiers.geometric import GeometricClassifier
from src.common.extractor import ShapeExtractor
from src.data.types import ShapeType

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--directory",
    required=True,
    help="path of directory with testing outlines"
)
arguments = parser.parse_args()

if not os.path.isdir(arguments.directory):
    raise ValueError("Supplied path is not a directory")

extractor = ShapeExtractor()
classifier = GeometricClassifier()

for (basedir, _, filenames) in os.walk(arguments.directory):
    results = {
        ShapeType.TRIANGLE: 0,
        ShapeType.SQUARE: 0,
        ShapeType.STAR: 0,
        ShapeType.CIRCLE: 0,
        None: 0
    }
    for filename in filenames:
        full_path = os.path.join(basedir, filename)
        image = cv2.imread(full_path, flags=cv2.IMREAD_GRAYSCALE)
        regions = extractor.get_regions(image)
        for region in regions:
            result = classifier.classify(region)
            results[result] += 1
    print(results)
    break
