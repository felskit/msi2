import argparse

import cv2
import os

from src.classifiers.geometric import GeometricClassifier
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
        # invert the image for now, since they are the opposite way we want them in the training set
        # should be white on black, not black on white
        image = cv2.bitwise_not(image)
        classifier = GeometricClassifier()
        result = classifier.classify(image)
        results[result] += 1
    print(results)
    break
