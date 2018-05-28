from src.classifiers.geometric import GeometricClassifier
from src.classifiers.network import NetworkClassifier
from src.common.config import config
from src.common.extractor import ShapeExtractor
from src.data.types import FillMode, ShapeType

import argparse
import cv2
import numpy as np
import os

CHOICES = [
    'geometric',
    'vector-network',
    'convolutional-network'
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--directory",
    required=True,
    help="path to directory with testing outlines"
)
parser.add_argument(
    "-c",
    "--classifier",
    type=str,
    choices=CHOICES,
    required=True,
    help="name of classifier to use; options available: geometric, vector-network, convolutional-network"
)
arguments = parser.parse_args()

if not os.path.isdir(arguments.directory):
    raise ValueError("Supplied path is not a directory")

lower = np.array([240])
upper = np.array([255])
image_size = config["image_size"]
extractor = ShapeExtractor(lower, upper, image_size)
if arguments.classifier == "geometric":
    classifier = GeometricClassifier()
    fill_mode = FillMode.WHITE_ON_BLACK
elif arguments.classifier == "vector-network":
    classifier = NetworkClassifier("model/shapes_model_1d_vec.h5", True)
    fill_mode = FillMode.BLACK_ON_WHITE
elif arguments.classifier == "convolutional-network":
    classifier = NetworkClassifier("model/shapes_model_2d_img.h5", False)
    fill_mode = FillMode.BLACK_ON_WHITE
else:
    raise ValueError

for subdir, _, files in os.walk(arguments.directory):
    results = {
        ShapeType.TRIANGLE: 0,
        ShapeType.SQUARE: 0,
        ShapeType.STAR: 0,
        ShapeType.CIRCLE: 0,
        None: 0
    }
    for file in files:
        full_path = os.path.join(subdir, file)
        image = cv2.imread(full_path, flags=cv2.IMREAD_GRAYSCALE)
        regions = extractor.get_regions(image, fill_mode=fill_mode)
        for region in regions:
            result = classifier.classify(region)
            results[result] += 1
    print("Summary:")
    for key, value in results.items():
        print('\t{:20} - {:2} occurrences'.format(key or "Other", value))
    break  # don't walk the directory tree further
