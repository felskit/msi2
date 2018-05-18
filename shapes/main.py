from src.classifiers.geometric import GeometricClassifier
from src.classifiers.network import NetworkClassifier
from src.common.config import config
from src.common.extractor import ShapeExtractor
from src.data.types import FillMode, ShapeType

import argparse
import cv2
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--directory",
    required=True,
    help="path to directory with testing outlines"
)
arguments = parser.parse_args()

if not os.path.isdir(arguments.directory):
    raise ValueError("Supplied path is not a directory")

lower = np.array([240])
upper = np.array([255])
image_size = config["image_size"]
extractor = ShapeExtractor(lower, upper, image_size)
classifier = GeometricClassifier()

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
        regions = extractor.get_regions(image, fill_mode=FillMode.WHITE_ON_BLACK)
        for region in regions:
            result = classifier.classify(region)
            results[result] += 1
    print(results)
    break  # test just one image
