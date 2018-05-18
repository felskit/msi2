from src.classifiers.geometric import GeometricClassifier
from src.classifiers.network import NetworkClassifier
from src.common.config import config
from src.common.extractor import ShapeExtractor
from src.data.types import FillMode, ContourProcessingMode
from src.utils.capture import CaptureWindow
import cv2
import numpy as np


capture = CaptureWindow(0)
lower = np.array([0, 50, 50])
upper = np.array([15, 255, 255])
extractor = ShapeExtractor(lower, upper)
classifier = NetworkClassifier(model_dir="./model/shapes_model_2d_vec.h5", flatten=False)
image_size = config["image_size_network"]

while capture.running:
    read, frame = capture.next_frame()
    if read:
        regions = extractor.get_regions(
            frame,
            fill_mode=FillMode.BLACK_ON_WHITE,
            contour_processing_mode=ContourProcessingMode.MORPHOLOGICAL_CLOSING,
            output_shape=(image_size, image_size)
        )
        if regions is not None:
            for region in regions:
                result = classifier.classify(region, True)
                if result is not None:
                    capture.draw_recognized_region(region, result)
        capture.show_frame()
        capture.process_keys()
