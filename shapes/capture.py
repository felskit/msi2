from src.classifiers.geometric import GeometricClassifier
from src.classifiers.network import NetworkClassifier
from src.common.config import config
from src.common.extractor import ShapeExtractor
from src.data.types import FillMode
import cv2
import numpy as np

# TODO: possibly refactor this a bit (some generic CameraCapture class that can be used later in the "time trial")

capture = cv2.VideoCapture(0)
lower = np.array([0, 50, 50])
upper = np.array([15, 255, 255])
highlight_color = (0, 255, 0)
extractor = ShapeExtractor(lower, upper)
classifier = NetworkClassifier(model_dir="./model/shapes_model_1d_vec.h5", flatten=True)
image_size = config["image_size_network"]


def is_key_pressed(key):
    return cv2.waitKey(1) & 0xff == ord(key)


def draw_recognized_region(frame, region, result):
    cv2.rectangle(
        frame,
        (region.x1, region.y1),
        (region.x2, region.y2),
        color=highlight_color,
        thickness=3
    )
    cv2.putText(
        frame,
        result.name,
        (region.x1 + 10, region.y2 - 10),
        cv2.FONT_HERSHEY_DUPLEX,
        1.5,
        highlight_color,
        2
    )


while True:
    read, frame = capture.read()
    if read:
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        regions = extractor.get_regions(hsv, mode=FillMode.BLACK_ON_WHITE, output_shape=(image_size, image_size))
        for region in regions:
            result = classifier.classify(region, True)
            if result is not None:
                draw_recognized_region(frame, region, result)
        cv2.imshow('Shapes', frame)
    if is_key_pressed('q'):
        break

capture.release()
cv2.destroyAllWindows()
