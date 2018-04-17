import cv2
import numpy as np

from src.classifiers.geometric import GeometricClassifier
from src.common.extractor import ShapeExtractor

capture = cv2.VideoCapture(0)
lower = np.array([0, 50, 50])
upper = np.array([15, 255, 255])
highlight_color = (0, 255, 0)
extractor = ShapeExtractor(lower, upper)
classifier = GeometricClassifier()


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
        regions = extractor.get_regions(hsv)
        for region in regions:
            result = classifier.classify(region)
            if result is not None:
                draw_recognized_region(frame, region, result)
        cv2.imshow('Shapes', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()