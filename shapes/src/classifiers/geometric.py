import cv2

from src.data.types import ShapeType


class GeometricClassifier:
    def classify(self, image):
        """
        Performs classification on a single input image.
        The input image should be thresholded (only 1 bit per pixel allowed).
        The input image should contain one white shape on a black background.

        :param image: Input image pixel data.
        :type image: numpy.ndarray
        :return: Type of shape classified on the input image.
        :rtype: ShapeType
        """
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) < 2:
            return None
        contours = contours[1]
        if len(contours) != 1:
            return None
        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 3:
            return ShapeType.TRIANGLE
        elif len(approximation) == 4:
            return ShapeType.SQUARE
        elif len(approximation) == 10:
            return ShapeType.STAR  # VERY dicey
        return ShapeType.CIRCLE  # also VERY dicey