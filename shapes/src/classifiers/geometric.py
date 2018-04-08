import cv2
import numpy as np

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
        approximation = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approximation) == 3:
            return ShapeType.TRIANGLE
        if len(approximation) == 4:
            return self.check_aspect_ratio(approximation)
        if len(approximation) == 10:
            return self.check_star(approximation)
        return self.check_circle(contour, perimeter)

    @staticmethod
    def check_aspect_ratio(approximation):
        """
        Checks whether the supplied approximated contour is roughly a square.

        :param approximation: Approximated contour.
        :return: Determined shape type.
        """
        (_, _, width, height) = cv2.boundingRect(approximation)
        aspect_ratio = width / height
        return ShapeType.SQUARE if 0.8 <= aspect_ratio <= 1.2 else None

    @staticmethod
    def check_star(approximation):
        """
        Checks whether the supplied approximated contour is roughly a star.

        :param approximation: Approximated contour.
        :return: Determined shape type.
        """
        moments = cv2.moments(approximation)
        # calculate centroid of the contour
        centroid = np.array((moments['m10'] / moments['m00'], moments['m01'] / moments['m00']))
        # if the shape is really a star, odd vertices should have different distances from the centroid
        # than the even ones
        # so calculate the distances first
        distances = np.sqrt(np.sum(np.power(approximation - centroid, 2), 2))
        # and then calculate the standard deviation over the even and odd vertices' distances
        first_deviation = np.std(distances[::2])
        second_deviation = np.std(distances[1::2])
        if first_deviation <= 5 and second_deviation <= 5:
            return ShapeType.STAR
        else:
            return None

    @staticmethod
    def check_circle(contour, perimeter):
        """
        Checks whether the supplied approximated contour is roughly a circle.

        :param contour: Approximated contour.
        :param perimeter: The perimeter of the contour.
        :return: Determined shape type.
        """
        moments = cv2.moments(contour)
        radius_from_perimeter = perimeter / (2 * np.pi)
        area = moments['m00']
        radius_from_area = np.sqrt(area / np.pi)
        if np.abs(radius_from_area - radius_from_perimeter) <= 25:
            return ShapeType.CIRCLE
        else:
            return ShapeType.TRIANGLE
