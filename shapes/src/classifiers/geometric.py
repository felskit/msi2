from src.data.types import ShapeType
import cv2
import numpy as np


class GeometricClassifier:
    """
    OpenCV geometric classifier.
    """

    def classify(self, region, verbose=False):
        """
        Performs classification on a single input image.
        The input region image should be thresholded (only 1 bit per pixel allowed).
        The input region image should contain one white shape on a black background.

        :param region: The region being classified.
        :type region: src.data.types.Region
        :param verbose: Specifies whether the classified image should be shown in a separate window.
        :type verbose: bool
        :return: Type of shape classified on the input image.
        :rtype: ShapeType
        """
        image = region.image
        if verbose:
            cv2.imshow("", image * 255)

        # find shape contours
        _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ignore if the image doesn't have exactly one shape
        if len(contours) != 1:
            return None

        # approximate the contour
        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

        # verify the approximation
        if len(approximation) == 3:
            return ShapeType.TRIANGLE
        if len(approximation) == 4:
            return self._check_square(approximation)
        if len(approximation) == 10:
            return self._check_star(approximation)
        return self._check_circle(contour, perimeter)

    @staticmethod
    def _check_square(approximation):
        """
        Checks whether the supplied approximated contour is roughly a square.

        :param approximation: Approximated contour.
        :return: Determined shape type.
        """
        (_, _, width, height) = cv2.boundingRect(approximation)
        aspect_ratio = width / height
        perimeter = cv2.arcLength(approximation, True)
        moments = cv2.moments(approximation)
        area = moments['m00']
        side_from_perimeter = perimeter / 4
        side_from_area = np.sqrt(area)
        if np.abs(side_from_perimeter - side_from_area) <= 15 and 0.8 <= aspect_ratio <= 1.2:
            return ShapeType.SQUARE
        else:
            return None

    @staticmethod
    def _check_star(approximation):
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
    def _check_circle(contour, perimeter):
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
            return None
