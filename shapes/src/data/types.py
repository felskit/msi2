from enum import Enum


class Region:
    """
    Represents a single thresholded region to be supplied to a classifier.
    """

    def __init__(self, image, x1, x2, y1, y2):
        """
        Constructor for the region.

        :param image: Thresholded input image region to classify.
        :type image: numpy.ndarray
        :param x1: Left boundary of the region, in input image coordinates.
        :type x1: int
        :param x2: Right boundary of the region, in input image coordinates.
        :type x2: int
        :param y1: Upper boundary of the region, in input image coordinates.
        :type y1: int
        :param y2: Lower boundary of the region, in input image coordinates.
        :type y2: int
        """
        self.image = image
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2


class ShapeType(Enum):
    """
    Enumeration type used for classification result.
    """
    TRIANGLE = "triangle"
    SQUARE = "square"
    STAR = "star"
    CIRCLE = "circle"


class FillMode(Enum):
    """
    Enumeration type used by ShapeExtractor to determine output color scheme.
    """
    BLACK_ON_WHITE = 0
    WHITE_ON_BLACK = 1
