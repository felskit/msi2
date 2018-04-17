import cv2
import numpy as np

from src.data.types import Region


class ShapeExtractor:
    """
    Module used to extract regions of image that contain potential shapes.
    """

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def get_regions(self, image):
        """
        Performs threshold filtering on provided image to find near-white shapes.
        Each shape is redrawn based on its contour in a separate image.
        Returned list contains single-channel images (white shape on black background).

        :param image: Input image pixel data.
        :type image: numpy.ndarray
        :return: List of images representing regions of the image with shapes.
        :rtype: list
        """
        # perform threshold filtering to get shape mask
        shape_mask = cv2.inRange(image, self.lower, self.upper)
        # get the contours of found shapes
        _, contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        regions = []
        # draw shapes on separate images
        for contour in contours:
            region = self._contour_to_image(contour)
            if region is not None:
                regions.append(region)
        return regions

    def _contour_to_image(self, contour):
        """
        Converts information about shape contour to single-channel black and white image.
        The shape will be white on black background with small margin around it.

        :param contour: Shape contour to convert.
        :type contour: numpy.ndarray
        :return: Image representing region of the image with the shape
        :rtype: Region
        """
        # calculate extreme points of the contour
        min_x = contour[contour[:, :, 0].argmin()][0][0]
        max_x = contour[contour[:, :, 0].argmax()][0][0]
        min_y = contour[contour[:, :, 1].argmin()][0][1]
        max_y = contour[contour[:, :, 1].argmax()][0][1]
        if max_x - min_x < 50 or max_y - min_y < 50:
            return None
        # TODO: change margins value or reduce to 1px at all
        # calculate the margin that shape will have on the output image
        x_margin = int(0.05 * (max_x - min_x))
        y_margin = int(0.05 * (max_y - min_y))
        # create a new single-channel black image
        image = np.uint8(np.zeros((max_y + y_margin, max_x + x_margin)))
        # fill provided contour
        cv2.fillPoly(image, pts=[contour], color=255)
        # crop the image based on margin
        x1 = max(min_x - x_margin, 0)
        x2 = min(max_x + x_margin, image.shape[1])
        y1 = max(min_y - y_margin, 0)
        y2 = min(max_y + y_margin, image.shape[0])
        image = image[y1:y2, x1:x2]
        return Region(image, x1, x2, y1, y2)
