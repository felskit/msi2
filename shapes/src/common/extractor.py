from src.common.config import config
from src.data.types import FillMode, Region
import cv2
import numpy as np


class ShapeExtractor:
    """
    Module used to extract regions of image that contain potential shapes.
    """

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.image_margin = config["image_margin"]
        self.bounding_box_margin = config["bounding_box_margin"]

    def get_regions(self, image, mode, output_shape):
        """
        Performs threshold filtering on provided image to find near-white shapes.
        Each shape is redrawn based on its contour in a separate image.
        Returned list contains single-channel images (white shape on black background).

        :param image: Input image pixel data.
        :type image: numpy.ndarray
        :param mode: Color scheme to be used during region extraction.
        :type mode: types.FillMode
        :param output_shape: Shape of the output region images.
        :type output_shape: tuple
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
            region = self._contour_to_image(contour, mode, output_shape, image.shape)
            if region is not None:
                regions.append(region)

        return regions

    def _contour_to_image(self, contour, mode, output_shape, input_shape):
        """
        Converts information about shape contour to single-channel black and white image.
        The shape will be white on black background with small margin around it.
        Returned image pixels are in [0,1] range.

        :param contour: Shape contour to convert.
        :type contour: numpy.ndarray
        :param mode: Color scheme to be used during region extraction.
        :type mode: types.FillMode
        :param output_shape: Shape of the output region image.
        :type output_shape: tuple
        :param input_shape: Shape of the captured image.
        :type input_shape: tuple
        :return: Image representing region of the image with the shape
        :rtype: types.Region
        """
        # calculate extreme points of the contour
        min_x = contour[contour[:, :, 0].argmin()][0][0]
        max_x = contour[contour[:, :, 0].argmax()][0][0]
        min_y = contour[contour[:, :, 1].argmin()][0][1]
        max_y = contour[contour[:, :, 1].argmax()][0][1]

        # TODO: improve this (contours that are scattered / have many holes should be ignored)
        # ignore this shape if it doesn't meet the requirements
        if max_x - min_x < 50 or max_y - min_y < 50:
            return None

        # calculate the margin that shape will have and the output image size
        x_width = max_x - min_x
        y_width = max_y - min_y
        if x_width < y_width:
            shape_size = y_width
            x_fill_offset = (y_width - x_width) / 2
            y_fill_offset = 0
        else:
            shape_size = x_width
            x_fill_offset = 0
            y_fill_offset = (x_width - y_width) / 2
        margin = int(self.image_margin * shape_size)
        output_size = shape_size + 2 * margin
        x_fill_offset += margin - min_x
        y_fill_offset += margin - min_y

        # create a new single-channel image
        if mode == FillMode.BLACK_ON_WHITE:
            background = np.ones((output_size, output_size))
            fill_color = 0
        elif mode == FillMode.WHITE_ON_BLACK:
            background = np.zeros((output_size, output_size))
            fill_color = 1
        else:
            raise ValueError("Expected FillMode.BLACK_ON_WHITE or FillMode.WHITE_ON_BLACK, got " + str(mode))
        image = np.uint8(background)

        # TODO: fill holes in the contour (in a way that doesn't change extreme points or we'll need to calculate again)

        # convert the contour to fit the output image
        for i, _ in enumerate(contour):
            contour[i][0][0] += x_fill_offset
            contour[i][0][1] += y_fill_offset

        # fill provided contour
        cv2.fillPoly(image, pts=[contour], color=fill_color)

        # resize the image to spare some memory
        image = cv2.resize(image, output_shape)

        # calculate the bounding box extreme points
        x1 = max(min_x - self.bounding_box_margin, 0)
        x2 = min(max_x + self.bounding_box_margin, input_shape[1])
        y1 = max(min_y - self.bounding_box_margin, 0)
        y2 = min(max_y + self.bounding_box_margin, input_shape[0])

        return Region(image, x1, x2, y1, y2)
