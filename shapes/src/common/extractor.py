from src.common.config import config
from src.data.types import FillMode, Region, ContourProcessingMode
import cv2
import numpy as np


class ShapeExtractor:
    """
    Module used to extract regions of image that contain potential shapes.
    """

    def __init__(self, lower, upper, image_size):
        self.lower = lower
        self.upper = upper
        self.bounding_box_margin = config["bounding_box_margin"]
        self.shape_margin = config["shape_margin"]
        self.output_shape = (image_size, image_size)

    def get_regions(self, image, fill_mode, contour_processing_mode=ContourProcessingMode.NONE):
        """
        Performs threshold filtering on provided image to find near-white shapes.
        Each shape is redrawn based on its contour in a separate image.
        Returned list contains single-channel images (white shape on black background).

        :param image: Input image pixel data.
        :type image: numpy.ndarray
        :param fill_mode: Color scheme to be used during region extraction.
        :type fill_mode: FillMode
        :param contour_processing_mode: Contour processing mode to use when thresholding input images.
        :type contour_processing_mode: ContourProcessingMode
        :return: List of images representing regions of the image with shapes.
        :rtype: list
        """
        # perform threshold filtering to get shape mask
        shape_mask = cv2.inRange(image, self.lower, self.upper)

        # morphological closing to fill up gaps
        if contour_processing_mode == ContourProcessingMode.MORPHOLOGICAL_CLOSING:
            kernel = np.ones((10, 10), np.uint8)
            shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_CLOSE, kernel)
        elif contour_processing_mode != ContourProcessingMode.NONE:
            raise ValueError(
                "Expected ContourProcessingMode.NONE or ContourProcessingMode.MORPHOLOGICAL_CLOSING, got "
                + str(contour_processing_mode)
            )

        # get the contours of found shapes
        _, contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        regions = []

        # draw shapes on separate images
        for contour in contours:
            region = self._contour_to_image(contour, fill_mode, image.shape)
            if region is not None:
                regions.append(region)

        return regions

    def _contour_to_image(self, contour, fill_mode, input_shape):
        """
        Converts information about shape contour to single-channel black and white image.
        The shape will be white on black background with small margin around it.
        Returned image pixels are in [0,1] range.

        :param contour: Shape contour to convert.
        :type contour: numpy.ndarray
        :param fill_mode: Color scheme to be used during region extraction.
        :type fill_mode: types.FillMode
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

        # ignore this shape if it doesn't meet the requirements
        if max_x - min_x < 50 or max_y - min_y < 50:
            return None

        # calculate the margin that shape will have and the temporary canvas image size
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
        margin = int(self.shape_margin * shape_size)
        canvas_size = shape_size + 2 * margin
        canvas_shape = (canvas_size, canvas_size)
        x_fill_offset += margin - min_x
        y_fill_offset += margin - min_y

        # create a new single-channel image
        if fill_mode == FillMode.BLACK_ON_WHITE:
            background = np.ones(canvas_shape)
            fill_color = 0
        elif fill_mode == FillMode.WHITE_ON_BLACK:
            background = np.zeros(canvas_shape)
            fill_color = 1
        else:
            raise ValueError(
                "Expected FillMode.BLACK_ON_WHITE or FillMode.WHITE_ON_BLACK, got "
                + str(fill_mode)
            )
        image = np.uint8(background)

        # convert the contour to fit the output image
        for i, _ in enumerate(contour):
            contour[i][0][0] += x_fill_offset
            contour[i][0][1] += y_fill_offset

        # fill provided contour
        cv2.fillPoly(image, pts=[contour], color=fill_color)

        # resize the image to spare some memory
        image = cv2.resize(image, self.output_shape)

        # calculate the bounding box extreme points
        x1 = max(min_x - self.bounding_box_margin, 0)
        x2 = min(max_x + self.bounding_box_margin, input_shape[1])
        y1 = max(min_y - self.bounding_box_margin, 0)
        y2 = min(max_y + self.bounding_box_margin, input_shape[0])

        return Region(image, x1, x2, y1, y2)
