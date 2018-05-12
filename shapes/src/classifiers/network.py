from keras.models import load_model
from src.data.types import ShapeType
import cv2
import numpy as np


class NetworkClassifier:
    """
    Neural network classifier.
    """

    def __init__(self, model_dir, flatten):
        """
        Classifier constructor.

        :param model_dir: Neural network model directory.
        :type model_dir: str
        :param flatten: Specifies whether input image should be flattened (for a model that recognizes 1D vectors)
                        or kept as a 2D image (for a model that recognizes 2D images).
        :type flatten: bool
        """
        self.model = load_model(model_dir)
        self.flatten = flatten
        self.prediction_threshold = 0.95

    def classify(self, region, verbose=False):
        """
        Performs classification on a single input image using a neural network.
        The input region image should be thresholded (only 1 bit per pixel allowed).
        The input region image should contain one black shape on a white background.

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

        # flatten the image
        if self.flatten:
            image = image.reshape(1, np.prod([image.shape]))

        # feed image into model
        prediction = self.model.predict(image)[0].tolist()

        # verify the prediction
        if max(prediction) > self.prediction_threshold:
            if prediction[0] == max(prediction):
                return ShapeType.CIRCLE
            if prediction[1] == max(prediction):
                return ShapeType.SQUARE
            if prediction[2] == max(prediction):
                return ShapeType.STAR
            if prediction[3] == max(prediction):
                return ShapeType.TRIANGLE
        else:
            return None
