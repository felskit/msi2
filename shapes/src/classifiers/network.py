from keras.models import load_model
from src.common.config import config
from src.data.types import ShapeType
import cv2
import numpy as np

img_size = 64
model = load_model('./model/shapes_model.h5')
data_dim = np.prod([img_size, img_size])


class NetworkClassifier:
    def __init__(self):
        self.image_size = 64

    def classify(self, region):
        image = region.image
        image_size = config["image_size"]
        image = cv2.resize(image, (image_size, image_size))

        # show image conditionally
        cv2.imshow("", image * 255)

        image = image.reshape(1, data_dim)
        image = image.astype('float32')

        # feed image into model
        prediction = model.predict(image)[0].tolist()

        p_val = .95
        if max(prediction) > p_val:
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
