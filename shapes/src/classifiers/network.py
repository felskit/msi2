import cv2
import numpy as np
from keras.models import load_model

from src.data.types import ShapeType

img_size = 64
model = load_model('./model/shapes_model.h5')
data_dim = np.prod([img_size, img_size])


class NetworkClassifier:
    def classify(self, region):
        image = region.image
        image = 255 - image

        cv2.imshow("", image)
        # cv2.waitKey(0)

        image = cv2.resize(image, (img_size, img_size))
        image = image.reshape(1, data_dim)
        image = image.astype('float32')
        image /= 255

        # feed image into model
        prediction = model.predict(image)[0].tolist()

        p_val = .95
        print(prediction)
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
