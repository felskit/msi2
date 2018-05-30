from keras.layers import Dense, Dropout, Conv2D, Flatten
from keras.models import Sequential
from src.common.config import config
from src.utils.trainer import NetworkTrainer
import argparse
import numpy as np
import os

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--directory",
    required=True,
    help="path to directory with training shapes"
)
parser.add_argument(
    "-t",
    "--train-size",
    required=True,
    help="fraction representing the size of the training set extracted from the provided data set "
         "(e.g. 0.8 means that 80% of images are going to be used for training)"
)
arguments = parser.parse_args()

if not os.path.isdir(arguments.directory):
    raise ValueError("Supplied path is not a directory")

train_size = float(arguments.train_size)
if not 0.0 < train_size <= 1.0:
    raise ValueError("Supplied training set size is not a float from range (0, 1]")

# load training data
image_size = config["image_size"]
trainer = NetworkTrainer(image_size)
image_shape, classes = trainer.load_data(arguments.directory, train_size)

# define the model (for 1D vectors)
model = Sequential()
model.add(Dense(256, activation="relu", input_shape=(np.prod(image_shape),)))
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(classes, activation="softmax"))

# train and save the model
trainer.train_and_save(model, "./model/shapes_model_1d_vec.h5", batch_size=256, epochs=50, flatten=True, verbose=1)

# define the convolutional model
model = Sequential()
model.add(Conv2D(
    activation="relu",
    input_shape=(image_shape[0], image_shape[1], 1),
    filters=1,
    kernel_size=8,
    padding="valid"
))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(classes, activation="softmax"))

# train and save the model
trainer.train_and_save(model, "./model/shapes_model_2d_img.h5", batch_size=256, epochs=25, flatten=False, verbose=1)
