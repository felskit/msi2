from keras.layers import Dense, Dropout
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
arguments = parser.parse_args()

if not os.path.isdir(arguments.directory):
    raise ValueError("Supplied path is not a directory")

# load training data
image_size = config["image_size_network"]
trainer = NetworkTrainer(image_size)
image_shape, classes = trainer.load_data(arguments.directory)

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
