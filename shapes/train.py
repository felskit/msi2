import numpy as np
import argparse
import cv2
import os

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling2D

from random import shuffle

image_size = 64
folders = []
images = []
labels = []

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


# TODO: common.image_processor.py
# converts each image from
def flatten(_data_dim, _images):
    _images = np.array(_images)
    _images = _images.reshape(len(_images), _data_dim)
    _images = _images.astype('float32')
    _images /= 255
    return _images


# -------------get train/test data-----------------
# get data
for subdir, _, files in os.walk(arguments.directory):
    if not files:
        continue
    folder = subdir[subdir.rfind('\\') + 1:]
    folders.append(folder)
    for file in files:
        image = cv2.imread(os.path.join(subdir, file), 0)  # load training images in greyscale
        images.append(cv2.resize(image, (image_size, image_size)))
        labels.append(folders.index(folder))

# break data into training and test sets
train_images = []
test_images = []
train_labels = []
test_labels = []

to_train = 0
image_label_pairs = list(zip(images, labels))
shuffle(image_label_pairs)
for image, label in image_label_pairs:
    if to_train < 5:
        train_images.append(image)
        train_labels.append(label)
        to_train += 1
    else:
        test_images.append(image)
        test_labels.append(label)
        to_train = 0

# flatten data
data_dim = np.prod(images[0].shape)
train_data = flatten(data_dim, train_images)
test_data = flatten(data_dim, test_images)

# change labels to categorical
train_labels_cat = to_categorical(np.array(train_labels))
test_labels_cat = to_categorical(np.array(test_labels))

# determine the number of classes
classes = np.unique(train_labels)
classes_count = len(classes)

# define the model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(data_dim,)))
# model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(classes_count, activation='softmax'))

# build the model
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
model.fit(
    train_data,
    train_labels_cat,
    batch_size=256,
    epochs=50,
    verbose=1,
    validation_data=(test_data, test_labels_cat)
)

# test model
[test_loss, test_acc] = model.evaluate(test_data, test_labels_cat)
print("Results for trained model on test data: loss = {}, accuracy = {}".format(test_loss, test_acc))

# save model
model.save('./model/shapes_model.h5')
