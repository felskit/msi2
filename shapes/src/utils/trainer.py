from keras.utils import to_categorical
from random import shuffle
import cv2
import numpy as np
import os


class NetworkTrainer:
    """
    Neural network trainer.
    """

    def __init__(self, image_size):
        """
        Trainer constructor.

        :param image_size: Size of the images to be used during network training.
        :type image_size: int
        """
        self.image_shape = (image_size, image_size)
        self.train_images, self.test_images = [], []
        self.train_labels, self.test_labels = [], []
        self.classes = 0

    @staticmethod
    def _normalize(images):
        """
        Converts images from [0,255] range to [0,1] range.

        :param images: List of images to be normalized.
        :type images: list
        :return: Numpy array representing the images in [0,1] range.
        :rtype: numpy.array
        """
        images = np.array(images)
        images = images.astype("float32")
        images /= 255
        return images

    @staticmethod
    def _flatten(images, shape):
        """
        Flattens images to 1D vectors.

        :param images: Numpy array of images to be flattened.
        :type images: numpy.array
        :param shape: Dimension of the output vector.
        :type shape: numpy.array
        :return: Numpy array of 1D vectors representing the images.
        :rtype: numpy.array
        """
        return images.reshape(len(images), shape)

    def _reshape_for_convolution(self, images):
        """
        Reshapes images for use with the convolutional model.

        :param images: Numpy array of images.
        :type images: numpy.array
        :return: Reshaped Numpy array of images.
        :type images: numpy.array
        """
        images = np.array(images)
        return images.reshape(len(images), self.image_shape[0], self.image_shape[1], 1)

    def load_data(self, data_dir):
        """
        Initializes the trainer with training data.

        :param data_dir: Directory to training data set.
        :type data_dir: str
        :return: A tuple of training images shape and number of classes.
        :rtype: tuple
        """
        folders, images, labels = [], [], []
        for subdir, _, files in os.walk(data_dir):
            if not files:
                continue
            folder = subdir[subdir.rfind("\\") + 1:]
            folders.append(folder)
            for file in files:
                image = cv2.imread(os.path.join(subdir, file), flags=cv2.IMREAD_GRAYSCALE)
                images.append(cv2.resize(image, self.image_shape))
                labels.append(folders.index(folder))

        # break data into training and test sets
        self.train_images, self.test_images = [], []
        self.train_labels, self.test_labels = [], []

        to_train = 0
        image_label_pairs = list(zip(images, labels))
        shuffle(image_label_pairs)
        for image, label in image_label_pairs:
            if to_train < 5:
                self.train_images.append(image)
                self.train_labels.append(label)
                to_train += 1
            else:
                self.test_images.append(image)
                self.test_labels.append(label)
                to_train = 0

        # normalize data (so that image pixels are in [0,1] range)
        self.train_images = self._normalize(self.train_images)
        self.test_images = self._normalize(self.test_images)

        # determine the number of classes
        self.classes = len(np.unique(self.train_labels))

        # convert labels to categorical
        self.train_labels = to_categorical(np.array(self.train_labels))
        self.test_labels = to_categorical(np.array(self.test_labels))

        # return the shape of training data and number of classes
        return self.image_shape, self.classes

    def train_and_save(self, model, path, batch_size, epochs, flatten, verbose=0):
        """
        Builds, trains, tests (if specified) and saves the neural network model.
        Can be reused to train multiple models after loading the training data once.

        :param model: Defined Sequential neural network model.
        :type model: keras.models.Sequential
        :param path: Path to output model file.
        :type path: str
        :param batch_size: Batch size during training.
        :type batch_size: int
        :param epochs: Number of epochs during training.
        :type epochs: int
        :param flatten: Specifies whether input images should be flattened (for a model that recognizes 1D vectors)
                        or kept as a 2D image (for a model that recognizes 2D images).
        :type flatten: bool
        :param verbose: Defines the verbosity level of training process (0 - none, 1 - verbose, 2 - one line per epoch).
        :type verbose: int
        """
        if flatten:
            data_shape = np.prod(self.image_shape)
            train_data = self._flatten(self.train_images, data_shape)
            test_data = self._flatten(self.test_images, data_shape)
        else:
            train_data = self._reshape_for_convolution(self.train_images)
            test_data = self._reshape_for_convolution(self.test_images)

        # build the model
        model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        # train the model
        model.fit(
            train_data,
            self.train_labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(test_data, self.test_labels)
        )

        # test the model
        if verbose > 0:
            [test_loss, test_acc] = model.evaluate(test_data, self.test_labels)
            print("Results for trained model on test data: loss = {}, accuracy = {}".format(test_loss, test_acc))

        # save the model
        model.save(path)
