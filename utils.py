"""
In this module auxiliary functions are defined for the main module.
"""
import logging
import os
from typing import Any, List

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_niblack, threshold_sauvola
from sklearn.metrics import confusion_matrix

from encoders import CircularThermometerEncoder, ThermometerEncoder

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def load_data(dataset_path: str) -> tuple:
    """
    Load the data from the data directory and return it as a tuple.

    Returns:
        tuple: A tuple containing the training and test data and labels.
    """

    def load(fname: str) -> np.ndarray:
        f: str = os.path.join(fname)
        return np.load(f)["arr_0"]

    X_train = load(os.path.join(dataset_path, 'kmnist-train-imgs.npz'))
    X_test = load(os.path.join(dataset_path, 'kmnist-test-imgs.npz'))
    y_train = [str(i) for i in load(os.path.join(dataset_path, 'kmnist-train-labels.npz'))]
    y_test = [str(i) for i in load(os.path.join(dataset_path, 'kmnist-test-labels.npz'))]

    return X_train, X_test, y_train, y_test


class Binarizer():
    """
    Implement binarization strategies.

    Implemented strategies:
        - basic_bin
        - simple_thermometer
        - circular_thermometer
        - sauvola
        - niblack
        - adaptive_thresh_mean
        - adaptive_thresh_gaussian
    """

    def __init__(
        self, strategy: str, threshold: int, resolution: int, window_size: int, constant_k: float,
            constant_c: float) -> None:
        self.strategy = strategy
        self.threshold = threshold
        self.resolution = resolution
        self.window_size = window_size
        self.constant_k = constant_k
        self.constant_c = constant_c

        assert self.strategy is not None, "Strategy must be specified."

    def transform(self, X) -> list:
        """
        Transform the data according to the specified strategy.

        Args:
            X (list): The data to be transformed.

        Raises:
            ValueError: If the strategy is not implemented.

        Returns:
            list: The transformed data.
        """
        if self.strategy == "basic_bin":
            return self.basic_bin(X, self.threshold)
        elif self.strategy == "simple_thermometer":
            return self.simple_thermometer(X, resolution=self.resolution)
        elif self.strategy == "circular_thermometer":
            return self.circular_thermometer(X, resolution=self.resolution)
        elif self.strategy == "sauvola":
            return self.sauvola(X, window_size=self.window_size)
        elif self.strategy == "niblack":
            return self.niblack(X, window_size=self.window_size, constant_k=self.constant_k)
        elif self.strategy == "adaptive_thresh_mean":
            return self.adaptive_thresh_mean(X, window_size=self.window_size, constant_c=self.constant_c)
        elif self.strategy == "adaptive_thresh_gaussian":
            return self.adaptive_thresh_gaussian(X, window_size=self.window_size, constant_c=self.constant_c)
        else:
            raise ValueError("Strategy not implemented.")

    def basic_bin(self, arr: np.ndarray, threshold: int = 128) -> list:
        return [list(np.where(x < threshold, 0, 1).flatten()) for x in arr]

    def simple_thermometer(self, arr: np.ndarray, minimum: int = 0, maximum: int = 255, resolution: int = 25) -> list:
        therm = ThermometerEncoder(maximum=maximum, minimum=minimum, resolution=resolution)
        return [np.uint8(therm.encode(x)).flatten() for x in arr]

    def circular_thermometer(self, arr: np.ndarray, minimum: int = 0, maximum: int = 255, resolution: int = 20) -> list:
        therm = CircularThermometerEncoder(maximum=maximum, minimum=minimum, resolution=resolution)
        return [np.uint8(therm.encode(x)).flatten() for x in arr]

    def sauvola(self, arr: np.ndarray, window_size: int = 11) -> list:
        bin_imgs = list()
        for x in arr:
            thresh_s = threshold_sauvola(x, window_size=window_size)
            binary_s = np.array(x > thresh_s, dtype=int)
            bin_imgs.append(binary_s.flatten())
        return bin_imgs

    def niblack(self, arr: np.ndarray, window_size: int = 11, constant_k: float = 0.8) -> list:
        bin_imgs = list()
        for x in arr:
            thresh_n = threshold_niblack(x, window_size=window_size, k=constant_k)
            binary_n = np.array(x > thresh_n, dtype=int)
            bin_imgs.append(binary_n.flatten())
        return bin_imgs

    def adaptive_thresh_mean(self, arr: np.ndarray, window_size: int = 11, constant_c: int = 2) -> list:
        return [cv.adaptiveThreshold(x, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, window_size, constant_c).flatten() for x in arr]

    def adaptive_thresh_gaussian(self, arr: np.ndarray, window_size: int = 11, constant_c: int = 2) -> list:
        return [cv.adaptiveThreshold(x, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, window_size, constant_c).flatten() for x in arr]

def plot_confusion_matrix(y_true, y_pred, name: str, experiment: Any, labels: List[str] = ['A', 'B', 'C', 'D']):
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15

    plt.rc('legend', **{'fontsize': 15})

    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = 100. * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Greys, clim=(0.0, 100.0))

    width, height = cm_normalized.shape

    for x in range(width):
        for y in range(height):
            if cm_normalized[x][y] < 50.:
                ax.annotate('%1.2f%%' % (cm_normalized[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')
            else:
                ax.annotate('%1.2f%%' % (cm_normalized[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center', color='white')

    ax.set_title(f'Confusion Matrix - {name}', fontweight='bold', fontsize=15)
    fig.colorbar(im)
    tick_marks = np.arange(len(labels))
    ax.xaxis.set_ticks(tick_marks)
    ax.xaxis.set_ticklabels(labels)

    ax.yaxis.set_ticks(tick_marks)
    ax.yaxis.set_ticklabels(labels)

    ax.set_ylabel('True Label', fontweight='bold', fontsize=15)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=15)

    experiment.log({f"confusion_matrix_{name}": plt})

def display_digits(images: dict, experiment: Any):
  fig, axs = plt.subplots(2, 5, squeeze=True)

  for i, img in enumerate(images.values()):
    img = np.reshape(img, (28, 28))
    img = img/np.max(img)
    fig.axes[i].imshow(img, cmap="gray", vmin=0., vmax=1.)
    fig.axes[i].xaxis.set_visible(False)
    fig.axes[i].yaxis.set_visible(False)

    experiment.log({f"mental_images": plt})