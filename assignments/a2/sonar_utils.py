"""Module that contains all Audio Signal Processing applied to Passive Sonar data."""

import logging
from typing import Dict, Tuple, Union

import cv2 as cv
import numpy as np
from poseidon.io.offline import SonarDict
from poseidon.signal.passivesonar import lofar
from scipy.signal import decimate
from skimage.filters import threshold_niblack, threshold_sauvola
from sklearn.metrics import recall_score

from encoders import CircularThermometerEncoder, ThermometerEncoder
from utils import Binarizer


class SonarBinarizer(Binarizer):
    def __init__(
        self, strategy: str, threshold: int, resolution: int, window_size: int, constant_k: float,
            constant_c: float):
        super().__init__(strategy, threshold, resolution, window_size, constant_k, constant_c)

    def basic_bin(self, arr: np.ndarray, threshold: int = None) -> list:
        _threshold = lambda x: threshold if threshold else np.mean(x)
        return [list(np.where(x < _threshold(x), 0, 1)) for x in arr]

    def simple_thermometer(self, arr: np.ndarray, minimum: int = 0, maximum: int = 255, resolution: int = 25) -> list:
        therm = ThermometerEncoder(maximum=maximum, minimum=minimum, resolution=resolution)
        return [np.uint8(therm.encode(x)) for x in arr]

    def circular_thermometer(self, arr: np.ndarray, minimum: int = 0, maximum: int = 255, resolution: int = 20) -> list:
        therm = CircularThermometerEncoder(maximum=maximum, minimum=minimum, resolution=resolution)
        return [np.uint8(therm.encode(x)) for x in arr]

    def sauvola(self, arr: np.ndarray, window_size: int = 11) -> list:
        bin_imgs = list()
        for x in arr:
            thresh_s = threshold_sauvola(x, window_size=window_size)
            binary_s = np.array(x > thresh_s, dtype=int)
            bin_imgs.append(binary_s)
        return bin_imgs

    def niblack(self, arr: np.ndarray, window_size: int = 11, constant_k: float = 0.8) -> list:
        bin_imgs = list()
        for x in arr:
            thresh_n = threshold_niblack(x, window_size=window_size, k=constant_k)
            binary_n = np.array(x > thresh_n, dtype=int)
            bin_imgs.append(binary_n)
        return bin_imgs

    def adaptive_thresh_mean(self, arr: np.ndarray, window_size: int = 11, constant_c: int = 2) -> list:
        return [cv.adaptiveThreshold(x, 1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, window_size, constant_c) for x in arr]

    def adaptive_thresh_gaussian(self, arr: np.ndarray, window_size: int = 11, constant_c: int = 2) -> list:
        return [cv.adaptiveThreshold(x, 1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, window_size, constant_c) for x in arr]


def generate_train_test_dataset(
    sonar_data: SonarDict,
    trgt_label_map: Dict[str, int],
    runs_distribution: Dict[str, Dict[str, int]],
    concatenate_runs: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Auxiliary function that generates a data-target pair from the sonar runs.

    params:
        sonar_data (SonarDict): nested dicionary in which the basic unit contains a record of the audio (signal key)
            in np.array format and the sample_rate (fs key) stored in floating point.
        trgt_label_map (dict): dictionary with target labels.
        runs_distribution (Dict[str, Dict[str, int]]): Dictionary that indicates which classes runs will make part
            of the test set.
        concatenate_runs_data (Optional[bool]): When true, the return is a concatenation of all runs data. Defaults to False.
    returns:
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: A tuple of data/target numpy arrays
    """
    if trgt_label_map is None:
        trgt_label_map = {
            'ClassA': 0,
            'ClassB': 1,
            'ClassC': 2,
            'ClassD': 3
        }

    _development_data = []
    _development_target = []

    _test_data = []
    _test_target = []

    for class_name, run in sonar_data.items():
        for i, (run_name, Sxx) in enumerate(run.items()):
            run = Sxx[0]
            class_target = trgt_label_map[class_name]
            data, target = run, np.ones(run.shape[0]) * class_target

            if i in runs_distribution['test'][class_name]:
                logging.debug("Attributing run %s to test set", run_name)
                _test_data.append(data)
                _test_target.append(target)
            else:
                logging.debug("Attributing run %s to development set", run_name)
                _development_data.append(data)
                _development_target.append(target)

    if concatenate_runs:
        _test_target = np.concatenate(_test_target)
        _test_data = np.concatenate(_test_data)
        _development_target = np.concatenate(_development_target)
        _development_data = np.concatenate(_development_data)

    return (_development_data, _development_target), (_test_data, _test_target)


def preprocess_rawdata(raw_data: SonarDict, dataset_config: dict):
    """
    Preprocessing function that apply the specified pipeline to all classes runs.

    params:
        raw_data (SonarDict):
            nested dicionary in which the basic unit contains
            a record of the audio (signal key) in np.array format and the
            sample_rate (fs key) stored in floating point.
        signal_proc_params (dict):
            dictionary with params to the signal processing step.
    """
    return (
        raw_data
        .apply(lambda rr: rr['signal'])
        .apply(decimate,
               dataset_config['preprocessing_decimation_rate'], 8, 'fir', -1, True
               )
        .apply(lofar,
               dataset_config['fs'],
               dataset_config['preprocessing_lofar_nfft'],
               dataset_config['preprocessing_lofar_noverlap'],
               dataset_config['preprocessing_lofar_spectrum_bins_left']
               )
    )


def sp_index(
        y_true: np.ndarray, y_pred: np.ndarray, method: str = 'argmax', threshold: float = None) -> Union[
        float, np.ndarray]:
    """
    Function that computes SP Index

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted target value
        method (str): Method to compute the SP Index. Available methods are: 'argmax', 'threshold'. Default is 'argmax'
        threshold (float): Threshold to compute the SP Index. Default is None
    Returns:
        (Union[float, np.ndarray]): Returns the computed sp index score
    """
    if method == 'argmax':
        y_true = y_true.argmax(1)
        y_pred = y_pred.argmax(1)
    elif method == 'threshold':
        raise NotImplementedError
    else:
        raise ValueError("Method not recognized")

    num_classes = len(np.unique(y_true))
    recall = recall_score(y_true, y_pred, average=None)
    sp = np.sqrt(np.sum(recall) / num_classes * np.power(np.prod(recall), 1.0 / float(num_classes)))
    return sp
