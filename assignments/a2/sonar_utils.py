"""Module that contains all Audio Signal Processing applied to Passive Sonar data."""

import logging
from typing import Dict, Tuple, Union

import cv2 as cv
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from poseidon.io.offline import SonarDict
from poseidon.signal.passivesonar import lofar
from scipy.signal import decimate
from sklearn import preprocessing
from sklearn.metrics import recall_score

from encoders import CircularThermometerEncoder, ThermometerEncoder
from utils import Binarizer


def get_balanced_dataset(X_data: np.ndarray, y_data: np.ndarray, strategy: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Auxiliary function that balances dataset based on the strategy passed as arguemnt.
    Args:
        X_data (np.ndarray): Model's input data
        y_data (np.ndarray): Model's target data
        strategy (Optional[str]): Balacing strategy to use. When not passed, the returns is equal to the input.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Balanced version of the input and its target based on the strategy passed.
    """
    X_data = pd.DataFrame(X_data.copy())
    y_data = pd.DataFrame(y_data.copy())

    if strategy == 'downsampling':
        x, y = NearMiss().fit_resample(X_data, y_data)
        x = pd.DataFrame(x).to_numpy()
        y = pd.DataFrame(y).to_numpy()
    elif strategy == 'oversampling':
        # https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
        x, y = SMOTE().fit_resample(X_data, y_data)
        x = pd.DataFrame(x).to_numpy()
        y = pd.DataFrame(y).to_numpy()
    else:
        x = X_data.copy().to_numpy()
        y = y_data.copy().to_numpy()
    return x, y

def scale_data(train_data: pd.DataFrame, test_data: pd.DataFrame, scaler: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Function that applies a scaler to the data. The scaler can be either ´[minmax, standard, robust]´.
    The scalers is first fitted to the training data and then applied to both the training and test data.
    Args:
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Test data.
        scaler (str): Scaler to be applied.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Scaled training and test data.
    """
    obj_std = None
    if scaler == 'mapstd':
        obj_std = preprocessing.StandardScaler().fit(train_data)
    elif scaler == 'mapstd_rob':
        obj_std = preprocessing.RobustScaler().fit(train_data)
    elif scaler == 'mapminmax':
        obj_std = preprocessing.MinMaxScaler().fit(train_data)
    else:
        return train_data, test_data

    train_data_scaled = obj_std.transform(train_data)
    test_data_scaled = obj_std.transform(test_data)
    
    return train_data_scaled, test_data_scaled

class SonarBinarizer(Binarizer):
    """
    Implement binarization strategies.

    Implemented strategies:
        - basic_bin
        - simple_thermometer
        - circular_thermometer
    """
    def __init__(
        self,
        strategy: str,
        threshold: int,
        resolution: int,
        window_size: int,
        minimum: int,
        maximum: int
    ):
        super().__init__(strategy, threshold, resolution, window_size, None, None)
        self.minimum = minimum
        self.maximum = maximum

    def basic_bin(self, arr: np.ndarray, threshold: int = None) -> list:
        _threshold = lambda x: threshold if threshold else np.mean(x)
        return [list(np.where(x < _threshold(x), 0, 1)) for x in arr]

    def simple_thermometer(self, arr: np.ndarray, minimum: int = 0, maximum: int = 8, resolution: int = 25) -> list:
        therm = ThermometerEncoder(maximum=self.maximum, minimum=self.minimum, resolution=resolution)
        return [np.uint8(therm.encode(x)).flatten() for x in arr]

    def circular_thermometer(self, arr: np.ndarray, minimum: int = 0, maximum: int = 8, resolution: int = 20) -> list:
        therm = CircularThermometerEncoder(maximum=self.maximum, minimum=self.minimum, resolution=resolution)
        return [np.uint8(therm.encode(x)).flatten() for x in arr]


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
        y_true = np.array(y_true).astype(int)
        y_pred = np.array(y_pred).astype(int)

    num_classes = len(np.unique(y_true))
    recall = recall_score(y_true, y_pred, average=None)
    sp = np.sqrt(np.sum(recall) / num_classes * np.power(np.prod(recall), 1.0 / float(num_classes)))
    return sp
