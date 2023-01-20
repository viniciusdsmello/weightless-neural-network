"""
In this module we define the model architecture and all the training steps.
"""
import math
import gc
import logging
import os
import time

import numpy as np
from poseidon.io.offline import load_raw_data
from sklearn.model_selection import StratifiedKFold, train_test_split

import wandb
from assignments.a2.sonar_utils import (generate_train_test_dataset,
                                        preprocess_rawdata, sp_index, SonarBinarizer, scale_data, get_balanced_dataset)
from model_trainer import ModelTrainer
from utils import plot_confusion_matrix

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

DATASET_PATH: str = os.getenv("DATASET_PATH")
DATASET_CONFIG = {
    "dataset": "4classes",
    # PreProcessing Parameters
    "fs": 22050,
    "preprocessing_decimation_rate": 3,
    "preprocessing_lofar_nfft": 1024,
    "preprocessing_lofar_noverlap": False,
    "preprocessing_lofar_spectrum_bins_left": 400,
    "preprocessing_filter_type": "fir",
    "preprocessing_filter_phase": False,
    "runs_distribution": {
        'test': {
            'Class1': [2],
            'Class2': [2],
            'Class3': [2],
            'Class4': [2]
        }
    }
}

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
)


def train():
    """
    This function is called by the wandb agent and contains the training loop.
    """
    config_defaults = {
        "training_type": "validation_split",
        "random_seed": 8080,
        "validation_split": 0.2,
        "folds": 5,
        "data_normalization": "mapstd",
        "data_balancing": None,
        "binarization_strategy": "circular_thermometer",
        "binarization_threshold": None, # If None, the threshold is computed using the mean value of each sample
        "binarization_resolution": 20,
        "binarization_window_size": 10,
        # Number of RAM addressing bits
        "wsd_address_size": 20,
        # RAMs ignores the address 0
        "wsd_ignore_zero": False,
        "wsd_verbose": False,
        "wsd_bleaching_activated": True,
        # when M (number of bits) is not divisible by n_i
        "wsd_complete_addressing": True,
        # returns the degree of similarity of each y
        "wsd_return_activation_degree": False,
        # returns the confidence level of each y
        "wsd_return_confidence": False,
        # confidence of each y in relation to each class
        "wsd_return_classes_degrees": False,
    }
    config_defaults.update(DATASET_CONFIG)
    with wandb.init(
        entity="viniciusdsmello",
        project="wnn-sonar"
    ) as run:
        run.config.setdefaults(config_defaults)
        config = wandb.config
        try:
            # Load data
            raw_data_path = os.path.join(DATASET_PATH, config['dataset'])
            raw_data = load_raw_data(raw_data_path, verbose=1)
            preprocessed_data = preprocess_rawdata(raw_data, config)
            logging.info("Splitting data into train and test sets...")
            development_set, test_set = generate_train_test_dataset(
                sonar_data=preprocessed_data,
                trgt_label_map={'Class1': 0, 'Class2': 1, 'Class3': 2, 'Class4': 3},
                concatenate_runs=True,
                runs_distribution=config['runs_distribution']
            )
            X, y = development_set
            X_test, y_test = test_set

            logging.info("X shape: %s", X.shape)
            logging.info("y shape: %s", len(y))
            logging.info("X_test shape: %s", X_test.shape)
            logging.info("y_test shape: %s", len(y_test))

            # Balance data
            if config['data_balancing'] is not None:
                X, y = get_balanced_dataset(X, y, config['data_balancing'])

            # Scale data
            X, X_test = scale_data(X, X_test, config['data_normalization'])

            # We need to convert the target variables to str due to a limitation of the WNN implementation
            y = y.astype(int).astype(str)
            y_test = y_test.astype(int).astype(str)

            # Binaryize data
            try:
                binarizer = SonarBinarizer(
                    strategy=config['binarization_strategy'],
                    threshold=config['binarization_threshold'],
                    resolution=config['binarization_resolution'],
                    window_size=config['binarization_window_size'],
                    minimum=math.ceil(X.min()),
                    maximum=math.floor(X.max())
                )
                X = binarizer.transform(X)
                X_test = binarizer.transform(X_test)
        
                logging.info("X shape after binarization: %s", np.array(X).shape)
                logging.info("X_test shape after binarization: %s", np.array(X_test).shape)
            except Exception:
                logging.exception("Failed to binarize the data")
                raise

            if config['training_type'] == 'kfold':
                logging.info("Training with k-fold cross validation.")

                folds = config["folds"]
                kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config["random_seed"])

                train_acc_scores = []
                val_acc_scores = []
                train_spindex_scores = []
                val_spindex_scores = []

                durations = []
                evaluation_durations = []
                for fold, (index_X_train, index_X_val) in enumerate(kf.split(X, y)):
                    logging.info(f"Training Fold {fold + 1}/{folds}")

                    # Split data
                    logging.info("Splitting data.")
                    X_train = [X[i] for i in index_X_train]
                    X_val = [X[i] for i in index_X_val]

                    y_train = [y[i] for i in index_X_train]
                    y_val = [y[i] for i in index_X_val]

                    # Train model
                    trainer = ModelTrainer(config)

                    start_time = time.time()
                    trainer.train(X_train, y_train)
                    durations.append(time.time() - start_time)

                    start_time = time.time()
                    val_acc: float = trainer.evaluate(X_val, y_val)
                    train_acc: float = trainer.evaluate(X_train, y_train)
                    
                    train_sp_index: float = sp_index(y_train, trainer.predict(X_train), method=None)
                    val_sp_index: float = sp_index(y_val, trainer.predict(X_val), method=None)

                    evaluation_durations.append(time.time() - start_time)

                    train_acc_scores.append(train_acc)
                    val_acc_scores.append(val_acc)

                    train_spindex_scores.append(train_sp_index)
                    val_spindex_scores.append(val_sp_index)

                    del trainer.model
                    del trainer
                    del X_train
                    del X_val
                    del y_train
                    del y_val
                    gc.collect()

                run.log({"train_acc_scores": train_acc_scores})
                run.log({"val_acc_scores": val_acc_scores})
                run.log({"train_spindex_scores": train_spindex_scores})
                run.log({"val_spindex_scores": val_spindex_scores})

                run.log({"mean_evaluation_duration": np.mean(evaluation_durations)})
                run.log({"std_evaluation_duration": np.std(evaluation_durations)})

                run.log({"mean_training_duration": np.mean(durations)})
                run.log({"std_training_duration": np.std(durations)})

                run.log({"mean_train_acc": np.mean(train_acc_scores)})
                run.log({"std_train_acc": np.std(train_acc_scores)})
                run.log({"mean_val_acc": np.mean(val_acc_scores)})
                run.log({"std_val_acc": np.std(val_acc_scores)})

                run.log({"mean_train_sp_index": np.mean(train_spindex_scores)})
                run.log({"std_train_sp_index": np.std(train_spindex_scores)})

                run.log({"mean_val_sp_index": np.mean(val_spindex_scores)})
                run.log({"std_val_sp_index": np.std(val_spindex_scores)})

            elif config['training_type'] == 'validation_split':
                logging.info("Training with validation split")
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    y,
                    test_size=config['validation_split'],
                    random_state=config["random_seed"]
                )
                logging.info("Launching training")
                trainer = ModelTrainer(config)
                start_time = time.time()
                trainer.train(X_train, y_train)
                logging.info("Training finished")
                training_duration = time.time() - start_time
                logging.info(f"Training time: {training_duration}")
                run.log({"training_duration": training_duration})

                logging.info("Evaluating model")
                start_time = time.time()
                train_acc: float = trainer.evaluate(X_train, y_train)
                val_acc: float = trainer.evaluate(X_val, y_val)
                evaluation_duration = time.time() - start_time
                logging.info(f"Evaluation time: {evaluation_duration}")
                run.log({"evaluation_duration": evaluation_duration})

                train_sp_index: float = sp_index(y_train, trainer.predict(X_train), method=None)
                val_sp_index: float = sp_index(y_val, trainer.predict(X_val), method=None)

                logging.info(f"Train Accuracy: {train_acc}")
                logging.info(f"Validation Accuracy: {val_acc}")
                logging.info(f"Train SP Index: {train_sp_index}")
                logging.info(f"Validation SP Index: {val_sp_index}")

                run.log({"train_acc": train_acc})
                run.log({"val_acc": val_acc})
                run.log({"train_sp_index": train_sp_index})
                run.log({"val_sp_index": val_sp_index})

            # Train with whole training set
            logging.info("Training with whole training set")
            try:
                trainer = ModelTrainer(config)
                trainer.train(X, y)
            except Exception:
                logging.exception("Failed to train model")
                raise
            train_acc: float = trainer.evaluate(X, y)
            test_acc: float = trainer.evaluate(X_test, y_test)
            train_sp_index: float = sp_index(y, trainer.predict(X), method=None)
            test_sp_index: float = sp_index(y_test, trainer.predict(X_test), method=None)

            logging.info(f"Train Accuracy: {train_acc}")
            logging.info(f"Test Accuracy: {test_acc}")
            logging.info(f"Train SP Index: {train_sp_index}")
            logging.info(f"Test SP Index: {test_sp_index}")

            run.log({"train_acc": train_acc})
            run.log({"test_acc": test_acc})
            run.log({"train_sp_index": train_sp_index})
            run.log({"test_sp_index": test_sp_index})

            plot_confusion_matrix(
                y_true=y,
                y_pred=trainer.predict(X),
                name='Training set',
                experiment=wandb,
                labels=sorted(set(y))
            )

            plot_confusion_matrix(
                y_true=y_test,
                y_pred=trainer.predict(X_test),
                name='Test set',
                experiment=wandb,
                labels=sorted(set(y_test))
            )
        except Exception:
            logging.exception("Error during training")
            raise


if __name__ == '__main__':
    try:
        train()
    except:
        logging.exception("An error occured during training")
        raise
