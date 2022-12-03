"""
In this module we define the model architecture and all the training steps.
"""
import gc
import logging
import os
import time

import numpy as np
from poseidon.io.offline import load_raw_data
from sklearn.model_selection import StratifiedKFold, train_test_split

import wandb
from assignments.a2.sonar_utils import (generate_train_test_dataset,
                                        preprocess_rawdata, sp_index, SonarBinarizer)
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
    "preprocessing_lofar_spectrum_bins_left": 200,
    "preprocessing_filter_type": "fir",
    "preprocessing_filter_phase": False,
    "runs_distribution": {
        'test': {
            'Class1': [1],
            'Class2': [1],
            'Class3': [1],
            'Class4': [1]
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
        "binarization_strategy": "basic_bin",
        "binarization_threshold": None, # If None, the threshold is computed using the mean value of each sample
        "binarization_resolution": 20,
        "binarization_window_size": 3,
        "binarization_constant_c": 2,
        "binarization_constant_k": 0.2,
        # Number of RAM addressing bits
        "wsd_address_size": 5,
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

        y = y.astype(int)
        y_test = y_test.astype(int)

        logging.info("X shape: %s", X.shape)
        logging.info("y shape: %s", len(y))
        logging.info("X_test shape: %s", X_test.shape)
        logging.info("y_test shape: %s", len(y_test))


        # Binaryize data
        try:
            binarizer = SonarBinarizer(
                strategy=config['binarization_strategy'],
                threshold=config['binarization_threshold'],
                resolution=config['binarization_resolution'],
                window_size=config['binarization_window_size'],
                constant_c=config['binarization_constant_c'],
                constant_k=config['binarization_constant_k']
            )
            X = binarizer.transform(X)
            X_test = binarizer.transform(X_test)
        except Exception:
            logging.exception("Failed to binarize the data")
            raise

        if config['training_type'] == 'kfold':
            logging.info("Training with k-fold cross validation.")

            folds = config["folds"]
            kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config["random_seed"])

            scores = []
            scores_train = []
            durations = []
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

                test_acc: float = trainer.evaluate(X_val, y_val)
                train_acc: float = trainer.evaluate(X_train, y_train)
                scores.append(test_acc)
                scores_train.append(train_acc)

                del trainer.model
                del trainer
                del X_train
                del X_val
                del y_train
                del y_val
                gc.collect()

            wandb.log({"mean_trainnig_duration": np.mean(durations)})
            wandb.log({"std_trainnig_duration": np.std(durations)})

            wandb.log({"mean_train_acc": np.mean(scores_train)})
            wandb.log({"std_train_acc": np.std(scores_train)})

            wandb.log({"mean_val_acc": np.mean(scores)})
            wandb.log({"std_val_acc": np.std(scores)})
        elif config['training_type'] == 'validation_split':
            logging.info("Training with validation split")
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=config['validation_split'],
                random_state=config["random_seed"]
            )
            logging.info("Launching training")
            start_time = time.time()
            trainer = ModelTrainer(config)
            trainer.train(X_train, y_train)
            logging.info("Training finished")
            training_duration = time.time() - start_time
            logging.info(f"Training time: {training_duration}")
            wandb.log({"training_duration": training_duration})
            logging.info("Evaluating model")
            train_acc: float = trainer.evaluate(X_train, y_train)
            val_acc: float = trainer.evaluate(X_val, y_val)
            train_sp_index: float = sp_index(y_train, trainer.predict(X_train))
            val_sp_index: float = sp_index(y_val, trainer.predict(X_val))

            logging.info(f"Train Accuracy: {train_acc}")
            logging.info(f"Validation Accuracy: {test_acc}")
            logging.info(f"Train SP Index: {train_sp_index}")
            logging.info(f"Validation SP Index: {val_sp_index}")

            wandb.log({"train_acc": train_acc})
            wandb.log({"val_acc": val_acc})
            wandb.log({"train_sp_index": train_sp_index})
            wandb.log({"val_sp_index": val_sp_index})

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
        train_sp_index: float = sp_index(y_train, trainer.predict(X_train))
        test_sp_index: float = sp_index(y_test, trainer.predict(X_test))
        logging.info(f"Train Accuracy: {train_acc}")
        logging.info(f"Test Accuracy: {test_acc}")
        wandb.log({"train_acc": train_acc})
        wandb.log({"test_acc": test_acc})
        logging.info(f"Train SP Index: {train_sp_index}")
        logging.info(f"Test SP Index: {test_sp_index}")
        wandb.log({"train_sp_index": train_sp_index})
        wandb.log({"test_sp_index": test_sp_index})

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
        try:
            display_digits(trainer.model.getMentalImages(), wandb)
        except:
            logging.exception("Error displaying mental images")


if __name__ == '__main__':
    try:
        train()
    except:
        logging.exception("An error occured during training")
        raise
