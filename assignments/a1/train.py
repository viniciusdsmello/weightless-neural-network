"""
In this module we define the model architecture and all the training steps.
"""
import time
import logging
import os
import numpy as np
import wandb
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils import load_data, Binarizer
from model_trainer import ModelTrainer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ModuleNotFoundError:
    pass

DATASET_PATH: str = os.getenv("DATASET_PATH")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
)


def train():
    """
    This function is called by the wandb agent and contains the training loop.
    """
    config_defaults = {
        "training_type": "kfold",
        "random_seed": 8080,
        "validation_split": 0.2,
        "folds": 10,
        "binarization_strategy": "basic_bin",
        "binarization_threshold": 128,
        "binarization_resolution": 20,
        "binarization_window_size": 3,
        "binarization_constant_c": 2,
        "binarization_constant_k": 0.2,
        # Number of RAM addressing bits
        "wsd_address_size": 20,
        # RAMs ignores the address 0
        "wsd_ignore_zero": False,
        "wsd_verbose": True,
        "wsd_bleaching_activated": True,
        # when M (number of bits) is not divisible by n_i
        "wsd_complete_addressing": True,
        # returns the degree of similarity of each y
        "wsd_return_activation_degree": False,
        # returns the confidence level of each y
        "wsd_return_confidence": False,
        # confidence of each y in relation to each class
        "wsd_return_classes_degrees": False
    }
    with wandb.init(
        entity="viniciusdsmello",
        project="wnn",
        config=config_defaults
    ) as run:
        config = wandb.config
        # Load data
        X, X_test, y, y_test = load_data(DATASET_PATH)

        logging.info("X shape: %s", X.shape)
        logging.info("y shape: %s", len(y))
        logging.info("X_test shape: %s", X_test.shape)
        logging.info("y_test shape: %s", len(y_test))

        # Binaryize data
        binarizer = Binarizer(
            strategy=config['binarization_strategy'],
            threshold=config['binarization_threshold'],
            resolution=config['binarization_resolution'],
            window_size=config['binarization_window_size'],
            constant_c=config['binarization_constant_c'],
            constant_k=config['binarization_constant_k']
        )
        X = binarizer.transform(X)
        X_test = binarizer.transform(X_test)

        if config['training_type'] == 'kfold':
            logging.info("Training with k-fold cross validation.")

            folds = config["folds"]
            kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config["random_seed"])

            scores = []
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
                test_acc: float = trainer.evaluate(X_val, y_val)

                durations.append(time.time() - start_time)
                scores.append(test_acc)

            wandb.log({"mean_trainnig_duration": np.mean(durations)})
            wandb.log({"str_trainnig_duration": np.std(durations)})

            wandb.log({"mean_val_acc": np.mean(scores)})
            wandb.log({"str_val_acc": np.std(scores)})
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
            test_acc: float = trainer.evaluate(X_val, y_val)

            logging.info(f"Train Accuracy: {train_acc}")
            logging.info(f"Test Accuracy: {test_acc}")

            wandb.log({"train_acc": train_acc})
            wandb.log({"val_acc": test_acc})

        # Train with whole training set
        logging.info("Training with whole training set")
        trainer = ModelTrainer(config)
        trainer.train(X, y)
        train_acc: float = trainer.evaluate(X, y)
        test_acc: float = trainer.evaluate(X_test, y_test)
        logging.info(f"Train Accuracy: {train_acc}")
        logging.info(f"Test Accuracy: {test_acc}")
        wandb.log({"train_acc": train_acc})
        wandb.log({"test_acc": test_acc})


if __name__ == '__main__':
    try:
        train()
    except:
        logging.exception("An error occured during training")
        raise
