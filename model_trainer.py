import logging
import wisardpkg as wp
from sklearn.metrics import accuracy_score


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self._is_model_trained: bool = False

        self.model = self._build_wisard_model()

    def _build_wisard_model(self):
        """
        Build a Wisard model based on the configuration.
        """
        try:
            logging.info("Building Wisard model")
            model = wp.Wisard(
                self.config["wsd_address_size"],
                ignoreZero=self.config["wsd_ignore_zero"],
                verbose=self.config["wsd_verbose"],
                bleachingActivated=self.config["wsd_bleaching_activated"],
                completeAddressing=self.config["wsd_complete_addressing"],
                returnActivationDegree=self.config["wsd_return_activation_degree"],
                returnConfidence=self.config["wsd_return_confidence"],
                returnClassesDegrees=self.config["wsd_return_classes_degrees"]
            )
        except ValueError:
            logging.error("Invalid configuration for Wisard model")
            raise
        except Exception as e:
            logging.exception("Fail on building wisard model")
            raise
        return model

    def train(self, X_train, y_train):
        """
        Train the model with the given data.
        """
        self.model.train(X_train, y_train)
        self._is_model_trained = True

    def predict(self, X: list) -> list:
        """
        Predict the labels for the given data

        Args:
            X (list): List of images

        Returns:
            list: List of predicted labels
        """
        return self.model.classify(X)

    def evaluate(self, X: list, y: list) -> float:
        """
        Evaluate the model on the given data and return the accuracy score

        Args:
            X (list): List of images
            y (list): List of labels

        Returns:
            float: Accuracy score
        """
        
        y_pred: list = self.predict(X)
        return accuracy_score(y, y_pred)
