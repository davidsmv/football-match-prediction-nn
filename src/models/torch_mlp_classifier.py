import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from loguru import logger


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def _build_model(self):
        nn.Sequential()

    def fit(self, X, y):
        # Implement the training logic for the MLP classifier here
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device_}")

    def predict(self, X):
        # Implement the prediction logic for the MLP classifier here
        pass

    def predict_proba(self, X):
        # Implement the probability prediction logic for the MLP classifier here
        pass
