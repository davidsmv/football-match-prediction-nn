import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from loguru import logger


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_size, output_size):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._build_model()

    def _build_model(self):
        self.W1 = torch.randn(
            self.input_size, self.hidden_size, requires_grad=True
        )
        self.b1 = torch.randn(
            self.hidden_size, requires_grad=True
        )
        self.W2 = torch.randn(
            self.hidden_size, self.output_size, requires_grad=True
        )
        self.b2 = torch.randn(
            self.output_size, requires_grad=True
        )

    def fit(self, X, y):
        X_t = (
            torch.tensor(X, dtype=torch.float32)
            .to(self.device)
        )
        y_t = (
            torch.tensor(y, dtype=torch.float32)
            .view(-1, 1)
            .to(self.device)
        )

    def predict(self, X):
        # Implement the prediction logic for the MLP classifier here
        pass

    def predict_proba(self, X):
        # Implement the probability prediction logic for the MLP classifier here
        pass

# TODO check on internet simple MLP classifier implementation in PyTorch and sklearn integration.
# https://medium.com/@mn05052002/building-a-simple-mlp-from-scratch-using-pytorch-7d50ca66512b

# TODO search how they explained in the course or how the code in the code worked