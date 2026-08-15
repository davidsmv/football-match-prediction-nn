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

    def _build_model(self) -> None:
        # Layer 1
        self.W1 = torch.randn(
            self.input_size,
            self.hidden_size,
            requires_grad=True,
            device=self.device
        )
        self.b1 = torch.randn(
            self.hidden_size,
            requires_grad=True,
            device=self.device
        )
        # Layer 2 (output layer)
        self.W2 = torch.randn(
            self.hidden_size,
            self.output_size,
            requires_grad=True,
            device=self.device
        )
        self.b2 = torch.randn(
            self.output_size,
            requires_grad=True,
            device=self.device
        )

    def forward(self, X):
        # Input -> hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = torch.relu(self.z1)

        # Hidden layer -> output layer
        self.z2 = self.a1 @ self.W2 + self.b2

        # Return raw logits
        return self.z2

    def backward(self, X, y, output, lr=0.01):
        m = X.shape[0]

        # Apply softmax to logits to get probabilities
        probs = torch.softmax(output, dim=1)

        # For multiclass cross-entropy, dz2 = probs - one_hot(y)
        # We'll compute this directly using y as class indices
        one_hot_y = torch.zeros_like(probs)
        one_hot_y.scatter_(1, y.unsqueeze(1), 1)

        dz2 = probs - one_hot_y

        # Gradients for W2 and b2
        dW2 = self.a1.T @ dz2 / m
        db2 = dz2.sum(axis=0) / m

        # Backpropagate to hidden layer
        da1 = dz2 @ self.W2.T

        # ReLU derivative: 1 where z1 > 0, else 0
        relu_deriv = (self.z1 > 0).float()
        dz1 = da1 * relu_deriv

        # Gradients for W1 and b1
        dW1 = X.T @ dz1 / m
        db1 = dz1.sum(axis=0) / m

        # Update parameters
        with torch.no_grad():
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

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