import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the TorchMLPClassifier.

        input_size: Number of input features.
        hidden_size: Number of hidden units (neurons) in the hidden layer.
        output_size: Number of output classes (for classification).
        """
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
        """
        Backpropagation to compute gradients and update weights.
        X: Input data (m x input_size)
        y: True labels (m x 1)
        output: Output from the forward pass (m x output_size)
        lr: Learning rate

        Conventions:
        - m: number of samples
        - input_size: number of input features
        - hidden_size: number of neurons in the hidden layer
        - output_size: number of classes (for classification)
        - lr: learning rate for gradient descent
        - one_hot_y: one-hot encoded true labels. e.g.
        if y = [0, 2], then one_hot_y = [[1, 0, 0], [0, 0, 1]]
        - d: denotes the derivative of the loss with respect to the
        corresponding variable.
        - gradient: All those derivatives (dW1, db1, dW2, db2) together
        form the gradient.

        Notes:
        - Derivate: how the function changes with respect to one variable.
        - Gradient: collection of all those partial derivatives for a function
        with multiple variables.
        """
        # Count of samples
        m = X.shape[0]

        # Apply softmax to logits to get probabilities
        probs = torch.softmax(output, dim=1)

        # For multiclass cross-entropy, dz2 = probs - one_hot(y)
        # We'll compute this directly using y as class indices
        # One-hot encode y is for creating the correct shape for subtraction,
        # For instance, if y = [0, 2], then one_hot_y = [[1, 0, 0], [0, 0, 1]]
        # So first you need a matrix of zeros with shape (m, output_size) and
        # then scatter 1s at the indices specified by y
        one_hot_y = torch.zeros_like(probs)
        one_hot_y.scatter_(1, y.unsqueeze(1), 1)

        # Calculate the output layer error
        # Why? dz2 tell us: How does the loss change if I change each output
        # neuron's pre-activation (z2)?"
        # SHAPE: (m, output_size)
        # Calculate the gradient of the loss with respect to the output layer's
        # pre-activation values (z2).
        # Recall the forward pass:
        #     z2    →    softmax    →    probs
        #                         →    predicted probabilities
        # For multiclass classification, we use:
        #     - softmax to convert the logits (z2) into probabilities
        #     - cross-entropy to measure how different those probabilities are
        #       from the true labels
        # The derivative of the softmax + cross-entropy combination simplifies
        # to:
        #
        #     dz2 = ∂L/∂z2
        #         = probs - one_hot_y
        #
        # In other words, we are comparing what the model predicted with what
        # the correct answer should have been.
        # Example:
        #
        #     probs      = [0.7, 0.2, 0.1]   # model prediction
        #     one_hot_y  = [0.0, 1.0, 0.0]   # true class = 1
        #
        #     dz2        = [0.7, -0.8, 0.1]
        #
        # This tells us how the loss changes with respect to each output
        # neuron's
        # pre-activation:
        #     +0.7 → the model gave too much probability to class 0
        #     -0.8 → the model needs to increase the output for the correct
    #               class
        #     +0.1 → the model gave some probability to an incorrect class
        # Therefore, dz2 is the error signal/gradient that starts the backward
        # propagation from the output layer. We will use it to calculate how
        # the weights and biases in the output layer should be changed.
        #
        # SHAPE:
        #     probs       → (m, output_size)
        #     one_hot_y  → (m, output_size)
        #     dz2         → (m, output_size)
        dz2 = probs - one_hot_y

        # Gradient of the loss with respect to W2
        # We want to calculate:
        #     dW2 = ∂L/∂W2
        # The output layer is:
        #     z2 = a1 @ W2 + b2
        # Using the chain rule:
        #     ∂L/∂W2 = (∂L/∂z2) * (∂z2/∂W2)
        # We already calculated:
        #     dz2 = ∂L/∂z2
        # Since:
        #     z2 = a1 @ W2 + b2
        # the derivative of z2 with respect to W2 depends on a1.
        # For all samples at once, this becomes:
        #     dW2 = a1.T @ dz2
        # Since we have m training examples, we average their gradients:
        #     dW2 = (a1.T @ dz2) / m
        # Therefore:
        #     dW2 = ∂L/∂W2
        # Interpretation:
        #     dW2 tells us how the loss changes when each weight in W2 changes.
        #     The sign tells us the direction in which the weight should move,
        #     and the magnitude tells us how strongly that weight affects the
        # loss.
        dW2 = (self.a1.T @ dz2) / m
        # Gradient of the loss with respect to the output biases (b2).
        # Since each bias is added directly to its corresponding output neuron:
        #     z2 = a1 @ W2 + b2
        # The derivative of z2 with respect to b2 is 1, so by the chain rule:
        #     db2 = ∂L/∂b2 = ∂L/∂z2 = dz2
        # For a batch, we sum the gradients from all samples because the same
        # bias is shared across every sample, then divide by m to get the
        # average gradient:
        #     db2 = (1/m) Σ dz2
        # In PyTorch, dim=0 sums across the samples, leaving one gradient
        # value for each output bias.
        db2 = dz2.sum(dim=0) / m

        # Gradient of the loss with respect to the hidden layer's activations
        # (a1).
        # We want to calculate:
        #     da1 = ∂L/∂a1
        # The output layer is:
        #     z2 = a1 @ W2 + b2
        # Using the chain rule:
        #     ∂L/∂a1 = (∂L/∂z2) * (∂z2/∂a1)
        # We already calculated:
        #     dz2 = ∂L/∂z2    SHAPE: (m, output_size)
        # Since:
        #     z2 = a1 @ W2 + b2
        # the derivative of z2 with respect to a1 depends on W2.
        # For all samples at once, this becomes:
        #     da1 = dz2 @ W2.T
        # In other words, we are "routing" the output error (dz2) backwards
        # through W2 to find out how much each hidden neuron contributed to
        # that error.
        # SHAPE:
        #     dz2     → (m, output_size)
        #     W2.T    → (output_size, hidden_size)
        #     da1     → (m, hidden_size)
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