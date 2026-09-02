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
        # ------------------------------------------------------------------
        # Backprop story (we walk the network backwards).
        #
        # Forward was:
        #     X → z1 → a1 (ReLU) → z2 → softmax → probs → loss
        #
        # Backward starts at the loss and asks, layer by layer:
        # "if I change this value, how does the loss change?"
        # Those answers (the gradients) tell us how to update W and b.
        #
        # One running example for every step below (m = 1):
        #     2 hidden neurons, 3 classes, true class = 1
        #     z2  = [2.0, 1.0, 0.0]
        #     a1  = [1.0, 0.5]
        #     W2  = [[0.5, 1.0, 0.0],     # hidden 0 → classes 0, 1, 2
        #            [0.2, 0.3, 0.4]]     # hidden 1 → classes 0, 1, 2
        #     y   = 1
        # ------------------------------------------------------------------

        # How many samples are in this batch. We will average gradients
        # over m so one example cannot dominate the update.
        # In the running example: 1 row in X → m = 1, so / m does nothing.
        m = X.shape[0]

        # Step 1 — turn logits into probabilities.
        # The forward pass ended at z2 (raw scores). Softmax turns those
        # scores into probs that sum to 1, so we can compare them to labels.
        #     z2    = [2.0, 1.0, 0.0]     # class 0 scored highest
        #     probs ≈ [0.7, 0.2, 0.1]     # rounded; they now sum to 1
        # Class 0 still looks most likely, but now we have a distribution.
        probs = torch.softmax(output, dim=1)

        # Step 2 — put the true labels in the same shape as probs.
        # y is a class index. Subtraction needs a vector of 0s and 1s
        # with a 1 only on the correct class.
        #     y         = 1
        #     one_hot_y = [0.0, 1.0, 0.0]
        one_hot_y = torch.zeros_like(probs)
        one_hot_y.scatter_(1, y.unsqueeze(1), 1)

        # Step 3 — start of backprop: the output error dz2 = ∂L/∂z2.
        # For softmax + cross-entropy this simplifies to:
        #     dz2 = probs - one_hot_y
        # We are comparing "what the model predicted" with "what was true".
        #
        #     probs      = [0.7, 0.2, 0.1]
        #     one_hot_y  = [0.0, 1.0, 0.0]
        #     dz2        = [0.7, -0.8, 0.1]
        #       +0.7 → too much probability on class 0 (push this down)
        #       -0.8 → too little on the correct class (push this up)
        #       +0.1 → some probability on class 2 (push this down)
        #
        # Everything below is just "who should take blame for this error?"
        # First we blame the output layer (W2, b2). Then we send the same
        # error one layer back so we can blame the hidden layer (W1, b1).
        #
        # SHAPE: probs, one_hot_y, dz2 → (m, output_size)
        dz2 = probs - one_hot_y

        # Step 4 — how should W2 change?  dW2 = ∂L/∂W2
        # Output layer:  z2 = a1 @ W2 + b2
        # Chain rule:    ∂L/∂W2 = (∂L/∂z2) * (∂z2/∂W2)
        # We already have ∂L/∂z2 = dz2.  z2 depends on W2 through a1,
        # so for a whole batch:
        #     dW2 = (a1.T @ dz2) / m
        # (divide by m = average over the batch; here m = 1)
        #
        #     a1  = [1.0, 0.5]
        #     dz2 = [0.7, -0.8, 0.1]
        # Each weight update is (hidden activation) * (output error):
        #     dW2 = [[1.0*0.7,  1.0*(-0.8),  1.0*0.1],
        #            [0.5*0.7,  0.5*(-0.8),  0.5*0.1]]
        #         = [[0.70, -0.80, 0.10],
        #            [0.35, -0.40, 0.05]]
        # Hidden neuron 0 fired more (1.0 vs 0.5), so its row is larger:
        # it takes more blame. Column 1 is negative → those weights
        # should increase (we needed more probability on the true class).
        #
        # Sign = which way to move the weight. Magnitude = how strongly
        # that weight affected the loss.
        dW2 = (self.a1.T @ dz2) / m

        # Step 5 — how should b2 change?  db2 = ∂L/∂b2
        # Bias is added directly: z2 = a1 @ W2 + b2, so ∂z2/∂b2 = 1.
        # Chain rule then says db2 is just dz2. The same bias is shared
        # by every sample, so we sum across the batch and average:
        #     db2 = (1/m) Σ dz2
        # dim=0 sums over samples, leaving one value per output neuron.
        # Here m = 1, so there is nothing to average:
        #     dz2 = [0.7, -0.8, 0.1]
        #     db2 = [0.7, -0.8, 0.1]
        db2 = dz2.sum(dim=0) / m

        # Step 6 — send the output error back to the hidden activations.
        # We now have dW2 and db2, so we know how to update the output
        # layer. W1 cannot be updated yet: it never touches z2 directly.
        # It only affects the loss through a1:
        #     a1 → W2 → z2 → loss
        # So we need da1 = ∂L/∂a1 first: "how much did each hidden
        # neuron contribute to the output error?"
        #
        # Same equation as before, other direction:
        #     z2 = a1 @ W2 + b2
        # Chain rule:  ∂L/∂a1 = (∂L/∂z2) * (∂z2/∂a1)
        # Going backward through W2 uses the transpose (same wires,
        # reversed):
        #     da1 = dz2 @ W2.T
        #
        #     dz2 = [0.7, -0.8, 0.1]
        #     W2  = [[0.5, 1.0, 0.0],
        #            [0.2, 0.3, 0.4]]
        # Hidden 0's da1 is dz2 dotted with its outgoing weights:
        #     0.7*0.5 + (-0.8)*1.0 + 0.1*0.0 = -0.45
        # Hidden 1:
        #     0.7*0.2 + (-0.8)*0.3 + 0.1*0.4 = -0.06
        #     da1 = [-0.45, -0.06]
        # Hidden 0 was wired more strongly into the class-1 error
        # (weight 1.0 vs 0.3), so it receives more of the blame.
        #
        # Next we will push da1 through ReLU to get dz1, then the same
        # pattern as steps 4–5 will give dW1 and db1.
        #
        # SHAPE:
        #     dz2  → (m, output_size)
        #     W2.T → (output_size, hidden_size)
        #     da1  → (m, hidden_size)
        da1 = dz2 @ self.W2.T

        # Step 7 — push da1 through ReLU to get dz1 = ∂L/∂z1.
        # Forward: a1 = ReLU(z1). ReLU is a gate:
        #     z1 > 0  →  a1 = z1  and  ∂a1/∂z1 = 1  (error may pass)
        #     z1 ≤ 0  →  a1 = 0   and  ∂a1/∂z1 = 0  (error is blocked)
        # Chain rule:
        #     dz1 = da1 * relu'(z1)
        # We already have da1 from step 6. Multiply element-wise by the
        # gate so only hidden neurons that actually fired can update W1.
        #
        # Our a1 was [1.0, 0.5], so both neurons were on:
        #     z1         = [1.0,  0.5]     # both > 0
        #     relu_deriv = [1.0,  1.0]
        #     da1        = [-0.45, -0.06]
        #     dz1        = [-0.45*1, -0.06*1] = [-0.45, -0.06]
        # Both errors pass. If a neuron had z1 ≤ 0, its relu_deriv would
        # be 0 and that slot of dz1 would be 0 — no blame for W1/b1 on
        # this sample.
        #
        # Next: same pattern as steps 4–5, but with X and dz1 → dW1, db1.
        #
        # SHAPE: z1, relu_deriv, da1, dz1 → (m, hidden_size)
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