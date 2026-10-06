import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from sklearn.utils.validation import check_array, check_X_y


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_size=32,
        epochs=1000,
        lr=0.01,
        class_weight=None,
        random_state=None,
        validation_fraction=0.0,
        early_stopping_patience=None,
    ):
        """
        Initialize the TorchMLPClassifier.

        hidden_size: Number of hidden units (neurons) in the hidden layer.
        epochs: Number of training epochs.
        lr: Learning rate for gradient descent.
        class_weight: None or "balanced" (inverse class frequency weights
        in the loss, so rare classes like draws count more).
        random_state: Seed for the weight initialisation.
        validation_fraction: Share of the last (most recent) training rows
        held out to track validation loss. 0 disables it.
        early_stopping_patience: Stop after this many epochs without a new
        best validation loss and restore the best weights. None disables it.

        input_size/output_size are not accepted here: they are inferred from
        X and y in fit() so this classifier can be used as a Pipeline step
        without knowing the transformed feature count in advance.
        """
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.class_weight = class_weight
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience

    def _build_model(self, input_size, output_size) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.input_size = input_size
        self.output_size = output_size
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

    def backward(self, X, y, output, lr=0.01, sample_weight=None):
        """
        Backpropagation to compute gradients and update weights.
        X: Input data (m x input_size)
        y: True labels (m x 1)
        output: Output from the forward pass (m x output_size)
        lr: Learning rate
        sample_weight: Optional per-sample loss weights (m,); None = all 1.

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
        # With sample weights, m is the sum of the weights (weighted mean).
        if sample_weight is None:
            sample_weight = torch.ones(X.shape[0], device=X.device)
        m = sample_weight.sum()

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
        dz2 = (probs - one_hot_y) * sample_weight.unsqueeze(1)

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
        # Step 6 blamed the hidden *activations* a1. But W1 and b1 do not
        # connect to a1 directly — they connect to z1:
        #     z1 → ReLU → a1 → W2 → z2 → loss
        # So we must go one step further back: "how does the loss change
        # if we change z1 (the pre-activation)?"
        #
        # Forward, ReLU is a switch:
        #     z1 > 0  →  a1 = z1,   relu'(z1) = 1   (neuron is ON)
        #     z1 ≤ 0  →  a1 = 0,    relu'(z1) = 0   (neuron is OFF)
        # If a neuron was OFF, its output was forced to 0. It could not
        # influence z2, so no error should flow back to its weights.
        #
        # Chain rule through the activation:
        #     dz1 = da1 * relu'(z1)
        # Element-wise multiply: each hidden neuron keeps its da1 only if
        # it was ON during the forward pass.
        #
        # Running example (both neurons were ON):
        #     z1         = [1.0,  0.5]      # from a1 = ReLU(z1)
        #     relu_deriv = [1.0,  1.0]      # 1 where z1 > 0, else 0
        #     da1        = [-0.45, -0.06]   # from step 6
        #     dz1        = [-0.45, -0.06]   # both gates open → unchanged
        #
        # If hidden 1 had been OFF instead (z1 = -0.2 → a1 = 0):
        #     relu_deriv = [1.0,  0.0]
        #     dz1        = [-0.45, 0.0]     # second neuron's error blocked
        # That neuron would not update W1 or b1 on this sample.
        #
        # Next: same pattern as steps 4–5, but with X and dz1 → dW1, db1.
        #
        # SHAPE: z1, relu_deriv, da1, dz1 → (m, hidden_size)
        relu_deriv = (self.z1 > 0).float()
        dz1 = da1 * relu_deriv

        # Step 8 — how should W1 change?  dW1 = ∂L/∂W1
        # Exact same reasoning as step 4, but one layer earlier.
        # Input layer:  z1 = X @ W1 + b1
        # Chain rule:   ∂L/∂W1 = (∂L/∂z1) * (∂z1/∂W1)
        # We already have ∂L/∂z1 = dz1 (from step 7). z1 depends on W1
        # through X, so for a whole batch:
        #     dW1 = (X.T @ dz1) / m
        # (divide by m = average over the batch)
        #
        # Running example:
        #     X   = [[x0, x1, ...]]     # row vector: one sample, input_size features
        #     dz1 = [[-0.45, -0.06]]    # row vector: one sample, hidden_size neurons
        # Each weight update is (input feature) * (hidden error):
        #     dW1[feature_i, neuron_j] = X[feature_i] * dz1[neuron_j]
        # Feature i should take as much blame as it contributed to the
        # error in neuron j.
        #
        # SHAPE:
        #     X.T  → (input_size, m)
        #     dz1  → (m, hidden_size)
        #     dW1  → (input_size, hidden_size)   matches W1
        dW1 = X.T @ dz1 / m

        # Step 9 — how should b1 change?  db1 = ∂L/∂b1
        # Same reasoning as step 5. The bias is added directly:
        #     z1 = X @ W1 + b1  →  ∂z1/∂b1 = 1
        # Chain rule then says db1 is just dz1, averaged over the batch:
        #     db1 = (1/m) Σ dz1
        # dim=0 sums over samples, leaving one value per hidden neuron.
        #
        # Running example (m = 1, nothing to average):
        #     dz1 = [[-0.45, -0.06]]
        #     db1 = [-0.45, -0.06]
        #
        # SHAPE: db1 → (hidden_size,)   matches b1
        db1 = dz1.sum(axis=0) / m

        # Step 10 — gradient descent: move each weight against its gradient.
        # Every gradient says "if I increase this weight, the loss increases
        # by this much." Subtracting lr * gradient nudges the weight in the
        # direction that reduces the loss.
        #     new_weight = old_weight - lr * gradient
        #
        # torch.no_grad() disables PyTorch's autograd engine during the
        # update so that these in-place operations are not recorded as
        # part of the computation graph (we computed gradients manually).
        #
        # lr (learning rate) scales how big each step is:
        #   too large  → overshoots the minimum, loss may diverge
        #   too small  → converges, but very slowly
        #
        # All four parameter tensors are updated in the same way:
        #     W1, b1 — hidden layer weights and biases
        #     W2, b2 — output layer weights and biases
        with torch.no_grad():
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            self.W2 -= lr * dW2
            self.b2 -= lr * db2

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # classes_ + searchsorted map arbitrary labels (e.g. -1/0/1) to the
        # 0..C-1 indices CrossEntropyLoss/argmax require.
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)

        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'.")
        n_val = int(len(X) * self.validation_fraction)
        if self.early_stopping_patience is not None and n_val == 0:
            raise ValueError(
                "early_stopping_patience requires validation_fraction > 0."
            )

        self._build_model(
            input_size=X.shape[1], output_size=len(self.classes_)
        )
        epochs, lr = self.epochs, self.lr

        # Rows are assumed time-ordered, so the tail is the most recent data.
        if n_val:
            X_fit, y_fit = X[:-n_val], y_idx[:-n_val]
        else:
            X_fit, y_fit = X, y_idx

        # Ensure tensors are on the correct device and dtype
        X_tensor = torch.tensor(X_fit, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_fit, dtype=torch.long, device=self.device)
        if n_val:
            X_val = torch.tensor(
                X[-n_val:], dtype=torch.float32, device=self.device
            )
            y_val = torch.tensor(
                y_idx[-n_val:], dtype=torch.long, device=self.device
            )

        class_weights = None
        sample_weight = None
        if self.class_weight == "balanced":
            counts = np.maximum(
                np.bincount(y_fit, minlength=len(self.classes_)), 1
            )
            class_weights = torch.tensor(
                len(y_fit) / (len(self.classes_) * counts),
                dtype=torch.float32,
                device=self.device,
            )
            sample_weight = class_weights[y_tensor]

        self.train_losses_, self.val_losses_ = [], []
        # Cross-Entropy Loss is used here because this is a multi-class
        # classification problem (Home win / Draw / Away win). CrossEntropyLoss
        # combines a log-softmax activation with negative log-likelihood loss
        # in a single numerically stable operation. It penalises the model when
        #  it assigns low probability to the correct class, pushing the raw
        # logits (the network's unnormalised output) towards high confidence
        # for the true label. Because of this, the network's final layer must
        # output one raw logit per class rather than a probability — the loss
        # function handles the softmax internally.
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        best_val, best_state, stale = float("inf"), None, 0
        params = ("W1", "b1", "W2", "b2")
        print_step = max(1, epochs // 10)
        for epoch in range(epochs):
            output = self.forward(X_tensor)
            #  Compute loss using Cross-Entropy Loss
            loss = criterion(output, y_tensor)
            self.train_losses_.append(loss.item())
            #  update weights
            self.backward(X_tensor, y_tensor, output, lr, sample_weight)

            val_loss = None
            if n_val:
                with torch.no_grad():
                    val_loss = criterion(self.forward(X_val), y_val).item()
                self.val_losses_.append(val_loss)
                if val_loss < best_val:
                    best_val, stale = val_loss, 0
                    best_state = {
                        p: getattr(self, p).detach().clone() for p in params
                    }
                else:
                    stale += 1

            if (epoch + 1) % print_step == 0:
                val_msg = f", Val loss: {val_loss:.4f}" if n_val else ""
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Loss: {loss.item():.4f}{val_msg}"
                )

            patience = self.early_stopping_patience
            if patience is not None and stale >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if self.early_stopping_patience is not None and best_state:
            with torch.no_grad():
                for p in params:
                    getattr(self, p).copy_(best_state[p])
        return self

    def predict(self, X):
        check_is_fitted(self, ["W1", "W2"])
        X = check_array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.forward(X_tensor)
            y_idx = torch.argmax(logits, dim=1).cpu().numpy()

        return self.classes_[y_idx]

    def predict_proba(self, X):
        check_is_fitted(self, ["W1", "W2"])
        X = check_array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.forward(X_tensor)
            y_proba = torch.softmax(logits, dim=1)

        return y_proba.cpu().numpy()

# TODO check on internet simple MLP classifier implementation in PyTorch and sklearn integration.
# https://medium.com/@mn05052002/building-a-simple-mlp-from-scratch-using-pytorch-7d50ca66512b
# https://optuna.org/

# TODO search how they explained in the course or how the code in the code worked