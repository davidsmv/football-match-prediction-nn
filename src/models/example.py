import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class TorchBinaryNN(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden1=16, hidden2=8, lr=0.001, epochs=30, batch_size=32):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

    def _build_model(self, n_features):
        return nn.Sequential(
            nn.Linear(n_features, self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2, 1),
            nn.Sigmoid()
        )

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_ = self._build_model(self.n_features_in_).to(self.device_)

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device_)
        y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device_)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model_(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["model_", "n_features_in_"])
        X = check_array(X)

        X_t = torch.tensor(X, dtype=torch.float32).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            probs = self.model_(X_t).cpu().numpy().ravel()

        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)