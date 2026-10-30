import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate = 0.1, epochs = 100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        n_samples, n_features = x.shape

        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_history = []

        for epoch in range(self.epochs):
            z = x @ self.w + self.b
            y_pred = self.sigmoid(z)

            y_pred_safe = np.clip(y_pred, 1e-9, 1 - 1e-9)
            loss = -np.mean(y * np.log(y_pred_safe) + (1 - y) * np.log(1 - y_pred_safe))
            self.loss_history.append(loss)

            dw = x.T @ (y_pred - y) / n_samples
            db = np.mean(y_pred - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, x):
        z = x @ self.w + self.b
        y_prob = self.sigmoid(z)

        return (y_prob >= 0.5).astype(int)
