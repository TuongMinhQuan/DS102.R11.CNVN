import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate = 0.1, epochs = 100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None
        self.loss_history = []

    def softmax(self, z):
        z = z - np.max(z, axis = 1, keepdims = True)
        exp_z = np.exp(z)

        return exp_z / np.sum(exp_z, axis = 1, keepdims = True)

    def fit(self, x, y):
        n_samples, n_features = x.shape
        n_classes = 10

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.loss_history = []

        y_one_hot = np.eye(n_classes)[y]

        for epoch in range(self.epochs):
            z = x @ self.W + self.b
            y_pred = self.softmax(z)

            y_pred_safe = np.clip(y_pred, 1e-10, 1.0)
            loss = -np.mean(np.sum(y_one_hot * np.log(y_pred_safe), axis = 1))
            self.loss_history.append(loss)

            dW = x.T @ (y_pred - y_one_hot) / n_samples
            db = np.mean(y_pred - y_one_hot, axis = 0)

            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    def predict(self, x):
        z = x @ self.W + self.b
        y_pred = self.softmax(z)

        return np.argmax(y_pred, axis = 1)