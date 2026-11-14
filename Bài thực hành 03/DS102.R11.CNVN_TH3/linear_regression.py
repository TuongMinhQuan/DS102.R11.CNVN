import numpy as np

class LinearRegression:

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X = np.asarray(X, dtype = float)
        y = np.asarray(y, dtype = float).reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X phải là mảng 2 chiều.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X và y phải có cùng số hàng.")

        X = np.c_[np.ones((X.shape[0], 1)), X]

        self.weights = np.linalg.pinv(X) @ y

        return self

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Mô hình cần được huấn luyện trước khi dự đoán.")

        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError("X phải là mảng 2 chiều.")

        X = np.c_[np.ones((X.shape[0], 1)), X]

        return (X @ self.weights).ravel()

    def evaluate(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype = float).ravel()
        y_pred = np.asarray(y_pred, dtype = float).ravel()

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true và y_pred phải có cùng kích thước.")

        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))