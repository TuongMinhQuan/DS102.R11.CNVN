import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_data():
    mnist = fetch_openml("mnist_784", version = 1, as_frame = False)

    x = mnist.data
    y = mnist.target

    mask = (y == "0") | (y == "1")
    x = x[mask].astype(float)
    y = y[mask].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, x_test, y_train, y_test