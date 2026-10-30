import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from data_utils import load_data
from logistic_regression import LogisticRegression

x_train, x_test, y_train, y_test = load_data()

model = LogisticRegression(learning_rate = 0.1, epochs = 100)
model.fit(x_train, y_train)

lr_pred = model.predict(x_test)

sk_model = SklearnLogisticRegression(max_iter = 1000)
sk_model.fit(x_train, y_train)
sk_pred = sk_model.predict(x_test)


def evaluate(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)

    return precision, recall, f1


lr_precision, my_recall, my_f1 = evaluate(y_test, lr_pred)
sk_precision, sk_recall, sk_f1 = evaluate(y_test, sk_pred)

print("Logistic Regression:")
print("Precision:", round(lr_precision, 4))
print("Recall:", round(my_recall, 4))
print("F1-score:", round(my_f1, 4))

print("\nScikit-learn:")
print("Precision:", round(sk_precision, 4))
print("Recall:", round(sk_recall, 4))
print("F1-score:", round(sk_f1, 4))

print("\nSo sánh:")
print("Độ lệch Precision:", round(sk_precision - lr_precision, 4))
print("Độ lệch Recall:", round(sk_recall - my_recall, 4))
print("Độ lệch F1-score:", round(sk_f1 - my_f1, 4))
