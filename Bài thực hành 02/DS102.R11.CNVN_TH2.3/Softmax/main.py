import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from data_utils import load_data
from softmax_regression import SoftmaxRegression

x_train, x_test, y_train, y_test = load_data()

model = SoftmaxRegression(learning_rate = 0.1, epochs = 10)
model.fit(x_train, y_train)

sm_pred = model.predict(x_test)

sk_model = SklearnLogisticRegression(max_iter = 100)
sk_model.fit(x_train, y_train)
sk_pred = sk_model.predict(x_test)


def evaluate(y_true, y_pred):
    confusion_matrix = np.zeros((10, 10), dtype=int)

    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1

    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(10):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0

        f1 = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return (
        np.mean(precision_list),
        np.mean(recall_list),
        np.mean(f1_list)
    )

sm_precision, softmax_recall, softmax_f1 = evaluate(y_test, sm_pred)
sk_precision, sk_recall, sk_f1 = evaluate(y_test, sk_pred)

print("Softmax Regression:")
print("Precision:", round(sm_precision, 4))
print("Recall:", round(softmax_recall, 4))
print("F1-score:", round(softmax_f1, 4))

print("\nScikit-learn:")
print("Precision:", round(sk_precision, 4))
print("Recall:", round(sk_recall, 4))
print("F1-score:", round(sk_f1, 4))

print("\nSo sánh:")
print("Độ lệch Precision:", round(sk_precision - sm_precision, 4))
print("Độ lệch Recall:", round(sk_recall - softmax_recall, 4))
print("Độ lệch F1-score:", round(sk_f1 - softmax_f1, 4))