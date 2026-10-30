import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data
from softmax_regression import SoftmaxRegression

x_train, x_test, y_train, y_test = load_data()

model = SoftmaxRegression(learning_rate = 0.1, epochs = 100)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

confusion_matrix = np.zeros((10, 10), dtype = int)

for i in range(len(y_test)):
    confusion_matrix[y_test[i], y_pred[i]] += 1

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

print("Precision:", round(np.mean(precision_list), 4))
print("Recall:", round(np.mean(recall_list), 4))
print("F1-score:", round(np.mean(f1_list), 4))

plt.plot(model.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()