import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data
from logistic_regression import LogisticRegression

x_train, x_test, y_train, y_test = load_data()

model = LogisticRegression(learning_rate = 0.1, epochs = 100)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

TP = np.sum((y_test == 1) & (y_pred == 1))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))

precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0

f1_score = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1_score, 4))

plt.plot(model.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()