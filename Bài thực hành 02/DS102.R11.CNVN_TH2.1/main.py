import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_data
from logistic_regression import LogisticRegression

x_train, x_test, y_train, y_test = load_data()

model = LogisticRegression(learning_rate = 0.1, epochs = 100)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

TP = np.sum((y_test == 1) & (y_pred == 1))
TN = np.sum((y_test == 0) & (y_pred == 0))
FP = np.sum((y_test == 0) & (y_pred == 1))
FN = np.sum((y_test == 1) & (y_pred == 0))

precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0

f1_score = (2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1_score, 4))

confusion_matrix = np.array([
    [TN, FP],
    [FN, TP]
])

print("\nConfusion Matrix:")
print(confusion_matrix)

fig, axes = plt.subplots(1, 2, figsize = (12, 5))

axes[0].imshow(confusion_matrix, cmap = "Blues")
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")
axes[0].set_xticks([0, 1], ["0", "1"])
axes[0].set_yticks([0, 1], ["0", "1"])

for i in range(2):
    for j in range(2):
        axes[0].text(
            j, i, confusion_matrix[i, j],
            ha = "center", va = "center", color = "red"
        )

axes[1].plot(model.loss_history)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Training Loss")

plt.tight_layout()
plt.show()
