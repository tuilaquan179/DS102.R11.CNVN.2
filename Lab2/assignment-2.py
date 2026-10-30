import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from softmax_regression import SoftmaxRegression, evaluate_metrics

train_images = idx2numpy.convert_from_file("data/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("data/train-labels.idx1-ubyte")
test_images = idx2numpy.convert_from_file("data/t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("data/t10k-labels.idx1-ubyte")

X_train = train_images.reshape(-1, 28 * 28) / 255.0
y_train = train_labels

X_test = test_images.reshape(-1, 28 * 28) / 255.0
y_test = test_labels

print(f"Kich thuoc tap train: {X_train.shape}")
print(f"Kich thuoc tap ttest : {X_test.shape}")

model = SoftmaxRegression(epoch=1000, lr=0.5, num_class=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy, precision, recall, f1 = evaluate_metrics(y_test, y_pred, num_class=10)

print("Ket qua")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")
print(f"Macro F1-Score: {f1:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(model.losses) + 1), model.losses)
plt.show()
