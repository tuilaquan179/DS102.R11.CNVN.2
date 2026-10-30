import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression, evaluate_metrics

train_images = idx2numpy.convert_from_file("data/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("data/train-labels.idx1-ubyte")
train_data = (train_images, train_labels)

test_images = idx2numpy.convert_from_file("data/t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("data/t10k-labels.idx1-ubyte")
test_data = (test_images, test_labels)

def filter_data(data, condition):
    images, labels = data
    new_images = images[labels == condition]
    new_labels = labels[labels == condition]
    return new_images, new_labels

train_images_0, train_labels_0 = filter_data(train_data, 0)
train_images_1, train_labels_1 = filter_data(train_data, 1)

train_images = np.concatenate((train_images_0, train_images_1), axis=0)
train_labels = np.concatenate((train_labels_0, train_labels_1), axis=0)

test_images_0, test_labels_0 = filter_data(test_data, 0)
test_images_1, test_labels_1 = filter_data(test_data, 1)

test_images = np.concatenate((test_images_0, test_images_1), axis=0)
test_labels = np.concatenate((test_labels_0, test_labels_1), axis=0)

X_train = train_images.reshape(-1, 28 * 28) / 255.0
y_train = train_labels

X_test = test_images.reshape(-1, 28 * 28) / 255.0
y_test = test_labels

model = LogisticRegression(epoch=1000, lr=0.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy, precision, recall, f1 = evaluate_metrics(y_test, y_pred)

print(" \n Ket qua \n ")
print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(model.losses) + 1), model.losses)
plt.show()