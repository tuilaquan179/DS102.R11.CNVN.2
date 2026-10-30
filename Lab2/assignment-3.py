import idx2numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train = idx2numpy.convert_from_file("data/train-images.idx3-ubyte").reshape(-1, 784) / 255.0
y_train = idx2numpy.convert_from_file("data/train-labels.idx1-ubyte")
X_test = idx2numpy.convert_from_file("data/t10k-images.idx3-ubyte").reshape(-1, 784) / 255.0
y_test = idx2numpy.convert_from_file("data/t10k-labels.idx1-ubyte")

# Logistic
train_mask, test_mask = (y_train <= 1), (y_test <= 1)
model_logistic = LogisticRegression(max_iter=1000).fit(X_train[train_mask], y_train[train_mask])
print("Logistic")
print(classification_report(y_test[test_mask], model_logistic.predict(X_test[test_mask]), digits=4))

# Softmax
model_softmax = LogisticRegression(multi_class='multinomial', max_iter=1000).fit(X_train, y_train)
print("Softmax")
print(classification_report(y_test, model_softmax.predict(X_test), digits=4))