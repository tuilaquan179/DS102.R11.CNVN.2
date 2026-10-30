import numpy as np

class SoftmaxRegression:
    def __init__(self, epoch=1000, lr=0.1, num_class = 10):
        self.epoch = epoch
        self.lr = lr
        self.num_class = num_class
        self.losses = []
        self.metrics = []
        self.weights = None
        self.bias = None

    def softmax(self, z):
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def loss_function(self, y, y_hat):
        epsilon = 1e-10 
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y * np.log(y_hat), axis=1))

    def accuracy(self, y, y_hat):
        return np.mean(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.softmax(z)
        return np.argmax(y_hat, axis=1)  

    def GradientDescent(self, X, y):
        n_sample, n_feature = X.shape
        if y.ndim == 1:
            y_one_hot = np.eye(self.num_class)[y]
        else:
            y_one_hot = y

        if self.weights is None:
            self.weights = np.zeros((n_feature, self.num_class))
            self.bias = np.zeros(self.num_class)
            
        for epoch in range(self.epoch):
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.softmax(z)

            dw = np.dot(X.T, (y_hat - y_one_hot)) / n_sample
            db = np.sum(y_hat - y_one_hot, axis=0) / n_sample

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = self.loss_function(y_one_hot, y_hat)
            acc = self.accuracy(y_one_hot, y_hat)

            self.losses.append(loss)
            self.metrics.append(acc)

            if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == self.epoch - 1:
                print(f"Epoch: {epoch+1}/{self.epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    def fit(self, X, y):
        self.GradientDescent(X, y)

def evaluate_metrics(y_true, y_pred, num_class=10):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    precisions, recalls, f1s = [], [], []

    for c in range(num_class):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        p = tp / (tp+fp) if (tp+fp) > 0 else 0.0
        r = tp / (tp+fn) if (tp+fn) > 0 else 0.0
        f1 = 2 * p * r / (p+r) if (p+r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    accuracy = np.mean(y_true == y_pred)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    return accuracy, macro_precision, macro_recall, macro_f1