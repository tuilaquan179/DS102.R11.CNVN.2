import numpy as np

class LogisticRegression:
    def __init__(self, epoch=1000, lr=0.1):
        self.epoch = epoch
        self.lr = lr
        self.losses = []
        self.metrics = []
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss_function(self, y, y_hat):
        epsilon = 1e-10 
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def accuracy(self, y, y_hat):
        return (y == y_hat).mean()

    def predict(self, X):
        y_hat = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return (y_hat >= 0.5).astype(int)  

    def GradientDescent(self, X, y):
        n_sample, n_feature = X.shape
        if self.weights is None:
            self.weights = np.zeros(n_feature)
            self.bias = 0.0

        for epoch in range(self.epoch):
            y_hat = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # Tinh weights va bias'
            dw = np.dot(X.T, (y_hat - y)) / n_sample
            db = np.sum(y_hat - y) / n_sample

            # Update weights va bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Tinh loss va accuracy
            loss = self.loss_function(y, y_hat)
            y_pred = (y_hat >= 0.5).astype(int)
            acc = self.accuracy(y, y_pred)

            self.losses.append(loss)
            self.metrics.append(acc)

            if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == self.epoch - 1:
                print(f"Epoch: {epoch+1}/{self.epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    def fit(self, X, y):
        self.GradientDescent(X, y)

def evaluate_metrics(y, y_hat):
    tp = np.sum((y == 1) & (y_hat == 1))
    fp = np.sum((y == 0) & (y_hat == 1))
    fn = np.sum((y == 1) & (y_hat == 0))
    tn = np.sum((y == 0) & (y_hat == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y)
    return accuracy, precision, recall, f1