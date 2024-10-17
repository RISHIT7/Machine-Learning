import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn import preprocessing
import sys

class WeightedLogisticRegression:
    def __init__(self, n_features, n_classes):
        self.W = np.zeros((n_features, n_classes), dtype=np.float64)
    
    def accuracy_score(self, y, y_pred):
        y_true = y.copy()

        if (y_true.shape[0]!=y_pred.shape[0]):
            print("Prediction file of wrong dimensions")
            exit()

        total_samples = y_true.shape[0]
        correct = 0

        for i in range(y_true.shape[0]):
            true = int(y_true[i][0])
            pred = int(y_pred[i][0])
            if (pred==true):
                correct+=1
        accuracy = correct/total_samples

        print("Accuracy obtained on the test set: " + str(accuracy))
    
    def _ternary_search(self, X, Y, freq, eta_0, W, g, L_w, iterations=20):
        eta_l = 0
        eta_h = eta_0
        while self._calculate_loss(softmax(X @ (W - eta_h * g), axis=1), Y, freq) < L_w:
            eta_h *= 2
        
        for _ in range(iterations):
            eta_1 = (2 * eta_l + eta_h) / 3
            eta_2 = (eta_l + 2 * eta_h) / 3

            loss_1 = self._calculate_loss(softmax(X @ (W - eta_1 * g), axis=1), Y, freq)
            loss_2 = self._calculate_loss(softmax(X @ (W - eta_2 * g), axis=1), Y, freq)
            
            if loss_1 > loss_2:
                eta_l = eta_1
            elif loss_2 > loss_1:
                eta_h = eta_2
            else:
                eta_l = eta_1
                eta_h = eta_2
        
        return (eta_l + eta_h) / 2

    def _calculate_loss(self, probs, Y, freq):
        log_probs = np.log(probs) / freq
        weighted_log_probs = log_probs * Y
        loss = -np.mean(weighted_log_probs.sum(axis=1)) /2
        
        return loss
    
    def _calculate_gradient(self, probs, X, Y, freq):
        error = (probs - Y)
        weighted_error = error / freq[np.argmax(Y, axis=1), None]
        gradient = (X.T @ weighted_error) / (2*X.shape[0])
        
        return gradient

    def _adam_update(self, gradient, m, v, t, beta1, beta2, eta, epsilon):
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            # Update weights
            return m, v, eta * m_hat / (np.sqrt(v_hat) + epsilon)

    def fit(self, X, Y, freq, batch_size, epochs, eta_0=1e-9, adaptive=1, decay=0.9, verbose=False):
        W = self.W
        m = np.zeros_like(W)
        v = np.zeros_like(W)

        for epoch in range(1, epochs+1):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                Y_batch = Y[i:i + batch_size]

                logits = X_batch @ W
                probs = softmax(logits, axis=1)

                if verbose or adaptive == 3:
                    loss = self._calculate_loss(probs, Y_batch, freq)

                gradient = self._calculate_gradient(probs, X_batch, Y_batch, freq)
                
                if adaptive == 1:
                    eta_t = eta_0
                elif adaptive == 2:
                    eta_t = eta_0 / (1 + (decay * epoch))
                elif adaptive == 3:
                    eta_t = self._ternary_search(X_batch, Y_batch, freq, eta_0, W, gradient, loss)

                elif adaptive == 4:
                    eta_t = eta_0
                    if epoch > epochs*(2/3):
                        eta_t = decay
                elif adaptive == 5:
                    # Adam
                    eta_t = eta_0
                    
                    m, v, update_value = self._adam_update(gradient, m, v, epoch, 0.9, 0.999, eta_0, 1e-12)
                    W -= update_value
                    
                if adaptive != 5:
                    W -= eta_t * gradient
                
                if verbose:
                    print(f"Epoch Number: {epoch}, Batch Number: {i // batch_size + 1}, Batch Loss (before updating weights): {loss}")

        return W
    
    def predict(self, X):
        return np.argmax(softmax(X @ self.W, axis=1), axis=1, keepdims=True)

def part_a():
    train_file = sys.argv[2]
    params_file = sys.argv[3]
    model_file = sys.argv[4]
    
    df = pd.read_csv(train_file)
    X = df.drop('Race', axis=1).values
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = df['Race']
    Y_oneHotEncoded = pd.get_dummies(y).values

    freq = np.sum(Y_oneHotEncoded, axis=0)
    
    n_features = X.shape[1]
    n_classes = Y_oneHotEncoded.shape[1]

    with open(params_file, 'r') as f:
        params = f.readlines()
        
        adaptive_type = int(params[0].strip())
        string = params[1].strip()
        
        if adaptive_type == 2:
            learning_rate, decay = map(float, string.split(','))
        else:
            learning_rate = float(string)
            decay = 0
        
        epochs = int(params[2].strip())
        batch_size = int(params[3].strip())

    model = WeightedLogisticRegression(n_features, n_classes)
    W_optimized_constant = model.fit(X, Y_oneHotEncoded, freq, batch_size, epochs, eta_0=learning_rate, adaptive=adaptive_type, decay=decay, verbose=False)

    # test 
    X_test = pd.read_csv("test1.csv").values
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    y_pred = model.predict(X_test)
    y_test = pd.read_csv("test_pred1.csv").values
    model.accuracy_score(y_test, y_pred)

    W_flatten = W_optimized_constant.flatten()
    np.savetxt(model_file, W_flatten)

def part_b():
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    model_file = sys.argv[4]
    pred_file = sys.argv[5]
    
    df = pd.read_csv(train_file)
    X = df.drop('Race', axis=1).values
    scaler = preprocessing.StandardScaler().fit(X)
    X_train = scaler.transform(X)
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    y = df['Race']
    Y_oneHotEncoded = pd.get_dummies(y).values

    freq = np.sum(Y_oneHotEncoded, axis=0)

    n_features = X_train.shape[1]
    n_classes = Y_oneHotEncoded.shape[1]

    adaptive_type = 4
    learning_rate, decay = 24, 8
    epochs = 25
    batch_size = 100

    model = WeightedLogisticRegression(n_features, n_classes)
    W_optimized_constant = model.fit(X_train, Y_oneHotEncoded, freq, batch_size, epochs, eta_0=learning_rate, adaptive=adaptive_type, decay=decay, verbose=True)
    W_flatten = W_optimized_constant.flatten()
    np.savetxt(model_file, W_flatten)

    # test 
    X_test = pd.read_csv(test_file).values
    X_test = scaler.transform(X_test)
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    y_pred = softmax(X_test @ W_optimized_constant, axis=1)
    np.savetxt(pred_file, y_pred, delimiter=',')


def main():
    part = sys.argv[1]
    if part == "a":
        part_a()
    elif part == "b":
        part_b()

if __name__ == "__main__":
    main()